import os
import time
import subprocess
import uuid
import torch
import torchaudio
import requests
import importlib.util
from cog import BasePredictor, Input, Path
from transformers import AutoModel, AutoProcessor


import concurrent.futures

def download_hf_model(repo_id, local_dir):
    """HuggingFace modelini Pget Manifest kullanarak en hızlı şekilde indir."""
    api_url = f"https://huggingface.co/api/models/{repo_id}"
    
    print(f"  Dosya listesi alınıyor: {api_url}")
    resp = requests.get(api_url, timeout=30)
    resp.raise_for_status()
    
    all_files = [s["rfilename"] for s in resp.json().get("siblings", [])]
    
    # Teknik olarak gerekmeyen dosyaları filtrele (Resimler, README, .gitattributes vb.)
    exclude_extensions = {".png", ".jpg", ".jpeg", ".md", ".gitattributes", ".jinja", ".txt"}
    critical_files = {"merges.txt", "vocab.json", "tokenizer.json", "added_tokens.json"}
    
    files_to_download = []
    for f in all_files:
        name_lower = f.lower()
        ext = os.path.splitext(name_lower)[1]
        if f in critical_files:
            files_to_download.append(f)
        elif ext in exclude_extensions:
            continue
        elif "images/" in name_lower or "test" in name_lower:
            continue
        else:
            files_to_download.append(f)

    print(f"  Toplam {len(all_files)} dosyadan {len(files_to_download)} tanesi seçildi.")
    os.makedirs(local_dir, exist_ok=True)
    
    has_pget = subprocess.run(["which", "pget"], capture_output=True).returncode == 0
    
    if has_pget:
        # Pget Manifest dosyası oluştur
        manifest_path = os.path.join("/tmp", f"manifest_{uuid.uuid4().hex}.txt")
        with open(manifest_path, "w") as f:
            for filename in files_to_download:
                filepath = os.path.abspath(os.path.join(local_dir, filename))
                url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
                f.write(f"{url} {filepath}\n")
        
        print(f"  ⚡ Pget Manifest ile indirme başlatılıyor...")
        try:
            # Replicate'in pget aracı manifest dosyasını kullanarak tüm dosyaları paralel indirir
            subprocess.run(["pget", "multidownload", manifest_path], check=True)
            print(f"  ✓ Tüm dosyalar başarıyla indirildi.")
            return
        except Exception as e:
            print(f"  ⚠️ Pget Manifest başarısız oldu, ThreadPool fallback'e geçiliyor: {e}")
        finally:
            if os.path.exists(manifest_path):
                os.remove(manifest_path)

    # Fallback: ThreadPool ile paralel indirme (Pget yoksa veya hata verdiyse)
    print(f"  ℹ️ ThreadPoolExecutor ile paralel indirme (max_workers=10)...")
    def _download_task(filename):
        filepath = os.path.join(local_dir, filename)
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            return
        url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
        _download_with_requests(url, filepath, 0, 0, filename)

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        list(executor.map(_download_task, files_to_download))


def _download_with_requests(url, filepath, idx, total, filename):
    """Yedek veya küçük dosya indirme yöntemi."""
    for attempt in range(3):
        try:
            with requests.get(url, stream=True, timeout=300, allow_redirects=True) as r:
                r.raise_for_status()
                with open(filepath, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        f.write(chunk)
            break
        except Exception as e:
            if attempt < 2:
                time.sleep(2)
            else:
                raise


class Predictor(BasePredictor):
    def setup(self):
        """Modeli belleğe yükle (Sadece bir kez çalışır)"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        # Disable the broken cuDNN SDPA backend
        torch.backends.cuda.enable_cudnn_sdp(False)
        # Keep these enabled as fallbacks
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)

        self.model_name = "OpenMOSS-Team/MOSS-TTS"
        self.audio_tokenizer_name = "OpenMOSS-Team/MOSS-Audio-Tokenizer"
        self.local_dir = "./moss_model_cache"
        self.audio_tokenizer_dir = "./moss_audio_tokenizer_cache"
        
        print(f"[{self.model_name}] indirme başlatılıyor (requests/urllib3)...")
        if not os.path.exists(self.local_dir) or not os.listdir(self.local_dir):
            download_hf_model(self.model_name, self.local_dir)
            print("MOSS-TTS dosyaları başarıyla indirildi!")
        else:
            print("MOSS-TTS diskte mevcut, atlandı.")
        
        print(f"[{self.audio_tokenizer_name}] indirme başlatılıyor...")
        if not os.path.exists(self.audio_tokenizer_dir) or not os.listdir(self.audio_tokenizer_dir):
            download_hf_model(self.audio_tokenizer_name, self.audio_tokenizer_dir)
            print("MOSS-Audio-Tokenizer dosyaları başarıyla indirildi!")
        else:
            print("MOSS-Audio-Tokenizer diskte mevcut, atlandı.")
        
        # Offline çalışabilmesi için Audio Tokenizer rotasını lokal önbellek (cache) ile güncelliyoruz.
        processing_file = os.path.join(self.local_dir, "processing_moss_tts.py")
        if os.path.exists(processing_file):
            with open(processing_file, 'r') as f:
                content = f.read()
            
            abs_audio_path = os.path.abspath(self.audio_tokenizer_dir)
            content = content.replace(
                '"OpenMOSS-Team/MOSS-Audio-Tokenizer"',
                f'"{abs_audio_path}"'
            )
            content = content.replace(
                "'OpenMOSS-Team/MOSS-Audio-Tokenizer'",
                f'"{abs_audio_path}"'
            )
            with open(processing_file, 'w') as f:
                f.write(content)
            print(f"processing_moss_tts.py yamalandı → Audio Tokenizer lokal yoldan yüklenecek.")
        
        # Değiştirilen dosyanın anında etki etmesi için Hugging Face önbelleğini temizliyoruz.
        cache_dir = "/root/.cache/huggingface/modules/transformers_modules"
        if os.path.exists(cache_dir):
            import shutil
            shutil.rmtree(cache_dir)
            print("Transformers modül cache'i temizlendi.")
        
        # Model ve tokenizer yüklendi, ağ isteklerini tamamen kapatıyoruz.
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
            
        print("Model RAM'e yükleniyor...")
        
        def resolve_attn_implementation() -> str:
            # Prefer FlashAttention 2 when package + device conditions are met.
            if (
                self.device == "cuda"
                and importlib.util.find_spec("flash_attn") is not None
                and self.dtype in {torch.float16, torch.bfloat16}
            ):
                major, _ = torch.cuda.get_device_capability()
                if major >= 8:
                    return "flash_attention_2"
        
            # CUDA fallback: use PyTorch SDPA kernels.
            if self.device == "cuda":
                return "sdpa"
        
            # CPU fallback.
            return "eager"

        attn_implementation = resolve_attn_implementation()
        print(f"[INFO] Using attn_implementation={attn_implementation}")
        
        self.processor = AutoProcessor.from_pretrained(self.local_dir, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            self.local_dir, 
            trust_remote_code=True,
            attn_implementation=attn_implementation,
            dtype=self.dtype
        ).to(self.device)
        self.model.eval()
        print("Model ve işlemci başarıyla RAM'e yüklendi, API isteklere hazır!")

    def predict(
        self,
        reference_audio: Path = Input(
            description="Ses klonlama için referans ses dosyası (WAV/MP3). Modelin bu sesi taklit etmesini sağlar. 3-30 saniye arası temiz bir kayıt önerilir. Boş bırakılırsa modelin varsayılan sesi kullanılır.",
            default=None
        ),
        text: str = Input(
            description="Sese çevrilecek metin. Türkçe, İngilizce, Çince, Almanca, Fransızca, İspanyolca, Japonca, Korece ve daha birçok dili destekler. Pinyin ve IPA girdileri de kabul edilir.",
            default="Merhaba, bugün hava çok güzel."
        ),
        audio_temperature: float = Input(
            description="Ses üretimi sıcaklığı. Düşük değerler (0.5-1.0) daha monoton ve kararlı bir ses üretir. Yüksek değerler (1.5-2.0) daha doğal, duygusal ve canlı bir ses üretir ancak tutarsızlık riski artar. Çince için 1.7, İngilizce için 1.5 önerilir.",
            default=1.7,
            ge=0.0,
            le=3.0
        ),
        audio_top_p: float = Input(
            description="Ses için Nucleus Sampling eşiği. Model bir sonraki ses parçasını seçerken, olasılık sıralamasında üst yüzde kaçlık dilimi kullanacağını belirler. 0.8 = en olası %80'lik dilimden seç. Düşük değerler daha tutarlı, yüksek değerler daha çeşitli sonuç verir.",
            default=0.8,
            ge=0.0,
            le=1.0
        ),
        audio_top_k: int = Input(
            description="Ses için Top-K Sampling. Model bir sonraki ses parçasını seçerken en olası kaç adayı değerlendireceğini belirler. 25 = en olası 25 aday arasından seç. Düşük değerler daha güvenli, yüksek değerler daha yaratıcı sonuç verir.",
            default=25,
            ge=1,
            le=200
        ),
        max_new_tokens: int = Input(
            description="Üretilebilecek maksimum token sayısı. Çıktı ses uzunluğunu kontrol eder. Kural: 1 saniye ≈ 12.5 token. Örnek: 2048 token ≈ ~163 saniye ses. Uzun metinler için artırın.",
            default=2048,
            ge=64,
            le=8192
        ),
        expected_duration_sec: float = Input(
            description="Beklenen çıktı ses süresi (saniye). 0 = otomatik (model kendisi belirler). Belirlenmişse modelin konuşma hızını bu süreye göre ayarlar. En iyi sonuç için metnin doğal okuma süresinin 0.5x-1.5x aralığında olmalıdır.",
            default=0,
            ge=0,
            le=120
        ),
        audio_repetition_penalty: float = Input(
            description="Ses tekrar cezası. 1.0 = ceza yok. 1.0'dan büyük değerler, modelin aynı ses kalıplarını tekrar etmesini engeller. Tekrarlayan/takılan ses sorunlarında artırın.",
            default=1.0,
            ge=0.5,
            le=3.0
        ),
        text_temperature: float = Input(
            description="[Deneysel] Metin motoru sıcaklığı. Modelin hangi kelimelerin/yapıların üretileceğine karar verme aşamasını kontrol eder. Varsayılan 1.5 değeri model geliştiricileri tarafından optimize edilmiştir, değiştirmek önerilmez.",
            default=1.5,
            ge=0.0,
            le=3.0
        ),
        text_top_p: float = Input(
            description="[Deneysel] Metin için Nucleus Sampling. Metin token seçiminde kullanılır. Varsayılan 1.0 değeri model geliştiricileri tarafından optimize edilmiştir.",
            default=1.0,
            ge=0.0,
            le=1.0
        ),
        text_top_k: int = Input(
            description="[Deneysel] Metin için Top-K Sampling. Metin token adaylarını sınırlar. Varsayılan 50 değeri model geliştiricileri tarafından optimize edilmiştir.",
            default=50,
            ge=1,
            le=200
        ),
        output_format: str = Input(
            description="Çıktı ses formatı.",
            default="mp3",
            choices=["mp3", "wav"]
        ),
        sample_rate: int = Input(
            description="Çıktı örnekleme hızı (Hz). 24000 modelin ham hızıdır, 44100 daha yüksek kalite için resample edilir.",
            default=44100,
            choices=[24000, 44100]
        ),
    ) -> Path:
        """MOSS-TTS ile metinden ses üret veya ses klonla."""
        
        print(f"İşlem başlıyor... Text: {text[:50]}...")

        ref_path = None
        if reference_audio:
            print(f"Referans ses kullanılıyor: {reference_audio}")
            ref_path = [str(reference_audio)]
        
        tokens_arg = None
        if expected_duration_sec > 0:
            tokens_arg = int(expected_duration_sec * 12.5)
            print(f"Beklenen süre: {expected_duration_sec}s → {tokens_arg} token")

        if ref_path:
             inputs = self.processor.build_user_message(text=text, reference=ref_path, tokens=tokens_arg)
        else:
             inputs = self.processor.build_user_message(text=text, tokens=tokens_arg)
            
        batch = self.processor([inputs], mode="generation")
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            text_temperature=text_temperature,
            text_top_p=text_top_p,
            text_top_k=text_top_k,
            audio_temperature=audio_temperature,
            audio_top_p=audio_top_p,
            audio_top_k=audio_top_k,
            audio_repetition_penalty=audio_repetition_penalty,
        )

        output_path = Path(f"/tmp/output_{uuid.uuid4().hex}.wav")
        if output_path.exists():
            os.remove(output_path)
            
        decoded = self.processor.decode(outputs)
        for message in decoded:
            if hasattr(message, 'audio_codes_list') and message.audio_codes_list:
                audio_tensor = message.audio_codes_list[0] # [tokens] or [1, tokens]
                if audio_tensor.ndim == 1:
                    audio_tensor = audio_tensor.unsqueeze(0) # [1, tokens]
                
                model_sr = getattr(self.processor.model_config, 'sampling_rate', getattr(self.model.config, 'sampling_rate', 24000))
                
                # Resampling if needed
                if sample_rate != model_sr:
                    print(f"Resampling: {model_sr}Hz -> {sample_rate}Hz")
                    resampler = torchaudio.transforms.Resample(model_sr, sample_rate).to(self.device)
                    audio_tensor = resampler(audio_tensor)
                
                ext = output_format.lower()
                final_path = f"/tmp/output_{uuid.uuid4().hex}.{ext}"
                
                if ext == "wav":
                    torchaudio.save(final_path, audio_tensor.cpu(), sample_rate)
                else:
                    # MP3 conversion using ffmpeg
                    temp_wav = f"/tmp/temp_{uuid.uuid4().hex}.wav"
                    torchaudio.save(temp_wav, audio_tensor.cpu(), sample_rate)
                    
                    try:
                        subprocess.run([
                            "ffmpeg", "-i", temp_wav,
                            "-codec:a", "libmp3lame",
                            "-b:a", "192k",
                            "-ar", str(sample_rate),
                            final_path, "-y"
                        ], check=True, capture_output=True)
                    finally:
                        if os.path.exists(temp_wav):
                            os.remove(temp_wav)
                
                return Path(final_path)
                
        raise Exception("Ses üretilemedi. Lütfen parametreleri kontrol edin.")

