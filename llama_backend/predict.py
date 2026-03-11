import os
import shutil
import subprocess
import time
import uuid
import torch
from pathlib import Path
from cog import BasePredictor, Input, Path as CogPath
import soundfile as sf

# LlamaCppPipeline imports
import sys
import requests
import fnmatch
# Ensure the cloned repo is in the path
sys.path.append("/src/MOSS-TTS")
from moss_tts_delay.llama_cpp import LlamaCppPipeline, PipelineConfig

def _download_with_requests(url, filepath, idx, total, filename):
    """Yedek indirme yöntemi: requests kütüphanesi."""
    for attempt in range(3):
        try:
            with requests.get(url, stream=True, timeout=600, allow_redirects=True) as r:
                r.raise_for_status()
                downloaded = 0
                with open(filepath, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        f.write(chunk)
                        downloaded += len(chunk)
                mb = downloaded / (1024 * 1024)
                print(f"  [{idx}/{total}] ✓ {filename} ({mb:.1f} MB)")
            break
        except Exception as e:
            print(f"  [{idx}/{total}] Deneme {attempt+1}/3 başarısız: {e}")
            if os.path.exists(filepath):
                os.remove(filepath)
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
            else:
                raise

def download_hf_model(repo_id, local_dir, allow_patterns=None):
    """HuggingFace modelini pget (hızlı) veya requests (yedek) ile indir."""
    api_url = f"https://huggingface.co/api/models/{repo_id}"
    print(f"[{repo_id}] Dosya listesi alınıyor: {api_url}")
    
    resp = requests.get(api_url, timeout=30)
    resp.raise_for_status()
    
    files = [s["rfilename"] for s in resp.json().get("siblings", [])]
    
    if allow_patterns:
        filtered_files = []
        for f in files:
            for p in allow_patterns:
                if fnmatch.fnmatch(f, p):
                    filtered_files.append(f)
                    break
        files = filtered_files
        
    print(f"[{repo_id}] Toplam {len(files)} dosya indirilecek -> {local_dir}")
    os.makedirs(local_dir, exist_ok=True)
    
    has_pget = subprocess.run(["which", "pget"], capture_output=True).returncode == 0
    if has_pget:
        print("  ⚡ pget bulundu! Paralel hızlı indirme aktif.")
    else:
        print("  ℹ️ pget bulunamadı, requests ile indiriliyor.")
    
    for i, filename in enumerate(files, 1):
        filepath = os.path.join(local_dir, filename)
        file_dir = os.path.dirname(filepath)
        if file_dir:
            os.makedirs(file_dir, exist_ok=True)
            
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            print(f"  [{i}/{len(files)}] {filename} - zaten mevcut, atlandı.")
            continue
            
        url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
        print(f"  [{i}/{len(files)}] İndiriliyor: {filename}...")
        
        if has_pget:
            try:
                subprocess.run(["pget", url, filepath, "-f"], check=True, timeout=600)
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                print(f"  [{i}/{len(files)}] ✓ {filename} ({size_mb:.1f} MB)")
            except Exception as e:
                print(f"  [{i}/{len(files)}] pget başarısız, requests'e düşülüyor: {e}")
                if os.path.exists(filepath):
                    os.remove(filepath)
            _download_with_requests(url, filepath, i, len(files), filename)
        else:
            _download_with_requests(url, filepath, i, len(files), filename)
            
    print(f"[{repo_id}] Tüm dosyalar başarıyla indirildi.")

class Predictor(BasePredictor):
    def setup(self):
        """Modeli belleğe yükle (Sadece bir kez çalışır)"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Ağırlık dizinleri
        self.weights_dir = "/src/weights"
        self.gguf_dir = os.path.join(self.weights_dir, "MOSS-TTS-GGUF")
        self.onnx_dir = os.path.join(self.weights_dir, "MOSS-Audio-Tokenizer-ONNX")
        
        os.makedirs(self.weights_dir, exist_ok=True)
        
        # 1. GGUF (Backbone, Embeddings, LM Heads) İndir
        if not os.path.exists(self.gguf_dir) or not os.listdir(self.gguf_dir):
            patterns = [
                "MOSS_TTS_Q4_K_M.gguf",
                "embeddings/*",
                "lm_heads/*",
                "tokenizer/*"
            ]
            download_hf_model("OpenMOSS-Team/MOSS-TTS-GGUF", self.gguf_dir, allow_patterns=patterns)
        else:
            print("MOSS-TTS-GGUF diskte mevcut, atlandı.")
            
        # 2. ONNX (Audio Tokenizer - Encoder, Decoder) İndir
        if not os.path.exists(self.onnx_dir) or not os.listdir(self.onnx_dir):
            download_hf_model("OpenMOSS-Team/MOSS-Audio-Tokenizer-ONNX", self.onnx_dir)
        else:
            print("MOSS-Audio-Tokenizer-ONNX diskte mevcut, atlandı.")

        # 2.5. TensorRT Motorlarını Derle (Maksimum Hız İçin)
        # TensorRT motorları çalıştırılacak karta (örn: L40S) özel olmaları için
        # build aşamasında değil, setup() aşamasında karta ilk binişte derlenir.
        self.trt_dir = os.path.join(self.weights_dir, "MOSS-Audio-Tokenizer-TRT")
        if not os.path.exists(self.trt_dir) or not os.listdir(self.trt_dir):
            print("⚡ TensorRT Motorları Derleniyor... (Bu işlem ilk açılışta kartınıza göre 1-5 dakika sürebilir)")
            os.makedirs(self.trt_dir, exist_ok=True)
            encoder_onnx = os.path.join(self.onnx_dir, "encoder.onnx")
            decoder_onnx = os.path.join(self.onnx_dir, "decoder.onnx")
            
            # build_engine.sh scriptini çalıştırıyoruz
            build_cmd = [
                "bash", 
                "/src/MOSS-TTS/moss_audio_tokenizer/trt/build_engine.sh",
                encoder_onnx,
                decoder_onnx,
                self.trt_dir
            ]
            try:
                subprocess.run(build_cmd, check=True)
                print("⚡ TensorRT motorları başarıyla GPU'nuza özel derlendi!")
            except subprocess.CalledProcessError as e:
                print(f"TensorRT derleme hatası yakalandı. Eğer çalışmazsa lütfen logları kontrol edin: {e}")
                raise
        else:
            print("TensorRT motorları zaten GPU için hazır, atlandı.")

        # 3. Ayar Dosyasının Dinamik Güncellenmesi
        config_path = "./configs/llama_cpp/trt.yaml"
        self.config = PipelineConfig.from_yaml(config_path)
        
        # Yolları manuel olarak konfigürasyona enjekte ediyoruz çünkü default YAML'da olmayabilir.
        # LlamaCppPipeline belgelerine göre; pipeline otomatik olarak gguf_dir ve onnx_dir beklerse,
        # LlamaCppBackbone vs. koduna bakılarak bu yollar ezilmeli.
        # Not: MOSS-TTS dokümantasyonuna göre pipeline args olarak path bekleyebilir veya 
        # config içinde tutabiliriz. Kod standart olarak weights/MOSS... gibi bekliyor.
        
        # Repo dizininde weights/ klasörü yoksa sembolik link atalım (garanti olsun diye)
        local_weights_symlink = "/src/MOSS-TTS/weights"
        if not os.path.exists(local_weights_symlink):
            os.symlink(self.weights_dir, local_weights_symlink)
            print(f"Symlink oluşturuldu: {local_weights_symlink} -> {self.weights_dir}")

        print("LlamaCppPipeline başlatılıyor...")
        # NOT: LlamaCppPipeline(config) kullanımı, model ağırlıklarını RAM/VRAM'e çeker.
        # Pipeline'ı tek bir instance olarak açık bırakacağız.
        self.pipeline = LlamaCppPipeline(self.config)
        self.pipeline.__enter__() # Context manager olarak tasarlandığı için manuel başlatıyoruz.
        print("Model başarıyla yüklendi ve LlamaCppPipeline hazır!")

    def predict(
        self,
        text: str = Input(
            description="Sese çevrilecek metin. Türkçe, İngilizce, Çince, Almanca vb. destekler.",
            default="Merhaba, bugün hava çok güzel. Yeni sistemimiz çok hızlı çalışıyor."
        ),
        reference_audio: CogPath = Input(
            description="Ses klonlama için referans ses dosyası (İsteğe bağlı).",
            default=None
        ),
        language: str = Input(
            description="Dil (örn: 'tr', 'en', 'zh', 'auto')",
            default="auto"
        )
    ) -> CogPath:
        """MOSS-TTS (Llama.cpp + ONNX) ile metinden ses üret."""
        
        print(f"İşlem başlıyor... Text: {text[:50]}...")
        
        ref_path = str(reference_audio) if reference_audio else None
        
        # Pipeline generate 
        t_start = time.time()
        
        waveform = self.pipeline.generate(
            text=text,
            reference_audio=ref_path,
            language=language if language != "auto" else None
        )
        
        t_gen = time.time() - t_start
        print(f"Çıkarım (Inference) tamamlandı. Süre: {t_gen:.2f} saniye")
        
        # Çıktıyı kaydetme
        output_path = CogPath(f"/tmp/output_{uuid.uuid4().hex}.wav")
        if output_path.exists():
            os.remove(output_path)
            
        # MOSS-TTS varsayılan örnekleme hızı 24000 Hz'dir
        sf.write(str(output_path), waveform, 24000)
        
        return output_path

    def __del__(self):
        # Kaynakları temizle
        if hasattr(self, 'pipeline'):
            self.pipeline.__exit__(None, None, None)
