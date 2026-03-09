# MOSS-TTS

[[Model]](https://huggingface.co/OpenMOSS-Team/MOSS-TTS)
[[Replicate Demo]](https://replicate.com/voiser-ai/moss-tts)
[[Resmi API]](https://studio.mosi.cn/docs/moss-tts)

MOSS-TTS, [OpenMOSS](https://huggingface.co/OpenMOSS-Team) tarafından geliştirilen son teknoloji bir Metinden Sese (Text-to-Speech) modelidir. Bu depo, modelin [Replicate](https://replicate.com) üzerinde [Cog](https://github.com/replicate/cog) kullanılarak bulut tabanlı API olarak çalıştırılmasını sağlar.


## Temel Özellikler

- **Sıfır-atışlı ses klonlama** — 3–30 saniyelik bir referans sesten herhangi bir sesi klonlayın, ince ayar gerekmez.
- **Çok dilli sentez** — Türkçe, İngilizce, Çince, Almanca, Fransızca, İspanyolca, Japonca, Korece dahil 20+ dil desteği.
- **Dil değiştirme (Code-switching)** — Tek bir cümle içinde doğal çok dilli üretim (örn. Türkçe–İngilizce karışık).
- **Süre kontrolü** — `expected_duration_sec` parametresi ile konuşma hızını ve süresini kontrol edin.
- **Telaffuz kontrolü** — Pinyin, IPA ve karışık girdiler ile fonem düzeyinde hassas telaffuz kontrolü.
- **Ultra uzun üretim** — Tek seferde 1 saate kadar kesintisiz konuşma üretin.


## Mimari

Bu deployment **MossTTSDelay-8B** modelini kullanır. Gecikme zamanlamalı (delay scheduling) çoklu kafa VQ tahmini yapan tek bir Transformer omurgasına sahiptir. Model `bfloat16` hassasiyetinde NVIDIA L40S GPU üzerinde çalışır.

| Bileşen | Detay |
|:-------:|:------|
| Model | MossTTSDelay-8B |
| Parametre Sayısı | ~8 Milyar |
| Ağırlık Boyutu | ~16 GB (MOSS-TTS) + ~7 GB (Audio Tokenizer) |
| Çıktı Formatı | 24 kHz WAV |
| GPU | NVIDIA L40S |
| Hassasiyet | bfloat16 |


## Kullanım

### Replicate Playground

Modeli kullanmanın en kolay yolu [Replicate Playground](https://replicate.com/voiser-ai/moss-tts) arayüzüdür:

1. Referans ses yükleyin (isteğe bağlı, ses klonlama için)
2. Metninizi girin
3. **Run** butonuna tıklayın

### Python

```python
import replicate

output = replicate.run(
    "voiser-ai/moss-tts",
    input={
        "text": "Merhaba, bugün hava çok güzel.",
        "audio_temperature": 1.7,
        "audio_top_p": 0.8,
        "audio_top_k": 25,
    }
)

# Ses dosyasını kaydet
with open("output.wav", "wb") as f:
    f.write(output.read())
```

### Ses Klonlama

```python
import replicate

output = replicate.run(
    "voiser-ai/moss-tts",
    input={
        "text": "Hello, the weather is great today.",
        "reference_audio": open("referans.wav", "rb"),
        "audio_temperature": 1.5,  # İngilizce için daha düşük
    }
)
```

### Süre Kontrolü

```python
output = replicate.run(
    "voiser-ai/moss-tts",
    input={
        "text": "Bu cümle yaklaşık beş saniye sürmeli.",
        "expected_duration_sec": 5.0,
    }
)
```

### cURL

```bash
curl -s -X POST "https://api.replicate.com/v1/predictions" \
  -H "Authorization: Bearer $REPLICATE_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "version": "<model_version_id>",
    "input": {
      "text": "Merhaba, bugün hava çok güzel.",
      "audio_temperature": 1.7,
      "audio_top_p": 0.8,
      "audio_top_k": 25
    }
  }'
```


## API Parametreleri

### Girdiler

| Parametre | Tip | Varsayılan | Açıklama |
|:----------|:---:|:----------:|:---------|
| `reference_audio` | Dosya | — | Ses klonlama için referans ses dosyası (WAV/MP3). 3–30 saniye arası temiz konuşma önerilir. |
| `text` | String | `"Merhaba, bugün hava çok güzel."` | Sese çevrilecek metin. Çok dilli, Pinyin ve IPA girdilerini destekler. |
| `audio_temperature` | Float | `1.7` | Ses üretimi sıcaklığı. Yüksek → daha ifadeli ama daha az kararlı. Çince: `1.7`, İngilizce: `1.5` önerilir. |
| `audio_top_p` | Float | `0.8` | Nucleus sampling eşiği. Düşük → daha tutarlı çıktı. |
| `audio_top_k` | Int | `25` | Top-K sampling. Her adımda değerlendirilen aday token sayısını sınırlar. |
| `max_new_tokens` | Int | `512` | Üretilecek maksimum token sayısı. Kural: **1 saniye ≈ 12.5 token**. |
| `expected_duration_sec` | Float | `0` | Beklenen çıktı süresi (saniye). `0` = otomatik. En iyi sonuç: doğal okuma süresinin 0.5×–1.5× aralığı. |
| `audio_repetition_penalty` | Float | `1.0` | Tekrarlayan ses kalıplarını cezalandırır. Döngüsel ses varsa artırın. |
| `text_temperature` | Float | `1.5` | **[Deneysel]** Metin kafası sıcaklığı. Model geliştiricilerin optimize ettiği varsayılan değerdir. |
| `text_top_p` | Float | `1.0` | **[Deneysel]** Metin tokenları için Nucleus sampling. |
| `text_top_k` | Int | `50` | **[Deneysel]** Metin tokenları için Top-K sampling. |

> **Not:** `text_*` parametreleri modelin dahili metin üretim kafasını kontrol eder ve model geliştiricilerinin önerdiği varsayılan değerlere ayarlanmıştır. Üretim davranışını deneysel olarak incelemiyorsanız değiştirmeniz önerilmez.

### Çıktı

| Parametre | Tip | Açıklama |
|:----------|:---:|:---------|
| `output` | Dosya | 24 kHz WAV formatında üretilmiş ses dosyası. |


## Nasıl Çalışır?

```
GitHub Push → GitHub Actions → Cog Build → Replicate
                                              ↓
                              İlk API İsteği (Cold Boot)
                                              ↓
                         HuggingFace'ten ağırlıkları indir (~23 GB)
                             pget ile paralel indirme (~1–3 dk)
                                              ↓
                              Modeli GPU belleğine yükle
                                              ↓
                                   Tahminlere hazır
```

1. **Derleme aşaması** — Yalnızca Python kodu ve bağımlılıklar Docker imajına paketlenir (~3 GB). Model ağırlıkları imaja **gömülmez**.
2. **Soğuk başlatma (Cold Boot)** — İlk istekte, model ağırlıkları (~16 GB MOSS-TTS + ~7 GB Audio Tokenizer) HuggingFace'ten [`pget`](https://github.com/replicate/pget) ile paralel olarak yüksek hızda indirilir.
3. **Sıcak tahminler** — Model GPU belleğine yüklendikten sonra, sonraki istekler saniyeler içinde yanıtlanır.


## Proje Yapısı

```
moss-tts-replicate/
├── predict.py                                # Cog predictor: indirme, kurulum, çıkarım
├── cog.yaml                                  # Cog ayarları: GPU, CUDA, Python, bağımlılıklar
├── .github/workflows/push_to_replicate.yaml  # CI/CD: main'e push'ta otomatik deploy
├── .gitignore                                # Build artifaktları ve model cache'i hariç tutar
└── README.md                                 # Bu dosya
```


## Yerel Geliştirme

### Gereksinimler

- [Docker](https://www.docker.com/)
- [Cog](https://github.com/replicate/cog)
- CUDA destekli NVIDIA GPU

### Yerelde Çalıştırma

```bash
git clone https://github.com/voiser-org/moss-tts-replicate.git
cd moss-tts-replicate
cog predict -i text="Merhaba, bu bir test."
```

### Replicate'a Push

```bash
cog login
cog push r8.im/voiser-ai/moss-tts
```


## CI/CD

Bu depo, sürekli dağıtım için GitHub Actions kullanır. `main` dalına yapılan her push, modeli otomatik olarak derleyip Replicate'a yükler.

### Kurulum

GitHub deponuza aşağıdaki secret'ı ekleyin (`Settings → Secrets → Actions`):

| Secret | Açıklama |
|:-------|:---------|
| `REPLICATE_API_TOKEN` | [Replicate API token](https://replicate.com/account/api-tokens) sayfanızdan alınır. |


## Önerilen Parametreler

Model geliştiricilerinin farklı diller için önerdiği varsayılan değerler:

| Dil | `audio_temperature` | `audio_top_p` | `audio_top_k` |
|:----|:-------------------:|:-------------:|:--------------:|
| Çince | `1.7` | `0.8` | `25` |
| İngilizce | `1.5` | `0.8` | `50` |

> **İpucu:** Varsayılan değerler çoğu senaryo için iyi çalışır. Önce varsayılanlarla başlayın, ses çeşitliliğini artırmak veya azaltmak istiyorsanız `audio_temperature` değerini ayarlayın.


## Lisans

Bu proje, [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) altında yayınlanan [OpenMOSS-Team/MOSS-TTS](https://huggingface.co/OpenMOSS-Team/MOSS-TTS) modelini kullanmaktadır. Tam lisans detayları için [HuggingFace model kartını](https://huggingface.co/OpenMOSS-Team/MOSS-TTS) ziyaret edin.
