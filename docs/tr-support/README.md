# Turkish Text Normalization

Bu katman, MOSS-TTS modeline ham rakam göndermek yerine desteklenen Türkçe tam sayıları konuşulabilir metne çevirir.

## Amaç

- Düz tam sayıları Türkçe yazıyla modele vermek
- Sayıların rakam rakam okunmasını azaltmak

## Etkinleştirme

`text_normalization_language="tr"` seçildiğinde Türkçe sayı normalizasyonu aktif olur.

Örnek:

```json
{
  "text": "Bugün 2024 kişi geldi.",
  "text_normalization_language": "tr"
}
```

Çıktı metni modele yaklaşık olarak şu şekilde gider:

```text
Bugün iki bin yirmi dört kişi geldi.
```

## Desteklenen Yapılar

- Düz tam sayılar
- Saf rakamdan oluşan standalone tokenlar

Örnekler:

- `0 -> sıfır`
- `5 -> beş`
- `21 -> yirmi bir`
- `105 -> yüz beş`
- `2024 -> iki bin yirmi dört`
- `1000000 -> bir milyon`

## Desteklenmeyen Yapılar

- Saatler
- Ondalık sayılar
- Grup ayraçlı sayılar
- İşaretli sayılar
- Sıra sayıları
- Tarih
- Para
- Yüzde
- Kesir
- Alfanümerik kodlar

Örnekler:

- `3,14`
- `1.000`
- `-42`
- `1.`
- `12/05/2026`
- `₺150`
- `%25`
- `1/2`
- `A320`

## Güvenli Kullanım

- Metinde yalnızca düz tam sayı varsa bu katmanı aç
- Saat, para veya tarih gibi kalıplarda dönüşüm bekleme
- Grup ayraçlı sayıları (`1.000`, `1,000`) olduğu gibi bırak

## Kapsam Özeti

Şu an Türkçe normalizasyon katmanı yalnızca düz tam sayıları yazıya çevirir. Saat özelliği projeden kaldırılmıştır.
