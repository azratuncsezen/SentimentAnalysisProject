# SentimentAnalysisProject

İngilizce ve Türkçe yorumlar (3000 + 3000) üzerinde üç duygu analizi yöntemini (VADER, BERT, LLM) karşılaştıran veri madenciliği projesi.

## Kurulum

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

API anahtarı gereken işler için `.env.example` dosyasını `.env` olarak kopyalayıp kendi anahtarınızı yazın (`.env` GitHub'a yüklenmez).

## Çalıştırma

Tüm scriptler repo kök klasöründen çalıştırılır:

```bash
python vader/vader_analysis.py          # VADER sonuçları -> results/vader_*.csv
python veri_hazirlama/veri_kontrol.py   # veri setlerini inceler
python veri_hazirlama/veri_hazirla.py   # DİKKAT: data/ dosyalarını yeniden üretir, normalde çalıştırmayın
```

`bert/` ve `llm/` klasörlerinin kodları ilgili kişiler tarafından eklenecektir.

## Plan

Klasör yapısı, ortak kurallar, görev dağılımı ve takvim için bkz. [PROJE_PLANI.md](PROJE_PLANI.md).
