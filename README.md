# SentimentAnalysisProject

İngilizce ve Türkçe metinlerde VADER, BERT ailesi ve LLM tabanlı duygu analizinin karşılaştırılması. Her dilde 1.000 negatif, 1.000 nötr ve 1.000 pozitif yorum kullanılmıştır. Aynı dilde tüm yöntemler aynı test kümesinde değerlendirilir.

## Ekip

- Azra Tunçsezen — 202403011048: veri hazırlığı, VADER ve karşılaştırma
- Berkay Şahin — 202403011072: RoBERTa ve BERTurk
- Furkan Başer — 202503011045: LLM

## Kurulum

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
python -m pip install -r requirements.txt
```

Canlı LLM/çeviri istekleri için `.env.example` dosyasını `.env` olarak kopyalayarak kendi API anahtarınızı ekleyin. `.env` dosyasını paylaşmayın. Kayıtlı LLM yanıtlarını `--offline` ile değerlendirmek API anahtarı veya yeni istek gerektirmez.

## Veriler ve yöntemler

- İngilizce: `frasem/sentiment-dataset`, TweetEval kökenli test verisi.
- Türkçe: `maydogan/Turkish_SentimentAnalysis_TRSAv1` içinden dengeli test örneklemi.
- VADER: ham İngilizce ve Türkçe metinlerde kural tabanlı sınıflandırma.
- İngilizce BERT ailesi: `cardiffnlp/twitter-roberta-base-sentiment`; proje içinde ek eğitim yok.
- Türkçe BERT: `dbmdz/bert-base-turkish-cased`; 27.000 eğitim / 3.000 doğrulama, ayrı 3.000 test. Birebir metin çakışması yok.
- LLM: `claude-sonnet-5-5`, zero-shot; tam çalışma kaydında temperature gönderilmemiştir.
- Ek deney: Haiku ile Türkçe→İngilizce çeviri, 25 elle düzeltme ve VADER.

## Çalıştırma

Komutları depo kökünden çalıştırın:

```bash
python veri_hazirlama/veri_kontrol.py
python vader/vader_analysis.py
python llm/llm_analysis.py --model sonnet --offline --no-temperature
python llm/translate.py --model haiku --offline
python vader/clean_translations.py
python vader/vader_turkish_translated.py
python bert/bert_english.py
python evaluate.py
```

İngilizce model ilk çalışmada indirilir. `evaluate.py`, yedi kayıtlı tahmin dosyasının kimlik/metin/etiket tutarlılığını denetler; sınıf bazlı hata, doğruluk, makro ölçütler ve karışıklık matrislerini `results/evaluation/` içine üretir. Model eğitmez veya API çağırmaz.

Türkçe modelin tam eğitimden kaydedilmiş dosyaları depoda bulunmaz. Paylaşılan model klasörünü `bert/models/berturk-trsa/` konumuna yerleştirip çalıştırın:

```bash
python bert/bert_turkish_predict.py
```

Model yoksa eğitim yeniden yapılabilir; farklı ortamda sonuçların birebir aynı olması garanti edilmez:

```bash
python bert/bert_turkish_train.py --smoke  # yalnızca kısa çalışma denemesi
python bert/bert_turkish_train.py          # tam eğitim
python bert/bert_turkish_predict.py
```

`veri_hazirlama/veri_hazirla.py`, ortak test dosyalarını yeniden yazar; kayıtlı deneyleri değerlendirmek için çalıştırılması gerekmez.

## Kayıtlı sonuçlar

Hatalar gerçek sınıfa göre sayılır; her sınıfın destek sayısı 1.000'dir.

| Yöntem / dil | Negatif hata | Nötr hata | Pozitif hata | Doğruluk |
|---|---:|---:|---:|---:|
| VADER / İngilizce | 452 | 572 | 285 | %56,37 |
| RoBERTa / İngilizce | 219 | 300 | 294 | %72,90 |
| LLM / İngilizce | 190 | 355 | 327 | %70,93 |
| VADER / Türkçe | 980 | 50 | 890 | %36,00 |
| BERTurk / Türkçe | 161 | 266 | 111 | %82,07 |
| LLM / Türkçe | 82 | 633 | 54 | %74,37 |
| Çeviri + VADER / Türkçe | 439 | 833 | 79 | %54,97 |

## Sınırlılıklar

LLM model seçimi için kullanılan 50'şer yorum nihai test kümesindendir. İngilizcede 13, Türkçede 45 geçersiz LLM yanıtı nötr sayılmıştır. Bu 58 yanıtın hepsinde çıktı token sayısı 50 sınırına ulaşır; `stop_reason` saklanmadığı için bitiş nedeni doğrudan doğrulanamaz. Türkçe ürün yorumları ve İngilizce tweetler farklı alanlardadır. Bağımlılıkların kesin sürümleri ve kaynak revision değerleri sabitlenmemiştir.

Ayrıntılar: [BERT](bert/README.md), [LLM](llm/README.md), [proje planı](PROJE_PLANI.md).
