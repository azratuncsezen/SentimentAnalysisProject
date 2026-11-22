# Proje Planı — Karşılaştırmalı Duygu Analizi

**Ders:** BİM 448 Veri Madenciliği ve Bilgi Keşfi (Doç. Dr. Aysun Güran)
**Ekip:** Berkay, Azra, Furkan
**Teslim:** 25 Kasım 2026 Çarşamba, ders saatinde — elden (imza karşılığı) **ve** e-posta ile

---

## 1. Ödevin istedikleri

Üç yöntemi, İngilizce ve Türkçe veri setlerinde ayrı ayrı uygulayıp karşılaştırmak:

1. **VADER** — kural/sözlük tabanlı
2. **BERT tabanlı** — İngilizce için fine-tuned BERT/RoBERTa, Türkçe için BERTurk
3. **LLM tabanlı** — prompting / API ile tahmin

**Zorunlu analiz:** Her yöntem × her dil × her sınıf (Pozitif, Negatif, Nötr) için **yanlış tahmin sayıları** tek tek verilmeli.

**Teslim edilecekler:**

| Materyal | İçerik |
|---|---|
| Word raporu (.docx) | Veri setlerinin kaynak ve detayları; 3 yöntemin açıklaması, mimari ve parametreleri; sonuçlar ve karşılaştırma |
| Kaynak kod (.py / notebook) | Ön işleme, model çalıştırma, değerlendirme — temiz ve çalıştırılabilir |
| Sunum videosu | En fazla 10 dakika, **üç kişinin de aktif yer aldığı** |

E-posta eki sınırı aşılırsa her şey bir OneDrive klasörüne yüklenip link e-postaya eklenir.

**Değerlendirme kriterleri:** veri setlerinin uygunluğu, 3 yöntemin eksiksiz uygulanması, sınıf bazlı hata analizi ve akademik yorum, rapor/sunum kalitesi, zamanında teslim.

---

## 2. Veri setleri

Her iki set de her sınıftan 1000 örnek olmak üzere **3000'er yorum** içerir (`data/` klasörü, `veri_hazirlama/veri_hazirla.py` ile `random_state=42` kullanılarak örneklendi). Tekrarlanan yorum yok.

### İngilizce — `frasem/sentiment-dataset`
- **Gerçek kaynak:** TweetEval'in *sentiment* görevi = SemEval-2017 Task 4 (Twitter).
  Raporda atıf: Rosenthal, Farra, Nakov (2017) *SemEval-2017 Task 4: Sentiment Analysis in Twitter*; Barbieri vd. (2020) *TweetEval*, Findings of EMNLP.
- **Bizim örneklem:** veri setinin **test** bölümünden alındı.
- **Alan:** tweetler, çoğunlukla siyaset/gündem; ortalama ~14 kelime.
- **Dikkat:** `@user`, link ve `&amp;` gibi temizlenmemiş kalıntılar var; "nötr" sınıfı tweetlerde doğası gereği belirsiz.

### Türkçe — `maydogan/Turkish_SentimentAnalysis_TRSAv1`
- **Kaynak:** Aydoğan M., Kocaman V. (2022) *TRSAv1: A new benchmark dataset for classifying user reviews on Turkish e-commerce websites*, Journal of Information Science. doi:10.1177/01655515221074328
- **Boyut:** 150.000 e-ticaret ürün yorumu (her sınıftan 50.000), tek `train` bölümü.
- **Dikkat — etiket gürültüsü:** Etiketlerin nasıl üretildiği veri seti sayfasında yazmıyor. Bazı örnekler etiketin yıldız puanından geldiğini düşündürüyor (ör. "güzel bir ürün seviyorum" → nötr, "Fena değil" → negatif). **Doğrulanmadı**; makaleden kontrol edilmeli ve 100 nötr yorumda elle kontrol yapılmalı.

### Raporda tartışılacak sınırlılık
İngilizce veri **tweet/siyaset**, Türkçe veri **ürün yorumu**. Diller arasındaki başarı farkının bir kısmı dilden değil **alan (domain) farkından** kaynaklanabilir.

---

## 3. Verilen kararlar

| # | Karar |
|---|---|
| 1 | Bu repo üzerinde çalışılır; herkes kendi branch'inde çalışır, `main`'e Pull Request ile birleştirilir. |
| 2 | Veri setleri değiştirilmez; zayıf yanları raporda "sınırlılıklar ve hata analizi" bölümünde tartışılır. |
| 3 | **VADER (Türkçe):** mevcut ham sonuç korunur (%36 doğruluk — VADER Türkçe bilmediği için çoğunu nötr sayıyor). Ek olarak Türkçe yorumlar İngilizceye çevrilip VADER tekrar uygulanır; ikisi karşılaştırılır. |
| 4 | **BERT:** İngilizce için hazır `cardiffnlp/twitter-roberta-base-sentiment` (TweetEval train ile eğitilmiş → bizim test örneklemimizi görmemiş). Türkçe için **BERTurk (`dbmdz/bert-base-turkish-cased`) fine-tuning**: TRSAv1'in `data/turkish.csv` dışında kalan yorumlarıyla eğitilir. **Test yorumları eğitime kesinlikle karışmaz.** Sabit seed kullanılır. |
| 5 | **LLM:** Claude API. Model (Opus / Sonnet / Haiku) 50 yorumluk deneme sonrası seçilecek. Tahminler bir kez alınıp dosyaya kaydedilir; rapordaki sayılar o dosyadan gelir. |

---

## 4. Görev dağılımı

| | **Berkay** | **Azra** | **Furkan** |
|---|---|---|---|
| **Yöntem** | BERT | VADER (+ çeviri sonrası VADER) | LLM (Claude) |
| **Klasör** | `bert/` | `vader/` | `llm/` |
| **Çıktılar** | `results/bert_english.csv`, `results/bert_turkish.csv` | `results/vader_english.csv`, `results/vader_turkish.csv` (+ `results/vader_turkish_translated.csv`) | `results/llm_english.csv`, `results/llm_turkish.csv` |
| **Ek iş** | API anahtarlarını ayarlar; GitHub birleştirmeleri | Veri seti bölümü; 100 nötr Türkçe yorumda etiket kontrolü | Prompt tasarımı, 50 yorumluk deneme, maliyet ölçümü, model önerisi |
| **Rapordaki bölümü** | BERT: mimari, eğitim parametreleri | Giriş, veri setleri, VADER | LLM: model, prompt, parametreler, maliyet |
| **Videodaki kısmı (~3 dk)** | BERT | Amaç, veri setleri, VADER | LLM |

**Ortak (en son, birlikte):** karşılaştırma notebook'u (`compare.ipynb`), Sonuçlar–Tartışma–Sonuç bölümleri, Word raporunun birleştirilmesi, slaytlar, video kaydı.

**Açık konu:** Türkçe çeviri API gerektiriyor. Furkan LLM koduyla çevirip dosyayı Azra'ya verebilir ya da Azra bu kısmı atlayıp sadece ham VADER'i tartışır.

---

## 5. Klasör yapısı

```
SentimentAnalysisProject/
├── data/                  ← test verisi — KİMSE DEĞİŞTİRMEZ
│   ├── english.csv
│   └── turkish.csv
├── veri_hazirlama/        ← veri setlerini indirip örnekleyen kodlar
│   ├── veri_hazirla.py
│   └── veri_kontrol.py
├── vader/                 ← Azra
│   └── vader_analysis.py
├── bert/                  ← Berkay
├── llm/                   ← Furkan
├── results/               ← herkesin tahmin dosyaları (ortak format)
├── compare.ipynb          ← en son: 3 yöntem × 2 dil karşılaştırması
├── requirements.txt       ← gerekli Python paketleri
├── .env.example           ← API anahtarı şablonu (gerçek anahtar .env'de)
├── .gitignore
├── README.md
└── PROJE_PLANI.md         ← bu dosya
```

Tüm scriptler **repo kök klasöründen** çalıştırılır, örn. `python vader/vader_analysis.py`.

---

## 6. Ortak kurallar

1. **Çıktı formatı (zorunlu):** her tahmin dosyası `results/<yontem>_<dil>.csv` adında ve şu kolonlarla:
   `id, text, label, prediction, correct`
   — `label` ve `prediction` değerleri yalnızca `positive`, `negative`, `neutral` olabilir. Kodlama `utf-8-sig`. (Azra'nın VADER dosyaları zaten bu formatta.)
2. **`data/` dosyalarına kimse dokunmaz.** Tüm yöntemler aynı 3000 + 3000 yorumla değerlendirilir.
3. **Kimse başkasının klasörüne yazmaz** → birleştirmede çakışma (conflict) çıkmaz.
4. **API anahtarı asla GitHub'a yüklenmez.** Sadece kendi bilgisayarındaki `.env` dosyasında durur (`.gitignore`'da). Şablon: `.env.example`.
5. Rastgelelik içeren her işte **sabit seed (42)** kullanılır.
6. Kodun başına kısa bir açıklama yazılır: ne yapıyor, nasıl çalıştırılır, hangi parametreler kullanıldı. Videoda ve soru-cevapta "bunu nasıl yaptınız" sorusuna herkes kendi kısmını anlatabilmeli.

---

## 7. Git akışı

```bash
git checkout main && git pull
git checkout -b berkay-bert        # azra-vader / furkan-llm
# ... çalış, kaydet ...
git add bert/ results/bert_*.csv
git commit -m "BERT: İngilizce RoBERTa sonuçları"
git push -u origin berkay-bert
```

Sonra GitHub'da **Pull Request** açılır, ekipten biri hızlıca bakar, `main`'e birleştirilir. `compare.ipynb` üç PR da birleştikten sonra `main` üzerinde hazırlanır.

---

## 8. API anahtarı

- Anahtar yalnızca **LLM** ve **çeviri** için gerekir; VADER ve BERT anahtar kullanmaz.
- Herkesin aynı makinede ya da aynı anahtarla çalışması **gerekmez** — anahtar sadece kimin hesabından ödeneceğini belirler, sonucu etkilemez.
- Berkay, Furkan için Anthropic Console'da **ayrı ve harcama limitli** bir anahtar oluşturur ve güvenli yoldan iletir (GitHub'a veya açık grup sohbetine yazılmaz). Proje bitince iptal edilir.
- Kullanım: proje klasöründe `.env` dosyası oluşturup içine `ANTHROPIC_API_KEY=...` yazmak.

**Tahmini maliyet** (6000 sınıflandırma + 3000 çeviri; kaba tahmin, ölçülmedi):

| Model | Normal | Batch API (%50 indirimli) |
|---|---|---|
| Claude Opus 5.5 | ~30 $ | ~15 $ |
| Claude Sonnet 5.5 | ~15 $ | ~7–8 $ |
| Claude Haiku 5.5 | ~1 $ | ~0,5 $ |

Tüm veri çalıştırılmadan önce 50 yorumluk deneme ile gerçek maliyet ölçülür.

---

## 9. Takvim

| Gün | Berkay | Azra | Furkan |
|---|---|---|---|
| **Pazar 22 Kasım** | Ortam kurulumu, İngilizce RoBERTa, Türkçe eğitimi başlatmak; Furkan'ın anahtarını oluşturmak | Veri seti bölümünü yazmak | Anahtarı almak, prompt tasarımı, 50 yorumluk deneme |
| **Pazartesi 23 Kasım** | Türkçe BERT sonuçları → PR | Çeviri + VADER, etiket kontrolü → PR | 6000 yorumu sınıflandırmak (+ çeviri) → PR |
| **Salı 24 Kasım** | Sabah: `compare.ipynb` ve rapor bölümleri · Akşam: **hep birlikte** rapor birleştirme, slaytlar, video kaydı | | |
| **Çarşamba 25 Kasım** | Son kontrol, çıktı, e-posta + elden teslim | | |

**Kritik ara hedef:** Pazartesi akşamı üç PR da `main`'de olmalı. Salı yalnızca rapor ve videoya ayrılır.

---

## 10. Karşılaştırma notebook'u (`compare.ipynb`) ne üretecek?

- Doğruluk, precision, recall, F1 tablosu (yöntem × dil)
- **Her yöntem × dil × sınıf için yanlış tahmin sayıları** (ödevin zorunlu kısmı)
- Karışıklık matrisleri (confusion matrix) ve grafikler
- Örnek hatalı tahminler ve yorumları ("hangi yorumu neden yanlış bildi")
- Türkçe VADER: ham vs. çeviri sonrası karşılaştırması

---

## 11. Henüz karar verilmemiş konular

- [ ] Claude modeli (50 yorumluk deneme sonrası)
- [ ] Türkçe çeviriyi kim yapacak (Furkan mı, atlanacak mı)
- [ ] Word raporunu kim birleştirecek
- [ ] Videoda konuşma sırası ve kayıt zamanı
