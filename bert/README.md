# BERT

Bu klasör **Berkay**'a aittir. BERT tabanlı duygu analizi kodları buraya konur; başkası bu klasöre yazmaz. Beklenen çıktılar `results/bert_english.csv` ve `results/bert_turkish.csv` dosyalarıdır. Zorunlu kolon formatı: `id, text, label, prediction, correct` (`label` ve `prediction` yalnızca `positive`, `negative`, `neutral` olabilir; kodlama `utf-8-sig`). Model checkpoint'leri (`bert/checkpoints/`, `bert/models/`) GitHub'a yüklenmez.

## Kullanılan modeller
- **İngilizce:** `cardiffnlp/twitter-roberta-base-sentiment` — hazır model, ek eğitim yok. TweetEval *train* ile eğitildiği için test örneklemimizle (TweetEval *test*) çakışma yoktur. Etiketler: LABEL_0=negative, LABEL_1=neutral, LABEL_2=positive (model kartına göre).
- **Türkçe:** `dbmdz/bert-base-turkish-cased` (BERTurk), TRSAv1 üzerinde fine-tune edildi.

## Ön işleme
- İngilizce: model kartındaki tweet ön işlemesi (`@kullanıcı` → `@user`, bağlantılar → `http`); max_length 128, batch 32.
- Türkçe: `veri_hazirla.py` ile aynı temizlik (boş/NaN at, strip, küçük harf etiket, tekrar eden metinleri at).

## Sızıntı kontrolü (Türkçe)
`data/turkish.csv` içindeki 3000 yorumun tamamı TRSAv1'den çıkarıldı (140.954 → 137.954 satır); kalan çakışma `assert` ile 0 olarak doğrulanır. Her sınıftan 10.000 yorum örneklendi (30.000), %10'u (3000) eğitim havuzundan stratified doğrulama için ayrıldı: 27.000 eğitim / 3.000 doğrulama.

## Hiperparametreler (Türkçe)
max_length 128, batch 16, lr 2e-5, 2 epoch, weight_decay 0.01, warmup %6, AdamW (Trainer varsayılanı), seed 42, her epoch değerlendirme, en iyi model doğrulama macro-F1'e göre.

## Donanım ve süre
Apple M4, 16 GB, `mps` cihazı. Eğitim süresi 16,7 dk. Doğrulama: epoch 1 acc 0,8153 / macro-F1 0,8122; epoch 2 acc 0,8233 / macro-F1 0,8221 (ayrıntı: `train_log.txt`).

## Çalıştırma (repo kökünden)
```bash
.venv/bin/python bert/bert_english.py
.venv/bin/python bert/bert_turkish_train.py     # model bert/models/berturk-trsa/ içine kaydedilir (GitHub'a yüklenmez)
.venv/bin/python bert/bert_turkish_predict.py
```

## Sonuçlar (3000 test yorumu / dil)
| Dil | Doğruluk | Yanlış: negative | Yanlış: neutral | Yanlış: positive |
|---|---|---|---|---|
| İngilizce | %72,90 | 219 / 1000 | 300 / 1000 | 294 / 1000 |
| Türkçe | %82,07 | 161 / 1000 | 266 / 1000 | 111 / 1000 |
