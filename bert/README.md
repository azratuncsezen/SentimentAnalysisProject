# BERT

Bu klasör **Berkay**'a aittir. BERT tabanlı duygu analizi kodları buraya konur; başkası bu klasöre yazmaz. Beklenen çıktılar `results/bert_english.csv` ve `results/bert_turkish.csv` dosyalarıdır. Zorunlu kolon formatı: `id, text, label, prediction, correct` (`label` ve `prediction` yalnızca `positive`, `negative`, `neutral` olabilir; kodlama `utf-8-sig`). Model checkpoint'leri (`bert/checkpoints/`, `bert/models/`) GitHub'a yüklenmez.
