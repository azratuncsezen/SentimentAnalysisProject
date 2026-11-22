# LLM

Bu klasör **Furkan**'a aittir. LLM (Claude API) tabanlı duygu analizi kodları buraya konur; başkası bu klasöre yazmaz. Beklenen çıktılar `results/llm_english.csv` ve `results/llm_turkish.csv` dosyalarıdır. Zorunlu kolon formatı: `id, text, label, prediction, correct` (`label` ve `prediction` yalnızca `positive`, `negative`, `neutral` olabilir; kodlama `utf-8-sig`). API anahtarı yalnızca yerel `.env` dosyasında durur (şablon: `.env.example`), asla GitHub'a yüklenmez.
