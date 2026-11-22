"""
translate.py — Türkçe yorumları Claude ile İngilizceye çevirir (Azra'nın
"çeviri sonrası VADER" deneyi için, PROJE_PLANI.md karar 3).

Çıktı:
    llm/turkish_translated.csv  (kolonlar: id, text, text_en, label — utf-8-sig)
    Azra bu dosyadaki text_en kolonuna VADER uygular.

Yöntem:
    Çevirmen talimatı, temperature = 0. Model yorumun duygusunu
    yorumlamaz/etiketlemez; sadece anlamı ve tonu koruyarak çevirir
    (yazım hataları, argo, ironi ve emojiler korunur).

Kullanım (repo kök klasöründen):
    python llm/translate.py --sample 20 --model haiku   # deneme, çeviriye göz at
    python llm/translate.py --model haiku --batch       # 3000 yorum, Batch API
    python llm/translate.py --model haiku --offline     # önbellekten CSV'yi yeniden yaz

Önbellek: llm/raw/translate_<model>.jsonl (yarıda kalırsa kaldığı yerden devam)
"""

import argparse
import sys

import pandas as pd

from claude_utils import (
    DATA, LLM_DIR, build_params, estimate_cost, load_cache, make_client,
    resolve_model, response_text, run_batch, run_realtime,
)

SEED = 42
MAX_TOKENS = 1024

SYSTEM_PROMPT = """You are a professional Turkish-to-English translator.

Translate the Turkish e-commerce product review given by the user into natural English.
- Preserve the meaning, tone and intensity exactly, including sarcasm, slang, complaints and praise.
- Keep emojis and numbers as they are. Do not correct the author's opinion or add anything.
- If the text is already in English or cannot be translated, return it unchanged.

Output only the English translation, with no quotes, notes or explanations."""

USER_TEMPLATE = "{text}"


def make_record(item_id, msg, batch):
    return {
        "id": int(item_id),
        "batch": batch,
        "text_en": response_text(msg),
        "input_tokens": getattr(msg.usage, "input_tokens", 0),
        "output_tokens": getattr(msg.usage, "output_tokens", 0),
    }


def main():
    ap = argparse.ArgumentParser(description="Türkçe yorumları İngilizceye çevir")
    ap.add_argument("--model", default="haiku")
    ap.add_argument("--sample", type=int, default=0, help="Deneme için yorum sayısı")
    ap.add_argument("--batch", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--no-temperature", action="store_true")
    args = ap.parse_args()

    model = resolve_model(args.model)
    temperature = None if args.no_temperature else 0
    client = None if args.offline else make_client()

    df = pd.read_csv(DATA / "turkish.csv", encoding="utf-8-sig")
    if args.sample:
        df = df.sample(n=args.sample, random_state=SEED).sort_values("id")

    cache_path = LLM_DIR / "raw" / f"translate_{model}.jsonl"
    cache = load_cache(cache_path)
    todo = [(int(r.id), str(r.text)) for r in df.itertuples() if int(r.id) not in cache]
    print(f"model={model}  yorum={len(df)}  önbellekte={len(df) - len(todo)}  gönderilecek={len(todo)}")

    if todo and args.offline:
        sys.exit(f"--offline: {len(todo)} yorumun çevirisi önbellekte yok.")

    def make_params(text):
        return build_params(model, SYSTEM_PROMPT, USER_TEMPLATE.format(text=text),
                            MAX_TOKENS, temperature)

    if todo:
        if args.batch:
            run_batch(client, todo, make_params,
                      lambda i, t, msg, err: make_record(i, msg, True),
                      cache_path, LLM_DIR / "raw" / f"translate_{model}_batch.json")
        else:
            run_realtime(client, todo, make_params,
                         lambda i, t, msg: make_record(i, msg, False),
                         cache_path, workers=args.workers)
        cache = load_cache(cache_path)

    missing = [i for i in df["id"] if int(i) not in cache]
    if missing:
        sys.exit(f"! {len(missing)} yorumun çevirisi eksik, scripti tekrar çalıştır.")

    recs = [cache[int(i)] for i in df["id"]]
    df = df.copy()
    df["text_en"] = [r["text_en"] for r in recs]
    in_tok = sum(r["input_tokens"] for r in recs)
    out_tok = sum(r["output_tokens"] for r in recs)
    empty = int((df["text_en"].str.strip() == "").sum())

    name = f"turkish_translated_sample{len(df)}.csv" if args.sample else "turkish_translated.csv"
    path = LLM_DIR / name
    df[["id", "text", "text_en", "label"]].to_csv(path, index=False, encoding="utf-8-sig")

    print(f"\nKaydedildi: llm/{name}  ({len(df)} satır, boş çeviri: {empty})")
    print(f"Token: giriş={in_tok}  çıkış={out_tok}")
    costs = [estimate_cost(model, r["input_tokens"], r["output_tokens"], bool(r.get("batch"))) for r in recs]
    cost = None if None in costs else sum(costs)
    if cost is not None:
        print(f"Tahmini maliyet: ${cost:.4f}")
    print("\nİlk 5 çeviri:")
    for r in df.head(5).itertuples():
        print(f"  TR: {r.text[:100]}\n  EN: {r.text_en[:100]}\n")


if __name__ == "__main__":
    main()
