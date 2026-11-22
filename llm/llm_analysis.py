"""
llm_analysis.py — Claude API ile duygu analizi (LLM yöntemi).

Ne yapar:
    data/english.csv ve data/turkish.csv'deki her yorumu Claude'a gönderir,
    yanıtı positive / negative / neutral olarak alır, doğruluk ve sınıf bazlı
    hata sayılarını yazdırır, sonuçları ortak formatta kaydeder:
        results/llm_english.csv, results/llm_turkish.csv
        (kolonlar: id, text, label, prediction, correct — utf-8-sig)

Yöntem:
    Zero-shot prompting. Her iki dil için AYNI talimat kullanılır; örnek
    (few-shot) verilmez, model eğitilmez. temperature = 0 (en tutarlı yanıt).
    Yanıt biçimi: structured outputs (output_config.format = json_schema).
    Modelin cevabı {"sentiment": ...} biçiminde bir JSON olmak zorundadır ve
    sentiment alanı yalnızca positive / negative / neutral değerlerini alabilir
    (enum). Böylece model açıklama yazamaz ve geçersiz yanıt oluşmaz.
    (İlk denemede serbest metin istenince Haiku bazı Türkçe yorumlarda etiket
    yerine açıklama yazmaya başladı; zorunlu araç çağrısı (tool_choice) ise
    Sonnet 5.5'te desteklenmiyor. Bu yüzden bu yönteme geçildi.)

Kullanım (repo kök klasöründen):
    # 1) Deneme: her dilden 50 yorum (dengeli, seed 42), maliyet ölçümü ve tahmin
    python llm/llm_analysis.py --sample 50 --model haiku
    python llm/llm_analysis.py --sample 50 --model sonnet
    python llm/llm_analysis.py --sample 50 --model opus

    # 2) Seçilen modelle tüm veri (Batch API: %50 ucuz, birkaç dakika–saat sürer)
    python llm/llm_analysis.py --model haiku --batch
    #    ya da anlık istekler (daha hızlı başlar, tam fiyat)
    python llm/llm_analysis.py --model haiku

    # 3) API'ye gitmeden önbellekten sonuçları yeniden üret / raporla
    python llm/llm_analysis.py --model haiku --offline

Önbellek:
    Her yanıt llm/raw/<dil>_<model>_json.jsonl dosyasına anında yazılır. Script
    yarıda kesilirse tekrar çalıştırınca sadece eksik yorumlar gönderilir.
    Rapordaki sayılar bu dosyalardan gelir (tahminler bir kez alınır).
    Çalıştırma bilgileri (model, prompt, token, maliyet): llm/run_info_<dil>.json
"""

import argparse
import json
import re
import sys
from datetime import datetime, timezone

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from claude_utils import (
    DATA, LLM_DIR, RESULTS, ROOT, PRICES, estimate_cost, load_cache, make_client,
    resolve_model, response_text, run_batch, run_realtime, build_params,
)

LABELS = ["negative", "neutral", "positive"]
SEED = 42
MAX_TOKENS = 50  # JSON yanıtı ~10 token

METHOD = "json"  # önbellek dosya adına eklenir (önceki denemelerle karışmasın)
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "sentiment": {
            "type": "string",
            "enum": ["positive", "negative", "neutral"],
        }
    },
    "required": ["sentiment"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You are a sentiment classification system.

Classify the overall sentiment expressed by the author of the given text into exactly one of three classes:
- positive: the author expresses a favorable, approving, happy or satisfied attitude.
- negative: the author expresses an unfavorable, critical, angry, sad or dissatisfied attitude.
- neutral: the text has no clear sentiment, is factual or informational, or positive and negative parts balance out.

The text may be an English tweet or a Turkish product review. Read it in its own language. Ignore usernames, links and hashtag symbols unless they carry sentiment.

Answer in the required JSON format: {"sentiment": "positive" | "negative" | "neutral"}."""

USER_TEMPLATE = "<text>\n{text}\n</text>"

# Model yanıtı beklenmedik biçimde gelirse (ör. "Positive.", "Nötr") eşleme
ALIASES = {
    "positive": "positive", "pozitif": "positive", "olumlu": "positive",
    "negative": "negative", "negatif": "negative", "olumsuz": "negative",
    "neutral": "neutral", "nötr": "neutral", "notr": "neutral", "tarafsız": "neutral",
}
FALLBACK = "neutral"  # geçersiz yanıtlar bu sınıfa sayılır ve raporlanır


def parse_label(raw):
    for word in re.findall(r"[a-zçğıöşü]+", raw.lower()):
        if word in ALIASES:
            return ALIASES[word], True
    return FALLBACK, False


def extract_answer(msg):
    """JSON yanıtındaki sentiment değerini, okunamazsa düz metni döndürür."""
    text = response_text(msg)
    try:
        value = json.loads(text).get("sentiment")
        if value:
            return str(value)
    except (ValueError, AttributeError):
        pass
    return text


def make_record(item_id, raw, usage, batch):
    prediction, valid = parse_label(raw)
    return {
        "id": int(item_id),
        "raw": raw,
        "prediction": prediction,
        "valid": valid,
        "batch": batch,
        "input_tokens": getattr(usage, "input_tokens", 0) if usage else 0,
        "output_tokens": getattr(usage, "output_tokens", 0) if usage else 0,
    }


def stratified_sample(df, n):
    per_class = n // len(LABELS)
    extra = n - per_class * len(LABELS)
    parts = []
    for i, label in enumerate(LABELS):
        k = per_class + (1 if i < extra else 0)
        parts.append(df[df["label"] == label].sample(n=k, random_state=SEED))
    return pd.concat(parts).sort_values("id")


def report(df, title):
    print(f"\n{'=' * 50}\n{title}")
    acc = accuracy_score(df["label"], df["prediction"])
    print(f"\nACCURACY: {acc:.4f} ({acc * 100:.2f}%)")
    print("\nCLASSIFICATION REPORT:")
    print(classification_report(df["label"], df["prediction"], labels=LABELS,
                                zero_division=0, digits=4))
    cm = confusion_matrix(df["label"], df["prediction"], labels=LABELS)
    print("CONFUSION MATRIX (satır = gerçek, sütun = tahmin):")
    print(pd.DataFrame(cm, index=LABELS, columns=LABELS))
    print("\nERROR COUNTS BY TRUE CLASS:")
    errors = {}
    for label in LABELS:
        sub = df[df["label"] == label]
        e = int((~sub["correct"]).sum())
        errors[label] = e
        print(f"{label}: {e} / {len(sub)}")
    return acc, errors


def main():
    ap = argparse.ArgumentParser(description="Claude API ile duygu analizi")
    ap.add_argument("--model", default="haiku",
                    help="haiku | sonnet | opus ya da tam model kimliği")
    ap.add_argument("--lang", choices=["english", "turkish", "all"], default="all")
    ap.add_argument("--sample", type=int, default=0,
                    help="Deneme: her dilden bu kadar yorum (dengeli). 0 = tümü")
    ap.add_argument("--batch", action="store_true",
                    help="Batch API kullan (%%50 ucuz, sonuç gecikmeli)")
    ap.add_argument("--workers", type=int, default=4,
                    help="Anlık modda paralel istek sayısı")
    ap.add_argument("--offline", action="store_true",
                    help="API çağırma, sadece önbellekteki yanıtlardan raporla")
    ap.add_argument("--no-temperature", action="store_true",
                    help="temperature parametresini gönderme")
    args = ap.parse_args()

    model = resolve_model(args.model)
    temperature = None if args.no_temperature else 0
    langs = ["english", "turkish"] if args.lang == "all" else [args.lang]
    client = None if args.offline else make_client()

    def make_params(text):
        params = build_params(model, SYSTEM_PROMPT, USER_TEMPLATE.format(text=text),
                              MAX_TOKENS, temperature)
        params["output_config"] = {"format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}}
        return params

    summary = []
    for lang in langs:
        df = pd.read_csv(DATA / f"{lang}.csv", encoding="utf-8-sig")
        if args.sample:
            df = stratified_sample(df, args.sample)

        cache_path = LLM_DIR / "raw" / f"{lang}_{model}_{METHOD}.jsonl"
        cache = load_cache(cache_path)
        todo = [(int(r.id), str(r.text)) for r in df.itertuples() if int(r.id) not in cache]
        print(f"\n[{lang}] model={model}  yorum={len(df)}  önbellekte={len(df) - len(todo)}  "
              f"gönderilecek={len(todo)}")

        if todo and args.offline:
            sys.exit(f"--offline: {len(todo)} yorumun yanıtı önbellekte yok.")

        if todo:
            if args.batch:
                state_path = LLM_DIR / "raw" / f"{lang}_{model}_{METHOD}_batch.json"
                run_batch(
                    client, todo, make_params,
                    lambda i, t, msg, err: make_record(i, extract_answer(msg), msg.usage, True),
                    cache_path, state_path,
                )
            else:
                run_realtime(
                    client, todo, make_params,
                    lambda i, t, msg: make_record(i, extract_answer(msg), msg.usage, False),
                    cache_path, workers=args.workers,
                )
            cache = load_cache(cache_path)

        missing = [i for i in df["id"] if int(i) not in cache]
        if missing:
            print(f"! {len(missing)} yorumun yanıtı eksik, scripti tekrar çalıştır.")
            continue

        recs = [cache[int(i)] for i in df["id"]]
        df = df.copy()
        df["prediction"] = [r["prediction"] for r in recs]
        df["correct"] = df["label"] == df["prediction"]
        invalid = sum(not r["valid"] for r in recs)
        in_tok = sum(r["input_tokens"] for r in recs)
        out_tok = sum(r["output_tokens"] for r in recs)

        title = f"LLM ({model}) — {lang.upper()}" + (f" — DENEME {len(df)}" if args.sample else "")
        acc, errors = report(df, title)

        n_batch = sum(bool(r.get("batch")) for r in recs)
        mode = "batch" if n_batch == len(recs) else ("realtime" if n_batch == 0 else "mixed")
        costs = [estimate_cost(model, r["input_tokens"], r["output_tokens"], bool(r.get("batch")))
                 for r in recs]
        cost = None if None in costs else sum(costs)
        print(f"\nGeçersiz yanıt: {invalid} (→ '{FALLBACK}' sayıldı)")
        print(f"Token: giriş={in_tok}  çıkış={out_tok}  (batch ile alınan: {n_batch}/{len(recs)})")
        if cost is not None:
            print(f"Tahmini maliyet: ${cost:.4f}")
            if args.sample:
                per = 3000 / len(df)
                print(f"3000 yoruma ölçeklenmiş tahmin: "
                      f"${estimate_cost(model, in_tok, out_tok, False) * per:.3f} (anlık), "
                      f"${estimate_cost(model, in_tok, out_tok, True) * per:.3f} (batch)")

        out = df[["id", "text", "label", "prediction", "correct"]]
        if args.sample:
            path = LLM_DIR / "trials" / f"{lang}_{model}_{len(df)}.csv"
        else:
            path = RESULTS / f"llm_{lang}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"\nKaydedildi: {path.relative_to(ROOT)}")

        info = {
            "language": lang,
            "model": model,
            "n": len(df),
            "sample": bool(args.sample),
            "mode": mode,
            "temperature": temperature,
            "max_tokens": MAX_TOKENS,
            "system_prompt": SYSTEM_PROMPT,
            "user_template": USER_TEMPLATE,
            "response_format": "structured outputs (json_schema, enum: positive/negative/neutral)",
            "output_schema": OUTPUT_SCHEMA,
            "accuracy": round(acc, 4),
            "errors_by_class": errors,
            "invalid_responses": invalid,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "estimated_cost_usd": None if cost is None else round(cost, 4),
            "prices_usd_per_mtok": PRICES.get(model),
            "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        suffix = f"_trial_{model}" if args.sample else ""
        (LLM_DIR / f"run_info_{lang}{suffix}.json").write_text(
            json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        summary.append((lang, acc, cost))

    if summary:
        print(f"\n{'=' * 50}\nÖZET ({model})")
        for lang, acc, cost in summary:
            c = "" if cost is None else f"  maliyet ${cost:.4f}"
            print(f"  {lang}: doğruluk {acc * 100:.2f}%{c}")


if __name__ == "__main__":
    main()
