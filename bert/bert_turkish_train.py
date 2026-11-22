"""Türkçe duygu analizi: BERTurk (dbmdz/bert-base-turkish-cased) fine-tuning.

Ne yapar: TRSAv1 verisini veri_hazirla.py ile aynı şekilde temizler, data/turkish.csv'deki
(test) tüm yorumları çıkarır (sızıntı kontrolü: çakışma 0 olmak ZORUNDA), her sınıftan 10.000
yorum örnekler, %10'unu doğrulama için ayırır ve modeli eğitir. En iyi model (doğrulama
macro-F1) bert/models/berturk-trsa/ içine kaydedilir; özet metrikler bert/train_log.txt'ye
eklenir.
Çalıştırma (repo kökünden):
  .venv/bin/python bert/bert_turkish_train.py            # tam eğitim
  .venv/bin/python bert/bert_turkish_train.py --smoke    # 300 örnek, hızlı deneme
Parametreler: max_length=128, batch=16, lr=2e-5, epoch=2, weight_decay=0.01, warmup=%6,
seed=42, her epoch sonunda değerlendirme, seçim ölçütü = doğrulama macro-F1. Cihaz: mps/cpu.
"""
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, Trainer,
                          TrainingArguments, set_seed)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
set_seed(SEED)

SMOKE = "--smoke" in sys.argv
ROOT = Path(__file__).resolve().parent.parent
BASE = "dbmdz/bert-base-turkish-cased"
MAX_LEN, BATCH, LR, EPOCHS, WD, WARMUP = 128, 16, 2e-5, 2, 0.01, 0.06
PER_CLASS = 100 if SMOKE else 10_000
LABELS = ["negative", "neutral", "positive"]
L2I = {l: i for i, l in enumerate(LABELS)}
OUT_DIR = ROOT / "bert" / ("models/smoke" if SMOKE else "models/berturk-trsa")
CKPT = ROOT / "bert" / "checkpoints" / ("smoke" if SMOKE else "berturk-trsa")
LOG = ROOT / "bert" / ("train_log_smoke.txt" if SMOKE else "train_log.txt")


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


# --- Veri: veri_hazirla.py ile aynı temizlik ---
tr = load_dataset("maydogan/Turkish_SentimentAnalysis_TRSAv1", split="train").to_pandas()
tr = tr.rename(columns={"review": "text", "score": "label"})[["text", "label"]]
tr["label"] = tr["label"].astype(str).str.strip().str.lower()
tr = tr.dropna(subset=["text", "label"]).copy()
tr["text"] = tr["text"].astype(str).str.strip()
tr = tr[tr["text"] != ""].drop_duplicates(subset=["text"])
n_before = len(tr)

# --- Sızıntı kontrolü: test yorumlarını çıkar ---
test_texts = set(pd.read_csv(ROOT / "data" / "turkish.csv")["text"].astype(str).str.strip())
tr = tr[~tr["text"].isin(test_texts)]
n_after = len(tr)
overlap = len(set(tr["text"]) & test_texts)
log(f"[{time.strftime('%F %T')}] smoke={SMOKE} temiz satır: {n_before}, test çıkarıldıktan sonra: {n_after}, "
    f"çıkarılan: {n_before - n_after}, test yorum sayısı: {len(test_texts)}, kalan çakışma: {overlap}")
assert overlap == 0, "SIZINTI: eğitim verisinde test yorumu var!"
assert set(tr["label"]) == set(LABELS), set(tr["label"])

pool = pd.concat([g.sample(n=PER_CLASS, random_state=SEED) for _, g in tr.groupby("label")])
pool["y"] = pool["label"].map(L2I)
train_df, val_df = train_test_split(pool, test_size=0.10, stratify=pool["y"], random_state=SEED)
assert not (set(train_df["text"]) | set(val_df["text"])) & test_texts
log(f"eğitim: {len(train_df)}, doğrulama: {len(val_df)}; sınıf dağılımı (eğitim): "
    f"{train_df['label'].value_counts().to_dict()}")

tok = AutoTokenizer.from_pretrained(BASE)


def to_ds(df):
    ds = Dataset.from_dict({"text": df["text"].tolist(), "labels": df["y"].tolist()})
    return ds.map(lambda b: tok(b["text"], truncation=True, max_length=MAX_LEN), batched=True,
                  remove_columns=["text"])


train_ds, val_ds = to_ds(train_df), to_ds(val_df)
model = AutoModelForSequenceClassification.from_pretrained(
    BASE, num_labels=3, id2label=dict(enumerate(LABELS)), label2id=L2I)


def metrics(p):
    pred = p.predictions.argmax(-1)
    return {"accuracy": accuracy_score(p.label_ids, pred),
            "macro_f1": f1_score(p.label_ids, pred, average="macro")}


args = TrainingArguments(
    output_dir=str(CKPT), num_train_epochs=EPOCHS, per_device_train_batch_size=BATCH,
    per_device_eval_batch_size=64, learning_rate=LR, weight_decay=WD, warmup_steps=WARMUP,
    eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
    load_best_model_at_end=True, metric_for_best_model="macro_f1", greater_is_better=True,
    seed=SEED, data_seed=SEED, logging_steps=50, report_to="none",
    max_steps=6 if SMOKE else -1,
)
trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                  data_collator=None, processing_class=tok, compute_metrics=metrics)
log(f"cihaz: {trainer.args.device}; parametreler: max_len={MAX_LEN} batch={BATCH} lr={LR} "
    f"epoch={EPOCHS} wd={WD} warmup={WARMUP} seed={SEED}")

t0 = time.time()
trainer.train()
elapsed = time.time() - t0
final = trainer.evaluate()
log(f"eğitim süresi: {elapsed / 60:.1f} dk; son doğrulama metrikleri (en iyi model): "
    f"acc={final['eval_accuracy']:.4f} macro_f1={final['eval_macro_f1']:.4f}")
seen = set()
for h in trainer.state.log_history:
    if "eval_macro_f1" in h and h["epoch"] not in seen:
        seen.add(h["epoch"])
        log(f"  epoch {h['epoch']:.0f}: val_acc={h['eval_accuracy']:.4f} val_macro_f1={h['eval_macro_f1']:.4f} "
            f"val_loss={h['eval_loss']:.4f}")
OUT_DIR.mkdir(parents=True, exist_ok=True)
trainer.save_model(str(OUT_DIR))
tok.save_pretrained(str(OUT_DIR))
log(f"model kaydedildi: {OUT_DIR.relative_to(ROOT)}")
