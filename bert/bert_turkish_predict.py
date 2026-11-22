"""Türkçe tahmin: fine-tune edilmiş BERTurk ile data/turkish.csv (3000 yorum) tahmini.

Ne yapar: bert/models/berturk-trsa/ içindeki modeli yükler, test yorumlarını sınıflandırır ve
results/bert_turkish.csv dosyasına yazar (id, text, label, prediction, correct; utf-8-sig).
Önce bert/bert_turkish_train.py çalıştırılmış olmalı.
Çalıştırma (repo kökünden): .venv/bin/python bert/bert_turkish_predict.py
Parametreler: max_length=128 (truncation), batch_size=64, seed=42, cihaz=mps (yoksa cpu).
"""
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "bert" / "models" / "berturk-trsa"
MAX_LEN, BATCH = 128, 64
LABELS = ["negative", "neutral", "positive"]

device = "mps" if torch.backends.mps.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device).eval()
id2label = {int(k): v for k, v in model.config.id2label.items()}
assert [id2label[i] for i in range(3)] == LABELS, id2label

df = pd.read_csv(ROOT / "data" / "turkish.csv")
texts = df["text"].astype(str).tolist()

preds = []
with torch.no_grad():
    for i in range(0, len(texts), BATCH):
        enc = tokenizer(texts[i:i + BATCH], truncation=True, max_length=MAX_LEN,
                        padding=True, return_tensors="pt").to(device)
        preds.extend(model(**enc).logits.argmax(dim=-1).cpu().tolist())

df["prediction"] = [id2label[p] for p in preds]
df["correct"] = df["label"] == df["prediction"]

acc = accuracy_score(df["label"], df["prediction"])
print(f"ACCURACY: {acc:.4f} ({acc * 100:.2f}%)")
print("ERROR COUNTS BY TRUE CLASS:")
for label in LABELS:
    sub = df[df["label"] == label]
    print(f"{label}: {(~sub['correct']).sum()} / {len(sub)}")

out = ROOT / "results" / "bert_turkish.csv"
df.to_csv(out, index=False, encoding="utf-8-sig")
print(f"Saved: {out.relative_to(ROOT)}")
