"""İngilizce duygu analizi: önceden eğitilmiş RoBERTa (fine-tuning YOK).

Ne yapar: data/english.csv (3000 tweet) üzerinde cardiffnlp/twitter-roberta-base-sentiment
ile tahmin üretir ve results/bert_english.csv dosyasına yazar.
Çalıştırma (repo kökünden): .venv/bin/python bert/bert_english.py
Parametreler: max_length=128 (truncation), batch_size=32, seed=42, cihaz=mps (yoksa cpu).
Etiket eşlemesi (model kartı): LABEL_0=negative, LABEL_1=neutral, LABEL_2=positive.
Ön işleme (model kartı): @kullanıcı -> "@user", http(s) bağlantıları -> "http".
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
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
MAX_LEN = 128
BATCH = 32
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
LABELS = ["negative", "neutral", "positive"]


def preprocess(text):
    # Model kartındaki tweet ön işlemesi
    words = []
    for t in str(text).split(" "):
        t = "@user" if t.startswith("@") and len(t) > 1 else t
        t = "http" if t.startswith("http") else t
        words.append(t)
    return " ".join(words)


device = "mps" if torch.backends.mps.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(device).eval()

df = pd.read_csv(ROOT / "data" / "english.csv")
texts = [preprocess(t) for t in df["text"]]

preds = []
with torch.no_grad():
    for i in range(0, len(texts), BATCH):
        enc = tokenizer(texts[i:i + BATCH], truncation=True, max_length=MAX_LEN,
                        padding=True, return_tensors="pt").to(device)
        logits = model(**enc).logits
        preds.extend(logits.argmax(dim=-1).cpu().tolist())

df["prediction"] = [ID2LABEL[p] for p in preds]
df["correct"] = df["label"] == df["prediction"]

acc = accuracy_score(df["label"], df["prediction"])
print(f"ACCURACY: {acc:.4f} ({acc * 100:.2f}%)")
print("ERROR COUNTS BY TRUE CLASS:")
for label in LABELS:
    sub = df[df["label"] == label]
    print(f"{label}: {(~sub['correct']).sum()} / {len(sub)}")

out = ROOT / "results" / "bert_english.csv"
df.to_csv(out, index=False, encoding="utf-8-sig")
print(f"Saved: {out.relative_to(ROOT)}")
