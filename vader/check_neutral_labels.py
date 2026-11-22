
from pathlib import Path
import pandas as pd

df = pd.read_csv("data/turkish.csv")

neutral = df[df["label"] == "neutral"]

sample = neutral.sample(n=100, random_state=42)

sample = sample[["id", "text", "label"]].copy()
sample["manual_label"] = ""
sample["notes"] = ""

output = Path("vader/neutral_label_review.csv")
output.parent.mkdir(parents=True, exist_ok=True)

sample.to_csv(output, index=False, encoding="utf-8-sig")

print("Toplam nötr yorum:", len(neutral))
print("Kontrol için seçilen:", len(sample))
print("Kaydedildi:", output)
