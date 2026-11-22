
from pathlib import Path

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

source = Path("vader/turkish_translated_clean.csv")
output = Path("results/vader_turkish_translated.csv")

df = pd.read_csv(source)
analyzer = SentimentIntensityAnalyzer()

def predict(text):
    score = analyzer.polarity_scores(str(text))["compound"]

    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    return "neutral"

df["prediction"] = df["text_en"].apply(predict)
df["correct"] = df["prediction"] == df["label"]

result = df[["id", "text", "label", "prediction", "correct"]]

assert len(result) == 3000
assert result["id"].is_unique

output.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(output, index=False, encoding="utf-8-sig")

accuracy = result["correct"].mean()

print(f"Toplam yorum: {len(result)}")
print(f"Doğru tahmin: {result['correct'].sum()}")
print(f"Yanlış tahmin: {(~result['correct']).sum()}")
print(f"Accuracy: {accuracy:.2%}")

print("\nGerçek sınıfa göre yanlış tahminler:")
print(
    result.loc[~result["correct"]]
    .groupby("label")
    .size()
    .reindex(["negative", "neutral", "positive"], fill_value=0)
)

print("\nKaydedildi:", output)
