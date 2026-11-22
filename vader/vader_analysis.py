
import pandas as pd
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

analyzer = SentimentIntensityAnalyzer()

def vader_predict(text):
    score = analyzer.polarity_scores(str(text))["compound"]

    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"

labels = ["negative", "neutral", "positive"]

for language in ["english", "turkish"]:
    print(f"\n{'=' * 50}")
    print(f"VADER ANALYSIS: {language.upper()}")

    df = pd.read_csv(DATA / f"{language}.csv")
    df["prediction"] = df["text"].apply(vader_predict)
    df["correct"] = df["label"] == df["prediction"]

    accuracy = accuracy_score(df["label"], df["prediction"])
    print(f"\nACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")

    print("\nCLASSIFICATION REPORT:")
    print(classification_report(
        df["label"],
        df["prediction"],
        labels=labels,
        zero_division=0
    ))

    cm = confusion_matrix(
        df["label"],
        df["prediction"],
        labels=labels
    )

    print("\nCONFUSION MATRIX:")
    print(pd.DataFrame(cm, index=labels, columns=labels))

    print("\nERROR COUNTS BY TRUE CLASS:")
    for label in labels:
        subset = df[df["label"] == label]
        errors = (~subset["correct"]).sum()
        print(f"{label}: {errors} / {len(subset)}")

    df.to_csv(
        RESULTS / f"vader_{language}.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nSaved: results/vader_{language}.csv")

print("\nVADER ANALYSIS COMPLETED!")
