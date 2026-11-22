
from datasets import load_dataset
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "data"
OUTPUT.mkdir(exist_ok=True)

LABELS = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

def prepare(df, language):
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]
    df = df.drop_duplicates(subset=["text"])

    print(f"\n{language} - CLEAN LABEL COUNTS:")
    print(df["label"].value_counts())

    samples = []
    for label in ["positive", "negative", "neutral"]:
        subset = df[df["label"] == label]

        if len(subset) < 1000:
            raise ValueError(
                f"{language}: {label} has only {len(subset)} rows"
            )

        samples.append(
            subset.sample(n=1000, random_state=42)
        )

    result = pd.concat(samples, ignore_index=True)
    result = result.sample(frac=1, random_state=42)
    result = result.reset_index(drop=True)

    result.insert(0, "id", range(1, len(result) + 1))
    result.to_csv(
        OUTPUT / f"{language}.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print(f"{language}: SAVED {len(result)} ROWS")
    print(result["label"].value_counts())


# ENGLISH
en = load_dataset(
    "frasem/sentiment-dataset",
    split="test"
).to_pandas()

en = en.rename(columns={"label": "original_label"})
en["label"] = en["original_label"].map(LABELS)
en = en[["text", "label"]]

prepare(en, "english")


# TURKISH
tr = load_dataset(
    "maydogan/Turkish_SentimentAnalysis_TRSAv1",
    split="train"
).to_pandas()

print("\nTURKISH ORIGINAL LABEL COUNTS:")
print(tr["score"].value_counts(dropna=False))

tr = tr.rename(
    columns={"review": "text", "score": "label"}
)

tr["label"] = (
    tr["label"]
    .astype(str)
    .str.strip()
    .str.lower()
)

tr["label"] = tr["label"].replace({
    "positive": "positive",
    "negative": "negative",
    "neutral": "neutral"
})

prepare(tr[["text", "label"]], "turkish")

print("\nALL DATASETS PREPARED SUCCESSFULLY!")
