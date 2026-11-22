
from datasets import load_dataset
from collections import Counter

datasets = {
    "English": "frasem/sentiment-dataset",
    "Turkish": "maydogan/Turkish_SentimentAnalysis_TRSAv1"
}

for language, name in datasets.items():
    print(f"\n{'=' * 40}")
    print(f"DATASET: {language}")

    try:
        ds = load_dataset(name)
        print(ds)

        for split in ds:
            data = ds[split]
            print(f"\nSPLIT: {split}")
            print("COLUMNS:", data.column_names)
            print("FIRST ROW:", data[0])

            if "label" in data.column_names:
                counts = Counter(data["label"])
                print("LABEL COUNTS:", dict(counts))

    except Exception as e:
        print("ERROR:", repr(e))
