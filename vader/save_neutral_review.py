
import pandas as pd

path = "vader/neutral_label_review.csv"
df = pd.read_csv(path)

labels = {
    1566:"positive", 2205:"neutral", 2210:"negative", 1984:"positive",
    1252:"neutral", 2036:"neutral", 1880:"negative", 1550:"neutral",
    2591:"negative", 453:"neutral", 2442:"neutral", 255:"negative",
    1925:"negative", 2934:"negative", 2834:"negative", 2707:"positive",
    853:"negative", 2671:"negative", 2290:"positive", 978:"negative",
    1667:"neutral", 562:"negative", 1123:"neutral", 1586:"neutral",
    666:"negative", 736:"negative", 330:"positive", 2960:"neutral",
    2718:"negative", 2858:"negative", 1051:"positive", 473:"neutral",
    1871:"positive", 1509:"negative", 1119:"neutral", 637:"neutral",
    2061:"neutral", 1749:"negative", 2716:"positive", 188:"negative",
    994:"negative", 318:"negative", 964:"neutral", 2938:"positive",
    913:"neutral", 844:"positive", 2783:"positive", 1810:"neutral",
    1328:"neutral", 2522:"positive", 1713:"negative", 2666:"neutral",
    812:"neutral", 1732:"negative", 80:"negative", 98:"neutral",
    1863:"positive", 34:"negative", 692:"negative", 2469:"positive",
    903:"neutral", 172:"neutral", 1648:"negative", 663:"neutral",
    1813:"positive", 2080:"positive", 1999:"neutral", 2622:"neutral",
    228:"negative", 1649:"positive", 347:"neutral", 1495:"neutral",
    1770:"neutral", 2218:"positive", 886:"negative", 874:"neutral",
    1967:"negative", 140:"negative", 1768:"positive", 941:"neutral",
    2039:"negative", 211:"negative", 842:"negative", 214:"neutral",
    977:"negative", 1665:"neutral", 2990:"positive", 2139:"positive",
    2268:"positive", 991:"neutral", 1171:"positive", 1354:"neutral",
    1567:"neutral", 683:"neutral", 2352:"negative", 1319:"positive",
    2295:"negative", 291:"negative", 207:"positive", 2484:"positive"
}

assert len(df) == 100
assert df["id"].is_unique
assert set(df["id"]) == set(labels)

df["manual_label"] = df["id"].map(labels)
df["notes"] = df.apply(
    lambda row: "Etiket uyumsuzluğu" if row["manual_label"] != row["label"]
    else "Etiket uyumlu",
    axis=1
)

df.to_csv(path, index=False, encoding="utf-8-sig")

print("İncelenen:", len(df))
print(df["manual_label"].value_counts())
print("Etiket uyuşmazlığı:", (df["manual_label"] != df["label"]).sum())
print("Kaydedildi:", path)
