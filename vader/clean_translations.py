
from pathlib import Path
import pandas as pd

source = Path("llm/turkish_translated.csv")
output = Path("vader/turkish_translated_clean.csv")

df = pd.read_csv(source)

# Elle incelenen sorunlu kayıtlar için temiz çeviriler
corrections = {
    2: "It doesn't work at all. The brush is nice, but it's sold everywhere for 10 TL anyway.",
    81: "It's really interesting, this product I use on my T-zone made it greasy :) Wasn't it supposed to dry it out? :(",
    255: "It doesn't provide any color, which was disappointing.",
    364: "It even picks up the finest hairs. It's ideal for the upper lip.",
    530: "The machine is very good, but it doesn't remove short hairs.",
    591: "It is a water-based product made for sensitive eyes, but it still stings.",
    1014: "Unfortunately, the cap wasn't tightened properly before packaging. A third of the product had leaked out by the time it arrived.",
    1078: "I have combination and sensitive skin. The results are quite good.",
    1429: "It's nice that it contains natural ingredients, but I can't say it stopped my hair loss.",
    1490: "It stopped working within two weeks. It didn't work even after charging. I don't recommend buying it.",
    1533: "They should ship it already.",
    1568: "It leaves a clean feeling, but it isn't effective.",
    1707: "I've used them so far. They're very nice and soft, and they don't cause irritation.",
    1921: "It doesn't whiten very much, but the soap cleans well and the cream moisturizes nicely.",
    2033: "The color is perfect, and the lipstick feels very light on the lips. Buy it.",
    2088: "It's very dark. People with fair skin shouldn't buy it.",
    2408: "I've started being able to manage my hair without styling products.",
    2480: "Women with small breasts probably shouldn't choose this. It looked like I was wearing an undershirt. I didn't like it and returned it.",
    2522: "It's very good. It makes hair extremely soft.",
    2560: "My favorite concealer arrived in two days. Thank you, Trendyol :)",
    2641: "It's a very good product and affordable. My hair doesn't turn orange.",
    2692: "It removes oil and cleanses, but it isn't very effective.",
    2784: "It doesn't arrive stretched onto a canvas. I returned it for this reason. It's difficult to work this way.",
    2798: "The color is very beautiful, with a slight brown undertone. It's soft.",
    2863: "I no longer need to dye my roots. Using it twice was enough. I recommend it to anyone who wants to lighten their hair naturally.",
}

assert len(df) == 3000
assert df["id"].is_unique
assert set(corrections).issubset(set(df["id"]))

for record_id, translation in corrections.items():
    df.loc[df["id"] == record_id, "text_en"] = translation

output.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output, index=False, encoding="utf-8-sig")

pattern = r"Wait, let me|Final translation:|Here is the translation|As an AI"
remaining = df["text_en"].str.contains(
    pattern, case=False, regex=True, na=False
).sum()

print("Toplam kayıt:", len(df))
print("Düzeltilen kayıt:", len(corrections))
print("Kalan tespit edilen açıklama:", remaining)
print("Kaydedildi:", output)
