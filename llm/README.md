# LLM

Bu klasör **Furkan**'a aittir. LLM (Claude API) tabanlı duygu analizi kodları buraya konur; başkası bu klasöre yazmaz. Beklenen çıktılar `results/llm_english.csv` ve `results/llm_turkish.csv` dosyalarıdır. Zorunlu kolon formatı: `id, text, label, prediction, correct` (`label` ve `prediction` yalnızca `positive`, `negative`, `neutral` olabilir; kodlama `utf-8-sig`). API anahtarı yalnızca yerel `.env` dosyasında durur (şablon: `.env.example`), asla GitHub'a yüklenmez.

## Dosyalar

| Dosya | Ne yapar |
|---|---|
| `llm_analysis.py` | Yorumları Claude ile sınıflandırır → `results/llm_<dil>.csv` |
| `translate.py` | Türkçe yorumları İngilizceye çevirir → `llm/turkish_translated.csv` (Azra'nın çeviri sonrası VADER deneyi için) |
| `claude_utils.py` | Ortak kod: API istemcisi, model adları, fiyatlar, önbellek, Batch API |
| `raw/*_json.jsonl` | Modelin her yorum için verdiği ham yanıt + token sayıları (önbellek; rapordaki sayılar buradan gelir) |
| `trials/` | 50 yorumluk deneme sonuçları (model seçimi için) |
| `run_info_*.json` | Her çalıştırmanın modeli, prompt'u, parametreleri, doğruluğu, sınıf bazlı hata sayıları ve maliyeti |

## Kurulum

```bash
pip install -r requirements.txt
cp .env.example .env        # içine ANTHROPIC_API_KEY=... yaz
```

## Çalıştırma sırası (repo kök klasöründen)

```bash
# 1) 50 yorumluk deneme — üç model (her dilden 50 yorum, sınıf başına ~17, seed 42)
python llm/llm_analysis.py --sample 50 --model haiku
python llm/llm_analysis.py --sample 50 --model sonnet
python llm/llm_analysis.py --sample 50 --model opus
#    → doğruluk + ölçülen maliyet + 3000 yoruma ölçeklenmiş tahmin yazdırılır

# 2) Seçilen modelle 3000 + 3000 yorum (Batch API, %50 ucuz)
python llm/llm_analysis.py --model <secilen> --batch

# 3) Türkçe çeviri (Azra için)
python llm/translate.py --sample 20 --model <secilen>     # önce birkaç çeviriye göz at
python llm/translate.py --model <secilen> --batch
```

- `--batch` ile script batch'i gönderir ve bitene kadar dakikada bir durumu yazdırır (genelde birkaç dakika–1 saat). Bilgisayar kapanırsa aynı komutu tekrar çalıştır; yeni batch açmaz, gönderileni bekler.
- `--batch` yazmazsan istekler anında, paralel gönderilir (tam fiyat).
- Yarıda kalan ya da hata veren istekler önbellekte olmadığı için komut tekrar çalıştırılınca yalnızca onlar gönderilir. Aynı yorum için iki kez ödeme yapılmaz.
- `--offline`: API'ye hiç gitmeden önbellekten CSV ve raporu yeniden üretir.
- Model `temperature` parametresini reddederse anlık modda kod kendisi onsuz devam eder; batch modunda `--no-temperature` ekle.

## Yöntem (rapor için)

- **Yaklaşım:** zero-shot prompting. Model eğitilmez, örnek (few-shot) verilmez.
- **Prompt:** iki dil için **aynı** İngilizce talimat (tam metni `llm_analysis.py` içindeki `SYSTEM_PROMPT` ve `run_info_*.json`). Yorum `<text>...</text>` etiketleri arasında verilir.
- **Yanıt biçimi:** structured outputs (`output_config.format = json_schema`). Modelin cevabı `{"sentiment": ...}` biçiminde JSON olmak zorundadır; `sentiment` alanı şemayla (enum) yalnızca `positive` / `negative` / `neutral` değerlerini alabilir, model açıklama yazamaz. İlk denemede serbest metinle tek kelime istendiğinde Haiku, 50 Türkçe yorumun 9'unda etiket yerine açıklama yazmaya başlamıştı; zorunlu araç çağrısı (`tool_choice`) ise Sonnet 5.5'te desteklenmiyor. Bu yüzden bu yönteme geçildi.
- **Parametreler:** `temperature = 0`, `max_tokens = 50`.
- **Yedek kural:** bir yanıtta yine de geçerli etiket bulunamazsa `neutral` sayılır; bu tür yanıtların sayısı çıktıda ve `run_info` dosyasında raporlanır.
- **Model kimlikleri:** `claude-haiku-5-5`, `claude-sonnet-5-5`, `claude-opus-5-5`.
- **Fiyatlar** (USD / milyon token, giriş–çıkış, 22 Kasım 2026): Haiku 5.5 0,10–0,50 · Sonnet 5.5 2–10 · Opus 5.5 4–20. Batch API yarı fiyat. Kaynak: <https://platform.claude.com/docs/en/about-claude/pricing>
- **Maliyet** kod tarafından gerçek token sayılarından hesaplanır; bu tablo fiyatları değişirse `claude_utils.py` içindeki `PRICES` güncellenmelidir. Faturadaki kesin tutar için Console'a bak.
