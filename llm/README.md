# LLM

Bu klasörde Furkan Başer tarafından yürütülen Claude API tabanlı duygu analizi yer alır. Kayıtlı tam sonuçlar Sonnet modeline aittir. Tahmin dosyaları `results/llm_english.csv` ve `results/llm_turkish.csv` konumundadır. API anahtarı yalnızca yerel `.env` dosyasında tutulur.

## Dosyalar

| Dosya | Ne yapar |
|---|---|
| `llm_analysis.py` | Yorumları Claude ile sınıflandırır → `results/llm_<dil>.csv` |
| `translate.py` | Türkçe yorumları İngilizceye çevirir → `llm/turkish_translated.csv` (Azra'nın çeviri sonrası VADER deneyi için) |
| `claude_utils.py` | Ortak kod: API istemcisi, model adları, fiyatlar, önbellek, Batch API |
| `raw/*_json.jsonl` | Ayrıştırılmış yanıt, tahmin, geçerlilik ve token sayıları (tam API yanıtı/stop_reason saklanmaz) (önbellek; rapordaki sayılar buradan gelir) |
| `trials/` | 50 yorumluk deneme sonuçları (model seçimi için) |
| `run_info_*.json` | Her çalıştırmanın modeli, prompt'u, parametreleri, doğruluğu, sınıf bazlı hata sayıları ve maliyeti |

## Kurulum

```bash
pip install -r requirements.txt
cp .env.example .env        # içine ANTHROPIC_API_KEY=... yaz
```

## Kayıtlı sonuçları yeniden üretme

```bash
python llm/llm_analysis.py --model sonnet --offline --no-temperature
python llm/translate.py --model haiku --offline
```

Bu komutlar yeni API isteği göndermez. Sınıflandırma kaydındaki `temperature: null` ile uyum için `--no-temperature` kullanılır.

## Yeni API çalışması (ücretli; repo kök klasöründen)

```bash
# 1) 50 yorumluk deneme — üç model (her dilden 50 yorum, sınıf başına ~17, seed 42)
python llm/llm_analysis.py --sample 50 --model haiku
python llm/llm_analysis.py --sample 50 --model sonnet
python llm/llm_analysis.py --sample 50 --model opus
#    → doğruluk + ölçülen maliyet + 3000 yoruma ölçeklenmiş tahmin yazdırılır

# 2) Seçilen modelle 3000 + 3000 yorum (Batch API, %50 ucuz)
python llm/llm_analysis.py --model sonnet --batch --no-temperature

# 3) Türkçe çeviri (Azra için)
python llm/translate.py --sample 20 --model haiku     # önce birkaç çeviriye göz at
python llm/translate.py --model haiku --batch
```

- `--batch` ile script batch'i gönderir ve bitene kadar dakikada bir durumu yazdırır (genelde birkaç dakika–1 saat). Bilgisayar kapanırsa aynı komutu tekrar çalıştır; yeni batch açmaz, gönderileni bekler.
- `--batch` yazmazsan istekler anında, paralel gönderilir (tam fiyat).
- Yarıda kalan ya da hata veren istekler önbellekte olmadığı için komut tekrar çalıştırılınca yalnızca onlar gönderilir. Aynı yorum için iki kez ödeme yapılmaz.
- `--offline`: API'ye hiç gitmeden önbellekten CSV ve raporu yeniden üretir.
- Model `temperature` parametresini reddederse anlık modda kod kendisi onsuz devam eder; batch modunda `--no-temperature` ekle.

## Yöntem (rapor için)

- **Yaklaşım:** zero-shot prompting. Model eğitilmez, örnek (few-shot) verilmez.
- **Prompt:** iki dil için **aynı** İngilizce talimat (tam metni `llm_analysis.py` içindeki `SYSTEM_PROMPT` ve `run_info_*.json`). Yorum `<text>...</text>` etiketleri arasında verilir.
- **Yanıt biçimi:** `output_config.format = json_schema` ile sentiment alanı üç sınıfla sınırlandırılmıştır. Buna rağmen 13 İngilizce ve 45 Türkçe kayıtta geçerli etiket ayrıştırılamamıştır. Bu 58 kaydın tamamı 50 çıktı tokenına ulaşmış; geçerli kayıtlar 14–15 token kullanmıştır. Bu, çıktı sınırının yetersizliğine güçlü işarettir; tam yanıt ve `stop_reason` olmadan kesin bitiş nedeni doğrulanamaz.
- **Tam deney parametreleri:** `max_tokens = 50`; `temperature` gönderilmemiştir (kayıtta null). Thinking ve effort açıkça ayarlanmamıştır; sağlayıcının model varsayılanları geçerlidir. Komut satırında `--no-temperature` verilmezse kodun başlangıç tercihi 0'dır; bu, kayıtlı tam deneyin ayarı değildir.
- **Yedek kural:** bir yanıtta yine de geçerli etiket bulunamazsa `neutral` sayılır; bu tür yanıtların sayısı çıktıda ve `run_info` dosyasında raporlanır.
- **Model kimlikleri:** `claude-haiku-5-5`, `claude-sonnet-5-5`, `claude-opus-5-5`.
- **Fiyatlar** (USD / milyon token, giriş–çıkış, 22 Kasım 2026): Haiku 5.5 0,10–0,50 · Sonnet 5.5 2–10 · Opus 5.5 4–20. Batch API yarı fiyat. Kaynak: <https://platform.claude.com/docs/en/about-claude/pricing>
- **Maliyet** kod tarafından gerçek token sayılarından hesaplanır; bu tablo fiyatları değişirse `claude_utils.py` içindeki `PRICES` güncellenmelidir. Faturadaki kesin tutar için Console'a bak.

## Deneyin sınırları ve maliyet

Model seçimi için 50'şer yorum nihai test kümesinden alınmıştır; gelecekte ayrı doğrulama kümesi kullanılmalıdır. Depoda tamamlanmış Opus denemesi bulunmaz. Önbellek kimlik tabanlıdır; metin/istem değişirse eski yanıtların yeniden kullanılmaması ayrıca kontrol edilmelidir.

Her dilde 50 kayıt tam fiyat, 2.950 kayıt Batch API ile %50 indirim katsayısıyla hesaplanmıştır. Sonnet için milyon token başına giriş 2 USD, çıkış 10 USD katsayılarıyla maliyet İngilizce 1,768021 USD, Türkçe 1,798396 USD, toplam 3,566417 USD'dir. Bunlar kayıtlı tokenlardan hesaplanan tahminlerdir; fatura tutarı değildir ve çeviri/diğer denemeleri kapsamaz.
