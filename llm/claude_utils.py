"""
claude_utils.py — llm_analysis.py ve translate.py'nin ortak kullandığı yardımcılar.

- Claude API istemcisini .env'deki ANTHROPIC_API_KEY ile kurar
- Model takma adları (haiku / sonnet / opus) ve token fiyatları
- Her isteğin sonucunu JSONL önbelleğine yazar; script yarıda kesilirse
  kaldığı yerden devam eder, aynı yorum için ikinci kez para ödenmez
- İki çalışma modu: anlık (paralel istekler) ve Batch API (%50 indirimli)

Doğrudan çalıştırılmaz; diğer scriptler içe aktarır.
"""

import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LLM_DIR = ROOT / "llm"
DATA = ROOT / "data"
RESULTS = ROOT / "results"

# API model kimlikleri
MODELS = {
    "haiku": "claude-haiku-5-5",
    "sonnet": "claude-sonnet-5-5",
    "opus": "claude-opus-5-5",
}

# USD / milyon token (giriş, çıkış) — platform.claude.com/docs/en/about-claude/pricing
# Kontrol tarihi: 22 Kasım 2026. Batch API bu fiyatların yarısıdır.
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}


def resolve_model(name):
    return MODELS.get(name, name)


def make_client():
    try:
        import anthropic
    except ImportError:
        sys.exit("'anthropic' paketi yüklü değil: pip install -r requirements.txt")
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit(
            "ANTHROPIC_API_KEY bulunamadı. .env.example dosyasını .env olarak "
            "kopyalayıp anahtarını yaz."
        )
    # max_retries: hız sınırı (429) ve geçici hatalarda SDK otomatik tekrar dener
    return anthropic.Anthropic(max_retries=8)


def estimate_cost(model, input_tokens, output_tokens, batch=False):
    if model not in PRICES:
        return None
    pin, pout = PRICES[model]
    cost = (input_tokens * pin + output_tokens * pout) / 1_000_000
    return cost / 2 if batch else cost


def response_text(msg):
    return "".join(
        getattr(block, "text", "") for block in msg.content
        if getattr(block, "type", "") == "text"
    ).strip()


# ---------------------------------------------------------------- önbellek

def load_cache(path):
    """JSONL önbelleğini {id: kayıt} olarak okur."""
    cache = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    cache[rec["id"]] = rec
    return cache


class CacheWriter:
    """Kayıtları satır satır ekler; paralel iş parçacıklarından güvenle çağrılır."""

    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")
        self.lock = threading.Lock()

    def write(self, rec):
        with self.lock:
            self.f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self.f.flush()

    def close(self):
        self.f.close()


# ---------------------------------------------------------------- istek

def build_params(model, system, user_text, max_tokens, temperature):
    params = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user_text}],
    }
    if temperature is not None:
        params["temperature"] = temperature
    return params


_temperature_supported = {}


def create_message(client, params):
    """Bazı modeller temperature kabul etmeyebilir; öyleyse bir kez onsuz dener."""
    model = params["model"]
    if _temperature_supported.get(model) is False:
        params = {k: v for k, v in params.items() if k != "temperature"}
    try:
        return _send(client, params)
    except Exception as e:  # anthropic.BadRequestError
        if "temperature" in params and "temperature" in str(e).lower():
            _temperature_supported[model] = False
            print(f"  ! {model} temperature kabul etmiyor, onsuz devam ediliyor")
            params = {k: v for k, v in params.items() if k != "temperature"}
            return _send(client, params)
        raise


def _send(client, params):
    """SDK sürümü output_config'i tanımıyorsa onu extra_body ile gönderir."""
    try:
        return client.messages.create(**params)
    except TypeError:
        if "output_config" not in params:
            raise
        rest = {k: v for k, v in params.items() if k != "output_config"}
        return client.messages.create(**rest, extra_body={"output_config": params["output_config"]})


# ---------------------------------------------------------------- çalıştırıcılar

def run_realtime(client, items, make_params, handle, cache_path, workers=4):
    """
    items: [(id, text)]  — önbellekte olmayanlar
    make_params(text) -> istek parametreleri
    handle(id, text, message) -> önbelleğe yazılacak kayıt (dict)
    """
    writer = CacheWriter(cache_path)
    done = 0
    total = len(items)
    start = time.time()

    def work(item):
        item_id, text = item
        msg = create_message(client, make_params(text))
        return handle(item_id, text, msg)

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(work, it): it for it in items}
            for fut in as_completed(futures):
                writer.write(fut.result())
                done += 1
                if done % 50 == 0 or done == total:
                    rate = done / max(time.time() - start, 1e-9)
                    print(f"  {done}/{total}  ({rate:.1f} istek/sn)")
    finally:
        writer.close()


def run_batch(client, items, make_params, handle_result, cache_path, state_path,
              poll_seconds=60):
    """
    Batch API: istekleri tek seferde gönderir, bitince sonuçları önbelleğe yazar.
    Gönderilen batch'in kimliği state_path'e kaydedilir; script kapanırsa
    tekrar çalıştırıldığında yeni batch açmak yerine aynısını bekler.

    handle_result(id, text, message_or_None, error_or_None) -> kayıt
    """
    texts = {str(i): t for i, t in items}

    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        batch_id = state["batch_id"]
        print(f"  Önceki batch devam ediyor: {batch_id}")
    else:
        requests = []
        for item_id, text in items:
            params = make_params(text)
            requests.append({"custom_id": str(item_id), "params": params})
        batch = client.messages.batches.create(requests=requests)
        batch_id = batch.id
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(
            json.dumps({"batch_id": batch_id, "count": len(requests)}),
            encoding="utf-8",
        )
        print(f"  Batch gönderildi: {batch_id} ({len(requests)} istek)")

    while True:
        batch = client.messages.batches.retrieve(batch_id)
        c = batch.request_counts
        print(
            f"  durum={batch.processing_status}  işleniyor={c.processing} "
            f"başarılı={c.succeeded} hatalı={c.errored}"
        )
        if batch.processing_status == "ended":
            break
        time.sleep(poll_seconds)

    writer = CacheWriter(cache_path)
    failed = 0
    first_error = None
    try:
        for res in client.messages.batches.results(batch_id):
            cid = res.custom_id
            if res.result.type == "succeeded":
                rec = handle_result(int(cid), texts.get(cid, ""), res.result.message, None)
            else:
                failed += 1
                if first_error is None:
                    first_error = getattr(res.result, "error", res.result.type)
                continue  # önbelleğe yazılmaz → sonraki çalıştırmada tekrar denenir
            writer.write(rec)
    finally:
        writer.close()
    state_path.unlink()
    if failed:
        print(f"  ! {failed} istek başarısız oldu; scripti tekrar çalıştırınca yeniden denenir.")
        print(f"    İlk hata: {first_error}")
        print("    Hata 'temperature' ile ilgiliyse --no-temperature ile tekrar çalıştır.")
    return failed
