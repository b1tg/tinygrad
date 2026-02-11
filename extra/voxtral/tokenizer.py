"""Tekken tokenizer (decode only) for Voxtral."""
import json, base64, os

def load_tokenizer(model_dir: str):
  """Load a minimal Tekken decoder from tekken.json. Returns a decode function."""
  tekken_path = os.path.join(model_dir, "tekken.json")
  with open(tekken_path, "r", encoding="utf-8") as f:
    data = json.load(f)

  vocab = data["vocab"]
  config = data.get("config", {})
  n_special = int(config.get("default_num_special_tokens", 1000))
  special_ids = {int(st["rank"]) for st in data.get("special_tokens", []) if "rank" in st}
  bytes_cache: dict[int, bytes] = {}

  def token_bytes(token_id: int) -> bytes:
    b = bytes_cache.get(token_id)
    if b is not None: return b
    if token_id < 0 or token_id < n_special or token_id in special_ids:
      bytes_cache[token_id] = b""
      return b""
    vocab_id = token_id - n_special
    if vocab_id < 0 or vocab_id >= len(vocab):
      bytes_cache[token_id] = b""
      return b""
    b = base64.b64decode(vocab[vocab_id]["token_bytes"])
    bytes_cache[token_id] = b
    return b

  def decode(token_ids: list[int]) -> str:
    out = bytearray()
    for token_id in token_ids:
      if token_id < n_special or token_id in special_ids: continue
      out += token_bytes(token_id)
    return out.decode("utf-8", errors="replace")

  return decode
