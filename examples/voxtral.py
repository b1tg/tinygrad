#!/usr/bin/env python3
# Voxtral Realtime 4B speech-to-text — mistralai/Voxtral-Mini-4B-Realtime-2602
import sys, os, math, struct, json, base64, functools, time, argparse, subprocess
from tinygrad import Tensor, nn, dtypes, UOp, TinyJit

SAMPLE_RATE, HOP_LENGTH, WINDOW_SIZE, NUM_MEL_BINS = 16000, 160, 400, 128
DOWNSAMPLE_FACTOR, VOCAB_SIZE, ADA_NORM_DIM = 4, 131072, 32
ENC_DIM, ENC_LAYERS, ENC_HEADS, ENC_HEAD_DIM, ENC_HIDDEN, ENC_KV_HEADS, ENC_WINDOW = 1280, 32, 32, 64, 5120, 32, 750
DEC_DIM, DEC_LAYERS, DEC_HEADS, DEC_HEAD_DIM, DEC_HIDDEN, DEC_KV_HEADS = 3072, 26, 32, 128, 9216, 8
NORM_EPS, ROPE_THETA, MAX_CONTEXT = 1e-5, 1_000_000.0, 512
TOKEN_BOS, TOKEN_EOS, TOKEN_PAD = 1, 2, 32
N_LEFT_PAD, N_DELAY, N_RIGHT_PAD, RAW_TOK_LEN = 32, 6, 17, 1280

@functools.cache
def precompute_rope(head_dim: int, max_len: int, theta: float = ROPE_THETA):
  freqs = 1.0 / (theta ** (Tensor.arange(0, head_dim, 2).float() / head_dim))
  angles = Tensor.arange(max_len).float().unsqueeze(-1) * freqs.unsqueeze(0)
  return angles.cos().contiguous(), angles.sin().contiguous()

def apply_rope(x: Tensor, cos_f: Tensor, sin_f: Tensor, n_heads: int, head_dim: int):
  B, T, _ = x.shape
  x = x.reshape(B, T, n_heads, head_dim)
  cos_f, sin_f = cos_f.reshape(1, T, 1, head_dim // 2), sin_f.reshape(1, T, 1, head_dim // 2)
  x1, x2 = x[..., ::2], x[..., 1::2]
  return Tensor.stack(x1 * cos_f - x2 * sin_f, x2 * cos_f + x1 * sin_f, dim=-1).flatten(-2).reshape(B, T, n_heads * head_dim)

def causal_conv1d(x: Tensor, weight: Tensor, bias: Tensor, stride: int):
  K, pad = weight.shape[-1], weight.shape[-1] - stride
  target = (math.ceil((x.shape[-1] - K + pad) / stride + 1) - 1) * stride + K - pad
  return x.pad((None, None, (pad, int(target - x.shape[-1])))).conv2d(weight, bias, stride=stride)

class EncoderLayer:
  def __init__(self):
    self.attention_wq = nn.Linear(ENC_DIM, ENC_HEADS * ENC_HEAD_DIM, bias=True)
    self.attention_wk = nn.Linear(ENC_DIM, ENC_KV_HEADS * ENC_HEAD_DIM, bias=False)
    self.attention_wv = nn.Linear(ENC_DIM, ENC_KV_HEADS * ENC_HEAD_DIM, bias=True)
    self.attention_wo = nn.Linear(ENC_HEADS * ENC_HEAD_DIM, ENC_DIM, bias=True)
    self.attention_norm = nn.RMSNorm(ENC_DIM, NORM_EPS)
    self.feed_forward_w1, self.feed_forward_w3 = nn.Linear(ENC_DIM, ENC_HIDDEN, bias=False), nn.Linear(ENC_DIM, ENC_HIDDEN, bias=False)
    self.feed_forward_w2 = nn.Linear(ENC_HIDDEN, ENC_DIM, bias=True)
    self.ffn_norm = nn.RMSNorm(ENC_DIM, NORM_EPS)

  def __call__(self, h: Tensor, rope_cos: Tensor, rope_sin: Tensor, k_cache: Tensor | None, v_cache: Tensor | None):
    x = self.attention_norm(h)
    q = apply_rope(self.attention_wq(x), rope_cos, rope_sin, ENC_HEADS, ENC_HEAD_DIM)
    k = apply_rope(self.attention_wk(x), rope_cos, rope_sin, ENC_KV_HEADS, ENC_HEAD_DIM)
    v = self.attention_wv(x)
    C = q.shape[1]
    q = q.reshape(1, C, ENC_HEADS, ENC_HEAD_DIM).transpose(1, 2)
    k_new = k.reshape(1, C, ENC_KV_HEADS, ENC_HEAD_DIM).transpose(1, 2)
    v_new = v.reshape(1, C, ENC_KV_HEADS, ENC_HEAD_DIM).transpose(1, 2)
    if k_cache is not None:
      k_all, v_all = k_cache.cat(k_new, dim=2), v_cache.cat(v_new, dim=2)
    else:
      k_all, v_all = k_new, v_new
    cache_len = k_all.shape[2] - C
    mask = Tensor.full((1, 1, C, k_all.shape[2]), float("-inf")).triu(cache_len + 1)
    attn = q.scaled_dot_product_attention(k_all, v_all, attn_mask=mask)
    h = h + self.attention_wo(attn.transpose(1, 2).reshape(1, C, -1))
    x = self.ffn_norm(h)
    h = (h + self.feed_forward_w2(self.feed_forward_w1(x).silu() * self.feed_forward_w3(x))).realize()
    if k_all.shape[2] > ENC_WINDOW:
      d = k_all.shape[2] - ENC_WINDOW
      k_all, v_all = k_all[:, :, d:, :], v_all[:, :, d:, :]
    return h, k_all.contiguous().realize(), v_all.contiguous().realize()

class Encoder:
  def __init__(self):
    self.conv_layers_0_conv = nn.Conv1d(NUM_MEL_BINS, ENC_DIM, 3, stride=1, bias=True)
    self.conv_layers_1_conv = nn.Conv1d(ENC_DIM, ENC_DIM, 3, stride=2, bias=True)
    self.transformer_layers = [EncoderLayer() for _ in range(ENC_LAYERS)]
    self.transformer_norm = nn.RMSNorm(ENC_DIM, NORM_EPS)

  def __call__(self, mel: Tensor):
    h = causal_conv1d(mel.unsqueeze(0), self.conv_layers_0_conv.weight, self.conv_layers_0_conv.bias, stride=1).gelu()
    h = causal_conv1d(h, self.conv_layers_1_conv.weight, self.conv_layers_1_conv.bias, stride=2).gelu()
    h = h.permute(0, 2, 1)
    trunc = h.shape[1] % DOWNSAMPLE_FACTOR
    if trunc > 0: h = h[:, trunc:]
    S = h.shape[1]
    rope_cos, rope_sin = precompute_rope(ENC_HEAD_DIM, S)
    for layer in self.transformer_layers:
      chunks, k_cache, v_cache = [], None, None
      for start in range(0, S, ENC_WINDOW):
        end = min(start + ENC_WINDOW, S)
        h_chunk, k_cache, v_cache = layer(h[:, start:end], rope_cos[start:end], rope_sin[start:end], k_cache, v_cache)
        chunks.append(h_chunk)
      h = Tensor.cat(*chunks, dim=1)
    return self.transformer_norm(h.squeeze(0))

class Adapter:
  def __init__(self):
    self.audio_language_projection_0 = nn.Linear(ENC_DIM * DOWNSAMPLE_FACTOR, DEC_DIM, bias=False)
    self.audio_language_projection_2 = nn.Linear(DEC_DIM, DEC_DIM, bias=False)
  def __call__(self, enc_out: Tensor):
    return self.audio_language_projection_2(self.audio_language_projection_0(
      enc_out.reshape(enc_out.shape[0] // DOWNSAMPLE_FACTOR, ENC_DIM * DOWNSAMPLE_FACTOR)).gelu())

class DecoderLayer:
  def __init__(self, max_context: int):
    self.attention_wq = nn.Linear(DEC_DIM, DEC_HEADS * DEC_HEAD_DIM, bias=False)
    self.attention_wk = nn.Linear(DEC_DIM, DEC_KV_HEADS * DEC_HEAD_DIM, bias=False)
    self.attention_wv = nn.Linear(DEC_DIM, DEC_KV_HEADS * DEC_HEAD_DIM, bias=False)
    self.attention_wo = nn.Linear(DEC_HEADS * DEC_HEAD_DIM, DEC_DIM, bias=False)
    self.attention_norm = nn.RMSNorm(DEC_DIM, NORM_EPS)
    self.feed_forward_w1, self.feed_forward_w3 = nn.Linear(DEC_DIM, DEC_HIDDEN, bias=False), nn.Linear(DEC_DIM, DEC_HIDDEN, bias=False)
    self.feed_forward_w2 = nn.Linear(DEC_HIDDEN, DEC_DIM, bias=False)
    self.ffn_norm = nn.RMSNorm(DEC_DIM, NORM_EPS)
    self.ada_rms_norm_t_cond_0 = nn.Linear(DEC_DIM, ADA_NORM_DIM, bias=False)
    self.ada_rms_norm_t_cond_2 = nn.Linear(ADA_NORM_DIM, DEC_DIM, bias=False)
    self.max_context = max_context

  def __call__(self, x: Tensor, start_pos):
    B, T, _ = x.shape
    x_norm = self.attention_norm(x)
    q, k, v = self.attention_wq(x_norm), self.attention_wk(x_norm), self.attention_wv(x_norm)
    rope_cos, rope_sin = precompute_rope(DEC_HEAD_DIM, self.max_context)
    q = apply_rope(q, rope_cos[start_pos:start_pos+T], rope_sin[start_pos:start_pos+T], DEC_HEADS, DEC_HEAD_DIM)
    k = apply_rope(k, rope_cos[start_pos:start_pos+T], rope_sin[start_pos:start_pos+T], DEC_KV_HEADS, DEC_HEAD_DIM)
    q = q.reshape(B, T, DEC_HEADS, DEC_HEAD_DIM).transpose(1, 2)
    k = k.reshape(B, T, DEC_KV_HEADS, DEC_HEAD_DIM).transpose(1, 2)
    v = v.reshape(B, T, DEC_KV_HEADS, DEC_HEAD_DIM).transpose(1, 2)
    if not hasattr(self, "cache_kv"):
      self.cache_kv = Tensor.zeros(2, B, DEC_KV_HEADS, self.max_context, DEC_HEAD_DIM, dtype=k.dtype).contiguous().realize()
    self.cache_kv[:, :, :, start_pos:start_pos+T, :].assign(Tensor.stack(k, v)).realize()
    keys, vals = self.cache_kv[0, :, :, :start_pos+T, :], self.cache_kv[1, :, :, :start_pos+T, :]
    mask = Tensor.full((1, 1, T, start_pos+T), float("-inf"), dtype=x.dtype).triu(int(start_pos)+1) if T > 1 else None
    attn = q.scaled_dot_product_attention(keys, vals, attn_mask=mask, enable_gqa=True).transpose(1, 2).reshape(B, T, -1)
    h = x + self.attention_wo(attn)
    h_norm = self.ffn_norm(h) * self.ada_scale
    return (h + self.feed_forward_w2(self.feed_forward_w1(h_norm).silu() * self.feed_forward_w3(h_norm))).contiguous()

class Decoder:
  def __init__(self, max_context: int = MAX_CONTEXT):
    self.tok_embeddings = nn.Embedding(VOCAB_SIZE, DEC_DIM)
    self.layers = [DecoderLayer(max_context) for _ in range(DEC_LAYERS)]
    self.norm = nn.RMSNorm(DEC_DIM, NORM_EPS)
    self.max_context, self.forward_jit = max_context, TinyJit(self._decode_step)

  def precompute_ada(self, t_value: float):
    inv_freq = (-math.log(10000.0) * Tensor.arange(DEC_DIM // 2).float() / (DEC_DIM // 2)).exp()
    emb = t_value * inv_freq
    t_cond = emb.cos().cat(emb.sin())
    for layer in self.layers:
      layer.ada_scale = (1 + layer.ada_rms_norm_t_cond_2(layer.ada_rms_norm_t_cond_0(t_cond).gelu()).reshape(1, 1, DEC_DIM)).realize()

  def _run_layers(self, h: Tensor, start_pos) -> Tensor:
    for layer in self.layers: h = layer(h, start_pos)
    return (self.norm(h)[:, -1, :] @ self.tok_embeddings.weight.T).argmax(-1, keepdim=True)

  def _decode_step(self, ada_out: Tensor, token_id: Tensor, start_pos) -> Tensor:
    h = (ada_out[start_pos:start_pos+1] + self.tok_embeddings(token_id)).reshape(1, 1, DEC_DIM)
    return self._run_layers(h, start_pos).realize()

  def prefill(self, input_embeds: Tensor):
    h = input_embeds.unsqueeze(0)
    for layer in self.layers:
      h = layer(h, 0)
      h = h.realize()
    return h

  def decode_first(self, embed: Tensor, pos: int) -> int:
    return int(self._run_layers(embed.reshape(1, 1, DEC_DIM), pos).item())

# mel spectrogram (Slaney-style, matching mistral_common)
def _h2m(f): return 3*f/200 if f < 1000 else 15 + 27/math.log(6.4) * math.log(f/1000)
def _m2h(m): return 200*m/3 if m < 15 else 1000 * math.exp(math.log(6.4)/27 * (m-15))

def compute_mel_filters():
  n_freq = 1 + WINDOW_SIZE // 2
  fft_f = [i * (SAMPLE_RATE // 2) / (n_freq - 1) for i in range(n_freq)]
  mel_min, mel_max = _h2m(0), _h2m(8000)
  ff = [_m2h(mel_min + i * (mel_max - mel_min) / (NUM_MEL_BINS + 1)) for i in range(NUM_MEL_BINS + 2)]
  fd = [ff[i+1] - ff[i] for i in range(len(ff)-1)]
  return [[max(0, min((fft_f[f] - ff[m]) / fd[m], (ff[m+2] - fft_f[f]) / fd[m+1])) * 2 / (ff[m+2] - ff[m])
           for m in range(NUM_MEL_BINS)] for f in range(n_freq)]

@functools.cache
def _mel_basis():
  n_freq = 1 + WINDOW_SIZE // 2
  mel_filters = compute_mel_filters()
  window = 0.5 * (1.0 - (2 * math.pi / WINDOW_SIZE * Tensor.arange(WINDOW_SIZE).float()).cos())
  angles = (2 * math.pi / WINDOW_SIZE) * Tensor.arange(n_freq).float().unsqueeze(1) * Tensor.arange(WINDOW_SIZE).float().unsqueeze(0)
  return Tensor(mel_filters).contiguous(), window.contiguous(), angles.cos().contiguous(), angles.sin().contiguous()

def compute_mel_spectrogram(audio: Tensor):
  mel_filters, window, dft_cos, dft_sin = _mel_basis()
  n_fft, hop, n_freq = WINDOW_SIZE, HOP_LENGTH, 1 + WINDOW_SIZE // 2
  audio = audio.pad(((n_fft // 2, n_fft // 2),))
  n_frames = 1 + (audio.shape[0] - n_fft) // hop
  frames = audio[(Tensor.arange(n_frames).unsqueeze(1) * hop + Tensor.arange(n_fft).unsqueeze(0)).flatten()].reshape(n_frames, n_fft) * window
  real, imag = frames @ dft_cos.T, frames @ (-dft_sin).T
  mel_spec = ((real.square() + imag.square())[:-1] @ mel_filters).T
  log_spec = mel_spec.clamp(min_=1e-10).log2() * math.log10(2)
  return (log_spec.maximum(Tensor.full(log_spec.shape, -6.5)) + 4.0) / 4.0

def load_audio(path: str):
  result = subprocess.run(["ffmpeg", "-i", path, "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-v", "quiet", "-"], capture_output=True)
  if result.returncode != 0: raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()}")
  return list(struct.unpack(f"<{len(result.stdout)//4}f", result.stdout))

def load_tokenizer(model_dir: str):
  with open(os.path.join(model_dir, "tekken.json")) as f: data = json.load(f)
  vocab, n_special = data["vocab"], int(data.get("config", {}).get("default_num_special_tokens", 1000))
  special_ids = {int(st["rank"]) for st in data.get("special_tokens", []) if "rank" in st}
  cache: dict[int, bytes] = {}
  def decode(tids):
    out = bytearray()
    for t in tids:
      if t < n_special or t in special_ids: continue
      if t not in cache: cache[t] = base64.b64decode(vocab[t - n_special]["token_bytes"]) if 0 <= t - n_special < len(vocab) else b""
      out += cache[t]
    return out.decode("utf-8", errors="replace")
  return decode

def load_voxtral(model_dir: str, max_context: int = MAX_CONTEXT):
  sd = nn.state.safe_load(os.path.join(model_dir, "consolidated.safetensors"))
  encoder, adapter, decoder = Encoder(), Adapter(), Decoder(max_context)
  EP, AP = "mm_streams_embeddings.embedding_module.whisper_encoder.", "mm_streams_embeddings.embedding_module."
  def r(k):
    for p in ["transformer.layers", "transformer.norm", "conv_layers.0.conv", "conv_layers.1.conv"]: k = k.replace(p, p.replace(".", "_"))
    for p in ["attention", "feed_forward", "t_cond", "projection"]: k = k.replace(p + ".", p + "_")
    return k
  nn.state.load_state_dict(encoder, {r(k[len(EP):]): v for k, v in sd.items() if k.startswith(EP)}, strict=True, verbose=False)
  nn.state.load_state_dict(adapter, {r(k[len(AP):]): v for k, v in sd.items() if k.startswith(AP+"audio")}, strict=True, verbose=False)
  dec = {r(k): v for k, v in sd.items() if k.startswith("layers.") or k == "norm.weight"}
  dec["tok_embeddings.weight"] = sd[AP+"tok_embeddings.weight"]
  nn.state.load_state_dict(decoder, dec, strict=True, verbose=False)
  for m in [encoder, adapter, decoder]:
    for s in nn.state.get_parameters(m): s.replace(s.contiguous())
  return encoder, adapter, decoder

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Voxtral Realtime 4B transcription")
  parser.add_argument("model_dir", help="Model directory with consolidated.safetensors and tekken.json")
  parser.add_argument("audio_path", help="Audio file to transcribe (WAV, OGG, MP3, etc.)")
  args = parser.parse_args()

  audio = load_audio(args.audio_path)
  print(f"Audio: {len(audio)} samples ({len(audio)/SAMPLE_RATE:.1f}s)", file=sys.stderr)

  prompt_ids = [TOKEN_BOS] + [TOKEN_PAD] * (N_LEFT_PAD + N_DELAY)
  n = len(audio)
  padded = [0.0] * (N_LEFT_PAD * RAW_TOK_LEN) + audio + [0.0] * ((RAW_TOK_LEN - n % RAW_TOK_LEN) % RAW_TOK_LEN + N_RIGHT_PAD * RAW_TOK_LEN)

  mel = compute_mel_spectrogram(Tensor(padded))
  if mel.shape[1] % 2 != 0: mel = mel[:, 1:]
  print(f"Mel: {mel.shape[1]} frames", file=sys.stderr)

  t_load = time.time()
  encoder, adapter, decoder = load_voxtral(args.model_dir, max_context=max(mel.shape[1] // 8 + 64, 256))
  print(f"Model load: {time.time()-t_load:.1f}s", file=sys.stderr)
  Tensor.no_grad = True
  mel = mel.realize()

  t0 = time.time()
  ada_out = adapter(encoder(mel)).realize()
  t_enc = time.time()
  print(f"Encoder+Adapter: {t_enc-t0:.1f}s", file=sys.stderr)
  n_audio, L = ada_out.shape[0], len(prompt_ids)
  decoder.precompute_ada(float(N_DELAY))
  prefix_embeds = ada_out[:L] + decoder.tok_embeddings(Tensor(prompt_ids, dtype=dtypes.int))

  if L > 1: decoder.prefill(prefix_embeds[:-1])
  token = decoder.decode_first(prefix_embeds[-1], pos=L - 1)
  generated = [token]
  t_prefill = time.time()
  print(f"Prefill: {t_prefill-t_enc:.1f}s", file=sys.stderr)

  v = UOp.variable("start_pos", 1, decoder.max_context - 1)
  tokenize = load_tokenizer(args.model_dir)
  sys.stdout.write(tokenize([token])); sys.stdout.flush()
  for pos in range(L, n_audio):
    if token == TOKEN_EOS: break
    token_id = decoder.forward_jit(ada_out, Tensor([token], dtype=dtypes.int), v.bind(pos))
    token = int(token_id.item())
    generated.append(token)
    sys.stdout.write(tokenize([token])); sys.stdout.flush()
  print()

  t1 = time.time()
  if generated and generated[-1] == TOKEN_EOS: generated = generated[:-1]
  print(f"{len(generated)} tokens in {t1-t0:.1f}s ({len(generated)/(t1-t0):.1f} tok/s), decode: {t1-t_prefill:.1f}s ({len(generated)/(t1-t_prefill):.1f} tok/s)", file=sys.stderr)
