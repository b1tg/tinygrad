"""Voxtral Realtime 4B — tinygrad implementation.

Model: mistralai/Voxtral-Mini-4B-Realtime-2602
Ported from extra/voxtral.c/python_simple_implementation.py
"""
import math, sys
from tinygrad import Tensor, nn, dtypes

# ============================================================================
# Model Constants
# ============================================================================

# Encoder
ENC_DIM, ENC_LAYERS, ENC_HEADS, ENC_HEAD_DIM = 1280, 32, 32, 64
ENC_HIDDEN, ENC_KV_HEADS, ENC_WINDOW = 5120, 32, 750
ENC_NORM_EPS, ENC_ROPE_THETA = 1e-5, 1_000_000.0

# Decoder
DEC_DIM, DEC_LAYERS, DEC_HEADS, DEC_HEAD_DIM = 3072, 26, 32, 128
DEC_HIDDEN, DEC_KV_HEADS, DEC_WINDOW = 9216, 8, 8192
DEC_NORM_EPS, DEC_ROPE_THETA = 1e-5, 1_000_000.0
VOCAB_SIZE = 131072

# Audio
SAMPLE_RATE, FRAME_RATE, NUM_MEL_BINS = 16000, 12.5, 128
HOP_LENGTH, WINDOW_SIZE = 160, 400
GLOBAL_LOG_MEL_MAX = 1.5
DOWNSAMPLE_FACTOR = 4

# Ada norm
ADA_NORM_DIM = 32

# Streaming
N_LEFT_PAD_TOKENS = 32
TRANSCRIPTION_DELAY_MS = 480
RAW_AUDIO_LENGTH_PER_TOK = int(SAMPLE_RATE // FRAME_RATE)  # 1280
AUDIO_LENGTH_PER_TOK = RAW_AUDIO_LENGTH_PER_TOK // HOP_LENGTH  # 8

# Special tokens
TOKEN_BOS, TOKEN_EOS = 1, 2
TOKEN_STREAMING_PAD = 32

def num_audio_tokens(audio_len):
  if audio_len % HOP_LENGTH != 0: audio_len = math.ceil(audio_len / HOP_LENGTH - 1)
  else: audio_len = audio_len // HOP_LENGTH
  return math.ceil(audio_len / AUDIO_LENGTH_PER_TOK)

N_DELAY_TOKENS = num_audio_tokens(int(TRANSCRIPTION_DELAY_MS / 1000.0 * SAMPLE_RATE))  # 6
N_RIGHT_PAD_TOKENS = (N_DELAY_TOKENS + 1) + 10  # 17

# ============================================================================
# RoPE
# ============================================================================

def compute_rope_freqs(positions: Tensor, head_dim: int, theta: float):
  freqs = 1.0 / (theta ** (Tensor.arange(0, head_dim, 2).float() / head_dim))
  angles = positions.float().unsqueeze(-1) * freqs.unsqueeze(0)
  return angles.cos(), angles.sin()

def apply_rope_interleaved(x: Tensor, cos_f: Tensor, sin_f: Tensor, n_heads: int, head_dim: int):
  """Interleaved (GPT-J style) RoPE: pairs (0,1), (2,3), ..."""
  seq_len = x.shape[0]
  x = x.reshape(seq_len, n_heads, head_dim)
  cos_f = cos_f.unsqueeze(1)
  sin_f = sin_f.unsqueeze(1)
  x1 = x[..., ::2]   # even indices
  x2 = x[..., 1::2]  # odd indices
  o1 = x1 * cos_f - x2 * sin_f
  o2 = x2 * cos_f + x1 * sin_f
  out = Tensor.stack(o1, o2, dim=-1).flatten(-2)
  return out.reshape(seq_len, n_heads * head_dim)

# ============================================================================
# Attention
# ============================================================================

def causal_attention(q: Tensor, k: Tensor, v: Tensor, n_heads: int, n_kv_heads: int, head_dim: int, window: int,
                     q_start_pos: int = 0, kv_start_pos: int = 0):
  seq_q, seq_kv = q.shape[0], k.shape[0]
  q = q.reshape(seq_q, n_heads, head_dim).permute(1, 0, 2).unsqueeze(0)      # [1, nh, sq, hd]
  k = k.reshape(seq_kv, n_kv_heads, head_dim).permute(1, 0, 2).unsqueeze(0)  # [1, nkv, skv, hd]
  v = v.reshape(seq_kv, n_kv_heads, head_dim).permute(1, 0, 2).unsqueeze(0)
  # GQA: repeat KV heads
  gqa_ratio = n_heads // n_kv_heads
  if gqa_ratio > 1:
    k = k.repeat_interleave(gqa_ratio, dim=1)
    v = v.repeat_interleave(gqa_ratio, dim=1)
  # Causal mask with sliding window
  qi_abs = (q_start_pos + Tensor.arange(seq_q)).unsqueeze(1)
  kv_abs = (kv_start_pos + Tensor.arange(seq_kv)).unsqueeze(0)
  mask = (kv_abs <= qi_abs) * (kv_abs >= (qi_abs - (window - 1)))
  attn_mask = mask.where(Tensor.zeros(1), Tensor.full((1,), float("-inf")))
  out = q.float().scaled_dot_product_attention(k.float(), v.float(), attn_mask=attn_mask.unsqueeze(0).unsqueeze(0))
  return out.squeeze(0).permute(1, 0, 2).reshape(seq_q, n_heads * head_dim)

# ============================================================================
# Causal Conv1d
# ============================================================================

def causal_conv1d(x: Tensor, weight: Tensor, bias: Tensor, stride: int):
  """x: [1, C_in, L], weight: [C_out, C_in, K], returns [1, C_out, L']"""
  kernel_size = weight.shape[-1]
  padding_total = kernel_size - stride
  n_frames = (x.shape[-1] - kernel_size + padding_total) / stride + 1
  target_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
  extra_padding = int(target_length - x.shape[-1])
  x = x.pad((None, None, (padding_total, extra_padding)))
  return x.conv2d(weight, bias, stride=stride)

# ============================================================================
# Time Embedding
# ============================================================================

def compute_time_embedding(t_value: float, dim: int, theta: float = 10000.0):
  half_dim = dim // 2
  inv_freq = (-math.log(theta) * Tensor.arange(half_dim).float() / half_dim).exp()
  emb = t_value * inv_freq
  return emb.cos().cat(emb.sin())

# ============================================================================
# Encoder
# ============================================================================

class EncoderLayer:
  def __init__(self):
    self.attention_wq = nn.Linear(ENC_DIM, ENC_HEADS * ENC_HEAD_DIM, bias=True)
    self.attention_wk = nn.Linear(ENC_DIM, ENC_KV_HEADS * ENC_HEAD_DIM, bias=False)
    self.attention_wv = nn.Linear(ENC_DIM, ENC_KV_HEADS * ENC_HEAD_DIM, bias=True)
    self.attention_wo = nn.Linear(ENC_HEADS * ENC_HEAD_DIM, ENC_DIM, bias=True)
    self.attention_norm = nn.RMSNorm(ENC_DIM, ENC_NORM_EPS)
    self.feed_forward_w1 = nn.Linear(ENC_DIM, ENC_HIDDEN, bias=False)
    self.feed_forward_w2 = nn.Linear(ENC_HIDDEN, ENC_DIM, bias=True)
    self.feed_forward_w3 = nn.Linear(ENC_DIM, ENC_HIDDEN, bias=False)
    self.ffn_norm = nn.RMSNorm(ENC_DIM, ENC_NORM_EPS)

  def __call__(self, h: Tensor, rope_cos: Tensor, rope_sin: Tensor):
    x_norm = self.attention_norm(h)
    q = self.attention_wq(x_norm)
    k = self.attention_wk(x_norm)
    v = self.attention_wv(x_norm)
    q = apply_rope_interleaved(q, rope_cos, rope_sin, ENC_HEADS, ENC_HEAD_DIM)
    k = apply_rope_interleaved(k, rope_cos, rope_sin, ENC_KV_HEADS, ENC_HEAD_DIM)
    attn_out = causal_attention(q, k, v, ENC_HEADS, ENC_KV_HEADS, ENC_HEAD_DIM, ENC_WINDOW)
    h = h + self.attention_wo(attn_out)
    x_norm = self.ffn_norm(h)
    gate = self.feed_forward_w1(x_norm).silu()
    up = self.feed_forward_w3(x_norm)
    h = h + self.feed_forward_w2(gate * up)
    return h

class Encoder:
  def __init__(self):
    self.conv_layers_0_conv = nn.Conv1d(NUM_MEL_BINS, ENC_DIM, 3, stride=1, bias=True)
    self.conv_layers_1_conv = nn.Conv1d(ENC_DIM, ENC_DIM, 3, stride=2, bias=True)
    self.transformer_layers = [EncoderLayer() for _ in range(ENC_LAYERS)]
    self.transformer_norm = nn.RMSNorm(ENC_DIM, ENC_NORM_EPS)

  def __call__(self, mel: Tensor):
    """mel: [128, frames] -> [seq, 1280]"""
    mel_3d = mel.unsqueeze(0)
    h = causal_conv1d(mel_3d, self.conv_layers_0_conv.weight, self.conv_layers_0_conv.bias, stride=1).gelu()
    h = causal_conv1d(h, self.conv_layers_1_conv.weight, self.conv_layers_1_conv.bias, stride=2).gelu()
    h = h.squeeze(0).permute(1, 0)  # [seq, 1280]
    conv_len = h.shape[0]
    trunc = conv_len % DOWNSAMPLE_FACTOR
    if trunc > 0: h = h[trunc:]
    seq_len = h.shape[0]
    print(f"  Conv stem: {mel.shape[1]} frames -> {conv_len}, left-trunc {trunc} -> {seq_len}", file=sys.stderr)
    positions = Tensor.arange(seq_len)
    rope_cos, rope_sin = compute_rope_freqs(positions, ENC_HEAD_DIM, ENC_ROPE_THETA)
    for i, layer in enumerate(self.transformer_layers):
      h = layer(h, rope_cos, rope_sin)
      if (i + 1) % 8 == 0: print(f"  Encoder layer {i+1}/{ENC_LAYERS}", file=sys.stderr)
    h = self.transformer_norm(h)
    return h

# ============================================================================
# Adapter
# ============================================================================

class Adapter:
  def __init__(self):
    self.audio_language_projection_0 = nn.Linear(ENC_DIM * DOWNSAMPLE_FACTOR, DEC_DIM, bias=False)
    self.audio_language_projection_2 = nn.Linear(DEC_DIM, DEC_DIM, bias=False)

  def __call__(self, enc_out: Tensor):
    """enc_out: [seq, 1280] -> [seq/4, 3072]"""
    seq_len = enc_out.shape[0]
    ds = enc_out.reshape(seq_len // DOWNSAMPLE_FACTOR, ENC_DIM * DOWNSAMPLE_FACTOR)
    out = self.audio_language_projection_0(ds).gelu()
    out = self.audio_language_projection_2(out)
    print(f"  Adapter: {seq_len} -> {out.shape[0]} (downsample {DOWNSAMPLE_FACTOR}x)", file=sys.stderr)
    return out

# ============================================================================
# Decoder
# ============================================================================

class DecoderLayer:
  def __init__(self):
    self.attention_wq = nn.Linear(DEC_DIM, DEC_HEADS * DEC_HEAD_DIM, bias=False)
    self.attention_wk = nn.Linear(DEC_DIM, DEC_KV_HEADS * DEC_HEAD_DIM, bias=False)
    self.attention_wv = nn.Linear(DEC_DIM, DEC_KV_HEADS * DEC_HEAD_DIM, bias=False)
    self.attention_wo = nn.Linear(DEC_HEADS * DEC_HEAD_DIM, DEC_DIM, bias=False)
    self.attention_norm = nn.RMSNorm(DEC_DIM, DEC_NORM_EPS)
    self.feed_forward_w1 = nn.Linear(DEC_DIM, DEC_HIDDEN, bias=False)
    self.feed_forward_w2 = nn.Linear(DEC_HIDDEN, DEC_DIM, bias=False)
    self.feed_forward_w3 = nn.Linear(DEC_DIM, DEC_HIDDEN, bias=False)
    self.ffn_norm = nn.RMSNorm(DEC_DIM, DEC_NORM_EPS)
    self.ada_rms_norm_t_cond_0 = nn.Linear(DEC_DIM, ADA_NORM_DIM, bias=False)
    self.ada_rms_norm_t_cond_2 = nn.Linear(ADA_NORM_DIM, DEC_DIM, bias=False)

  def __call__(self, h: Tensor, pos: int, full_k: Tensor, full_v: Tensor, t_cond: Tensor | None = None):
    seq_len = h.shape[0]
    x_norm = self.attention_norm(h)
    q = self.attention_wq(x_norm)
    k = self.attention_wk(x_norm)
    v = self.attention_wv(x_norm)
    positions = Tensor.arange(pos, pos + seq_len)
    rope_cos, rope_sin = compute_rope_freqs(positions, DEC_HEAD_DIM, DEC_ROPE_THETA)
    q = apply_rope_interleaved(q.float(), rope_cos, rope_sin, DEC_HEADS, DEC_HEAD_DIM)
    k = apply_rope_interleaved(k.float(), rope_cos, rope_sin, DEC_KV_HEADS, DEC_HEAD_DIM)
    # Update KV cache
    full_k = full_k.cat(k, dim=0) if full_k.shape[0] > 0 else k
    full_v = full_v.cat(v, dim=0) if full_v.shape[0] > 0 else v
    if full_k.shape[0] > DEC_WINDOW:
      full_k = full_k[-DEC_WINDOW:]
      full_v = full_v[-DEC_WINDOW:]
    kv_start_pos = (pos + seq_len - 1) - (full_k.shape[0] - 1)
    attn_out = causal_attention(q, full_k, full_v, DEC_HEADS, DEC_KV_HEADS, DEC_HEAD_DIM, DEC_WINDOW,
                                q_start_pos=pos, kv_start_pos=kv_start_pos)
    h = h + self.attention_wo(attn_out)
    h_norm = self.ffn_norm(h)
    if t_cond is not None:
      ada_hidden = self.ada_rms_norm_t_cond_0(t_cond).gelu()
      ada_scale = self.ada_rms_norm_t_cond_2(ada_hidden)
      h_norm = h_norm * (1 + ada_scale.unsqueeze(0))
    gate = self.feed_forward_w1(h_norm).silu()
    up = self.feed_forward_w3(h_norm)
    h = h + self.feed_forward_w2(gate * up)
    return h, full_k, full_v

class Decoder:
  def __init__(self):
    self.tok_embeddings = nn.Embedding(VOCAB_SIZE, DEC_DIM)
    self.layers = [DecoderLayer() for _ in range(DEC_LAYERS)]
    self.norm = nn.RMSNorm(DEC_DIM, DEC_NORM_EPS)
    self.kv_cache: list[tuple[Tensor, Tensor]] | None = None

  def reset_cache(self):
    kv_dim = DEC_KV_HEADS * DEC_HEAD_DIM
    self.kv_cache = [(Tensor.zeros(0, kv_dim), Tensor.zeros(0, kv_dim)) for _ in range(DEC_LAYERS)]

  def embed_token(self, token_id: int):
    return self.tok_embeddings.weight[token_id]

  def embed_tokens(self, token_ids: list[int]):
    return self.tok_embeddings(Tensor(token_ids, dtype=dtypes.int))

  def prefill(self, input_embeds: Tensor, t_cond: Tensor):
    self.reset_cache()
    assert self.kv_cache is not None
    h = input_embeds
    for i, layer in enumerate(self.layers):
      k_cache, v_cache = self.kv_cache[i]
      h, k_cache, v_cache = layer(h, 0, k_cache, v_cache, t_cond)
      self.kv_cache[i] = (k_cache, v_cache)
      if i < 4 or (i + 1) % 8 == 0:
        print(f"  Decoder prefill layer {i+1}/{DEC_LAYERS}", file=sys.stderr)
    return h

  def forward_one(self, embed: Tensor, pos: int, t_cond: Tensor):
    assert self.kv_cache is not None
    h = embed.unsqueeze(0) if embed.ndim == 1 else embed
    for i, layer in enumerate(self.layers):
      k_cache, v_cache = self.kv_cache[i]
      h, k_cache, v_cache = layer(h, pos, k_cache, v_cache, t_cond)
      self.kv_cache[i] = (k_cache, v_cache)
    h = self.norm(h)
    logits = h.float().squeeze(0) @ self.tok_embeddings.weight.float().T
    return logits

# ============================================================================
# Weight loading
# ============================================================================

def load_voxtral(model_dir: str):
  """Load model from directory containing consolidated.safetensors."""
  import os
  sf_path = os.path.join(model_dir, "consolidated.safetensors")
  print(f"Loading model from {sf_path}", file=sys.stderr)
  state_dict = nn.state.safe_load(sf_path)

  encoder = Encoder()
  adapter = Adapter()
  decoder = Decoder()

  enc_prefix = "mm_streams_embeddings.embedding_module.whisper_encoder"
  ada_prefix = "mm_streams_embeddings.embedding_module"
  tok_prefix = "mm_streams_embeddings.embedding_module"

  # Map safetensors names to model attributes
  enc_map = {}
  for i in range(ENC_LAYERS):
    sp = f"{enc_prefix}.transformer.layers.{i}"
    dp = f"transformer_layers.{i}"
    enc_map.update({
      f"{sp}.attention.wq.weight": f"{dp}.attention_wq.weight",
      f"{sp}.attention.wq.bias": f"{dp}.attention_wq.bias",
      f"{sp}.attention.wk.weight": f"{dp}.attention_wk.weight",
      f"{sp}.attention.wv.weight": f"{dp}.attention_wv.weight",
      f"{sp}.attention.wv.bias": f"{dp}.attention_wv.bias",
      f"{sp}.attention.wo.weight": f"{dp}.attention_wo.weight",
      f"{sp}.attention.wo.bias": f"{dp}.attention_wo.bias",
      f"{sp}.attention_norm.weight": f"{dp}.attention_norm.weight",
      f"{sp}.feed_forward.w1.weight": f"{dp}.feed_forward_w1.weight",
      f"{sp}.feed_forward.w2.weight": f"{dp}.feed_forward_w2.weight",
      f"{sp}.feed_forward.w2.bias": f"{dp}.feed_forward_w2.bias",
      f"{sp}.feed_forward.w3.weight": f"{dp}.feed_forward_w3.weight",
      f"{sp}.ffn_norm.weight": f"{dp}.ffn_norm.weight",
    })
  enc_map[f"{enc_prefix}.conv_layers.0.conv.weight"] = "conv_layers_0_conv.weight"
  enc_map[f"{enc_prefix}.conv_layers.0.conv.bias"] = "conv_layers_0_conv.bias"
  enc_map[f"{enc_prefix}.conv_layers.1.conv.weight"] = "conv_layers_1_conv.weight"
  enc_map[f"{enc_prefix}.conv_layers.1.conv.bias"] = "conv_layers_1_conv.bias"
  enc_map[f"{enc_prefix}.transformer.norm.weight"] = "transformer_norm.weight"

  enc_state = {enc_map[k]: v for k, v in state_dict.items() if k in enc_map}
  nn.state.load_state_dict(encoder, enc_state, strict=True, verbose=False)
  print("  Encoder weights loaded", file=sys.stderr)

  ada_state = {
    "audio_language_projection_0.weight": state_dict[f"{ada_prefix}.audio_language_projection.0.weight"],
    "audio_language_projection_2.weight": state_dict[f"{ada_prefix}.audio_language_projection.2.weight"],
  }
  nn.state.load_state_dict(adapter, ada_state, strict=True, verbose=False)
  print("  Adapter weights loaded", file=sys.stderr)

  dec_map = {}
  for i in range(DEC_LAYERS):
    sp = f"layers.{i}"
    dp = f"layers.{i}"
    dec_map.update({
      f"{sp}.attention_norm.weight": f"{dp}.attention_norm.weight",
      f"{sp}.attention.wq.weight": f"{dp}.attention_wq.weight",
      f"{sp}.attention.wk.weight": f"{dp}.attention_wk.weight",
      f"{sp}.attention.wv.weight": f"{dp}.attention_wv.weight",
      f"{sp}.attention.wo.weight": f"{dp}.attention_wo.weight",
      f"{sp}.ffn_norm.weight": f"{dp}.ffn_norm.weight",
      f"{sp}.feed_forward.w1.weight": f"{dp}.feed_forward_w1.weight",
      f"{sp}.feed_forward.w2.weight": f"{dp}.feed_forward_w2.weight",
      f"{sp}.feed_forward.w3.weight": f"{dp}.feed_forward_w3.weight",
      f"{sp}.ada_rms_norm_t_cond.0.weight": f"{dp}.ada_rms_norm_t_cond_0.weight",
      f"{sp}.ada_rms_norm_t_cond.2.weight": f"{dp}.ada_rms_norm_t_cond_2.weight",
    })
  dec_map[f"{tok_prefix}.tok_embeddings.weight"] = "tok_embeddings.weight"
  dec_map["norm.weight"] = "norm.weight"

  dec_state = {dec_map[k]: v for k, v in state_dict.items() if k in dec_map}
  nn.state.load_state_dict(decoder, dec_state, strict=True, verbose=False)
  print("  Decoder weights loaded", file=sys.stderr)

  return encoder, adapter, decoder
