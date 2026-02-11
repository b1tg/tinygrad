"""Voxtral Realtime 4B — tinygrad implementation.

Model: mistralai/Voxtral-Mini-4B-Realtime-2602
Ported from extra/voxtral.c/python_simple_implementation.py
"""
import math, sys, functools
from tinygrad import Tensor, nn, dtypes, UOp, TinyJit

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

# Max context for KV cache (decoder only needs audio_tokens + some margin)
MAX_CONTEXT = 512

def num_audio_tokens(audio_len):
  if audio_len % HOP_LENGTH != 0: audio_len = math.ceil(audio_len / HOP_LENGTH - 1)
  else: audio_len = audio_len // HOP_LENGTH
  return math.ceil(audio_len / AUDIO_LENGTH_PER_TOK)

N_DELAY_TOKENS = num_audio_tokens(int(TRANSCRIPTION_DELAY_MS / 1000.0 * SAMPLE_RATE))  # 6
N_RIGHT_PAD_TOKENS = (N_DELAY_TOKENS + 1) + 10  # 17

# ============================================================================
# RoPE (precomputed, cached)
# ============================================================================

@functools.cache
def precompute_rope_interleaved(head_dim: int, max_len: int, theta: float):
  """Precompute interleaved RoPE cos/sin table. Returns [max_len, head_dim//2] cos and sin."""
  freqs = 1.0 / (theta ** (Tensor.arange(0, head_dim, 2).float() / head_dim))
  positions = Tensor.arange(max_len).float()
  angles = positions.unsqueeze(-1) * freqs.unsqueeze(0)  # [max_len, head_dim//2]
  return angles.cos().contiguous(), angles.sin().contiguous()

def apply_rope_interleaved(x: Tensor, cos_f: Tensor, sin_f: Tensor, n_heads: int, head_dim: int):
  """Interleaved (GPT-J style) RoPE: pairs (0,1), (2,3), ..."""
  B, T, _ = x.shape
  x = x.reshape(B, T, n_heads, head_dim)
  cos_f = cos_f.reshape(1, T, 1, head_dim // 2)
  sin_f = sin_f.reshape(1, T, 1, head_dim // 2)
  x1 = x[..., ::2]
  x2 = x[..., 1::2]
  o1 = x1 * cos_f - x2 * sin_f
  o2 = x2 * cos_f + x1 * sin_f
  return Tensor.stack(o1, o2, dim=-1).flatten(-2).reshape(B, T, n_heads * head_dim)

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
    # h is [seq, dim] for encoder — needs batch dim for rope
    q = self.attention_wq(x_norm).unsqueeze(0)
    k = self.attention_wk(x_norm).unsqueeze(0)
    v = self.attention_wv(x_norm).unsqueeze(0)
    q = apply_rope_interleaved(q, rope_cos, rope_sin, ENC_HEADS, ENC_HEAD_DIM).squeeze(0)
    k = apply_rope_interleaved(k, rope_cos, rope_sin, ENC_KV_HEADS, ENC_HEAD_DIM).squeeze(0)
    # Attention (no sliding window for encoder, use full causal)
    seq_len = q.shape[0]
    q = q.reshape(seq_len, ENC_HEADS, ENC_HEAD_DIM).permute(1, 0, 2).unsqueeze(0)
    k = k.reshape(seq_len, ENC_KV_HEADS, ENC_HEAD_DIM).permute(1, 0, 2).unsqueeze(0)
    v = v.squeeze(0).reshape(seq_len, ENC_KV_HEADS, ENC_HEAD_DIM).permute(1, 0, 2).unsqueeze(0)
    mask = Tensor.full((1, 1, seq_len, seq_len), float("-inf")).triu(1)
    attn_out = q.float().scaled_dot_product_attention(k.float(), v.float(), attn_mask=mask)
    attn_out = attn_out.squeeze(0).permute(1, 0, 2).reshape(seq_len, ENC_HEADS * ENC_HEAD_DIM)
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
    rope_cos, rope_sin = precompute_rope_interleaved(ENC_HEAD_DIM, seq_len, ENC_ROPE_THETA)
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
# Streaming Encoder (incremental conv stem + KV cache)
# ============================================================================

class StreamingConvStem:
  """Incremental conv stem with boundary-correct tail buffers.

  Conv0: kernel=3, stride=1, causal -> needs 2-frame mel tail between chunks
  Conv1: kernel=3, stride=2 -> needs even input (residual tracking) + 2-frame conv0 tail
  """
  def __init__(self, encoder: Encoder):
    self.encoder = encoder
    self.initialized = False
    self.mel_tail: Tensor | None = None       # [128, 2]
    self.conv0_tail: Tensor | None = None     # [1280, 2]
    self.conv0_residual: Tensor | None = None # [1280, 1]
    self.conv0_residual_count = 0
    self._conv1_initialized = False

  def process(self, mel_new: Tensor) -> Tensor | None:
    """Process new mel frames [128, n_new] -> [new_seq, 1280] or None."""
    n_new = mel_new.shape[1]
    if n_new <= 0: return None

    # Phase 1: Conv0
    if not self.initialized:
      mel_3d = mel_new.unsqueeze(0)
      conv0_out = causal_conv1d(mel_3d, self.encoder.conv_layers_0_conv.weight, self.encoder.conv_layers_0_conv.bias, stride=1).gelu()
      conv0_new = conv0_out.squeeze(0)  # [1280, conv0_len]
      self.mel_tail = mel_new[:, -2:].contiguous().realize() if n_new >= 2 else mel_new.pad((None, (2 - n_new, 0))).contiguous().realize()
      self.initialized = True
    else:
      padded = self.mel_tail.cat(mel_new, dim=1)
      padded_3d = padded.unsqueeze(0)
      conv0_full = causal_conv1d(padded_3d, self.encoder.conv_layers_0_conv.weight, self.encoder.conv_layers_0_conv.bias, stride=1).gelu()
      conv0_new = conv0_full.squeeze(0)[:, 2:]  # discard overlap
      self.mel_tail = mel_new[:, -2:].contiguous().realize() if n_new >= 2 else self.mel_tail

    # Phase 2: Stride alignment (even count for conv1 stride=2)
    conv0_new_len = conv0_new.shape[1]
    total_avail = self.conv0_residual_count + conv0_new_len
    new_res = total_avail & 1
    feed_from_new = conv0_new_len - (new_res if self.conv0_residual_count == 0 else (total_avail - self.conv0_residual_count) - (total_avail - new_res - self.conv0_residual_count))

    # Simpler: figure out how many total to feed (even number), then how many from new
    feed_total = total_avail - new_res
    if feed_total <= 0:
      if new_res and conv0_new_len > 0:
        self.conv0_residual = conv0_new[:, -1:].contiguous().realize()
      self.conv0_residual_count = new_res
      return None

    # Build feed buffer
    parts = []
    if self.conv0_residual_count == 1 and self.conv0_residual is not None:
      parts.append(self.conv0_residual)
      from_new = feed_total - 1
    else:
      from_new = feed_total
    parts.append(conv0_new[:, :from_new])
    feed = parts[0].cat(*parts[1:], dim=1) if len(parts) > 1 else parts[0]

    # Save new residual
    if new_res and conv0_new_len > from_new:
      self.conv0_residual = conv0_new[:, -1:].contiguous().realize()
    else:
      self.conv0_residual = None
    self.conv0_residual_count = new_res

    # Phase 3: Conv1
    if not self._conv1_initialized:
      conv1_in = feed.unsqueeze(0)
      conv1_discard = 0
      self._conv1_initialized = True
      self.conv0_tail = feed[:, -2:].contiguous().realize() if feed.shape[1] >= 2 else feed.pad((None, (2 - feed.shape[1], 0))).contiguous().realize()
    else:
      conv1_in = self.conv0_tail.cat(feed, dim=1).unsqueeze(0)
      conv1_discard = 1
      self.conv0_tail = feed[:, -2:].contiguous().realize() if feed.shape[1] >= 2 else self.conv0_tail

    conv1_out = causal_conv1d(conv1_in, self.encoder.conv_layers_1_conv.weight, self.encoder.conv_layers_1_conv.bias, stride=2).gelu()
    conv1_out = conv1_out.squeeze(0)

    if conv1_discard > 0 and conv1_out.shape[1] > conv1_discard:
      conv1_out = conv1_out[:, conv1_discard:]
    elif conv1_discard > 0:
      return None

    if conv1_out.shape[1] <= 0: return None
    return conv1_out.permute(1, 0)  # [seq, 1280]


class StreamingEncoder:
  """Incremental encoder with per-layer KV cache, matching C vox_encoder_forward_incremental."""
  def __init__(self, encoder: Encoder):
    self.encoder = encoder
    self.conv_stem = StreamingConvStem(encoder)
    self.cache_k: list[Tensor | None] = [None] * ENC_LAYERS
    self.cache_v: list[Tensor | None] = [None] * ENC_LAYERS
    self.cache_len = 0
    self.pos_offset = 0
    self.enc_residual: Tensor | None = None
    self.enc_residual_count = 0

  def reset(self):
    self.conv_stem = StreamingConvStem(self.encoder)
    self.cache_k = [None] * ENC_LAYERS
    self.cache_v = [None] * ENC_LAYERS
    self.cache_len = 0
    self.pos_offset = 0
    self.enc_residual = None
    self.enc_residual_count = 0

  def _compact_cache(self):
    if self.cache_len <= ENC_WINDOW: return
    discard = self.cache_len - ENC_WINDOW
    for i in range(ENC_LAYERS):
      if self.cache_k[i] is not None:
        self.cache_k[i] = self.cache_k[i][:, :, discard:, :].contiguous().realize()
        self.cache_v[i] = self.cache_v[i][:, :, discard:, :].contiguous().realize()
    self.pos_offset += discard
    self.cache_len = ENC_WINDOW

  def _encoder_forward_incremental(self, x: Tensor) -> Tensor:
    """Process [new_len, 1280] through transformer layers with KV cache. Returns [new_len, 1280]."""
    new_len = x.shape[0]
    if self.cache_len + new_len > ENC_WINDOW:
      self._compact_cache()
    cache_len = self.cache_len
    logical_start = self.pos_offset + cache_len

    # RoPE for new positions
    rope_cos, rope_sin = precompute_rope_interleaved(ENC_HEAD_DIM, logical_start + new_len, ENC_ROPE_THETA)
    rope_cos_new = rope_cos[logical_start:logical_start + new_len]
    rope_sin_new = rope_sin[logical_start:logical_start + new_len]

    h = x
    for i, layer in enumerate(self.encoder.transformer_layers):
      x_norm = layer.attention_norm(h)
      q = layer.attention_wq(x_norm).unsqueeze(0)
      k = layer.attention_wk(x_norm).unsqueeze(0)
      v = layer.attention_wv(x_norm).unsqueeze(0)

      q = apply_rope_interleaved(q, rope_cos_new, rope_sin_new, ENC_HEADS, ENC_HEAD_DIM)
      k = apply_rope_interleaved(k, rope_cos_new, rope_sin_new, ENC_KV_HEADS, ENC_HEAD_DIM)

      q = q.squeeze(0).reshape(new_len, ENC_HEADS, ENC_HEAD_DIM).permute(1, 0, 2).unsqueeze(0)
      k_new = k.squeeze(0).reshape(new_len, ENC_KV_HEADS, ENC_HEAD_DIM).permute(1, 0, 2).unsqueeze(0)
      v_new = v.squeeze(0).reshape(new_len, ENC_KV_HEADS, ENC_HEAD_DIM).permute(1, 0, 2).unsqueeze(0)

      if self.cache_k[i] is not None:
        full_k = self.cache_k[i].cat(k_new, dim=2)
        full_v = self.cache_v[i].cat(v_new, dim=2)
      else:
        full_k, full_v = k_new, v_new

      self.cache_k[i] = full_k.contiguous().realize()
      self.cache_v[i] = full_v.contiguous().realize()

      total_kv = cache_len + new_len
      mask = Tensor.full((1, 1, new_len, total_kv), float("-inf")).triu(cache_len + 1)
      attn_out = q.float().scaled_dot_product_attention(full_k.float(), full_v.float(), attn_mask=mask)
      attn_out = attn_out.squeeze(0).permute(1, 0, 2).reshape(new_len, ENC_HEADS * ENC_HEAD_DIM)
      h = h + layer.attention_wo(attn_out)

      x_norm = layer.ffn_norm(h)
      gate = layer.feed_forward_w1(x_norm).silu()
      up = layer.feed_forward_w3(x_norm)
      h = h + layer.feed_forward_w2(gate * up)

    h = self.encoder.transformer_norm(h)
    self.cache_len = cache_len + new_len
    return h

  def process_mel(self, mel_new: Tensor) -> Tensor | None:
    """Process new mel [128, n_frames] -> [aligned_seq, 1280] or None."""
    conv_out = self.conv_stem.process(mel_new)
    if conv_out is None: return None

    enc_out = self._encoder_forward_incremental(conv_out)
    enc_out_len = enc_out.shape[0]

    # Combine with residual, align to DOWNSAMPLE_FACTOR
    if self.enc_residual is not None and self.enc_residual_count > 0:
      combined = self.enc_residual.cat(enc_out, dim=0)
    else:
      combined = enc_out
    total = combined.shape[0]
    usable = (total // DOWNSAMPLE_FACTOR) * DOWNSAMPLE_FACTOR
    leftover = total - usable

    if leftover > 0:
      self.enc_residual = combined[usable:].contiguous().realize()
    else:
      self.enc_residual = None
    self.enc_residual_count = leftover

    if usable <= 0: return None
    return combined[:usable]

# ============================================================================
# Decoder (with pre-allocated KV cache + JIT)
# ============================================================================

class DecoderLayer:
  def __init__(self, max_context: int):
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
    self.max_context = max_context

  def __call__(self, x: Tensor, start_pos: int | UOp, t_cond: Tensor):
    B, T, _ = x.shape
    x_norm = self.attention_norm(x)
    q, k, v = self.attention_wq(x_norm), self.attention_wk(x_norm), self.attention_wv(x_norm)

    # RoPE
    rope_cos, rope_sin = precompute_rope_interleaved(DEC_HEAD_DIM, self.max_context, DEC_ROPE_THETA)
    q = apply_rope_interleaved(q, rope_cos[start_pos:start_pos+T], rope_sin[start_pos:start_pos+T], DEC_HEADS, DEC_HEAD_DIM)
    k = apply_rope_interleaved(k, rope_cos[start_pos:start_pos+T], rope_sin[start_pos:start_pos+T], DEC_KV_HEADS, DEC_HEAD_DIM)

    # Reshape for attention
    q = q.reshape(B, T, DEC_HEADS, DEC_HEAD_DIM).transpose(1, 2)       # [B, H, T, Hd]
    k = k.reshape(B, T, DEC_KV_HEADS, DEC_HEAD_DIM).transpose(1, 2)    # [B, KvH, T, Hd]
    v = v.reshape(B, T, DEC_KV_HEADS, DEC_HEAD_DIM).transpose(1, 2)    # [B, KvH, T, Hd]

    # KV cache: write new K/V into pre-allocated buffer
    if not hasattr(self, "cache_kv"):
      self.cache_kv = Tensor.zeros(2, B, DEC_KV_HEADS, self.max_context, DEC_HEAD_DIM, dtype=k.dtype).contiguous().realize()
    self.cache_kv[:, :, :, start_pos:start_pos+T, :].assign(Tensor.stack(k, v)).realize()
    k = self.cache_kv[0, :, :, 0:start_pos+T, :]
    v = self.cache_kv[1, :, :, 0:start_pos+T, :]

    # Causal mask
    mask = Tensor.full((1, 1, T, start_pos+T), float("-inf"), dtype=x.dtype).triu(int(start_pos)+1) if T > 1 else None
    attn = q.scaled_dot_product_attention(k, v, attn_mask=mask, enable_gqa=True)
    attn = attn.transpose(1, 2).reshape(B, T, -1)
    h = x + self.attention_wo(attn)

    # Adaptive RMS norm + FFN
    h_norm = self.ffn_norm(h)
    ada_hidden = self.ada_rms_norm_t_cond_0(t_cond).gelu()
    ada_scale = self.ada_rms_norm_t_cond_2(ada_hidden)
    h_norm = h_norm * (1 + ada_scale.reshape(1, 1, DEC_DIM))

    gated = self.feed_forward_w1(h_norm).silu().contiguous() * self.feed_forward_w3(h_norm)
    return (h + self.feed_forward_w2(gated)).contiguous()

class Decoder:
  def __init__(self, max_context: int = MAX_CONTEXT):
    self.tok_embeddings = nn.Embedding(VOCAB_SIZE, DEC_DIM)
    self.layers = [DecoderLayer(max_context) for _ in range(DEC_LAYERS)]
    self.norm = nn.RMSNorm(DEC_DIM, DEC_NORM_EPS)
    self.max_context = max_context
    self.forward_jit = TinyJit(self._forward_one)

  def embed_token(self, token_id: int):
    return self.tok_embeddings.weight[token_id]

  def embed_tokens(self, token_ids: list[int]):
    return self.tok_embeddings(Tensor(token_ids, dtype=dtypes.int))

  def _forward_one(self, embed: Tensor, start_pos: int | UOp, t_cond: Tensor) -> Tensor:
    h = embed  # [1, 1, DEC_DIM]
    for layer in self.layers:
      h = layer(h, start_pos, t_cond)
    h = self.norm(h)
    logits = h[:, -1, :].float() @ self.tok_embeddings.weight.float().T
    return logits.argmax(-1, keepdim=True)

  def prefill(self, input_embeds: Tensor, t_cond: Tensor):
    """input_embeds: [seq, dim], runs all prefix tokens through decoder."""
    h = input_embeds.unsqueeze(0)  # [1, seq, dim]
    for i, layer in enumerate(self.layers):
      h = layer(h, 0, t_cond)
      if i < 4 or (i + 1) % 8 == 0:
        print(f"  Decoder prefill layer {i+1}/{DEC_LAYERS}", file=sys.stderr)
    return h

  def decode_token(self, embed: Tensor, pos: int, t_cond: Tensor, use_jit: bool = True) -> int:
    """Single token decode. embed: [DEC_DIM]. Returns token id."""
    h = embed.reshape(1, 1, DEC_DIM)  # [1, 1, DEC_DIM]
    v_start_pos = UOp.variable("start_pos", 1, self.max_context - 1)
    if use_jit:
      token = self.forward_jit(h, v_start_pos.bind(pos), t_cond)
    else:
      token = self._forward_one(h, pos, t_cond)
    return int(token.item())

# ============================================================================
# Weight loading
# ============================================================================

def load_voxtral(model_dir: str, max_context: int = MAX_CONTEXT):
  """Load model from directory containing consolidated.safetensors."""
  import os
  sf_path = os.path.join(model_dir, "consolidated.safetensors")
  print(f"Loading model from {sf_path}", file=sys.stderr)
  state_dict = nn.state.safe_load(sf_path)

  encoder = Encoder()
  adapter = Adapter()
  decoder = Decoder(max_context)

  enc_prefix = "mm_streams_embeddings.embedding_module.whisper_encoder"
  ada_prefix = "mm_streams_embeddings.embedding_module"
  tok_prefix = "mm_streams_embeddings.embedding_module"

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

  # Make all params contiguous for efficiency
  for s in nn.state.get_parameters(decoder): s.replace(s.contiguous())
  for s in nn.state.get_parameters(encoder): s.replace(s.contiguous())
  for s in nn.state.get_parameters(adapter): s.replace(s.contiguous())

  return encoder, adapter, decoder
