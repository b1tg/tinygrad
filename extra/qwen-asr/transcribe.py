#!/usr/bin/env python3
"""
Pure tinygrad inference for Qwen3-ASR (0.6B and 1.7B).

Usage:
  python extra/qwen-asr/transcribe.py <model_dir> <audio.wav>
"""

from __future__ import annotations
import argparse, json, math, os, sys, time, wave
from typing import Callable

import numpy as np
from tinygrad import Device, Tensor, dtypes
from tinygrad.nn.state import safe_load

# ============================================================================
# Constants
# ============================================================================

SAMPLE_RATE = 16000
NUM_MEL_BINS = 128
HOP_LENGTH = 160
WINDOW_SIZE = 400

TOKEN_IM_START = 151644
TOKEN_IM_END = 151645
TOKEN_AUDIO_START = 151669
TOKEN_AUDIO_END = 151670
TOKEN_AUDIO_PAD = 151676
TOKEN_ENDOFTEXT = 151643
TOKEN_ASR_TEXT = 151704

EOS_TOKEN_IDS = {TOKEN_ENDOFTEXT, TOKEN_IM_END}

PROMPT_PREFIX = [TOKEN_IM_START, 8948, 198, TOKEN_IM_END, 198, TOKEN_IM_START, 872, 198, TOKEN_AUDIO_START]
PROMPT_SUFFIX = [TOKEN_AUDIO_END, TOKEN_IM_END, 198, TOKEN_IM_START, 77091, 198]


# ============================================================================
# Config and IO helpers
# ============================================================================

def load_config(model_dir: str) -> dict:
  with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as f:
    cfg = json.load(f)

  tc = cfg["thinker_config"]
  ac = tc["audio_config"]
  txc = tc["text_config"]

  return {
    "enc_d_model": ac["d_model"],
    "enc_layers": ac["encoder_layers"],
    "enc_heads": ac["encoder_attention_heads"],
    "enc_ffn_dim": ac["encoder_ffn_dim"],
    "enc_output_dim": ac["output_dim"],
    "enc_downsample_hidden": ac["downsample_hidden_size"],
    "enc_num_mel_bins": ac["num_mel_bins"],
    "enc_max_source_pos": ac["max_source_positions"],
    "enc_n_window": ac["n_window"],
    "enc_n_window_infer": ac["n_window_infer"],
    "enc_conv_chunksize": ac.get("conv_chunksize", 500),
    "dec_hidden_size": txc["hidden_size"],
    "dec_layers": txc["num_hidden_layers"],
    "dec_heads": txc["num_attention_heads"],
    "dec_kv_heads": txc["num_key_value_heads"],
    "dec_head_dim": txc["head_dim"],
    "dec_intermediate": txc["intermediate_size"],
    "dec_rms_norm_eps": txc["rms_norm_eps"],
    "dec_rope_theta": txc["rope_theta"],
    "dec_mrope_section": txc["rope_scaling"]["mrope_section"],
    "dec_vocab_size": txc["vocab_size"],
    "audio_start_token_id": tc["audio_start_token_id"],
    "audio_end_token_id": tc["audio_end_token_id"],
    "audio_token_id": tc["audio_token_id"],
  }


def load_audio(path: str) -> tuple[np.ndarray, int]:
  try:
    import soundfile as sf
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
      audio = audio.mean(axis=1)
    return np.asarray(audio, dtype=np.float32), int(sr)
  except Exception:
    with wave.open(path, "rb") as wf:
      sr = wf.getframerate()
      channels = wf.getnchannels()
      width = wf.getsampwidth()
      nframes = wf.getnframes()
      raw = wf.readframes(nframes)
    if width == 1:
      audio = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
      audio = (audio - 128.0) / 128.0
    elif width == 2:
      audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 4:
      audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
      raise RuntimeError(f"unsupported wav sample width: {width}")
    if channels > 1:
      audio = audio.reshape(-1, channels).mean(axis=1)
    return audio.astype(np.float32), int(sr)


def linear_resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
  if src_sr == dst_sr:
    return audio.astype(np.float32)
  ratio = dst_sr / src_sr
  new_len = int(len(audio) * ratio)
  idx = np.linspace(0, len(audio) - 1, new_len, dtype=np.float32)
  return np.interp(idx, np.arange(len(audio), dtype=np.float32), audio).astype(np.float32)


# ============================================================================
# Mel filterbank + spectrogram
# ============================================================================

def hertz_to_mel(freq: np.ndarray | float) -> np.ndarray | float:
  min_log_hertz = 1000.0
  min_log_mel = 15.0
  logstep = 27.0 / np.log(6.4)
  mels = 3.0 * freq / 200.0
  if isinstance(freq, np.ndarray):
    log_region = freq >= min_log_hertz
    mels[log_region] = min_log_mel + np.log(freq[log_region] / min_log_hertz) * logstep
    return mels
  if freq >= min_log_hertz:
    return min_log_mel + np.log(freq / min_log_hertz) * logstep
  return mels


def mel_to_hertz(mels: np.ndarray) -> np.ndarray:
  min_log_hertz = 1000.0
  min_log_mel = 15.0
  logstep = np.log(6.4) / 27.0
  freq = 200.0 * mels / 3.0
  log_region = mels >= min_log_mel
  freq[log_region] = min_log_hertz * np.exp(logstep * (mels[log_region] - min_log_mel))
  return freq


def compute_mel_filters() -> np.ndarray:
  num_frequency_bins = 1 + WINDOW_SIZE // 2
  fft_freqs = np.linspace(0, SAMPLE_RATE // 2, num_frequency_bins)
  mel_min = hertz_to_mel(0.0)
  mel_max = hertz_to_mel(8000.0)
  mel_freqs = np.linspace(mel_min, mel_max, NUM_MEL_BINS + 2)
  filter_freqs = mel_to_hertz(mel_freqs)
  filter_diff = np.diff(filter_freqs)
  slopes = np.expand_dims(filter_freqs, 0) - np.expand_dims(fft_freqs, 1)
  down_slopes = -slopes[:, :-2] / filter_diff[:-1]
  up_slopes = slopes[:, 2:] / filter_diff[1:]
  fb = np.maximum(np.zeros(1), np.minimum(down_slopes, up_slopes))
  enorm = 2.0 / (filter_freqs[2:NUM_MEL_BINS + 2] - filter_freqs[:NUM_MEL_BINS])
  fb *= np.expand_dims(enorm, 0)
  return fb.astype(np.float32)


def _hann_window_periodic(n: int) -> Tensor:
  t = Tensor.arange(n, dtype=dtypes.float32)
  return 0.5 - 0.5 * (2.0 * math.pi * t / n).cos()


def compute_mel_spectrogram(audio: Tensor, mel_filters: Tensor) -> Tensor:
  """
  audio: [samples], mel_filters: [freq_bins, mel_bins]
  returns: [mel_bins, frames]

  This mirrors torch.stft(..., center=True, pad_mode="reflect") + power->mel->log.
  """
  if audio.ndim != 1:
    raise ValueError("audio must be 1D")

  pad = WINDOW_SIZE // 2
  if audio.shape[0] < 2:
    audio = Tensor.zeros(WINDOW_SIZE, dtype=dtypes.float32)

  left = audio[1:pad + 1].flip(0)
  right = audio[-pad - 1:-1].flip(0)
  padded = left.cat(audio, dim=0).cat(right, dim=0)

  frames = padded.reshape(1, 1, -1)._pool((WINDOW_SIZE,), stride=(HOP_LENGTH,)).reshape(-1, WINDOW_SIZE)
  if frames.shape[0] > 1:
    frames = frames[:-1]

  win = _hann_window_periodic(WINDOW_SIZE)
  frames = frames * win

  n_bins = WINDOW_SIZE // 2 + 1
  k = Tensor.arange(n_bins, dtype=dtypes.float32).reshape(n_bins, 1)
  n = Tensor.arange(WINDOW_SIZE, dtype=dtypes.float32).reshape(1, WINDOW_SIZE)
  angle = -2.0 * math.pi * (k @ n) / WINDOW_SIZE
  dft_real = angle.cos()
  dft_imag = angle.sin()

  real = frames @ dft_real.T
  imag = frames @ dft_imag.T
  power = real.square() + imag.square()

  mel_spec = power @ mel_filters
  log_spec = (mel_spec.maximum(1e-10).log() / math.log(10.0))
  maxv = log_spec.max()
  log_spec = log_spec.maximum(maxv - 8.0)
  log_spec = (log_spec + 4.0) / 4.0
  return log_spec.T


# ============================================================================
# Weights
# ============================================================================

class MultiSafetensors:
  def __init__(self, model_dir: str):
    self.model_dir = model_dir
    self.weight_map = None
    self.shards: dict[str, dict[str, Tensor]] = {}

    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    single_path = os.path.join(model_dir, "model.safetensors")

    if os.path.exists(index_path):
      with open(index_path, encoding="utf-8") as f:
        index = json.load(f)
      self.weight_map = index["weight_map"]
      self.single_path = None
    elif os.path.exists(single_path):
      self.weight_map = None
      self.single_path = single_path
      self.single = safe_load(single_path)
    else:
      raise FileNotFoundError(f"no safetensors found in {model_dir}")

  def _load_shard(self, shard: str) -> dict[str, Tensor]:
    if shard not in self.shards:
      self.shards[shard] = safe_load(os.path.join(self.model_dir, shard))
    return self.shards[shard]

  def get_tensor(self, name: str) -> Tensor:
    if self.weight_map is None:
      return self.single[name]
    shard = self.weight_map[name]
    return self._load_shard(shard)[name]


def get_weight(sf: MultiSafetensors, name: str) -> Tensor:
  t = sf.get_tensor(name)
  if isinstance(t.device, str) and t.device.startswith("DISK:"):
    t = t.to(Device.DEFAULT)
  return t.float() if t.dtype == dtypes.bfloat16 else t


# ============================================================================
# Math helpers
# ============================================================================

def linear(x: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
  out = x.linear(weight.transpose())
  return out + bias if bias is not None else out


def layer_norm(x: Tensor, weight: Tensor, bias: Tensor, eps: float = 1e-5) -> Tensor:
  return x.layernorm(axis=-1, eps=eps) * weight + bias


def rms_norm(x: Tensor, weight: Tensor, eps: float = 1e-6) -> Tensor:
  xx = x.float()
  var = xx.square().mean(-1, keepdim=True)
  return (xx * (var + eps).rsqrt()) * weight.float()


def sinusoidal_position_embedding(length: int, channels: int, max_timescale: float = 10000) -> Tensor:
  log_timescale_increment = math.log(max_timescale) / (channels // 2 - 1)
  inv_timescales = (-log_timescale_increment * Tensor.arange(channels // 2, dtype=dtypes.float32)).exp()
  scaled_time = Tensor.arange(length, dtype=dtypes.float32).reshape(length, 1) * inv_timescales.reshape(1, -1)
  return scaled_time.sin().cat(scaled_time.cos(), dim=1)


def compute_rope_freqs(positions: Tensor, head_dim: int, theta: float) -> tuple[Tensor, Tensor]:
  inv_freq = 1.0 / (theta ** (Tensor.arange(0, head_dim, 2, dtype=dtypes.float32) / head_dim))
  angles = positions.float().reshape(-1, 1) * inv_freq.reshape(1, -1)
  emb = angles.cat(angles, dim=-1)
  return emb.cos(), emb.sin()


def apply_rope_neox(x: Tensor, cos_f: Tensor, sin_f: Tensor, head_dim: int) -> Tensor:
  cos_f = cos_f.unsqueeze(1)
  sin_f = sin_f.unsqueeze(1)
  half = head_dim // 2
  x1 = x[..., :half]
  x2 = x[..., half:]
  rotated = (-x2).cat(x1, dim=-1)
  return x * cos_f + rotated * sin_f


def full_attention(q: Tensor, k: Tensor, v: Tensor, n_heads: int, n_kv_heads: int, head_dim: int,
                   cu_seqlens: list[int] | None = None) -> Tensor:
  seq_len = q.shape[0]

  if cu_seqlens is not None and len(cu_seqlens) > 2:
    chunks: list[Tensor] = []
    for i in range(len(cu_seqlens) - 1):
      start, end = cu_seqlens[i], cu_seqlens[i + 1]
      chunks.append(full_attention(q[start:end], k[start:end], v[start:end], n_heads, n_kv_heads, head_dim, None))
    out = chunks[0]
    for c in chunks[1:]:
      out = out.cat(c, dim=0)
    return out

  qq = q.reshape(seq_len, n_heads, head_dim).transpose(0, 1).unsqueeze(0)
  kk = k.reshape(seq_len, n_kv_heads, head_dim).transpose(0, 1).unsqueeze(0)
  vv = v.reshape(seq_len, n_kv_heads, head_dim).transpose(0, 1).unsqueeze(0)

  out = qq.float().scaled_dot_product_attention(kk.float(), vv.float(), enable_gqa=(n_heads != n_kv_heads))
  return out.squeeze(0).transpose(0, 1).reshape(seq_len, n_heads * head_dim)


def causal_attention(q: Tensor, k: Tensor, v: Tensor, n_heads: int, n_kv_heads: int, head_dim: int,
                     q_start_pos: int = 0, kv_start_pos: int = 0) -> Tensor:
  seq_q = q.shape[0]
  seq_kv = k.shape[0]

  qq = q.reshape(seq_q, n_heads, head_dim).transpose(0, 1).unsqueeze(0)
  kk = k.reshape(seq_kv, n_kv_heads, head_dim).transpose(0, 1).unsqueeze(0)
  vv = v.reshape(seq_kv, n_kv_heads, head_dim).transpose(0, 1).unsqueeze(0)

  qi_abs = (Tensor.arange(seq_q, dtype=dtypes.int32) + q_start_pos).reshape(seq_q, 1)
  kv_abs = (Tensor.arange(seq_kv, dtype=dtypes.int32) + kv_start_pos).reshape(1, seq_kv)
  attn_mask = (kv_abs <= qi_abs).reshape(1, 1, seq_q, seq_kv)

  out = qq.float().scaled_dot_product_attention(
    kk.float(),
    vv.float(),
    attn_mask=attn_mask,
    enable_gqa=(n_heads != n_kv_heads),
  )
  return out.squeeze(0).transpose(0, 1).reshape(seq_q, n_heads * head_dim)


# ============================================================================
# Encoder
# ============================================================================

def encoder_forward(mel: Tensor, sf: MultiSafetensors, cfg: dict, verbose: bool = True) -> Tensor:
  prefix = "thinker.audio_tower"
  d_model = cfg["enc_d_model"]
  n_layers = cfg["enc_layers"]
  n_heads = cfg["enc_heads"]
  head_dim = d_model // n_heads
  n_window = cfg["enc_n_window"]
  chunk_size = n_window * 2

  conv1_w, conv1_b = get_weight(sf, f"{prefix}.conv2d1.weight"), get_weight(sf, f"{prefix}.conv2d1.bias")
  conv2_w, conv2_b = get_weight(sf, f"{prefix}.conv2d2.weight"), get_weight(sf, f"{prefix}.conv2d2.bias")
  conv3_w, conv3_b = get_weight(sf, f"{prefix}.conv2d3.weight"), get_weight(sf, f"{prefix}.conv2d3.bias")
  conv_out_w = get_weight(sf, f"{prefix}.conv_out.weight")

  total_frames = mel.shape[1]
  chunk_outputs: list[Tensor] = []

  for start in range(0, total_frames, chunk_size):
    end = min(start + chunk_size, total_frames)
    chunk_mel = mel[:, start:end]

    x = chunk_mel.unsqueeze(0).unsqueeze(0)
    x = x.conv2d(conv1_w, conv1_b, stride=2, padding=1).gelu()
    x = x.conv2d(conv2_w, conv2_b, stride=2, padding=1).gelu()
    x = x.conv2d(conv3_w, conv3_b, stride=2, padding=1).gelu()

    b, c, f, t = x.shape
    x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)
    chunk_outputs.append(x.squeeze(0).realize())

  x = chunk_outputs[0]
  for c in chunk_outputs[1:]:
    x = x.cat(c, dim=0)

  if verbose:
    print(f"  Conv output: {total_frames} frames -> {x.shape[0]} tokens (chunks of {chunk_size})", file=sys.stderr)

  x = linear(x, conv_out_w).realize()
  seq_len = x.shape[0]
  if verbose:
    print(f"  After conv_out projection: [{seq_len}, {d_model}]", file=sys.stderr)

  tokens_per_chunk = chunk_outputs[0].shape[0]
  pos_emb = sinusoidal_position_embedding(tokens_per_chunk, d_model)
  offset = 0
  merged: list[Tensor] = []
  for co in chunk_outputs:
    chunk_len = co.shape[0]
    merged.append(x[offset:offset + chunk_len] + pos_emb[:chunk_len])
    offset += chunk_len
  x = merged[0]
  for c in merged[1:]:
    x = x.cat(c, dim=0)

  n_window_infer = cfg["enc_n_window_infer"]
  tokens_per_infer_window = tokens_per_chunk * (n_window_infer // chunk_size)

  cu_seqlens = [0]
  pos = 0
  while pos < seq_len:
    window_end = min(pos + tokens_per_infer_window, seq_len)
    cu_seqlens.append(window_end)
    pos = window_end
  if verbose:
    print(f"  Attention windows (cu_seqlens): {cu_seqlens}", file=sys.stderr)

  for layer in range(n_layers):
    lp = f"{prefix}.layers.{layer}"

    ln_w = get_weight(sf, f"{lp}.self_attn_layer_norm.weight")
    ln_b = get_weight(sf, f"{lp}.self_attn_layer_norm.bias")
    x_norm = layer_norm(x, ln_w, ln_b)

    wq, wq_b = get_weight(sf, f"{lp}.self_attn.q_proj.weight"), get_weight(sf, f"{lp}.self_attn.q_proj.bias")
    wk, wk_b = get_weight(sf, f"{lp}.self_attn.k_proj.weight"), get_weight(sf, f"{lp}.self_attn.k_proj.bias")
    wv, wv_b = get_weight(sf, f"{lp}.self_attn.v_proj.weight"), get_weight(sf, f"{lp}.self_attn.v_proj.bias")
    wo, wo_b = get_weight(sf, f"{lp}.self_attn.out_proj.weight"), get_weight(sf, f"{lp}.self_attn.out_proj.bias")

    q = linear(x_norm, wq, wq_b)
    k = linear(x_norm, wk, wk_b)
    v = linear(x_norm, wv, wv_b)

    attn_out = full_attention(q, k, v, n_heads, n_heads, head_dim, cu_seqlens=cu_seqlens)
    x = (x + linear(attn_out, wo, wo_b)).realize()

    ffn_ln_w = get_weight(sf, f"{lp}.final_layer_norm.weight")
    ffn_ln_b = get_weight(sf, f"{lp}.final_layer_norm.bias")
    x_norm = layer_norm(x, ffn_ln_w, ffn_ln_b)

    fc1_w, fc1_b = get_weight(sf, f"{lp}.fc1.weight"), get_weight(sf, f"{lp}.fc1.bias")
    fc2_w, fc2_b = get_weight(sf, f"{lp}.fc2.weight"), get_weight(sf, f"{lp}.fc2.bias")

    ffn_out = linear(linear(x_norm, fc1_w, fc1_b).gelu(), fc2_w, fc2_b)
    x = (x + ffn_out).realize()

    if verbose and ((layer + 1) % 6 == 0 or layer == 0):
      print(f"  Encoder layer {layer + 1}/{n_layers}: range [{x.min().item():.2f}, {x.max().item():.2f}]", file=sys.stderr)

  ln_post_w = get_weight(sf, f"{prefix}.ln_post.weight")
  ln_post_b = get_weight(sf, f"{prefix}.ln_post.bias")
  x = layer_norm(x, ln_post_w, ln_post_b)

  proj1_w, proj1_b = get_weight(sf, f"{prefix}.proj1.weight"), get_weight(sf, f"{prefix}.proj1.bias")
  proj2_w, proj2_b = get_weight(sf, f"{prefix}.proj2.weight"), get_weight(sf, f"{prefix}.proj2.bias")

  x = linear(linear(x, proj1_w, proj1_b).gelu(), proj2_w, proj2_b).realize()

  if verbose:
    print(f"  Encoder final output: [{x.shape[0]}, {x.shape[1]}]", file=sys.stderr)
  return x


# ============================================================================
# Decoder
# ============================================================================

class Decoder:
  def __init__(self, sf: MultiSafetensors, cfg: dict, verbose: bool = True):
    self.sf = sf
    self.cfg = cfg
    self.hidden_size = cfg["dec_hidden_size"]
    self.n_layers = cfg["dec_layers"]
    self.n_heads = cfg["dec_heads"]
    self.n_kv_heads = cfg["dec_kv_heads"]
    self.head_dim = cfg["dec_head_dim"]
    self.eps = cfg["dec_rms_norm_eps"]
    self.rope_theta = cfg["dec_rope_theta"]

    self.tok_embeddings = get_weight(sf, "thinker.model.embed_tokens.weight")
    self.lm_head = get_weight(sf, "thinker.lm_head.weight")
    self.final_norm = get_weight(sf, "thinker.model.norm.weight")

    self.layers = []
    for i in range(self.n_layers):
      self.layers.append(self._load_layer(i))
      if verbose and (i + 1) % 8 == 0:
        print(f"  Decoder layer {i + 1}/{self.n_layers} loaded", file=sys.stderr)

    self.kv_cache: dict[int, tuple[Tensor, Tensor]] = {}

  def _load_layer(self, i: int) -> dict[str, Tensor]:
    lp = f"thinker.model.layers.{i}"
    return {
      "input_layernorm": get_weight(self.sf, f"{lp}.input_layernorm.weight"),
      "post_attention_layernorm": get_weight(self.sf, f"{lp}.post_attention_layernorm.weight"),
      "q_proj": get_weight(self.sf, f"{lp}.self_attn.q_proj.weight"),
      "k_proj": get_weight(self.sf, f"{lp}.self_attn.k_proj.weight"),
      "v_proj": get_weight(self.sf, f"{lp}.self_attn.v_proj.weight"),
      "o_proj": get_weight(self.sf, f"{lp}.self_attn.o_proj.weight"),
      "q_norm": get_weight(self.sf, f"{lp}.self_attn.q_norm.weight"),
      "k_norm": get_weight(self.sf, f"{lp}.self_attn.k_norm.weight"),
      "gate_proj": get_weight(self.sf, f"{lp}.mlp.gate_proj.weight"),
      "up_proj": get_weight(self.sf, f"{lp}.mlp.up_proj.weight"),
      "down_proj": get_weight(self.sf, f"{lp}.mlp.down_proj.weight"),
    }

  def embed_token(self, token_id: int) -> Tensor:
    return self.tok_embeddings[token_id]

  def embed_tokens(self, token_ids: Tensor) -> Tensor:
    return self.tok_embeddings[token_ids]

  def _layer_forward(self, h: Tensor, layer_idx: int, pos: int) -> Tensor:
    l = self.layers[layer_idx]
    seq_len = h.shape[0]

    x_norm = rms_norm(h, l["input_layernorm"], self.eps)

    q = linear(x_norm, l["q_proj"])
    k = linear(x_norm, l["k_proj"])
    v = linear(x_norm, l["v_proj"])

    q = q.reshape(seq_len, self.n_heads, self.head_dim)
    k = k.reshape(seq_len, self.n_kv_heads, self.head_dim)

    q = rms_norm(q, l["q_norm"], self.eps)
    k = rms_norm(k, l["k_norm"], self.eps)

    positions = Tensor.arange(pos, pos + seq_len, dtype=dtypes.int32)
    rope_cos, rope_sin = compute_rope_freqs(positions, self.head_dim, self.rope_theta)
    q = apply_rope_neox(q, rope_cos, rope_sin, self.head_dim)
    k = apply_rope_neox(k, rope_cos, rope_sin, self.head_dim)

    q = q.reshape(seq_len, self.n_heads * self.head_dim)
    k = k.reshape(seq_len, self.n_kv_heads * self.head_dim)
    v = v.reshape(seq_len, self.n_kv_heads * self.head_dim)

    if layer_idx not in self.kv_cache:
      k_cache, v_cache = k, v
    else:
      kc, vc = self.kv_cache[layer_idx]
      k_cache, v_cache = kc.cat(k, dim=0), vc.cat(v, dim=0)
    self.kv_cache[layer_idx] = (k_cache.realize(), v_cache.realize())

    kv_start_pos = (pos + seq_len - 1) - (k_cache.shape[0] - 1)
    attn_out = causal_attention(
      q,
      k_cache,
      v_cache,
      self.n_heads,
      self.n_kv_heads,
      self.head_dim,
      q_start_pos=pos,
      kv_start_pos=kv_start_pos,
    )

    h = (h + linear(attn_out, l["o_proj"])).realize()

    x_norm = rms_norm(h, l["post_attention_layernorm"], self.eps)
    gate = linear(x_norm, l["gate_proj"]).silu()
    up = linear(x_norm, l["up_proj"])
    h = (h + linear(gate * up, l["down_proj"])).realize()
    return h

  def prefill(self, input_embeds: Tensor, verbose: bool = True) -> Tensor:
    self.kv_cache = {}
    h = input_embeds
    for layer in range(self.n_layers):
      h = self._layer_forward(h, layer, 0)
      if verbose and (layer < 2 or (layer + 1) % 8 == 0):
        print(f"  Decoder prefill layer {layer + 1}/{self.n_layers}: [{h.min().item():.2f}, {h.max().item():.2f}]", file=sys.stderr)
    return h

  def forward_one(self, embed: Tensor, pos: int) -> Tensor:
    h = embed.unsqueeze(0) if embed.ndim == 1 else embed
    for layer in range(self.n_layers):
      h = self._layer_forward(h, layer, pos)
    h = rms_norm(h, self.final_norm, self.eps)
    return linear(h.float().squeeze(0), self.lm_head)


# ============================================================================
# Tokenizer decode
# ============================================================================

def bytes_to_unicode() -> dict[int, str]:
  bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) + list(range(ord("\xae"), ord("\xff") + 1))
  cs = bs[:]
  n = 0
  for b in range(256):
    if b not in bs:
      bs.append(b)
      cs.append(256 + n)
      n += 1
  return dict(zip(bs, [chr(c) for c in cs]))


def load_tokenizer(model_dir: str) -> Callable[[list[int]], str]:
  with open(os.path.join(model_dir, "vocab.json"), encoding="utf-8") as f:
    vocab = json.load(f)
  id_to_token = {v: k for k, v in vocab.items()}

  special_tokens: set[int] = set()
  tc_path = os.path.join(model_dir, "tokenizer_config.json")
  if os.path.exists(tc_path):
    with open(tc_path, encoding="utf-8") as f:
      tc = json.load(f)
    for tid_str in tc.get("added_tokens_decoder", {}).keys():
      special_tokens.add(int(tid_str))

  byte_decoder = {v: k for k, v in bytes_to_unicode().items()}

  def decode(token_ids: list[int]) -> str:
    pieces: list[str] = []
    for tid in token_ids:
      if tid in special_tokens:
        if tid == TOKEN_ASR_TEXT:
          pieces.append("<asr_text>")
        continue
      tok = id_to_token.get(tid, "")
      if tok:
        pieces.append(tok)
    text = "".join(pieces)
    raw = bytearray([byte_decoder[c] for c in text if c in byte_decoder])
    return raw.decode("utf-8", errors="replace")

  return decode


def parse_asr_text(text: str) -> str:
  if "<asr_text>" in text:
    return text.split("<asr_text>", 1)[1]
  lower = text.lower()
  if lower.startswith("language "):
    parts = text.split(maxsplit=2)
    if len(parts) >= 3:
      return parts[2]
  return text


# ============================================================================
# Pipeline
# ============================================================================

def transcribe(model_dir: str, wav_path: str, max_new_tokens: int = 1024, verbose: bool = True) -> dict:
  t0 = time.perf_counter()

  audio_array, sr = load_audio(wav_path)
  if sr != SAMPLE_RATE:
    if verbose:
      print(f"Audio sample rate is {sr}, resampling to {SAMPLE_RATE}", file=sys.stderr)
    audio_array = linear_resample(audio_array, sr, SAMPLE_RATE)

  if verbose:
    print(f"Audio: {len(audio_array)} samples ({len(audio_array) / SAMPLE_RATE:.1f}s)", file=sys.stderr)

  cfg = load_config(model_dir)
  if verbose:
    print(
      f"Model: enc_d={cfg['enc_d_model']}, enc_layers={cfg['enc_layers']}, dec_hidden={cfg['dec_hidden_size']}, "
      f"dec_layers={cfg['dec_layers']}",
      file=sys.stderr,
    )

  mel_filters = Tensor(compute_mel_filters(), dtype=dtypes.float32)
  audio_tensor = Tensor(audio_array, dtype=dtypes.float32)
  t_mel0 = time.perf_counter()
  mel = compute_mel_spectrogram(audio_tensor, mel_filters).realize()
  mel_s = time.perf_counter() - t_mel0
  if verbose:
    print(f"Mel spectrogram: [{mel.shape[0]}, {mel.shape[1]}]", file=sys.stderr)

  if verbose:
    print(f"Loading model from {model_dir}...", file=sys.stderr)
  sf_file = MultiSafetensors(model_dir)

  if verbose:
    print("Running encoder...", file=sys.stderr)
  t_enc0 = time.perf_counter()
  audio_embeds = encoder_forward(mel, sf_file, cfg, verbose=verbose).realize()
  enc_s = time.perf_counter() - t_enc0

  n_audio = audio_embeds.shape[0]
  if verbose:
    print(f"Audio embeddings: [{n_audio}, {audio_embeds.shape[1]}]", file=sys.stderr)

  input_ids = PROMPT_PREFIX + [TOKEN_AUDIO_PAD] * n_audio + PROMPT_SUFFIX
  if verbose:
    print(f"Prompt length: {len(input_ids)} tokens ({n_audio} audio pads)", file=sys.stderr)

  if verbose:
    print("Loading decoder...", file=sys.stderr)
  decoder = Decoder(sf_file, cfg, verbose=verbose)

  prefix_ids = Tensor(np.array(PROMPT_PREFIX, dtype=np.int32), dtype=dtypes.int32)
  suffix_ids = Tensor(np.array(PROMPT_SUFFIX, dtype=np.int32), dtype=dtypes.int32)
  input_embeds = decoder.embed_tokens(prefix_ids).cat(audio_embeds, dim=0).cat(decoder.embed_tokens(suffix_ids), dim=0)

  prompt_len = len(input_ids)
  if verbose:
    print(f"Running decoder prefill ({prompt_len} tokens)...", file=sys.stderr)

  t_dec0 = time.perf_counter()
  if prompt_len > 1:
    _ = decoder.prefill(input_embeds[:-1], verbose=verbose)
  logits = decoder.forward_one(input_embeds[-1], pos=prompt_len - 1).realize()
  token = int(logits.argmax().item())
  generated = [token]

  if verbose:
    print(f"  First token: {token}", file=sys.stderr)
    print("Running decoder generation...", file=sys.stderr)

  for step in range(max_new_tokens - 1):
    if token in EOS_TOKEN_IDS:
      break
    pos = prompt_len + step
    embed = decoder.embed_token(token)
    logits = decoder.forward_one(embed, pos=pos).realize()
    token = int(logits.argmax().item())
    generated.append(token)

  dec_s = time.perf_counter() - t_dec0
  if verbose:
    print(f"Generated {len(generated)} tokens", file=sys.stderr)

  while generated and generated[-1] in EOS_TOKEN_IDS:
    generated = generated[:-1]

  decode = load_tokenizer(model_dir)
  text = parse_asr_text(decode(generated)).strip()

  wall_s = time.perf_counter() - t0
  return {
    "text": text,
    "audio_seconds": len(audio_array) / SAMPLE_RATE,
    "wall_seconds": wall_s,
    "mel_seconds": mel_s,
    "encoder_seconds": enc_s,
    "decoder_seconds": dec_s,
    "generated_tokens": len(generated),
    "tokens_per_second": (len(generated) / dec_s) if dec_s > 0 else float("inf"),
  }


def main() -> None:
  parser = argparse.ArgumentParser(description="Qwen3-ASR tinygrad transcriber")
  parser.add_argument("model_dir", type=str)
  parser.add_argument("audio", type=str)
  parser.add_argument("--max-new-tokens", type=int, default=1024)
  parser.add_argument("--silent", action="store_true", help="only print transcript")
  parser.add_argument("--timings-json", action="store_true", help="print timing JSON to stderr")
  args = parser.parse_args()

  out = transcribe(args.model_dir, args.audio, max_new_tokens=args.max_new_tokens, verbose=not args.silent)
  print(out["text"])

  if args.timings_json:
    print(json.dumps({k: v for k, v in out.items() if k != "text"}, sort_keys=True), file=sys.stderr)


if __name__ == "__main__":
  main()
