#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, math, time
from pathlib import Path
from typing import Optional, Union

import librosa
import numpy as np
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.append(str(REPO_ROOT))

from tinygrad import Device, Tensor, TinyJit, Variable, dtypes, nn
from tinygrad.helpers import getenv
from tinygrad.nn.state import load_state_dict, safe_load

from extra.models.llama import apply_rotary_emb, convert_from_huggingface, fix_bf16, precompute_freqs_cis, repeat_kv


def load_hf_index_weights(repo_id: str, index_file: str = "model.safetensors.index.json") -> dict[str, Tensor]:
  index_path = hf_hub_download(repo_id, index_file)
  with open(index_path) as f:
    weight_map = json.load(f)["weight_map"]
  shard_names = sorted(set(weight_map.values()))
  shards = {name: safe_load(str(hf_hub_download(repo_id, name))) for name in shard_names}
  return {k: shards[shard_name][k] for k, shard_name in weight_map.items()}


def activation(name: str, x: Tensor) -> Tensor:
  if name == "gelu": return x.gelu()
  if name == "silu": return x.silu()
  raise ValueError(f"unsupported activation: {name}")


class Conv1dCacheLayer:
  def __init__(self):
    self.cache: Optional[Tensor] = None
    self.left_pad: int = 0
    self.in_channels: int = 0

  def lazy_initialization(self, hidden_states: Tensor, conv_module: "CausalConv1d"):
    self.left_pad = conv_module.left_pad
    self.in_channels = conv_module.in_channels
    self.cache = Tensor.zeros(
      hidden_states.shape[0], self.in_channels, self.left_pad, dtype=hidden_states.dtype, device=hidden_states.device
    ).realize()

  def update(self, hidden_states: Tensor, conv_module: Optional["CausalConv1d"] = None) -> Tensor:
    if self.cache is None:
      if conv_module is None: raise ValueError("cache is not initialized")
      self.lazy_initialization(hidden_states, conv_module)
    assert self.cache is not None

    if self.left_pad > 0:
      shortfall = max(0, self.left_pad - hidden_states.shape[-1])
      if shortfall > 0:
        padding_states = self.cache[:, :, -shortfall:].cat(hidden_states, dim=-1)
      else:
        padding_states = hidden_states[:, :, -self.left_pad:]
    else:
      padding_states = Tensor.empty(hidden_states.shape[0], self.in_channels, 0, dtype=hidden_states.dtype, device=hidden_states.device)

    current_cache = self.cache.clone()
    if self.left_pad > 0:
      self.cache.assign(padding_states).realize()
    return current_cache


class Conv1dPaddingCache:
  def __init__(self):
    self.layers: dict[str, Conv1dCacheLayer] = {}

  def update(self, hidden_states: Tensor, cache_key: str, conv_module: "CausalConv1d") -> Tensor:
    if cache_key not in self.layers:
      self.layers[cache_key] = Conv1dCacheLayer()
    padding_states = self.layers[cache_key].update(hidden_states, conv_module)
    return padding_states.cat(hidden_states, dim=-1)


class CausalConv1d:
  def __init__(
    self,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    cache_key: str,
    stride: int = 1,
    dilation: int = 1,
    groups: int = 1,
  ):
    self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, groups=groups, bias=True)
    self.causal_padding = (kernel_size - 1) * dilation - (stride - 1)
    if self.causal_padding < 0:
      raise ValueError(f"invalid causal padding {self.causal_padding}")
    self.cache_key = cache_key
    self.in_channels = in_channels
    self.left_pad = self.causal_padding

  def __call__(self, hidden_states: Tensor, padding_cache: Optional[Conv1dPaddingCache] = None) -> Tensor:
    if padding_cache is not None:
      hidden_states = padding_cache.update(hidden_states, self.cache_key, self)
    else:
      hidden_states = hidden_states.pad((None, None, (self.left_pad, 0)))
    return self.conv(hidden_states)


class AcousticFeedForward:
  def __init__(self, config: dict, hidden_size: int):
    self.linear1 = nn.Linear(hidden_size, config["ffn_expansion"] * hidden_size, bias=True)
    self.linear2 = nn.Linear(config["ffn_expansion"] * hidden_size, hidden_size, bias=True)
    self.hidden_act = config["hidden_act"]

  def __call__(self, hidden_states: Tensor) -> Tensor:
    return self.linear2(activation(self.hidden_act, self.linear1(hidden_states)))


class AcousticConvNext1dLayer:
  def __init__(self, config: dict, hidden_size: int, dilation: int = 1, stride: int = 1, layer_idx: Optional[int] = None):
    self.norm = nn.RMSNorm(hidden_size, config["rms_norm_eps"])
    self.ffn_norm = nn.RMSNorm(hidden_size, config["rms_norm_eps"])
    self.ffn = AcousticFeedForward(config, hidden_size)
    self.gamma = (Tensor.ones(hidden_size) * config["layer_scale_init_value"]).realize()
    self.ffn_gamma = (Tensor.ones(hidden_size) * config["layer_scale_init_value"]).realize()
    self.mixer = CausalConv1d(
      in_channels=hidden_size,
      out_channels=hidden_size,
      kernel_size=config["kernel_size"],
      cache_key=f"convnext_layer_{layer_idx}",
      groups=hidden_size,
      dilation=dilation,
      stride=stride,
    )

  def __call__(self, hidden_states: Tensor, padding_cache: Optional[Conv1dPaddingCache] = None) -> Tensor:
    residual = hidden_states
    hidden_states = self.norm(hidden_states.transpose(1, 2)).transpose(1, 2)
    hidden_states = self.mixer(hidden_states, padding_cache=padding_cache)
    hidden_states = hidden_states * self.gamma.reshape(1, -1, 1)
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.ffn_norm(hidden_states.transpose(1, 2))
    hidden_states = self.ffn(hidden_states).transpose(1, 2)
    hidden_states = hidden_states * self.ffn_gamma.reshape(1, -1, 1)
    return residual + hidden_states


class AcousticTokenizerEncoderStem:
  def __init__(self, config: dict):
    self.conv = CausalConv1d(
      in_channels=config["channels"],
      out_channels=config["num_filters"],
      kernel_size=config["kernel_size"],
      cache_key="encoder_stem",
    )
    self.stage = [
      AcousticConvNext1dLayer(config, hidden_size=config["num_filters"], layer_idx=layer_idx)
      for layer_idx in range(1, config["depths"][0] + 1)
    ]

  def __call__(self, hidden_states: Tensor, padding_cache: Optional[Conv1dPaddingCache] = None) -> Tensor:
    hidden_states = self.conv(hidden_states, padding_cache=padding_cache)
    for block in self.stage:
      hidden_states = block(hidden_states, padding_cache=padding_cache)
    return hidden_states


class AcousticTokenizerEncoderLayer:
  def __init__(self, config: dict, stage_idx: int):
    depth_idx = stage_idx + 1
    layer_idx = sum(depth + 1 for depth in config["depths"][:depth_idx])
    intermediate_channels = int(config["num_filters"] * (2 ** depth_idx))

    self.conv = CausalConv1d(
      in_channels=int(config["num_filters"] * (2 ** stage_idx)),
      out_channels=intermediate_channels,
      kernel_size=int(config["downsampling_ratios"][stage_idx] * 2),
      cache_key=f"encoder_layer_{stage_idx}",
      stride=config["downsampling_ratios"][stage_idx],
    )
    self.stage = [
      AcousticConvNext1dLayer(config, hidden_size=intermediate_channels, layer_idx=layer_idx + offset)
      for offset in range(1, config["depths"][depth_idx] + 1)
    ]

  def __call__(self, hidden_states: Tensor, padding_cache: Optional[Conv1dPaddingCache] = None) -> Tensor:
    hidden_states = self.conv(hidden_states, padding_cache=padding_cache)
    for block in self.stage:
      hidden_states = block(hidden_states, padding_cache=padding_cache)
    return hidden_states


class AcousticTokenizerEncoder:
  def __init__(self, config: dict):
    self.stem = AcousticTokenizerEncoderStem(config)
    self.conv_layers = [AcousticTokenizerEncoderLayer(config, stage_idx) for stage_idx in range(len(config["downsampling_ratios"]))]
    self.head = CausalConv1d(
      in_channels=int(config["num_filters"] * (2 ** len(config["downsampling_ratios"]))),
      out_channels=config["hidden_size"],
      kernel_size=config["kernel_size"],
      cache_key="encoder_head",
    )

  def __call__(
    self,
    hidden_states: Tensor,
    padding_cache: Optional[Conv1dPaddingCache] = None,
    use_cache: bool = False,
  ) -> tuple[Tensor, Optional[Conv1dPaddingCache]]:
    if use_cache and padding_cache is None:
      padding_cache = Conv1dPaddingCache()

    hidden_states = self.stem(hidden_states, padding_cache=padding_cache)
    for layer in self.conv_layers:
      hidden_states = layer(hidden_states, padding_cache=padding_cache)
    hidden_states = self.head(hidden_states, padding_cache=padding_cache)
    latents = hidden_states.permute(0, 2, 1)
    return latents, padding_cache


class MultiModalProjector:
  def __init__(self, config: dict):
    text_hidden = config["text_config"]["hidden_size"]
    ac_hidden = config["acoustic_tokenizer_encoder_config"]["hidden_size"]
    se_hidden = config["semantic_tokenizer_encoder_config"]["hidden_size"]

    self.acoustic_linear_1 = nn.Linear(ac_hidden, text_hidden, bias=True)
    self.acoustic_norm = nn.RMSNorm(text_hidden, 1e-6)
    self.acoustic_linear_2 = nn.Linear(text_hidden, text_hidden, bias=True)

    self.semantic_linear_1 = nn.Linear(se_hidden, text_hidden, bias=True)
    self.semantic_norm = nn.RMSNorm(text_hidden, 1e-6)
    self.semantic_linear_2 = nn.Linear(text_hidden, text_hidden, bias=True)

  def __call__(self, acoustic_latents: Tensor, semantic_latents: Tensor) -> Tensor:
    acoustic_features = self.acoustic_linear_1(acoustic_latents)
    acoustic_features = self.acoustic_norm(acoustic_features)
    acoustic_features = self.acoustic_linear_2(acoustic_features)

    semantic_features = self.semantic_linear_1(semantic_latents)
    semantic_features = self.semantic_norm(semantic_features)
    semantic_features = self.semantic_linear_2(semantic_features)
    return acoustic_features + semantic_features


class QwenAttention:
  def __init__(self, dim: int, n_heads: int, n_kv_heads: int, max_context: int):
    self.n_heads = n_heads
    self.n_kv_heads = n_kv_heads
    self.head_dim = dim // n_heads
    self.n_rep = n_heads // n_kv_heads
    self.max_context = max_context

    self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=True)
    self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=True)
    self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=True)
    self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

  def __call__(self, x: Tensor, start_pos: Union[Variable, int], freqs_cis: Tensor, mask: Optional[Tensor]) -> Tensor:
    xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
    xq = xq.reshape(xq.shape[0], xq.shape[1], self.n_heads, self.head_dim)
    xk = xk.reshape(xk.shape[0], xk.shape[1], self.n_kv_heads, self.head_dim)
    xv = xv.reshape(xv.shape[0], xv.shape[1], self.n_kv_heads, self.head_dim)
    xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

    bsz, seqlen, _, _ = xq.shape
    if not hasattr(self, "cache_kv"):
      self.cache_kv = Tensor.zeros(2, bsz, self.max_context, self.n_kv_heads, self.head_dim, dtype=x.dtype).contiguous().realize()

    self.cache_kv[:, :, start_pos:start_pos + seqlen, :, :].assign(Tensor.stack(xk, xv)).realize()
    keys = self.cache_kv[0, :, 0:start_pos + seqlen, :, :]
    values = self.cache_kv[1, :, 0:start_pos + seqlen, :, :]

    keys, values = repeat_kv(keys, self.n_rep), repeat_kv(values, self.n_rep)
    xq, keys, values = xq.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2)
    attn = xq.scaled_dot_product_attention(keys, values, mask).transpose(1, 2)
    return self.wo(attn.reshape(bsz, seqlen, -1))


class QwenFeedForward:
  def __init__(self, dim: int, hidden_dim: int):
    self.w1 = nn.Linear(dim, hidden_dim, bias=False)
    self.w2 = nn.Linear(hidden_dim, dim, bias=False)
    self.w3 = nn.Linear(dim, hidden_dim, bias=False)

  def __call__(self, x: Tensor) -> Tensor:
    return self.w2(self.w1(x).silu() * self.w3(x))


class QwenBlock:
  def __init__(self, dim: int, hidden_dim: int, n_heads: int, n_kv_heads: int, norm_eps: float, max_context: int):
    self.attention = QwenAttention(dim, n_heads, n_kv_heads, max_context)
    self.feed_forward = QwenFeedForward(dim, hidden_dim)
    self.attention_norm = nn.RMSNorm(dim, norm_eps)
    self.ffn_norm = nn.RMSNorm(dim, norm_eps)

  def __call__(self, x: Tensor, start_pos: Union[Variable, int], freqs_cis: Tensor, mask: Optional[Tensor]) -> Tensor:
    h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
    return (h + self.feed_forward(self.ffn_norm(h))).contiguous()


class QwenTransformer:
  def __init__(
    self,
    dim: int,
    hidden_dim: int,
    n_heads: int,
    n_kv_heads: int,
    n_layers: int,
    norm_eps: float,
    vocab_size: int,
    rope_theta: float,
    max_context: int,
    jit: bool = True,
  ):
    self.layers = [QwenBlock(dim, hidden_dim, n_heads, n_kv_heads, norm_eps, max_context) for _ in range(n_layers)]
    self.norm = nn.RMSNorm(dim, norm_eps)
    self.tok_embeddings = nn.Embedding(vocab_size, dim)
    self.output = nn.Linear(dim, vocab_size, bias=False)
    self.max_context = max_context
    self.n_heads = n_heads
    self.freqs_cis = precompute_freqs_cis(dim // n_heads, max_context * 2, rope_theta).contiguous().requires_grad_(False)
    self.jit = jit
    self.forward_tokens_jit = TinyJit(self.forward_tokens_impl)
    self.next_token_jit = TinyJit(self.next_token_impl)

  def forward_with_embeds(self, inputs_embeds: Tensor, start_pos: Union[Variable, int]) -> Tensor:
    _bsz, seqlen, _ = inputs_embeds.shape
    h = inputs_embeds.contiguous()
    freqs_cis = self.freqs_cis.cast(h.dtype)[:, start_pos:start_pos + seqlen, :, :, :]
    mask = Tensor.full((1, 1, seqlen, start_pos + seqlen), float("-inf"), dtype=h.dtype, device=h.device).triu(start_pos + 1) if seqlen > 1 else None
    for layer in self.layers:
      h = layer(h, start_pos, freqs_cis, mask)
    # Prefill/generation only needs next-token logits, avoid projecting all sequence positions.
    h_last = self.norm(h[:, -1, :]).contiguous()
    return self.output(h_last)

  def forward_tokens_impl(self, tokens: Tensor, start_pos: Union[Variable, int]) -> Tensor:
    return self.forward_with_embeds(self.tok_embeddings(tokens), start_pos)

  def forward_tokens(self, tokens: Tensor, start_pos: int) -> Tensor:
    if self.jit and getenv("JIT", 1) and tokens.shape == (1, 1) and start_pos != 0:
      print("jit")
      return self.forward_tokens_jit(tokens, Variable("start_pos", 1, self.max_context - 1).bind(start_pos))
    return self.forward_tokens_impl(tokens, start_pos)

  def next_token_impl(self, tokens: Tensor, start_pos: Union[Variable, int]) -> Tensor:
    return self.forward_tokens_impl(tokens, start_pos).argmax(axis=-1)

  def next_token(self, tokens: Tensor, start_pos: int) -> int:
    if self.jit and getenv("JIT", 1) and tokens.shape == (1, 1) and start_pos != 0:
      tok = self.next_token_jit(tokens, Variable("start_pos", 1, self.max_context - 1).bind(start_pos))
    else:
      tok = self.next_token_impl(tokens, start_pos)
    return int(tok.item())


class VibeVoiceAsrTiny:
  def __init__(self, config: dict, max_context: int, jit: bool = True):
    self.config = config
    self.audio_token_id = config["audio_token_id"]
    self.acoustic_chunk_size = config["acoustic_tokenizer_chunk_size"]
    self.acoustic_hop_length = int(math.prod(config["acoustic_tokenizer_encoder_config"]["downsampling_ratios"]))
    self.acoustic_vae_std = float(config["acoustic_tokenizer_encoder_config"]["vae_std"])

    self.acoustic_tokenizer_encoder = AcousticTokenizerEncoder(config["acoustic_tokenizer_encoder_config"])
    self.semantic_tokenizer_encoder = AcousticTokenizerEncoder(config["semantic_tokenizer_encoder_config"])
    self.multi_modal_projector = MultiModalProjector(config)

    text_config = config["text_config"]
    self.language_model = QwenTransformer(
      dim=text_config["hidden_size"],
      hidden_dim=text_config["intermediate_size"],
      n_heads=text_config["num_attention_heads"],
      n_kv_heads=text_config["num_key_value_heads"],
      n_layers=text_config["num_hidden_layers"],
      norm_eps=text_config["rms_norm_eps"],
      vocab_size=text_config["vocab_size"],
      rope_theta=text_config["rope_parameters"]["rope_theta"],
      max_context=max_context,
      jit=jit,
    )

  def get_audio_features(self, input_values: Tensor, num_audio_placeholders: int, acoustic_chunk_size: Optional[int] = None) -> Tensor:
    if acoustic_chunk_size is None:
      acoustic_chunk_size = self.acoustic_chunk_size

    acoustic_encoder_cache: Optional[Conv1dPaddingCache] = None
    semantic_encoder_cache: Optional[Conv1dPaddingCache] = None
    acoustic_latents: list[Tensor] = []
    semantic_latents: list[Tensor] = []

    for offset in range(0, input_values.shape[-1], acoustic_chunk_size):
      chunk = input_values[:, :, offset:offset + acoustic_chunk_size]
      ac_latents, acoustic_encoder_cache = self.acoustic_tokenizer_encoder(chunk, padding_cache=acoustic_encoder_cache, use_cache=True)
      se_latents, semantic_encoder_cache = self.semantic_tokenizer_encoder(chunk, padding_cache=semantic_encoder_cache, use_cache=True)
      acoustic_latents.append(ac_latents)
      semantic_latents.append(se_latents)

    acoustic_latents = acoustic_latents[0].cat(*acoustic_latents[1:], dim=1) if len(acoustic_latents) > 1 else acoustic_latents[0]
    semantic_latents = semantic_latents[0].cat(*semantic_latents[1:], dim=1) if len(semantic_latents) > 1 else semantic_latents[0]

    if self.acoustic_vae_std > 0:
      noise_std = self.acoustic_vae_std * Tensor.randn(
        acoustic_latents.shape[0], device=acoustic_latents.device, dtype=acoustic_latents.dtype
      )
      acoustic_latents = acoustic_latents + noise_std.reshape(-1, 1, 1) * Tensor.randn(
        *acoustic_latents.shape, device=acoustic_latents.device, dtype=acoustic_latents.dtype
      )

    combined = self.multi_modal_projector(acoustic_latents, semantic_latents).realize()
    if combined.shape[1] < num_audio_placeholders:
      raise RuntimeError(
        f"not enough audio features ({combined.shape[1]}) for placeholders ({num_audio_placeholders})"
      )
    return combined[:, :num_audio_placeholders, :].reshape(-1, combined.shape[-1]).realize()

  def prefill(self, input_ids: np.ndarray, audio_embeds: Tensor) -> Tensor:
    tok_embeds = self.language_model.tok_embeddings(Tensor(input_ids, dtype=dtypes.int32)).realize()
    audio_positions = np.where(input_ids[0] == self.audio_token_id)[0]
    if len(audio_positions) == 0:
      raise RuntimeError("no audio placeholder tokens found in input_ids")
    if audio_embeds.shape[0] < len(audio_positions):
      raise RuntimeError(
        f"not enough audio embeddings ({audio_embeds.shape[0]}) for placeholders ({len(audio_positions)})"
      )

    tok_np = tok_embeds.numpy()
    audio_np = audio_embeds[:len(audio_positions)].numpy()
    tok_np[0, audio_positions, :] = audio_np
    inputs_embeds = Tensor(tok_np, dtype=tok_embeds.dtype, device=tok_embeds.device)
    return self.language_model.forward_with_embeds(inputs_embeds, 0).realize()

  def generate_stream(
    self,
    input_ids: np.ndarray,
    audio_embeds: Tensor,
    max_new_tokens: int,
    eos_token_id: int,
  ):
    logits = self.prefill(input_ids, audio_embeds)
    start_pos = input_ids.shape[1]
    next_token = int(logits.argmax(axis=-1).item())
    yield next_token
    if next_token == eos_token_id: return

    generated_count = 1
    while generated_count < max_new_tokens and start_pos < self.language_model.max_context:
      token_tensor = Tensor([[next_token]], dtype=dtypes.int32)
      next_token = self.language_model.next_token(token_tensor, start_pos)
      yield next_token
      generated_count += 1
      start_pos += 1
      if next_token == eos_token_id:
        break


class JsonLineStreamer:
  def __init__(self):
    self.buffer = ""
    self.scan_pos = 0
    self.in_array = False
    self.in_string = False
    self.escape = False
    self.depth = 0
    self.obj_start = -1
    self.done = False
    self.items: list[dict] = []

  def push(self, text: str) -> list[dict]:
    self.buffer += text
    out: list[dict] = []
    i = self.scan_pos
    while i < len(self.buffer):
      ch = self.buffer[i]
      if not self.in_array:
        if ch == "[":
          self.in_array = True
        i += 1
        continue

      if self.in_string:
        if self.escape:
          self.escape = False
        elif ch == "\\":
          self.escape = True
        elif ch == '"':
          self.in_string = False
        i += 1
        continue

      if ch == '"':
        self.in_string = True
      elif ch == "{":
        if self.depth == 0:
          self.obj_start = i
        self.depth += 1
      elif ch == "}":
        if self.depth > 0:
          self.depth -= 1
          if self.depth == 0 and self.obj_start != -1:
            obj_text = self.buffer[self.obj_start:i + 1]
            try:
              obj = json.loads(obj_text)
              if isinstance(obj, dict):
                out.append(obj)
                self.items.append(obj)
            except Exception:
              pass
      elif ch == "]" and self.depth == 0:
        self.done = True
      i += 1

    self.scan_pos = i
    return out


def normalize_audio(audio: np.ndarray, target_db_fs: float = -25.0, eps: float = 1e-6) -> np.ndarray:
  rms = np.sqrt(np.mean(audio**2))
  audio = audio * (10 ** (target_db_fs / 20.0) / (rms + eps))
  max_val = np.max(np.abs(audio))
  if max_val > 1.0:
    audio = audio / (max_val + eps)
  return audio.astype(np.float32, copy=False)


def prepare_audio(
  audio_path: Path,
  sampling_rate: int = 24000,
  pad_to_multiple_of: int = 3200,
) -> tuple[np.ndarray, int, float]:
  waveform, _ = librosa.load(str(audio_path), sr=sampling_rate, mono=True)
  waveform = normalize_audio(waveform)
  original_len = int(waveform.shape[0])
  padded_len = int(math.ceil(original_len / pad_to_multiple_of) * pad_to_multiple_of)
  padded = np.zeros((1, 1, padded_len), dtype=np.float32)
  padded[0, 0, :original_len] = waveform
  return padded, original_len, original_len / sampling_rate


def build_transcription_prompt(duration_seconds: float, num_audio_tokens: int, extra_prompt: Optional[str]) -> str:
  system_prompt = "You are a helpful assistant that transcribes audio input into text output in JSON format."
  audio_token = "<|box_start|>" * num_audio_tokens
  if extra_prompt:
    request = (
      f"This is a {duration_seconds:.2f} seconds audio, with extra info: {extra_prompt}\n\n"
      "Please transcribe it with these keys: Start time, End time, Speaker ID, Content"
    )
  else:
    request = (
      f"This is a {duration_seconds:.2f} seconds audio, please transcribe it with these keys: "
      "Start time, End time, Speaker ID, Content"
    )
  return (
    "<|im_start|>system\n"
    f"{system_prompt}<|im_end|>\n"
    "<|im_start|>user\n"
    f"<|object_ref_start|>{audio_token}<|object_ref_end|>\n"
    f"{request}<|im_end|>\n"
  )


def extract_json_like_output(text: str) -> tuple[Optional[list[dict]], str]:
  out = text.strip()
  if out.startswith("assistant"):
    out = out[len("assistant"):].strip()
  for marker in ("<|im_end|>", "<|endoftext|>"):
    if marker in out:
      out = out.split(marker, 1)[0].strip()

  if "[" in out and "]" in out:
    out = out[out.find("["):out.rfind("]") + 1]

  try:
    parsed = json.loads(out)
    if isinstance(parsed, list):
      return parsed, out
  except Exception:
    pass
  return None, out


def load_config(repo_id: str) -> dict:
  config_path = hf_hub_download(repo_id, "config.json")
  with open(config_path) as f:
    return json.load(f)


def build_model(repo_id: str, max_context: int, jit: bool) -> tuple[VibeVoiceAsrTiny, dict]:
  config = load_config(repo_id)
  model = VibeVoiceAsrTiny(config, max_context=max_context, jit=jit)

  weights = load_hf_index_weights(repo_id)
  lm_hf = {k[len("language_model."):]: v for k, v in weights.items() if k.startswith("language_model.")}
  lm_weights = convert_from_huggingface(
    lm_hf,
    n_layers=config["text_config"]["num_hidden_layers"],
    n_heads=config["text_config"]["num_attention_heads"],
    n_kv_heads=config["text_config"]["num_key_value_heads"],
  )

  other_weights = {k: v.to(device=Device.DEFAULT) for k, v in weights.items() if not k.startswith("language_model.")}
  lm_weights = fix_bf16(lm_weights)
  other_weights = fix_bf16(other_weights)

  state = {**other_weights, **{f"language_model.{k}": v for k, v in lm_weights.items()}}
  load_state_dict(model, state, strict=False, consume=True, verbose=False)
  return model, config


def main():
  parser = argparse.ArgumentParser(description="Pure tinygrad VibeVoice-ASR inference")
  parser.add_argument("audio", type=Path, help="Path to audio file")
  parser.add_argument("--repo", type=str, default="microsoft/VibeVoice-ASR-HF")
  parser.add_argument("--prompt", type=str, default=None, help="Optional context prompt/hotwords")
  parser.add_argument("--max-context", type=int, default=8192)
  parser.add_argument("--max-new-tokens", type=int, default=4096)
  parser.add_argument("--pad-to-multiple-of", type=int, default=3200)
  parser.add_argument("--acoustic-chunk-size", type=int, default=None)
  parser.add_argument("--audio-token-stride", type=int, default=1, help="Compress audio tokens by averaging every N audio embeddings (N>1 is faster, may reduce quality)")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--no-jit", action="store_true")
  parser.add_argument("--stream-lines", action=argparse.BooleanOptionalAction, default=True, help="Stream parsed JSON items line-by-line during generation")
  args = parser.parse_args()
  if args.audio_token_stride < 1:
    raise ValueError("--audio-token-stride must be >= 1")

  np.random.seed(args.seed)
  Tensor.manual_seed(args.seed)

  tokenizer = AutoTokenizer.from_pretrained(args.repo)
  model, config = build_model(args.repo, max_context=args.max_context, jit=not args.no_jit)

  input_audio, original_len, audio_seconds = prepare_audio(
    args.audio,
    sampling_rate=24000,
    pad_to_multiple_of=args.pad_to_multiple_of,
  )

  t0 = time.perf_counter()
  input_values = Tensor(input_audio, dtype=dtypes.float16)
  full_audio_tokens = int(math.ceil(original_len / args.pad_to_multiple_of))
  audio_embeds = model.get_audio_features(
    input_values,
    num_audio_placeholders=full_audio_tokens,
    acoustic_chunk_size=args.acoustic_chunk_size,
  )

  if args.audio_token_stride > 1:
    emb_np = audio_embeds.numpy()
    stride = args.audio_token_stride
    pooled = [emb_np[i:i+stride].mean(axis=0) for i in range(0, emb_np.shape[0], stride)]
    emb_np = np.stack(pooled, axis=0).astype(np.float16, copy=False)
    audio_embeds = Tensor(emb_np, dtype=dtypes.float16)
    num_audio_tokens = emb_np.shape[0]
  else:
    num_audio_tokens = full_audio_tokens

  prompt_text = build_transcription_prompt(audio_seconds, num_audio_tokens, args.prompt)
  input_ids = np.array([tokenizer.encode(prompt_text, add_special_tokens=False)], dtype=np.int32)

  generated_ids: list[int] = []
  streamer = JsonLineStreamer()
  if args.stream_lines:
    print("--- 🔄 Streaming ---")
  for tok in model.generate_stream(
    input_ids=input_ids,
    audio_embeds=audio_embeds,
    max_new_tokens=args.max_new_tokens,
    eos_token_id=tokenizer.eos_token_id,
  ):
    generated_ids.append(tok)
    if args.stream_lines:
      piece = tokenizer.decode([tok], skip_special_tokens=False)
      for obj in streamer.push(piece):
        print(json.dumps(obj, ensure_ascii=False, separators=(",", ":")), flush=True)
  elapsed = time.perf_counter() - t0

  decoded = tokenizer.decode(generated_ids, skip_special_tokens=False)
  parsed, cleaned = extract_json_like_output(decoded)
  if parsed is None and len(streamer.items):
    parsed = streamer.items

  prompt_tokens = int(input_ids.shape[1])
  completion_tokens = int(len(generated_ids))
  total_tokens = prompt_tokens + completion_tokens
  tok_per_s = (completion_tokens / elapsed) if elapsed > 0 else 0.0

  print("--- ✅ Transcription Complete ---")
  print(f"⏱️ Time: {elapsed:.2f}s | 🎵 Audio: {audio_seconds:.2f}s")
  print(f"📊 Tokens: {prompt_tokens} (prompt) + {completion_tokens} (completion) = {total_tokens} (total)")
  print(f"⚡ Speed: {tok_per_s:.1f} tokens/s")
  print("---")
  if parsed is not None:
    print(json.dumps(parsed, ensure_ascii=False, separators=(",", ":")))
  else:
    print(cleaned if cleaned else decoded)


if __name__ == "__main__":
  main()
