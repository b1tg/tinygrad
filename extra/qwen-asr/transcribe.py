#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, os, time, wave
from typing import Callable

from tinygrad import Device, Tensor, TinyJit, Variable, dtypes
from tinygrad.nn.state import safe_load

SR, N_MELS, HOP, N_FFT = 16000, 128, 160, 400
TOK_IM_START, TOK_IM_END = 151644, 151645
TOK_AUDIO_START, TOK_AUDIO_END, TOK_AUDIO_PAD = 151669, 151670, 151676
TOK_EOT, TOK_ASR_TEXT = 151643, 151704
EOS = {TOK_EOT, TOK_IM_END}
PROMPT_PREFIX = [TOK_IM_START, 8948, 198, TOK_IM_END, 198, TOK_IM_START, 872, 198, TOK_AUDIO_START]
PROMPT_SUFFIX = [TOK_AUDIO_END, TOK_IM_END, 198, TOK_IM_START, 77091, 198]


def cfg(model_dir: str) -> dict:
  c = json.load(open(os.path.join(model_dir, "config.json"), encoding="utf-8"))
  ac, tc = c["thinker_config"]["audio_config"], c["thinker_config"]["text_config"]
  return {
    "ed": ac["d_model"], "el": ac["encoder_layers"], "eh": ac["encoder_attention_heads"],
    "enw": ac["n_window"], "enwi": ac["n_window_infer"],
    "dh": tc["hidden_size"], "dl": tc["num_hidden_layers"], "dah": tc["num_attention_heads"],
    "dkvh": tc["num_key_value_heads"], "dhd": tc["head_dim"], "deps": tc["rms_norm_eps"], "theta": tc["rope_theta"],
  }


def _pcm_to_floats(raw: bytes, sw: int, ch: int) -> list[float]:
  if sw not in (1, 2, 4):
    raise RuntimeError(f"unsupported wav sample width: {sw}")
  frame = sw * ch
  if len(raw) % frame != 0:
    raw = raw[:len(raw) - (len(raw) % frame)]
  out: list[float] = []
  for i in range(0, len(raw), frame):
    s = 0.0
    for c in range(ch):
      off = i + c * sw
      if sw == 1:
        v = raw[off] - 128
        s += v / 128.0
      elif sw == 2:
        v = int.from_bytes(raw[off:off+2], "little", signed=True)
        s += v / 32768.0
      else:
        v = int.from_bytes(raw[off:off+4], "little", signed=True)
        s += v / 2147483648.0
    out.append(s / ch)
  return out


def load_audio(path: str) -> tuple[list[float], int]:
  with wave.open(path, "rb") as wf:
    sr, ch, sw = wf.getframerate(), wf.getnchannels(), wf.getsampwidth()
    raw = wf.readframes(wf.getnframes())
  return _pcm_to_floats(raw, sw, ch), int(sr)


def resample(x: list[float], src: int, dst: int) -> list[float]:
  if src == dst:
    return x
  if len(x) <= 1:
    return x
  out_len = max(1, int(len(x) * (dst / src)))
  scale = (len(x) - 1) / (out_len - 1) if out_len > 1 else 0.0
  out: list[float] = []
  for i in range(out_len):
    idx = i * scale
    j = int(idx)
    if j >= len(x) - 1:
      out.append(x[-1])
      continue
    t = idx - j
    out.append(x[j] * (1.0 - t) + x[j + 1] * t)
  return out


def _hz_to_mel(freq: float) -> float:
  return (3.0 * freq / 200.0) if freq < 1000.0 else (15.0 + math.log(freq / 1000.0) * (27.0 / math.log(6.4)))


def _mel_to_hz(mel: float) -> float:
  return (200.0 * mel / 3.0) if mel < 15.0 else (1000.0 * math.exp((math.log(6.4) / 27.0) * (mel - 15.0)))


def compute_mel_filters() -> Tensor:
  # Slaney mel filters. Output shape: [n_fft//2+1, n_mels]
  n_bins = 1 + N_FFT // 2
  fft = Tensor.linspace(0.0, float(SR // 2), n_bins, dtype=dtypes.float32)
  m0, m1 = _hz_to_mel(0.0), _hz_to_mel(SR / 2)
  mel_pts = Tensor.linspace(m0, m1, N_MELS + 2, dtype=dtypes.float32)
  hz_pts = Tensor([_mel_to_hz(float(v)) for v in mel_pts.tolist()], dtype=dtypes.float32)
  fd = hz_pts[1:] - hz_pts[:-1]
  ramps = hz_pts.reshape(1, -1) - fft.reshape(-1, 1)
  lower = -ramps[:, :-2] / fd[:-1].reshape(1, -1)
  upper = ramps[:, 2:] / fd[1:].reshape(1, -1)
  w = lower.minimum(upper).maximum(0.0)
  return (w * (2.0 / (hz_pts[2:] - hz_pts[:-2])).reshape(1, -1)).float()


def compute_mel_spectrogram(audio: Tensor, mel_filters: Tensor) -> Tensor:
  if audio.ndim != 1: raise ValueError("audio must be 1D")
  if audio.shape[0] < 2: audio = Tensor.zeros(N_FFT, dtype=dtypes.float32)
  spec = audio.stft(
    n_fft=N_FFT, hop_length=HOP, win_length=N_FFT, window=Tensor.hann_window(N_FFT, dtype=dtypes.float32),
    center=True, pad_mode="reflect", return_complex=True
  )
  if spec.shape[1] > 1: spec = spec[:, :-1]
  m = (spec[:, :, 0].square() + spec[:, :, 1].square()).transpose(0, 1) @ mel_filters
  x = (m.maximum(1e-10).log() / math.log(10.0))
  return (x.maximum(x.max() - 8.0) + 4.0).div(4.0).T


class Weights:
  def __init__(self, model_dir: str):
    self.model_dir, self.cache = model_dir, {}
    idx = os.path.join(model_dir, "model.safetensors.index.json")
    s = os.path.join(model_dir, "model.safetensors")
    self.wmap = json.load(open(idx, encoding="utf-8"))["weight_map"] if os.path.exists(idx) else None
    self.single = safe_load(s) if self.wmap is None else None

  def __getitem__(self, k: str) -> Tensor:
    if self.wmap is None:
      t = self.single[k]
    else:
      shard = self.wmap[k]
      if shard not in self.cache:
        self.cache[shard] = safe_load(os.path.join(self.model_dir, shard))
      t = self.cache[shard][k]
    if isinstance(t.device, str) and t.device.startswith("DISK:"): t = t.to(Device.DEFAULT)
    return t.float() if t.dtype == dtypes.bfloat16 else t


def lin(x: Tensor, w: Tensor, b: Tensor | None = None) -> Tensor:
  y = x.linear(w.transpose())
  return y + b if b is not None else y


def ln(x: Tensor, w: Tensor, b: Tensor, eps=1e-5) -> Tensor: return x.layernorm(-1, eps) * w + b

def rms(x: Tensor, w: Tensor, eps=1e-6) -> Tensor:
  xf = x.float()
  return xf * (xf.square().mean(-1, keepdim=True) + eps).rsqrt() * w.float()


def sinpos(n: int, d: int, mx=10000) -> Tensor:
  its = (-math.log(mx) / (d // 2 - 1) * Tensor.arange(d // 2, dtype=dtypes.float32)).exp()
  a = Tensor.arange(n, dtype=dtypes.float32).reshape(n, 1) * its.reshape(1, -1)
  return a.sin().cat(a.cos(), dim=-1)


def rope(pos: Tensor, hd: int, theta: float) -> tuple[Tensor, Tensor]:
  inv = 1.0 / (theta ** (Tensor.arange(0, hd, 2, dtype=dtypes.float32) / hd))
  a = pos.float().reshape(-1, 1) * inv.reshape(1, -1)
  e = a.cat(a, dim=-1)
  return e.cos(), e.sin()


def apply_rope(x: Tensor, c: Tensor, s: Tensor, hd: int) -> Tensor:
  h = hd // 2
  x1, x2 = x[..., :h], x[..., h:]
  return x * c.unsqueeze(1) + (-x2).cat(x1, dim=-1) * s.unsqueeze(1)


def full_attn(q: Tensor, k: Tensor, v: Tensor, h: int, kh: int, hd: int, cs: list[int] | None = None) -> Tensor:
  if cs is not None and len(cs) > 2:
    outs = [full_attn(q[cs[i]:cs[i+1]], k[cs[i]:cs[i+1]], v[cs[i]:cs[i+1]], h, kh, hd) for i in range(len(cs)-1)]
    o = outs[0]
    for z in outs[1:]: o = o.cat(z, dim=0)
    return o
  sq = q.shape[0]
  qq = q.reshape(sq, h, hd).transpose(0, 1).unsqueeze(0)
  kk = k.reshape(sq, kh, hd).transpose(0, 1).unsqueeze(0)
  vv = v.reshape(sq, kh, hd).transpose(0, 1).unsqueeze(0)
  o = qq.float().scaled_dot_product_attention(kk.float(), vv.float(), enable_gqa=(h != kh))
  return o.squeeze(0).transpose(0, 1).reshape(sq, h * hd)


def causal_attn(q: Tensor, k: Tensor, v: Tensor, h: int, kh: int, hd: int, qp=0, kp=0) -> Tensor:
  sq, sk = q.shape[0], k.shape[0]
  qq = q.reshape(sq, h, hd).transpose(0, 1).unsqueeze(0)
  kk = k.reshape(sk, kh, hd).transpose(0, 1).unsqueeze(0)
  vv = v.reshape(sk, kh, hd).transpose(0, 1).unsqueeze(0)
  qi = (Tensor.arange(sq, dtype=dtypes.int32) + qp).reshape(sq, 1)
  ki = (Tensor.arange(sk, dtype=dtypes.int32) + kp).reshape(1, sk)
  m = (ki <= qi).reshape(1, 1, sq, sk)
  o = qq.float().scaled_dot_product_attention(kk.float(), vv.float(), attn_mask=m, enable_gqa=(h != kh))
  return o.squeeze(0).transpose(0, 1).reshape(sq, h * hd)


def encode(mel: Tensor, w: Weights, c: dict, use_jit=False, verbose=True) -> Tensor:
  p, ed, el, eh = "thinker.audio_tower", c["ed"], c["el"], c["eh"]
  hd, chunk = ed // eh, c["enw"] * 2
  conv = [
    (w[f"{p}.conv2d1.weight"], w[f"{p}.conv2d1.bias"]),
    (w[f"{p}.conv2d2.weight"], w[f"{p}.conv2d2.bias"]),
    (w[f"{p}.conv2d3.weight"], w[f"{p}.conv2d3.bias"]),
  ]
  outp = w[f"{p}.conv_out.weight"]

  chunks = []
  for s in range(0, mel.shape[1], chunk):
    x = mel[:, s:min(s + chunk, mel.shape[1])].unsqueeze(0).unsqueeze(0)
    for cw, cb in conv: x = x.conv2d(cw, cb, stride=2, padding=1).gelu()
    b, ch, f, t = x.shape
    chunks.append(x.permute(0, 3, 1, 2).reshape(b, t, ch * f).squeeze(0).realize())
  x = chunks[0]
  for z in chunks[1:]: x = x.cat(z, dim=0)
  if verbose: print(f"  Conv output: {mel.shape[1]} frames -> {x.shape[0]} tokens", file=os.sys.stderr)

  x = lin(x, outp).realize()
  tpc = chunks[0].shape[0]
  pe = sinpos(tpc, ed)
  off, ys = 0, []
  for z in chunks:
    ys.append(x[off:off + z.shape[0]] + pe[:z.shape[0]])
    off += z.shape[0]
  x = ys[0]
  for z in ys[1:]: x = x.cat(z, dim=0)

  win = tpc * (c["enwi"] // chunk)
  cs, pos = [0], 0
  while pos < x.shape[0]:
    pos = min(pos + win, x.shape[0])
    cs.append(pos)
  attn = TinyJit(lambda q, k, v: full_attn(q, k, v, eh, eh, hd, cs)) if use_jit else None

  for i in range(el):
    lp = f"{p}.layers.{i}"
    xn = ln(x, w[f"{lp}.self_attn_layer_norm.weight"], w[f"{lp}.self_attn_layer_norm.bias"])
    q = lin(xn, w[f"{lp}.self_attn.q_proj.weight"], w[f"{lp}.self_attn.q_proj.bias"])
    k = lin(xn, w[f"{lp}.self_attn.k_proj.weight"], w[f"{lp}.self_attn.k_proj.bias"])
    v = lin(xn, w[f"{lp}.self_attn.v_proj.weight"], w[f"{lp}.self_attn.v_proj.bias"])
    a = attn(q, k, v) if attn is not None else full_attn(q, k, v, eh, eh, hd, cs)
    x = (x + lin(a, w[f"{lp}.self_attn.out_proj.weight"], w[f"{lp}.self_attn.out_proj.bias"]))
    xn = ln(x, w[f"{lp}.final_layer_norm.weight"], w[f"{lp}.final_layer_norm.bias"])
    ff = lin(lin(xn, w[f"{lp}.fc1.weight"], w[f"{lp}.fc1.bias"]).gelu(), w[f"{lp}.fc2.weight"], w[f"{lp}.fc2.bias"])
    x = (x + ff).realize()

  x = ln(x, w[f"{p}.ln_post.weight"], w[f"{p}.ln_post.bias"])
  x = lin(lin(x, w[f"{p}.proj1.weight"], w[f"{p}.proj1.bias"]).gelu(), w[f"{p}.proj2.weight"], w[f"{p}.proj2.bias"]).realize()
  if verbose: print(f"  Encoder final output: [{x.shape[0]}, {x.shape[1]}]", file=os.sys.stderr)
  return x


class Decoder:
  def __init__(self, w: Weights, c: dict, use_jit=False, verbose=True):
    self.w, self.c, self.use_jit = w, c, use_jit
    self.h, self.kh, self.hd, self.l, self.eps, self.theta = c["dah"], c["dkvh"], c["dhd"], c["dl"], c["deps"], c["theta"]
    self.emb, self.lm, self.norm = w["thinker.model.embed_tokens.weight"], w["thinker.lm_head.weight"], w["thinker.model.norm.weight"]
    self.layers = []
    for i in range(self.l):
      p = f"thinker.model.layers.{i}"
      self.layers.append({
        "in": w[f"{p}.input_layernorm.weight"], "post": w[f"{p}.post_attention_layernorm.weight"],
        "q": w[f"{p}.self_attn.q_proj.weight"], "k": w[f"{p}.self_attn.k_proj.weight"], "v": w[f"{p}.self_attn.v_proj.weight"],
        "o": w[f"{p}.self_attn.o_proj.weight"], "qn": w[f"{p}.self_attn.q_norm.weight"], "kn": w[f"{p}.self_attn.k_norm.weight"],
        "g": w[f"{p}.mlp.gate_proj.weight"], "u": w[f"{p}.mlp.up_proj.weight"], "d": w[f"{p}.mlp.down_proj.weight"],
      })
      if verbose and (i + 1) % 8 == 0: print(f"  Decoder layer {i + 1}/{self.l} loaded", file=os.sys.stderr)
    self.kv, self.jits = {}, {}

  def tok(self, ids: Tensor | int) -> Tensor: return self.emb[ids]

  def _ca_jit(self, q: Tensor, k: Tensor, v: Tensor, qp: Variable | int, kp: Variable | int) -> Tensor:
    return causal_attn(q, k, v, self.h, self.kh, self.hd, qp, kp)

  def layer(self, h: Tensor, i: int, pos: int) -> Tensor:
    l, s = self.layers[i], h.shape[0]
    xn = rms(h, l["in"], self.eps)
    q, k, v = lin(xn, l["q"]), lin(xn, l["k"]), lin(xn, l["v"])
    q, k = q.reshape(s, self.h, self.hd), k.reshape(s, self.kh, self.hd)
    q, k = rms(q, l["qn"], self.eps), rms(k, l["kn"], self.eps)
    c, s0 = rope(Tensor.arange(pos, pos + s, dtype=dtypes.int32), self.hd, self.theta)
    q, k = apply_rope(q, c, s0, self.hd), apply_rope(k, c, s0, self.hd)
    q, k, v = q.reshape(s, self.h * self.hd), k.reshape(s, self.kh * self.hd), v.reshape(s, self.kh * self.hd)

    if i not in self.kv: kc, vc = k, v
    else:
      ok, ov = self.kv[i]
      kc, vc = ok.cat(k, dim=0), ov.cat(v, dim=0)
    self.kv[i] = (kc.realize(), vc.realize())
    kp = (pos + s - 1) - (kc.shape[0] - 1)

    if self.use_jit:
      key = (q.shape[0], kc.shape[0])
      if key not in self.jits: self.jits[key] = TinyJit(self._ca_jit)
      qs, ks = Variable("qpos", 0, 32768).bind(pos), Variable("kpos", 0, 32768).bind(kp)
      a = self.jits[key](q, kc, vc, qs, ks)
    else:
      a = causal_attn(q, kc, vc, self.h, self.kh, self.hd, pos, kp)

    h = (h + lin(a, l["o"])).realize()
    xn = rms(h, l["post"], self.eps)
    h = (h + lin(lin(xn, l["g"]).silu() * lin(xn, l["u"]), l["d"])).realize()
    return h

  def prefill(self, x: Tensor, verbose=True) -> Tensor:
    self.kv = {}
    for i in range(self.l):
      x = self.layer(x, i, 0)
      if verbose and (i < 2 or (i + 1) % 8 == 0): print(f"  Decoder prefill layer {i+1}/{self.l}", file=os.sys.stderr)
    return x

  def step(self, emb: Tensor, pos: int) -> Tensor:
    h = emb.unsqueeze(0) if emb.ndim == 1 else emb
    for i in range(self.l): h = self.layer(h, i, pos)
    return lin(rms(h, self.norm, self.eps).float().squeeze(0), self.lm)


# tinygrad/apps/llm.py-compatible byte mapping

def bytes_to_unicode() -> dict[int, str]:
  bs = [*range(33, 127), *range(161, 173), *range(174, 256)]
  return {b: chr(b) for b in bs} | {b: chr(256 + i) for i, b in enumerate(x for x in range(256) if x not in bs)}


def tokenizer(model_dir: str) -> Callable[[list[int]], str]:
  vocab = json.load(open(os.path.join(model_dir, "vocab.json"), encoding="utf-8"))
  id2tok = {v: k for k, v in vocab.items()}
  special = set()
  tc = os.path.join(model_dir, "tokenizer_config.json")
  if os.path.exists(tc):
    special |= {int(k) for k in json.load(open(tc, encoding="utf-8")).get("added_tokens_decoder", {}).keys()}
  bdec = {v: k for k, v in bytes_to_unicode().items()}

  def dec(ids: list[int]) -> str:
    pieces = ["<asr_text>" if i == TOK_ASR_TEXT else "" for i in ids if i in special]
    pieces += [id2tok.get(i, "") for i in ids if i not in special]
    raw = bytearray([bdec[c] for c in "".join(pieces) if c in bdec])
    return raw.decode("utf-8", errors="replace")

  return dec


def parse_asr_text(s: str) -> str:
  if "<asr_text>" in s: return s.split("<asr_text>", 1)[1]
  return s.split(maxsplit=2)[2] if s.lower().startswith("language ") and len(s.split(maxsplit=2)) >= 3 else s


def transcribe(model_dir: str, wav: str, max_new_tokens=1024, verbose=True, use_jit=False) -> dict:
  t0 = time.perf_counter()
  a, sr = load_audio(wav)
  if sr != SR:
    if verbose: print(f"Audio sample rate is {sr}, resampling to {SR}", file=os.sys.stderr)
    a = resample(a, sr, SR)
  if verbose: print(f"Audio: {len(a)} samples ({len(a)/SR:.1f}s)", file=os.sys.stderr)

  c = cfg(model_dir)
  w = Weights(model_dir)
  mfb = compute_mel_filters()
  tm = time.perf_counter()
  m = compute_mel_spectrogram(Tensor(a, dtype=dtypes.float32), mfb).realize()
  mel_s = time.perf_counter() - tm

  te = time.perf_counter()
  ae = encode(m, w, c, use_jit=use_jit, verbose=verbose).realize()
  enc_s = time.perf_counter() - te

  ids = PROMPT_PREFIX + [TOK_AUDIO_PAD] * ae.shape[0] + PROMPT_SUFFIX
  dec = Decoder(w, c, use_jit=use_jit, verbose=verbose)
  pre = dec.tok(Tensor(PROMPT_PREFIX, dtype=dtypes.int32))
  suf = dec.tok(Tensor(PROMPT_SUFFIX, dtype=dtypes.int32))
  emb = pre.cat(ae, dim=0).cat(suf, dim=0)

  td = time.perf_counter()
  if len(ids) > 1: dec.prefill(emb[:-1], verbose=verbose)
  t = int(dec.step(emb[-1], len(ids) - 1).realize().argmax().item())
  out = [t]
  for i in range(max_new_tokens - 1):
    if t in EOS: break
    t = int(dec.step(dec.tok(t), len(ids) + i).realize().argmax().item())
    out.append(t)
  dec_s = time.perf_counter() - td

  while out and out[-1] in EOS: out.pop()
  txt = parse_asr_text(tokenizer(model_dir)(out)).strip()
  return {
    "text": txt,
    "audio_seconds": len(a) / SR,
    "wall_seconds": time.perf_counter() - t0,
    "mel_seconds": mel_s,
    "encoder_seconds": enc_s,
    "decoder_seconds": dec_s,
    "generated_tokens": len(out),
    "tokens_per_second": (len(out) / dec_s) if dec_s > 0 else float("inf"),
  }


def main() -> None:
  p = argparse.ArgumentParser(description="Qwen3-ASR tinygrad transcriber")
  p.add_argument("model_dir")
  p.add_argument("audio")
  p.add_argument("--max-new-tokens", type=int, default=1024)
  p.add_argument("--silent", action="store_true")
  p.add_argument("--timings-json", action="store_true")
  p.add_argument("--jit", action="store_true", help="enable TinyJit attention path")
  a = p.parse_args()

  o = transcribe(a.model_dir, a.audio, max_new_tokens=a.max_new_tokens, verbose=not a.silent, use_jit=a.jit)
  print(o["text"])
  if a.timings_json: print(json.dumps({k: v for k, v in o.items() if k != "text"}, sort_keys=True), file=os.sys.stderr)


if __name__ == "__main__":
  main()
