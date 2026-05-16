from __future__ import annotations
import functools, itertools, math, pathlib
from dataclasses import dataclass, replace
from tinygrad import Device, Tensor, nn, UOp, TinyJit, dtypes, getenv, function
from tinygrad.llm.gguf import gguf_load, block_device
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, resolve

@functools.cache
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, device:str|None=None) -> Tensor:
  freqs = 1.0 / (theta ** (Tensor.arange(0, dim, 2, device=device)[:(dim // 2)] / dim))
  freqs = Tensor.arange(end, device=device).unsqueeze(dim=1) * freqs.unsqueeze(dim=0)
  return freqs.cos().cat(freqs.sin(), dim=-1).contiguous()

def _expert_matmul_kernel(output:UOp, x:UOp, weight:UOp, sel:UOp) -> UOp:
  B, T, K, OUT, IN, xk = output.shape[-4], output.shape[-3], output.shape[-2], output.shape[-1], weight.shape[-1], x.shape[-2]
  output, x, weight, sel = output.flatten(), x.flatten(), weight.flatten(), sel.flatten()
  b = UOp.range(B, 1, AxisType.LOOP)
  t = UOp.range(T, 2, AxisType.LOOP)
  k = UOp.range(K, 3, AxisType.LOOP)
  o = UOp.range(OUT, 4, AxisType.LOOP)
  r = UOp.range(IN, 0, AxisType.REDUCE)
  e = sel.index(b*T*K + t*K + k).cast(dtypes.weakint)
  xv = x.index(b*T*xk*IN + t*xk*IN + (k if xk != 1 else 0)*IN + r)
  wv = weight.index(e*OUT*IN + o*IN + r)
  ret = (xv * wv).cast(dtypes.float).reduce(r, arg=Ops.ADD).cast(output.dtype.base)
  off = b*T*K*OUT + t*K*OUT + k*OUT + o if len(output.shape) == 4 else b*T*K*OUT + t*K*OUT + k*OUT + o
  return output.index(off, ptr=True).store(ret).end(b, t, k, o).sink(arg=KernelInfo(name=f"expert_matmul_{x.shape}_{weight.shape}", beam=-1))

def _q4k(raw:UOp, idx:UOp) -> UOp:
  block, inb = idx // 256, idx % 256
  base, g, j = block * 144, inb // 32, inb % 32
  def u8(off): return raw.index(off).cast(dtypes.uint16)
  def half(off): return (u8(off) | (u8(off+1) << 8)).bitcast(dtypes.float16).cast(dtypes.float)
  qb = u8(base + 16 + (g//2)*32 + j)
  q = (g % 2).eq(0).where(qb & 15, qb >> 4).cast(dtypes.float)
  sc = (g < 4).where(u8(base+4+g) & 63, (u8(base+8+g) & 15) | ((u8(base+g) >> 6) << 4)).cast(dtypes.float)
  mn = (g < 4).where(u8(base+8+g) & 63, (u8(base+8+g) >> 4) | ((u8(base+4+g) >> 6) << 4)).cast(dtypes.float)
  return half(base) * sc * q - half(base+2) * mn

def _q5k(raw:UOp, idx:UOp) -> UOp:
  block, inb = idx // 256, idx % 256
  base, g, j = block * 176, inb // 32, inb % 32
  def u8(off): return raw.index(off).cast(dtypes.uint16)
  def half(off): return (u8(off) | (u8(off+1) << 8)).bitcast(dtypes.float16).cast(dtypes.float)
  qb = u8(base + 48 + (g//2)*32 + j)
  q = ((g % 2).eq(0).where(qb & 15, qb >> 4) + (((u8(base+16+j) >> g) & 1) << 4)).cast(dtypes.float)
  sc = (g < 4).where(u8(base+4+g) & 63, (u8(base+8+g) & 15) | ((u8(base+g) >> 6) << 4)).cast(dtypes.float)
  mn = (g < 4).where(u8(base+8+g) & 63, (u8(base+8+g) >> 4) | ((u8(base+4+g) >> 6) << 4)).cast(dtypes.float)
  return half(base) * sc * q - half(base+2) * mn

def _qk(raw:UOp, idx:UOp, typ:int) -> UOp:
  if typ == 2:
    block, inb = idx // 32, idx % 32
    base = block * 18
    def u8(off): return raw.index(off).cast(dtypes.uint16)
    qb = u8(base + 2 + inb % 16)
    q = (inb < 16).where(qb & 15, qb >> 4).cast(dtypes.float)
    return (u8(base) | (u8(base+1) << 8)).bitcast(dtypes.float16).cast(dtypes.float) * (q - 8)
  if typ == 12: return _q4k(raw, idx)
  if typ == 13: return _q5k(raw, idx)
  block, inb = idx // 256, idx % 256
  base, g, j = block * 136, inb // 32, inb % 32
  def u8(off): return raw.index(off).cast(dtypes.uint16)
  def half(off): return (u8(off) | (u8(off+1) << 8)).bitcast(dtypes.float16).cast(dtypes.float)
  qb = u8(base + 8 + g*16 + j % 16)
  q = (j < 16).where(qb & 15, qb >> 4).cast(dtypes.weakint)
  lut = q.eq(0).where(-127, q.eq(1).where(-104, q.eq(2).where(-83, q.eq(3).where(-65, q.eq(4).where(-49, q.eq(5).where(-35,
        q.eq(6).where(-22, q.eq(7).where(-10, q.eq(8).where(1, q.eq(9).where(13, q.eq(10).where(25, q.eq(11).where(38,
        q.eq(12).where(53, q.eq(13).where(69, q.eq(14).where(89, 113)))))))))))))))
  sl = (g % 2).eq(0).where(u8(base+4+g//2) & 15, u8(base+4+g//2) >> 4)
  sc = (sl | (((u8(base+2) | (u8(base+3) << 8)) >> (g*2) & 3) << 4)).cast(dtypes.float) - 32
  return half(base) * sc * lut.cast(dtypes.float)

def _expert_matmul_q4k_kernel(output:UOp, x:UOp, raw:UOp, sel:UOp, in_features:int, typ:int=12) -> UOp:
  B, T, K, OUT, xk = output.shape[-4], output.shape[-3], output.shape[-2], output.shape[-1], x.shape[-2]
  output, x, raw, sel = output.flatten(), x.flatten(), raw.flatten(), sel.flatten()
  b = UOp.range(B, 1, AxisType.LOOP)
  t = UOp.range(T, 2, AxisType.LOOP)
  k = UOp.range(K, 3, AxisType.LOOP)
  o = UOp.range(OUT, 4, AxisType.LOOP)
  r = UOp.range(in_features, 0, AxisType.REDUCE)
  e = sel.index(b*T*K + t*K + k).cast(dtypes.weakint)
  xv = x.index(b*T*xk*in_features + t*xk*in_features + (k if xk != 1 else 0)*in_features + r)
  ret = (xv * _qk(raw, e*OUT*in_features + o*in_features + r, typ)).cast(dtypes.float).reduce(r, arg=Ops.ADD).cast(output.dtype.base)
  off = b*T*K*OUT + t*K*OUT + k*OUT + o
  return output.index(off, ptr=True).store(ret).end(b, t, k, o).sink(arg=KernelInfo(name=f"expert_matmul_q4k_{x.shape}_{OUT}_{in_features}", beam=-1))

def _expert_ffn_q4k_kernel(output:UOp, x:UOp, gate_raw:UOp, up_raw:UOp, sel:UOp, in_features:int, typ:int=12) -> UOp:
  B, T, K, OUT, xk = output.shape[0], output.shape[1], output.shape[2], output.shape[3], x.shape[-2]
  output, x, gate_raw, up_raw, sel = output.flatten(), x.flatten(), gate_raw.flatten(), up_raw.flatten(), sel.flatten()
  b = UOp.range(B, 1, AxisType.LOOP)
  t = UOp.range(T, 2, AxisType.LOOP)
  k = UOp.range(K, 3, AxisType.LOOP)
  o = UOp.range(OUT, 4, AxisType.LOOP)
  r = UOp.range(in_features, 0, AxisType.REDUCE)
  e = sel.index(b*T*K + t*K + k).cast(dtypes.weakint)
  xv = x.index(b*T*xk*in_features + t*xk*in_features + (k if xk != 1 else 0)*in_features + r)
  off = e*OUT*in_features + o*in_features + r
  gate = (xv * _qk(gate_raw, off, typ)).cast(dtypes.float).reduce(r, arg=Ops.ADD)
  up = (xv * _qk(up_raw, off, typ)).cast(dtypes.float).reduce(r, arg=Ops.ADD)
  ret = (gate * (gate.const_like(1.0) + (gate * gate.const_like(-1/math.log(2))).alu(Ops.EXP2)).alu(Ops.RECIPROCAL) * up).cast(output.dtype.base)
  return output.index(b*T*K*OUT + t*K*OUT + k*OUT + o, ptr=True).store(ret).end(b, t, k, o).sink(
    arg=KernelInfo(name=f"expert_ffn_q4k_{x.shape}_{OUT}_{in_features}", beam=-1))

def _expert_output(shape:tuple[int, ...], ref:Tensor, axis:int) -> Tensor:
  local = tuple(s // len(ref.device) if i == axis else s for i,s in enumerate(shape))
  return Tensor(UOp.empty(local, ref.dtype, ref.device, axis), dtype=ref.dtype, device=ref.device)

def _reshard_last(x:Tensor, devices:tuple[str, ...]) -> Tensor:
  if isinstance(x.device, tuple) and x.uop.axis == x.ndim-1: return x
  return (x.to(devices[0]) if isinstance(x.device, tuple) else x).shard(devices, axis=-1)

def _expert_ffn_q4k(gate:ExpertWeights, up:ExpertWeights, sel:Tensor, x:Tensor) -> Tensor|None:
  if not (getenv("EXPERT_Q4K_FUSED", 1) and isinstance(gate.weight.device, tuple) and gate.weight.device == up.weight.device): return None
  if not (getattr(gate.weight, "_gguf_type", None) == getattr(up.weight, "_gguf_type", None) in (2, 12, 13, 23) and
          gate.weight.uop.axis == up.weight.uop.axis == 1): return None
  if not isinstance(x.device, tuple): x = x.to(gate.weight.device)
  if not isinstance(sel.device, tuple): sel = sel.to(gate.weight.device)
  return Tensor.custom_kernel(_expert_output((*sel.shape, gate.weight.shape[1]), gate.weight, len(sel.shape)), x,
                              gate.weight._gguf_raw, up.weight._gguf_raw, sel,
                              fxn=functools.partial(_expert_ffn_q4k_kernel, in_features=gate.weight.shape[2], typ=gate.weight._gguf_type))[0]

class ExpertWeights:
  """Like nn.Linear but with num_experts dimension. Weight shape: (num_experts, out_features, in_features)."""
  def __init__(self, num_experts:int, in_features:int, out_features:int):
    self.weight = Tensor.zeros(num_experts, out_features, in_features)
  def __call__(self, sel:Tensor, x:Tensor) -> Tensor:
    # sel: (B, T, k), x: (B, T, 1, in) or (B, T, k, in) -> output: (B, T, k, out)
    if isinstance(self.weight.device, tuple):
      if self.weight.uop.axis == 2: x = _reshard_last(x, self.weight.device)
      elif not isinstance(x.device, tuple): x = x.to(self.weight.device)
      if not isinstance(sel.device, tuple): sel = sel.to(self.weight.device)
      if getenv("EXPERT_Q4K_CUSTOM", 1) and getattr(self.weight, "_gguf_type", None) in (2, 12, 13, 23) and self.weight.uop.axis == 1:
        return Tensor.custom_kernel(_expert_output((*sel.shape, self.weight.shape[1]), self.weight, len(sel.shape)), x,
                                    self.weight._gguf_raw, sel, fxn=functools.partial(_expert_matmul_q4k_kernel,
                                                                                      in_features=self.weight.shape[2],
                                                                                      typ=self.weight._gguf_type))[0]
      if getenv("EXPERT_Q4K_CUSTOM", 1) and getattr(self.weight, "_gguf_type", None) in (2, 12, 13, 23) and self.weight.uop.axis == 2:
        ret = _expert_output((len(self.weight.device), *sel.shape, self.weight.shape[1]), self.weight, 0)
        return Tensor.custom_kernel(ret, x, self.weight._gguf_raw, sel, fxn=functools.partial(_expert_matmul_q4k_kernel,
                                    in_features=self.weight.uop.shard_shape[2], typ=self.weight._gguf_type))[0].sum(axis=0)
      if self.weight.uop.base.op is Ops.BUFFER and x.shape[-1] == self.weight.shape[-1] and self.weight.uop.axis == 1:
        return Tensor.custom_kernel(_expert_output((*sel.shape, self.weight.shape[1]), self.weight, len(sel.shape)), x, self.weight, sel,
                                    fxn=_expert_matmul_kernel)[0]
      if self.weight.uop.base.op is Ops.BUFFER and x.shape[-1] == self.weight.shape[-1] and self.weight.uop.axis == 2:
        ret = _expert_output((len(self.weight.device), *sel.shape, self.weight.shape[1]), self.weight, 0)
        return Tensor.custom_kernel(ret, x, self.weight, sel, fxn=_expert_matmul_kernel)[0].sum(axis=0)
    return (x.unsqueeze(-2) @ self.weight[sel].transpose(-1, -2)).contiguous().squeeze(-2)

def apply_rope(x:Tensor, freqs_cis:Tensor) -> Tensor:
  assert x.shape[-1] % 2 == 0
  cos, sin = freqs_cis.reshape(1, 1, x.shape[2], -1).chunk(2, dim=-1)
  x1, x2 = x.chunk(2, dim=-1)
  return (x1 * cos - x2 * sin).cat(x2 * cos + x1 * sin, dim=-1)

def pairwise_topk(x: Tensor, k: int) -> tuple[Tensor, Tensor]:
  n = x.shape[-1]
  vals = Tensor.arange(n, device=x.device).reshape(1,1,n).cast(x.dtype).expand(x.shape)
  cmp = (x.unsqueeze(-1) > x.unsqueeze(-2)) | ((x.unsqueeze(-1) == x.unsqueeze(-2)) & \
    (Tensor.arange(n, device=x.device).reshape(1,1,n,1) < Tensor.arange(n, device=x.device).reshape(1,1,1,n)))
  sel = Tensor.zeros_like(x).scatter(-1, cmp.sum(axis=-1).cast('int32'), vals)[:,:,n-k:].cast('int32')
  return x.gather(-1, sel), sel

@dataclass(frozen=True)
class SSMConfig:
  conv_kernel: int
  state_size: int
  group_count: int
  time_step_rank: int
  inner_size: int

@dataclass(frozen=True)
class TransformerConfig:
  num_blocks: int
  dim: int
  hidden_dim: int
  n_heads: int
  n_kv_heads: int
  norm_eps: float
  vocab_size: int
  head_dim: int
  rope_theta: float
  rope_dim: int
  v_head_dim: int
  max_context: int = 0
  qk_norm: int = 0
  num_experts: int = 0
  num_experts_per_tok: int = 0
  norm_topk_prob: bool = False
  q_lora_rank: int = 0
  kv_lora_rank: int = 0
  shared_expert_dim: int = 0
  full_attention_interval: int = 0
  attn_output_gate: bool = False
  ssm: SSMConfig|None = None
  shared_expert_gate: bool = True
  leading_dense_blocks: int = 0
  dense_hidden_dim: int = 0
  routed_scaling_factor: float = 1.0
  qkv_bias: bool = False
  expert_bias: bool = False

class FFNBlock:
  def __init__(self, config:TransformerConfig):
    self.config = config

    # --- RMSNorms --------------------------------------------------------
    self.attn_norm   = nn.RMSNorm(config.dim, config.norm_eps)
    self.ffn_norm    = nn.RMSNorm(config.dim, config.norm_eps)

    # --- feed-forward (MoE or dense) -------------------------------------
    if config.num_experts > 0:
      self.ffn_gate_inp = nn.Linear(config.dim, config.num_experts, bias=False)  # router
      if config.expert_bias: self.exp_probs_b = {"bias": Tensor.zeros(config.num_experts)}
      self.ffn_gate_exps = ExpertWeights(config.num_experts, config.dim, config.hidden_dim)
      self.ffn_up_exps = ExpertWeights(config.num_experts, config.dim, config.hidden_dim)
      self.ffn_down_exps = ExpertWeights(config.num_experts, config.hidden_dim, config.dim)
      if config.shared_expert_dim > 0:
        self.ffn_gate_shexp = nn.Linear(config.dim, config.shared_expert_dim, bias=False)
        self.ffn_up_shexp = nn.Linear(config.dim, config.shared_expert_dim, bias=False)
        self.ffn_down_shexp = nn.Linear(config.shared_expert_dim, config.dim, bias=False)
        if config.shared_expert_gate: self.ffn_gate_inp_shexp = {"weight": Tensor.zeros(config.dim)}
    else:
      self.ffn_gate    = nn.Linear(config.dim, config.hidden_dim, bias=False)
      self.ffn_up      = nn.Linear(config.dim, config.hidden_dim, bias=False)
      self.ffn_down    = nn.Linear(config.hidden_dim, config.dim, bias=False)

  def _feed_forward(self, x:Tensor) -> Tensor:
    if hasattr(self, 'ffn_gate_exps'):
      dev = x.device
      h = x.unsqueeze(2)  # (B, T, 1, D) - add expert dim for broadcasting
      logits = self.ffn_gate_inp(x)
      if hasattr(self, 'exp_probs_b'):
        probs = logits.sigmoid()
        _, sel = pairwise_topk(probs + self.exp_probs_b["bias"], self.config.num_experts_per_tok)
        probs = probs.gather(-1, sel)
        if self.config.norm_topk_prob: probs = probs / probs.sum(axis=-1, keepdim=True)
      else:
        vals, sel = pairwise_topk(logits, self.config.num_experts_per_tok)
        probs = vals.softmax(-1) if self.config.norm_topk_prob else logits.softmax(-1).gather(-1, sel)
      probs = probs * self.config.routed_scaling_factor
      h_exp = _expert_ffn_q4k(self.ffn_gate_exps, self.ffn_up_exps, sel, h)
      if h_exp is None: h_exp = (self.ffn_gate_exps(sel, h).silu() * self.ffn_up_exps(sel, h)).contiguous()
      x_down = self.ffn_down_exps(sel, h_exp)  # (B, T, k, D)
      if x_down.device != probs.device: probs = probs.to(x_down.device)
      out = (x_down * probs.unsqueeze(-1)).sum(axis=2)  # (B, T, D)
      if hasattr(self, 'ffn_gate_shexp'):
        sx = x.to(self.ffn_gate_shexp.weight.device) if (
          isinstance(self.ffn_gate_shexp.weight.device, tuple) and not isinstance(x.device, tuple)) else x
        shexp = self.ffn_down_shexp(self.ffn_gate_shexp(sx).silu().contiguous() * self.ffn_up_shexp(sx))
        if out.device != shexp.device: out = out.to(shexp.device)
        if hasattr(self, 'ffn_gate_inp_shexp'):
          shexp = shexp * (sx * self.ffn_gate_inp_shexp["weight"].to(sx.device)).sum(axis=-1, keepdim=True).sigmoid()
        out = out + shexp
      return out.to(dev) if out.device != dev else out
    # TODO: remove the need for this contiguous
    dev = x.device
    if isinstance(self.ffn_gate.weight.device, tuple) and not isinstance(x.device, tuple): x = x.to(self.ffn_gate.weight.device)
    h = self.ffn_gate(x).silu().contiguous() * self.ffn_up(x)
    ret = self.ffn_down(h)
    return ret.to(dev) if ret.device != dev else ret

  # given the token-prefix match, return how much cached state this block can still reuse
  def _reusable_prefix_len(self, prefix_len:int, cached_len:int) -> int: return prefix_len
  # return writes that reset this block's state after a cache mismatch
  def _state_reset_ops(self) -> list[Tensor]: return []
  def _init_state(self, x:Tensor): raise NotImplementedError
  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor: raise NotImplementedError

  def __call__(self, x: Tensor, start_pos: int|UOp):
    self._init_state(x)
    # we pass in the weights implicitly so we unpack the GGUF on the fly
    @function(precompile=True, allow_implicit=True)
    def _run(x:Tensor, start_pos:int|UOp):
      h =     x + self._attention(self.attn_norm(x), start_pos)
      return (h + self._feed_forward(self.ffn_norm(h))).contiguous()
    return _run(x, start_pos)

class TransformerBlock(FFNBlock):
  def __init__(self, config:TransformerConfig):
    super().__init__(config)
    assert config.v_head_dim == config.head_dim, "TransformerBlock requires v_head_dim == head_dim"

    # --- attention projections (all linear, bias-free) ------------------
    q_proj_out       = config.head_dim * config.n_heads * (2 if config.attn_output_gate else 1)
    kv_proj_out      = config.head_dim * config.n_kv_heads
    self.attn_q      = nn.Linear(config.dim, q_proj_out,  bias=config.qkv_bias)
    self.attn_k      = nn.Linear(config.dim, kv_proj_out, bias=config.qkv_bias)
    self.attn_v      = nn.Linear(config.dim, kv_proj_out, bias=config.qkv_bias)
    self.attn_output = nn.Linear(config.head_dim * config.n_heads, config.dim, bias=False)
    if config.qk_norm: self.attn_q_norm, self.attn_k_norm = nn.RMSNorm(config.qk_norm, config.norm_eps), nn.RMSNorm(config.qk_norm, config.norm_eps)

  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    q, k, v = self.attn_q(x), self.attn_k(x), self.attn_v(x)
    if self.config.qk_norm and self.config.qk_norm != self.config.head_dim: q, k = self.attn_q_norm(q), self.attn_k_norm(k)

    B, T, _ = x.shape
    if self.config.attn_output_gate:
      qg = q.reshape(B, T, self.config.n_heads, 2, self.config.head_dim)
      q, gate = qg[:, :, :, 0, :], qg[:, :, :, 1, :].reshape(B, T, self.config.n_heads * self.config.head_dim)
    q = q.reshape(B, T, self.config.n_heads,    self.config.head_dim).transpose(1, 2)  # (B,H,T,Hd)
    k = k.reshape(B, T, self.config.n_kv_heads, self.config.head_dim).transpose(1, 2)  # (B,KvH,T,Hd)
    v = v.reshape(B, T, self.config.n_kv_heads, self.config.head_dim).transpose(1, 2)  # (B,KvH,T,Hd)
    if self.config.qk_norm == self.config.head_dim: q, k = self.attn_q_norm(q), self.attn_k_norm(k)

    q = apply_rope(q[..., :self.config.rope_dim], self.freqs_cis[start_pos:start_pos+T]).cat(q[..., self.config.rope_dim:], dim=-1)
    k = apply_rope(k[..., :self.config.rope_dim], self.freqs_cis[start_pos:start_pos+T]).cat(k[..., self.config.rope_dim:], dim=-1)

    # NOTE: we don't want to change self.cache_kv, the function API doesn't support this well
    assigned_kv = Tensor(self.cache_kv.uop.after(self.cache_kv[:, :, :, start_pos:start_pos+T, :].uop.store(Tensor.stack(k, v).uop)))
    k = assigned_kv[0, :, :, 0:start_pos+T, :]
    v = assigned_kv[1, :, :, 0:start_pos+T, :]

    #self.cache_kv[:, :, :, start_pos:start_pos+T, :].assign(Tensor.stack(k, v))
    #k = self.cache_kv[0, :, :, 0:start_pos+T, :]
    #v = self.cache_kv[1, :, :, 0:start_pos+T, :]

    # NOTE: this mask is causal_lower_right, not the causal_upper_left generated by is_casual = True
    # TODO: this if statement should be removed and it shouldn't generate extra kernels
    mask = Tensor.full((1, 1, T, start_pos+T), float("-inf"), dtype=x.dtype, device=x.device).triu(start_pos+1) if resolve(T != 1) else None
    attn = q.scaled_dot_product_attention(k, v, attn_mask=mask, enable_gqa=True)     # (B,H,T,Hd)
    attn = attn.transpose(1, 2).reshape(B, T, -1)                                    # back to (B,T,D)
    return self.attn_output(attn if not self.config.attn_output_gate else (attn * gate.sigmoid()))

  def _init_state(self, x:Tensor):
    if not hasattr(self, "cache_kv"):
      # TODO: how is the dtype of this determined?
      self.cache_kv = Tensor.empty(2, x.shape[0], self.config.n_kv_heads, self.config.max_context, self.config.head_dim, device=x.device)
      self.freqs_cis = precompute_freqs_cis(self.config.rope_dim, self.config.max_context, self.config.rope_theta, device=x.device)

class MLATransformerBlock(FFNBlock):
  def __init__(self, config:TransformerConfig):
    super().__init__(config)
    qk_nope_head_dim = config.head_dim - config.rope_dim
    if config.q_lora_rank > 0:
      self.attn_q_a = nn.Linear(config.dim, config.q_lora_rank, bias=False)
      self.attn_q_a_norm = nn.RMSNorm(config.q_lora_rank, config.norm_eps)
      self.attn_q_b = nn.Linear(config.q_lora_rank, config.n_heads * config.head_dim, bias=False)
    else:
      self.attn_q = nn.Linear(config.dim, config.n_heads * config.head_dim, bias=False)
    self.attn_kv_a_mqa = nn.Linear(config.dim, config.kv_lora_rank + config.rope_dim, bias=False)
    self.attn_kv_a_norm = nn.RMSNorm(config.kv_lora_rank, config.norm_eps)
    self.attn_k_b = {"weight": Tensor.zeros(config.n_heads, config.kv_lora_rank, qk_nope_head_dim)}
    self.attn_v_b = {"weight": Tensor.zeros(config.n_heads, config.v_head_dim, config.kv_lora_rank)}
    self.attn_output = nn.Linear(config.n_heads * config.v_head_dim, config.dim, bias=False)

  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    B, T, _ = x.shape
    q_nope_head_dim = self.config.head_dim - self.config.rope_dim
    q_proj = self.attn_q_b(self.attn_q_a_norm(self.attn_q_a(x))) if self.config.q_lora_rank > 0 else self.attn_q(x)
    q = q_proj.reshape(B, T, self.config.n_heads, self.config.head_dim).transpose(1, 2)
    q_nope, q_rope = q[..., :q_nope_head_dim], q[..., q_nope_head_dim:]
    q = (q_nope @ self.attn_k_b["weight"].transpose(-1, -2)).cat(apply_rope(q_rope, self.freqs_cis[start_pos:start_pos+T]), dim=-1)

    kv_a = self.attn_kv_a_mqa(x)
    c_kv = self.attn_kv_a_norm(kv_a[..., :self.config.kv_lora_rank])
    k_rope = apply_rope(
      kv_a[..., self.config.kv_lora_rank:].reshape(B, T, 1, self.config.rope_dim).transpose(1, 2),
      self.freqs_cis[start_pos:start_pos+T])

    k_store = c_kv.reshape(B, 1, T, self.config.kv_lora_rank).cat(k_rope.reshape(B, 1, T, self.config.rope_dim), dim=-1)
    k = Tensor(self.cache_k.uop.after(self.cache_k[:, :, start_pos:start_pos+T, :].uop.store(k_store.uop)))[:, :, 0:start_pos+T, :]
    v = k[..., :self.config.kv_lora_rank]

    mask = Tensor.full((1, 1, T, start_pos+T), float("-inf"), dtype=x.dtype, device=x.device).triu(start_pos+1) if resolve(T != 1) else None
    attn = q @ k.transpose(-1, -2) * (1.0 / self.config.head_dim ** 0.5)
    if mask is not None: attn = attn + mask
    attn = attn.softmax(-1)
    attn = ((attn @ v) @ self.attn_v_b["weight"].transpose(-1, -2)).transpose(1, 2).reshape(B, T, -1)
    return self.attn_output(attn)

  def _init_state(self, x:Tensor):
    if not hasattr(self, "cache_k"):
      self.cache_k = Tensor.empty(x.shape[0], 1, self.config.max_context, self.config.kv_lora_rank + self.config.rope_dim, device=x.device)
      self.freqs_cis = precompute_freqs_cis(self.config.rope_dim, self.config.max_context, self.config.rope_theta, device=x.device)

class GatedDeltaNetBlock(FFNBlock):
  def __init__(self, config:TransformerConfig, ssm:SSMConfig):
    super().__init__(config)
    self.head_k_dim, self.num_k_heads, self.num_v_heads = ssm.state_size, ssm.group_count, ssm.time_step_rank
    assert self.num_v_heads % self.num_k_heads == 0
    self.head_v_dim, self.ssm_conv_kernel = ssm.inner_size // ssm.time_step_rank, ssm.conv_kernel
    self.conv_channels, self.q_dim = ssm.inner_size + 2*ssm.group_count*ssm.state_size, ssm.state_size*ssm.group_count
    self.attn_qkv, self.attn_gate = nn.Linear(config.dim, self.conv_channels, bias=False), nn.Linear(config.dim, ssm.inner_size, bias=False)
    self.ssm_alpha, self.ssm_beta = nn.Linear(config.dim, self.num_v_heads, bias=False), nn.Linear(config.dim, self.num_v_heads, bias=False)
    self.ssm_conv1d = {"weight": Tensor.zeros(self.conv_channels, self.ssm_conv_kernel)}
    self.ssm_dt = {"bias": Tensor.zeros(self.num_v_heads)}
    self.ssm_a = Tensor.zeros(self.num_v_heads)
    self.ssm_norm, self.ssm_out = nn.RMSNorm(self.head_v_dim, config.norm_eps), nn.Linear(ssm.inner_size, config.dim, bias=False)

  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    B, T, _ = x.shape
    assert T == 1, "GatedDeltaNetBlock currently only supports T=1"

    # input processing
    x = x.half()
    out_gate = self.attn_gate(x).reshape(B, 1, self.num_v_heads, self.head_v_dim)
    beta = self.ssm_beta(x).sigmoid().reshape(B, self.num_v_heads, 1, 1)
    alpha = ((self.ssm_alpha(x).float() + self.ssm_dt["bias"]).softplus() * self.ssm_a).reshape(B, self.num_v_heads, 1, 1).exp()

    # qkv conv
    conv_window = self.conv_state.cat(self.attn_qkv(x), dim=1)
    conv_out = (conv_window * self.ssm_conv1d["weight"].T.unsqueeze(0)).sum(1).silu()
    q, k, v = conv_out.split([self.q_dim, self.q_dim, self.conv_channels - 2*self.q_dim], dim=-1)
    q = q.reshape(B, self.num_k_heads, self.head_k_dim).normalize(dim=-1).repeat(1, self.num_v_heads//self.num_k_heads, 1)
    k = k.reshape(B, self.num_k_heads, self.head_k_dim).normalize(dim=-1).repeat(1, self.num_v_heads//self.num_k_heads, 1)
    v = v.reshape(B, self.num_v_heads, self.head_v_dim)
    q, k, v = q.mul(self.head_k_dim**-0.5).unsqueeze(-1), k.unsqueeze(-1), v.unsqueeze(-1)

    # recurrent
    recurrent_state = self.recurrent_state * alpha
    recurrent_state = recurrent_state + ((v - recurrent_state@k) * beta)@k.transpose(-1, -2)

    # store the updated state
    conv_state_store = self.conv_state.uop.store(conv_window[:, 1:, :].cast(self.conv_state.dtype).uop)
    recurrent_state_store = self.recurrent_state.uop.store(recurrent_state.cast(self.recurrent_state.dtype).uop)
    recurrent_state = Tensor(self.recurrent_state.uop.after(recurrent_state_store, conv_state_store))

    # output
    core_attn_out = self.ssm_norm((recurrent_state@q).squeeze(-1).reshape(B, 1, self.num_v_heads, self.head_v_dim))
    return self.ssm_out((core_attn_out * out_gate.silu()).reshape(B, 1, -1).cast(x.dtype))

  # recurrent state can't be partially reused after divergence, force a full rebuild
  def _state_reset_ops(self):
    return [self.conv_state.assign(Tensor.zeros_like(self.conv_state)),
            self.recurrent_state.assign(Tensor.zeros_like(self.recurrent_state))] if hasattr(self, "conv_state") else []
  def _reusable_prefix_len(self, prefix_len:int, cached_len:int) -> int: return 0 if prefix_len != cached_len else prefix_len

  def _init_state(self, x):
    if not hasattr(self, "conv_state"):
      self.conv_state = Tensor.zeros(x.shape[0], self.ssm_conv_kernel-1, self.conv_channels, device=x.device).clone()
      self.recurrent_state = Tensor.zeros(x.shape[0], self.num_v_heads, self.head_v_dim, self.head_v_dim, device=x.device).clone()

class Transformer:
  def __init__(self, config:TransformerConfig):
    dense_config = replace(config, num_experts=0, num_experts_per_tok=0, shared_expert_dim=0, hidden_dim=config.dense_hidden_dim or config.hidden_dim)
    if config.ssm: config = replace(config, qk_norm=config.head_dim)
    block_cls = MLATransformerBlock if config.kv_lora_rank > 0 else TransformerBlock
    self.blk:list[FFNBlock] = [GatedDeltaNetBlock(config, config.ssm) if config.ssm and (i+1) % config.full_attention_interval != 0 else
                               block_cls(dense_config if i < config.leading_dense_blocks else config) for i in range(config.num_blocks)]
    self.token_embd  = nn.Embedding(config.vocab_size, config.dim)
    self.output_norm = nn.RMSNorm(config.dim, config.norm_eps)
    self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
    self.max_context = config.max_context
    self.has_recurrent_block = any(isinstance(b, GatedDeltaNetBlock) for b in self.blk)
    self._cached_tokens: list[int] = []
    # we specialize the JIT for prefill and rollout
    self.prefill_jit = TinyJit(self.forward)
    self.rollout_jit = TinyJit(self.forward)

  def forward(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor) -> Tensor:
    x = self.token_embd(tokens.to(self.token_embd.weight.device)).float()                   # (B, T, D)
    for block in self.blk: x = block(x.to(getattr(block.attn_norm, "weight").device), start_pos)
    logits = self.output(self.output_norm(x.to(self.output.weight.device)))[:, -1, :]
    # Gumbel-max trick: argmax(logits/temp - log(-log(uniform))) is equivalent to sampling from softmax(logits/temp)
    return (logits / temperature.to(logits.device).maximum(1e-12) -
            (Tensor.rand_like(logits).maximum(1e-12).log().neg()).log()).argmax(-1, keepdim=True)

  def __call__(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor) -> Tensor:
    return (self.prefill_jit if resolve(tokens.shape[1] != 1) else self.rollout_jit)(tokens.contiguous(), start_pos, temperature)

  @staticmethod
  def from_gguf(gguf:Tensor|str|pathlib.Path, max_context:int|None=None,
                realize=bool(getenv("REALIZE", 0)), shard:int=1) -> tuple[Transformer, dict]:
    devices = tuple(f"{Device.DEFAULT}:{i}" for i in range(shard)) if shard > 1 else None
    load_dtype = 'float16' if devices and getenv("HALF", 1) else None
    kv, state_dict = gguf_load(gguf, devices=devices, shard_axis=_gguf_shard_axis if devices else None, dtype=load_dtype)
    arch = kv['general.architecture']
    if devices and getenv("GGUF_REALIZE_SHARDED", 0) and kv.get('general.file_type') in (0, 1, 7, 32): realize = True

    # all state items should be float16, not float32
    if load_dtype is None: state_dict = {k:v.cast('float16') if getenv("HALF", 1) else v for k,v in state_dict.items()}

    # some models like Llama 3.2 don't have an output.weight, they just tie to the token_embd.weight
    if 'output.weight' not in state_dict: state_dict['output.weight'] = state_dict['token_embd.weight']

    max_context = min(max_context, kv[f'{arch}.context_length']) if max_context is not None else kv[f'{arch}.context_length']
    n_heads, n_kv_heads = kv[f'{arch}.attention.head_count'], kv[f'{arch}.attention.head_count_kv']

    ssm = None
    if arch in ('qwen35', 'qwen35moe'):
      ssm = SSMConfig(**{k: kv[f'{arch}.ssm.{k}'] for k in ('conv_kernel','state_size','group_count','time_step_rank','inner_size')})
    if arch in ('qwen35', 'qwen35moe', 'glm4moe'):
      state_dict = {k.replace('post_attention_norm', 'ffn_norm'):v for k,v in state_dict.items()}

    kv_lora_rank = kv.get(f'{arch}.attention.kv_lora_rank', 0)
    head_dim = kv.get(f'{arch}.attention.key_length_mla', kv.get(f'{arch}.attention.key_length', kv[f'{arch}.embedding_length'] // n_heads))
    rope_dim = kv.get(f'{arch}.rope.dimension_count', head_dim)

    # Permute RoPE weights from interleaved to half-split layout.
    for name in state_dict:
      if ('attn_q.weight' in name or 'attn_q_b.weight' in name) and (arch == 'llama' or kv_lora_rank):
        w = state_dict[name].reshape(n_heads, state_dict[name].shape[0]//n_heads, -1)
        prefix = head_dim-rope_dim
        state_dict[name] = w[:, :prefix].cat(w[:, prefix:].rearrange("n (h two) d -> n (two h) d", two=2), dim=1).reshape(-1, w.shape[-1])
      elif arch == 'llama' and 'attn_k.weight' in name:
        w = state_dict[name].reshape(n_kv_heads, state_dict[name].shape[0]//n_kv_heads, -1)
        state_dict[name] = w.rearrange("n (h two) d -> n (two h) d", two=2).reshape(-1, w.shape[-1])
      elif kv_lora_rank and 'attn_kv_a_mqa.weight' in name:
        state_dict[name] = state_dict[name][:kv_lora_rank].cat(state_dict[name][kv_lora_rank:].rearrange("(h two) d -> (two h) d", two=2), dim=0)
    config = TransformerConfig(
      num_blocks=kv[f'{arch}.block_count'] - kv.get(f'{arch}.nextn_predict_layers', 0), dim=kv[f'{arch}.embedding_length'],
      hidden_dim=kv.get(f'{arch}.expert_feed_forward_length', kv.get(f'{arch}.feed_forward_length', 0)),
      n_heads=n_heads, n_kv_heads=n_kv_heads, norm_eps=kv[f'{arch}.attention.layer_norm_rms_epsilon'],
      vocab_size=len(kv['tokenizer.ggml.tokens']),
      head_dim=head_dim,
      rope_theta=kv[f'{arch}.rope.freq_base'],
      rope_dim=rope_dim,
      v_head_dim=kv.get(f'{arch}.attention.value_length_mla', kv.get(f'{arch}.attention.value_length', head_dim)),
      max_context=max_context,
      qk_norm=int(state_dict['blk.0.attn_q_norm.weight'].shape[0]) if 'blk.0.attn_q_norm.weight' in state_dict else 0,
      num_experts=kv.get(f'{arch}.expert_count', 0), num_experts_per_tok=kv.get(f'{arch}.expert_used_count', 0),
      norm_topk_prob=kv.get(f'{arch}.expert_weights_norm', arch in ('qwen3moe', 'qwen35moe')),
      kv_lora_rank=kv_lora_rank, q_lora_rank=kv.get(f'{arch}.attention.q_lora_rank', 0),
      leading_dense_blocks=kv.get(f'{arch}.leading_dense_block_count', 0),
      shared_expert_dim=kv.get(
        f'{arch}.expert_shared_feed_forward_length',
        kv.get(f'{arch}.expert_shared_count', 0) * kv.get(f'{arch}.expert_feed_forward_length', 0)),
      shared_expert_gate=f"blk.{kv.get(f'{arch}.leading_dense_block_count', 0)}.ffn_gate_inp_shexp.weight" in state_dict,
      dense_hidden_dim=kv.get(f'{arch}.feed_forward_length', 0) if kv.get(f'{arch}.leading_dense_block_count', 0) else 0,
      routed_scaling_factor=kv.get(f'{arch}.expert_weights_scale', 1.0), attn_output_gate=arch in ('qwen35', 'qwen35moe'), ssm=ssm,
      full_attention_interval=kv.get(f'{arch}.full_attention_interval', 0),
      qkv_bias='blk.0.attn_q.bias' in state_dict,
      expert_bias=f"blk.{kv.get(f'{arch}.leading_dense_block_count', 0)}.exp_probs_b.bias" in state_dict)
    model = Transformer(config)
    if devices: shard_weights(model, devices)
    nn.state.load_state_dict(model, state_dict, verbose=False, consume=True, realize=False)  # NOTE: rope_freqs.weight (32,) is unused
    # NOTE: without this contiguous, it unpacks the weights from the model every time. we shouldn't need this, but for now it's faster
    if realize:
      params = nn.state.get_parameters(model)
      for s in params: s.replace(s.contiguous())
      if devices:
        for s in params: s.realize()
      else:
        Tensor.realize(*params)
    if devices and arch == "deepseek2" and config.num_experts and kv.get('general.file_type') not in (0, 1, 32):
      model.rollout_jit.beam = 1
    return model, kv

  def get_start_pos(self, tokens:list[int]) -> int:
    prefix_len = sum(1 for _ in itertools.takewhile(lambda ab: ab[0] == ab[1], zip(tokens[:-1], self._cached_tokens)))
    return min(block._reusable_prefix_len(prefix_len, len(self._cached_tokens)) for block in self.blk)

  def generate(self, tokens:list[int], chunk_size:int=32, temperature:float=0.0):
    if self.has_recurrent_block: chunk_size = 1
    sharded = any(isinstance(x.device, tuple) for x in nn.state.get_parameters(self))
    if sharded and len(tokens) > 1: chunk_size = len(tokens)
    v_start_pos = UOp.variable("start_pos", 0, self.max_context-1)
    v_toks = UOp.variable("toks", 1, chunk_size)
    # TODO: use UOp.variable for temperature once float variables are supported
    temp = Tensor(temperature).contiguous()
    # assign all input tokens once, then slice from start_pos for the model call
    t = Tensor(tokens + [0] * (self.max_context - len(tokens)), dtype="int32").reshape(1, self.max_context)
    # recompute start_pos from what's currently valid in the caches
    start_pos = self.get_start_pos(tokens)
    if start_pos < len(self._cached_tokens) and (resets := [r for b in self.blk for r in b._state_reset_ops()]): Tensor.realize(*resets)
    out, prompt_len = None, len(tokens)
    while len(tokens) < self.max_context:
      sp, nt = v_start_pos.bind(start_pos), v_toks.bind(min(chunk_size, len(tokens) - start_pos))
      if sharded:
        inp = t[:, start_pos:start_pos+nt.val] if start_pos < prompt_len else out
        out = self(inp, start_pos if start_pos < prompt_len else sp, temp).to(t.device).realize()
      else:
        out = self(t[:, sp:sp+nt] if start_pos < prompt_len or out is None else out, sp, temp).realize()
      start_pos += nt.val
      # chunked prefill: keep processing until all prompt tokens are consumed
      if start_pos < len(tokens): continue
      tokens.append(int(out.item()))
      self._cached_tokens = tokens[:-1]
      yield tokens[-1]

def _shard_axis_key(k:str, ndim:int, expert_axis:int|None=0, attention:bool=False) -> int|None:
  if k.endswith(".bias"): return 0 if attention and (".attn_q." in k or ".attn_k." in k or ".attn_v." in k) else None
  if ndim <= 1 or "norm" in k or "scale" in k: return None
  if k == "output.weight": return 0
  if attention and ".attn_q.weight" in k: return 0
  if attention and ".attn_q_b.weight" in k: return 0
  if attention and (".attn_k_b.weight" in k or ".attn_v_b.weight" in k): return 0
  if attention and ".attn_output.weight" in k: return -1
  if ".attn_q_a.weight" in k or ".attn_kv_a_mqa.weight" in k: return None
  if ".ffn_gate_inp.weight" in k: return None
  if "_exps.weight" in k:
    return None if expert_axis is None else (-1 if ".ffn_down_exps.weight" in k or getenv("EXPERT_GATEUP_INPUT_SHARD", 0) else 1)
  if ".ffn_down" in k: return -1
  if ".ffn_gate" in k or ".ffn_up" in k: return 0
  return None

def _shard_axis(k:str, v:Tensor, expert_axis:int|None=0) -> int|None: return _shard_axis_key(k, v.ndim, expert_axis)

def _gguf_shard_axis(k:str, dims:tuple[int, ...], typ:int, kv:dict) -> int|None:
  arch = kv["general.architecture"]
  return _shard_axis_key(k, len(dims), attention=not kv.get(f"{arch}.ssm.conv_kernel", 0))

def shard_weights(model:Transformer, devices:tuple[str, ...], expert_axis:int|None=0):
  tensor_attn = {i: type(b) in (TransformerBlock, MLATransformerBlock) for i,b in enumerate(model.blk)}
  for k,v in nn.state.get_state_dict(model).items():
    bid = int(k.split(".")[1]) if k.startswith("blk.") else None
    if (axis:=_shard_axis_key(k, v.ndim, expert_axis, tensor_attn.get(bid, False))) is not None: v.shard_(devices, axis=axis)
    elif bid is not None and tensor_attn[bid]: v.shard_(devices, axis=None)
    elif k.startswith("blk."): v.to_(block_device(devices, int(k.split(".")[1]), len(model.blk)))
    elif k == "token_embd.weight": v.to_(devices[0])
    else: v.shard_(devices, axis=None)
