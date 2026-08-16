from __future__ import annotations
import enum, functools, itertools, math, pathlib
from dataclasses import dataclass, replace
from tinygrad import Tensor, nn, UOp, TinyJit, getenv, function, dtypes
from tinygrad.nn import Linear
from tinygrad.helpers import ceildiv
from tinygrad.llm.gguf import gguf_load
from tinygrad.uop.ops import resolve, smax

class ExpertGating(enum.IntEnum):
  SOFTMAX = 1
  SIGMOID = 2
  SOFTMAX_WEIGHT = 3  # softmax over the top-k selected logits
  SQRT_SOFTPLUS = 4

@functools.cache
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, device:str|None=None,
                         yarn:tuple[float, int, float, float]|None=None) -> Tensor:
  freqs = 1.0 / (theta ** (Tensor.arange(0, dim, 2)[:(dim // 2)] / dim))
  if yarn is not None:
    factor, original_context, beta_fast, beta_slow = yarn
    def correction(rotations:float): return dim * math.log(original_context / (rotations * 2 * math.pi)) / (2 * math.log(theta))
    low, high = max(math.floor(correction(beta_fast)), 0), min(math.ceil(correction(beta_slow)), dim-1)
    smooth = 1 - ((Tensor.arange(dim//2) - low) / (high - low if high != low else 0.001)).clip(0, 1)
    freqs = freqs / factor * (1 - smooth) + freqs * smooth
  freqs = Tensor.arange(end).unsqueeze(dim=1) * freqs.unsqueeze(dim=0)
  return freqs.cos().cat(freqs.sin(), dim=-1).clone(device)

class ExpertWeights:
  """Like Linear but with num_experts dimension. Weight shape: (num_experts, out_features, in_features)."""
  def __init__(self, num_experts:int, in_features:int, out_features:int):
    self.weight = Tensor.zeros(num_experts, out_features, in_features)
  def __call__(self, sel:Tensor, x:Tensor) -> Tensor:
    # sel: (B, T, k), x: (B, T, 1, in) or (B, T, k, in) -> output: (B, T, k, out)
    return (x.unsqueeze(-2) @ self.weight[sel].transpose(-1, -2)).contiguous().squeeze(-2)

class MXFP4ExpertWeights:
  def __init__(self, num_experts:int, in_features:int, out_features:int):
    assert in_features % 32 == 0, f"MXFP4 input features must be divisible by 32, got {in_features}"
    self.weight = Tensor.zeros(num_experts, out_features, in_features//32, 17, dtype="uint8")
  def __call__(self, sel:Tensor, x:Tensor) -> Tensor:
    blocks = self.weight[sel]
    e = blocks[..., 0].cast(dtypes.uint32)
    d = (e < 2).where(Tensor([0x00200000, 0x00400000], dtype=dtypes.uint32, device=blocks.device)[e.clip(0, 1)],
                      (e - 1) * 0x00800000).bitcast(dtypes.float32).unsqueeze(-1)
    codes = blocks[..., 1:].unsqueeze(-1).div(Tensor.const((1, 16), dtypes.uint8), rounding_mode="trunc").bitwise_and(15)
    codes = codes.transpose(-1, -2).flatten(-2)
    values = Tensor([0., 1., 2., 3., 4., 6., 8., 12., -0., -1., -2., -3., -4., -6., -8., -12.],
                    device=blocks.device)[codes]
    weights = (values*d).flatten(-2).cast(x.dtype)
    return (x.unsqueeze(-2) @ weights.transpose(-1, -2)).squeeze(-2)

def swiglu(gate:Tensor, up:Tensor, limit:float|None=None) -> Tensor:
  if limit is not None and limit > 0: gate, up = gate.clamp(max_=limit), up.clamp(-limit, limit)
  return gate.silu() * up

def apply_rope(x:Tensor, freqs_cis:Tensor, interleaved:bool=False, inverse:bool=False) -> Tensor:
  assert x.shape[-1] % 2 == 0
  if interleaved:
    cos, sin = freqs_cis.reshape((1, freqs_cis.shape[0]) + (1,)*(x.ndim-3) + (x.shape[-1],)).chunk(2, dim=-1)
    x1, x2 = x[..., ::2], x[..., 1::2]
  else:
    cos, sin = freqs_cis.reshape(1, 1, x.shape[2], -1).chunk(2, dim=-1)
    x1, x2 = x.chunk(2, dim=-1)
  if inverse: sin = -sin
  x1, x2 = x1*cos-x2*sin, x1*sin+x2*cos
  return Tensor.stack(x1, x2, dim=-1).flatten(-2) if interleaved else x1.cat(x2, dim=-1)

def hadamard_transform(x:Tensor) -> Tensor:
  size, width = int(x.shape[-1]), 1
  assert size > 0 and size & (size-1) == 0
  while width < size:
    y = x.reshape(x.shape[:-1] + (size//(2*width), 2, width))
    a, b = y[..., 0, :], y[..., 1, :]
    x = Tensor.stack(a+b, a-b, dim=-2).flatten(-3).contiguous()
    width *= 2
  return x * size**-0.5

def pairwise_topk(x: Tensor, k: int) -> tuple[Tensor, Tensor]:
  n = x.shape[-1]
  vals = Tensor.arange(n).reshape(1,1,n).cast(x.dtype).expand(x.shape)
  cmp = (x.unsqueeze(-1) > x.unsqueeze(-2)) | ((x.unsqueeze(-1) == x.unsqueeze(-2)) & \
    (Tensor.arange(n).reshape(1,1,n,1) < Tensor.arange(n).reshape(1,1,1,n)))
  sel = x.const_like(0).scatter(-1, cmp.sum(axis=-1).cast('int32'), vals)[:,:,n-k:].cast('int32')
  return x.gather(-1, sel), sel

@dataclass(frozen=True)
class SSMConfig:
  conv_kernel: int
  state_size: int
  group_count: int
  time_step_rank: int
  inner_size: int
  kda: bool = False

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
  sliding_window: int = 0
  yarn: tuple[float, int, float, float]|None = None
  qk_norm: int = 0
  num_experts: int = 0
  num_experts_per_tok: int = 0
  norm_topk_prob: bool = False
  expert_gating_func: ExpertGating = ExpertGating.SOFTMAX
  q_lora_rank: int = 0
  kv_lora_rank: int = 0
  shared_expert_dim: int = 0
  ssm_layers: tuple[bool, ...] = ()
  attn_output_gate: bool = False
  attn_sinks: bool = False
  ssm: SSMConfig|None = None
  shared_expert_gate: bool = True
  leading_dense_blocks: int = 0
  dense_hidden_dim: int = 0
  routed_scaling_factor: float = 1.0
  swiglu_clamp: tuple[float, ...] = ()
  shared_swiglu_clamp: tuple[float, ...] = ()
  qkv_bias: bool = False
  expert_bias: bool = False
  mxfp4_experts: bool = False

@dataclass(frozen=True, kw_only=True)
class DeepSeek4Config(TransformerConfig):
  compress_ratios: tuple[int, ...]
  index_n_heads: int
  index_head_dim: int
  index_topk: int
  output_groups: int
  output_lora_rank: int
  hc_mult: int
  hc_sinkhorn_iters: int
  hc_eps: float
  hash_layers: int
  compress_rope_theta: float

class FFNBlock:
  def __init__(self, config:TransformerConfig):
    self.config = config
    self.swiglu_clamp: float|None = None
    self.shared_swiglu_clamp: float|None = None

    # --- RMSNorms --------------------------------------------------------
    self.attn_norm   = nn.RMSNorm(config.dim, config.norm_eps)
    self.ffn_norm    = nn.RMSNorm(config.dim, config.norm_eps)

    # --- feed-forward (MoE or dense) -------------------------------------
    if config.num_experts > 0:
      expert_weights = MXFP4ExpertWeights if config.mxfp4_experts else ExpertWeights
      self.ffn_gate_inp = Linear(config.dim, config.num_experts, bias=False)  # router
      if config.expert_bias: self.exp_probs_b = {"bias": Tensor.zeros(config.num_experts)}
      self.ffn_gate_exps = expert_weights(config.num_experts, config.dim, config.hidden_dim)
      self.ffn_up_exps = expert_weights(config.num_experts, config.dim, config.hidden_dim)
      self.ffn_down_exps = expert_weights(config.num_experts, config.hidden_dim, config.dim)
      if config.shared_expert_dim > 0:
        self.ffn_gate_shexp = Linear(config.dim, config.shared_expert_dim, bias=False)
        self.ffn_up_shexp = Linear(config.dim, config.shared_expert_dim, bias=False)
        self.ffn_down_shexp = Linear(config.shared_expert_dim, config.dim, bias=False)
        if config.shared_expert_gate: self.ffn_gate_inp_shexp = {"weight": Tensor.zeros(config.dim)}
    else:
      self.ffn_gate    = Linear(config.dim, config.hidden_dim, bias=False)
      self.ffn_up      = Linear(config.dim, config.hidden_dim, bias=False)
      self.ffn_down    = Linear(config.hidden_dim, config.dim, bias=False)

  def _feed_forward(self, x:Tensor, tokens:Tensor|None=None) -> Tensor:
    if not hasattr(self, 'ffn_gate_exps'):
      # TODO: remove the need for this contiguous
      return self.ffn_down(self.ffn_gate(x).silu().contiguous() * self.ffn_up(x))

    h, logits = x.unsqueeze(2), self.ffn_gate_inp(x)
    bias = self.exp_probs_b["bias"] if hasattr(self, 'exp_probs_b') else None
    gating, normalize_topk = self.config.expert_gating_func, self.config.norm_topk_prob
    if bias is not None and gating == ExpertGating.SOFTMAX: gating = ExpertGating.SIGMOID
    # fast path: without selection bias, normalized SOFTMAX is equivalent to SOFTMAX_WEIGHT
    if gating == ExpertGating.SOFTMAX and bias is None and normalize_topk:
      gating, normalize_topk = ExpertGating.SOFTMAX_WEIGHT, False
    if   gating == ExpertGating.SOFTMAX_WEIGHT: scores = logits
    elif gating == ExpertGating.SOFTMAX:        scores = logits.softmax(-1)
    elif gating == ExpertGating.SIGMOID:        scores = logits.sigmoid()
    elif gating == ExpertGating.SQRT_SOFTPLUS:  scores = logits.softplus().sqrt()
    if hasattr(self, "ffn_gate_tid2eid"):
      assert tokens is not None
      sel = self.ffn_gate_tid2eid["weight"][tokens]
    else: _, sel = pairwise_topk(scores if bias is None else scores + bias, self.config.num_experts_per_tok)
    probs = scores.gather(-1, sel)
    if gating == ExpertGating.SOFTMAX_WEIGHT: probs = probs.softmax(-1)
    if normalize_topk: probs = probs / probs.sum(axis=-1, keepdim=True)
    probs = probs * self.config.routed_scaling_factor

    x_down = self.ffn_down_exps(sel, swiglu(self.ffn_gate_exps(sel, h), self.ffn_up_exps(sel, h),
                                             getattr(self, "swiglu_clamp", None)).contiguous())
    out = (x_down * probs.unsqueeze(-1)).sum(axis=2)

    if hasattr(self, 'ffn_gate_shexp'):
      shexp = self.ffn_down_shexp(swiglu(self.ffn_gate_shexp(x), self.ffn_up_shexp(x),
                                         getattr(self, "shared_swiglu_clamp", None)).contiguous())
      if hasattr(self, 'ffn_gate_inp_shexp'):
        shexp = shexp * (x * self.ffn_gate_inp_shexp["weight"]).sum(axis=-1, keepdim=True).sigmoid()
      out = out + shexp
    return out

  # given the token-prefix match, return how much cached state this block can still reuse
  def _reusable_prefix_len(self, prefix_len:int, cached_len:int) -> int: return prefix_len
  def _init_state(self, x:Tensor): raise NotImplementedError
  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor: raise NotImplementedError
  def _apply_attention_sink(self, q:Tensor, k:Tensor, v:Tensor, mask:Tensor|None) -> tuple[Tensor, Tensor, Tensor|None]:
    if hasattr(self, 'attn_sinks'):
      B, H, T, _ = q.shape
      k, v = k.cat(k[..., :1, :].const_like(0), dim=-2), v.cat(v[..., :1, :].const_like(0), dim=-2)
      sink_col = self.attn_sinks["weight"].reshape(1, H, 1, 1).expand(B, H, T, 1)
      if mask is None: mask = Tensor.zeros(B, 1, T, k.shape[-2]-1, dtype=q.dtype, buffer=False)
      mask = mask.expand(B, H, T, k.shape[-2]-1).cat(sink_col, dim=-1)
    return k, v, mask
  def _attention_residual(self, x:Tensor, start_pos:int|UOp) -> Tensor: return x + self._attention(self.attn_norm(x), start_pos)
  def _feed_forward_residual(self, x:Tensor, tokens:Tensor|None) -> Tensor: return x + self._feed_forward(self.ffn_norm(x), tokens)

  def __call__(self, x: Tensor, start_pos: int|UOp, tokens: Tensor|None=None):
    self._init_state(x)
    # we pass in the weights implicitly so we unpack the GGUF on the fly
    @function(precompile=True, allow_implicit=True)
    def _run(x:Tensor, start_pos:int|UOp, tokens:Tensor|None):
      return self._feed_forward_residual(self._attention_residual(x, start_pos), tokens).contiguous()
    return _run(x, start_pos, tokens)

class TransformerBlock(FFNBlock):
  def __init__(self, config:TransformerConfig):
    super().__init__(config)
    assert config.v_head_dim == config.head_dim, "TransformerBlock requires v_head_dim == head_dim"

    # --- attention projections (all linear, bias-free) ------------------
    q_proj_out       = config.head_dim * config.n_heads * (2 if config.attn_output_gate else 1)
    kv_proj_out      = config.head_dim * config.n_kv_heads
    self.attn_q      = Linear(config.dim, q_proj_out,  bias=config.qkv_bias)
    self.attn_k      = Linear(config.dim, kv_proj_out, bias=config.qkv_bias)
    self.attn_v      = Linear(config.dim, kv_proj_out, bias=config.qkv_bias)
    self.attn_output = Linear(config.head_dim * config.n_heads, config.dim, bias=False)
    if config.qk_norm: self.attn_q_norm, self.attn_k_norm = nn.RMSNorm(config.qk_norm, config.norm_eps), nn.RMSNorm(config.qk_norm, config.norm_eps)
    if config.attn_sinks: self.attn_sinks = {"weight": Tensor.zeros(config.n_heads)}

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
    mask = Tensor.full((1, 1, T, start_pos+T), float("-inf"), dtype=x.dtype, buffer=False).triu(start_pos+1) \
      if resolve(T != 1) else None
    k, v, mask = self._apply_attention_sink(q, k, v, mask)
    attn = q.scaled_dot_product_attention(k, v, attn_mask=mask, enable_gqa=True)     # (B,H,T,Hd)
    attn = attn.transpose(1, 2).reshape(B, T, -1)                                    # back to (B,T,D)
    return self.attn_output(attn if not self.config.attn_output_gate else (attn * gate.sigmoid()))

  def _init_state(self, x:Tensor):
    if not hasattr(self, "cache_kv"):
      self.cache_kv = Tensor.empty(2, x.shape[0], self.config.n_kv_heads, self.config.max_context, self.config.head_dim,
                                   dtype=dtypes.default_float, device=x.device)
      self.freqs_cis = precompute_freqs_cis(self.config.rope_dim, self.config.max_context, self.config.rope_theta,
                                            device=x.device, yarn=self.config.yarn)

class MLABlock(FFNBlock):
  def __init__(self, config:TransformerConfig):
    super().__init__(config)
    if config.q_lora_rank > 0:
      self.attn_q_a = Linear(config.dim, config.q_lora_rank, bias=False)
      self.attn_q_a_norm = nn.RMSNorm(config.q_lora_rank, config.norm_eps)
      self.attn_q_b = Linear(config.q_lora_rank, config.n_heads * config.head_dim, bias=False)
    else:
      self.attn_q = Linear(config.dim, config.n_heads * config.head_dim, bias=False)
    if config.attn_sinks: self.attn_sinks = {"weight": Tensor.zeros(config.n_heads)}

  def _q_input(self, x:Tensor) -> Tensor: return self.attn_q_a_norm(self.attn_q_a(x)) if self.config.q_lora_rank > 0 else x
  def _q_projection(self, x:Tensor) -> Tensor:
    q_input = self._q_input(x)
    return self.attn_q_b(q_input) if self.config.q_lora_rank > 0 else self.attn_q(q_input)

class MLATransformerBlock(MLABlock):
  def __init__(self, config:TransformerConfig):
    super().__init__(config)
    qk_nope_head_dim = config.head_dim - config.rope_dim
    self.attn_kv_a_mqa = Linear(config.dim, config.kv_lora_rank + config.rope_dim, bias=False)
    self.attn_kv_a_norm = nn.RMSNorm(config.kv_lora_rank, config.norm_eps)
    self.attn_k_b = {"weight": Tensor.zeros(config.n_heads, config.kv_lora_rank, qk_nope_head_dim)}
    self.attn_v_b = {"weight": Tensor.zeros(config.n_heads, config.v_head_dim, config.kv_lora_rank)}
    self.attn_output = Linear(config.n_heads * config.v_head_dim, config.dim, bias=False)

  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    B, T, _ = x.shape
    q = self._q_projection(x)
    q = q.reshape(B, T, self.config.n_heads, self.config.head_dim).transpose(1, 2)
    q_nope, q_rope = q[..., :-self.config.rope_dim], q[..., -self.config.rope_dim:]
    freqs_cis = self.freqs_cis[start_pos:start_pos+T]
    if not self.config.ssm or not self.config.ssm.kda: q_rope = apply_rope(q_rope, freqs_cis)
    q = (q_nope @ self.attn_k_b["weight"].transpose(-1, -2)).cat(q_rope, dim=-1)

    kv_a = self.attn_kv_a_mqa(x)
    c_kv = self.attn_kv_a_norm(kv_a[..., :self.config.kv_lora_rank])
    k_rope = kv_a[..., self.config.kv_lora_rank:].unsqueeze(1)
    if not self.config.ssm or not self.config.ssm.kda: k_rope = apply_rope(k_rope, freqs_cis)
    kv = c_kv.cat(k_rope.squeeze(1), dim=-1).unsqueeze(1)
    k = Tensor(self.cache_k.uop.after(self.cache_k[:, :, start_pos:start_pos+T, :].uop.store(kv.uop)))[:, :, 0:start_pos+T, :]
    v = k[..., :self.config.kv_lora_rank]
    mask = Tensor.full((1, 1, T, start_pos+T), float("-inf"), dtype=x.dtype, buffer=False).triu(start_pos+1) \
      if resolve(T != 1) else None
    k, v, mask = self._apply_attention_sink(q, k, v, mask)
    scores = q @ k.transpose(-1, -2) * (1.0 / self.config.head_dim ** 0.5)
    out = (scores + mask if mask is not None else scores).softmax(-1) @ v
    out = (out @ self.attn_v_b["weight"].transpose(-1, -2)).transpose(1, 2).reshape(B, T, -1)
    return self.attn_output(out)

  def _init_state(self, x:Tensor):
    if not hasattr(self, "cache_k"):
      self.cache_k = Tensor.empty(x.shape[0], 1, self.config.max_context, self.config.kv_lora_rank + self.config.rope_dim, device=x.device)
      self.freqs_cis = precompute_freqs_cis(self.config.rope_dim, self.config.max_context, self.config.rope_theta,
                                            device=x.device, yarn=self.config.yarn)

class GatedDeltaNetBlock(FFNBlock):
  def __init__(self, config:TransformerConfig, ssm:SSMConfig):
    super().__init__(config)
    self.head_k_dim, self.num_k_heads, self.num_v_heads = ssm.state_size, ssm.group_count, ssm.time_step_rank
    assert self.num_v_heads % self.num_k_heads == 0
    self.head_v_dim, self.ssm_conv_kernel = ssm.inner_size // ssm.time_step_rank, ssm.conv_kernel
    self.conv_channels, self.q_dim = ssm.inner_size + 2*ssm.group_count*ssm.state_size, ssm.state_size*ssm.group_count
    self.attn_qkv = Linear(config.dim, self.conv_channels, bias=False)
    if ssm.kda:
      self.ssm_g_a, self.ssm_g_b = Linear(config.dim, self.head_v_dim, bias=False), Linear(self.head_v_dim, ssm.inner_size, bias=False)
      self.ssm_f_a, self.ssm_f_b = Linear(config.dim, self.head_k_dim, bias=False), Linear(self.head_k_dim, ssm.inner_size, bias=False)
    else:
      self.attn_gate = Linear(config.dim, ssm.inner_size, bias=False)
      self.ssm_alpha = Linear(config.dim, self.num_v_heads, bias=False)
    self.ssm_beta = Linear(config.dim, self.num_v_heads, bias=False)
    self.ssm_conv1d = {"weight": Tensor.zeros(self.conv_channels, self.ssm_conv_kernel)}
    self.ssm_dt = {"bias": Tensor.zeros(ssm.inner_size if ssm.kda else self.num_v_heads)}
    self.ssm_a = Tensor.zeros(self.num_v_heads, 1) if ssm.kda else Tensor.zeros(self.num_v_heads)
    self.ssm_norm, self.ssm_out = nn.RMSNorm(self.head_v_dim, config.norm_eps), Linear(ssm.inner_size, config.dim, bias=False)

  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    B, T, _ = x.shape
    # bind ints to a variable so the reset flag stays a runtime value (it toggles when generation restarts at position 0)
    start_pos = start_pos if isinstance(start_pos, UOp) else UOp.variable("start_pos", 0, self.config.max_context-1).bind(start_pos)
    initial = Tensor(start_pos).eq(0)
    is_kda = hasattr(self, "ssm_g_a")
    symbolic = isinstance(T, UOp)
    T_pad = x.max_shape[1]  # symbolic chunks are padded to their max size: one graph serves every size

    # input processing
    x = x.half()
    out_gate = self.ssm_g_b(self.ssm_g_a(x)) if is_kda else self.attn_gate(x)
    out_gate = out_gate.reshape(B, T, self.num_v_heads, self.head_v_dim)
    beta = self.ssm_beta(x).sigmoid().reshape(B, T, self.num_v_heads)
    alpha = self.ssm_f_b(self.ssm_f_a(x)) if is_kda else self.ssm_alpha(x)
    log_alpha = ((alpha.float() + self.ssm_dt["bias"]).softplus().reshape(B, T, self.num_v_heads, -1) *
                 self.ssm_a.reshape(self.num_v_heads, -1))

    # qkv conv, conv_state is reset when starting from position 0
    conv_state = initial.where(0, self.conv_state)
    # assemble the conv window in a static-size buffer: [conv_state | qkv rows | zero-pad].
    # padded steps are exact no-ops: beta=0 (delta rule off), log_alpha=0 (decay 1 after exp)
    win = Tensor.zeros(B, self.ssm_conv_kernel-1 + T_pad, self.conv_channels).uop
    win = win.after(win[:, :self.ssm_conv_kernel-1].store(conv_state.cast(win.dtype).uop))
    win = win.after(win[:, self.ssm_conv_kernel-1:self.ssm_conv_kernel-1+T].store(self.attn_qkv(x).cast(win.dtype).uop))
    conv_window = Tensor(win)
    # the last conv_kernel-1 columns of the window become the next conv state
    conv_state_store = self.conv_state.uop.store(conv_window[:, T:T+self.ssm_conv_kernel-1].cast(self.conv_state.dtype).uop)

    conv_out = functools.reduce(lambda a,b: a+b,
      (conv_window[:, i:i+T_pad] * self.ssm_conv1d["weight"][:, i] for i in range(self.ssm_conv_kernel))).silu()
    if symbolic:
      out_gate = out_gate.pad_to((B, T_pad, self.num_v_heads, self.head_v_dim))
      beta, log_alpha = beta.pad_to((B, T_pad, self.num_v_heads)), log_alpha.pad_to((B, T_pad, *log_alpha.shape[2:]))
    q, k, v = conv_out.split([self.q_dim, self.q_dim, self.conv_channels - 2*self.q_dim], dim=-1)
    qk_eps = 1e-12 if is_kda else 1e-6
    q, k = (z.reshape(B, T_pad, self.num_k_heads, self.head_k_dim).normalize(dim=-1, eps=qk_eps)
            .repeat(1, 1, self.num_v_heads//self.num_k_heads, 1) for z in (q, k))
    v = v.reshape(B, T_pad, self.num_v_heads, self.head_v_dim)
    # layout the per-step operands to broadcast against the (B, H, V, K) state
    q, k, v, beta = (z.transpose(1, 2).float() for z in (q, k, v, beta))
    q, k, v, beta = q.unsqueeze(-2) * self.head_k_dim**-0.5, k.unsqueeze(-2), v.unsqueeze(-1), beta.unsqueeze(-1).unsqueeze(-1)
    alpha = log_alpha.transpose(1, 2).exp().unsqueeze(-1)  # per-channel decay for kda, per-head otherwise (B, H, T, V|1, 1)

    # recurrent: scan over the (padded) tokens, updating the recurrent state. collect the per-step outputs
    state = Tensor(self.recurrent_state.uop.after(conv_state_store)).float()  # carry the conv write into this graph
    state = initial.where(0, state)
    outs = []
    for t in range(T_pad):
      s1 = state * alpha[:, :, t]  # decay the state
      delta = (v[:, :, t] - (s1*k[:, :, t]).sum(-1, keepdim=True)) * beta[:, :, t]  # the delta rule update
      state = s1 + delta * k[:, :, t]
      outs.append((state * q[:, :, t]).sum(-1))

    # store the updated recurrent state in place, then read the stacked outputs after the write
    core = Tensor(outs[0].stack(*outs[1:], dim=1).contiguous().uop.after(self.recurrent_state.uop.store(state.cast(self.recurrent_state.dtype).uop)))

    # output; undo the padding before the output projection
    z = (self.ssm_norm(core) * (out_gate.sigmoid() if is_kda else out_gate.silu())).cast(x.dtype).contiguous()
    if symbolic: z = z[:, :T]
    return self.ssm_out(z.reshape(B, T, -1))

  def _init_state(self, x):
    if not hasattr(self, "conv_state"):
      self.conv_state = Tensor.zeros(x.shape[0], self.ssm_conv_kernel-1, self.conv_channels, device=x.device).clone()
      self.recurrent_state = Tensor.zeros(x.shape[0], self.num_v_heads, self.head_v_dim, self.head_k_dim, device=x.device).clone()

class HyperConnection:
  def __init__(self, config:DeepSeek4Config):
    width = (2+config.hc_mult)*config.hc_mult
    self.fn = {"weight": Tensor.zeros(width, config.hc_mult*config.dim)}
    self.base, self.scale = {"weight": Tensor.zeros(width)}, {"weight": Tensor.zeros(3)}
    self.hc, self.norm_eps = config.hc_mult, config.norm_eps
    self.eps, self.iters = config.hc_eps, config.hc_sinkhorn_iters

  def prepare(self, x:Tensor) -> tuple[Tensor, Tensor, Tensor]:
    flat = x.flatten(2)
    mixes = (flat @ self.fn["weight"].T) * (flat.square().mean(-1, keepdim=True)+self.norm_eps).rsqrt()
    scale, base = self.scale["weight"], self.base["weight"]
    B, T, _ = mixes.shape
    pre = (mixes[..., :self.hc]*scale[0]+base[:self.hc]).sigmoid()+self.eps
    post = (mixes[..., self.hc:2*self.hc]*scale[1]+base[self.hc:2*self.hc]).sigmoid()*2
    comb = (mixes[..., 2*self.hc:]*scale[2]+base[2*self.hc:]).reshape(B, T, self.hc, self.hc).softmax(-1)+self.eps
    comb = comb/(comb.sum(-2, keepdim=True)+self.eps)
    for _ in range(1, self.iters):
      comb = comb/(comb.sum(-1, keepdim=True)+self.eps)
      comb = comb/(comb.sum(-2, keepdim=True)+self.eps)
    return (pre.unsqueeze(-1)*x).sum(2).cast(x.dtype), post, comb

  @staticmethod
  def mix(x:Tensor, residual:Tensor, post:Tensor, comb:Tensor) -> Tensor:
    return (post.unsqueeze(-1)*x.unsqueeze(-2) + comb.transpose(-1, -2) @ residual).cast(x.dtype)

class HyperConnectionOutput:
  def __init__(self, config:DeepSeek4Config):
    self.fn = {"weight": Tensor.zeros(config.hc_mult, config.hc_mult*config.dim)}
    self.base, self.scale = {"weight": Tensor.zeros(config.hc_mult)}, {"weight": Tensor.zeros(1)}
    self.hc, self.norm_eps, self.eps = config.hc_mult, config.norm_eps, config.hc_eps

  def __call__(self, x:Tensor) -> Tensor:
    flat = x.flatten(2)
    mixes = (flat @ self.fn["weight"].T) * (flat.square().mean(-1, keepdim=True)+self.norm_eps).rsqrt()
    pre = (mixes*self.scale["weight"]+self.base["weight"]).sigmoid()+self.eps
    return (pre.unsqueeze(-1)*x).sum(2)

class DeepSeek4Compressor:
  def __init__(self, config:TransformerConfig, ratio:int, head_dim:int, rotate:bool=False):
    self.ratio, self.head_dim, self.rope_dim, self.rotate = ratio, head_dim, config.rope_dim, rotate
    self.overlap = ratio == 4
    channels = head_dim * (1+self.overlap)
    self.kv, self.gate = nn.Linear(config.dim, channels, bias=False), nn.Linear(config.dim, channels, bias=False)
    self.ape, self.norm = {"weight": Tensor.zeros(ratio, channels)}, nn.RMSNorm(head_dim, config.norm_eps)

  def init_state(self, x:Tensor):
    if hasattr(self, "kv_state"): return
    mult = 1+self.overlap
    self.kv_state = Tensor.zeros(x.shape[0], mult*self.ratio, mult*self.head_dim, device=x.device).clone()
    self.score_state = Tensor.full(self.kv_state.shape, float("-inf"), device=x.device).clone()

  def __call__(self, x:Tensor, start_pos:int|UOp, freqs_cis:Tensor, kv_cache:Tensor, cache_offset:int=0) -> Tensor:
    ratio, state, score_state = self.ratio, self.kv_state, self.score_state
    kv, score = self.kv(x), self.gate(x)+self.ape["weight"][start_pos % ratio]
    slot = ratio + start_pos % ratio if self.overlap else start_pos % ratio
    if self.overlap:
      rollover = (Tensor.arange(1).to(x.device) == start_pos % ratio).reshape(1, 1, 1)
      state = rollover.where(state[:, ratio:].cat(state[:, ratio:], dim=1), state)
      score_state = rollover.where(score_state[:, ratio:].cat(score_state[:, ratio:], dim=1), score_state)
    slot_mask = Tensor.arange(state.shape[1]).to(x.device).reshape(1, -1, 1) == slot
    updated, updated_score = slot_mask.where(kv, state), slot_mask.where(score, score_state)
    should_compress = (Tensor.arange(1).to(x.device) == (start_pos+1) % ratio).reshape(1, 1, 1)
    if self.overlap:
      pooled = updated[:, :ratio, :self.head_dim].cat(updated[:, ratio:, self.head_dim:], dim=1)
      pooled_score = updated_score[:, :ratio, :self.head_dim].cat(updated_score[:, ratio:, self.head_dim:], dim=1)
    else: pooled, pooled_score = updated, updated_score
    compressed = self.norm((pooled*pooled_score.softmax(1)).sum(1, keepdim=True).cast(x.dtype))
    compress_pos = smax(start_pos+1-ratio, 0)
    compressed = compressed[..., :-self.rope_dim].cat(
      apply_rope(compressed[..., -self.rope_dim:], freqs_cis[compress_pos:compress_pos+1], interleaved=True), dim=-1)
    if self.rotate: compressed = hadamard_transform(compressed)
    cache_pos = cache_offset + start_pos//ratio
    old = kv_cache[:, cache_pos:cache_pos+1]
    kv_cache = Tensor(kv_cache.uop.after(old.uop.store(should_compress.where(compressed.cast(kv_cache.dtype), old).uop)))
    return Tensor(kv_cache.uop.after(self.kv_state.uop.store(updated.uop), self.score_state.uop.store(updated_score.uop)))

  def reset_ops(self) -> list[Tensor]:
    return [self.kv_state.assign(self.kv_state.const_like(0)), self.score_state.assign(self.score_state.const_like(float("-inf")))]

class DeepSeek4Indexer:
  def __init__(self, config:DeepSeek4Config, ratio:int):
    self.n_heads, self.head_dim, self.topk = config.index_n_heads, config.index_head_dim, config.index_topk
    self.rope_dim, self.ratio, self.max_context = config.rope_dim, ratio, config.max_context
    self.proj = nn.Linear(config.dim, self.n_heads, bias=False)
    self.attn_q_b = nn.Linear(config.q_lora_rank, self.n_heads*self.head_dim, bias=False)
    self.compressor = DeepSeek4Compressor(config, ratio, self.head_dim, rotate=True)

  def init_state(self, x:Tensor):
    if hasattr(self, "kv_cache"): return
    self.kv_cache = Tensor.zeros(x.shape[0], max(1, ceildiv(self.max_context, self.ratio)), self.head_dim, dtype=x.dtype, device=x.device).clone()
    self.compressor.init_state(x)

  def __call__(self, q_input:Tensor, x:Tensor, start_pos:int|UOp, freqs_cis:Tensor) -> Tensor:
    cache = self.compressor(x, start_pos, freqs_cis, self.kv_cache)
    B, T, _ = x.shape
    q = self.attn_q_b(q_input).reshape(B, T, self.n_heads, self.head_dim)
    q = q[..., :-self.rope_dim].cat(apply_rope(q[..., -self.rope_dim:], freqs_cis[start_pos:start_pos+T], interleaved=True), dim=-1)
    q = hadamard_transform(q)
    weights = self.proj(x) * (self.head_dim**-0.5 * self.n_heads**-0.5)
    scores = (q @ cache.transpose(-1, -2).unsqueeze(1)).relu()
    scores = (scores*weights.unsqueeze(-1)).sum(2)
    count = (Tensor(start_pos).to(x.device)+1)//self.ratio
    scores = (Tensor.arange(cache.shape[1]).to(x.device).reshape(1, 1, -1) < count).where(scores, float("-inf"))
    selected = scores.topk(min(self.topk, int(cache.shape[1])))[1]
    return (selected < count).where(selected, -1)

class DeepSeek4Block(MLABlock):
  def __init__(self, config:DeepSeek4Config, layer_id:int):
    super().__init__(config)
    self.compress_ratio = config.compress_ratios[layer_id]
    self.swiglu_clamp, self.shared_swiglu_clamp = config.swiglu_clamp[layer_id], config.shared_swiglu_clamp[layer_id]
    self.attn_kv_a_mqa = nn.Linear(config.dim, config.head_dim, bias=False)
    self.attn_kv_a_norm = nn.RMSNorm(config.head_dim, config.norm_eps)
    self.attn_output_a = nn.Linear(config.n_heads*config.head_dim//config.output_groups,
                                   config.output_groups*config.output_lora_rank, bias=False)
    self.attn_output_b = nn.Linear(config.output_groups*config.output_lora_rank, config.dim, bias=False)

    self.hc_attn, self.hc_ffn = HyperConnection(config), HyperConnection(config)

    if self.compress_ratio:
      self.compressor = DeepSeek4Compressor(config, self.compress_ratio, config.head_dim)
      if self.compress_ratio == 4: self.indexer = DeepSeek4Indexer(config, self.compress_ratio)

    if layer_id < config.hash_layers:
      self.ffn_gate_tid2eid = {"weight": Tensor.zeros(config.vocab_size, config.num_experts_per_tok, dtype="int32")}
    else: self.exp_probs_b = {"bias": Tensor.zeros(config.num_experts)}

  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    B, T, _ = x.shape
    assert T == 1
    q_input = self._q_input(x)
    q = self.attn_q_b(q_input)
    q = q.reshape(B, T, self.config.n_heads, self.config.head_dim).transpose(1, 2)
    q = q * (q.square().mean(-1, keepdim=True)+self.config.norm_eps).rsqrt()
    q_nope, q_rope = q[..., :-self.config.rope_dim], q[..., -self.config.rope_dim:]
    freqs_cis = self.freqs_cis[start_pos:start_pos+T]
    q_rope = apply_rope(q_rope.transpose(1, 2), freqs_cis, interleaved=True).transpose(1, 2)
    q = q_nope.cat(q_rope, dim=-1)

    kv = self.attn_kv_a_norm(self.attn_kv_a_mqa(x))
    c_kv, k_rope = kv[..., :-self.config.rope_dim], kv[..., -self.config.rope_dim:]
    k_rope = apply_rope(k_rope, freqs_cis, interleaved=True)
    k, v, mask = self._cache_attention(q_input, x, start_pos, c_kv.cat(k_rope, dim=-1))
    k, v, mask = self._apply_attention_sink(q, k, v, mask)
    out = q.scaled_dot_product_attention(k, v, attn_mask=mask, enable_gqa=True).transpose(1, 2)
    out = out[..., :-self.config.rope_dim].cat(
      apply_rope(out[..., -self.config.rope_dim:], freqs_cis, interleaved=True, inverse=True), dim=-1).contiguous()
    return self._attention_output(out)

  def _compressed_indices(self, q_input:Tensor, x:Tensor, start_pos:int|UOp, cache:Tensor) -> tuple[Tensor, Tensor]:
    cache = self.compressor(x, start_pos, self.freqs_cis, cache, self.config.sliding_window)
    if hasattr(self, "indexer"): return cache, self.indexer(q_input, x, start_pos, self.freqs_cis)
    count = (Tensor(start_pos).to(x.device)+1)//self.compress_ratio
    idx = Tensor.arange(cache.shape[1]-self.config.sliding_window).to(x.device).reshape(1, 1, -1)
    return cache, (idx < count).where(idx, -1)

  def _cache_attention(self, q_input:Tensor, x:Tensor, start_pos:int|UOp, kv:Tensor) -> tuple[Tensor, Tensor, Tensor|None]:
    B, T, _ = x.shape
    window = self.config.sliding_window
    window_pos = start_pos % window
    cache = Tensor(self.kv_cache.uop.after(self.kv_cache[:, window_pos:window_pos+1].uop.store(kv.uop)))
    window_idx = Tensor(start_pos, device=x.device)-window+1+Tensor.arange(window).to(x.device)
    window_idx = (window_idx >= 0).where(window_idx % window, -1).reshape(1, 1, -1)
    if self.compress_ratio:
      cache, compressed_idx = self._compressed_indices(q_input, x, start_pos, cache)
      compressed_idx = (compressed_idx >= 0).where(compressed_idx+window, -1)
      indices = window_idx.cat(compressed_idx, dim=-1)
    else: indices = window_idx
    indices = indices.expand(B, T, indices.shape[-1])
    valid, safe_indices = indices >= 0, indices.maximum(0)
    selected = cache.unsqueeze(1).expand(B, T, cache.shape[1], cache.shape[2]).gather(
      2, safe_indices.unsqueeze(-1).expand(B, T, indices.shape[-1], cache.shape[-1]))
    selected = valid.unsqueeze(-1).where(selected, 0.0)
    return selected, selected, valid.unsqueeze(1).where(0.0, float("-inf"))

  def _attention_output(self, out:Tensor) -> Tensor:
    B, T = out.shape[:2]
    groups, rank = self.config.output_groups, self.config.output_lora_rank
    out = out.reshape(B, T, groups, 1, -1)
    weight = self.attn_output_a.weight.reshape(groups, rank, -1).transpose(-1, -2)
    # (B, T, groups, 1, group_dim) @ (groups, group_dim, rank) -> (B, T, groups * rank)
    out = (out @ weight).flatten(2)
    return self.attn_output_b(out)

  def _attention_residual(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    h, post, comb = self.hc_attn.prepare(x)
    return self.hc_attn.mix(self._attention(self.attn_norm(h), start_pos), x, post, comb)

  def _feed_forward_residual(self, x:Tensor, tokens:Tensor|None) -> Tensor:
    h, post, comb = self.hc_ffn.prepare(x)
    return self.hc_ffn.mix(self._feed_forward(self.ffn_norm(h), tokens), x, post, comb)

  def _init_state(self, x:Tensor):
    if hasattr(self, "kv_cache"): return
    cache_size = self.config.sliding_window + (max(1, ceildiv(self.config.max_context, self.compress_ratio)) if self.compress_ratio else 0)
    self.kv_cache = Tensor.zeros(x.shape[0], cache_size, self.config.head_dim, dtype=x.dtype, device=x.device).clone()
    rope_theta = self.config.compress_rope_theta if self.compress_ratio else self.config.rope_theta
    yarn = self.config.yarn if self.compress_ratio else None
    self.freqs_cis = precompute_freqs_cis(self.config.rope_dim, self.config.max_context, rope_theta, device=x.device, yarn=yarn)
    if not self.compress_ratio: return
    self.compressor.init_state(x)
    if hasattr(self, "indexer"): self.indexer.init_state(x)

  def _state_reset_ops(self) -> list[Tensor]:
    if not hasattr(self, "kv_cache") or not self.compress_ratio: return []
    return self.compressor.reset_ops() + (self.indexer.compressor.reset_ops() if hasattr(self, "indexer") else [])

  def _reusable_prefix_len(self, prefix_len:int, cached_len:int) -> int: return 0 if prefix_len != cached_len else prefix_len

class Transformer:
  def __init__(self, config:TransformerConfig):
    self.output_hc: HyperConnectionOutput|None = None
    if isinstance(config, DeepSeek4Config):
      self.blk: list[FFNBlock] = [DeepSeek4Block(config, i) for i in range(config.num_blocks)]
      self.output_hc = HyperConnectionOutput(config)
    else:
      dense_config = replace(config, num_experts=0, num_experts_per_tok=0, shared_expert_dim=0,
                             hidden_dim=config.dense_hidden_dim or config.hidden_dim)
      if config.ssm: config = replace(config, qk_norm=config.head_dim)
      block_cls = MLATransformerBlock if config.kv_lora_rank > 0 else TransformerBlock
      self.blk = [GatedDeltaNetBlock(dense_config if i < config.leading_dense_blocks else config, config.ssm)
                  if config.ssm and config.ssm_layers[i] else
                  block_cls(dense_config if i < config.leading_dense_blocks else config) for i in range(config.num_blocks)]
    self.token_embd  = nn.Embedding(config.vocab_size, config.dim)
    self.output_norm = nn.RMSNorm(config.dim, config.norm_eps)
    self.output = Linear(config.dim, config.vocab_size, bias=False)
    self.max_context = config.max_context
    self.has_recurrent_block = any(isinstance(b, (GatedDeltaNetBlock, DeepSeek4Block)) for b in self.blk)
    self.has_hash_router = any(hasattr(block, "ffn_gate_tid2eid") for block in self.blk)
    self._cached_tokens: list[int] = []
    # we specialize the JIT for prefill and rollout
    self.prefill_jit = TinyJit(self.forward)
    self.rollout_jit = TinyJit(self.forward)

  def forward(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor) -> Tensor:
    x = self.token_embd(tokens).float()                   # (B, T, D)
    router_tokens = tokens if self.has_hash_router else None
    if self.output_hc is not None:
      x = x.unsqueeze(2).expand(*x.shape[:2], self.output_hc.hc, x.shape[-1]).contiguous()
    for block in self.blk: x = block(x, start_pos, router_tokens)
    if self.output_hc is not None: x = self.output_hc(x)
    # only run the output projection on the last token
    logits = self.output(self.output_norm(x[:, -1:]))[:, -1, :]
    # Gumbel-max trick: argmax(logits/temp - log(-log(uniform))) is equivalent to sampling from softmax(logits/temp)
    return (logits / temperature.maximum(1e-12) - (Tensor.rand_like(logits).maximum(1e-12).log().neg()).log()).argmax(-1, keepdim=True)

  def __call__(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor) -> Tensor:
    return (self.prefill_jit if resolve(tokens.shape[1] != 1) else self.rollout_jit)(tokens.contiguous(), start_pos, temperature)

  @staticmethod
  def from_gguf(gguf:Tensor|str|pathlib.Path, max_context:int|None=None,
                realize=bool(getenv("REALIZE", 0))) -> tuple[Transformer, dict]:
    # TODO: remove the need for copy to default device
    expert_weights = ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")
    kv, state_dict = gguf_load(gguf.to(None).realize() if isinstance(gguf, Tensor) else gguf,
                               preserve_quantized=lambda name, typ: typ == 39 and name.endswith(expert_weights))

    # all state items should be float16, not float32
    state_dict = {k:v.cast('float16') if getenv("HALF", 1) and v.is_floating_point() else v for k,v in state_dict.items()}
    expert_tensors = [v for k,v in state_dict.items() if k.endswith(expert_weights)]
    mxfp4_experts = any(v.dtype == dtypes.uint8 for v in expert_tensors)
    if mxfp4_experts and not all(v.dtype == dtypes.uint8 for v in expert_tensors):
      raise ValueError("mixed packed MXFP4 and dequantized expert weights are not supported")

    # some models like Llama 3.2 don't have an output.weight, they just tie to the token_embd.weight
    if 'output.weight' not in state_dict: state_dict['output.weight'] = state_dict['token_embd.weight']

    arch = kv['general.architecture']
    max_context = min(max_context, kv[f'{arch}.context_length']) if max_context is not None else kv[f'{arch}.context_length']
    n_heads, n_kv_heads = kv[f'{arch}.attention.head_count'], kv[f'{arch}.attention.head_count_kv']

    ssm = None
    ssm_layers: tuple[bool, ...] = ()
    if arch in ('qwen35', 'qwen35moe'):
      ssm = SSMConfig(**{k: kv[f'{arch}.ssm.{k}'] for k in ('conv_kernel','state_size','group_count','time_step_rank','inner_size')})
      ssm_layers = tuple((i+1) % kv[f'{arch}.full_attention_interval'] != 0 for i in range(kv[f'{arch}.block_count']))
    elif arch == 'kimi-linear':
      ssm_layers = tuple(x == 0 for x in n_kv_heads)
      n_kv_heads = max(n_kv_heads)
      ssm = SSMConfig(kv[f'{arch}.ssm.conv_kernel'], kv[f'{arch}.kda.head_dim'], n_heads, n_heads, n_heads*kv[f'{arch}.kda.head_dim'], kda=True)
      for i, is_ssm in enumerate(ssm_layers):
        if not is_ssm: continue
        state_dict[f"blk.{i}.attn_qkv.weight"] = state_dict.pop(f"blk.{i}.attn_q.weight").cat(
          state_dict.pop(f"blk.{i}.attn_k.weight"), state_dict.pop(f"blk.{i}.attn_v.weight"), dim=0).contiguous()
        state_dict[f"blk.{i}.ssm_conv1d.weight"] = state_dict.pop(f"blk.{i}.ssm_conv1d_q.weight").cat(
          state_dict.pop(f"blk.{i}.ssm_conv1d_k.weight"), state_dict.pop(f"blk.{i}.ssm_conv1d_v.weight"), dim=0).squeeze(1).contiguous()
        state_dict[f"blk.{i}.ssm_out.weight"] = state_dict.pop(f"blk.{i}.attn_output.weight")
    if arch in ('qwen35', 'qwen35moe', 'glm4moe'):
      state_dict = {k.replace('post_attention_norm', 'ffn_norm'):v for k,v in state_dict.items()}

    kv_lora_rank = kv.get(f'{arch}.attention.kv_lora_rank', 0)
    head_dim = kv.get(f'{arch}.attention.key_length_mla', kv.get(f'{arch}.attention.key_length', kv[f'{arch}.embedding_length'] // n_heads))
    rope_dim = kv.get(f'{arch}.rope.dimension_count', head_dim)

    deepseek4_kwargs, yarn = {}, None
    sliding_window = kv.get(f'{arch}.attention.sliding_window', 0)
    swiglu_clamp:tuple[float, ...] = ()
    shared_swiglu_clamp:tuple[float, ...] = ()
    if arch == 'deepseek4':
      num_blocks = kv[f'{arch}.block_count']
      def as_layers(value): return tuple(value) if isinstance(value, list) else (value,)*num_blocks
      swiglu_clamp = as_layers(kv[f'{arch}.swiglu_clamp_exp'])
      shared_swiglu_clamp = as_layers(kv.get(f'{arch}.swiglu_clamp_shexp', kv[f'{arch}.swiglu_clamp_exp']))
      yarn = (kv[f'{arch}.rope.scaling.factor'], kv[f'{arch}.rope.scaling.original_context_length'],
              kv[f'{arch}.rope.scaling.yarn_beta_fast'], kv[f'{arch}.rope.scaling.yarn_beta_slow'])
      deepseek4_kwargs = dict(compress_ratios=tuple(kv[f'{arch}.attention.compress_ratios'][:num_blocks]),
        index_n_heads=kv[f'{arch}.attention.indexer.head_count'], index_head_dim=kv[f'{arch}.attention.indexer.key_length'],
        index_topk=kv[f'{arch}.attention.indexer.top_k'], output_groups=kv[f'{arch}.attention.output_group_count'],
        output_lora_rank=kv[f'{arch}.attention.output_lora_rank'], hc_mult=kv[f'{arch}.hyper_connection.count'],
        hc_sinkhorn_iters=kv[f'{arch}.hyper_connection.sinkhorn_iterations'], hc_eps=kv[f'{arch}.hyper_connection.epsilon'],
        hash_layers=kv[f'{arch}.hash_layer_count'], compress_rope_theta=kv[f'{arch}.attention.compress_rope_freq_base'])
      state_dict = {k.replace('.attn_kv.weight', '.attn_kv_a_mqa.weight').replace('attn_compressor_', 'compressor.')
                     .replace('indexer_compressor_', 'indexer.compressor.').replace('.hc_attn_', '.hc_attn.')
                     .replace('.hc_ffn_', '.hc_ffn.').replace('output_hc_', 'output_hc.'):v for k,v in state_dict.items()}

    # Permute RoPE weights from interleaved to half-split layout.
    for name in state_dict:
      if arch in ('kimi-linear', 'deepseek4'): continue
      if ('attn_q.weight' in name or 'attn_q_b.weight' in name) and (arch == 'llama' or kv_lora_rank):
        w = state_dict[name].reshape(n_heads, state_dict[name].shape[0]//n_heads, -1)
        prefix = head_dim-rope_dim
        state_dict[name] = w[:, :prefix].cat(w[:, prefix:].rearrange("n (h two) d -> n (two h) d", two=2), dim=1).reshape(-1, w.shape[-1])
      elif arch == 'llama' and 'attn_k.weight' in name:
        w = state_dict[name].reshape(n_kv_heads, state_dict[name].shape[0]//n_kv_heads, -1)
        state_dict[name] = w.rearrange("n (h two) d -> n (two h) d", two=2).reshape(-1, w.shape[-1])
      elif kv_lora_rank and 'attn_kv_a_mqa.weight' in name:
        state_dict[name] = state_dict[name][:kv_lora_rank].cat(state_dict[name][kv_lora_rank:].rearrange("(h two) d -> (two h) d", two=2), dim=0)
    config_cls = DeepSeek4Config if arch == 'deepseek4' else TransformerConfig
    config = config_cls(
      num_blocks=kv[f'{arch}.block_count'] - kv.get(f'{arch}.nextn_predict_layers', 0), dim=kv[f'{arch}.embedding_length'],
      hidden_dim=kv.get(f'{arch}.expert_feed_forward_length', kv.get(f'{arch}.feed_forward_length', 0)),
      n_heads=n_heads, n_kv_heads=n_kv_heads, norm_eps=kv[f'{arch}.attention.layer_norm_rms_epsilon'],
      vocab_size=len(kv['tokenizer.ggml.tokens']),
      head_dim=head_dim,
      rope_theta=kv[f'{arch}.rope.freq_base'],
      rope_dim=rope_dim,
      v_head_dim=kv.get(f'{arch}.attention.value_length_mla', kv.get(f'{arch}.attention.value_length', head_dim)),
      max_context=max_context, sliding_window=sliding_window, yarn=yarn,
      qk_norm=int(state_dict['blk.0.attn_q_norm.weight'].shape[0]) if 'blk.0.attn_q_norm.weight' in state_dict else 0,
      num_experts=kv.get(f'{arch}.expert_count', 0), num_experts_per_tok=kv.get(f'{arch}.expert_used_count', 0),
      norm_topk_prob=kv.get(f'{arch}.expert_weights_norm', arch in ('qwen3moe', 'qwen35moe', 'kimi-linear')),
      expert_gating_func=ExpertGating(kv.get(f'{arch}.expert_gating_func', ExpertGating.SOFTMAX)),
      kv_lora_rank=kv_lora_rank, q_lora_rank=kv.get(f'{arch}.attention.q_lora_rank', 0),
      leading_dense_blocks=kv.get(f'{arch}.leading_dense_block_count', 0),
      shared_expert_dim=kv.get(
        f'{arch}.expert_shared_feed_forward_length',
        kv.get(f'{arch}.expert_shared_count', 0) * kv.get(f'{arch}.expert_feed_forward_length', 0)),
      shared_expert_gate=f"blk.{kv.get(f'{arch}.leading_dense_block_count', 0)}.ffn_gate_inp_shexp.weight" in state_dict,
      dense_hidden_dim=kv.get(f'{arch}.feed_forward_length', 0) if kv.get(f'{arch}.leading_dense_block_count', 0) else 0,
      routed_scaling_factor=kv.get(f'{arch}.expert_weights_scale', 1.0), swiglu_clamp=swiglu_clamp,
      shared_swiglu_clamp=shared_swiglu_clamp, attn_output_gate=arch in ('qwen35', 'qwen35moe'), ssm=ssm,
      ssm_layers=ssm_layers,
      qkv_bias='blk.0.attn_q.bias' in state_dict,
      expert_bias=f"blk.{kv.get(f'{arch}.leading_dense_block_count', 0)}.exp_probs_b.bias" in state_dict,
      mxfp4_experts=mxfp4_experts,
      attn_sinks='blk.0.attn_sinks.weight' in state_dict, **deepseek4_kwargs)
    model = Transformer(config)
    nn.state.load_state_dict(model, state_dict, verbose=False, consume=True, realize=False)  # NOTE: rope_freqs.weight (32,) is unused
    # NOTE: without this contiguous, it unpacks the weights from the model every time. we shouldn't need this, but for now it's faster
    if realize:
      for s in (params:=nn.state.get_parameters(model)): s.replace(s.contiguous())
      Tensor.realize(*params)
    return model, kv

  def warmup(self):
    for _ in range(2): list(zip(range(2), self.generate([0])))

  def get_start_pos(self, tokens:list[int]) -> int:
    # recurrent state can't be partially reused after divergence: reuse it only when tokens extend the cached prefix
    if self.has_recurrent_block:
      return len(self._cached_tokens) if self._cached_tokens and len(self._cached_tokens) < len(tokens) \
        and tokens[:len(self._cached_tokens)] == self._cached_tokens else 0
    prefix_len = sum(1 for _ in itertools.takewhile(lambda ab: ab[0] == ab[1], zip(tokens[:-1], self._cached_tokens)))
    return min(block._reusable_prefix_len(prefix_len, len(self._cached_tokens)) for block in self.blk)

  def generate(self, tokens:list[int], chunk_size:int=32, temperature:float=0.0):
    if self.has_recurrent_block: chunk_size = 1
    v_start_pos = UOp.variable("start_pos", 0, self.max_context-1)
    v_toks = UOp.variable("toks", 1, chunk_size)
    # TODO: use UOp.variable for temperature once float variables are supported
    temp = Tensor([temperature])
    # assign all input tokens once, then slice from start_pos for the model call
    t = Tensor(tokens + [0] * (self.max_context - len(tokens)), dtype="int32").reshape(1, self.max_context)
    # recompute start_pos from what's currently valid in the caches
    start_pos = self.get_start_pos(tokens)
    out, prompt_len = None, len(tokens)
    while len(tokens) < self.max_context:
      n_toks = min(chunk_size, len(tokens) - start_pos)
      sp, nt = v_start_pos.bind(start_pos), v_toks.bind(n_toks)
      out = self(t[:, sp:sp+nt] if start_pos < prompt_len or out is None else out, sp, temp).realize()
      start_pos += n_toks
      # chunked prefill: keep processing until all prompt tokens are consumed
      if start_pos < len(tokens): continue
      tokens.append(int(out.item()))
      self._cached_tokens = tokens[:-1]
      yield tokens[-1]
