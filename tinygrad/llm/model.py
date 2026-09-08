from __future__ import annotations
import enum, functools, itertools, pathlib
from dataclasses import dataclass, replace
from tinygrad import Device, Tensor, nn, UOp, TinyJit, getenv, function, dtypes
from tinygrad.llm.kernels.amd import Linear, expert_quant_linear, gated_delta_prefill, flash_attention, amd_custom_kernels_supported
from tinygrad.llm.gguf import ggml_data_to_tensor, gguf_load
from tinygrad.uop.ops import KernelInfo, Ops, resolve

class ExpertGating(enum.IntEnum):
  SOFTMAX = 1
  SIGMOID = 2
  SOFTMAX_WEIGHT = 3  # softmax over the top-k selected logits
  SQRT_SOFTPLUS = 4

@functools.cache
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, device:str|None=None) -> Tensor:
  freqs = 1.0 / (theta ** (Tensor.arange(0, dim, 2)[:(dim // 2)] / dim))
  freqs = Tensor.arange(end).unsqueeze(dim=1) * freqs.unsqueeze(dim=0)
  return freqs.cos().cat(freqs.sin(), dim=-1).clone(device).realize()

@functools.cache
def _device_arange(end:int, device:str) -> Tensor:
  # arange has no device argument; realize its transfer once so routed blocks never retain a default-device source.
  return Tensor.arange(end).to(device).realize()

class ExpertWeights:
  """Like Linear but with num_experts dimension. Weight shape: (num_experts, out_features, in_features)."""
  _PACKED_BLOCK_BYTES = {17: 74, 18: 98, 23: 136}  # IQ2_XS, IQ3_XS, IQ4_XS
  use_custom_quant = True
  def __init__(self, num_experts:int, in_features:int, out_features:int):
    self.num_experts, self.in_features, self.out_features = num_experts, in_features, out_features
    self.weight = Tensor.zeros(num_experts, out_features, in_features)
    self.ggml_type: int|None = None

  def _set_quantized(self):
    # Keep routed IQ weights packed and gather only selected experts before decoding. Decoding all 288 experts
    # at once needs ~4.5 GiB per projection and cannot fit beside a 20 GiB pipeline stage on a 24 GiB card.
    packed_sizes = {self.weight.numel() // 256 * size:typ for typ,size in self._PACKED_BLOCK_BYTES.items()}
    raw = next((u for u in self.weight.uop.toposort() if u.device == self.weight.device
                and u.op in (Ops.BUFFER, Ops.SHRINK) and u.dtype == dtypes.uint8 and u.max_numel() in packed_sizes), None)
    if raw is None: return
    self.ggml_type = packed_sizes[raw.max_numel()]
    block_bytes = self._PACKED_BLOCK_BYTES[self.ggml_type]
    self.weight = Tensor(raw).reshape(self.num_experts, self.out_features, self.in_features//256, block_bytes)

  def __call__(self, sel:Tensor, x:Tensor) -> Tensor:
    # sel: (B, T, k), x: (B, T, 1, in) or (B, T, k, in) -> output: (B, T, k, out)
    if self.ggml_type is None: self._set_quantized()
    if self.ggml_type is not None and self.use_custom_quant and amd_custom_kernels_supported(self.weight.device):
      return expert_quant_linear(self.weight, self.ggml_type, sel, x, self.out_features, self.in_features)
    if self.ggml_type is not None:
      n = sel.numel() * self.out_features * self.in_features
      weight = ggml_data_to_tensor(self.weight[sel].flatten(), n, self.ggml_type)
      weight = weight.reshape(*sel.shape, self.out_features, self.in_features).cast(x.dtype)
    else:
      weight = self.weight[sel]
    return (x.unsqueeze(-2) @ weight.transpose(-1, -2)).contiguous().squeeze(-2)

def apply_rope(x:Tensor, freqs_cis:Tensor) -> Tensor:
  assert x.shape[-1] % 2 == 0
  cos, sin = freqs_cis.reshape(1, 1, x.shape[2], -1).chunk(2, dim=-1)
  x1, x2 = x.chunk(2, dim=-1)
  return (x1 * cos - x2 * sin).cat(x2 * cos + x1 * sin, dim=-1)

def pairwise_topk(x: Tensor, k: int) -> tuple[Tensor, Tensor]:
  n = x.shape[-1]
  assert isinstance(x.device, str)
  indices = _device_arange(n, x.device)
  vals = indices.reshape(1,1,n).cast(x.dtype).expand(x.shape)
  cmp = (x.unsqueeze(-1) > x.unsqueeze(-2)) | ((x.unsqueeze(-1) == x.unsqueeze(-2)) & \
    (indices.reshape(1,1,n,1) < indices.reshape(1,1,1,n)))
  sel = x.const_like(0).scatter(-1, cmp.sum(axis=-1).cast('int32'), vals)[:,:,n-k:].cast('int32')
  return x.gather(-1, sel), sel

def bitonic_topk(x:Tensor, k:int) -> tuple[Tensor, Tensor]:
  """Top-k with propagated indices, avoiding Tensor.sort's quadratic index-recovery tensor."""
  n = x.shape[-1]
  if not isinstance(n, int): raise ValueError(f"top-k dimension must be static, got {n}")
  if not 0 < k <= n: raise ValueError(f"selected index k={k} is out of range for {n}")
  if not isinstance(x.device, str): raise ValueError("top-k requires a concrete device")
  if n == 1: return x, x.const_like(0, dtypes.int32)
  stages, padded = (n-1).bit_length(), 1 << (n-1).bit_length()
  idx = _device_arange(n, x.device).reshape((1,)*(x.ndim-1)+(n,)).expand(x.shape)
  if padded != n:
    pads = (None,)*(x.ndim-1)+((0, padded-n),)
    x, idx = x.pad(pads, value=x.dtype.min), idx.pad(pads, value=n)
  x, idx = x.unflatten(-1, (2,)*stages), idx.unflatten(-1, (2,)*stages)
  base_dim = x.ndim-stages
  for stage in range(1, stages+1):
    if stage != stages:
      crossover_dim = base_dim + stages-stage-1
      x0, x1 = x.split(1, crossover_dim)
      i0, i1 = idx.split(1, crossover_dim)
      flip_dims = tuple(-i for i in range(1, stage+1))
      x, idx = x0.cat(x1.flip(flip_dims), dim=crossover_dim), i0.cat(i1.flip(flip_dims), dim=crossover_dim)
    for substage in range(stage-1, -1, -1):
      partner_dim = base_dim + stages-substage-1
      xa, xb = x.split(1, partner_dim)
      ia, ib = idx.split(1, partner_dim)
      a_first = (xa > xb) | ((xa == xb) & (ia < ib))
      larger, smaller = a_first.where(xa, xb), a_first.where(xb, xa)
      larger_i, smaller_i = a_first.where(ia, ib), a_first.where(ib, ia)
      x, idx = larger.cat(smaller, dim=partner_dim).contiguous(), larger_i.cat(smaller_i, dim=partner_dim).contiguous()
    if stage != stages:
      x0, x1 = x.split(1, crossover_dim)
      i0, i1 = idx.split(1, crossover_dim)
      x, idx = x0.cat(x1.flip(flip_dims), dim=crossover_dim), i0.cat(i1.flip(flip_dims), dim=crossover_dim)
  return x.flatten(base_dim)[..., :k], idx.flatten(base_dim)[..., :k].cast(dtypes.int32)

@functools.cache
def _gather_rows_kernel(out:UOp, src:UOp, idx:UOp) -> UOp:
  # src: (B,N,D), idx: (B,T,K), out: (B,T,K,D)
  B, T, K, D = out.shape
  r = UOp.range(out.numel(), 0)
  d, z = r%D, r//D
  k, z = z%K, z//K
  t, b = z%T, z//T
  return out.flatten()[r].store(src[b, idx[b, t, k], d]).end(r).sink(arg=KernelInfo(name="gather_rows"))

def gather_rows(src:Tensor, idx:Tensor) -> Tensor:
  """Indirectly gather src[B,N,D] with idx[B,T,K] without a one-hot expansion."""
  if src.ndim != 3 or idx.ndim != 3 or src.shape[0] != idx.shape[0]:
    raise ValueError(f"expected src[B,N,D] and idx[B,T,K], got {src.shape=} {idx.shape=}")
  symbolic, orig_shape = not all(isinstance(s, int) for s in idx.shape), idx.shape
  if symbolic: idx = idx.pad_to(idx.max_shape)
  out = Tensor.empty(src.shape[0], idx.shape[1], idx.shape[2], src.shape[2], dtype=src.dtype, device=src.device)
  out = Tensor.custom_kernel(out, src, idx.cast(dtypes.int32), fxn=_gather_rows_kernel)[0]
  return out if not symbolic else out.shrink(tuple((0, s) for s in (*orig_shape, src.shape[2])))

@dataclass(frozen=True)
class SSMConfig:
  conv_kernel: int
  state_size: int
  group_count: int
  time_step_rank: int
  inner_size: int
  kda: bool = False
  gate_lower_bound: float|None = None
  norm_eps: float = 1e-12
  split_qkv: bool = False

@dataclass(frozen=True)
class IndexerConfig:
  top_k: int
  head_dim: int
  n_heads: int
  kpool: int

def _kda_log_decay(g:Tensor, converted_a:Tensor, lower_bound:float|None) -> Tensor:
  # GGUF stores -exp(A_log). GLM's bounded gate uses +exp(A_log), while the original KDA gate uses the negative value directly.
  return lower_bound * (g * -converted_a).sigmoid() if lower_bound is not None else g.softplus() * converted_a

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
  expert_gating_func: ExpertGating = ExpertGating.SOFTMAX
  q_lora_rank: int = 0
  kv_lora_rank: int = 0
  shared_expert_dim: int = 0
  ssm_layers: tuple[bool, ...] = ()
  attn_output_gate: bool = False
  ssm: SSMConfig|None = None
  shared_expert_gate: bool = True
  leading_dense_blocks: int = 0
  dense_hidden_dim: int = 0
  routed_scaling_factor: float = 1.0
  qkv_bias: bool = False
  expert_bias: bool = False
  swiglu_limit: float = 0.0
  hc_mult: int = 0
  hc_eps: float = 0.0
  hc_sinkhorn_iters: int = 0
  indexer: IndexerConfig|None = None

class HyperConnection:
  def __init__(self, config:TransformerConfig):
    width = (2 + config.hc_mult) * config.hc_mult
    self.fn = {"weight": Tensor.zeros(width, config.hc_mult * config.dim)}
    self.base, self.scale = {"weight": Tensor.zeros(width)}, {"weight": Tensor.zeros(3)}
    self.hc, self.norm_eps = config.hc_mult, config.norm_eps
    self.eps, self.iters = config.hc_eps, config.hc_sinkhorn_iters

  def prepare(self, x:Tensor) -> tuple[Tensor, Tensor, Tensor]:
    flat = x.flatten(2).float()
    flat = flat * (flat.square().mean(-1, keepdim=True) + self.norm_eps).rsqrt()
    mixes = flat @ self.fn["weight"].float().T
    scale, base = self.scale["weight"].float(), self.base["weight"].float()
    B, T, _ = mixes.shape
    pre = (mixes[..., :self.hc] * scale[0] + base[:self.hc]).sigmoid() + self.eps
    post = (mixes[..., self.hc:2*self.hc] * scale[1] + base[self.hc:2*self.hc]).sigmoid() * 2
    comb = (mixes[..., 2*self.hc:] * scale[2] + base[2*self.hc:]).reshape(B, T, self.hc, self.hc).softmax(-1) + self.eps
    comb = comb / (comb.sum(-2, keepdim=True) + self.eps)
    for _ in range(1, self.iters):
      comb = comb / (comb.sum(-1, keepdim=True) + self.eps)
      comb = comb / (comb.sum(-2, keepdim=True) + self.eps)
    return (pre.unsqueeze(-1) * x).sum(2).cast(x.dtype), post, comb

  @staticmethod
  def mix(x:Tensor, residual:Tensor, post:Tensor, comb:Tensor) -> Tensor:
    return (post.unsqueeze(-1) * x.unsqueeze(-2) + comb.transpose(-1, -2).cast(x.dtype) @ residual).cast(x.dtype)

class FFNBlock:
  def __init__(self, config:TransformerConfig):
    self.config = config
    self.device: str|None = None

    # --- RMSNorms --------------------------------------------------------
    self.attn_norm   = nn.RMSNorm(config.dim, config.norm_eps)
    self.ffn_norm    = nn.RMSNorm(config.dim, config.norm_eps)
    if config.hc_mult:
      self.hc_attn, self.hc_ffn = HyperConnection(config), HyperConnection(config)

    # --- feed-forward (MoE or dense) -------------------------------------
    if config.num_experts > 0:
      self.ffn_gate_inp = Linear(config.dim, config.num_experts, bias=False)  # router
      if config.expert_bias: self.exp_probs_b = {"bias": Tensor.zeros(config.num_experts)}
      self.ffn_gate_exps = ExpertWeights(config.num_experts, config.dim, config.hidden_dim)
      self.ffn_up_exps = ExpertWeights(config.num_experts, config.dim, config.hidden_dim)
      self.ffn_down_exps = ExpertWeights(config.num_experts, config.hidden_dim, config.dim)
      if config.shared_expert_dim > 0:
        self.ffn_gate_shexp = Linear(config.dim, config.shared_expert_dim, bias=False)
        self.ffn_up_shexp = Linear(config.dim, config.shared_expert_dim, bias=False)
        self.ffn_down_shexp = Linear(config.shared_expert_dim, config.dim, bias=False)
        if config.shared_expert_gate: self.ffn_gate_inp_shexp = {"weight": Tensor.zeros(config.dim)}
    else:
      self.ffn_gate    = Linear(config.dim, config.hidden_dim, bias=False)
      self.ffn_up      = Linear(config.dim, config.hidden_dim, bias=False)
      self.ffn_down    = Linear(config.hidden_dim, config.dim, bias=False)

  def _swiglu(self, gate:Tensor, up:Tensor) -> Tensor:
    if self.config.swiglu_limit:
      gate = gate.minimum(self.config.swiglu_limit)
      up = up.clip(-self.config.swiglu_limit, self.config.swiglu_limit)
    return gate.silu() * up

  def _feed_forward(self, x:Tensor) -> Tensor:
    if hasattr(self, 'ffn_gate_exps'):
      h = x.unsqueeze(2)  # (B, T, 1, D) - add expert dim for broadcasting
      logits = self.ffn_gate_inp(x)
      bias = self.exp_probs_b["bias"] if hasattr(self, 'exp_probs_b') else None
      gating, normalize_topk = self.config.expert_gating_func, self.config.norm_topk_prob
      # fast path: without selection bias, normalized SOFTMAX is equivalent to SOFTMAX_WEIGHT
      if gating == ExpertGating.SOFTMAX and bias is None and normalize_topk:
        gating, normalize_topk = ExpertGating.SOFTMAX_WEIGHT, False
      if   gating == ExpertGating.SOFTMAX_WEIGHT: scores = logits
      elif gating == ExpertGating.SOFTMAX:        scores = logits.softmax(-1)
      elif gating == ExpertGating.SIGMOID:        scores = logits.sigmoid()
      elif gating == ExpertGating.SQRT_SOFTPLUS:  scores = logits.softplus().sqrt()

      _, sel = pairwise_topk(scores if bias is None else scores + bias, self.config.num_experts_per_tok)
      probs = scores.gather(-1, sel)
      # SOFTMAX_WEIGHT applies softmax after top-k selection
      if gating == ExpertGating.SOFTMAX_WEIGHT: probs = probs.softmax(-1)
      if normalize_topk: probs = probs / probs.sum(axis=-1, keepdim=True)
      probs = probs * self.config.routed_scaling_factor
      x_down = self.ffn_down_exps(sel, self._swiglu(self.ffn_gate_exps(sel, h), self.ffn_up_exps(sel, h)).contiguous())  # (B, T, k, D)
      out = (x_down * probs.unsqueeze(-1)).sum(axis=2)  # (B, T, D)
      if hasattr(self, 'ffn_gate_shexp'):
        shexp = self.ffn_down_shexp(self._swiglu(self.ffn_gate_shexp(x), self.ffn_up_shexp(x)).contiguous())
        if hasattr(self, 'ffn_gate_inp_shexp'): shexp = shexp * (x * self.ffn_gate_inp_shexp["weight"]).sum(axis=-1, keepdim=True).sigmoid()
        out = out + shexp
      return out
    # TODO: remove the need for this contiguous
    return self.ffn_down(self._swiglu(self.ffn_gate(x), self.ffn_up(x)).contiguous())

  # given the token-prefix match, return how much cached state this block can still reuse
  def _reusable_prefix_len(self, prefix_len:int, cached_len:int) -> int: return prefix_len
  def _init_state(self, x:Tensor): raise NotImplementedError
  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor: raise NotImplementedError

  def __call__(self, x: Tensor, start_pos: int|UOp):
    self._init_state(x)
    # we pass in the weights implicitly so we unpack the GGUF on the fly
    @function(precompile=True, allow_implicit=True)
    def _run(x:Tensor, start_pos:int|UOp):
      if self.config.hc_mult:
        residual = x
        h, post, comb = self.hc_attn.prepare(x)
        x = self.hc_attn.mix(self._attention(self.attn_norm(h), start_pos), residual, post, comb)
        residual = x
        h, post, comb = self.hc_ffn.prepare(x)
        return self.hc_ffn.mix(self._feed_forward(self.ffn_norm(h)), residual, post, comb).contiguous()
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
    self.attn_q      = Linear(config.dim, q_proj_out,  bias=config.qkv_bias)
    self.attn_k      = Linear(config.dim, kv_proj_out, bias=config.qkv_bias)
    self.attn_v      = Linear(config.dim, kv_proj_out, bias=config.qkv_bias)
    self.attn_output = Linear(config.head_dim * config.n_heads, config.dim, bias=False)
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
    store = self.cache_kv[:, :, :, start_pos:start_pos+T, :].uop.store(Tensor.stack(k, v).cast(dtypes.half).uop)
    assigned_kv = Tensor(self.cache_kv.uop.after(store))
    # on RDNA3, hybrid models use custom flash attention kernels on the KV cache
    if amd_custom_kernels_supported(x.device) and self.config.ssm is not None:
      attn = flash_attention(q, assigned_kv, start_pos+T)
      attn = attn.transpose(1, 2).reshape(B, T, -1)                                    # back to (B,T,D)
      return self.attn_output(attn if not self.config.attn_output_gate else (attn * gate.sigmoid()))
    k = assigned_kv[0, :, :, 0:start_pos+T, :]
    v = assigned_kv[1, :, :, 0:start_pos+T, :]

    #self.cache_kv[:, :, :, start_pos:start_pos+T, :].assign(Tensor.stack(k, v))
    #k = self.cache_kv[0, :, :, 0:start_pos+T, :]
    #v = self.cache_kv[1, :, :, 0:start_pos+T, :]

    # NOTE: this mask is causal_lower_right, not the causal_upper_left generated by is_casual = True
    # TODO: this if statement should be removed and it shouldn't generate extra kernels
    mask = Tensor.full((1, 1, T, start_pos+T), float("-inf"), dtype=x.dtype, buffer=False).triu(start_pos+1) \
      if resolve(T != 1) else None
    attn = q.scaled_dot_product_attention(k, v, attn_mask=mask, enable_gqa=True)     # (B,H,T,Hd)
    attn = attn.transpose(1, 2).reshape(B, T, -1)                                    # back to (B,T,D)
    return self.attn_output(attn if not self.config.attn_output_gate else (attn * gate.sigmoid()))

  def _init_state(self, x:Tensor):
    if not hasattr(self, "cache_kv"):
      # zeroed so the flash kernels can safely read whole tiles past the valid region (masked lanes multiply by 0)
      self.cache_kv = Tensor.zeros(2, x.shape[0], self.config.n_kv_heads, self.config.max_context, self.config.head_dim,
                                   dtype=dtypes.half, device=x.device)
      if self.config.rope_dim:
        self.freqs_cis = precompute_freqs_cis(self.config.rope_dim, self.config.max_context, self.config.rope_theta, device=x.device)

class AttentionIndexer:
  """GLM-5.3 Flash KPool-DSA indexer."""
  def __init__(self, config:TransformerConfig):
    assert config.indexer is not None and config.q_lora_rank > 0
    self.config, self.index_config = config, config.indexer
    ic = self.index_config
    if ic.top_k % ic.kpool: raise ValueError(f"indexer top_k={ic.top_k} must be divisible by kpool={ic.kpool}")
    self.attn_q_b = Linear(config.q_lora_rank, ic.n_heads * ic.head_dim, bias=False)
    self.attn_k = Linear(config.dim, ic.head_dim, bias=False)
    self.k_norm = nn.LayerNorm(ic.head_dim, eps=1e-6)
    self.proj = Linear(config.dim, ic.n_heads, bias=False)
    self.compressor_ape = Tensor.zeros(ic.kpool, ic.head_dim)
    self.compressor_gate = Linear(config.dim, ic.head_dim, bias=False)

  def _init_state(self, x:Tensor):
    if not hasattr(self, "cache"):
      ic = self.index_config
      self.cache = Tensor.zeros(x.shape[0], self.config.max_context, 2*ic.head_dim+1, dtype=dtypes.half, device=x.device).clone()

  def __call__(self, hidden_states:Tensor, q_resid:Tensor, start_pos:int|UOp) -> Tensor:
    self._init_state(hidden_states)
    B, T, _ = hidden_states.shape
    ic, device = self.index_config, hidden_states.device
    assert isinstance(device, str)

    q = self.attn_q_b(q_resid).reshape(B, T, ic.n_heads, ic.head_dim)
    k = self.k_norm(self.attn_k(hidden_states))
    gate_scores = self.compressor_gate(hidden_states)
    valid = Tensor.ones(B, T, 1, dtype=hidden_states.dtype, device=device)
    packed = k.cat(gate_scores, valid, dim=-1).cast(self.cache.dtype)
    store = self.cache[:, start_pos:start_pos+T].uop.store(packed.uop)
    cached = Tensor(self.cache.uop.after(store))

    pool_count = (self.config.max_context + ic.kpool - 1) // ic.kpool
    padded_len = pool_count * ic.kpool
    if padded_len != self.config.max_context: cached = cached.pad((None, (0, padded_len-self.config.max_context), None))
    grouped = cached.reshape(B, pool_count, ic.kpool, 2*ic.head_dim+1)
    keys, gates, key_valid = grouped[..., :ic.head_dim], grouped[..., ic.head_dim:2*ic.head_dim], grouped[..., -1] != 0
    logits = key_valid.unsqueeze(-1).where(gates.float() + self.compressor_ape.float().reshape(1, 1, ic.kpool, ic.head_dim), -1e30)
    probabilities = logits.softmax(2).cast(keys.dtype)
    pool_keys = (probabilities * keys * key_valid.unsqueeze(-1)).sum(2)
    pool_valid = key_valid.all(2)

    scores = (q.float().unsqueeze(-2) * pool_keys.float().reshape(B, 1, 1, pool_count, ic.head_dim)).sum(-1)
    scores = (scores * (ic.head_dim**-0.5)).relu()
    weights = self.proj(hidden_states).float() * (ic.n_heads**-0.5)
    index_scores = (weights.unsqueeze(-1) * scores).sum(-2)

    q_offsets = _device_arange(hidden_states.max_shape[1], device)[:T]
    q_positions = q_offsets + Tensor(start_pos, device=device)
    pool_ends = _device_arange(pool_count, device) * ic.kpool + (ic.kpool-1)
    candidates = pool_valid.reshape(B, 1, pool_count) & (pool_ends.reshape(1, 1, pool_count) <= q_positions.reshape(1, T, 1))
    index_scores = candidates.where(index_scores, index_scores.const_like(index_scores.dtype.min))

    select_k = min(ic.top_k // ic.kpool, pool_count)
    if select_k == pool_count:
      selected = _device_arange(pool_count, device).reshape(1, 1, pool_count).expand(B, T, pool_count).cast(dtypes.int32)
      selected_scores = index_scores
    else: selected_scores, selected = bitonic_topk(index_scores, select_k)
    selected_valid = selected_scores != selected_scores.const_like(selected_scores.dtype.min)
    offsets = _device_arange(ic.kpool, device).reshape(1, 1, 1, ic.kpool)
    topk_indices = selected.unsqueeze(-1) * ic.kpool + offsets
    topk_indices = selected_valid.unsqueeze(-1).where(topk_indices, -1).flatten(-2)
    if topk_indices.shape[-1] < ic.top_k:
      topk_indices = topk_indices.pad((None, None, (0, ic.top_k-topk_indices.shape[-1])), value=-1)
    elif topk_indices.shape[-1] > ic.top_k: topk_indices = topk_indices[..., :ic.top_k]

    # A complete pool is selected as a unit. The current incomplete pool is always appended token-by-token.
    tail_width = ic.kpool-1
    if tail_width:
      tail_count = (q_positions + 1) % ic.kpool
      tail_start = q_positions + 1 - tail_count
      tail_offsets = _device_arange(tail_width, device).reshape(1, tail_width)
      tail = tail_start.reshape(T, 1) + tail_offsets
      tail = (tail_offsets < tail_count.reshape(T, 1)).where(tail, -1)
      topk_indices = topk_indices.cat(tail.reshape(1, T, tail_width).expand(B, T, tail_width), dim=-1)
    return topk_indices.cast(dtypes.int32)

class MLATransformerBlock(FFNBlock):
  def __init__(self, config:TransformerConfig):
    super().__init__(config)
    qk_nope_head_dim = config.head_dim - config.rope_dim
    if config.q_lora_rank > 0:
      self.attn_q_a = Linear(config.dim, config.q_lora_rank, bias=False)
      self.attn_q_a_norm = nn.RMSNorm(config.q_lora_rank, config.norm_eps)
      self.attn_q_b = Linear(config.q_lora_rank, config.n_heads * config.head_dim, bias=False)
    else:
      self.attn_q = Linear(config.dim, config.n_heads * config.head_dim, bias=False)
    self.attn_kv_a_mqa = Linear(config.dim, config.kv_lora_rank + config.rope_dim, bias=False)
    self.attn_kv_a_norm = nn.RMSNorm(config.kv_lora_rank, config.norm_eps)
    self.attn_k_b = {"weight": Tensor.zeros(config.n_heads, config.kv_lora_rank, qk_nope_head_dim)}
    self.attn_v_b = {"weight": Tensor.zeros(config.n_heads, config.v_head_dim, config.kv_lora_rank)}
    self.attn_output = Linear(config.n_heads * config.v_head_dim, config.dim, bias=False)
    if config.indexer is not None: self.indexer = AttentionIndexer(config)

  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    B, T, _ = x.shape
    q_nope_head_dim = self.config.head_dim - self.config.rope_dim
    q_resid = self.attn_q_a_norm(self.attn_q_a(x)) if self.config.q_lora_rank > 0 else None
    q_proj = self.attn_q_b(q_resid) if q_resid is not None else self.attn_q(x)
    q = q_proj.reshape(B, T, self.config.n_heads, self.config.head_dim).transpose(1, 2)
    q_nope, q_rope = q[..., :q_nope_head_dim], q[..., q_nope_head_dim:]
    if self.config.rope_dim and (not self.config.ssm or not self.config.ssm.kda):
      q_rope = apply_rope(q_rope, self.freqs_cis[start_pos:start_pos+T])
    q = (q_nope @ self.attn_k_b["weight"].transpose(-1, -2)).cat(q_rope, dim=-1)

    kv_a = self.attn_kv_a_mqa(x)
    c_kv = self.attn_kv_a_norm(kv_a[..., :self.config.kv_lora_rank])
    k_rope = kv_a[..., self.config.kv_lora_rank:].reshape(B, T, 1, self.config.rope_dim).transpose(1, 2)
    if self.config.rope_dim and (not self.config.ssm or not self.config.ssm.kda):
      k_rope = apply_rope(k_rope, self.freqs_cis[start_pos:start_pos+T])

    k_store = c_kv.reshape(B, 1, T, self.config.kv_lora_rank).cat(k_rope.reshape(B, 1, T, self.config.rope_dim), dim=-1)
    cached = Tensor(self.cache_k.uop.after(self.cache_k[:, :, start_pos:start_pos+T, :].uop.store(k_store.cast(self.cache_k.dtype).uop)))

    if hasattr(self, "indexer"):
      assert q_resid is not None
      indices = self.indexer(x, q_resid, start_pos)
      valid = indices >= 0
      selected = gather_rows(cached[:, 0], indices.clip(0, self.config.max_context-1))
      q_selected = q.transpose(1, 2)
      attn = (q_selected.unsqueeze(-2) * selected.unsqueeze(2)).sum(-1) * (1.0 / self.config.head_dim ** 0.5)
      attn = valid.unsqueeze(2).where(attn, attn.const_like(float("-inf"))).softmax(-1)
      latent = (attn.unsqueeze(-2) @ selected[..., :self.config.kv_lora_rank].unsqueeze(2)).squeeze(-2)
      value_weight = self.attn_v_b["weight"].transpose(-1, -2).reshape(1, 1, self.config.n_heads,
                                                                 self.config.kv_lora_rank, self.config.v_head_dim)
      out = (latent.unsqueeze(-2) @ value_weight).squeeze(-2).reshape(B, T, -1)
      return self.attn_output(out)

    k = cached[:, :, 0:start_pos+T, :]
    v = k[..., :self.config.kv_lora_rank]

    mask = Tensor.full((1, 1, T, start_pos+T), float("-inf"), dtype=x.dtype, buffer=False).triu(start_pos+1) \
      if resolve(T != 1) else None
    attn = q @ k.transpose(-1, -2) * (1.0 / self.config.head_dim ** 0.5)
    if mask is not None: attn = attn + mask
    attn = attn.softmax(-1)
    attn = ((attn @ v) @ self.attn_v_b["weight"].transpose(-1, -2)).transpose(1, 2).reshape(B, T, -1)
    return self.attn_output(attn)

  def _init_state(self, x:Tensor):
    if not hasattr(self, "cache_k"):
      self.cache_k = Tensor.empty(x.shape[0], 1, self.config.max_context, self.config.kv_lora_rank + self.config.rope_dim,
                                  dtype=dtypes.half, device=x.device)
      if self.config.rope_dim:
        self.freqs_cis = precompute_freqs_cis(self.config.rope_dim, self.config.max_context, self.config.rope_theta, device=x.device)

class GatedDeltaNetBlock(FFNBlock):
  def __init__(self, config:TransformerConfig, ssm:SSMConfig):
    super().__init__(config)
    self.head_k_dim, self.num_k_heads, self.num_v_heads = ssm.state_size, ssm.group_count, ssm.time_step_rank
    assert self.num_v_heads % self.num_k_heads == 0
    self.head_v_dim, self.ssm_conv_kernel = ssm.inner_size // ssm.time_step_rank, ssm.conv_kernel
    self.conv_channels, self.q_dim = ssm.inner_size + 2*ssm.group_count*ssm.state_size, ssm.state_size*ssm.group_count
    if ssm.split_qkv:
      self.attn_q, self.attn_k = Linear(config.dim, self.q_dim, bias=False), Linear(config.dim, self.q_dim, bias=False)
      self.attn_v = Linear(config.dim, self.conv_channels - 2*self.q_dim, bias=False)
    else: self.attn_qkv = Linear(config.dim, self.conv_channels, bias=False)
    if ssm.kda:
      self.ssm_g_a, self.ssm_g_b = Linear(config.dim, self.head_v_dim, bias=False), Linear(self.head_v_dim, ssm.inner_size, bias=False)
      self.ssm_f_a, self.ssm_f_b = Linear(config.dim, self.head_k_dim, bias=False), Linear(self.head_k_dim, ssm.inner_size, bias=False)
    else:
      self.attn_gate = Linear(config.dim, ssm.inner_size, bias=False)
      self.ssm_alpha = Linear(config.dim, self.num_v_heads, bias=False)
    self.ssm_beta = Linear(config.dim, self.num_v_heads, bias=False)
    self.ssm_conv1d = {"weight": Tensor.zeros(self.conv_channels, self.ssm_conv_kernel)}
    self.ssm_dt = {"bias": Tensor.zeros(ssm.inner_size if ssm.kda else self.num_v_heads)}
    self.ssm_a = Tensor.zeros(self.num_v_heads, 1) if ssm.kda and ssm.gate_lower_bound is None else Tensor.zeros(self.num_v_heads)
    self.ssm_norm, self.ssm_out = nn.RMSNorm(self.head_v_dim, config.norm_eps), Linear(ssm.inner_size, config.dim, bias=False)

  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    B, T, _ = x.shape
    # bind ints to a variable so the reset flag stays a runtime value (it toggles when generation restarts at position 0)
    start_pos = start_pos if isinstance(start_pos, UOp) else UOp.variable("start_pos", 0, self.config.max_context-1).bind(start_pos)
    initial = Tensor(start_pos, device=x.device).eq(0)
    is_kda = hasattr(self, "ssm_g_a")
    symbolic = isinstance(T, UOp)
    T_pad = x.max_shape[1]  # symbolic chunks are padded to their max size: one graph serves every size

    # input processing
    x = x.half()
    out_gate = self.ssm_g_b(self.ssm_g_a(x)) if is_kda else self.attn_gate(x)
    out_gate = out_gate.reshape(B, T, self.num_v_heads, self.head_v_dim)
    beta = self.ssm_beta(x).sigmoid().reshape(B, T, self.num_v_heads)
    alpha = (self.ssm_f_b(self.ssm_f_a(x)) if is_kda else self.ssm_alpha(x)).float() + self.ssm_dt["bias"]
    alpha = alpha.reshape(B, T, self.num_v_heads, -1)
    log_alpha = _kda_log_decay(alpha, self.ssm_a.reshape(self.num_v_heads, -1), self.config.ssm.gate_lower_bound if self.config.ssm else None)

    # qkv conv, conv_state is reset when starting from position 0
    conv_state = initial.where(0, self.conv_state)
    qkv = self.attn_q(x).cat(self.attn_k(x), self.attn_v(x), dim=-1) if hasattr(self, "attn_q") else self.attn_qkv(x)
    # assemble the conv window in a static-size buffer: [conv_state | qkv rows | zero-pad].
    # padded steps are exact no-ops: beta=0 (delta rule off), log_alpha=0 (decay 1 after exp)
    win = Tensor.zeros(B, self.ssm_conv_kernel-1 + T_pad, self.conv_channels, device=x.device).uop
    win = win.after(win[:, :self.ssm_conv_kernel-1].store(conv_state.cast(win.dtype).uop))
    win = win.after(win[:, self.ssm_conv_kernel-1:self.ssm_conv_kernel-1+T].store(qkv.cast(win.dtype).uop))
    conv_window = Tensor(win)
    # the last conv_kernel-1 columns of the window become the next conv state
    conv_state_store = self.conv_state.uop.store(conv_window[:, T:T+self.ssm_conv_kernel-1].cast(self.conv_state.dtype).uop)

    conv_out = functools.reduce(lambda a,b: a+b,
      (conv_window[:, i:i+T_pad] * self.ssm_conv1d["weight"][:, i] for i in range(self.ssm_conv_kernel))).silu()
    if symbolic:
      out_gate = out_gate.pad_to((B, T_pad, self.num_v_heads, self.head_v_dim))
      beta, log_alpha = beta.pad_to((B, T_pad, self.num_v_heads)), log_alpha.pad_to((B, T_pad, *log_alpha.shape[2:]))
    q, k, v = conv_out.split([self.q_dim, self.q_dim, self.conv_channels - 2*self.q_dim], dim=-1)
    qk_eps = self.config.ssm.norm_eps if self.config.ssm and is_kda else 1e-6
    q, k = (z.reshape(B, T_pad, self.num_k_heads, self.head_k_dim).normalize(dim=-1, eps=qk_eps)
            .repeat(1, 1, self.num_v_heads//self.num_k_heads, 1) for z in (q, k))
    v = v.reshape(B, T_pad, self.num_v_heads, self.head_v_dim)
    # layout the per-step operands to broadcast against the (B, H, V, K) state
    q, k, v, beta = (z.transpose(1, 2).float() for z in (q, k, v, beta))
    q = q * self.head_k_dim**-0.5
    alpha = log_alpha.transpose(1, 2).exp()  # per-channel decay for kda, per-head otherwise (B, H, T, V|1)

    # recurrent: scan over the (padded) tokens, updating the recurrent state. collect the per-step outputs
    state = Tensor(self.recurrent_state.uop.after(conv_state_store))  # carry the conv write into this graph
    if self.head_k_dim % 32 == 0 and self.head_v_dim % 4 == 0 and amd_custom_kernels_supported(x.device):
      # one fused kernel for the whole scan; it resets and updates the recurrent state in place (RDNA3)
      core = gated_delta_prefill(q, k, v, beta, alpha, state, Tensor(start_pos, device=x.device)).transpose(1, 2)
    else:
      q, k, v, beta = q.unsqueeze(-2), k.unsqueeze(-2), v.unsqueeze(-1), beta.unsqueeze(-1).unsqueeze(-1)
      alpha = alpha.unsqueeze(-1)
      state = initial.where(0, state.float())
      outs = []
      for t in range(T_pad):
        s1 = state * alpha[:, :, t]  # decay the state
        delta = (v[:, :, t] - (s1*k[:, :, t]).sum(-1, keepdim=True)) * beta[:, :, t]  # the delta rule update
        state = s1 + delta * k[:, :, t]
        outs.append((state * q[:, :, t]).sum(-1))

      # store the updated recurrent state in place, then read the stacked outputs after the write
      state_store = self.recurrent_state.uop.store(state.cast(self.recurrent_state.dtype).uop)
      core = Tensor(outs[0].stack(*outs[1:], dim=1).contiguous().uop.after(state_store))

    # output; undo the padding before the output projection
    z = (self.ssm_norm(core) * (out_gate.sigmoid() if is_kda else out_gate.silu())).cast(x.dtype).contiguous()
    if symbolic: z = z[:, :T]
    return self.ssm_out(z.reshape(B, T, -1))

  def _init_state(self, x):
    if not hasattr(self, "conv_state"):
      self.conv_state = Tensor.zeros(x.shape[0], self.ssm_conv_kernel-1, self.conv_channels, device=x.device).clone()
      self.recurrent_state = Tensor.zeros(x.shape[0], self.num_v_heads, self.head_v_dim, self.head_k_dim, device=x.device).clone()

class Transformer:
  def __init__(self, config:TransformerConfig):
    dense_config = replace(config, num_experts=0, num_experts_per_tok=0, shared_expert_dim=0, hidden_dim=config.dense_hidden_dim or config.hidden_dim)
    if config.ssm: config = replace(config, qk_norm=config.head_dim)
    block_cls = MLATransformerBlock if config.kv_lora_rank > 0 else TransformerBlock
    self.blk:list[FFNBlock] = [GatedDeltaNetBlock(dense_config if i < config.leading_dense_blocks else config, config.ssm)
                               if config.ssm and config.ssm_layers[i] else
                               block_cls(dense_config if i < config.leading_dense_blocks else config) for i in range(config.num_blocks)]
    self.token_embd  = nn.Embedding(config.vocab_size, config.dim)
    self.output_norm = nn.RMSNorm(config.dim, config.norm_eps)
    self.output = Linear(config.dim, config.vocab_size, bias=False)
    self.max_context = config.max_context
    self.hc_mult = config.hc_mult
    # Selected-expert decoding keeps GLM's prefill scratch roughly linear in tokens. Eight tokens stays below
    # the headroom of the fullest 24 GiB pipeline stage while avoiding dozens of two-token pipeline passes.
    self.prefill_chunk_size = 8 if config.num_experts and config.hc_mult else 32
    self.devices: tuple[str, ...]|None = None
    self.has_recurrent_block = any(isinstance(b, GatedDeltaNetBlock) for b in self.blk)
    self._cached_tokens: list[int] = []
    # we specialize the JIT for prefill and rollout
    self.prefill_jit = TinyJit(self.forward)
    self.rollout_jit = TinyJit(self.forward)

  def forward(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor) -> Tensor:
    x = self.token_embd(tokens.to(self.token_embd.weight.device)).float()                   # (B, T,D)
    if self.hc_mult: x = x.unsqueeze(2).expand(*x.shape[:2], self.hc_mult, x.shape[-1]).contiguous()
    for block in self.blk:
      # Complete pipeline transfers before building the destination block graph. Combining an SDMA copy and
      # destination kernels in one schedule leaves ordinary kernels with buffers from two AMD devices.
      if (device := getattr(block, "device", None)) is not None and x.device != device: x = x.to(device).contiguous().realize()
      x = block(x, start_pos)
      if self.hc_mult:
        x = x.contiguous().realize()  # materialize a standalone output and bound scratch to one GLM block
        buffer = x.uop.buf_uop.buffer
        assert buffer is not None
        x = Tensor(UOp.from_buffer(buffer))[:x.numel()].reshape(x.shape)  # discard executed dependencies before stage copies
    if self.hc_mult: x = x.mean(2)
    if x.device != self.output.weight.device: x = x.to(self.output.weight.device).contiguous().realize()
    # only run the output projection on the last token
    logits = self.output(self.output_norm(x[:, -1:]))[:, -1, :]
    temperature = temperature.to(logits.device).realize()
    # Gumbel-max trick: argmax(logits/temp - log(-log(uniform))) is equivalent to sampling from softmax(logits/temp)
    return (logits / temperature.maximum(1e-12) - (Tensor.rand_like(logits).maximum(1e-12).log().neg()).log()).argmax(-1, keepdim=True)

  def __call__(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor) -> Tensor:
    if self.hc_mult and self.devices is not None:
      return self.forward(tokens.contiguous(), start_pos, temperature)
    return (self.prefill_jit if resolve(tokens.shape[1] != 1) else self.rollout_jit)(tokens.contiguous(), start_pos, temperature)

  @staticmethod
  def from_gguf(gguf:Tensor|str|pathlib.Path, max_context:int|None=None,
                realize=bool(getenv("REALIZE", 0)), shard:int=1) -> tuple[Transformer, dict]:
    if shard < 1: raise ValueError("shard must be at least 1")
    devices = tuple(Device.canonicalize(f"{Device.DEFAULT}:{i}") for i in range(shard)) if shard > 1 else None
    kv, state_dict = gguf_load(gguf, devices=devices)

    # all state items should be float16, not float32
    state_dict = {k:v.cast('float16') if getenv("HALF", 1) else v for k,v in state_dict.items()}

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
    elif arch in ('kimi-linear', 'glm5next'):
      ssm_layers = tuple(x == 0 for x in n_kv_heads)
      n_kv_heads = max(n_kv_heads)
      ssm = SSMConfig(kv[f'{arch}.ssm.conv_kernel'], kv[f'{arch}.kda.head_dim'], n_heads, n_heads,
                      n_heads*kv[f'{arch}.kda.head_dim'], kda=True,
                      gate_lower_bound=kv.get(f'{arch}.kda.gate_lower_bound'),
                      norm_eps=1e-6 if arch == 'glm5next' else 1e-12, split_qkv=arch == 'glm5next')
      if arch == 'glm5next':
        state_dict = {k.replace('.hc_attn_', '.hc_attn.').replace('.hc_ffn_', '.hc_ffn.')
          .replace('.indexer_compressor_ape.weight', '.indexer.compressor_ape')
          .replace('.indexer_compressor_gate.weight', '.indexer.compressor_gate.weight'):v for k,v in state_dict.items()}
      for i, is_ssm in enumerate(ssm_layers):
        if not is_ssm: continue
        if arch != 'glm5next': state_dict[f"blk.{i}.attn_qkv.weight"] = state_dict.pop(f"blk.{i}.attn_q.weight").cat(
          state_dict.pop(f"blk.{i}.attn_k.weight"), state_dict.pop(f"blk.{i}.attn_v.weight"), dim=0).contiguous()
        state_dict[f"blk.{i}.ssm_conv1d.weight"] = state_dict.pop(f"blk.{i}.ssm_conv1d_q.weight").cat(
          state_dict.pop(f"blk.{i}.ssm_conv1d_k.weight"), state_dict.pop(f"blk.{i}.ssm_conv1d_v.weight"), dim=0).squeeze(1).contiguous()
        state_dict[f"blk.{i}.ssm_out.weight"] = state_dict.pop(f"blk.{i}.attn_output.weight")
    if arch in ('qwen35', 'qwen35moe', 'glm4moe'):
      state_dict = {k.replace('post_attention_norm', 'ffn_norm'):v for k,v in state_dict.items()}

    kv_lora_rank = kv.get(f'{arch}.attention.kv_lora_rank', 0)
    head_dim = kv.get(f'{arch}.attention.key_length_mla', kv.get(f'{arch}.attention.key_length', kv[f'{arch}.embedding_length'] // n_heads))
    rope_dim = kv.get(f'{arch}.rope.dimension_count', head_dim)

    # Permute RoPE weights from interleaved to half-split layout.
    for name in state_dict:
      if arch == 'kimi-linear' or rope_dim == 0: continue
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
      rope_theta=kv.get(f'{arch}.rope.freq_base', 10000.0),
      rope_dim=rope_dim,
      v_head_dim=kv.get(f'{arch}.attention.value_length_mla', kv.get(f'{arch}.attention.value_length', head_dim)),
      max_context=max_context,
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
      routed_scaling_factor=kv.get(f'{arch}.expert_weights_scale', 1.0), attn_output_gate=arch in ('qwen35', 'qwen35moe'), ssm=ssm,
      ssm_layers=ssm_layers,
      qkv_bias='blk.0.attn_q.bias' in state_dict,
      expert_bias=f"blk.{kv.get(f'{arch}.leading_dense_block_count', 0)}.exp_probs_b.bias" in state_dict,
      swiglu_limit=(clamps[0] if isinstance(clamps := kv.get(f'{arch}.swiglu_clamp_exp', 0.0), list) else clamps),
      hc_mult=kv.get(f'{arch}.hyper_connection.count', 0),
      hc_eps=kv.get(f'{arch}.hyper_connection.epsilon', 0.0),
      hc_sinkhorn_iters=kv.get(f'{arch}.hyper_connection.sinkhorn_iterations', 0),
      indexer=IndexerConfig(
        top_k=kv[f'{arch}.attention.indexer.top_k'], head_dim=kv[f'{arch}.attention.indexer.key_length'],
        n_heads=kv[f'{arch}.attention.indexer.head_count'], kpool=kv[f'{arch}.attention.indexer.kpool']) if arch == 'glm5next' else None)
    model = Transformer(config)
    if devices is not None:
      model.devices = devices
      # Move placeholders to the device holding their packed GGUF source before loading. Otherwise load_state_dict
      # would silently copy every decoded layer back to device zero.
      for name, param in nn.state.get_state_dict(model).items():
        if (source := state_dict.get(name)) is not None and param.device != source.device:
          param.replace(Tensor.empty(*param.shape, dtype=param.dtype, device=source.device))
      for i, block in enumerate(model.blk):
        if (source := state_dict.get(f"blk.{i}.attn_norm.weight")) is not None:
          assert isinstance(source.device, str)
          block.device = source.device
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
    if self.has_recurrent_block and not amd_custom_kernels_supported(self.token_embd.weight.device): chunk_size = 1
    chunk_size = min(chunk_size, self.prefill_chunk_size)
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
