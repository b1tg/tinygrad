from __future__ import annotations
import enum, functools, itertools, pathlib, math, re
from dataclasses import dataclass, replace
from tinygrad import Tensor, nn, UOp, TinyJit, getenv, function, dtypes
from tinygrad.llm.kernels.amd import Linear, gated_delta_prefill, flash_attention, amd_custom_kernels_supported
from tinygrad.llm.gguf import gguf_load, ggml_data_to_tensor
from tinygrad.helpers import prod
from tinygrad.uop.ops import Ops, resolve

class ExpertGating(enum.IntEnum):
  SOFTMAX = 1
  SIGMOID = 2
  SOFTMAX_WEIGHT = 3  # softmax over the top-k selected logits
  SQRT_SOFTPLUS = 4

@functools.cache
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, device:str|None=None) -> Tensor:
  freqs = 1.0 / (theta ** (Tensor.arange(0, dim, 2)[:(dim // 2)] / dim))
  freqs = Tensor.arange(end).unsqueeze(dim=1) * freqs.unsqueeze(dim=0)
  return freqs.cos().cat(freqs.sin(), dim=-1).clone(device)

class ExpertWeights:
  """Like Linear but with num_experts dimension. Weight shape: (num_experts, out_features, in_features)."""
  def __init__(self, num_experts:int, in_features:int, out_features:int):
    self.weight = Tensor.zeros(num_experts, out_features, in_features)
  def __call__(self, sel:Tensor, x:Tensor) -> Tensor:
    # sel: (B, T, k), x: (B, T, 1, in) or (B, T, k, in) -> output: (B, T, k, out)
    return (x.unsqueeze(-2) @ self.weight[sel].transpose(-1, -2)).contiguous().squeeze(-2)

def apply_rope(x:Tensor, freqs_cis:Tensor) -> Tensor:
  assert x.shape[-1] % 2 == 0
  cos, sin = freqs_cis.reshape(1, 1, x.shape[2], -1).chunk(2, dim=-1)
  x1, x2 = x.chunk(2, dim=-1)
  return (x1 * cos - x2 * sin).cat(x2 * cos + x1 * sin, dim=-1)

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
class Gemma4Config:
  hidden_dims: tuple[int, ...]
  sliding_layers: tuple[bool, ...]
  sliding_window: int
  swa_head_dim: int
  swa_rope_theta: float
  shared_kv_layers: int
  per_layer_embed_dim: int
  final_logit_softcap: float

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
  ssm_output_gate_sigmoid: bool = False
  ssm: SSMConfig|None = None
  shared_expert_gate: bool = True
  leading_dense_blocks: int = 0
  dense_hidden_dim: int = 0
  routed_scaling_factor: float = 1.0
  qkv_bias: bool = False
  expert_bias: bool = False
  hyper_connection_count: int = 0
  hyper_connection_low_rank: int = 0
  indexer_heads: int = 0
  indexer_head_dim: int = 0
  indexer_top_k: int = 0
  indexer_compress_ratio: int = 0
  ple_layers: tuple[int, ...] = ()
  ple_ngram_size: int = 0
  ple_heads_per_ngram: int = 0
  ple_conv_kernel: int = 0
  ple_eos_token_id: int = 0
  ple_row_dim: int = 0
  ple_vocab_size: int = 0
  ple_layer_multipliers: tuple[int, ...] = ()
  ple_head_offsets: tuple[int, ...] = ()
  ple_head_vocab_sizes: tuple[int, ...] = ()
  gemma4: Gemma4Config|None = None

class FFNBlock:
  def __init__(self, config:TransformerConfig):
    self.config = config

    # --- RMSNorms --------------------------------------------------------
    self.attn_norm   = nn.RMSNorm(config.dim, config.norm_eps)
    self.ffn_norm    = nn.RMSNorm(config.dim, config.norm_eps)

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
      x_down = self.ffn_down_exps(sel, (self.ffn_gate_exps(sel, h).silu() * self.ffn_up_exps(sel, h)).contiguous())  # (B, T, k, D)
      out = (x_down * probs.unsqueeze(-1)).sum(axis=2)  # (B, T, D)
      if hasattr(self, 'ffn_gate_shexp'):
        shexp = self.ffn_down_shexp(self.ffn_gate_shexp(x).silu().contiguous() * self.ffn_up_shexp(x))
        if hasattr(self, 'ffn_gate_inp_shexp'): shexp = shexp * (x * self.ffn_gate_inp_shexp["weight"]).sum(axis=-1, keepdim=True).sigmoid()
        out = out + shexp
      return out
    # TODO: remove the need for this contiguous
    return self.ffn_down(self.ffn_gate(x).silu().contiguous() * self.ffn_up(x))

  # given the token-prefix match, return how much cached state this block can still reuse
  def _reusable_prefix_len(self, prefix_len:int, cached_len:int) -> int: return prefix_len
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
    if amd_custom_kernels_supported(x.device) and self.config.ssm is not None and not hasattr(self, 'indexer'):
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
    mask = self.indexer(x, start_pos, self.freqs_cis).unsqueeze(1).where(0, float('-inf')).cast(x.dtype) if hasattr(self, 'indexer') else \
      (Tensor.full((1, 1, T, start_pos+T), float('-inf'), dtype=x.dtype, buffer=False).triu(start_pos+1) if resolve(T != 1) else None)
    attn = q.scaled_dot_product_attention(k, v, attn_mask=mask, enable_gqa=True)     # (B,H,T,Hd)
    attn = attn.transpose(1, 2).reshape(B, T, -1)                                    # back to (B,T,D)
    return self.attn_output(attn if not self.config.attn_output_gate else (attn * gate.sigmoid()))

  def _init_state(self, x:Tensor):
    if not hasattr(self, "cache_kv"):
      # zeroed so the flash kernels can safely read whole tiles past the valid region (masked lanes multiply by 0)
      self.cache_kv = Tensor.zeros(2, x.shape[0], self.config.n_kv_heads, self.config.max_context, self.config.head_dim,
                                   dtype=dtypes.half, device=x.device)
      self.freqs_cis = precompute_freqs_cis(self.config.rope_dim, self.config.max_context, self.config.rope_theta, device=x.device)

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

  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    B, T, _ = x.shape
    q_nope_head_dim = self.config.head_dim - self.config.rope_dim
    q_proj = self.attn_q_b(self.attn_q_a_norm(self.attn_q_a(x))) if self.config.q_lora_rank > 0 else self.attn_q(x)
    q = q_proj.reshape(B, T, self.config.n_heads, self.config.head_dim).transpose(1, 2)
    q_nope, q_rope = q[..., :q_nope_head_dim], q[..., q_nope_head_dim:]
    if not self.config.ssm or not self.config.ssm.kda: q_rope = apply_rope(q_rope, self.freqs_cis[start_pos:start_pos+T])
    q = (q_nope @ self.attn_k_b["weight"].transpose(-1, -2)).cat(q_rope, dim=-1)

    kv_a = self.attn_kv_a_mqa(x)
    c_kv = self.attn_kv_a_norm(kv_a[..., :self.config.kv_lora_rank])
    k_rope = kv_a[..., self.config.kv_lora_rank:].reshape(B, T, 1, self.config.rope_dim).transpose(1, 2)
    if not self.config.ssm or not self.config.ssm.kda: k_rope = apply_rope(k_rope, self.freqs_cis[start_pos:start_pos+T])

    k_store = c_kv.reshape(B, 1, T, self.config.kv_lora_rank).cat(k_rope.reshape(B, 1, T, self.config.rope_dim), dim=-1)
    k = Tensor(self.cache_k.uop.after(self.cache_k[:, :, start_pos:start_pos+T, :].uop.store(k_store.uop)))[:, :, 0:start_pos+T, :]
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
      self.cache_k = Tensor.empty(x.shape[0], 1, self.config.max_context, self.config.kv_lora_rank + self.config.rope_dim, device=x.device)
      self.freqs_cis = precompute_freqs_cis(self.config.rope_dim, self.config.max_context, self.config.rope_theta, device=x.device)

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
    q = q * self.head_k_dim**-0.5
    alpha = log_alpha.transpose(1, 2).exp()  # per-channel decay for kda, per-head otherwise (B, H, T, V|1)

    # recurrent: scan over the (padded) tokens, updating the recurrent state. collect the per-step outputs
    state = Tensor(self.recurrent_state.uop.after(conv_state_store))  # carry the conv write into this graph
    if self.head_k_dim % 32 == 0 and self.head_v_dim % 4 == 0 and amd_custom_kernels_supported(x.device):
      # one fused kernel for the whole scan; it resets and updates the recurrent state in place (RDNA3)
      core = gated_delta_prefill(q, k, v, beta, alpha, state, Tensor(start_pos)).transpose(1, 2)
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
    z = (self.ssm_norm(core) * (out_gate.sigmoid() if is_kda or self.config.ssm_output_gate_sigmoid else out_gate.silu())).cast(x.dtype).contiguous()
    if symbolic: z = z[:, :T]
    return self.ssm_out(z.reshape(B, T, -1))

  def _init_state(self, x):
    if not hasattr(self, "conv_state"):
      self.conv_state = Tensor.zeros(x.shape[0], self.ssm_conv_kernel-1, self.conv_channels, device=x.device).clone()
      self.recurrent_state = Tensor.zeros(x.shape[0], self.num_v_heads, self.head_v_dim, self.head_k_dim, device=x.device).clone()

class GroupRMSNorm(nn.RMSNorm):
  def __init__(self, groups:int, dim:int, eps:float):
    super().__init__(groups*dim, eps)
    self.groups, self.dim = groups, dim
  def __call__(self, x:Tensor) -> Tensor:
    y = x.reshape(*x.shape[:-1], self.groups, self.dim)
    return (self._norm(y.float()).cast(x.dtype) * self.weight.reshape(self.groups, self.dim)).flatten(-2)

class PLEEmbedding:
  # nn.Embedding's non-atomic fallback is a one-hot reduction over the full vocabulary, which is prohibitive for PLE.
  def __init__(self, vocab_size:int, embed_size:int): self.weight, self.vocab_size, self.embed_size = Tensor.empty(vocab_size, embed_size), vocab_size, embed_size
  def __call__(self, idx:Tensor) -> Tensor:
    # Decode only selected quantized rows. Materializing either PLE table would require tens of GiB.
    for typ, block_size, block_nbytes in ((13, 256, 176), (20, 32, 18)):
      packed = self.vocab_size * self.embed_size // block_size * block_nbytes
      if (raw := next((u for u in self.weight.uop.toposort() if u.op is Ops.SHRINK and u.dtype == dtypes.uint8 and prod(u.shape) == packed), None)) is not None:
        blocks = Tensor(raw).reshape(self.vocab_size, self.embed_size//block_size, block_nbytes)[idx].reshape(-1)
        return ggml_data_to_tensor(blocks, prod(idx.shape)*self.embed_size, typ).reshape(*idx.shape, self.embed_size).cast(self.weight.dtype)
    return self.weight[idx]

class GatedResidual:
  def __init__(self, config:TransformerConfig, combine:bool=True):
    hdim = config.hyper_connection_count * config.dim
    self.norm = GroupRMSNorm(config.hyper_connection_count, config.dim, config.norm_eps)
    self.down, self.up = Linear(hdim, config.hyper_connection_low_rank, bias=False), Linear(config.hyper_connection_low_rank, hdim, bias=False)
    if combine: self.inject = Linear(hdim, config.hyper_connection_count, bias=False)
    self.groups, self.dim = config.hyper_connection_count, config.dim
  def __call__(self, x:Tensor):
    xn = self.norm(x)
    mix = self.up((self.down(xn) / self.groups).silu()).sigmoid().reshape(*x.shape[:-1], self.groups, self.dim)
    mixed = (mix * xn.reshape(*x.shape[:-1], self.groups, self.dim)).mean(-2)
    if not hasattr(self, 'inject'): return mixed
    return mixed, x, 2 * (self.inject(xn) / self.groups).sigmoid()

class Qwen4ExpPLE:
  def __init__(self, config:TransformerConfig):
    self.config, self.groups, self.dim = config, config.hyper_connection_count, config.dim
    self.embedding = PLEEmbedding(config.ple_vocab_size, config.ple_row_dim)
    embed_dim, hdim = config.ple_row_dim * len(config.ple_head_offsets), self.groups * self.dim
    self.key, self.value = Linear(embed_dim, hdim, bias=False), Linear(embed_dim, self.dim, bias=False)
    self.norm_key = GroupRMSNorm(self.groups, self.dim, config.norm_eps)
    self.norm_query = GroupRMSNorm(self.groups, self.dim, config.norm_eps)
    self.norm_conv = GroupRMSNorm(self.groups, self.dim, config.norm_eps)
    self.conv1d = {"weight": Tensor.zeros(hdim, config.ple_conv_kernel)}

  def _init_state(self, x:Tensor):
    if not hasattr(self, "token_state"):
      self.token_state = Tensor.full((x.shape[0], self.config.ple_ngram_size-1), self.config.ple_eos_token_id,
                                     dtype=dtypes.int64, device=x.device).clone()
      state_len = (self.config.ple_conv_kernel-1) * self.config.ple_ngram_size
      self.conv_state = Tensor.zeros(x.shape[0], state_len, self.groups*self.dim, device=x.device).clone()

  def __call__(self, x:Tensor, tokens:Tensor, start_pos:int|UOp) -> Tensor:
    B, T = tokens.shape
    T_pad, eos = tokens.max_shape[1], self.config.ple_eos_token_id
    start_pos = start_pos if isinstance(start_pos, UOp) else UOp.variable("start_pos", 0, self.config.max_context-1).bind(start_pos)
    initial = Tensor(start_pos).eq(0)
    state = initial.where(Tensor.full(self.token_state.shape, eos, dtype=dtypes.int64, device=x.device), self.token_state)
    toks, rows = tokens.cast(dtypes.int64).pad_to((B, T_pad)), []
    multipliers = [Tensor(v, dtype=dtypes.int64, device=x.device) for v in self.config.ple_layer_multipliers]
    offsets = Tensor(self.config.ple_head_offsets, dtype=dtypes.int64, device=x.device)
    sizes = Tensor(self.config.ple_head_vocab_sizes, dtype=dtypes.int64, device=x.device)
    for i in range(T_pad):
      tok = toks[:, i]
      shifted = [tok] + [state[:, j] for j in range(self.config.ple_ngram_size-1)]
      ids = []
      for ngram in range(2, self.config.ple_ngram_size+1):
        mixed = shifted[0] * multipliers[0]
        for j in range(1, ngram): mixed = mixed.bitwise_xor(shifted[j] * multipliers[j])
        lo, hi = (ngram-2)*self.config.ple_heads_per_ngram, (ngram-1)*self.config.ple_heads_per_ngram
        ids.append((mixed.unsqueeze(-1) % sizes[lo:hi]) + offsets[lo:hi])
      rows.append(self.embedding(ids[0].cat(*ids[1:], dim=-1)).flatten(-2))
      nxt = tok.unsqueeze(-1).cat(state[:, :-1], dim=-1)
      nxt = (tok == eos).unsqueeze(-1).where(Tensor.full(nxt.shape, eos, dtype=dtypes.int64, device=x.device), nxt)
      if isinstance(T, UOp): nxt = Tensor(i < T).where(nxt, state)
      state = nxt
    embeddings = rows[0].stack(*rows[1:], dim=1)[:, :T]
    token_store = self.token_state.uop.store(state.uop)

    key = self.norm_key(self.key(embeddings)).reshape(B, T, self.groups, self.dim)
    value = self.value(embeddings)
    query = self.norm_query(x).reshape(B, T, self.groups, self.dim)
    gate = (key * query).sum(-1, keepdim=True) / math.sqrt(self.dim)
    gate = gate.abs().maximum(1e-6).sqrt() * gate.sign()
    gated = (gate.sigmoid() * value.unsqueeze(-2)).flatten(-2)
    conv_in = self.norm_conv(gated)

    state_len, dilation = self.conv_state.shape[1], self.config.ple_ngram_size
    win = Tensor.zeros(B, state_len + T_pad, self.groups*self.dim).uop
    old_state = initial.where(0, self.conv_state)
    win = win.after(win[:, :state_len].store(old_state.uop))
    win = win.after(win[:, state_len:state_len+T].store(conv_in.uop))
    conv_window = Tensor(win)
    conv_store = self.conv_state.uop.store(conv_window[:, T:T+state_len].cast(self.conv_state.dtype).uop)
    conv = functools.reduce(lambda a,b: a+b, (conv_window[:, i*dilation:i*dilation+T_pad] * self.conv1d["weight"][:, i]
                                              for i in range(self.config.ple_conv_kernel))).silu()[:, :T]
    return Tensor((gated + conv).contiguous().uop.after(token_store, conv_store))

class QSAIndexer:
  def __init__(self, config:TransformerConfig):
    self.config = config
    self.q_proj = Linear(config.dim, config.indexer_heads*config.indexer_head_dim, bias=False)
    self.k_proj = Linear(config.dim, config.indexer_head_dim, bias=False)
    self.q_norm, self.k_norm = nn.RMSNorm(config.indexer_head_dim, config.norm_eps), nn.RMSNorm(config.indexer_head_dim, config.norm_eps)
    self.cache_len = (config.max_context + config.indexer_compress_ratio-1) // config.indexer_compress_ratio * config.indexer_compress_ratio

  def _init_state(self, x:Tensor):
    if not hasattr(self, "cache_k"):
      self.cache_k = Tensor.zeros(x.shape[0], self.cache_len, self.config.indexer_head_dim, dtype=dtypes.half, device=x.device)

  def __call__(self, x:Tensor, start_pos:int|UOp, freqs_cis:Tensor) -> Tensor:
    B, T, C, ratio = *x.shape[:2], self.config.indexer_head_dim, self.config.indexer_compress_ratio
    q = self.q_norm(self.q_proj(x).reshape(B, T, self.config.indexer_heads, C)).transpose(1, 2)
    q = apply_rope(q[..., :self.config.rope_dim], freqs_cis[start_pos:start_pos+T]).cat(q[..., self.config.rope_dim:], dim=-1)
    raw_k = self.k_proj(x)
    store = self.cache_k[:, start_pos:start_pos+T].uop.store(raw_k.cast(self.cache_k.dtype).uop)
    keys = Tensor(self.cache_k.uop.after(store))
    blocks = self.cache_len // ratio
    pooled = self.k_norm(keys.reshape(B, blocks, ratio, C).float().mean(2).cast(x.dtype)).unsqueeze(1)
    pooled = apply_rope(pooled[..., :self.config.rope_dim], freqs_cis[:self.cache_len:ratio]).cat(pooled[..., self.config.rope_dim:], dim=-1)
    scores = (q.float() @ pooled.transpose(-1, -2).float()).relu().sum(1) / math.sqrt(C)
    positions = Tensor.arange(T).reshape(1, T, 1) + start_pos
    complete = Tensor.arange(blocks).reshape(1, 1, blocks) * ratio + ratio-1 <= positions
    scores = complete.where(scores, float("-inf"))
    selected = scores.topk(min(self.config.indexer_top_k//ratio, blocks), dim=-1)[1]
    selected_blocks = Tensor.zeros(B, T, blocks, dtype=dtypes.int32, device=x.device).scatter(-1, selected, 1)
    key_ids = Tensor.arange(self.cache_len)[:start_pos+T]
    block_ids = (key_ids // ratio).reshape(1, 1, -1).expand(B, T, -1)
    mask = selected_blocks.gather(-1, block_ids).bool()
    tail = block_ids == ((positions+1)//ratio)
    return (mask | tail) & (key_ids.reshape(1, 1, -1) <= positions)

class Qwen4ExpBlockMixin:
  def _init_qwen4exp(self, config:TransformerConfig, layer_idx:int):
    del self.attn_norm, self.ffn_norm
    self.hc_attn, self.hc_ffn = GatedResidual(config), GatedResidual(config)
    if layer_idx in config.ple_layers: self.ple = Qwen4ExpPLE(config)

  def __call__(self, x:Tensor, tokens:Tensor, start_pos:int|UOp):
    self._init_state(x)
    if hasattr(self, 'ple'): self.ple._init_state(x)
    @function(precompile=True, allow_implicit=True)
    def _run(x:Tensor, tokens:Tensor, start_pos:int|UOp):
      if hasattr(self, 'ple'): x = x + self.ple(x, tokens, start_pos)
      h, residual, inject = self.hc_attn(x)
      x = residual + (self._attention(h, start_pos).unsqueeze(-2) * inject.unsqueeze(-1)).flatten(-2)
      h, residual, inject = self.hc_ffn(x)
      return (residual + (self._feed_forward(h).unsqueeze(-2) * inject.unsqueeze(-1)).flatten(-2)).contiguous()
    return _run(x, tokens, start_pos)

class Qwen4ExpLinearBlock(Qwen4ExpBlockMixin, GatedDeltaNetBlock):
  def __init__(self, config:TransformerConfig, ssm:SSMConfig, layer_idx:int):
    GatedDeltaNetBlock.__init__(self, config, ssm)
    self._init_qwen4exp(config, layer_idx)

class Qwen4ExpAttentionBlock(Qwen4ExpBlockMixin, TransformerBlock):
  def __init__(self, config:TransformerConfig, layer_idx:int):
    TransformerBlock.__init__(self, config)
    self.indexer = QSAIndexer(config)
    self._init_qwen4exp(config, layer_idx)

  def _init_state(self, x:Tensor):
    TransformerBlock._init_state(self, x)
    self.indexer._init_state(x)

class Gemma4Block(FFNBlock):
  def __init__(self, config:TransformerConfig, layer_idx:int):
    assert config.gemma4 is not None
    super().__init__(replace(config, hidden_dim=config.gemma4.hidden_dims[layer_idx]))
    self.sliding = config.gemma4.sliding_layers[layer_idx]
    self.head_dim = config.gemma4.swa_head_dim if self.sliding else config.head_dim
    self.rope_theta = config.gemma4.swa_rope_theta if self.sliding else config.rope_theta
    self.attn_q_norm, self.attn_k_norm = nn.RMSNorm(self.head_dim, config.norm_eps), nn.RMSNorm(self.head_dim, config.norm_eps)
    self.post_attention_norm = nn.RMSNorm(config.dim, config.norm_eps)
    self.post_ffw_norm, self.post_norm = nn.RMSNorm(config.dim, config.norm_eps), nn.RMSNorm(config.dim, config.norm_eps)
    self.attn_q = Linear(config.dim, config.n_heads*self.head_dim, bias=False)
    self.attn_k = Linear(config.dim, config.n_kv_heads*self.head_dim, bias=False)
    self.attn_v = Linear(config.dim, config.n_kv_heads*self.head_dim, bias=False)
    self.attn_output = Linear(config.n_heads*self.head_dim, config.dim, bias=False)
    self.inp_gate = Linear(config.dim, config.gemma4.per_layer_embed_dim, bias=False)
    self.proj = Linear(config.gemma4.per_layer_embed_dim, config.dim, bias=False)
    self.layer_output_scale = {"weight": Tensor.ones(1)}
    if not self.sliding: self.rope_freqs = {"weight": Tensor.ones(self.head_dim//2)}
    self.shared_kv, self.store_shared_kv = False, False

  def _feed_forward(self, x:Tensor) -> Tensor: return self.ffn_down(self.ffn_gate(x).gelu() * self.ffn_up(x))

  def _init_state(self, x:Tensor):
    if not hasattr(self, "freqs_cis"):
      inv_freq = 1.0 / (self.rope_theta ** (Tensor.arange(0, self.head_dim, 2) / self.head_dim))
      if not self.sliding: inv_freq = inv_freq / self.rope_freqs["weight"].float()
      angles = Tensor.arange(self.config.max_context).unsqueeze(1) * inv_freq.unsqueeze(0)
      self.freqs_cis = angles.cos().cat(angles.sin(), dim=-1).clone(x.device)
    if not self.shared_kv and not hasattr(self, "cache_kv"):
      self.cache_kv = Tensor.zeros(2, x.shape[0], self.config.n_kv_heads, self.config.max_context, self.head_dim,
                                   dtype=dtypes.half, device=x.device)

  def _attention(self, x:Tensor, start_pos:int|UOp, shared_kv:Tensor|None) -> Tensor:
    B, T, _ = x.shape
    q = self.attn_q_norm(self.attn_q(x).reshape(B, T, self.config.n_heads, self.head_dim)).transpose(1, 2)
    q = apply_rope(q, self.freqs_cis[start_pos:start_pos+T])
    if shared_kv is None:
      k = self.attn_k_norm(self.attn_k(x).reshape(B, T, self.config.n_kv_heads, self.head_dim)).transpose(1, 2)
      v = self.attn_v(x).reshape(B, T, self.config.n_kv_heads, self.head_dim).transpose(1, 2)
      vf = v.float()
      v = (vf * ((vf*vf).mean(-1, keepdim=True) + self.config.norm_eps).rsqrt()).cast(x.dtype)
      k = apply_rope(k, self.freqs_cis[start_pos:start_pos+T])
      store = self.cache_kv[:, :, :, start_pos:start_pos+T, :].uop.store(Tensor.stack(k, v).cast(dtypes.half).uop)
      kv = Tensor(self.cache_kv.uop.after(store))
    else:
      kv = shared_kv
    k, v = kv[0, :, :, :start_pos+T], kv[1, :, :, :start_pos+T]
    positions = (Tensor.arange(T) + start_pos).reshape(T, 1)
    key_ids = Tensor.arange(self.config.max_context)[:start_pos+T].reshape(1, -1)
    valid = key_ids <= positions
    if self.sliding: valid = valid & (key_ids > positions-self.config.gemma4.sliding_window)
    mask = valid.unsqueeze(0).unsqueeze(0).where(0, float("-inf")).cast(x.dtype)
    attn = (q * math.sqrt(self.head_dim)).scaled_dot_product_attention(k, v, attn_mask=mask, enable_gqa=True)
    return self.attn_output(attn.transpose(1, 2).reshape(B, T, -1))

  def __call__(self, x:Tensor, per_layer_input:Tensor, start_pos:int|UOp, shared_kv:Tensor|None=None):
    self._init_state(x)
    @function(precompile=True, allow_implicit=True)
    def _run(x:Tensor, per_layer_input:Tensor, start_pos:int|UOp, shared_kv:Tensor|None):
      h = x + self.post_attention_norm(self._attention(self.attn_norm(x), start_pos, shared_kv))
      h = h + self.post_ffw_norm(self._feed_forward(self.ffn_norm(h)))
      h = h + self.post_norm(self.proj(self.inp_gate(h).gelu() * per_layer_input))
      return (h * self.layer_output_scale["weight"]).contiguous()
    return _run(x, per_layer_input, start_pos, shared_kv)

class Transformer:
  def __init__(self, config:TransformerConfig):
    dense_config = replace(config, num_experts=0, num_experts_per_tok=0, shared_expert_dim=0, hidden_dim=config.dense_hidden_dim or config.hidden_dim)
    if config.ssm: config = replace(config, qk_norm=config.head_dim)
    if config.gemma4:
      self.blk = [Gemma4Block(config, i) for i in range(config.num_blocks)]
      first_shared = config.num_blocks-config.gemma4.shared_kv_layers
      sources:dict[bool, int] = {}
      for i in range(first_shared): sources[self.blk[i].sliding] = i
      if set(sources) != set(config.gemma4.sliding_layers): raise ValueError("gemma4 shared KV has no source for one attention type")
      for i, block in enumerate(self.blk):
        block.shared_kv = i >= first_shared
        block.store_shared_kv = i in sources.values()
      self.per_layer_token_embd = PLEEmbedding(config.vocab_size, config.num_blocks*config.gemma4.per_layer_embed_dim)
      self.per_layer_model_proj = Linear(config.dim, config.num_blocks*config.gemma4.per_layer_embed_dim, bias=False)
      self.per_layer_proj_norm = nn.RMSNorm(config.gemma4.per_layer_embed_dim, config.norm_eps)
    elif config.hyper_connection_count:
      self.blk = [Qwen4ExpLinearBlock(config, config.ssm, i) if config.ssm_layers[i] else Qwen4ExpAttentionBlock(config, i)
                  for i in range(config.num_blocks)]
      self.output_hc = GatedResidual(config, combine=False)
    else:
      block_cls = MLATransformerBlock if config.kv_lora_rank > 0 else TransformerBlock
      self.blk:list[FFNBlock] = [GatedDeltaNetBlock(dense_config if i < config.leading_dense_blocks else config, config.ssm)
                                 if config.ssm and config.ssm_layers[i] else
                                 block_cls(dense_config if i < config.leading_dense_blocks else config) for i in range(config.num_blocks)]
    self.token_embd  = nn.Embedding(config.vocab_size, config.dim)
    if not config.hyper_connection_count: self.output_norm = nn.RMSNorm(config.dim, config.norm_eps)
    self.output = Linear(config.dim, config.vocab_size, bias=False)
    self.max_context = config.max_context
    self.has_recurrent_block = any(isinstance(b, GatedDeltaNetBlock) for b in self.blk)
    self._cached_tokens: list[int] = []
    # we specialize the JIT for prefill and rollout
    self.prefill_jit = TinyJit(self.forward)
    self.rollout_jit = TinyJit(self.forward)

  def forward(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor) -> Tensor:
    x = self.token_embd(tokens).float()                   # (B, T, D)
    if hasattr(self, 'per_layer_token_embd'):
      gemma4 = self.blk[0].config.gemma4
      x = x * math.sqrt(self.token_embd.weight.shape[1])
      token_ple = self.per_layer_token_embd(tokens).reshape(*tokens.shape, len(self.blk), gemma4.per_layer_embed_dim)
      context_ple = (self.per_layer_model_proj(x) / math.sqrt(self.token_embd.weight.shape[1])).reshape(*token_ple.shape)
      per_layer_input = (self.per_layer_proj_norm(context_ple) + token_ple*math.sqrt(gemma4.per_layer_embed_dim)) / math.sqrt(2)
      shared_kv:dict[bool, Tensor] = {}
      for i, block in enumerate(self.blk):
        x = block(x, per_layer_input[:, :, i], start_pos, shared_kv[block.sliding] if block.shared_kv else None)
        if block.store_shared_kv: shared_kv[block.sliding] = block.cache_kv
      x = self.output_norm(x)
    elif hasattr(self, 'output_hc'):
      x = x.repeat(1, 1, self.blk[0].config.hyper_connection_count)
      for block in self.blk: x = block(x, tokens, start_pos)
      x = self.output_hc(x)
    else:
      for block in self.blk: x = block(x, start_pos)
    # only run the output projection on the last token
    logits = self.output(x[:, -1:] if hasattr(self, 'output_hc') or hasattr(self, 'per_layer_token_embd') else self.output_norm(x[:, -1:]))[:, -1, :]
    if hasattr(self, 'per_layer_token_embd') and gemma4.final_logit_softcap:
      logits = (logits / gemma4.final_logit_softcap).tanh() * gemma4.final_logit_softcap
    # Gumbel-max trick: argmax(logits/temp - log(-log(uniform))) is equivalent to sampling from softmax(logits/temp)
    return (logits / temperature.maximum(1e-12) - (Tensor.rand_like(logits).maximum(1e-12).log().neg()).log()).argmax(-1, keepdim=True)

  def __call__(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor) -> Tensor:
    return (self.prefill_jit if resolve(tokens.shape[1] != 1) else self.rollout_jit)(tokens.contiguous(), start_pos, temperature)

  @staticmethod
  def from_gguf(gguf:Tensor|str|pathlib.Path, max_context:int|None=None,
                realize=bool(getenv("REALIZE", 0))) -> tuple[Transformer, dict]:
    # TODO: remove the need for copy to default device
    kv, state_dict = gguf_load(gguf.to(None).realize() if isinstance(gguf, Tensor) else gguf)

    # all state items should be float16, not float32
    state_dict = {k:v.cast('float16') if getenv("HALF", 1) else v for k,v in state_dict.items()}

    # some models like Llama 3.2 don't have an output.weight, they just tie to the token_embd.weight
    if 'output.weight' not in state_dict: state_dict['output.weight'] = state_dict['token_embd.weight']

    arch = kv['general.architecture']
    max_context = min(max_context, kv[f'{arch}.context_length']) if max_context is not None else kv[f'{arch}.context_length']
    n_heads, n_kv_heads = kv[f'{arch}.attention.head_count'], kv[f'{arch}.attention.head_count_kv']

    ssm = None
    ssm_layers: tuple[bool, ...] = ()
    if arch in ('qwen35', 'qwen35moe', 'qwen4exp'):
      ssm = SSMConfig(**{k: kv[f'{arch}.ssm.{k}'] for k in ('conv_kernel','state_size','group_count','time_step_rank','inner_size')})
      ssm_layers = tuple(x == 0 for x in kv[f'{arch}.attention.compress_ratios']) if arch == 'qwen4exp' else \
        tuple((i+1) % kv[f'{arch}.full_attention_interval'] != 0 for i in range(kv[f'{arch}.block_count']))
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
    ple_layers: tuple[int, ...] = ()
    if arch == 'qwen4exp':
      ple_layers = tuple(sorted({int(m.group(1)) for k in state_dict if (m := re.match(r'blk\.(\d+)\.ple_', k))}))
      remap = {'output_hc_norm.':'output_hc.norm.', 'output_hc_down.':'output_hc.down.', 'output_hc_up.':'output_hc.up.'}
      for i in range(kv[f'{arch}.block_count']):
        for p in ('hc_attn', 'hc_ffn'):
          remap |= {f'blk.{i}.{p}_norm.':f'blk.{i}.{p}.norm.', f'blk.{i}.{p}_down.':f'blk.{i}.{p}.down.',
                    f'blk.{i}.{p}_up.':f'blk.{i}.{p}.up.', f'blk.{i}.{p}_inject.':f'blk.{i}.{p}.inject.'}
      for i in ple_layers:
        remap |= {f'blk.{i}.ple_key.':f'blk.{i}.ple.key.', f'blk.{i}.ple_value.':f'blk.{i}.ple.value.',
                  f'blk.{i}.ple_norm_key.':f'blk.{i}.ple.norm_key.', f'blk.{i}.ple_norm_query.':f'blk.{i}.ple.norm_query.',
                  f'blk.{i}.ple_norm_conv.':f'blk.{i}.ple.norm_conv.', f'blk.{i}.ple_conv1d.':f'blk.{i}.ple.conv1d.'}
      state_dict = {next((v+k[len(p):] for p,v in remap.items() if k.startswith(p)), k):w for k,w in state_dict.items()}
      if len(ple_layers) != 1: raise ValueError(f"qwen4exp currently requires exactly one PLE layer, got {ple_layers}")
      state_dict[f'blk.{ple_layers[0]}.ple.embedding.weight'] = state_dict.pop('per_layer_token_embd.weight')

    gemma4 = None
    if arch == 'gemma4':
      sliding_layers = tuple(kv[f'{arch}.attention.sliding_window_pattern'])
      rope_freqs = state_dict.pop('rope_freqs.weight')
      for i, sliding in enumerate(sliding_layers):
        if not sliding: state_dict[f'blk.{i}.rope_freqs.weight'] = rope_freqs
      hidden_dims = kv[f'{arch}.feed_forward_length']
      if not isinstance(hidden_dims, (list, tuple)): hidden_dims = [hidden_dims] * kv[f'{arch}.block_count']
      gemma4 = Gemma4Config(tuple(hidden_dims), sliding_layers, kv[f'{arch}.attention.sliding_window'],
        kv[f'{arch}.attention.key_length_swa'], kv[f'{arch}.rope.freq_base_swa'], kv[f'{arch}.attention.shared_kv_layers'],
        kv[f'{arch}.embedding_length_per_layer_input'], kv[f'{arch}.final_logit_softcapping'])

    kv_lora_rank = kv.get(f'{arch}.attention.kv_lora_rank', 0)
    head_dim = kv.get(f'{arch}.attention.key_length_mla', kv.get(f'{arch}.attention.key_length', kv[f'{arch}.embedding_length'] // n_heads))
    rope_dim = kv.get(f'{arch}.rope.dimension_count', head_dim)

    # Permute RoPE weights from interleaved to half-split layout.
    for name in state_dict:
      if arch == 'kimi-linear': continue
      if ('attn_q.weight' in name or 'attn_q_b.weight' in name) and (arch == 'llama' or kv_lora_rank):
        w = state_dict[name].reshape(n_heads, state_dict[name].shape[0]//n_heads, -1)
        prefix = head_dim-rope_dim
        state_dict[name] = w[:, :prefix].cat(w[:, prefix:].rearrange("n (h two) d -> n (two h) d", two=2), dim=1).reshape(-1, w.shape[-1])
      elif arch == 'llama' and 'attn_k.weight' in name:
        w = state_dict[name].reshape(n_kv_heads, state_dict[name].shape[0]//n_kv_heads, -1)
        state_dict[name] = w.rearrange("n (h two) d -> n (two h) d", two=2).reshape(-1, w.shape[-1])
      elif kv_lora_rank and 'attn_kv_a_mqa.weight' in name:
        state_dict[name] = state_dict[name][:kv_lora_rank].cat(state_dict[name][kv_lora_rank:].rearrange("(h two) d -> (two h) d", two=2), dim=0)
    hidden_dim = kv.get(f'{arch}.expert_feed_forward_length', kv.get(f'{arch}.feed_forward_length', 0))
    if isinstance(hidden_dim, (list, tuple)): hidden_dim = max(hidden_dim)
    config = TransformerConfig(
      num_blocks=kv[f'{arch}.block_count'] - kv.get(f'{arch}.nextn_predict_layers', 0), dim=kv[f'{arch}.embedding_length'],
      hidden_dim=hidden_dim,
      n_heads=n_heads, n_kv_heads=n_kv_heads, norm_eps=kv[f'{arch}.attention.layer_norm_rms_epsilon'],
      vocab_size=len(kv['tokenizer.ggml.tokens']),
      head_dim=head_dim,
      rope_theta=kv[f'{arch}.rope.freq_base'],
      rope_dim=rope_dim,
      v_head_dim=kv.get(f'{arch}.attention.value_length_mla', kv.get(f'{arch}.attention.value_length', head_dim)),
      max_context=max_context,
      qk_norm=int(state_dict['blk.0.attn_q_norm.weight'].shape[0]) if 'blk.0.attn_q_norm.weight' in state_dict else 0,
      num_experts=kv.get(f'{arch}.expert_count', 0), num_experts_per_tok=kv.get(f'{arch}.expert_used_count', 0),
      norm_topk_prob=kv.get(f'{arch}.expert_weights_norm', arch in ('qwen3moe', 'qwen35moe', 'kimi-linear', 'qwen4exp')),
      expert_gating_func=ExpertGating(kv.get(f'{arch}.expert_gating_func', ExpertGating.SOFTMAX)),
      kv_lora_rank=kv_lora_rank, q_lora_rank=kv.get(f'{arch}.attention.q_lora_rank', 0),
      leading_dense_blocks=kv.get(f'{arch}.leading_dense_block_count', 0),
      shared_expert_dim=kv.get(
        f'{arch}.expert_shared_feed_forward_length',
        kv.get(f'{arch}.expert_shared_count', 0) * kv.get(f'{arch}.expert_feed_forward_length', 0)),
      shared_expert_gate=f"blk.{kv.get(f'{arch}.leading_dense_block_count', 0)}.ffn_gate_inp_shexp.weight" in state_dict,
      dense_hidden_dim=kv.get(f'{arch}.feed_forward_length', 0) if kv.get(f'{arch}.leading_dense_block_count', 0) else 0,
      routed_scaling_factor=kv.get(f'{arch}.expert_weights_scale', 1.0), attn_output_gate=arch in ('qwen35', 'qwen35moe', 'qwen4exp'),
      ssm_output_gate_sigmoid=arch == 'qwen4exp', ssm=ssm,
      ssm_layers=ssm_layers,
      qkv_bias='blk.0.attn_q.bias' in state_dict,
      expert_bias=f"blk.{kv.get(f'{arch}.leading_dense_block_count', 0)}.exp_probs_b.bias" in state_dict,
      hyper_connection_count=kv.get(f'{arch}.hyper_connection.count', 0),
      hyper_connection_low_rank=kv.get(f'{arch}.hyper_connection.low_rank', 0),
      indexer_heads=kv.get(f'{arch}.attention.indexer.head_count', 0),
      indexer_head_dim=kv.get(f'{arch}.attention.indexer.key_length', 0),
      indexer_top_k=kv.get(f'{arch}.attention.indexer.top_k', 0),
      indexer_compress_ratio=max(kv.get(f'{arch}.attention.compress_ratios', (0,))),
      ple_layers=ple_layers, ple_ngram_size=kv.get(f'{arch}.ple.ngram_size', 0),
      ple_heads_per_ngram=kv.get(f'{arch}.ple.heads_per_ngram', 0), ple_conv_kernel=kv.get(f'{arch}.ple.conv_kernel', 0),
      ple_eos_token_id=kv.get(f'{arch}.ple.eos_token_id', 0),
      ple_row_dim=kv.get(f'{arch}.embedding_length_per_layer_input', 0),
      ple_vocab_size=state_dict[f'blk.{ple_layers[0]}.ple.embedding.weight'].shape[0] if ple_layers else 0,
      ple_layer_multipliers=tuple(kv.get(f'{arch}.ple.layer_multipliers', ())),
      ple_head_offsets=tuple(kv.get(f'{arch}.ple.head_offsets', ())),
      ple_head_vocab_sizes=tuple(kv.get(f'{arch}.ple.head_vocab_sizes', ())),
      gemma4=gemma4)
    model = Transformer(config)
    nn.state.load_state_dict(model, state_dict, verbose=False, consume=True, realize=False)
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
