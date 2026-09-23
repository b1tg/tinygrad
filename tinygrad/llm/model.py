from __future__ import annotations
import functools, itertools, pathlib, re
from dataclasses import dataclass, replace
from tinygrad import Device, Tensor, nn, UOp, TinyJit, getenv, function
from tinygrad.helpers import ceildiv, prod
from tinygrad.nn import Linear
from tinygrad.llm.gguf import GGUFQuantizedTensor, gguf_load
from tinygrad.uop.ops import resolve
_TP_LAYOUT: dict[str, int|str] = {
  "output.weight":0, "output_norm.weight":"replicate",
  "attn_q.weight":0, "attn_q.bias":0, "attn_k.weight":0, "attn_k.bias":0, "attn_v.weight":0, "attn_v.bias":0,
  "attn_q_b.weight":0, "attn_k_b.weight":0, "attn_v_b.weight":0, "attn_output.weight":1, "attn_gate.weight":0,
  "ssm_conv1d_q.weight":0, "ssm_conv1d_k.weight":0, "ssm_conv1d_v.weight":0,
  "ssm_g.weight":0, "ssm_g_a.weight":"replicate", "ssm_g_b.weight":0,
  "ssm_f_a.weight":"replicate", "ssm_f_b.weight":0, "ssm_beta.weight":0, "ssm_a":0, "ssm_dt.bias":0, "ssm_norm.weight":"replicate",
  "attn_norm.weight":"replicate", "attn_q_norm.weight":"replicate", "attn_k_norm.weight":"replicate",
  "attn_q_a.weight":"replicate", "attn_q_a_norm.weight":"replicate",
  "attn_kv_a_mqa.weight":"replicate", "attn_kv_a_norm.weight":"replicate",
  "ffn_gate.weight":0, "ffn_up.weight":0, "ffn_down.weight":1,
  "ffn_gate_exps.weight":1, "ffn_up_exps.weight":1, "ffn_down_exps.weight":2,
  "ffn_routed_down.weight":"replicate", "ffn_routed_up.weight":"replicate", "ffn_routed_norm.weight":"replicate",
  "ffn_norm.weight":"replicate", "ffn_gate_inp.weight":"replicate", "exp_probs_b.bias":"replicate",
  "ffn_gate_shexp.weight":"replicate", "ffn_up_shexp.weight":"replicate", "ffn_down_shexp.weight":"replicate",
  "ffn_gate_inp_shexp.weight":"replicate",
  "attn_res_score.weight":"replicate", "ffn_res_score.weight":"replicate", "output_res_score.weight":"replicate",
}
_PREDEQUANT_SUFFIXES: dict[str, tuple[str, ...]] = {
  "router":("ffn_gate_inp.weight",),
  "attention":("attn_q.weight", "attn_q_a.weight", "attn_q_b.weight", "attn_k.weight", "attn_k_b.weight", "attn_v.weight",
               "attn_v_b.weight", "attn_kv_a_mqa.weight", "attn_output.weight", "attn_gate.weight"),
  "output":("output.weight",),
  "shared":("ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight", "ffn_gate_inp_shexp.weight"),
  "dense":("ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"),
}
def _predequant_category(name:str) -> str|None:
  key = name.split(".", 2)[-1] if name.startswith("blk.") else name
  return next((category for category,suffixes in _PREDEQUANT_SUFFIXES.items() if key in suffixes), None)
def _tp_policy(name:str) -> int|str|None:
  return _TP_LAYOUT.get(name.split(".", 2)[-1] if name.startswith("blk.") else name)
def _fuse_policy(name:str) -> tuple[str, str]|None:
  if name.startswith("blk.") and name.endswith(".ffn_gate_exps.weight"):
    return name.replace("ffn_gate_exps", "ffn_up_exps"), name.replace("ffn_gate_exps", "ffn_gateup_exps")
  return None
def _raw_policy(name:str, ggml_type:int) -> bool:
  expert = name.startswith("blk.") and name.endswith((".ffn_gate_exps.weight", ".ffn_up_exps.weight", ".ffn_down_exps.weight"))
  q8_gemv = getenv("CUSTOM_Q8_0_GEMV", 0) and name.startswith("blk.") and \
    name.endswith(".attn_q_a.weight") and ggml_type == 8
  return (expert and (ggml_type == 17 or (getenv("CUSTOM_Q4_0_EXPERT", 0) and ggml_type == 2))) or q8_gemv
def _place_model(model, state_dict:dict[str, Tensor], devices:tuple[str, ...]|None):
  model.devices = devices
  if devices is None: return
  for name,target in nn.state.get_state_dict(model).items():
    if (source := state_dict.get(name)) is not None and isinstance(source.device, tuple):
      target.replace(Tensor.empty(*target.shape, dtype=target.dtype, device=source.device[0]).shard(source.device, axis=source.uop.axis))
  if not isinstance(model.output.weight.device, tuple):
    model.output.weight.replace(Tensor.empty(*model.output.weight.shape, dtype=model.output.weight.dtype, device=devices[0]).shard(devices, axis=0))
@functools.cache
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, device:str|None=None) -> Tensor:
  freqs = 1.0 / (theta ** (Tensor.arange(0, dim, 2)[:(dim // 2)] / dim))
  freqs = Tensor.arange(end).unsqueeze(dim=1) * freqs.unsqueeze(dim=0)
  return freqs.cos().cat(freqs.sin(), dim=-1).clone(device)

class ExpertWeights:
  """Like Linear but with num_experts dimension. Weight shape: (num_experts, out_features, in_features)."""
  def __init__(self, num_experts:int, in_features:int, out_features:int):
    self.weight: Tensor|GGUFQuantizedTensor = Tensor.zeros(num_experts, out_features, in_features)
  def __call__(self, sel:Tensor, x:Tensor) -> Tensor:
    # sel: (B, T, k), x: (B, T, 1, in) or (B, T, k, in) -> output: (B, T, k, out)
    if isinstance(self.weight, GGUFQuantizedTensor) and self.weight.ggml_type == 2 and getenv("CUSTOM_Q4_0_EXPERT", 0):
      return self.weight.q4_0_expert_linear(sel, x)
    weight = self.weight.decode(sel).cast(x.dtype) if isinstance(self.weight, GGUFQuantizedTensor) else self.weight[sel]
    return (x.unsqueeze(-2) @ weight.transpose(-1, -2)).contiguous().squeeze(-2)

class GGUFLinear(Linear):
  weight: Tensor|GGUFQuantizedTensor
  def __call__(self, x:Tensor) -> Tensor:
    if isinstance(self.weight, GGUFQuantizedTensor): return self.weight.q8_0_linear(x)
    return super().__call__(x)

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

def gathered_argmax(x:Tensor) -> Tensor:
  assert isinstance(x.device, tuple)
  candidates, offset = [], 0
  for part,device in zip(x.chunk(len(x.device), dim=-1), x.device):
    part = part.to(device)
    local_idx = part.argmax(-1, keepdim=True)
    candidates.append(part.gather(-1, local_idx).cat((local_idx + offset).cast(part.dtype), dim=-1).contiguous().to(x.device[0]))
    offset += int(part.shape[-1])
  candidate = candidates[0].stack(*candidates[1:], dim=-2)
  winner = candidate[..., 0].argmax(-1, keepdim=True)
  return candidate[..., 1].gather(-1, winner).cast('int32')

@dataclass(frozen=True)
class AttnResSpec:
  prev_valid_blocks: int
  block_write_idx: int|None

def apply_attn_res(prefix_sum:Tensor, residuals:Tensor, score_norm:nn.RMSNorm, valid_blocks:int) -> Tensor:
  if valid_blocks == 0: return prefix_sum
  values = residuals[:, :, :valid_blocks].cat(prefix_sum.unsqueeze(2), dim=2).float()
  scores = score_norm(values).sum(axis=-1)
  return (values * scores.softmax(-1).unsqueeze(-1)).sum(axis=2).cast(prefix_sum.dtype)

@dataclass(frozen=True)
class SSMConfig:
  conv_kernel: int
  state_size: int
  group_count: int
  time_step_rank: int
  inner_size: int
  kda: bool = False
  full_rank_gate: bool = False
  gate_lower_bound: float|None = None

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
  mla_nope: bool = False
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
  expert_latent_dim: int = 0
  expert_latent_norm: bool = False
  situ_beta: float = 0.0
  situ_linear_beta: float = 0.0
  attn_res_block_size: int = 0

class FFNBlock:
  def __init__(self, config:TransformerConfig):
    self.config = config

    # --- RMSNorms --------------------------------------------------------
    self.attn_norm   = nn.RMSNorm(config.dim, config.norm_eps)
    self.ffn_norm    = nn.RMSNorm(config.dim, config.norm_eps)

    if config.attn_res_block_size:
      self.attn_res_score, self.ffn_res_score = nn.RMSNorm(config.dim, config.norm_eps), nn.RMSNorm(config.dim, config.norm_eps)

    # --- feed-forward (MoE or dense) -------------------------------------
    if config.num_experts > 0:
      self.ffn_gate_inp = Linear(config.dim, config.num_experts, bias=False)  # router
      if config.expert_bias: self.exp_probs_b = {"bias": Tensor.zeros(config.num_experts)}
      expert_dim = config.expert_latent_dim or config.dim
      if config.expert_latent_dim:
        self.ffn_routed_down, self.ffn_routed_up = Linear(config.dim, expert_dim, bias=False), Linear(expert_dim, config.dim, bias=False)
        if config.expert_latent_norm: self.ffn_routed_norm = nn.RMSNorm(expert_dim, config.norm_eps)
      self.ffn_gate_exps = ExpertWeights(config.num_experts, expert_dim, config.hidden_dim)
      self.ffn_up_exps = ExpertWeights(config.num_experts, expert_dim, config.hidden_dim)
      self.ffn_down_exps = ExpertWeights(config.num_experts, config.hidden_dim, expert_dim)
      if config.shared_expert_dim > 0:
        self.ffn_gate_shexp = Linear(config.dim, config.shared_expert_dim, bias=False)
        self.ffn_up_shexp = Linear(config.dim, config.shared_expert_dim, bias=False)
        self.ffn_down_shexp = Linear(config.shared_expert_dim, config.dim, bias=False)
        if config.shared_expert_gate: self.ffn_gate_inp_shexp = {"weight": Tensor.zeros(config.dim)}
    else:
      self.ffn_gate    = Linear(config.dim, config.hidden_dim, bias=False)
      self.ffn_up      = Linear(config.dim, config.hidden_dim, bias=False)
      self.ffn_down    = Linear(config.hidden_dim, config.dim, bias=False)

  def _activate(self, gate:Tensor, up:Tensor, *, materialize_gate:bool=False) -> Tensor:
    if not self.config.situ_beta:
      gate = gate.silu()
      return (gate.contiguous() if materialize_gate else gate) * up
    dtype, beta = gate.dtype, self.config.situ_beta
    gate = gate.float()
    gate = beta * (gate / beta).tanh() * gate.sigmoid()
    if self.config.situ_linear_beta:
      up = self.config.situ_linear_beta * (up.float() / self.config.situ_linear_beta).tanh()
    return (gate * up).cast(dtype)

  def _feed_forward(self, x:Tensor) -> Tensor:
    if hasattr(self, 'ffn_gate_exps') or hasattr(self, 'ffn_gateup_exps'):
      h = (self.ffn_routed_down(x) if hasattr(self, 'ffn_routed_down') else x).unsqueeze(2)
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
      if hasattr(self, 'ffn_gateup_exps'):
        assert isinstance(self.ffn_gateup_exps.weight.device, tuple)
        ndev = len(self.ffn_gateup_exps.weight.device)
        gate_up = self.ffn_gateup_exps(sel, h)
        gate_up = gate_up.reshape(*gate_up.shape[:-1], ndev, 2, gate_up.shape[-1]//(2*ndev))
        hidden = self._activate(gate_up[..., 0, :], gate_up[..., 1, :])
        hidden = hidden.reshape(*hidden.shape[:3], hidden.shape[3]*hidden.shape[4])
        hidden = hidden.contiguous()
      else:
        hidden = self._activate(self.ffn_gate_exps(sel, h), self.ffn_up_exps(sel, h)).contiguous()
      out = (self.ffn_down_exps(sel, hidden) * probs.unsqueeze(-1)).sum(axis=2)
      if hasattr(self, 'ffn_routed_up'):
        if hasattr(self, 'ffn_routed_norm'): out = self.ffn_routed_norm(out)
        out = self.ffn_routed_up(out)
      if hasattr(self, 'ffn_gate_shexp'):
        shexp = self.ffn_down_shexp(self._activate(self.ffn_gate_shexp(x), self.ffn_up_shexp(x), materialize_gate=True))
        if hasattr(self, 'ffn_gate_inp_shexp'): shexp = shexp * (x * self.ffn_gate_inp_shexp["weight"]).sum(axis=-1, keepdim=True).sigmoid()
        out = out + shexp
      return out
    # TODO: remove the need for this contiguous
    return self.ffn_down(self._activate(self.ffn_gate(x), self.ffn_up(x), materialize_gate=True))

  # given the token-prefix match, return how much cached state this block can still reuse
  def _reusable_prefix_len(self, prefix_len:int, cached_len:int) -> int: return prefix_len
  # return writes that reset this block's state after a cache mismatch
  def _state_reset_ops(self) -> list[Tensor]: return []
  def _init_state(self, x:Tensor): raise NotImplementedError
  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor: raise NotImplementedError

  def _normalized_attention(self, x:Tensor, start_pos:int|UOp, materialize:bool=False) -> Tensor:
    x = self.attn_norm(x)
    return self._attention(x.contiguous() if materialize else x, start_pos)

  def _call_attn_res(self, prefix:Tensor, start_pos:int|UOp, residuals:Tensor, spec:AttnResSpec):
    self._init_state(prefix)
    @function(precompile=True, allow_implicit=True)
    def _run_attn_res(prefix:Tensor, start_pos:int|UOp, residuals:Tensor):
      attention_input = apply_attn_res(prefix, residuals, self.attn_res_score, spec.prev_valid_blocks)
      # Materialize norm outputs: fusing them into the following GEMVs produces slower schedules.
      attention = self._normalized_attention(attention_input, start_pos, materialize=True)
      if spec.block_write_idx is not None:
        write = residuals[:, :, spec.block_write_idx:spec.block_write_idx+1].uop.store(prefix.unsqueeze(2).uop)
        residuals, next_prefix = Tensor(residuals.uop.after(write)), attention
      else:
        next_prefix = prefix + attention

      valid_blocks = spec.prev_valid_blocks + (spec.block_write_idx is not None)
      ffn_input = apply_attn_res(next_prefix, residuals, self.ffn_res_score, valid_blocks)
      output = next_prefix + self._feed_forward(self.ffn_norm(ffn_input).contiguous())
      return output.contiguous(), residuals
    return _run_attn_res(prefix, start_pos, residuals)

  def __call__(self, x: Tensor, start_pos: int|UOp):
    self._init_state(x)
    # we pass in the weights implicitly so we unpack the GGUF on the fly
    @function(precompile=True, allow_implicit=True)
    def _run(x:Tensor, start_pos:int|UOp):
      h =     x + self._normalized_attention(x, start_pos)
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
    attn = q.scaled_dot_product_attention(k, v, attn_mask=mask, enable_gqa=True)     # (B,H,T,Hd)
    attn = attn.transpose(1, 2).reshape(B, T, -1)                                    # back to (B,T,D)
    return self.attn_output(attn if not self.config.attn_output_gate else (attn * gate.sigmoid()))

  def _init_state(self, x:Tensor):
    if not hasattr(self, "cache_kv"):
      shape = (2, x.shape[0], self.config.n_kv_heads, self.config.max_context, self.config.head_dim)
      self.cache_kv = Tensor.zeros(*shape, dtype=x.dtype, device=x.device[0]).contiguous().shard(x.device, axis=2).realize() \
        if isinstance(x.device, tuple) else Tensor.empty(*shape, dtype=x.dtype, device=x.device)
      self.freqs_cis = precompute_freqs_cis(self.config.rope_dim, self.config.max_context, self.config.rope_theta, device=x.device)

class MLATransformerBlock(FFNBlock):
  def __init__(self, config:TransformerConfig):
    super().__init__(config)
    qk_nope_head_dim = config.head_dim - config.rope_dim
    if config.q_lora_rank > 0:
      self.attn_q_a = GGUFLinear(config.dim, config.q_lora_rank, bias=False)
      self.attn_q_a_norm = nn.RMSNorm(config.q_lora_rank, config.norm_eps)
      self.attn_q_b = Linear(config.q_lora_rank, config.n_heads * config.head_dim, bias=False)
    else:
      self.attn_q = Linear(config.dim, config.n_heads * config.head_dim, bias=False)
    self.attn_kv_a_mqa = GGUFLinear(config.dim, config.kv_lora_rank + config.rope_dim, bias=False)
    self.attn_kv_a_norm = nn.RMSNorm(config.kv_lora_rank, config.norm_eps)
    self.attn_k_b = {"weight": Tensor.zeros(config.n_heads, config.kv_lora_rank, qk_nope_head_dim)}
    self.attn_v_b = {"weight": Tensor.zeros(config.n_heads, config.v_head_dim, config.kv_lora_rank)}
    self.attn_output = Linear(config.n_heads * config.v_head_dim, config.dim, bias=False)
    if config.attn_output_gate: self.attn_gate = Linear(config.dim, config.n_heads * config.v_head_dim, bias=False)

  def _normalized_attention(self, x:Tensor, start_pos:int|UOp, materialize:bool=False) -> Tensor:
    q_weight = getattr(self, "attn_q_a", None)
    if q_weight is not None: q_weight = q_weight.weight
    if isinstance(q_weight, GGUFQuantizedTensor) and prod(x.uop.max_shape[:-1]) == 1 and not hasattr(self, "attn_gate"):
      if materialize: x = x.contiguous()
      xf = x.float()
      denominator = (xf.square().mean(-1, keepdim=True) + self.attn_norm.eps).sqrt()
      q_a = q_weight.q8_0_rmsnorm_linear(x, self.attn_norm.weight, denominator)
      return self._attention(x, start_pos, q_a)
    return super()._normalized_attention(x, start_pos, materialize)

  def _attention(self, x:Tensor, start_pos:int|UOp, q_a:Tensor|None=None) -> Tensor:
    B, T, _ = x.shape
    q_nope_head_dim = self.config.head_dim - self.config.rope_dim
    q_proj = self.attn_q_b(self.attn_q_a_norm(self.attn_q_a(x) if q_a is None else q_a)) if self.config.q_lora_rank > 0 else self.attn_q(x)
    q = q_proj.reshape(B, T, self.config.n_heads, self.config.head_dim).transpose(1, 2)
    q_nope, q_rope = q[..., :q_nope_head_dim], q[..., q_nope_head_dim:]
    if not self.config.mla_nope: q_rope = apply_rope(q_rope, self.freqs_cis[start_pos:start_pos+T])
    q = (q_nope @ self.attn_k_b["weight"].transpose(-1, -2)).cat(q_rope, dim=-1)

    kv_a = self.attn_kv_a_mqa(x)
    c_kv = self.attn_kv_a_norm(kv_a[..., :self.config.kv_lora_rank])
    k_rope = kv_a[..., self.config.kv_lora_rank:].reshape(B, T, 1, self.config.rope_dim).transpose(1, 2)
    if not self.config.mla_nope: k_rope = apply_rope(k_rope, self.freqs_cis[start_pos:start_pos+T])

    k_store = c_kv.reshape(B, 1, T, self.config.kv_lora_rank).cat(k_rope.reshape(B, 1, T, self.config.rope_dim), dim=-1)
    k = Tensor(self.cache_k.uop.after(self.cache_k[:, :, start_pos:start_pos+T, :].uop.store(k_store.uop)))[:, :, 0:start_pos+T, :]
    v = k[..., :self.config.kv_lora_rank]

    mask = Tensor.full((1, 1, T, start_pos+T), float("-inf"), dtype=x.dtype, buffer=False).triu(start_pos+1) \
      if resolve(T != 1) else None
    attn = q @ k.transpose(-1, -2) * (1.0 / self.config.head_dim ** 0.5)
    if mask is not None: attn = attn + mask
    attn = attn.softmax(-1)
    attn = ((attn @ v) @ self.attn_v_b["weight"].transpose(-1, -2)).transpose(1, 2).reshape(B, T, -1)
    if hasattr(self, "attn_gate"): attn = attn * self.attn_gate(x).sigmoid()
    return self.attn_output(attn)

  def _init_state(self, x:Tensor):
    if not hasattr(self, "cache_k"):
      shape = (x.shape[0], 1, self.config.max_context, self.config.kv_lora_rank + self.config.rope_dim)
      self.cache_k = Tensor.zeros(*shape, dtype=x.dtype, device=x.device[0]).contiguous().shard(x.device).realize() \
        if isinstance(x.device, tuple) else Tensor.empty(*shape, dtype=x.dtype, device=x.device)
      if not self.config.mla_nope: self.freqs_cis = precompute_freqs_cis(self.config.rope_dim, self.config.max_context, self.config.rope_theta, device=x.device)

class GatedDeltaNetBlock(FFNBlock):
  def __init__(self, config:TransformerConfig, ssm:SSMConfig):
    super().__init__(config)
    self.head_k_dim, self.num_k_heads, self.num_v_heads = ssm.state_size, ssm.group_count, ssm.time_step_rank
    assert self.num_v_heads % self.num_k_heads == 0
    self.head_v_dim, self.ssm_conv_kernel = ssm.inner_size // ssm.time_step_rank, ssm.conv_kernel
    self.conv_channels, self.q_dim = ssm.inner_size + 2*ssm.group_count*ssm.state_size, ssm.state_size*ssm.group_count
    self.gate_lower_bound = ssm.gate_lower_bound
    self.attn_qkv = Linear(config.dim, self.conv_channels, bias=False)
    if ssm.kda:
      if ssm.full_rank_gate: self.ssm_g = Linear(config.dim, ssm.inner_size, bias=False)
      else: self.ssm_g_a, self.ssm_g_b = Linear(config.dim, self.head_v_dim, bias=False), Linear(self.head_v_dim, ssm.inner_size, bias=False)
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
    assert T == 1, "GatedDeltaNetBlock currently only supports T=1"

    # input processing
    x = x.half()
    out_gate = self.ssm_g(x) if hasattr(self, "ssm_g") else self.ssm_g_b(self.ssm_g_a(x)) if hasattr(self, "ssm_g_a") else self.attn_gate(x)
    out_gate = out_gate.reshape(B, 1, self.num_v_heads, self.head_v_dim)
    beta = self.ssm_beta(x).sigmoid().reshape(B, self.num_v_heads, 1, 1)
    alpha = (self.ssm_f_b(self.ssm_f_a(x)) if hasattr(self, "ssm_f_a") else self.ssm_alpha(x)).float() + self.ssm_dt["bias"]
    alpha = alpha.reshape(B, self.num_v_heads, -1)
    alpha = ((-self.ssm_a.reshape(1, self.num_v_heads, -1) * alpha).sigmoid() * self.gate_lower_bound
             if self.gate_lower_bound is not None else alpha.softplus() * self.ssm_a.reshape(1, self.num_v_heads, -1)).exp().unsqueeze(-2)

    # qkv conv
    conv_window = self.conv_state.cat(self.attn_qkv(x), dim=1)
    conv_out = (conv_window * self.ssm_conv1d["weight"].T.unsqueeze(0)).sum(1).silu()
    if hasattr(self, "ssm_f_a"):
      qkv = conv_out.reshape(B, -1, 3)
      q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]
    else:
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
    out_gate = out_gate.sigmoid() if hasattr(self, "ssm_f_a") else out_gate.silu()
    return self.ssm_out((core_attn_out * out_gate).reshape(B, 1, -1).cast(x.dtype))

  # recurrent state can't be partially reused after divergence, force a full rebuild
  def _state_reset_ops(self):
    return [self.conv_state.assign(self.conv_state.const_like(0)),
            self.recurrent_state.assign(self.recurrent_state.const_like(0))] if hasattr(self, "conv_state") else []
  def _reusable_prefix_len(self, prefix_len:int, cached_len:int) -> int: return 0 if prefix_len != cached_len else prefix_len

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
    if config.attn_res_block_size:
      block_size = config.attn_res_block_size
      self.output_res_score = nn.RMSNorm(config.dim, config.norm_eps)
      self.attn_res_specs = tuple(AttnResSpec(ceildiv(i, block_size), i//block_size if i % block_size == 0 else None)
                                  for i in range(config.num_blocks))
      self.num_attn_res_blocks = ceildiv(config.num_blocks, block_size)
    else:
      self.attn_res_specs, self.num_attn_res_blocks = (), 0
    self.max_context = config.max_context
    self.devices: tuple[str, ...]|None = None
    self.has_recurrent_block = any(isinstance(b, GatedDeltaNetBlock) for b in self.blk)
    self._cached_tokens: list[int] = []
    # we specialize the JIT for prefill and rollout
    self.prefill_jit = TinyJit(self.forward)
    self.rollout_jit = TinyJit(self.forward)

  def forward(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor) -> Tensor:
    x = self.token_embd(tokens).float()                   # (B, T, D)
    if self.devices is not None: x = x.to(self.devices)
    if self.num_attn_res_blocks:
      residuals = Tensor.empty(*x.shape[:2], self.num_attn_res_blocks, x.shape[-1], dtype=x.dtype, device=x.device)
      for block, spec in zip(self.blk, self.attn_res_specs): x, residuals = block._call_attn_res(x, start_pos, residuals, spec)
      x = apply_attn_res(x, residuals, self.output_res_score, self.num_attn_res_blocks)
    else:
      for block in self.blk: x = block(x, start_pos)
    sharded_head = self.devices is not None and isinstance(self.output.weight.device, tuple)
    x = self.output_norm(x)
    if sharded_head: x = x.contiguous()
    if self.devices is not None and not sharded_head: x = x.to(self.devices[0])
    logits = self.output(x)[:, -1, :]
    if sharded_head: temperature = temperature.to(self.devices)
    # Gumbel-max trick: argmax(logits/temp - log(-log(uniform))) is equivalent to sampling from softmax(logits/temp)
    scores = logits / temperature.maximum(1e-12) - (Tensor.rand_like(logits).maximum(1e-12).log().neg()).log()
    out = gathered_argmax(scores) if sharded_head and getenv("GATHERED_ARGMAX", 0) else scores.argmax(-1, keepdim=True)
    return out.to(self.devices[0]) if sharded_head else out

  def __call__(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor) -> Tensor:
    return (self.prefill_jit if resolve(tokens.shape[1] != 1) else self.rollout_jit)(tokens.contiguous(), start_pos, temperature)

  @staticmethod
  def from_gguf(gguf:Tensor|str|pathlib.Path, max_context:int|None=None,
                realize=bool(getenv("REALIZE", 0)), shard:int=1) -> tuple[Transformer, dict]:
    devices = tuple(Device.canonicalize(f"{Device.DEFAULT}:{i}") for i in range(shard)) if shard > 1 else None
    source = gguf.to(None).realize() if devices is None and isinstance(gguf, Tensor) else gguf
    layer_limit, split_limit = getenv("L", 0), getenv("S", 0)
    predequant = {x.strip() for x in getenv("PREDEQUANT", "").split(",") if x.strip()}
    if unknown := predequant - _PREDEQUANT_SUFFIXES.keys(): raise ValueError(f"unknown PREDEQUANT categories: {sorted(unknown)}")
    if layer_limit < 0: raise ValueError(f"L must be non-negative, got {layer_limit}")
    tensor_filter = (lambda name: (m:=re.match(r"blk\.(\d+)\.", name)) is None or int(m.group(1)) < layer_limit) if layer_limit else None
    kv, loaded = gguf_load(source, devices, _tp_policy if devices is not None else None,
                           _fuse_policy if devices is not None else None, _raw_policy,
                           split_limit=split_limit or None, tensor_filter=tensor_filter)
    raw_weights = {k:v for k,v in loaded.items() if isinstance(v, GGUFQuantizedTensor)}
    state_dict = {k:v for k,v in loaded.items() if isinstance(v, Tensor)}

    # all state items should be float16, not float32
    state_dict = {k:v.cast('float16') if getenv("HALF", 1) else v for k,v in state_dict.items()}
    if raw_weights and getenv("HALF", 1) and (routers := {k:v.contiguous() for k,v in state_dict.items() if k.endswith("ffn_gate_inp.weight")}):
      Tensor.realize(*routers.values())
      state_dict.update(routers)

    # some models like Llama 3.2 don't have an output.weight, they just tie to the token_embd.weight
    if 'output.weight' not in state_dict and 'token_embd.weight' in state_dict: state_dict['output.weight'] = state_dict['token_embd.weight']

    arch = kv['general.architecture']
    num_blocks = kv[f'{arch}.block_count'] - kv.get(f'{arch}.nextn_predict_layers', 0)
    if layer_limit: num_blocks = min(num_blocks, layer_limit)
    max_context = min(max_context, kv[f'{arch}.context_length']) if max_context is not None else kv[f'{arch}.context_length']
    n_heads, n_kv_heads = kv[f'{arch}.attention.head_count'], kv[f'{arch}.attention.head_count_kv']

    ssm = None
    ssm_layers: tuple[bool, ...] = ()
    if arch in ('qwen35', 'qwen35moe'):
      ssm = SSMConfig(**{k: kv[f'{arch}.ssm.{k}'] for k in ('conv_kernel','state_size','group_count','time_step_rank','inner_size')})
      ssm_layers = tuple((i+1) % kv[f'{arch}.full_attention_interval'] != 0 for i in range(num_blocks))
    elif arch in ('kimi-linear', 'kimi-k3'):
      ssm_layers = tuple(x == 0 for x in n_kv_heads[:num_blocks])
      n_kv_heads = max(n_kv_heads)
      ssm = SSMConfig(kv[f'{arch}.ssm.conv_kernel'], kv[f'{arch}.kda.head_dim'], n_heads, n_heads,
                      n_heads*kv[f'{arch}.kda.head_dim'], kda=True, full_rank_gate=arch == 'kimi-k3',
                      gate_lower_bound=kv.get(f'{arch}.kda.gate_lower_bound'))
      for i, is_ssm in enumerate(ssm_layers):
        if not is_ssm: continue
        q = state_dict.pop(f"blk.{i}.attn_q.weight")
        state_dict[f"blk.{i}.attn_qkv.weight"] = q.stack(state_dict.pop(f"blk.{i}.attn_k.weight"),
          state_dict.pop(f"blk.{i}.attn_v.weight"), dim=1).reshape(-1, q.shape[-1]).contiguous()
        q = state_dict.pop(f"blk.{i}.ssm_conv1d_q.weight").squeeze(1)
        state_dict[f"blk.{i}.ssm_conv1d.weight"] = q.stack(state_dict.pop(f"blk.{i}.ssm_conv1d_k.weight").squeeze(1),
          state_dict.pop(f"blk.{i}.ssm_conv1d_v.weight").squeeze(1), dim=1).reshape(-1, q.shape[-1]).contiguous()
        state_dict[f"blk.{i}.ssm_out.weight"] = state_dict.pop(f"blk.{i}.attn_output.weight")
        if arch == 'kimi-k3': state_dict[f"blk.{i}.ssm_a"] = state_dict[f"blk.{i}.ssm_a"].unsqueeze(-1)
    if arch in ('qwen35', 'qwen35moe', 'glm4moe'):
      state_dict = {k.replace('post_attention_norm', 'ffn_norm'):v for k,v in state_dict.items()}

    kv_lora_rank = kv.get(f'{arch}.attention.kv_lora_rank', 0)
    head_dim = kv.get(f'{arch}.attention.key_length_mla', kv.get(f'{arch}.attention.key_length', kv[f'{arch}.embedding_length'] // n_heads))
    rope_dim = kv.get(f'{arch}.rope.dimension_count', head_dim)

    # Permute RoPE weights from interleaved to half-split layout.
    for name in state_dict:
      if arch in ('kimi-linear', 'kimi-k3'): continue
      if ('attn_q.weight' in name or 'attn_q_b.weight' in name) and (arch == 'llama' or kv_lora_rank):
        w = state_dict[name].reshape(n_heads, state_dict[name].shape[0]//n_heads, -1)
        prefix = head_dim-rope_dim
        state_dict[name] = w[:, :prefix].cat(w[:, prefix:].rearrange("n (h two) d -> n (two h) d", two=2), dim=1).reshape(-1, w.shape[-1])
      elif arch == 'llama' and 'attn_k.weight' in name:
        w = state_dict[name].reshape(n_kv_heads, state_dict[name].shape[0]//n_kv_heads, -1)
        state_dict[name] = w.rearrange("n (h two) d -> n (two h) d", two=2).reshape(-1, w.shape[-1])
      elif kv_lora_rank and 'attn_kv_a_mqa.weight' in name:
        state_dict[name] = state_dict[name][:kv_lora_rank].cat(state_dict[name][kv_lora_rank:].rearrange("(h two) d -> (two h) d", two=2), dim=0)
    if predequant and (selected := {name:weight.contiguous() for name,weight in state_dict.items()
                                   if _predequant_category(name) in predequant}):
      Tensor.realize(*selected.values())
      state_dict.update(selected)
      print(f"predequantized {len(selected)} tensors ({sum(x.nbytes() for x in selected.values())/1e9:.2f} GB logical FP16)")
    config = TransformerConfig(
      num_blocks=num_blocks, dim=kv[f'{arch}.embedding_length'],
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
      norm_topk_prob=kv.get(f'{arch}.expert_weights_norm', arch in ('qwen3moe', 'qwen35moe', 'kimi-linear', 'kimi-k3')),
      mla_nope=arch == 'kimi-k3',
      kv_lora_rank=kv_lora_rank, q_lora_rank=kv.get(f'{arch}.attention.q_lora_rank', 0),
      leading_dense_blocks=kv.get(f'{arch}.leading_dense_block_count', 0),
      shared_expert_dim=kv.get(
        f'{arch}.expert_shared_feed_forward_length',
        kv.get(f'{arch}.expert_shared_count', 0) * kv.get(f'{arch}.expert_feed_forward_length', 0)),
      shared_expert_gate=f"blk.{kv.get(f'{arch}.leading_dense_block_count', 0)}.ffn_gate_inp_shexp.weight" in state_dict,
      dense_hidden_dim=kv.get(f'{arch}.feed_forward_length', 0) if kv.get(f'{arch}.leading_dense_block_count', 0) else 0,
      routed_scaling_factor=kv.get(f'{arch}.expert_weights_scale', 1.0), attn_output_gate=arch in ('qwen35', 'qwen35moe', 'kimi-k3'), ssm=ssm,
      ssm_layers=ssm_layers,
      qkv_bias='blk.0.attn_q.bias' in state_dict,
      expert_bias=f"blk.{kv.get(f'{arch}.leading_dense_block_count', 0)}.exp_probs_b.bias" in state_dict,
      expert_latent_dim=kv.get(f'{arch}.expert_latent_length', 0),
      expert_latent_norm=f"blk.{kv.get(f'{arch}.leading_dense_block_count', 0)}.ffn_routed_norm.weight" in state_dict,
      situ_beta=kv.get(f'{arch}.activation.situ_beta', 0.0), situ_linear_beta=kv.get(f'{arch}.activation.situ_linear_beta', 0.0),
      attn_res_block_size=kv.get(f'{arch}.attn_res.block_size', 0))
    model = Transformer(config)
    for i, block in enumerate(model.blk):
      for attr in ("attn_q_a", "attn_kv_a_mqa"):
        name = f"blk.{i}.{attr}.weight"
        if hasattr(block, attr) and (weight := raw_weights.get(name)) is not None:
          getattr(block, attr).weight = weight
          state_dict.update({f"{name}.data.{j}": raw for j, raw in enumerate(weight.data)})
      fused_name = f"blk.{i}.ffn_gateup_exps.weight"
      if fused_name in state_dict or fused_name in raw_weights:
        block.ffn_gateup_exps = ExpertWeights(block.config.num_experts, block.config.expert_latent_dim or block.config.dim, 2*block.config.hidden_dim)
        del block.ffn_gate_exps, block.ffn_up_exps
      for attr in ("ffn_gate_exps", "ffn_up_exps", "ffn_gateup_exps", "ffn_down_exps"):
        name = f"blk.{i}.{attr}.weight"
        if (weight := raw_weights.get(name)) is None: continue
        getattr(block, attr).weight = weight
        state_dict.update({f"{name}.data.{j}": raw for j, raw in enumerate(weight.data)})
    model.num_params = sum(int(x.numel()) for x in nn.state.get_parameters(model)) + \
      sum(weight.numel() - sum(int(x.numel()) for x in weight.data) for weight in raw_weights.values())
    _place_model(model, state_dict, devices)
    nn.state.load_state_dict(model, state_dict, strict=not (layer_limit or split_limit), verbose=False, consume=True, realize=False)
    # NOTE: without this contiguous, it unpacks the weights from the model every time. we shouldn't need this, but for now it's faster
    if realize:
      for s in (params:=nn.state.get_parameters(model)): s.replace(s.contiguous())
      Tensor.realize(*params)
    return model, kv

  def warmup(self):
    for _ in range(2): list(zip(range(2), self.generate([0])))

  def get_start_pos(self, tokens:list[int]) -> int:
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
    if start_pos < len(self._cached_tokens) and (resets := [r for b in self.blk for r in b._state_reset_ops()]): Tensor.realize(*resets)
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
