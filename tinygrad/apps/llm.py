from __future__ import annotations
import sys, argparse, typing, re, unicodedata, json, uuid, time, functools
from tinygrad import Tensor, nn, UOp, TinyJit, getenv, function
from tinygrad.helpers import partition, DEBUG, Timing, GlobalCounters, stderr_log, colored
from tinygrad.viz.serve import TCPServerWithReuse, HTTPRequestHandler

class SimpleTokenizer:
  def __init__(self, normal_tokens:dict[str, int], special_tokens:dict[str, int], preset:str="llama3"):
    if preset not in ("llama3","llama-v3","llama-bpe","qwen2","qwen35","olmo"): raise ValueError(f"Invalid tokenizer preset '{preset}'")
    # https://github.com/openai/gpt-2/blob/9b63575ef42771a015060c964af2c3da4cf7c8ab/src/encoder.py#L9
    bs = [*range(33, 127), *range(161, 173), *range(174, 256)]  # bytes that map to themselves
    self._byte_decoder = {chr(b): b for b in bs} | {chr(256+i): b for i,b in enumerate(b for b in range(256) if b not in bs)}

    # https://github.com/ggml-org/llama.cpp/blob/94933c8c2eeaa9a7983e3f6c08af76bd86724094/src/llama-vocab.cpp#L286
    # 0x323b0 is one past the max codepoint in unicode categories L/N/Z (0x323af is max L)
    def ucat_range(pre: str): return "".join(re.escape(chr(cp)) for cp in range(0x323b0) if unicodedata.category(chr(cp)).startswith(pre))
    r_ws, r_p_N, r_p_L = r"\t\n\x0b\x0c\r\x85" + ucat_range("Z"), ucat_range("N"), ucat_range("L")
    self._split_to_word = re.compile("(?i:'s|'t|'re|'ve|'m|'ll|'d)|" + \
      f"[^\\r\\n{r_p_N}{r_p_L}]?[{r_p_L}]+|[{r_p_N}]{{1,3}}| ?[^{r_ws}{r_p_N}{r_p_L}]+[\\r\\n]*|[{r_ws}]*[\\r\\n]+|[{r_ws}]+(?![^{r_ws}])|[{r_ws}]+")
    self._split_to_sentence = re.compile("|".join(re.escape(tok) for tok in special_tokens.keys()) if special_tokens else r"(?!)")

    self._normal_tokens = {bytes(self._byte_decoder[c] for c in tok): tid for tok, tid in normal_tokens.items()}
    self._special_tokens = special_tokens
    self._tok2bytes = {tid: tok for tok, tid in self._normal_tokens.items()} | {tid: tok.encode() for tok, tid in self._special_tokens.items()}
    self.preset = preset

  @staticmethod
  def from_gguf_kv(kv:dict):
    # https://github.com/ggml-org/llama.cpp/blob/94933c8c2eeaa9a7983e3f6c08af76bd86724094/src/llama-vocab.cpp#L1818-L1820
    vocab: typing.Iterable[tuple[str, int]] = ((tok, idx) for idx, tok in enumerate(kv["tokenizer.ggml.tokens"]))
    normal_tokens, special_tokens = partition(vocab, lambda e: kv["tokenizer.ggml.token_type"][e[1]] == 1)
    return SimpleTokenizer(dict(normal_tokens), dict(special_tokens), kv["tokenizer.ggml.pre"])

  def _encode_word(self, word:bytes) -> list[int]:
    if (early_token:=self._normal_tokens.get(word)) is not None: return [early_token]
    parts = [bytes([b]) for b in word]
    # greedily merge any parts that we can
    while True:
      i = min([(sys.maxsize, -1)] + [(self._normal_tokens.get(parts[j]+parts[j+1], sys.maxsize), j) for j in range(len(parts)-1)])[1]
      if i == -1: break
      parts[i:i+2] = [parts[i] + parts[i+1]]
    try: return [self._normal_tokens[p] for p in parts]
    except KeyError: raise RuntimeError("token not found")
  def _encode_sentence(self, chunk:str) -> list[int]:
    return [tok for word in self._split_to_word.findall(chunk) for tok in self._encode_word(word.encode())]
  def encode(self, text:str) -> list[int]:
    tokens: list[int] = []
    pos = 0
    for match in self._split_to_sentence.finditer(text):
      tokens.extend(self._encode_sentence(text[pos:match.start(0)]) + [self._special_tokens[text[match.start(0):match.end(0)]]])
      pos = match.end(0)
    return tokens + self._encode_sentence(text[pos:])

  def decode(self, ids:list[int]) -> str: return b''.join(self._tok2bytes[tid] for tid in ids).decode(errors='replace')
  def role(self, role:str):
    if self.preset == 'olmo': return self.encode("<|" + role + "|>\n")  # OLMoE Instruct format
    if self.preset in ('qwen2', 'qwen35'): return self.encode("<|im_start|>" + role + "\n")
    return self.encode("<|start_header_id|>" + role + "<|end_header_id|>\n\n")
  def assistant_prompt(self):
    # Qwen3.5 template injects an empty think block by default when add_generation_prompt is used.
    if self.preset == 'qwen35': return self.encode("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    return self.role("assistant")
  def end_turn(self, eos_id:int):
    if self.preset == 'olmo': return self.encode("\n")
    if self.preset in ('qwen2', 'qwen35'): return [eos_id] + self.encode("\n")
    return [eos_id]

@functools.cache
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> Tensor:
  freqs = 1.0 / (theta ** (Tensor.arange(0, dim, 2)[:(dim // 2)] / dim))
  freqs = Tensor.arange(end).unsqueeze(dim=1) * freqs.unsqueeze(dim=0)
  return freqs.cos().cat(freqs.sin(), dim=-1).contiguous()

class ExpertWeights:
  """Like nn.Linear but with num_experts dimension. Weight shape: (num_experts, out_features, in_features)."""
  def __init__(self, num_experts:int, in_features:int, out_features:int):
    self.weight = Tensor.zeros(num_experts, out_features, in_features)
  def __call__(self, sel:Tensor, x:Tensor) -> Tensor:
    # sel: (B, T, k), x: (B, T, 1, in) or (B, T, k, in) -> output: (B, T, k, out)
    return (x.unsqueeze(-2) @ self.weight[sel].transpose(-1, -2)).squeeze(-2)

def apply_rope(x:Tensor, freqs_cis:Tensor) -> Tensor:
  assert x.shape[-1] % 2 == 0
  cos, sin = freqs_cis.reshape(1, 1, x.shape[2], -1).chunk(2, dim=-1)
  x1, x2 = x.chunk(2, dim=-1)
  return (x1 * cos - x2 * sin).cat(x2 * cos + x1 * sin, dim=-1)

def apply_rope_partial(x:Tensor, freqs_cis:Tensor, rope_dim:int) -> Tensor:
  if rope_dim == x.shape[-1]: return apply_rope(x, freqs_cis)
  x_rope, x_pass = x[..., :rope_dim], x[..., rope_dim:]
  return apply_rope(x_rope, freqs_cis).cat(x_pass, dim=-1)

def l2_normalize(x:Tensor, eps:float=1e-6) -> Tensor:
  return x * (x.square().sum(-1, keepdim=True) + eps).rsqrt()

class TransformerBlock:
  def __init__(self, dim:int, hidden_dim:int, n_heads:int, n_kv_heads:int, norm_eps:float, head_dim:int, rope_theta:float,
               max_context:int=0, qk_norm:int=0, num_experts:int=0, num_experts_per_tok:int=0):
    self.n_heads      = n_heads
    self.n_kv_heads   = n_kv_heads
    self.head_dim     = head_dim
    self.rope_theta   = rope_theta
    self.max_context  = max_context
    self.qk_norm      = qk_norm

    # --- attention projections (all linear, bias-free) ------------------
    q_proj_out       = self.head_dim * n_heads
    kv_proj_out      = self.head_dim * n_kv_heads
    self.attn_q      = nn.Linear(dim, q_proj_out,  bias=False)
    self.attn_k      = nn.Linear(dim, kv_proj_out, bias=False)
    self.attn_v      = nn.Linear(dim, kv_proj_out, bias=False)
    self.attn_output = nn.Linear(q_proj_out, dim,  bias=False)

    # --- RMSNorms --------------------------------------------------------
    self.attn_norm   = nn.RMSNorm(dim, norm_eps)
    self.ffn_norm    = nn.RMSNorm(dim, norm_eps)
    if qk_norm: self.attn_q_norm, self.attn_k_norm = nn.RMSNorm(qk_norm, norm_eps), nn.RMSNorm(qk_norm, norm_eps)

    # --- feed-forward (MoE or dense) -------------------------------------
    if num_experts > 0:
      self.num_experts_per_tok = num_experts_per_tok
      self.ffn_gate_inp = nn.Linear(dim, num_experts, bias=False)  # router
      self.ffn_gate_exps = ExpertWeights(num_experts, dim, hidden_dim)
      self.ffn_up_exps = ExpertWeights(num_experts, dim, hidden_dim)
      self.ffn_down_exps = ExpertWeights(num_experts, hidden_dim, dim)
    else:
      self.ffn_gate    = nn.Linear(dim, hidden_dim, bias=False)
      self.ffn_up      = nn.Linear(dim, hidden_dim, bias=False)
      self.ffn_down    = nn.Linear(hidden_dim, dim, bias=False)

  @function
  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    x_norm = self.attn_norm(x)                       # (B,T,D)
    q, k, v = self.attn_q(x_norm), self.attn_k(x_norm), self.attn_v(x_norm)
    if self.qk_norm and self.qk_norm != self.head_dim: q, k = self.attn_q_norm(q), self.attn_k_norm(k)

    B, T, _ = x.shape
    q = q.reshape(B, T, self.n_heads,    self.head_dim).transpose(1, 2)  # (B,H,T,Hd)
    k = k.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)  # (B,KvH,T,Hd)
    v = v.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)  # (B,KvH,T,Hd)
    if self.qk_norm == self.head_dim: q, k = self.attn_q_norm(q), self.attn_k_norm(k)

    freqs_cis = precompute_freqs_cis(self.head_dim, self.max_context, self.rope_theta)[start_pos:start_pos+T]
    q = apply_rope(q, freqs_cis)
    k = apply_rope(k, freqs_cis)

    # TODO: fix assign to behave like this
    assigned_kv = self.cache_kv.uop.after(self.cache_kv[:, :, :, start_pos:start_pos+T, :].uop.assign(Tensor.stack(k, v).contiguous().uop))
    tensor_assigned_kv = Tensor(assigned_kv, device=assigned_kv.device)
    k = tensor_assigned_kv[0, :, :, 0:start_pos+T, :]
    v = tensor_assigned_kv[1, :, :, 0:start_pos+T, :]

    #self.cache_kv[:, :, :, start_pos:start_pos+T, :].assign(Tensor.stack(k, v))
    #k = self.cache_kv[0, :, :, 0:start_pos+T, :]
    #v = self.cache_kv[1, :, :, 0:start_pos+T, :]

    # NOTE: this mask is causal_lower_right, not the causal_upper_left generated by is_casual = True
    # TODO: this if statement should be removed and it shouldn't generate extra kernels
    mask = Tensor.full((1, 1, T, start_pos+T), float("-inf"), dtype=x.dtype, device=x.device).triu(start_pos+1) if T > 1 else None
    attn = q.scaled_dot_product_attention(k, v, attn_mask=mask, enable_gqa=True)     # (B,H,T,Hd)
    attn = attn.transpose(1, 2).reshape(B, T, -1)                                    # back to (B,T,D)
    attn = self.attn_output(attn)
    return x + attn

  @function
  def _feed_forward(self, h: Tensor) -> Tensor:
    h_norm = self.ffn_norm(h)
    if hasattr(self, 'ffn_gate_exps'):
      x = h_norm.unsqueeze(2)  # (B, T, 1, D) - add expert dim for broadcasting
      probs, sel = self.ffn_gate_inp(h_norm).softmax(-1).topk(self.num_experts_per_tok)  # (B, T, k) each
      x_down = self.ffn_down_exps(sel, self.ffn_gate_exps(sel, x).silu() * self.ffn_up_exps(sel, x))  # (B, T, k, D)
      return h + (x_down * probs.unsqueeze(-1)).sum(axis=2)  # (B, T, D)
    # TODO: remove the need for this contiguous
    gated  = self.ffn_gate(h_norm).silu().contiguous() * self.ffn_up(h_norm)
    return h + self.ffn_down(gated)

  def __call__(self, x: Tensor, start_pos: int|UOp):
    if not hasattr(self, "cache_kv"):
      # TODO: how is the dtype of this determined?
      self.cache_kv = Tensor.zeros(2, x.shape[0], self.n_kv_heads, self.max_context, self.head_dim, device=x.device).contiguous().realize()
    return self._feed_forward(self._attention(x, start_pos)).contiguous()

class WeightOnly:
  def __init__(self, *shape:int): self.weight = Tensor.zeros(*shape)

class BiasOnly:
  def __init__(self, dim:int): self.bias = Tensor.zeros(dim)

class Qwen35Block:
  def __init__(self, dim:int, hidden_dim:int, n_heads:int, n_kv_heads:int, norm_eps:float, head_dim:int, rope_theta:float, rope_dim:int,
               max_context:int, is_recurrent:bool, ssm_conv_kernel:int, ssm_state_size:int, ssm_group_count:int, ssm_time_step_rank:int,
               ssm_inner_size:int):
    self.n_heads, self.n_kv_heads, self.head_dim = n_heads, n_kv_heads, head_dim
    self.rope_theta, self.rope_dim, self.max_context = rope_theta, rope_dim, max_context
    self.is_recurrent = is_recurrent
    self.norm_eps = norm_eps

    self.attn_norm = nn.RMSNorm(dim, norm_eps)
    self.post_attention_norm = nn.RMSNorm(dim, norm_eps)
    self.ffn_gate = nn.Linear(dim, hidden_dim, bias=False)
    self.ffn_up = nn.Linear(dim, hidden_dim, bias=False)
    self.ffn_down = nn.Linear(hidden_dim, dim, bias=False)

    if self.is_recurrent:
      self.head_k_dim = ssm_state_size
      self.num_k_heads = ssm_group_count
      self.num_v_heads = ssm_time_step_rank
      self.head_v_dim = ssm_inner_size // self.num_v_heads
      self.ssm_conv_kernel = ssm_conv_kernel
      self.conv_channels = ssm_inner_size + 2*self.num_k_heads*self.head_k_dim
      self.q_dim = self.head_k_dim * self.num_k_heads

      self.attn_qkv = nn.Linear(dim, self.conv_channels, bias=False)
      self.attn_gate = nn.Linear(dim, ssm_inner_size, bias=False)
      self.ssm_alpha = nn.Linear(dim, self.num_v_heads, bias=False)
      self.ssm_beta = nn.Linear(dim, self.num_v_heads, bias=False)
      # NOTE: gguf_load reverses tensor dims, so this matches blk.*.ssm_conv1d.weight
      self.ssm_conv1d = WeightOnly(self.conv_channels, self.ssm_conv_kernel)
      self.ssm_dt = BiasOnly(self.num_v_heads)
      self.ssm_a = Tensor.zeros(self.num_v_heads)
      self.ssm_norm = nn.RMSNorm(self.head_v_dim, norm_eps)
      self.ssm_out = nn.Linear(ssm_inner_size, dim, bias=False)
    else:
      self.attn_q = nn.Linear(dim, self.n_heads * self.head_dim * 2, bias=False)
      self.attn_k = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
      self.attn_v = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
      self.attn_output = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
      self.attn_q_norm = nn.RMSNorm(self.head_dim, norm_eps)
      self.attn_k_norm = nn.RMSNorm(self.head_dim, norm_eps)

  def _full_attention(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    B, T, _ = x.shape
    qg = self.attn_q(x).reshape(B, T, self.n_heads, 2, self.head_dim)
    q = self.attn_q_norm(qg[:, :, :, 0, :].transpose(1, 2))
    gate = qg[:, :, :, 1, :].reshape(B, T, self.n_heads * self.head_dim)
    k = self.attn_k_norm(self.attn_k(x).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2))
    v = self.attn_v(x).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

    freqs_cis = precompute_freqs_cis(self.rope_dim, self.max_context, self.rope_theta)[start_pos:start_pos+T]
    q = apply_rope_partial(q, freqs_cis, self.rope_dim)
    k = apply_rope_partial(k, freqs_cis, self.rope_dim)

    # TODO: fix assign to behave like this
    assigned_kv = self.cache_kv.uop.after(self.cache_kv[:, :, :, start_pos:start_pos+T, :].uop.assign(Tensor.stack(k, v).contiguous().uop))
    tensor_assigned_kv = Tensor(assigned_kv, device=assigned_kv.device)
    k = tensor_assigned_kv[0, :, :, 0:start_pos+T, :]
    v = tensor_assigned_kv[1, :, :, 0:start_pos+T, :]

    mask = Tensor.full((1, 1, T, start_pos+T), float("-inf"), dtype=x.dtype, device=x.device).triu(start_pos+1) if T > 1 else None
    attn = q.scaled_dot_product_attention(k, v, attn_mask=mask, enable_gqa=True).transpose(1, 2).reshape(B, T, -1)
    return self.attn_output(attn * gate.sigmoid())

  def _linear_attention(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    B, T, _ = x.shape
    if not hasattr(self, "conv_cache") or self.conv_cache.shape[0] != B:
      self.conv_cache = Tensor.zeros(B, self.ssm_conv_kernel-1, self.conv_channels, device=x.device).contiguous().realize()
      self.ssm_state = Tensor.zeros(B, self.num_v_heads, self.head_v_dim, self.head_v_dim, device=x.device).contiguous().realize()
    if isinstance(start_pos, int) and start_pos == 0:
      self.conv_cache = Tensor.zeros(*self.conv_cache.shape, device=x.device).contiguous().realize()
      self.ssm_state = Tensor.zeros(*self.ssm_state.shape, device=x.device).contiguous().realize()

    if not hasattr(self, "_fused_w_qkvz"):
      self._fused_w_qkvz = self.attn_qkv.weight.cat(self.attn_gate.weight, dim=0).contiguous().realize()
      self._fused_w_ab = self.ssm_alpha.weight.cat(self.ssm_beta.weight, dim=0).contiguous().realize()
    qkvz = x.linear(self._fused_w_qkvz.transpose()).float()
    qkv = qkvz[..., :self.conv_channels]
    z = qkvz[..., self.conv_channels:].reshape(B, T, self.num_v_heads, self.head_v_dim)
    ab = x.linear(self._fused_w_ab.transpose()).float()
    alpha = (ab[..., :self.num_v_heads] + self.ssm_dt.bias).softplus()
    beta = ab[..., self.num_v_heads:].sigmoid()
    gate = alpha * self.ssm_a

    conv_w = self.ssm_conv1d.weight.transpose().float().reshape(1, self.ssm_conv_kernel, self.conv_channels)
    state = self.ssm_state.float()
    conv_cache = self.conv_cache.float()
    outs: list[Tensor] = []
    q_scale = self.head_k_dim**-0.5
    k_repeat = self.num_v_heads // self.num_k_heads
    for i in range(T):
      conv_input = conv_cache.cat(qkv[:, i:i+1, :], dim=1)
      conv_out = (conv_input * conv_w).sum(axis=1).silu()
      conv_cache = conv_input[:, 1:, :]

      q = conv_out[:, :self.q_dim].reshape(B, self.num_k_heads, self.head_k_dim)
      k = conv_out[:, self.q_dim:2*self.q_dim].reshape(B, self.num_k_heads, self.head_k_dim)
      v = conv_out[:, 2*self.q_dim:].reshape(B, self.num_v_heads, self.head_v_dim)
      q, k = l2_normalize(q, self.norm_eps), l2_normalize(k, self.norm_eps)
      if self.num_k_heads != self.num_v_heads:
        # GGUF qwen3.5 linear-attn weights use tiled V-head order when num_v_heads > num_k_heads.
        q = q.unsqueeze(1).expand(B, k_repeat, self.num_k_heads, self.head_k_dim).reshape(B, self.num_v_heads, self.head_k_dim)
        k = k.unsqueeze(1).expand(B, k_repeat, self.num_k_heads, self.head_k_dim).reshape(B, self.num_v_heads, self.head_k_dim)

      q, k, v = (q*q_scale).unsqueeze(-1), k.unsqueeze(-1), v.unsqueeze(-1)
      state = state * gate[:, i, :].reshape(B, self.num_v_heads, 1, 1).exp()
      d = (v - state@k) * beta[:, i, :].reshape(B, self.num_v_heads, 1, 1)
      state = state + d@k.transpose(-1, -2)
      out = (state@q).squeeze(-1).reshape(B, 1, self.num_v_heads, self.head_v_dim)
      outs.append(self.ssm_out((self.ssm_norm(out) * z[:, i:i+1].silu()).reshape(B, 1, -1).cast(x.dtype)))

    # Keep cache buffers stable across JIT runs; side-effect assign without replacing Tensor.uop.
    assigned_conv_uop = self.conv_cache.uop.after(self.conv_cache.uop.assign(conv_cache.cast(self.conv_cache.dtype).uop))
    assigned_state_uop = self.ssm_state.uop.after(self.ssm_state.uop.assign(state.cast(self.ssm_state.dtype).uop))
    assigned_conv, assigned_state = Tensor(assigned_conv_uop, device=assigned_conv_uop.device), Tensor(assigned_state_uop, device=assigned_state_uop.device)
    ret = outs[0] if len(outs) == 1 else outs[0].cat(*outs[1:], dim=1)
    return ret + (assigned_conv[:, :1, :1].sum() + assigned_state[:, :, :1, :1].sum()).cast(ret.dtype) * 0

  def __call__(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    x_norm = self.attn_norm(x)
    if self.is_recurrent: attn_out = self._linear_attention(x_norm, start_pos)
    else:
      if not hasattr(self, "cache_kv"):
        self.cache_kv = Tensor.zeros(2, x.shape[0], self.n_kv_heads, self.max_context, self.head_dim, device=x.device).contiguous().realize()
      attn_out = self._full_attention(x_norm, start_pos)
    h = x + attn_out
    # TODO: remove contiguous requirement when linear path no longer needs it.
    ffn = self.ffn_down(self.ffn_gate((x_norm:=self.post_attention_norm(h))).silu().contiguous() * self.ffn_up(x_norm))
    return h + ffn

class Transformer:
  def __init__(self, *, num_blocks, dim, hidden_dim, n_heads, n_kv_heads, norm_eps, vocab_size, head_dim:int, rope_theta:float,
               max_context:int=0, qk_norm:int=0, num_experts:int=0, num_experts_per_tok:int=0, qwen35:bool=False, qwen35_rope_dim:int=0,
               qwen35_full_attention_interval:int=0, qwen35_ssm_conv_kernel:int=0, qwen35_ssm_state_size:int=0,
               qwen35_ssm_group_count:int=0, qwen35_ssm_time_step_rank:int=0, qwen35_ssm_inner_size:int=0):
    self.no_prompt_overlap = qwen35
    self.supports_jit = True
    self.supports_sym_start_pos = True
    if qwen35:
      self.blk = [Qwen35Block(dim, hidden_dim, n_heads, n_kv_heads, norm_eps, head_dim, rope_theta, qwen35_rope_dim, max_context,
                              (i+1) % qwen35_full_attention_interval != 0, qwen35_ssm_conv_kernel, qwen35_ssm_state_size,
                              qwen35_ssm_group_count, qwen35_ssm_time_step_rank, qwen35_ssm_inner_size) for i in range(num_blocks)]
    else:
      self.blk = [TransformerBlock(dim, hidden_dim, n_heads, n_kv_heads, norm_eps, head_dim, rope_theta, max_context, qk_norm,
                                   num_experts, num_experts_per_tok) for _ in range(num_blocks)]
    self.token_embd  = nn.Embedding(vocab_size, dim)
    self.output_norm = nn.RMSNorm(dim, norm_eps)
    self.output = nn.Linear(dim, vocab_size, bias=False)
    self.max_context = max_context
    # JIT is used if T=1 and start_pos is a UOp. TODO: make this not needed by including T in the JIT and making start_pos always a UOp
    self.forward_jit = TinyJit(self.forward)

  def forward(self, tokens:Tensor, start_pos:int|UOp) -> Tensor:
    x = self.token_embd(tokens)                           # (B, T, D)
    for block in self.blk: x = block(x, start_pos)
    # TODO: add temperature
    return self.output(self.output_norm(x))[:, -1, :].softmax(-1, dtype="float").argmax(-1, keepdim=True)

  def __call__(self, tokens:Tensor, start_pos:int|UOp=0) -> Tensor:
    use_jit = self.supports_jit and getenv("JIT", 1) and tokens.shape[1] == 1 and isinstance(start_pos, UOp)
    return (self.forward_jit if use_jit else self.forward)(tokens, start_pos)

  @staticmethod
  def from_gguf(gguf:Tensor, max_context:int|None=None, realize=bool(getenv("REALIZE", 1))) -> tuple[Transformer, dict]:
    # TODO: remove the need for copy to default device
    kv, state_dict = nn.state.gguf_load(gguf.to(None).realize())

    # all state items should be float16, not float32
    state_dict = {k:v.cast('float16') if getenv("HALF", 1) else v for k,v in state_dict.items()}

    # some models like Llama 3.2 don't have an output.weight, they just tie to the token_embd.weight
    if 'output.weight' not in state_dict: state_dict['output.weight'] = state_dict['token_embd.weight']

    arch = kv['general.architecture']
    max_context = min(max_context, kv[f'{arch}.context_length']) if max_context is not None else kv[f'{arch}.context_length']
    n_heads, n_kv_heads = kv[f'{arch}.attention.head_count'], kv[f'{arch}.attention.head_count_kv']

    if arch == 'qwen35':
      model = Transformer(num_blocks=kv[f'{arch}.block_count'], dim=kv[f'{arch}.embedding_length'],
                          hidden_dim=kv[f'{arch}.feed_forward_length'], n_heads=n_heads, n_kv_heads=n_kv_heads,
                          norm_eps=kv[f'{arch}.attention.layer_norm_rms_epsilon'], vocab_size=len(kv['tokenizer.ggml.tokens']),
                          head_dim=kv[f'{arch}.attention.key_length'], rope_theta=kv[f'{arch}.rope.freq_base'],
                          max_context=max_context, qwen35=True, qwen35_rope_dim=kv[f'{arch}.rope.dimension_count'],
                          qwen35_full_attention_interval=kv[f'{arch}.full_attention_interval'],
                          qwen35_ssm_conv_kernel=kv[f'{arch}.ssm.conv_kernel'], qwen35_ssm_state_size=kv[f'{arch}.ssm.state_size'],
                          qwen35_ssm_group_count=kv[f'{arch}.ssm.group_count'], qwen35_ssm_time_step_rank=kv[f'{arch}.ssm.time_step_rank'],
                          qwen35_ssm_inner_size=kv[f'{arch}.ssm.inner_size'])
    else:
      # Permute Q/K weights from interleaved to half-split RoPE layout (llama-style models only)
      if arch == 'llama':
        for name in state_dict:
          if 'attn_q.weight' in name: state_dict[name] = state_dict[name].rearrange("(n h two) d -> (n two h) d", n=n_heads, two=2)
          if 'attn_k.weight' in name: state_dict[name] = state_dict[name].rearrange("(n h two) d -> (n two h) d", n=n_kv_heads, two=2)

      model = Transformer(num_blocks=kv[f'{arch}.block_count'], dim=kv[f'{arch}.embedding_length'],
                          hidden_dim=kv.get(f'{arch}.expert_feed_forward_length', kv[f'{arch}.feed_forward_length']),
                          n_heads=n_heads, n_kv_heads=n_kv_heads, norm_eps=kv[f'{arch}.attention.layer_norm_rms_epsilon'],
                          vocab_size=len(kv['tokenizer.ggml.tokens']),
                          head_dim=kv.get(f'{arch}.attention.key_length', kv[f'{arch}.embedding_length'] // n_heads),
                          rope_theta=kv[f'{arch}.rope.freq_base'], max_context=max_context,
                          qk_norm=int(state_dict['blk.0.attn_q_norm.weight'].shape[0]) if 'blk.0.attn_q_norm.weight' in state_dict else 0,
                          num_experts=kv.get(f'{arch}.expert_count', 0), num_experts_per_tok=kv.get(f'{arch}.expert_used_count', 0))
    nn.state.load_state_dict(model, state_dict, verbose=False, consume=True, realize=False)  # NOTE: rope_freqs.weight (32,) is unused
    # NOTE: without this contiguous, it unpacks the weights from the model every time. we shouldn't need this, but for now it's faster
    if realize:
      for s in (params:=nn.state.get_parameters(model)): s.replace(s.contiguous())
      Tensor.realize(*params)
    return model, kv

  def generate(self, tokens:list[int], start_pos=0):
    # Recurrent qwen35 blocks do not support re-processing one-token overlap from previous generate call.
    if self.no_prompt_overlap and start_pos > 0: start_pos += 1
    v_start_pos = UOp.variable("start_pos", 1, self.max_context-1)
    t = Tensor([tokens[start_pos:]], dtype="int32")
    while len(tokens) < self.max_context:
      use_sym = self.supports_sym_start_pos and getenv("SYM", 1) and start_pos != 0 and t.shape[-1] == 1
      t = self(t, v_start_pos.bind(start_pos) if use_sym else start_pos)
      next_id = int(t.item())
      tokens.append(next_id)
      start_pos = len(tokens) - 1
      yield next_id

models = {
  "llama3.2:1b": "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q6_K.gguf",
  "llama3.2:1b-q4": "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
  "llama3.2:3b": "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q6_K.gguf",
  "llama3.2:3b-f16": "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-f16.gguf",
  "llama3.1:8b": "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
  "qwen3:0.6b": "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf",
  "qwen3.5:0.8b": "https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q8_0.gguf",
  "qwen3:1.7b": "https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q4_K_M.gguf",
  "qwen3:8b": "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf",
  "qwen3.5:9b": "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q6_K.gguf",
  "qwen3.5:27b": "https://huggingface.co/unsloth/Qwen3.5-27B-GGUF/resolve/main/Qwen3.5-27B-Q4_K_M.gguf",
  "qwen3:30b-a3b": "https://huggingface.co/Qwen/Qwen3-30B-A3B-GGUF/resolve/main/Qwen3-30B-A3B-Q4_K_M.gguf",
  "olmoe": "https://huggingface.co/allenai/OLMoE-1B-7B-0924-Instruct-GGUF/resolve/main/olmoe-1b-7b-0924-instruct-q4_k_m.gguf",
}

# *** simple OpenAI compatible server on 11434 to match ollama ***
# OPENAI_BASE_URL=http://localhost:11434/v1 OPENAI_API_KEY=ollama uvx --from gpt-command-line gpt

CHAT_HTML = b'''<!DOCTYPE html><html><head><title>tinygrad chat</title><style>
  * { margin: 0 }
  body { background: #212121; color: #e3e3e3; font-family: system-ui;
         height: 100vh; display: flex; flex-direction: column }
  #chat { flex: 1; overflow-y: auto; padding: 20px }
  .msg { padding: 10px 16px; margin: 8px 0; white-space: pre-wrap; border-radius: 18px }
  .user { background: #2f2f2f; margin-left: auto; width: fit-content; max-width: 70% }
  #input { max-width: 768px; width: 100%; margin: 20px auto; padding: 14px 20px;
           background: #2f2f2f; color: inherit; font: inherit;
           border: none; outline: none; resize: none; border-radius: 24px; field-sizing: content }
</style></head><body><div id="chat"></div>
<textarea id="input" rows="1" placeholder="Ask anything"></textarea>
<script>
  input.onkeydown = (e) => { if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); send() } }
  const msgs = [];
  async function send() {
    if (!input.value.trim()) return;
    msgs.push({role: 'user', content: input.value.trim()});
    chat.innerHTML += '<div class="msg user">' + input.value.trim().replace(/</g, '&lt;') + '</div>';
    input.value = '';
    const d = document.createElement('div'); d.className = 'msg'; chat.appendChild(d);
    const r = await fetch('/v1/chat/completions', {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({model: 'llama', messages: msgs, stream: true})});
    for (const rd = r.body.getReader(), dec = new TextDecoder();;) {
      const {done, value} = await rd.read();
      if (done) break;
      for (const ln of dec.decode(value).split('\\n'))
        if (ln.startsWith('data: ') && !ln.includes('[DONE]'))
          try { d.textContent += JSON.parse(ln.slice(6)).choices[0]?.delta?.content || '' } catch {}
      chat.scrollTop = chat.scrollHeight;
    }
    msgs.push({role: 'assistant', content: d.textContent});
  }
</script></body></html>'''

class Handler(HTTPRequestHandler):
  def log_request(self, code='-', size='-'): pass
  def do_GET(self): self.send_data(CHAT_HTML, content_type="text/html")
  def run_model(self, ids:list[int], model_name:str, include_usage=False):
    stderr_log(f"{self.path}  {colored('--', 'BLACK')}  in:{len(ids):5d}  {colored('--', 'BLACK')}  ")
    tmpl = {"id":f"chatcmpl-{uuid.uuid4().hex[:24]}", "object":"chat.completion.chunk", "created":int(time.time()), "model":model_name}
    yield {"choices": [{"index":0, "delta":{"role":"assistant","content":""}, "finish_reason":None}], **tmpl}
    out: list[int] = []
    st = time.perf_counter()
    for next_id in model.generate(ids):
      if len(out) == 0: stderr_log(f"prefill:{len(ids)/((pt:=time.perf_counter())-st):4.0f} tok/s  {colored('--', 'BLACK')}  ")
      if next_id == eos_id: break
      out.append(next_id)
      yield {"choices": [{"index":0, "delta":{"content":tok.decode([next_id])}, "finish_reason":None}], **tmpl}
    yield {"choices": [{"index":0, "delta":{},"finish_reason":"stop"}], **tmpl}
    if include_usage:
      yield {"choices": [], "usage": {"prompt_tokens": len(ids), "completion_tokens": len(out), "total_tokens": len(ids) + len(out)}, **tmpl}
    stderr_log(f"out:{len(out):5d}  {colored('--', 'BLACK')}  gen: {len(out)/(time.perf_counter()-pt):4.0f} tok/s\n")

  def do_POST(self):
    raw_body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
    body: dict[str, typing.Any] = json.loads(raw_body.decode("utf-8"))
    if DEBUG >= 1: print(json.dumps(body, indent=2))
    if self.path == "/v1/chat/completions":
      # extract tokens
      ids: list[int] = [bos_id] if bos_id is not None else []
      for msg in body["messages"]:
        ids += tok.role(msg["role"])
        # content can be a str or a list
        content = msg["content"]
        if isinstance(content, str): ids += tok.encode(content)
        elif isinstance(content, list):
          for c in content:
            if c["type"] == "text": ids += tok.encode(c["text"])
            else: raise RuntimeError(f"unhandled type: {c['type']}")
        else: raise RuntimeError(f"unknown content type: {type(content)}")
        ids += tok.end_turn(eos_id)
      ids += tok.assistant_prompt()

      # reply
      chunks = self.run_model(ids, body["model"], not body.get("stream") or body.get("stream_options",{}).get("include_usage", False))
      if body.get("stream"): self.stream_json(chunks)
      else:
        out = []
        for c in chunks: out.append(c["choices"][0]["delta"].get("content", "") if c["choices"] else "")
        self.send_data(json.dumps({**c, "object":"chat.completion",
          "choices":[{"index":0, "message":{"role":"assistant","content":"".join(out)}, "finish_reason":"stop"}]}).encode())
    else:
      raise RuntimeError(f"unhandled path {self.path}")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", "-m", choices=list(models.keys()), default=list(models.keys())[0], help="Model choice")
  parser.add_argument("--max_context", type=int, default=4096, help="Max Context Length")
  parser.add_argument("--serve", nargs='?', type=int, const=11434, metavar="PORT", help="Run OpenAI compatible API (optional port, default 11434)")
  parser.add_argument("--benchmark", nargs='?', type=int, const=20, metavar="COUNT", help="Benchmark tok/s (optional count, default 20)")
  args = parser.parse_args()

  # load the model
  raw_model = Tensor.from_url(models[args.model])
  model, kv = Transformer.from_gguf(raw_model, args.max_context)
  if DEBUG >= 1 or args.benchmark:
    print(f"using model {args.model} with {raw_model.nbytes():,} bytes and {sum(x.numel() for x in nn.state.get_parameters(model)):,} params")
  del raw_model

  # TODO: why this is required to free the RAM of the GGUF copy?
  import gc
  gc.collect()

  # do benchmark
  if args.benchmark:
    gen = model.generate([0], 0)
    for _ in range(args.benchmark):
      GlobalCounters.reset()
      with Timing(on_exit=lambda x: f", {1e9/x:6.2f} tok/s, {GlobalCounters.global_mem/x:7.2f} GB/s,"
                  f" {GlobalCounters.global_mem//1000000}/{GlobalCounters.mem_used//1000000} MB"): next(gen)
    exit(0)

  # extract some metadata
  tok = SimpleTokenizer.from_gguf_kv(kv)
  bos_id: int|None = kv.get('tokenizer.ggml.bos_token_id') if kv.get('tokenizer.ggml.add_bos_token', True) else None
  eos_id: int = kv['tokenizer.ggml.eos_token_id']

  # start server
  if args.serve: TCPServerWithReuse(('', args.serve), Handler).serve_forever()

  ids: list[int] = [bos_id] if bos_id is not None else []
  while 1:
    start_pos = max(len(ids) - 1, 0)
    try:
      ids += tok.role("user") + tok.encode(input('>>> ')) + tok.end_turn(eos_id) + tok.assistant_prompt()
    except EOFError:
      break
    for next_id in model.generate(ids, start_pos):
      sys.stdout.write(tok.decode([next_id]) if next_id != eos_id else "\n\n")
      sys.stdout.flush()
      if next_id == eos_id: break
