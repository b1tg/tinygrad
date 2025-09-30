from __future__ import annotations
import sys, argparse, typing, re, itertools, unicodedata
from tinygrad import Tensor, nn, UOp, TinyJit, getenv, helpers

def gpt2_decode_vocab(voc: dict[str, int]): # https://github.com/openai/gpt-2/blob/9b63575ef42771a015060c964af2c3da4cf7c8ab/src/encoder.py#L9
  c2b = { chr(cp): cp for cp in itertools.chain(range(ord("!"), ord("~")+1), range(ord("¡"), ord("¬")+1), range(ord("®"), ord("ÿ")+1)) }
  c2b.update({ chr(256+off): cp for off, cp in enumerate(cp for cp in range(256) if chr(cp) not in c2b) })
  return { bytes(c2b[c] for c in tok): tid for tok, tid in voc.items() }

def get_llama_re():
  def ucat_range(pre: str): return "".join(re.escape(chr(cp)) for cp in range(sys.maxunicode + 1) if unicodedata.category(chr(cp)).startswith(pre))
  r_ws, r_p_N, r_p_L = r"\t\n\x0b\x0c\r\x85" + ucat_range("Z"), ucat_range("N"), ucat_range("L")
  # https://github.com/ggml-org/llama.cpp/blob/94933c8c2eeaa9a7983e3f6c08af76bd86724094/src/llama-vocab.cpp#L286
  return "(?i:'s|'t|'re|'ve|'m|'ll|'d)|" + \
    f"[^\\r\\n{r_p_N}{r_p_L}]?[{r_p_L}]+|[{r_p_N}]{{1,3}}| ?[^{r_ws}{r_p_N}{r_p_L}]+[\\r\\n]*|[{r_ws}]*[\\r\\n]+|[{r_ws}]+(?![^{r_ws}])|[{r_ws}]+"

class SimpleTokenizer:
  def __init__(self, pat: str, normal_tokens: dict[bytes, int], special_tokens: dict[str, int]):
    self._normal_tokens, self._special_tokens, self._pat = normal_tokens, special_tokens, re.compile(pat)
    self._tok2str = { tid: tok.encode() for tok, tid in special_tokens.items() } | { tid: tok for tok, tid in normal_tokens.items()  }
    self._special_re = re.compile("|".join(re.escape(tok) for tok in self._special_tokens.keys()) if special_tokens else r"(?!)")

  @staticmethod
  def from_gguf_kv(kv: dict):
    # Accept any preset; default to LLaMA-style regex segmentation
    vocab: typing.Iterable[tuple[str, int]] = ((tok, idx) for idx, tok in enumerate(kv["tokenizer.ggml.tokens"]))
    normal_tokens, special_tokens = helpers.partition(vocab, lambda e: kv["tokenizer.ggml.token_type"][e[1]] == 1)
    return SimpleTokenizer(get_llama_re(), gpt2_decode_vocab(dict(normal_tokens)), dict(special_tokens))

  def encode(self, text: str):
    tokens: list[int] = []
    pos = 0
    for match in self._special_re.finditer(text):
      tokens.extend(self._encode_sentence(text[pos:match.start(0)]) + [self._special_tokens[text[match.start(0):match.end(0)]]])
      pos = match.end(0)
    return tokens + self._encode_sentence(text[pos:])

  def decode(self, ids: list[int]) -> str: return b''.join(self._tok2str[tid] for tid in ids).decode()
  def role(self, role:str): return self.encode("<|start_header_id|>" + role + "<|end_header_id|>\n\n")

  def _encode_sentence(self, chunk: str): return [ tok for word in self._pat.findall(chunk) for tok in self._encode_word(word.encode()) ]
  def _encode_word(self, word: bytes):
    if (early_token:=self._normal_tokens.get(word)) is not None: return [early_token]
    parts = [word[i:i+1] for i in range(len(word))]
    while True:
      min_tid, min_idx = 2**32, -1
      for idx, (p1, p2) in enumerate(zip(parts[:-1], parts[1:])):
        tid = self._normal_tokens.get(p1 + p2, min_tid)
        if tid < min_tid: min_tid, min_idx = tid, idx
      if min_idx == -1: break
      parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx+1]] + parts[min_idx+2:]
    try: return [ self._normal_tokens[p] for p in parts ]
    except KeyError: raise RuntimeError("token not found")

def _yarn_concentration_and_inv_freq(head_dim:int, base:float, initial_context_length:int, scaling_factor:float, ntk_alpha:float, ntk_beta:float, device):
  freq = (base ** (Tensor.arange(0, head_dim, 2, dtype="float32", device=device) / head_dim))
  if scaling_factor > 1.0:
    concentration = 0.1 * Tensor([scaling_factor], dtype="float32", device=device).log().item() + 1.0
    d_half = head_dim / 2
    low = d_half * Tensor([initial_context_length], dtype="float32", device=device).log() / Tensor([base], dtype="float32", device=device).log() - d_half * Tensor([(ntk_beta * 2 * 3.141592653589793)], dtype="float32", device=device).log() / Tensor([base], dtype="float32", device=device).log()
    high = d_half * Tensor([initial_context_length], dtype="float32", device=device).log() / Tensor([base], dtype="float32", device=device).log() - d_half * Tensor([(ntk_alpha * 2 * 3.141592653589793)], dtype="float32", device=device).log() / Tensor([base], dtype="float32", device=device).log()
    # construct ramp
    ramp = (Tensor.arange(d_half, dtype="float32", device=device) - low.item()) / (high.item() - low.item())
    mask = 1 - ramp.clamp(0, 1)
    interpolation = 1.0 / (scaling_factor * freq)
    extrapolation = 1.0 / freq
    inv_freq = interpolation * (1 - mask) + extrapolation * mask
  else:
    concentration = 1.0
    inv_freq = 1.0 / freq
  return concentration, inv_freq

def apply_rope(x:Tensor, start_pos:int|UOp, *, head_dim:int, base:float, initial_context_length:int, scaling_factor:float, ntk_alpha:float, ntk_beta:float) -> Tensor:
  B, H, T, Hd = x.shape
  assert Hd == head_dim, "head_dim mismatch"
  assert (Hd & 1) == 0, "RoPE requires an even head dimension"
  half = Hd // 2
  concentration, inv_freq = _yarn_concentration_and_inv_freq(half*2, base, initial_context_length, scaling_factor, ntk_alpha, ntk_beta, x.device)
  t = (Tensor.arange(T, dtype="float32", device=x.device) + start_pos)[:, None]
  freqs = t.matmul(inv_freq[None, :])
  cos = (freqs.cos() * concentration).reshape(1, 1, T, half).cast(x.dtype)
  sin = (freqs.sin() * concentration).reshape(1, 1, T, half).cast(x.dtype)
  x_pairs = x.reshape(B, H, T, half, 2)
  return Tensor.stack(x_pairs[..., 0] * cos - x_pairs[..., 1] * sin,
                      x_pairs[..., 0] * sin + x_pairs[..., 1] * cos, dim=-1).reshape(B, H, T, Hd)

class TransformerBlock:
  def __init__(self, layer_idx: int, dim:int, hidden_dim:int, n_heads:int, n_kv_heads:int, head_dim:int, norm_eps:float, max_context:int=0,
               rope_base:float=150000.0, rope_scaling_factor:float=32.0, rope_ntk_alpha:float=1.0, rope_ntk_beta:float=32.0,
               initial_context_length:int=4096, low_mem_moe:bool=True, num_experts:int=32):
    self.n_heads      = n_heads
    self.n_kv_heads   = n_kv_heads
    self.head_dim     = head_dim
    self.max_context  = max_context
    self.layer_idx = layer_idx
    self.rope_base = rope_base
    self.rope_scaling_factor = rope_scaling_factor
    self.rope_ntk_alpha = rope_ntk_alpha
    self.rope_ntk_beta = rope_ntk_beta
    self.initial_context_length = initial_context_length

    # --- attention projections (match GGUF shapes) ----------------------
    kv_proj_out      = self.head_dim * n_kv_heads
    self.attn_q      = nn.Linear(dim, self.head_dim * n_heads,    bias=True)
    self.attn_k      = nn.Linear(dim, kv_proj_out,                 bias=True)
    self.attn_v      = nn.Linear(dim, kv_proj_out,                 bias=True)
    self.attn_output = nn.Linear(self.head_dim * n_heads, dim,     bias=True)

    # sinks parameter per head (matches checkpoint name attn_sinks.weight)
    class _Param:
      def __init__(self, shape, dtype, device): self.weight = Tensor.zeros(*shape, dtype=dtype, device=device)
    self.attn_sinks = _Param((n_heads,), "float16", None)

    # --- RMSNorms (names match GGUF) -----------------------------------
    self.attn_norm            = nn.RMSNorm(dim, norm_eps)
    self.post_attention_norm  = nn.RMSNorm(dim, norm_eps)

    # --- MoE feed-forward (E=32 for 20B, E=128 for 120B) for gpt-oss ---------------------------------
    self.num_experts = num_experts
    self.ffn_gate_inp = nn.Linear(dim, self.num_experts, bias=True)
    class _MoEParam:
      def __init__(self, out:int, inp:int, num_experts:int=32, dtype:str="float16", device=None):
        self.weight = Tensor.zeros(num_experts, out, inp, dtype=dtype, device=device)
        self.bias   = Tensor.zeros(num_experts, out,      dtype=dtype, device=device)
    # weights stored as (E, out, in) in GGUF
    # place MoE expert banks on CPU if low_mem_moe
    moe_dev = "CPU" if low_mem_moe else None
    self.ffn_gate_exps = _MoEParam(dim, dim, self.num_experts, device=moe_dev)
    self.ffn_up_exps   = _MoEParam(dim, dim, self.num_experts, device=moe_dev)
    self.ffn_down_exps = _MoEParam(dim, dim, self.num_experts, device=moe_dev)
    self.low_mem_moe = low_mem_moe

  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    x_norm = self.attn_norm(x)                       # (B,T,D)
    q, k, v = self.attn_q(x_norm), self.attn_k(x_norm), self.attn_v(x_norm)
    sliding_window = 128 if self.layer_idx % 2 == 0 else 0

    B, T, _ = x.shape
    q = q.reshape(B, T, self.n_heads,    self.head_dim).transpose(1, 2)  # (B,H,T,Hd)
    k = k.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)  # (B,KvH,T,Hd)
    v = v.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)  # (B,KvH,T,Hd)

    q = apply_rope(q, start_pos, head_dim=self.head_dim, base=self.rope_base, initial_context_length=self.initial_context_length,
                   scaling_factor=self.rope_scaling_factor, ntk_alpha=self.rope_ntk_alpha, ntk_beta=self.rope_ntk_beta)
    k = apply_rope(k, start_pos, head_dim=self.head_dim, base=self.rope_base, initial_context_length=self.initial_context_length,
                   scaling_factor=self.rope_scaling_factor, ntk_alpha=self.rope_ntk_alpha, ntk_beta=self.rope_ntk_beta)

    # TODO: remove these kv cache realizes
    if not hasattr(self, "cache_kv"):
      self.cache_kv = Tensor.zeros(2, B, self.n_kv_heads, self.max_context, self.head_dim, dtype=k.dtype, device=k.device).contiguous().realize()
    self.cache_kv[:, :, :, start_pos:start_pos+T, :].assign(Tensor.stack(k, v)).realize()  # type: ignore
    k = self.cache_kv[0, :, :, 0:start_pos+T, :]
    v = self.cache_kv[1, :, :, 0:start_pos+T, :]

    # Manual attention with GQA + sinks bias
    # Repeat kv to full heads
    rep = self.n_heads // self.n_kv_heads
    k_full = k.repeat_interleave(rep, dim=1)   # (B,H,Tk,Hd)
    v_full = v.repeat_interleave(rep, dim=1)   # (B,H,Tk,Hd)
    # Build causal/sliding mask: (1,1,T,Tk)
    if T > 1 or sliding_window > 0:
      q_idx = (Tensor.arange(T, dtype="int32", device=x.device) + start_pos).reshape(1, 1, T, 1)
      k_idx = Tensor.arange(start_pos+T, dtype="int32", device=x.device).reshape(1, 1, 1, start_pos+T)
      causal_ok = k_idx <= q_idx
      if sliding_window > 0:
        window_ok = k_idx >= (q_idx - (sliding_window - 1))
        ok = causal_ok & window_ok
      else:
        ok = causal_ok
      mask = ok.where(0, float("-inf")).cast(x.dtype)
    else:
      mask = None
    # logits = (B,H,T,Tk)
    logits = q.matmul(k_full.transpose(-2, -1)) / (self.head_dim ** 0.5)
    if mask is not None: logits = logits + mask
    # Handle sinks bias - add it to logits and renormalize  
    # Instead of concatenating and slicing, incorporate sinks into the attention computation
    S = self.attn_sinks.weight.reshape(1, self.n_heads, 1, 1).cast(logits.dtype).expand(B, self.n_heads, T, 1)
    
    # Method: Add sinks to the logits, compute softmax, then ignore sinks in output
    # This matches the mathematical effect of the torch implementation
    logits_max = logits.max(-1, keepdim=True)
    S_adj = S + logits_max  # Adjust sinks to prevent numerical issues
    
    # Compute attention weights with sinks contribution
    logits_shifted = logits - logits_max
    S_shifted = S_adj - logits_max
    
    exp_logits = logits_shifted.exp()
    exp_sinks = S_shifted.exp()
    
    # Denominator includes sinks contribution
    denom = exp_logits.sum(-1, keepdim=True) + exp_sinks.sum(-1, keepdim=True)
    
    # But only apply attention to sequence tokens (ignore sinks for output)
    W = exp_logits / denom
    attn = W.matmul(v_full)                       # (B,H,T,Hd)
    attn = attn.transpose(1, 2).reshape(B, T, -1)                                    # back to (B,T,D)
    attn = self.attn_output(attn)
    return x + attn

  def _feed_forward(self, h: Tensor) -> Tensor:
    # MoE feed-forward following gpt-oss torch implementation more closely
    x = self.post_attention_norm(h)                    # (B,T,D)
    B, T, D = x.shape
    
    # Router: top-4 experts per token
    g = self.ffn_gate_inp(x)                           # (B,T,E)
    topk_vals, topk_idx = g.topk(4, dim=-1)           # (B,T,4)
    expert_weights = topk_vals.softmax(-1)             # (B,T,4)
    expert_indices = topk_idx                          # (B,T,4)
    
    # Move expert weights to device and get expert-specific weights
    gate_weights = self.ffn_gate_exps.weight.to(x.device)    # (E, D, D)
    gate_bias = self.ffn_gate_exps.bias.to(x.device)         # (E, D)
    up_weights = self.ffn_up_exps.weight.to(x.device)        # (E, D, D) 
    up_bias = self.ffn_up_exps.bias.to(x.device)             # (E, D)
    down_weights = self.ffn_down_exps.weight.to(x.device)    # (E, D, D)
    down_bias = self.ffn_down_exps.bias.to(x.device)         # (E, D)
    
    # Select weights for chosen experts: (B,T,4) -> (B,T,4,D,D)
    selected_gate_w = gate_weights[expert_indices]     # (B,T,4,D,D)
    selected_gate_b = gate_bias[expert_indices]        # (B,T,4,D)
    selected_up_w = up_weights[expert_indices]         # (B,T,4,D,D)
    selected_up_b = up_bias[expert_indices]            # (B,T,4,D)
    selected_down_w = down_weights[expert_indices]     # (B,T,4,D,D)
    selected_down_b = down_bias[expert_indices]        # (B,T,4,D)
    
    # Expand input for expert computation: (B,T,D) -> (B,T,4,D)
    x_expanded = x.unsqueeze(2).expand(B, T, 4, D)
    
    # Gate and Up projections: einsum equivalent to "btd,btdh->bth"
    gate_out = (x_expanded.unsqueeze(-1) * selected_gate_w.transpose(-1, -2)).sum(-2) + selected_gate_b  # (B,T,4,D)
    up_out = (x_expanded.unsqueeze(-1) * selected_up_w.transpose(-1, -2)).sum(-2) + selected_up_b        # (B,T,4,D)
    
    # SwiGLU activation with clamping
    limit = 7.0
    alpha = 1.702
    x_glu = gate_out.clamp(max_=limit)
    x_linear = up_out.clamp(min_=-limit, max_=limit)
    h_moe = (alpha * x_glu).sigmoid() * x_glu * (x_linear + 1)  # (B,T,4,D)
    
    # Down projection
    down_out = (h_moe.unsqueeze(-1) * selected_down_w.transpose(-1, -2)).sum(-2) + selected_down_b  # (B,T,4,D)
    
    # Weighted sum of experts: (B,T,4,D) * (B,T,4,1) -> (B,T,D)
    output = (expert_weights.unsqueeze(-1) * down_out).sum(-2)  # (B,T,D)
    
    return h + output

  def __call__(self, x: Tensor, start_pos: int|UOp):
    return self._feed_forward(self._attention(x, start_pos))

class Transformer:
  def __init__(self, *, num_blocks, dim, hidden_dim, n_heads, n_kv_heads, head_dim, norm_eps, vocab_size, max_context,
               rope_base:float=150000.0, rope_scaling_factor:float=32.0, rope_ntk_alpha:float=1.0, rope_ntk_beta:float=32.0,
               initial_context_length:int=4096, num_experts:int=32):
    self.blk = [TransformerBlock(i, dim, hidden_dim, n_heads, n_kv_heads, head_dim, norm_eps, max_context,
                                 rope_base, rope_scaling_factor, rope_ntk_alpha, rope_ntk_beta, initial_context_length, num_experts=num_experts) for i in range(num_blocks)]
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
    return self.output(self.output_norm(x))[:, -1, :].softmax(-1).argmax(-1, keepdim=True)

  def __call__(self, tokens:Tensor, start_pos:int|UOp=0) -> Tensor:
    return (self.forward_jit if getenv("JIT", 1) and tokens.shape[1] == 1 and isinstance(start_pos, UOp) else self.forward)(tokens, start_pos)

  @staticmethod
  def from_gguf(gguf:Tensor, max_context:int|None=None) -> tuple[Transformer, dict]:
    # TODO: remove the need for copy to default device
    kv, state_dict = nn.state.gguf_load(gguf.to(None))

    # all state items should be float16, not float32
    state_dict = {k:v.cast('float16') for k,v in state_dict.items()}

    # some models like Llama 3.2 don't have an output.weight, they just tie to the token_embd.weight
    if 'output.weight' not in state_dict: state_dict['output.weight'] = state_dict['token_embd.weight']

    arch = kv['general.architecture']
    max_context = min(max_context, kv[f'{arch}.context_length']) if max_context is not None else kv[f'{arch}.context_length']
    rope_base = kv.get(f'{arch}.rope.freq_base', 150000.0)
    # GGUF uses rope.scaling.* with type yarn
    rope_scaling_factor = kv.get(f'{arch}.rope.scaling.factor', 32.0)
    # For YaRN we approximate alpha/beta as defaults if not present
    rope_ntk_alpha = kv.get(f'{arch}.rope.yarn_ntk_alpha', 1.0)
    rope_ntk_beta = kv.get(f'{arch}.rope.yarn_ntk_beta', 32.0)
    initial_context_length = kv.get(f'{arch}.rope.scaling.original_context_length', kv.get(f'{arch}.context_length', 4096))
    # infer head_dim from attn_q.weight rows: rows = n_heads * head_dim
    aq = state_dict.get('blk.0.attn_q.weight')
    n_heads = kv[f'{arch}.attention.head_count']
    n_kv_heads = kv[f'{arch}.attention.head_count_kv']
    head_dim = (aq.shape[0] // n_heads) if aq is not None else (kv[f'{arch}.embedding_length'] // n_heads)
    
    # infer number of experts from ffn_gate_inp.weight shape
    gate_inp = state_dict.get('blk.0.ffn_gate_inp.weight')
    num_experts = gate_inp.shape[0] if gate_inp is not None else 32
    
    model = Transformer(num_blocks=kv[f'{arch}.block_count'], dim=kv[f'{arch}.embedding_length'], hidden_dim=kv.get(f'{arch}.feed_forward_length', kv[f'{arch}.embedding_length']),
                        n_heads=n_heads, n_kv_heads=n_kv_heads, head_dim=head_dim,
                        norm_eps=kv.get(f'{arch}.attention.layer_norm_rms_epsilon', 1e-5), vocab_size=len(kv['tokenizer.ggml.tokens']), max_context=max_context,
                        rope_base=rope_base, rope_scaling_factor=rope_scaling_factor, rope_ntk_alpha=rope_ntk_alpha, rope_ntk_beta=rope_ntk_beta,
                        initial_context_length=initial_context_length, num_experts=num_experts)
    import json
    with open("gpt_oss.json", "w") as f:
      json.dump(list(state_dict.keys()), f)
    # Adjust 2D weight orientation if needed (GGUF is (out,in) already; tinygrad Linear expects (in,out))
    model_sd = nn.state.get_state_dict(model)
    adjusted = {}
    for k, v in state_dict.items():
      if k in model_sd and v.ndim == 2 and v.shape != model_sd[k].shape and v.shape[::-1] == model_sd[k].shape:
        adjusted[k] = v.T
      else:
        adjusted[k] = v
    nn.state.load_state_dict(model, adjusted, verbose=False, consume=True, realize=False)
    return model, kv

  def generate(self, tokens:list[int], start_pos=0):
    v_start_pos = UOp.variable("start_pos", 1, self.max_context-1)
    start_pos = 0
    t = Tensor([tokens[start_pos:]], dtype="int32")
    self.forward_jit.reset()  # TODO: why is this required? root cause the issue and make it not be needed
    while len(tokens) < self.max_context:
      t = self(t, v_start_pos.bind(start_pos) if getenv("SYM", 1) and start_pos != 0 and t.shape[-1] == 1 else start_pos)
      next_id = int(t.item())
      tokens.append(next_id)
      start_pos = len(tokens) - 1
      yield next_id

models = {
  # "1B": "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q6_K.gguf",
  # "1B": "/home/amax/llama/openai_gpt-oss-20b-MXFP4.gguf",
    "1B": "https://huggingface.co/ggml-org/gpt-oss-20b-GGUF/resolve/main/gpt-oss-20b-mxfp4.gguf?download=true",
  "3B": "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q6_K.gguf",
  "3B_f16": "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-f16.gguf",
  "8B": "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
}

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--size", choices=list(models.keys()), default=list(models.keys())[0], help="Model size")
  parser.add_argument("--max_context", type=int, default=4096, help="Max Context Length")
  parser.add_argument("--debug_tokens", action="store_true", help="Print token ids and decoded repr while generating")
  args = parser.parse_args()

  # load the model
  model, kv = Transformer.from_gguf(Tensor.from_url(models[args.size]), args.max_context)

  # prefer Harmony o200k tokenizer for GPT-OSS
  use_harmony = False
  try:
    from openai_harmony import Conversation, Message, Role, StreamableParser, StreamState, load_harmony_encoding, HarmonyEncodingName
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    use_harmony = True
  except Exception:
    pass

  if use_harmony:
    # while 1:
    if 1:
      try:
        # usr = input('>>> ')
        usr = "test"
      except EOFError:
        pass
        # break
      conv = Conversation.from_messages([Message.from_role_and_content(Role.USER, usr)])
      ids: list[int] = encoding.render_conversation_for_completion(conv, Role.ASSISTANT)
      start_pos = len(ids) - 1
      stop_tokens = set(encoding.stop_tokens_for_assistant_actions())
      parser = StreamableParser(encoding, role=Role.ASSISTANT)
      field_created = False
      current_output_text = ""
      output_text_delta_buffer = ""
      for next_id in model.generate(ids, start_pos):
        if next_id in stop_tokens:
          sys.stdout.write("\n\n")
          sys.stdout.flush()
          break
        if args.debug_tokens:
          sys.stdout.write(f"[id={next_id} repr={repr(encoding.decode([next_id]))}]\n")
          sys.stdout.flush()
        parser.process(next_id)
        if parser.state == StreamState.EXPECT_START and field_created:
          sys.stdout.write("\n")
          field_created = False
        if not parser.last_content_delta:
          continue
        if not field_created:
          field_created = True
        output_text_delta_buffer += parser.last_content_delta
        sys.stdout.write(output_text_delta_buffer)
        sys.stdout.flush()
        current_output_text += output_text_delta_buffer
        output_text_delta_buffer = ""
  else:
    1/0
    # fallback to GGUF tokenizer (may produce suboptimal spacing)
    tok = SimpleTokenizer.from_gguf_kv(kv)
    bos_id: int = kv['tokenizer.ggml.bos_token_id']
    eos_id: int = kv['tokenizer.ggml.eos_token_id']
    ids: list[int] = []
    while 1:
      start_pos = len(ids) - 1
      try:
        usr = input('>>> ')
        ids = [bos_id] + tok.encode(usr)
      except EOFError:
        break
      for next_id in model.generate(ids, start_pos):
        if next_id == eos_id:
          sys.stdout.write("\n\n")
          sys.stdout.flush()
          break
        if args.debug_tokens:
          sys.stdout.write(f"[id={next_id} repr={repr(tok.decode([next_id]))}]\n")
        else:
          sys.stdout.write(tok.decode([next_id]))
        sys.stdout.flush()
      ids = []
print("over")