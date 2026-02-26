from __future__ import annotations
import sys, argparse, typing, re, unicodedata, json, uuid, time, functools, numpy as np
from tinygrad import Tensor, nn, UOp, TinyJit, getenv, function
from tinygrad.helpers import partition, DEBUG, Timing, GlobalCounters, stderr_log, colored
from tinygrad.viz.serve import TCPServerWithReuse, HTTPRequestHandler

class SimpleTokenizer:
  def __init__(self, normal_tokens:dict[str, int], special_tokens:dict[str, int], preset:str="llama3"):
    if preset not in ("llama3","llama-v3","llama-bpe","qwen2","olmo"): raise ValueError(f"Invalid tokenizer preset '{preset}'")
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
    if self.preset == 'qwen2': return self.encode("<|im_start|>" + role + "\n")
    return self.encode("<|start_header_id|>" + role + "<|end_header_id|>\n\n")
  def end_turn(self, eos_id:int):
    if self.preset == 'olmo': return self.encode("\n")
    if self.preset == 'qwen2': return [eos_id] + self.encode("\n")
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

    self.cache_kv[:, :, :, start_pos:start_pos+T, :].assign(Tensor.stack(k, v))
    k = self.cache_kv[0, :, :, 0:start_pos+T, :]
    v = self.cache_kv[1, :, :, 0:start_pos+T, :]

    # NOTE: this mask is causal_lower_right, not the causal_upper_left generated by is_casual = True
    mask = Tensor.full((1, 1, T, start_pos+T), float("-inf"), dtype=x.dtype, device=x.device).triu(int(start_pos)+1) if T > 1 else None
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

class Transformer:
  def __init__(self, *, num_blocks, dim, hidden_dim, n_heads, n_kv_heads, norm_eps, vocab_size, head_dim:int, rope_theta:float,
               max_context:int=0, qk_norm:int=0, num_experts:int=0, num_experts_per_tok:int=0):
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
    return (self.forward_jit if getenv("JIT", 1) and tokens.shape[1] == 1 and isinstance(start_pos, UOp) else self.forward)(tokens, start_pos)

  @staticmethod
  def from_gguf(gguf:Tensor, max_context:int|None=None, realize=True) -> tuple[Transformer, dict]:
    # TODO: remove the need for copy to default device
    kv, state_dict = nn.state.gguf_load(gguf.to(None))

    # all state items should be float16, not float32
    state_dict = {k:v.cast('float16') if getenv("HALF", 1) else v for k,v in state_dict.items()}

    # some models like Llama 3.2 don't have an output.weight, they just tie to the token_embd.weight
    if 'output.weight' not in state_dict: state_dict['output.weight'] = state_dict['token_embd.weight']

    arch = kv['general.architecture']
    max_context = min(max_context, kv[f'{arch}.context_length']) if max_context is not None else kv[f'{arch}.context_length']
    n_heads, n_kv_heads = kv[f'{arch}.attention.head_count'], kv[f'{arch}.attention.head_count_kv']

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
    for s in (params:=nn.state.get_parameters(model)): s.replace(s.contiguous())
    if realize: Tensor.realize(*params)
    return model, kv

  def generate(self, tokens:list[int], start_pos=0):
    v_start_pos = UOp.variable("start_pos", 1, self.max_context-1)
    t = Tensor([tokens[start_pos:]], dtype="int32")
    while len(tokens) < self.max_context:
      t = self(t, v_start_pos.bind(start_pos) if getenv("SYM", 1) and start_pos != 0 and t.shape[-1] == 1 else start_pos)
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
  "qwen3:1.7b": "https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q4_K_M.gguf",
  "qwen3:8b": "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf",
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
  input.onkeydown = (e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() } }
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
      ids += tok.role("assistant")

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

# *** abliteration: remove safety refusal behavior from GGUF models ***
# Usage:
#   python -m tinygrad.apps.llm --model llama3.2:1b --abliterate \
#     --harmful-path harmful.jsonl --harmless-path harmless.jsonl
#
# JSONL files should have one JSON object per line with a "text" field containing a prompt.
# Options:
#   --abliterate-num-prompts N   number of prompt pairs to use (default: 64, lower = faster)
#   --abliterate-strength F      projection strength (default: 2.5)
#   --abliterate-threshold F     min explained variance to modify a layer (default: 0.25)
#   --abliterate-skip-layers N   skip first/last N layers (default: 2)
#
# Use DEBUG=1 to see per-layer explained variance ratios.

def load_prompts(path: str) -> list[str]:
  prompts: list[str] = []
  with open(path, "r", encoding="utf-8") as f:
    for line in f:
      if not line.strip(): continue
      item = json.loads(line)
      prompt = item.get("text", item.get("prompt", ""))
      if prompt: prompts.append(prompt)
  return prompts

def capture_activations(model: Transformer, tok: SimpleTokenizer, prompts: list[str], bos_id: int|None, eos_id: int,
                         num_prompts: int) -> dict[str, list[Tensor]]:
  activations: dict[str, list[Tensor]] = {}
  for prompt in prompts[:num_prompts]:
    tokens = ([bos_id] if bos_id is not None else []) + tok.role("user") + tok.encode(prompt) + tok.end_turn(eos_id) + tok.role("assistant")
    x = model.token_embd(Tensor([tokens], dtype="int32")).cast("float32")
    T = x.shape[1]
    for i, blk in enumerate(model.blk):
      # --- attention (no KV cache, plain SDPA) ---
      x_norm = blk.attn_norm(x)
      q, k, v = blk.attn_q(x_norm), blk.attn_k(x_norm), blk.attn_v(x_norm)
      if blk.qk_norm and blk.qk_norm != blk.head_dim: q, k = blk.attn_q_norm(q), blk.attn_k_norm(k)
      q = q.reshape(1, T, blk.n_heads, blk.head_dim).transpose(1, 2)
      k = k.reshape(1, T, blk.n_kv_heads, blk.head_dim).transpose(1, 2)
      v = v.reshape(1, T, blk.n_kv_heads, blk.head_dim).transpose(1, 2)
      if blk.qk_norm == blk.head_dim: q, k = blk.attn_q_norm(q), blk.attn_k_norm(k)
      freqs_cis = precompute_freqs_cis(blk.head_dim, blk.max_context, blk.rope_theta)[:T]
      q, k = apply_rope(q, freqs_cis), apply_rope(k, freqs_cis)
      mask = Tensor.full((1, 1, T, T), float("-inf"), dtype=x.dtype).triu(1) if T > 1 else None
      attn = q.scaled_dot_product_attention(k, v, attn_mask=mask, enable_gqa=True)
      attn = attn.transpose(1, 2).reshape(1, T, -1)
      attn_out = blk.attn_output(attn)
      activations.setdefault(f"{i}.attn", []).append(attn_out[0, -1].realize())
      x = x + attn_out

      # --- feed-forward ---
      h_norm = blk.ffn_norm(x)
      if hasattr(blk, 'ffn_gate_exps'):
        # MoE path
        x_exp = h_norm.unsqueeze(2)
        probs, sel = blk.ffn_gate_inp(h_norm).softmax(-1).topk(blk.num_experts_per_tok)
        x_down = blk.ffn_down_exps(sel, blk.ffn_gate_exps(sel, x_exp).silu() * blk.ffn_up_exps(sel, x_exp))
        ffn_out = (x_down * probs.unsqueeze(-1)).sum(axis=2)
      else:
        gated = blk.ffn_gate(h_norm).silu().contiguous() * blk.ffn_up(h_norm)
        ffn_out = blk.ffn_down(gated)
        activations.setdefault(f"{i}.ffn", []).append(ffn_out[0, -1].realize())
      x = (x + ffn_out).contiguous()
  return activations

def pca_direction(harmless_acts: list[Tensor], harmful_acts: list[Tensor]) -> tuple[Tensor, float]:
  # move to numpy for SVD — tinygrad's SVD hits codegen bugs on small matrices
  harmless_np = np.stack([a.numpy() for a in harmless_acts])  # (N1, D)
  harmful_np = np.stack([a.numpy() for a in harmful_acts])    # (N2, D)
  all_np = np.concatenate([harmless_np, harmful_np], axis=0)
  global_mean = all_np.mean(axis=0)
  M = all_np - global_mean
  _, S, Vt = np.linalg.svd(M, full_matrices=False)
  direction = Vt[0].astype(np.float32)
  explained_variance = float((S[0] ** 2) / (S ** 2).sum())
  # orient so it points from harmless to harmful
  mean_diff = harmful_np.mean(axis=0) - harmless_np.mean(axis=0)
  if np.dot(direction, mean_diff) < 0: direction = -direction
  # orthogonalize against harmless mean direction
  hm = harmless_np.mean(axis=0)
  hm_norm = np.linalg.norm(hm)
  if hm_norm > 0:
    hm_dir = hm / hm_norm
    direction = direction - np.dot(direction, hm_dir) * hm_dir
  return Tensor(direction), explained_variance

def abliterate_weight(weight: Tensor, direction: Tensor, strength: float) -> Tensor:
  eps = 1e-12
  orig_dtype = weight.dtype
  w = weight.cast("float32")
  d = (direction / (direction * direction).sum().sqrt().maximum(eps)).cast("float32")  # unit direction, shape (out,)
  row_norms = (w * w).sum(axis=-1).sqrt()  # (out,)
  w_dirs = w / row_norms.maximum(eps).unsqueeze(-1)  # unit direction per row, (out, in)
  # norm-preserving biprojected abliteration: project out refusal direction from output space
  proj = d @ w_dirs  # (in,)
  w_dirs = w_dirs - strength * d.unsqueeze(-1) * proj.unsqueeze(0)  # subtract outer(d, proj)
  updated_norms = (w_dirs * w_dirs).sum(axis=-1).sqrt().maximum(eps)
  result = (w_dirs / updated_norms.unsqueeze(-1)) * row_norms.unsqueeze(-1)
  return result.cast(orig_dtype)

def abliterate_model(model: Transformer, tok: SimpleTokenizer, bos_id: int|None, eos_id: int, args):
  harmful_prompts = load_prompts(args.harmful_path)
  harmless_prompts = load_prompts(args.harmless_path)
  print(f"Loaded {len(harmful_prompts)} harmful and {len(harmless_prompts)} harmless prompts")

  print("Collecting activations...")
  harmful_acts = capture_activations(model, tok, harmful_prompts, bos_id, eos_id, args.abliterate_num_prompts)
  harmless_acts = capture_activations(model, tok, harmless_prompts, bos_id, eos_id, args.abliterate_num_prompts)

  print("Applying abliteration...")
  num_layers = len(model.blk)
  skip = args.abliterate_skip_layers
  updates = 0
  for i in range(skip, num_layers - skip):
    # attention
    key = f"{i}.attn"
    if key in harmful_acts and key in harmless_acts:
      direction, evr = pca_direction(harmless_acts[key], harmful_acts[key])
      if evr > args.abliterate_threshold:
        model.blk[i].attn_output.weight.replace(abliterate_weight(model.blk[i].attn_output.weight, direction, args.abliterate_strength))
        updates += 1
        if DEBUG >= 1: print(f"  layer {i} attn: evr={evr:.4f}")
    # ffn (skip MoE)
    key = f"{i}.ffn"
    if key in harmful_acts and key in harmless_acts:
      direction, evr = pca_direction(harmless_acts[key], harmful_acts[key])
      if evr > args.abliterate_threshold:
        model.blk[i].ffn_down.weight.replace(abliterate_weight(model.blk[i].ffn_down.weight, direction, args.abliterate_strength))
        updates += 1
        if DEBUG >= 1: print(f"  layer {i} ffn:  evr={evr:.4f}")

  # realize all modified parameters
  Tensor.realize(*nn.state.get_parameters(model))
  print(f"Abliteration complete: {updates} weight matrices modified across layers {skip}-{num_layers - skip - 1}")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", choices=list(models.keys()), default=list(models.keys())[0], help="Model choice")
  parser.add_argument("--max_context", type=int, default=4096, help="Max Context Length")
  parser.add_argument("--serve", nargs='?', type=int, const=11434, metavar="PORT", help="Run OpenAI compatible API (optional port, default 11434)")
  parser.add_argument("--benchmark", nargs='?', type=int, const=20, metavar="COUNT", help="Benchmark tok/s (optional count, default 20)")
  parser.add_argument("--abliterate", action="store_true", help="Enable abliteration to remove refusal behavior")
  parser.add_argument("--harmful-path", default="harmful.jsonl", help="Path to harmful prompts JSONL")
  parser.add_argument("--harmless-path", default="harmless.jsonl", help="Path to harmless prompts JSONL")
  parser.add_argument("--abliterate-strength", type=float, default=2.5, help="Abliteration strength")
  parser.add_argument("--abliterate-threshold", type=float, default=0.25, help="Explained variance threshold")
  parser.add_argument("--abliterate-skip-layers", type=int, default=2, help="Layers to skip at start/end")
  parser.add_argument("--abliterate-num-prompts", type=int, default=64, help="Number of prompt pairs to use")
  args = parser.parse_args()

  # load the model
  model, kv = Transformer.from_gguf(Tensor.from_url(models[args.model]), args.max_context)
  if DEBUG >= 1: print(f"using model {args.model}")

  # extract some metadata
  tok = SimpleTokenizer.from_gguf_kv(kv)
  bos_id: int|None = kv.get('tokenizer.ggml.bos_token_id') if kv.get('tokenizer.ggml.add_bos_token', True) else None
  eos_id: int = kv['tokenizer.ggml.eos_token_id']

  # abliterate if requested
  if args.abliterate: abliterate_model(model, tok, bos_id, eos_id, args)

  # do benchmark
  if args.benchmark:
    param_bytes = sum(x.nbytes() for x in nn.state.get_parameters(model))
    for b in model.blk:
      if hasattr(b, 'ffn_gate_exps'):
        expert_bytes = b.ffn_gate_exps.weight.nbytes() + b.ffn_up_exps.weight.nbytes() + b.ffn_down_exps.weight.nbytes()
        param_bytes -= int(expert_bytes * (1 - b.num_experts_per_tok / b.ffn_gate_exps.weight.shape[0]))
    gen = model.generate([0], 0)
    for _ in range(args.benchmark):
      GlobalCounters.reset()
      with Timing(on_exit=lambda x: f", {1e9/x:6.2f} tok/s, {GlobalCounters.global_mem/x:7.2f} GB/s, param {param_bytes/x:7.2f} GB/s"): next(gen)
    exit(0)

  # start server
  if args.serve: TCPServerWithReuse(('', args.serve), Handler).serve_forever()

  ids: list[int] = [bos_id] if bos_id is not None else []
  while 1:
    start_pos = max(len(ids) - 1, 0)
    try:
      ids += tok.role("user") + tok.encode(input('>>> ')) + tok.end_turn(eos_id) + tok.role("assistant")
    except EOFError:
      break
    for next_id in model.generate(ids, start_pos):
      sys.stdout.write(tok.decode([next_id]) if next_id != eos_id else "\n\n")
      sys.stdout.flush()
      if next_id == eos_id: break
