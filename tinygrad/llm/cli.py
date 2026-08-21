from __future__ import annotations
import sys, argparse, codecs, itertools, typing, re, unicodedata, json, time
from typing import TYPE_CHECKING
from tinygrad import nn
from tinygrad.uop.ops import UOp, Ops
from tinygrad.helpers import partition, DEBUG, Timing, GlobalCounters, Context, fetch, profile_marker, getenv
from tinygrad.llm.model import Transformer
if TYPE_CHECKING:
  import jinja2

class SimpleTokenizer:
  def __init__(self, normal_tokens:dict[str, int], special_tokens:dict[str, int], preset:str="llama3",
               bos_id:int|None=None, eos_id:int=0, eot_id:int|None=None, merges:list[str]|None=None):
    preset = {"qwen35moe":"qwen35","chatglm-bpe":"glm4","llama-v3":"llama3","llama-bpe":"llama3"}.get(preset, preset)
    if preset not in ("llama3","qwen2","qwen35","olmo","kimi-k2","tekken","glm4"):
      raise ValueError(f"Invalid tokenizer preset '{preset}'")
    # https://github.com/openai/gpt-2/blob/9b63575ef42771a015060c964af2c3da4cf7c8ab/src/encoder.py#L9
    bs = [*range(33, 127), *range(161, 173), *range(174, 256)]  # bytes that map to themselves
    self._byte_decoder = {chr(b): b for b in bs} | {chr(256+i): b for i,b in enumerate(b for b in range(256) if b not in bs)}

    # https://github.com/ggml-org/llama.cpp/blob/94933c8c2eeaa9a7983e3f6c08af76bd86724094/src/llama-vocab.cpp#L286
    # 0x323b0 is one past the max codepoint in unicode categories L/N/Z (0x323af is max L)
    # compact adjacent codepoints into ranges: listing them all makes re spend seconds on large prompts
    def ucat_range(pre:str, hi:int=0x323b0, exclude=None) -> str:
      cps = enumerate(cp for cp in range(hi) if unicodedata.category(chr(cp)).startswith(pre) and not (exclude and exclude(cp)))
      runs = [list(g) for _, g in itertools.groupby(cps, lambda e: e[1]-e[0])]
      return "".join(re.escape(chr(g[0][1])) + (f"-{re.escape(chr(g[-1][1]))}" if len(g) > 1 else "") for g in runs)
    r_ws, r_p_N, r_p_L = r"\t\n\x0b\x0c\r\x85" + ucat_range("Z"), ucat_range("N"), ucat_range("L")
    r_L = r_p_L + ucat_range("M", 0xE0200) if preset == "qwen35" else r_p_L  # qwen35 treats marks as letters
    r_c, r_l = "(?i:'s|'t|'re|'ve|'m|'ll|'d)", f"[^\\r\\n{r_p_N}{r_p_L}]?"
    r_n, r_p = f"[{r_p_N}]{{1,3}}", f" ?[^{r_ws}{r_p_N}{r_L}]+[\\r\\n]*"
    r_w = f"{r_c}|{r_l}[{r_L}]+"
    r_t = f"[{r_ws}]*[\\r\\n]+|[{r_ws}]+(?![^{r_ws}])|[{r_ws}]+"
    if preset in ("qwen2", "qwen35", "tekken"): r_n = f"[{r_p_N}]"
    if preset == "kimi-k2":  # Han first, letters exclude Han: llama.cpp unicode_regex_split_custom_kimi_k2
      han = ((0x3400,0x4DBF),(0x4E00,0x9FFF),(0xF900,0xFAFF),(0x20000,0x2A6DF),(0x2A700,0x2B73F),
             (0x2B740,0x2B81F),(0x2B820,0x2CEAF),(0x2CEB0,0x2EBEF),(0x2F800,0x2FA1F))
      r_han = "".join(f"{chr(a)}-{chr(b)}" for a, b in han)
      r_w = f"[{r_han}]+|{r_l}[{ucat_range('L', exclude=lambda cp: any(a <= cp <= b for a, b in han))}]+{r_c}?"
    elif preset == "tekken":  # llama.cpp runs this on collapsed text: non-ascii letters match both cases, marks are punctuation
      r_up, r_lo = ucat_range("L", exclude=lambda cp: 0x61 <= cp <= 0x7A), ucat_range("L", exclude=lambda cp: 0x41 <= cp <= 0x5A)
      r_w = f"{r_l}[{r_up}]*[{r_lo}]+|{r_l}[{r_up}]+[{r_lo}]*"
      r_p = f" ?[^{r_ws}{r_p_N}{r_p_L}]+[\\r\\n/]*"
    elif preset == "olmo":  # llama.cpp reuses its gpt2 splitter: no whitespace/newline grouping
      r_w, r_n, r_p = f"'s|'t|'re|'ve|'m|'ll|'d| ?[{r_p_L}]+", f" ?[{r_p_N}]+", f" ?[^{r_ws}{r_p_N}{r_p_L}]+"
      r_t = f"[{r_ws}]+(?![^{r_ws}])|[{r_ws}]+"
    self._split_to_word = re.compile(f"{r_w}|{r_n}|{r_p}|{r_t}")
    self._split_to_sentence = re.compile("|".join(map(re.escape, sorted(filter(None, special_tokens), key=len, reverse=True))) or r"(?!)")

    self._normal_tokens = {bytes(self._byte_decoder[c] for c in tok): tid for tok, tid in normal_tokens.items()}
    self._special_tokens = special_tokens
    self._tok2bytes = {tid: tok for tok, tid in self._normal_tokens.items()} | {tid: tok.encode() for tok, tid in self._special_tokens.items()}
    self._bpe_ranks: dict[tuple[bytes, bytes], int] = {}
    for i, m in enumerate(merges or ()):
      if (pos := m.find(" ", 1)) < 0: continue
      try: self._bpe_ranks[bytes(self._byte_decoder[c] for c in m[:pos]), bytes(self._byte_decoder[c] for c in m[pos+1:])] = i
      except KeyError: pass
    self.preset = preset
    self.bos_id, self.eos_id, self.eot_id = bos_id, eos_id, eot_id

  @staticmethod
  def from_gguf_kv(kv:dict):
    # https://github.com/ggml-org/llama.cpp/blob/94933c8c2eeaa9a7983e3f6c08af76bd86724094/src/llama-vocab.cpp#L1818-L1820
    vocab: typing.Iterable[tuple[str, int]] = ((tok, idx) for idx, tok in enumerate(kv["tokenizer.ggml.tokens"]))
    # llama.cpp special cache: UNKNOWN=2, CONTROL=3, USER_DEFINED=4
    normal_tokens, special_tokens = partition(vocab, lambda e: kv["tokenizer.ggml.token_type"][e[1]] not in (2, 3, 4))
    special_tokens_dict = dict(special_tokens)
    add_bos = kv.get('tokenizer.ggml.add_bos_token', kv["tokenizer.ggml.pre"] in ("llama3","llama-v3","llama-bpe","tekken"))
    return SimpleTokenizer(dict(normal_tokens), special_tokens_dict, kv["tokenizer.ggml.pre"],
      bos_id=kv.get('tokenizer.ggml.bos_token_id') if add_bos else None,
      eos_id=kv.get('tokenizer.ggml.eos_token_id', 0), eot_id=kv.get('tokenizer.ggml.eot_token_id', special_tokens_dict.get('<|im_end|>')),
      merges=kv.get("tokenizer.ggml.merges"))

  def _encode_word(self, word:bytes) -> list[int]:
    if self.preset in ("llama3","tekken") and (early_token:=self._normal_tokens.get(word)) is not None: return [early_token]
    parts = [bytes([b]) for b in word]
    rank = (lambda a,b: self._bpe_ranks.get((a,b), sys.maxsize)) if self._bpe_ranks else (lambda a,b: self._normal_tokens.get(a+b, sys.maxsize))
    # greedily merge any parts that we can
    while True:
      i = min([(sys.maxsize, -1)] + [(rank(parts[j], parts[j+1]), j) for j in range(len(parts)-1)])[1]
      if i == -1: break
      parts[i:i+2] = [parts[i] + parts[i+1]]
    out = []
    for p in parts:
      if (tid:=self._normal_tokens.get(p)) is not None: out.append(tid)
      else: out += [t for b in p if (t:=self._normal_tokens.get(bytes([b]))) is not None]
    if not out: raise RuntimeError("token not found")
    return out
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
  def stream_decoder(self) -> typing.Callable[..., str]:
    dec = codecs.getincrementaldecoder('utf-8')('replace')
    def _decode(tid:int|None=None) -> str: return dec.decode(self._tok2bytes[tid]) if tid is not None else dec.decode(b'', final=True)
    return _decode
  def is_end(self, token_id:int) -> bool: return token_id in (self.eos_id, self.eot_id)

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
  "qwen3.5:0.8b": "https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q8_0.gguf",
  "qwen3.5:4b": "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf",
  "qwen3.5:9b": "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q4_K_M.gguf",
  "qwen3.6:27b": "https://huggingface.co/unsloth/Qwen3.6-27B-GGUF/resolve/main/Qwen3.6-27B-Q4_K_M.gguf",
  "qwen3.6:35b-a3b": "https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
  "olmoe": "https://huggingface.co/allenai/OLMoE-1B-7B-0924-Instruct-GGUF/resolve/main/olmoe-1b-7b-0924-instruct-q4_k_m.gguf",
  "moonlight": "https://huggingface.co/gabriellarson/Moonlight-16B-A3B-Instruct-GGUF/resolve/main/Moonlight-16B-A3B-Instruct-Q4_K_M.gguf",
  "glm-4.7-flash": "https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF/resolve/main/GLM-4.7-Flash-Q4_K_M.gguf",
}

class FallbackTemplate:
  # minimal jinja2.Template-compatible chat template without jinja2, no tool calling support
  def __init__(self, tok:SimpleTokenizer): self.tok = tok
  def role(self, role:str) -> str:
    if self.tok.preset == 'olmo': return "<|" + role + "|>\n"  # OLMoE Instruct format
    if self.tok.preset == 'kimi-k2': return "<|im_" + role + "|>" + role + "<|im_middle|>"
    if self.tok.preset.startswith('qwen'): return "<|im_start|>" + role + "\n"
    if self.tok.preset == 'glm4': return "<|" + role + "|>"
    if self.tok.preset == 'tekken':
      if role == 'user': return "[INST]"
      if role == 'assistant': return ""
      raise ValueError(f"Unsupported role '{role}' for tokenizer preset '{self.tok.preset}'")
    return "<|start_header_id|>" + role + "<|end_header_id|>\n\n"
  def end_turn(self) -> str:
    if self.tok.preset == 'olmo': return "\n"
    if self.tok.preset == 'kimi-k2': return self.tok.decode([self.tok.eos_id])
    if self.tok.preset.startswith('qwen'): return self.tok.decode([self.tok.eos_id]) + "\n"
    if self.tok.preset == 'glm4': return ""
    if self.tok.preset == 'tekken': return "[/INST]"
    return self.tok.decode([self.tok.eos_id])
  def render(self, messages:list[dict], tools=None, add_generation_prompt:bool=True, preserve_thinking:bool=False) -> str:
    out = self.tok.decode([] if self.tok.bos_id is None else [self.tok.bos_id]) + ("<sop>" if self.tok.preset == 'glm4' else "")
    for msg in messages:
      out += self.role(msg["role"])
      content = msg.get("content")
      if isinstance(content, str): out += content
      elif isinstance(content, list):
        for c in content:
          if c["type"] == "text": out += c["text"]
          else: raise RuntimeError(f"unhandled type: {c['type']}")
      elif content is not None: raise RuntimeError(f"unknown content type: {type(content)}")
      out += self.end_turn()
    return out + self.role("assistant") if add_generation_prompt else out

from tinygrad.llm.serve import LLMServer

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", "-m", default=list(models.keys())[0], help=f"Model choice ({', '.join(models.keys())}) or path to a local GGUF file")
  parser.add_argument("--max_context", type=int, default=4096, help="Max Context Length")
  parser.add_argument("--serve", nargs='?', type=int, const=8000, metavar="PORT", help="Run OpenAI compatible API (optional port, default 8000)")
  parser.add_argument("--warmup", action="store_true", help="warmup the JIT")
  parser.add_argument("--benchmark", nargs='?', type=int, const=20, metavar="COUNT", help="Benchmark tok/s (optional count, default 20)")
  args = parser.parse_args()

  # load the model
  model, kv = Transformer.from_gguf(fetch(models.get(args.model, args.model)), args.max_context)
  model_name = kv.get('general.name') or kv.get('general.basename') or args.model
  file_sizes = [y.nbytes() for y in UOp.sink(*[x.uop for x in nn.state.get_parameters(model)]).toposort() if y.op is Ops.BUFFER]
  print(f"using model \"{model_name}\" with {sum(file_sizes):,} bytes and {sum(x.numel() for x in nn.state.get_parameters(model)):,} params, "
        f"max context {args.max_context} on {nn.state.get_parameters(model)[0].device}")

  # get tokenizer
  tok = SimpleTokenizer.from_gguf_kv(kv)

  # use the model's chat template if jinja2 is available (enables model-specific formatting)
  template: jinja2.Template|FallbackTemplate = FallbackTemplate(tok)
  if (ct := kv.get('tokenizer.chat_template')) is not None:
    try:
      import jinja2
      env = jinja2.Environment()
      env.filters['tojson'] = lambda obj, **kwargs: json.dumps(obj, **kwargs)  # jinja2's tojson escapes <>& for HTML safety
      env.globals['raise_exception'] = lambda msg: (_ for _ in ()).throw(RuntimeError(msg))
      env.globals['strftime_now'] = lambda fmt: time.strftime(fmt)
      env.globals['bos_token'] = tok.decode([tok.bos_id]) if tok.bos_id is not None else ""
      env.globals['eos_token'] = tok.decode([tok.eos_id])
      template = env.from_string(ct)
    except ImportError: print("warning: jinja2 is not installed, the model's chat template is disabled")

  # warmup the JIT
  if args.warmup or args.serve:
    with Context(DEBUG=max(DEBUG.value, 1)): model.warmup()

  # start server
  if args.serve: LLMServer(('', args.serve), model, model_name, tok, template).serve_forever()

  # do benchmark
  if args.benchmark is not None:
    gen = model.generate(toks:=[tok.bos_id or 0])
    for i in range(args.benchmark):
      profile_marker(f"decode @ {i}")
      GlobalCounters.reset()
      if (log:=getenv("BENCHMARK_LOG", "")): from extra.bench_log import WallTimeEvent, BenchEvent
      with Timing(on_exit=lambda x: f", {1e9/x:6.2f} tok/s, {GlobalCounters.global_mem/x:7.2f} GB/s,"
                  f" {GlobalCounters.global_mem//1000000}/{GlobalCounters.mem_used//1000000} MB  --  "+\
                  tok.decode(toks).replace("\n", "\\n")):
        if log:
          with WallTimeEvent(BenchEvent.STEP): next(gen)
        else: next(gen)
    exit(0)

  # interactive chat
  messages: list[dict] = []
  while 1:
    try: messages.append({"role":"user", "content":input('>>> ')})
    except EOFError: break
    ids = tok.encode(template.render(messages=messages, add_generation_prompt=True))
    reply, dec = "", tok.stream_decoder()
    for next_id in model.generate(ids):
      if tok.is_end(next_id):
        sys.stdout.write(dec() + "\n\n")
        break
      reply += (piece := dec(next_id))
      sys.stdout.write(piece)
      sys.stdout.flush()
    messages.append({"role":"assistant", "content":reply})

if __name__ == "__main__": main()
