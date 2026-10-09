from __future__ import annotations
import itertools, json, pathlib, queue, re, socketserver, threading, time, typing, uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from tinygrad.helpers import DEBUG, colored, stderr_log
from tinygrad.llm.model import Generation
from tinygrad.viz.serve import TCPServerWithReuse, Handler as VizHandler
if TYPE_CHECKING:
  from tinygrad.llm.cli import SimpleTokenizer
  from tinygrad.llm.model import Transformer

def parse_tool_call(s:str) -> tuple[str, typing.Any]|None:
  s = s.strip()
  if s.startswith("{"):  # hermes JSON format: {"name": ..., "arguments": {...}}
    try:
      call = json.loads(s)
      return call["name"], call.get("arguments", call.get("parameters", {}))
    except (json.JSONDecodeError, KeyError): return None
  # XML format: <function=name>\n<parameter=key>\nvalue\n</parameter>...</function>
  # GLM format: name<arg_key>key</arg_key><arg_value>value</arg_value>...
  patterns = (
    (r"<function=([^>]+)>\s*(.*?)\s*(?:</function>)?$", r"<parameter=([^>]+)>(.*?)</parameter>"),
    (r"([\w.-]+)\s*((?:<arg_key>[^<]+</arg_key>\s*<arg_value>.*?</arg_value>\s*)*)$", r"<arg_key>([^<]+)</arg_key>\s*<arg_value>(.*?)</arg_value>"))
  for call_pattern, arg_pattern in patterns:
    if (fm := re.match(call_pattern, s, re.DOTALL)):
      args = {}
      for pm in re.finditer(arg_pattern, fm.group(2), re.DOTALL):
        value = re.sub(r"^\r?\n|\r?\n\Z", "", pm.group(2))
        try: args[pm.group(1)] = json.loads(value)
        except json.JSONDecodeError: args[pm.group(1)] = value
      return fm.group(1), args
  return None

def normalize_messages(messages:list[dict]) -> None:
  # chat templates expect tool_call arguments as dicts (OpenAI clients send JSON strings)
  for m in messages:
    for tc in m.get("tool_calls") or []:
      if "function" in tc and isinstance(args := tc["function"].get("arguments"), str):
        try: tc["function"]["arguments"] = json.loads(args)
        except json.JSONDecodeError: pass
    # string-only templates need text parts joined and explicit null content emptied
    if m.get("content", "") is None: m["content"] = ""
    elif isinstance(c := m.get("content"), list) and all(p.get("type") == "text" for p in c): m["content"] = "".join(p["text"] for p in c)

class StreamRouter:
  # routes streamed output text to (field, text) deltas, keeping tool_call regions in .buf for the final parse
  def __init__(self, reasoning:bool=False):
    self.buf = ""
    self.mode = "reasoning" if reasoning else "undecided"  # output inside a think block is sent as reasoning_content
  def split(self, tag:str, final:bool) -> tuple[str, bool]:
    # split buf on the first full tag, holding back a partial tag at the end unless final
    if tag in self.buf:
      before, self.buf = self.buf.split(tag, 1)
      return before, True
    hold = max((i for i in range(1, min(len(self.buf), len(tag))+1) if tag.startswith(self.buf[-i:])), default=0) if not final else 0
    emit, self.buf = self.buf[:len(self.buf)-hold], self.buf[len(self.buf)-hold:]
    return emit, False
  def route(self, piece:str, final:bool=False) -> typing.Iterator[tuple[str, str]]:
    self.buf += piece
    if self.mode == "undecided":  # decide whether the output starts with a think block
      if not final and len(self.buf) < len("<think>") and "<think>".startswith(self.buf): return
      self.mode, self.buf = ("reasoning", self.buf[len("<think>"):]) if self.buf.startswith("<think>") else ("content", self.buf)
    if self.mode == "reasoning":
      emit, done = self.split("</think>", final)
      if emit: yield "reasoning_content", emit
      if not done: return
      self.mode = "content"
    if self.mode == "tool": return
    emit, found = self.split("<tool_call>", final)
    if emit: yield "content", emit
    if found: self.mode, self.buf = "tool", "<tool_call>" + self.buf

@dataclass
class Request(Generation):
  output: queue.SimpleQueue = field(default_factory=queue.SimpleQueue)

class Handler(VizHandler):
  server: LLMServer
  def log_request(self, code='-', size='-'): pass
  def do_GET(self):
    if self.path == "/v1/models": self.send_data(json.dumps({"object":"list","data":[{"id":self.server.model_name,"object":"model"}]}).encode())
    elif self.path.startswith("/assets/"): super().do_GET()
    else: self.send_data((pathlib.Path(__file__).parent / "chat.html").read_bytes(), content_type="text/html")
  def run_model(self, ids:list[int], model_name:str, include_usage=False, max_tokens:int|None=None, temperature:float=0.0,
                reasoning:bool=False, request_st:float|None=None):
    prompt_tokens = len(ids)
    tmpl: dict = {"id":f"chatcmpl-{uuid.uuid4().hex[:24]}", "object":"chat.completion.chunk", "created":int(time.time()), "model":model_name}
    def chunk(d:dict): return {"choices": [{"index":0, "delta":d, "finish_reason":None}], **tmpl}
    out, finish_reason, pt = [], "cancelled", None
    st = last = time.perf_counter()
    if request_st is None: request_st = st
    dec = self.server.tok.stream_decoder()
    router = StreamRouter(reasoning)
    self.server.pending.put(req := Request(ids, temperature, max_tokens, self.server.tok.is_end))
    try:
      yield chunk({"role":"assistant", "content":""})
      for next_id in iter(req.output.get, None):
        if isinstance(next_id, Exception): raise next_id
        if pt is None: pt = time.perf_counter()
        out.append(next_id)
        last = time.perf_counter()
        for field, delta in router.route(dec(next_id)): yield chunk({field:delta})
      finish_reason = req.finish_reason or "cancelled"
      for field, delta in router.route(dec(), final=True): yield chunk({field:delta})
      tool_calls: list[dict] = []
      for m in re.finditer(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", router.buf, re.DOTALL):
        if (parsed := parse_tool_call(m.group(1))) is None:
          stderr_log(f"[{tmpl['id'][-8:]}] /v1/chat/completions invalid tool call: {m.group(0)[:200]!r}\n")
          yield chunk({"content":m.group(0)})  # don't silently drop output the client can't use
        else:
          name, args = parsed
          tool_calls.append({"index":len(tool_calls), "id":f"call_{uuid.uuid4().hex[:24]}", "type":"function",
                             "function":{"name":name, "arguments":args if isinstance(args, str) else json.dumps(args)}})
      if tool_calls:
        yield chunk({"tool_calls":tool_calls})
        if finish_reason == "stop": finish_reason = "tool_calls"
      yield {"choices": [{"index":0, "delta":{},"finish_reason":finish_reason}], **tmpl}
      if include_usage:
        yield {"choices": [], "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": len(out),
                                        "total_tokens": prompt_tokens + len(out)}, **tmpl}
    except Exception:
      finish_reason = "error"
      raise
    finally:
      ttft = f"{pt-request_st:.2f}s" if pt is not None else "-"
      decode = f"{(len(out)-1)/(last-pt):.0f} tok/s" if len(out) > 1 and pt is not None and last > pt else "-"
      stderr_log(f"[{tmpl['id'][-8:]}] /v1/chat/completions in:{prompt_tokens} out:{len(out)} prep:{(st-request_st)*1e3:.0f}ms "
                 f"ttft:{ttft} decode:{decode} total:{time.perf_counter()-request_st:.2f}s finish:{finish_reason}\n")
      req.finish_reason = req.finish_reason or finish_reason

  def do_POST(self):
    request_st = time.perf_counter()
    raw_body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
    body: dict[str, typing.Any] = json.loads(raw_body.decode("utf-8"))
    if DEBUG >= 1: print(json.dumps(body, indent=2))
    if self.path == "/v1/chat/completions":
      # render and tokenize
      normalize_messages(body["messages"])
      rendered = self.server.template.render(messages=body["messages"], tools=body.get("tools"), add_generation_prompt=True, preserve_thinking=True)
      ids: list[int] = self.server.tok.encode(rendered)
      if len(ids) >= self.server.model.max_context:
        stderr_log(f"{colored('context length exceeded', 'red')}  in:{len(ids):5d}  max:{self.server.model.max_context:5d}\n")
        return self.send_data(json.dumps({"error":{"message":f"prompt has {len(ids)} tokens, but the model context is "
          f"{self.server.model.max_context}", "type":"invalid_request_error", "param":"messages", "code":"context_length_exceeded"}}).encode(),
          status_code=400)

      # reply
      max_tokens = body.get("max_completion_tokens") or body.get("max_tokens")
      chunks = self.run_model(ids, body["model"], not body.get("stream") or body.get("stream_options",{}).get("include_usage", False),
                              max_tokens=max_tokens, temperature=float(body.get("temperature", 0.0)),
                              reasoning=rendered.rstrip().endswith("<think>"), request_st=request_st)
      if body.get("stream"): self.stream_json(chunks)
      else:
        out, reasoning, tool_calls, finish_reason = [], [], [], "stop"
        for c in chunks:
          if not c["choices"]: continue
          choice = c["choices"][0]
          if (delta := choice.get("delta", {})):
            if delta.get("content"): out.append(delta["content"])
            if delta.get("reasoning_content"): reasoning.append(delta["reasoning_content"])
            tool_calls += [{k:v for k, v in tc.items() if k != "index"} for tc in delta.get("tool_calls", [])]
          if choice.get("finish_reason"): finish_reason = choice["finish_reason"]
        message: dict[str, typing.Any] = {"role":"assistant", "content":"".join(out) or None}
        if reasoning: message["reasoning_content"] = "".join(reasoning)
        if tool_calls: message["tool_calls"] = tool_calls
        self.send_data(json.dumps({**c, "object":"chat.completion",
          "choices":[{"index":0, "message":message, "finish_reason":finish_reason}]}).encode())
    else:
      raise RuntimeError(f"unhandled path {self.path}")

class LLMServer(socketserver.ThreadingMixIn, TCPServerWithReuse):
  daemon_threads = True
  def __init__(self, server_address:tuple, model:Transformer, model_name:str, tok:SimpleTokenizer, template:typing.Any, batch_size:int=1):
    self.model, self.model_name, self.tok, self.template = model, model_name, tok, template
    self.pending, self.stopping = queue.Queue[Request](), threading.Event()
    super().__init__(server_address, Handler)
    self.worker = threading.Thread(target=self.run, args=(batch_size,), name='llm-inference')
    self.worker.start()

  def run(self, batch_size):
    jobs = [Request([], finish_reason="stop") for _ in range(batch_size)]
    source = self.model.generate_batch(jobs)
    while not self.stopping.is_set():
      while free := [i for i, job in enumerate(jobs) if job.finish_reason is not None]:
        try: job = self.pending.get(timeout=0.1 if len(free) == len(jobs) else 0)
        except queue.Empty: break
        i = max(free, key=lambda i: sum(1 for _ in itertools.takewhile(lambda ab: ab[0] == ab[1], zip(job.tokens, jobs[i].tokens))))
        jobs[i] = job
      if not (active := [(i, job) for i, job in enumerate(jobs) if job.finish_reason is None]): continue
      try: tokens = next(source)
      except Exception as exc:
        tokens = [None if isinstance(exc, StopIteration) else exc]*batch_size
        for _, job in active: job.finish_reason = job.finish_reason or "error"
        source = self.model.generate_batch(jobs)
      for i, job in active:
        if tokens[i] is not None: job.output.put(tokens[i])
        if job.finish_reason is not None: job.output.put(None)
    source.close()
    for job in jobs + list(self.pending.queue): job.output.put(None)

  def server_close(self):
    self.stopping.set()
    self.worker.join()
    super().server_close()
