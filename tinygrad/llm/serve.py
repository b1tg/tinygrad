from __future__ import annotations
import json, pathlib, re, time, typing, uuid
from typing import TYPE_CHECKING
from tinygrad.helpers import DEBUG, colored, stderr_log
from tinygrad.viz.serve import TCPServerWithReuse, HTTPRequestHandler
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
  if (fm := re.match(r"<function=([^>]+)>\s*(.*?)\s*(?:</function>)?$", s, re.DOTALL)):
    args = {}
    for pm in re.finditer(r"<parameter=([^>]+)>(.*?)</parameter>", fm.group(2), re.DOTALL):
      value = re.sub(r"^\r?\n|\r?\n\Z", "", pm.group(2))
      try: args[pm.group(1)] = json.loads(value)
      except json.JSONDecodeError: args[pm.group(1)] = value
    return fm.group(1), args
  return None

def normalize_messages(messages:list[dict], harmony:bool=False) -> None:
  # chat templates expect tool_call arguments as dicts (OpenAI clients send JSON strings)
  for m in messages:
    for tc in m.get("tool_calls") or []:
      if "function" in tc and isinstance(args := tc["function"].get("arguments"), str):
        try: tc["function"]["arguments"] = json.loads(args)
        except json.JSONDecodeError: pass
    # collapse text-only OpenAI content parts into a plain string, chat templates expect strings
    if isinstance(parts := m.get("content"), list):
      if any(not isinstance(c, dict) or c.get("type") != "text" or not isinstance(c.get("text"), str) for c in parts):
        raise ValueError("only text content parts are supported")
      m["content"] = "".join(c["text"] for c in parts)
    # harmony templates read CoT from 'thinking'; clients send it as 'reasoning_content'. Only map tool call turns:
    # the unsloth template misroutes content of plain turns with 'thinking' set (upstream llama.cpp maps the same way)
    if harmony and m.get("role") == "assistant" and m.get("tool_calls") and not m.get("content") \
       and isinstance(rc := m.get("reasoning_content"), str) and "thinking" not in m: m["thinking"] = rc

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
  def tool_calls(self) -> typing.Iterator[tuple[str, typing.Any]|str]:
    # yields (name, arguments) for each tool call region in .buf, or the raw text if parsing failed
    for m in re.finditer(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", self.buf, re.DOTALL):
      yield parse_tool_call(m.group(1)) or m.group(0)

class HarmonyRouter(StreamRouter):
  # routes harmony format (gpt-oss) output to (field, text) deltas, accumulating tool call arguments in .buf
  # state machine: header -> {tool (stop), reasoning -> header, content -> header}
  # message format:
  # response contains multiple messages
  # one message: <|start|>{header}<|message|>{content}<|end|>
  # {header} with channel: assistant<|channel|>final
  # assistant tool call: <|start|>assistant to=functions.{name}<|channel|>commentary json<|message|>{args}<|call|>
  # tool result: <|start|>functions.{name} to=assistant<|channel|>commentary<|message|>{output}<|end|>
  # headers end at <|message|>: [<|start|>assistant][ to=NAME][<|channel|>CHANNEL]
  def __init__(self): self.buf, self.mode, self.tool_name = "", "header", ""
  def route(self, piece:str, final:bool=False) -> typing.Iterator[tuple[str, str]]:
    self.buf += piece
    while self.buf:  # every iteration either breaks (needs more input) or consumes a delimiter from self.buf
      if self.mode == "header":
        if "<|message|>" not in self.buf and not final: break  # hold the whole header, it must be parsed intact
        hdr, _ = self.split("<|message|>", final)
        if "to=" in hdr:  # tool call: to=functions.NAME[<|constrain|>json| code][<|channel|>commentary json]<|message|>ARGS<|call|>
          self.mode, self.tool_name = "tool", hdr.split("to=")[1].split("<|")[0].split()[0].removeprefix("functions.")
        else: self.mode = "reasoning" if hdr.rsplit("<|channel|>")[-1].strip() == "analysis" else "content"
      elif self.mode == "tool": break  # tool call arguments accumulate in .buf
      else:  # reasoning or content message body until <|end|>, then a new header follows
        emit, done = self.split("<|end|>", final)
        if emit: yield ("reasoning_content" if self.mode == "reasoning" else "content"), emit
        if not done: break
        self.mode = "header"
  def tool_calls(self) -> typing.Iterator[tuple[str, typing.Any]|str]:
    if self.mode == "tool":  # generation stops at <|call|>, so there is at most one call, and .buf holds its arguments
      try: yield self.tool_name, json.loads(self.buf.strip() or "{}")  # empty arguments mean {}
      except json.JSONDecodeError: yield self.buf

class Handler(HTTPRequestHandler):
  server: LLMServer
  def log_request(self, code='-', size='-'): pass
  def do_GET(self):
    if self.path == "/v1/models": self.send_data(json.dumps({"object":"list","data":[{"id":self.server.model_name,"object":"model"}]}).encode())
    else: self.send_data((pathlib.Path(__file__).parent / "chat.html").read_bytes(), content_type="text/html")
  def run_model(self, ids:list[int], model_name:str, include_usage=False, max_tokens:int|None=None, temperature:float=0.0,
                reasoning:bool=False):
    model, tok = self.server.model, self.server.tok
    prompt_tokens = len(ids)
    cache_start_pos = model.get_start_pos(ids)
    stderr_log(f"in:{colored(f'{cache_start_pos:5d}', 'green')} +{len(ids)-cache_start_pos:5d}  {colored('--', 'BLACK')}  ")
    tmpl = {"id":f"chatcmpl-{uuid.uuid4().hex[:24]}", "object":"chat.completion.chunk", "created":int(time.time()), "model":model_name}
    def chunk(d:dict): return {"choices": [{"index":0, "delta":d, "finish_reason":None}], **tmpl}
    out: list[int] = []
    finish_reason = "stop"
    st = pt = time.perf_counter()
    dec = tok.stream_decoder()
    router = HarmonyRouter() if self.server.harmony else StreamRouter(reasoning)
    def log_stats(interrupted:bool=False):
      et = time.perf_counter()
      total = f"total:{et-st:6.2f}s"
      stderr_log(f"gen:{len(out)/(et-pt) if len(out) > 1 else 0:4.0f} tok/s  {colored('--', 'BLACK')}  "
                 f"out:{len(out):5d}  {colored('--', 'BLACK')}  {colored(total, 'red') if interrupted else total}\n")
    completed = False
    try:
      yield chunk({"role":"assistant", "content":""})
      for next_id in model.generate(ids, temperature=temperature):
        if len(out) == 0:
          stderr_log(f"prefill:{(prompt_tokens-cache_start_pos)/((pt:=time.perf_counter())-st):4.0f} tok/s  {colored('--', 'BLACK')}  ")
        if tok.is_end(next_id): break
        out.append(next_id)
        for field, delta in router.route(dec(next_id)): yield chunk({field:delta})
        if max_tokens is not None and len(out) >= max_tokens:
          finish_reason = "length"
          break
      for field, delta in router.route(dec(), final=True): yield chunk({field:delta})
      tool_calls: list[dict] = []
      for tc in router.tool_calls():
        if isinstance(tc, str):  # unparseable tool call, don't silently drop output the client can't use
          stderr_log(f"failed to parse tool call: {tc[:200]}")
          yield chunk({"content":tc})
        else:
          name, args = tc
          tool_calls.append({"index":len(tool_calls), "id":f"call_{uuid.uuid4().hex[:24]}", "type":"function",
                             "function":{"name":name, "arguments":args if isinstance(args, str) else json.dumps(args)}})
      if tool_calls:
        yield chunk({"tool_calls":tool_calls})
        if finish_reason == "stop": finish_reason = "tool_calls"
      completed = True
      yield {"choices": [{"index":0, "delta":{},"finish_reason":finish_reason}], **tmpl}
      if include_usage:
        yield {"choices": [], "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": len(out),
                                        "total_tokens": prompt_tokens + len(out)}, **tmpl}
      log_stats()
    except GeneratorExit:
      if not completed: log_stats(interrupted=True)
      raise

  def do_POST(self):
    request_st = time.perf_counter()
    stderr_log(f"{self.path}  {colored('--', 'BLACK')}  ")
    raw_body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
    body: dict[str, typing.Any] = json.loads(raw_body.decode("utf-8"))
    if DEBUG >= 1: print(json.dumps(body, indent=2))
    if self.path == "/v1/chat/completions":
      # render and tokenize
      try: normalize_messages(body["messages"], harmony=self.server.harmony)
      except ValueError as e:
        return self.send_data(json.dumps({"error":{"message":str(e), "type":"invalid_request_error", "param":"messages",
          "code":"invalid_message"}}).encode(), status_code=400)
      rendered = self.server.template.render(messages=body["messages"], tools=body.get("tools"), add_generation_prompt=True, preserve_thinking=True)
      ids: list[int] = self.server.tok.encode(rendered)
      stderr_log(f"prep:{(time.perf_counter()-request_st)*1e3:5.0f} ms  {colored('--', 'BLACK')}  ")
      if len(ids) >= self.server.model.max_context:
        stderr_log(f"{colored('context length exceeded', 'red')}  in:{len(ids):5d}  max:{self.server.model.max_context:5d}\n")
        return self.send_data(json.dumps({"error":{"message":f"prompt has {len(ids)} tokens, but the model context is "
          f"{self.server.model.max_context}", "type":"invalid_request_error", "param":"messages", "code":"context_length_exceeded"}}).encode(),
          status_code=400)

      # reply
      max_tokens = body.get("max_completion_tokens") or body.get("max_tokens")
      # harmony (gpt-oss) degenerates at temperature 0, default to 1.0 for it
      chunks = self.run_model(ids, body["model"], not body.get("stream") or body.get("stream_options",{}).get("include_usage", False),
                              max_tokens=max_tokens, temperature=float(body.get("temperature", 1.0 if self.server.harmony else 0.0)),
                              reasoning=rendered.rstrip().endswith("<think>"))
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

class LLMServer(TCPServerWithReuse):
  def __init__(self, server_address:tuple, model:Transformer, model_name:str, tok:SimpleTokenizer, template:typing.Any, harmony:bool=False):
    self.model, self.model_name, self.tok, self.template, self.harmony = model, model_name, tok, template, harmony
    super().__init__(server_address, Handler)
