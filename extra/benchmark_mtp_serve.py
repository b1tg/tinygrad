"""Measure client-observed SSE decode throughput from the real --serve entry point."""
import argparse, http.client, json, os, pathlib, statistics, subprocess, sys, time


def request(port, prompt, count, request_body=None):
  conn = http.client.HTTPConnection('127.0.0.1', port, timeout=1800)
  body = dict(model='local', messages=[dict(role='user', content=prompt)], temperature=0,
              max_tokens=count, stream=True, stream_options=dict(include_usage=True))
  if request_body is not None:
    body = dict(request_body, stream=True, stream_options=dict(include_usage=True))
  start = time.perf_counter()
  conn.request('POST', '/v1/chat/completions', json.dumps(body), {'Content-Type': 'application/json'})
  response = conn.getresponse()
  if response.status != 200: raise RuntimeError(response.read().decode())
  events, usage, finish, finished_at, tool_calls = [], None, None, None, []
  try:
    for line in response:
      if not line.startswith(b'data:'): continue
      payload = line[5:].strip()
      if payload == b'[DONE]': break
      item = json.loads(payload)
      now = time.perf_counter()
      if 'usage' in item: usage = item['usage']
      for choice in item['choices']:
        if choice.get('finish_reason'): finish, finished_at = choice['finish_reason'], now-start
        delta = choice.get('delta', {})
        if delta.get('tool_calls'):
          tool_calls += [dict(type=t.get('type'), function=t.get('function')) for t in delta['tool_calls']]
          events.append(dict(time=now-start, field='tool_calls', text=json.dumps(delta['tool_calls'])))
        for field in ('content', 'reasoning_content'):
          if delta.get(field): events.append(dict(time=now-start, field=field, text=delta[field]))
  finally: conn.close()
  if not usage or not events or finished_at is None: raise RuntimeError('missing usage or streamed output')
  # SSE chunks need not correspond one-to-one to tokens. Use server completion-token usage, never chunk count.
  elapsed = finished_at-events[0]['time']
  separable = any(e['field'] != 'tool_calls' for e in events)
  if not separable: raise RuntimeError('tool-only response is buffered: client cannot separate prefill and decode latency')
  return dict(tok_s=(usage['completion_tokens']-1)/elapsed, decode_seconds=elapsed, ttft=events[0]['time'],
              total_seconds=time.perf_counter()-start, usage=usage, finish_reason=finish, tool_calls=tool_calls,
              content=''.join(e['text'] for e in events if e['field']=='content'),
              reasoning=''.join(e['text'] for e in events if e['field']=='reasoning_content'), events=events)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--model', required=True)
  parser.add_argument('--port', type=int, default=18350)
  parser.add_argument('--max-context', type=int, default=4096)
  parser.add_argument('--draft-tokens', type=int, default=3)
  parser.add_argument('--decode-tokens', type=int, default=128)
  parser.add_argument('--repeat', type=int, default=3)
  parser.add_argument('--warmup', type=int, default=2)
  parser.add_argument('--prompt', action='append')
  parser.add_argument('--output', required=True)
  parser.add_argument('--request-json', help='replay an actual client request including its messages and tools')
  args = parser.parse_args()
  prompts = args.prompt or ['Write a Python function that returns all prime numbers less than n. Explain how it works.',
                           'Explain how rain forms and why clouds do not always produce rain.',
                           '请介绍中国的四大发明，以及它们对世界历史的影响。']
  request_body = json.loads(pathlib.Path(args.request_json).read_text()) if args.request_json else None
  if request_body is not None: prompts = ['pi: explain rangify']
  output = pathlib.Path(args.output)
  report = dict(config=vars(args), measurement='client SSE first nonempty delta to finish event, including buffered tool calls',
                lookup=False, runs=[])
  for mode in (0, args.draft_tokens):
    log_path = output.with_suffix(f'.mtp{mode}.server.log')
    env = dict(os.environ, MTP=str(mode), MTP_LOOKUP='0', TMPDIR='/tmp/1', PYTHONUNBUFFERED='1')
    with log_path.open('w') as log:
      proc = subprocess.Popen([sys.executable, '-m', 'tinygrad.llm', '--model', args.model,
                               '--max_context', str(args.max_context), '--serve', str(args.port)], env=env, stdout=log, stderr=log)
      try:
        deadline = time.monotonic()+1800
        while True:
          if proc.poll() is not None: raise RuntimeError(f'server exited: {log_path}')
          try:
            conn = http.client.HTTPConnection('127.0.0.1', args.port, timeout=1)
            conn.request('GET', '/v1/models')
            ready = conn.getresponse().status == 200
            conn.close()
            if ready: break
          except OSError: pass
          if time.monotonic() > deadline: raise TimeoutError('server startup')
          time.sleep(1)
        for prompt in prompts:
          for i in range(args.warmup):
            print(f'warmup mtp={mode} request={i} prompt={prompt!r}', flush=True)
            request(args.port, prompt, args.decode_tokens, request_body)
          for i in range(args.repeat):
            result = dict(mtp=mode, prompt=prompt, iteration=i, **request(args.port, prompt, args.decode_tokens, request_body))
            report['runs'].append(result)
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
            print(json.dumps({k:v for k,v in result.items() if k not in ('events','content','reasoning')}, ensure_ascii=False), flush=True)
      finally:
        proc.terminate()
        try: proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
          proc.kill()
          proc.wait()
  report['summary'] = []
  for prompt in prompts:
    runs = [r for r in report['runs'] if r['prompt']==prompt]
    reference = runs[0]
    equal = all((r['content'], r['reasoning'], r['tool_calls'], r['usage']) ==
                (reference['content'], reference['reasoning'], reference['tool_calls'], reference['usage']) for r in runs)
    rates = {mode:statistics.median(r['tok_s'] for r in runs if r['mtp']==mode) for mode in (0,args.draft_tokens)}
    summary = dict(prompt=prompt, no_mtp_tok_s=rates[0], mtp_tok_s=rates[args.draft_tokens],
                   speedup=rates[args.draft_tokens]/rates[0], equal=equal)
    report['summary'].append(summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
  output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
  if any(not r['equal'] or r['speedup'] < 1.5 for r in report['summary']): raise SystemExit(1)
