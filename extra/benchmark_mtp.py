"""Compare warmed greedy decoding with and without Qwen3.5 speculative decoding."""
import argparse, itertools, json, statistics, time
from tinygrad import Device
from tinygrad.llm.cli import SimpleTokenizer
from tinygrad.llm.model import Transformer


def compare_decode(model, tokens, draft_tokens, decode_tokens=128, repeat=3, chunk_size=32):
  if not tokens or len(tokens)+decode_tokens+1 > model.max_context:
    raise ValueError('prompt and requested output must fit in max_context')
  if decode_tokens < 1 or repeat < 1: raise ValueError('decode_tokens and repeat must be positive')
  if not 1 <= draft_tokens <= 7: raise ValueError('draft_tokens must be between 1 and 7')

  def run(mode, count):
    model._cached_tokens = []
    gen = model.generate(tokens.copy(), chunk_size=chunk_size, mtp=mode)
    try:
      first = next(gen)
      Device[Device.DEFAULT].synchronize()
      start = time.perf_counter()
      output = [first] + list(itertools.islice(gen, count))
      Device[Device.DEFAULT].synchronize()
      elapsed = time.perf_counter()-start
      if len(output) != count+1: raise RuntimeError('generation ended before the requested token count')
      return output, count/elapsed, dict(model.mtp_stats) if mode else None
    finally: gen.close()

  # Use the actual prompt and full decode length so later draft shapes are captured before timing.
  for mode in (0, draft_tokens):
    for i in range(2):
      print(f'warm mtp={mode} iteration={i}', flush=True)
      run(mode, decode_tokens)
  reference, records = None, []
  # Alternate order to reduce systematic clock/temperature bias between the modes.
  for iteration in range(repeat):
    for mode in ((0, draft_tokens) if iteration%2 == 0 else (draft_tokens, 0)):
      output, rate, stats = run(mode, decode_tokens)
      if reference is None: reference = output
      equal = output == reference
      record = dict(mtp=mode, tok_s=rate, equal=equal, stats=stats)
      if not equal:
        record['first_mismatch'] = next(i for i,(a,b) in enumerate(zip(reference, output)) if a != b)
        record['reference'], record['output'] = reference, output
      print(json.dumps(record), flush=True)
      records.append(record)
  rates = {mode:statistics.median(r['tok_s'] for r in records if r['mtp'] == mode) for mode in (0, draft_tokens)}
  return dict(no_mtp_tok_s=rates[0], mtp_tok_s=rates[draft_tokens], speedup=rates[draft_tokens]/rates[0],
              equal=all(r['equal'] for r in records), records=records)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--model', required=True)
  parser.add_argument('--max-context', type=int, default=2048)
  parser.add_argument('--decode-tokens', type=int, default=128)
  parser.add_argument('--draft-tokens', type=int, default=3)
  parser.add_argument('--repeat', type=int, default=3)
  parser.add_argument('--prompt', action='append')
  parser.add_argument('--prompt-tokens', type=int, help='use the synthetic prompt from benchmark_llm')
  parser.add_argument('--chunk-size', type=int, default=32)
  parser.add_argument('--min-speedup', type=float, default=0, help='fail if any prompt is below this speedup')
  parser.add_argument('--json', help='write the complete benchmark result')
  args = parser.parse_args()
  if args.prompt_tokens is not None and args.prompt_tokens < 1: parser.error('--prompt-tokens must be positive')
  m, kv = Transformer.from_gguf(args.model, args.max_context)
  tokenizer = SimpleTokenizer.from_gguf_kv(kv)
  prompts = args.prompt or [
    'Write a Python function that returns all prime numbers less than n. Explain how it works.',
    'Explain how rain forms and why clouds do not always produce rain.',
    '请介绍中国的四大发明，以及它们对世界历史的影响。',
  ]
  if args.prompt_tokens is not None: prompts = [f'synthetic:{args.prompt_tokens}']
  report = dict(model=args.model, device=Device.DEFAULT, target=str(Device[Device.DEFAULT].target),
                max_context=args.max_context, decode_tokens=args.decode_tokens, draft_tokens=args.draft_tokens,
                repeat=args.repeat, results=[])
  print(json.dumps({k:v for k,v in report.items() if k != 'results'}), flush=True)
  for prompt in prompts:
    tokens = tokenizer.encode('<|im_start|>user\n'+prompt+'<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')
    if args.prompt_tokens is not None: tokens = [257]+[1000+i%1000 for i in range(args.prompt_tokens-1)]
    result = compare_decode(m, tokens, args.draft_tokens, args.decode_tokens, args.repeat, args.chunk_size)
    report['results'].append(dict(prompt=prompt, prompt_tokens=len(tokens), **result))
    print(f"speedup {result['speedup']:.3f}x equal={result['equal']}", flush=True)
  if args.json:
    with open(args.json, 'w') as f: json.dump(report, f, ensure_ascii=False, indent=2)
  if any(not r['equal'] or r['speedup'] < args.min_speedup for r in report['results']): raise SystemExit(1)
