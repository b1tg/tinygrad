"""Compare warmed decoding with and without MTP using a synthetic prompt."""
import argparse, itertools, json, statistics, sys, time
from tinygrad import Device
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
      return output, count/elapsed
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
      output, rate = run(mode, decode_tokens)
      if reference is None: reference = output
      equal = output == reference
      record = dict(mtp=mode, tok_s=rate, equal=equal)
      if not equal:
        record['first_mismatch'] = next(i for i,(a,b) in enumerate(zip(reference, output)) if a != b)
        record['reference'], record['output'] = reference, output
      print(json.dumps(record), flush=True)
      records.append(record)
  rates = {mode:statistics.median(r['tok_s'] for r in records if r['mtp'] == mode) for mode in (0, draft_tokens)}
  return dict(no_mtp_tok_s=rates[0], mtp_tok_s=rates[draft_tokens], speedup=rates[draft_tokens]/rates[0],
              equal=all(r['equal'] for r in records), records=records)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--model', required=True)
  parser.add_argument('--max-context', type=int, default=2048)
  parser.add_argument('--prompt-tokens', type=int, default=1024)
  parser.add_argument('--decode-tokens', type=int, default=128)
  parser.add_argument('--draft-tokens', type=int, default=3)
  parser.add_argument('--repeat', type=int, default=3)
  parser.add_argument('--chunk-size', type=int, default=32)
  parser.add_argument('--json', help='write the complete benchmark result')
  args = parser.parse_args()
  if args.prompt_tokens < 1: parser.error('--prompt-tokens must be positive')
  model, _ = Transformer.from_gguf(args.model, args.max_context)
  tokens = [257]+[1000+i%1000 for i in range(args.prompt_tokens-1)]
  result = dict(vars(args), device=Device.DEFAULT, target=str(Device[Device.DEFAULT].renderer.target),
                **compare_decode(model, tokens, args.draft_tokens, args.decode_tokens, args.repeat, args.chunk_size))
  print(json.dumps({k:v for k,v in result.items() if k != 'records'}), flush=True)
  if args.json:
    with open(args.json, 'w') as f: json.dump(result, f, indent=2)


if __name__ == '__main__':
  try: main()
  except KeyboardInterrupt: sys.exit(1)
