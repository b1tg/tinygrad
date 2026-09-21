import argparse, sys, time
from tinygrad.llm.model import Transformer
from tinygrad.helpers import profile_marker

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", required=True, help="path to gguf model")
  parser.add_argument("--max-context", type=int, default=8192, help="max context length (default: %(default)s)")
  parser.add_argument("--prompt-tokens", type=int, default=1024, help="number of prompt tokens (default: %(default)s)")
  parser.add_argument("--decode-tokens", type=int, default=16, help="number of tokens to decode (default: %(default)s)")
  parser.add_argument("--chunk-size", type=int, default=None, help="chunk size for prefill (default: model setting)")
  args = parser.parse_args()

  profile_marker("load start")
  st = time.perf_counter()
  model, metadata = Transformer.from_gguf(args.model, args.max_context)
  del metadata  # tokenizer metadata destruction must not be charged to the first decode-loop iteration
  print(f"load {time.perf_counter()-st:.3f}s", flush=True)
  profile_marker("load end")

  profile_marker("warmup start")
  st = time.perf_counter()
  model.warmup(chunk_size=args.chunk_size)
  print(f"warm {time.perf_counter()-st:.3f}s", flush=True)
  profile_marker("warmup end")

  prompt = [257] + [1000+i%1000 for i in range(args.prompt_tokens-1)]
  gen = model.generate(prompt, chunk_size=args.chunk_size)
  profile_marker("prefill start")
  st = time.perf_counter()
  # first token is time-to-first-token; counted as part of prefill
  output = [next(gen)]
  pt = time.perf_counter()
  profile_marker("prefill end")
  print(f"prefill {args.prompt_tokens/(pt-st):.3f} tok/s", flush=True)

  profile_marker("decode start")
  dt = time.perf_counter()
  for _ in range(args.decode_tokens): output.append(next(gen))
  et = time.perf_counter()
  profile_marker("decode end")
  print(f"decode {args.decode_tokens/(et-dt):.3f} tok/s output {output}", flush=True)

if __name__ == "__main__":
  try: main()
  except KeyboardInterrupt: sys.exit(1)
