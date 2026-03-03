#!/usr/bin/env python3
import argparse, itertools, json, os, re, statistics, subprocess, sys

def parse_int_list(s:str) -> list[int]:
  return [int(x.strip()) for x in s.split(",") if x.strip()]

def run_cmd(cmd:list[str], env:dict[str,str]) -> str:
  cp = subprocess.run(cmd, env=env, capture_output=True, text=True)
  if cp.returncode != 0:
    raise RuntimeError(f"command failed ({cp.returncode}): {' '.join(cmd)}\nSTDOUT:\n{cp.stdout}\nSTDERR:\n{cp.stderr}")
  return cp.stdout + cp.stderr

def run_e2e_once(model:str, bench_cnt:int, warmup_skip:int, mv_threads:int, mv_block:int, mv_rows:int) -> dict:
  env = dict(os.environ)
  env.update({
    "MV_THREADS_PER_ROW": str(mv_threads),
    "MV_BLOCKSIZE": str(mv_block),
    "MV_ROWS_PER_THREAD": str(mv_rows),
    "PYTHONPATH": ".",
  })
  out = run_cmd([sys.executable, "tinygrad/apps/llm.py", "--model", model, "--benchmark", str(bench_cnt)], env)
  tok_s = [float(x) for x in re.findall(r"([0-9]+\.?[0-9]*) tok/s", out)]
  ms = [float(x) for x in re.findall(r"\n\s*([0-9]+\.?[0-9]*) ms,", out)]
  if len(tok_s) == 0: raise RuntimeError(f"no tok/s parsed from output:\n{out}")
  steady_tok_s = tok_s[warmup_skip:] if len(tok_s) > warmup_skip else tok_s
  steady_ms = ms[warmup_skip:] if len(ms) > warmup_skip else ms
  return {
    "steady_avg_tok_s": sum(steady_tok_s)/len(steady_tok_s),
    "steady_med_tok_s": statistics.median(steady_tok_s),
    "steady_avg_ms": sum(steady_ms)/len(steady_ms) if len(steady_ms) else None,
    "raw_tok_s": tok_s,
  }

def run_kernel_probe_once(mv_threads:int, mv_block:int, mv_rows:int, dim:int, hidden_dim:int, vocab_size:int, n_kv_heads:int, head_dim:int) -> dict:
  env = dict(os.environ)
  env.update({
    "MV_THREADS_PER_ROW": str(mv_threads),
    "MV_BLOCKSIZE": str(mv_block),
    "MV_ROWS_PER_THREAD": str(mv_rows),
    "PYTHONPATH": ".",
  })
  out = run_cmd([sys.executable, __file__, "--probe-kernel",
                 "--dim", str(dim), "--hidden-dim", str(hidden_dim), "--vocab-size", str(vocab_size),
                 "--n-kv-heads", str(n_kv_heads), "--head-dim", str(head_dim)], env)
  return json.loads(out)

def probe_kernel_process(args:argparse.Namespace):
  from tinygrad import Tensor, Device
  from tinygrad.engine.realize import get_program

  kv_dim = args.n_kv_heads * args.head_dim
  dt = "float16"
  # weights reflect decode usage per block (+ final head): q/o(2x), k/v(2x), gate/up(2x), down(1x), lm_head(1x)
  cases = [
    ("proj_dim_dim", args.dim, args.dim, 2.0),
    ("proj_dim_kv", args.dim, kv_dim, 2.0),
    ("ffn_up", args.dim, args.hidden_dim, 2.0),
    ("ffn_down", args.hidden_dim, args.dim, 1.0),
    ("lm_head", args.dim, args.vocab_size, 1.0),
  ]

  ret: dict[str, dict] = {}
  weighted = 0.0
  for name, in_dim, out_dim, w in cases:
    x = Tensor.rand(1, in_dim, dtype=dt).realize()
    wt = Tensor.rand(in_dim, out_dim, dtype=dt).realize()
    y = x @ wt
    p = get_program(y.schedule()[-1].ast, renderer=Device[Device.DEFAULT].renderer, opts=None)
    y.realize(); Device[Device.DEFAULT].synchronize()
    warmup = 2
    iters = 6 if out_dim >= 100000 else 12
    tms = []
    for _ in range(warmup):
      z = x @ wt; z.realize(); Device[Device.DEFAULT].synchronize()
    import time
    for _ in range(iters):
      st = time.perf_counter()
      z = x @ wt; z.realize(); Device[Device.DEFAULT].synchronize()
      tms.append((time.perf_counter() - st) * 1e3)
    mn = min(tms)
    weighted += mn * w
    ret[name] = {"in": in_dim, "out": out_dim, "weight": w, "min_ms": mn, "avg_ms": sum(tms)/len(tms), "opts": [repr(o) for o in p.applied_opts]}

  print(json.dumps({"weighted_min_ms": weighted, "cases": ret}))

def main():
  p = argparse.ArgumentParser(description="Autotune MV_* for tinygrad/apps/llm.py decode.")
  p.add_argument("--model", default="llama3.2:1b")
  p.add_argument("--threads", default="32,64,96,128,160", help="MV_THREADS_PER_ROW candidates")
  p.add_argument("--blocks", default="2,4,8", help="MV_BLOCKSIZE candidates")
  p.add_argument("--rows", default="1,2,4,8", help="MV_ROWS_PER_THREAD candidates")
  p.add_argument("--benchmark-count", type=int, default=16, help="llm.py --benchmark count for e2e")
  p.add_argument("--warmup-skip", type=int, default=4, help="ignore first N benchmark lines")
  p.add_argument("--repeats", type=int, default=2, help="e2e repeats per candidate")
  p.add_argument("--topk", type=int, default=6, help="in auto mode, run e2e only on top K kernel candidates")
  p.add_argument("--mode", choices=["auto", "kernel", "e2e"], default="auto")
  p.add_argument("--full-e2e", action="store_true", help="in auto mode, run e2e on all candidates")
  # Kernel shape parameters (defaults for llama3.2:1b)
  p.add_argument("--dim", type=int, default=2048)
  p.add_argument("--hidden-dim", type=int, default=8192)
  p.add_argument("--vocab-size", type=int, default=128256)
  p.add_argument("--n-kv-heads", type=int, default=8)
  p.add_argument("--head-dim", type=int, default=64)
  p.add_argument("--probe-kernel", action="store_true", help=argparse.SUPPRESS)
  args = p.parse_args()

  if args.probe_kernel:
    probe_kernel_process(args)
    return

  threads, blocks, rows = parse_int_list(args.threads), parse_int_list(args.blocks), parse_int_list(args.rows)
  combos = [{"MV_THREADS_PER_ROW": t, "MV_BLOCKSIZE": b, "MV_ROWS_PER_THREAD": r} for t,b,r in itertools.product(threads, blocks, rows)]
  print(f"candidate_count={len(combos)}")

  kernel_res = []
  if args.mode in ("auto", "kernel"):
    for c in combos:
      kr = run_kernel_probe_once(c["MV_THREADS_PER_ROW"], c["MV_BLOCKSIZE"], c["MV_ROWS_PER_THREAD"],
                                 args.dim, args.hidden_dim, args.vocab_size, args.n_kv_heads, args.head_dim)
      kernel_res.append({**c, **kr})
      print(json.dumps({"phase":"kernel", **c, "weighted_min_ms": kr["weighted_min_ms"]}))
    kernel_res.sort(key=lambda x: x["weighted_min_ms"])

  if args.mode == "kernel":
    best = kernel_res[0]
    print("\nBEST (kernel):", json.dumps({k: best[k] for k in ("MV_THREADS_PER_ROW","MV_BLOCKSIZE","MV_ROWS_PER_THREAD","weighted_min_ms")}, indent=2))
    print(f"export MV_THREADS_PER_ROW={best['MV_THREADS_PER_ROW']} MV_BLOCKSIZE={best['MV_BLOCKSIZE']} MV_ROWS_PER_THREAD={best['MV_ROWS_PER_THREAD']}")
    return

  if args.mode == "e2e":
    e2e_candidates = combos
  else:
    if args.full_e2e:
      e2e_candidates = combos
    else:
      # Keep top-K by kernel score, but also keep per-dimension winners to avoid pruning away good e2e configs.
      seed = kernel_res[:args.topk]
      seed += [min([x for x in kernel_res if x["MV_THREADS_PER_ROW"] == t], key=lambda y: y["weighted_min_ms"]) for t in threads]
      seed += [min([x for x in kernel_res if x["MV_BLOCKSIZE"] == b], key=lambda y: y["weighted_min_ms"]) for b in blocks]
      seed += [min([x for x in kernel_res if x["MV_ROWS_PER_THREAD"] == r], key=lambda y: y["weighted_min_ms"]) for r in rows]
      e2e_candidates = list({(x["MV_THREADS_PER_ROW"], x["MV_BLOCKSIZE"], x["MV_ROWS_PER_THREAD"]): {
        "MV_THREADS_PER_ROW": x["MV_THREADS_PER_ROW"], "MV_BLOCKSIZE": x["MV_BLOCKSIZE"], "MV_ROWS_PER_THREAD": x["MV_ROWS_PER_THREAD"]}
        for x in seed}.values())

  e2e_res = []
  for c in e2e_candidates:
    runs = [run_e2e_once(args.model, args.benchmark_count, args.warmup_skip,
                         c["MV_THREADS_PER_ROW"], c["MV_BLOCKSIZE"], c["MV_ROWS_PER_THREAD"]) for _ in range(args.repeats)]
    med_toks = statistics.median([r["steady_avg_tok_s"] for r in runs])
    mean_toks = sum(r["steady_avg_tok_s"] for r in runs) / len(runs)
    rec = {**c, "steady_avg_tok_s_runs": [r["steady_avg_tok_s"] for r in runs], "steady_avg_tok_s_median": med_toks,
           "steady_avg_tok_s_mean": mean_toks}
    print(json.dumps({"phase":"e2e", **rec}))
    e2e_res.append(rec)
  e2e_res.sort(key=lambda x: x["steady_avg_tok_s_median"], reverse=True)

  print("\nTop results by median steady tok/s:")
  for r in e2e_res[:min(10, len(e2e_res))]:
    print(json.dumps(r))
  best = e2e_res[0]
  print(f"\nBEST: MV_THREADS_PER_ROW={best['MV_THREADS_PER_ROW']} MV_BLOCKSIZE={best['MV_BLOCKSIZE']} MV_ROWS_PER_THREAD={best['MV_ROWS_PER_THREAD']}")
  print(f"export MV_THREADS_PER_ROW={best['MV_THREADS_PER_ROW']} MV_BLOCKSIZE={best['MV_BLOCKSIZE']} MV_ROWS_PER_THREAD={best['MV_ROWS_PER_THREAD']}")

if __name__ == "__main__":
  main()
