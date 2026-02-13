"""Benchmark matmul shapes used in BERT-Large FP8 training.

BERT-Large config: hidden=1024, intermediate=4096, heads=16, seq=512

Usage:
  python extra/gemm/bert_fp8_matmul.py                    # benchmark all shapes
  python extra/gemm/bert_fp8_matmul.py --shape qkv        # benchmark specific shape
  python extra/gemm/bert_fp8_matmul.py --bs 12             # custom batch size per GPU
  python extra/gemm/bert_fp8_matmul.py --backward          # include backward pass shapes
  HALF=1 python extra/gemm/bert_fp8_matmul.py              # compare with fp16 baseline
  FP8E4M3FNUZ=1 python extra/gemm/bert_fp8_matmul.py      # force specific fp8 variant
"""
import argparse, time
import numpy as np
from tinygrad import Tensor, dtypes, Device
from tinygrad.helpers import getenv
from tinygrad.dtype import _to_np_dtype

# BERT-Large dimensions
HIDDEN = 1024
INTERMEDIATE = 4096
HEADS = 16
HEAD_DIM = HIDDEN // HEADS  # 64
SEQ = 512

def get_dtype():
  if getenv("HALF"): return dtypes.half
  if getenv("BFLOAT16"): return dtypes.bfloat16
  if getenv("FP8E4M3"): return dtypes.fp8e4m3
  if getenv("FP8E5M2"): return dtypes.fp8e5m2
  if getenv("FP8E4M3FNUZ"): return dtypes.fp8e4m3fnuz
  if getenv("FP8E5M2FNUZ"): return dtypes.fp8e5m2fnuz
  return dtypes.fp8e4m3_hw()

def init_matrix(rows, cols, dtype):
  np_dtype = _to_np_dtype(dtype) or np.float32
  return Tensor(np.random.default_rng().random((rows, cols), dtype=np.float32).astype(np_dtype) - 0.5).cast(dtype).realize()

def benchmark_matmul(M, K, N, dtype, acc_dtype=None, cnt=10, warmup=3, label=""):
  a, b = init_matrix(M, K, dtype), init_matrix(K, N, dtype)
  # warmup
  for _ in range(warmup):
    c = a.matmul(b, dtype=acc_dtype).realize()
  # benchmark
  times = []
  for _ in range(cnt):
    Device[Device.DEFAULT].synchronize()
    st = time.perf_counter()
    c = a.matmul(b, dtype=acc_dtype).realize()
    Device[Device.DEFAULT].synchronize()
    times.append(time.perf_counter() - st)
  median_s = sorted(times)[len(times) // 2]
  flops = 2 * M * N * K
  tflops = flops / median_s / 1e12
  print(f"  {label:40s}  ({M:5d}, {K:5d}) x ({K:5d}, {N:5d})  "
        f"median: {median_s*1e3:7.2f} ms  {tflops:6.2f} TFLOPS")
  return median_s, tflops

def get_shapes(bs, seq=SEQ):
  """Return all matmul shapes in BERT-Large training as (M, K, N, label)."""
  tokens = bs * seq  # flattened batch*seq dimension
  shapes = {}

  # === Forward pass (linear layers) ===
  shapes["qkv"] = (tokens, HIDDEN, HIDDEN, "fwd: QKV projection (x3)")
  shapes["attn_out"] = (tokens, HIDDEN, HIDDEN, "fwd: attention output proj")
  shapes["ffn_up"] = (tokens, HIDDEN, INTERMEDIATE, "fwd: FFN up-projection")
  shapes["ffn_down"] = (tokens, INTERMEDIATE, HIDDEN, "fwd: FFN down-projection")

  # === Backward pass: grad_input = grad @ weight ===
  shapes["bwd_gi_qkv"] = (tokens, HIDDEN, HIDDEN, "bwd_gi: QKV grad_input (x3)")
  shapes["bwd_gi_attn_out"] = (tokens, HIDDEN, HIDDEN, "bwd_gi: attn_out grad_input")
  shapes["bwd_gi_ffn_up"] = (tokens, INTERMEDIATE, HIDDEN, "bwd_gi: FFN up grad_input")
  shapes["bwd_gi_ffn_down"] = (tokens, HIDDEN, INTERMEDIATE, "bwd_gi: FFN down grad_input")

  # === Backward pass: grad_weight = grad.T @ input ===
  shapes["bwd_gw_qkv"] = (HIDDEN, tokens, HIDDEN, "bwd_gw: QKV grad_weight (x3)")
  shapes["bwd_gw_attn_out"] = (HIDDEN, tokens, HIDDEN, "bwd_gw: attn_out grad_weight")
  shapes["bwd_gw_ffn_up"] = (INTERMEDIATE, tokens, HIDDEN, "bwd_gw: FFN up grad_weight")
  shapes["bwd_gw_ffn_down"] = (HIDDEN, tokens, INTERMEDIATE, "bwd_gw: FFN down grad_weight")

  return shapes

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Benchmark BERT-Large FP8 matmul shapes")
  parser.add_argument("--bs", type=int, default=8, help="batch size per GPU (default: 8)")
  parser.add_argument("--seq", type=int, default=SEQ, help=f"sequence length (default: {SEQ})")
  parser.add_argument("--shape", type=str, default=None, help="benchmark specific shape (e.g. qkv, ffn_up)")
  parser.add_argument("--backward", action="store_true", help="include backward pass shapes")
  parser.add_argument("--cnt", type=int, default=getenv("CNT", 10), help="iterations per shape")
  parser.add_argument("--warmup", type=int, default=3, help="warmup iterations")
  parser.add_argument("--acc-float", action="store_true", help="accumulate in float32 (default for fp8)")
  args = parser.parse_args()

  dtype = get_dtype()
  acc_dtype = dtypes.float if (args.acc_float or dtype in (dtypes.fp8e4m3, dtypes.fp8e5m2, dtypes.fp8e4m3fnuz, dtypes.fp8e5m2fnuz)) else None
  shapes = get_shapes(args.bs, args.seq)

  print(f"Device: {Device.DEFAULT}, dtype: {dtype}, acc_dtype: {acc_dtype or 'default'}")
  print(f"BERT-Large: hidden={HIDDEN}, intermediate={INTERMEDIATE}, heads={HEADS}, seq={args.seq}, bs={args.bs}")
  print(f"Tokens per batch: {args.bs * args.seq}")
  print(f"Iterations: {args.cnt} (warmup: {args.warmup})")
  print()

  if args.shape:
    if args.shape not in shapes:
      print(f"Unknown shape '{args.shape}'. Available: {', '.join(shapes.keys())}")
      raise SystemExit(1)
    M, K, N, label = shapes[args.shape]
    benchmark_matmul(M, K, N, dtype, acc_dtype, args.cnt, args.warmup, label)
  else:
    # forward pass
    print("=== Forward Pass (per layer, x24 layers) ===")
    fwd_keys = ["qkv", "attn_out", "ffn_up", "ffn_down"]
    for key in fwd_keys:
      M, K, N, label = shapes[key]
      benchmark_matmul(M, K, N, dtype, acc_dtype, args.cnt, args.warmup, label)

    if args.backward:
      print("\n=== Backward Pass: grad_input (per layer, x24 layers) ===")
      for key in [k for k in shapes if k.startswith("bwd_gi")]:
        M, K, N, label = shapes[key]
        benchmark_matmul(M, K, N, dtype, acc_dtype, args.cnt, args.warmup, label)

      print("\n=== Backward Pass: grad_weight (per layer, x24 layers) ===")
      for key in [k for k in shapes if k.startswith("bwd_gw")]:
        M, K, N, label = shapes[key]
        benchmark_matmul(M, K, N, dtype, acc_dtype, args.cnt, args.warmup, label)
