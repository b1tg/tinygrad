"""Benchmark FP8Linear vs nn.Linear on BERT-Large shapes (including quantization + backward).

Usage:
  BEAM=3 python extra/gemm/bert_fp8_linear_bench.py
  BEAM=3 python extra/gemm/bert_fp8_linear_bench.py --bs 12
  BEAM=3 python extra/gemm/bert_fp8_linear_bench.py --backward
"""
import argparse, time
from tinygrad import Tensor, dtypes, nn, Device, TinyJit
from tinygrad.helpers import getenv
from extra.fp8.fp8_linear import FP8Linear

HIDDEN = 1024
INTERMEDIATE = 4096
SEQ = 512

SHAPES = [
  ("QKV projection",    HIDDEN, HIDDEN),
  ("Attention output",   HIDDEN, HIDDEN),
  ("FFN up-projection",  HIDDEN, INTERMEDIATE),
  ("FFN down-projection", INTERMEDIATE, HIDDEN),
]

def bench_forward(layer, x, cnt, warmup):
  @TinyJit
  def step():
    return layer(x).realize()
  for _ in range(warmup):
    step()
  times = []
  for _ in range(cnt):
    Device[Device.DEFAULT].synchronize()
    st = time.perf_counter()
    step()
    Device[Device.DEFAULT].synchronize()
    times.append(time.perf_counter() - st)
  return sorted(times)[len(times) // 2]

def bench_forward_backward(layer, x, cnt, warmup):
  layer.weight.requires_grad_(True)
  @TinyJit
  def step():
    y = layer(x)
    y.sum().backward()
    Tensor.realize(y, x.grad, layer.weight.grad)
  for _ in range(warmup):
    step()
  times = []
  for _ in range(cnt):
    Device[Device.DEFAULT].synchronize()
    st = time.perf_counter()
    step()
    Device[Device.DEFAULT].synchronize()
    times.append(time.perf_counter() - st)
  return sorted(times)[len(times) // 2]

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--bs", type=int, default=12)
  parser.add_argument("--seq", type=int, default=SEQ)
  parser.add_argument("--cnt", type=int, default=getenv("CNT", 10))
  parser.add_argument("--warmup", type=int, default=3)
  parser.add_argument("--backward", action="store_true", help="benchmark forward+backward")
  args = parser.parse_args()

  mode = "forward+backward" if args.backward else "forward only"
  bench_fn = bench_forward_backward if args.backward else bench_forward

  print(f"Device: {Device.DEFAULT}, BEAM={getenv('BEAM', 0)}, bs={args.bs}, seq={args.seq}, mode={mode}")
  print(f"Tokens per batch: {args.bs * args.seq}")
  print(f"Iterations: {args.cnt} (warmup: {args.warmup})\n")

  results = []
  for name, in_f, out_f in SHAPES:
    # nn.Linear (half)
    linear_half = nn.Linear(in_f, out_f)
    linear_half.weight = Tensor.randn(out_f, in_f).half()
    linear_half.bias = Tensor.randn(out_f).half()
    x_half = Tensor.randn(args.bs, args.seq, in_f).half().requires_grad_(args.backward)
    t_half = bench_fn(linear_half, x_half, args.cnt, args.warmup)
    x_half.grad = None

    # FP8Linear
    fp8_layer = FP8Linear(in_f, out_f)
    fp8_layer.weight = Tensor.randn(out_f, in_f)
    fp8_layer.bias = Tensor.randn(out_f)
    x_fp8 = Tensor.randn(args.bs, args.seq, in_f).requires_grad_(args.backward)
    t_fp8 = bench_fn(fp8_layer, x_fp8, args.cnt, args.warmup)
    x_fp8.grad = None

    flops = 2 * args.bs * args.seq * in_f * out_f
    if args.backward: flops *= 3  # fwd + grad_input + grad_weight
    tflops_half = flops / t_half / 1e12
    tflops_fp8 = flops / t_fp8 / 1e12
    speedup = t_half / t_fp8

    print(f"  {name:25s}  ({in_f:4d} -> {out_f:4d})  "
          f"half: {t_half*1e3:7.2f} ms ({tflops_half:5.2f} TF)  "
          f"fp8: {t_fp8*1e3:7.2f} ms ({tflops_fp8:5.2f} TF)  "
          f"speedup: {speedup:.2f}x")
    results.append((name, in_f, out_f, t_half, tflops_half, t_fp8, tflops_fp8, speedup))

  print(f"\n  {'TOTAL (per layer)':25s}  {'':16s}  "
        f"half: {sum(r[3] for r in results)*1e3:7.2f} ms  "
        f"fp8: {sum(r[5] for r in results)*1e3:7.2f} ms  "
        f"speedup: {sum(r[3] for r in results)/sum(r[5] for r in results):.2f}x")
