#!/usr/bin/env python3
"""
Comprehensive FP8Linear benchmark tool.

Compares FP8 quantization vs float32/float16 for shapes found in:
- BERT-large
- Llama-8B (prefill and decode)
- Llama-405B (prefill and decode)

Reports:
- Forward latency (ms)
- Forward + backward latency (ms)
- GFLOPS
- Speedup vs FP32/FP16
- Quantization overhead breakdown
"""

import time
from tinygrad import Tensor, dtypes, Device
from tinygrad.nn import Linear
from examples.mlperf.initializers import LinearBert as Linear
from tinygrad.device import is_dtype_supported
from extra.fp8 import FP8Linear
import os

# Benchmark settings
WARMUP = 3
ITERATIONS = 10


def benchmark_layer(layer_class, input_shape, in_features, out_features, dtype=dtypes.float32):
  # if dtype == dtypes.half:
  #   os.environ["DEFAULT_FLOAT"] ="HALF"
  # else:
  #   os.environ["DEFAULT_FLOAT"] ="FLOAT"
  """
  Benchmark a linear layer.

  Returns:
      dict with 'fwd_ms', 'fwd_bwd_ms', 'gflops_fwd'
  """
  # Create model
  model = layer_class(in_features, out_features, bias=True)
  model.weight.assign(Tensor.randn(out_features, in_features, dtype=dtype) * 0.02)
  if model.bias is not None:
    model.bias.assign(Tensor.zeros(out_features, dtype=dtype))
  model.weight.requires_grad = True
  if model.bias is not None:
    model.bias.requires_grad = True

  # Create input
  x = Tensor.randn(*input_shape, dtype=dtype, requires_grad=True)
  print(f"{dtype=}, {x.dtype=}, {model.weight.dtype=}, {model.bias.dtype=}")

  # Warmup
  for _ in range(WARMUP):
    y = model(x)
    y.sum().backward()
    if x.grad is not None:
      x.grad.realize()

  # Benchmark forward only
  Device[Device.DEFAULT].synchronize()
  times_fwd = []
  for _ in range(ITERATIONS):
    start = time.perf_counter()
    y = model(x)
    y.realize()
    Device[Device.DEFAULT].synchronize()
    times_fwd.append(time.perf_counter() - start)

  avg_fwd_ms = sum(times_fwd) / len(times_fwd) * 1000

  # Benchmark forward + backward
  Device[Device.DEFAULT].synchronize()
  times_fwd_bwd = []
  for _ in range(ITERATIONS):
    # Clear gradients
    model.weight.grad = None
    if model.bias is not None:
      model.bias.grad = None
    x.grad = None

    start = time.perf_counter()
    y = model(x)
    y.realize()
    y.sum().backward()
    if model.weight.grad is not None:
      model.weight.grad.realize()
    if model.bias is not None and model.bias.grad is not None:
      model.bias.grad.realize()
    if x.grad is not None:
      x.grad.realize()
    Device[Device.DEFAULT].synchronize()
    times_fwd_bwd.append(time.perf_counter() - start)

  avg_fwd_bwd_ms = sum(times_fwd_bwd) / len(times_fwd_bwd) * 1000

  # Calculate GFLOPS
  # Assuming 3D input: (batch, seq, features)
  if len(input_shape) == 3:
    batch, seq, _ = input_shape
    M = batch * seq
  else:
    # 2D input: (batch, features)
    M = input_shape[0]

  K = in_features
  N = out_features

  # Forward matmul FLOPs: 2 * M * K * N
  flops_fwd = 2 * M * K * N
  gflops_fwd = flops_fwd / (avg_fwd_ms / 1000) / 1e9

  return {
    'fwd_ms': avg_fwd_ms,
    'fwd_bwd_ms': avg_fwd_bwd_ms,
    'gflops_fwd': gflops_fwd,
  }


def print_comparison_table(results_by_dtype):
  """Print a formatted comparison table"""
  # Extract base results
  fp8_res = results_by_dtype.get('fp8')
  fp32_res = results_by_dtype.get('fp32')
  fp16_res = results_by_dtype.get('fp16')

  print(f"{'':20} {'Forward (ms)':>15} {'Fwd+Bwd (ms)':>15} {'GFLOPS':>12} {'Speedup':>10}")
  print("-" * 75)

  # FP32 baseline
  if fp32_res:
    print(f"{'FP32':20} {fp32_res['fwd_ms']:>15.2f} {fp32_res['fwd_bwd_ms']:>15.2f} "
          f"{fp32_res['gflops_fwd']:>12.2f} {'1.00x':>10}")

  # FP16
  if fp16_res:
    speedup_fwd = fp32_res['fwd_ms'] / fp16_res['fwd_ms'] if fp32_res else 1.0
    print(f"{'FP16':20} {fp16_res['fwd_ms']:>15.2f} {fp16_res['fwd_bwd_ms']:>15.2f} "
          f"{fp16_res['gflops_fwd']:>12.2f} {speedup_fwd:>9.2f}x")

  # FP8
  if fp8_res:
    speedup_vs_fp32 = fp32_res['fwd_ms'] / fp8_res['fwd_ms'] if fp32_res else 1.0
    speedup_vs_fp16 = fp16_res['fwd_ms'] / fp8_res['fwd_ms'] if fp16_res else 1.0
    speedup_str = f"{speedup_vs_fp32:.2f}x"
    if fp16_res:
      speedup_str += f" ({speedup_vs_fp16:.2f}x vs FP16)"

    print(f"{'FP8':20} {fp8_res['fwd_ms']:>15.2f} {fp8_res['fwd_bwd_ms']:>15.2f} "
          f"{fp8_res['gflops_fwd']:>12.2f} {speedup_str:>10}")

  print()


def benchmark_shape(name, input_shape, in_features, out_features):
  """Benchmark a specific shape with multiple dtypes"""
  print(f"\n{'='*75}")
  print(f"Shape: {name}")
  print(f"  Input: {input_shape}, Linear: ({in_features}, {out_features})")
  print(f"{'='*75}")

  results = {}

  # Benchmark FP32
  # print("\n[1/3] Benchmarking FP32...")
  # try:
  #   res = benchmark_layer(Linear, input_shape, in_features, out_features, dtype=dtypes.float32)
  #   results['fp32'] = res
  # except Exception as e:
  #   print(f"  FP32 failed: {e}")

  # Benchmark FP16
  print("[2/3] Benchmarking FP16...")
  try:
    res = benchmark_layer(Linear, input_shape, in_features, out_features, dtype=dtypes.float16)
    results['fp16'] = res
  except Exception as e:
    print(f"  FP16 failed: {e}")

  # Benchmark FP8
  if is_dtype_supported(dtypes.fp8e4m3):
    print("[3/3] Benchmarking FP8...")
    try:
      res = benchmark_layer(FP8Linear, input_shape, in_features, out_features, dtype=dtypes.float32)
      results['fp8'] = res
    except Exception as e:
      print(f"  FP8 failed: {e}")
  else:
    print("[3/3] FP8 not supported on this device, skipping")

  # Print results
  print("\nResults:")
  print_comparison_table(results)

  return results


def main():
  """Run all benchmarks"""
  print("="*75)
  print("FP8Linear Comprehensive Benchmark")
  print(f"Device: {Device.DEFAULT}")
  print(f"Warmup iterations: {WARMUP}")
  print(f"Benchmark iterations: {ITERATIONS}")
  print("="*75)

  all_results = {}

  # ========================================================================
  # BERT-large shapes
  # ========================================================================
  print("\n" + "="*75)
  print("BERT-LARGE (hidden=1024, intermediate=4096, batch=32, seq=512)")
  print("="*75)

  all_results['bert_qkv'] = benchmark_shape(
    "BERT-large QKV projection",
    input_shape=(1024, 512, 1024),
    in_features=1024,
    out_features=1024
  )

  all_results['bert_ffn_up'] = benchmark_shape(
    "BERT-large FFN up (1024 -> 4096)",
    input_shape=(1024, 512, 1024),
    in_features=1024,
    out_features=4096
  )

  all_results['bert_ffn_down'] = benchmark_shape(
    "BERT-large FFN down (4096 -> 1024)",
    input_shape=(1024, 512, 4096),
    in_features=4096,
    out_features=1024
  )
  if 0:
    # ========================================================================
    # Llama-8B shapes
    # ========================================================================
    print("\n" + "="*75)
    print("LLAMA-8B (dim=4096, hidden_dim=14336)")
    print("="*75)

    all_results['llama8b_prefill_qkv'] = benchmark_shape(
      "Llama-8B Prefill QKV (seq=512)",
      input_shape=(1, 512, 4096),
      in_features=4096,
      out_features=4096
    )

    all_results['llama8b_prefill_ffn'] = benchmark_shape(
      "Llama-8B Prefill FFN (seq=512)",
      input_shape=(1, 512, 4096),
      in_features=4096,
      out_features=14336
    )

    all_results['llama8b_decode_qkv'] = benchmark_shape(
      "Llama-8B Decode QKV (seq=1)",
      input_shape=(1, 1, 4096),
      in_features=4096,
      out_features=4096
    )

    all_results['llama8b_decode_ffn'] = benchmark_shape(
      "Llama-8B Decode FFN (seq=1)",
      input_shape=(1, 1, 4096),
      in_features=4096,
      out_features=14336
    )

    # ========================================================================
    # Llama-405B shapes
    # ========================================================================
    print("\n" + "="*75)
    print("LLAMA-405B (dim=16384, hidden_dim=53248)")
    print("="*75)

    all_results['llama405b_prefill_qkv'] = benchmark_shape(
      "Llama-405B Prefill QKV (seq=512)",
      input_shape=(1, 512, 16384),
      in_features=16384,
      out_features=16384
    )

    all_results['llama405b_prefill_ffn'] = benchmark_shape(
      "Llama-405B Prefill FFN (seq=512)",
      input_shape=(1, 512, 16384),
      in_features=16384,
      out_features=53248
    )

    all_results['llama405b_decode_qkv'] = benchmark_shape(
      "Llama-405B Decode QKV (seq=1)",
      input_shape=(1, 1, 16384),
      in_features=16384,
      out_features=16384
    )

    all_results['llama405b_decode_ffn'] = benchmark_shape(
      "Llama-405B Decode FFN (seq=1)",
      input_shape=(1, 1, 16384),
      in_features=16384,
      out_features=53248
    )

  # ========================================================================
  # Summary
  # ========================================================================
  print("\n" + "="*75)
  print("SUMMARY: Average Speedup vs fp16")
  print("="*75)

  fp8_speedups = []
  for name, results in all_results.items():
    if 'fp8' in results and 'fp16' in results:
      speedup = results['fp16']['fwd_ms'] / results['fp8']['fwd_ms']
      fp8_speedups.append(speedup)
      print(f"{name:30} {speedup:>6.2f}x")

  if fp8_speedups:
    print(f"\n{'Average FP8 speedup':30} {sum(fp8_speedups)/len(fp8_speedups):>6.2f}x")
  print()


if __name__ == '__main__':
  main()
