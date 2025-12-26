#!/usr/bin/env python3
"""Breakdown FP8 performance: quantization overhead vs matmul speed"""

import time
from tinygrad import Tensor, dtypes, Device
from tinygrad.helpers import colored
from examples.mlperf.initializers import quantize_to_fp8

# Test shape: (BS*SEQ, hidden) @ (hidden, hidden)
M, K, N = 33792, 1024, 1024

WARMUP = 3
ITERATIONS = 20

def benchmark_op(name, fn, *args):
    """Benchmark a single operation"""
    # Warmup
    for _ in range(WARMUP):
        result = fn(*args)
        if isinstance(result, tuple):
            for r in result:
                r.realize()
        else:
            result.realize()

    # Benchmark
    Device[Device.DEFAULT].synchronize()
    times = []
    for _ in range(ITERATIONS):
        start = time.perf_counter()
        result = fn(*args)
        if isinstance(result, tuple):
            for r in result:
                r.realize()
        else:
            result.realize()
        Device[Device.DEFAULT].synchronize()
        times.append(time.perf_counter() - start)

    avg_time = sum(times) / len(times) * 1000  # ms
    print(f"  {name:<40} {avg_time:>8.3f} ms")
    return avg_time

def main():
    print(f"{colored('='*80, 'blue')}")
    print(f"{colored('FP8 Performance Breakdown', 'blue')}")
    print(f"{colored('='*80, 'blue')}")
    print(f"\nDevice: {Device.DEFAULT}")
    print(f"Shape: ({M}, {K}) @ ({K}, {N})")
    print(f"Iterations: {ITERATIONS}\n")

    # Create tensors
    x = Tensor.rand(M, K, dtype=dtypes.half)
    w = Tensor.rand(N, K, dtype=dtypes.half)

    print(f"{colored('Step 1: Quantization overhead', 'yellow')}")
    # Quantize input
    t_quant_x = benchmark_op("Quantize input (33792, 1024)", quantize_to_fp8, x)

    # Quantize weight
    t_quant_w = benchmark_op("Quantize weight (1024, 1024)", quantize_to_fp8, w)

    total_quant = t_quant_x + t_quant_w
    print(f"  {colored('Total quantization overhead:', 'cyan')} {total_quant:.3f} ms\n")

    # Pre-quantize for matmul tests
    x_fp8, x_scale = quantize_to_fp8(x)
    w_fp8, w_scale = quantize_to_fp8(w)

    print(f"{colored('Step 2: Matmul performance', 'yellow')}")

    # FP16 matmul (baseline)
    def fp16_matmul():
        return x.dot(w.T, dtype=dtypes.float)
    t_fp16 = benchmark_op("FP16 matmul", fp16_matmul)
    gflops_fp16 = (2 * M * N * K) / (t_fp16 / 1000) / 1e9

    # FP8 matmul (no quantization)
    def fp8_matmul():
        return x_fp8.dot(w_fp8.T, dtype=dtypes.float)
    t_fp8 = benchmark_op("FP8 matmul (pre-quantized)", fp8_matmul)
    gflops_fp8 = (2 * M * N * K) / (t_fp8 / 1000) / 1e9

    # FP8 matmul with scaling
    def fp8_matmul_scaled():
        y = x_fp8.dot(w_fp8.T, dtype=dtypes.float)
        return y * x_scale * w_scale
    t_fp8_scaled = benchmark_op("FP8 matmul + scaling", fp8_matmul_scaled)
    gflops_fp8_scaled = (2 * M * N * K) / (t_fp8_scaled / 1000) / 1e9

    print(f"\n  FP16 GFLOPS: {gflops_fp16:.1f}")
    print(f"  FP8 GFLOPS (no scale): {gflops_fp8:.1f}")
    print(f"  FP8 GFLOPS (with scale): {gflops_fp8_scaled:.1f}\n")

    print(f"{colored('Step 3: Complete FP8Linear operation', 'yellow')}")

    # Full FP8Linear (quantize + matmul + scale)
    def full_fp8_linear():
        x1, s_x = quantize_to_fp8(x)
        w1, s_w = quantize_to_fp8(w)
        y = x1.dot(w1.T, dtype=dtypes.float)
        return y * s_x * s_w
    t_full = benchmark_op("Full FP8Linear (quant + matmul + scale)", full_fp8_linear)

    print(f"\n{colored('='*80, 'blue')}")
    print(f"{colored('ANALYSIS', 'blue')}")
    print(f"{colored('='*80, 'blue')}\n")

    # Breakdown
    print(f"Time breakdown for FP8Linear:")
    print(f"  Quantization:        {total_quant:>8.3f} ms ({total_quant/t_full*100:>5.1f}%)")
    print(f"  FP8 matmul+scale:    {t_fp8_scaled:>8.3f} ms ({t_fp8_scaled/t_full*100:>5.1f}%)")
    print(f"  Other overhead:      {t_full - total_quant - t_fp8_scaled:>8.3f} ms ({(t_full - total_quant - t_fp8_scaled)/t_full*100:>5.1f}%)")
    print(f"  {'-'*60}")
    print(f"  Total:               {t_full:>8.3f} ms\n")

    # Speedup comparison
    speedup = t_fp16 / t_full
    color = 'green' if speedup > 1.0 else 'red'
    print(f"FP8Linear vs FP16: {colored(f'{speedup:.2f}x', color)} ({'faster' if speedup > 1.0 else 'slower'})")

    # What if we cached quantized weights?
    t_cached = t_quant_x + t_fp8_scaled
    speedup_cached = t_fp16 / t_cached
    color_cached = 'green' if speedup_cached > 1.0 else 'red'
    print(f"FP8Linear (cached weights) vs FP16: {colored(f'{speedup_cached:.2f}x', color_cached)} ({'faster' if speedup_cached > 1.0 else 'slower'})\n")

    # Recommendations
    print(f"{colored('RECOMMENDATIONS:', 'yellow')}")

    if t_fp8 < t_fp16:
        print(f"  ✓ FP8 matmul IS faster than FP16 ({t_fp8:.3f}ms vs {t_fp16:.3f}ms)")
        print(f"    BUT quantization overhead ({total_quant:.3f}ms) makes it slower overall")
        print(f"    {colored('→ Solution: Cache quantized weights!', 'green')}")
    else:
        print(f"  ✗ FP8 matmul is NOT faster than FP16 ({t_fp8:.3f}ms vs {t_fp16:.3f}ms)")
        print(f"    {colored('→ FP8 GEMM kernel may not be using hardware acceleration', 'red')}")
        print(f"    {colored('→ Check if rocBLAS FP8 kernels are available and being used', 'red')}")

    if total_quant > t_fp16:
        pct = total_quant / t_fp16 * 100
        print(f"  ✗ Quantization overhead ({total_quant:.3f}ms) is {pct:.0f}% of FP16 matmul time!")
        print(f"    {colored('→ Quantization is too expensive for this use case', 'red')}")

if __name__ == "__main__":
    main()
