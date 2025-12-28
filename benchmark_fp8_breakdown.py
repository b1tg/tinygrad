#!/usr/bin/env python3
"""Breakdown FP8 performance: quantization overhead vs matmul speed

Compare power-of-2 scaling (DeepSeek style) vs arbitrary scaling.

"""

"""
FP8LinearBertBasic x.shape=(1024, 512, 1024), self.weight.shape=(4096, 1024)
FP8LinearBertBasic x.shape=(1024, 512, 4096), self.weight.shape=(1024, 4096)
"""

import time
from tinygrad import Tensor, dtypes, Device
from tinygrad.helpers import colored
# from examples.mlperf.initializers import quantize_to_fp8
from extra.fp8 import quantize_to_fp8, FP8Linear
from examples.mlperf.initializers import LinearBert

# 33792= 66 * 512
# Test shape: (BS*SEQ, hidden) @ (hidden, hidden)
M, K, N = 33792, 1024, 1024
M, K, N = 33792, 8192, 8192
# M, K, N = 1024*512, 2048, 2048

# M, K, N = 33792, 4096, 1024
# M, K, N = 1024*512, 1024, 4096
batch_size = 1024
# batch_size = 128

M, K, N = 1024*512, 1024, 4096
M, K, N = batch_size*512, 4096, 1024

WARMUP = 3
ITERATIONS = 20

# Wrapper functions for different scaling methods
def quantize_pow2(x, axis=None, dtype=dtypes.fp8e4m3):
  """Power-of-2 scaling (DeepSeek style)"""
  return quantize_to_fp8(x, axis=axis, dtype=dtype, power_of_2_scale=True)

def quantize_arbitrary(x, axis=None, dtype=dtypes.fp8e4m3):
  """Arbitrary scaling (original method)"""
  return quantize_to_fp8(x, axis=axis, dtype=dtype, power_of_2_scale=False)

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
    print(f"{colored('FP8 Performance Breakdown - Forward & Backward', 'blue')}")
    print(f"{colored('='*80, 'blue')}")
    print(f"\nDevice: {Device.DEFAULT}")
    print(f"Shape: ({M}, {K}) @ ({K}, {N})")
    print(f"Iterations: {ITERATIONS}\n")

    # Create tensors
    x = Tensor.rand(M, K, dtype=dtypes.half, requires_grad=True)
    w = Tensor.rand(N, K, dtype=dtypes.half, requires_grad=True)

    print(f"{colored('Step 1: Quantization overhead comparison', 'yellow')}")
    print(f"\n  {colored('Power-of-2 scaling (DeepSeek style):', 'cyan')}")
    # Quantize input - power of 2
    t_quant_x_pow2 = benchmark_op(f"Quantize input {x.shape} (pow2)", quantize_pow2, x)
    # Quantize weight - power of 2
    t_quant_w_pow2 = benchmark_op(f"Quantize weight {w.shape} (pow2)", quantize_pow2, w)
    total_quant_pow2 = t_quant_x_pow2 + t_quant_w_pow2
    print(f"  {colored('Total quantization (pow2):', 'cyan')} {total_quant_pow2:.3f} ms")

    print(f"\n  {colored('Arbitrary scaling (original):', 'cyan')}")
    # Quantize input - arbitrary
    t_quant_x_arb = benchmark_op(f"Quantize input {x.shape} (arbitrary)", quantize_arbitrary, x)
    # Quantize weight - arbitrary
    t_quant_w_arb = benchmark_op(f"Quantize weight {w.shape} (arbitrary)", quantize_arbitrary, w)
    total_quant_arb = t_quant_x_arb + t_quant_w_arb
    print(f"  {colored('Total quantization (arbitrary):', 'cyan')} {total_quant_arb:.3f} ms")

    # Compare
    quant_diff = total_quant_pow2 - total_quant_arb
    quant_ratio = total_quant_pow2 / total_quant_arb if total_quant_arb > 0 else float('inf')
    color = 'green' if quant_ratio <= 1.05 else 'yellow' if quant_ratio <= 1.2 else 'red'
    print(f"\n  {colored('Pow2 vs Arbitrary:', color)} {quant_ratio:.3f}x ({'+' if quant_diff > 0 else ''}{quant_diff:.3f} ms)")

    # Use power-of-2 for subsequent tests (default)
    total_quant = total_quant_pow2
    t_quant_x = t_quant_x_pow2
    t_quant_w = t_quant_w_pow2

    # Pre-quantize for matmul tests (using power-of-2 scaling)
    x_fp8, x_scale = quantize_pow2(x)
    w_fp8, w_scale = quantize_pow2(w)
    # x_fp8, x_scale = quantize_arbitrary(x)
    # w_fp8, w_scale = quantize_arbitrary(w)

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
        return y.contiguous() * x_scale * w_scale
    t_fp8_scaled = benchmark_op("FP8 matmul + scaling", fp8_matmul_scaled)
    # FP8 matmul with scaling
    # def fp8_matmul_scaled_0():
    #     y = x_fp8_0.dot(x_fp8_0.T, dtype=dtypes.float)
    #     return y * x_scale_0 * w_scale_0
    # t_fp8_scaled = benchmark_op("FP8 matmul + scaling (arbitary)", fp8_matmul_scaled_0)
    gflops_fp8_scaled = (2 * M * N * K) / (t_fp8_scaled / 1000) / 1e9

    print(f"\n  FP16 GFLOPS: {gflops_fp16:.1f}")
    print(f"  FP8 GFLOPS (no scale): {gflops_fp8:.1f}")
    print(f"  FP8 GFLOPS (with scale): {gflops_fp8_scaled:.1f}\n")

    print(f"{colored('Step 3: Complete FP8Linear operation', 'yellow')}")

    # Full FP8Linear with power-of-2 scaling
    def full_fp8_linear_pow2():
        x1, s_x = quantize_pow2(x)
        w1, s_w = quantize_pow2(w)
        y = x1.dot(w1.T, dtype=dtypes.float)
        return y.contiguous() * s_x * s_w
    t_full_pow2 = benchmark_op("Full FP8Linear (pow2 scale)", full_fp8_linear_pow2)

    # Full FP8Linear with arbitrary scaling
    def full_fp8_linear_arb():
        x1, s_x = quantize_arbitrary(x)
        w1, s_w = quantize_arbitrary(w)
        y = x1.dot(w1.T, dtype=dtypes.float)
        return y.contiguous() * s_x * s_w
    t_full_arb = benchmark_op("Full FP8Linear (arbitrary scale)", full_fp8_linear_arb)

    # Compare
    full_diff = t_full_pow2 - t_full_arb
    full_ratio = t_full_pow2 / t_full_arb if t_full_arb > 0 else float('inf')
    color = 'green' if full_ratio <= 1.05 else 'yellow' if full_ratio <= 1.2 else 'red'
    print(f"\n  {colored('Full pow2 vs arbitrary:', color)} {full_ratio:.3f}x ({'+' if full_diff > 0 else ''}{full_diff:.3f} ms)")

    # Use pow2 for subsequent analysis
    t_full = t_full_pow2
    # return

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

    # ========================================================================
    # BACKWARD PASS ANALYSIS (using custom kernel)
    # ========================================================================
    print(f"\n{colored('='*80, 'blue')}")
    print(f"{colored('BACKWARD PASS ANALYSIS (Custom Kernel)', 'blue')}")
    print(f"{colored('='*80, 'blue')}\n")

    # Reshape to 3D for FP8Linear (batch, seq, features)
    # batch_size = 16
    seq_len = M // batch_size
    x_3d = x.reshape(batch_size, seq_len, K)
    print(f"{x_3d.shape=}")

    # Create FP8Linear layer with custom kernel
    fp8_layer = FP8Linear(K, N, bias=False, use_custom_kernel=False)
    fp8_layer.weight = w  # FP8Linear expects (out_features, in_features)

    fp16_layer = LinearBert(K, N, bias=False)
    fp16_layer.weight = w

    print(f"{colored('Step 4: Custom kernel forward+backward', 'yellow')}")

    # FP8 custom kernel forward+backward
    def fp8_custom_fwd_bwd():
        x_3d_grad = x_3d.detach()
        x_3d_grad.requires_grad = True

        # Forward pass with custom kernel
        y = fp8_layer(x_3d_grad)

        # Create gradient for backward
        # grad_output = Tensor.ones_like(y)
        # print(f"{grad_output.shape=}")

        # Backward pass (triggers custom_linear_backward)
        y.sum().backward()

        return y, x_3d_grad.grad, fp8_layer.weight.grad

    t_fp8_fwd_bwd = benchmark_op("FP8 custom kernel (fwd+bwd)", fp8_custom_fwd_bwd)

    # FP16 baseline forward+backward
    def fp16_fwd_bwd_():
        x_3d_grad = x_3d.detach().cast(dtypes.half)
        x_3d_grad.requires_grad = True
        w_grad = w.detach().cast(dtypes.half)
        w_grad.requires_grad = True

        # Forward: matmul
        # y = x_3d_grad.cast(dtypes.float).dot(w_grad.T.cast(dtypes.float), dtype=dtypes.float)
        y = x_3d_grad.dot(w_grad.T, dtype=dtypes.float)

        # Backward
        # grad_output = Tensor.ones_like(y)
        # y.backward(grad_output)
        y.sum().backward()

        return y, x_3d_grad.grad, w_grad.grad
    def fp16_fwd_bwd():
        x_3d_grad = x_3d.detach().cast(dtypes.half)
        x_3d_grad.requires_grad = True

        # Forward pass with custom kernel
        y = fp16_layer(x_3d_grad)

        # Create gradient for backward
        # grad_output = Tensor.empty_like(y)
        # print(f"{grad_output.shape=}")

        # Backward pass (triggers custom_linear_backward)
        # y.backward(grad_output)
        y.sum().backward()

        return y, x_3d_grad.grad, fp16_layer.weight.grad

    t_fp16_fwd_bwd = benchmark_op("FP16 baseline (fwd+bwd)", fp16_fwd_bwd)
    return

    print(f"\n{colored('Step 5: Backward pass component breakdown', 'yellow')}")

    # Create gradient tensor (output gradient) - 2D for simpler analysis
    grad_out = Tensor.rand(M, N, dtype=dtypes.half)

    # Quantize gradient - compare both methods
    t_quant_grad_pow2 = benchmark_op("Quantize gradient (pow2)", quantize_pow2, grad_out)
    t_quant_grad_arb = benchmark_op("Quantize gradient (arbitrary)", quantize_arbitrary, grad_out)
    t_quant_grad = t_quant_grad_pow2  # Use pow2 for subsequent tests

    # Pre-quantize for component tests
    grad_fp8, grad_scale = quantize_pow2(grad_out)

    # FP8 backward - input gradient: grad_fp8 @ w_fp8
    def fp8_backward_input():
        y = grad_fp8.dot(w_fp8, dtype=dtypes.float)
        return y * grad_scale
    t_fp8_bwd_input = benchmark_op("FP8 backward (input grad)", fp8_backward_input)
    gflops_fp8_bwd_input = (2 * M * N * K) / (t_fp8_bwd_input / 1000) / 1e9

    # FP8 backward - weight gradient: grad_fp8.T @ x_fp8
    def fp8_backward_weight():
        y = grad_fp8.T.dot(x_fp8, dtype=dtypes.float)
        return y * grad_scale
    t_fp8_bwd_weight = benchmark_op("FP8 backward (weight grad)", fp8_backward_weight)
    gflops_fp8_bwd_weight = (2 * N * M * K) / (t_fp8_bwd_weight / 1000) / 1e9

    print(f"\n  FP8 backward input GFLOPS: {gflops_fp8_bwd_input:.1f}")
    print(f"  FP8 backward weight GFLOPS: {gflops_fp8_bwd_weight:.1f}\n")

    # Estimate component times
    t_bwd_matmuls = t_fp8_bwd_input + t_fp8_bwd_weight
    t_full_bwd_estimate = t_quant_grad + t_bwd_matmuls

    print(f"\n{colored('='*80, 'blue')}")
    print(f"{colored('BACKWARD PASS BREAKDOWN', 'blue')}")
    print(f"{colored('='*80, 'blue')}\n")

    # Component breakdown (estimated from individual measurements)
    print(f"Component breakdown (from separate measurements):")
    print(f"  Gradient quantization:  {t_quant_grad:>8.3f} ms ({t_quant_grad/t_full_bwd_estimate*100:>5.1f}%)")
    print(f"  FP8 matmuls+scale:      {t_bwd_matmuls:>8.3f} ms ({t_bwd_matmuls/t_full_bwd_estimate*100:>5.1f}%)")
    print(f"  {'-'*60}")
    print(f"  Estimated total:        {t_full_bwd_estimate:>8.3f} ms\n")

    # Custom kernel actual timings
    # Extract backward time: total - forward
    t_fp8_bwd_actual = t_fp8_fwd_bwd - t_full  # Approximate backward time
    t_fp16_bwd_actual = t_fp16_fwd_bwd - t_fp16

    print(f"Actual end-to-end timings:")
    print(f"  FP8 custom kernel (fwd+bwd):   {t_fp8_fwd_bwd:>8.3f} ms")
    print(f"  FP16 baseline (fwd+bwd):       {t_fp16_fwd_bwd:>8.3f} ms")
    print(f"  {'-'*60}")
    print(f"  Estimated FP8 backward only:   {t_fp8_bwd_actual:>8.3f} ms")
    print(f"  Estimated FP16 backward only:  {t_fp16_bwd_actual:>8.3f} ms\n")

    # Speedup comparisons
    speedup_fwd_bwd = t_fp16_fwd_bwd / t_fp8_fwd_bwd
    color_fwd_bwd = 'green' if speedup_fwd_bwd > 1.0 else 'red'
    print(f"FP8 custom kernel vs FP16 (fwd+bwd): {colored(f'{speedup_fwd_bwd:.2f}x', color_fwd_bwd)} ({'faster' if speedup_fwd_bwd > 1.0 else 'slower'})")

    if t_fp8_bwd_actual > 0 and t_fp16_bwd_actual > 0:
        speedup_bwd = t_fp16_bwd_actual / t_fp8_bwd_actual
        color_bwd = 'green' if speedup_bwd > 1.0 else 'red'
        print(f"FP8 backward vs FP16 backward:       {colored(f'{speedup_bwd:.2f}x', color_bwd)} ({'faster' if speedup_bwd > 1.0 else 'slower'})\n")

    # Recommendations
    print(f"{colored('RECOMMENDATIONS:', 'yellow')}")

    print(f"\n  {colored('Forward pass:', 'cyan')}")
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

    print(f"\n  {colored('Backward pass (custom kernel):', 'cyan')}")
    if speedup_fwd_bwd > 1.0:
        print(f"  ✓ FP8 custom kernel IS faster for fwd+bwd ({speedup_fwd_bwd:.2f}x speedup)")
    else:
        print(f"  ✗ FP8 custom kernel is slower for fwd+bwd ({speedup_fwd_bwd:.2f}x)")
        print(f"    {colored('→ Quantization overhead dominates the benefits', 'yellow')}")

    bwd_quant_pct = t_quant_grad / t_full_bwd_estimate * 100
    if bwd_quant_pct > 20:
        print(f"  ⚠ Gradient quantization is {bwd_quant_pct:.1f}% of backward components")
        print(f"    {colored('→ Consider if gradient quantization overhead is acceptable', 'yellow')}")
    else:
        print(f"  ✓ Gradient quantization overhead is acceptable ({bwd_quant_pct:.1f}% of backward)")

    # Summary: Power-of-2 vs Arbitrary scaling
    print(f"\n{colored('='*80, 'blue')}")
    print(f"{colored('POWER-OF-2 vs ARBITRARY SCALING SUMMARY', 'blue')}")
    print(f"{colored('='*80, 'blue')}\n")

    print(f"Quantization time comparison:")
    print(f"  Input quantization:    pow2={t_quant_x_pow2:.3f}ms, arb={t_quant_x_arb:.3f}ms, diff={t_quant_x_pow2-t_quant_x_arb:+.3f}ms")
    print(f"  Weight quantization:   pow2={t_quant_w_pow2:.3f}ms, arb={t_quant_w_arb:.3f}ms, diff={t_quant_w_pow2-t_quant_w_arb:+.3f}ms")
    print(f"  Gradient quantization: pow2={t_quant_grad_pow2:.3f}ms, arb={t_quant_grad_arb:.3f}ms, diff={t_quant_grad_pow2-t_quant_grad_arb:+.3f}ms")
    print(f"  {'-'*70}")
    total_pow2 = t_quant_x_pow2 + t_quant_w_pow2 + t_quant_grad_pow2
    total_arb = t_quant_x_arb + t_quant_w_arb + t_quant_grad_arb
    print(f"  Total (fwd+bwd quant): pow2={total_pow2:.3f}ms, arb={total_arb:.3f}ms, diff={total_pow2-total_arb:+.3f}ms")

    print(f"\nFull FP8Linear comparison:")
    print(f"  Forward pass: pow2={t_full_pow2:.3f}ms, arb={t_full_arb:.3f}ms, diff={t_full_pow2-t_full_arb:+.3f}ms")

    # Overall assessment
    overhead_pct = (total_pow2 - total_arb) / total_arb * 100 if total_arb > 0 else 0
    color = 'green' if abs(overhead_pct) < 5 else 'yellow' if abs(overhead_pct) < 15 else 'red'
    print(f"\n{colored('Power-of-2 scaling overhead:', color)} {overhead_pct:+.1f}%")

    print(f"\n{colored('Benefits of power-of-2 scaling:', 'cyan')}")
    print(f"  ✓ Compatible with AMD RDNA4 block exponent scaling (MFMA_SCALE instructions)")
    print(f"  ✓ No extra quantization error from non-power-of-2 scales (DeepSeek)")
    print(f"  ✓ Integer exponent arithmetic is faster on some hardware")
    if abs(overhead_pct) < 10:
        print(f"  ✓ Minimal performance overhead ({overhead_pct:+.1f}%)")
    else:
        print(f"  ⚠ Noticeable performance overhead ({overhead_pct:+.1f}%) - may need optimization")

if __name__ == "__main__":
    main()
