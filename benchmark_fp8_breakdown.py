#!/usr/bin/env python3
"""Breakdown FP8 performance: quantization overhead vs matmul speed"""

import time
from tinygrad import Tensor, dtypes, Device
from tinygrad.helpers import colored
# from examples.mlperf.initializers import quantize_to_fp8
from extra.fp8 import quantize_to_fp8, FP8Linear

# 33792= 66 * 512
# Test shape: (BS*SEQ, hidden) @ (hidden, hidden)
M, K, N = 33792, 1024, 1024
M, K, N = 33792, 8192, 8192
# M, K, N = 1024*512, 2048, 2048

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
    print(f"{colored('FP8 Performance Breakdown - Forward & Backward', 'blue')}")
    print(f"{colored('='*80, 'blue')}")
    print(f"\nDevice: {Device.DEFAULT}")
    print(f"Shape: ({M}, {K}) @ ({K}, {N})")
    print(f"Iterations: {ITERATIONS}\n")

    # Create tensors
    x = Tensor.rand(M, K, dtype=dtypes.half, requires_grad=True)
    w = Tensor.rand(N, K, dtype=dtypes.half, requires_grad=True)

    print(f"{colored('Step 1: Quantization overhead', 'yellow')}")
    # Quantize input
    t_quant_x = benchmark_op(f"Quantize input {x.shape}", quantize_to_fp8, x)

    # Quantize weight
    t_quant_w = benchmark_op(f"Quantize weight {w.shape}", quantize_to_fp8, w)

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
    batch_size = 16
    seq_len = M // batch_size
    x_3d = x.reshape(batch_size, seq_len, K)

    # Create FP8Linear layer with custom kernel
    fp8_layer = FP8Linear(K, N, bias=False, use_custom_kernel=True)
    fp8_layer.weight = w.T  # FP8Linear expects (out_features, in_features)

    print(f"{colored('Step 4: Custom kernel forward+backward', 'yellow')}")

    # FP8 custom kernel forward+backward
    def fp8_custom_fwd_bwd():
        x_3d_grad = x_3d.detach()
        x_3d_grad.requires_grad = True

        # Forward pass with custom kernel
        y = fp8_layer(x_3d_grad)

        # Create gradient for backward
        grad_output = Tensor.ones_like(y)

        # Backward pass (triggers custom_linear_backward)
        y.backward(grad_output)

        return y, x_3d_grad.grad, fp8_layer.weight.grad

    t_fp8_fwd_bwd = benchmark_op("FP8 custom kernel (fwd+bwd)", fp8_custom_fwd_bwd)

    # FP16 baseline forward+backward
    def fp16_fwd_bwd():
        x_3d_grad = x_3d.detach().cast(dtypes.half)
        x_3d_grad.requires_grad = True
        w_grad = w.T.detach().cast(dtypes.half)
        w_grad.requires_grad = True

        # Forward: matmul
        # y = x_3d_grad.cast(dtypes.float).dot(w_grad.T.cast(dtypes.float), dtype=dtypes.float)
        y = x_3d_grad.dot(w_grad.T, dtype=dtypes.float)

        # Backward
        grad_output = Tensor.ones_like(y)
        y.backward(grad_output)

        return y, x_3d_grad.grad, w_grad.grad

    t_fp16_fwd_bwd = benchmark_op("FP16 baseline (fwd+bwd)", fp16_fwd_bwd)

    print(f"\n{colored('Step 5: Backward pass component breakdown', 'yellow')}")

    # Create gradient tensor (output gradient) - 2D for simpler analysis
    grad_out = Tensor.rand(M, N, dtype=dtypes.half)

    # Quantize gradient
    t_quant_grad = benchmark_op("Quantize gradient", quantize_to_fp8, grad_out)

    # Pre-quantize for component tests
    grad_fp8, grad_scale = quantize_to_fp8(grad_out)

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

if __name__ == "__main__":
    main()
