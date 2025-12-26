#!/usr/bin/env python3
"""Benchmark FP8Linear vs normal Linear for BERT shapes"""

import time
import os
os.environ['FP8'] = '1'

from tinygrad import Tensor, dtypes, Device
from tinygrad.helpers import getenv, colored
from examples.mlperf.initializers import FP8LinearBert, LinearBert

# BERT-large config
HIDDEN_SIZE = 16384
INTERMEDIATE_SIZE = 16384
BS = 66*2
SEQ = 512
BATCH_SEQ = (BS, SEQ)  # 33792

# Warmup and benchmark settings
WARMUP = 3
ITERATIONS = 10

def benchmark_linear(linear_class, input_shape, in_features, out_features, name):
    """Benchmark a linear layer with given shapes"""

    # Create model
    model = linear_class(in_features, out_features, bias=True)
    model.weight.assign(Tensor.rand(out_features, in_features, dtype=dtypes.default_float))
    model.bias.assign(Tensor.rand(out_features, dtype=dtypes.default_float))
    model.weight.requires_grad = True
    model.bias.requires_grad = True

    # Create input
    x = Tensor.rand(*input_shape, dtype=dtypes.default_float)
    x.requires_grad = True

    # Warmup
    print(f"  Warming up {name}...")
    for _ in range(WARMUP):
        y = model(x)
        # y.realize()
        y.sum().backward()
        # if model.weight.grad is not None:
        #     model.weight.grad.realize()
        if x.grad is not None:
            x.grad.realize()

    # Benchmark forward
    print(f"  Benchmarking {name} forward...")
    Device[Device.DEFAULT].synchronize()
    times_fwd = []
    for _ in range(ITERATIONS):
        start = time.perf_counter()
        y = model(x)
        y.realize()
        Device[Device.DEFAULT].synchronize()
        times_fwd.append(time.perf_counter() - start)

    avg_fwd = sum(times_fwd) / len(times_fwd) * 1000  # ms

    # Benchmark forward + backward
    print(f"  Benchmarking {name} forward+backward...")
    Device[Device.DEFAULT].synchronize()
    times_fwd_bwd = []
    for _ in range(ITERATIONS):
        # Clear grads
        model.weight.grad = None
        model.bias.grad = None
        x.grad = None

        start = time.perf_counter()
        y = model(x)
        y.realize()
        y.sum().backward()
        if model.weight.grad is not None:
            model.weight.grad.realize()
        if model.bias.grad is not None:
            model.bias.grad.realize()
        if x.grad is not None:
            x.grad.realize()
        Device[Device.DEFAULT].synchronize()
        times_fwd_bwd.append(time.perf_counter() - start)

    avg_fwd_bwd = sum(times_fwd_bwd) / len(times_fwd_bwd) * 1000  # ms

    # Calculate FLOPS
    # Forward: 2 * M * N * K (matmul) + M * N (bias)
    # Backward: similar for weight grad and input grad
    M, K = input_shape[0], in_features
    N = out_features
    flops_fwd = 2 * M * N * K
    gflops_fwd = flops_fwd / (avg_fwd / 1000) / 1e9

    return {
        'avg_fwd': avg_fwd,
        'avg_fwd_bwd': avg_fwd_bwd,
        'gflops_fwd': gflops_fwd,
    }

def print_results(shape_name, fp8_results, normal_results):
    """Print comparison results"""
    print(f"\n{colored(shape_name, 'yellow')}")
    print(f"  {'Metric':<25} {'FP8Linear':<15} {'NormalLinear':<15} {'Speedup':<10}")
    print(f"  {'-'*70}")

    # Forward pass
    speedup_fwd = normal_results['avg_fwd'] / fp8_results['avg_fwd']
    color_fwd = 'green' if speedup_fwd > 1.0 else 'red'
    print(f"  {'Forward (ms)':<25} {fp8_results['avg_fwd']:<15.3f} {normal_results['avg_fwd']:<15.3f} {colored(f'{speedup_fwd:.2f}x', color_fwd):<10}")

    # Forward + Backward pass
    speedup_fwd_bwd = normal_results['avg_fwd_bwd'] / fp8_results['avg_fwd_bwd']
    color_fwd_bwd = 'green' if speedup_fwd_bwd > 1.0 else 'red'
    print(f"  {'Forward+Backward (ms)':<25} {fp8_results['avg_fwd_bwd']:<15.3f} {normal_results['avg_fwd_bwd']:<15.3f} {colored(f'{speedup_fwd_bwd:.2f}x', color_fwd_bwd):<10}")

    # GFLOPS
    print(f"  {'GFLOPS (forward)':<25} {fp8_results['gflops_fwd']:<15.1f} {normal_results['gflops_fwd']:<15.1f}")

    return speedup_fwd, speedup_fwd_bwd

def main():
    print(f"{colored('='*80, 'blue')}")
    print(f"{colored('FP8Linear vs NormalLinear Benchmark for BERT Shapes', 'blue')}")
    print(f"{colored('='*80, 'blue')}")
    print(f"\nDevice: {Device.DEFAULT}")
    print(f"Default dtype: {dtypes.default_float}")
    print(f"BERT Config: hidden={HIDDEN_SIZE}, intermediate={INTERMEDIATE_SIZE}, BS={BS}, SEQ={SEQ}")
    print(f"Warmup iterations: {WARMUP}, Benchmark iterations: {ITERATIONS}\n")

    # Test shapes (similar to BERT layers)
    test_cases = [
        # ("QKV Projection", (*BATCH_SEQ, HIDDEN_SIZE), HIDDEN_SIZE, HIDDEN_SIZE),
        # ("Attention Output", (*BATCH_SEQ, HIDDEN_SIZE), HIDDEN_SIZE, HIDDEN_SIZE),
        # ("FFN Intermediate", (*BATCH_SEQ, HIDDEN_SIZE), HIDDEN_SIZE, INTERMEDIATE_SIZE),
        ("FFN Output", (*BATCH_SEQ, INTERMEDIATE_SIZE), INTERMEDIATE_SIZE, HIDDEN_SIZE),
    ]

    speedups_fwd = []
    speedups_fwd_bwd = []

    for shape_name, input_shape, in_features, out_features in test_cases:
        print(f"\n{colored(f'Testing {shape_name}: {input_shape} @ ({in_features}, {out_features})', 'cyan')}")

        # Benchmark FP8Linear
        print(f"{colored('FP8Linear:', 'yellow')}")
        print(f"{input_shape=}, {in_features=} {out_features=}")
        fp8_results = benchmark_linear(FP8LinearBert, input_shape, in_features, out_features, "FP8Linear")

        # Benchmark Normal Linear
        print(f"{colored('NormalLinear:', 'yellow')}")
        normal_results = benchmark_linear(LinearBert, input_shape, in_features, out_features, "NormalLinear")

        # Print comparison
        speedup_fwd, speedup_fwd_bwd = print_results(shape_name, fp8_results, normal_results)
        speedups_fwd.append(speedup_fwd)
        speedups_fwd_bwd.append(speedup_fwd_bwd)

    # Summary
    avg_speedup_fwd = sum(speedups_fwd) / len(speedups_fwd)
    avg_speedup_fwd_bwd = sum(speedups_fwd_bwd) / len(speedups_fwd_bwd)

    print(f"\n{colored('='*80, 'blue')}")
    print(f"{colored('SUMMARY', 'blue')}")
    print(f"{colored('='*80, 'blue')}")
    print(f"Average speedup (forward): {colored(f'{avg_speedup_fwd:.2f}x', 'green' if avg_speedup_fwd > 1.0 else 'red')}")
    print(f"Average speedup (forward+backward): {colored(f'{avg_speedup_fwd_bwd:.2f}x', 'green' if avg_speedup_fwd_bwd > 1.0 else 'red')}")

    if avg_speedup_fwd > 1.0:
        print(f"\n{colored('✓ FP8Linear is faster on average!', 'green')}")
    else:
        print(f"\n{colored('✗ FP8Linear is slower on average. Potential issues:', 'red')}")
        print(f"  - Quantization overhead may be too high")
        print(f"  - FP8 GEMM kernel may not be optimized for these shapes")
        print(f"  - Check if rocBLAS FP8 kernels are being used")

if __name__ == "__main__":
    main()
