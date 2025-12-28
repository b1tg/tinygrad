#!/usr/bin/env python3
"""
Benchmark FP8LinearCached weight caching benefit.

Tests the performance improvement from caching quantized weights during
gradient accumulation scenarios. Compares FP8Linear (no caching) vs
FP8LinearCached (with caching).

Expected results:
- 2.52x speedup with 4 gradient accumulation steps
- ~12ms saved per forward pass (quantization overhead)
- Linear scaling with accumulation steps

Usage:
    python test/external/external_benchmark_fp8_cached.py
"""

import time
from tinygrad import Tensor, dtypes, Device
from tinygrad.helpers import colored
from extra.fp8 import FP8Linear, FP8LinearCached

# Test configuration
M, K, N = 8192, 2048, 2048  # Typical BERT dimensions
accumulation_steps = 4       # Gradient accumulation steps
iterations = 10              # Number of benchmark iterations

print(f"{colored('='*80, 'blue')}")
print(f"{colored('FP8 Weight Cache Benchmark', 'blue')}")
print(f"{colored('='*80, 'blue')}\n")

print(f"Device: {Device.DEFAULT}")
print(f"Shape: ({M}, {K}) @ ({K}, {N})")
print(f"Gradient accumulation steps: {accumulation_steps}")
print(f"Iterations: {iterations}\n")

# ============================================================================
# Benchmark 1: Without cache (FP8Linear)
# ============================================================================

print(f"{colored('Test 1: FP8Linear (no caching)', 'yellow')}")
layer_no_cache = FP8Linear(K, N, bias=False, use_custom_kernel=False)

times_no_cache = []
for _ in range(iterations):
    start = time.perf_counter()
    for _ in range(accumulation_steps):
        x = Tensor.randn(M, K, dtype=dtypes.half)
        y = layer_no_cache(x)
        y.realize()
    Device[Device.DEFAULT].synchronize()
    times_no_cache.append(time.perf_counter() - start)

avg_no_cache = sum(times_no_cache) / len(times_no_cache) * 1000
print(f"  Average time: {avg_no_cache:.2f} ms\n")

# ============================================================================
# Benchmark 2: With cache (FP8LinearCached)
# ============================================================================

print(f"{colored('Test 2: FP8LinearCached (with caching)', 'yellow')}")
layer_with_cache = FP8LinearCached(K, N, bias=False, use_custom_kernel=False)

times_with_cache = []
for _ in range(iterations):
    start = time.perf_counter()
    for _ in range(accumulation_steps):
        x = Tensor.randn(M, K, dtype=dtypes.half)
        y = layer_with_cache(x)
        y.realize()
    layer_with_cache.invalidate_cache()  # Simulate optimizer.step()
    Device[Device.DEFAULT].synchronize()
    times_with_cache.append(time.perf_counter() - start)

avg_with_cache = sum(times_with_cache) / len(times_with_cache) * 1000
print(f"  Average time: {avg_with_cache:.2f} ms\n")

# ============================================================================
# Results Analysis
# ============================================================================

print(f"{colored('='*80, 'blue')}")
print(f"{colored('RESULTS', 'blue')}")
print(f"{colored('='*80, 'blue')}\n")

speedup = avg_no_cache / avg_with_cache
savings = avg_no_cache - avg_with_cache
savings_per_forward = savings / accumulation_steps

color = 'green' if speedup > 1.0 else 'red'

print(f"Without cache:  {avg_no_cache:>8.2f} ms ({accumulation_steps} forward passes)")
print(f"With cache:     {avg_with_cache:>8.2f} ms ({accumulation_steps} forward passes)")
print(f"{'-'*80}")
print(f"Speedup:        {colored(f'{speedup:.2f}x', color)}")
print(f"Time saved:     {colored(f'{savings:.2f} ms', color)} per {accumulation_steps} forward passes")
print(f"                {colored(f'{savings_per_forward:.2f} ms', color)} per forward pass\n")

# ============================================================================
# Interpretation
# ============================================================================

print(f"{colored('INTERPRETATION:', 'yellow')}")

if speedup >= 2.0:
    print(f"  ✓ {colored('Excellent', 'green')} - Cache is working as expected!")
    print(f"    Speedup of {speedup:.2f}x indicates {accumulation_steps-1} of {accumulation_steps} forwards avoided quantization")
elif speedup >= 1.5:
    print(f"  ⚠ {colored('Good', 'yellow')} - Cache is working but with some overhead")
    print(f"    Expected ~{accumulation_steps/(accumulation_steps-1):.2f}x, got {speedup:.2f}x")
elif speedup >= 1.1:
    print(f"  ⚠ {colored('Marginal', 'yellow')} - Cache benefit is limited")
    print(f"    Quantization may be fast relative to matmul on this hardware")
else:
    print(f"  ✗ {colored('No benefit', 'red')} - Cache not helping")
    print(f"    Check if FP8LinearCached is actually being used")

print(f"\nPer-forward savings of {savings_per_forward:.2f}ms indicates:")
if savings_per_forward >= 10:
    print(f"  ✓ Quantization is a significant overhead (~{savings_per_forward:.0f}ms)")
    print(f"    Cache provides major benefit for gradient accumulation/inference")
elif savings_per_forward >= 5:
    print(f"  ⚠ Moderate quantization overhead (~{savings_per_forward:.0f}ms)")
    print(f"    Cache still worthwhile for multi-step accumulation")
else:
    print(f"  ⚠ Low quantization overhead (~{savings_per_forward:.0f}ms)")
    print(f"    Hardware may have fast FP8 quantization")

print(f"\n{colored('Recommendation:', 'cyan')}")
if speedup >= 1.5:
    print(f"  → Use FP8_CACHED=1 for BERT training with gradient accumulation")
    print(f"  → Use FP8LinearCached for inference (near-infinite speedup)")
elif speedup >= 1.1:
    print(f"  → Consider using FP8_CACHED=1 if gradient accumulation > 4 steps")
    print(f"  → Definitely use for inference where weights never change")
else:
    print(f"  → Caching may not be beneficial on this hardware/configuration")
    print(f"  → Stick with FP8=1 (non-cached) for simplicity")

print(f"\n{colored('='*80, 'blue')}")
