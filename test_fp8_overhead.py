#!/usr/bin/env python3
"""Test quantization overhead"""

import time
from tinygrad import Tensor, dtypes, Device
from examples.mlperf.initializers import quantize_to_fp8

M, K = 66*512, 1024

# Create input
x = Tensor.empty(M, K, dtype=dtypes.half)
x.requires_grad = True

WARMUP = 5
ITERATIONS = 20
ITERATIONS = 1

def bench():
    # x_fp8, scale = quantize_to_fp8(x)
    # x_fp8.realize()
    # scale.realize()
    # x_fp8.sum().backward()
    # x.grad.realize()
    # return
    # Warmup
    if 0:
        for _ in range(WARMUP):
            x_fp8, scale = quantize_to_fp8(x)
            # x_fp8.realize()
            # scale.realize()
            x_fp8.sum().backward()
            x.grad.realize()


    # Benchmark
    Device[Device.DEFAULT].synchronize()
    times = []
    for _ in range(ITERATIONS):
        start = time.perf_counter()
        x_fp8, scale = quantize_to_fp8(x)
        # x_fp8.realize()
        scale.realize()
        x_fp8.sum().backward()
        x.grad.realize()
        Device[Device.DEFAULT].synchronize()
        times.append(time.perf_counter() - start)

    avg_time = sum(times) / len(times) * 1000
    print(f"Quantize ({M}, {K}): {avg_time:.3f} ms")

def break_down():
    # Now test individual ops
    print("\nBreakdown:")

    # abs
    def test_abs():
        return x.abs()
    Device[Device.DEFAULT].synchronize()
    times = []
    for _ in range(ITERATIONS):
        start = time.perf_counter()
        y = test_abs()
        y.realize()
        Device[Device.DEFAULT].synchronize()
        times.append(time.perf_counter() - start)
    print(f"  abs():          {sum(times)/len(times)*1000:.3f} ms")

    # max
    def test_max():
        return x.abs().max(keepdim=True)
    Device[Device.DEFAULT].synchronize()
    times = []
    for _ in range(ITERATIONS):
        start = time.perf_counter()
        y = test_max()
        y.realize()
        Device[Device.DEFAULT].synchronize()
        times.append(time.perf_counter() - start)
    print(f"  abs().max():    {sum(times)/len(times)*1000:.3f} ms")

    # scale and clamp
    def test_scale_clamp():
        scale = 448. / (x.abs().max(keepdim=True).detach() + 1e-8)
        return (x * scale).maximum(-448.0).minimum(448.0)
    Device[Device.DEFAULT].synchronize()
    times = []
    for _ in range(ITERATIONS):
        start = time.perf_counter()
        y = test_scale_clamp()
        y.realize()
        Device[Device.DEFAULT].synchronize()
        times.append(time.perf_counter() - start)
    print(f"  scale+clamp:    {sum(times)/len(times)*1000:.3f} ms")

    # cast
    def test_cast():
        scale = 448. / (x.abs().max(keepdim=True).detach() + 1e-8)
        y = (x * scale).maximum(-448.0).minimum(448.0)
        return y.cast(dtypes.fp8e4m3)
    Device[Device.DEFAULT].synchronize()
    times = []
    for _ in range(ITERATIONS):
        start = time.perf_counter()
        y = test_cast()
        y.realize()
        Device[Device.DEFAULT].synchronize()
        times.append(time.perf_counter() - start)
    print(f"  +cast:          {sum(times)/len(times)*1000:.3f} ms")


if __name__ == "__main__":
    bench()