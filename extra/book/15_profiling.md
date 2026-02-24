# Chapter 15: Command Queue & Profiling

Understanding performance requires measurement. This chapter covers how tinygrad dispatches work to the GPU (command queues) and how to measure kernel performance (profiling).

## Command Queues

GPUs don't execute kernels immediately when you call them. Instead, commands are placed in a **queue** and the GPU processes them asynchronously:

```
CPU:  submit(kernel1) -> submit(kernel2) -> submit(kernel3) -> wait
GPU:  ... executing kernel1 ... | kernel2 ... | kernel3 ... | done
```

Tinygrad manages command queues through the `HWQueue` abstraction in each backend:

```python
# Simplified from ops_metal.py / ops_amd.py
class HWQueue:
    def submit(self, program, bufs, global_size, local_size): ...
    def signal(self, fence): ...
    def wait(self, fence): ...
```

### Compute vs Copy Queues

Most backends have separate queues for compute and memory transfers:

```
Compute Queue:  kernel1 -> kernel2 -> kernel3
Copy Queue:     copy_in1 -> copy_in2 -> copy_out1

# These run in parallel! The GPU can copy data while computing.
```

This overlap is critical for performance — while one kernel runs, the next kernel's data can be copied in.

## Profiling Basics

### DEBUG=2: Quick Performance Check

The simplest way to measure performance:

```bash
DEBUG=2 python -c "
from tinygrad import Tensor
(Tensor.randn(1024, 1024) @ Tensor.randn(1024, 1024)).realize()
"
```

Output:
```
*** METAL  3  r_64_16_64_4_4_4                arg  3 mem  0.01 GB tm  123.45us/  1.23ms (  17.5 GFLOPS   12|34  GB/s)
```

Reading the output:
- `r_64_16_64_4_4_4` — kernel name (r = reduction, numbers = axis sizes)
- `arg 3` — 3 buffer arguments
- `mem 0.01 GB` — total memory accessed
- `tm 123.45us` — kernel execution time
- `1.23ms` — wall clock time since program start
- `17.5 GFLOPS` — floating point operations per second
- `12|34 GB/s` — memory bandwidth (read|write)

### PROFILE=1: Hardware Profiling

For precise GPU timing:

```bash
PROFILE=1 python my_script.py
```

This uses hardware performance counters (Metal's `sampleTimestamps`, AMD's SQTT, NVIDIA's timing events) to measure exact kernel duration without CPU overhead.

The profile data can be viewed with:
```bash
# Generates a perfetto trace
PROFILE=1 python my_script.py
# Open the generated trace file in https://ui.perfetto.dev/
```

## Key Performance Metrics

### GFLOPS (Giga Floating-Point Operations per Second)

Measures compute throughput:
```
GFLOPS = (total FLOPs) / (kernel time in seconds) / 1e9

# For matmul MxNxK: FLOPs = 2 * M * N * K
# 1024x1024x1024 matmul = 2 * 1024^3 = 2.1 billion FLOPs
# If it runs in 100us: 2.1e9 / 100e-6 / 1e9 = 21,000 GFLOPS = 21 TFLOPS
```

### GB/s (Memory Bandwidth)

Measures memory throughput:
```
GB/s = (bytes read + bytes written) / (kernel time in seconds) / 1e9

# Loading two 1024x1024 float32 matrices = 2 * 4MB = 8MB
# Storing one 1024x1024 float32 result = 4MB
# Total = 12MB. If kernel runs in 10us: 12e6 / 10e-6 / 1e9 = 1200 GB/s
```

### Roofline Model

A kernel is either **compute-bound** or **memory-bound**:

```
Arithmetic Intensity = FLOPs / Bytes

If AI > device_peak_GFLOPS / device_peak_GB_s:
    -> compute bound (limited by ALU throughput)
Else:
    -> memory bound (limited by memory bandwidth)
```

For matmul: AI = 2*N (for large square matrices) — highly compute-bound.
For element-wise add: AI = 0.33 (1 FLOP per 12 bytes) — memory-bound.

## Optimizing with BEAM

Use profiling to guide optimization:

```bash
# See default performance
DEBUG=2 python -c "
from tinygrad import Tensor
(Tensor.randn(1024,1024) @ Tensor.randn(1024,1024)).realize()
"

# Try BEAM search for better kernels
BEAM=5 DEBUG=2 python -c "
from tinygrad import Tensor
(Tensor.randn(1024,1024) @ Tensor.randn(1024,1024)).realize()
"
```

## The Profiling Infrastructure

Tinygrad's profiling system works across all backends:

1. **Timestamps**: Each kernel dispatch records start/end GPU timestamps
2. **Estimates**: Before running, tinygrad estimates FLOPs and memory from the kernel AST
3. **Trace output**: Results can be exported as Perfetto traces for visualization

```python
# From tinygrad/renderer/__init__.py
@dataclass
class Estimates:
    flops: sint = 0       # estimated floating-point operations
    mem: sint = 0         # estimated memory bytes accessed
    lds: sint = 0         # estimated local memory usage
```

These estimates are computed statically from the kernel structure, then compared against actual timings.

## Exercises

1. **Profile a matmul**: Run `DEBUG=2` on matmul at different sizes (64, 256, 1024, 4096). Plot GFLOPS vs size.

2. **Compare to peak**: Look up your GPU's theoretical peak GFLOPS. What percentage does tinygrad achieve on a large matmul?

3. **Memory vs compute**: Run `DEBUG=2` on `Tensor.randn(10000) + Tensor.randn(10000)` (element-wise, memory-bound) vs `Tensor.randn(100,100) @ Tensor.randn(100,100)` (matmul, compute-bound). Compare GFLOPS and GB/s.

4. **Perfetto trace**: Run `PROFILE=1` on a small model and open the trace in Perfetto. Identify gaps between kernels.

## Source Code Map

| File | What to read |
|------|-------------|
| `tinygrad/helpers.py` | `PROFILE`, `DEBUG` environment variables |
| `tinygrad/renderer/__init__.py` | `Estimates` — FLOP/memory estimation |
| `tinygrad/engine/realize.py` | Kernel dispatch and timing |
| `tinygrad/runtime/ops_metal.py` | Metal profiling implementation |
| `tinygrad/runtime/ops_nv.py` | NVIDIA profiling implementation |
| `tinygrad/runtime/ops_amd.py` | AMD profiling implementation |
