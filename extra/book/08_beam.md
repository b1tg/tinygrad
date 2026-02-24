# Chapter 8: BEAM Search — Kernel Optimization

A naive kernel works, but a fast kernel requires choosing the right loop structure, vectorization, and thread mapping. Tinygrad uses **BEAM search** to automatically find the best optimization for each kernel.

## The Problem

Consider summing a 4x4 matrix along axis 1. The naive kernel:

```c
kernel void r_4_4(device float* data0, device float* data1, ...) {
  int gidx0 = gid.x; /* 4 */
  float acc0 = 0.0f;
  for (int ridx0 = 0; ridx0 < 4; ridx0++) {
    float val0 = *(data1+((gidx0<<2)+ridx0));
    acc0 = (acc0+val0);
  }
  *(data0+gidx0) = acc0;
}
```

This launches 4 threads, each looping 4 times. But we could also:
- **Upcast**: Remove the loop entirely, use vectorized `float4` loads
- **Unroll the globals**: Launch 1 thread that handles all 4 rows
- **Use local memory**: Use threadgroup shared memory for the reduction

Each choice produces different code. BEAM search tries many combinations and picks the fastest.

## Optimization Actions

Tinygrad has a small set of optimization primitives (called `Opt`s):

```python
from tinygrad.codegen.opt import Opt, OptOps

# UPCAST: unroll a loop into explicit per-element ops (enables vectorization)
Opt(OptOps.UPCAST, axis=0, arg=4)
# Effect: a loop of 4 becomes 4 explicit operations
# Benefit: enables float4 loads/stores

# LOCAL: map a global axis to local threads
Opt(OptOps.LOCAL, axis=0, arg=16)
# Effect: blockDim.x = 16, each thread handles fewer elements
# Benefit: parallelism within a workgroup

# GROUP: group a reduction across local threads
Opt(OptOps.GROUP, axis=0, arg=8)
# Effect: 8 threads each do partial reduction, then combine
# Benefit: parallel reduction

# UNROLL: fully unroll a loop
Opt(OptOps.UNROLL, axis=0, arg=4)
# Effect: loop body is repeated 4 times
# Benefit: removes loop overhead, enables instruction-level parallelism

# TC: use tensor cores (matrix multiply hardware)
Opt(OptOps.TC, axis=0, arg=...)
# Effect: replaces multiply-accumulate with WMMA instructions
# Benefit: massive throughput for matmul
```

These Opts are **composable** — you can apply multiple to the same kernel. For a matmul, you might apply:
- `LOCAL` to tile the output
- `UPCAST` to vectorize loads
- `GROUP` to parallelize the reduction
- `TC` to use tensor cores

## BEAM Search Algorithm

BEAM search explores the space of possible optimizations:

```
1. Start with the unoptimized kernel
2. Generate all valid single-step optimizations
3. Apply each one, measure execution time on real hardware
4. Keep the top-K (beam width) fastest kernels
5. For each survivor, generate next-step optimizations
6. Repeat until no more improvements
7. Return the fastest kernel found
```

```bash
# Enable BEAM search with width 5
BEAM=5 python -c "
from tinygrad import Tensor
(Tensor.ones(1024, 1024) @ Tensor.ones(1024, 1024)).realize()
"
```

The search is exhaustive within the beam — it actually runs each kernel variant on the GPU and measures wall-clock time. This is slow (many compilations and kernel launches) but finds genuinely optimal configurations.

## Heuristic Optimization

For when BEAM is too slow, tinygrad has hand-coded heuristics in `tinygrad/codegen/opt/heuristic.py`:

```python
# Heuristic rules (simplified):
# 1. If the kernel is small, upcast everything
# 2. If there's a reduction, try GROUP
# 3. For matmul-like patterns, try tensor cores
# 4. Map large axes to GLOBAL, small axes to LOCAL
```

The default (without `BEAM=N` or `NOOPT=1`) uses these heuristics.

## Upcast: Loop Unrolling for Vectorization

Upcast is the most common optimization. It converts a loop into explicit operations:

```c
// Before upcast (NOOPT):
for (int gidx0 = 0; gidx0 < 16; gidx0++) {
  float val0 = *(data1+gidx0);
  *(data0+gidx0) = (val0+1.0f);
}

// After UPCAST(axis=0, arg=4):
// Now 4 threads, each handling 4 elements with float4
int lidx0 = lid.x; /* 4 */
float4 val0 = *((float4*)(data1+(lidx0<<2)));
*((float4*)(data0+(lidx0<<2))) = float4(
  (val0.x+1.0f), (val0.y+1.0f),
  (val0.z+1.0f), (val0.w+1.0f));
```

The vectorized version does 4x fewer memory transactions (one 128-bit load instead of four 32-bit loads), which is significantly faster on GPUs.

## GROUP: Parallel Reduction

For reductions, GROUP distributes work across local threads:

```c
// Before GROUP (single thread reduces):
float acc = 0.0f;
for (int i = 0; i < 1024; i++) {
  acc += data[i];
}
out = acc;

// After GROUP(axis=0, arg=32):
// 32 threads each reduce 32 elements, then combine
float acc = 0.0f;
for (int i = lid.x * 32; i < (lid.x + 1) * 32; i++) {
  acc += data[i];
}
shared[lid.x] = acc;
barrier();
// Thread 0 sums the 32 partial results
if (lid.x == 0) {
  float total = 0.0f;
  for (int i = 0; i < 32; i++) total += shared[i];
  out = total;
}
```

## Tensor Cores

For matmul, the TC optimization maps multiply-accumulate to hardware matrix units (WMMA on NVIDIA, WMMA on AMD):

```c
// Before TC: scalar multiply-accumulate
for (int k = 0; k < K; k++) {
  acc += a[i][k] * b[k][j];
}

// After TC: hardware matrix multiply
// 16x16x16 matrix multiply in one instruction
wmma::mma_sync(d_frag, a_frag, b_frag, c_frag);
```

This can be 10-100x faster than scalar code.

## Seeing Optimizations Applied

```bash
# Compare unoptimized vs optimized
NOOPT=1 DEBUG=4 python -c "
from tinygrad import Tensor; (Tensor.ones(4,4).sum(1)).realize()
"

# vs
DEBUG=4 python -c "
from tinygrad import Tensor; (Tensor.ones(4,4).sum(1)).realize()
"
```

With `DEBUG=5`, you'll see the `Opt` actions that were applied:

```
(Opt(op=OptOps.UPCAST, axis=0, arg=4),)
```

## The Search Space

For a matmul-like kernel with shape `(M, N, K)`, the search space includes:

- GLOBAL/LOCAL splits for M (e.g., 64 blocks x 16 threads)
- GLOBAL/LOCAL splits for N
- UPCAST for M (1, 2, 4, 8)
- UPCAST for N (1, 2, 4, 8)
- UPCAST for K (1, 2, 4, 8)
- GROUP for the reduction (various sizes)
- TC enable/disable

This can be thousands of configurations. BEAM search with width 5 evaluates ~100 of them, which on a 1024x1024 matmul takes a few seconds.

## Caching

Optimized kernels are cached. The cache key is the kernel's UOp AST hash — if the same computation pattern appears again (even with different data), the cached optimization is reused.

```bash
# First run: slow (searches for optimal config)
BEAM=5 python -c "
from tinygrad import Tensor; (Tensor.randn(1024,1024) @ Tensor.randn(1024,1024)).realize()
"

# Second run: fast (uses cached result)
BEAM=5 python -c "
from tinygrad import Tensor; (Tensor.randn(1024,1024) @ Tensor.randn(1024,1024)).realize()
"
```

## Exercises

1. **Compare speeds**: Run a 1024x1024 matmul with `NOOPT=1`, default heuristics, and `BEAM=5`. Compare the GFLOPS in `DEBUG=2` output.

2. **Read the Opts**: Run `DEBUG=5` on a matmul and find the list of Opt actions applied. What does each one do?

3. **Try different beam widths**: Run `BEAM=1`, `BEAM=3`, `BEAM=10` on the same matmul. Does higher beam width always produce faster kernels?

## Source Code Map

| File | What to read |
|------|-------------|
| `tinygrad/codegen/opt/postrange.py` | `apply_opts()` — applies Opt actions to the AST |
| `tinygrad/codegen/opt/search.py` | BEAM search implementation |
| `tinygrad/codegen/opt/heuristic.py` | Hand-coded optimization heuristics |
| `tinygrad/codegen/opt/tc.py` | Tensor core detection and application |
| `tinygrad/codegen/opt/__init__.py` | `Opt`, `OptOps` definitions |
