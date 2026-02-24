# Chapter 11: Tensor Cores / WMMA

Modern GPUs have dedicated hardware for matrix multiplication. NVIDIA calls them Tensor Cores, AMD calls them WMMA (Wave Matrix Multiply-Accumulate) units. This chapter explains what they are, how tinygrad detects when to use them, and what the generated code looks like.

## What are Tensor Cores?

Normal GPU arithmetic does one operation per thread per cycle: one add, one multiply. Tensor Cores do **matrix multiply-accumulate** — a small matrix multiply (like 16x16x16) in a single instruction across all threads in a warp/wave.

```
Normal ALU:  1 FLOP per thread per cycle
Tensor Core: 16*16*16 = 4096 FLOPs per warp per cycle (for float16)
```

This is why GPUs advertise "tensor TFLOPS" numbers that are 4-16x higher than their "scalar TFLOPS."

## The WMMA Instruction

WMMA (Wave/Warp Matrix Multiply-Accumulate) computes:

```
D = A × B + C
```

Where A, B, C, D are small matrices distributed across the threads of a warp (32 threads on NVIDIA, 32 on AMD RDNA):

```
NVIDIA:  16x16x16 (fp16) or 8x8x4 (tf32) or 16x16x16 (int8)
AMD:     16x16x16 (fp16) or 16x16x16 (bf16)
```

Each thread holds a few elements of each matrix. After the WMMA instruction, each thread has a few elements of the result.

## How Tinygrad Detects Tensor Core Opportunities

The tensor core pass (`tinygrad/codegen/opt/tc.py`) looks for patterns in the kernel AST that match a matmul-like structure:

1. There's a reduction (SUM) over a dimension
2. Inside the reduction, there's a multiply of two loaded values
3. The shapes are compatible with WMMA tile sizes

```python
# The TC optimization is applied as:
Opt(OptOps.TC, axis=0, arg=(tile_m, tile_n, tile_k, dtype, ...))
```

When this is applied, the optimizer:
1. Tiles the loops to WMMA dimensions (e.g., 16x16x16)
2. Replaces the multiply-accumulate loop body with a WMMA UOp
3. Adjusts load/store patterns for the distributed matrix layout

## The Generated Code

### Without Tensor Cores (scalar):

```c
// 16x16 matmul, scalar
for (int ridx0 = 0; ridx0 < 16; ridx0++) {
  float val0 = *(data1 + (gidx0*16 + ridx0));   // A[i][k]
  float val1 = *(data2 + (ridx0*16 + gidx1));   // B[k][j]
  acc += val0 * val1;
}
```

### With Tensor Cores (CUDA PTX):

```ptx
// Load matrix fragments
wmma.load.a.sync.aligned.m16n16k16.global.row.f16 {%r0, ...}, [data1];
wmma.load.b.sync.aligned.m16n16k16.global.col.f16 {%r8, ...}, [data2];
// Multiply-accumulate
wmma.mma.sync.aligned.m16n16k16.row.col.f32.f16.f16.f32
    {%r16, ...}, {%r0, ...}, {%r8, ...}, {%r16, ...};
// Store result
wmma.store.d.sync.aligned.m16n16k16.global.row.f32 [data0], {%r16, ...};
```

### With WMMA (AMD RDNA3):

```
v_wmma_f32_16x16x16_f16 v[acc], v[a_frag], v[b_frag], v[acc]
```

One instruction replaces 16*16 = 256 multiply-accumulate operations.

## The UOp for WMMA

In tinygrad's IR, tensor core operations are represented by the `Ops.WMMA` UOp:

```python
UOp(Ops.WMMA, dtypes.float.vec(8),
    src=(a_fragment, b_fragment, acc_fragment),
    arg=(16, 16, 16, dtypes.half, "AMD"))
```

The arg contains: `(M, N, K, input_dtype, device_name)`.

## Data Layout

Tensor cores require specific data layouts. Each thread in a warp holds specific elements of the input matrices. The mapping depends on the hardware:

**NVIDIA (sm_80, 16x16x16 fp16)**:
- Thread `i` in the warp holds 8 elements of A and 8 elements of B
- After WMMA, thread `i` holds 8 elements of the result

**AMD (RDNA3, 16x16x16 fp16)**:
- 32 lanes, each holding fragments according to AMD's WMMA spec
- `v_wmma_f32_16x16x16_f16` operates on VGPRs directly

Tinygrad handles the data layout transformation automatically in the TC optimization pass.

## Performance Impact

Tensor cores provide massive speedups for matmul-heavy workloads:

```bash
# Without tensor cores
NOOPT=1 DEBUG=2 python -c "
from tinygrad import Tensor
(Tensor.ones(1024,1024) @ Tensor.ones(1024,1024)).realize()
"
# ~X GFLOPS

# With tensor cores (if supported by hardware)
DEBUG=2 python -c "
from tinygrad import Tensor
(Tensor.ones(1024,1024).half() @ Tensor.ones(1024,1024).half()).realize()
"
# ~16X GFLOPS (or more)
```

Note: tensor cores typically require half-precision (fp16/bf16) inputs. Some newer hardware supports fp32 or int8.

## When Tensor Cores are Used

Tensor cores are used when:
1. The hardware supports them (NVIDIA sm_70+, AMD RDNA3+)
2. The data type is compatible (usually fp16/bf16)
3. The shapes are compatible with tile sizes (multiples of 16)
4. The optimizer's TC pass successfully matches the matmul pattern

```python
# Likely to use tensor cores:
(Tensor.ones(256, 256).half() @ Tensor.ones(256, 256).half()).realize()

# Won't use tensor cores (float32 input on most hardware):
(Tensor.ones(256, 256) @ Tensor.ones(256, 256)).realize()

# Won't use tensor cores (too small for tile size):
(Tensor.ones(3, 3).half() @ Tensor.ones(3, 3).half()).realize()
```

## Exercises

1. **Check support**: Run `DEBUG=5` on a half-precision matmul. Look for `WMMA` in the AST output to see if tensor cores are being used.

2. **Compare speeds**: Run `DEBUG=2` on a 1024x1024 matmul with float32 vs float16 inputs. What's the speedup?

3. **Read the code**: Look at `tinygrad/codegen/opt/tc.py` and find the WMMA tile sizes for your hardware.

## Source Code Map

| File | What to read |
|------|-------------|
| `tinygrad/codegen/opt/tc.py` | Tensor core detection and optimization |
| `tinygrad/renderer/ptx.py` | PTX WMMA instruction emission (NVIDIA) |
| `tinygrad/renderer/cstyle.py` | CUDA WMMA intrinsic calls |
| `tinygrad/renderer/amd/` | AMD WMMA instruction encoding |
