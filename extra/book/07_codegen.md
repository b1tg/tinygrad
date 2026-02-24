# Chapter 7: Codegen — UOps to Source Code

This chapter traces the full path from a kernel's UOp AST to the generated GPU source code. By the end, you'll understand every rewrite pass in the pipeline and be able to read the generated code.

## The Big Picture

After scheduling produces a kernel AST (a `SINK` node), codegen transforms it through ~15 pattern-matching passes into renderable source code:

```
Kernel AST (SINK with RANGE/LOAD/STORE)
  |  graph_rewrite passes (lowering)
  v
Linear UOp list (flat instruction sequence)
  |  renderer (pattern matching on UOps -> strings)
  v
Source code string (C/CUDA/Metal/PTX)
  |  compiler
  v
Binary (GPU executable)
```

## See It Yourself

```bash
DEBUG=5 NOOPT=1 python -c "
from tinygrad import Tensor
(Tensor.ones(4) + Tensor.ones(4)).realize()
"
```

The output shows the AST and the rendered code. Let's trace through a simple example.

## The Kernel AST

After rangeify, a simple element-wise add of two size-4 vectors looks like:

```python
c0 = UOp(Ops.PARAM, dtypes.float.ptr(4), (), 0)   # output buffer
c2 = UOp.range(4, 0, AxisType.LOOP)                 # loop variable i: 0..3
c4 = UOp(Ops.PARAM, dtypes.float.ptr(4), (), 1)    # input buffer a
c6 = UOp(Ops.PARAM, dtypes.float.ptr(4), (), 2)    # input buffer b
c8 = c4.index(c2) + c6.index(c2)                    # a[i] + b[i]
c10 = c0.index(c2, ptr=True).store(c8).end(c2)      # out[i] = result
ast = c10.sink(arg=KernelInfo(...))
```

This is the input to `full_rewrite_to_sink()` in `tinygrad/codegen/__init__.py`.

## The Rewrite Passes

### Pass 1-5: Early Lowering

These passes handle movement ops, simplify range expressions, and optimize index math:

```
pm_mops               # Convert movement ops to index expressions
pm_syntactic_sugar     # Clean up syntactic patterns
pm_load_collapse       # Merge redundant loads
symbolic               # Simplify math (x*1 -> x, x+0 -> x)
pm_simplify_ranges     # Simplify range bounds
```

### Pass 6: Optimization (apply_opts)

This is where BEAM search or heuristics apply optimization actions:

```python
from tinygrad.codegen.opt import Opt, OptOps

# Available optimizations:
Opt(OptOps.UPCAST, axis=0, arg=4)   # Unroll loop 4x (vectorize)
Opt(OptOps.LOCAL, axis=0, arg=16)   # Map to 16 local threads
Opt(OptOps.GROUP, axis=0, arg=8)    # Group reduction
Opt(OptOps.UNROLL, axis=0, arg=4)   # Fully unroll a loop
Opt(OptOps.TC, axis=0, arg=(...))   # Use tensor cores
```

With `NOOPT=1`, this pass is skipped and the kernel stays in its simplest form. See Chapter 8 for details.

### Pass 7: Expander

Unrolls loops marked as UPCAST or UNROLL. A range with UPCAST becomes explicit per-element operations:

```
Before: for i in range(4): out[i] = a[i] + b[i]
After:  out[0] = a[0] + b[0]
        out[1] = a[1] + b[1]
        out[2] = a[2] + b[2]
        out[3] = a[3] + b[3]
```

This enables vectorization — the renderer can emit `float4` loads/stores instead of scalar ones.

### Pass 8-9: Buffers and Reductions

Add local memory buffers (for workgroup-shared data) and lower reduction operations (REDUCE) into accumulator + loop patterns.

### Pass 10: GPU Dimensions (gpudims)

Map RANGE loops to GPU hardware dimensions:

```python
# AxisType.GLOBAL  -> blockIdx.x/y/z  (threadgroup_position)
# AxisType.LOCAL   -> threadIdx.x/y/z (thread_position)
# AxisType.WARP    -> warp-level operations
# AxisType.LOOP    -> actual for loops
# AxisType.REDUCE  -> reduction loops
# AxisType.UPCAST  -> unrolled (no loop)
```

A RANGE with `AxisType.GLOBAL` becomes `gid.x` in Metal or `blockIdx.x` in CUDA. A RANGE with `AxisType.LOCAL` becomes `lid.x` or `threadIdx.x`.

### Pass 11-12: Loads and Devectorization

Add explicit LOAD instructions (replacing INDEX references) and handle vectorized access patterns.

### Pass 13: Decompositions

Lower complex operations into primitives:

```python
# DIV(x, y) -> MUL(x, RECIPROCAL(y))
# EXP(x)    -> EXP2(x * LOG2E)
# LOG(x)    -> LOG2(x) * LN2
# SIGMOID(x) -> RECIPROCAL(1 + EXP2(-x * LOG2E))
```

### Pass 14-15: Final Rewrite and Control Flow

Insert actual control flow — `RANGE`/`END` pairs become `for` loops, `IF`/`ENDIF` become conditionals:

```
Before: RANGE(r0, 0, 4) ... END(r0)
After:  for (int r0 = 0; r0 < 4; r0++) { ... }
```

## Linearization

After all passes, the DAG is flattened into a **linear list of UOps** — one instruction per line, in execution order:

```python
# The linearize() function in codegen/late/linearizer.py
# topologically sorts the UOp DAG into a flat list:
[
    UOp(Ops.PARAM, ptr, (), 0),        # data0
    UOp(Ops.PARAM, ptr, (), 1),        # data1
    UOp(Ops.PARAM, ptr, (), 2),        # data2
    UOp(Ops.SPECIAL, int, (), gidx0),  # thread index
    UOp(Ops.INDEX, ptr, ...),          # &data1[gidx0]
    UOp(Ops.LOAD, float, ...),         # val0 = data1[gidx0]
    UOp(Ops.INDEX, ptr, ...),          # &data2[gidx0]
    UOp(Ops.LOAD, float, ...),         # val1 = data2[gidx0]
    UOp(Ops.ADD, float, ...),          # val0 + val1
    UOp(Ops.INDEX, ptr, ...),          # &data0[gidx0]
    UOp(Ops.STORE, void, ...),         # data0[gidx0] = result
]
```

## Rendering

The renderer walks the linear UOp list and emits source code. Each UOp has a rendering rule defined in a PatternMatcher:

```python
# Simplified from tinygrad/renderer/cstyle.py

# PARAM -> function parameter
Ops.PARAM:  "data{arg}"

# SPECIAL -> hardware thread index
Ops.SPECIAL: "gid.x" / "blockIdx.x" / etc.

# LOAD -> pointer dereference
Ops.LOAD:  "*(data1+gidx0)"

# ADD -> infix operator
Ops.ADD:   "(val0+val1)"

# STORE -> pointer write
Ops.STORE: "*(data0+gidx0) = (val0+val1);"

# RANGE/END -> for loop
Ops.RANGE: "for (int ridx0 = 0; ridx0 < 4; ridx0++) {"
Ops.END:   "}"
```

The actual renderer handles many more cases (vectorized types, image types, special hardware functions), but the core is pattern matching on UOp types.

## Different Renderers

Tinygrad has renderers for different targets:

| Renderer | Target | File |
|----------|--------|------|
| `CUDARenderer` | NVIDIA CUDA C++ | `renderer/cstyle.py` |
| `MetalRenderer` | Apple Metal C++ | `renderer/cstyle.py` |
| `OpenCLRenderer` | OpenCL C | `renderer/cstyle.py` |
| `WGSLRenderer` | WebGPU WGSL | `renderer/wgsl.py` |
| `PTXRenderer` | NVIDIA PTX assembly | `renderer/ptx.py` |
| `LLVMRenderer` | LLVM IR (CPU) | `renderer/llvmir.py` |
| `AMDRenderer` | AMD ISA assembly | `renderer/amd/` |

All C-style renderers share the same base class and most rules — the differences are mainly in function signatures, thread indexing, and hardware-specific intrinsics.

## The Colors

When you see kernel names in `DEBUG=2` output, the colors encode information:

```
E_4_4     # E = elementwise
r_4_4     # r = has reduction

# Colors of the numbers indicate axis types:
# Blue   = GLOBAL (mapped to GPU blocks)
# Cyan   = LOCAL (mapped to GPU threads)
# Yellow = UPCAST (unrolled/vectorized)
# Red    = REDUCE (reduction loop)
# White  = LOOP (regular loop)
```

## End-to-End Example

Let's trace a sum reduction through the entire pipeline:

```bash
DEBUG=5 NOOPT=1 python -c "
from tinygrad import Tensor
Tensor.ones(4).sum().realize()
"
```

**Kernel AST** (after rangeify):
```
SINK
  └─ STORE(param0, idx, value).END(r0)
       └─ REDUCE(+, r0)
            └─ LOAD(param1, r0)
```

**After lowering** (linearized):
```
PARAM(0)          -> data0
PARAM(1)          -> data1
CONST(0.0)        -> accumulator init
RANGE(0, 4)       -> for ridx0 = 0..3
  LOAD(data1, ridx0)  -> val0
  ADD(acc, val0)       -> new acc
END               -> end loop
STORE(data0, 0, acc)   -> write result
```

**Rendered** (Metal):
```c
kernel void r_4(device float* data0, device float* data1, ...) {
  float acc0 = 0.0f;
  for (int ridx0 = 0; ridx0 < 4; ridx0++) {
    float val0 = *(data1+ridx0);
    acc0 = (acc0+val0);
  }
  *(data0+0) = acc0;
}
```

## Exercises

1. **Read the output**: Run `DEBUG=5 NOOPT=1` on `Tensor.ones(4,4).sum(axis=0).realize()`. Identify each UOp in the AST and map it to the generated code.

2. **Compare optimized**: Run `DEBUG=4` (without NOOPT) on the same expression. How does the optimized kernel differ?

3. **Different backends**: If you have access to CUDA, compare the Metal and CUDA output for the same kernel. What's the same? What's different?

4. **Trace a pass**: Add `VIZ=1` and run a simple kernel. The visualizer shows each rewrite pass.

## Source Code Map

| File | What to read |
|------|-------------|
| `tinygrad/codegen/__init__.py` | `full_rewrite_to_sink()` — the complete lowering pipeline |
| `tinygrad/codegen/__init__.py` | `get_program()` — top-level entry point |
| `tinygrad/codegen/late/expander.py` | Loop unrolling and vectorization |
| `tinygrad/codegen/late/linearizer.py` | DAG to linear list conversion |
| `tinygrad/codegen/late/devectorizer.py` | Load/store indexing |
| `tinygrad/codegen/gpudims.py` | GPU dimension assignment |
| `tinygrad/renderer/__init__.py` | `Renderer` base class, `ProgramSpec` |
| `tinygrad/renderer/cstyle.py` | C-style code emission (~1000 lines) |
