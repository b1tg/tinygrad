# Chapter 5: Rangeify — From Shapes to Loops

This is the chapter that explains the core insight of tinygrad. If you understand rangeify, you understand how a 10,000-line framework can compete with million-line ones.

## The Problem

In PyTorch, when you write:

```python
a = torch.ones(2, 3)
b = a.T            # transpose
c = b.reshape(6)   # flatten
d = c.sum()        # reduce
```

What actually happens? PyTorch allocates a new tensor for the transpose, another for the reshape, another for the sum. Each step reads and writes memory.

Tinygrad does something different: **it doesn't execute anything until you ask for the result.** When you call `.realize()` or `.item()`, tinygrad looks at the entire chain of operations and generates a single GPU kernel that does everything at once. The question is: how?

The answer is **rangeify** — the algorithm that converts high-level shape manipulations (reshape, permute, expand, etc.) into explicit loop variables that a GPU can execute.

## See It in Action

Before we understand the theory, let's see what rangeify actually produces. Run this:

```python
# run with: DEBUG_RANGEIFY=1 python this_file.py
from tinygrad import Tensor

a = Tensor.ones(2, 3)
b = a.permute(1, 0)  # transpose: (2,3) -> (3,2)
c = b.reshape(6)     # flatten:   (3,2) -> (6,)
d = c.sum().realize() # reduce:   (6,)  -> (1,)
```

With `DEBUG_RANGEIFY=1`, you'll see output like:

```
***  1 Ops.ASSIGN           (1,)                   [0]
     1 Ops.REDUCE_AXIS      (1,)                   [r0 -> 0]
     1 Ops.RESHAPE          (6,)                   [(r0//2)][(r0%2)] -> [r0]
     1 Ops.PERMUTE          (3, 2)                 [(r0%2) -> (r0//2)][(r0//2) -> (r0%2)]
     1 Ops.EXPAND           (2, 3)                 [0 -> (r0%2)][0 -> (r0//2)]
     1 Ops.CONST            ()
     1 Ops.PARAM            (1,)                   [0]
```

Read it bottom-up. There's one loop variable `r0` that goes from 0 to 5. Each movement op transforms how `r0` maps to dimensions:

- **EXPAND** creates the (2,3) shape with indices `(r0%2, r0//2)`
- **PERMUTE** swaps them to `(r0//2, r0%2)`
- **RESHAPE** combines them back into a single `r0`
- **REDUCE_AXIS** sums over `r0`, producing a scalar

No memory was allocated for the transpose or reshape. The entire computation is: *"loop `r0` from 0 to 5, sum up `ones[r0//2, r0%2]`"*. That's one kernel, one loop.

## The Building Blocks

### UOps: Tinygrad's Universal IR

Everything in tinygrad is a **UOp** (micro-operation). A UOp is a node in a directed acyclic graph (DAG) with four fields:

```python
UOp(op=Ops.ADD, dtype=dtypes.float, src=(uop_a, uop_b), arg=None)
```

- `op`: what operation (ADD, MUL, RESHAPE, REDUCE_AXIS, ...)
- `dtype`: the data type
- `src`: tuple of input UOps (the edges of the DAG)
- `arg`: extra data (reshape dimensions, reduce axes, etc.)

When you write `Tensor([1,2,3]) + Tensor([4,5,6])`, tinygrad builds a UOp DAG — no computation happens yet:

```python
from tinygrad import Tensor

a = Tensor([1.0, 2.0, 3.0, 4.0])
b = Tensor([5.0, 6.0, 7.0, 8.0])
c = a + b

# c.uop is:
# UOp(Ops.ADD, dtypes.float,
#   src=(
#     UOp(Ops.COPY, ..., src=(UOp(Ops.PARAM, ..., shape=(4,)), ...)),  # a
#     UOp(Ops.COPY, ..., src=(UOp(Ops.PARAM, ..., shape=(4,)), ...)),  # b
#   ))

print(c.uop.op)    # Ops.ADD
print(c.uop.shape)  # (4,)
```

### Movement Ops

Movement ops change the *logical shape* of a tensor without moving data. There are six:

| Op | What it does | Example |
|----|-------------|---------|
| `RESHAPE` | Change shape, same total size | `(6,)` -> `(2,3)` |
| `PERMUTE` | Reorder dimensions | `(2,3)` -> `(3,2)` (transpose) |
| `EXPAND` | Broadcast a size-1 dim | `(1,3)` -> `(4,3)` |
| `SHRINK` | Take a slice | `(10,)` -> `(3,)` (items 2..5) |
| `PAD` | Add zeros around edges | `(3,)` -> `(5,)` (1 on each side) |
| `FLIP` | Reverse a dimension | `[1,2,3]` -> `[3,2,1]` |

In PyTorch, these are called "view operations" — they create a new *view* of the data. In tinygrad, they become UOp nodes in the lazy graph.

### RANGE Nodes

A `RANGE` UOp represents a loop variable:

```python
from tinygrad.uop.ops import UOp, Ops, AxisType
from tinygrad.dtype import dtypes

# A loop variable from 0 to 9 (exclusive)
r = UOp.range(10, 0, AxisType.LOOP)  # range_id=0, goes 0..9
print(r)  # UOp(Ops.RANGE, dtypes.index, src=(10,), arg=(0, AxisType.LOOP))
```

When this eventually becomes GPU code, it becomes an actual loop:

```c
for (int r0 = 0; r0 < 10; r0++) {
  // ... body uses r0 ...
}
```

Or, on a GPU, the loop dimensions get mapped to thread indices.

## The Algorithm

Rangeify is implemented in `tinygrad/schedule/indexing.py:run_rangeify()`. Here's how it works:

### Step 1: Find Realization Points

First, tinygrad decides which tensors need to be *materialized* in memory (i.e., stored to a buffer). The `pm_generate_realize_map` PatternMatcher marks:

- The sources of `SINK` (final outputs)
- Any `COPY`, `ASSIGN`, `CONTIGUOUS` nodes
- Sources of `COPY` (data being sent across devices)

Everything else is fused — it won't have its own buffer.

### Step 2: Assign Ranges (Bottom-Up)

The core loop walks through all nodes in **reverse topological order** (outputs first, inputs last). For each node, it answers: "what are the loop variables for this node's dimensions?"

There are three cases:

#### Case 1: Realized Node -> Create New Ranges

If a node is in the realize map (it will be stored to memory), we create fresh loop variables:

```
ASSIGN shape=(4,4) -> [r0][r1]   # new ranges r0=0..3, r1=0..3
```

This is where loops are born. Each dimension gets its own loop variable.

#### Case 2: Single Consumer -> Inherit Ranges

If a node has exactly one consumer, it just uses whatever ranges the consumer assigned:

```
MUL shape=(4,4,4) -> [r0][r1][r2]   # inherited from REDUCE_AXIS above
```

No new ranges needed — the computation is fused into the consumer's loops.

#### Case 3: Multiple Consumers -> Merge or Create

If a node has multiple consumers, tinygrad tries to merge their ranges. If all consumers use the same range for a given axis, it's reused. If not, a new range is created and that axis is partially realized:

```
# Node with 2 consumers that agree on axis 0 but disagree on axis 1:
Consumer A: [r0][r1]
Consumer B: [r0][r3]
Result:     [r0][r_new]  # axis 1 gets a new range, will be buffered
```

### Step 3: Transform Ranges Through Movement Ops

When the algorithm hits a movement op, it transforms the *output ranges* into *input ranges* using the movement op's semantics. This is where the magic happens.

Let's trace through our example:

```python
a = Tensor.ones(2, 3)      # shape (2, 3)
b = a.permute(1, 0)        # shape (3, 2) — the UOp graph has PERMUTE
c = b.reshape(6)           # shape (6,)   — the UOp graph has RESHAPE
d = c.sum()                # shape (1,)   — the UOp graph has REDUCE_AXIS
```

The UOp graph (bottom to top):
```
CONST shape=()
  -> EXPAND shape=(2,3)
    -> PERMUTE(1,0) shape=(3,2)
      -> RESHAPE(6) shape=(6,)
        -> REDUCE_AXIS(+, axis=0) shape=(1,)
          -> ASSIGN shape=(1,)      <-- realized
```

Rangeify walks top-down (reverse topo):

1. **ASSIGN** `shape=(1,)`: Realized. Create `r_out = [0]` (size 1, no loop needed).

2. **REDUCE_AXIS** `shape=(1,)`: Inherits `out_rngs = [0]` from ASSIGN. Since it reduces axis 0, it creates a new REDUCE range: `in_rngs = [r0]` where `r0` goes 0..5. This `r0` will become the reduction loop.

3. **RESHAPE(6)** from `(3,2)`: The output is `[r0]`. Reshape decomposes `r0` into the original dimensions: `in_rngs = [r0//2, r0%2]`.

4. **PERMUTE(1,0)** from `(2,3)` to `(3,2)`: Swaps the ranges: `in_rngs = [r0%2, r0//2]`.

5. **EXPAND** from `(1,1)` to `(2,3)`: Expanded dims get constant 0: `in_rngs = [0, 0]`. This is correct — the CONST is the same value everywhere, so we don't need to index into it.

The final kernel reads: "for r0 in 0..5, accumulate const[0,0]" — which is just `1.0 * 6 = 6.0`.

## Movement Op Semantics in Detail

The function `apply_movement_op()` in `indexing.py` defines how each movement op transforms ranges. Let's look at each one:

### SHRINK (slicing)

```python
# a[2:5] -> SHRINK with arg=((2, 5),)
# If output range is r0 (goes 0..2), input range is r0+2
case Ops.SHRINK: rngs = tuple(a+ss for a,(ss,_) in zip(rngs, arg))
```

Example:
```python
from tinygrad import Tensor
a = Tensor([10, 20, 30, 40, 50])
b = a[1:4]  # SHRINK arg=((1,4),)
# output range r0: 0,1,2
# input range:     r0+1 = 1,2,3  -> accesses a[1], a[2], a[3]
```

### PERMUTE (reorder dimensions)

```python
# a.permute(1,0) -> PERMUTE with arg=(1,0)
# Reorders the ranges
case Ops.PERMUTE: rngs = tuple(rngs[p] for p in argsort(arg))
```

Example:
```python
a = Tensor.ones(2, 3)
b = a.permute(1, 0)  # PERMUTE arg=(1,0)
# output ranges: [r0][r1]  (shape 3,2)
# input ranges:  [r1][r0]  (shape 2,3) — swapped back
```

### EXPAND (broadcast)

```python
# a.expand(4,3) where a has shape (1,3) -> EXPAND with arg=(4,3)
# Expanded dims get constant 0 (the value is the same for all indices)
case Ops.EXPAND: rngs = tuple(a if in_sh==out_sh else a.const_like(0) ...)
```

Example:
```python
a = Tensor.ones(1, 3)
b = a.expand(4, 3)  # EXPAND arg=(4,3)
# output ranges: [r0][r1]  (shape 4,3)
# input ranges:  [0][r1]   (shape 1,3) — axis 0 is constant
```

### FLIP (reverse)

```python
# a.flip(0) -> FLIP
# Reverses the indexing: r -> (size-1) - r
case Ops.FLIP: rngs = tuple(((s-1)-a) if f else a ...)
```

### PAD (zero-padding)

```python
# F.pad(a, (1,1)) -> PAD with arg=((1,1),)
# Shifts the range and adds a validity check
case Ops.PAD: rngs = tuple(... r.where(r-s, invalid()) ...)
```

Padding is special because some indices are "invalid" — they refer to the padded region. Rangeify represents this with a validity mask that gets turned into a `WHERE` (conditional) later.

### RESHAPE (change shape)

```python
# a.reshape(2,3) where a has shape (6,)
# Uses div/mod decomposition
case Ops.RESHAPE: rngs = _apply_reshape(in_shape, arg, sink.substitute(...)).src
```

Reshape is the most complex. It flattens the output ranges into a single linear index, then decomposes that index into the input dimensions using integer division and modulo:

```
output shape (6,) with range r0
-> linear index: r0
-> input shape (2,3): axis0 = r0 // 3, axis1 = r0 % 3
```

## A Real Example: Matrix Multiply

Let's see rangeify on the most important operation in ML:

```python
# run with: DEBUG_RANGEIFY=1 NOOPT=1 python this_file.py
from tinygrad import Tensor

a = Tensor.ones(4, 4)
b = Tensor.ones(4, 4)
c = (a @ b).realize()
```

The `@` operator expands to: reshape, expand, multiply, reduce. Here's what `DEBUG_RANGEIFY=1` shows:

```
***  1 Ops.ASSIGN           (4, 4, 1)   [r0][r1][0]
     1 Ops.REDUCE_AXIS      (4, 4, 1)   [r0][r1][r2 -> 0]
     1 Ops.MUL              (4, 4, 4)   [r0][r1][r2]
     1 Ops.EXPAND           (4, 4, 4)   [0 -> r0][r1][r2]
     1 Ops.PERMUTE          (1, 4, 4)   [0][r2 -> r1][r1 -> r2]
     1 Ops.RESHAPE          (1, 4, 4)   [r2][r1] -> [0][r2][r1]
     1 Ops.EXPAND           (4, 4, 4)   [r0][0 -> r1][r2]
     1 Ops.RESHAPE          (4, 1, 4)   [r0][r2] -> [r0][0][r2]
```

Read it bottom-up. `a` (shape 4x4 stored as flat 16 elements) gets:
1. **RESHAPE** to (4,1,4): ranges split into `[r0][0][r2]`
2. **EXPAND** to (4,4,4): axis 1 broadcast, `[r0][r1][r2]`

And `b` gets:
1. **RESHAPE** to (1,4,4): ranges become `[0][r2][r1]`
2. **PERMUTE**: swap last two axes to align for multiply
3. **EXPAND** to (4,4,4): axis 0 broadcast

Then:
- **MUL**: element-wise multiply with ranges `[r0][r1][r2]`
- **REDUCE_AXIS** over axis 2: `r2` becomes the reduction loop
- **ASSIGN**: output shape (4,4), ranges `[r0][r1]`

The generated kernel:
```c
// Pseudocode (actual output will be Metal/CUDA/etc)
for r0 in 0..3:     // output row
  for r1 in 0..3:   // output col
    acc = 0
    for r2 in 0..3: // reduction dim
      acc += a[r0*4 + r2] * b[r2*4 + r1]
    out[r0*4 + r1] = acc
```

This is a standard matrix multiply — and tinygrad derived it automatically from shape operations.

## The Generated Kernel

Let's actually see the generated code:

```python
# run with: DEBUG=5 NOOPT=1 python this_file.py
from tinygrad import Tensor

a = Tensor.ones(4, 4)
b = Tensor.ones(4, 4)
c = (a @ b).realize()
```

With `DEBUG=5`, you'll see the full UOp AST and the rendered kernel. The AST after rangeify looks like:

```
c0 = UOp(Ops.PARAM, dtypes.float.ptr(16), (), 0)  # output buffer
c2 = UOp.range(4, 0, AxisType.LOOP)                 # r0: output row
c4 = UOp.range(4, 1, AxisType.LOOP)                 # r1: output col
c6 = UOp.range(4, 2, AxisType.REDUCE)               # r2: reduction dim
c8 = UOp(Ops.PARAM, dtypes.float.ptr(16), (), 1)   # a buffer
c10 = UOp(Ops.PARAM, dtypes.float.ptr(16), (), 2)  # b buffer
# indexing: a[r0*4 + r2] * b[r2*4 + r1], reduced over r2
c12 = c8.index(c2*4+c6) + c10.index(c6*4+c4)
```

Notice:
- `r0` and `r1` are `AxisType.LOOP` — they become the output indices
- `r2` is `AxisType.REDUCE` — it becomes the accumulation loop
- The indexing expressions `r0*4+r2` and `r2*4+r1` come directly from how rangeify decomposed the movement ops

## Key Insights

### 1. No Data Movement for Views

Reshape, permute, expand — none of these allocate memory. They just change how ranges map to buffer indices. The only operations that touch memory are LOAD (read from buffer) and STORE (write to buffer).

### 2. Fusion is Free

Because rangeify works on the entire UOp graph at once, operations naturally fuse. If you write `(a @ b).relu()`, the relu becomes part of the same kernel — it's just applied to the accumulator before writing to the output buffer.

### 3. The Realize Map Controls Kernel Boundaries

The realize map determines where one kernel ends and another begins. If an operation is not in the realize map, it gets fused into its consumer's kernel. The `pm_remove_bufferize` pass can even remove unnecessary materializations to create larger fused kernels.

### 4. Symbolic Math Does the Heavy Lifting

The `_apply_reshape` function relies heavily on symbolic simplification. When you reshape `(6,)` to `(2,3)`, it generates expressions like `r0//3` and `r0%3`. The `symbolic` PatternMatcher simplifies these — for example, `(r0%3)//3` simplifies to `0`. This is critical for fusing chains of reshapes.

## Hands-On: Tracing Rangeify Yourself

Here's a script that lets you explore rangeify on any tensor expression:

```python
"""
Run with: DEBUG_RANGEIFY=1 python extra/book/trace_rangeify.py
Try changing the expression to see different rangeify outputs.
"""
from tinygrad import Tensor
import os
os.environ['DEBUG_RANGEIFY'] = '1'
os.environ['NOOPT'] = '1'

# === Try these expressions ===

# 1. Simple element-wise
# c = (Tensor.ones(4) + Tensor.ones(4)).realize()

# 2. Reshape + sum
# c = Tensor.ones(2,3).reshape(6).sum().realize()

# 3. Transpose + matmul
# c = (Tensor.ones(4,4).T @ Tensor.ones(4,4)).realize()

# 4. Broadcasting
# c = (Tensor.ones(4,1) + Tensor.ones(1,4)).realize()

# 5. Convolution-like pattern
# c = Tensor.ones(1,1,4).expand(1,1,4).reshape(4).sum().realize()

# 6. Pad + sum
# c = Tensor.ones(3).pad(((1,1),)).sum().realize()

# Default: matrix multiply
c = (Tensor.ones(4,4) @ Tensor.ones(4,4)).realize()
```

## What Comes After Rangeify

After rangeify converts movement ops to ranges, the kernel goes through several more passes:

1. **Symbolic simplification**: Simplify range expressions (e.g., `(r0*4+r1)//4` -> `r0`)
2. **Buffer removal**: Try to remove unnecessary intermediate buffers (`pm_remove_bufferize`)
3. **Buffer assignment**: Convert remaining `BUFFERIZE` to `STORE` + `BUFFER`
4. **Kernel splitting**: Each `STORE` becomes a separate kernel (`split_kernels`)
5. **Optimization**: BEAM search applies tiling, upcast, unroll, tensor cores (Chapter 8)
6. **Codegen**: Lower to GPU source code (Chapter 7)

The full pipeline is in `tinygrad/schedule/rangeify.py:get_kernel_graph()`.

## Source Code Map

| File | What to read |
|------|-------------|
| `tinygrad/schedule/indexing.py` | `run_rangeify()` — the main algorithm |
| `tinygrad/schedule/indexing.py:142` | `apply_movement_op()` — how each movement op transforms ranges |
| `tinygrad/schedule/indexing.py:126` | `_apply_reshape()` — the reshape decomposition |
| `tinygrad/schedule/rangeify.py:483` | `get_kernel_graph()` — the full pipeline |
| `tinygrad/schedule/rangeify.py:458` | `split_store()` — how the graph is split into kernels |
| `tinygrad/uop/ops.py:16` | `AxisType` — the enum for range types (LOOP, REDUCE, GLOBAL, etc.) |

## Exercises

1. **Trace a transpose**: Run `DEBUG_RANGEIFY=1` on `Tensor.ones(3,4).permute(1,0).contiguous().realize()`. What ranges does the PERMUTE produce? Why is `contiguous()` needed?

2. **Understand fusion**: Run `DEBUG_RANGEIFY=1` on `(Tensor.ones(4,4) @ Tensor.ones(4,4)).relu().realize()`. Does the relu create a new kernel or fuse with the matmul?

3. **Count the kernels**: Run `DEBUG=2` on `a = Tensor.ones(4,4); b = (a @ a).realize(); c = (b + b).realize()`. How many kernels are generated? Why?

4. **Read the code**: Open `tinygrad/schedule/indexing.py` and find the `run_rangeify` function. Identify where each of the three cases (realized, single consumer, multiple consumers) is handled.
