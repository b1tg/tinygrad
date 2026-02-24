# Chapter 2: The UOp — Tinygrad's Universal IR

Everything in tinygrad is a UOp. Tensors are UOps. Kernels are UOps. The generated GPU code is a UOp. The compiled binary is a UOp. Understanding UOps is understanding tinygrad.

## What is a UOp?

A UOp (micro-operation) is a node in a directed acyclic graph (DAG). It has four fields:

```python
from tinygrad.uop.ops import UOp, Ops
from tinygrad.dtype import dtypes

# Create a UOp manually
node = UOp(
    op=Ops.CONST,      # what operation
    dtype=dtypes.float, # data type
    src=(),             # input UOps (tuple)
    arg=42.0            # extra data
)
print(node.op)     # Ops.CONST
print(node.dtype)  # dtypes.float
print(node.arg)    # 42.0
print(node.src)    # ()
```

UOps form a graph through the `src` field. An `ADD` node points to its two operands:

```python
a = UOp(Ops.CONST, dtypes.float, arg=1.0)
b = UOp(Ops.CONST, dtypes.float, arg=2.0)
c = UOp(Ops.ADD, dtypes.float, src=(a, b))

print(c.op)        # Ops.ADD
print(c.src[0].arg) # 1.0
print(c.src[1].arg) # 2.0
```

## UOps are Singletons (Hash-Consed)

A critical property: **if you create two UOps with identical (op, dtype, src, arg), you get the same object**:

```python
x = UOp(Ops.CONST, dtypes.float, arg=1.0)
y = UOp(Ops.CONST, dtypes.float, arg=1.0)
print(x is y)  # True — same Python object!
```

This is called **hash consing**. It means:
- The UOp graph is automatically deduplicated
- You can compare UOps with `is` instead of `==`
- Memory is shared — the same subexpression used in 100 places only exists once

The implementation uses a global cache in `UOpMetaClass.__call__` (`tinygrad/uop/ops.py:86`).

## The Ops Enum

The `Ops` enum defines every operation tinygrad knows about. They're organized by level:

```python
from tinygrad.uop import Ops

# Level 1: Definitions
# DEFINE_VAR, SPECIAL, DEFINE_LOCAL — kernel setup

# Level 2: Infrastructure
# PARAM, CALL — function parameters
# SINK, AFTER — ordering/grouping
# GEP, VECTORIZE — vector access

# Level 3: Memory
# INDEX — pointer arithmetic
# LOAD, STORE — memory access

# Level 4: Math
# Unary:  CAST, EXP2, LOG2, SIN, SQRT, RECIPROCAL, NEG
# Binary: ADD, MUL, MAX, CMPLT, CMPNE, AND, OR, XOR
# Ternary: WHERE, MULACC
# Tensor: WMMA (matrix multiply)

# Level 5: Control Flow
# RANGE, END — loops
# IF, ENDIF — conditionals
# BARRIER — synchronization
# CONST — constants

# Level 6: Tensor Graph (high-level, don't exist in final kernels)
# RESHAPE, PERMUTE, EXPAND, PAD, SHRINK, FLIP — movement ops
# REDUCE_AXIS — reduction
# COPY, BUFFER, ASSIGN — buffer management
# CONTIGUOUS — force materialization
```

The same `UOp` class represents both high-level tensor ops (like `RESHAPE`) and low-level kernel ops (like `LOAD`). The compilation pipeline is a sequence of graph rewrites that lower high-level ops into low-level ones.

## Tensors are UOps

When you create a `Tensor`, you're building a UOp graph:

```python
from tinygrad import Tensor

a = Tensor([1.0, 2.0, 3.0])
print(a.uop.op)      # Ops.COPY
print(a.uop.shape)    # (3,)
print(a.uop.dtype)    # dtypes.float

b = a + 1
print(b.uop.op)       # Ops.ADD
print(b.uop.src[0].op)  # Ops.COPY  (the tensor a)
```

Every tensor method appends to this graph:

```python
a = Tensor.ones(2, 3)       # CONST -> EXPAND
b = a.reshape(3, 2)          # RESHAPE wrapping a
c = b.permute(1, 0)          # PERMUTE wrapping b
d = c.sum()                  # REDUCE_AXIS wrapping c
# d.uop is a DAG: REDUCE_AXIS -> PERMUTE -> RESHAPE -> EXPAND -> CONST
```

Nothing is computed until you call `.realize()`, `.numpy()`, or `.item()`.

## Key UOp Properties

UOps have several useful computed properties:

```python
from tinygrad import Tensor

a = Tensor.ones(4, 4)
b = a.sum(axis=1)

# Shape: the logical shape of the tensor
print(b.uop.shape)   # (4, 1)

# Device: where the tensor lives
print(b.uop.device)  # 'METAL' or 'CUDA' etc.

# The graph structure
print(b.uop.op)               # Ops.REDUCE_AXIS
print(b.uop.src[0].op)        # Ops.EXPAND
print(b.uop.arg)              # (Ops.ADD, (1,))  — sum over axis 1
```

## UOp Graph Visualization

For any UOp graph, you can use `pretty_print` to see the structure:

```python
from tinygrad.uop.ops import UOp, Ops, pretty_print
from tinygrad.dtype import dtypes

a = UOp(Ops.CONST, dtypes.float, arg=1.0)
b = UOp(Ops.CONST, dtypes.float, arg=2.0)
c = UOp(Ops.ADD, dtypes.float, src=(a, b))
d = UOp(Ops.MUL, dtypes.float, src=(c, c))

print(pretty_print(d))
```

Or use `VIZ=1` to open the interactive graph visualizer:

```bash
VIZ=1 python -c "from tinygrad import Tensor; (Tensor.ones(4) + Tensor.ones(4)).realize()"
```

## The Two Roles of UOps

UOps serve two very different roles, and understanding this duality is key:

### Role 1: Lazy Tensor Graph

Before `.realize()`, UOps represent the **logical computation** — what operations to perform:

```
REDUCE_AXIS(+, axis=1)
  └─ EXPAND (4, 4)
       └─ RESHAPE (1, 1)
            └─ CONST 1.0
```

These UOps have ops like `RESHAPE`, `EXPAND`, `REDUCE_AXIS`. They represent *what* to compute.

### Role 2: Kernel AST

After scheduling and codegen, UOps represent the **physical computation** — how to execute on hardware:

```
SINK
  └─ STORE(ptr, idx, value)
       ├─ PARAM(0)              # output buffer pointer
       ├─ RANGE(0..3)           # loop variable
       └─ REDUCE(+)
            └─ LOAD(ptr, idx)
                 ├─ PARAM(1)    # input buffer pointer
                 └─ RANGE(0..3) # another loop variable
```

These UOps have ops like `LOAD`, `STORE`, `RANGE`, `PARAM`. They represent *how* to compute.

The compilation pipeline transforms Role 1 UOps into Role 2 UOps through a series of pattern-matching rewrites (Chapter 3).

## Building UOp Graphs by Hand

You can construct and render UOp graphs manually. This is useful for understanding the codegen pipeline:

```python
from tinygrad.uop.ops import UOp, Ops
from tinygrad.dtype import dtypes

# Build a simple kernel AST: out[i] = 1.0 + 2.0
const1 = UOp(Ops.CONST, dtypes.float, arg=1.0)
const2 = UOp(Ops.CONST, dtypes.float, arg=2.0)
add = UOp(Ops.ADD, dtypes.float, src=(const1, const2))

print(add)
# UOp(Ops.ADD, dtypes.float, arg=None, src=(
#   UOp(Ops.CONST, dtypes.float, arg=1.0, src=()),
#   UOp(Ops.CONST, dtypes.float, arg=2.0, src=()),))
```

## The Toposort

Since UOps form a DAG, we can topologically sort them — process children before parents. This is used everywhere in tinygrad:

```python
from tinygrad.uop.ops import UOp, Ops
from tinygrad.dtype import dtypes

a = UOp(Ops.CONST, dtypes.float, arg=1.0)
b = UOp(Ops.CONST, dtypes.float, arg=2.0)
c = UOp(Ops.ADD, dtypes.float, src=(a, b))
d = UOp(Ops.MUL, dtypes.float, src=(c, a))

for u in d.toposort():
    print(u.op, u.arg if u.op is Ops.CONST else "")
# Ops.CONST 1.0
# Ops.CONST 2.0
# Ops.ADD
# Ops.MUL
```

## Exercises

1. **Build a graph**: Create a UOp graph representing `(a + b) * (a - b)` where a=3.0 and b=2.0. Print the graph.

2. **Verify singleton**: Create two identical `UOp(Ops.ADD, ...)` graphs and verify they return the same object.

3. **Inspect a tensor**: Create `t = Tensor.ones(3,3).sum()` and walk `t.uop` manually — print each node's `op` and `shape` by following `src` pointers.

4. **Count ops**: Use `.toposort()` on `(Tensor.ones(4,4) @ Tensor.ones(4,4)).uop` to count how many UOp nodes are in a matmul graph before realization.

## Source Code Map

| File | What to read |
|------|-------------|
| `tinygrad/uop/ops.py:84` | `UOpMetaClass` — the singleton/hash-consing implementation |
| `tinygrad/uop/ops.py:100` | `UOp` class — all methods and properties |
| `tinygrad/uop/__init__.py` | `Ops` enum — all operation types |
| `tinygrad/uop/spec.py` | Validation rules for UOp graphs |
| `tinygrad/uop/symbolic.py` | Symbolic simplification of UOp expressions |
