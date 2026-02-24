# Chapter 4: Scheduling — When Does Computation Happen?

In tinygrad, computation is lazy. This chapter explains *when* things actually get computed, how the framework decides which operations to fuse into a single kernel, and what the schedule looks like.

## The Lazy Graph

When you write tensor operations, tinygrad builds a graph of UOps — no computation happens:

```python
from tinygrad import Tensor

a = Tensor.ones(4, 4)
b = Tensor.ones(4, 4)
c = a + b         # just adds an ADD node to the graph
d = c * 2         # just adds a MUL node
e = d.sum()       # just adds a REDUCE_AXIS node
# Nothing has been computed yet
```

Computation is triggered by:

```python
e.realize()   # force computation, keep result on device
e.numpy()     # force computation, copy to CPU
e.item()      # force computation, return scalar
```

## What is a Schedule?

When you call `.realize()`, tinygrad creates a **schedule** — a list of operations to execute in order. Each item is either:
- A **kernel**: generated GPU code to run
- A **copy**: data transfer between devices
- An **allocation**: buffer creation

```bash
# See the schedule
DEBUG=2 python -c "
from tinygrad import Tensor
a = Tensor.ones(4, 4)
b = Tensor.ones(4, 4)
c = ((a + b) * 2).sum().realize()
"
```

You'll see output like:

```
*** METAL  1  copy  64, METAL <- PYTHON    # copy a to GPU
*** METAL  2  copy  64, METAL <- PYTHON    # copy b to GPU
*** METAL  3  r_16                         # the kernel: add, mul, sum
```

Three operations: two copies to bring data to the GPU, then one kernel that does `(a+b)*2` and sums — all fused into a single kernel.

## Kernel Fusion

The key optimization in scheduling is **fusion** — combining multiple operations into one kernel to avoid intermediate memory allocations:

```python
# Without fusion: 3 kernels, 2 intermediate buffers
# Kernel 1: c = a + b    (write temp1)
# Kernel 2: d = temp1 * 2  (read temp1, write temp2)
# Kernel 3: e = sum(temp2) (read temp2, write result)

# With fusion: 1 kernel, 0 intermediate buffers
# Kernel 1: e = sum((a + b) * 2)  (read a,b, write result)
```

Tinygrad fuses operations that can share the same loop structure. The general rule: **if the output shape matches or is a reduction of the input shape, it can be fused.**

### What gets fused:
- Elementwise chains: `(a + b) * c - d`
- Elementwise followed by reduction: `(a + b).sum()`
- Reshapes and permutes (they're free — just change indexing)

### What forces a new kernel:
- `CONTIGUOUS` — explicitly forces materialization
- `COPY` — data transfer across devices
- Multiple consumers — if a value is used by two separate reductions
- The `ASSIGN` boundary — in-place operations

```python
from tinygrad import Tensor

# Fused into 1 kernel:
x = Tensor.ones(4, 4)
y = ((x + 1) * 2).sum()
y.realize()  # 1 kernel

# Two kernels (matmul forces materialization):
a = Tensor.ones(4, 4)
b = a @ a          # kernel 1: matmul
c = (b + 1).sum()  # kernel 2: add + sum
c.realize()
```

## The Scheduling Pipeline

Here's how a `.realize()` call turns into executed kernels:

```
Tensor.realize()
  └─ schedule_with_vars()
       └─ complete_create_schedule_with_vars(big_sink)
            │
            ├─ transform_to_call(big_sink)
            │    Converts tensor UOp graph into CALL nodes
            │    with explicit buffer assignments
            │
            ├─ get_kernel_graph(function)  [rangeify.py]
            │    Converts movement ops to RANGE loops
            │    Splits graph into discrete kernels
            │    (This is Chapter 5: Rangeify)
            │
            └─ create_schedule(kernel_graph)
                 Topological sort of kernels
                 Returns list[ExecItem]
```

Each `ExecItem` has:
- `.ast` — the UOp kernel AST (a `SINK` node)
- `.bufs` — the buffers it reads and writes

## The Realize Map

Tinygrad determines which operations need their own buffer (and thus their own kernel) by building a **realize map**. An operation is realized if:

1. It's the **final output** (a `SINK` source)
2. It's a **COPY** or **ASSIGN** (must be materialized)
3. It's a **CONTIGUOUS** (user explicitly requested materialization)
4. It has **multiple consumers** that can't share ranges

Everything else is fused into its consumer's kernel.

## Seeing the Schedule

Use `DEBUG=2` to see what kernels get generated:

```bash
# Element-wise fusion
DEBUG=2 python -c "
from tinygrad import Tensor
x = Tensor.ones(1000)
y = ((x + 1) * 2 - 3).realize()
"
# Output: 1 kernel (E_1000)

# Reduction fusion
DEBUG=2 python -c "
from tinygrad import Tensor
x = Tensor.ones(1000)
y = ((x + 1) * 2).sum().realize()
"
# Output: 1 kernel (r_1000)
# The 'r' prefix means it has a reduction
```

Kernel naming convention:
- `E_N` — elementwise kernel, N total elements
- `r_N_M` — reduction kernel, N output elements, M reduction elements
- Numbers in the name represent the axis sizes

## Memory Planning

After scheduling, tinygrad runs a **memory planner** that reuses buffers whose lifetimes don't overlap:

```python
# Without memory planning:
# Buffer A: used by kernel 1-2
# Buffer B: used by kernel 3-4
# Buffer C: used by kernel 5-6
# Total: 3 buffers allocated

# With memory planning:
# Buffer A: used by kernel 1-2, then reused for kernel 5-6
# Buffer B: used by kernel 3-4
# Total: 2 buffers allocated
```

This is handled by `tinygrad/engine/memory.py`.

## Exercises

1. **Count kernels**: Run `DEBUG=2` on various expressions and predict how many kernels will be generated:
   - `(Tensor.ones(100) + Tensor.ones(100)).realize()`
   - `(Tensor.ones(100) + Tensor.ones(100)).sum().realize()`
   - `x = Tensor.ones(4,4); (x @ x + x).realize()`

2. **Force splits**: Use `.contiguous()` to force an intermediate materialization. Compare `DEBUG=2` output with and without it.

3. **Read the schedule**: Set `DEBUG=3` to see the kernel ASTs. Identify which UOp nodes represent loads, stores, and computation.

## Source Code Map

| File | What to read |
|------|-------------|
| `tinygrad/engine/schedule.py` | `complete_create_schedule_with_vars()` — the main scheduler |
| `tinygrad/engine/realize.py` | `run_schedule()` — kernel execution |
| `tinygrad/engine/allocations.py` | `transform_to_call()` — buffer assignment |
| `tinygrad/engine/memory.py` | Memory planner (buffer reuse) |
| `tinygrad/schedule/rangeify.py` | `get_kernel_graph()` — movement ops to kernels |
