# Chapter 1: Introduction

This chapter gives you the mental model for how tinygrad works. By the end, you'll understand what tinygrad is, how it differs from PyTorch, and what happens when you run a simple tensor operation.

## What is Tinygrad?

Tinygrad is a deep learning framework in ~10,000 lines of Python. It can train and run neural networks on GPUs (NVIDIA, AMD, Apple Metal, etc.), just like PyTorch — but with a fundamentally simpler architecture.

Where PyTorch has separate subsystems for autograd, dispatch, memory management, and dozens of hand-tuned kernels, tinygrad has **one core idea**: represent everything as a graph of micro-operations (UOps), then rewrite that graph into GPU code using pattern matching.

## Your First Tinygrad Program

```python
from tinygrad import Tensor

a = Tensor([1.0, 2.0, 3.0, 4.0])
b = Tensor([5.0, 6.0, 7.0, 8.0])
c = a + b
print(c.numpy())  # [ 6.  8. 10. 12.]
```

This looks like PyTorch, and that's intentional. The API is familiar. The difference is underneath.

## Lazy Evaluation

The most important thing to understand: **tinygrad is lazy**. When you write `c = a + b`, nothing is computed. Tinygrad just builds a graph:

```python
c = a + b
print(c)  # <Tensor <UOp METAL (4,) float (<Ops.ADD: 44>, None)> on METAL with grad None>
```

The addition hasn't happened yet. `c` is just a node in a computation graph that says "I am the result of adding `a` and `b`." Computation only happens when you explicitly ask for a result:

- `c.numpy()` — compute and return as numpy array
- `c.realize()` — compute and keep on GPU
- `c.item()` — compute and return as Python scalar
- `c.tolist()` — compute and return as Python list

This lazy evaluation is what enables tinygrad to fuse operations and generate efficient kernels.

## What Happens When You Realize

Let's trace what happens when you call `c.numpy()` on `c = a + b`:

```
1. Build UOp graph:  ADD(COPY(a), COPY(b))
2. Schedule:         Determine what kernels to run
3. Rangeify:         Convert shapes to loop variables
4. Codegen:          Lower UOps to GPU source code
5. Compile:          Compile source to GPU binary
6. Execute:          Dispatch kernel on GPU
7. Copy back:        Transfer result to CPU for numpy()
```

You can see the generated GPU code by setting `DEBUG=4`:

```bash
DEBUG=4 NOOPT=1 python -c "
from tinygrad import Tensor
a = Tensor([1.0, 2.0, 3.0, 4.0])
b = Tensor([5.0, 6.0, 7.0, 8.0])
print((a + b).numpy())
"
```

On Apple Metal, you'll see something like:

```c
#include <metal_stdlib>
using namespace metal;
kernel void E_4(device float* data0, device float* data1, device float* data2,
                uint3 gid [[threadgroup_position_in_grid]],
                uint3 lid [[thread_position_in_threadgroup]]) {
  int gidx0 = gid.x; /* 4 */
  float val0 = *(data1+gidx0);
  float val1 = *(data2+gidx0);
  *(data0+gidx0) = (val0+val1);
}
```

This kernel launches 4 GPU threads (one per element). Each thread loads one element from `data1` and `data2`, adds them, and stores the result to `data0`.

On CUDA, you'd see nearly identical code but with `blockIdx.x` instead of `gid.x`.

## The Two Halves of Tinygrad

Tinygrad has two conceptual halves:

**1. The ML Framework** (tensor.py, nn/)
- Tensor operations: matmul, conv2d, relu, softmax, etc.
- Autograd: automatic differentiation for backprop
- Neural network layers: Linear, Conv2d, BatchNorm, etc.

This part works like a minimal PyTorch. If you know PyTorch, you already know how to use it.

**2. The Compiler** (schedule/, codegen/, renderer/, runtime/)
- Scheduling: deciding which operations to fuse into which kernels
- Code generation: turning operations into GPU source code
- Rendering: emitting the final source string
- Runtime: compiling and executing on specific hardware

This is the part unique to tinygrad, and what this book focuses on.

## Everything is Elementwise + Reduce

The key insight in tinygrad: almost every tensor operation can be expressed as a combination of **elementwise operations** and **reductions**, plus zero-cost **shape movements**.

```python
# Elementwise: output has same shape as input
c = a + b           # ADD
c = a * b           # MUL
c = a.relu()        # MAX(a, 0)
c = a.exp()         # EXP

# Reduction: output has fewer elements
c = a.sum(axis=0)   # SUM along axis
c = a.max(axis=1)   # MAX along axis

# Shape movements: no data moves, just reinterpret
c = a.reshape(2, 2) # RESHAPE
c = a.T             # PERMUTE (transpose)
c = a.expand(4, 4)  # EXPAND (broadcast)
```

Even complex operations decompose into these primitives:

```python
# Mean = sum / count
def mean(x, axis):
    return x.sum(axis) / x.shape[axis]

# Softmax = exp(x - max(x)) / sum(exp(x - max(x)))
def softmax(x, axis=-1):
    e = (x - x.max(axis, keepdim=True)).exp()
    return e / e.sum(axis, keepdim=True)

# Matrix multiply = reshape + expand + multiply + sum
# (covered in detail in Chapter 9)
```

This decomposition is what makes tinygrad small. Instead of implementing 200 optimized kernels, tinygrad has a general-purpose compiler that handles the ~20 primitive operations.

## DEBUG Levels

Throughout this book, we'll use environment variables to inspect tinygrad's internals:

| Variable | What it shows |
|----------|-------------|
| `DEBUG=1` | Kernel names and memory |
| `DEBUG=2` | Kernel timing and GFLOPS |
| `DEBUG=3` | Kernel AST (the UOp graph) |
| `DEBUG=4` | Generated source code |
| `DEBUG=5` | Full UOp tree + source |
| `NOOPT=1` | Disable kernel optimizations |
| `DEBUG_RANGEIFY=1` | Show how shapes become loops |
| `VIZ=1` | Open the graph visualizer |

Try them:

```bash
# See kernel timing
DEBUG=2 python -c "from tinygrad import Tensor; (Tensor.ones(1000,1000) @ Tensor.ones(1000,1000)).realize()"

# See generated code for matmul
DEBUG=4 NOOPT=1 python -c "from tinygrad import Tensor; (Tensor.ones(4,4) @ Tensor.ones(4,4)).realize()"
```

## Exercises

1. **Compare outputs**: Run `DEBUG=4 NOOPT=1` on `Tensor.ones(4).sum().realize()`. Read the generated kernel — what does the loop do?

2. **Count kernels**: Run `DEBUG=2` on `(Tensor.ones(4,4) @ Tensor.ones(4,4)).relu().realize()`. How many kernels are generated? Is relu a separate kernel or fused?

3. **Try different operations**: Run `DEBUG=4 NOOPT=1` on `Tensor.ones(4,4).sum(axis=0).realize()` vs `Tensor.ones(4,4).sum(axis=1).realize()`. How do the kernels differ?

## Source Code Map

| File | What it does |
|------|-------------|
| `tinygrad/tensor.py` | The public Tensor API (~5000 lines) |
| `tinygrad/dtype.py` | Data type definitions |
| `tinygrad/device.py` | Device/Buffer/Compiler abstractions |
| `tinygrad/nn/__init__.py` | Neural network layers |
| `tinygrad/gradient.py` | Autograd implementation |
