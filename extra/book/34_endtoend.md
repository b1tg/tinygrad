# Chapter 34: End-to-End Trace — Following a Computation Through tinygrad

This chapter traces a single operation through the entire tinygrad pipeline, from Python API to GPU execution and back. This ties together everything from Parts 1-5.

## The Operation

```python
from tinygrad import Tensor
result = Tensor.ones(4, 4).sum().item()
# result = 16.0
```

Simple: create a 4×4 matrix of ones, sum them, get the number. But under the hood, this touches every layer of tinygrad.

## Stage 1: Tensor Creation

`Tensor.ones(4, 4)` calls:

```python
@staticmethod
def ones(*shape, **kwargs):
    return Tensor.full(shape, 1.0, **kwargs)

@staticmethod
def full(shape, fill_value, **kwargs):
    return Tensor(fill_value, **kwargs)._broadcast_to(shape)
```

This creates a `UOp.const(dtypes.float32, 1.0)` and expands it to shape `(4, 4)`. No buffer is allocated — it's just a constant in the graph.

The UOp graph at this point:

```
CONST(1.0, dtype=float32)
  → RESHAPE to (1, 1)
  → EXPAND to (4, 4)
```

## Stage 2: Sum

`.sum()` creates a reduce operation:

```python
def sum(self, axis=None, keepdim=False):
    return self._reduce(Ops.ADD, axis, keepdim)
```

The UOp graph grows:

```
CONST(1.0) → RESHAPE(1,1) → EXPAND(4,4) → REDUCE_AXIS(ADD, axes=(0,1))
```

Still no computation. The graph just records "sum all elements of a 4×4 matrix of ones."

## Stage 3: `.item()` Triggers Realization

```python
def item(self):
    assert self.numel() == 1
    return self.data()[(0,) * len(self.shape)]
```

`.data()` calls `._buffer()`, which calls `.realize()`. Now things get real.

## Stage 4: Scheduling

`realize()` calls `schedule_with_vars()`:

```python
def schedule_with_vars(self, *lst):
    big_sink = UOp.sink(*[x.uop for x in (self,) + lst])
    becomes_map, schedule, var_vals = complete_create_schedule_with_vars(big_sink)
    _apply_map_to_tensors(becomes_map, name="buffers")
    return schedule, var_vals
```

`complete_create_schedule_with_vars` does the heavy lifting:

1. **`transform_to_call`**: Wraps the graph in a CALL node, separating the computation from its buffer arguments
2. **`get_kernel_graph`**: The core scheduling pipeline (next section)
3. **`create_schedule`**: Linearizes the kernel graph into execution order
4. **`memory_planner`**: Optimizes buffer allocation

## Stage 5: Rangeify (The Core Transformation)

Inside `get_kernel_graph`, the UOp graph is transformed through several passes:

### Pass 1: Movement ops to ranges

The `EXPAND(4,4)` and `REDUCE_AXIS(ADD, (0,1))` become explicit loops:

```
Before rangeify:
  CONST(1.0) → EXPAND(4,4) → REDUCE_AXIS(ADD, (0,1))

After rangeify:
  RANGE(0, 4)  ← loop variable i
  RANGE(0, 4)  ← loop variable j
  CONST(1.0)   ← the value at every position
  REDUCE(ADD, over ranges i and j)  ← sum over both loops
```

This is the key insight from Chapter 5: shapes become loops.

### Pass 2: Symbolic simplification

The pattern matcher simplifies the graph. Since we're summing a constant:

```
sum(1.0 for i in range(4) for j in range(4)) = 1.0 * 4 * 4 = 16.0
```

tinygrad's symbolic engine may constant-fold this entirely.

### Pass 3: Bufferize

Add buffer operations — where to read inputs from and write outputs to:

```
BUFFER(output, size=1, dtype=float32)
  STORE: result of the reduction
```

### Pass 4: Split into kernels

The graph is split at kernel boundaries (see Chapter 33 on fusion). For this simple case, there's just one kernel.

## Stage 6: Codegen

The kernel UOp graph is lowered to source code (Chapter 7). For a Metal GPU:

```c
#include <metal_stdlib>
kernel void r_16(device float* data0, uint3 gid [[threadgroup_position_in_grid]]) {
  float acc = 0.0f;
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      acc += 1.0f;
    }
  }
  *(data0) = acc;
}
```

(In practice, the compiler may optimize this further — the constant folding might eliminate the loops entirely.)

## Stage 7: Compilation

The source code is compiled to GPU binary:

```python
class Compiler:
    def compile_cached(self, src):
        # Check disk cache first
        if (lib := diskcache_get(self.cachekey, src)) is None:
            lib = self.compile(src)  # actually compile
            diskcache_put(self.cachekey, src, lib)
        return lib
```

The compiled binary is cached on disk, so the same kernel won't be recompiled.

## Stage 8: Execution

`run_schedule` processes each `ExecItem`:

```python
def run_schedule(schedule, var_vals=None):
    while len(schedule):
        ei = schedule.pop(0).lower()  # lower AST → compiled program
        ei.run(var_vals)              # execute on GPU
```

`ExecItem.run()`:
1. Ensures all buffers are allocated
2. Calls the compiled program with the buffer pointers
3. Updates global statistics (kernel count, FLOPs, memory bandwidth)

## Stage 9: Data Extraction

After `realize()`, the tensor's UOp points to a realized buffer. `._buffer()` returns it:

```python
def _buffer(self):
    x = self.cast(self.dtype.base).contiguous()
    return cast(Buffer, x.realize().uop.buffer).ensure_allocated()
```

Then `.data()` copies the result from GPU to CPU:

```python
def data(self):
    return self._buffer().as_memoryview().cast('f', self.shape)
```

And `.item()` extracts the single value:

```python
return self.data()[(0,)]  # → 16.0
```

## The Complete Pipeline

```
Python: Tensor.ones(4,4).sum().item()
  │
  ├─ Tensor.ones(4,4)     → UOp: CONST(1.0) → EXPAND(4,4)
  ├─ .sum()                → UOp: → REDUCE_AXIS(ADD)
  ├─ .item()               → triggers realize()
  │
  ├─ schedule_with_vars()
  │   ├─ transform_to_call()   → wrap in CALL
  │   ├─ get_kernel_graph()
  │   │   ├─ rangeify           → shapes become loops
  │   │   ├─ symbolic           → simplify expressions
  │   │   ├─ bufferize          → add buffer read/write
  │   │   └─ split_kernels      → one kernel
  │   ├─ create_schedule()      → topological sort
  │   └─ memory_planner()       → optimize buffers
  │
  ├─ run_schedule()
  │   ├─ lower()               → codegen → compile
  │   └─ run()                 → execute on GPU
  │
  └─ copyout → 16.0
```

## Seeing It All with DEBUG

```bash
DEBUG=4 python -c "from tinygrad import Tensor; print(Tensor.ones(4,4).sum().item())"
```

`DEBUG=4` shows the generated kernel source code. `DEBUG=2` shows kernel execution stats. `DEBUG=5` shows the full UOp graph at each stage.

## Exercises

1. **Run the trace**: Execute the command above with `DEBUG=4`. Read the generated kernel code. Does it have loops, or did constant folding eliminate them?

2. **Bigger example**: Try `Tensor.rand(4,4).sum().item()` with `DEBUG=4`. This can't be constant-folded — you should see actual loops in the kernel.

3. **Two kernels**: Try `Tensor.rand(4,4).sum().sqrt().item()`. How many kernels? (The sqrt should fuse with the sum.)

4. **Pipeline stages**: Set `VIZ=1` and run the example. The visualizer shows the UOp graph at each transformation stage.

## Source Code Map

| File | What to read |
|------|-------------|
| `tinygrad/tensor.py:252-292` | `schedule_with_vars` and `realize` |
| `tinygrad/engine/schedule.py:81-138` | `complete_create_schedule_with_vars` |
| `tinygrad/schedule/rangeify.py:483-514` | `get_kernel_graph` — the full pipeline |
| `tinygrad/engine/realize.py:156-212` | `ExecItem.run` and `run_schedule` |
