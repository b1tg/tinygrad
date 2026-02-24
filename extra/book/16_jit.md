# Chapter 16: The JIT

Compiling GPU kernels is expensive. The **JIT** (Just-In-Time compiler) caches compiled programs and, more importantly, captures entire computation graphs for replay — making repeated operations (like training loops) fast.

## Two Levels of Caching

### Level 1: Kernel Compilation Cache

Every kernel is cached by its source code hash. If the same UOp AST appears again, the compiled binary is reused:

```python
# First time: compile from source
# (a + b).realize()  -> compiles kernel E_4, ~10ms

# Second time: cache hit, no recompilation
# (c + d).realize()  -> reuses kernel E_4, ~0.01ms
```

This happens automatically. You don't need TinyJit for this.

### Level 2: TinyJit — Full Graph Capture

`TinyJit` goes further: it captures the entire schedule (what kernels to run, in what order, with what buffers) and replays it:

```python
from tinygrad import Tensor, TinyJit

@TinyJit
def add_and_sum(a, b):
    return (a + b).sum().realize()

# First call: builds the schedule, compiles kernels
result = add_and_sum(Tensor.ones(1000), Tensor.ones(1000))

# Second call: replays the captured schedule (much faster!)
result = add_and_sum(Tensor.randn(1000), Tensor.randn(1000))

# Third call: replay again
result = add_and_sum(Tensor.randn(1000), Tensor.randn(1000))
```

On the first call, TinyJit records every kernel launch and buffer operation. On subsequent calls, it replays the exact same sequence, only swapping in the new input buffers.

## Why TinyJit Matters for Training

A training step looks like this:

```python
@TinyJit
def train_step(x, y):
    pred = model(x)
    loss = (pred - y).square().mean()
    loss.backward()
    optimizer.step()
    return loss.realize()
```

Without JIT: each call rebuilds the UOp graph, reschedules, and recompiles (~100ms overhead).
With JIT: the first call records everything, subsequent calls just dispatch the same kernels (~0.1ms overhead).

## How It Works

```
Call 1 (capture):
  1. Run the function normally
  2. Record every kernel dispatch: (program, buffers, global_size, local_size)
  3. Identify which buffers are "variable" (change between calls) vs "fixed"
  4. Store the captured execution list

Call 2+ (replay):
  1. Substitute new input buffers for the variable ones
  2. Replay the recorded kernel dispatches
  3. Skip scheduling, codegen, compilation entirely
```

The replay is essentially a list of GPU kernel launches with no Python overhead.

## Constraints

TinyJit requires that the **computation structure stays the same** between calls:

```python
@TinyJit
def f(x):
    if x.shape[0] > 100:  # WRONG: shape-dependent control flow
        return x.sum()
    return x.mean()
```

Since the captured schedule depends on the shape, changing shapes between calls will fail. The rule: **same shapes, same ops, same graph structure**.

Variable inputs (different data but same shape) are fine. That's exactly what happens in training — same model, same batch size, different data.

## The Cache Key

TinyJit identifies when to replay by hashing:
- Input tensor shapes and dtypes
- Input tensor devices
- The function being called

If any of these change, it re-captures instead of replaying.

## Exercises

1. **Measure the speedup**: Time a function with and without `@TinyJit`:
   ```python
   import time
   from tinygrad import Tensor, TinyJit

   def f(x): return (x @ x).realize()
   jf = TinyJit(f)

   x = Tensor.randn(256, 256)
   # Warm up
   f(x); jf(x); jf(x)

   t0 = time.perf_counter()
   for _ in range(100): f(x)
   print(f"Without JIT: {(time.perf_counter()-t0)*10:.1f}ms per call")

   t0 = time.perf_counter()
   for _ in range(100): jf(x)
   print(f"With JIT: {(time.perf_counter()-t0)*10:.1f}ms per call")
   ```

2. **See the capture**: Use `DEBUG=2` with a JIT function. Notice that kernels are compiled on the first call but just dispatched on subsequent calls.

## Source Code Map

| File | What to read |
|------|-------------|
| `tinygrad/engine/jit.py` | `TinyJit` implementation |
| `tinygrad/engine/realize.py` | `run_schedule()` — where JIT intercepts |
