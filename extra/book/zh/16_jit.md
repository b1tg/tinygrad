# 第16章：JIT

编译 GPU 内核的开销很大。**JIT**（即时编译器）缓存已编译的程序，更重要的是，它能捕获整个计算图并重放——使重复操作（如训练循环）变得很快。

## 两级缓存

### 第1级：内核编译缓存

每个内核按其源代码哈希值缓存。如果相同的 UOp AST 再次出现，则复用已编译的二进制文件：

```python
# First time: compile from source
# (a + b).realize()  -> compiles kernel E_4, ~10ms

# Second time: cache hit, no recompilation
# (c + d).realize()  -> reuses kernel E_4, ~0.01ms
```

这是自动发生的。你不需要 TinyJit 就能获得这个功能。

### 第2级：TinyJit — 完整计算图捕获

`TinyJit` 更进一步：它捕获整个调度（运行哪些内核、以什么顺序、使用哪些缓冲区）并重放：

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

在第一次调用时，TinyJit 记录每次内核启动和缓冲区操作。在后续调用中，它重放完全相同的序列，只替换新的输入缓冲区。

## 为什么 TinyJit 对训练很重要

一个训练步骤看起来像这样：

```python
@TinyJit
def train_step(x, y):
    pred = model(x)
    loss = (pred - y).square().mean()
    loss.backward()
    optimizer.step()
    return loss.realize()
```

没有 JIT：每次调用都重建 UOp 图、重新调度、重新编译（约 100ms 开销）。
有 JIT：第一次调用记录所有内容，后续调用只需调度相同的内核（约 0.1ms 开销）。

## 工作原理

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

重放本质上是一系列 GPU 内核启动，没有 Python 开销。

## 约束条件

TinyJit 要求调用之间**计算结构保持不变**：

```python
@TinyJit
def f(x):
    if x.shape[0] > 100:  # WRONG: shape-dependent control flow
        return x.sum()
    return x.mean()
```

由于捕获的调度依赖于形状，在调用之间改变形状会失败。规则是：**相同的形状、相同的操作、相同的图结构**。

可变输入（不同数据但相同形状）是可以的。这正是训练中发生的情况——相同的模型、相同的批次大小、不同的数据。

## 缓存键

TinyJit 通过哈希以下内容来判断何时重放：
- 输入张量的形状和数据类型
- 输入张量的设备
- 被调用的函数

如果其中任何一项发生变化，它会重新捕获而不是重放。

## 练习

1. **测量加速效果**：对比有无 `@TinyJit` 的函数执行时间：
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

2. **查看捕获过程**：对 JIT 函数使用 `DEBUG=2`。注意内核在第一次调用时编译，但在后续调用中只是被调度。

## 源代码导航

| 文件 | 阅读内容 |
|------|----------|
| `tinygrad/engine/jit.py` | `TinyJit` 实现 |
| `tinygrad/engine/realize.py` | `run_schedule()` — JIT 拦截的位置 |
