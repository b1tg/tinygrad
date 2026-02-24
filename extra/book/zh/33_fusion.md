# 第33章：Kernel Fusion —— 当操作合并时

Kernel fusion 是深度学习编译器中最重要的优化。它不再为每个操作运行一个 GPU kernel，而是将融合后的操作共享一个 kernel —— 消除了中间的内存读写。

## 为什么 Fusion 很重要

没有 fusion 时，`(a + b) * c` 将会是：

```
Kernel 1: read a, read b → compute a+b → write temp
Kernel 2: read temp, read c → compute temp*c → write result

Memory traffic: 5 reads + 2 writes
```

有了 fusion：

```
Kernel 1: read a, read b, read c → compute (a+b)*c → write result

Memory traffic: 3 reads + 1 write
```

GPU 计算很快，内存访问很慢。Fusion 削减了内存流量，通常可达 2-5 倍。

## Fusion 在 tinygrad 中如何工作

Fusion 发生在调度阶段（第4章），具体是在 `remove_bufferize` 函数中。核心思想是：

**如果从中间缓冲区读取的代价比重新计算更高，则移除该中间缓冲区。**

### 中间缓冲区

当调度器遇到如下操作时：

```python
a = Tensor.rand(1000)
b = a + 1        # creates an intermediate buffer for a+1
c = b * 2        # reads from that intermediate buffer
```

计算图最初包含：
```
BUFFER(a) → ADD(1) → BUFFERIZE → INDEX → MUL(2) → STORE(result)
```

`BUFFERIZE` 标记了将要创建中间缓冲区的位置。`remove_bufferize` 决定是保留还是移除它。

### Fusion 决策

```python
def remove_bufferize(src, buf, idx):
    # 1. Never remove user-requested contiguous buffers
    if src.op in ALWAYS_RUN_OPS or not buf.arg.removable:
        return None  # keep the buffer

    # 2. Count accessed buffers
    accessed_buffers = [...]  # all buffers the fused kernel would access
    if len(accessed_buffers) > 3:
        return None  # too many buffers → keep separate

    # 3. Check if reduces access buffers
    if buffer_in_reduce:
        return None  # reducing over buffered data → keep separate

    # 4. If we get here, remove the buffer (fuse!)
    return src.substitute(range_mapping)
```

有三个条件会阻止 fusion：

**1. 输入缓冲区过多（> 3）**

融合后的 kernel 如果有太多缓冲区参数，会因寄存器压力而变慢：
```python
# Won't fuse: would need 4+ input buffers
result = a + b + c + d + e
```

**2. 访问缓冲区的 reduce 操作**

如果一个 reduce 操作从缓冲区中读取数据，融合将意味着每个 reduce 步骤都要重新读取该缓冲区：
```python
# Won't fuse across the reduce:
temp = big_matrix @ weight     # this becomes a buffer
result = temp.sum(axis=1)      # reduce reads temp many times
```

**3. 用户请求的 contiguous 操作**

当你调用 `.contiguous()` 时，tinygrad 总是会实体化一个缓冲区。

### 当 Fusion 成功时

Fusion 通过将中间缓冲区的范围变量替换为消费者的索引来实现：

```python
# Before fusion:
# Kernel 1: for i in range(N): buf[i] = a[i] + 1
# Kernel 2: for i in range(N): result[i] = buf[i] * 2

# After fusion:
# Kernel 1: for i in range(N): result[i] = (a[i] + 1) * 2
```

替换操作将缓冲区读取替换为生成该数据的计算。

## 实际中哪些操作会被融合

### 逐元素链：总是被融合

```python
x = Tensor.rand(1000)
y = x.relu().sigmoid().tanh()  # all fused into one kernel
```

### Reduce + 逐元素：会被融合

```python
x = Tensor.rand(100, 100)
y = x.sum(axis=1).relu()  # sum + relu in one kernel
```

### 逐元素 + reduce：会被融合

```python
x = Tensor.rand(100, 100)
y = (x * 2).sum(axis=1)  # multiply + sum in one kernel
```

### Reduce + reduce：不会被融合

```python
x = Tensor.rand(100, 100)
y = x.sum(axis=1).sum()  # two separate kernels
```

两个 reduce 操作无法共享一个 kernel，因为它们需要不同的同步模式。

### Reshape 和 permute：零开销

移动操作根本不会创建 kernel —— 它们只是改变索引的计算方式：

```python
x = Tensor.rand(4, 8)
y = x.reshape(2, 16).permute(1, 0)  # no kernel, just index math
z = y.sum()  # the reshape+permute are folded into this kernel's indexing
```

## 使用 DEBUG 观察 Fusion

```bash
DEBUG=2 python -c "
from tinygrad import Tensor
x = Tensor.rand(1000)
y = (x + 1).relu().sum()
print(y.item())
"
```

使用 `DEBUG=2` 时，你会看到 kernel 信息。一条融合链会显示为单个 kernel。

## 在流水线中的位置

```
Tensor ops → UOp graph → Scheduling → Rangeify → Codegen → GPU
                              ↑
                        Fusion happens here
                    (remove_bufferize decides
                     which intermediates to keep)
```

## 部分连续性（PCONTIG）

对于高级场景，tinygrad 通过 `PCONTIG` 支持部分 fusion：

```python
# With PCONTIG > 2, some dimensions can be fused while others are buffered
# This is useful when the output-to-input size ratio is very large
out_in_ratio = prod(buf.shape) / sum(x.size for x in accessed_buffers)
if out_in_ratio < 10: return None  # don't fuse
```

这处理了某些维度适合融合而其他维度不适合的情况。

## 练习

1. **统计 kernel 数量**：运行 `DEBUG=2 python -c "from tinygrad import Tensor; x = Tensor.rand(100,100); y = (x*2+1).relu().sum(); print(y.item())"`。有多少个 kernel？（提示：应该是 2 个 —— 一个用于 rand，一个用于融合后的乘法+加法+relu+sum。）

2. **打破 fusion**：什么会强制产生 kernel 边界？尝试在链中插入 `.contiguous()` 或 `.realize()`，然后重新统计 kernel 数量。

3. **阅读代价模型**：在 `tinygrad/schedule/rangeify.py` 中找到 `remove_bufferize`。在阻止 fusion 之前，访问缓冲区的最大数量是多少？

4. **Reduce 屏障**：为什么两个 reduce 操作不能融合为一个 kernel？想想在 reduce 过程中 GPU 线程在做什么。

## 源代码索引

| 文件 | 阅读内容 |
|------|---------|
| `tinygrad/schedule/rangeify.py:167-229` | `remove_bufferize` —— fusion 决策函数 |
| `tinygrad/schedule/rangeify.py:483-514` | `get_kernel_graph` —— 编排整个 kernel 图 |
| `tinygrad/engine/schedule.py:18-63` | `create_schedule` —— 将 kernel 图线性化为执行顺序 |
