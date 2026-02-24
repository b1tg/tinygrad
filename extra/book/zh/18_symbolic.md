# 第18章：符号数学与 Memoryview

tinygrad 需要对尚未确定为具体数值的表达式进行数学运算——循环边界、张量维度、索引计算。本章介绍 tinygrad 的符号数学系统和基于 memoryview 的内存模型。

## 符号表达式

当 tinygrad 编译一个 kernel 时，它需要计算如下的索引表达式：

```python
# For a (M, N) matmul where M and N might be symbolic:
index = row * N + col
```

如果 `N` 是编译时常量（例如 64），则可以简化为 `row * 64 + col`。但如果 `N` 是运行时变量（例如来自动态 batch 大小），它将保持符号形式。

tinygrad 使用 UOp 来表示符号表达式：

```python
from tinygrad.uop.ops import UOp, Ops
from tinygrad.dtype import dtypes

# Create a symbolic variable
N = UOp(Ops.DEFINE_VAR, dtypes.int, arg="N")

# Build symbolic expressions
idx = N * UOp(Ops.CONST, dtypes.int, arg=2)  # N * 2
print(idx.op)  # Ops.MUL
```

## 符号简化器

`symbolic` PatternMatcher 自动简化表达式：

```python
# x + 0 -> x
# x * 1 -> x
# x * 0 -> 0
# (x + c1) + c2 -> x + (c1 + c2)
# (x * c1) * c2 -> x * (c1 * c2)
# x // 1 -> x
# x % 1 -> 0
# (x * c) // c -> x  (when c > 0)
# (x * c) % c -> 0
```

这些规则对于 rangeify 至关重要。当一次 reshape 将 `r0` 转换为 `r0 // 3` 和 `r0 % 3`，而随后的 reshape 又将它们合并回来时，简化器必须能够还原出 `r0`。

示例：
```
reshape(6) -> reshape(2, 3): indices = [r0 // 3, r0 % 3]
reshape(2, 3) -> reshape(6): index = (r0 // 3) * 3 + (r0 % 3)

Simplification: (r0 // 3) * 3 + (r0 % 3) = r0  ✓
```

## 变量边界

符号变量具有边��（最小值和最大值），这使得基于范围的优化成为可能：

```python
# If r0 ranges from 0 to 5:
# r0 // 6 is always 0  (since r0 < 6)
# r0 % 6 is always r0  (since r0 < 6)
# r0 >= 0 is always true
# r0 < 0 is always false
```

这些边界会在表达式中传播：

```python
# r0 in [0, 5], r1 in [0, 3]
# r0 + r1 in [0, 8]
# r0 * r1 in [0, 15]
# r0 * 4 + r1 in [0, 23]
```

边界跟踪使 tinygrad 能够消除死代码路径：
```python
# if (r0 < 0) { ... }  // dead code, r0 >= 0 always
# if (r0 < 6) { ... }  // always true, can remove the if
```

## 整数除法与取模

整数除法和取模是最难简化的运算。tinygrad 在 `tinygrad/uop/divandmod.py` 中有专门的处理逻辑：

```python
# Key identities:
# (a * b + c) // b = a + c // b   (when 0 <= c < b)
# (a * b + c) % b = c % b         (when 0 <= c < b)
# (a // b) // c = a // (b * c)
# (a % (b * c)) // b = (a // b) % c
```

这些恒等式正是使连续 reshape 链能够编译为高效代码的关键所在。

## Memoryview：零拷贝内存访问

tinygrad 使用 Python 的 `memoryview` 实现 CPU 端零拷贝数据访问：

```python
from tinygrad import Tensor

t = Tensor([1.0, 2.0, 3.0, 4.0])
t.realize()

# Get a memoryview into the GPU buffer (if possible)
mv = t.lazydata.buffer.as_memoryview()
# This is a zero-copy view — modifying mv modifies the buffer directly
```

### Buffer 内存工作原理

tinygrad 中的 `Buffer` 封装了设备特定的内存分配：

```python
from tinygrad.device import Buffer
from tinygrad.dtype import dtypes

# Allocate 16 floats on CPU
buf = Buffer('CPU', 16, dtypes.float)
buf.ensure_allocated()

# Get a memoryview for direct access
mv = buf.as_memoryview()
mv = mv.cast('f')  # interpret as float32
mv[0] = 42.0
print(mv[0])  # 42.0
```

该机制被广泛应用于：
- AMD 模拟器（第12章）——WaveState 使用指向 Buffer 对象的 memoryview 来访问寄存器文件
- 数据加载——将 numpy 数组拷贝到 GPU 时通过 memoryview 进行
- PYTHON 后端——纯 Python 执行使用 memoryview 进行内存访问

### cast() 技巧

Python 的 `memoryview.cast()` 可以在不拷贝数据的情况下将字节重新解释为不同类型：

```python
import struct

# Create bytes representing a float32
data = struct.pack('f', 3.14)
mv = memoryview(bytearray(data))

# Interpret as float
print(mv.cast('f')[0])  # 3.14

# Interpret as uint32 (same bits, different interpretation)
print(mv.cast('I')[0])  # 1078523331 (IEEE 754 encoding of 3.14)
```

这就是 tinygrad 在不进行数据拷贝的情况下处理位转换和数据类型重新解释的方式。

## 练习

1. **手动简化**：给定 `r0` 的范围为 [0, 11]，简化 `(r0 // 4) * 4 + (r0 % 4)`。验证它等于 `r0`。

2. **边界传播**：如果 `r0` 的范围为 [0, 3]，`r1` 的范围为 [0, 7]，那么 `r0 * 8 + r1` 的边界是什么？我们能否保证 `r0 * 8 + r1 < 32`？

3. **Memoryview 操作**：在 CPU 上创建一个 tinygrad Buffer，通过 memoryview 写入值，然后读取回来：
   ```python
   from tinygrad.device import Buffer
   from tinygrad.dtype import dtypes
   buf = Buffer('CPU', 4, dtypes.float).ensure_allocated()
   mv = buf.as_memoryview(force_zero_copy=True).cast('f')
   mv[0] = 1.0; mv[1] = 2.0; mv[2] = 3.0; mv[3] = 4.0
   print(list(mv))
   ```

## 源码导航

| 文件 | 阅读要点 |
|------|----------|
| `tinygrad/uop/symbolic.py` | 符号简化器 PatternMatcher |
| `tinygrad/uop/divandmod.py` | 整数除法/取模简化规则 |
| `tinygrad/device.py` | `Buffer` 类和 `as_memoryview()` |
| `tinygrad/dtype.py` | DType 系统，存储格式 |
