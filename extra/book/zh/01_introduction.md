# 第一章：简介

本章为你建立 tinygrad 的工作原理的心智模型。读完后，你将理解 tinygrad 是什么、它与 PyTorch 有何不同，以及当你运行一个简单的 Tensor 操作时会发生什么。

## 什么是 Tinygrad？

Tinygrad 是一个约 10,000 行 Python 代码的深度学习框架。它可以在 GPU（NVIDIA、AMD、Apple Metal 等）上训练和运行神经网络，就像 PyTorch 一样——但架构从根本上更简单。

PyTorch 拥有独立的自动求导、调度、内存管理子系统以及数十个手工调优的内核，而 tinygrad 只有**一个核心思想**：将一切表示为微操作（UOps）图，然后通过 Pattern Matcher 将该图重写为 GPU 代码。

## 你的第一个 Tinygrad 程序

```python
from tinygrad import Tensor

a = Tensor([1.0, 2.0, 3.0, 4.0])
b = Tensor([5.0, 6.0, 7.0, 8.0])
c = a + b
print(c.numpy())  # [ 6.  8. 10. 12.]
```

这看起来像 PyTorch，这是有意为之的。API 是熟悉的，不同之处在底层。

## 惰性求值

最重要的一点：**tinygrad 是惰性的**。当你写 `c = a + b` 时，什么都不会被计算。Tinygrad 只是构建了一个图：

```python
c = a + b
print(c)  # <Tensor <UOp METAL (4,) float (<Ops.ADD: 44>, None)> on METAL with grad None>
```

加法还没有发生。`c` 只是计算图中的一个节点，表示"我是 `a` 和 `b` 相加的结果"。只有当你显式请求结果时，计算才会发生：

- `c.numpy()` — 计算并返回 numpy 数组
- `c.realize()` — 计算并保留在 GPU 上
- `c.item()` — 计算并返回 Python 标量
- `c.tolist()` — 计算并返回 Python 列表

这种惰性求值使 tinygrad 能够融合操作并生成高效的内核。

## 调用 Realize 时发生了什么

让我们追踪当你对 `c = a + b` 调用 `c.numpy()` 时发生了什么：

```
1. 构建 UOp 图：    ADD(COPY(a), COPY(b))
2. 调度：           确定要运行哪些内核
3. Rangeify：       将形状转换为循环变量
4. 代码生成：       将 UOps 降级为 GPU 源代码
5. 编译：           将源代码编译为 GPU 二进制文件
6. 执行：           在 GPU 上调度内核
7. 回传：           将结果传回 CPU 供 numpy() 使用
```

你可以通过设置 `DEBUG=4` 来查看生成的 GPU 代码：

```bash
DEBUG=4 NOOPT=1 python -c "
from tinygrad import Tensor
a = Tensor([1.0, 2.0, 3.0, 4.0])
b = Tensor([5.0, 6.0, 7.0, 8.0])
print((a + b).numpy())
"
```

在 Apple Metal 上，你会看到类似这样的内容：

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

这个内核启动 4 个 GPU 线程（每个元素一个）。每个线程从 `data1` 和 `data2` 加载一个元素，将它们相加，并将结果存储到 `data0`。

在 CUDA 上，你会看到几乎相同的代码，只是用 `blockIdx.x` 代替 `gid.x`。

## Tinygrad 的两个部分

Tinygrad 在概念上分为两个部分：

**1. 机器学习框架**（tensor.py, nn/）
- Tensor 操作：matmul、conv2d、relu、softmax 等
- 自动求导：用于反向传播的自动微分
- 神经网络层：Linear、Conv2d、BatchNorm 等

这部分就像一个精简版的 PyTorch。如果你了解 PyTorch，你已经知道如何使用它。

**2. 编译器**（schedule/、codegen/、renderer/、runtime/）
- 调度：决定将哪些操作融合到哪些内核中
- 代码生成：将操作转换为 GPU 源代码
- 渲染：输出最终的源代码字符串
- 运行时：在特定硬件上编译和执行

这是 tinygrad 独特的部分，也是本书的重点。

## 一切都是逐元素操作 + 归约

Tinygrad 的关键洞察：几乎所有 Tensor 操作都可以表示为**逐元素操作**和**归约**的组合，加上零开销的**形状变换**。

```python
# 逐元素操作：输出与输入形状相同
c = a + b           # ADD
c = a * b           # MUL
c = a.relu()        # MAX(a, 0)
c = a.exp()         # EXP

# 归约：输出元素更少
c = a.sum(axis=0)   # 沿轴求和
c = a.max(axis=1)   # 沿轴取最大值

# 形状变换：数据不移动，只是重新解释
c = a.reshape(2, 2) # RESHAPE
c = a.T             # PERMUTE（转置）
c = a.expand(4, 4)  # EXPAND（广播）
```

即使是复杂的操作也可以分解为这些原语：

```python
# 均值 = 求和 / 计数
def mean(x, axis):
    return x.sum(axis) / x.shape[axis]

# Softmax = exp(x - max(x)) / sum(exp(x - max(x)))
def softmax(x, axis=-1):
    e = (x - x.max(axis, keepdim=True)).exp()
    return e / e.sum(axis, keepdim=True)

# 矩阵乘法 = reshape + expand + multiply + sum
# （在第 9 章中详细介绍）
```

这种分解使 tinygrad 保持精简。tinygrad 不需要实现 200 个优化内核，而是拥有一个通用编译器来处理约 20 个原始操作。

## DEBUG 级别

在本书中，我们将使用环境变量来检查 tinygrad 的内部机制：

| 变量 | 显示内容 |
|----------|-------------|
| `DEBUG=1` | 内核名称和内存 |
| `DEBUG=2` | 内核计时和 GFLOPS |
| `DEBUG=3` | 内核 AST（UOp 图） |
| `DEBUG=4` | 生成的源代码 |
| `DEBUG=5` | 完整的 UOp 树 + 源代码 |
| `NOOPT=1` | 禁用内核优化 |
| `DEBUG_RANGEIFY=1` | 显示形状如何变为循环 |
| `VIZ=1` | 打开图可视化工具 |

试试看：

```bash
# 查看内核计时
DEBUG=2 python -c "from tinygrad import Tensor; (Tensor.ones(1000,1000) @ Tensor.ones(1000,1000)).realize()"

# 查看矩阵乘法的生成代码
DEBUG=4 NOOPT=1 python -c "from tinygrad import Tensor; (Tensor.ones(4,4) @ Tensor.ones(4,4)).realize()"
```

## 练习

1. **比较输出**：对 `Tensor.ones(4).sum().realize()` 运行 `DEBUG=4 NOOPT=1`。阅读生成的内核——循环做了什么？

2. **计算内核数量**：对 `(Tensor.ones(4,4) @ Tensor.ones(4,4)).relu().realize()` 运行 `DEBUG=2`。生成了多少个内核？relu 是单独的内核还是被融合了？

3. **尝试不同操作**：对 `Tensor.ones(4,4).sum(axis=0).realize()` 和 `Tensor.ones(4,4).sum(axis=1).realize()` 运行 `DEBUG=4 NOOPT=1`。内核有何不同？

## 源代码导航

| 文件 | 功能 |
|------|-------------|
| `tinygrad/tensor.py` | 公共 Tensor API（约 5000 行） |
| `tinygrad/dtype.py` | 数据类型定义 |
| `tinygrad/device.py` | Device/Buffer/Compiler 抽象 |
| `tinygrad/nn/__init__.py` | 神经网络层 |
| `tinygrad/gradient.py` | 自动求导实现 |
