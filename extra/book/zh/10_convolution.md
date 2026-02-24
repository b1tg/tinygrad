# 第10章：卷积

卷积是深度学习中第二重要的操作（仅次于矩阵乘法）。本章展示 tinygrad 如何使用相同的 reshape/expand/sum 原语来实现 conv2d——不需要专门的卷积内核。

## 卷积做了什么

二维卷积将一个小的权重核在输入图像上滑动，在每个位置计算逐元素乘法和求和：

```python
from tinygrad import Tensor

# 输入：1个批次，1个通道，4x4 图像
inp = Tensor([[[[0, 1, 2, 3],
                [4, 5, 6, 7],
                [8, 9, 10, 11],
                [12, 13, 14, 15]]]], dtype='float')

# 权重：1个输出通道，1个输入通道，3x3 核
weight = Tensor.ones(1, 1, 3, 3)

out = inp.conv2d(weight)
print(out.shape)   # (1, 1, 2, 2)
print(out.numpy())
# [[[[45. 54.]
#    [81. 90.]]]]
```

位置 (0,0) 的输出是左上角 3x3 区域的和：`0+1+2+4+5+6+8+9+10 = 45`。

## _pool 技巧

tinygrad 使用一个叫做 `_pool` 的辅助函数来实现卷积。它重新排列输入，使得每个"窗口"（核滑过的区域）变成一个单独的切片：

```python
from tinygrad import Tensor

inp = Tensor([[[[0, 1, 2, 3],
                [4, 5, 6, 7],
                [8, 9, 10, 11],
                [12, 13, 14, 15]]]], dtype='float')

pooled = inp._pool(k_=(3, 3), stride=1, dilation=1)
print(pooled.shape)  # (1, 1, 2, 2, 3, 3)
```

池化输出的形状为 `(batch, channels, out_h, out_w, kernel_h, kernel_w)`。每个 `(out_h, out_w)` 位置包含核会看到的 3x3 区域：

```
pooled[0, 0, 0, 0] = [[0,  1,  2],   # 左上角 3x3 区域
                       [4,  5,  6],
                       [8,  9, 10]]

pooled[0, 0, 0, 1] = [[1,  2,  3],   # 向右移动1
                       [5,  6,  7],
                       [9, 10, 11]]

pooled[0, 0, 1, 0] = [[4,  5,  6],   # 向下移动1
                       [8,  9, 10],
                       [12, 13, 14]]

pooled[0, 0, 1, 1] = [[5,  6,  7],   # 向右和向下移动
                       [9, 10, 11],
                       [13, 14, 15]]
```

一旦区域排列好，卷积就变成逐元素乘法 + 求和：

```python
# 卷积 = pooled * weight，然后对核维度求和
result = (pooled * weight).sum(axis=(-2, -1))
print(result.numpy())
# [[[[45. 54.]
#    [81. 90.]]]]
```

## _pool 的工作原理（无数据移动！）

神奇之处在于 `_pool` 只使用移动操作——reshape、expand、shrink——来创建区域。**没有数据被复制。** 这些区域是原始数据的虚拟视图：

```python
# 简化的 _pool 实现：
def _pool(x, k_, stride, dilation):
    # 1. 通过步幅技巧用 expand 创建重叠窗口
    # 2. 用 shrink 选择有效区域
    # 3. reshape 为 (batch, channels, out_h, out_w, kernel_h, kernel_w)

    # expand 使用 stride=stride 来创建滑动窗口效果
    # shrink 移除填充/溢出
    # 结果：每个输出位置都有自己的核大小区域视图
    pass
```

底层通过操纵步幅，使相邻输出位置指向输入的重叠区域——就像 NumPy 的 `as_strided` 一样。

## 步幅和膨胀

**步幅**控制核在位置之间移动的距离：

```python
# 步幅1：核每次滑动1个像素
pooled = inp._pool(k_=(2, 2), stride=1, dilation=1)
print(pooled.shape)  # (1, 1, 3, 3, 2, 2)  -- 3x3 个输出位置

# 步幅2：核每次滑动2个像素
pooled = inp._pool(k_=(2, 2), stride=2, dilation=1)
print(pooled.shape)  # (1, 1, 2, 2, 2, 2)  -- 2x2 个输出位置
```

**膨胀**在核模式中创建间隔：

```python
# 膨胀1：普通卷积
# 核看到：[0,1], [4,5]

# 膨胀2：跳过每隔一个元素
# 核看到：[0,2], [8,10]
pooled = inp._pool(k_=(2, 2), stride=1, dilation=2)
```

## 完整的 conv2d 流水线

完整的 `conv2d` 操作：

```python
# 简化自 tensor.py
def conv2d(x, weight, stride=1, padding=0, dilation=1, groups=1):
    # 1. 如果需要，填充输入
    if padding:
        x = x.pad(...)

    # 2. 池化：创建滑动窗口视图
    x = x._pool(k_=weight.shape[-2:], stride=stride, dilation=dilation)
    # shape: (batch, in_channels, out_h, out_w, kernel_h, kernel_w)

    # 3. reshape 以与权重相乘
    # x: (batch, groups, in_channels//groups, out_h, out_w, kernel_h, kernel_w)
    # w: (groups, out_channels//groups, in_channels//groups, kernel_h, kernel_w)

    # 4. 乘法并对 (in_channels, kernel_h, kernel_w) 求和
    return (x * weight).sum(axis=(-3, -2, -1))
```

生成的内核将所有这些融合为一个 GPU 程序——池化视图没有中间分配。

## 填充

填充在输入边界周围添加零。在 tinygrad 中，这使用 `PAD` 移动操作：

```python
from tinygrad import Tensor

x = Tensor.ones(1, 1, 3, 3)
# 在空间维度的每一侧填充1个像素
out = x.conv2d(Tensor.ones(1, 1, 3, 3), padding=1)
print(out.shape)  # (1, 1, 3, 3) -- 与输入相同的空间大小
```

PAD 操作在 ShapeTracker 中创建一个掩码。原始数据之外的索引返回0，这正是零填充所做的。

## 分组卷积

分组将输入和输出通道分成独立的集合：

```python
# groups=2 意味着：
# - 前半部分输出通道只看到前半部分输入通道
# - 后半部分输出通道只看到后半部分输入通道
out = inp.conv2d(weight, groups=2)
```

深度可分离卷积（用于 MobileNet）是极端情况，其中 `groups = in_channels`，意味着每个通道独立卷积。

## 与矩阵乘法的联系

卷积可以看作是伪装的矩阵乘法。`_pool` 操作创建一个 im2col 矩阵，乘法+求和就是矩阵乘法：

```
im2col: (batch*out_h*out_w) x (in_channels*kernel_h*kernel_w)
weight: (out_channels) x (in_channels*kernel_h*kernel_w)
output: (batch*out_h*out_w) x (out_channels)
```

tinygrad 的方法是等价的，但不显式创建 im2col 矩阵——它通过 `_pool` 的形状/步幅操纵隐式实现。

## 练习

1. **手动卷积**：仅使用 `._pool()`、`*` 和 `.sum()` 实现 `[1,2,3,4,5]` 与核 `[1,1,1]` 的一维卷积。用 `Tensor.conv2d` 验证。

2. **追踪 _pool**：对 (1,1,4,4) 输入、核 (2,2)、步幅2，打印 `_pool` 内部每一步的形状。

3. **生成的代码**：对一个小的 conv2d 运行 `DEBUG=4 NOOPT=1`。阅读内核——识别加载模式和累加循环。

## 源代码导航

| 文件 | 阅读内容 |
|------|-------------|
| `tinygrad/tensor.py` | `Tensor.conv2d()` -- 卷积 API |
| `tinygrad/tensor.py` | `Tensor._pool()` -- 滑动窗口辅助函数 |
