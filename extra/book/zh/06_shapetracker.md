# 第六章：ShapeTracker — 无需拷贝的视图

ShapeTracker 是 tinygrad 在不移动内存中数据的情况下表示 tensor 形状、步长和视图的方式。如果你用过 PyTorch，你知道 `.reshape()`、`.transpose()` 和 `.expand()` 是"免费"操作。ShapeTracker 就是实现这一点的机制。

## 问题

一个 2x3 矩阵 `[[1,2,3],[4,5,6]]` 在内存中存储为平坦数组：`[1, 2, 3, 4, 5, 6]`。要访问元素 `[row, col]`，你计算内存偏移量：`row * 3 + col`。

数字 `3` 和 `1` 是**步长（strides）**——沿每个维度移动一步时要跳过多少个元素：

```
形状:   (2, 3)
步长:   (3, 1)
公式:   index = row * 3 + col * 1
```

```python
from tinygrad import Tensor

a = Tensor([[1, 2, 3], [4, 5, 6]])
# 内存布局: [1, 2, 3, 4, 5, 6]
# a[1, 2] -> 偏移量 = 1*3 + 2*1 = 5 -> 值为 6
```

## 转置是免费的

要转置这个矩阵，不需要重新排列内存。只需交换步长：

```
原始:     shape=(2,3), strides=(3,1)  -> index = row*3 + col*1
转置后:   shape=(3,2), strides=(1,3)  -> index = row*1 + col*3
```

两个公式访问相同的内存，只是顺序不同：

```python
# 原始:  [0,0]=0, [0,1]=1, [0,2]=2, [1,0]=3, [1,1]=4, [1,2]=5
# 转置:  [0,0]=0, [0,1]=3, [1,0]=1, [1,1]=4, [2,0]=2, [2,1]=5
```

没有拷贝数据。"转置"只是对相同内存的不同解读方式。

## View

**View** 是核心数据结构，包含：

```python
# from tinygrad/shape/view.py
@dataclass
class View:
    shape: tuple[int, ...]     # 逻辑形状
    strides: tuple[int, ...]   # 内存步长
    offset: int                # 内存中的起始偏移量
    mask: tuple|None           # 有效区域（���于填充）
    contiguous: bool           # 内存是否顺序排列
```

### 常见 View

```python
# 连续的 2x3 矩阵
# shape=(2,3), strides=(3,1), offset=0
# index = row*3 + col

# 转置后 (3x2)
# shape=(3,2), strides=(1,3), offset=0
# index = row*1 + col*3

# 切片（4x4 的第 1-2 行）
# shape=(2,4), strides=(4,1), offset=4
# index = 4 + row*4 + col

# 广播 (1x4 扩展为 3x4)
# shape=(3,4), strides=(0,1), offset=0
# index = row*0 + col*1 = col  （忽略 row！）

# 标量广播
# shape=(4,4), strides=(0,0), offset=0
# index = 0  （始终读取同一个元素）
```

注意广播时步长为 `0`——意味着"不沿这个维度移动"，所以每一行都读取相同的数据。

## 移动操作作为 View 变换

每个移动操作变换 View：

### RESHAPE

改变形状但保持线性索引映射：

```python
from tinygrad import Tensor
a = Tensor.ones(6)        # shape=(6,), strides=(1,)
b = a.reshape(2, 3)       # shape=(2,3), strides=(3,1)
c = b.reshape(3, 2)       # shape=(3,2), strides=(2,1)
# 三者以相同顺序访问相同的 6 个元素
```

### PERMUTE（转置）

通过重排步长来重排维度：

```python
a = Tensor.ones(2, 3, 4)  # strides=(12, 4, 1)
b = a.permute(2, 0, 1)    # strides=(1, 12, 4), shape=(4, 2, 3)
```

### EXPAND（广播）

将扩展维度的步长设为 0：

```python
a = Tensor.ones(1, 4)     # strides=(4, 1) 或 (0, 1)
b = a.expand(3, 4)        # strides=(0, 1), shape=(3, 4)
# 所有 3 行指向相同的 4 个元素
```

### SHRINK（切片）

调整偏移量和形状：

```python
a = Tensor.ones(10)       # strides=(1,), offset=0
b = a[3:7]                # strides=(1,), offset=3, shape=(4,)
```

### PAD

添加掩码来指示有效区域：

```python
a = Tensor.ones(3)        # [1, 1, 1]
b = a.pad(((1, 1),))      # [0, 1, 1, 1, 0], mask=((1, 4),)
# 掩码外的元素被视为零
```

### FLIP

取反步长并调整偏移量：

```python
a = Tensor.ones(4)        # strides=(1,), offset=0
b = a.flip(0)             # strides=(-1,), offset=3
# 反向访问元素: 3, 2, 1, 0
```

## 多视图 ShapeTracker

有时单个 View 不够用。如果你对非连续 tensor（如转置后的矩阵）做 reshape，需要两个视图：

```python
a = Tensor.ones(2, 3)     # View 1: shape=(2,3), strides=(3,1)
b = a.permute(1, 0)       # View 1: shape=(3,2), strides=(1,3)  -- 非连续！
c = b.reshape(6)           # 无法用单个视图对非连续张量做 reshape
# 这就是 CONTIGUOUS 的用途 -- 它强制拷贝使其连续
# 或者 rangeify 通过 div/mod 分解来处理
```

在当前的 tinygrad（rangeify 时代之后）中，多视图 ShapeTracker 主要由 rangeify 的索引分解处理。当你对非连续 tensor 做 reshape 时，rangeify 生成 `div` 和 `mod` 表达式来计算正确的内存偏移量。

## 合并维度

一个重要的优化：具有兼容步长的相邻维度可以合并。如果 `stride[i] == shape[i+1] * stride[i+1]`，维度 `i` 和 `i+1` 可以合并：

```python
# shape=(2, 3, 4), strides=(12, 4, 1)
# 维度 0 步长 (12) == shape[1] * stride[1] (3 * 4 = 12) ✓
# 维度 1 步长 (4) == shape[2] * stride[2] (4 * 1 = 4) ✓
# 可以全部合并 -> shape=(24,), strides=(1,)
```

这用于简化 kernel 索引表达式。内存中连续的 3D tensor 可以被当作 1D 数组处理，生成更简单的 GPU 代码。

## 与 Rangeify 的关系

ShapeTracker 的概念就是 rangeify（第 5 章）转换为循环的内容。当 rangeify 处理移动操作时：

1. **RESHAPE** 中 range `[r0]` 变为 `[r0//3, r0%3]` — 除法/取模分解
2. **PERMUTE** 中 range `[r0, r1]` 变为 `[r1, r0]` — 交换 range
3. **EXPAND** 中 range `[r0, r1]` 变为 `[0, r1]` — 广播维度用常量
4. **SHRINK** 中 range `[r0]` 变为 `[r0 + offset]` — 平移
5. **FLIP** 中 range `[r0]` 变为 `[size-1-r0]` — 反转

ShapeTracker 的基于步长的索引直接转化为 rangeify 产生的 range 表达式。

## 练习

1. **计算步长**：对于形状 `(2, 3, 4)` 且行优先（C）顺序，步长是什么？验证：元素 `[1, 2, 3]` 应在偏移量 `1*12 + 2*4 + 3*1 = 23` 处。

2. **转置步长**：对于形状 `(4, 5)` 步长 `(5, 1)` 的矩阵，`.permute(1, 0)` 后步长是什么？

3. **广播步长**：对于形状 `(1, 5)` 步长 `(0, 1)` 的向量，扩展为形状 `(3, 5)` 后，元素 `[2, 3]` 访问内存中的哪个值？

4. **何时需要 contiguous？**：尝试 `Tensor.ones(2,3).permute(1,0).reshape(6)`。这行得通吗？为什么？rangeify 如何处理它？

## 源代码索引

| 文件 | 阅读内容 |
|------|---------|
| `tinygrad/shape/view.py` | `View` 类 — 核心形状/步长/偏移结构 |
| `tinygrad/schedule/indexing.py:142` | `apply_movement_op()` — 移动操作如何变换 range |
| `tinygrad/schedule/indexing.py:126` | `_apply_reshape()` — reshape 的除法/取模分解 |
