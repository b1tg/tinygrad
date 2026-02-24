# 第五章：Rangeify — 从形状到循环

这是解释 tinygrad 核心思想的章节。如果你理解了 rangeify，你就理解了一个 10,000 行的框架如何与百万行的框架竞争。

## 问题

在 PyTorch 中，当你写下：

```python
a = torch.ones(2, 3)
b = a.T            # 转置
c = b.reshape(6)   # 展平
d = c.sum()        # 归约
```

实际发生了什么？PyTorch 为转置分配一个新 tensor，为 reshape 再分配一个，为 sum 又分配一个。每一步都读写内存。

tinygrad 做了不同的事情：**在你要求结果之前，它什么都不执行。** 当你调用 `.realize()` 或 `.item()` 时，tinygrad 查看整个操作链，生成一个一次性完成所有事情的 GPU kernel。问题是：怎么做到的？

答案是 **rangeify** — 将高层形状操作（reshape、permute、expand 等）转换为 GPU 可以执行的显式循环变量的算法。

## 实际运行

在理解理论之前，让我们看看 rangeify 实际产生了什么。运行这个：

```python
# 运行命令: DEBUG_RANGEIFY=1 python this_file.py
from tinygrad import Tensor

a = Tensor.ones(2, 3)
b = a.permute(1, 0)  # 转置: (2,3) -> (3,2)
c = b.reshape(6)     # 展平:   (3,2) -> (6,)
d = c.sum().realize() # 归约:   (6,)  -> (1,)
```

使用 `DEBUG_RANGEIFY=1`，你会看到类似这样的输出：

```
***  1 Ops.ASSIGN           (1,)                   [0]
     1 Ops.REDUCE_AXIS      (1,)                   [r0 -> 0]
     1 Ops.RESHAPE          (6,)                   [(r0//2)][(r0%2)] -> [r0]
     1 Ops.PERMUTE          (3, 2)                 [(r0%2) -> (r0//2)][(r0//2) -> (r0%2)]
     1 Ops.EXPAND           (2, 3)                 [0 -> (r0%2)][0 -> (r0//2)]
     1 Ops.CONST            ()
     1 Ops.PARAM            (1,)                   [0]
```

从下往上读。有一个循环变量 `r0`，从 0 到 5。每个移动操作变换 `r0` 到维度的映射方式：

- **EXPAND** 创建 (2,3) 形状，索引为 `(r0%2, r0//2)`
- **PERMUTE** 交换它们为 `(r0//2, r0%2)`
- **RESHAPE** 将它们合并回单个 `r0`
- **REDUCE_AXIS** 对 `r0` 求和，产生标量

转置和 reshape 没有分配任何内存。整个计算是：*"循环 `r0` 从 0 到 5，累加 `ones[r0//2, r0%2]`"*。一个 kernel，一个循环。

## 构建模块

### UOp：tinygrad 的通用中间表示

tinygrad 中的一切都是 **UOp**（微操作）。UOp 是有向无环图（DAG）中的节点，有四个字段：

```python
UOp(op=Ops.ADD, dtype=dtypes.float, src=(uop_a, uop_b), arg=None)
```

- `op`：什么操作（ADD、MUL、RESHAPE、REDUCE_AXIS 等）
- `dtype`：数据类型
- `src`：输入 UOp 的元组（DAG 的边）
- `arg`：额外数据（reshape 维度、reduce 轴等）

当你写 `Tensor([1,2,3]) + Tensor([4,5,6])` 时，tinygrad 构建一个 UOp DAG — 还没有发生任何计算：

```python
from tinygrad import Tensor

a = Tensor([1.0, 2.0, 3.0, 4.0])
b = Tensor([5.0, 6.0, 7.0, 8.0])
c = a + b

# c.uop 是:
# UOp(Ops.ADD, dtypes.float,
#   src=(
#     UOp(Ops.COPY, ..., src=(UOp(Ops.PARAM, ..., shape=(4,)), ...)),  # a
#     UOp(Ops.COPY, ..., src=(UOp(Ops.PARAM, ..., shape=(4,)), ...)),  # b
#   ))

print(c.uop.op)    # Ops.ADD
print(c.uop.shape)  # (4,)
```

### 移动操作

移动操作改变 tensor 的*逻辑形状*而不移动数据。有六种：

| 操作 | 功能 | 示例 |
|----|-------------|---------|
| `RESHAPE` | 改变形状，总大小不变 | `(6,)` -> `(2,3)` |
| `PERMUTE` | 重排维度 | `(2,3)` -> `(3,2)` (转置) |
| `EXPAND` | 广播大小为 1 的维度 | `(1,3)` -> `(4,3)` |
| `SHRINK` | 取切片 | `(10,)` -> `(3,)` (第 2..5 个元素) |
| `PAD` | 在边缘添加零 | `(3,)` -> `(5,)` (每侧补 1) |
| `FLIP` | 反转一个维度 | `[1,2,3]` -> `[3,2,1]` |

在 PyTorch 中，这些称为"视图操作"——它们创建数据的新*视图*。在 tinygrad 中，它们成为惰性图中的 UOp 节点。

### RANGE 节点

`RANGE` UOp 表示一个循环变量：

```python
from tinygrad.uop.ops import UOp, Ops, AxisType
from tinygrad.dtype import dtypes

# 从 0 到 9（不含）的循环变量
r = UOp.range(10, 0, AxisType.LOOP)  # range_id=0, 范围 0..9
print(r)  # UOp(Ops.RANGE, dtypes.index, src=(10,), arg=(0, AxisType.LOOP))
```

当最终变成 GPU 代码时，它变成一个实际的循环：

```c
for (int r0 = 0; r0 < 10; r0++) {
  // ... 循环体使用 r0 ...
}
```

或者在 GPU 上，循环维度被映射到线程索引。

## 算法

Rangeify 实现在 `tinygrad/schedule/indexing.py:run_rangeify()` 中。工作原理如下：

### 步骤 1：找到物化点

首先，tinygrad 决定哪些 tensor 需要*物化*到内存中（即存储到缓冲区）。`pm_generate_realize_map` PatternMatcher 标记：

- `SINK` 的源（最终输出）
- 任何 `COPY`、`ASSIGN`、`CONTIGUOUS` 节点
- `COPY` 的源（正在跨设备发送的数据）

其他所有内容都被融合——不会有自己的缓冲区。

### 步骤 2：分配 Range（自底向上）

核心循环按**逆拓扑序**（先输出，后输入）遍历所有节点。对于每个节点，它回答："这个节点的维度对应哪些循环变量？"

有三种情况：

#### 情况 1：已物化的节点 -> 创建新 Range

如果节点在 realize map 中（将被存储到内存），我们创建新的循环变量：

```
ASSIGN shape=(4,4) -> [r0][r1]   # 新 range r0=0..3, r1=0..3
```

这是循环诞生的地方。每个维度获得自己的循环变量。

#### 情况 2：单个消费者 -> 继承 Range

如果节点只有一个消费者，它直接使用消费者分配的 range：

```
MUL shape=(4,4,4) -> [r0][r1][r2]   # 从上面的 REDUCE_AXIS 继承
```

不需要新的 range——计算被融合到消费者的循环中。

#### 情况 3：多个消费者 -> 合并或创建

如果节点有多个消费者，tinygrad 尝试合并它们的 range。如果所有消费者对某个轴使用相同的 range，则复用。否则，创建新 range 并部分物化该轴：

```
# 有 2 个消费者的节点，它们在轴 0 上一致但在轴 1 上不一致：
Consumer A: [r0][r1]
Consumer B: [r0][r3]
结果:       [r0][r_new]  # 轴 1 获得新 range，将被缓冲
```

### 步骤 3：通过移动操作变换 Range

当算法遇到移动操作时，它使用移动操作的语义将*输出 range* 变换为*输入 range*。这就是魔法发生的地方。

让我们追踪我们的例子：

```python
a = Tensor.ones(2, 3)      # 形状 (2, 3)
b = a.permute(1, 0)        # 形状 (3, 2) — UOp 图有 PERMUTE
c = b.reshape(6)           # 形状 (6,)   — UOp 图有 RESHAPE
d = c.sum()                # 形状 (1,)   — UOp 图有 REDUCE_AXIS
```

UOp 图（从底到顶）：
```
CONST shape=()
  -> EXPAND shape=(2,3)
    -> PERMUTE(1,0) shape=(3,2)
      -> RESHAPE(6) shape=(6,)
        -> REDUCE_AXIS(+, axis=0) shape=(1,)
          -> ASSIGN shape=(1,)      <-- 已物化
```

Rangeify 自顶向下遍历（逆拓扑序）：

1. **ASSIGN** `shape=(1,)`：已物化。创建 `r_out = [0]`（大小 1，不需要循环）。

2. **REDUCE_AXIS** `shape=(1,)`：从 ASSIGN 继承 `out_rngs = [0]`。因为它归约轴 0，创建新的 REDUCE range：`in_rngs = [r0]`，其中 `r0` 从 0 到 5。这个 `r0` 将成为归约循环。

3. **RESHAPE(6)** 从 `(3,2)`：输出是 `[r0]`。Reshape 将 `r0` 分解为原始维度：`in_rngs = [r0//2, r0%2]`。

4. **PERMUTE(1,0)** 从 `(2,3)` 到 `(3,2)`：交换 range：`in_rngs = [r0%2, r0//2]`。

5. **EXPAND** 从 `(1,1)` 到 `(2,3)`：扩展的维度得到常量 0：`in_rngs = [0, 0]`。这是正确的——CONST 在所有位置都是相同值，所以不需要索引。

最终 kernel 是："对 r0 从 0 到 5 循环，累加 const[0,0]"——就是 `1.0 * 6 = 6.0`。

## 移动操作语义详解

`indexing.py` 中的 `apply_movement_op()` 函数定义了每个移动操作如何变换 range。让我们逐一了解：

### SHRINK（切片）

```python
# a[2:5] -> SHRINK with arg=((2, 5),)
# 如果输出 range 是 r0（范围 0..2），输入 range 是 r0+2
case Ops.SHRINK: rngs = tuple(a+ss for a,(ss,_) in zip(rngs, arg))
```

示例：
```python
from tinygrad import Tensor
a = Tensor([10, 20, 30, 40, 50])
b = a[1:4]  # SHRINK arg=((1,4),)
# 输出 range r0: 0,1,2
# 输入 range:    r0+1 = 1,2,3  -> 访问 a[1], a[2], a[3]
```

### PERMUTE（重排维度）

```python
# a.permute(1,0) -> PERMUTE with arg=(1,0)
# 重排 range
case Ops.PERMUTE: rngs = tuple(rngs[p] for p in argsort(arg))
```

示例：
```python
a = Tensor.ones(2, 3)
b = a.permute(1, 0)  # PERMUTE arg=(1,0)
# 输出 range: [r0][r1]  (形状 3,2)
# 输入 range: [r1][r0]  (形状 2,3) — 交换回来
```

### EXPAND（广播）

```python
# a.expand(4,3) 其中 a 的形状为 (1,3) -> EXPAND with arg=(4,3)
# 扩展的维度得到常量 0（值对所有索引相同）
case Ops.EXPAND: rngs = tuple(a if in_sh==out_sh else a.const_like(0) ...)
```

示例：
```python
a = Tensor.ones(1, 3)
b = a.expand(4, 3)  # EXPAND arg=(4,3)
# 输出 range: [r0][r1]  (形状 4,3)
# 输入 range: [0][r1]   (形状 1,3) — 轴 0 是常量
```

### FLIP（反转）

```python
# a.flip(0) -> FLIP
# 反转索引: r -> (size-1) - r
case Ops.FLIP: rngs = tuple(((s-1)-a) if f else a ...)
```

### PAD（零填充）

```python
# F.pad(a, (1,1)) -> PAD with arg=((1,1),)
# 移动 range 并添加有效性检查
case Ops.PAD: rngs = tuple(... r.where(r-s, invalid()) ...)
```

填充比较特殊，因为某些索引是"无效的"——它们指向填充区域。Rangeify 用有效性掩码表示这一点，后续会转换为 `WHERE`（条件操作）。

### RESHAPE（改变形状）

```python
# a.reshape(2,3) 其中 a 的形状为 (6,)
# 使用除法/取模分解
case Ops.RESHAPE: rngs = _apply_reshape(in_shape, arg, sink.substitute(...)).src
```

Reshape 是最复杂的。它将输出 range 展平为单个线性索引，然后使用整数除法和取模将该索引分解为输入维度：

```
输出形状 (6,)，range 为 r0
-> 线性索引: r0
-> 输入形状 (2,3): axis0 = r0 // 3, axis1 = r0 % 3
```

## 实际示例：矩阵乘法

让我们看看 rangeify 在 ML 中最重要的操作上的表现：

```python
# 运行命令: DEBUG_RANGEIFY=1 NOOPT=1 python this_file.py
from tinygrad import Tensor

a = Tensor.ones(4, 4)
b = Tensor.ones(4, 4)
c = (a @ b).realize()
```

`@` 运算符展开为：reshape、expand、multiply、reduce。以下是 `DEBUG_RANGEIFY=1` 的输出：

```
***  1 Ops.ASSIGN           (4, 4, 1)   [r0][r1][0]
     1 Ops.REDUCE_AXIS      (4, 4, 1)   [r0][r1][r2 -> 0]
     1 Ops.MUL              (4, 4, 4)   [r0][r1][r2]
     1 Ops.EXPAND           (4, 4, 4)   [0 -> r0][r1][r2]
     1 Ops.PERMUTE          (1, 4, 4)   [0][r2 -> r1][r1 -> r2]
     1 Ops.RESHAPE          (1, 4, 4)   [r2][r1] -> [0][r2][r1]
     1 Ops.EXPAND           (4, 4, 4)   [r0][0 -> r1][r2]
     1 Ops.RESHAPE          (4, 1, 4)   [r0][r2] -> [r0][0][r2]
```

从下往上读。`a`（形状 4x4，存储为 16 个平坦元素）经过：
1. **RESHAPE** 到 (4,1,4)：range 分为 `[r0][0][r2]`
2. **EXPAND** 到 (4,4,4)：轴 1 广播，`[r0][r1][r2]`

`b` 经过：
1. **RESHAPE** 到 (1,4,4)：range 变为 `[0][r2][r1]`
2. **PERMUTE**：交换最后两个轴以对齐乘法
3. **EXPAND** 到 (4,4,4)：轴 0 广播

然后：
- **MUL**：range 为 `[r0][r1][r2]` 的逐元素乘法
- **REDUCE_AXIS** 在轴 2 上：`r2` 成为归约循环
- **ASSIGN**：输出形状 (4,4)，range `[r0][r1]`

生成的 kernel：
```c
// 伪代码（实际输出是 Metal/CUDA 等）
for r0 in 0..3:     // 输出行
  for r1 in 0..3:   // 输出列
    acc = 0
    for r2 in 0..3: // 归约维度
      acc += a[r0*4 + r2] * b[r2*4 + r1]
    out[r0*4 + r1] = acc
```

这是标准矩阵乘法——tinygrad 从形状操作自动推导出来的。

## 生成的 Kernel

让我们实际看看生成的代码：

```python
# 运行命令: DEBUG=5 NOOPT=1 python this_file.py
from tinygrad import Tensor

a = Tensor.ones(4, 4)
b = Tensor.ones(4, 4)
c = (a @ b).realize()
```

使用 `DEBUG=5`，你会看到完整的 UOp AST 和渲染的 kernel。rangeify 之后的 AST 如下：

```
c0 = UOp(Ops.PARAM, dtypes.float.ptr(16), (), 0)  # 输出缓冲区
c2 = UOp.range(4, 0, AxisType.LOOP)                 # r0: 输出行
c4 = UOp.range(4, 1, AxisType.LOOP)                 # r1: 输出列
c6 = UOp.range(4, 2, AxisType.REDUCE)               # r2: 归约维度
c8 = UOp(Ops.PARAM, dtypes.float.ptr(16), (), 1)   # a 缓冲区
c10 = UOp(Ops.PARAM, dtypes.float.ptr(16), (), 2)  # b 缓冲区
# 索引: a[r0*4 + r2] * b[r2*4 + r1]，在 r2 上归约
c12 = c8.index(c2*4+c6) + c10.index(c6*4+c4)
```

注意：
- `r0` 和 `r1` 是 `AxisType.LOOP` — 它们成为输出索引
- `r2` 是 `AxisType.REDUCE` — 它成为累加循环
- 索引表达式 `r0*4+r2` 和 `r2*4+r1` 直接来自 rangeify 分解移动操作的方式

## 关键洞察

### 1. 视图不需要数据移动

Reshape、permute、expand — 这些都不分配内存。它们只改变 range 映射到缓冲区索引的方式。唯一接触内存的操作是 LOAD（从缓冲区读取）和 STORE（写入缓冲区）。

### 2. 融合是免费的

因为 rangeify 一次处理整个 UOp 图，操作自然融合。如果你写 `(a @ b).relu()`，relu 成为同一个 kernel 的一部分——它只是在写入输出缓冲区之前应用到累加器上。

### 3. Realize Map 控制 Kernel 边界

Realize map 决定一个 kernel 在哪里结束，另一个在哪里开始。如果操作不在 realize map 中，它就被融合到消费者的 kernel 中。`pm_remove_bufferize` pass 甚至可以移除不必要的物化，创建更大的融合 kernel。

### 4. 符号数学承担了重任

`_apply_reshape` 函数严重依赖符号简化。当你将 `(6,)` reshape 为 `(2,3)` 时，它生成 `r0//3` 和 `r0%3` 这样的表达式。`symbolic` PatternMatcher 简化它们——例如，`(r0%3)//3` 简化为 `0`。这对融合 reshape 链至关重要。

## 动手实践：自己追踪 Rangeify

这是一个脚本，让你探索任意 tensor 表达式的 rangeify：

```python
"""
运行命令: DEBUG_RANGEIFY=1 python extra/book/trace_rangeify.py
尝试修改表达式以查看不同的 rangeify 输出。
"""
from tinygrad import Tensor
import os
os.environ['DEBUG_RANGEIFY'] = '1'
os.environ['NOOPT'] = '1'

# === 尝试这些表达式 ===

# 1. 简单逐元素
# c = (Tensor.ones(4) + Tensor.ones(4)).realize()

# 2. Reshape + 求和
# c = Tensor.ones(2,3).reshape(6).sum().realize()

# 3. 转置 + 矩阵乘法
# c = (Tensor.ones(4,4).T @ Tensor.ones(4,4)).realize()

# 4. 广播
# c = (Tensor.ones(4,1) + Tensor.ones(1,4)).realize()

# 5. 类卷积模式
# c = Tensor.ones(1,1,4).expand(1,1,4).reshape(4).sum().realize()

# 6. 填充 + 求和
# c = Tensor.ones(3).pad(((1,1),)).sum().realize()

# 默认: 矩阵乘法
c = (Tensor.ones(4,4) @ Tensor.ones(4,4)).realize()
```

## Rangeify 之后

rangeify 将移动操作转换为 range 后，kernel 还要经过几个 pass：

1. **符号简化**：简化 range 表达式（例如 `(r0*4+r1)//4` -> `r0`）
2. **缓冲区移除**：尝试移除不必要的中间缓冲区（`pm_remove_bufferize`）
3. **缓冲区分配**：将剩余的 `BUFFERIZE` 转换为 `STORE` + `BUFFER`
4. **Kernel 拆分**：每个 `STORE` 成为单独的 kernel（`split_kernels`）
5. **优化**：BEAM 搜索应用分块、upcast、unroll、tensor core（第 8 章）
6. **代码生成**：降低到 GPU 源代码（第 7 章）

完整流水线在 `tinygrad/schedule/rangeify.py:get_kernel_graph()` 中。

## 源代码索引

| 文件 | 阅读内容 |
|------|---------|
| `tinygrad/schedule/indexing.py` | `run_rangeify()` — 主算法 |
| `tinygrad/schedule/indexing.py:142` | `apply_movement_op()` — 每个移动操作如何变换 range |
| `tinygrad/schedule/indexing.py:126` | `_apply_reshape()` — reshape 分解 |
| `tinygrad/schedule/rangeify.py:483` | `get_kernel_graph()` — 完整流水线 |
| `tinygrad/schedule/rangeify.py:458` | `split_store()` — 图如何拆分为 kernel |
| `tinygrad/uop/ops.py:16` | `AxisType` — range 类型枚举（LOOP、REDUCE、GLOBAL 等）|

## 练习

1. **追踪转置**：对 `Tensor.ones(3,4).permute(1,0).contiguous().realize()` 运行 `DEBUG_RANGEIFY=1`。PERMUTE 产生了什么 range？为什么需要 `contiguous()`？

2. **理解融合**：对 `(Tensor.ones(4,4) @ Tensor.ones(4,4)).relu().realize()` 运行 `DEBUG_RANGEIFY=1`。relu 是创建了新 kernel 还是与 matmul 融合了？

3. **计算 kernel 数量**：对 `a = Tensor.ones(4,4); b = (a @ a).realize(); c = (b + b).realize()` 运行 `DEBUG=2`。生成了多少个 kernel？为什么？

4. **阅读代码**：打开 `tinygrad/schedule/indexing.py`，找到 `run_rangeify` 函数。识别三种情况（已物化、单消费者、多消费者）分别在哪里处理。
