# 第四章：调度 — 计算何时发生？

在 tinygrad 中，计算是惰性的。本章解释计算*何时*真正发生、框架如何决定将哪些操作融合为单个 kernel，以及调度（schedule）长什么样。

## 惰性图

当你编写 tensor 操作时，tinygrad 构建一个 UOp 图——不进行任何计算：

```python
from tinygrad import Tensor

a = Tensor.ones(4, 4)
b = Tensor.ones(4, 4)
c = a + b         # 只是在图中添加一个 ADD 节点
d = c * 2         # 只是添加一个 MUL 节点
e = d.sum()       # 只是添加一个 REDUCE_AXIS 节点
# 到目前为止什么都没有计算
```

计算由以下操作触发：

```python
e.realize()   # 强制计算，结果保留在设备上
e.numpy()     # 强制计算，拷贝到 CPU
e.item()      # 强制计算，返回标量
```

## 什么是 Schedule？

当你调用 `.realize()` 时，tinygrad 创建一个 **schedule** —— 一个按顺序执行的操作列表。每个条目要么是：
- 一个 **kernel**：生成的 GPU 代码
- 一个 **copy**：设备间的数据传输
- 一个 **allocation**：缓冲区创建

```bash
# 查看 schedule
DEBUG=2 python -c "
from tinygrad import Tensor
a = Tensor.ones(4, 4)
b = Tensor.ones(4, 4)
c = ((a + b) * 2).sum().realize()
"
```

你会看到类似这样的输出：

```
*** METAL  1  copy  64, METAL <- PYTHON    # 将 a 拷贝到 GPU
*** METAL  2  copy  64, METAL <- PYTHON    # 将 b 拷贝到 GPU
*** METAL  3  r_16                         # kernel：加法、乘法、求和
```

三个操作：两次拷贝将数据送到 GPU，然后一个 kernel 完成 `(a+b)*2` 并求和——全部融合在单个 kernel 中。

## Kernel 融合

调度中最关键的优化是**融合**——将多个操作合并为一个 kernel，以避免中间内存分配：

```python
# 不融合：3 个 kernel，2 个中间缓冲区
# Kernel 1: c = a + b    （写入 temp1）
# Kernel 2: d = temp1 * 2  （读取 temp1，写入 temp2）
# Kernel 3: e = sum(temp2) （读取 temp2，写入 result）

# 融合后：1 个 kernel，0 个中间缓冲区
# Kernel 1: e = sum((a + b) * 2)  （读取 a,b，写入 result）
```

tinygrad 融合可以共享相同循环结构的操作。一般规则：**如果输出形状与输入形状匹配或是输入形状的归约，就可以融合。**

### 可以融合的情况：
- 逐元素链：`(a + b) * c - d`
- 逐元素后接归约：`(a + b).sum()`
- reshape 和 permute（它们是免费的——只改变索引方式）

### 强制产生新 kernel 的情况：
- `CONTIGUOUS` —— 显式强制物化
- `COPY` —— 跨设备数据传输
- 多个消费者 —— 如果一个值被两个独立的归约使用
- `ASSIGN` 边界 —— 原地操作

```python
from tinygrad import Tensor

# 融合为 1 个 kernel：
x = Tensor.ones(4, 4)
y = ((x + 1) * 2).sum()
y.realize()  # 1 个 kernel

# 两个 kernel（matmul 强制物化）：
a = Tensor.ones(4, 4)
b = a @ a          # kernel 1: 矩阵乘法
c = (b + 1).sum()  # kernel 2: 加法 + 求和
c.realize()
```

## 调度流水线

以下是 `.realize()` 调用如何变为执行的 kernel：

```
Tensor.realize()
  └─ schedule_with_vars()
       └─ complete_create_schedule_with_vars(big_sink)
            │
            ├─ transform_to_call(big_sink)
            │    将 tensor UOp 图转换为 CALL 节点
            │    并显式分配缓冲区
            │
            ├─ get_kernel_graph(function)  [rangeify.py]
            │    将移动操作转换为 RANGE 循环
            │    将图拆分为离散的 kernel
            │    （这是第 5 章：Rangeify）
            │
            └─ create_schedule(kernel_graph)
                 对 kernel 进行拓扑排序
                 返回 list[ExecItem]
```

每个 `ExecItem` 包含：
- `.ast` —— kernel 的 UOp AST（一个 `SINK` 节点）
- `.bufs` —— 它读写的缓冲区

## Realize Map

tinygrad 通过构建 **realize map** 来确定哪些操作需要自己的缓冲区（因此需要自己的 kernel）。一个操作被 realize 的条件是：

1. 它是**最终输出**（一个 `SINK` 源）
2. 它是 **COPY** 或 **ASSIGN**（必须物化）
3. 它是 **CONTIGUOUS**（用户显式请求物化）
4. 它有**多个消费者**且无法共享 range

其他所有操作都被融合到其消费者的 kernel 中。

## 查看 Schedule

使用 `DEBUG=2` 查看生成了哪些 kernel：

```bash
# 逐元素融合
DEBUG=2 python -c "
from tinygrad import Tensor
x = Tensor.ones(1000)
y = ((x + 1) * 2 - 3).realize()
"
# 输出：1 个 kernel（E_1000）

# 归约融合
DEBUG=2 python -c "
from tinygrad import Tensor
x = Tensor.ones(1000)
y = ((x + 1) * 2).sum().realize()
"
# 输出：1 个 kernel（r_1000）
# 'r' 前缀表示包含归约操作
```

Kernel 命名约定：
- `E_N` —— 逐元素 kernel，N 个总元素
- `r_N_M` —— 归约 kernel，N 个输出元素，M 个归约元素
- 名称中的数字代表各轴大小

## 内存规划

调度完成后，tinygrad 运行**内存规划器**，复用生命周期不重叠的缓冲区：

```python
# 不做内存规划：
# Buffer A：被 kernel 1-2 使用
# Buffer B：被 kernel 3-4 使用
# Buffer C：被 kernel 5-6 使用
# 总计：分配 3 个缓冲区

# 做了内存规划：
# Buffer A：被 kernel 1-2 使用，然后复用给 kernel 5-6
# Buffer B：被 kernel 3-4 使用
# 总计：分配 2 个缓冲区
```

这由 `tinygrad/engine/memory.py` 处理。

## 练习

1. **计算 kernel 数量**：对各种表达式运行 `DEBUG=2`，预测会生成多少个 kernel：
   - `(Tensor.ones(100) + Tensor.ones(100)).realize()`
   - `(Tensor.ones(100) + Tensor.ones(100)).sum().realize()`
   - `x = Tensor.ones(4,4); (x @ x + x).realize()`

2. **强制拆分**：使用 `.contiguous()` 强制物化中间结果。比较有和没有它时的 `DEBUG=2` 输出。

3. **阅读 schedule**：设置 `DEBUG=3` 查看 kernel AST。识别哪些 UOp 节点代表加载、存储和计算。

## 源代码索引

| 文件 | 阅读内容 |
|------|---------|
| `tinygrad/engine/schedule.py` | `complete_create_schedule_with_vars()` —— 主调度器 |
| `tinygrad/engine/realize.py` | `run_schedule()` —— kernel 执行 |
| `tinygrad/engine/allocations.py` | `transform_to_call()` —— 缓冲区分配 |
| `tinygrad/engine/memory.py` | 内存规划器（缓冲区复用） |
| `tinygrad/schedule/rangeify.py` | `get_kernel_graph()` —— 移动操作到 kernel |
