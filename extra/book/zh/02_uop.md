# 第二章：UOp — Tinygrad 的通用中间表示

Tinygrad 中的一切都是 UOp。Tensor 是 UOp。内核是 UOp。生成的 GPU 代码是 UOp。编译后的二进制文件是 UOp。理解 UOp 就是理解 tinygrad。

## 什么是 UOp？

UOp（微操作）是有向无环图（DAG）中的一个节点。它有四个字段：

```python
from tinygrad.uop.ops import UOp, Ops
from tinygrad.dtype import dtypes

# 手动创建一个 UOp
node = UOp(
    op=Ops.CONST,      # 什么操作
    dtype=dtypes.float, # 数据类型
    src=(),             # 输入 UOps（元组）
    arg=42.0            # 额外数据
)
print(node.op)     # Ops.CONST
print(node.dtype)  # dtypes.float
print(node.arg)    # 42.0
print(node.src)    # ()
```

UOps 通过 `src` 字段形成图。一个 `ADD` 节点指向它的两个操作数：

```python
a = UOp(Ops.CONST, dtypes.float, arg=1.0)
b = UOp(Ops.CONST, dtypes.float, arg=2.0)
c = UOp(Ops.ADD, dtypes.float, src=(a, b))

print(c.op)        # Ops.ADD
print(c.src[0].arg) # 1.0
print(c.src[1].arg) # 2.0
```

## UOps 是单例的（哈希共识）

一个关键属性：**如果你创建两个具有相同 (op, dtype, src, arg) 的 UOps，你会得到同一个对象**：

```python
x = UOp(Ops.CONST, dtypes.float, arg=1.0)
y = UOp(Ops.CONST, dtypes.float, arg=1.0)
print(x is y)  # True — 同一个 Python 对象！
```

这被称为**哈希共识（hash consing）**。这意味着：
- UOp 图自动去重
- 你可以用 `is` 而不是 `==` 来比较 UOps
- 内存是共享的——在 100 个地方使用的相同子表达式只存在一份

实现使用了 `UOpMetaClass.__call__`（`tinygrad/uop/ops.py:86`）中的全局缓存。

## Ops 枚举

`Ops` 枚举定义了 tinygrad 知道的每一种操作。它们按层级组织：

```python
from tinygrad.uop import Ops

# 第 1 层：定义
# DEFINE_VAR, SPECIAL, DEFINE_LOCAL — 内核设置

# 第 2 层：基础设施
# PARAM, CALL — 函数参数
# SINK, AFTER — 排序/分组
# GEP, VECTORIZE — 向量访问

# 第 3 层：内存
# INDEX — 指针运算
# LOAD, STORE — 内存访问

# 第 4 层：数学运算
# 一元：CAST, EXP2, LOG2, SIN, SQRT, RECIPROCAL, NEG
# 二元：ADD, MUL, MAX, CMPLT, CMPNE, AND, OR, XOR
# 三元：WHERE, MULACC
# Tensor：WMMA（矩阵乘法）

# 第 5 层：控制流
# RANGE, END — 循环
# IF, ENDIF — 条件
# BARRIER — 同步
# CONST — 常量

# 第 6 层：Tensor 图（高层级，在最终内核中不存在）
# RESHAPE, PERMUTE, EXPAND, PAD, SHRINK, FLIP — 移动操作
# REDUCE_AXIS — 归约
# COPY, BUFFER, ASSIGN — 缓冲区管理
# CONTIGUOUS — 强制物化
```

同一个 `UOp` 类既表示高层级 Tensor 操作（如 `RESHAPE`），也表示低层级内核操作（如 `LOAD`）。编译流水线是一系列图重写，将高层级操作降级为低层级操作。

## Tensor 就是 UOps

当你创建一个 `Tensor` 时，你就在构建一个 UOp 图：

```python
from tinygrad import Tensor

a = Tensor([1.0, 2.0, 3.0])
print(a.uop.op)      # Ops.COPY
print(a.uop.shape)    # (3,)
print(a.uop.dtype)    # dtypes.float

b = a + 1
print(b.uop.op)       # Ops.ADD
print(b.uop.src[0].op)  # Ops.COPY  (tensor a)
```

每个 Tensor 方法都会追加到这个图中：

```python
a = Tensor.ones(2, 3)       # CONST -> EXPAND
b = a.reshape(3, 2)          # RESHAPE 包裹 a
c = b.permute(1, 0)          # PERMUTE 包裹 b
d = c.sum()                  # REDUCE_AXIS 包裹 c
# d.uop 是一个 DAG：REDUCE_AXIS -> PERMUTE -> RESHAPE -> EXPAND -> CONST
```

在你调用 `.realize()`、`.numpy()` 或 `.item()` 之前，什么都不会被计算。

## 关键 UOp 属性

UOps 有几个有用的计算属性：

```python
from tinygrad import Tensor

a = Tensor.ones(4, 4)
b = a.sum(axis=1)

# Shape：Tensor 的逻辑形状
print(b.uop.shape)   # (4, 1)

# Device：Tensor 所在的设备
print(b.uop.device)  # 'METAL' 或 'CUDA' 等

# 图结构
print(b.uop.op)               # Ops.REDUCE_AXIS
print(b.uop.src[0].op)        # Ops.EXPAND
print(b.uop.arg)              # (Ops.ADD, (1,))  — 沿轴 1 求和
```

## UOp 图可视化

对于任何 UOp 图，你可以使用 `pretty_print` 来查看结构：

```python
from tinygrad.uop.ops import UOp, Ops, pretty_print
from tinygrad.dtype import dtypes

a = UOp(Ops.CONST, dtypes.float, arg=1.0)
b = UOp(Ops.CONST, dtypes.float, arg=2.0)
c = UOp(Ops.ADD, dtypes.float, src=(a, b))
d = UOp(Ops.MUL, dtypes.float, src=(c, c))

print(pretty_print(d))
```

或者使用 `VIZ=1` 打开交互式图可视化工具：

```bash
VIZ=1 python -c "from tinygrad import Tensor; (Tensor.ones(4) + Tensor.ones(4)).realize()"
```

## UOps 的两种角色

UOps 扮演两种截然不同的角色，理解这种双重性是关键：

### 角色 1：惰性 Tensor 图

在 `.realize()` 之前，UOps 表示**逻辑计算**——要执行什么操作：

```
REDUCE_AXIS(+, axis=1)
  └─ EXPAND (4, 4)
       └─ RESHAPE (1, 1)
            └─ CONST 1.0
```

这些 UOps 包含 `RESHAPE`、`EXPAND`、`REDUCE_AXIS` 等操作。它们表示*要计算什么*。

### 角色 2：内核 AST

在调度和代码生成之后，UOps 表示**物理计算**——如何在硬件上执行：

```
SINK
  └─ STORE(ptr, idx, value)
       ├─ PARAM(0)              # 输出缓冲区指针
       ├─ RANGE(0..3)           # 循环变量
       └─ REDUCE(+)
            └─ LOAD(ptr, idx)
                 ├─ PARAM(1)    # 输入缓冲区指针
                 └─ RANGE(0..3) # 另一个循环变量
```

这些 UOps 包含 `LOAD`、`STORE`、`RANGE`、`PARAM` 等操作。它们表示*如何计算*。

编译流水线通过一系列 Pattern Matcher 重写将角色 1 的 UOps 转换为角色 2 的 UOps（第三章）。

## 手动构建 UOp 图

你可以手动构建和渲染 UOp 图。这对于理解代码生成流水线很有用：

```python
from tinygrad.uop.ops import UOp, Ops
from tinygrad.dtype import dtypes

# 构建一个简单的内核 AST：out[i] = 1.0 + 2.0
const1 = UOp(Ops.CONST, dtypes.float, arg=1.0)
const2 = UOp(Ops.CONST, dtypes.float, arg=2.0)
add = UOp(Ops.ADD, dtypes.float, src=(const1, const2))

print(add)
# UOp(Ops.ADD, dtypes.float, arg=None, src=(
#   UOp(Ops.CONST, dtypes.float, arg=1.0, src=()),
#   UOp(Ops.CONST, dtypes.float, arg=2.0, src=()),))
```

## 拓扑排序

由于 UOps 形成 DAG，我们可以对它们进行拓扑排序——先处理子节点再处理父节点。这在 tinygrad 中随处可见：

```python
from tinygrad.uop.ops import UOp, Ops
from tinygrad.dtype import dtypes

a = UOp(Ops.CONST, dtypes.float, arg=1.0)
b = UOp(Ops.CONST, dtypes.float, arg=2.0)
c = UOp(Ops.ADD, dtypes.float, src=(a, b))
d = UOp(Ops.MUL, dtypes.float, src=(c, a))

for u in d.toposort():
    print(u.op, u.arg if u.op is Ops.CONST else "")
# Ops.CONST 1.0
# Ops.CONST 2.0
# Ops.ADD
# Ops.MUL
```

## 练习

1. **构建一个图**：创建一个表示 `(a + b) * (a - b)` 的 UOp 图，其中 a=3.0，b=2.0。打印该图。

2. **验证单例**：创建两个相同的 `UOp(Ops.ADD, ...)` 图，验证它们返回同一个对象。

3. **检查一个 Tensor**：创建 `t = Tensor.ones(3,3).sum()` 并手动遍历 `t.uop`——通过跟踪 `src` 指针打印每个节点的 `op` 和 `shape`。

4. **计算操作数**：对 `(Tensor.ones(4,4) @ Tensor.ones(4,4)).uop` 使用 `.toposort()` 来计算矩阵乘法图在 realize 之前有多少个 UOp 节点。

## 源代码导航

| 文件 | 阅读内容 |
|------|-------------|
| `tinygrad/uop/ops.py:84` | `UOpMetaClass` — 单例/哈希共识实现 |
| `tinygrad/uop/ops.py:100` | `UOp` 类 — 所有方法和属性 |
| `tinygrad/uop/__init__.py` | `Ops` 枚举 — 所有操作类型 |
| `tinygrad/uop/spec.py` | UOp 图的验证规则 |
| `tinygrad/uop/symbolic.py` | UOp 表达式的符号化简 |
