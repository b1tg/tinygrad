# 第三章：Pattern Matcher — 图重写引擎

Tinygrad 的整个编译流水线建立在一个机制之上：**模式匹配和图重写**。tinygrad 不是编写一个包含解析、优化和代码生成等独立阶段的传统编译器，而是将一切表达为"在 UOp 图中找到这个模式，用那个替换它"。

## 核心思想

Pattern Matcher 接受两样东西：
1. 一个**模式** — 在图中要查找什么
2. 一个**重写函数** — 用什么来替换它

```python
from tinygrad.uop.ops import UOp, Ops, PatternMatcher, UPat
from tinygrad.dtype import dtypes

# 模式：找到 ADD(CONST(0), x) -> 替换为 x
# "任何东西加零还是它本身"
pm = PatternMatcher([
    (UPat(Ops.ADD, src=(UPat(Ops.CONST, arg=0), UPat.var("x"))),
     lambda x: x),
])

# 构建一个图：0 + 42
zero = UOp(Ops.CONST, dtypes.int, arg=0)
val = UOp(Ops.CONST, dtypes.int, arg=42)
expr = UOp(Ops.ADD, dtypes.int, src=(zero, val))

# 应用 Pattern Matcher
result = pm.rewrite(expr)
print(result.op, result.arg)  # Ops.CONST 42
```

模式 `UPat(Ops.ADD, src=(UPat(Ops.CONST, arg=0), UPat.var("x")))` 表示："匹配任何第一个源是值为 0 的 CONST 的 ADD 节点，并将第二个源命名为 `x`。"重写函数 `lambda x: x` 表示："用 `x` 替换整个匹配。"

## UPat：模式语言

`UPat` 是 tinygrad 的模式 DSL。以下是关键形式：

```python
from tinygrad.uop.ops import UPat, Ops

# 匹配特定操作
UPat(Ops.ADD)                    # 任何 ADD 节点

# 用命名变量匹配（为重写函数捕获）
UPat(Ops.ADD, name="a")         # 任何 ADD 节点，命名为 "a"

# 匹配任意节点
UPat.var("x")                    # 任何节点，命名为 "x"

# 匹配常量
UPat.cvar("c")                   # 任何 CONST 节点，命名为 "c"

# 匹配特定源
UPat(Ops.ADD, src=(              # ADD，其中：
    UPat.var("x"),               #   第一个源是任意节点
    UPat(Ops.CONST, name="c"),   #   第二个源是 CONST
))

# 匹配多种操作
UPat((Ops.ADD, Ops.MUL), name="op")  # ADD 或 MUL

# 带数据类型约束的匹配
UPat(Ops.CONST, dtype=dtypes.float)  # 仅匹配 float 类型的 CONST
```

## graph_rewrite：将模式应用到图上

`graph_rewrite` 将 PatternMatcher 应用到整个 UOp 图上，反复重写直到没有更多模式匹配为止：

```python
from tinygrad.uop.ops import UOp, Ops, PatternMatcher, UPat, graph_rewrite
from tinygrad.dtype import dtypes

# 常量折叠：将 ADD(CONST, CONST) 替换为 CONST
def fold_add(a, b):
    return UOp(Ops.CONST, dtypes.int, arg=a.arg + b.arg)

pm = PatternMatcher([
    (UPat(Ops.ADD, src=(UPat(Ops.CONST, name="a"), UPat(Ops.CONST, name="b"))), fold_add),
])

# 构建：(1 + 2) + (3 + 4)
c1 = UOp(Ops.CONST, dtypes.int, arg=1)
c2 = UOp(Ops.CONST, dtypes.int, arg=2)
c3 = UOp(Ops.CONST, dtypes.int, arg=3)
c4 = UOp(Ops.CONST, dtypes.int, arg=4)
expr = UOp(Ops.ADD, dtypes.int, src=(
    UOp(Ops.ADD, dtypes.int, src=(c1, c2)),
    UOp(Ops.ADD, dtypes.int, src=(c3, c4)),
))

# 反复应用直到不动点
result = graph_rewrite(expr, pm)
print(result.op, result.arg)  # Ops.CONST 10
```

`graph_rewrite` 函数默认自底向上遍历图，在每个节点尝试每个模式，直到不再有重写可能（达到不动点）。

## 渲染器的工作原理

Tinygrad 中的代码生成就是一个 PatternMatcher。渲染器遍历线性化的 UOp 列表，将每个节点匹配到代码生成规则：

```python
# 简化自 tinygrad/renderer/cstyle.py
# 这些模式将 UOps 转换为 C 代码字符串：

# CONST -> 字面值
(UPat(Ops.CONST, name="u"), lambda u: f"{u.arg}f")

# ADD -> 中缀 +
(UPat(Ops.ADD, name="u"), lambda u: f"({u.src[0].rendered}+{u.src[1].rendered})")

# LOAD -> 指针解引用
(UPat(Ops.LOAD, name="u"), lambda u: f"*(data+{u.src[1].rendered})")

# STORE -> 指针写入
(UPat(Ops.STORE, name="u"), lambda u: f"*(data+{u.src[1].rendered}) = {u.src[2].rendered};")
```

这是一个大幅简化，但实际的渲染器（`tinygrad/renderer/cstyle.py`）正是基于这个原理工作的——约 100 条模式规则将 UOps 转换为 C/CUDA/Metal 代码字符串。

## 编译流水线即 Pattern Matcher

整个 tinygrad 编译器是一系列 Pattern Matcher 阶段。在 `tinygrad/codegen/__init__.py:full_rewrite_to_sink()` 中：

```
阶段 1:   pm_mops + pm_syntactic_sugar    # 移动操作重写
阶段 2:   pm_load_collapse                # 合并加载
阶段 3:   pm_split_ranges                 # 拆分范围表达式
阶段 4:   symbolic                        # 简化数学表达式
阶段 5:   pm_simplify_ranges              # 简化范围边界
阶段 6:   apply_opts                      # BEAM 搜索 / 启发式优化
阶段 7:   expander                        # 展开循环，扩展向量
阶段 8:   pm_add_buffers_local            # 添加本地内存缓冲区
阶段 9:   pm_reduce                       # 降级归约操作
阶段 10:  pm_add_gpudims                  # 分配 GPU 线程维度
阶段 11:  pm_add_loads                    # 添加加载指令
阶段 12:  devectorize                     # 降级向量操作
阶段 13:  decompositions                  # 分解复杂操作
阶段 14:  pm_final_rewrite                # 最终清理
阶段 15:  pm_add_control_flow             # 添加循环和条件
```

每个阶段都是一个 `graph_rewrite(ast, some_pattern_matcher)`。UOp 图从顶部作为高层级 Tensor 操作进入，从底部作为扁平的 GPU 指令列表输出。

## 编写你自己的模式

你可以用 `+` 组合 PatternMatcher：

```python
from tinygrad.uop.ops import PatternMatcher, UPat, Ops

pm1 = PatternMatcher([
    (UPat(Ops.ADD, src=(UPat(Ops.CONST, arg=0), UPat.var("x"))), lambda x: x),  # x + 0 = x
])
pm2 = PatternMatcher([
    (UPat(Ops.MUL, src=(UPat(Ops.CONST, arg=1), UPat.var("x"))), lambda x: x),  # x * 1 = x
])

combined = pm1 + pm2  # 匹配两种模式
```

## 返回 None 表示"不匹配"

如果重写函数返回 `None`，该模式被视为不匹配：

```python
pm = PatternMatcher([
    (UPat(Ops.ADD, src=(UPat(Ops.CONST, name="c"), UPat.var("x"))),
     lambda c, x: x if c.arg == 0 else None),  # 仅当常量为 0 时匹配
])
```

这对于条件重写很有用——你通过结构进行模式匹配，然后在函数中检查值。

## 上下文参数

某些 Pattern Matcher 需要共享状态。你可以传递一个 `ctx` 对象：

```python
from tinygrad.uop.ops import PatternMatcher, UPat, Ops, graph_rewrite

def count_adds(ctx, a):
    ctx.append(a)
    return None  # 不重写，只收集

pm = PatternMatcher([
    (UPat(Ops.ADD, name="a"), count_adds),
])

counts = []
graph_rewrite(some_graph, pm, ctx=counts)
print(f"找到 {len(counts)} 个 ADD 节点")
```

## 实际示例：符号化简器

Tinygrad 的符号数学化简器就是一个 PatternMatcher。以下是 `tinygrad/uop/symbolic.py` 中的一些规则：

```python
# x + 0 = x
(UPat(Ops.ADD, src=(UPat.var("x"), UPat(Ops.CONST, arg=0))), lambda x: x)

# x * 0 = 0
(UPat(Ops.MUL, src=(UPat.var(), UPat(Ops.CONST, arg=0, name="z"))), lambda z: z)

# x * 1 = x
(UPat(Ops.MUL, src=(UPat.var("x"), UPat(Ops.CONST, arg=1))), lambda x: x)

# (x + c1) + c2 = x + (c1 + c2)  — 通过结合律进行常量折叠
(UPat(Ops.ADD, src=(UPat(Ops.ADD, src=(UPat.var("x"), UPat.cvar("c1"))), UPat.cvar("c2"))),
 lambda x, c1, c2: x + UOp.const(c1.dtype, c1.arg + c2.arg))
```

这些规则在编译过程中自动触发，简化索引表达式，例如将 `(r0 * 4 + 0)` 简化为 `(r0 * 4)`。

## 自底向上 vs 自顶向下

`graph_rewrite` 支持两种遍历顺序：

- **`bottom_up=False`**（默认）：从输入向输出处理节点。适用于降级阶段，你希望在子节点重写后再重写父节点。
- **`bottom_up=True`**：从输出向输入处理节点。适用于结构重写，父节点的上下文很重要。

## 练习

1. **编写一个化简器**：创建一个 PatternMatcher，将整数 UOps 的 `x - x` 简化为 `0`。

2. **常量折叠器**：扩展 fold_add 示例以处理 MUL。用 `(2 * 3) + (4 * 5)` 测试。

3. **计数模式**：编写一个 PatternMatcher，计算内核 AST 中出现了多少个 LOAD、STORE 和 ADD 操作。用 `DEBUG=5` 输出进行测试。

4. **阅读渲染器**：打开 `tinygrad/renderer/cstyle.py`，找到渲染 `Ops.ADD` 的 PatternMatcher。Metal 渲染器对 `a + b` 生成什么？

## 源代码导航

| 文件 | 阅读内容 |
|------|-------------|
| `tinygrad/uop/ops.py` | `PatternMatcher` 类，`graph_rewrite()` 函数 |
| `tinygrad/uop/upat.py` | `UPat` — 模式匹配 DSL |
| `tinygrad/uop/symbolic.py` | 符号数学化简器（实际模式） |
| `tinygrad/uop/decompositions.py` | 操作分解模式 |
| `tinygrad/renderer/cstyle.py` | 作为 PatternMatcher 的代码渲染器 |
| `tinygrad/codegen/__init__.py` | `full_rewrite_to_sink()` — 完整流水线 |
