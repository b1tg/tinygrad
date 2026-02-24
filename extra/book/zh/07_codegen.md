# 第7章：代码生成 -- 从 UOps 到源代码

本章追踪从内核的 UOp AST 到生成的 GPU 源代码的完整路径。读完本章后，你将理解流水线中的每个重写阶段，并能够阅读生成的代码。

## 全局概览

在调度产生内核 AST（一个 `SINK` 节点）之后，codegen 通过约15个模式匹配阶段将其转换为可渲染的源代码：

```
内核 AST（包含 RANGE/LOAD/STORE 的 SINK）
  |  graph_rewrite 阶段（降级）
  v
线性 UOp 列表（扁平的指令序列）
  |  渲染器（对 UOps 进行模式匹配 -> 字符串）
  v
源代码字符串（C/CUDA/Metal/PTX）
  |  编译器
  v
二进制文件（GPU 可执行文件）
```

## 亲自查看

```bash
DEBUG=5 NOOPT=1 python -c "
from tinygrad import Tensor
(Tensor.ones(4) + Tensor.ones(4)).realize()
"
```

输出会显示 AST 和渲染后的代码。让我们通过一个简单的例子来追踪整个过程。

## 内核 AST

经过 rangeify 之后，两个大小为4的向量的简单逐元素加法看起来像这样：

```python
c0 = UOp(Ops.PARAM, dtypes.float.ptr(4), (), 0)   # 输出缓冲区
c2 = UOp.range(4, 0, AxisType.LOOP)                 # 循环变量 i: 0..3
c4 = UOp(Ops.PARAM, dtypes.float.ptr(4), (), 1)    # 输入缓冲区 a
c6 = UOp(Ops.PARAM, dtypes.float.ptr(4), (), 2)    # 输入缓冲区 b
c8 = c4.index(c2) + c6.index(c2)                    # a[i] + b[i]
c10 = c0.index(c2, ptr=True).store(c8).end(c2)      # out[i] = result
ast = c10.sink(arg=KernelInfo(...))
```

这是 `tinygrad/codegen/__init__.py` 中 `full_rewrite_to_sink()` 的输入。

## 重写阶段

### 阶段 1-5：早期降级

这些阶段处理移动操作、简化范围表达式并优化索引运算：

```
pm_mops               # 将移动操作转换为索引表达式
pm_syntactic_sugar     # 清理语法模式
pm_load_collapse       # 合并冗余加载
symbolic               # 简化数学运算（x*1 -> x, x+0 -> x）
pm_simplify_ranges     # 简化范围边界
```

### 阶段 6：优化（apply_opts）

这是 BEAM 搜索或启发式方法应用优化动作的地方：

```python
from tinygrad.codegen.opt import Opt, OptOps

# 可用的优化：
Opt(OptOps.UPCAST, axis=0, arg=4)   # 展开循环4次（向量化）
Opt(OptOps.LOCAL, axis=0, arg=16)   # 映射到16个本地线程
Opt(OptOps.GROUP, axis=0, arg=8)    # 分组归约
Opt(OptOps.UNROLL, axis=0, arg=4)   # 完全展开循环
Opt(OptOps.TC, axis=0, arg=(...))   # 使用 Tensor Cores
```

使用 `NOOPT=1` 时，此阶段会被跳过，内核保持最简形式。详见第8章。

### 阶段 7：展开器

展开标记为 UPCAST 或 UNROLL 的循环。带有 UPCAST 的范围变成显式的逐元素操作：

```
之前: for i in range(4): out[i] = a[i] + b[i]
之后: out[0] = a[0] + b[0]
      out[1] = a[1] + b[1]
      out[2] = a[2] + b[2]
      out[3] = a[3] + b[3]
```

这使得向量化成为可能——渲染器可以发出 `float4` 加载/存储而不是标量操作。

### 阶段 8-9：缓冲区和归约

添加本地内存缓冲区（用于工作组共享数据）并将归约操作（REDUCE）降级为累加器 + 循环模式。

### 阶段 10：GPU 维度（gpudims）

将 RANGE 循环映射到 GPU 硬件维度：

```python
# AxisType.GLOBAL  -> blockIdx.x/y/z  (threadgroup_position)
# AxisType.LOCAL   -> threadIdx.x/y/z (thread_position)
# AxisType.WARP    -> warp 级操作
# AxisType.LOOP    -> 实际的 for 循环
# AxisType.REDUCE  -> 归约循环
# AxisType.UPCAST  -> 已展开（无循环）
```

带有 `AxisType.GLOBAL` 的 RANGE 在 Metal 中变成 `gid.x`，在 CUDA 中变成 `blockIdx.x`。带有 `AxisType.LOCAL` 的 RANGE 变成 `lid.x` 或 `threadIdx.x`。

### 阶段 11-12：加载和反向量化

添加显式的 LOAD 指令（替换 INDEX 引用）并处理向量化访问模式。

### 阶段 13：分解

将复杂操作降级为基本操作：

```python
# DIV(x, y) -> MUL(x, RECIPROCAL(y))
# EXP(x)    -> EXP2(x * LOG2E)
# LOG(x)    -> LOG2(x) * LN2
# SIGMOID(x) -> RECIPROCAL(1 + EXP2(-x * LOG2E))
```

### 阶段 14-15：最终重写和控制流

插入实际的控制流——`RANGE`/`END` 对变成 `for` 循环，`IF`/`ENDIF` 变成条件语句：

```
之前: RANGE(r0, 0, 4) ... END(r0)
之后: for (int r0 = 0; r0 < 4; r0++) { ... }
```

## 线性化

经过所有阶段后，DAG 被展平为一个**线性 UOp 列表**——每行一条指令，按执行顺序排列：

```python
# codegen/late/linearizer.py 中的 linearize() 函数
# 对 UOp DAG 进行拓扑排序，生成扁平列表：
[
    UOp(Ops.PARAM, ptr, (), 0),        # data0
    UOp(Ops.PARAM, ptr, (), 1),        # data1
    UOp(Ops.PARAM, ptr, (), 2),        # data2
    UOp(Ops.SPECIAL, int, (), gidx0),  # 线程索引
    UOp(Ops.INDEX, ptr, ...),          # &data1[gidx0]
    UOp(Ops.LOAD, float, ...),         # val0 = data1[gidx0]
    UOp(Ops.INDEX, ptr, ...),          # &data2[gidx0]
    UOp(Ops.LOAD, float, ...),         # val1 = data2[gidx0]
    UOp(Ops.ADD, float, ...),          # val0 + val1
    UOp(Ops.INDEX, ptr, ...),          # &data0[gidx0]
    UOp(Ops.STORE, void, ...),         # data0[gidx0] = result
]
```

## 渲染

渲染器遍历线性 UOp 列表并发出源代码。每个 UOp 都有一个在 PatternMatcher 中定义的渲染规则：

```python
# 简化自 tinygrad/renderer/cstyle.py

# PARAM -> 函数参数
Ops.PARAM:  "data{arg}"

# SPECIAL -> 硬件线程索引
Ops.SPECIAL: "gid.x" / "blockIdx.x" / 等

# LOAD -> 指针解引用
Ops.LOAD:  "*(data1+gidx0)"

# ADD -> 中缀运算符
Ops.ADD:   "(val0+val1)"

# STORE -> 指针写入
Ops.STORE: "*(data0+gidx0) = (val0+val1);"

# RANGE/END -> for 循环
Ops.RANGE: "for (int ridx0 = 0; ridx0 < 4; ridx0++) {"
Ops.END:   "}"
```

实际的渲染器处理更多情况（向量化类型、图像类型、特殊硬件函数），但核心是对 UOp 类型的模式匹配。

## 不同的渲染器

tinygrad 有针对不同目标的渲染器：

| 渲染器 | 目标 | 文件 |
|----------|--------|------|
| `CUDARenderer` | NVIDIA CUDA C++ | `renderer/cstyle.py` |
| `MetalRenderer` | Apple Metal C++ | `renderer/cstyle.py` |
| `OpenCLRenderer` | OpenCL C | `renderer/cstyle.py` |
| `WGSLRenderer` | WebGPU WGSL | `renderer/wgsl.py` |
| `PTXRenderer` | NVIDIA PTX 汇编 | `renderer/ptx.py` |
| `LLVMRenderer` | LLVM IR（CPU） | `renderer/llvmir.py` |
| `AMDRenderer` | AMD ISA 汇编 | `renderer/amd/` |

所有 C 风格渲染器共享相同的基类和大部分规则——差异主要在函数签名、线程索引和硬件特定的内建函数上。

## 颜色

当你在 `DEBUG=2` 输出中看到内核名称时，颜色编码了信息：

```
E_4_4     # E = 逐元素操作
r_4_4     # r = 包含归约

# 数字的颜色表示轴类型：
# 蓝色   = GLOBAL（映射到 GPU 块）
# 青色   = LOCAL（映射到 GPU 线程）
# 黄色   = UPCAST（展开/向量化）
# 红色   = REDUCE（归约循环）
# 白色   = LOOP（普通循环）
```

## 端到端示例

让我们追踪一个求和归约通过整个流水线的过程：

```bash
DEBUG=5 NOOPT=1 python -c "
from tinygrad import Tensor
Tensor.ones(4).sum().realize()
"
```

**内核 AST**（rangeify 之后）：
```
SINK
  └─ STORE(param0, idx, value).END(r0)
       └─ REDUCE(+, r0)
            └─ LOAD(param1, r0)
```

**降级之后**（线性化）：
```
PARAM(0)          -> data0
PARAM(1)          -> data1
CONST(0.0)        -> 累加器初始化
RANGE(0, 4)       -> for ridx0 = 0..3
  LOAD(data1, ridx0)  -> val0
  ADD(acc, val0)       -> 新的 acc
END               -> 循环结束
STORE(data0, 0, acc)   -> 写入结果
```

**渲染后**（Metal）：
```c
kernel void r_4(device float* data0, device float* data1, ...) {
  float acc0 = 0.0f;
  for (int ridx0 = 0; ridx0 < 4; ridx0++) {
    float val0 = *(data1+ridx0);
    acc0 = (acc0+val0);
  }
  *(data0+0) = acc0;
}
```

## 练习

1. **阅读输出**：对 `Tensor.ones(4,4).sum(axis=0).realize()` 运行 `DEBUG=5 NOOPT=1`。识别 AST 中的每个 UOp 并将其映射到生成的代码。

2. **比较优化后的版本**：对相同的表达式运行 `DEBUG=4`（不带 NOOPT）。优化后的内核有什么不同？

3. **不同后端**：如果你有 CUDA 访问权限，比较同一内核的 Metal 和 CUDA 输出。哪些相同？哪些不同？

4. **追踪一个阶段**：添加 `VIZ=1` 并运行一个简单的内核。可视化工具会显示每个重写阶段。

## 源代码导航

| 文件 | 阅读内容 |
|------|-------------|
| `tinygrad/codegen/__init__.py` | `full_rewrite_to_sink()` -- 完整的降级流水线 |
| `tinygrad/codegen/__init__.py` | `get_program()` -- 顶层入口点 |
| `tinygrad/codegen/late/expander.py` | 循环展开和向量化 |
| `tinygrad/codegen/late/linearizer.py` | DAG 到线性列表的转换 |
| `tinygrad/codegen/late/devectorizer.py` | 加载/存储索引 |
| `tinygrad/codegen/gpudims.py` | GPU 维度分配 |
| `tinygrad/renderer/__init__.py` | `Renderer` 基类，`ProgramSpec` |
| `tinygrad/renderer/cstyle.py` | C 风格代码发射（约1000行） |
