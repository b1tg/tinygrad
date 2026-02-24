# 第17章：VIZ — 图可视化工具

调试 tinygrad 时，你经常需要查看 UOp 图——有哪些节点、它们如何连接、以及每个重写 pass 如何变换它们。VIZ 是 tinygrad 内置的交互式图可视化工具。

## 快速开始

```bash
VIZ=1 python -c "
from tinygrad import Tensor
(Tensor.ones(4, 4) + Tensor.ones(4, 4)).realize()
"
```

这会打开一个浏览器，显示编译各阶段 UOp 图的交互式可视化。你可以：
- 在重写 pass 之间导航
- 点击节点查看详细信息
- 查看每次模式匹配的前后对比
- 缩放和平移图

## VIZ 显示什么

VIZ 在每个 `graph_rewrite` pass 处捕获 UOp 图。对于简单的 `a + b`：

```
Pass 1: "earliest rewrites"     - high-level tensor graph
Pass 2: "rangeify"              - movement ops -> ranges
Pass 3: "symbolic+debuf"        - symbolic simplification
Pass 4: "bufferize to store"    - add store instructions
Pass 5: "split kernels"         - split into individual kernels
Pass 6: "early movement ops"    - lower movement ops
Pass 7: "symbolic"              - more simplification
Pass 8: "apply_opts"            - BEAM/heuristic optimization
Pass 9: "expander"              - unroll loops
Pass 10: "add gpudims"          - assign thread dimensions
Pass 11: "devectorize"          - lower vectors
Pass 12: "decompositions"       - decompose ops
Pass 13: "final rewrite"        - cleanup
Pass 14: "add control flow"     - insert loops/conditions
```

你可以逐步查看每个 pass，精确了解图是如何变换的。

## 阅读图

节点按操作类型着色：
- **绿色**：内存操作（LOAD、STORE）
- **蓝色**：数学操作（ADD、MUL 等）
- **红色**：控制流（RANGE、END、IF）
- **灰色**：常量和参数

边表示数据依赖——箭头从输入指向输出。

## 使用 VIZ 调试

VIZ 在以下情况特别有用：

1. **内核产生错误结果**：逐步查看各 pass，找到图与预期不符的地方。

2. **模式匹配未触发**：VIZ 显示哪些模式匹配了以及替换了什么。如果你的模式没有匹配，你可以看到原因。

3. **理解优化**：查看 BEAM 搜索或启发式方法对图结构做了什么。

```bash
# See optimization applied to a matmul
VIZ=1 python -c "
from tinygrad import Tensor
(Tensor.ones(64, 64) @ Tensor.ones(64, 64)).realize()
"
```

## 实现原理

VIZ 通过挂钩 `graph_rewrite` 工作。当 `VIZ=1` 时，每次 `graph_rewrite` 调用都会记录 UOp 图的前后状态。这些快照通过本地 Web 服务器提供，使用 dagre（一个 JavaScript 图布局库）进行可视化。

可视化工具的代码位于 `tinygrad/viz/`。

## 练习

1. **探索矩阵乘法**：对 4x4 矩阵乘法运行 `VIZ=1`。在 "rangeify" pass 中，找到 RANGE 节点并追踪它们如何映射到归约循环。

2. **比较 pass**：查看 "apply_opts" 前后的图。发生了什么变化？

3. **调试 bug**：如果你编写了自定义模式匹配器，使用 VIZ 验证它是否正确匹配和替换。

## 源代码导航

| 文件 | 阅读内容 |
|------|----------|
| `tinygrad/viz/` | 可视化工具实现 |
| `tinygrad/uop/ops.py` | 带 VIZ 钩子的 `graph_rewrite()` |
| `tinygrad/helpers.py` | `VIZ` 环境变量 |
