# 第34章：端到端追踪 —— 跟踪一次计算在 tinygrad 中的完整流程

本章将追踪一个单一操作在整个 tinygrad 流水线中的执行过程，从 Python API 到 GPU 执行再到结果返回。这将把第1至第5部分的所有内容串联起来。

## 操作

```python
from tinygrad import Tensor
result = Tensor.ones(4, 4).sum().item()
# result = 16.0
```

很简单：创建一个 4x4 的全1矩阵，求和，获取数值。但在底层，这会触及 tinygrad 的每一层。

## 阶段 1：Tensor 创建

`Tensor.ones(4, 4)` 调用了：

```python
@staticmethod
def ones(*shape, **kwargs):
    return Tensor.full(shape, 1.0, **kwargs)

@staticmethod
def full(shape, fill_value, **kwargs):
    return Tensor(fill_value, **kwargs)._broadcast_to(shape)
```

这会创建一个 `UOp.const(dtypes.float32, 1.0)` 并将其扩展到形状 `(4, 4)`。此时没有分配任何 buffer —— 它只是计算图中的一个常量。

此时的 UOp 图：

```
CONST(1.0, dtype=float32)
  → RESHAPE to (1, 1)
  → EXPAND to (4, 4)
```

## 阶段 2：求和

`.sum()` 创建一个归约操作：

```python
def sum(self, axis=None, keepdim=False):
    return self._reduce(Ops.ADD, axis, keepdim)
```

UOp 图继续增长：

```
CONST(1.0) → RESHAPE(1,1) → EXPAND(4,4) → REDUCE_AXIS(ADD, axes=(0,1))
```

仍然没有进行计算。图只是记录了"对一个 4x4 全1矩阵的所有元素求和"。

## 阶段 3：`.item()` 触发 realize

```python
def item(self):
    assert self.numel() == 1
    return self.data()[(0,) * len(self.shape)]
```

`.data()` 调用 `._buffer()`，后者调用 `.realize()`。现在事情真正开始了。

## 阶段 4：调度（Scheduling）

`realize()` 调用 `schedule_with_vars()`：

```python
def schedule_with_vars(self, *lst):
    big_sink = UOp.sink(*[x.uop for x in (self,) + lst])
    becomes_map, schedule, var_vals = complete_create_schedule_with_vars(big_sink)
    _apply_map_to_tensors(becomes_map, name="buffers")
    return schedule, var_vals
```

`complete_create_schedule_with_vars` 承担了主要工作：

1. **`transform_to_call`**：将图包装在一个 CALL 节点中，将计算与其 buffer 参数分离
2. **`get_kernel_graph`**：核心调度流水线（见下一节）
3. **`create_schedule`**：将 kernel 图线性化为执行顺序
4. **`memory_planner`**：优化 buffer 分配

## 阶段 5：Rangeify（核心变换）

在 `get_kernel_graph` 内部，UOp 图经过多个 pass 的变换：

### Pass 1：移动操作转换为范围

`EXPAND(4,4)` 和 `REDUCE_AXIS(ADD, (0,1))` 变成显式循环：

```
Before rangeify:
  CONST(1.0) → EXPAND(4,4) → REDUCE_AXIS(ADD, (0,1))

After rangeify:
  RANGE(0, 4)  ← loop variable i
  RANGE(0, 4)  ← loop variable j
  CONST(1.0)   ← the value at every position
  REDUCE(ADD, over ranges i and j)  ← sum over both loops
```

这就是第5章的核心洞察：形状变成循环。

### Pass 2：符号化简

模式匹配器简化图。由于我们是对常量求和：

```
sum(1.0 for i in range(4) for j in range(4)) = 1.0 * 4 * 4 = 16.0
```

tinygrad 的符号引擎可能会将其完全常量折叠。

### Pass 3：Bufferize

添加 buffer 操作 —— 从哪里读取输入以及将输出写入哪里：

```
BUFFER(output, size=1, dtype=float32)
  STORE: result of the reduction
```

### Pass 4：拆分为 kernel

图在 kernel 边界处被拆分（参见第33章关于融合的内容）。对于这个简单的例子，只有一个 kernel。

## 阶段 6：Codegen

kernel UOp 图被降级为源代码（第7章）。对于 Metal GPU：

```c
#include <metal_stdlib>
kernel void r_16(device float* data0, uint3 gid [[threadgroup_position_in_grid]]) {
  float acc = 0.0f;
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      acc += 1.0f;
    }
  }
  *(data0) = acc;
}
```

（在实际中，编译器可能会进一步优化 —— 常量折叠可能会完全消除循环。）

## 阶段 7：编译

源代码被编译为 GPU 二进制文件：

```python
class Compiler:
    def compile_cached(self, src):
        # Check disk cache first
        if (lib := diskcache_get(self.cachekey, src)) is None:
            lib = self.compile(src)  # actually compile
            diskcache_put(self.cachekey, src, lib)
        return lib
```

编译后的二进制文件会缓存到磁盘上，因此相同的 kernel 不会被重复编译。

## 阶段 8：执行

`run_schedule` 处理每个 `ExecItem`：

```python
def run_schedule(schedule, var_vals=None):
    while len(schedule):
        ei = schedule.pop(0).lower()  # lower AST → compiled program
        ei.run(var_vals)              # execute on GPU
```

`ExecItem.run()`：
1. 确保所有 buffer 已分配
2. 使用 buffer 指针调用编译后的程序
3. 更新全局统计信息（kernel 数量、FLOPs、内存带宽）

## 阶段 9：数据提取

`realize()` 完成后，tensor 的 UOp 指向一个已实现的 buffer。`._buffer()` 返回它：

```python
def _buffer(self):
    x = self.cast(self.dtype.base).contiguous()
    return cast(Buffer, x.realize().uop.buffer).ensure_allocated()
```

然后 `.data()` 将结果从 GPU 复制到 CPU：

```python
def data(self):
    return self._buffer().as_memoryview().cast('f', self.shape)
```

`.item()` 提取单个值：

```python
return self.data()[(0,)]  # → 16.0
```

## 完整流水线

```
Python: Tensor.ones(4,4).sum().item()
  │
  ├─ Tensor.ones(4,4)     → UOp: CONST(1.0) → EXPAND(4,4)
  ├─ .sum()                → UOp: → REDUCE_AXIS(ADD)
  ├─ .item()               → triggers realize()
  │
  ├─ schedule_with_vars()
  │   ├─ transform_to_call()   → wrap in CALL
  │   ├─ get_kernel_graph()
  │   │   ├─ rangeify           → shapes become loops
  │   │   ├─ symbolic           → simplify expressions
  │   │   ├─ bufferize          → add buffer read/write
  │   │   └─ split_kernels      → one kernel
  │   ├─ create_schedule()      → topological sort
  │   └─ memory_planner()       → optimize buffers
  │
  ├─ run_schedule()
  │   ├─ lower()               → codegen → compile
  │   └─ run()                 → execute on GPU
  │
  └─ copyout → 16.0
```

## 使用 DEBUG 查看全部过程

```bash
DEBUG=4 python -c "from tinygrad import Tensor; print(Tensor.ones(4,4).sum().item())"
```

`DEBUG=4` 显示生成的 kernel 源代码。`DEBUG=2` 显示 kernel 执行统计信息。`DEBUG=5` 显示每个阶段的完整 UOp 图。

## 练习

1. **运行追踪**：使用 `DEBUG=4` 执行上面的命令。阅读生成的 kernel 代码。它是否有循环，还是常量折叠消除了它们？

2. **更大的例子**：使用 `DEBUG=4` 尝试 `Tensor.rand(4,4).sum().item()`。这个例子无法进行常量折叠 —— 你应该能在 kernel 中看到实际的循环。

3. **两个 kernel**：尝试 `Tensor.rand(4,4).sum().sqrt().item()`。有几个 kernel？（sqrt 应该与 sum 融合。）

4. **流水线各阶段**：设置 `VIZ=1` 并运行示例。可视化工具会显示每个变换阶段的 UOp 图。

## 源代码索引

| 文件 | 阅读内容 |
|------|---------|
| `tinygrad/tensor.py:252-292` | `schedule_with_vars` 和 `realize` |
| `tinygrad/engine/schedule.py:81-138` | `complete_create_schedule_with_vars` |
| `tinygrad/schedule/rangeify.py:483-514` | `get_kernel_graph` —— 完整流水线 |
| `tinygrad/engine/realize.py:156-212` | `ExecItem.run` 和 `run_schedule` |
