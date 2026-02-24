# 第29章：Tensor 类 — tinygrad 的公共 API

`Tensor` 类是大多数用户唯一需要交互的东西。它封装了一个 UOp（第2章）并提供了类似 NumPy/PyTorch 的 API。本章解释 `Tensor` 如何建立在 UOp 图之上，惰性求值如何工作，以及计算实际何时发生。

## 三个属性

一个 Tensor 恰好有三个字段：

```python
class Tensor:
    __slots__ = "uop", "requires_grad", "grad"
```

就这些。没有 shape 数组，没有 stride 追踪器，没有 device 字段。一切都从 `uop` 派生：

```python
@property
def device(self) -> str|tuple[str, ...]: return self.uop.device

@property
def shape(self) -> tuple[sint, ...]: return self.uop.shape

@property
def dtype(self) -> DType: return self.uop.dtype
```

Tensor 是一个薄封装。UOp 完成所有工作。

## 创建 Tensor

当你写 `Tensor([1, 2, 3])` 时，以下是发生的事情：

```python
def __init__(self, data, device=None, dtype=None, requires_grad=None):
    # 1. Figure out dtype from the data
    if dtype is None:
        dtype = dtypes.default_int  # for [1, 2, 3]

    # 2. Pack the Python list into bytes
    data = _frompy(data, dtype)
    # _frompy creates a UOp.new_buffer on device "PYTHON"
    # and copies the data into it

    # 3. Copy to target device if needed
    if data.device != device:
        data = data.copy_to_device(device)

    self.uop = data
    self.grad = None
    self.requires_grad = requires_grad
```

不同的输入类型走不同的路径：

| 输入 | 发生了什么 |
|-------|-------------|
| `Tensor(3.14)` | 创建一个 `UOp.const` — 没有缓冲区，只是图中的一个常量 |
| `Tensor([1,2,3])` | 通过 `struct.pack` 打包为字节，在 `"PYTHON"` 设备上创建缓冲区 |
| `Tensor(numpy_array)` | 在 `"NPY"` 设备上创建指向 numpy 数据的缓冲区 |
| `Tensor(existing_uop)` | 直接使用该 UOp |
| `Tensor(pathlib.Path)` | 创建一个 `DISK` 缓冲区 — 数据在需要时才从磁盘读取 |

## 惰性求值

关于 Tensor 最重要的一点：**在你请求结果之前，不会进行任何计算。**

```python
a = Tensor([1, 2, 3])   # just builds a graph
b = Tensor([4, 5, 6])   # just builds a graph
c = a + b                # just builds a graph — no addition happens!
c = c * 2               # still just building the graph
print(c.numpy())         # NOW computation happens
```

每个操作都调用 `_apply_uop`，它在图中创建一个新的 UOp：

```python
def _apply_uop(self, fxn, *x, extra_args=(), **kwargs):
    srcs = (self,) + x
    new_uop = fxn(*[t.uop for t in srcs], *extra_args, **kwargs)

    # Create the result Tensor (no computation yet!)
    ret = Tensor.__new__(Tensor)
    ret.uop = new_uop
    ret.grad = None

    # Propagate requires_grad
    needs_input_grad = [t.requires_grad for t in srcs]
    ret.requires_grad = True if any(needs_input_grad) else \
                         None if None in needs_input_grad else False
    return ret
```

`_binop` 方法展示了二元操作的工作方式：

```python
def _binop(self, op, x, reverse):
    lhs, rhs = self._broadcasted(x, reverse)  # handle broadcasting
    return lhs._apply_uop(lambda *u: u[0].alu(op, *u[1:]), rhs)
```

当你写 `a + b` 时，Python 调用 `a.__add__(b)`，它调用 `a._binop(Ops.ADD, b, False)`，这会创建一个带有 `Ops.ADD` 的 UOp。没有数学运算发生。

## 计算何时发生

计算由三个方法触发：

### 1. `.realize()` — 显式触发

```python
c = (a + b) * 2
c.realize()  # forces computation
```

以下是 `realize` 的工作方式：

```python
def realize(self, *lst, do_update_stats=True):
    # 1. Handle pending assigns (for in-place operations)
    # 2. Check if already realized
    if self.uop.has_buffer_identity():
        return self  # already a concrete buffer, nothing to do

    # 3. Create schedule and run it
    run_schedule(*Tensor.schedule_with_vars(self))
    return self
```

### 2. `.numpy()` / `.item()` / `.data()` — 数据提取

这些方法调用 `._buffer()`，它在内部调用 `.realize()`：

```python
def _buffer(self):
    x = self.cast(self.dtype.base).contiguous()
    return cast(Buffer, x.realize().uop.buffer).ensure_allocated()

def item(self):
    assert self.numel() == 1, "must have one element for item"
    return self.data()[(0,) * len(self.shape)]

def numpy(self):
    return self._buffer().numpy().reshape(self.shape)
```

### 3. `.backward()` — 梯度计算

这会触发梯度图的 realize（在第30章中介绍）。

## 调度

当计算被触发时，`schedule_with_vars` 将 UOp 图转换为可执行项的列表：

```python
def schedule_with_vars(self, *lst):
    # Collect all UOps from all tensors being realized
    big_sink = UOp.sink(*[x.uop for x in (self,) + lst])

    # This is the big function — see Chapter 4 (Scheduling)
    becomes_map, schedule, var_vals = complete_create_schedule_with_vars(big_sink)

    # Update all live tensors to point to realized buffers
    _apply_map_to_tensors(becomes_map, name="buffers")

    return schedule, var_vals
```

调度完成后，UOp 图被转换为 `ExecItem` 列表 — 每一项都是一个需要运行的 GPU 内核（或内存拷贝）。

## 全局 Tensor 注册表

tinygrad 追踪所有存活的 Tensor：

```python
all_tensors: dict[weakref.ref[Tensor], None] = {}
```

当一个 Tensor 被创建时，它会被添加：
```python
all_tensors[weakref.ref(self)] = None
```

当它被垃圾回收时，它会被移除：
```python
def __del__(self):
    all_tensors.pop(weakref.ref(self), None)
```

这个注册表被 `_apply_map_to_tensors` 使用 — realize 之后，所有存活的 Tensor 需要将它们的 UOp 更新为指向已 realize 的缓冲区，而不是旧的惰性图：

```python
def _apply_map_to_tensors(applied_map, name):
    # Find tensors whose UOps reference anything in applied_map
    scope_tensors = [t for tref in all_tensors
                     if (t := tref()) is not None
                     and t.uop.topovisit(visitor, in_scope)]

    # Substitute old UOps with new ones
    sink = UOp.sink(*[t.uop for t in scope_tensors])
    new_sink = sink.substitute(applied_map)

    # Update each tensor's UOp
    for t, s, ns in zip(scope_tensors, sink.src, new_sink.src):
        if s is ns: continue
        t.uop = ns
```

## `.assign()` — 原地操作

原地操作使用 `assign`，它创建一个待处理的写入：

```python
def assign(self, x):
    # Validate shapes and devices match
    assert self.shape == x.shape
    assert self.device == x.device
    assert self.dtype == x.dtype

    result = self._apply_uop(UOp.assign, x)

    # Track as pending assign — will be realized when the buffer is next read
    _pending_assigns.setdefault(buf_uop, []).append(result.uop)
    return self.replace(result)
```

这就是 `Tensor.assign` 与优化器配合工作的方式：

```python
# In Adam optimizer:
self.b1_running *= self.b1  # creates pending assign
self.b2_running *= self.b2  # creates pending assign
# Both are realized when the next forward pass reads these buffers
```

## `.detach()` — 停止梯度

`detach` 创建一个不传播梯度的新 Tensor：

```python
def detach(self):
    return Tensor(self.uop.detach(), device=self.device, requires_grad=False)
```

UOp 图中的 `Ops.DETACH` 操作充当屏障 — `compute_gradient` 不会穿过它进行遍历。

## 广播

当操作数具有不同的形状时，tinygrad 会自动进行广播：

```python
a = Tensor.ones(3, 4)    # shape (3, 4)
b = Tensor.ones(4)       # shape (4,)
c = a + b                # b is broadcast to (3, 4)
```

广播的工作方式：
1. 左对齐形状：`(4,)` 变为 `(1, 4)`
2. 扩展：`(1, 4)` 通过 `Ops.EXPAND` 变为 `(3, 4)`

不会复制数据 — 扩展只是改变了相同数据的索引方式。

## 多 Tensor 同时 realize

你可以一次 realize 多个 Tensor 以获得更好的融合：

```python
a = Tensor([1, 2, 3])
b = a + 1
c = a * 2
# Realize both at once — the scheduler can optimize across them
b.realize(c)
```

## PyTorch 对比

| 特性 | PyTorch | tinygrad |
|---------|---------|---------|
| 求值方式 | 即时求值（Eager） | 惰性求值（Lazy） |
| 数据结构 | `torch.Tensor` (C++) | `Tensor` 封装 `UOp` (Python) |
| 形状追踪 | 存储在 Tensor 中 | 从 UOp 图派生 |
| 设备 | 存储在 Tensor 中 | 从 UOp 图派生 |
| 自动微分 | 基于 Tape | 图重写（第30章） |
| 代码量 | 约300万行 | 约1万行 |

## 练习

1. **追踪创建过程**：运行 `DEBUG=4 python -c "from tinygrad import Tensor; t = Tensor([1,2,3]); print(t.numpy())"`。调度了多少个内核？

2. **惰性求值证明**：创建 `a = Tensor([1,2,3]); b = a + a + a + a`。在调用 `.numpy()` 之前，不会发生任何计算。通过检查 `a.uop.op` 来验证 — 它应该仍然是一个缓冲区，而不是计算结果。

3. **多 Tensor realize**：比较 `a.realize(); b.realize()` 和 `a.realize(b)`。使用 `DEBUG=2`，计算内核数量。同时 realize 可能由于融合而产生更少的内核。

4. **阅读源码**：在 `tinygrad/tensor.py` 中，找到 `__init__` 方法。追踪每种输入类型（int、list、numpy 数组、UOp）的处理过程。

## 源码导航

| 文件 | 阅读内容 |
|------|-------------|
| `tinygrad/tensor.py:102-175` | `Tensor.__init__` — 所有创建路径 |
| `tinygrad/tensor.py:179-190` | `_apply_uop` — 操作如何构建图 |
| `tinygrad/tensor.py:252-292` | `schedule_with_vars` 和 `realize` — 计算何时发生 |
| `tinygrad/tensor.py:332-401` | `_buffer`、`data`、`item`、`numpy` — 提取数据 |
