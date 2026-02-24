# 第32章：Buffer 与内存管理

每个 tensor 的数据都存储在一个 `Buffer` 中。本章介绍 GPU 内存是如何分配、缓存、复用和释放的。

## Buffer 类

```python
class Buffer:
    def __init__(self, device, size, dtype, opaque=None, options=None,
                 initial_value=None, base=None, offset=0):
        self.device = device    # "METAL", "CUDA", "CPU", etc.
        self.size = size        # number of elements
        self.dtype = dtype      # element type
        self.options = options  # BufferSpec (image, uncached, etc.)
        self.offset = offset    # byte offset for views
        self._base = base       # parent buffer for views
```

Buffer 是设备内存中一块区域的句柄。它不直接持有数据——而是持有一个不透明的 `_buf` 对象，由设备的 Allocator 来解释和操作。

## Buffer 生命周期

```
Buffer()          → created (not yet allocated)
  .allocate()     → GPU memory allocated
  .copyin(mv)     → data copied from CPU to GPU
  .copyout(mv)    → data copied from GPU to CPU
  .deallocate()   → GPU memory freed
  .__del__()      → automatic cleanup via garbage collection
```

### 分配

```python
def allocate(self, opaque=None):
    assert not self.is_initialized(), "can't allocate twice"
    self.allocator = Device[self.device].allocator

    if self._base is not None:
        # View: share memory with parent buffer
        self._base.ensure_allocated()
        self._buf = self.allocator._offset(self.base._buf, self.nbytes, self.offset)
    else:
        # New allocation
        self._buf = opaque if opaque is not None else self.allocator.alloc(self.nbytes, self.options)
        GlobalCounters.mem_used += self.nbytes
    return self
```

### 释放

```python
def deallocate(self):
    if self._base is None:
        GlobalCounters.mem_used -= self.nbytes
        self.allocator.free(self._buf, self.nbytes, self.options)
    del self._buf

def __del__(self):
    if hasattr(self, '_buf'):
        self.deallocate()
```

当 Buffer 被垃圾回收时，其 GPU 内存会自动释放。

## Buffer 视图

视图与父 Buffer 共享内存，但具有不同的偏移量：

```python
def view(self, size, dtype, offset):
    return Buffer(self.device, size, dtype, base=self.base, offset=self.offset + offset)
```

视图用于切片等操作——不会复制数据：

```python
t = Tensor([1, 2, 3, 4, 5])
s = t[1:4]  # view of the same buffer, offset by 1 element
```

## Allocator 层级结构

```
Allocator (base class)
  └── LRUAllocator (caching layer)
        └── MetalAllocator, CUDAAllocator, etc. (device-specific)
```

### 基础 Allocator

```python
class Allocator:
    def alloc(self, size, options=None):
        return self._alloc(size, options or self.default_buffer_spec)

    def free(self, opaque, size, options=None):
        self._free(opaque, options or self.default_buffer_spec)

    # Implemented by each device:
    def _alloc(self, size, options): raise NotImplementedError
    def _free(self, opaque, options): pass
    def _copyin(self, dest, src: memoryview): raise NotImplementedError
    def _copyout(self, dest: memoryview, src): raise NotImplementedError
```

### LRU Allocator —— 缓存层

GPU 内存分配开销很大。LRU Allocator 会缓存已释放的 Buffer 以便复用：

```python
class LRUAllocator(Allocator):
    def __init__(self, dev):
        self.cache: dict[tuple[int, BufferSpec|None], Any] = defaultdict(list)

    def alloc(self, size, options=None):
        # Try to reuse a cached buffer of the same size
        if len(c := self.cache[(size, options)]):
            return c.pop()
        # No cached buffer — allocate new
        try:
            return super().alloc(size, options)
        except (RuntimeError, MemoryError):
            # Out of memory — free cache and retry
            self.free_cache()
            return super().alloc(size, options)

    def free(self, opaque, size, options=None):
        if LRU:  # LRU caching enabled
            self.cache[(size, options)].append(opaque)
        else:
            super().free(opaque, size, options)
```

当你释放一个 Buffer 时，它会进入缓存。下次需要相同大小的 Buffer 时，可以立即复用——无需调用 GPU 分配。

如果 GPU 内存耗尽，`free_cache()` 会真正释放所有缓存的 Buffer 并重试。

## BufferSpec —— Buffer 选项

```python
@dataclass(frozen=True)
class BufferSpec:
    image: ImageDType|None = None   # use texture memory
    uncached: bool = False          # bypass LRU cache
    cpu_access: bool = False        # CPU-accessible GPU memory
    host: bool = False              # host-pinned memory
    nolru: bool = False             # don't cache on free
    external_ptr: int|None = None   # use externally allocated memory
```

## 数据传输

将数据传入和传出 GPU 内存：

```python
# CPU → GPU
buf = Buffer("METAL", 1024, dtypes.float32)
buf.allocate()
buf.copyin(memoryview(bytearray(4096)))  # 1024 floats × 4 bytes

# GPU → CPU
mv = memoryview(bytearray(4096))
buf.copyout(mv)

# Zero-copy (when possible)
mv = buf.as_memoryview(allow_zero_copy=True)
```

## 内存规划器

在调度之后，内存规划器会优化 Buffer 分配：

```python
from tinygrad.engine.memory import memory_planner
schedule = memory_planner(schedule)
```

内存规划器的工作原理：
1. 分析 Buffer 的生命周期（每个 Buffer 首次写入和最后读取的时间）
2. 复用在时间上不重叠的 Buffer
3. 降低峰值内存使用量

如果不使用内存规划器，每个中间结果都会获得自己的 Buffer。使用后，Buffer 可以共享：

```
Without planner:  buf1 [████████]
                  buf2     [████████]
                  buf3         [████████]
Peak: 3 buffers

With planner:     buf1 [████████]
                  buf1         [████████]  (reused!)
                  buf2     [████████]
Peak: 2 buffers
```

## GlobalCounters

tinygrad 全局跟踪内存使用情况：

```python
class GlobalCounters:
    mem_used: int = 0      # current GPU memory in bytes
    kernel_count: int = 0  # total kernels executed
    global_ops: int = 0    # total FLOPs
    global_mem: int = 0    # total memory bandwidth
```

你可以检查内存使用情况：
```python
from tinygrad.helpers import GlobalCounters
print(f"GPU memory: {GlobalCounters.mem_used / 1e6:.1f} MB")
```

## 练习

1. **跟踪内存**：创建大小递增的 tensor，并在每次 `.realize()` 之后打印 `GlobalCounters.mem_used`。内存何时被释放？

2. **LRU 实战**：创建并 realize 一个 tensor，删除它，再创建一个相同大小的 tensor。使用 `DEBUG=7`，观察第二次分配是否复用了缓存的 Buffer。

3. **视图**：创建 `t = Tensor.arange(10)` 和 `s = t[3:7]`。它们是否由同一个 Buffer 支持？（提示：检查 `s` 是否使用了带偏移量的视图。）

## 源代码索引

| 文件 | 阅读内容 |
|------|---------|
| `tinygrad/device.py:95-204` | `Buffer` 类——分配、释放、视图 |
| `tinygrad/device.py:221-263` | `Allocator` 和 `LRUAllocator` |
| `tinygrad/device.py:73-80` | `BufferSpec` 选项 |
| `tinygrad/engine/memory.py` | 内存规划器 |
