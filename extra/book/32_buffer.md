# Chapter 32: Buffers and Memory Management

Every tensor's data lives in a `Buffer`. This chapter explains how GPU memory is allocated, cached, reused, and freed.

## The Buffer Class

```python
class Buffer:
    def __init__(self, device, size, dtype, opaque=None, options=None,
                 initial_value=None, base=None, offset=0):
        self.device = device    # "METAL", "CUDA", "CPU", etc.
        self.size = size        # number of elements
        self.dtype = dtype      # element type
        self.options = options   # BufferSpec (image, uncached, etc.)
        self.offset = offset    # byte offset for views
        self._base = base       # parent buffer for views
```

A Buffer is a handle to a chunk of device memory. It doesn't hold the data directly — it holds an opaque `_buf` object that the device allocator understands.

## Buffer Lifecycle

```
Buffer()          → created (not yet allocated)
  .allocate()     → GPU memory allocated
  .copyin(mv)     → data copied from CPU to GPU
  .copyout(mv)    → data copied from GPU to CPU
  .deallocate()   → GPU memory freed
  .__del__()      → automatic cleanup via garbage collection
```

### Allocation

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

### Deallocation

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

When a Buffer is garbage collected, its GPU memory is freed automatically.

## Buffer Views

A view shares memory with a parent buffer at a different offset:

```python
def view(self, size, dtype, offset):
    return Buffer(self.device, size, dtype, base=self.base, offset=self.offset + offset)
```

Views are used for operations like slicing — no data is copied:

```python
t = Tensor([1, 2, 3, 4, 5])
s = t[1:4]  # view of the same buffer, offset by 1 element
```

## The Allocator Hierarchy

```
Allocator (base class)
  └── LRUAllocator (caching layer)
        └── MetalAllocator, CUDAAllocator, etc. (device-specific)
```

### Base Allocator

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

### LRU Allocator — The Caching Layer

GPU allocation is expensive. The LRU allocator caches freed buffers for reuse:

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

When you free a buffer, it goes into the cache. Next time you need a buffer of the same size, it's reused instantly — no GPU allocation call needed.

If the GPU runs out of memory, `free_cache()` actually frees all cached buffers and retries.

## BufferSpec — Buffer Options

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

## Data Transfer

Getting data in and out of GPU memory:

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

## Memory Planner

After scheduling, the memory planner optimizes buffer allocation:

```python
from tinygrad.engine.memory import memory_planner
schedule = memory_planner(schedule)
```

The memory planner:
1. Analyzes buffer lifetimes (when each buffer is first written and last read)
2. Reuses buffers that don't overlap in time
3. Reduces peak memory usage

Without the memory planner, each intermediate result gets its own buffer. With it, buffers are shared:

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

tinygrad tracks memory usage globally:

```python
class GlobalCounters:
    mem_used: int = 0      # current GPU memory in bytes
    kernel_count: int = 0  # total kernels executed
    global_ops: int = 0    # total FLOPs
    global_mem: int = 0    # total memory bandwidth
```

You can check memory usage:
```python
from tinygrad.helpers import GlobalCounters
print(f"GPU memory: {GlobalCounters.mem_used / 1e6:.1f} MB")
```

## Exercises

1. **Track memory**: Create tensors of increasing size and print `GlobalCounters.mem_used` after each `.realize()`. When does memory get freed?

2. **LRU in action**: Create and realize a tensor, delete it, create another of the same size. With `DEBUG=7`, observe that the second allocation reuses the cached buffer.

3. **Views**: Create `t = Tensor.arange(10)` and `s = t[3:7]`. Are they backed by the same buffer? (Hint: check if `s` uses a view with an offset.)

## Source Code Map

| File | What to read |
|------|-------------|
| `tinygrad/device.py:95-204` | `Buffer` class — allocation, deallocation, views |
| `tinygrad/device.py:221-263` | `Allocator` and `LRUAllocator` |
| `tinygrad/device.py:73-80` | `BufferSpec` options |
| `tinygrad/engine/memory.py` | Memory planner |
