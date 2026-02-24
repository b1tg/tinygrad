# Chapter 18: Symbolic Math & Memoryview

Tinygrad needs to do math on expressions that aren't numbers yet — loop bounds, tensor dimensions, index calculations. This chapter covers tinygrad's symbolic math system and the memoryview-based memory model.

## Symbolic Expressions

When tinygrad compiles a kernel, it needs to compute index expressions like:

```python
# For a (M, N) matmul where M and N might be symbolic:
index = row * N + col
```

If `N` is a compile-time constant (say 64), this simplifies to `row * 64 + col`. But if `N` is a runtime variable (e.g., from dynamic batch sizes), it stays symbolic.

Tinygrad represents symbolic expressions as UOps:

```python
from tinygrad.uop.ops import UOp, Ops
from tinygrad.dtype import dtypes

# Create a symbolic variable
N = UOp(Ops.DEFINE_VAR, dtypes.int, arg="N")

# Build symbolic expressions
idx = N * UOp(Ops.CONST, dtypes.int, arg=2)  # N * 2
print(idx.op)  # Ops.MUL
```

## The Symbolic Simplifier

The `symbolic` PatternMatcher automatically simplifies expressions:

```python
# x + 0 -> x
# x * 1 -> x
# x * 0 -> 0
# (x + c1) + c2 -> x + (c1 + c2)
# (x * c1) * c2 -> x * (c1 * c2)
# x // 1 -> x
# x % 1 -> 0
# (x * c) // c -> x  (when c > 0)
# (x * c) % c -> 0
```

These rules are critical for rangeify. When a reshape converts `r0` into `r0 // 3` and `r0 % 3`, and then a subsequent reshape combines them back, the simplifier must recover `r0`.

Example:
```
reshape(6) -> reshape(2, 3): indices = [r0 // 3, r0 % 3]
reshape(2, 3) -> reshape(6): index = (r0 // 3) * 3 + (r0 % 3)

Simplification: (r0 // 3) * 3 + (r0 % 3) = r0  ✓
```

## Variable Bounds

Symbolic variables have bounds (min and max values), which enable range-based optimizations:

```python
# If r0 ranges from 0 to 5:
# r0 // 6 is always 0  (since r0 < 6)
# r0 % 6 is always r0  (since r0 < 6)
# r0 >= 0 is always true
# r0 < 0 is always false
```

These bounds propagate through expressions:

```python
# r0 in [0, 5], r1 in [0, 3]
# r0 + r1 in [0, 8]
# r0 * r1 in [0, 15]
# r0 * 4 + r1 in [0, 23]
```

Bound tracking lets tinygrad eliminate dead code paths:
```python
# if (r0 < 0) { ... }  // dead code, r0 >= 0 always
# if (r0 < 6) { ... }  // always true, can remove the if
```

## Integer Division and Modulo

Integer division and modulo are the most complex operations to simplify. Tinygrad has dedicated logic in `tinygrad/uop/divandmod.py`:

```python
# Key identities:
# (a * b + c) // b = a + c // b   (when 0 <= c < b)
# (a * b + c) % b = c % b         (when 0 <= c < b)
# (a // b) // c = a // (b * c)
# (a % (b * c)) // b = (a // b) % c
```

These identities are what make chains of reshapes compile to efficient code.

## Memoryview: Zero-Copy Memory Access

Tinygrad uses Python's `memoryview` for zero-copy CPU-side data access:

```python
from tinygrad import Tensor

t = Tensor([1.0, 2.0, 3.0, 4.0])
t.realize()

# Get a memoryview into the GPU buffer (if possible)
mv = t.lazydata.buffer.as_memoryview()
# This is a zero-copy view — modifying mv modifies the buffer directly
```

### How Buffer Memory Works

A `Buffer` in tinygrad wraps a device-specific allocation:

```python
from tinygrad.device import Buffer
from tinygrad.dtype import dtypes

# Allocate 16 floats on CPU
buf = Buffer('CPU', 16, dtypes.float)
buf.ensure_allocated()

# Get a memoryview for direct access
mv = buf.as_memoryview()
mv = mv.cast('f')  # interpret as float32
mv[0] = 42.0
print(mv[0])  # 42.0
```

This is used extensively in:
- The AMD emulator (Chapter 12) — WaveState uses memoryviews into Buffer objects for register files
- Data loading — copying numpy arrays to GPU goes through memoryview
- The PYTHON backend — pure Python execution uses memoryview for memory access

### The cast() Trick

Python's `memoryview.cast()` reinterprets bytes as different types without copying:

```python
import struct

# Create bytes representing a float32
data = struct.pack('f', 3.14)
mv = memoryview(bytearray(data))

# Interpret as float
print(mv.cast('f')[0])  # 3.14

# Interpret as uint32 (same bits, different interpretation)
print(mv.cast('I')[0])  # 1078523331 (IEEE 754 encoding of 3.14)
```

This is how tinygrad handles bitcasts and dtype reinterpretation without data copies.

## Exercises

1. **Simplify by hand**: Given `r0` in [0, 11], simplify `(r0 // 4) * 4 + (r0 % 4)`. Verify it equals `r0`.

2. **Bound propagation**: If `r0` in [0, 3] and `r1` in [0, 7], what are the bounds of `r0 * 8 + r1`? Can we guarantee `r0 * 8 + r1 < 32`?

3. **Memoryview**: Create a tinygrad Buffer on CPU, write values through a memoryview, then read them back:
   ```python
   from tinygrad.device import Buffer
   from tinygrad.dtype import dtypes
   buf = Buffer('CPU', 4, dtypes.float).ensure_allocated()
   mv = buf.as_memoryview(force_zero_copy=True).cast('f')
   mv[0] = 1.0; mv[1] = 2.0; mv[2] = 3.0; mv[3] = 4.0
   print(list(mv))
   ```

## Source Code Map

| File | What to read |
|------|-------------|
| `tinygrad/uop/symbolic.py` | The symbolic simplifier PatternMatcher |
| `tinygrad/uop/divandmod.py` | Integer div/mod simplification rules |
| `tinygrad/device.py` | `Buffer` class and `as_memoryview()` |
| `tinygrad/dtype.py` | DType system, storage formats |
