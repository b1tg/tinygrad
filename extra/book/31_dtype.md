# Chapter 31: Dtype System — Types, Promotion, and Precision

Every number in tinygrad has a type. This chapter explains the dtype system — how types are defined, how they interact, and how exotic formats like bfloat16 and fp8 work.

## DType: The Type Object

```python
@dataclass(frozen=True, eq=False)
class DType(metaclass=DTypeMetaClass):
    priority: int   # determines upcasting order
    bitsize: int    # bits per element
    name: str       # C type name ("float", "int", etc.)
    fmt: str|None   # struct.pack format character
    count: int      # vector width (1 for scalars)
    _scalar: DType|None  # scalar version of vector types
```

Every dtype is a singleton — `DTypeMetaClass` caches instances so `dtypes.float32` always returns the same object:

```python
class DTypeMetaClass(type):
    dcache: dict[tuple, DType] = {}
    def __call__(cls, *args, **kwargs):
        if (ret := dcache.get(args)) is not None: return ret
        dcache[args] = ret = super().__call__(*args)
        return ret
```

## The Type Hierarchy

```python
class dtypes:
    bool:    DType = DType.new(0,  1,   "bool",           '?')
    int8:    DType = DType.new(1,  8,   "signed char",    'b')
    uint8:   DType = DType.new(2,  8,   "unsigned char",  'B')
    int16:   DType = DType.new(3,  16,  "short",          'h')
    uint16:  DType = DType.new(4,  16,  "unsigned short", 'H')
    int32:   DType = DType.new(5,  32,  "int",            'i')
    uint32:  DType = DType.new(6,  32,  "unsigned int",   'I')
    int64:   DType = DType.new(7,  64,  "long",           'q')
    uint64:  DType = DType.new(8,  64,  "unsigned long",  'Q')
    fp8e4m3: DType = DType.new(9,  8,   "float8_e4m3",    None)
    fp8e5m2: DType = DType.new(10, 8,   "float8_e5m2",    None)
    float16: DType = DType.new(11, 16,  "half",           'e')
    bfloat16:DType = DType.new(12, 16,  "__bf16",         None)
    float32: DType = DType.new(13, 32,  "float",          'f')
    float64: DType = DType.new(14, 64,  "double",         'd')
```

The `priority` field determines promotion order. Higher priority types "win" when two types meet.

## Type Promotion Lattice

When you add an int32 to a float32, what type is the result? tinygrad follows JAX's type promotion rules:

```python
promo_lattice = {
    dtypes.bool:    [dtypes.int8, dtypes.uint8],
    dtypes.int8:    [dtypes.int16],
    dtypes.uint8:   [dtypes.int16, dtypes.uint16],
    dtypes.int16:   [dtypes.int32],
    dtypes.uint16:  [dtypes.int32, dtypes.uint32],
    dtypes.int32:   [dtypes.int64],
    dtypes.uint32:  [dtypes.int64, dtypes.uint64],
    dtypes.int64:   [dtypes.uint64],
    dtypes.uint64:  [dtypes.fp8e4m3, dtypes.fp8e5m2],
    dtypes.fp8e4m3: [dtypes.float16, dtypes.bfloat16],
    dtypes.fp8e5m2: [dtypes.float16, dtypes.bfloat16],
    dtypes.float16: [dtypes.float32],
    dtypes.bfloat16:[dtypes.float32],
    dtypes.float32: [dtypes.float64],
}
```

`least_upper_dtype` finds the smallest type that can represent both inputs:

```python
least_upper_dtype(dtypes.int32, dtypes.float16)  # → float32
least_upper_dtype(dtypes.uint8, dtypes.int8)     # → int16
least_upper_dtype(dtypes.bool, dtypes.float32)   # → float32
```

It works by finding the intersection of all ancestor sets in the lattice, then picking the minimum.

## Defaults

```python
dtypes.default_float = dtypes.float32  # what Tensor(3.14) creates
dtypes.default_int = dtypes.int32      # what Tensor(42) creates
```

You can override with `DEFAULT_FLOAT=half python ...` to run everything in float16.

## Exotic Float Formats

### bfloat16 (Brain Float)

Same exponent range as float32 (8 bits), but only 7 mantissa bits instead of 23:

```
float32:  1 sign + 8 exponent + 23 mantissa = 32 bits
bfloat16: 1 sign + 8 exponent +  7 mantissa = 16 bits
float16:  1 sign + 5 exponent + 10 mantissa = 16 bits
```

bfloat16 has less precision than float16 but much wider range. It's the standard for ML training because gradients can be very large or very small.

### fp8 (8-bit floats)

Two variants used for quantization:

```
fp8e4m3: 1 sign + 4 exponent + 3 mantissa  (range: ±448)
fp8e5m2: 1 sign + 5 exponent + 2 mantissa  (range: ±57344)
```

e4m3 has more precision (good for weights), e5m2 has more range (good for gradients).

### No `fmt` for Exotic Types

Notice that bfloat16 and fp8 have `fmt=None` — Python's `struct` module doesn't support them. tinygrad handles conversion manually:

```python
def float_to_bf16(x):
    u = struct.unpack('I', struct.pack('f', x))[0]
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000  # round to nearest even
    return struct.unpack('f', struct.pack('I', u))[0]
```

## Vector Types

For SIMD operations, dtypes can be vectorized:

```python
dtypes.float32.vec(4)   # float4 — four floats packed together
dtypes.half.vec(2)      # half2 — two halves packed together
```

Vector types are used in codegen for GPU operations that process multiple elements at once.

## PtrDType — Pointer Types

Buffers in GPU memory are represented as pointer types:

```python
@dataclass(frozen=True, eq=False)
class PtrDType(DType):
    _base: DType           # what the pointer points to
    addrspace: AddrSpace   # GLOBAL, LOCAL, or REG
    v: int                 # vector width
    size: int              # number of elements (-1 = unlimited)
```

Address spaces matter for GPU programming:
- `GLOBAL`: main GPU memory (slow, large)
- `LOCAL`: shared memory within a workgroup (fast, small)
- `REG`: registers (fastest, smallest)

## ImageDType — Texture Memory

For OpenCL image optimizations:

```python
@dataclass(frozen=True, eq=False)
class ImageDType(PtrDType):
    shape: tuple[int, ...] = ()  # image dimensions (height, width)
```

Image types use GPU texture units for memory access, which can be faster for certain access patterns.

## Type Checking Helpers

```python
dtypes.is_float(dtypes.float32)    # True
dtypes.is_float(dtypes.int32)      # False
dtypes.is_int(dtypes.uint8)        # True
dtypes.is_unsigned(dtypes.uint8)   # True
dtypes.is_unsigned(dtypes.int8)    # False

dtypes.min(dtypes.uint8)    # 0
dtypes.max(dtypes.uint8)    # 255
dtypes.min(dtypes.float32)  # -inf
dtypes.max(dtypes.float32)  # inf
```

## Accumulation Dtypes

When summing many small numbers, you need a wider accumulator to avoid overflow:

```python
def sum_acc_dtype(dt):
    if dtypes.is_unsigned(dt): return least_upper_dtype(dt, dtypes.uint)
    if dtypes.is_int(dt) or dt == dtypes.bool: return least_upper_dtype(dt, dtypes.int)
    return least_upper_dtype(dt, dtypes.float32)  # default: accumulate in float32
```

So `Tensor([1, 2, 3], dtype=dtypes.uint8).sum()` accumulates in uint32, not uint8.

## Exercises

1. **Promotion**: What is `least_upper_dtype(dtypes.int8, dtypes.uint8)`? Trace through the lattice.

2. **Range**: fp8e4m3 has 4 exponent bits and 3 mantissa bits. What's the largest representable number? (Answer: 448)

3. **Why bfloat16?**: Why does ML training prefer bfloat16 over float16? (Hint: think about gradient magnitudes.)

4. **Accumulation**: Why does summing uint8 values accumulate in uint32? What would go wrong with uint8 accumulation?

## Source Code Map

| File | What to read |
|------|-------------|
| `tinygrad/dtype.py:54-84` | `DType` class definition |
| `tinygrad/dtype.py:143-233` | `dtypes` class with all type definitions |
| `tinygrad/dtype.py:244-257` | Type promotion lattice and `least_upper_dtype` |
| `tinygrad/dtype.py:287-338` | fp8 conversion functions |
