# 第31章：DType 系统 — 类型、提升与精度

tinygrad 中的每个数字都有类型。本章讲解 dtype 系统 — 类型如何定义、如何交互，以及 bfloat16 和 fp8 等特殊格式的工作原理。

## DType：类型对象

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

每个 dtype 都是单例 — `DTypeMetaClass` 会缓存实例，因此 `dtypes.float32` 始终返回同一个对象：

```python
class DTypeMetaClass(type):
    dcache: dict[tuple, DType] = {}
    def __call__(cls, *args, **kwargs):
        if (ret := dcache.get(args)) is not None: return ret
        dcache[args] = ret = super().__call__(*args)
        return ret
```

## 类型层次结构

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

`priority` 字段决定提升顺序。当两种类型相遇时，优先级更高的类型"获胜"。

## 类型提升格

当你将一个 int32 与一个 float32 相加时，结果是什么类型？tinygrad 遵循 JAX 的类型提升规则：

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

`least_upper_dtype` 找到能同时表示两个输入的最小类型：

```python
least_upper_dtype(dtypes.int32, dtypes.float16)  # → float32
least_upper_dtype(dtypes.uint8, dtypes.int8)     # → int16
least_upper_dtype(dtypes.bool, dtypes.float32)   # → float32
```

它的工作原理是在格中找到所有祖先集合的交集，然后选择最小的那个。

## 默认类型

```python
dtypes.default_float = dtypes.float32  # what Tensor(3.14) creates
dtypes.default_int = dtypes.int32      # what Tensor(42) creates
```

你可以通过 `DEFAULT_FLOAT=half python ...` 来覆盖默认值，从而以 float16 运行所有计算。

## 特殊浮点格式

### bfloat16（Brain Float）

与 float32 相同的指数范围（8位），但尾数只有7位而非23位：

```
float32:  1 sign + 8 exponent + 23 mantissa = 32 bits
bfloat16: 1 sign + 8 exponent +  7 mantissa = 16 bits
float16:  1 sign + 5 exponent + 10 mantissa = 16 bits
```

bfloat16 的精度低于 float16，但范围大得多。它是机器学习训练的标准格式，因为梯度值可能非常大或非常小。

### fp8（8位浮点数）

用于量化的两种变体：

```
fp8e4m3: 1 sign + 4 exponent + 3 mantissa  (range: ±448)
fp8e5m2: 1 sign + 5 exponent + 2 mantissa  (range: ±57344)
```

e4m3 精度更高（适合权重），e5m2 范围更大（适合梯度）。

### 特殊类型没有 `fmt`

注意 bfloat16 和 fp8 的 `fmt=None` — Python 的 `struct` 模块不支持它们。tinygrad 手动处理转换：

```python
def float_to_bf16(x):
    u = struct.unpack('I', struct.pack('f', x))[0]
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000  # round to nearest even
    return struct.unpack('f', struct.pack('I', u))[0]
```

## 向量类型

对于 SIMD 操作，dtype 可以被向量化：

```python
dtypes.float32.vec(4)   # float4 — four floats packed together
dtypes.half.vec(2)      # half2 — two halves packed together
```

向量类型用于代码生成中的 GPU 操作，一次处理多个元素。

## PtrDType — 指针类型

GPU 显存中的缓冲区用指针类型表示：

```python
@dataclass(frozen=True, eq=False)
class PtrDType(DType):
    _base: DType           # what the pointer points to
    addrspace: AddrSpace   # GLOBAL, LOCAL, or REG
    v: int                 # vector width
    size: int              # number of elements (-1 = unlimited)
```

地址空间对 GPU 编程很重要：
- `GLOBAL`：GPU 主显存（慢，容量大）
- `LOCAL`：工作组内的共享内存（快，容量小）
- `REG`：寄存器（最快，容量最小）

## ImageDType — 纹理内存

用于 OpenCL 图像优化：

```python
@dataclass(frozen=True, eq=False)
class ImageDType(PtrDType):
    shape: tuple[int, ...] = ()  # image dimensions (height, width)
```

图像类型利用 GPU 纹理单元进行内存访问，对于某些访问模式可以更快。

## 类型检查辅助函数

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

## 累加 DType

当对大量小数求和时，需要更宽的累加器以避免溢出：

```python
def sum_acc_dtype(dt):
    if dtypes.is_unsigned(dt): return least_upper_dtype(dt, dtypes.uint)
    if dtypes.is_int(dt) or dt == dtypes.bool: return least_upper_dtype(dt, dtypes.int)
    return least_upper_dtype(dt, dtypes.float32)  # default: accumulate in float32
```

因此 `Tensor([1, 2, 3], dtype=dtypes.uint8).sum()` 会以 uint32 而非 uint8 进行累加。

## 练习

1. **类型提升**：`least_upper_dtype(dtypes.int8, dtypes.uint8)` 的结果是什么？在格中追踪推导过程。

2. **范围**：fp8e4m3 有4位指数和3位尾数。最大可表示的数是多少？（答案：448）

3. **为什么选择 bfloat16？**：为什么机器学习训练更倾向于使用 bfloat16 而非 float16？（提示：思考梯度的数量级。）

4. **累加**：为什么对 uint8 值求和要以 uint32 进行累加？如果用 uint8 累加会出什么问题？

## 源代码索引

| 文件 | 阅读内容 |
|------|----------|
| `tinygrad/dtype.py:54-84` | `DType` 类定义 |
| `tinygrad/dtype.py:143-233` | `dtypes` 类及所有类型定义 |
| `tinygrad/dtype.py:244-257` | 类型提升格与 `least_upper_dtype` |
| `tinygrad/dtype.py:287-338` | fp8 转换函数 |
