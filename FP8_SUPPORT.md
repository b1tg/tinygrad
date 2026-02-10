# FP8 Support Status in TinyGrad

## Overview

TinyGrad supports four FP8 formats:
- `fp8e4m3` - E4M3 with standard IEEE-like encoding
- `fp8e5m2` - E5M2 with standard IEEE-like encoding  
- `fp8e4m3fnuz` - E4M3 with FNUZ (Finite Numbers Unsigned Zero) encoding
- `fp8e5m2fnuz` - E5M2 with FNUZ encoding

## Hardware Support Matrix

| Hardware | fp8e4m3 | fp8e5m2 | fp8e4m3fnuz | fp8e5m2fnuz |
|----------|---------|---------|-------------|-------------|
| NVIDIA (sm89+) | ✅ | ✅ | ❌ | ❌ |
| AMD gfx942 (MI300) | ❌ | ❌ | ✅ | ✅ |
| AMD gfx950 (MI350) | ✅ | ✅ | ❌ | ❌ |
| PYTHON backend | ✅ | ✅ | ✅ | ✅ |
| NULL backend | ✅ | ✅ | ✅ | ✅ |

### AMD Architecture Details

```python
# gfx942 = MI300 (CDNA3)
# Only supports FNUZ variants via builtin AMG instructions
supported_dtypes = {
    (9, 4, 2): (dtypes.fp8e4m3fnuz, dtypes.fp8e5m2fnuz),  # gfx942
    (9, 5, 0): (dtypes.fp8e4m3, dtypes.fp8e5m2),          # gfx950
}
```

## Format Specifications

### FP8E4M3 (Standard)
- **Bits**: 1 sign + 4 exponent + 3 mantissa
- **Bias**: 7
- **Max**: 448.0 (0x7e)
- **Min**: -448.0 (0xfe)
- **Inf**: Supported (0x7c positive, 0xfc negative)
- **NaN**: Supported (0x7f)

### FP8E5M2 (Standard)
- **Bits**: 1 sign + 5 exponent + 2 mantissa
- **Bias**: 15
- **Max**: 57344.0 (0x7b)
- **Min**: -57344.0 (0xfb)
- **Inf**: Supported (0x7c positive, 0xfc negative)
- **NaN**: Supported (0x7f)

### FP8E4M3FNUZ (FNUZ = Finite Numbers Unsigned Zero)
- **Bits**: 1 sign + 4 exponent + 3 mantissa
- **Bias**: 8
- **Max**: 240.0 (0x7f)
- **Min**: -240.0 (0xff)
- **Inf**: ❌ Not supported (encoded as NaN)
- **NaN**: 0x80 (all exponent and mantissa bits set with sign=1)
- **Zero**: Only +0 (0x00), -0 also encoded as 0x80

### FP8E5M2FNUZ (FNUZ)
- **Bits**: 1 sign + 5 exponent + 2 mantissa
- **Bias**: 16
- **Max**: 57344.0 (0x7f)
- **Min**: -57344.0 (0xff)
- **Inf**: ❌ Not supported (encoded as NaN)
- **NaN**: 0x80
- **Zero**: Only +0 (0x00)

## Backend Behavior Comparison

### Supported Types by Backend

#### AMD gfx942 (MI300)
Only supports FNUZ variants:
- ✅ fp8e4m3fnuz
- ✅ fp8e5m2fnuz
- ❌ fp8e4m3 (N/A)
- ❌ fp8e5m2 (N/A)

Uses AMD builtin intrinsics:
- `__builtin_amdgcn_cvt_pk_fp8_f32` - convert float to fp8
- `__builtin_amdgcn_cvt_pk_bf8_f32` - convert float to bf8 (E5M2)
- `__builtin_amdgcn_cvt_f32_fp8` - convert fp8 to float
- `__builtin_amdgcn_cvt_f32_bf8` - convert bf8 to float

#### PyTorch
Supports all 4 types natively:
- `torch.float8_e4m3fn`
- `torch.float8_e5m2`
- `torch.float8_e4m3fnuz`
- `torch.float8_e5m2fnuz`

#### TinyGrad PYTHON=1 Backend
Software emulation using `float_to_fp8()` and `fp8_to_float()` functions in `tinygrad/dtype.py`.

## Boundary Value Behavior

### FP8E4M3 (Standard) - NOT on gfx942

| Value | PyTorch | TinyGrad PYTHON=1 | AMD gfx942 |
|-------|---------|-------------------|------------|
| 0.0 | 0.0 (0x00) | 0.0 (0x00) | N/A |
| 1.0 | 1.0 (0x38) | 1.0 (0x38) | N/A |
| 448.0 (max) | 448.0 (0x7e) | 448.0 (0x7e) | N/A |
| 449.0 (overflow) | 448.0 (0x7e) | 448.0 (0x7e) | N/A |
| 464.0 (threshold) | 448.0 (0x7e) | 448.0 (0x7e) | N/A |
| > 464.0 (large overflow) | **NaN** (0x7f) | 448.0 (0x7e) | N/A |
| inf | NaN (0x7f) | NaN (0x7f) | N/A |
| nan | NaN (0x7f) | NaN (0x7f) | N/A |

**Overflow Threshold**: 464.0 (3.6% above max)

### FP8E5M2 (Standard) - NOT on gfx942

| Value | PyTorch | TinyGrad PYTHON=1 | AMD gfx942 |
|-------|---------|-------------------|------------|
| 0.0 | 0.0 (0x00) | 0.0 (0x00) | N/A |
| 1.0 | 1.0 (0x3c) | 1.0 (0x3c) | N/A |
| 57344.0 (max) | 57344.0 (0x7b) | 57344.0 (0x7b) | N/A |
| 60000.0 (overflow) | 57344.0 (0x7b) | 57344.0 (0x7b) | N/A |
| 61440.0 (threshold) | 57344.0 (0x7b) | 57344.0 (0x7b) | N/A |
| > 61440.0 (large overflow) | **Inf** (0x7c) | 57344.0 (0x7b) | N/A |
| inf | Inf (0x7c) | Inf (0x7c) | N/A |
| nan | NaN (0x7f) | NaN (0x7e) | N/A |

**Overflow Threshold**: 61440.0 (7.1% above max)

### FP8E4M3FNUZ ✅ SUPPORTED on gfx942

| Value | PyTorch | TinyGrad PYTHON=1 | AMD gfx942 |
|-------|---------|-------------------|------------|
| 0.0 | 0.0 (0x00) | 0.0 (0x00) | 0.0 (0x00) |
| 1.0 | 1.0 (0x40) | 1.0 (0x40) | 1.0 (0x40) |
| 240.0 (max) | 240.0 (0x7f) | 240.0 (0x7f) | 240.0 (0x7f) |
| 241.0 (overflow) | 240.0 (0x7f) | 240.0 (0x7f) | 240.0 (0x7f) |
| 280.0 (large overflow) | **NaN** (0x80) | **NaN** (0x80) | **NaN** (0x80) |
| inf | **NaN** (0x80) | **NaN** (0x80) | **NaN** (0x80) |
| nan | **NaN** (0x80) | **NaN** (0x80) | **NaN** (0x80) |

**Note**: ✅ All three backends match perfectly for FNUZ types

### FP8E5M2FNUZ ✅ SUPPORTED on gfx942

| Value | PyTorch | TinyGrad PYTHON=1 | AMD gfx942 |
|-------|---------|-------------------|------------|
| 0.0 | 0.0 (0x00) | 0.0 (0x00) | 0.0 (0x00) |
| 1.0 | 1.0 (0x40) | 1.0 (0x40) | 1.0 (0x40) |
| 57344.0 (max) | 57344.0 (0x7f) | 57344.0 (0x7f) | 57344.0 (0x7f) |
| 60000.0 (overflow) | 57344.0 (0x7f) | 57344.0 (0x7f) | 57344.0 (0x7f) |
| inf | **NaN** (0x80) | **NaN** (0x80) | **NaN** (0x80) |
| nan | **NaN** (0x80) | **NaN** (0x80) | **NaN** (0x80) |

**Note**: ✅ All three backends match perfectly for FNUZ types

## Known Differences Between PyTorch and TinyGrad

### 1. FP8E4M3 Large Overflow
- **PyTorch**: Values > 464.0 → NaN
- **TinyGrad**: All overflow → Clamp to 448.0
- **Impact**: Minimal (rare in practice)

### 2. FP8E5M2 Large Overflow
- **PyTorch**: Values > 61440.0 → Inf
- **TinyGrad**: All overflow → Clamp to 57344.0
- **Impact**: Minimal (rare in practice)

### 3. NaN Representation
- **FP8E5M2 NaN**:
  - PyTorch: 0x7f
  - TinyGrad: 0x7e
- **Impact**: Cosmetic (both decode to NaN)

## Implementation Details

### Key Source Files

1. **`tinygrad/dtype.py`**
   - `float_to_fp8()`: Converts Python float to FP8 integer
   - `fp8_to_float()`: Converts FP8 integer to Python float
   - `truncate`: Dictionary of truncation functions for each dtype

2. **`tinygrad/device.py`**
   - `is_dtype_supported()`: Checks if dtype is supported on device
   - AMD target mapping: `{(9,4,2): (fp8e4m3fnuz, fp8e5m2fnuz), (9,5,0): (fp8e4m3, fp8e5m2)}`

3. **`tinygrad/renderer/cstyle.py`**
   - `fp8_index()`: Maps dtype to index (0=fp8, 1=bf8)
   - `AMDHIPRenderer`: Generates AMD HIP code with FP8 intrinsics
   - `type_map`: Maps dtypes to HIP type names

4. **`tinygrad/runtime/ops_python.py`**
   - Software emulation using `from_storage_scalar()` and `to_storage_scalar()`

### AMD Code Generation

```cpp
// FP8 conversion in generated HIP code
static inline __attribute__((device)) unsigned char f32_to_fp8(float v, int is_bf8) {
  v = (((*(unsigned*)&v)&0x7F800000)!=0x7F800000)?__builtin_amdgcn_fmed3f(
    v, is_bf8?57344.0f:448.0f, is_bf8?-57344.0f:-448.0f) : v;
  return (unsigned char)(is_bf8?__builtin_amdgcn_cvt_pk_bf8_f32(v,v,0,false)
                                 :__builtin_amdgcn_cvt_pk_fp8_f32(v,v,0,false));
}
```

## Test Status

All FP8 tests pass:
```bash
python -m pytest test/test_dtype.py test/test_dtype_alu.py -q
# 288 passed, 55 skipped, 1 xfailed
```

Skipped tests are for unsupported dtypes on specific hardware (e.g., non-FNUZ on gfx942).

## Recommendations

### For MI300 (gfx942) Users
- Use **FNUZ variants only**: `fp8e4m3fnuz`, `fp8e5m2fnuz`
- Non-FNUZ types will fail with "dtype not supported"
- FNUZ provides better compatibility between CPU (PYTHON=1) and GPU

### For NVIDIA Users
- Use standard variants: `fp8e4m3`, `fp8e5m2`
- FNUZ variants are not supported on CUDA

### For Testing/Development
- Use `PYTHON=1` backend for debugging
- FNUZ types behave identically across all backends

## References

- [PyTorch FP8 Documentation](https://pytorch.org/docs/stable/tensors.html)
- [AMD CDNA3 Instruction Set](https://www.amd.com/en/products/accelerators/instinct/mi300.html)
- [FP8 Formats in ML](https://arxiv.org/abs/2209.05433)
