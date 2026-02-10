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

#### NVIDIA RTX 4090 (sm_89, CUDA=1)

Tested on: 5x NVIDIA GeForce RTX 4090, CUDA Driver 12.8, NVRTC 12.8

Only supports standard variants via `cuda_fp8.h` (`__nv_fp8_e4m3`, `__nv_fp8_e5m2`):
- ✅ fp8e4m3
- ✅ fp8e5m2
- ❌ fp8e4m3fnuz (NVRTC CompileError: `float8_e4m3fnuz` undefined)
- ❌ fp8e5m2fnuz (NVRTC CompileError: `float8_e5m2fnuz` undefined)

| Feature | fp8e4m3 | fp8e5m2 | fp8e4m3fnuz | fp8e5m2fnuz |
|---------|---------|---------|-------------|-------------|
| Cast float32 -> fp8 | ✅ | ✅ | ❌ | ❌ |
| Cast fp8 -> float32 | ✅ | ✅ | ❌ | ❌ |
| Cast fp8 -> float16 | ✅ | ✅ | ❌ | ❌ |
| Cast fp8 -> bfloat16 | ✅ | ✅ | ❌ | ❌ |
| Element-wise ops (+, *) | ✅ | ✅ | ❌ | ❌ |
| Matmul (fp8 @ fp8) | ✅ | ✅ | ❌ | ❌ |
| Tensor Core (WMMA) | ✅ (m16n8k32) | ✅ | ❌ | ❌ |
| FP8 Linear (extra/) | ✅ (6/6 pass) | ✅ | ❌ | ❌ |

**Tensor Core Details**: FP8 matmul uses WMMA instructions on sm_89 (Ada Lovelace):
- Instruction: `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`
- TC config: `cuda_81632_f8` (8x16x32 tile, FP8 inputs, FP32 accumulator)
- Automatically selected for matrices >= 64x64

**Matmul Accuracy** (32x32, inputs scaled by 0.1):

| Type | Max Abs Error | Mean Abs Error | Mean Rel Error |
|------|---------------|----------------|----------------|
| fp8e4m3 | 0.019 | 0.002 | 0.35 |
| fp8e5m2 | 0.021 | 0.004 | 0.48 |

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

### FP8E4M3 (Standard)

| Value | PyTorch | TinyGrad PYTHON=1 | CUDA (RTX 4090) | AMD gfx942 |
|-------|---------|-------------------|-----------------|------------|
| 0.0 | 0.0 (0x00) | 0.0 (0x00) | 0.0 (0x00) | N/A |
| 1.0 | 1.0 (0x38) | 1.0 (0x38) | 1.0 (0x38) | N/A |
| 448.0 (max) | 448.0 (0x7e) | 448.0 (0x7e) | 448.0 (0x7e) | N/A |
| 449.0 (overflow) | 448.0 (0x7e) | 448.0 (0x7e) | 448.0 (0x7e) | N/A |
| 464.0 (threshold) | 448.0 (0x7e) | 448.0 (0x7e) | 448.0 (0x7e) | N/A |
| 500.0 (large overflow) | **NaN** (0x7f) | 448.0 (0x7e) | **448.0** (0x7e) | N/A |
| inf | NaN (0x7f) | NaN (0x7f) | **448.0** (0x7e) | N/A |
| nan | NaN (0x7f) | NaN (0x7f) | NaN (0x7f) | N/A |

**Key difference**: CUDA saturates `inf` to max (448.0, 0x7e), while PYTHON=1 encodes it as NaN (0x7f). PyTorch also encodes it as NaN (0x7f).

### FP8E5M2 (Standard)

| Value | PyTorch | TinyGrad PYTHON=1 | CUDA (RTX 4090) | AMD gfx942 |
|-------|---------|-------------------|-----------------|------------|
| 0.0 | 0.0 (0x00) | 0.0 (0x00) | 0.0 (0x00) | N/A |
| 1.0 | 1.0 (0x3c) | 1.0 (0x3c) | 1.0 (0x3c) | N/A |
| 57344.0 (max) | 57344.0 (0x7b) | 57344.0 (0x7b) | 57344.0 (0x7b) | N/A |
| 60000.0 (overflow) | 57344.0 (0x7b) | 57344.0 (0x7b) | 57344.0 (0x7b) | N/A |
| 61440.0 (threshold) | 57344.0 (0x7b) | 57344.0 (0x7b) | 57344.0 (0x7b) | N/A |
| 65536.0 (large overflow) | **Inf** (0x7c) | 57344.0 (0x7b) | **57344.0** (0x7b) | N/A |
| inf | Inf (0x7c) | Inf (0x7c) | **57344.0** (0x7b) | N/A |
| nan | NaN (0x7f) | NaN (0x7e) | NaN (0x7f) | N/A |

**Key differences**:
- CUDA saturates `inf` to max (57344.0, 0x7b), while PYTHON=1 preserves Inf (0x7c) and PyTorch preserves Inf
- NaN encoding: CUDA 0x7f, PYTHON 0x7e, PyTorch 0x7f (all decode to NaN)

### FP8E4M3FNUZ

| Value | PyTorch | TinyGrad PYTHON=1 | CUDA (RTX 4090) | AMD gfx942 |
|-------|---------|-------------------|-----------------|------------|
| 0.0 | 0.0 (0x00) | 0.0 (0x00) | N/A | 0.0 (0x00) |
| 1.0 | 1.0 (0x40) | 1.0 (0x40) | N/A | 1.0 (0x40) |
| 240.0 (max) | 240.0 (0x7f) | 240.0 (0x7f) | N/A | 240.0 (0x7f) |
| 241.0 (overflow) | 240.0 (0x7f) | 240.0 (0x7f) | N/A | 240.0 (0x7f) |
| 280.0 (large overflow) | **NaN** (0x80) | **NaN** (0x80) | N/A | **NaN** (0x80) |
| inf | **NaN** (0x80) | **NaN** (0x80) | N/A | **NaN** (0x80) |
| nan | **NaN** (0x80) | **NaN** (0x80) | N/A | **NaN** (0x80) |

**Note**: ✅ PyTorch, PYTHON=1 and gfx942 all match perfectly. CUDA does not support FNUZ.

### FP8E5M2FNUZ

| Value | PyTorch | TinyGrad PYTHON=1 | CUDA (RTX 4090) | AMD gfx942 |
|-------|---------|-------------------|-----------------|------------|
| 0.0 | 0.0 (0x00) | 0.0 (0x00) | N/A | 0.0 (0x00) |
| 1.0 | 1.0 (0x40) | 1.0 (0x40) | N/A | 1.0 (0x40) |
| 57344.0 (max) | 57344.0 (0x7f) | 57344.0 (0x7f) | N/A | 57344.0 (0x7f) |
| 60000.0 (overflow) | 57344.0 (0x7f) | 57344.0 (0x7f) | N/A | 57344.0 (0x7f) |
| inf | **NaN** (0x80) | **NaN** (0x80) | N/A | **NaN** (0x80) |
| nan | **NaN** (0x80) | **NaN** (0x80) | N/A | **NaN** (0x80) |

**Note**: ✅ PyTorch, PYTHON=1 and gfx942 all match perfectly. CUDA does not support FNUZ.

## Known Differences

### 1. CUDA vs PYTHON=1: inf Handling
- **CUDA**: `inf` → Saturates to max value (fp8e4m3: 448.0, fp8e5m2: 57344.0)
- **PYTHON=1**: `inf` → NaN for fp8e4m3 (0x7f), Inf for fp8e5m2 (0x7c)
- **Impact**: Only affects edge cases with inf inputs

### 2. CUDA vs PyTorch: Large Overflow
- **PyTorch**: Values > 464.0 → NaN (fp8e4m3), Values > 61440.0 → Inf (fp8e5m2)
- **CUDA (tinygrad)**: All overflow → Clamp to max
- **Impact**: Minimal (rare in practice)

### 3. NaN Encoding
- **FP8E5M2**: CUDA 0x7f, PYTHON 0x7e, PyTorch 0x7f
- **Impact**: Cosmetic (all decode to NaN)

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

```bash
# CUDA=1 (RTX 4090, after is_dtype_supported fix)
CUDA=1 python -m pytest test/test_dtype.py -k "fp8 or Fp8"
# 56 passed, 4 skipped

CUDA=1 python -m pytest test/testextra/test_fp8_linear.py
# 6 passed (forward_2d, forward_3d, backward_2d, backward_3d, filter, multi_gpu)

# PYTHON=1 / AMD gfx942
python -m pytest test/test_dtype.py test/test_dtype_alu.py -q
# 288 passed, 55 skipped, 1 xfailed
```

Skipped tests are for unsupported dtypes on specific hardware (e.g., FNUZ on CUDA, non-FNUZ on gfx942).

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
