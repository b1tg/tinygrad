# Adreno a7xx (a730) Porting Notes

## Status

| Feature | Status | Notes |
|---|---|---|
| Device init / GPU detection | ✅ | chip_id=0x07030002, parsed as a730 |
| llvm-qcom compilation | ✅ | Same binary as CL's clBuildProgram |
| Register mapping (NDRANGE, UPDATE_CNTL, etc.) | ✅ | All matched against CL driver via ioctl hook |
| Signal / timestamp / cache flush | ✅ | CP_EVENT_WRITE7, CACHE_FLUSH7, EV_WRITE_ALWAYSON |
| WIE_CNTL_0/1 (work ID config) | ✅ | Required for get_global_id / multi-wave |
| THREAD128 | ✅ | a730 CL driver uses THREAD128, not THREAD64 |
| CONST_CONFIG constlen | ✅ | a7xx field is 8-bit (max 255), use 1024//16=64 |
| test_tiny | ✅ 17/17 | |
| GEMM up to 1024x1024 | ✅ | |
| LLM (NOOPT=1) | ✅ | Correct output, ~0.09 tok/s |
| LLM (NOOPT=0, optimized) | ❌ | Wrong output, see "Remaining Bug" below |
| CL backend | ✅ | `OPENCL_PATH=/system/vendor/lib64/libOpenCL.so DEV=CL`, 2.67 tok/s |

## Files Changed

- `tinygrad/runtime/ops_qcom.py` — a7xx register mapping, dispatch, cache flush
- `tinygrad/runtime/support/compiler_qcom.py` — dynamic chip_id from arch string
- `tinygrad/runtime/support/compiler_mesa.py` — IR3Compiler accepts new arch format
- `extra/qcom_gpu_driver/opencl_ioctl.py` — mesa fallback, indirect buffer parsing, a7xx compare

## Key a7xx Register Differences from a6xx

### Addresses that moved

| Register | a6xx | a7xx |
|---|---|---|
| SP_CS_NDRANGE_0~6 | 0xb990~0xb996 | 0xa9d4~0xa9da |
| SP_CS_WGE_CNTL | 0xb998 | 0xa9db |
| SP_CS_KERNEL_GROUP_X/Y/Z | 0xb999~0xb99b | 0xa9dc~0xa9de |
| SP_CS_CONST_CONFIG | 0xb987 | 0xa9cd |
| SP_UPDATE_CNTL | 0xbb08 | 0xab1f |
| SP_REG_PROG_ID_0 | 0xb983 | 0xa9c8 |
| SP_CS_UAV_BASE | 0xa9f2 | 0xa9f8 |

### New a7xx registers

- `SP_CS_NDRANGE_7` (0xa9df) — last workgroup size for unaligned dispatch
- `SP_CS_VGS_CNTL` (0xa9c5) — set to 0

### Shared addresses (unchanged)

SP_CS_CNTL_0/1 (0xa9b0), SP_CS_CONFIG (0xa9bb), SP_CS_TSIZE (0xa9ba), SP_CS_USIZE (0xaa00), SP_MODE_CNTL (0xab00), TPL1_MODE_CNTL (0xb309), SP_CS_PVT_MEM_* (0xa9b6+), SP_CS_INSTR_SIZE (0xa9bc), WIE_CNTL_0/1 (0xa9c2/c3)

### CONST_CONFIG field width

a7xx `CONSTLEN` is 8-bit (mask 0xff). Writing `1024 // 4 = 256` overflows into the ENABLED bit. Correct value: `1024 // 16 = 64`.

## Critical Fixes Discovered

1. **WIE_CNTL_0** (0xa9c2) = 0xccc0cf — required for get_global_id/get_group_id/get_local_id. Without it, all work IDs return 0. On a6xx this was handled by CONST_CONFIG_0 (0xb997) in the NDRANGE consecutive write block.

2. **WIE_CNTL_1** (0xa9c3) — must set `threadsize=THREAD128` + `linearlocalidregid=0xfc` for multi-wave work groups (>64 items). Without it, only 1 wave (64 threads) executes per work group.

3. **THREAD128** — a730's CL driver uses THREAD128 everywhere (SP_CS_CNTL_0 bit 20, WGE_CNTL bit 9, WIE_CNTL_1 bit 8). THREAD64 also works for computation but doesn't match CL.

4. **Signal**: `CP_EVENT_WRITE7` + `CACHE_CLEAN` + `write_enabled` replaces a6xx `CP_EVENT_WRITE` + `CACHE_FLUSH_TS`.

5. **Timestamp**: `CP_EVENT_WRITE7` + `EV_WRITE_ALWAYSON` replaces `CP_REG_TO_MEM` + `CP_ALWAYS_ON_COUNTER`.

6. **Cache flush**: `CACHE_FLUSH7` (atomic write-back + invalidate), `CACHE_CLEAN` (write-back only), `CACHE_INVALIDATE7` (invalidate only). a6xx used `CACHE_FLUSH_TS` + `CACHE_INVALIDATE`.

7. **MODE_CNTL / TPL1_MODE_CNTL** — CL driver sets 0x0a / 0x09 (extra bit 3 vs a6xx's 0x02 / 0x01). Currently hardcoded for a7xx.

## Remaining Bug: LLM NOOPT=0

**Symptom**: With kernel optimizations enabled (NOOPT=0, the default), LLM produces wrong output. NOOPT=1 is correct. Affects both Qwen3 (standard transformer) and Qwen3.5 (SSM/DeltaNet).

**What's verified correct**:
- Binary: CL and llvm-qcom produce byte-identical binaries for the same source ✓
- Registers: all values match CL driver (verified via ioctl hook + indirect buffer parsing) ✓
- Single kernel dispatch: CL and QCOM produce identical results (tested with shared memory + barrier reduction kernel) ✓
- Kernel names: CL and QCOM generate the same kernel schedule (same names) ✓

**What fails**: In multi-kernel pipelines (e.g., `_attention` block with ~19 kernels), the output diverges from CL. `diff = 0.207` on block 0 attention output. Deterministic (same diff every run).

**Key experiment** (`compare_cl_qcom.py`):
```
NOOPT=1 ref:  [-0.686, -0.00344, ...]  sum=-0.8623
QCOM NOOPT=0: [-0.893, -0.0318, ...]   diff=0.207
CL   NOOPT=0: [-0.686, -0.00344, ...]  diff=0.000  ← CL matches ref perfectly
```

**Hypothesis**: Inter-kernel buffer coherency on a730. When tinygrad chains kernels, the buffer written by kernel N may not be fully visible to kernel N+1 under our PM4 command flow. CL's `clEnqueueNDRangeKernel` handles this internally. Individual kernel dispatch is correct because each gets a full `memory_barrier()`.

**Not the cause** (ruled out):
- Compiler/binary differences (identical)
- Register values (matched via ioctl hook)
- THREAD64 vs THREAD128 (tried both)
- Cache flush variants (CACHE_CLEAN, CACHE_FLUSH7, CCHE_INVALIDATE)
- tinygrad kernel cache (tested with CACHELEVEL=0)
- llvm-qcom instance state (tested with fresh instance per compile)
- Kernel optimization options (-cl-opt-disable, -O0)
- Register write ordering (tried CL's order)
- UPDATE_CNTL preamble reset (0x01fffeff)
- branchstack minimum
- max_threads alignment

**Next steps**: Dump per-kernel input/output buffer contents in the multi-kernel pipeline, find the first kernel whose output diverges.

## Device Info

```
chip_id: 0x07030002
gpu_model: Adreno730v3
gmem: 2048 KB
llvm-qcom: /vendor/lib64/libllvm-qcom.so (30 MB)
OpenCL: /system/vendor/lib64/libOpenCL.so
```

## Quick Start

```bash
# Setup on Android/Termux
source extra/cl_android.sh

# CL backend (fully working)
OPENCL_PATH=/system/vendor/lib64/libOpenCL.so DEV=CL python3 tinygrad/llm/cli.py -m qwen3.5:0.8b

# QCOM backend (correct with NOOPT=1)
NOOPT=1 DEV=QCOM python3 tinygrad/llm/cli.py -m qwen3.5:0.8b

# Run tests
DEV=QCOM python3 -m pytest test/test_tiny.py -x -q

# Debug: capture register dumps
OPENCL_PATH=/system/vendor/lib64/libOpenCL.so python3 extra/qcom_gpu_driver/scripts/capture_regs.py
OPENCL_PATH=/system/vendor/lib64/libOpenCL.so python3 extra/qcom_gpu_driver/scripts/compare_cl_qcom.py
```

## Diagnostic Scripts

- `scripts/probe_gpu.py` — detect GPU model, chip_id, llvm-qcom availability
- `scripts/capture_regs.py` — ioctl hook to dump per-kernel register writes (follows indirect buffers)
- `scripts/compare_cl_qcom.py` — compare CL vs QCOM _attention output for LLM block 0
- `scripts/diff_dispatch.py` — compile same kernel source, dispatch via CL and QCOM, compare results
