# CDNA4 MOCKGPU Support - Bounty Tasks

## Goal
Add CDNA4 MOCKGPU support + all tests passing in emulator in CI with MOCKGPU_ARCH=cdna4

## Current Status (Updated 2026-02-14 night)
- **17/17 tests pass** in test_tiny.py on CDNA4
- RDNA3 backward compatibility maintained (16/16 pass, mnist_backward segfault is pre-existing)
- BITOP3 instruction support added (CDNA-specific)
- Partial test_ops.py comparison (50/416 tests): CDNA4 37P/12F/1S vs RDNA3 28P/17F/5S
  - CDNA4 actually has FEWER failures than RDNA3 in the first 50 tests
  - Many failures are pre-existing on both architectures
- Only `tinygrad/renderer/amd/emu.py` modified

## Changes Made

### 1. VCC 64-bit Fix (GEMM/GEMV root cause)
VCC was read/written as 32-bit only. CDNA4 uses 64-lane waves needing all 64 VCC bits.
VOPC wrote only VCC_LO, and v_cndmask read only VCC_LO → lanes 32-63 always saw VCC=0.

**What was changed (working for both CDNA4 and RDNA3):**
- Import VCC_HI, EXEC_HI at top (line 68)
- New helpers: `rvcc()`, `wvcc()`, `wexec()`, `rsgpr_pair()`, `wsgpr_pair()`
- `_set_lane_bit` uses uint64
- `_compile_vopc` writes VCC/EXEC as 64-bit pairs
- `_compile_vop3` v_cndmask reads VCC as 64-bit pair via `rsgpr_pair`
- `_compile_vopd` reads VCC as 64-bit via `rvcc()`
- `compile_vop_pcode` VCC lane stores and mask writes via `wsgpr_pair`

**What was NOT changed (kept 32-bit to avoid RDNA3 segfault):**
- `compile_sop_pcode` VCC/EXEC reads
- `scalar_stores` EXEC/VCC writes
- `compile_lane_pcode` EXEC read
- `_compile_sopp` VCC read
- `compile_vop_pcode` VCC read (line ~500)
- `_compile_vop3sd` VCC read/write — THIS was the RDNA3 segfault culprit
  (VOP3SD's `vcc_in_off` is dynamic, `rsgpr_pair` reads `reg+1` causing OOB)

### 2. BITOP3 Instruction Support (CDNA-specific)
CDNA pcode for `v_bitop3_b32` references `INST.OMOD/ABS/NEG` to build a truth table.
Fix: pre-compute TTBL from instruction fields, strip TTBL assignment from pcode, skip src mods.

**Changes:**
- `_pcode_fixes` dict: strip `TTBL = { INST... }` line for V_BITOP3_B32/B16
- `_compile_vop3`: detect BITOP3, skip src mod application, compute TTBL dynamically:
  `TTBL = (omod << 6) | (abs << 3) | neg`

## Remaining Work

### Priority 1: Full test_ops.py Comparison
Need to complete the subprocess-isolated test run to identify CDNA4-specific failures.
Script at `/tmp/run_tests.py` works but is slow (~10min per 50 tests).

```bash
# Run comparison (takes ~80min per arch with 416 tests)
MOCKGPU_ARCH=cdna4 python /tmp/run_tests.py 2>&1 | tee /tmp/cdna4_results.log
MOCKGPU_ARCH=rdna3 python /tmp/run_tests.py 2>&1 | tee /tmp/rdna3_results.log

# Compare results
diff <(sort /tmp/test_ops_cdna4.txt) <(sort /tmp/test_ops_rdna3.txt)
```

### Priority 2: VOP3SD 64-bit VCC for CDNA4
Still uses 32-bit VCC. May cause issues with carry instructions (V_ADD_CO_U32, etc.)
on 64-lane waves. Fix needs bounds-guarding the `reg+1` read.

### Priority 3: dtype_alu Failures
`test_bfloat16_unary` (sin returns 0) fails on BOTH RDNA3 and CDNA4 — pre-existing.

## Commands to Resume

```bash
# Quick smoke test
AMD_LLVM=1 MOCKGPU_ARCH=cdna4 MOCKGPU=1 python -c "from tinygrad import Tensor; print(Tensor([1,2,3]).numpy())"

# Full test_tiny (should be 17 passed, 2 skipped)
AMD_LLVM=1 MOCKGPU_ARCH=cdna4 MOCKGPU=1 python -m pytest test/test_tiny.py -v

# RDNA3 compat (skip pre-existing segfault)
AMD_LLVM=1 MOCKGPU_ARCH=rdna3 MOCKGPU=1 python -m pytest test/test_tiny.py -v -k "not mnist_backward"

# Subprocess-isolated test_ops comparison
MOCKGPU_ARCH=cdna4 python /tmp/run_tests.py
MOCKGPU_ARCH=rdna3 python /tmp/run_tests.py

# Check git diff
git diff tinygrad/renderer/amd/emu.py
```

## Checklist
- [x] All tests in test_tiny.py pass on CDNA4 (17/17)
- [x] Fix gemm/gemv failures (VCC 64-bit fix)
- [x] RDNA3 backward compatibility (16/16 pass, mnist_backward pre-existing)
- [x] BITOP3 instruction support
- [ ] Complete test_ops.py comparison (partial: CDNA4 looks similar or better than RDNA3)
- [ ] VOP3SD 64-bit VCC handling without breaking RDNA3
- [ ] dtype_alu test_bfloat16_unary (pre-existing on both arches)
- [ ] CI test: `AMD_LLVM=1 MOCKGPU_ARCH=cdna4 python -m pytest test/test_tiny.py -v --durations 20`
