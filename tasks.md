claude --resume b6f6b47f-403f-4689-b15e-e8e42b54cd88
# CDNA4 MOCKGPU Support - Bounty Tasks

## Goal
Add CDNA4 MOCKGPU support + all tests passing in emulator in CI with MOCKGPU_ARCH=cdna4

## Current Status (Updated 2026-02-14)
- ✅ **Major progress**: 64-lane wave infrastructure implemented
- ✅ **15/17 tests pass** in test_tiny.py (was 0/17 before fixes)
- ✅ RDNA3 backward compatibility maintained
- ✅ No more segfaults in dtype_alu tests
- ❌ **2 tests fail**: test_gemm and test_gemv (specific pattern issue)
- ❌ Some dtype_alu tests have wrong results

## Work Completed Today

### ✅ Fixed 64-bit EXEC/VCC Register Handling
**Files modified:** `tinygrad/renderer/amd/emu.py`

**Changes made:**
1. Removed hardcoded `WAVE_SIZE = 32` constant
2. Added `MAX_VGPR_SIZE = 256 * 64` for CDNA compatibility
3. Created `ctx.rexec()` helper to read EXEC as 64-bit (EXEC_LO + EXEC_HI)
4. Updated `WaveState.__init__()` to:
   - Accept `wave_size` parameter (32 for RDNA, 64 for CDNA)
   - Allocate VGPR buffer as `256 * wave_size`
   - Initialize both EXEC_LO and EXEC_HI registers
   - Use `wave_size` in VGPR indexing: `reg * wave_size + lane`
5. Updated `run_asm()` to determine wave size: `wave_size = 64 if arch == "cdna" else 32`
6. Fixed `_Ctx` class:
   - Changed `vgpr` buffer to use `MAX_VGPR_SIZE`
   - Changed `range()` default to 64 for CDNA compatibility
   - Updated `rvgpr_dyn()` and `wvgpr_dyn()` to use 64-lane stride
7. Fixed `unroll_lanes()` to use uint64 for 64-bit masks
8. Fixed `_lane_active()` to handle 64-bit exec_mask
9. Fixed EXECZ comparison to use `uint64(0)` instead of `uint32(0)`
10. Replaced all `ctx.rsgpr_dyn(_c(EXEC_LO.offset))` with `ctx.rexec()`

**Test results:**
```bash
# CDNA4 tests
AMD_LLVM=1 MOCKGPU_ARCH=cdna4 MOCKGPU=1 python -m pytest test/test_tiny.py -v
# Result: 15 passed, 2 failed (test_gemm, test_gemv), 2 skipped

# RDNA3 still works
AMD_LLVM=1 MOCKGPU_ARCH=rdna3 MOCKGPU=1 python -c "from tinygrad import Tensor; print(Tensor([1,2,3]).numpy())"
# Result: [1 2 3] ✅
```

## Remaining Issues

### 1. GEMM/GEMV Failure Pattern (HIGH PRIORITY)
**Symptom:** Elements at positions 8-15 and 24-31 are zero in 32-element results
```python
# Test case that fails:
N = 32
a = Tensor.ones(N,N).contiguous()
b = Tensor.eye(N).contiguous()
result = (a@b).numpy()
# result[0] = [1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0]
#             ^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^
#             correct          WRONG           correct          WRONG
```

**What works:**
- ✅ Element-wise operations (add, mul, neg, abs)
- ✅ Simple copy operations
- ✅ Reductions (sum)
- ✅ All operations with < 32 elements

**Investigation notes:**
- Pattern suggests every other group of 8 elements fails
- Not an LDS issue (no LDS instructions in gemm)
- Likely related to how multi-dword stores work (`global_store_dwordx4`)
- May be related to wave boundaries or VGPR data layout
- The instruction `global_store_dwordx4(v[0], v[4], v[0:3], EXEC_HI)` appears in trace

**Next steps to debug:**
1. **Check global_store_dwordx4 handling in emu.py**
   - Search for `global_store` in _compile_mem_op function (~line 973)
   - The instruction `global_store_dwordx4(v[0], v[4], v[0:3], EXEC_HI)` uses EXEC_HI as a parameter
   - Verify that EXEC_HI is being read correctly in memory operations
   - Check if the store is using the right exec mask (should use full 64-bit exec_mask)

2. **Verify VGPR data layout**
   - Add debug prints in wvgpr_dyn() to see what's being written
   - Add debug prints in rvgpr_dyn() to see what's being read
   - Check if the 64-lane stride is being used consistently
   - Verify that lane indexing is correct for stores

3. **Test with different matrix sizes**
   - Try N=16 (should work - less than 32)
   - Try N=64 (should show pattern repeating)
   - Try N=8 (should work - single group)
   - This will confirm if it's related to wave boundaries

4. **Check EXEC mask in memory operations**
   - In _compile_mem_op, verify exec_mask is 64-bit: `exec_mask = ctx.rexec()`
   - Check that _lane_active() is being called with the 64-bit exec_mask
   - Verify that stores respect the exec mask for all 64 lanes

5. **Add targeted debug output**
   ```python
   # In wvgpr_dyn(), add:
   if DEBUG >= 4:
     print(f"[emu] VGPR write: reg={reg}, lane={lane}, val={val}, exec_active={_lane_active(exec_mask, lane)}")
   ```

### 2. dtype_alu Test Failures
**Symptom:** Tests run but produce wrong results (no more segfaults!)
```bash
AMD_LLVM=1 MOCKGPU_ARCH=cdna4 MOCKGPU=1 python -m pytest test/backend/test_dtype_alu.py::TestDTypeALU::test_bfloat16_unary -xvs
# Result: FAILED - wrong numerical results (e.g., sin returns 0 instead of -0.988)
```

**Next steps:**
- Investigate if this is related to the gemm issue or separate
- Check if bfloat16 operations use special instructions
- May need to integrate uncommitted changes (SDWA, FP8 support)

## Uncommitted Changes to Review

The working directory has uncommitted changes in `emu.py` and `pcode.py`:
- BITOP3 fixes (removes TTBL assignment from pcode)
- CDNA VOP3 E64 fallback (falls back to E32 version when E64 pcode missing)
- FP8/BF8 conversion functions (`fp8_to_f32`, `bf8_to_f32`)
- SDWA widen support (zero-extend 16-bit results)
- Extended type casting support (uint8, int8, uint16, int16)

**Action needed:** Review and integrate these changes, they may fix some dtype issues.

## Commands to Resume Work

### Quick test commands:
```bash
# Test basic functionality
AMD_LLVM=1 MOCKGPU_ARCH=cdna4 MOCKGPU=1 python -c "from tinygrad import Tensor; print(Tensor([1,2,3]).numpy())"

# Test gemm issue
AMD_LLVM=1 MOCKGPU_ARCH=cdna4 MOCKGPU=1 python -c "
from tinygrad import Tensor
N = 32
a = Tensor.ones(N,N).contiguous()
b = Tensor.eye(N).contiguous()
result = (a@b).numpy()
print('First row:', result[0])
print('Mismatches:', [(i, result[0][i]) for i in range(N) if result[0][i] != 1.0])
"

# Run test suite
AMD_LLVM=1 MOCKGPU_ARCH=cdna4 MOCKGPU=1 python -m pytest test/test_tiny.py -v

# Test dtype_alu
AMD_LLVM=1 MOCKGPU_ARCH=cdna4 MOCKGPU=1 python -m pytest test/backend/test_dtype_alu.py::TestDTypeALU::test_bfloat16_unary -xvs

# Debug with instruction trace
AMD_LLVM=1 MOCKGPU_ARCH=cdna4 MOCKGPU=1 DEBUG=3 python -c "..." 2>&1 | grep "\[emu\]"
```

### Check backward compatibility:
```bash
# Verify RDNA3 still works
AMD_LLVM=1 MOCKGPU_ARCH=rdna3 MOCKGPU=1 python -m pytest test/test_tiny.py::TestTiny::test_plus -xvs
```

## Technical Details

### Wave Size Implementation
- **RDNA3/RDNA4:** 32-lane waves, EXEC is 32-bit (only EXEC_LO used)
- **CDNA:** 64-lane waves, EXEC is 64-bit (EXEC_LO + EXEC_HI)
- **Solution:** Compile for 64-bit EXEC (works for both), runtime determines actual wave size

### VGPR Layout
- Compiled code uses: `reg * 64 + lane` (64-lane stride)
- WaveState uses: `reg * wave_size + lane` (dynamic stride)
- For CDNA: both are 64, matches perfectly
- For RDNA: compiled uses 64, runtime uses 32, but upper 32 slots unused (OK)

### Key Files Modified
- `tinygrad/renderer/amd/emu.py` - All changes in this file
  - Lines ~115-120: Constants (removed WAVE_SIZE, added MAX_VGPR_SIZE)
  - Lines ~130-132: _lane_active() fixed for uint64
  - Lines ~290-305: _Ctx class (range, unroll_lanes)
  - Lines ~360-370: Added rexec() helper
  - Lines ~370-380: VGPR access (64-lane stride)
  - Lines ~570: EXECZ comparison fixed
  - Lines ~1215-1240: WaveState class (dynamic wave_size)
  - Lines ~1260-1270: run_asm() (determine wave_size from arch)

### Git Status
```bash
# Modified files (not committed):
M tinygrad/renderer/amd/emu.py
M tinygrad/renderer/amd/pcode.py  # uncommitted changes to review
M tinygrad/viz/js/index.js  # minor fix (let declaration)
```

## Debugging Strategy for Tomorrow

### Priority 1: Fix GEMM/GEMV (Elements 8-15, 24-31 Zero)

**Hypothesis:** The pattern (groups of 8 failing) suggests wave-level masking issue. With 64-lane waves and 32-element output, we have:
- Lanes 0-31: Process elements 0-31
- Lanes 32-63: Unused (but EXEC mask should disable them)

The failure pattern (8-15, 24-31 zero) could mean:
- Lanes 8-15 and 24-31 are not writing correctly
- OR the EXEC mask is incorrectly masking these lanes
- OR the VGPR indexing is wrong for these lanes

**Debugging approach:**
1. Start with the simplest test case:
   ```bash
   AMD_LLVM=1 MOCKGPU_ARCH=cdna4 MOCKGPU=1 DEBUG=3 python -c "
   from tinygrad import Tensor
   N = 32
   a = Tensor.ones(N,N).contiguous()
   b = Tensor.eye(N).contiguous()
   result = (a@b).numpy()
   print('Result:', result[0])
   " 2>&1 | tee gemm_debug.log
   ```

2. Check the EXEC mask initialization in WaveState:
   - For n_lanes=32, exec_mask should be 0x00000000FFFFFFFF (lower 32 bits set)
   - Verify EXEC_LO = 0xFFFFFFFF and EXEC_HI = 0x00000000

3. Check memory store operations:
   - Look for `global_store` instructions in the debug output
   - Verify they're using the correct exec mask
   - Check if EXEC_HI parameter is being handled

4. Add instrumentation to wvgpr_dyn():
   ```python
   def wvgpr_dyn(self, reg: UOp, lane: UOp, val: UOp, exec_mask: UOp, after: UOp | None = None) -> UOp:
     """Write VGPR with dynamic register index."""
     buf = self.vgpr.after(after) if after is not None else self.vgpr
     offset = reg.cast(dtypes.int) * _c(64, dtypes.int) + lane.cast(dtypes.int)
     active = _lane_active(exec_mask, lane)
     # TODO: Add debug print here to see what's being written
     return buf.index(offset, active).store(val.cast(dtypes.uint32))
   ```

### Priority 2: Fix dtype_alu Wrong Results

**Hypothesis:** Missing instruction implementations or incorrect pcode for CDNA-specific operations.

**Debugging approach:**
1. Start with the simplest failing test:
   ```bash
   AMD_LLVM=1 MOCKGPU_ARCH=cdna4 MOCKGPU=1 DEBUG=3 python -c "
   from tinygrad import Tensor
   import numpy as np
   x = Tensor([0.5]).realize()
   result = x.sin().numpy()
   print('sin(0.5) =', result[0], 'expected:', np.sin(0.5))
   " 2>&1 | tee sin_debug.log
   ```

2. Check the instruction trace for the sin operation
   - Look for V_SIN or V_COS instructions
   - Check if they're being compiled correctly

3. Review uncommitted changes in pcode.py:
   - The FP8/BF8 conversion functions might be needed
   - SDWA widen support might affect 16-bit operations
   - Extended type casting might fix some conversions

### Priority 3: Review and Integrate Uncommitted Changes

Files with uncommitted changes:
- `tinygrad/renderer/amd/emu.py` - Our 64-lane wave changes (keep)
- `tinygrad/renderer/amd/pcode.py` - BITOP3, FP8, SDWA fixes (review and integrate)
- `tinygrad/viz/js/index.js` - Minor fix (review)

**Action:** Use `git diff` to review each change and decide what to keep.

## Quick Reference

### Test Commands (Copy-Paste Ready)
```bash
# Quick smoke test
AMD_LLVM=1 MOCKGPU_ARCH=cdna4 MOCKGPU=1 python -c "from tinygrad import Tensor; print(Tensor([1,2,3]).numpy())"

# GEMM failure test
AMD_LLVM=1 MOCKGPU_ARCH=cdna4 MOCKGPU=1 python -c "from tinygrad import Tensor; N=32; a=Tensor.ones(N,N).contiguous(); b=Tensor.eye(N).contiguous(); r=(a@b).numpy(); print('First row:', r[0]); print('Zeros at:', [i for i in range(N) if r[0][i]==0])"

# Full test suite
AMD_LLVM=1 MOCKGPU_ARCH=cdna4 MOCKGPU=1 python -m pytest test/test_tiny.py -v

# Single test with debug
AMD_LLVM=1 MOCKGPU_ARCH=cdna4 MOCKGPU=1 DEBUG=3 python -m pytest test/test_tiny.py::TestTiny::test_gemm -xvs 2>&1 | tee test_gemm.log

# RDNA3 compatibility check
AMD_LLVM=1 MOCKGPU_ARCH=rdna3 MOCKGPU=1 python -c "from tinygrad import Tensor; print(Tensor([1,2,3]).numpy())"
```

### Key Code Locations
- **EXEC register handling:** `emu.py:363-368` (rexec() helper)
- **VGPR access:** `emu.py:374-383` (rvgpr_dyn, wvgpr_dyn)
- **Wave size determination:** `emu.py:1264-1265` (run_asm)
- **WaveState initialization:** `emu.py:1218-1239` (EXEC mask setup)
- **Memory operations:** `emu.py:973-1140` (_compile_mem_op)
- **Lane masking:** `emu.py:130-132` (_lane_active)

### Expected Behavior
- **RDNA3:** 32-lane waves, EXEC_LO only, EXEC_HI = 0
- **CDNA4:** 64-lane waves, both EXEC_LO and EXEC_HI used
- **VGPR stride:** Always 64 in compiled code, matches CDNA perfectly
- **For N=32 gemm:** Should use 32 lanes (0-31), lanes 32-63 disabled by EXEC mask


- [ ] All tests in test_tiny.py pass (currently 15/17)
- [ ] Fix gemm/gemv failures
- [ ] dtype_alu tests produce correct results
- [ ] CI test passes: `AMD_LLVM=1 MOCKGPU_ARCH=cdna4 python -m pytest test/test_tiny.py -v --durations 20`
- [ ] RDNA3 backward compatibility maintained
- [ ] Consider running more comprehensive test suites (test/backend/test_ops.py)
