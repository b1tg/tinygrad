# CDNA4 MockGPU CI Bounty Tasks

Last updated: 2026-02-24

## Goal

Make `MOCKGPU_ARCH=cdna4` emulator behavior match rdna3/rdna4 baseline (real emulation semantics, not disable/fake pass), and get the AMD CI test bundle green.

## Current Status

- Overall bounty: `IN PROGRESS`
- Confirmed blocker in CI bundle scope: `test/backend/test_ops.py::TestOps::test_padded_conv2d_bs1`
- Current symptom: backward weight gradient mismatch on cdna4 emulation (`backend=amd` and `backend=amdllvm`)

## Landed Fixes (this round)

- [x] Fixed pcode `<>` compare semantics to ordered-not-equal (`_cmp_nan`) for CDNA cmp paths.
- [x] Fixed VOP3P MIX detection to include both `FMA_MIX` and `MAD_MIX`.
- [x] Added missing `V_ACCVGPR_MOV_B32_{E32,E64}` runtime fallback support.
- [x] Added VOP3SD `D1` carry-out recognition/writeback path (per-lane mask write to `sdst`).
- [x] Made `MAD24` Python fallback opt-in (`MOCKGPU_ENABLE_MAD24_FALLBACK`) to avoid silently overriding canonical path.
- [x] Added ACCVGPR to main compiled execution path (`_Ctx.acc`, `raccvgpr_dyn`, `waccvgpr_dyn`) and wired `V_ACCVGPR_{READ,WRITE,MOV_B32}` to compiled semantics.
- [x] Added `acc=1` handling to compiled GLOBAL/FLAT/SCRATCH mem path (read/write through ACCVGPR in non-LDS paths).
- [x] Fixed compiled unaligned dword semantics:
  - Non-byte memory load path now handles unaligned `DWORD/DWORDX*` reconstruction in `test/mockgpu/amd/pcode.py`.
  - Non-byte memory store path now handles unaligned `DWORD` split writes in `test/mockgpu/amd/emu.py`.

## Verified Results

1. `test/backend/test_dtype_alu.py` (+ shard smoke)
   - Command:

```bash
AMD=1 MOCKGPU=1 MOCKGPU_ARCH=cdna4 AMD_LLVM=1 python -m pytest -n=2 -q test/backend/test_dtype_alu.py test/backend/test_multitensor.py::TestMultiTensor::test_shard
```

   - Result: `43 passed, 5 skipped, 1 xfailed`

2. Target blocker reproduction (still failing)
   - Command:

```bash
AMD=1 MOCKGPU=1 MOCKGPU_ARCH=cdna4 AMD_LLVM=1 SKIP_SLOW_TEST=1 python -m pytest test/backend/test_ops.py::TestOps::test_padded_conv2d_bs1 -q
```

   - Result: `FAILED` (backward tensor 1 mismatch)

3. Main-path memory progress check
   - Command:

```bash
AMD=1 MOCKGPU=1 MOCKGPU_ARCH=cdna4 AMD_LLVM=1 SKIP_SLOW_TEST=1 MOCKGPU_DISABLE_CDNA_MEM_FALLBACK=1 python -m pytest test/backend/test_dtype.py::TestEmulatedUInt64DType::test_upcast_to_ops -q
```

   - Result: `1 passed` (previously failed; fixed by unaligned dword load/store semantics)

4. Latest blocker diagnostics (`test_padded_conv2d_bs1`, `-n=2`)
   - Repro command:

```bash
AMD=1 MOCKGPU=1 MOCKGPU_ARCH=cdna4 AMD_LLVM=1 SKIP_SLOW_TEST=1 python -m pytest -n=2 test/backend/test_ops.py::TestOps::test_padded_conv2d_bs1 -q
```

   - Result: still `FAILED` (backward tensor 1 mismatch: `84/108`, max abs diff `7.114667`)
   - Kernel-level finding (fallback counters on failing run):
     - `global_load`: `210`
     - `global_store`: `3`
     - `scratch_mem`: `64`
     - `accvgpr`: `524`
   - Dominant op mix in failing kernel:
     - `V_PK_ADD_F32`, `V_PK_MOV_B32`, `V_ACCVGPR_READ/WRITE` (plus scratch load/store)
   - A/B outcomes:
     - Disabling scratch fallback (`MOCKGPU_DISABLE_CDNA_SCRATCH_FALLBACK=1`) makes error worse (max abs ~`23.08`).
     - Disabling global fallback (`MOCKGPU_DISABLE_CDNA_GLOBAL_FALLBACK=1`) does not improve baseline mismatch.
     - Multiple PK remap experiments (`PK_ADD` fixed lanes, `PK_MOV` direct, opsel remaps, fallback-only variants) did not converge to correct result.

## Next Focus

- [ ] Root-cause `test_padded_conv2d_bs1` (cdna4-only mismatch).
- [ ] Replace PK/ACCVGPR transitional behavior in failing kernel path with architecture-correct semantics (no debug toggles as final behavior).
- [ ] Re-run full CI bundle (`-n=2`) after blocker fix.
