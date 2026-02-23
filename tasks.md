# CDNA4 MockGPU CI Bounty Tasks

Last updated: 2026-02-23

## Goal

Make `MOCKGPU_ARCH=cdna4` emulator behavior match rdna3/rdna4 baseline (real emulation semantics, not disable/fake pass), and get the AMD CI test bundle green.

## Status Summary

- Overall bounty: `IN PROGRESS`
- Core cdna4 emulation fixes: `DONE` (commit `d7f91f8d1`)
- Full CI bundle in provided workflow: `NOT DONE YET`

## Completed

- [x] Fix CDNA mem op `acc` semantics (`GLOBAL/FLAT/SCRATCH`) to correctly read/write ACCVGPR when `acc=1`.
- [x] Remove dependence on disabling tensor-core packed paths for the key failing ops.
- [x] Stop using buggy default CDNA `V_PK_*` Python fallback; use canonical pcode path by default.
- [x] Add missing pcode bitfield suffix semantics for `.uN/.iN/.bN` (including `.u24/.i24` cases).
- [x] Reproduce and fix major `test_ops` regressions on cdna4:
  - [x] `test_avg_pool2d_asymmetric_padding`
  - [x] `test_conv2d`
  - [x] `test_grouped_conv2d`
- [x] Verify `test/backend/test_ops.py` passes on cdna4 (`350 passed`).

## In Progress

- [ ] Stabilize cdna4 for the full workflow test list under parallel pytest (`-n=auto` equivalent behavior).

## Remaining

- [ ] Fix remaining failures in dtype/fp8-related paths seen in cdna4 runs:
  - [ ] `test/backend/test_dtype.py` (multiple failures)
  - [ ] pcode parse/semantic gaps around fp8 + SDWA-related expressions
- [ ] Run the full test bundle from workflow for cdna4:
  - [ ] `test/backend/test_ops.py`
  - [ ] `test/backend/test_dtype.py`
  - [ ] `test/backend/test_dtype_alu.py`
  - [ ] `test/backend/test_linearizer.py`
  - [ ] `test/backend/test_randomness.py`
  - [ ] `test/backend/test_jit.py`
  - [ ] `test/backend/test_graph.py`
  - [ ] `test/backend/test_multitensor.py`
  - [ ] `test/device/test_hcq.py`
  - [ ] `test/testextra/test_cfg_viz.py`
  - [ ] `test/external/external_test_am.py`
- [ ] Validate both backends in matrix for cdna4:
  - [ ] `backend=amd`
  - [ ] `backend=amdllvm`

## Verification Commands

```bash
AMD=1 MOCKGPU=1 MOCKGPU_ARCH=cdna4 SKIP_SLOW_TEST=1 python -m pytest -q test/backend/test_ops.py
AMD=1 MOCKGPU=1 MOCKGPU_ARCH=cdna4 SKIP_SLOW_TEST=1 python -m pytest -q test/backend/test_dtype.py -x
```

