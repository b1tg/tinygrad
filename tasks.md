# CDNA4 MockGPU CI Bounty Tasks

Last updated: 2026-02-23

## Goal

Make `MOCKGPU_ARCH=cdna4` emulator behavior match rdna3/rdna4 baseline (real emulation semantics, not disable/fake pass), and get the AMD CI test bundle green.

## Status Summary

- Overall bounty: `IN PROGRESS`
- Core cdna4 emulation fixes: `DONE` (commit `d7f91f8d1`)
- Full CI bundle in provided workflow: `NOT DONE YET`
- Latest full cdna4 run (`-n=2`, `backend=amd`): `27 failed, 842 passed, 139 skipped, 5 xfailed, 29 errors` (27m28s)
- Current primary blocker: emu compile failure on `v_exp_f16_e32(...)` causing `test_hcq` errors and many `test_cfg_viz` failures

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

- [ ] Stabilize cdna4 for the full workflow test list under parallel pytest (`-n=2`/`-n=auto` behavior).
- [ ] Fix emu decode/compile path for `v_exp_f16_e32` on cdna4/rdna4 emu arch translation path.

## Remaining

- [ ] Fix remaining failing tests from latest full run.

### Failing Tests (latest full run, `backend=amd`, `-n=2`)

- [ ] DType / fp paths:
  - [ ] `test/backend/test_dtype_alu.py::TestDTypeALU::test_emulated_fp8e4m3_unary`
  - [ ] `test/backend/test_dtype.py::TestDoubleDType::test_float64_increased_precision`
  - [ ] `test/backend/test_dtype_alu.py::TestDTypeALU::test_emulated_fp8e5m2_unary`
  - [ ] `test/backend/test_dtype_alu.py::TestDTypeALU::test_float16_unary`
- [ ] Randomness:
  - [ ] `test/backend/test_randomness.py::TestRandomness::test_randn_finite`
  - [ ] `test/backend/test_randomness.py::TestRandomness::test_scaled_uniform`
  - [ ] `test/backend/test_randomness.py::TestRandomness::test_threefry_against_reference_full`
- [ ] MultiTensor:
  - [ ] `test/backend/test_multitensor.py::TestMultiTensor::test_matmul_shard_0_0`
  - [ ] `test/backend/test_multitensor.py::TestMultiTensor::test_matmul_shard_0_1`
  - [ ] `test/backend/test_multitensor.py::TestMultiTensor::test_matmul_shard_1_0`
  - [ ] `test/backend/test_multitensor.py::TestMultiTensor::test_matmul_shard_1_1`
  - [ ] `test/backend/test_multitensor.py::TestMultiTensor::test_matmul_shard_W_0`
  - [ ] `test/backend/test_multitensor.py::TestMultiTensor::test_matmul_shard_W_1`
  - [ ] `test/backend/test_multitensor.py::TestMultiTensor::test_matmul_shard_X_0`
  - [ ] `test/backend/test_multitensor.py::TestMultiTensor::test_matmul_shard_X_1`
  - [ ] `test/backend/test_multitensor.py::TestMultiTensor::test_matmul_shard_none`
  - [ ] `test/backend/test_multitensor.py::TestMultiTransformer::test_transformer`
- [ ] CFG viz (all fail from same emu compile error family):
  - [ ] `test/testextra/test_cfg_viz.py::TestCfg::test_colored_blocks`
  - [ ] `test/testextra/test_cfg_viz.py::TestCfg::test_diamond`
  - [ ] `test/testextra/test_cfg_viz.py::TestCfg::test_hit_count`
  - [ ] `test/testextra/test_cfg_viz.py::TestCfg::test_jump_back_to_end`
  - [ ] `test/testextra/test_cfg_viz.py::TestCfg::test_loop`
  - [ ] `test/testextra/test_cfg_viz.py::TestCfg::test_loop_branch`
  - [ ] `test/testextra/test_cfg_viz.py::TestCfg::test_loop_break`
  - [ ] `test/testextra/test_cfg_viz.py::TestCfg::test_ping_pong`
  - [ ] `test/testextra/test_cfg_viz.py::TestCfg::test_simple`
  - [ ] `test/testextra/test_cfg_viz.py::TestCfg::test_switch`
- [ ] HCQ (29 errors, all blocked by emu compile failure path):
  - [ ] `test/device/test_hcq.py::TestHCQ::*` (29 errors)

### Workflow Coverage Tracking

- [x] Run full workflow test bundle for cdna4 with `backend=amd` and parallel pytest (`-n=2`).
- [ ] Re-run full workflow test bundle for cdna4 with `backend=amd` after fixes (target: zero failures/errors).
- [ ] Run full workflow test bundle for cdna4 with `backend=amdllvm`.

## Verification Commands

```bash
AMD=1 MOCKGPU=1 MOCKGPU_ARCH=cdna4 SKIP_SLOW_TEST=1 python -m pytest -q test/backend/test_ops.py
AMD=1 MOCKGPU=1 MOCKGPU_ARCH=cdna4 SKIP_SLOW_TEST=1 python -m pytest -q test/backend/test_dtype.py -x
AMD=1 MOCKGPU=1 MOCKGPU_ARCH=cdna4 SKIP_SLOW_TEST=1 AMD_LLVM=0 python -m pytest -n=2 test/backend/test_ops.py test/backend/test_dtype.py test/backend/test_dtype_alu.py test/backend/test_linearizer.py test/backend/test_randomness.py test/backend/test_jit.py test/backend/test_graph.py test/backend/test_multitensor.py test/device/test_hcq.py test/testextra/test_cfg_viz.py test/external/external_test_am.py --durations=20
```
