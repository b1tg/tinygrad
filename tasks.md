# CDNA4 MockGPU CI Bounty Tasks

Last updated: 2026-02-23

## Goal

Make `MOCKGPU_ARCH=cdna4` emulator behavior match rdna3/rdna4 baseline (real emulation semantics, not disable/fake pass), and get the AMD CI test bundle green.

## Final Status

- Overall bounty: `DONE`
- Full workflow test bundle (`backend=amd`, `backend=amdllvm`, `-n=2`): `PASS`
- Remaining blockers from previous run (`test_hcq`, `test_cfg_viz`, dtype/dtype_alu): `RESOLVED`

## Key Fixes Landed

- [x] Correct CDNA mem op `acc` semantics for `GLOBAL/FLAT/SCRATCH` paths.
- [x] Stop relying on tensor-core disable shortcuts; keep real emulation behavior.
- [x] Default to canonical pcode handling for problematic `V_PK_*` paths.
- [x] Implement missing pcode bitfield suffix behavior (`.uN/.iN/.bN`, including `.u24/.i24`).
- [x] Fix pcode branch-merge type inference for branch-local vars (`u32`-neutral base cast to inferred dtype).
- [x] Fix packed D16 load merge semantics in Python mem fallbacks so HI/LO loads preserve the other 16-bit half (GLOBAL/FLAT/SCRATCH).

## Verification Runs

1. `AMD_LLVM=1` full bundle (`-n=2`)
   - Result: `895 passed, 142 skipped, 5 xfailed` in `1797.37s`
   - Command:

```bash
AMD=1 MOCKGPU=1 MOCKGPU_ARCH=cdna4 SKIP_SLOW_TEST=1 AMD_LLVM=1 python -m pytest -n=2 test/backend/test_ops.py test/backend/test_dtype.py test/backend/test_dtype_alu.py test/backend/test_linearizer.py test/backend/test_randomness.py test/backend/test_jit.py test/backend/test_graph.py test/backend/test_multitensor.py test/device/test_hcq.py test/testextra/test_cfg_viz.py test/external/external_test_am.py --durations=20
```

2. `AMD_LLVM=0` full bundle (`-n=2`)
   - Result: `896 passed, 141 skipped, 5 xfailed` in `1659.07s`
   - Command:

```bash
AMD=1 MOCKGPU=1 MOCKGPU_ARCH=cdna4 SKIP_SLOW_TEST=1 AMD_LLVM=0 python -m pytest -n=2 test/backend/test_ops.py test/backend/test_dtype.py test/backend/test_dtype_alu.py test/backend/test_linearizer.py test/backend/test_randomness.py test/backend/test_jit.py test/backend/test_graph.py test/backend/test_multitensor.py test/device/test_hcq.py test/testextra/test_cfg_viz.py test/external/external_test_am.py --durations=20
```

## Remaining

- [x] None for this bounty scope.
