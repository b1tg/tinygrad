---
name: Ascend NPU backend on CANN 8.1.RC1 (910B2) — working path
description: End-to-end working recipe for tinygrad Ascend backend on Atlas 910B2. Skeleton + renderer live in tinygrad/runtime/ops_ascend.py + AscendRenderer.
type: project
originSessionId: dbfaa30a-1994-4c0b-ae91-3d6e79382b76
---
Current state: `DEV=ASCEND python -m pytest test/test_tiny.py` passes **16/17** (only test_mnist fails, UB size > 192KB for Conv2d kernels).

## Working recipe (910B2 / CANN 8.1.RC1)

1. **Kernel name must end with `_2147483647`** (INT32_MAX = AscendC generic tiling key). Any other name causes `rtRegisterAllKernel` to return 107000.
2. **Body must use `LocalTensor::GetValue(i)` / `SetValue(i, v)` — NOT `__ubuf__ float*` pointer deref.** Raw UB-pointer scalar access via `*(ub+i)` crashes the vector core (507035 aivec exception) even though it looks like it should work. The `_ascend_rewrite` PatternMatcher in `cstyle.py` overrides LOAD/STORE for PARAM buffers to emit `{name}_l.Get/SetValue(idx)`.
3. **All GM↔UB traffic via `DataCopy` in TPipe + TQue framework.** AIV scalar unit cannot directly access GM. Renderer auto-wraps the user body with TPipe boilerplate: per-buffer `TQue<VECIN|VECOUT>` + `pipe.InitBuffer(q, 1, N*sizeof(T))` + `AllocTensor` + `DataCopy(l, g, N)` → body → `EnQue/DeQue` → `DataCopy(g, l, N)`. TILE is buffer size padded to multiple of 8 floats (32-byte DataCopy alignment).
4. **Register/launch via `libascendc_runtime.a`'s `RegisterAscendBinary` + `LaunchAscendKernel`**, NOT `aclrtBinaryLoad` + `aclrtLaunchKernel`. The ACL-level binary load path does NOT work for bisheng-compiled kernels — it's for AOT-precompiled ops only. The runtime ABI:
   - `RegisterAscendBinary(bytes, sz, type=1, &handle)` — type=1 means AIV vector core.
   - `LaunchAscendKernel(handle, tilingKey=0x7FFFFFFF, blockDim, &argsBuf, argSize, stream)`.
   A tiny C wrapper (`asc_wrap.cc` built in `~/.cache/tinygrad/ascend/asc_wrap.so`) links `libascendc_runtime.a` + `libascendcl` + `libruntime` + `libmsprofiler` + `liberror_manager` — then ctypes calls into it.
5. **Launch args are a packed struct, not `void**`**. Layout: `[void* arg0][void* arg1]...[int arg_k (4B + 4B pad)]...[void* __ascendc_overflow]`. The trailing `__ascendc_overflow` slot is a device-mem pointer to an 8-byte overflow-status buffer (allocate once per program with `aclrtMalloc`). Missing this slot = incorrect args reading, wrong results.
6. **Compile + link chain**: `bisheng -c --cce-aicore-arch=dav-c220-vec --cce-aicore-only -fcce-kernel-type-section --cce-auto-sync -O2 -std=c++17 -I${TIKCFW}(+interface+impl) -I${ASCEND_HOME}/include -DTILING_KEY_VAR=0 -x cce k.cce -o k.o` → `ld.lld -r -Ttext=0 k.o -o k_r.o` → `ld.lld -Ttext=0 k_r.o -static -o k_final.o`. The 2-pass ld.lld is required; `-fcce-kernel-type-section` emits the `.ascend.meta.<name>` NOTE section that AscendC runtime needs.
7. **No scalar exp/log/sin on AIV** — tinygrad's `TRANSCENDENTAL` pattern rewrites them to polynomial approximations. Remove these ops from `code_for_op` to let the default fallback kick in.
8. **`supports_float4 = False`**. AIV doesn't have the vectorized types tinygrad expects.
9. **Bool→float cast** must be rewritten to ternary `(b ? 1.0f : 0.0f)` — bisheng aicore rejects `(float)(bool_expr)`.
10. **`#define INFINITY/NAN`** via `__builtin_bit_cast(float, (int)0x7f800000)` / `0x7fc00000` — `<math.h>` doesn't compile inside `[aicore]` context.

## Known limitation

`test_mnist` fails without BEAM — Conv2d generates kernels with buffers up to 36864 floats (144KB each). Multiple such buffers in one kernel exceed the 192KB UB budget per AIV core.

**Workaround**: `BEAM=1 IGNORE_BEAM_CACHE=1` — tinygrad finds a smaller tiled kernel, test passes (~10min BEAM search).

**Real fix** (not done): implement block-level tiling + per-block DataCopy in the renderer. Either:
- Enable `has_threads = True` + `global_max = (24,1,1)` to trigger tinygrad's THREAD opt — this successfully hoists outer loops to block level (seen kernel `r_20_5_8_4_4_32_5_5` → 8 threads). BUT: each block still DataCopies the FULL buffer to UB, so total UB stays unchanged, AND each block overwrites the output with its partial compute (mutating the full UB) → final GM has garbage except for the last-writing block's slice.
- To correctly benefit from THREAD, the renderer must:
  1. Allocate only `size/blocks` UB per sliceable buffer
  2. Rewrite access indices: `core_id * stride + i` → `i` (body thinks it's operating on local tile)
  3. DataCopy GM↔UB with per-block offset (only read/write block's slice)
  Which requires static index analysis of the UOp graph — non-trivial codegen work, better paired with tinygrad's own tile-aware opt.

## Key files

- `tinygrad/runtime/ops_ascend.py`: Device, Compiler, Program, Allocator. Also builds `asc_wrap.so` (~/.cache/tinygrad/ascend/) on first use.
- `tinygrad/renderer/cstyle.py::AscendRenderer`: TPipe wrapper codegen + `_ascend_rewrite` PatternMatcher.
- `tinygrad/device.py::ALL_DEVICES`: has `"ASCEND"` appended.

## Hardware/env

- 910B2 (dav-c220-vec arch), 64GB HBM, CANN 8.1.RC1 at `/usr/local/Ascend/ascend-toolkit/latest`.
- Remote env: `source /usr/local/Ascend/ascend-toolkit/set_env.sh` + `LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64:$LD_LIBRARY_PATH`.
- Python 3.11 env at `/root/miniconda3/envs/tg/` (3.10 lacks `typing.Self`).

## Artifacts collected

`/tmp/1/ascend_artifacts/`: ACL headers, bisheng help, `bisheng_intf.cmake`, working ref_abs.o, our test kernels (add.cce/add_simple.cce/add_tpipe_tk.cce), `util_scripts/` (extract_host_stub.py etc), `asc_wrap.cc`.
