---
name: MUSA backend bring-up plan
description: Plan and verification gates for adding MTT S4000 (MUSA) backend to tinygrad
type: project
originSessionId: 6b0ec650-60aa-467e-b86a-51c5fd81ef3d
---
MTT S4000 / MUSA tinygrad backend — **working as of 2026-04-22**.

**Why:** Remote SSH machine (connect.sha1.seetacloud.com:28017) has S4000 + MUSA SDK 3.1.0. Local only edits/rsyncs.

**How to apply:**
- Workflow: edit local → rsync to remote → run on remote. Local scratch under `/tmp/1/` only; remote scratch path is unconstrained (pick whatever's convenient on the remote machine).
- Path: clone `ops_cuda.py` (131 lines) → `ops_musa.py` with sed `cu→mu` `CU→MU` `cuda→musa`. Reuse `CUDARenderer`. Compiler shells out to `mcc -x musa -mtgpu --cuda-gpu-arch=mp_31` with diskcache.
- autogen `tinygrad/runtime/autogen/musa.py` from `musa.h` / `musa_runtime_api.h` via clang2py.
- **Gate before llm testing:** `MUSA=1 python -m pytest test/test_tiny.py -v` must be 100% green. No `@skip` to bypass. Failures logged to `/tmp/1/musa_known_issues.md` and root-caused.
- Env var: use `DEV=MUSA` (NOT `MUSA=1` — deprecated).
- Remote has no HF/internet access; use self-contained workloads. Remote Python is py3.10 (too old); use conda env `tgm` (py3.12): `source /root/miniconda3/etc/profile.d/conda.sh && conda activate tgm`.
- Remote tinygrad path: `/root/tinygrad/` (pre-existing session area; rsync without --delete to be safe).
- Key facts: arch=`mp_22`, warpSize=128, 56 SM, 48GB. bf16 type=`__mt_bfloat16` (NOT `nv_bfloat16`). Includes: `musa_fp16.h`/`musa_bf16.h`.
- Compile cmd: `mcc -x musa -mtgpu --offload-arch=mp_22 -O2 --cuda-device-only -o out.fatbin src.mu`. fatbin loads via `muModuleLoadData` directly.
- Out of scope (do NOT attempt): HCQ direct driver, `/dev/mtgpu` ioctl, MTLink multi-GPU, muBLAS/muDNN external calls, Musify CUDA translation.

## Measured perf (2026-04-22, final)
- Qwen3-0.6B-Q8_0: no BEAM 32 tok/s (22 GB/s); JITBEAM=2 ~48 tok/s (34 GB/s, +55%). BEAM warmup ~2 min.
- Qwen3.5-0.8B-Q8_0: 29 tok/s (28 GB/s). **Use Q8 on MUSA**.
- Qwen3.5-0.8B-Q4_K_M: 1.58 tok/s (1.05 GB/s) — **18× slower than Q8 on same model**. Dequant codegen pathology.
- Qwen3.5-9B-Q4_K_M: no BEAM 0.27 tok/s. BEAM=2 safe (preflight) but picks worse kernels than baseline — high-coalescing candidates are resource-rejected.
- GEMV fp16 8192: baseline 73 GB/s (14%), BEAM=2 **169 GB/s (33%)**. muDNN reference 505 GB/s (hardware ceiling).

## BEAM action space + MUSARenderer hardware constants
- Upstream BEAM `actions` LOCAL/GROUPTOP/GROUP amts are NVIDIA-tuned (LOCAL caps at 29 + special case 32). mp_22 warp=128.
- **First attempt (REVERTED)**: added `32/64/128/256/512/1024` to LOCAL, `128/256/512` to GROUPTOP, `32/64/128/256` to GROUP across 3–6 axes each (~57 new candidates). BEAM search time exploded >20× with NO wall-clock benefit — tinygrad BEAM is width-2 greedy and each candidate costs 500 ms of mcc compile.
- **Final kept**:
  - single BEAM action `Opt(LOCAL, axis=0, arg=128)` — canonical mp_22 block start
  - `MUSARenderer.shared_max = 73728` (72 KB, vs CUDARenderer's 48 KB)
  - `MUSARenderer.global_max = (2**31-1, 2**31-1, 2**31-1)` (vs CUDARenderer y/z=65535)
  - `MUSARenderer.local_max = (1024, 1024, 1024)` (vs CUDARenderer z=64)
- These three Renderer constants match musaInfo: `sharedMemPerBlock`, `maxGridSize`, `maxThreadsDim`. Prevents BEAM from pruning candidates that the hardware actually supports.
- **Effect**: BEAM=2 on 8192 fp16 GEMV: 73 → 169 GB/s (2.3×). Search stays ~1 min/shape.
- **Lesson**: tinygrad BEAM is already near-optimal within its IR's expressible space. Adding more action candidates just slows search without unlocking qualitatively new kernels. Hardware-constant overrides are correctness — they don't add perf, they just stop artificial NVIDIA-caps from shrinking the legitimate search space.
- **MATVEC heuristic CAST-unwrap hack was REVERTED**: tinygrad's philosophy is BEAM should find the optimal kernel. Heuristic is only a fallback. Putting fixes at heuristic level is wrong layer.

## Hardware-constant audit (2026-04-22)
musaInfo fields vs tinygrad Renderer usage:

| spec field | value | used? | via |
|---|---|---|---|
| compute cap 2.2 | mp_22 | ✓ | `arch=f"mp_{M}{m}"` in MUSADevice |
| maxThreadsPerBlock | 1024 | ✓ | per-kernel `muFuncGetAttribute` preflight |
| sharedMemPerBlock | 72 KB | ✓ | `MUSARenderer.shared_max=73728` |
| maxGridSize x/y/z | INT_MAX/INT_MAX/INT_MAX | ✓ | `global_max = (2**31-1,)*3` |
| maxThreadsDim x/y/z | 1024/1024/1024 | ✓ | `local_max = (1024,)*3` |
| warpSize | 128 | partial | single BEAM action `LOCAL arg=128` (no `warp_size` attr on Renderer class to set) |
| multiProcessorCount | 56 | ✗ | BEAM doesn't know; ideal global_size is SM_count multiple |
| maxThreadsPerMultiprocessor | 6144 | ✗ | occupancy heuristic absent |
| regsPerBlock | 262144 | ✗ | BEAM_UPCAST_MAX not arch-aware |
| sharedMemPerMultiprocessor | 72 KB | ✗ | same as per-block → 1 full-smem block occupies whole MP; BEAM doesn't weigh occupancy vs tile size |
| l2CacheSize | 24 MB | ✗ | big-tensor tiling heuristic absent |
| concurrentKernels | 1 | ✗ | may mean "no overlap" → MUSAGraph's disabled status is correct anyway |
| totalConstMem | 8192 bytes | ✗ | extremely small, no const-cache usage yet |
| totalGlobalMem | 47.91 GB | ✗ | MAX_BUFFER_SIZE not set |
| isMultiGpuBoard | 1 | ✗ | MTLink untested |

TODO: occupancy-aware BEAM estimator is the real lever for last-mile perf; currently BEAM only times kernels, doesn't model hardware.

## MUDNN=1 fast-path — BUILT AND REMOVED
- User asked for a `MUDNN=1` debug branch to get muDNN-speed matmul via external call.
- Implemented end-to-end: `extra/musa/mudnn_wrapper.cc` (extern "C" over `musa::dnn::MatMul`), `runtime/support/mudnn.py` (ctypes), `ops_musa.py::{detect_matmul, MUDNNMatmulRunner}`, hook in `engine/realize.py::get_runner`.
- Verified: fp16 8192 GEMV wall-clock 127→215 GB/s (+69%), GEMM 8192 5→15.3 TF (3×), first-token warmup 30 s→7.8 s (4×).
- Did NOT help Q8/Q4 LLM decode (dequant-fused matmul is 4-PARAM AST, detector rejects).
- **User then removed it**: violates tinygrad's no-external-lib philosophy. All files deleted. Finding preserved: hardware isn't the bottleneck (muDNN hits 65% HBM peak), tinygrad's 15-33% codegen ceiling is IR-expressiveness limited. Can rebuild this pathway in ~1 day if ever needed for reference.

## Native bfloat16 fix (2-5x on bf16 workloads)
- `tinygrad/device.py:327` `is_dtype_supported(bfloat16, MUSA)` was falling through to default `return False`. That triggered `uop/decompositions.py`'s `pm_float_decomp` to rewrite every bf16 op into `load ushort → bitcast to uint → shift → fp32 math → cast_float_to_bf16` (the software IEEE754 round-trip `cast_float_to_bf16` function in cstyle.py:87-92).
- **Fix**: add `"MUSA"` to the `case "AMD" | "CL" | "PYTHON" | "NULL"` line for bfloat16.
- **Verified**: generated kernel now uses `__mt_bfloat16*` buffers with native arithmetic. GEMV 8192 23→46 GB/s, GEMM 8192 0.96→4.65 TFLOPS.
- **Lesson**: when adding a new tinygrad backend, `is_dtype_supported` in device.py is a mandatory checklist — silently triggers heavy software fallback if wrong.

## Launch preflight (fixes SIGSEGV)
- `MUSAProgram.__init__` queries `MU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK`, `__call__` raises RuntimeError if `prod(local_size)` exceeds it.
- BEAM's existing `try/except RuntimeError` then skips the candidate cleanly.
- Gate: test_tiny.py remains 17/17.

## MUSAGraph status
- Wrote full impl at `tinygrad/runtime/graph/musa.py` (mirror of `cuda.py`).
- **Disabled** — MUSA SDK 3.1.0 returns 801 "operation not supported" on both `muGraphExecKernelNodeSetParams` AND `muGraphExecUpdate`. Only workable path is destroy+re-instantiate per call, which is slower than direct launches (28 vs 32 tok/s on 0.6B). Re-enable when driver gains exec-update support.

## Half/bf16 intrinsics on mp_22 (important)
- `hexp2/hsin` declarations are gated by `__MUSA_ARCH__ >= 800` in musa_fp16.h → undeclared on mp_22.
- `hsqrt/hrcp/hlog2` declared, but only hlog2/hrcp/hsqrt have bodies in musa_fp16_mtgpu.h; `htrunc` decl only, no body.
- Fix: MUSARenderer `code_for_op` routes TRUNC/SIN/LOG2/EXP2/SQRT/RECIPROCAL through float: `((half)op((float)x))` for dtypes.half and `((__mt_bfloat16)op((float)x))` for bfloat16.

## check() pitfall
- `muGetErrorString(status, const char**)` — the second arg is `POINTER(POINTER(c_char))` (autogen type). MUST use `p = ctypes.POINTER(ctypes.c_char)(); byref(p); string_at(p).decode()`. Using `ctypes.c_char_p` directly raises `expected LP_LP_c_char instance instead of pointer to c_char_p` and masks the real launch error.
