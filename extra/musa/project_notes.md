---
name: MUSA backend bring-up plan
description: Plan and verification gates for adding MTT S4000 (MUSA) backend to tinygrad
type: project
originSessionId: 6b0ec650-60aa-467e-b86a-51c5fd81ef3d
---
MTT S4000 / MUSA tinygrad backend — **working as of 2026-04-18**.

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

## Measured perf (2026-04-18)
- Qwen3-0.6B-Q8_0: no BEAM 32 tok/s (22 GB/s); JITBEAM=2 48 tok/s (34 GB/s, +55%). BEAM warmup adds ~120s cold compile.
- Qwen3.5-9B-Q4_K_M: no BEAM 0.27 tok/s (1.49 GB/s). BEAM=2 now safe (preflight converts SIGSEGV → RuntimeError) but gives 0.18 tok/s — the rejected candidates were the good ones, leaving BEAM's search space strictly worse than baseline. Hot kernel root cause: default opt picks 32 threads/block with lidx-indexed byte-stride loads, killing coalescing on Q4_K_M dequant. Per-kernel BW <3 GB/s on a ~500 GB/s card.

## BEAM action space for mp_22 (codegen/opt/search.py)
- `actions` LOCAL/GROUPTOP/GROUP amts were NVIDIA-tuned (LOCAL caps at 29 with only one `32` special case). mp_22 warp=128 wants larger blocks.
- **Fix**: add `32/64/128/256` to `LOCAL/GROUPTOP/GROUP` amts. BEAM now picks `GROUP(0, 128)` on GEMV when available.
- **Effect**: marginal wall-clock gain (BEAM search already hits its IR-ceiling before these help much). Kept because it's the correct direction structurally — NVIDIA-only defaults were wrong.
- **MATVEC heuristic CAST-unwrap hack was REVERTED**: user correctly pointed out tinygrad's design philosophy is that BEAM should find optimal kernels. Heuristic is only a fallback/baseline. Putting fixes in heuristic is not the right layer.

## MUDNN=1 debug fast-path (engine/realize.py + runtime/ops_musa.py + extra/musa/mudnn_wrapper.cc)
- User request: "add a MUDNN=1 debug branch to get LLM speed up; we'll replace it with native codegen later."
- Wrote `extra/musa/mudnn_wrapper.cc` — extern "C" shim over `musa::dnn::MatMul`. Built on first use via `runtime/support/mudnn.py::_build_wrapper`.
- `ops_musa.py::detect_matmul(ast)` scans AST for the matmul pattern (3 PARAMs, 1 REDUCE ADD, MUL of 2 INDEX with CAST unwrap, no random-gen ops). Returns (dtype, M, N, K, a_arg, b_arg, out_arg) or None.
- `ops_musa.py::MUDNNMatmulRunner` — implements Runner interface, direct muDNN call.
- `engine/realize.py::get_runner` — when `MUDNN=1` env var and device starts with "MUSA", checks detect_matmul; if matches, returns MUDNNMatmulRunner instead of CompiledRunner.
- **Verified effect** (wall-clock bench): fp16 8192 GEMV 127→215 GB/s (+69%), fp16 8192 GEMM 5→15.3 TFLOPS (3x), warmup 30s→7.8s (4x).
- **Does not help GGUF Q8/Q4 LLM decode**: dequant fused into matmul AST has 4 PARAMs (int8_weight + scale + input + out) and CAST/MUL chains that detect_matmul rejects. Steady-state LLM decode unchanged (still 29 tok/s on Qwen3.5-0.8B-Q8).
- **TODO for Q-format**: extend detect_matmul to recognize the dequant-matmul pattern.

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
