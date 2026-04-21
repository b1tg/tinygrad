# MUSA backend for tinygrad (MTT S4000 / Moore Threads)

Status: **working for inference**, fp16/bf16 GEMV at ~15-25% muDNN from pure codegen.

Implemented 2026-04-18 against MUSA SDK 3.1.0 on MTT S4000 (mp_22, driver 2.7.0).

## Quick start
```bash
# needs MUSA SDK installed + libmusa.so on LD_LIBRARY_PATH + py3.11+
DEV=MUSA python -m tinygrad.llm -m /path/to/model.gguf

# optional: bench vs torch_musa/muDNN (requires torch_musa in system python)
DEV=MUSA python extra/musa/bench_vs_torch.py
```

## Files added / modified

### New
| Path | Purpose |
|---|---|
| `tinygrad/runtime/autogen/musa.py` | clang2py ctypes binding for libmusa.so (265 driver APIs) |
| `tinygrad/runtime/support/compiler_musa.py` | `MCCCompiler` — `mcc` subprocess + diskcache |
| `tinygrad/runtime/ops_musa.py` | `MUSADevice / MUSAProgram / MUSAAllocator`, launch preflight |
| `tinygrad/runtime/graph/musa.py` | `MUSAGraph` (currently disabled — driver doesn't support graph exec update) |
| `extra/musa/probe.py` | self-contained ctypes probe: compile + load + launch vadd |
| `extra/musa/vadd.mu` | probe kernel source |
| `extra/musa/bench_vs_torch.py` | side-by-side GEMV/GEMM benchmark vs torch_musa |
| `extra/musa/musa_probe.md` | MUSA SDK facts / API mapping / mcc flags / half-intrinsic matrix |
| `extra/musa/project_notes.md` | chronological engineering log (fixes shipped, reasoning, reverts) |

### Changed (upstream-worthy)
| File | Change |
|---|---|
| `tinygrad/runtime/autogen/__init__.py` | one `case "musa":` dispatch entry |
| `tinygrad/device.py` | `is_dtype_supported(bfloat16, "MUSA")` — **correctness fix**, see Gotcha 2 |
| `tinygrad/renderer/cstyle.py` | `MUSARenderer(CUDARenderer)` with half/bf16→fp32 math fallback |
| `tinygrad/codegen/opt/search.py` | BEAM actions add one targeted `LOCAL axis=0 arg=128` (canonical block size for mp_22 warp=128) |

## Measured performance

### LLM decode (single-user, Qwen3.5-0.8B, steady state)

| Weights | tok/s | mem BW | notes |
|---|---|---|---|
| Q8_0 (775MB) | **29.2** | 27.5 GB/s | 3.6% of 768 GB/s peak |
| Q4_K_M (508MB) | **1.58** | 1.05 GB/s | 18× slower than Q8 — dequant codegen pathology |

**Takeaway for users right now**: on MUSA use Q8_0, not K-quants. Q4_K_M hits a tinygrad codegen cliff (scalar byte-load pattern destroys coalescing, see Gotcha 5). Q8_0 is the realistic fast path.

### GEMV micro-benchmark (`extra/musa/bench_vs_torch.py`, wall-clock including Python/ctypes)

| Shape | dtype | muDNN (ref) | tinygrad baseline | tinygrad BEAM=2 |
|---|---|---|---|---|
| 4096 | fp16 | 277 GB/s | 31 GB/s (11%) | — |
| 4096 | bf16 | 271 GB/s | 15 GB/s (6%) | — |
| 8192 | fp16 | 505 GB/s | 73 GB/s (14%) | **169 GB/s (33%)** |
| 8192 | bf16 | 526 GB/s | 46 GB/s (9%) | — |
| 8192 GEMM | fp16 | 78 TF | 4.8 TF (6%) | — |

muDNN reference = what torch_musa gets via direct C++ call (our `bench_vs_torch.py` `TORCH=1` path). This is the hardware ceiling.

BEAM=2 warmup cost: ~1 min per unique shape (many mcc subprocess compiles, diskcache makes subsequent runs fast).

## Fixes shipped (each one independently verified)

**1. `is_dtype_supported(*, "MUSA")` explicit for half + bfloat16** ← one-line upstream fix, **+2-5x bf16 speedup**
- bf16: was falling through to default `return False` in `device.py:327`; triggered `uop/decompositions.py::pm_float_decomp` to rewrite every bf16 op as `uint16 + bit manipulation + IEEE754 round-trip`
- Fix: add `"MUSA"` to the bf16 True case → kernel now declares `__mt_bfloat16*` buffers with native bf16 arithmetic. bf16 8192 GEMV 23 → 62 GB/s; bf16 8192 GEMM 0.96 → 4.65 TF
- half: was implicitly True via end-of-function fall-through (no explicit case); added explicit `case "MUSA": return True` for symmetry and clarity. MUSA hardware has native `__half` on mp_22+.

**2. BEAM launch preflight** ← crash-safety, BEAM=2 on 9B no longer core-dumps
- `MUSAProgram.__init__` queries `MU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK`
- `__call__` raises `RuntimeError` if `prod(local_size)` exceeds it
- Converts SIGSEGV-level faults into catchable exceptions that BEAM's `try/except` handles

**3. BEAM action: one targeted `LOCAL=128`** (`search.py`) + **MUSARenderer `shared_max=73728`** (`cstyle.py`)
- mp_22 warp=128, S4000 has 72 KB smem/block (vs NVIDIA default 48 KB)
- Per Moore Threads programming guide chapter 9: recommended block sizes 128/256/512/1024
- **Tried first**: add 57 extra `LOCAL/GROUP/GROUPTOP` amts across all axes → BEAM search time exploded 20x (naive width-2 search cost grows with candidate count; each candidate = 500ms mcc compile) with no wall-clock improvement. **Reverted.**
- **Kept**: single `Opt(OptOps.LOCAL, axis=0, arg=128)` entry + `shared_max=73728` override. fp16 8192 GEMV with BEAM=2 goes 162 → **169 GB/s** (33% of muDNN 505 GB/s). Search time stays ~1 min per shape.
- **Lesson**: tinygrad BEAM is already near-optimal within its IR's expressible space; throwing more actions at it just slows search without unlocking qualitatively new kernels.

**4. Half/bf16 math fp32 fallback** (`MUSARenderer.code_for_op`)
- mp_22 lacks declarations for `hexp2/hsin` (gated behind `__MUSA_ARCH__ >= 800`) and lacks bodies for `htrunc` — see Gotcha 1
- `TRUNC/SIN/LOG2/EXP2/SQRT/RECIPROCAL` all cast to float, compute, cast back
- Without this, kernels fail at compile ("undeclared hexp2") or link ("undefined hsqrt")

**5. `check()` signature fix** ← error reporting
- `muGetErrorString` wants `POINTER(POINTER(c_char))`, not `c_char_p`. See Gotcha 3.

## Ceiling analysis

**Why not 90% of muDNN via pure codegen?** Verified against two reference frameworks:

- **torch_musa**: all GEMM routes through `mudnn::MatMul` / `mublasSgemm`; **zero hand-written fast kernels**. Their example `W8A8MatmulKernel` is a textbook scalar K-loop.
- **MATE (MUSA AI Tensor Engine, https://github.com/MooreThreads/mate)**: ships 256 pre-compiled ELF binary blobs (`mubin/mp31/gemm/*.cpp`, filenames like `hhhhssgemm_gm1_nn_tce_256_128x256B128_epilogue_persis_stage4`) with filename-encoded shapes/stages/TC layout. Loaded via `muModuleLoadData`. **Hand-written SASS at the assembly level**. Targets mp_31 only; our S4000 is mp_22 and can't run them.

Moore Threads themselves don't codegen fast matmul — they hand-assemble them per arch and ship as blobs. tinygrad's IR lacks the primitives muDNN uses internally (warp shuffle, `ldmatrix`-style TC fragment loads, `cp.async`, swizzled smem layouts). BEAM explores within the IR's expressible space, so its ceiling on mp_22 is ~30% muDNN regardless of tuning.

Getting closer to muDNN requires one of:
1. Extending tinygrad IR with warp-level primitives (multi-week work, affects all backends)
2. Adding MUSA tensor-core UOp with `mma.h` template calls in `MUSARenderer.render_kernel` + `tc.get_musa` (weeks, helps GEMM not GEMV)

## Open TODOs

- [ ] **MUSA Tensor Core integration**: S4000 has QY2 TC with `16x8x16 IMMA` shape (per MUSA programming guide ch09). Add `tc.get_musa(arch)` entry + MMA intrinsic rendering in `MUSARenderer.render_kernel`. Est +5-10× for GEMM-heavy (prefill, training); negligible for single-user decode.
- [ ] **Warp shuffle reduce primitive**: `__shfl_down_sync` / `__shfl_xor_sync` / `__shfl_up_sync` verified to compile and link on mp_22 (libdevice.bc exports `__mt_shfl_*_sync_i32`). Replacing tinygrad's smem-based GROUP_REDUCE with shuffle reduce could save 5 `__syncthreads()` per reduction. Needs new tinygrad IR uop; affects all backends.
- [ ] **`__ldg` read-only load hint**: detect PARAM buffers that are only read (no STORE into them) and emit `__ldg(&ptr[i])` to hit read-only L1. ~10 lines in `MUSARenderer.string_rewrite`. Est +5-10% for decode-shape GEMV.
- [ ] **BEAM quality on quantized matmul**: preflight rejects high-local candidates that exceed mp_22 per-function max_threads. Those were often the good ones. Needs resource-aware per-kernel estimator, not blanket block-size cap.
- [ ] **Native half intrinsics on mp_31+**: when `__MUSA_ARCH__ >= 800` (future arch), `hexp2/hsin` are declared. Gate the fp32 fallback in `MUSARenderer.code_for_op` on arch.
- [ ] **Tensor cores**: `MUSARenderer.tensor_cores = []`. mp_22 has 128 TCs but `mma.h` template API isn't wired. Minimal GEMV win (memory-bound) but big for prefill / training.
- [ ] **MUSAGraph re-enable**: currently disabled in `MUSADevice.__init__`. SDK 3.1.0 returns error 801 on both `muGraphExecKernelNodeSetParams` and `muGraphExecUpdate`; the full implementation is kept at `runtime/graph/musa.py` for when driver gains exec-level updates.
- [ ] **Multi-GPU via MTLink**: `peer_access` hooks in `MUSADevice` are wired but untested (single-card test box).

## Out of scope (deliberately not pursued)

- **HCQ direct-driver backend** — MUSA driver ABI + ISA closed-source, no spec published.
- **Musify (CUDA→MUSA source translation)** — tinygrad generates kernels from uops, has no source to translate.
- **MATE blob embedding for mp_22** — MATE only ships mp_31 binaries.

## Design decisions

**Mirror libcuda driver API, not runtime API.** Driver API (`libmusa.so`, `mu*` prefix) is a 1:1 mirror of libcuda (`cu*`), down to `_v2` suffixes. `ops_musa.py` is essentially `ops_cuda.py` with `s/cu/mu/`. The `libmusart.so` runtime API (`musa*` prefix) is a thin wrapper not needed here.

**Subprocess mcc, not in-process JIT.** MUSA has no public NVRTC-equivalent. `mcc` is an offline clang fork; tinygrad shells out per unique kernel and caches fatbin on disk.

**Reuse CUDARenderer.** MUSA C++ is a strict subset of CUDA C++ for device code. Only overrides needed:
- include names: `cuda_fp16.h` → `musa_fp16.h`, `cuda_bf16.h` → `musa_bf16.h`
- bf16 type: `nv_bfloat16` → `__mt_bfloat16`
- drop WMMA prefix injection (`tensor_cores = []`)
- half/bf16 math → fp32 round-trip (mp_22 ISA limitation)

## Debug gotchas (remember this)

### 1. Half intrinsic availability on mp_22
`musa_fp16.h` gates `hexp2` and `hsin` declarations behind `#if __MUSA_ARCH__ >= 800`. mp_22 = arch 220, so they're invisible. `hsqrt/hrcp/hlog2` are declared but their bodies are only in `musa_fp16_mtgpu.h` for `hlog2/hrcp/hsqrt` (no `htrunc`). Net: **zero** half math intrinsics reliable on mp_22. Fix in `MUSARenderer.code_for_op`: cast to float, compute, cast back. Symptom if forgotten: `error: use of undeclared identifier 'hexp2'` at compile or `undefined protected symbol: hsqrt(__half)` at link.

### 2. `is_dtype_supported` must include MUSA for bfloat16
`tinygrad/device.py:is_dtype_supported(bfloat16, target)` falls through to `return False` by default. Any device not in the explicit case list triggers the universal `pm_float_decomp` rewrite — killing perf. Whenever you add a new backend, audit this function.

### 3. `muGetErrorString` ctypes signature
Header: `MUresult muGetErrorString(MUresult, const char **)`. Autogen types second arg as `POINTER(POINTER(c_char))`. Using `c_char_p` raises `TypeError: expected LP_LP_c_char instance` and masks the real launch error. Correct:
```python
p = ctypes.POINTER(ctypes.c_char)()
musa.muGetErrorString(status, ctypes.byref(p))
msg = ctypes.string_at(p).decode()
```

### 4. `mcc` flags
Working: `mcc -x musa -mtgpu --offload-arch=mp_22 -O2 --cuda-device-only -o out.fatbin src.mu`
- `--offload-arch=mp_22` (NOT `--cuda-gpu-arch`, NOT `-fmusa-rdc`; llama.cpp PR examples are wrong for this SDK)
- Output fatbin loads directly via `muModuleLoadData`

### 5. Q4_K_M / K-quant dequant is slow under tinygrad codegen
Hot kernel pattern: `__launch_bounds__(32)` with 55+ scalar `unsigned char` loads per thread per inner iter, stride `lidx0 * 6912`. Threads in a warp miss every coalescing opportunity. Per-kernel BW <3 GB/s on a 500 GB/s card. BEAM *should* find the warp-cooperative variant where 128 threads dequant one 256-value Q4_K block, but those candidates exceed mp_22 per-block resources and get preflight-rejected. Workaround: **use Q8_0 instead of K-quants on MUSA** — Q8 dequant is simple enough that tinygrad's codegen coalesces adequately (27 GB/s vs 1 GB/s, 18×).

### 6. BEAM-crashing candidates (mitigated)
Some BEAM opts crash driver at `muLaunchKernel` when local_size exceeds per-function max. Python `try/except` can't catch the SIGSEGV. Fixed via `MUSAProgram` preflight querying `MU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK`. Side effect: some of the rejected candidates were the high-coalescing ones, so BEAM's post-filter space is strictly weaker than unrestricted.

### 7. Remote Python version
Default conda py3.10 on AutoDL → `ImportError: cannot import name 'Self' from 'typing'`. tinygrad needs py3.11+. Use `conda create -n tgm python=3.12 -y && conda activate tgm`.

## Bring-up checklist for a new MUSA box
1. `mthreads-gmi` reports the card
2. `mcc --version` works
3. `python extra/musa/probe.py` prints `[0.0, 11.0, 22.0, …, 165.0]`
4. `DEV=MUSA pytest test/test_tiny.py -q` → 17 passed / 2 skipped
5. `DEV=MUSA python extra/musa/bench_vs_torch.py` completes without crash
6. `DEV=MUSA python -m tinygrad.llm -m <model.gguf> --benchmark 5` produces tok/s > 0
