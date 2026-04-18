# MUSA backend for tinygrad (MTT S4000 / Moore Threads)

Status: **working for inference**. Implemented 2026-04-18 against MUSA SDK 3.1.0 on MTT S4000 (mp_22, driver 2.7.0).

## Quick start
```bash
# on a box with MUSA SDK installed and libmusa.so on LD_LIBRARY_PATH
DEV=MUSA python -m tinygrad.llm -m /path/to/model.gguf

# regenerate autogen ctypes if SDK version changes:
python -c "from tinygrad.runtime.support.autogen import gen; \
  open('tinygrad/runtime/autogen/musa.py','w').write( \
  gen('musa', \"'/usr/local/musa/lib/libmusa.so'\", ['/usr/local/musa/include/musa.h'], parse_macros=False))"
```

## Files added
- `tinygrad/runtime/autogen/musa.py` — clang2py-generated ctypes binding for libmusa.so (265 driver functions)
- `tinygrad/runtime/autogen/__init__.py` — one `case "musa":` dispatch line
- `tinygrad/runtime/support/compiler_musa.py` — `MCCCompiler` (subprocess `mcc` → fatbin, with diskcache via `Compiler.cache_key`)
- `tinygrad/runtime/ops_musa.py` — `MUSADevice / MUSAProgram / MUSAAllocator` (driver-API mirror of ops_cuda.py)
- `tinygrad/renderer/cstyle.py` — `MUSARenderer(CUDARenderer)` with half→fp32 math fallback
- `extra/musa/probe.py` — standalone ctypes probe: compile a .mu, load, launch, verify result
- `extra/musa/musa_probe.md` — collected facts about MUSA SDK / MTT S4000 (device specs, SDK layout, API mapping, mcc flags, half intrinsic availability matrix, etc.)

## Done ✓
- [x] Phase 0: hardware/toolchain probe; confirmed mp_22 / 47.91 GB / warpSize=128
- [x] Phase 1: autogen ctypes (`musa.py`, 1477 lines, all driver APIs)
- [x] Phase 2: `ops_musa.py` minimal device+program+allocator+renderer
- [x] Phase 2.5 gate: `DEV=MUSA pytest test/test_tiny.py` → 17 passed / 2 skipped (OpenCL-only)
- [x] Phase 3: `python -m tinygrad.llm` runs Qwen3-0.6B-Q8_0 and Qwen3.5-9B-Q4_K_M end-to-end; `3×23=69` correct
- [x] Refactor: MUSARenderer relocated into `tinygrad/renderer/cstyle.py` (standard home)
- [x] Half/bf16 intrinsic fallback: `TRUNC/SIN/LOG2/EXP2/SQRT/RECIPROCAL` go via fp32 on mp_22
- [x] `check()` error-reporting signature fix (`POINTER(POINTER(c_char))`)

## Measured performance
| Model | Config | tok/s | mem BW |
|---|---|---|---|
| Qwen3-0.6B-Q8_0 (751M, 639MB) | baseline | 31 | 22 GB/s |
| Qwen3-0.6B-Q8_0 | JITBEAM=2 | **48** | 34 GB/s |
| Qwen3.5-9B-Q4_K_M (8.9B, 5.4GB) | baseline | ~running, not precisely measured | — |

Warmup is dominated by `mcc` subprocess compile (~500 ms per kernel). diskcache (`Compiler.cache_key="musa_mp_22"`) makes subsequent runs fast.

## TODO / not done
- [ ] **BEAM on large models**: JITBEAM=2 on 9B triggers a process-level crash (core dump / hang) from at least one opt candidate. `beam_search`'s `try/except` cannot catch SIGSEGV. Root fix: isolate `_time_program` in a child process via `multiprocessing.Process` per candidate so a hardware fault only kills the worker. Alternatively: pre-filter opts that exceed mp_22 register/smem/local-size limits before launch. JITBEAM=1 runs to completion but picks degraded kernels (~0.14 tok/s) — the early-stop logic in search.py may not collect enough samples to rank MUSA kernels well.
- [ ] **Native half intrinsics on future archs**: current renderer forces fp32 round-trip for math ops. On mp_31+ (S4000 successors), re-enable `hexp2/hsin/…` once `__MUSA_ARCH__ >= 800` guards allow declarations AND bodies ship in the runtime.
- [ ] **Tensor cores**: `MUSARenderer.tensor_cores = []`. MUSA has MMA intrinsics in `mma.h`; plugging them into `tc.get_cuda`-style config + `render_kernel` WMMA prefix is future work (needs opcodes documented for mp_22/mp_31).
- [ ] **Graph runner**: `MUSADevice` passes `None` for graph; no `runtime/graph/musa.py`. If MUSA has a CUDA-Graph equivalent (`muGraph*` appears to exist in libmusa.so symbol dump) we could get one more speedup.
- [ ] **Multi-GPU via MTLink**: `peer_access` hooks in `MUSADevice.__init__` are wired but untested (single-card test box).
- [ ] **muBLAS/muDNN external calls**: we render everything from uops. A `cuBLAS`-style external GEMM path could close the gap with vendor libs for large matmul.

## Out of scope (not pursued, would not work)
- HCQ (direct `/dev/mtgpu` ioctl) backend — MUSA driver + ISA are closed-source, no spec published. Same situation as Ascend research (see `project_ascend_backend_research.md` in memory).
- Musify CUDA-to-MUSA translation — tinygrad is a source generator, it has no pre-existing CUDA source to translate.

## Design decisions

**Mirror libcuda driver API, not runtime API.** Driver API (`libmusa.so`, `mu*` prefix) is a 1:1 mirror of libcuda (`cu*`), down to `_v2` suffixes. ops_musa.py is essentially ops_cuda.py with `s/cu/mu/`. The higher-level `libmusart.so` runtime API (`musa*` prefix, mirrors `cuda*`) was not used — driver API is lower-latency and sufficient.

**Subprocess mcc, not an in-process JIT.** MUSA has no public NVRTC-equivalent (no `libmurtc.so`). `mcc` is the offline clang fork; tinygrad shells out to it per unique kernel and caches the fatbin on disk. First compile is slow (~500ms), cache hits are instant.

**Reuse CUDARenderer.** MUSA C++ is a strict subset of CUDA C++ for the kernel body. The only renderer overrides needed:
- swap `#include <cuda_fp16.h>` → `<musa_fp16.h>` (same for bf16)
- swap `nv_bfloat16` → `__mt_bfloat16`
- drop WMMA prefix injection by setting `tensor_cores = []`
- route half/bf16 math through fp32 (mp_22 ISA limitation, see below)

## Debug gotchas (remember this)

### 1. Half intrinsic availability on mp_22
`musa_fp16.h` gates `hexp2` and `hsin` declarations behind `#if __MUSA_ARCH__ >= 800`. mp_22 = arch 220, so they're invisible. `hsqrt/hrcp/hlog2` are declared but their bodies are only in `musa_fp16_mtgpu.h` for `hlog2/hrcp/hsqrt` (no `htrunc`). Net effect: **zero** half math intrinsics are reliable on mp_22.

Fix in `MUSARenderer.code_for_op`: cast to float, compute, cast back.
```python
def _musa_h2f(op): return lambda x,dtype: f"((half){op}((float){x}))" if dtype==dtypes.half else ...
```

Symptom if you forget: `error: use of undeclared identifier 'hexp2'` at compile, or `undefined protected symbol: hsqrt(__half)` at link.

### 2. `muGetErrorString` ctypes signature
The header is `MUresult muGetErrorString(MUresult, const char **pStr)`. Autogen types this as `POINTER(POINTER(c_char))`. Passing `ctypes.c_char_p` or `byref(c_char_p())` raises `TypeError: expected LP_LP_c_char instance` — and crucially, this masks the real launch error the caller was trying to report.

Correct pattern:
```python
p = ctypes.POINTER(ctypes.c_char)()
musa.muGetErrorString(status, ctypes.byref(p))
msg = ctypes.string_at(p).decode()
```

### 3. `mcc` flags
Working command: `mcc -x musa -mtgpu --offload-arch=mp_22 -O2 --cuda-device-only -o out.fatbin src.mu`
- `-x musa` — dialect
- `-mtgpu` — enable mtgpu target
- `--offload-arch=mp_22` — NOT `--cuda-gpu-arch` (some references use that; `--offload-arch` is the canonical flag for this SDK)
- `--cuda-device-only` — emit device-only fatbin; host stub is not needed since we dlopen via `muModuleLoadData`
- `-fmusa-rdc` from llama.cpp PR does NOT exist in this SDK — removing it was required
- Output is a fatbin; `file` reports "data"; `muModuleLoadData` loads it directly

### 4. arch string format
Compute capability from `muDeviceComputeCapability`: major=2, minor=2 → arch string `mp_22` (NOT `sm_22`, NOT `mt_22`). MUSARenderer passes this through `Target.arch` to `MCCCompiler(arch, …)` which substitutes into the flag.

### 5. BEAM candidates that SIGSEGV
Some BEAM-explored opts crash the driver at `muLaunchKernel`. Python's `try/except` around `_time_program` (search.py:164) cannot catch SIGSEGV — the whole process dies. Reproducer: JITBEAM=2 on Qwen3.5-9B. Workaround for now: don't use BEAM on large models. Proper fix: process isolation for each timing call.

### 6. Remote Python version
MUSA SDK ships with default conda py3.10 on AutoDL; tinygrad needs py3.11+ (uses `typing.Self`). Use a py3.12 conda env:
```bash
conda create -n tgm python=3.12 -y && conda activate tgm && pip install numpy pytest
```

## Verification script
`extra/musa/probe.py` is a self-contained ~40-line ctypes probe that compiles `extra/musa/vadd.mu` via `mcc`, loads via `muModuleLoadData`, launches via `muLaunchKernel`, and verifies `a[i]+b[i]==c[i]`. Run it first when bringing up a new MUSA box — if this passes, tinygrad will work; if it fails, the SDK install is broken.

```bash
cd extra/musa/
mcc -x musa -mtgpu --offload-arch=mp_22 -O2 --cuda-device-only -o vadd.fatbin vadd.mu
python probe.py
# expect: [0.0, 11.0, 22.0, ..., 165.0]
```
