# LLM raw quant dot notes

Date: 2026-05-18

Scope: AMD decode speed for GGUF Q8_0, Q4_K, and Q5_K dense linear layers in `tinygrad/llm`.

## Goal

Implement llama.cpp-style quant weight x q8 activation decode without dequantizing weights to fp16. Keep the result mergeable:

- no custom kernel
- no new public op for dot4
- no environment-variable branch
- no tensor `_gguf_type` hack
- minimal persistent memory growth

## Source references

Local llama.cpp evidence:

- `ggml/src/ggml-cuda/vecdotq.cuh`
  - Q8_0 uses `ggml_cuda_dp4a(v[i], u[i], sumi)`.
  - Q4_K keeps packed int weights and computes `(v >> shift) & 0x0F0F0F0F`.
  - Q4_K computes both main dot and min dot in the same function:
    - `dot1 = q4 * q8_1`
    - `dot2 = 1 * q8_1`
    - output term is `d * dot1 * scale - dmin * dot2 * min`.
- `ggml/src/ggml-cuda/mmvq.cu`
  - per-quant-type VDR constants select dot grouping.
- `ggml/src/ggml-cuda/mmq.cu`
  - `MUL_MAT_ID` selected experts are handled with ids and bounds before quant MMQ.
- `ggml/src/ggml-cuda/ggml-cuda.cu`
  - gate/up/GLU patterns can fuse into `mul_mat_vec_q`.

## Current implementation

`tinygrad/llm/gguf.py`

- Adds `GGUFQuantizedTensor`.
- Raw-loads selected dense 2D GGUF tensors for types Q8_0, Q4_K, Q5_K.
- Keeps dense raw quant only for large embeddings and skips:
  - token embeddings
  - norms
  - experts
  - selected llama rope-sensitive q/k tensors
  - MLA attention tensors
- Q8_0:
  - stores fp16 block scales
  - stores packed int32 qs
- Q4_K:
  - stores d/dmin and predecoded 8 scale + 8 min bytes
  - stores packed int32 qs
- Q5_K:
  - stores d/dmin and predecoded 8 scale + 8 min bytes
  - stores qh and qs in the original packed layout

`tinygrad/llm/model.py`

- Adds `Linear`, a thin subclass of `nn.Linear`.
- If a weight has `gguf_quant`, it uses raw quant GEMV; otherwise it falls back to normal `nn.Linear`.
- Activations are quantized to q8 blocks once and cached on the input tensor.
- Q8_0 uses packed int8 dot4 and scale multiplication.
- Q4_K/Q5_K use q8 activation x packed quant weight with scale/min.
- Q4_K min term is now inside the same raw linear expression as the main Q4 dot.

`tinygrad/renderer/cstyle.py`

- HIP renderer recognizes dot4 patterns and renders:
  - CDNA `gfx942/gfx950`: `__builtin_amdgcn_sdot4`
  - RDNA3 `gfx1100/gfx1101/gfx1102`: `__builtin_amdgcn_sudot4`
- Recognized patterns:
  - vector int8x4 multiply + reduce
  - scalar lane expression using shifts and masks
  - sum of 4 signed int8 lanes, rendered as dot4 with `0x01010101`

## Attempts

### Q8 raw dot

Implemented q8 activation quantization and Q8_0 raw dot using packed int32. This avoids fp16 weight dequant for dense Q8_0 weights.

Observed on `Qwen3.5-9B-Q8_0.gguf`:

- master baseline historical: about 110-112 tok/s in one run, but with different load path and memory accounting
- current raw path historical local runs: about 86-90 tok/s normal benchmark
- `DEBUG=2` profile: about 82-85 tok/s because kernel timing forces waits
- memory stayed around the model size, about 9841 MB slash-side memory

### Q4 scalar dot matcher

Initial Q4_K raw path unpacked nibbles in the expression and used scalar lane patterns so the renderer could still emit dot4.

Observed on `Qwen3.5-9B-Q4_K_M.gguf`:

- about 76-78 tok/s
- about 419-427 GB/s
- about 5994 MB slash-side memory

Observed on `Qwen3.5-27B-Q4_K_M.gguf`:

- about 29.8-30.4 tok/s
- about 509-518 GB/s
- about 17429 MB slash-side memory

### Failed packed Q4 attempt

Tried to keep Q4_K fully packed and directly reinterpret the byte layout. This was wrong:

- output quality broke, for example nonsensical mixed text after a few tokens
- speed was worse, about 60 tok/s on 9B Q4

This path was reverted.

Root cause: the Q4_K byte grouping was not equivalent to llama.cpp `q4[0]` and `q4[4]` indexing for all sub-blocks.

### Q4_K scale/min predecode

Moved Q4_K/Q5_K 6-bit scale/min unpacking from decode kernels into GGUF load.

Observed on `Qwen3.5-9B-Q4_K_M.gguf`:

- before: about 76-78 tok/s, about 419-427 GB/s, about 5994 MB
- after: about 80-81 tok/s, about 445-450 GB/s, about 6090 MB

Memory cost:

- 9B Q4: about +96 MB
- 27B Q4: about +335 MB

The cost is from storing 20 metadata bytes per Q4_K/Q5_K block instead of 16 bytes.

### Q4_K int32 packed qs

Changed Q4_K `k_qs` storage to int32 packed loads while keeping the same number of bytes. Decode uses direct int32 load plus mask/shift.

Observed on `Qwen3.5-9B-Q4_K_M.gguf`:

- about 83-85 tok/s
- about 458-472 GB/s
- about 6090 MB

Observed on `Qwen3.5-27B-Q4_K_M.gguf`:

- about 30.6-30.8 tok/s
- about 533-536 GB/s
- about 17764 MB

### Q4_K min dot fusion

Removed the separately materialized activation sum and expressed the min term in the same raw linear expression.

First version put the min reduction in the same kernel but generated scalar byte-lane sums. Then the HIP renderer was extended to lower 4-lane signed byte sum to dot4 with `0x01010101`.

Generated code was checked with a small Q4_K raw Linear:

- main Q4 dot emits `sudot4(..., val & 0x0F0F0F0F, ...)`
- min dot emits `sudot4(..., 0x1010101, ...)`
- no separate `r_8_8` activation-sum kernel remains for the min term

Observed on `Qwen3.5-9B-Q4_K_M.gguf`:

- before fusion: about 82-84 tok/s, best about 84.5 tok/s
- after fusion: 85-86 tok/s, best 86.14 tok/s
- bandwidth best increased to about 479 GB/s
- memory stayed about 6090 MB

Observed on `Qwen3.5-27B-Q4_K_M.gguf`:

- before fusion: about 30.6-30.8 tok/s, about 533-536 GB/s
- after fusion: about 31.2-31.6 tok/s, about 543-550 GB/s
- memory stayed about 17764 MB

### Failed vector min-dot attempt

Tried to express min dot with `_q8_dot4` and a vector dot constant. This exposed a scalar-constant `fold_bitcast` issue:

`TypeError: 'int' object is not iterable`

This was reverted. The final path keeps model code scalar and lets the renderer rewrite sum4 to dot4.

## Current benchmark data

All CLI bandwidth numbers are tinygrad `GlobalCounters.global_mem / wall_time`, not hardware HBM counters.

### Qwen3.5-9B-Q4_K_M.gguf

Command:

```sh
JITBEAM=2 PYTHONPATH=. python3 tinygrad/llm/cli.py --model ./Qwen3.5-9B-Q4_K_M.gguf --benchmark 12
```

Current representative decode rows:

```text
11.78 ms, 84.86 tok/s, 472.19 GB/s, 5564/6090 MB
11.71 ms, 85.40 tok/s, 475.20 GB/s, 5564/6090 MB
11.61 ms, 86.14 tok/s, 479.41 GB/s, 5565/6090 MB
11.61 ms, 86.14 tok/s, 479.42 GB/s, 5565/6090 MB
```

### Qwen3.5-27B-Q4_K_M.gguf

Command:

```sh
JITBEAM=2 PYTHONPATH=. python3 tinygrad/llm/cli.py --model ./Qwen3.5-27B-Q4_K_M.gguf --benchmark 8
```

Current representative decode rows:

```text
32.07 ms, 31.19 tok/s, 542.70 GB/s, 17402/17764 MB
31.88 ms, 31.36 tok/s, 545.85 GB/s, 17403/17764 MB
31.64 ms, 31.60 tok/s, 550.03 GB/s, 17404/17764 MB
31.69 ms, 31.56 tok/s, 549.27 GB/s, 17406/17764 MB
```

### Qwen3.5-9B-Q8_0.gguf

Command used for debug profile:

```sh
DEBUG=2 JITBEAM=2 PYTHONPATH=. python3 tinygrad/llm/cli.py --model ./Qwen3.5-9B-Q8_0.gguf --benchmark 4
```

Representative DEBUG=2 rows:

```text
12.20 ms, 81.95 tok/s, 721.42 GB/s, 8803/9841 MB
11.81 ms, 84.65 tok/s, 745.27 GB/s, 8803/9841 MB
```

Normal non-DEBUG raw Q8 runs were historically around 86-90 tok/s.

## Tests run

```sh
ruff check tinygrad/llm/model.py tinygrad/llm/gguf.py tinygrad/renderer/cstyle.py test/unit/test_gguf.py test/null/test_uops.py
DEV=CPU python3 -m pytest -q test/unit/test_gguf.py::TestGGUFGEMV
python3 -m pytest -q test/null/test_uops.py
git diff --check
```

Results:

- `TestGGUFGEMV`: 13 passed
- `test/null/test_uops.py`: 41 passed, 2 skipped, 1 xfailed
- `git diff --check`: passed

## Remaining work

- MoE selected-expert raw quant path, analogous to llama.cpp `MUL_MAT_ID`.
- gate/up/GLU fusion for dense FFN and shared experts.
- Better Q5_K packed path; current Q5_K still expands more than Q4_K.
- More systematic profile data with hardware counters on MI300X for Kimi/Moonlight.
