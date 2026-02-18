# Vox Status

## Benchmark Comparison (2026-02-18)

Test audio: `tinycorp-meetings/test.wav` — 215.5s, 2706 tokens

### RTX 4090 (6x, Linux)

| Implementation | Model Load | Encoder | Prefill | Decode | ms/step | Transcribe (no load) |
|---|---|---|---|---|---|---|
| tinygrad Voxtral (JITBEAM=2) | 43.9s | 33.3s | 5.1s | 48.4s (55.9 tok/s) | 17.9 | **86.8s** |
| tinygrad Voxtral (no BEAM) | 43.9s | 33.3s | 5.1s | 196.0s (13.8 tok/s) | 72.4 | **234.4s** |
| Whisper Large v3 (PyTorch) | 19.0s | — | — | — | — | **63.6s** |
| voxtral.c (BLAS/CPU) | ~0s (mmap) | >107min | — | — | — | N/A (CPU only) |

### Mac M4 (to fill in)

| Implementation | Model Load | Encoder | Prefill | Decode | ms/step | Transcribe (no load) |
|---|---|---|---|---|---|---|
| tinygrad Voxtral | ___s | ___s | ___s | ___s (___ tok/s) | ___ | **___s** |
| voxtral.c (MPS) | ~0s (mmap) | ___s | ___s | ___s | ___ | **___s** |
| Whisper Large v3 (PyTorch, MPS) | ___s | — | — | — | — | **___s** |

### M3 Max (from voxtral.c README, 3.6s audio extrapolated to 215.5s)

| Implementation | Encoder | Prefill | Decode | ms/step |
|---|---|---|---|---|
| voxtral.c (MPS) | ~17s (est.) | ~0.3s | ~85s (est.) | 23.5-31.6 |
| voxtral.c (BLAS/CPU) | ~480s (est.) | ~72s (est.) | ~907s (est.) | 335 |

### Notes

- **JITBEAM matters**: without BEAM kernel optimization, tinygrad decode is 5.8x slower (17.9 vs 72.4 ms/step). BEAM results are NOT cached across runs — must set `JITBEAM=2` every time.
- **Encoder is tinygrad's bottleneck**: 33.3s for 215.5s audio. voxtral.c MPS encoder is ~17s (est.) for the same audio on M3 Max.
- **Model size**: Whisper Large is 1.55B params, Voxtral is 4B (2.6x larger). Voxtral produces more detailed transcription.
- **voxtral.c BLAS**: CPU-only with bf16→f32 on-the-fly conversion. Not comparable to GPU benchmarks. Killed after 107min on encoder alone.
- **Model loading**: voxtral.c uses mmap (near-instant), tinygrad loads safetensors into GPU (43.9s), PyTorch whisper loads to GPU (19.0s).
- **Streaming**: tinygrad voxtral and voxtral.c both support streaming token output. Whisper outputs all text at once.

## Previous Benchmark (CPU, 2026-02-15)

| File | Duration | Mel | Encoder | Prefill | Decode | Total | ms/tok |
|------|----------|-----|---------|---------|--------|-------|--------|
| test_speech.wav | 3.6s | 3.6s | 0.67s | 13.8s | 8.4s (57 tok) | 68s | 148 |
| jfk.wav | 11.0s | 3.8s | 0.71s | 14.1s | 16.7s (149 tok) | 77s | 112 |
| antirez...short.ogg | 60.0s | 5.6s | 0.67s | 22.0s | 126.0s (761 tok) | 196s | 166 |
| I_have_a_dream.ogg | 180.0s | 9.0s | 0.68s | 38.5s | 381.2s (2262 tok) | 471s | 169 |

## Known Issues

1. **JITBEAM not cached** — BEAM-optimized kernels don't persist in cache.db across runs. Must set `JITBEAM=2` on every invocation.
2. **Model load ~44s per run** — no weight caching between invocations. voxtral.c solves this with mmap.
3. **Encoder is slow** — 33.3s for 215.5s audio on RTX 4090. Main bottleneck vs Whisper.
4. **Prefill scales with audio length** — 5.1s for 215.5s audio.
