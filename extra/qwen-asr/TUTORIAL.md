# Qwen-ASR Tinygrad Tutorial

This tutorial runs the pure tinygrad Qwen3-ASR port in `extra/qwen-asr/transcribe.py`,
then compares speed against the C baseline in `extra/antirez-qwen-asr`.

## 1. Build and Download

Build the C baseline:

```bash
make -C extra/antirez-qwen-asr blas
```

Download the 0.6B model:

```bash
cd extra/antirez-qwen-asr
./download_model.sh --model small
cd ../..
```

This creates `extra/antirez-qwen-asr/qwen3-asr-0.6b`.

## 2. Run Tinygrad Transcription

```bash
python extra/qwen-asr/transcribe.py \
  extra/antirez-qwen-asr/qwen3-asr-0.6b \
  extra/antirez-qwen-asr/samples/test_speech.wav
```

Silent output (transcript only):

```bash
python extra/qwen-asr/transcribe.py \
  extra/antirez-qwen-asr/qwen3-asr-0.6b \
  extra/antirez-qwen-asr/samples/test_speech.wav \
  --silent
```

Get machine-readable timing data (printed to stderr):

```bash
python extra/qwen-asr/transcribe.py \
  extra/antirez-qwen-asr/qwen3-asr-0.6b \
  extra/antirez-qwen-asr/samples/test_speech.wav \
  --silent --timings-json
```

Enable JIT attention path:

```bash
python extra/qwen-asr/transcribe.py \
  extra/antirez-qwen-asr/qwen3-asr-0.6b \
  extra/antirez-qwen-asr/samples/test_speech.wav \
  --silent --timings-json --jit
```

## 3. Benchmark Tinygrad vs C

### Fair Comparison (CPU vs CPU)

Use this when comparing the tinygrad port to `antirez-qwen-asr` fairly, since the C baseline is CPU BLAS only.

```bash
CPU=1 python extra/qwen-asr/benchmark.py \
  --model-dir extra/antirez-qwen-asr/qwen3-asr-0.6b \
  --audio extra/antirez-qwen-asr/samples/test_speech.wav \
  --warmup 1 --runs 3
```

CPU result (`samples/test_speech.wav`, 3 runs):

```text
tinygrad mean=27.993s p50=27.966s min=27.959s max=28.053s
antirez  mean=0.881s  p50=0.830s  min=0.773s  max=1.040s
speedup antirez/tinygrad (mean wall): 31.78x
```

CPU with JIT enabled for tinygrad:

```bash
CPU=1 python extra/qwen-asr/benchmark.py \
  --model-dir extra/antirez-qwen-asr/qwen3-asr-0.6b \
  --audio extra/antirez-qwen-asr/samples/test_speech.wav \
  --tinygrad-args "--jit" \
  --warmup 1 --runs 3
```

### Tinygrad Best Speed (METAL=1)

Use this when you want fastest tinygrad on Apple hardware:

```bash
METAL=1 python extra/qwen-asr/benchmark.py \
  --model-dir extra/antirez-qwen-asr/qwen3-asr-0.6b \
  --audio extra/antirez-qwen-asr/samples/test_speech.wav \
  --tinygrad-args "--jit" \
  --warmup 1 --runs 3
```

METAL result (`samples/test_speech.wav`, 3 runs):

```text
tinygrad mean=20.391s p50=20.278s min=19.762s max=21.133s
antirez  mean=1.358s  p50=1.421s  min=1.209s  max=1.446s
speedup antirez/tinygrad (mean wall): 15.01x
```

Single-run METAL timings seen locally:

```text
without --jit: wall_seconds ~22.31s
with --jit:    wall_seconds ~20.06s
```

### Output Format

Measured on:
- `Apple M4`
- macOS `Darwin 24.5.0`
- Python `3.12.11`

The benchmark script reports:
- mean/p50/min/max wall time for tinygrad and C
- parsed realtime info from C stderr when available
- mean wall-time ratio (`antirez/tinygrad`)

## 4. Notes

- This port is intentionally minimal and currently targets offline transcription.
- It reuses Qwen3-ASR prompt formatting and tokenizer decode behavior from the upstream Python reference.
- The benchmark uses wall-clock process time for both implementations.

## 5. Troubleshooting

If model files are missing:

```bash
ls -lh extra/antirez-qwen-asr/qwen3-asr-0.6b
```

If C benchmark binary is missing:

```bash
make -C extra/antirez-qwen-asr blas
```

If `soundfile` is unavailable, the tinygrad script falls back to stdlib `wave` input for WAV files.
