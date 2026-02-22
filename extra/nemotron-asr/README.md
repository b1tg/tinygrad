## Nemotron ASR — tinygrad

tinygrad implementation of NVIDIA's [nemotron-speech-streaming-en-0.6b](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) conformer-transducer (RNN-T) model. Ported from the C++ / ggml reference in `extra/nemotron-asr.cpp/`.

### Usage

```bash
# download model + test wav and run
python extra/nemotron-asr/test.py

# manual
python extra/nemotron-asr/transcribe.py <model.gguf|model.nemo> <audio.wav>

# live microphone input (segmented mode)
python extra/nemotron-asr/transcribe.py <model.gguf|model.nemo> --live

# useful live knobs
#   --mic-gain 2.0     : boost microphone input if volume is low
#   --chunk-seconds 1.0 --vad-threshold 0.01 --vad-end-seconds 1.5 : segment mode sensitivity
#   --vad-ratio 1.4 --vad-preroll-seconds 1.0 : adaptive threshold + capture leading words
#   --segment-max-seconds 12 : force flush for continuous speech/no-silence streams
#   --segment-overlap-seconds 1.0 : preserve context across forced flush boundaries
#   --live-queue-chunks 256 : keep recording while decode is busy (prevents speech loss during long decode)
#   --live-debug-level 0..3 : 0 transcript only (default), 1 segment events, 2 +timing, 3 +VAD/queue
# debug-level 0 keeps output minimal and prints "..." heartbeat during decode
```

### Architecture

- **Preprocessing**: STFT via `Tensor._pool` + DFT matrix multiply → mel spectrogram (tinygrad)
- **Encoder**: 24-layer conformer — FFN½ → RelPosMHA → ConvModule → FFN½ → LayerNorm (tinygrad)
- **Decoder**: 2-layer LSTM greedy RNN-T decode with TinyJit-cached step (pure tinygrad, no numpy)
- **Weight formats**: GGUF (Q8_0, Q4_0, F16, F32) and .nemo

### Key constants

| param | value |
|-------|-------|
| d_model | 1024 |
| n_heads | 8 |
| d_ff | 4096 |
| n_layers | 24 |
| kernel_size | 9 |
| vocab_size | 1025 (1024 tokens + blank) |
| decoder_dim | 640 |
| sample_rate | 16kHz |

### TinyJit for decode

The RNN-T decoder runs hundreds of sequential LSTM steps. Each step needs `.item()` to check the predicted token (blank → advance time, non-blank → emit and loop). `TinyJit` caches the LSTM step kernel after 2 warmup calls, eliminating per-step graph build overhead. LSTM states are maintained in persistent buffers updated via `.assign()` (JIT output buffers get reused across calls, so `.contiguous()` alone doesn't create independent copies).

### Benchmark

Tested on 215.52s audio, Q8_0 GGUF model, Apple M-series Mac.

| implementation | backend | processing time | RTF |
|---|---|---|---|
| C++ / ggml (streaming) | CPU (4 threads) | 104.73s | 0.486x |
| tinygrad (TinyJit) | Metal GPU | 109.63s | 0.509x |

RTF = processing time / audio duration (lower is better, <1.0 = faster than real-time).

**Breakdown (tinygrad)**:

| stage | time |
|---|---|
| preprocess (STFT + mel) | 0.12s |
| encoder (24-layer conformer) | 1.86s |
| decode (greedy RNN-T) | 107.65s |

The decode stage dominates because each LSTM step requires `.item()` to check if the predicted token is blank (GPU→CPU sync, ~25ms per call on Metal). The JIT step kernel itself runs in 0.23ms — the bottleneck is the synchronous readback, not compute.

### Files

- `transcribe.py` — model definition, preprocessing, GGUF/nemo loading, greedy decode
- `live.py` — live microphone segmented transcription pipeline (capture, queueing, VAD, retries)
- `test.py` — downloads Q8_0 GGUF + test WAV via `fetch()`, runs end-to-end transcription
