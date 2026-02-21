# How Voxtral Turns Sound Into Text

A step-by-step walkthrough of `examples/voxtral.py` — Mistral's Voxtral Realtime 4B
speech-to-text model, implemented in ~280 lines of tinygrad.

**Prerequisites**: You should be comfortable with matrix multiplication, basic PyTorch
(tensors, linear layers, softmax), and have a rough idea of what a transformer is.

## The Big Picture

Voxtral converts audio to text in three stages:

```
Audio waveform ──► Mel spectrogram ──► Encoder ──► Adapter ──► Decoder ──► Text tokens
     [N]           [128, T]          [S, 1280]    [S/4, 3072]              [token_ids]
```

1. **Mel spectrogram** — convert raw audio samples into a 2D "image" of frequency vs time
2. **Encoder** — a 32-layer transformer that reads the spectrogram and produces audio embeddings
3. **Adapter** — a small MLP that reshapes encoder output to match the decoder's dimension
4. **Decoder** — a 26-layer transformer (basically a language model) that generates text tokens one at a time

Let's trace what happens to a 10-second audio clip, from raw samples to printed text.

---

## Stage 1: Audio Loading

```python
SAMPLE_RATE = 16000  # 16 kHz mono

def load_audio(path: str):
  result = subprocess.run(["ffmpeg", "-i", path, "-f", "f32le", "-acodec", "pcm_f32le",
                           "-ac", "1", "-ar", str(SAMPLE_RATE), "-v", "quiet", "-"], capture_output=True)
  return list(struct.unpack(f"<{len(result.stdout)//4}f", result.stdout))
```

ffmpeg decodes any audio format (MP3, WAV, OGG, etc.) into raw 32-bit floats at 16 kHz mono.
A 10-second clip becomes 160,000 floating-point numbers, each between roughly -1.0 and 1.0.

**Why 16 kHz?** Human speech rarely has useful information above 8 kHz. By the
[Nyquist theorem](https://en.wikipedia.org/wiki/Nyquist%E2%80%93Shannon_sampling_theorem),
16 kHz captures everything up to 8 kHz — enough for speech, while keeping the data small.

---

## Stage 2: Mel Spectrogram

This is the most math-heavy preprocessing step. The goal: turn a 1D waveform into a 2D
representation that shows **which frequencies are present at each moment in time**.

### 2a. Short-Time Fourier Transform (STFT)

Audio changes over time — the frequencies in "hello" are different from "world". We can't
just take one FFT of the whole signal. Instead, we slide a window across the audio and
take the FFT of each window:

```python
WINDOW_SIZE = 400   # 25ms at 16 kHz — one "frame"
HOP_LENGTH  = 160   # 10ms — how far the window moves each step

window = Tensor.hann_window(WINDOW_SIZE)   # smooth tapering window
stft = audio.stft(WINDOW_SIZE, HOP_LENGTH, window=window, pad_mode="constant", return_complex=True)
# stft shape: (201, n_frames, 2)  — 201 frequency bins, last dim is [real, imag]
```

**What's happening step by step:**

1. **Pad** the audio with zeros on both sides (by `WINDOW_SIZE // 2 = 200` samples)
2. **Slice** into overlapping frames of 400 samples, each shifted by 160 samples
3. **Multiply** each frame by a Hann window (a smooth bell curve) to avoid edge artifacts
4. **DFT** each frame — a matrix multiply that decomposes the 400 time-domain samples
   into 201 frequency bins (the positive half of the spectrum)

The DFT is the key operation. For each frequency bin `k` and each frame `t`:

```
real[k, t] = sum(frame[n] * cos(2*pi*k*n / N) for n in range(N))
imag[k, t] = sum(frame[n] * -sin(2*pi*k*n / N) for n in range(N))
```

This is just a matrix multiplication: `frames @ DFT_matrix.T`. The DFT matrix has shape
`(201, 400)` where each row is a cosine (or sine) wave at a specific frequency.

The output has shape `(201, n_frames, 2)` — 201 frequency bins, `n_frames` time steps,
and 2 for real and imaginary parts.

### 2b. Power Spectrum

```python
magnitudes = stft[..., :-1, :].square().sum(-1)   # (201, n_frames-1)
```

`real^2 + imag^2` gives the **power** (energy) at each frequency. We drop the last frame
(edge artifact from padding). The result tells us: "at time step `t`, frequency bin `k`
has this much energy."

### 2c. Mel Filterbank

Human hearing is not linear — we're much better at distinguishing low frequencies
(200 Hz vs 400 Hz sounds very different) than high frequencies (6000 Hz vs 6200 Hz
sounds nearly the same). The **mel scale** models this.

```python
NUM_MEL_BINS = 128
mel_filters = Tensor(compute_mel_filters())   # shape: (201, 128)
mel_spec = mel_filters.T @ magnitudes          # (128, n_frames-1)
```

The mel filterbank is a `(201, 128)` matrix of overlapping triangular filters. Each of
the 128 mel bins covers a range of frequency bins, with the ranges spaced evenly on
the mel scale (wide at high frequencies, narrow at low). The matrix multiply
`mel_filters.T @ magnitudes` sums up the power in each mel range.

The result: a `(128, n_frames)` matrix — 128 mel frequency bands over time. This is the
mel spectrogram, basically a "heatmap" of audio energy.

### 2d. Log Compression and Normalization

```python
log_spec = mel_spec.clamp(min_=1e-10).log2() * math.log10(2)   # = log10(mel_spec)
return (log_spec.maximum(Tensor.full(log_spec.shape, -6.5)) + 4.0) / 4.0
```

We take `log10` of the power (clamped to avoid `log(0)`). This matches human perception —
we hear loudness on a logarithmic scale (decibels). Then we clip to a floor of -6.5 and
normalize to roughly [0, 1].

**For a 10-second clip**: 160,000 samples become a `(128, ~999)` mel spectrogram — 128
frequency bands over ~999 time frames (one every 10ms).

---

## Stage 3: Padding and Alignment

```python
N_LEFT_PAD, N_DELAY, N_RIGHT_PAD, RAW_TOK_LEN = 32, 6, 17, 1280

prompt_ids = [TOKEN_BOS] + [TOKEN_PAD] * (N_LEFT_PAD + N_DELAY)
padded = [0.0] * (N_LEFT_PAD * RAW_TOK_LEN) + audio + [0.0] * (...)
```

Voxtral is a "realtime" model — it's designed to transcribe audio **as it streams in**,
not just after the whole clip is available. The padding simulates the buffering that
happens in streaming mode:

- **Left padding** (32 tokens worth of silence): gives the encoder context to "warm up"
- **Delay** (6 tokens): the model waits a bit before starting to emit text — it needs
  some future context to transcribe accurately
- **Right padding** (17 tokens): ensures the final words have enough trailing context

Each "token" in audio space corresponds to `RAW_TOK_LEN = 1280` raw samples (80ms).

---

## Stage 4: Encoder (Audio Transformer)

The encoder takes the mel spectrogram `(128, T)` and produces a sequence of embedding
vectors `(S, 1280)`.

### 4a. Convolutional Front-End

```python
# conv1: (128, T) -> (1280, T)     stride=1, kernel=3
h = causal_conv1d(mel, self.conv_layers_0_conv.weight, ..., stride=1).gelu()
# conv2: (1280, T) -> (1280, T//2)  stride=2, kernel=3
h = causal_conv1d(h, self.conv_layers_1_conv.weight, ..., stride=2).gelu()
h = h.permute(0, 2, 1)   # -> (1, T//2, 1280)
```

Two 1D convolutions serve as a "feature extractor":
- `conv1` projects 128 mel bands up to 1280 dimensions (the encoder's hidden size)
- `conv2` downsamples by 2x with stride=2, halving the sequence length

Both use **causal** padding (only left-pad) so the model can't peek into the future —
important for streaming.

After this, we have a sequence of 1280-dimensional vectors, one every 20ms of audio.

### 4b. Transformer Layers with Sliding Window

```python
ENC_LAYERS, ENC_HEADS, ENC_HEAD_DIM, ENC_WINDOW = 32, 32, 64, 750

for layer in self.transformer_layers:
  chunks, k_cache, v_cache = [], None, None
  for start in range(0, S, ENC_WINDOW):
    h_chunk, k_cache, v_cache = layer(h[:, start:end], ..., k_cache, v_cache)
    chunks.append(h_chunk)
  h = Tensor.cat(*chunks, dim=1)
```

Each of the 32 encoder layers is a standard transformer block:

1. **RMSNorm** — normalize the input (simpler than LayerNorm, no mean subtraction)
2. **Self-attention** — each position attends to other positions to mix information
3. **Add & Norm** — residual connection + another RMSNorm
4. **Feed-forward network** — two linear layers with SiLU activation (SwiGLU pattern)
5. **Add** — another residual connection

The key detail is the **sliding window of 750 positions**. Instead of attending to the
entire sequence (which would be O(n^2) in memory), each layer processes the sequence in
chunks. Each chunk can attend to the current chunk plus the KV cache from the previous
chunk. This caps memory at O(750^2) per layer regardless of audio length.

**RoPE (Rotary Position Embeddings)** encode position information by rotating the query
and key vectors:

```python
x1, x2 = x[..., ::2], x[..., 1::2]   # split into pairs
rotated = stack(x1 * cos - x2 * sin,
                x2 * cos + x1 * sin)   # 2D rotation per pair
```

This elegantly encodes relative positions — the dot product between two rotated vectors
depends only on their distance, not absolute position.

**Output**: `(S, 1280)` — one 1280-dimensional vector for every ~20ms of audio.

---

## Stage 5: Adapter

```python
class Adapter:
  def __call__(self, enc_out: Tensor):
    # reshape: (S, 1280) -> (S/4, 5120)  — group 4 consecutive vectors
    x = enc_out.reshape(enc_out.shape[0] // DOWNSAMPLE_FACTOR, ENC_DIM * DOWNSAMPLE_FACTOR)
    return self.proj2(self.proj1(x).gelu())
    # (S/4, 5120) -> (S/4, 3072) -> (S/4, 3072)
```

The adapter does two things:
1. **4x downsampling**: groups every 4 consecutive encoder vectors by concatenating them
   (`4 * 1280 = 5120` dimensions). This reduces the sequence length by 4x.
2. **Dimension projection**: two linear layers map from 5120 to 3072 (the decoder's
   hidden size).

After the adapter, each vector represents ~80ms of audio and lives in the same vector
space as text token embeddings.

---

## Stage 6: Decoder (Language Model)

The decoder is essentially a causal language model (like a small LLaMA) that generates
text tokens conditioned on the audio embeddings.

### 6a. Input Embedding

```python
# Audio embeddings + token embeddings are ADDED together
prefix_embeds = ada_out[:L] + decoder.tok_embeddings(Tensor(prompt_ids))
```

This is the key insight: audio and text share the same sequence. The first `L` positions
get both an audio embedding (from the adapter) and a token embedding (BOS + padding tokens).
The decoder processes this fused sequence.

### 6b. Adaptive Normalization

```python
def precompute_ada(self, t_value: float):
  # sinusoidal encoding of the "delay" parameter
  inv_freq = (-log(10000) * arange(DEC_DIM // 2) / (DEC_DIM // 2)).exp()
  t_cond = cat(cos(t_value * inv_freq), sin(t_value * inv_freq))
  # per-layer scale factor
  for layer in self.layers:
    layer.ada_scale = (1 + layer.t_cond_proj(t_cond)).reshape(1, 1, DEC_DIM)
```

The decoder uses **adaptive normalization** — the delay parameter `t_value` (how many
tokens ahead the model is "looking") is encoded as a sinusoidal embedding and used to
scale the FFN input in each layer. This lets the same model work with different
streaming latency settings.

### 6c. Transformer Layers

Each of the 26 decoder layers follows the same pattern as the encoder, with two
differences:

- **Grouped Query Attention (GQA)**: 32 query heads but only 8 key/value heads.
  Each KV head is shared by 4 query heads. This reduces KV cache memory by 4x.
- **KV cache**: Keys and values are cached in a pre-allocated buffer so we don't
  recompute them for past positions during autoregressive generation.

```python
# Pre-allocate cache: (2, batch, 8, max_context, 128)
self.cache_kv = Tensor.zeros(2, B, DEC_KV_HEADS, self.max_context, DEC_HEAD_DIM)
# Write new KV into cache at current position
self.cache_kv[:, :, :, start_pos:start_pos+T, :].assign(Tensor.stack(k, v))
# Read all KV up to current position
keys, vals = self.cache_kv[0, :, :, :start_pos+T, :], self.cache_kv[1, :, :, :start_pos+T, :]
```

### 6d. Token Generation

```python
def _run_layers(self, h, start_pos):
  for layer in self.layers: h = layer(h, start_pos)
  # Project to vocabulary and pick the highest-scoring token
  return (self.norm(h)[:, -1, :] @ self.tok_embeddings.weight.T).argmax(-1)
```

After all layers, the final hidden state is projected to vocabulary size (131,072 tokens)
using the embedding matrix as the output projection (weight tying). `argmax` picks the
most likely token.

### 6e. The Decode Loop

```python
# Prefill: process all prompt positions at once
decoder.prefill(prefix_embeds[:-1])

# First token
token = decoder.decode_first(prefix_embeds[-1], pos=L-1)

# Autoregressive loop: one token at a time
for pos in range(L, n_audio):
  if token == TOKEN_EOS: break
  token = decoder.forward_jit(ada_out, Tensor([token]), v.bind(pos))
  sys.stdout.write(tokenize([token]))   # print as we go
```

1. **Prefill**: Process all prompt tokens (BOS + padding) in parallel to fill the KV cache
2. **First decode**: Generate the first text token from the last prompt position
3. **Autoregressive loop**: For each remaining audio position, generate one text token.
   Each step reads the audio embedding at that position (`ada_out[pos]`), adds the
   previous token's embedding, and runs through all 26 decoder layers.

`TinyJit` compiles the decode step into an optimized kernel after the first call, making
subsequent steps much faster.

---

## Putting It All Together

Here's the complete data flow for transcribing "Hello world" from a 2-second clip:

```
Raw audio:     [32000 float32 samples]
                    |
                    v
Mel spectrogram: (128, ~199)           128 mel bands x ~199 time frames
                    |
                    v
Conv front-end:  (1, ~99, 1280)        stride-2 downsampling
                    |
                    v
32x Encoder:     (1, ~99, 1280)        sliding-window self-attention
                    |
                    v
Adapter:         (~24, 3072)           4x downsampling + projection
                    |
                    v
+ Token embeds:  (~24, 3072)           audio + text fused in same sequence
                    |
                    v
26x Decoder:     tokens one at a time  KV-cached autoregressive generation
                    |
                    v
Output:          "Hello world"
```

The entire model is ~4B parameters, runs in a single file, and needs no dependencies
beyond tinygrad and ffmpeg.

---

## Key Dimensions at a Glance

| Constant | Value | Meaning |
|----------|-------|---------|
| `SAMPLE_RATE` | 16,000 | Audio samples per second |
| `WINDOW_SIZE` | 400 | STFT window = 25ms |
| `HOP_LENGTH` | 160 | STFT hop = 10ms |
| `NUM_MEL_BINS` | 128 | Mel frequency bands |
| `ENC_DIM` | 1,280 | Encoder hidden size |
| `ENC_LAYERS` | 32 | Encoder transformer layers |
| `ENC_WINDOW` | 750 | Sliding window size (~15s of audio) |
| `DEC_DIM` | 3,072 | Decoder hidden size |
| `DEC_LAYERS` | 26 | Decoder transformer layers |
| `DEC_HEADS` / `DEC_KV_HEADS` | 32 / 8 | GQA: 4 queries per KV head |
| `DOWNSAMPLE_FACTOR` | 4 | Adapter groups 4 encoder vectors into 1 |
| `VOCAB_SIZE` | 131,072 | Tekken tokenizer vocabulary |

## Running It

```bash
# Download the model (requires Hugging Face access)
huggingface-cli download mistralai/Voxtral-Mini-4B-Realtime-2602 --local-dir voxtral-model

# Transcribe an audio file
python examples/voxtral.py voxtral-model audio.wav
```
