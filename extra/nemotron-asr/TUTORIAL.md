# How Speech Recognition Works: A Walk Through Nemotron ASR

This tutorial explains how an automatic speech recognition (ASR) model turns a WAV file into text. We'll follow a single audio file through every stage of NVIDIA's [Nemotron 0.6B](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) conformer-transducer, referencing the actual tinygrad implementation in `transcribe.py`.

**Prerequisites**: You should know what matrix multiplication does, and have basic familiarity with PyTorch-style tensor operations (reshape, permute, linear layers, softmax). No audio/speech background needed.

## The Big Picture

```
WAV file ──→ mel spectrogram ──→ encoder ──→ decoder ──→ text
            (preprocessing)    (conformer)   (RNN-T)
```

The model has three parts:

1. **Preprocessing** turns raw audio samples into a 2D image-like representation called a mel spectrogram
2. **Encoder** (a conformer -- convolution + transformer) reads that spectrogram and produces a sequence of feature vectors, one per ~80ms of audio
3. **Decoder** (an LSTM + joint network) converts those feature vectors into text tokens, one at a time

Let's trace a real example. Say we have a 10-second WAV file at 16kHz. That's 160,000 audio samples -- just a list of 160,000 floating point numbers between -1.0 and 1.0.

## Stage 1: WAV to Numbers

```
WAV file → [160000] float32 samples
```

A WAV file stores audio as integers (usually 16-bit, range -32768 to 32767). We convert to float by dividing by 32768:

```python
# transcribe.py:408-410
audio = Tensor(list(struct.unpack(fmt, raw)), dtype=dtypes.float32) / scale
```

If the sample rate isn't 16kHz, we resample with linear interpolation. The model expects exactly 16,000 samples per second of audio.

## Stage 2: Mel Spectrogram

This is where it gets interesting. We need to turn a 1D waveform into a 2D representation that captures "what frequencies are present at each moment in time." This happens in the `preprocess()` function (`transcribe.py:37-75`).

### Step 2a: Pre-emphasis

```python
# transcribe.py:40-41
audio = audio_float[1:] - 0.97 * audio_float[:-1]
```

This is a simple high-pass filter. Each sample gets most of the previous sample subtracted from it. Why? Human speech has more energy in low frequencies, and this filter boosts the high frequencies to give the model a more balanced view. It's just `y[t] = x[t] - 0.97 * x[t-1]`.

### Step 2b: Framing (STFT)

We can't just FFT the entire audio at once -- that would tell us what frequencies exist *somewhere* in the file, but not *when*. Instead, we chop the audio into short overlapping windows and FFT each one separately. This is called the Short-Time Fourier Transform (STFT).

```
  audio: [.......................................] 160000 samples

  frame 0: [----512----]
  frame 1:    [----512----]          (hop 160 samples = 10ms forward)
  frame 2:       [----512----]
  ...
  frame 999:                  [----512----]
```

Each frame is 512 samples (~32ms of audio). We advance by 160 samples (10ms) between frames. So frames overlap heavily -- this is intentional, it gives us smooth time resolution.

```python
# transcribe.py:64 -- extract overlapping frames using _pool
frames = audio.reshape(1, 1, -1)._pool((N_FFT,), stride=(HOP_LENGTH,))
# result: [n_frames, 512]
```

Before FFT, each frame is multiplied by a window function (Hann window) that tapers the edges to zero. This prevents spectral leakage -- artifacts from the frame boundaries.

### Step 2c: DFT as Matrix Multiply

Here's the key insight: the Discrete Fourier Transform is just a matrix multiply. We build a DFT matrix where each row represents a different frequency:

```python
# transcribe.py:57-61
k = Tensor.arange(n_bins).reshape(n_bins, 1)   # frequency index [257, 1]
n = Tensor.arange(N_FFT).reshape(1, N_FFT)     # time index      [1, 512]
angle = -2.0 * math.pi * k * n / N_FFT         # [257, 512]
dft_real = angle.cos()
dft_imag = angle.sin()
```

Then we just matmul each frame against this matrix:

```python
# transcribe.py:68-70
real = frames @ dft_real.T    # [n_frames, 257]
imag = frames @ dft_imag.T    # [n_frames, 257]
spec = real * real + imag * imag  # power spectrum
```

The result is the power spectrum: for each frame, how much energy is at each of 257 frequency bins (0 Hz to 8000 Hz).

### Step 2d: Mel Filterbank

Humans don't perceive frequencies linearly. The difference between 100Hz and 200Hz sounds much larger than between 5100Hz and 5200Hz. The mel scale compresses high frequencies to match human perception.

The mel filterbank is a `[128, 257]` matrix (stored as a learned weight in the model). It groups the 257 frequency bins into 128 mel bands:

```python
# transcribe.py:73-74
mel = spec @ filterbank.T     # [n_frames, 257] @ [257, 128] → [n_frames, 128]
mel = (mel + LOG_ZERO_GUARD).log()
```

The log is important -- humans perceive loudness logarithmically (doubling the amplitude sounds like a fixed increase in volume, not a doubling).

**Final result for our 10s example**: `[999, 128]` -- 999 time frames, each with 128 mel features. You can think of this as a 999x128 grayscale image where x-axis is time, y-axis is frequency, and brightness is energy.

## Stage 3: Encoder (Conformer)

The encoder converts the mel spectrogram `[T, 128]` into a sequence of high-level feature vectors `[T/8, 1024]`. It has two parts: conv subsampling and 24 conformer layers.

### Step 3a: Conv Subsampling

The mel has too many time steps -- one every 10ms. Three strided 2D convolutions reduce the time dimension by 8x:

```python
# transcribe.py:101-124
class ConvSubsampling:
  # conv0: stride 2 → time/2
  # conv2+conv3 (depthwise+pointwise): stride 2 → time/4
  # conv5+conv6 (depthwise+pointwise): stride 2 → time/8
  self.out = nn.Linear(256 * 17, D_MODEL)  # project to d_model=1024
```

The input mel `[1, 1, 999, 128]` is treated like a 1-channel image. After three stride-2 convs:
- Time: 999 → 500 → 250 → 125 frames (now one per ~80ms)
- Freq: 128 → 64 → 32 → 17 (mel bins get compressed)
- Channels: 1 → 256

Then flatten channels*freq = 256*17 = 4352 and project to 1024 with a linear layer.

**Output**: `[1, 125, 1024]` -- 125 time steps, each a 1024-dim vector.

### Step 3b: Conformer Layers (x24)

Each conformer layer combines the strengths of transformers (global attention) and CNNs (local patterns):

```python
# transcribe.py:223-228
def __call__(self, x, pos_emb):
  x = x + self.feed_forward1(self.norm_feed_forward1(x)) * 0.5   # FFN (half)
  x = x + self.self_attn(self.norm_self_att(x), pos_emb)         # attention
  x = x + self.conv(self.norm_conv(x))                            # conv
  x = x + self.feed_forward2(self.norm_feed_forward2(x)) * 0.5   # FFN (half)
  return self.norm_out(x)
```

This is a "macaron" structure -- two half-weighted FFN layers sandwiching attention and convolution. Every sublayer uses a residual connection (`x = x + sublayer(x)`).

#### FFN (Feed-Forward Network)

```python
# transcribe.py:131-132
def __call__(self, x):
  return self.linear2(self.linear1(x).silu())  # 1024 → 4096 → 1024
```

Expand to 4x width, apply SiLU activation, project back. This is where the model stores "knowledge" -- the `[1024, 4096]` and `[4096, 1024]` weight matrices are the biggest parameters per layer.

#### Relative-Position Multi-Head Attention

This is self-attention with relative position encoding. Instead of adding absolute position to the input (like in vanilla transformers), the model adds position information directly to the attention scores. This is better for variable-length audio.

```python
# transcribe.py:144-163
# Standard Q, K, V projections, split into 8 heads of 128 dims each
q = self.linear_q(x).reshape(B, T, 8, 128)  # [B, 8, T, 128]
k = self.linear_k(x).reshape(B, T, 8, 128)
v = self.linear_v(x).reshape(B, T, 8, 128)

# Two attention terms:
content_attn = (q + pos_bias_u) @ k.T     # "what is at each position?"
pos_attn = (q + pos_bias_v) @ pos.T       # "how far apart are positions?"

attn = softmax((content_attn + pos_attn) / sqrt(128))
output = attn @ v                          # weighted sum of values
```

The content attention is standard Q*K^T (which frames should attend to which). The position attention adds awareness of relative distance -- "the frame 3 steps to my left" rather than "frame 47." The `_rel_shift` function (`transcribe.py:166-172`) is a clever reshaping trick that aligns the position scores correctly.

#### ConvModule

```python
# transcribe.py:185-209
# pointwise conv (1024 → 2048) + GLU gating (→ 1024)
# depthwise conv (kernel=9, causal padding)
# batch norm + SiLU
# pointwise conv (1024 → 1024)
```

Attention sees the whole sequence globally but can miss local patterns. The conv module scans a local window of 9 frames (~720ms) with a depthwise separable convolution. GLU (Gated Linear Unit) is the gating mechanism: split into two halves, one controls how much of the other passes through: `output = a * sigmoid(b)`.

The convolution is **causal** -- it only looks at past and current frames (left-padded by kernel_size-1). This allows the model to be used for streaming.

### Encoder Summary

After 24 conformer layers, each time step has been transformed from a raw acoustic feature into a rich representation that captures phonetic content, speaker characteristics, and context from surrounding audio.

**Input**: `[1, 999, 128]` mel spectrogram
**Output**: `[1, 125, 1024]` encoder features (8x fewer time steps, 8x wider vectors)

## Stage 4: Decoder (RNN-T)

Now we need to convert the encoder's output into text. This model uses **RNN-T** (Recurrent Neural Network Transducer), which is fundamentally different from the attention-based decoder you might know from seq2seq models.

### Why RNN-T?

In attention-based models (like Whisper), the decoder attends to the full encoder output at each step. This requires seeing the complete audio before generating any text.

RNN-T works differently: it processes encoder frames left-to-right, deciding at each frame whether to emit a token or stay silent (blank). This makes it naturally **streamable** -- it can start outputting text before the audio finishes.

### The Three Networks

RNN-T has three components:

```
encoder output ──→ [enc projection] ──→ enc_proj [640]
                                              ↓
                                    ┌─────────┴─────────┐
                                    │   joint network    │
                                    │  relu(enc + dec)   │
                                    │  → logits [1025]   │
                                    └─────────┬─────────┘
                                              ↑
previous token ──→ [embedding] ──→ [LSTM x2] ──→ dec_proj [640]
```

1. **Encoder projection**: Linear layer mapping each encoder frame from 1024 → 640 dims
2. **Prediction network**: Embedding + 2-layer LSTM that tracks what tokens have been emitted so far
3. **Joint network**: Combines encoder and decoder, outputs logits over 1025 classes (1024 tokens + blank)

### The LSTM

The prediction network is a 2-layer LSTM with hidden size 640. If you know RNNs, an LSTM cell computes:

```python
# transcribe.py:294-299
def _lstm_cell(x, h, c, w_ih, w_hh, b_ih, b_hh):
  gates = x @ w_ih.T + b_ih + h @ w_hh.T + b_hh   # [4*640]
  i, f, g, o = split_into_4_gates(gates)            # each [640]
  c_new = sigmoid(f) * c + sigmoid(i) * tanh(g)     # cell state update
  h_new = sigmoid(o) * tanh(c_new)                  # hidden state
  return h_new, c_new
```

Four gates from two matmuls:
- **Input gate** (i): what new info to store
- **Forget gate** (f): what old info to discard
- **Cell gate** (g): candidate new values
- **Output gate** (o): what to expose as output

The LSTM's job is to remember the history of emitted tokens. This is the "language model" part -- it knows that after seeing "the" and "cat", the next token is likely "sat" rather than "xqz".

### Greedy Decode Loop

Here's the actual decoding algorithm:

```python
# transcribe.py:316-342
for t in range(time_steps):          # for each encoder frame (~80ms of audio)
  enc_proj = project(encoder_out[t])  # what does the audio say here?

  for _ in range(10):                 # try emitting tokens at this frame
    emb = embed(prev_token)           # what was the last token?
    h0, c0 = lstm_layer_0(emb, h0, c0)
    h1, c1 = lstm_layer_1(h0, h1, c1)

    # joint: combine audio evidence + language model
    logits = relu(enc_proj + project(h1)) @ output_weight  # [1025]
    best = argmax(logits)

    if best == BLANK (1024):
      break                           # nothing to emit, move to next frame
    tokens.append(best)               # emit this token
    prev_token = best                 # feed it back to LSTM
```

The key insight: at each encoder frame, the model can emit **zero or more** tokens. Most frames emit blank (silence or continuation of a sound). When a frame does contain a new phoneme, the model might emit one or several subword tokens before returning to blank.

The inner loop has a cap of 10 -- in practice, a single 80ms frame almost never produces more than 2-3 tokens.

### The Joint Network

The joint network is surprisingly simple:

```python
logits = relu(enc_proj + dec_proj) @ W_out + b_out
```

It just adds the encoder projection and decoder projection, applies ReLU, then projects to vocab size. The addition is the key -- the encoder says "the audio sounds like X" and the decoder says "given the text so far, the next token should be Y" and they vote together.

## Stage 5: Detokenization

The model uses SentencePiece tokenization with 1024 subword tokens. Tokens starting with `▁` (Unicode 0x2581) indicate a word boundary:

```python
# transcribe.py:344-353
for tid in token_ids:
  piece = vocab[tid]
  if piece.startswith("▁"):     # word boundary marker
    text += " " + piece[1:]     # add space before word
  else:
    text += piece               # append subword
```

Example token sequence: `["▁the", "▁cat", "▁s", "at"]` → `"the cat sat"`

## End-to-End Example

Let's trace concrete tensor shapes through a 10-second audio:

```
WAV file (10s, 16kHz)
  → [160000] float32 samples

Pre-emphasis
  → [160000] filtered samples

STFT (512-sample frames, 160-sample hop)
  → [999, 257] power spectrum (999 frames x 257 freq bins)

Mel filterbank
  → [999, 128] mel spectrogram

Conv subsampling (3x stride-2)
  → [125, 1024] subsampled + projected

24 conformer layers
  → [125, 1024] encoder output

RNN-T greedy decode
  → [~50] token IDs (variable length)

Detokenize
  → "the cat sat on the mat"
```

## Parameter Count

Where do the 600M parameters live?

| component | params | notes |
|---|---|---|
| conv subsampling | ~1M | 3 conv layers + linear |
| conformer x24 | ~580M | bulk of the model |
| - FFN x2 per layer | ~8M each | 1024x4096 + 4096x1024 |
| - attention | ~5M | Q,K,V,pos,out projections |
| - conv module | ~2M | pointwise + depthwise |
| LSTM decoder | ~6M | 2 layers, 640 hidden |
| joint network | ~2M | enc proj + pred proj + output |
| embedding | ~0.7M | 1024 tokens x 640 dims |

The conformer FFN layers dominate -- 24 layers x 2 FFNs x ~8M params = ~384M, which is 64% of the model.

## What Makes This Different From Whisper?

| | Nemotron (this model) | Whisper |
|---|---|---|
| encoder | conformer (attn + conv) | transformer (attn only) |
| decoder | RNN-T (LSTM, greedy) | attention-based (autoregressive) |
| streaming | yes (causal convs) | no (needs full audio) |
| decode style | blank/emit per frame | cross-attention to full encoder |
| strength | low latency, streaming | better accuracy, multilingual |

The conformer's convolutions give it local pattern awareness that pure transformers lack. The RNN-T decoder enables streaming but makes decoding inherently sequential (each step depends on the previous token), which is why the decode loop is the bottleneck in GPU implementations.
