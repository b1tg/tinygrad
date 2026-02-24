# Chapter 25: Whisper — Speech Recognition

Whisper is OpenAI's speech recognition model. Given audio, it produces text. This chapter explains how audio processing works and walks through tinygrad's Whisper implementation.

## How Speech Recognition Works

The pipeline:

```
Audio waveform (16kHz samples)
    │
    ├─→ Mel spectrogram (80 frequency bins × 3000 time steps)
    │
    ├─→ Audio Encoder (transformer)
    │       outputs: (1, 1500, 512)
    │
    └─→ Text Decoder (transformer, autoregressively generates text)
            outputs: tokens → "the cat sat on the mat"
```

## Step 1: Audio to Mel Spectrogram

Raw audio is a sequence of amplitude values sampled 16,000 times per second. We need to convert this to a representation that captures *frequency content over time*.

### STFT (Short-Time Fourier Transform)

The STFT slides a window across the audio and computes the frequency spectrum at each position:

```python
# Window: 400 samples (25ms at 16kHz)
# Hop: 160 samples (10ms) — the window advances by this much each step
# This gives 100 frames per second of audio
stft = librosa.stft(waveform, n_fft=400, hop_length=160, window='hann')
magnitudes = np.absolute(stft) ** 2
```

The output is a 2D matrix: frequency bins × time steps.

### Mel Scale

Humans hear logarithmically — we perceive the difference between 100Hz and 200Hz the same as 1000Hz and 2000Hz. The Mel scale models this:

```python
# Project 201 frequency bins onto 80 mel-spaced bins
mel_spec = mel_filterbank @ magnitudes  # (80, time_steps)
```

This gives us 80 perceptually-spaced frequency channels.

### Log Compression

```python
log_spec = np.log10(np.clip(mel_spec, 1e-10, None))
log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
log_spec = (log_spec + 4.0) / 4.0
```

Taking the log compresses the dynamic range. A whisper and a shout differ by 10,000x in amplitude, but only ~4x in log scale.

### The Result

A 30-second audio clip becomes an `(80, 3000)` tensor — 80 frequency bins, 3000 time steps (one every 10ms). Think of it like an image where the x-axis is time and the y-axis is frequency.

## Step 2: Audio Encoder

The encoder processes the mel spectrogram into a sequence of feature vectors:

```python
class AudioEncoder:
    def __init__(self, n_mels, n_audio_ctx, n_audio_state, n_audio_head, n_audio_layer):
        self.conv1 = nn.Conv1d(n_mels, n_audio_state, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(n_audio_state, n_audio_state, kernel_size=3, stride=2, padding=1)
        self.blocks = [ResidualAttentionBlock(n_audio_state, n_audio_head)
                       for _ in range(n_audio_layer)]
        self.ln_post = nn.LayerNorm(n_audio_state)
        self.positional_embedding = Tensor.empty(n_audio_ctx, n_audio_state)

    def __call__(self, x):
        x = self.conv1(x).gelu()       # (80, 3000) -> (512, 3000)
        x = self.conv2(x).gelu()       # (512, 3000) -> (512, 1500) [stride=2 halves time]
        x = x.permute(0, 2, 1)         # (B, 1500, 512) — now a sequence of 1500 vectors
        x = x + self.positional_embedding[:x.shape[1]]
        x = x.sequential(self.blocks)  # transformer layers
        return self.ln_post(x)
```

Key points:
- Two 1D convolutions compress the mel spectrogram. The stride=2 in conv2 halves the time dimension (3000 → 1500 frames).
- Sinusoidal positional embeddings tell the transformer where each frame is in time.
- Self-attention blocks let every frame attend to every other frame, capturing long-range dependencies (e.g., a pause mid-word).

## Step 3: Text Decoder

The decoder generates text tokens one at a time, attending to the encoded audio:

```python
class TextDecoder:
    def __init__(self, n_vocab, n_text_ctx, n_text_state, n_text_head, n_text_layer):
        self.token_embedding = nn.Embedding(n_vocab, n_text_state)
        self.positional_embedding = Tensor.empty(n_text_ctx, n_text_state)
        self.blocks = [ResidualAttentionBlock(n_text_state, n_text_head,
                       is_decoder_block=True) for _ in range(n_text_layer)]
        self.ln = nn.LayerNorm(n_text_state)

    def forward(self, x, pos, encoded_audio):
        x = self.token_embedding(x) + self.positional_embedding[pos:pos+seqlen]
        for block in self.blocks:
            x = block(x, xa=encoded_audio)  # cross-attention to audio!
        return (self.ln(x) @ self.token_embedding.weight.T)
```

### Cross-Attention

The decoder blocks have **self-attention** (text attends to text) and **cross-attention** (text attends to audio):

```python
class ResidualAttentionBlock:
    def __call__(self, x, xa=None):
        x = x + self.attn(self.attn_ln(x))              # self-attention (causal)
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa)  # cross-attention to audio
        x = x + self.mlp_ln(x).sequential(self.mlp)     # feed-forward
        return x
```

Cross-attention uses queries from the text and keys/values from the audio encoder output. This is how the decoder "looks at" the audio while generating text.

### KV Caching

Like LLaMA, the Whisper decoder caches keys and values for efficient autoregressive generation:

```python
if self.kv_caching == 'self':
    # Append new KV to cache
    k = self.cache_k[:, :len].cat(k, dim=1)
    v = self.cache_v[:, :len].cat(v, dim=1)
    # Update cache
    self.cache_k.assign(k.pad((None, (0, padding), None)))
    self.cache_v.assign(v.pad((None, (0, padding), None)))
```

For cross-attention, the cache is set once (from the audio) and reused for every text token:

```python
if self.kv_caching == 'cross':
    if xa is not None:  # first call: compute and cache
        self.cache_k, self.cache_v = self.key(xa), self.value(xa)
    else:               # subsequent calls: use cache
        k, v = self.cache_k, self.cache_v
```

## The Decoding Loop

```python
def transcribe_waveform(model, enc, waveforms):
    # 1. Compute mel spectrogram
    log_spec = prep_audio(waveforms, model.batch_size)

    # 2. Process 30-second segments
    for curr_frame in range(0, log_spec.shape[-1], 3000):
        # Encode this segment of audio
        encoded_audio = model.encoder.encode(
            Tensor(log_spec[:, :, curr_frame:curr_frame + 3000]))

        # 3. Autoregressive decoding
        start_tokens = [startoftranscript, notimestamps]
        ctx = np.array([start_tokens])
        for i in range(max_tokens):
            logits = model.decoder(Tensor(ctx), pos, encoded_audio)
            next_token = logits[:, -1].argmax(axis=-1).numpy()
            ctx = np.concatenate((ctx, next_token), axis=1)
            if next_token == endoftext: break

    return enc.decode(ctx)
```

### Special Tokens

Whisper uses special tokens to control behavior:

```
<|startoftranscript|>  — begin transcription
<|en|>                 — language (English)
<|transcribe|>         — task (transcribe vs translate)
<|notimestamps|>       — no timestamp markers
<|endoftext|>          — stop generating
```

For multilingual models, the language token selects the target language.

## Whisper Model Sizes

```python
MODEL_URLS = {
    "tiny.en":  "...",    # 39M params, English only
    "tiny":     "...",    # 39M params, multilingual
    "base.en":  "...",    # 74M params
    "small.en": "...",    # 244M params
    "medium":   "...",    # 769M params
    "large-v2": "...",    # 1550M params
}
```

Larger models are more accurate but slower. `tiny.en` runs in real-time even on CPU.

## Live Transcription

Whisper can transcribe from a microphone in real-time:

```python
# From examples/whisper.py
if len(sys.argv) > 1:
    # Transcribe a file
    print(transcribe_file(model, enc, sys.argv[1]))
else:
    # Live from microphone
    # Records audio chunks, feeds them to the model continuously
    p = multiprocessing.Process(target=listener, args=(q,))
    # ... processes audio chunks as they arrive
```

## Running Whisper

```bash
# Transcribe a file
python examples/whisper.py audio_file.wav

# Use the small model for better accuracy
SMALL=1 python examples/whisper.py audio_file.wav

# Live transcription (requires pyaudio)
python examples/whisper.py
```

## Encoder-Decoder vs Decoder-Only

Whisper uses an **encoder-decoder** architecture (like the original Transformer paper). This is different from GPT/LLaMA which are **decoder-only**:

| | Encoder-Decoder | Decoder-Only |
|--|----------------|--------------|
| Use | Translation, ASR | Text generation |
| Input | Processed by encoder | Concatenated with output |
| Attention | Cross-attention from decoder to encoder | Self-attention only |
| Examples | Whisper, T5, BERT | GPT-2, LLaMA |

Encoder-decoder models naturally handle tasks where the input (audio) and output (text) are fundamentally different modalities.

## Exercises

1. **Transcribe audio**: Record or download a WAV file and run `python examples/whisper.py your_file.wav`.

2. **Compare model sizes**: Transcribe the same audio with `tiny.en` and `SMALL=1`. Compare accuracy and speed.

3. **Trace the architecture**: In the Whisper model, how many transformer layers does the encoder have vs the decoder for the "tiny" model? (Hint: check the `dims` dictionary.)

4. **Understand cross-attention**: In `ResidualAttentionBlock`, where do the queries come from in cross-attention? Where do keys and values come from?

5. **Spectrogram**: The mel spectrogram has shape `(80, 3000)` for 30 seconds of audio. How many milliseconds does each time step represent? How many Hz does each mel bin approximately span?

## Source Code Map

| File | What to read |
|------|-------------|
| `examples/whisper.py` | Full Whisper pipeline (model, tokenizer, audio processing, generation) |
| `examples/audio_helpers.py` | Mel filterbank computation |
| `extra/models/rnnt.py` | RNN-T — alternative speech recognition architecture |
