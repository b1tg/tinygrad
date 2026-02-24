# 第25章：Whisper — 语音识别

Whisper 是 OpenAI 的语音识别模型。给定音频，它生成文本。本章解释音频处理的工作原理，并逐步讲解 tinygrad 的 Whisper 实现。

## 语音识别的工作原理

处理流程：

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

## 第1步：音频转 Mel Spectrogram

原始音频是每秒采样 16,000 次的振幅值序列。我们需要将其转换为能够捕捉*随时间变化的频率内容*的表示形式。

### STFT（短时傅里叶变换）

STFT 在音频上滑动一个窗口，并在每个位置计算频谱：

```python
# Window: 400 samples (25ms at 16kHz)
# Hop: 160 samples (10ms) — the window advances by this much each step
# This gives 100 frames per second of audio
stft = librosa.stft(waveform, n_fft=400, hop_length=160, window='hann')
magnitudes = np.absolute(stft) ** 2
```

输出是一个二维矩阵：频率 bins × 时间步。

### Mel 尺度

人类的听觉是对数式的——我们感知 100Hz 和 200Hz 之间的差异与 1000Hz 和 2000Hz 之间的差异相同。Mel 尺度对此进行建模：

```python
# Project 201 frequency bins onto 80 mel-spaced bins
mel_spec = mel_filterbank @ magnitudes  # (80, time_steps)
```

这给出了 80 个感知均匀分布的频率通道。

### 对数压缩

```python
log_spec = np.log10(np.clip(mel_spec, 1e-10, None))
log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
log_spec = (log_spec + 4.0) / 4.0
```

取对数可以压缩动态范围。耳语和喊叫在振幅上相差 10,000 倍，但在对数尺度上仅相差约 4 倍。

### 结果

一段 30 秒的音频片段变成一个 `(80, 3000)` 的张量——80 个频率 bins，3000 个时间步（每 10ms 一个）。可以把它想象成一张图像，x 轴是时间，y 轴是频率。

## 第2步：Audio Encoder

Encoder 将 mel spectrogram 处理为一系列特征向量：

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

要点：
- 两个一维卷积压缩 mel spectrogram。conv2 中的 stride=2 将时间维度减半（3000 → 1500 帧）。
- 正弦位置编码告诉 transformer 每一帧在时间上的位置。
- 自注意力模块让每一帧都能关注其他所有帧，捕捉长距离依赖关系（例如，单词中间的停顿）。

## 第3步：Text Decoder

Decoder 逐个生成文本 token，同时关注编码后的音频：

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

Decoder 模块包含**自注意力**（文本关注文本）和**交叉注意力**（文本关注音频）：

```python
class ResidualAttentionBlock:
    def __call__(self, x, xa=None):
        x = x + self.attn(self.attn_ln(x))              # self-attention (causal)
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa)  # cross-attention to audio
        x = x + self.mlp_ln(x).sequential(self.mlp)     # feed-forward
        return x
```

Cross-attention 使用来自文本的 query 和来自 audio encoder 输出的 key/value。这就是 decoder 在生成文本时"查看"音频的方式。

### KV 缓存

与 LLaMA 类似，Whisper decoder 缓存 key 和 value 以实现高效的自回归生成：

```python
if self.kv_caching == 'self':
    # Append new KV to cache
    k = self.cache_k[:, :len].cat(k, dim=1)
    v = self.cache_v[:, :len].cat(v, dim=1)
    # Update cache
    self.cache_k.assign(k.pad((None, (0, padding), None)))
    self.cache_v.assign(v.pad((None, (0, padding), None)))
```

对于 cross-attention，缓存只设置一次（来自音频），然后在每个文本 token 生成时复用：

```python
if self.kv_caching == 'cross':
    if xa is not None:  # first call: compute and cache
        self.cache_k, self.cache_v = self.key(xa), self.value(xa)
    else:               # subsequent calls: use cache
        k, v = self.cache_k, self.cache_v
```

## 解码循环

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

### 特殊 Token

Whisper 使用特殊 token 来控制行为：

```
<|startoftranscript|>  — begin transcription
<|en|>                 — language (English)
<|transcribe|>         — task (transcribe vs translate)
<|notimestamps|>       — no timestamp markers
<|endoftext|>          — stop generating
```

对于多语言模型，语言 token 用于选择目标语言。

## Whisper 模型大小

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

更大的模型更准确但更慢。`tiny.en` 即使在 CPU 上也能实时运行。

## 实时转录

Whisper 可以从麦克风实时转录：

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

## 运行 Whisper

```bash
# Transcribe a file
python examples/whisper.py audio_file.wav

# Use the small model for better accuracy
SMALL=1 python examples/whisper.py audio_file.wav

# Live transcription (requires pyaudio)
python examples/whisper.py
```

## Encoder-Decoder 与 Decoder-Only 架构对比

Whisper 使用 **encoder-decoder** 架构（与原始 Transformer 论文相同）。这与 GPT/LLaMA 的 **decoder-only** 架构不同：

| | Encoder-Decoder | Decoder-Only |
|--|----------------|--------------|
| 用途 | 翻译、语音识别 | 文本生成 |
| 输入 | 由 encoder 处理 | 与输出拼接 |
| 注意力 | 从 decoder 到 encoder 的 cross-attention | 仅自注意力 |
| 示例 | Whisper, T5, BERT | GPT-2, LLaMA |

Encoder-decoder 模型天然适合处理输入（音频）和输出（文本）属于根本不同模态的任务。

## 练习

1. **转录音频**：录制或下载一个 WAV 文件，运行 `python examples/whisper.py your_file.wav`。

2. **比较模型大小**：用 `tiny.en` 和 `SMALL=1` 转录同一段音频。比较准确率和速度。

3. **追踪架构**：在 Whisper 模型中，"tiny" 模型的 encoder 有多少个 transformer 层？decoder 有多少个？（提示：查看 `dims` 字典。）

4. **理解 cross-attention**：在 `ResidualAttentionBlock` 中，cross-attention 的 query 来自哪里？key 和 value 来自哪里？

5. **Spectrogram**：Mel spectrogram 对于 30 秒音频的形状为 `(80, 3000)`。每个时间步代表多少毫秒？每个 mel bin 大约跨越多少 Hz？

## 源代码索引

| 文件 | 阅读内容 |
|------|---------|
| `examples/whisper.py` | 完整的 Whisper 流程（模型、分词器、音频处理、生成） |
| `examples/audio_helpers.py` | Mel 滤波器组计算 |
| `extra/models/rnnt.py` | RNN-T — 另一种语音识别架构 |
