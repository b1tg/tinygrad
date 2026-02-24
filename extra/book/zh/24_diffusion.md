# 第24章：Stable Diffusion — 从文本生成图像

Stable Diffusion 从文本提示生成图像。本章介绍扩散过程、VAE、UNet 和 CLIP — 使其工作的四个组件。

## 扩散的工作原理

核心思想很简单：从随机噪声开始，逐步去除噪声以获得图像。

### 训练（前向过程）

取一张真实图像，在 T 步内逐步添加高斯噪声，直到它变成纯噪声：

```
Step 0:   clear image of a cat
Step 250: slightly noisy cat
Step 500: very noisy, barely recognizable
Step 750: almost pure noise
Step 1000: pure Gaussian noise
```

### 推理（逆向过程）

神经网络学习在每一步预测并去除噪声：

```
Step 1000: pure noise -> model predicts noise -> subtract -> slightly less noisy
Step 750:  model predicts noise -> subtract -> some structure visible
Step 500:  model predicts noise -> subtract -> cat-like shape
Step 250:  model predicts noise -> subtract -> clear cat
Step 0:    final image
```

### 数学原理

在每个时间步 `t`，带噪声的图像为：

```
x_t = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t) * noise
```

其中 `alpha_t` 随着 `t` 从 0 到 1000 而从 1 递减到约 0。模型预测噪声分量，然后我们求解 `x_0`。

```python
def get_alphas_cumprod(beta_start=0.00085, beta_end=0.0120, n_training_steps=1000):
    betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, n_training_steps) ** 2
    alphas = 1.0 - betas
    return Tensor(np.cumprod(alphas))
```

## 三个组件

Stable Diffusion 不直接在图像上操作，而是在压缩的**潜在空间**中工作：

```
Text Prompt ──> CLIP ──> text embedding (77, 768)
                              │
Random noise (4, 64, 64)──> UNet ──> denoised latent (4, 64, 64)
                              │
                           Decoder ──> image (3, 512, 512)
```

### 1. CLIP：理解文本

CLIP（Contrastive Language-Image Pre-Training）将文本转换为捕获其含义的向量：

```python
# Tokenize the prompt
tokenizer = Tokenizer.ClipTokenizer()
tokens = Tensor([tokenizer.encode("a horse sized cat eating a bagel")])

# Run through CLIP text encoder
context = model.cond_stage_model.transformer.text_model(tokens)
print(context.shape)  # (1, 77, 768) — 77 token positions, 768 dimensions
```

文本编码器是一个 Transformer（类似于 GPT-2，但与图像编码器在数百万图文对上联合训练）。生成的 768 维向量告诉 UNet *要生成什么*。

**无分类器引导（Classifier-Free Guidance）**：为了增强文本的影响力，我们将 UNet 运行两次 — 一次使用提示，一次不使用 — 然后放大差异：

```python
def get_model_output(self, unconditional_context, context, latent, timestep, guidance_scale):
    # Run both contexts through the same UNet
    latents = self.model.diffusion_model(
        latent.expand(2, *latent.shape[1:]),
        timestep,
        unconditional_context.cat(context, dim=0)
    )
    unconditional, conditional = latents[0:1], latents[1:2]

    # Amplify the text's effect
    return unconditional + guidance_scale * (conditional - unconditional)
```

更高的 `guidance_scale`（默认 7.5）使图像更紧密地遵循文本，但会降低多样性。

### 2. UNet：去噪器

UNet 预测每一步需要减去的噪声。它具有带跳跃连接的编码器-解码器结构：

```
Input: noisy latent (4, 64, 64) + timestep + text embedding
                │
        ┌───────┼───── Encoder ─────┐
        │  64x64 → 32x32 → 16x16 → 8x8
        │       ↓         ↓        ↓
        │  ┌── Skip ─── Skip ── Skip ──┐
        │  │                            │
        │  └── Decoder ────────────────┘
        │  8x8 → 16x16 → 32x32 → 64x64
        │
Output: predicted noise (4, 64, 64)
```

UNet 有三种类型的模块：

**ResBlock** 处理空间计算：

```python
class ResBlock:
    def __init__(self, channels, emb_channels, out_channels):
        self.in_layers = [GroupNorm(32, channels), lambda x: x.silu(), Conv2d(channels, out_channels, 3, padding=1)]
        self.emb_layers = [lambda x: x.silu(), Linear(emb_channels, out_channels)]
        self.out_layers = [GroupNorm(32, out_channels), lambda x: x.silu(), Conv2d(out_channels, out_channels, 3, padding=1)]
```

**CrossAttention** 模块让 UNet 关注文本嵌入：

```python
class CrossAttention:
    def __call__(self, x, context):
        q = self.to_q(x)                # query from the image features
        k = self.to_k(context)           # key from the text
        v = self.to_v(context)           # value from the text
        return self.to_out(q.scaled_dot_product_attention(k, v))
```

这就是文本引导图像生成的方式 — UNet 的特征（查询）关注文本描述（键/值）。

**时间步嵌入（Timestep Embedding）** 告诉 UNet 当前处于哪个去噪步骤：

```python
def timestep_embedding(timesteps, dim):
    half = dim // 2
    freqs = (-math.log(10000) * Tensor.arange(half) / half).exp()
    args = timesteps * freqs
    return Tensor.cat(args.cos(), args.sin())
```

### 3. VAE：压缩到潜在空间

VAE 不直接处理完整的 512x512 图像（786,432 个值），而是压缩到 64x64 的潜在表示（16,384 个值）— 压缩了 48 倍：

```python
class AutoencoderKL:
    def __init__(self):
        self.encoder = Encoder()    # 3x512x512 -> 4x64x64
        self.decoder = Decoder()    # 4x64x64 -> 3x512x512
```

编码器将图像映射到 4 通道的潜在空间。解码器从潜在表示重建图像。在生成过程中，我们只使用解码器（UNet 完全在潜在空间中工作）。

解码器由一系列 ResNet 模块和上采样层组成：

```python
class Decoder:
    def __call__(self, x):
        x = self.conv_in(x)          # 4 -> 512 channels
        x = self.mid(x)              # middle blocks

        for l in self.up[::-1]:      # progressively upsample
            for b in l['block']: x = b(x)
            if 'upsample' in l:
                # 2x nearest neighbor upsampling
                bs, c, py, px = x.shape
                x = x.reshape(bs, c, py, 1, px, 1).expand(bs, c, py, 2, px, 2).reshape(bs, c, py*2, px*2)
                x = l['upsample']['conv'](x)

        return self.conv_out(self.norm_out(x).swish())
```

## 生成流水线

以下是来自 `examples/stable_diffusion.py` 的完整流水线：

```python
# 1. Encode the text prompt with CLIP
context = model.cond_stage_model.transformer.text_model(prompt_tokens)
unconditional_context = model.cond_stage_model.transformer.text_model(empty_tokens)

# 2. Start with random noise in latent space
latent = Tensor.randn(1, 4, 64, 64)

# 3. Iteratively denoise
timesteps = list(range(1, 1000, 1000 // num_steps))
for timestep in reversed(timesteps):
    latent = model(unconditional_context, context, latent,
                   Tensor([timestep]), alphas[t], alphas_prev[t],
                   Tensor([guidance_scale]))

# 4. Decode the final latent to an image
image = model.decode(latent)  # (3, 512, 512) uint8

# 5. Save
from PIL import Image
Image.fromarray(image.numpy()).save("output.png")
```

### 为什么少量步骤就能工作

原始扩散论文使用 1000 步。Stable Diffusion 使用 DDIM 或 DPM++ 采样器，通过智能跳步仅需 6-50 步。`get_x_prev_and_pred_x0` 方法实现了这一点：

```python
def get_x_prev_and_pred_x0(self, x, e_t, a_t, a_prev):
    # Predict x_0 from current noisy x_t and predicted noise e_t
    pred_x0 = (x - (1 - a_t).sqrt() * e_t) / a_t.sqrt()

    # Compute direction pointing to x_t
    dir_xt = (1.0 - a_prev).sqrt() * e_t

    # Jump directly to x_{t-k} (skipping steps)
    x_prev = a_prev.sqrt() * pred_x0 + dir_xt
    return x_prev, pred_x0
```

## 运行 Stable Diffusion

```bash
# Generate an image (downloads ~4GB model on first run)
python examples/stable_diffusion.py \
    --prompt "a horse sized cat eating a bagel" \
    --steps 6 --seed 42

# Use float16 for faster inference / less memory
python examples/stable_diffusion.py \
    --prompt "a sunset over mountains" \
    --steps 20 --fp16
```

## Stable Diffusion 变体

tinygrad 包含多个版本：

| 版本 | 文件 | 分辨率 | 文本编码器 |
|---------|------|-----------|-------------|
| SD 1.4 | `examples/stable_diffusion.py` | 512x512 | CLIP ViT-L/14 |
| SD 2.0 | `examples/sdv2.py` | 768x768 | OpenCLIP ViT-H/14 |
| SDXL | `examples/sdxl.py` | 1024x1024 | Dual CLIP |

## 练习

1. **生成图像**：使用不同的提示和种子运行 Stable Diffusion。`--guidance` 如何影响输出？

2. **理解 UNet**：在 `extra/models/unet.py` 中，追踪时间步嵌入如何在网络中流动。哪些层接收了它？

3. **潜在空间**：对于 `(3, 512, 512)` 的图像，潜在表示为 `(4, 64, 64)`。空间压缩比是多少？总压缩比是多少？

4. **CrossAttention**：在 UNet 中找到 CrossAttention 模块。查询、键和值分别是什么？文本嵌入从哪里进入？

5. **比较步数**：分别使用 4、8、20 和 50 步生成相同的图像。质量在什么时候趋于稳定？

## 源代码索引

| 文件 | 阅读内容 |
|------|-------------|
| `examples/stable_diffusion.py` | 完整的 SD v1 流水线 |
| `examples/sdxl.py` | SDXL 流水线 |
| `extra/models/unet.py` | UNet 模型（ResBlock、CrossAttention、SpatialTransformer） |
| `extra/models/clip.py` | CLIP 文本编码器 |
