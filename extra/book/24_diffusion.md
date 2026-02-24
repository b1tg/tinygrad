# Chapter 24: Stable Diffusion — Generating Images from Text

Stable Diffusion generates images from text prompts. This chapter explains the diffusion process, the VAE, the UNet, and CLIP — the four components that make it work.

## How Diffusion Works

The core idea is simple: start with random noise and gradually remove it to get an image.

### Training (Forward Process)

Take a real image and progressively add Gaussian noise over T steps until it becomes pure noise:

```
Step 0:   clear image of a cat
Step 250: slightly noisy cat
Step 500: very noisy, barely recognizable
Step 750: almost pure noise
Step 1000: pure Gaussian noise
```

### Inference (Reverse Process)

A neural network learns to predict and remove the noise at each step:

```
Step 1000: pure noise -> model predicts noise -> subtract -> slightly less noisy
Step 750:  model predicts noise -> subtract -> some structure visible
Step 500:  model predicts noise -> subtract -> cat-like shape
Step 250:  model predicts noise -> subtract -> clear cat
Step 0:    final image
```

### The Math

At each timestep `t`, the noisy image is:

```
x_t = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t) * noise
```

where `alpha_t` decreases from 1 to ~0 as `t` goes from 0 to 1000. The model predicts the noise component, and we solve for `x_0`.

```python
def get_alphas_cumprod(beta_start=0.00085, beta_end=0.0120, n_training_steps=1000):
    betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, n_training_steps) ** 2
    alphas = 1.0 - betas
    return Tensor(np.cumprod(alphas))
```

## The Three Components

Stable Diffusion doesn't operate directly on images. Instead, it works in a compressed **latent space**:

```
Text Prompt ──> CLIP ──> text embedding (77, 768)
                              │
Random noise (4, 64, 64)──> UNet ──> denoised latent (4, 64, 64)
                              │
                           Decoder ──> image (3, 512, 512)
```

### 1. CLIP: Understanding the Text

CLIP (Contrastive Language-Image Pre-Training) converts text into a vector that captures its meaning:

```python
# Tokenize the prompt
tokenizer = Tokenizer.ClipTokenizer()
tokens = Tensor([tokenizer.encode("a horse sized cat eating a bagel")])

# Run through CLIP text encoder
context = model.cond_stage_model.transformer.text_model(tokens)
print(context.shape)  # (1, 77, 768) — 77 token positions, 768 dimensions
```

The text encoder is a transformer (similar to GPT-2 but trained jointly with an image encoder on millions of image-text pairs). The resulting 768-dimensional vectors tell the UNet *what* to generate.

**Classifier-Free Guidance**: To strengthen the text's influence, we run the UNet twice — once with the prompt and once without — then amplify the difference:

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

Higher `guidance_scale` (default 7.5) makes the image follow the text more closely but reduces diversity.

### 2. UNet: The Denoiser

The UNet predicts the noise to subtract at each step. It has an encoder-decoder structure with skip connections:

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

The UNet has three types of blocks:

**ResBlocks** handle the spatial processing:

```python
class ResBlock:
    def __init__(self, channels, emb_channels, out_channels):
        self.in_layers = [GroupNorm(32, channels), lambda x: x.silu(), Conv2d(channels, out_channels, 3, padding=1)]
        self.emb_layers = [lambda x: x.silu(), Linear(emb_channels, out_channels)]
        self.out_layers = [GroupNorm(32, out_channels), lambda x: x.silu(), Conv2d(out_channels, out_channels, 3, padding=1)]
```

**CrossAttention** blocks let the UNet attend to the text embedding:

```python
class CrossAttention:
    def __call__(self, x, context):
        q = self.to_q(x)                # query from the image features
        k = self.to_k(context)           # key from the text
        v = self.to_v(context)           # value from the text
        return self.to_out(q.scaled_dot_product_attention(k, v))
```

This is how the text guides the image generation — the UNet's features (queries) attend to the text description (keys/values).

**Timestep Embedding** tells the UNet which denoising step it's at:

```python
def timestep_embedding(timesteps, dim):
    half = dim // 2
    freqs = (-math.log(10000) * Tensor.arange(half) / half).exp()
    args = timesteps * freqs
    return Tensor.cat(args.cos(), args.sin())
```

### 3. VAE: Compressing to Latent Space

Instead of working with full 512x512 images (786,432 values), the VAE compresses to 64x64 latents (16,384 values) — a 48x reduction:

```python
class AutoencoderKL:
    def __init__(self):
        self.encoder = Encoder()    # 3x512x512 -> 4x64x64
        self.decoder = Decoder()    # 4x64x64 -> 3x512x512
```

The encoder maps images to a 4-channel latent space. The decoder reconstructs images from latents. During generation, we only use the decoder (the UNet works entirely in latent space).

The decoder is a series of ResNet blocks and upsampling layers:

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

## The Generation Pipeline

Here's the full pipeline from `examples/stable_diffusion.py`:

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

### Why Few Steps Work

The original diffusion paper uses 1000 steps. Stable Diffusion uses 6-50 steps with DDIM or DPM++ samplers that skip steps intelligently. The `get_x_prev_and_pred_x0` method implements this:

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

## Running Stable Diffusion

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

## Stable Diffusion Variants

Tinygrad includes multiple versions:

| Version | File | Resolution | Text Encoder |
|---------|------|-----------|-------------|
| SD 1.4 | `examples/stable_diffusion.py` | 512x512 | CLIP ViT-L/14 |
| SD 2.0 | `examples/sdv2.py` | 768x768 | OpenCLIP ViT-H/14 |
| SDXL | `examples/sdxl.py` | 1024x1024 | Dual CLIP |

## Exercises

1. **Generate images**: Run Stable Diffusion with different prompts and seeds. How does `--guidance` affect the output?

2. **Understand the UNet**: In `extra/models/unet.py`, trace how the timestep embedding flows through the network. Which layers receive it?

3. **Latent space**: The latent is `(4, 64, 64)` for a `(3, 512, 512)` image. What is the spatial compression ratio? What is the total compression ratio?

4. **CrossAttention**: Find the CrossAttention block in the UNet. What are the query, key, and value? Where does the text embedding enter?

5. **Compare steps**: Generate the same image with 4, 8, 20, and 50 steps. At what point does quality plateau?

## Source Code Map

| File | What to read |
|------|-------------|
| `examples/stable_diffusion.py` | Full SD v1 pipeline |
| `examples/sdxl.py` | SDXL pipeline |
| `extra/models/unet.py` | UNet model (ResBlock, CrossAttention, SpatialTransformer) |
| `extra/models/clip.py` | CLIP text encoder |
