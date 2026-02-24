# Chapter 23: Large Language Models — LLaMA

This chapter covers how production LLMs work, using tinygrad's LLaMA 3 implementation. If you understood GPT-2 from the previous chapter, LLaMA adds three key improvements: RoPE, Grouped-Query Attention, and SwiGLU.

## GPT-2 vs LLaMA: What Changed?

| Feature | GPT-2 | LLaMA |
|---------|-------|-------|
| Norm | LayerNorm | RMSNorm |
| Position encoding | Learned embeddings | RoPE (rotary) |
| Activation | GELU | SwiGLU |
| Attention | Multi-head | Grouped-query (GQA) |
| FFN | 2 linear layers | 3 linear layers (gated) |
| Training data | 40GB text | Trillions of tokens |

## RoPE: Rotary Position Embeddings

GPT-2 adds learned position vectors. LLaMA uses **RoPE**, which encodes position through rotation in 2D subspaces.

The intuition: instead of adding a position vector, rotate the Query and Key vectors by an angle proportional to their position. Two tokens close together have similar rotations (small relative angle), making their attention score higher.

```python
def precompute_freqs_cis(dim, end, theta=10000.0):
    freqs = 1.0 / (theta ** (Tensor.arange(0, dim, 2)[:(dim // 2)] / dim))
    freqs = Tensor.arange(end).unsqueeze(1) * freqs.unsqueeze(0)
    # Returns cos and sin for each position and dimension pair
    return Tensor.stack(freqs.cos(), freqs.sin(), dim=-1).reshape(1, end, 1, dim//2, 2)
```

`theta = 10000` controls how quickly the rotations vary across dimensions. Low-dimensional pairs rotate fast (capturing local patterns), high-dimensional pairs rotate slowly (capturing long-range patterns).

### Applying Rotary Embeddings

RoPE treats consecutive pairs of dimensions as 2D coordinates and rotates them:

```python
def apply_rotary_emb(xq, xk, freqs_cis):
    # Reshape to pairs: (..., dim) -> (..., dim//2, 2)
    xq = xq.reshape(*xq.shape[:-1], -1, 2)
    xk = xk.reshape(*xk.shape[:-1], -1, 2)
    # Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    c, d = freqs_cis[..., 0:1], freqs_cis[..., 1:2]
    xq_out = complex_mult(xq, c, d)
    xk_out = complex_mult(xk, c, d)
    return xq_out.flatten(3), xk_out.flatten(3)
```

Why is this better than learned positions?
1. **Generalizes to longer sequences** — no fixed maximum length
2. **Relative position** — attention naturally depends on distance between tokens, not absolute positions
3. **No extra parameters** — the frequencies are computed, not learned

## RMSNorm

LLaMA uses RMSNorm instead of LayerNorm. It's simpler and faster — it normalizes by the root mean square, without centering:

```python
class RMSNorm:
    def __init__(self, dim, eps=1e-6):
        self.eps = eps
        self.weight = Tensor.ones(dim)

    def __call__(self, x):
        # Normalize by RMS (no mean subtraction)
        rms = (x.square().mean(-1, keepdim=True) + self.eps).rsqrt()
        return x * rms * self.weight
```

In practice, RMSNorm gives similar quality to LayerNorm with fewer operations.

## SwiGLU Feed-Forward

LLaMA's FFN uses a gated architecture with three weight matrices:

```python
class FeedForward:
    def __init__(self, dim, hidden_dim):
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)   # gate projection
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)    # down projection
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)    # up projection

    def __call__(self, x):
        return self.w2(self.w1(x).silu() * self.w3(x))
```

The gating mechanism (`w1(x).silu() * w3(x)`) lets the network learn which parts of the hidden representation to keep. SiLU (Sigmoid Linear Unit, also called Swish) is `x * sigmoid(x)`.

## Grouped-Query Attention (GQA)

Standard multi-head attention has separate K and V projections for each head. GQA shares K and V across groups of heads:

```python
class Attention:
    def __init__(self, dim, n_heads, n_kv_heads):
        self.n_heads = n_heads          # e.g., 32 query heads
        self.n_kv_heads = n_kv_heads    # e.g., 8 key/value heads
        self.n_rep = n_heads // n_kv_heads  # 4 query heads per KV head

        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * head_dim, bias=False)  # fewer!
        self.wv = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
```

With 32 query heads and 8 KV heads, each KV head is shared by 4 query heads. This reduces the KV cache size by 4x — critical for inference with long sequences.

The keys and values are repeated to match the query head count:

```python
def repeat_kv(x, n_rep):
    if n_rep == 1: return x
    return x.repeat((1, 1, 1, n_rep)).reshape(bs, seqlen, n_kv_heads * n_rep, head_dim)
```

## LLaMA Model Sizes

```python
MODEL_PARAMS = {
    "1B":   {"dim": 2048,  "n_heads": 32,  "n_kv_heads": 8,  "n_layers": 16,  "hidden_dim": 8192},
    "8B":   {"dim": 4096,  "n_heads": 32,  "n_kv_heads": 8,  "n_layers": 32,  "hidden_dim": 14336},
    "70B":  {"dim": 8192,  "n_heads": 64,  "n_kv_heads": 8,  "n_layers": 80,  "hidden_dim": 28672},
    "405B": {"dim": 16384, "n_heads": 128, "n_kv_heads": 8,  "n_layers": 126, "hidden_dim": 53248},
}
```

Notice `n_kv_heads=8` for all sizes — GQA keeps the KV cache manageable even for 405B.

## Quantization

Full-precision weights are expensive. A 70B model at float16 needs 140 GB of memory. Quantization reduces this:

### Int8 Quantization

Scale each weight row to fit in int8 (-128 to 127):

```python
class Int8Linear:
    def __init__(self, in_features, out_features):
        self.weight = Tensor.ones(out_features, in_features, dtype=dtypes.int8)
        self.scale = Tensor.ones(out_features, dtype=dtypes.half)

    def __call__(self, x):
        return x.dot(self.weight.cast(self.scale.dtype).T * self.scale)
```

Quantization:
```python
scale = v.abs().max(axis=1) / 127.0
int8_weight = (v.T / scale).T.round().cast(dtypes.int8)
```

This gives ~2x memory reduction with minimal quality loss.

### NF4 Quantization

NF4 (4-bit NormalFloat) packs two weights per byte using a 16-value lookup table:

```python
CODE = [-1.0, -0.696, -0.525, ..., 0.723, 1.0]  # 16 optimal values
```

Each weight is quantized to the nearest of these 16 values. This gives ~4x memory reduction, enough to run 70B models on consumer GPUs.

### FP8 Quantization

FP8 uses an 8-bit floating point format with 4 exponent bits and 3 mantissa bits:

```python
def quantize_to_fp8(x, dtype=dtypes.fp8e4m3):
    scale = 448.0 / x.abs().max()
    return (x * scale).clamp(-448.0, 448.0).cast(dtype), scale.reciprocal()
```

## Running LLaMA

### Interactive Chat

```bash
python examples/llama3.py --size 1B
# Downloads the model and starts a chat interface
```

### API Server

```bash
python examples/llama3.py --size 8B --port 7776
# Starts an OpenAI-compatible API server at localhost:7776
```

### Multi-GPU

```bash
python examples/llama3.py --size 70B --shard 6 --quantize int8
# Shards the 70B model across 6 GPUs with int8 quantization
```

## The Generation Loop

```python
# 1. Prefill: process the entire prompt
start_pos = prefill(model, prompt_tokens)

# 2. Generate: one token at a time
while True:
    tok = model(Tensor([[last_tok]]), start_pos, temperature)
    start_pos += 1
    last_tok = tok.item()
    if tok in stop_tokens: break
    print(tokenizer.decode([tok]), end="", flush=True)
```

**Prefill** processes all prompt tokens at once (parallel). **Generation** produces one token per step (sequential). The KV cache stores intermediate results so each generation step only processes the new token.

## Chat Template

LLaMA 3 uses a specific format for multi-turn conversations:

```
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|>
<|start_header_id|>user<|end_header_id|>

What is 2+2?<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
```

These special tokens are handled by the `Tokenizer` class in `examples/llama3.py`.

## Weight Loading

Tinygrad supports multiple weight formats:

```python
# Safetensors (preferred)
from tinygrad.nn.state import safe_load
weights = safe_load("model.safetensors")

# PyTorch checkpoints
from tinygrad.nn.state import torch_load
weights = torch_load("model.pth")

# GGUF (quantized)
from tinygrad.nn.state import gguf_load
kv_data, weights = gguf_load(tensor)

# Load into model
from tinygrad.nn.state import load_state_dict
load_state_dict(model, weights)
```

## Exercises

1. **Run LLaMA 1B**: `python examples/llama3.py --size 1B --no_api`. Chat with it. Notice the speed difference between prefill and generation.

2. **Compare quantization**: Run the same prompt with `--quantize int8` and without. Can you tell the difference in output quality?

3. **Read the attention**: In `extra/models/llama.py`, find where the KV cache is created and updated. What shape is it? How does GQA reduce its size?

4. **Understand RoPE**: Compute `precompute_freqs_cis(64, 10)` and print the result. How do the frequencies differ between the first and last dimension pairs?

5. **Measure throughput**: Run with `--benchmark`. How many tokens per second does your hardware achieve?

## Source Code Map

| File | What to read |
|------|-------------|
| `examples/llama3.py` | LLaMA 3 inference, quantization, API server |
| `extra/models/llama.py` | `Transformer`, `Attention`, `FeedForward`, RoPE |
| `tinygrad/nn/state.py` | `safe_load`, `torch_load`, `gguf_load`, `load_state_dict` |
| `tinygrad/nn/__init__.py` | `RMSNorm`, `Embedding` |
