# Chapter 22: Transformers & GPT-2

The transformer is the architecture behind GPT, BERT, LLaMA, and nearly every modern AI system. This chapter explains the transformer from scratch and walks through tinygrad's GPT-2 implementation.

## The Key Idea: Attention

Consider translating "The cat sat on the mat" to French. When predicting the French word for "sat", the model needs to focus on "cat" (subject agreement) and "sat" (the word itself), but not "the" or "mat".

**Attention** lets the model dynamically decide which parts of the input to focus on.

### Scaled Dot-Product Attention

Given a sequence of tokens, we create three vectors for each token:
- **Query (Q)**: "What am I looking for?"
- **Key (K)**: "What do I contain?"
- **Value (V)**: "What information do I provide?"

```
Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V
```

Step by step:
1. `Q @ K.T` — compute similarity between every pair of tokens
2. `/ sqrt(d_k)` — scale to prevent softmax from saturating
3. `softmax(...)` — convert to probabilities (each row sums to 1)
4. `... @ V` — weighted sum of values

```python
from tinygrad import Tensor

# Sequence of 5 tokens, each with dimension 64
Q = Tensor.rand(1, 5, 64)
K = Tensor.rand(1, 5, 64)
V = Tensor.rand(1, 5, 64)

# Manual attention
scores = Q @ K.transpose(-2, -1) / (64 ** 0.5)  # (1, 5, 5)
weights = scores.softmax(-1)                      # (1, 5, 5)
output = weights @ V                              # (1, 5, 64)
```

### Multi-Head Attention

Instead of one attention function, we run multiple in parallel ("heads"), each looking at a different aspect of the data:

```python
from tinygrad import nn

class Attention:
    def __init__(self, dim, n_heads):
        self.c_attn = nn.Linear(dim, 3 * dim)  # Q, K, V in one projection
        self.c_proj = nn.Linear(dim, dim)
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

    def __call__(self, x):
        B, T, C = x.shape
        # Project to Q, K, V and split into heads
        qkv = self.c_attn(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = [qkv[:, :, i].transpose(1, 2) for i in range(3)]
        # (B, n_heads, T, head_dim) for each

        # Attention + combine heads
        out = q.scaled_dot_product_attention(k, v)
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.c_proj(out)
```

With 12 heads and dim=768, each head operates on 64-dimensional slices. One head might track syntax, another semantics, another position.

Tinygrad provides `Tensor.scaled_dot_product_attention()` which handles the math (and can use Flash Attention optimizations).

### Causal Masking

For language models, a token at position `t` must only attend to tokens at positions `0..t` (not future tokens). This is enforced with a **causal mask**:

```python
# Upper triangular mask filled with -infinity
mask = Tensor.full((T, T), float("-inf")).triu(1)
# After softmax, -inf becomes 0, preventing attention to future tokens
```

## The Transformer Block

A transformer block combines attention with a feed-forward network:

```python
class TransformerBlock:
    def __init__(self, dim, n_heads):
        self.attn = Attention(dim, n_heads)
        self.mlp = FeedForward(dim, 4 * dim)
        self.ln_1 = nn.LayerNorm(dim)
        self.ln_2 = nn.LayerNorm(dim)

    def __call__(self, x):
        x = x + self.attn(self.ln_1(x))   # attention + residual
        x = x + self.mlp(self.ln_2(x))    # feed-forward + residual
        return x
```

Key patterns:
- **Pre-norm**: LayerNorm is applied *before* attention/MLP (not after, as in the original paper). This is more stable for training.
- **Residual connections**: The `x + ...` pattern. Same idea as ResNet — gradients flow easily through the skip connection.

### Feed-Forward Network

The FFN is just two linear layers with an activation function:

```python
class FeedForward:
    def __init__(self, dim, hidden_dim):
        self.c_fc = nn.Linear(dim, hidden_dim)
        self.c_proj = nn.Linear(hidden_dim, dim)

    def __call__(self, x):
        return self.c_proj(self.c_fc(x).gelu())
```

This operates on each token independently (no inter-token interaction). The hidden dimension is typically 4x the model dimension, giving the network more capacity.

**GELU** (Gaussian Error Linear Unit) is like ReLU but smoother — it doesn't have a hard zero cutoff.

### Layer Normalization

LayerNorm normalizes each token's features to zero mean and unit variance:

```python
norm = nn.LayerNorm(768)
x = Tensor.rand(1, 10, 768)  # 10 tokens, 768 features each
y = norm(x)  # each token's 768 features are normalized
```

Unlike BatchNorm (which normalizes across the batch), LayerNorm normalizes across features. This makes it work with variable-length sequences.

## GPT-2: The Full Model

GPT-2 is a stack of transformer blocks with token + position embeddings:

```python
class Transformer:
    def __init__(self, dim, n_heads, n_layers, vocab_size, max_seq_len=1024):
        self.wte = nn.Embedding(vocab_size, dim)        # token embeddings
        self.wpe = nn.Embedding(max_seq_len, dim)       # position embeddings
        self.h = [TransformerBlock(dim, n_heads) for _ in range(n_layers)]
        self.ln_f = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
```

### Token Embeddings

Each word (or sub-word) in the vocabulary gets a learned vector:

```python
# Vocabulary: 50,257 tokens -> 768-dimensional vectors
wte = nn.Embedding(50257, 768)
# Token index 42 maps to a 768-dim vector
embedding = wte(Tensor([42]))  # shape: (1, 768)
```

### Position Embeddings

Attention has no sense of order — "cat sat mat" and "mat sat cat" look the same. Position embeddings add location information:

```python
wpe = nn.Embedding(1024, 768)
positions = Tensor.arange(0, seq_len)  # [0, 1, 2, ..., seq_len-1]
pos_emb = wpe(positions)  # (seq_len, 768)
```

Each position gets a learned vector. The input to the first transformer block is `token_embedding + position_embedding`.

### Forward Pass

```python
def forward(self, tokens, start_pos):
    tok_emb = self.wte(tokens)
    pos_emb = self.wpe(positions)
    h = tok_emb + pos_emb

    # Causal mask for parallel token processing
    mask = Tensor.full((seq_len, seq_len), float("-inf")).triu(1)

    for block in self.h:
        h = block(h, mask)

    logits = self.lm_head(self.ln_f(h))
    return logits  # (batch, seq_len, vocab_size)
```

The output `logits` is a score for every token in the vocabulary at every position. To predict the next token, take `logits[:, -1, :]` (the last position) and pick the highest score.

### GPT-2 Sizes

```python
MODEL_PARAMS = {
    'gpt2':        dict(n_layers=12, n_heads=12, dim=768),    # 124M params
    'gpt2-medium': dict(n_layers=24, n_heads=16, dim=1024),   # 350M params
    'gpt2-large':  dict(n_layers=36, n_heads=20, dim=1280),   # 774M params
    'gpt2-xl':     dict(n_layers=48, n_heads=25, dim=1600),   # 1558M params
}
```

Same architecture, just different sizes.

## The KV Cache

During generation, we produce one token at a time. Without caching, generating token `t` requires re-computing attention for all previous tokens. With the **KV cache**, we store the Keys and Values from previous steps:

```python
# First call (prompt): process all prompt tokens
# Stores K, V for each layer: cache_kv shape = (2, batch, max_context, n_heads, head_dim)

# Subsequent calls: process only the new token
# Read cached K, V; append new K, V; compute attention against all
self.cache_kv[:, :, start_pos:start_pos+1].assign(Tensor.stack(xk, xv))
keys = self.cache_kv[0][:, :start_pos+1]
values = self.cache_kv[1][:, :start_pos+1]
```

This makes generation O(n) per token instead of O(n^2).

## Tokenization

Language models don't work on words directly — they use **subword tokenization**. GPT-2 uses Byte-Pair Encoding (BPE):

```
"tinygrad is cool" -> [22714, 9744, 318, 3608]
```

Common words get a single token. Rare words are split into subwords. This gives a fixed vocabulary (50,257 for GPT-2) that can represent any text.

## Text Generation

Generation works by repeatedly predicting the next token:

```
Prompt: "The cat sat on"
Step 1: model("The cat sat on") -> "the"
Step 2: model("The cat sat on the") -> "mat"
Step 3: model("The cat sat on the mat") -> "."
...
```

### Temperature Sampling

Instead of always picking the most likely token (greedy), we can sample from the probability distribution:

```python
if temperature < 1e-6:
    ret = logits.argmax(-1)                    # greedy
else:
    ret = (logits / temperature).softmax().multinomial()  # sample
```

- `temperature = 0` → always pick the top token (deterministic)
- `temperature = 1` → sample according to model's distribution
- `temperature > 1` → more random (creative)
- `temperature < 1` → more focused (conservative)

## Running GPT-2

```bash
# Generate text with GPT-2 medium
python examples/gpt2.py --model_size gpt2-medium \
    --prompt "The meaning of life is" --count 50 --temperature 0.8
```

## A Toy Transformer: Learning Addition

Before tackling GPT-2, tinygrad includes a tiny transformer that learns to add two-digit numbers:

```python
# From examples/transformer.py
# Input:  "42 + 35" encoded as [4, 2, 3, 5]
# Output: "077" encoded as [0, 7, 7]

model = Transformer(10, 6, 2, 128, 4, 32)
# 10 symbols (digits 0-9), 6 positions, 2 layers, 128 dim, 4 heads, 32 ff_dim
```

This is a great starting point for understanding transformers — same architecture, tiny scale.

## Exercises

1. **Run GPT-2**: Generate text with different temperatures. How does output quality change?

2. **Trace attention**: In the GPT-2 `Attention` class, trace the tensor shapes through the forward pass for `batch=1, seq_len=10, dim=768, n_heads=12`.

3. **Count FLOPs**: For a forward pass with sequence length `T` and model dimension `d`, the attention computation is O(T^2 * d). For GPT-2 medium (d=1024, T=128), how many FLOPs is one attention layer?

4. **Run the toy transformer**: `python examples/transformer.py`. It learns to add two-digit numbers. Watch the accuracy increase.

5. **Understand the KV cache**: In `examples/gpt2.py`, find where `cache_kv` is created and updated. Why does the JIT only work for single-token inference (not prompt processing)?

## Source Code Map

| File | What to read |
|------|-------------|
| `examples/gpt2.py` | GPT-2 with KV cache and text generation |
| `examples/transformer.py` | Toy transformer that learns addition |
| `extra/models/transformer.py` | Generic TransformerBlock class |
| `tinygrad/nn/__init__.py` | `LayerNorm`, `Embedding`, `Linear` |
