# 第22章：Transformer 与 GPT-2

Transformer 是 GPT、BERT、LLaMA 以及几乎所有现代 AI 系统背后的架构。本章从零开始讲解 Transformer，并逐步解析 tinygrad 的 GPT-2 实现。

## 核心思想：Attention

考虑将 "The cat sat on the mat" 翻译成法语。在预测 "sat" 对应的法语单词时，模型需要关注 "cat"（主谓一致）和 "sat"（单词本身），而不需要关注 "the" 或 "mat"。

**Attention** 让模型能够动态地决定关注输入的哪些部分。

### Scaled Dot-Product Attention

给定一个 token 序列，我们为每个 token 创建三个向量：
- **Query (Q)**："我在寻找什么？"
- **Key (K)**："我包含什么？"
- **Value (V)**："我提供什么信息？"

```
Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V
```

逐步分解：
1. `Q @ K.T` —— 计算每对 token 之间的相似度
2. `/ sqrt(d_k)` —— 缩放以防止 softmax 饱和
3. `softmax(...)` —— 转换为概率（每行之和为 1）
4. `... @ V` —— 对 Value 进行加权求和

```python
from tinygrad import Tensor

# 5 个 token 的序列，每个维度为 64
Q = Tensor.rand(1, 5, 64)
K = Tensor.rand(1, 5, 64)
V = Tensor.rand(1, 5, 64)

# 手动计算 attention
scores = Q @ K.transpose(-2, -1) / (64 ** 0.5)  # (1, 5, 5)
weights = scores.softmax(-1)                      # (1, 5, 5)
output = weights @ V                              # (1, 5, 64)
```

### Multi-Head Attention

我们不使用单个 Attention 函数，而是并行运行多个（称为"头"），每个头关注数据的不同方面：

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

当有 12 个头且 dim=768 时，每个头在 64 维的切片上操作。一个头可能追踪语法，另一个追踪语义，还有一个追踪位置。

tinygrad 提供了 `Tensor.scaled_dot_product_attention()`，它封装了数学运算（并且可以使用 Flash Attention 优化）。

### Causal Masking

对于语言模型，位置 `t` 处的 token 只能关注位置 `0..t` 的 token（不能关注未来的 token）。这通过 **causal mask** 来实现：

```python
# Upper triangular mask filled with -infinity
mask = Tensor.full((T, T), float("-inf")).triu(1)
# After softmax, -inf becomes 0, preventing attention to future tokens
```

## Transformer Block

一个 Transformer block 将 Attention 与前馈网络结合在一起：

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

关键模式：
- **Pre-norm**：LayerNorm 在 Attention/MLP *之前*应用（而非原始论文中的之后）。这使训练更加稳定。
- **残差连接**：`x + ...` 模式。与 ResNet 相同的思路 —— 梯度通过跳跃连接可以轻松流动。

### 前馈网络

前馈网络只是两个线性层加一个激活函数：

```python
class FeedForward:
    def __init__(self, dim, hidden_dim):
        self.c_fc = nn.Linear(dim, hidden_dim)
        self.c_proj = nn.Linear(hidden_dim, dim)

    def __call__(self, x):
        return self.c_proj(self.c_fc(x).gelu())
```

它独立地对每个 token 进行操作（没有 token 间的交互）。隐藏维度通常是模型维度的 4 倍，为网络提供更大的容量。

**GELU**（Gaussian Error Linear Unit）类似于 ReLU 但更平滑 —— 它没有硬零截断。

### Layer Normalization

LayerNorm 将每个 token 的特征归一化为零均值和单位方差：

```python
norm = nn.LayerNorm(768)
x = Tensor.rand(1, 10, 768)  # 10 tokens, 768 features each
y = norm(x)  # each token's 768 features are normalized
```

与 BatchNorm（跨批次归一化）不同，LayerNorm 跨特征进行归一化。这使它能够处理可变长度的序列。

## GPT-2：完整模型

GPT-2 是由 token Embedding + position Embedding 加上一系列 Transformer block 堆叠而成的：

```python
class Transformer:
    def __init__(self, dim, n_heads, n_layers, vocab_size, max_seq_len=1024):
        self.wte = nn.Embedding(vocab_size, dim)        # token embeddings
        self.wpe = nn.Embedding(max_seq_len, dim)       # position embeddings
        self.h = [TransformerBlock(dim, n_heads) for _ in range(n_layers)]
        self.ln_f = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
```

### Token Embedding

词汇表中的每个单词（或子词）都有一个学习到的向量：

```python
# Vocabulary: 50,257 tokens -> 768-dimensional vectors
wte = nn.Embedding(50257, 768)
# Token index 42 maps to a 768-dim vector
embedding = wte(Tensor([42]))  # shape: (1, 768)
```

### Position Embedding

Attention 没有顺序感知 —— "cat sat mat" 和 "mat sat cat" 看起来一样。Position Embedding 添加了位置信息：

```python
wpe = nn.Embedding(1024, 768)
positions = Tensor.arange(0, seq_len)  # [0, 1, 2, ..., seq_len-1]
pos_emb = wpe(positions)  # (seq_len, 768)
```

每个位置都有一个学习到的向量。第一个 Transformer block 的输入是 `token_embedding + position_embedding`。

### 前向传播

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

输出 `logits` 是词汇表中每个 token 在每个位置上的分数。要预测下一个 token，取 `logits[:, -1, :]`（最后一个位置）并选择最高分数。

### GPT-2 模型规格

```python
MODEL_PARAMS = {
    'gpt2':        dict(n_layers=12, n_heads=12, dim=768),    # 124M params
    'gpt2-medium': dict(n_layers=24, n_heads=16, dim=1024),   # 350M params
    'gpt2-large':  dict(n_layers=36, n_heads=20, dim=1280),   # 774M params
    'gpt2-xl':     dict(n_layers=48, n_heads=25, dim=1600),   # 1558M params
}
```

相同的架构，只是规模不同。

## KV Cache

在生成过程中，我们每次生成一个 token。如果不使用缓存，生成第 `t` 个 token 需要重新计算所有之前 token 的 Attention。使用 **KV cache** 后，我们存储之前步骤的 Key 和 Value：

```python
# First call (prompt): process all prompt tokens
# Stores K, V for each layer: cache_kv shape = (2, batch, max_context, n_heads, head_dim)

# Subsequent calls: process only the new token
# Read cached K, V; append new K, V; compute attention against all
self.cache_kv[:, :, start_pos:start_pos+1].assign(Tensor.stack(xk, xv))
keys = self.cache_kv[0][:, :start_pos+1]
values = self.cache_kv[1][:, :start_pos+1]
```

这使得生成的复杂度从每个 token O(n^2) 降低到 O(n)。

## 分词

语言模型不直接处理单词 —— 它们使用**子词分词**。GPT-2 使用字节对编码（BPE）：

```
"tinygrad is cool" -> [22714, 9744, 318, 3608]
```

常见单词获得一个 token。罕见单词被拆分成子词。这提供了一个固定的词汇表（GPT-2 为 50,257），可以表示任何文本。

## 文本生成

生成的工作方式是反复预测下一个 token：

```
Prompt: "The cat sat on"
Step 1: model("The cat sat on") -> "the"
Step 2: model("The cat sat on the") -> "mat"
Step 3: model("The cat sat on the mat") -> "."
...
```

### Temperature 采样

我们可以不总是选择最可能的 token（贪心策略），而是从概率分布中采样：

```python
if temperature < 1e-6:
    ret = logits.argmax(-1)                    # greedy
else:
    ret = (logits / temperature).softmax().multinomial()  # sample
```

- `temperature = 0` -> 始终选择最高概率的 token（确定性）
- `temperature = 1` -> 按照模型的分布进行采样
- `temperature > 1` -> 更随机（更有创造性）
- `temperature < 1` -> 更集中（更保守）

## 运行 GPT-2

```bash
# Generate text with GPT-2 medium
python examples/gpt2.py --model_size gpt2-medium \
    --prompt "The meaning of life is" --count 50 --temperature 0.8
```

## 玩具 Transformer：学习加法

在学习 GPT-2 之前，tinygrad 包含了一个微型 Transformer，它学习两位数加法：

```python
# From examples/transformer.py
# Input:  "42 + 35" encoded as [4, 2, 3, 5]
# Output: "077" encoded as [0, 7, 7]

model = Transformer(10, 6, 2, 128, 4, 32)
# 10 symbols (digits 0-9), 6 positions, 2 layers, 128 dim, 4 heads, 32 ff_dim
```

这是理解 Transformer 的绝佳起点 —— 相同的架构，极小的规模。

## 练习

1. **运行 GPT-2**：使用不同的 temperature 生成文本。输出质量会如何变化？

2. **追踪 Attention**：在 GPT-2 的 `Attention` 类中，追踪前向传播过程中 tensor 的形状变化，参数为 `batch=1, seq_len=10, dim=768, n_heads=12`。

3. **计算 FLOPs**：对于序列长度为 `T`、模型维度为 `d` 的前向传播，Attention 计算的复杂度为 O(T^2 * d)。对于 GPT-2 medium（d=1024, T=128），一个 Attention 层需要多少 FLOPs？

4. **运行玩具 Transformer**：`python examples/transformer.py`。它学习两位数加法。观察准确率的提升。

5. **理解 KV cache**：在 `examples/gpt2.py` 中，找到 `cache_kv` 创建和更新的位置。为什么 JIT 只适用于单 token 推理（而不是 prompt 处理）？

## 源代码索引

| 文件 | 阅读内容 |
|------|---------|
| `examples/gpt2.py` | 带有 KV cache 和文本生成的 GPT-2 |
| `examples/transformer.py` | 学习加法的玩具 Transformer |
| `extra/models/transformer.py` | 通用 TransformerBlock 类 |
| `tinygrad/nn/__init__.py` | `LayerNorm`、`Embedding`、`Linear` |
