# 第23章：大语言模型 — LLaMA

本章介绍生产级 LLM 的工作原理，以 tinygrad 的 LLaMA 3 实现为例。如果你已经理解了上一章的 GPT-2，那么 LLaMA 在此基础上增加了三个关键改进：RoPE、Grouped-Query Attention 和 SwiGLU。

## GPT-2 与 LLaMA：有什么变化？

| 特性 | GPT-2 | LLaMA |
|---------|-------|-------|
| 归一化 | LayerNorm | RMSNorm |
| 位置编码 | 学习的嵌入 | RoPE（旋转） |
| 激活函数 | GELU | SwiGLU |
| 注意力 | 多头 | 分组查询（GQA） |
| FFN | 2个线性层 | 3个线性层（门控） |
| 训练数据 | 40GB 文本 | 数万亿 token |

## RoPE：旋转位置嵌入

GPT-2 使用学习的位置向量。LLaMA 使用 **RoPE**，通过在二维子空间中的旋转来编码位置信息。

直觉理解：不是添加位置向量，而是将 Query 和 Key 向量按与其位置成正比的角度进行旋转。两个距离较近的 token 具有相似的旋转（相对角度较小），使得它们的注意力分数更高。

```python
def precompute_freqs_cis(dim, end, theta=10000.0):
    freqs = 1.0 / (theta ** (Tensor.arange(0, dim, 2)[:(dim // 2)] / dim))
    freqs = Tensor.arange(end).unsqueeze(1) * freqs.unsqueeze(0)
    # Returns cos and sin for each position and dimension pair
    return Tensor.stack(freqs.cos(), freqs.sin(), dim=-1).reshape(1, end, 1, dim//2, 2)
```

`theta = 10000` 控制旋转在不同维度上变化的速度。低维度对旋转快（捕捉局部模式），高维度对旋转慢（捕捉长距离模式）。

### 应用旋转嵌入

RoPE 将连续的维度对视为二维坐标并对其进行旋转：

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

为什么这比学习的位置编码更好？
1. **可泛化到更长序列** — 没有固定的最大长度限制
2. **相对位置** — 注意力自然地依赖于 token 之间的距离，而非绝对位置
3. **无额外参数** — 频率是计算得到的，而非学习得到的

## RMSNorm

LLaMA 使用 RMSNorm 代替 LayerNorm。它更简单、更快 — 通过均方根进行归一化，无需中心化：

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

在实践中，RMSNorm 以更少的运算量达到了与 LayerNorm 相近的质量。

## SwiGLU 前馈网络

LLaMA 的 FFN 使用带有三个权重矩阵的门控架构：

```python
class FeedForward:
    def __init__(self, dim, hidden_dim):
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)   # gate projection
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)    # down projection
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)    # up projection

    def __call__(self, x):
        return self.w2(self.w1(x).silu() * self.w3(x))
```

门控机制（`w1(x).silu() * w3(x)`）让网络学习保留隐藏表示的哪些部分。SiLU（Sigmoid Linear Unit，也称为 Swish）即 `x * sigmoid(x)`。

## Grouped-Query Attention（GQA）

标准多头注意力为每个头设置独立的 K 和 V 投影。GQA 在多组头之间共享 K 和 V：

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

当有 32 个 query 头和 8 个 KV 头时，每个 KV 头被 4 个 query 头共享。这将 KV cache 大小减少了 4 倍 — 对于长序列推理至关重要。

Key 和 Value 被重复以匹配 query 头的数量：

```python
def repeat_kv(x, n_rep):
    if n_rep == 1: return x
    return x.repeat((1, 1, 1, n_rep)).reshape(bs, seqlen, n_kv_heads * n_rep, head_dim)
```

## LLaMA 模型尺寸

```python
MODEL_PARAMS = {
    "1B":   {"dim": 2048,  "n_heads": 32,  "n_kv_heads": 8,  "n_layers": 16,  "hidden_dim": 8192},
    "8B":   {"dim": 4096,  "n_heads": 32,  "n_kv_heads": 8,  "n_layers": 32,  "hidden_dim": 14336},
    "70B":  {"dim": 8192,  "n_heads": 64,  "n_kv_heads": 8,  "n_layers": 80,  "hidden_dim": 28672},
    "405B": {"dim": 16384, "n_heads": 128, "n_kv_heads": 8,  "n_layers": 126, "hidden_dim": 53248},
}
```

注意所有尺寸的模型都使用 `n_kv_heads=8` — GQA 使得即使是 405B 模型的 KV cache 也保持在可控范围内。

## 量化

全精度权重开销很大。一个 70B 模型在 float16 下需要 140 GB 内存。量化可以减少这一开销：

### Int8 量化

将每行权重缩放到 int8 范围（-128 到 127）：

```python
class Int8Linear:
    def __init__(self, in_features, out_features):
        self.weight = Tensor.ones(out_features, in_features, dtype=dtypes.int8)
        self.scale = Tensor.ones(out_features, dtype=dtypes.half)

    def __call__(self, x):
        return x.dot(self.weight.cast(self.scale.dtype).T * self.scale)
```

量化过程：
```python
scale = v.abs().max(axis=1) / 127.0
int8_weight = (v.T / scale).T.round().cast(dtypes.int8)
```

这在质量损失极小的情况下实现了约 2 倍的内存缩减。

### NF4 量化

NF4（4-bit NormalFloat）使用一个包含 16 个值的查找表，将两个权重打包到一个字节中：

```python
CODE = [-1.0, -0.696, -0.525, ..., 0.723, 1.0]  # 16 optimal values
```

每个权重被量化为这 16 个值中最接近的一个。这实现了约 4 倍的内存缩减，足以在消费级 GPU 上运行 70B 模型。

### FP8 量化

FP8 使用 8 位浮点格式，包含 4 个指数位和 3 个尾数位：

```python
def quantize_to_fp8(x, dtype=dtypes.fp8e4m3):
    scale = 448.0 / x.abs().max()
    return (x * scale).clamp(-448.0, 448.0).cast(dtype), scale.reciprocal()
```

## 运行 LLaMA

### 交互式对话

```bash
python examples/llama3.py --size 1B
# Downloads the model and starts a chat interface
```

### API 服务器

```bash
python examples/llama3.py --size 8B --port 7776
# Starts an OpenAI-compatible API server at localhost:7776
```

### 多 GPU

```bash
python examples/llama3.py --size 70B --shard 6 --quantize int8
# Shards the 70B model across 6 GPUs with int8 quantization
```

## 生成循环

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

**Prefill** 一次性处理所有提示 token（并行）。**Generation** 每步生成一个 token（顺序）。KV cache 存储中间结果，使得每个生成步骤只需处理新的 token。

## 对话模板

LLaMA 3 使用特定格式进行多轮对话：

```
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|>
<|start_header_id|>user<|end_header_id|>

What is 2+2?<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
```

这些特殊 token 由 `examples/llama3.py` 中的 `Tokenizer` 类处理。

## 权重加载

tinygrad 支持多种权重格式：

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

## 练习

1. **运行 LLaMA 1B**：`python examples/llama3.py --size 1B --no_api`。与它对话。注意 prefill 和 generation 之间的速度差异。

2. **比较量化效果**：分别使用 `--quantize int8` 和不使用量化运行相同的提示。你能分辨出输出质量的差异吗？

3. **阅读注意力代码**：在 `extra/models/llama.py` 中，找到 KV cache 创建和更新的位置。它的形状是什么？GQA 如何减小其大小？

4. **理解 RoPE**：计算 `precompute_freqs_cis(64, 10)` 并打印结果。第一个和最后一个维度对之间的频率有何不同？

5. **测量吞吐量**：使用 `--benchmark` 运行。你的硬件能达到每秒多少 token？

## 源代码索引

| 文件 | 阅读内容 |
|------|-------------|
| `examples/llama3.py` | LLaMA 3 推理、量化、API 服务器 |
| `extra/models/llama.py` | `Transformer`、`Attention`、`FeedForward`、RoPE |
| `tinygrad/nn/state.py` | `safe_load`、`torch_load`、`gguf_load`、`load_state_dict` |
| `tinygrad/nn/__init__.py` | `RMSNorm`、`Embedding` |
