# Kimi TP-only 和 vLLM 对比

当前对比目标：Kimi/DeepSeek MLA + MoE 的 TP-only 路径，不包含 EP。

下面使用 Kimi K2.6 的典型形状：

```text
dim = 7168
n_heads = 64
head_dim = 192
v_head_dim = 128
q_lora_rank = 1536
kv_lora_rank = 512
rope_dim = 64
TP = 8
local_heads = 8
```

## MLA Attention

| tensor/module | 完整形状 | vLLM TP-only | tinygrad 当前 |
| --- | ---: | --- | --- |
| `attn_q_a.weight` | `[1536, 7168]` | copy / replicated，`disable_tp=True` | copy / replicated，`weights[rank]` 每卡一整份 |
| `attn_q_a_norm.weight` | `[1536]` | copy / replicated | copy；local path 中把 norm weight 搬到 rank device |
| `attn_q_b.weight` | `[12288, 1536]` = `[64*192,1536]` | shard axis0，`ColumnParallelLinear` | shard axis0，`weights[rank]` shape `[1536,1536]` = `[8*192,1536]` |
| `attn_kv_a_mqa.weight` | `[576, 7168]` = `[512+64,7168]` | copy / replicated，`disable_tp=True` 或 `ReplicatedLinear` | copy / replicated，`weights[rank]` 每卡一整份 |
| `attn_kv_a_norm.weight` | `[512]` | copy / replicated | copy；local path 中把 norm weight 搬到 rank device |
| `attn_k_b.weight` | `[64,512,128]` | 通过 `kv_b_proj` 的 `ColumnParallelLinear` 按 head axis0 切 | shard axis0，`weights[rank]` shape `[8,512,128]` |
| `attn_v_b.weight` | `[64,128,512]` | 通过 `kv_b_proj` 的 `ColumnParallelLinear` 按 head axis0 切 | shard axis0，`weights[rank]` shape `[8,128,512]` |
| `attn_output.weight` | `[7168,8192]` = `[7168,64*128]` | shard axis1，`RowParallelLinear` | shard axis1，`weights[rank]` shape `[7168,1024]` = `[7168,8*128]` |
| MLA latent cache | `[B,1,T,576]` | rank-local replicated latent cache | rank-local replicated list：`cache_k[rank]`，shape `[B,1,T,576]` |
| RoPE freqs | `[T,64]` | rank-local / local read | 每 rank 一份 copy：`freqs_cis[rank]` |
| attention output reduce | 每 rank partial `[B,T,7168]` | collective allreduce/reduce | Python reduce：`sum(part.to(first_device))` |

## Dense FFN / Shared Expert

| tensor/module | 完整形状 | vLLM TP-only | tinygrad 当前 |
| --- | ---: | --- | --- |
| `ffn_gate.weight` | `[hidden,7168]` | 和 up 合并，shard axis0，`MergedColumnParallelLinear` | shard axis0，gate 单独 GEMM |
| `ffn_up.weight` | `[hidden,7168]` | 和 gate 合并，shard axis0，`MergedColumnParallelLinear` | shard axis0，up 单独 GEMM |
| `ffn_down.weight` | `[7168,hidden]` | shard axis1，`RowParallelLinear` | shard axis1，partial reduce |
| shared expert gate/up/down | 同 dense FFN 模式 | `DeepseekV2MLP(... reduce_results=False)` | 目前按已有 dense/sharded linear 路径走；还没有 fused/merged |

## Routed MoE Experts

| tensor/module | 完整形状 | vLLM `FusedMoE` | tinygrad 当前 |
| --- | ---: | --- | --- |
| router `ffn_gate_inp.weight` | `[num_experts,7168]` | copy / replicated gate | copy / replicated gate |
| expert gate `ffn_gate_exps.weight` | `[E,hidden,7168]` | shard intermediate axis1 / w1 dim0 | shard axis1，`weights[rank]` shape `[E,hidden/TP,7168]` |
| expert up `ffn_up_exps.weight` | `[E,hidden,7168]` | shard intermediate axis1 / w3 dim0 | shard axis1，`weights[rank]` shape `[E,hidden/TP,7168]` |
| expert down `ffn_down_exps.weight` | `[E,7168,hidden]` | shard intermediate input axis2 / w2 dim1 | shard axis2，`weights[rank]` shape `[E,7168,hidden/TP]` |
| expert dispatch | selected tokens 在 `FusedMoE` 内部处理 | fused kernel / TP local intermediate | Python loop over ranks，expert id 选择语义一致 |
| MoE output reduce | 每 rank partial `[B,T,7168]` | collective allreduce/reduce | Python reduce 到 first device |

## 剩余差距

| 项 | vLLM | tinygrad 当前 |
| --- | --- | --- |
| `q_a + kv_a` | fused `DeepSeekV2FusedQkvAProjLinear(disable_tp=True)` | 两个独立 replicated linear |
| `q_a/kv_a` kernel | 可以走 custom/min-latency GEMM | 普通 tinygrad GEMM |
| gate+up | `MergedColumnParallelLinear` | gate/up 两个独立 GEMM |
| MoE | `FusedMoE` | Python/tinygrad expert loop |
| reduce | NCCL/RCCL collective | `.to(first_device)` 加法 |
| loader | 每个 rank process 加载自己的 shard | 单进程 GGUF loader 手动 slice/copy |
| KV cache DCP | 可选 DCP/KV cache shard | 未实现；TP path 使用 rank-local replicated latent cache |

总结：当前 shard axis 基本已经和 vLLM TP-only 对齐；主要差距在 fused kernel、collective、loader 机制和 custom kernel。
