
# TODO 目标是给llm.py推理加速，达到llama.cpp的水平（只关注 tok/s, tg128，暂时不管prefill）


注意：一次只能运行一个benchmark命令，且BEAM不要超过4，防止gpu竞态下出bug

为什么qwen3.5:0.8b,qwen3.5:2b  llm.py速度快于 llama.cpp，但qwen3.5:4b不如，趋势就是模型越大速度越慢。

测试命令：(注意不能选择gpu0，已经损坏; 注意benchmark前几个数字小不代表什么，关键看第五个以后的)

HCQ_VISIBLE_DEVICES="3" REALIZE=1 JITBEAM=2 python tinygrad/apps/llm.py -m qwen3.5:2b --benchmark
CUDA_VISIBLE_DEVICES="3" ../llama-b8272/llama-bench -m /home/b1tg/.cache/tinygrad/downloads/19fde856ab4eba92293cf71fc511af4e


已经测试数据：

tinygrad:
qwen3.5:0.8b
  2.62 ms, 381.84 tok/s,  636.35 GB/s, 1666/1628 MB  --  !\n!\n!\n!\n!\n!\n!\n!\n!\n!\n!
qwen3.5:2b
  3.29 ms, 303.79 tok/s, 1194.05 GB/s, 3930/3887 MB  --  ! !\n!\n!\n!\n!\n!\n!\n!\n!\n!\n
qwen3.5:4b
  5.83 ms, 171.38 tok/s, 1512.65 GB/s, 8826/8737 MB  --  ![]()\n\n# 1. Introduction\n\n## 1.1. Background\n\nThe rapid advancement
qwen3.5:35b-a3b (60 tok/s before exp fusion)
  13.35 ms, 74.91 tok/s,  460.83 GB/s, 6152/112257 MB  --  ! Cl
llama-bench
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 0.8B Q8_0               | 763.78 MiB |   752.39 M | ROCm       |  99 |           pp512 |     18201.11 ± 52.36 |
| qwen35 0.8B Q8_0               | 763.78 MiB |   752.39 M | ROCm       |  99 |           tg128 |       313.87 ± 17.94 |
| qwen35 2B Q4_K - Medium        |   1.18 GiB |     1.88 B | ROCm       |  99 |           pp512 |     14757.15 ± 62.74 |
| qwen35 2B Q4_K - Medium        |   1.18 GiB |     1.88 B | ROCm       |  99 |           tg128 |       284.62 ± 11.35 |
| qwen35 4B Q4_K - Medium        |   2.54 GiB |     4.21 B | ROCm       |  99 |           pp512 |      8648.69 ± 31.72 |
| qwen35 4B Q4_K - Medium        |   2.54 GiB |     4.21 B | ROCm       |  99 |           tg128 |        186.37 ± 3.83 |
| qwen35moe 35B.A3B Q4_K -Medium |  20.49 GiB |    34.66 B | ROCm       |  99 |       1 |           pp512 |      4602.16 ± 58.87 |
| qwen35moe 35B.A3B Q4_K -Medium |  20.49 GiB |    34.66 B | ROCm       |  99 |       1 |           tg128 |       119.60 ± 15.10 |
| deepseek2 30B.A3B Q4_K - Medium |  17.05 GiB |    29.94 B | ROCm       |  99 |       1 |           pp512 |       4638.61 ± 9.50 | (glm4.7)
| deepseek2 30B.A3B Q4_K - Medium |  17.05 GiB |    29.94 B | ROCm       |  99 |       1 |           tg128 |       134.91 ± 10.89 |


# [WIP] GLM-4.7 推理加速

**Current:** 86 tok/s (JITBEAM=2), ~87 tok/s (JITBEAM=4) | **Target:** 135 tok/s | **Progress:** 53→86 tok/s (1.62x)
**对比:** Qwen3.5:4b 相似模型大小(8.5 vs 8.8 GB)，171 tok/s @ 1513 GB/s
**Kernels:** 1410 per step (30 per MoE layer)

## GLM-4.7 架构 (deepseek2)
- 47 层: 1 dense + 46 MoE, 全部 MLA attention
- MoE: 64 experts, 4 active/token, 1 shared expert, expert_dim=1536
- MLA: q_lora_rank=768, kv_lora_rank=512, 20 heads
- Vocab: 154,880

## [DONE] 已完成的优化 (53→86 tok/s, +62%)

### 1. 移除 MLA binary swap hook (+10%: 48→53 tok/s)
- `from tinygrad.apps.mla_binary_swap import *` 每次 kernel 编译都会触发 hook 检查

### 2. MoE gate+up expert weight fusion (+9.4%: 53→58 tok/s)
- 合并 `ffn_gate_exps.weight.cat(ffn_up_exps.weight, dim=1)` 为 `_gate_up_w`
- 两次 gather+matmul → 一次 cat+gather+matmul，减少 92 kernels

### 3. Shared expert gate+up fusion (+5.2%: 58→61 tok/s)
- 合并 `ffn_gate_shexp.weight.cat(ffn_up_shexp.weight, dim=0)` 为 `_shexp_gate_up_w`
- 两次 matmul → 一次 matmul + chunk

### 4. Topk-before-softmax (+8.2%: 61→66 tok/s)
- 原来: `logits.softmax(-1).topk(k)` — 64元素softmax + topk
- 优化: `logits.topk(k)` then `softmax(top_4)` — 只对选中的4个值做softmax
- 因为 probs 后续会 renormalize，所以 topk 选择结果完全等价
- 减少 46 kernels (2470→2424)

### 5. 迭代 argmax 替代 bitonic sort (+12%: 66→74 tok/s) ⭐ 最大单项优化
- tinygrad 的 `topk()` 用 bitonic sort 全排序 64 个 expert — 每层产生 ~30 个 contiguous() kernel
- 改为 4 次迭代 argmax: `argmax → mask → repeat` — 每层只需 ~10 kernels
- **Kernel count: 2424→1550 (-36%)**，这是最大的提速来源
- 仅在 probs renormalize 的模型生效 (GLM-4.7, Qwen3.5-MoE)

### 6. 简化 iterative argmax (移除max, 用gather) (+7%: 74→79 tok/s)
- 移除每次迭代中的 `.max()` 调用，简化 masking 逻辑
- 改用 `logits.gather(-1, sel).softmax(-1)` 替代逐步收集 probs
- Kernel count 不变(1550)，但 kernel 更简单 → 更快

### 7. Fuse q_a + kv_a input projections (+2%: 79→80.5 tok/s)
- 合并 `attn_q_a.weight.cat(attn_kv_a_mqa.weight)` 为 `_qkv_a_w`
- 一次 matmul + split 替代两次 matmul
- **Kernel count: 1550→1503 (-47)**

### 8. Fuse router + shared expert gate_up (+5.6%: 80.5→85 tok/s)
- 合并 `ffn_gate_inp.weight.cat(_shexp_gate_up_w)` 为 `_router_shexp_w`
- 一次 matmul + split 替代两次 matmul
- **Kernel count: 1503→1457 (-46)**

### 9. 移除 KV cache .contiguous() (+1.2%: 85→86 tok/s)
- MLA cache write 中的 `k_new.contiguous()` 不必要
- **Kernel count: 1457→1410 (-47)**

## Profiling 分析

### 瓶颈根因: kernel dispatch overhead
- 1410 kernels, 每个 kernel ~5-10μs dispatch overhead
- 理论 dispatch time: 1410 × 7μs ≈ 10ms → 当前 11.6ms, dispatch 占 ~85%
- 实际 compute (matmuls) 只用 ~2-3ms, 其余是 dispatch + tiny kernel overhead
- Bandwidth 利用率: ~630 GB/s (实际 active data ~1.57 GB → 7.3ms 中只用了 ~1.57/5300 = 0.3ms)
- 模型是 dispatch-limited, 不是 bandwidth-limited

### Per-layer kernel breakdown (30 kernels per MoE layer)
| 类别 | 数量 | 备注 |
|------|------|------|
| Matmuls (irreducible) | 10 | q_a+kv_a, q_b, k_b, 2×cache_matmul, attn_output, router+shexp, gate_up, down, down_shexp |
| Iterative argmax | 8 | 4×argmax + 3×mask + 1×gather |
| RMSNorms | 6 | 3 norms × 2 kernels each (reduce + ewise) |
| KV cache ops | 3 | assign + read ops with start_pos |
| Elementwise | 3 | RoPE, silu*up, residual add etc. |

### 关键 kernel 频率 (from DEBUG=4 listing)
- `E_16_32_4`: 94× (2/layer) — tiny norm constants
- `E_768_16_8_2_4_4`: 91× — norm elementwise
- `E_49152_16_8_2_4_4`: 91× — large matmul outputs
- `r_*` reduces: 大量 46× kernels (各种 reduce ops in MLA/MoE)
- 126 unique kernel types total

## 下一步方向 (86→135 tok/s, 还需 +57%)

### 恢复时的即时TODO
1. 达到 135 tok/s 需要 ~590 kernels (13/layer), 当前 1410 (30/layer), 需减少 58%
2. 最大单项目标: iterative argmax 8 kernels/layer × 46 = 368 kernels (占总量 26%)
3. 需要评估: 是否可以用 custom kernel 替代 iterative argmax (单 kernel top-4)
4. 或者研究 tinygrad scheduler 如何 fuse 更多 elementwise → 可能需要 tinygrad core 改动

### 可行的优化方向 (按预期收益排序)

#### P0: 减少 iterative argmax kernels (368 kernels, 潜在减少 ~300)
- 当前: 4 次 argmax + 3 次 mask + gather = 8 kernels/layer
- 方案A: 用 Tensor.topk 但 patch tinygrad 使用更高效的 partial sort
- 方案B: 写 custom reduce kernel 一次性返回 top-4 indices
- 方案C: 研究是否有更少 kernel 的 tensor-level topk 实现

#### P1: RMSNorm fusion (282 kernels, 潜在减少 ~140)
- 3 norms × 2 kernels × 47 layers = 282 kernels
- 如果 scheduler 能 fuse norm 的 reduce+elementwise 为 1 kernel → 省 141 kernels
- 需要 tinygrad scheduler 改动

#### P2: MLA absorbed attention
- 预存 k_absorbed 在 cache → 减少 2 matmul/layer
- 分析显示 net savings 只有 0-1 kernel/layer, 不值得

#### P3: 多 GCD 并行 (最有潜力但最复杂)
- MI300X 有 8 GCDs, 每个有独立 dispatch 能力
- 如果 dispatch 能并行到 2 个 GCD → 理论上 dispatch time 减半 → ~6ms → 167 tok/s
- 需要 tinygrad multi-device 支持

### 速度预测模型
- 当前: 1410 kernels × ~8μs avg = 11.3ms → 86 tok/s
- 目标: 135 tok/s → 7.4ms → ~925 kernels (at 8μs avg)
- 或者保持 1410 kernels 但减少 avg dispatch to ~5.2μs → 需要 HCQ 优化

## 已排除的路径
- @function 边界合并: 不影响 kernel 数量
- JIT_BATCH_SIZE 调整: 单 mega-batch 反而更慢
- PCONTIG=1/2/3: AMD compiler crash
- MLA custom kernel: 只占 2.2%，非瓶颈
- JITBEAM=2→4 without kernel reduction: 仅 +1.7% (在 1550 kernels 时)
- precompute_freqs_cis half=True: 代码已加,对tok/s无影响
- MLA absorbed attention (预存 k_b/v_b 结果到 cache): 分析后 net ~0 kernel savings, 不值得
- JITBEAM=4 目前: ~87 tok/s (相比 JITBEAM=2 的 86, 仅 +1%)

## 相关文件
- `tinygrad/apps/llm.py` - 主文件，包含所有优化
- `test_mla_kernel.py` - MLA kernel benchmark（已修正）

# [DONE] 目标是给 llm.py 添加GLM-4.7-Flash支持

重要要求：
- 改动最小化，在添加新架构时尽量保持和老架构的兼容性同时考虑减少代码行数增加 (参考git diff master qwen35)
- you should only stop when you run the test command with no error and high quality answer

目标：
- 实现目标是
    1. 基础实现 echo "hello" | HCQ_VISIBLE_DEVICES="2" REALIZE=1 JITBEAM=2 python tinygrad/apps/llm.py -m glm4.7 输出合理答案
    2. 针对"why sky is blue" 这样的问题能输出正确的回答
    3. 进行--benchmark测试，目标速度是130tok/s，仔细分析性能瓶颈进行优化
- 在修改代码时，反复问自己，这行修改/增加是必须的吗？能不能更简单地实现、使用更少的行数
- ruff check --fix tinygrad/apps/llm.py
- 功能确认完成之后，重新阅读diff，反复问自己，是否每一行的修改/增加是必须的吗？能不能更简单地实现、使用更少的行数


注意事项：
- remeber to run llm.py with HCQ_VISIBLE_DEVICES="2" because gpu0 is broken

参考：
- transformer库：https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm4_moe/modeling_glm4_moe.py
- llama.cpp
- https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF
