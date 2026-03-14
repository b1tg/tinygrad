
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

**Current:** 82 tok/s (JITBEAM=4), 74 tok/s (JITBEAM=2) | **Target:** 130 tok/s | **Progress:** 53→82 tok/s (1.55x)
**对比:** Qwen3.5:4b 相似模型大小(8.5 vs 8.8 GB)，171 tok/s @ 1513 GB/s

## GLM-4.7 架构 (deepseek2)
- 47 层: 1 dense + 46 MoE, 全部 MLA attention
- MoE: 64 experts, 4 active/token, 1 shared expert, expert_dim=1536
- MLA: q_lora_rank=768, kv_lora_rank=512, 20 heads
- Vocab: 154,880

## [DONE] 已完成的优化 (53→82 tok/s, +55%)

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

### 6. JITBEAM=4 (+11%: 74→82 tok/s)
- JITBEAM=2→4 在 kernel 数量减少后效果更明显

## Profiling 分析

### 瓶颈根因: kernel dispatch overhead
- 2470 kernels 中 2237 个 (<15μs, ~10μs each) 被 dispatch overhead 占据
- 实际 compute 只用 30-40% 的时间，60-70% 是 kernel launch overhead
- 单个 tiny kernel: ~5μs dispatch + ~5μs compute = 10μs
- 优化后 1550 kernels，dispatch overhead 显著降低

### 现有 profiling (优化后, JITBEAM=2, 1550 kernels)
| 类别 | 代表 kernel | 备注 |
|------|------------|------|
| Expert down matmul | `r_256_4_8_8_96_12_4_4_4_4` (131μs×46) | 最大单个 kernel |
| Expert gate_up matmul | `r_2_1536_8_8_2_2_2_4_4` (28μs×46) | |
| MLA attention matmuls | `r_640_32_8_6_4` etc. (15-18μs×47) | |
| ~1200 tiny elementwise/reduce | `E_32_2n2`, `r_32_16_4` etc. (~10μs each) | dispatch 为主 |

## 下一步方向 (82→130 tok/s, 还需 +59%)

### 恢复时的即时TODO
1. 确认 precompute_freqs_cis half=True 变更对 kernel count 的影响 (已改代码，未确认kernel数)
2. 获取详细 per-layer kernel 分解 (每层MLA多少kernel, MoE多少, norm多少)
3. 重点方向: 减少 ~1200 个 tiny kernel

### 关键思路
- 每个 MoE layer ~33 kernels (16 MLA + 17 MoE FFN)
- 每个 dense layer (layer 0) 的 kernel 数用于对比
- 主要 tiny kernel 来源: RMSNorm(2 kernels each × 3-4/layer), softmax(3 kernels × 47), RoPE, iterative argmax(11/MoE layer)
- 需要研究 tinygrad 的 kernel fusion 策略, 看能否手动合并某些操作

### P0: 继续减少 tiny kernel 数量
- 当前 1550 kernels 中仍有 ~1200 个 tiny (<15μs)
- 目标: 减到 ~800 → 预计 ~100 tok/s
- 方向: 减少 MLA attention 中的小操作 (RoPE, norm, softmax 各有 3-5 个小 kernel)

### P1: MLA absorbed attention 预计算
- 预存 `k_projected = k_cache @ k_b` 在 KV cache
- 将 2-step matmul 变 1-step，但 KV cache 增大 16.6x

### P2: tinygrad scheduler 改进
- PCONTIG 在 AMD 上 crash (compiler error)
- 需要更好的 elementwise fusion: reduce 后的 scale 应 inline 到下一个 matmul

### P3: 多 GCD 并行
- MI300X 有 8 GCDs，当前只用 1 个

## 已排除的路径
- @function 边界合并: 不影响 kernel 数量
- JIT_BATCH_SIZE 调整: 单 mega-batch 反而更慢
- PCONTIG=1/2/3: AMD compiler crash
- MLA custom kernel: 只占 2.2%，非瓶颈
- JITBEAM=2→4 without kernel reduction: 仅 +1.7%
- precompute_freqs_cis half=True: 代码已加,对tok/s无影响(仍74 tok/s JITBEAM=2)

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
