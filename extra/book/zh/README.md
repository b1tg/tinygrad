# Tinygrad 内部原理：从第一性原理出发的指南

**面向了解 Python、基础线性代数和 PyTorch 的 CS/ML 应届毕业生。**

Tinygrad 是一个约 10,000 行代码的深度学习框架。不同于 PyTorch 的 300 万行以上代码，你可以阅读并理解它的全部。本书教你如何做到。

## 前置知识

- Python（熟悉类、装饰器、dataclass）
- 基础 ML（知道什么是矩阵乘法，什么是训练神经网络）
- 基础 PyTorch（用过 `torch.tensor`、`loss.backward()`、`optimizer.step()`）
- 一个终端和 `pip install tinygrad`

## 目录

### 第一部分：基础

| # | 章节 | 你将学到 |
|---|------|---------|
| 01 | [简介](01_introduction.md) | tinygrad 是什么，它的设计哲学，与 PyTorch 的区别 |
| 02 | [UOp](02_uop.md) | 一切构建于其上的单一数据结构 |
| 03 | [Pattern Matcher](03_pattern_matcher.md) | 驱动所有变换的图重写引擎 |

### 第二部分：流水线

| # | 章节 | 你将学到 |
|---|------|---------|
| 04 | [调度](04_scheduling.md) | 惰性求值如何工作——计算何时真正发生？ |
| 05 | [Rangeify](05_rangeify.md) | 形状操作如何变成循环嵌套（核心洞察） |
| 06 | [ShapeTracker](06_shapetracker.md) | tinygrad 如何在不移动数据的情况下追踪形状、步长和视图 |
| 07 | [代码生成](07_codegen.md) | UOp 图如何变成 GPU 源代码 |
| 08 | [BEAM 搜索](08_beam.md) | tinygrad 如何自动调优 kernel 性能 |

### 第三部分：实战技巧

| # | 章节 | 你将学到 |
|---|------|---------|
| 09 | [矩阵乘法](09_matmul.md) | matmul 如何表示为 reshape+expand+reduce |
| 10 | [卷积](10_convolution.md) | 卷积技巧——im2col 作为 reshape+expand |
| 11 | [Tensor Core](11_tensor_cores.md) | 使用硬件矩阵乘法单元（WMMA） |

### 第四部分：硬件与后端

| # | 章节 | 你将学到 |
|---|------|---------|
| 12 | [AMD GPU 模拟器](12_amd_emulator.md) | tinygrad 如何在 CPU 上模拟完整 GPU |
| 13 | [后端](13_backends.md) | CUDA、Metal、AMD、CPU——它们如何协同工作 |
| 14 | [多 GPU](14_multigpu.md) | 计算如何跨多个设备 |
| 15 | [性能分析](15_profiling.md) | 如何测量和优化性能 |

### 第五部分：高级主题

| # | 章节 | 你将学到 |
|---|------|---------|
| 16 | [JIT](16_jit.md) | TinyJit 如何捕获和重放计算图 |
| 17 | [VIZ](17_viz.md) | 图可视化器——看到 tinygrad 在做什么 |
| 18 | [符号数学](18_symbolic.md) | tinygrad 如何处理符号形状和表达式 |
| 19 | [底层 PTX](19_ptx.md) | LOP3 指令和 NVIDIA 汇编内部原理 |

### 第六部分：模型与应用

| # | 章节 | 你将学到 |
|---|------|---------|
| 20 | [MNIST](20_mnist.md) | 你的第一个神经网络——训练循环、优化器、损失函数 |
| 21 | [CNN](21_cnn.md) | ResNet、EfficientNet——图像分类如何工作 |
| 22 | [Transformer](22_transformer.md) | 自注意力、GPT-2——现代 AI 背后的架构 |
| 23 | [大语言模型](23_llm.md) | LLaMA——RoPE、GQA、KV cache、量化、服务 |
| 24 | [Stable Diffusion](24_diffusion.md) | 文生图——VAE、UNet、CLIP、扩散过程 |
| 25 | [Whisper](25_whisper.md) | 语音识别——mel 频谱图、编码器-解码器 |
| 26 | [YOLO](26_yolo.md) | 目标检测——FPN、无锚检测、NMS |
| 27 | [GAN](27_gan.md) | 在 MNIST 上的生成对抗网络 |
| 28 | [强化学习](28_rl.md) | CartPole 上的 PPO——Actor-Critic、优势函数、回报 |

### 第七部分：深入探索

| # | 章节 | 你将学到 |
|---|------|---------|
| 29 | [Tensor 类](29_tensor.md) | Tensor 如何包装 UOp、惰性求值、计算何时发生 |
| 30 | [Autograd](30_autograd.md) | `backward()` 如何通过图重写计算梯度 |
| 31 | [Dtype 系统](31_dtype.md) | 类型、提升格、bfloat16、fp8、向量类型 |
| 32 | [Buffer 与内存](32_buffer.md) | GPU 内存分配、LRU 缓存、内存规划器 |
| 33 | [Kernel 融合](33_fusion.md) | 操作何时合并为单个 GPU kernel |
| 34 | [端到端追踪](34_endtoend.md) | 跟踪 `Tensor.ones(4,4).sum().item()` 穿越整个流水线 |

## 如何阅读本书

**如果你是编译器/系统新手：** 从第 01 章开始顺序阅读。每章都建立在前一章的基础上。

**如果你想理解核心思想：** 直接跳到第 05 章（Rangeify）。这是 tinygrad 的核心——理解形状如何变成循环就能解锁其他一切。

**如果你是硬件方向：** 从第 12 章（AMD 模拟器）开始，看 tinygrad 如何与真实 GPU ISA 交互。

**如果你想学习 ML 模型：** 从第六部分（第 20-28 章）开始。每章从第一性原理解释一个模型，并展示 tinygrad 如何实现它。

**如果你想贡献代码：** 阅读第 01-07 章，然后选择你感兴趣的领域。

## 运行示例

本书中的所有代码示例都可以直接运行：

```bash
cd /path/to/tinygrad
python -c "
from tinygrad import Tensor
# 书中的示例代码放在这里
"
```

一些示例使用环境变量来输出调试信息：
```bash
DEBUG=4 python -c "from tinygrad import Tensor; Tensor.ones(4,4).sum().item()"
```

模型示例可以作为独立脚本运行：
```bash
python examples/beautiful_mnist.py         # 训练数字识别器
python examples/gpt2.py                    # 用 GPT-2 生成文本
python examples/whisper.py audio.wav       # 语音转文字
python examples/stable_diffusion.py        # 文生图
python examples/yolov8.py image.jpg        # 图像目标检测
```

## 源代码参考

本书中我们引用了具体的源文件。关键入口点：

| 文件 | 功能 |
|------|-----|
| `tinygrad/tensor.py` | 公共 API——`Tensor.matmul()` 所在之处 |
| `tinygrad/uop/ops.py` | UOp 类和 Ops 枚举——核心 IR |
| `tinygrad/schedule/rangeify.py` | 移动操作到 kernel 循环 |
| `tinygrad/schedule/indexing.py` | rangeify 算法本身 |
| `tinygrad/codegen/__init__.py` | UOp 图到 GPU 源代码 |
| `tinygrad/renderer/cstyle.py` | C 风格代码渲染器（CUDA/Metal/OpenCL） |
| `tinygrad/nn/__init__.py` | 神经网络层（Conv2d、Linear 等） |
| `tinygrad/nn/optim.py` | 优化器（Adam、SGD、Muon） |
| `extra/models/` | 模型实现（ResNet、LLaMA、UNet 等） |
| `examples/` | 可运行示例（MNIST、GPT-2、Whisper 等） |
| `test/mockgpu/amd/emu.py` | AMD GPU 模拟器 |
