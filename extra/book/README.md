# Tinygrad Internals: A First-Principles Guide

**For CS/ML new grads who know Python, basic linear algebra, and PyTorch.**

Tinygrad is a deep learning framework in ~10k lines of code. Unlike PyTorch's 3M+ lines, you can read and understand the entire thing. This book teaches you how.

## Prerequisites

- Python (comfortable with classes, decorators, dataclasses)
- Basic ML (what a matmul is, what training a neural net means)
- Basic PyTorch (you've done `torch.tensor`, `loss.backward()`, `optimizer.step()`)
- A terminal and `pip install tinygrad`

## Table of Contents

### Part 1: Foundations

| # | Chapter | What you'll learn |
|---|---------|-------------------|
| 01 | [Introduction](01_introduction.md) | What tinygrad is, its philosophy, how it differs from PyTorch |
| 02 | [The UOp](02_uop.md) | The single data structure everything is built on |
| 03 | [Pattern Matcher](03_pattern_matcher.md) | The graph rewriting engine that powers all transformations |

### Part 2: The Pipeline

| # | Chapter | What you'll learn |
|---|---------|-------------------|
| 04 | [Scheduling](04_scheduling.md) | How lazy evaluation works — when does computation actually happen? |
| 05 | [Rangeify](05_rangeify.md) | How shape manipulations become loop nests (the core insight) |
| 06 | [ShapeTracker](06_shapetracker.md) | How tinygrad tracks shapes, strides, and views without moving data |
| 07 | [Codegen](07_codegen.md) | How the UOp graph becomes GPU source code |
| 08 | [BEAM Search](08_beam.md) | How tinygrad auto-tunes kernel performance |

### Part 3: Tricks of the Trade

| # | Chapter | What you'll learn |
|---|---------|-------------------|
| 09 | [Matrix Multiplication](09_matmul.md) | How matmul is expressed as reshape+expand+reduce |
| 10 | [Convolution](10_convolution.md) | The conv trick — im2col as reshape+expand |
| 11 | [Tensor Cores](11_tensor_cores.md) | Using hardware matrix multiply units (WMMA) |

### Part 4: Hardware & Backends

| # | Chapter | What you'll learn |
|---|---------|-------------------|
| 12 | [AMD GPU Emulator](12_amd_emulator.md) | How tinygrad emulates a full GPU on your CPU |
| 13 | [Backends](13_backends.md) | CUDA, Metal, AMD, CPU — how they all fit together |
| 14 | [Multi-GPU](14_multigpu.md) | How computation spans multiple devices |
| 15 | [Profiling](15_profiling.md) | How to measure and optimize performance |

### Part 5: Advanced Topics

| # | Chapter | What you'll learn |
|---|---------|-------------------|
| 16 | [The JIT](16_jit.md) | How TinyJit captures and replays computation graphs |
| 17 | [VIZ](17_viz.md) | The graph visualizer — seeing what tinygrad does |
| 18 | [Symbolic Math](18_symbolic.md) | How tinygrad handles symbolic shapes and expressions |
| 19 | [Low-Level PTX](19_ptx.md) | LOP3 instructions and NVIDIA assembly internals |

### Part 6: Models & Applications

| # | Chapter | What you'll learn |
|---|---------|-------------------|
| 20 | [MNIST](20_mnist.md) | Your first neural network — training loops, optimizers, loss functions |
| 21 | [CNNs](21_cnn.md) | ResNet, EfficientNet — how image classification works |
| 22 | [Transformers](22_transformer.md) | Self-attention, GPT-2 — the architecture behind modern AI |
| 23 | [Large Language Models](23_llm.md) | LLaMA — RoPE, GQA, KV cache, quantization, serving |
| 24 | [Stable Diffusion](24_diffusion.md) | Text-to-image — VAE, UNet, CLIP, the diffusion process |
| 25 | [Whisper](25_whisper.md) | Speech recognition — mel spectrograms, encoder-decoder |
| 26 | [YOLO](26_yolo.md) | Object detection — FPN, anchor-free detection, NMS |
| 27 | [GANs](27_gan.md) | Generative adversarial networks on MNIST |
| 28 | [Reinforcement Learning](28_rl.md) | PPO on CartPole — actor-critic, advantage, reward-to-go |

### Part 7: Deep Dives

| # | Chapter | What you'll learn |
|---|---------|-------------------|
| 29 | [The Tensor Class](29_tensor.md) | How Tensor wraps UOps, lazy evaluation, when computation happens |
| 30 | [Autograd](30_autograd.md) | How `backward()` computes gradients via graph rewriting |
| 31 | [Dtype System](31_dtype.md) | Types, promotion lattice, bfloat16, fp8, vector types |
| 32 | [Buffers & Memory](32_buffer.md) | GPU memory allocation, LRU caching, memory planner |
| 33 | [Kernel Fusion](33_fusion.md) | When operations merge into a single GPU kernel |
| 34 | [End-to-End Trace](34_endtoend.md) | Following `Tensor.ones(4,4).sum().item()` through the entire pipeline |

## How to Read This Book

**If you're new to compilers/systems:** Start from Chapter 01 and read sequentially. Each chapter builds on the previous.

**If you want to understand the core idea:** Jump to Chapter 05 (Rangeify). This is the heart of tinygrad — understanding how shapes become loops unlocks everything else.

**If you're a hardware person:** Start with Chapter 12 (AMD Emulator) to see how tinygrad interfaces with real GPU ISAs.

**If you want to learn ML models:** Start with Part 6 (Chapters 20-28). Each chapter explains a model from first principles and shows how tinygrad implements it.

**If you want to contribute:** Read Chapters 01-07, then pick the area you're interested in.

## Running the Examples

All code examples in this book are designed to be run directly:

```bash
cd /path/to/tinygrad
python -c "
from tinygrad import Tensor
# examples from the book go here
"
```

Some examples use environment variables for debug output:
```bash
DEBUG=4 python -c "from tinygrad import Tensor; Tensor.ones(4,4).sum().item()"
```

Model examples can be run as standalone scripts:
```bash
python examples/beautiful_mnist.py         # Train a digit recognizer
python examples/gpt2.py                    # Generate text with GPT-2
python examples/whisper.py audio.wav       # Transcribe speech
python examples/stable_diffusion.py        # Generate images from text
python examples/yolov8.py image.jpg        # Detect objects in images
```

## Source Code References

Throughout this book, we reference specific source files. Key entry points:

| File | What it does |
|------|-------------|
| `tinygrad/tensor.py` | The public API — where `Tensor.matmul()` lives |
| `tinygrad/uop/ops.py` | The UOp class and Ops enum — the core IR |
| `tinygrad/schedule/rangeify.py` | Movement ops to kernel loops |
| `tinygrad/schedule/indexing.py` | The rangeify algorithm itself |
| `tinygrad/codegen/__init__.py` | UOp graph to GPU source code |
| `tinygrad/renderer/cstyle.py` | C-style code renderer (CUDA/Metal/OpenCL) |
| `tinygrad/nn/__init__.py` | Neural network layers (Conv2d, Linear, etc.) |
| `tinygrad/nn/optim.py` | Optimizers (Adam, SGD, Muon) |
| `extra/models/` | Model implementations (ResNet, LLaMA, UNet, etc.) |
| `examples/` | Runnable examples (MNIST, GPT-2, Whisper, etc.) |
| `test/mockgpu/amd/emu.py` | The AMD GPU emulator |
