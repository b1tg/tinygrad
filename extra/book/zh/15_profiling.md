# 第15章：命令队列与性能分析

理解性能需要测量。本章介绍 tinygrad 如何将工作分发到 GPU（命令队列）以及如何测量内核性能（性能分析）。

## 命令队列

GPU 不会在你调用时立即执行内核。相反，命令被放入一个**队列**中，GPU 异步处理它们：

```
CPU:  submit(kernel1) -> submit(kernel2) -> submit(kernel3) -> wait
GPU:  ... executing kernel1 ... | kernel2 ... | kernel3 ... | done
```

tinygrad 通过每个后端中的 `HWQueue` 抽象来管理命令队列：

```python
# Simplified from ops_metal.py / ops_amd.py
class HWQueue:
    def submit(self, program, bufs, global_size, local_size): ...
    def signal(self, fence): ...
    def wait(self, fence): ...
```

### 计算队列与拷贝队列

大多数后端有独立的计算和内存传输队列：

```
Compute Queue:  kernel1 -> kernel2 -> kernel3
Copy Queue:     copy_in1 -> copy_in2 -> copy_out1

# These run in parallel! The GPU can copy data while computing.
```

这种重叠对性能至关重要——当一个内核运行时，下一个内核的数据可以同时拷贝进来。

## 性能分析基础

### DEBUG=2：快速性能检查

最简单的性能测量方式：

```bash
DEBUG=2 python -c "
from tinygrad import Tensor
(Tensor.randn(1024, 1024) @ Tensor.randn(1024, 1024)).realize()
"
```

输出：
```
*** METAL  3  r_64_16_64_4_4_4                arg  3 mem  0.01 GB tm  123.45us/  1.23ms (  17.5 GFLOPS   12|34  GB/s)
```

输出解读：
- `r_64_16_64_4_4_4` — 内核名称（r = 归约，数字 = 各轴大小）
- `arg 3` — 3 个缓冲区参数
- `mem 0.01 GB` — 总内存访问量
- `tm 123.45us` — 内核执行时间
- `1.23ms` — 程序启动以来的墙钟时间
- `17.5 GFLOPS` — 每秒浮点运算次数
- `12|34 GB/s` — 内存带宽（读|写）

### PROFILE=1：硬件级性能分析

获取精确的 GPU 计时：

```bash
PROFILE=1 python my_script.py
```

这使用硬件性能计数器（Metal 的 `sampleTimestamps`、AMD 的 SQTT、NVIDIA 的计时事件）来测量精确的内核持续时间，不受 CPU 开销影响。

性能分析数据可以通过以下方式查看：
```bash
# Generates a perfetto trace
PROFILE=1 python my_script.py
# Open the generated trace file in https://ui.perfetto.dev/
```

## 关键性能指标

### GFLOPS（每秒十亿次浮点运算）

衡量计算吞吐量：
```
GFLOPS = (total FLOPs) / (kernel time in seconds) / 1e9

# For matmul MxNxK: FLOPs = 2 * M * N * K
# 1024x1024x1024 matmul = 2 * 1024^3 = 2.1 billion FLOPs
# If it runs in 100us: 2.1e9 / 100e-6 / 1e9 = 21,000 GFLOPS = 21 TFLOPS
```

### GB/s（内存带宽）

衡量内存吞吐量：
```
GB/s = (bytes read + bytes written) / (kernel time in seconds) / 1e9

# Loading two 1024x1024 float32 matrices = 2 * 4MB = 8MB
# Storing one 1024x1024 float32 result = 4MB
# Total = 12MB. If kernel runs in 10us: 12e6 / 10e-6 / 1e9 = 1200 GB/s
```

### Roofline 模型

一个内核要么是**计算受限**的，要么是**内存受限**的：

```
Arithmetic Intensity = FLOPs / Bytes

If AI > device_peak_GFLOPS / device_peak_GB_s:
    -> compute bound (limited by ALU throughput)
Else:
    -> memory bound (limited by memory bandwidth)
```

对于矩阵乘法：AI = 2*N（对于大型方阵）——高度计算受限。
对于逐元素加法：AI = 0.33（每 12 字节 1 次浮点运算）——内存受限。

## 使用 BEAM 优化

使用性能分析来指导优化：

```bash
# See default performance
DEBUG=2 python -c "
from tinygrad import Tensor
(Tensor.randn(1024,1024) @ Tensor.randn(1024,1024)).realize()
"

# Try BEAM search for better kernels
BEAM=5 DEBUG=2 python -c "
from tinygrad import Tensor
(Tensor.randn(1024,1024) @ Tensor.randn(1024,1024)).realize()
"
```

## 性能分析基础设施

tinygrad 的性能分析系统跨所有后端工作：

1. **时间戳**：每次内核调度记录 GPU 的开始/结束时间戳
2. **估算**：运行前，tinygrad 从内核 AST 估算 FLOPs 和内存
3. **Trace 输出**：结果可以导出为 Perfetto trace 进行可视化

```python
# From tinygrad/renderer/__init__.py
@dataclass
class Estimates:
    flops: sint = 0       # estimated floating-point operations
    mem: sint = 0         # estimated memory bytes accessed
    lds: sint = 0         # estimated local memory usage
```

这些估算从内核结构静态计算，然后与实际计时进行比较。

## 练习

1. **分析矩阵乘法**：在不同大小（64、256、1024、4096）上运行 `DEBUG=2` 的矩阵乘法。绘制 GFLOPS 与大小的关系图。

2. **与峰值比较**：查找你的 GPU 的理论峰值 GFLOPS。tinygrad 在大型矩阵乘法上达到了多少百分比？

3. **内存受限 vs 计算受限**：在 `Tensor.randn(10000) + Tensor.randn(10000)`（逐元素，内存受限）和 `Tensor.randn(100,100) @ Tensor.randn(100,100)`（矩阵乘法，计算受限）上运行 `DEBUG=2`。比较 GFLOPS 和 GB/s。

4. **Perfetto trace**：在一个小模型上运行 `PROFILE=1`，并在 Perfetto 中打开 trace。找出内核之间的间隙。

## 源代码导航

| 文件 | 阅读内容 |
|------|----------|
| `tinygrad/helpers.py` | `PROFILE`、`DEBUG` 环境变量 |
| `tinygrad/renderer/__init__.py` | `Estimates` — FLOP/内存估算 |
| `tinygrad/engine/realize.py` | 内核调度与计时 |
| `tinygrad/runtime/ops_metal.py` | Metal 性能分析实现 |
| `tinygrad/runtime/ops_nv.py` | NVIDIA 性能分析实现 |
| `tinygrad/runtime/ops_amd.py` | AMD 性能分析实现 |
