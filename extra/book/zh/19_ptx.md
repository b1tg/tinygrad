# 第19章：底层 PTX 与 LOP3

本章深入探讨 tinygrad 代码生成的最底层：NVIDIA PTX 汇编和 LOP3 指令。这是面向希望了解 tinygrad 如何生成硬件特定代码的读者的高级内容。

## 什么是 PTX？

PTX（Parallel Thread eXecution）是 NVIDIA 的虚拟指令集架构——一种用于 NVIDIA GPU 的汇编语言。虽然 CUDA C++ 由 nvcc/nvrtc 编译为 PTX，但 tinygrad 可以直接生成 PTX：

```ptx
.version 7.8
.target sm_89
.address_size 64

.visible .entry E_4(
    .param .u64 data0,
    .param .u64 data1,
    .param .u64 data2
) {
    .reg .u32 %r<10>;
    .reg .f32 %f<10>;
    .reg .u64 %rd<10>;

    mov.u32 %r0, %ctaid.x;          // block index
    ld.param.u64 %rd0, [data1];      // load pointer
    mad.wide.u32 %rd1, %r0, 4, %rd0; // address = ptr + idx*4
    ld.global.f32 %f0, [%rd1];       // load float

    ld.param.u64 %rd2, [data2];
    mad.wide.u32 %rd3, %r0, 4, %rd2;
    ld.global.f32 %f1, [%rd3];

    add.f32 %f2, %f0, %f1;          // add

    ld.param.u64 %rd4, [data0];
    mad.wide.u32 %rd5, %r0, 4, %rd4;
    st.global.f32 [%rd5], %f2;       // store result
}
```

tinygrad 的 PTX 渲染器（`tinygrad/renderer/ptx.py`）从 UOp 列表生成上述代码。

## 为什么选择 PTX 而不是 CUDA C++？

1. **无需依赖 nvcc**：PTX 可以由 NVIDIA 驱动直接加载，不需要编译器 SDK
2. **更精细的控制**：可以直接访问硬件功能，如 WMMA、共享内存屏障、warp shuffle
3. **更小的开销**：无需 C++ 解析/编译步骤
4. **可预测的输出**：你写的（大致上）就是执行的

## LOP3 指令

LOP3 是 NVIDIA 特有的指令，能够在单个时钟周期内计算**任意3输���布尔函数**。它用于位运算操作。

### 核心思想

任何3个输入（a, b, c）的布尔函数都可以编码为一个8位真值表：

```
Inputs:  a=0,b=0,c=0 | a=0,b=0,c=1 | a=0,b=1,c=0 | ... | a=1,b=1,c=1
Output:  bit 0        | bit 1        | bit 2        | ... | bit 7
```

共有256种可能的3输入布尔函数（2^8）。LOP3 将真值表作为立即操作数：

```ptx
lop3.b32 %r0, %r1, %r2, %r3, 0xCA;  // %r0 = LOP3(%r1, %r2, %r3, 0xCA)
```

字节 `0xCA` 编码的是：`(a & b) | (~a & c)` ——这实际上是一个2:1多路复用器！

### 常见的 LOP3 编码

| 函数 | 真值表 | 十六进制 |
|----------|-------------|--------|
| `a & b & c` | `10000000` | `0x80` |
| `a \| b \| c` | `11111110` | `0xFE` |
| `a ^ b ^ c` | `10010110` | `0x96` |
| `(a & b) \| c` | `11101010` | `0xEA` |
| `a ? b : c` | `11001010` | `0xCA` |
| `a & ~b` | `00001100` | `0x0C` |

### 为什么它很重要

如果没有 LOP3，计算 `(a & b) | (~a & c)` 需要4条指令：
```ptx
not.b32 %t0, %r1;           // ~a
and.b32 %t1, %r1, %r2;      // a & b
and.b32 %t2, %t0, %r3;      // ~a & c
or.b32  %r0, %t1, %t2;      // (a & b) | (~a & c)
```

使用 LOP3，只需1条指令：
```ptx
lop3.b32 %r0, %r1, %r2, %r3, 0xCA;
```

减少4倍指令数，减少4倍寄存器读取，对于位运算密集的代码可能快4倍。

### SASS 与 PTX

LOP3 同时存在于 PTX 和 SASS（实际的机器码）中。在 SASS 中，它更加灵活——可以与寄存器移动和谓词执行组合使用。NVIDIA 汇编器会将 PTX 的 `lop3.b32` 翻译为相应的 SASS 指令。

## tinygrad 如何使用 LOP3

tinygrad 的 PTX 渲染器会检测位运算操作链，并将其融合为 LOP3：

```python
# Before: three separate instructions
# AND(a, b) -> XOR(result, c) -> OR(result, d)

# After: fused into LOP3 where possible
# lop3.b32(a, b, c, truth_table)
```

真值表通过对所有8种输入组合求值布尔函数来计算。

## PTX 渲染器架构

`tinygrad/renderer/ptx.py` 中的 PTX 渲染器与 C 风格渲染器有所不同：

1. **寄存器分配**：PTX 使用虚拟寄存器（`%r0`、`%f0`、`%rd0`），由 NVIDIA 汇编器将其映射到物理寄存器
2. **显式类型**：每个操作都指定其类型（`.f32`、`.u32`、`.b32`）
3. **内存模型**：使用显式的 `ld.global`、`st.global`、`ld.shared` 来访问不同的地址空间
4. **屏障指令**：`bar.sync` 用于 warp 同步
5. **WMMA**：直接访问 Tensor Core 指令

## 练习

1. **计算真值表**：对于函数 `(a | b) & c`，计算8位真值表。你会使用什么 LOP3 立即数值？

2. **阅读渲染器代码**：打开 `tinygrad/renderer/ptx.py`，找到 `Ops.ADD` 在 float32 下是如何渲染的。它生成了什么 PTX 指令？

3. **对比输出**：在一个简单的内核上使用 `DEBUG=4` 运行，分别使用 C 风格渲染器和 PTX 渲染器进行对比（如果你有 NVIDIA 硬件的话）。

## 源代码索引

| 文件 | 阅读内容 |
|------|----------|
| `tinygrad/renderer/ptx.py` | PTX 渲染器——生成 NVIDIA 汇编 |
| `tinygrad/renderer/cstyle.py` | C 风格渲染器，用于对比参考 |
| `tinygrad/runtime/ops_nv.py` | 原生 NVIDIA 驱动（直接加载 PTX） |
| `tinygrad/runtime/ops_cuda.py` | CUDA 运行时（也通过 nvrtc 加载 PTX） |
