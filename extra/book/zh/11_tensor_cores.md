# 第十一章：Tensor Core / WMMA

现代 GPU 有专用的矩阵乘法硬件。NVIDIA 称之为 Tensor Core，AMD 称之为 WMMA（Wave Matrix Multiply-Accumulate）单元。本章解释它们是什么，tinygrad 如何检测何时使用它们，以及生成的代码长什么样。

## 什么是 Tensor Core？

普通 GPU 算术每个线程每个周期做一次运算：一次加法，一次乘法。Tensor Core 做**矩阵乘法-累加** — 在一条指令中跨 warp/wave 中的所有线程完成一个小矩阵乘法（如 16x16x16）。

```
普通 ALU:    每个线程每个周期 1 FLOP
Tensor Core: 每个 warp 每个周期 16*16*16 = 4096 FLOPs（对于 float16）
```

这就是为什么 GPU 宣传的"tensor TFLOPS"数字比它们的"标量 TFLOPS"高 4-16 倍。

## WMMA 指令

WMMA（Wave/Warp Matrix Multiply-Accumulate）计算：

```
D = A × B + C
```

其中 A、B、C、D 是分布在 warp 线程中的小矩阵（NVIDIA 32 个线程，AMD RDNA 32 个线程）：

```
NVIDIA:  16x16x16 (fp16) 或 8x8x4 (tf32) 或 16x16x16 (int8)
AMD:     16x16x16 (fp16) 或 16x16x16 (bf16)
```

每个线程持有每个矩阵的几个元素。WMMA 指令执行后，每个线程持有结果的几个元素。

## tinygrad 如何检测 Tensor Core 机会

tensor core pass（`tinygrad/codegen/opt/tc.py`）在 kernel AST 中寻找匹配矩阵乘法结构的模式：

1. 有一个维度上的归约（SUM）
2. 归约内部有两个加载值的乘法
3. 形状与 WMMA tile 大小兼容

```python
# TC 优化应用为：
Opt(OptOps.TC, axis=0, arg=(tile_m, tile_n, tile_k, dtype, ...))
```

应用时，优化器：
1. 将循环分块为 WMMA 维度（如 16x16x16）
2. 用 WMMA UOp 替换乘法-累加循环体
3. ���整加载/存储模式以适应分布式矩阵布局

## 生成的代码

### 不使用 Tensor Core（标量）：

```c
// 16x16 矩阵乘法，标量
for (int ridx0 = 0; ridx0 < 16; ridx0++) {
  float val0 = *(data1 + (gidx0*16 + ridx0));   // A[i][k]
  float val1 = *(data2 + (ridx0*16 + gidx1));   // B[k][j]
  acc += val0 * val1;
}
```

### 使用 Tensor Core（CUDA PTX）：

```ptx
// 加载矩阵片段
wmma.load.a.sync.aligned.m16n16k16.global.row.f16 {%r0, ...}, [data1];
wmma.load.b.sync.aligned.m16n16k16.global.col.f16 {%r8, ...}, [data2];
// 乘法-累加
wmma.mma.sync.aligned.m16n16k16.row.col.f32.f16.f16.f32
    {%r16, ...}, {%r0, ...}, {%r8, ...}, {%r16, ...};
// 存储结果
wmma.store.d.sync.aligned.m16n16k16.global.row.f32 [data0], {%r16, ...};
```

### 使用 WMMA（AMD RDNA3）：

```
v_wmma_f32_16x16x16_f16 v[acc], v[a_frag], v[b_frag], v[acc]
```

一条指令替代了 16*16 = 256 次乘法-累加操作。

## WMMA 的 UOp

在 tinygrad 的 IR 中，tensor core 操作由 `Ops.WMMA` UOp 表示：

```python
UOp(Ops.WMMA, dtypes.float.vec(8),
    src=(a_fragment, b_fragment, acc_fragment),
    arg=(16, 16, 16, dtypes.half, "AMD"))
```

arg 包含：`(M, N, K, input_dtype, device_name)`。

## 数据布局

Tensor core 需要特定的数据布局。warp 中的每个线程持有输入矩阵的特定元素。映射取决于硬件：

**NVIDIA（sm_80, 16x16x16 fp16）**：
- warp 中的线程 `i` 持有 A 的 8 个元素和 B 的 8 个元素
- WMMA 之后，线程 `i` 持有结果的 8 个元素

**AMD（RDNA3, 16x16x16 fp16）**：
- 32 个 lane，每个按照 AMD 的 WMMA 规范持有片段
- `v_wmma_f32_16x16x16_f16` 直接操作 VGPR

tinygrad 在 TC 优化 pass 中自动处理数据布局变换。

## 性能影响

Tensor core 对矩阵乘法密集的工作负载提供巨大加速：

```bash
# 不使用 tensor core
NOOPT=1 DEBUG=2 python -c "
from tinygrad import Tensor
(Tensor.ones(1024,1024) @ Tensor.ones(1024,1024)).realize()
"
# ~X GFLOPS

# 使用 tensor core（如果硬件支持）
DEBUG=2 python -c "
from tinygrad import Tensor
(Tensor.ones(1024,1024).half() @ Tensor.ones(1024,1024).half()).realize()
"
# ~16X GFLOPS（或更多）
```

注意：tensor core 通常需要半精度（fp16/bf16）输入。一些较新硬件支持 fp32 或 int8。

## 何时使用 Tensor Core

使用条件：
1. 硬件支持（NVIDIA sm_70+，AMD RDNA3+）
2. 数据类型兼容（通常是 fp16/bf16）
3. 形状与 tile 大小兼容（16 的倍数）
4. 优化器的 TC pass 成功匹配矩阵乘法模式

```python
# 可能使用 tensor core：
(Tensor.ones(256, 256).half() @ Tensor.ones(256, 256).half()).realize()

# 不会使用 tensor core（大多数硬件上的 float32 输入）：
(Tensor.ones(256, 256) @ Tensor.ones(256, 256)).realize()

# 不会使用 tensor core（太小，不满足 tile 大小）：
(Tensor.ones(3, 3).half() @ Tensor.ones(3, 3).half()).realize()
```

## 练习

1. **检查支持**：对半精度矩阵乘法运行 `DEBUG=5`。在 AST 输出中寻找 `WMMA` 来确认是否使用了 tensor core。

2. **比较速度**：对 1024x1024 矩阵乘法分别使用 float32 和 float16 输入运行 `DEBUG=2`。加速比是多少？

3. **阅读代码**：查看 `tinygrad/codegen/opt/tc.py`，找到你的硬件的 WMMA tile 大小。

## 源代码索引

| 文件 | 阅读内容 |
|------|---------|
| `tinygrad/codegen/opt/tc.py` | Tensor core 检测和优化 |
| `tinygrad/renderer/ptx.py` | PTX WMMA 指令发射（NVIDIA） |
| `tinygrad/renderer/cstyle.py` | CUDA WMMA intrinsic 调用 |
| `tinygrad/renderer/amd/` | AMD WMMA 指令编码 |
