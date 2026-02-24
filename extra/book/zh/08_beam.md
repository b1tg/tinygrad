# 第8章：BEAM 搜索 -- 内核优化

朴素的内核可以工作，但快速的内核需要选择正确的循环结构、向量化和线程映射。tinygrad 使用 **BEAM 搜索**来自动为每个内核找到最佳优化方案。

## 问题

考虑对一个 4x4 矩阵沿轴1求和。朴素的内核：

```c
kernel void r_4_4(device float* data0, device float* data1, ...) {
  int gidx0 = gid.x; /* 4 */
  float acc0 = 0.0f;
  for (int ridx0 = 0; ridx0 < 4; ridx0++) {
    float val0 = *(data1+((gidx0<<2)+ridx0));
    acc0 = (acc0+val0);
  }
  *(data0+gidx0) = acc0;
}
```

这启动了4个线程，每个循环4次。但我们也可以：
- **Upcast**：完全消除循环，使用向量化的 `float4` 加载
- **展开全局维度**：启动1个线程处理所有4行
- **使用本地内存**：使用线程组共享内存进行归约

每种选择产生不同的代码。BEAM 搜索尝试多种组合并选择最快的。

## 优化动作

tinygrad 有一小组优化原语（称为 `Opt`）：

```python
from tinygrad.codegen.opt import Opt, OptOps

# UPCAST：将循环展开为显式的逐元素操作（启用向量化）
Opt(OptOps.UPCAST, axis=0, arg=4)
# 效果：4次循环变成4个显式操作
# 好处：启用 float4 加载/存储

# LOCAL：将全局轴映射到本地线程
Opt(OptOps.LOCAL, axis=0, arg=16)
# 效果：blockDim.x = 16，每个线程处理更少的元素
# 好处：工作组内的并行性

# GROUP：跨本地线程分组归约
Opt(OptOps.GROUP, axis=0, arg=8)
# 效果：8个线程各做部分归约，然后合并
# 好处：并行归约

# UNROLL：完全展开循环
Opt(OptOps.UNROLL, axis=0, arg=4)
# 效果：循环体重复4次
# 好处：消除循环开销，启用指令级并行

# TC：使用 Tensor Cores（矩阵乘法硬件）
Opt(OptOps.TC, axis=0, arg=...)
# 效果：用 WMMA 指令替换乘加操作
# 好处：矩阵乘法的巨大吞吐量
```

这些 Opt 是**可组合的**——你可以对同一个内核应用多个。对于矩阵乘法，你可能会应用：
- `LOCAL` 来分块输出
- `UPCAST` 来向量化加载
- `GROUP` 来并行化归约
- `TC` 来使用 Tensor Cores

## BEAM 搜索算法

BEAM 搜索探索可能的优化空间：

```
1. 从未优化的内核开始
2. 生成所有有效的单步优化
3. 应用每一个，在真实硬件上测量执行时间
4. 保留前 K 个（beam 宽度）最快的内核
5. 对每个幸存者，生成下一步优化
6. 重复直到没有更多改进
7. 返回找到的最快内核
```

```bash
# 启用宽度为5的 BEAM 搜索
BEAM=5 python -c "
from tinygrad import Tensor
(Tensor.ones(1024, 1024) @ Tensor.ones(1024, 1024)).realize()
"
```

搜索在 beam 范围内是穷举的——它实际上在 GPU 上运行每个内核变体并测量实际时间。这很慢（需要多次编译和内核启动），但能找到真正最优的配置。

## 启发式优化

当 BEAM 太慢时，tinygrad 在 `tinygrad/codegen/opt/heuristic.py` 中有手写的启发式规则：

```python
# 启发式规则（简化）：
# 1. 如果内核很小，全部 upcast
# 2. 如果有归约，尝试 GROUP
# 3. 对于类似矩阵乘法的模式，尝试 Tensor Cores
# 4. 将大轴映射到 GLOBAL，小轴映射到 LOCAL
```

默认情况下（不使用 `BEAM=N` 或 `NOOPT=1`）使用这些启发式规则。

## Upcast：用于向量化的循环展开

Upcast 是最常见的优化。它将循环转换为显式操作：

```c
// Upcast 之前（NOOPT）：
for (int gidx0 = 0; gidx0 < 16; gidx0++) {
  float val0 = *(data1+gidx0);
  *(data0+gidx0) = (val0+1.0f);
}

// UPCAST(axis=0, arg=4) 之后：
// 现在4个线程，每个用 float4 处理4个元素
int lidx0 = lid.x; /* 4 */
float4 val0 = *((float4*)(data1+(lidx0<<2)));
*((float4*)(data0+(lidx0<<2))) = float4(
  (val0.x+1.0f), (val0.y+1.0f),
  (val0.z+1.0f), (val0.w+1.0f));
```

向量化版本减少了4倍的内存事务（一次128位加载代替四次32位加载），在 GPU 上显著更快。

## GROUP：并行归约

对于归约操作，GROUP 将工作分配到本地线程：

```c
// GROUP 之前（单线程归约）：
float acc = 0.0f;
for (int i = 0; i < 1024; i++) {
  acc += data[i];
}
out = acc;

// GROUP(axis=0, arg=32) 之后：
// 32个线程各归约32个元素，然后合并
float acc = 0.0f;
for (int i = lid.x * 32; i < (lid.x + 1) * 32; i++) {
  acc += data[i];
}
shared[lid.x] = acc;
barrier();
// 线程0对32个部分结果求和
if (lid.x == 0) {
  float total = 0.0f;
  for (int i = 0; i < 32; i++) total += shared[i];
  out = total;
}
```

## Tensor Cores

对于矩阵乘法，TC 优化将乘加操作映射到硬件矩阵单元（NVIDIA 上的 WMMA，AMD 上的 WMMA）：

```c
// TC 之前：标量乘加
for (int k = 0; k < K; k++) {
  acc += a[i][k] * b[k][j];
}

// TC 之后：硬件矩阵乘法
// 一条指令完成 16x16x16 矩阵乘法
wmma::mma_sync(d_frag, a_frag, b_frag, c_frag);
```

这可以比标量代码快 10-100 倍。

## 查看应用的优化

```bash
# 比较未优化和优化后的版本
NOOPT=1 DEBUG=4 python -c "
from tinygrad import Tensor; (Tensor.ones(4,4).sum(1)).realize()
"

# 对比
DEBUG=4 python -c "
from tinygrad import Tensor; (Tensor.ones(4,4).sum(1)).realize()
"
```

使用 `DEBUG=5`，你会看到应用的 `Opt` 动作：

```
(Opt(op=OptOps.UPCAST, axis=0, arg=4),)
```

## 搜索空间

对于形状为 `(M, N, K)` 的类矩阵乘法内核，搜索空间包括：

- M 的 GLOBAL/LOCAL 分割（例如，64个块 x 16个线程）
- N 的 GLOBAL/LOCAL 分割
- M 的 UPCAST（1, 2, 4, 8）
- N 的 UPCAST（1, 2, 4, 8）
- K 的 UPCAST（1, 2, 4, 8）
- 归约的 GROUP（各种大小）
- TC 启用/禁用

这可能有数千种配置。宽度为5的 BEAM 搜索评估约100种，在 1024x1024 矩阵乘法上需要几秒钟。

## 缓存

优化后的内核会被缓存。缓存键是内核的 UOp AST 哈希——如果相同的计算模式再次出现（即使数据不同），缓存的优化会被复用。

```bash
# 第一次运行：慢（搜索最优配置）
BEAM=5 python -c "
from tinygrad import Tensor; (Tensor.randn(1024,1024) @ Tensor.randn(1024,1024)).realize()
"

# 第二次运行：快（使用缓存结果）
BEAM=5 python -c "
from tinygrad import Tensor; (Tensor.randn(1024,1024) @ Tensor.randn(1024,1024)).realize()
"
```

## 练习

1. **比较速度**：用 `NOOPT=1`、默认启发式和 `BEAM=5` 运行 1024x1024 矩阵乘法。比较 `DEBUG=2` 输出中的 GFLOPS。

2. **阅读 Opts**：对矩阵乘法运行 `DEBUG=5`，找到应用的 Opt 动作列表。每个做了什么？

3. **尝试不同的 beam 宽度**：对同一个矩阵乘法运行 `BEAM=1`、`BEAM=3`、`BEAM=10`。更高的 beam 宽度是否总能产生更快的内核？

## 源代码导航

| 文件 | 阅读内容 |
|------|-------------|
| `tinygrad/codegen/opt/postrange.py` | `apply_opts()` -- 将 Opt 动作应用到 AST |
| `tinygrad/codegen/opt/search.py` | BEAM 搜索实现 |
| `tinygrad/codegen/opt/heuristic.py` | 手写的优化启发式规则 |
| `tinygrad/codegen/opt/tc.py` | Tensor Core 检测和应用 |
| `tinygrad/codegen/opt/__init__.py` | `Opt`、`OptOps` 定义 |
