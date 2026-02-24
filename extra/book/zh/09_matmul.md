# 第9章：矩阵乘法

矩阵乘法是深度学习中最重要的操作。每个线性层、每个注意力头、每个卷积（降级后）都是矩阵乘法。本章展示 tinygrad 如何仅使用 reshape、expand、multiply 和 sum 来实现矩阵乘法——与其他所有操作使用相同的原语。

## 朴素的视角

你可能学过矩阵乘法是"行和列的点积"：

```python
# C[i,j] = sum_k A[i,k] * B[k,j]
def naive_matmul(A, B):
    M, K = A.shape
    K2, N = B.shape
    C = [[0]*N for _ in range(M)]
    for i in range(M):
        for j in range(N):
            for k in range(K):
                C[i][j] += A[i][k] * B[k][j]
    return C
```

这有三层嵌套循环。tinygrad 做同样的事情，但用张量操作来表达。

## tinygrad 的方式：Reshape + Expand + Multiply + Sum

关键洞察：如果正确设置形状，矩阵乘法就是逐元素乘法加归约。

```python
from tinygrad import Tensor
import numpy as np

A = Tensor([[1, 2], [3, 4]])  # shape (2, 2)
B = Tensor([[5, 6], [7, 8]])  # shape (2, 2)

# 步骤1：将 A reshape 为 (M, 1, K) = (2, 1, 2)
a = A.reshape(2, 1, 2)

# 步骤2：将 B reshape 为 (1, N, K) 然后 permute 为 (1, K, N)... 实际上：
# 将 B reshape 为 (1, K, N) = (1, 2, 2)
b = B.permute(1, 0).reshape(1, 2, 2)

# 步骤3：将两者 expand 到 (M, N, K) = (2, 2, 2)
a = a.expand(2, 2, 2)  # 沿 N 广播
b = b.expand(2, 2, 2)  # 沿 M 广播

# 步骤4：逐元素乘法
c = a * b  # shape (2, 2, 2)

# 步骤5：沿 K（轴2）求和
result = c.sum(axis=2)  # shape (2, 2)

print(result.numpy())
# [[19. 22.]
#  [43. 50.]]

# 用 @ 验证：
print((A @ B).numpy())
# [[19. 22.]
#  [43. 50.]]
```

让我们理解为什么这样做有效：

```
A = [[1, 2],     B = [[5, 6],
     [3, 4]]          [7, 8]]

reshape 之后：
a = [[[1, 2]],    shape (2, 1, 2) = (M, 1, K)
     [[3, 4]]]

b = [[[5, 7]],    shape (1, 2, 2) = (1, N, K)（B 转置后 reshape）
     [[6, 8]]]    等等，让我们看看实际发生了什么...
```

实际上，让我们看看 tinygrad 内部真正做了什么：

```python
# tinygrad 的 Tensor.dot() 实现（简化）：
def dot(self, w):
    # self shape: (..., M, K)
    # w shape:    (..., K, N)
    x = self.reshape(*self.shape[:-1], 1, self.shape[-1])    # (..., M, 1, K)
    w = w.reshape(*w.shape[:-2], 1, w.shape[-2], w.shape[-1]) # (..., 1, K, N)
    # 现在乘法（沿 M 和 N 广播）并对 K 求和
    return (x * w).sum(-2)  # 对 K 维度求和
```

形状对齐以进行广播：
```
x: (M, 1, K)   ->  expand 到 (M, N, K)
w: (1, K, N)   ->  expand 到 (M, K, N)
x * w: (M, K, N) -- 等等，K 需要对齐
```

实际上，让我们用 `DEBUG_RANGEIFY=1` 更仔细地追踪：

```bash
DEBUG_RANGEIFY=1 NOOPT=1 python -c "
from tinygrad import Tensor
(Tensor.ones(4,4) @ Tensor.ones(4,4)).realize()
"
```

```
***  1 Ops.ASSIGN           (4, 4, 1)   [r0][r1][0]
     1 Ops.REDUCE_AXIS      (4, 4, 1)   [r0][r1][r2 -> 0]
     1 Ops.MUL              (4, 4, 4)   [r0][r1][r2]
     1 Ops.EXPAND           (4, 4, 4)   [0 -> r0][r1][r2]
     1 Ops.PERMUTE          (1, 4, 4)   [0][r2 -> r1][r1 -> r2]
     1 Ops.RESHAPE          (1, 4, 4)   [r2][r1] -> [0][r2][r1]
     1 Ops.EXPAND           (4, 4, 4)   [r0][0 -> r1][r2]
     1 Ops.RESHAPE          (4, 1, 4)   [r0][r2] -> [r0][0][r2]
```

从下往上读：
1. **A** (4x4) 被 RESHAPE 为 (4,1,4)，然后 EXPAND 为 (4,4,4)：`A[r0, :, r2]` 沿维度1广播
2. **B** (4x4) 被 RESHAPE 为 (1,4,4)，PERMUTE，然后 EXPAND 为 (4,4,4)：`B[r2, r1]` 沿维度0广播
3. **MUL** 在 (4,4,4)：对所有 (r0, r1, r2) 计算 `A[r0, r2] * B[r2, r1]`
4. **REDUCE_AXIS** 对 r2 求和：`sum_r2 A[r0, r2] * B[r2, r1]`
5. **ASSIGN** 在 (4,4)：输出 `C[r0, r1]`

这正是矩阵乘法公式：`C[i,j] = sum_k A[i,k] * B[k,j]`

## 生成的内核

```bash
DEBUG=4 NOOPT=1 python -c "
from tinygrad import Tensor
(Tensor.ones(4,4) @ Tensor.ones(4,4)).realize()
"
```

```c
kernel void r_4_4_4(device float* data0, device float* data1, device float* data2, ...) {
  int gidx0 = gid.x; /* 16 */
  float acc0 = 0.0f;
  for (int ridx0 = 0; ridx0 < 4; ridx0++) {
    float val0 = *(data1+((gidx0/4)*4+ridx0));   // A[i][k]
    float val1 = *(data2+(ridx0*4+(gidx0%4)));     // B[k][j]
    acc0 = (acc0+(val0*val1));
  }
  *(data0+gidx0) = acc0;
}
```

索引表达式 `(gidx0/4)*4+ridx0` 和 `ridx0*4+(gidx0%4)` 直接来自 rangeify 将 reshape+expand+permute 操作分解为算术运算。

## 为什么这样设计？

这种方法的优美之处在于：**tinygrad 没有专门的矩阵乘法内核**。它对以下所有操作使用相同的 reshape/expand/mul/sum 流水线：
- 矩阵-向量乘法
- 批量矩阵乘法
- 外积
- 点积
- 卷积（通过 `_pool`）

优化器（BEAM 搜索，第8章）然后自动进行分块、向量化和映射到 Tensor Cores。

## 点积：最简单的情况

```python
from tinygrad import Tensor
a = Tensor([1, 2, 3])
b = Tensor([4, 5, 6])
print((a * b).sum().item())  # 32 = 1*4 + 2*5 + 3*6
```

点积是 M=1, N=1 的矩阵乘法：只需乘法和求和。

## 批量矩阵乘法

tinygrad 自然地处理批量矩阵乘法，因为形状会广播：

```python
from tinygrad import Tensor

# 8个矩阵的批次，每个 4x4
A = Tensor.ones(8, 4, 4)
B = Tensor.ones(8, 4, 4)
C = A @ B  # shape (8, 4, 4)
print(C.shape)  # (8, 4, 4)
```

reshape/expand 模式可以扩展到任意批次维度。

## 练习

1. **手动矩阵乘法**：仅使用 `.reshape()`、`.expand()`、`*` 和 `.sum()` 实现 (3, 4) @ (4, 5) 的矩阵乘法。验证结果与 `@` 一致。

2. **追踪索引**：对于 2x3 @ 3x2 的矩阵乘法，写出 rangeify 为 `A[i,k]` 和 `B[k,j]` 生成的索引表达式。

3. **比较内核**：对 64x64 矩阵乘法分别使用和不使用 `NOOPT=1` 运行 `DEBUG=4`。启发式方法应用了哪些优化？

## 源代码导航

| 文件 | 阅读内容 |
|------|-------------|
| `tinygrad/tensor.py` | `Tensor.dot()` -- 矩阵乘法实现 |
| `tinygrad/tensor.py` | `Tensor.matmul()` / `Tensor.__matmul__()` |
