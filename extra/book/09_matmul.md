# Chapter 9: Matrix Multiplication

Matrix multiplication is the most important operation in deep learning. Every linear layer, every attention head, every convolution (when lowered) is a matmul. This chapter shows how tinygrad implements matmul using only reshape, expand, multiply, and sum — the same primitives used for everything else.

## The Naive View

You probably learned matmul as "dot products of rows and columns":

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

This has three nested loops. Tinygrad does the same thing, but expresses it as tensor operations.

## The Tinygrad Way: Reshape + Expand + Multiply + Sum

The key insight: matmul is element-wise multiplication followed by reduction, if you set up the shapes correctly.

```python
from tinygrad import Tensor
import numpy as np

A = Tensor([[1, 2], [3, 4]])  # shape (2, 2)
B = Tensor([[5, 6], [7, 8]])  # shape (2, 2)

# Step 1: Reshape A to (M, 1, K) = (2, 1, 2)
a = A.reshape(2, 1, 2)

# Step 2: Reshape B to (1, N, K) then permute to (1, K, N)... actually:
# Reshape B to (1, K, N) = (1, 2, 2)
b = B.permute(1, 0).reshape(1, 2, 2)

# Step 3: Expand both to (M, N, K) = (2, 2, 2)
a = a.expand(2, 2, 2)  # broadcast along N
b = b.expand(2, 2, 2)  # broadcast along M

# Step 4: Element-wise multiply
c = a * b  # shape (2, 2, 2)

# Step 5: Sum along K (axis 2)
result = c.sum(axis=2)  # shape (2, 2)

print(result.numpy())
# [[19. 22.]
#  [43. 50.]]

# Verify with @:
print((A @ B).numpy())
# [[19. 22.]
#  [43. 50.]]
```

Let's understand why this works:

```
A = [[1, 2],     B = [[5, 6],
     [3, 4]]          [7, 8]]

After reshaping:
a = [[[1, 2]],    shape (2, 1, 2) = (M, 1, K)
     [[3, 4]]]

b = [[[5, 7]],    shape (1, 2, 2) = (1, N, K)  (B transposed then reshaped)
     [[6, 8]]]    wait, actually let's look at what really happens...
```

Actually, let's look at what tinygrad really does internally:

```python
# tinygrad's Tensor.dot() implementation (simplified):
def dot(self, w):
    # self shape: (..., M, K)
    # w shape:    (..., K, N)
    x = self.reshape(*self.shape[:-1], 1, self.shape[-1])    # (..., M, 1, K)
    w = w.reshape(*w.shape[:-2], 1, w.shape[-2], w.shape[-1]) # (..., 1, K, N)
    # Now multiply (broadcasts along M and N) and sum over K
    return (x * w).sum(-2)  # sum over K dimension
```

The shapes line up for broadcasting:
```
x: (M, 1, K)   ->  expand to (M, N, K)
w: (1, K, N)   ->  expand to (M, K, N)
x * w: (M, K, N) -- wait, K needs to align
```

Actually, let's trace it more carefully with `DEBUG_RANGEIFY=1`:

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

Reading bottom-up:
1. **A** (4x4) gets RESHAPE to (4,1,4), then EXPAND to (4,4,4): `A[r0, :, r2]` broadcasts along dim 1
2. **B** (4x4) gets RESHAPE to (1,4,4), PERMUTE, then EXPAND to (4,4,4): `B[r2, r1]` broadcasts along dim 0
3. **MUL** at (4,4,4): `A[r0, r2] * B[r2, r1]` for all (r0, r1, r2)
4. **REDUCE_AXIS** sums over r2: `sum_r2 A[r0, r2] * B[r2, r1]`
5. **ASSIGN** at (4,4): output `C[r0, r1]`

This is exactly the matmul formula: `C[i,j] = sum_k A[i,k] * B[k,j]`

## The Generated Kernel

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

The index expressions `(gidx0/4)*4+ridx0` and `ridx0*4+(gidx0%4)` come directly from rangeify decomposing the reshape+expand+permute operations into arithmetic.

## Why This Design?

The beauty of this approach: **tinygrad doesn't have a special matmul kernel**. It uses the same reshape/expand/mul/sum pipeline for:
- Matrix-vector multiply
- Batched matmul
- Outer products
- Dot products
- Convolutions (with `_pool`)

The optimizer (BEAM search, Chapter 8) then tiles, vectorizes, and maps to tensor cores automatically.

## Dot Product: The Simplest Case

```python
from tinygrad import Tensor
a = Tensor([1, 2, 3])
b = Tensor([4, 5, 6])
print((a * b).sum().item())  # 32 = 1*4 + 2*5 + 3*6
```

The dot product is matmul with M=1, N=1: just multiply and sum.

## Batched Matmul

Tinygrad handles batched matmul naturally because shapes broadcast:

```python
from tinygrad import Tensor

# Batch of 8 matrices, each 4x4
A = Tensor.ones(8, 4, 4)
B = Tensor.ones(8, 4, 4)
C = A @ B  # shape (8, 4, 4)
print(C.shape)  # (8, 4, 4)
```

The reshape/expand pattern extends to arbitrary batch dimensions.

## Exercises

1. **Manual matmul**: Implement matmul using only `.reshape()`, `.expand()`, `*`, and `.sum()` for (3, 4) @ (4, 5). Verify the result matches `@`.

2. **Trace the indices**: For a 2x3 @ 3x2 matmul, write out the index expressions that rangeify produces for `A[i,k]` and `B[k,j]`.

3. **Compare kernels**: Run `DEBUG=4` with and without `NOOPT=1` on a 64x64 matmul. What optimizations does the heuristic apply?

## Source Code Map

| File | What to read |
|------|-------------|
| `tinygrad/tensor.py` | `Tensor.dot()` — the matmul implementation |
| `tinygrad/tensor.py` | `Tensor.matmul()` / `Tensor.__matmul__()` |
