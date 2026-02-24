# 第14章：多 GPU

当一个 GPU 不够用时，tinygrad 可以将计算分配到多个设备上。本章介绍多 GPU 的工作原理，从 Tensor API 到调度层。

## API

使用多个 GPU 非常简单：

```python
from tinygrad import Tensor, Device

# Shard a tensor across 2 GPUs along axis 0
a = Tensor.ones(8, 4).shard([f"{Device.DEFAULT}:0", f"{Device.DEFAULT}:1"], axis=0)
# First GPU has rows 0-3, second GPU has rows 4-7

b = Tensor.ones(4, 4).shard([f"{Device.DEFAULT}:0", f"{Device.DEFAULT}:1"])
# Both GPUs have the full tensor (replicated)

c = a @ b  # matmul happens on both GPUs in parallel
print(c.numpy())  # results are gathered back
```

## 分片策略

有两种分布数据的方式：

### 1. 分片（沿某个轴切分）
```python
# Split a (1024, 1024) matrix across 4 GPUs along axis 0
# GPU 0: rows 0-255
# GPU 1: rows 256-511
# GPU 2: rows 512-767
# GPU 3: rows 768-1023
t = Tensor.ones(1024, 1024).shard(devices, axis=0)
```

每个 GPU 持有数据的不同切片。

### 2. 复制（拷贝到所有设备）
```python
# Every GPU has the full (1024, 1024) matrix
t = Tensor.ones(1024, 1024).shard(devices)  # no axis = replicate
```

每个 GPU 持有完全相同的副本。

## 多 GPU 矩阵乘法的工作原理

考虑 `C = A @ B`，其中 A 沿行分片：

```
GPU 0: A[0:256]   @ B  = C[0:256]
GPU 1: A[256:512] @ B  = C[256:512]
GPU 2: A[512:768] @ B  = C[512:768]
GPU 3: A[768:1024] @ B = C[768:1024]
```

每个 GPU：
1. 持有 A 的切片
2. 持有 B 的副本（复制）
3. 独立计算 C 的切片
4. 不需要 GPU 间通信！

这是最简单的情况。当归约维度被分片时，GPU 之间需要通信：

```
# Shard A along columns (the K dimension):
GPU 0: A[:, 0:256]   @ B[0:256]   = partial_C_0
GPU 1: A[:, 256:512] @ B[256:512] = partial_C_1
# Need ALLREDUCE: C = partial_C_0 + partial_C_1
```

## ALLREDUCE

当不同 GPU 的部分结果需要合并时，tinygrad 使用 `ALLREDUCE`：

```python
# ALLREDUCE sums (or maxes) across all devices
# After ALLREDUCE, every GPU has the full result

# This happens automatically when you shard along a reduction dimension
a = Tensor.ones(4, 8).shard(devices, axis=1)
result = a.sum(axis=1)  # requires ALLREDUCE across devices
```

## MULTI 操作

在内部，多 GPU 张量使用 `Ops.MULTI` UOp 表示。一个 MULTI 节点包装了多个按设备划分的 UOps：

```python
# A sharded tensor:
MULTI(
    src=(
        UOp(device="GPU:0", shape=(256, 1024)),  # slice on GPU 0
        UOp(device="GPU:1", shape=(256, 1024)),  # slice on GPU 1
    ),
    arg=(axis=0,)  # sharded along axis 0
)
```

多 GPU 模式匹配器（`tinygrad/schedule/multi.py`）将 MULTI 节点上的操作重写为按设备的操作，加上必要的通信：

```
MULTI(ADD(a, b))  ->  ADD(a_gpu0, b_gpu0) on GPU:0
                      ADD(a_gpu1, b_gpu1) on GPU:1

MULTI(REDUCE(x))  ->  REDUCE(x_gpu0) on GPU:0
                       REDUCE(x_gpu1) on GPU:1
                       ALLREDUCE(partial_0, partial_1)
```

## 训练中的数据并行

多 GPU 在机器学习中最常见的模式是**数据并行**：

```python
from tinygrad import Tensor, nn, Device

devices = [f"{Device.DEFAULT}:{i}" for i in range(4)]

# Model weights are replicated on all GPUs
model = SomeModel()
for p in nn.state.get_parameters(model):
    p.replace(p.shard(devices))

# Training data is sharded across GPUs (each gets different batch)
for batch in dataloader:
    x = batch.shard(devices, axis=0)  # split batch across GPUs
    loss = model(x).sum()
    loss.backward()
    # Gradients are automatically allreduced across GPUs
    optimizer.step()
```

## 练习

1. **尝试多 GPU**：如果你有多个 GPU，运行：
   ```python
   from tinygrad import Tensor, Device
   devices = [f"{Device.DEFAULT}:0", f"{Device.DEFAULT}:1"]
   a = Tensor.ones(8, 4).shard(devices, axis=0)
   print((a.sum()).numpy())
   ```

2. **统计内核数量**：使用 `DEBUG=2` 查看分片矩阵乘法在各设备上启动了多少个内核。

## 源代码导航

| 文件 | 阅读内容 |
|------|----------|
| `tinygrad/tensor.py` | `Tensor.shard()` — 多 GPU API |
| `tinygrad/schedule/multi.py` | `multi_pm` — 多 GPU 模式匹配器 |
| `tinygrad/uop/ops.py` | `Ops.MULTI`、`Ops.ALLREDUCE` — 多 GPU UOps |
