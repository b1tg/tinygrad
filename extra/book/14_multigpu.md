# Chapter 14: Multi-GPU

When one GPU isn't enough, tinygrad can split computation across multiple devices. This chapter explains how multi-GPU works, from the tensor API down to the scheduling.

## The API

Using multiple GPUs is straightforward:

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

## Sharding Strategies

There are two ways to distribute data:

### 1. Shard (split along an axis)
```python
# Split a (1024, 1024) matrix across 4 GPUs along axis 0
# GPU 0: rows 0-255
# GPU 1: rows 256-511
# GPU 2: rows 512-767
# GPU 3: rows 768-1023
t = Tensor.ones(1024, 1024).shard(devices, axis=0)
```

Each GPU holds a different slice of the data.

### 2. Replicate (copy to all devices)
```python
# Every GPU has the full (1024, 1024) matrix
t = Tensor.ones(1024, 1024).shard(devices)  # no axis = replicate
```

Every GPU holds an identical copy.

## How Multi-GPU Matmul Works

Consider `C = A @ B` where A is sharded along rows:

```
GPU 0: A[0:256]   @ B  = C[0:256]
GPU 1: A[256:512] @ B  = C[256:512]
GPU 2: A[512:768] @ B  = C[512:768]
GPU 3: A[768:1024] @ B = C[768:1024]
```

Each GPU:
1. Has its slice of A
2. Has a copy of B (replicated)
3. Computes its slice of C independently
4. No inter-GPU communication needed!

This is the simplest case. When the reduction dimension is sharded, GPUs need to communicate:

```
# Shard A along columns (the K dimension):
GPU 0: A[:, 0:256]   @ B[0:256]   = partial_C_0
GPU 1: A[:, 256:512] @ B[256:512] = partial_C_1
# Need ALLREDUCE: C = partial_C_0 + partial_C_1
```

## ALLREDUCE

When partial results from different GPUs need to be combined, tinygrad uses `ALLREDUCE`:

```python
# ALLREDUCE sums (or maxes) across all devices
# After ALLREDUCE, every GPU has the full result

# This happens automatically when you shard along a reduction dimension
a = Tensor.ones(4, 8).shard(devices, axis=1)
result = a.sum(axis=1)  # requires ALLREDUCE across devices
```

## The MULTI Op

Internally, multi-GPU tensors are represented using the `Ops.MULTI` UOp. A MULTI node wraps multiple per-device UOps:

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

The multi-GPU pattern matcher (`tinygrad/schedule/multi.py`) rewrites operations on MULTI nodes into per-device operations plus any necessary communication:

```
MULTI(ADD(a, b))  ->  ADD(a_gpu0, b_gpu0) on GPU:0
                      ADD(a_gpu1, b_gpu1) on GPU:1

MULTI(REDUCE(x))  ->  REDUCE(x_gpu0) on GPU:0
                       REDUCE(x_gpu1) on GPU:1
                       ALLREDUCE(partial_0, partial_1)
```

## Data Parallelism for Training

The most common multi-GPU pattern in ML is **data parallelism**:

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

## Exercises

1. **Try multi-GPU**: If you have multiple GPUs, run:
   ```python
   from tinygrad import Tensor, Device
   devices = [f"{Device.DEFAULT}:0", f"{Device.DEFAULT}:1"]
   a = Tensor.ones(8, 4).shard(devices, axis=0)
   print((a.sum()).numpy())
   ```

2. **Count kernels**: Use `DEBUG=2` to see how many kernels are launched across devices for a sharded matmul.

## Source Code Map

| File | What to read |
|------|-------------|
| `tinygrad/tensor.py` | `Tensor.shard()` — the multi-GPU API |
| `tinygrad/schedule/multi.py` | `multi_pm` — multi-GPU pattern matcher |
| `tinygrad/uop/ops.py` | `Ops.MULTI`, `Ops.ALLREDUCE` — multi-GPU UOps |
