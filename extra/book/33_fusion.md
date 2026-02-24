# Chapter 33: Kernel Fusion — When Operations Merge

Kernel fusion is the most important optimization in a deep learning compiler. Instead of running one GPU kernel per operation, fused operations share a single kernel — eliminating intermediate memory reads and writes.

## Why Fusion Matters

Without fusion, `(a + b) * c` would be:

```
Kernel 1: read a, read b → compute a+b → write temp
Kernel 2: read temp, read c → compute temp*c → write result

Memory traffic: 5 reads + 2 writes
```

With fusion:

```
Kernel 1: read a, read b, read c → compute (a+b)*c → write result

Memory traffic: 3 reads + 1 write
```

GPU computation is fast. Memory access is slow. Fusion cuts memory traffic, often by 2-5x.

## How Fusion Works in tinygrad

Fusion happens during scheduling (Chapter 4), specifically in the `remove_bufferize` function. The key idea:

**An intermediate buffer is removed if reading from it would be more expensive than recomputing it.**

### The Intermediate Buffer

When the scheduler encounters an operation like:

```python
a = Tensor.rand(1000)
b = a + 1        # creates an intermediate buffer for a+1
c = b * 2        # reads from that intermediate buffer
```

The graph initially has:
```
BUFFER(a) → ADD(1) → BUFFERIZE → INDEX → MUL(2) → STORE(result)
```

`BUFFERIZE` marks where an intermediate buffer would be created. `remove_bufferize` decides whether to keep or remove it.

### The Fusion Decision

```python
def remove_bufferize(src, buf, idx):
    # 1. Never remove user-requested contiguous buffers
    if src.op in ALWAYS_RUN_OPS or not buf.arg.removable:
        return None  # keep the buffer

    # 2. Count accessed buffers
    accessed_buffers = [...]  # all buffers the fused kernel would access
    if len(accessed_buffers) > 3:
        return None  # too many buffers → keep separate

    # 3. Check if reduces access buffers
    if buffer_in_reduce:
        return None  # reducing over buffered data → keep separate

    # 4. If we get here, remove the buffer (fuse!)
    return src.substitute(range_mapping)
```

Three conditions prevent fusion:

**1. Too many input buffers (> 3)**

A fused kernel with too many buffer arguments is slow due to register pressure:
```python
# Won't fuse: would need 4+ input buffers
result = a + b + c + d + e
```

**2. Reduces that access buffers**

If a reduction reads from a buffer, fusing would mean re-reading that buffer for every reduction step:
```python
# Won't fuse across the reduce:
temp = big_matrix @ weight     # this becomes a buffer
result = temp.sum(axis=1)      # reduce reads temp many times
```

**3. User-requested contiguous**

When you call `.contiguous()`, tinygrad always materializes a buffer.

### When Fusion Succeeds

Fusion succeeds by substituting the intermediate buffer's range variables with the consumer's index:

```python
# Before fusion:
# Kernel 1: for i in range(N): buf[i] = a[i] + 1
# Kernel 2: for i in range(N): result[i] = buf[i] * 2

# After fusion:
# Kernel 1: for i in range(N): result[i] = (a[i] + 1) * 2
```

The substitute replaces the buffer read with the computation that produced it.

## What Gets Fused in Practice

### Element-wise chains: Always fused

```python
x = Tensor.rand(1000)
y = x.relu().sigmoid().tanh()  # all fused into one kernel
```

### Reduce + element-wise: Fused

```python
x = Tensor.rand(100, 100)
y = x.sum(axis=1).relu()  # sum + relu in one kernel
```

### Element-wise + reduce: Fused

```python
x = Tensor.rand(100, 100)
y = (x * 2).sum(axis=1)  # multiply + sum in one kernel
```

### Reduce + reduce: NOT fused

```python
x = Tensor.rand(100, 100)
y = x.sum(axis=1).sum()  # two separate kernels
```

Two reductions can't share a kernel because they need different synchronization patterns.

### Reshapes and permutes: Free

Movement operations don't create kernels at all — they just change how indices are computed:

```python
x = Tensor.rand(4, 8)
y = x.reshape(2, 16).permute(1, 0)  # no kernel, just index math
z = y.sum()  # the reshape+permute are folded into this kernel's indexing
```

## Seeing Fusion with DEBUG

```bash
DEBUG=2 python -c "
from tinygrad import Tensor
x = Tensor.rand(1000)
y = (x + 1).relu().sum()
print(y.item())
"
```

With `DEBUG=2`, you'll see kernel information. A fused chain shows as a single kernel.

## The Pipeline in Context

```
Tensor ops → UOp graph → Scheduling → Rangeify → Codegen → GPU
                              ↑
                        Fusion happens here
                    (remove_bufferize decides
                     which intermediates to keep)
```

## Partial Contiguity (PCONTIG)

For advanced cases, tinygrad supports partial fusion via `PCONTIG`:

```python
# With PCONTIG > 2, some dimensions can be fused while others are buffered
# This is useful when the output-to-input size ratio is very large
out_in_ratio = prod(buf.shape) / sum(x.size for x in accessed_buffers)
if out_in_ratio < 10: return None  # don't fuse
```

This handles cases where fusing would be beneficial for some dimensions but not others.

## Exercises

1. **Count kernels**: Run `DEBUG=2 python -c "from tinygrad import Tensor; x = Tensor.rand(100,100); y = (x*2+1).relu().sum(); print(y.item())"`. How many kernels? (Hint: should be 2 — one for rand, one for the fused multiply+add+relu+sum.)

2. **Break fusion**: What forces a kernel boundary? Try inserting `.contiguous()` or `.realize()` in the chain and count kernels again.

3. **Read the cost model**: In `tinygrad/schedule/rangeify.py`, find `remove_bufferize`. What's the maximum number of accessed buffers before fusion is prevented?

4. **Reduce barriers**: Why can't two reductions fuse into one kernel? Think about what a GPU thread does during a reduction.

## Source Code Map

| File | What to read |
|------|-------------|
| `tinygrad/schedule/rangeify.py:167-229` | `remove_bufferize` — the fusion decision function |
| `tinygrad/schedule/rangeify.py:483-514` | `get_kernel_graph` — orchestrates the entire kernel graph |
| `tinygrad/engine/schedule.py:18-63` | `create_schedule` — linearizes the kernel graph into execution order |
