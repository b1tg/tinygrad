# Chapter 10: Convolution

Convolution is the second most important operation in deep learning (after matmul). This chapter shows how tinygrad implements conv2d using the same reshape/expand/sum primitives — no special convolution kernel needed.

## What Convolution Does

A 2D convolution slides a small weight kernel over an input image, computing element-wise multiply and sum at each position:

```python
from tinygrad import Tensor

# Input: 1 batch, 1 channel, 4x4 image
inp = Tensor([[[[0, 1, 2, 3],
                [4, 5, 6, 7],
                [8, 9, 10, 11],
                [12, 13, 14, 15]]]], dtype='float')

# Weight: 1 output channel, 1 input channel, 3x3 kernel
weight = Tensor.ones(1, 1, 3, 3)

out = inp.conv2d(weight)
print(out.shape)   # (1, 1, 2, 2)
print(out.numpy())
# [[[[45. 54.]
#    [81. 90.]]]]
```

The output at position (0,0) is the sum of the top-left 3x3 patch: `0+1+2+4+5+6+8+9+10 = 45`.

## The _pool Trick

Tinygrad implements convolution using a helper called `_pool`. It rearranges the input so that each "window" (the region the kernel slides over) becomes a separate slice:

```python
from tinygrad import Tensor

inp = Tensor([[[[0, 1, 2, 3],
                [4, 5, 6, 7],
                [8, 9, 10, 11],
                [12, 13, 14, 15]]]], dtype='float')

pooled = inp._pool(k_=(3, 3), stride=1, dilation=1)
print(pooled.shape)  # (1, 1, 2, 2, 3, 3)
```

The pooled output has shape `(batch, channels, out_h, out_w, kernel_h, kernel_w)`. Each `(out_h, out_w)` position contains the 3x3 patch that the kernel would see:

```
pooled[0, 0, 0, 0] = [[0,  1,  2],   # top-left 3x3 patch
                       [4,  5,  6],
                       [8,  9, 10]]

pooled[0, 0, 0, 1] = [[1,  2,  3],   # shifted right by 1
                       [5,  6,  7],
                       [9, 10, 11]]

pooled[0, 0, 1, 0] = [[4,  5,  6],   # shifted down by 1
                       [8,  9, 10],
                       [12, 13, 14]]

pooled[0, 0, 1, 1] = [[5,  6,  7],   # shifted right and down
                       [9, 10, 11],
                       [13, 14, 15]]
```

Once you have the patches laid out, convolution becomes element-wise multiply + sum:

```python
# Convolution = pooled * weight, then sum over kernel dims
result = (pooled * weight).sum(axis=(-2, -1))
print(result.numpy())
# [[[[45. 54.]
#    [81. 90.]]]]
```

## How _pool Works (No Data Movement!)

The magic is that `_pool` uses only movement ops — reshape, expand, shrink — to create the patches. **No data is copied.** The patches are virtual views of the original data:

```python
# Simplified _pool implementation:
def _pool(x, k_, stride, dilation):
    # 1. Expand to create overlapping windows via stride tricks
    # 2. Shrink to select valid regions
    # 3. Reshape to (batch, channels, out_h, out_w, kernel_h, kernel_w)

    # The expand uses stride=stride to create the sliding window effect
    # Shrink removes the padding/overflow
    # Result: each output position has its own view of the kernel-sized patch
    pass
```

Under the hood, it manipulates strides so that adjacent output positions point to overlapping regions of the input — exactly like NumPy's `as_strided`.

## Stride and Dilation

**Stride** controls how far the kernel moves between positions:

```python
# Stride 1: kernel slides 1 pixel at a time
pooled = inp._pool(k_=(2, 2), stride=1, dilation=1)
print(pooled.shape)  # (1, 1, 3, 3, 2, 2)  -- 3x3 output positions

# Stride 2: kernel slides 2 pixels at a time
pooled = inp._pool(k_=(2, 2), stride=2, dilation=1)
print(pooled.shape)  # (1, 1, 2, 2, 2, 2)  -- 2x2 output positions
```

**Dilation** creates gaps in the kernel pattern:

```python
# Dilation 1: normal convolution
# Kernel sees: [0,1], [4,5]

# Dilation 2: skip every other element
# Kernel sees: [0,2], [8,10]
pooled = inp._pool(k_=(2, 2), stride=1, dilation=2)
```

## Full conv2d Pipeline

The complete `conv2d` operation:

```python
# Simplified from tensor.py
def conv2d(x, weight, stride=1, padding=0, dilation=1, groups=1):
    # 1. Pad input if needed
    if padding:
        x = x.pad(...)

    # 2. Pool: create sliding window views
    x = x._pool(k_=weight.shape[-2:], stride=stride, dilation=dilation)
    # shape: (batch, in_channels, out_h, out_w, kernel_h, kernel_w)

    # 3. Reshape for multiplication with weights
    # x: (batch, groups, in_channels//groups, out_h, out_w, kernel_h, kernel_w)
    # w: (groups, out_channels//groups, in_channels//groups, kernel_h, kernel_w)

    # 4. Multiply and sum over (in_channels, kernel_h, kernel_w)
    return (x * weight).sum(axis=(-3, -2, -1))
```

The generated kernel fuses all of this into a single GPU program — no intermediate allocations for the pooled views.

## Padding

Padding adds zeros around the input border. In tinygrad, this uses the `PAD` movement op:

```python
from tinygrad import Tensor

x = Tensor.ones(1, 1, 3, 3)
# Pad 1 pixel on each side of the spatial dimensions
out = x.conv2d(Tensor.ones(1, 1, 3, 3), padding=1)
print(out.shape)  # (1, 1, 3, 3) -- same spatial size as input
```

The PAD op creates a mask in the ShapeTracker. Indices outside the original data return 0, which is exactly what zero-padding does.

## Grouped Convolution

Groups split input and output channels into independent sets:

```python
# groups=2 means:
# - First half of output channels only sees first half of input channels
# - Second half of output channels only sees second half of input channels
out = inp.conv2d(weight, groups=2)
```

Depthwise convolution (used in MobileNet) is the extreme case where `groups = in_channels`, meaning each channel is convolved independently.

## Connection to Matmul

Convolution can be viewed as a matmul in disguise. The `_pool` operation creates an im2col matrix, and the multiply+sum is the matmul:

```
im2col: (batch*out_h*out_w) x (in_channels*kernel_h*kernel_w)
weight: (out_channels) x (in_channels*kernel_h*kernel_w)
output: (batch*out_h*out_w) x (out_channels)
```

Tinygrad's approach is equivalent but doesn't explicitly create the im2col matrix — it's implicit through the shape/stride manipulation of `_pool`.

## Exercises

1. **Manual conv**: Implement a 1D convolution of `[1,2,3,4,5]` with kernel `[1,1,1]` using only `._pool()`, `*`, and `.sum()`. Verify with `Tensor.conv2d`.

2. **Trace _pool**: Print the shape at each step inside `_pool` for a (1,1,4,4) input with kernel (2,2), stride 2.

3. **Generated code**: Run `DEBUG=4 NOOPT=1` on a small conv2d. Read the kernel — identify the load patterns and the accumulation loop.

## Source Code Map

| File | What to read |
|------|-------------|
| `tinygrad/tensor.py` | `Tensor.conv2d()` — the convolution API |
| `tinygrad/tensor.py` | `Tensor._pool()` — the sliding window helper |
