# Chapter 6: ShapeTracker — Views Without Copies

ShapeTracker is how tinygrad represents tensor shapes, strides, and views without ever moving data in memory. If you've used PyTorch, you know that `.reshape()`, `.transpose()`, and `.expand()` are "free" operations. ShapeTracker is the mechanism that makes this possible.

## The Problem

A 2x3 matrix `[[1,2,3],[4,5,6]]` is stored in memory as a flat array: `[1, 2, 3, 4, 5, 6]`. To access element `[row, col]`, you compute the memory offset: `row * 3 + col`.

The numbers `3` and `1` are the **strides** — how many elements to skip when you move one step along each dimension:

```
Shape:   (2, 3)
Strides: (3, 1)
Formula: index = row * 3 + col * 1
```

```python
from tinygrad import Tensor

a = Tensor([[1, 2, 3], [4, 5, 6]])
# Memory layout: [1, 2, 3, 4, 5, 6]
# a[1, 2] -> offset = 1*3 + 2*1 = 5 -> value 6
```

## Transpose is Free

To transpose this matrix, you don't need to rearrange memory. Just swap the strides:

```
Original:   shape=(2,3), strides=(3,1)  -> index = row*3 + col*1
Transposed: shape=(3,2), strides=(1,3)  -> index = row*1 + col*3
```

Both formulas access the same memory, just in different order:

```python
# Original:  [0,0]=0, [0,1]=1, [0,2]=2, [1,0]=3, [1,1]=4, [1,2]=5
# Transposed: [0,0]=0, [0,1]=3, [1,0]=1, [1,1]=4, [2,0]=2, [2,1]=5
```

No data was copied. The "transpose" is just a different way of interpreting the same memory.

## The View

A **View** is the core data structure. It contains:

```python
# from tinygrad/shape/view.py
@dataclass
class View:
    shape: tuple[int, ...]     # logical shape
    strides: tuple[int, ...]   # memory strides
    offset: int                # starting offset in memory
    mask: tuple|None           # valid region (for padding)
    contiguous: bool           # whether memory is laid out sequentially
```

### Common Views

```python
# Contiguous 2x3 matrix
# shape=(2,3), strides=(3,1), offset=0
# index = row*3 + col

# Transposed (3x2)
# shape=(3,2), strides=(1,3), offset=0
# index = row*1 + col*3

# Sliced (rows 1-2 of a 4x4)
# shape=(2,4), strides=(4,1), offset=4
# index = 4 + row*4 + col

# Broadcasted (1x4 expanded to 3x4)
# shape=(3,4), strides=(0,1), offset=0
# index = row*0 + col*1 = col  (row is ignored!)

# Scalar broadcast
# shape=(4,4), strides=(0,0), offset=0
# index = 0  (always reads the same element)
```

Notice the stride of `0` for broadcasting — it means "don't move along this dimension," so every row reads the same data.

## Movement Ops as View Transformations

Each movement op transforms the View:

### RESHAPE

Changes shape but preserves the linear index mapping:

```python
from tinygrad import Tensor
a = Tensor.ones(6)        # shape=(6,), strides=(1,)
b = a.reshape(2, 3)       # shape=(2,3), strides=(3,1)
c = b.reshape(3, 2)       # shape=(3,2), strides=(2,1)
# All three access the same 6 elements in the same order
```

### PERMUTE (transpose)

Reorders dimensions by reordering strides:

```python
a = Tensor.ones(2, 3, 4)  # strides=(12, 4, 1)
b = a.permute(2, 0, 1)    # strides=(1, 12, 4), shape=(4, 2, 3)
```

### EXPAND (broadcast)

Sets stride to 0 for expanded dimensions:

```python
a = Tensor.ones(1, 4)     # strides=(4, 1) or (0, 1)
b = a.expand(3, 4)        # strides=(0, 1), shape=(3, 4)
# All 3 rows point to the same 4 elements
```

### SHRINK (slice)

Adjusts offset and shape:

```python
a = Tensor.ones(10)       # strides=(1,), offset=0
b = a[3:7]                # strides=(1,), offset=3, shape=(4,)
```

### PAD

Adds a mask to indicate valid regions:

```python
a = Tensor.ones(3)        # [1, 1, 1]
b = a.pad(((1, 1),))      # [0, 1, 1, 1, 0], mask=((1, 4),)
# Elements outside the mask are treated as zero
```

### FLIP

Negates the stride and adjusts offset:

```python
a = Tensor.ones(4)        # strides=(1,), offset=0
b = a.flip(0)             # strides=(-1,), offset=3
# Accesses elements in reverse: 3, 2, 1, 0
```

## Multi-View ShapeTrackers

Sometimes a single View isn't enough. If you reshape a non-contiguous tensor (like a transposed matrix), you need two views:

```python
a = Tensor.ones(2, 3)     # View 1: shape=(2,3), strides=(3,1)
b = a.permute(1, 0)       # View 1: shape=(3,2), strides=(1,3)  -- non-contiguous!
c = b.reshape(6)           # Can't reshape non-contiguous with one view
# This is where CONTIGUOUS comes in -- it forces a copy to make it contiguous
# Or rangeify handles it with div/mod decomposition
```

In the current tinygrad (post-rangeify era), multi-view ShapeTrackers are largely handled by rangeify's index decomposition. When you reshape a non-contiguous tensor, rangeify generates `div` and `mod` expressions to compute the correct memory offsets.

## Merge Dimensions

An important optimization: adjacent dimensions with compatible strides can be merged. If `stride[i] == shape[i+1] * stride[i+1]`, dimensions `i` and `i+1` can be collapsed:

```python
# shape=(2, 3, 4), strides=(12, 4, 1)
# Dimension 0 stride (12) == shape[1] * stride[1] (3 * 4 = 12) ✓
# Dimension 1 stride (4) == shape[2] * stride[2] (4 * 1 = 4) ✓
# Can merge all three -> shape=(24,), strides=(1,)
```

This is used to simplify kernel index expressions. A 3D tensor that's contiguous in memory can be treated as a 1D array, producing simpler GPU code.

## Connection to Rangeify

ShapeTracker concepts are what rangeify (Chapter 5) transforms into loops. When rangeify processes a movement op:

1. **RESHAPE** with ranges `[r0]` becomes `[r0//3, r0%3]` — div/mod decomposition
2. **PERMUTE** with ranges `[r0, r1]` becomes `[r1, r0]` — swap ranges
3. **EXPAND** with ranges `[r0, r1]` becomes `[0, r1]` — constant for broadcast dims
4. **SHRINK** with ranges `[r0]` becomes `[r0 + offset]` — shift
5. **FLIP** with ranges `[r0]` becomes `[size-1-r0]` — reverse

The stride-based indexing from ShapeTracker translates directly into the range expressions that rangeify produces.

## Exercises

1. **Compute strides**: For shape `(2, 3, 4)` with row-major (C) ordering, what are the strides? Verify: element `[1, 2, 3]` should be at offset `1*12 + 2*4 + 3*1 = 23`.

2. **Transpose strides**: For a matrix with shape `(4, 5)` and strides `(5, 1)`, what are the strides after `.permute(1, 0)`?

3. **Broadcast strides**: For a vector with shape `(1, 5)` and strides `(0, 1)`, expanded to shape `(3, 5)`, what value does element `[2, 3]` access in memory?

4. **When is contiguous needed?**: Try `Tensor.ones(2,3).permute(1,0).reshape(6)`. Does this work? Why or why not? What does rangeify do to handle it?

## Source Code Map

| File | What to read |
|------|-------------|
| `tinygrad/shape/view.py` | `View` class — the core shape/stride/offset structure |
| `tinygrad/schedule/indexing.py:142` | `apply_movement_op()` — how movement ops transform ranges |
| `tinygrad/schedule/indexing.py:126` | `_apply_reshape()` — reshape as div/mod decomposition |
