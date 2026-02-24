# Chapter 29: The Tensor Class — tinygrad's Public API

The `Tensor` class is the only thing most users interact with. It wraps a UOp (Chapter 2) and provides a familiar NumPy/PyTorch-like API. This chapter explains how `Tensor` sits on top of the UOp graph, how lazy evaluation works, and when computation actually happens.

## The Three Attributes

A Tensor has exactly three fields:

```python
class Tensor:
    __slots__ = "uop", "requires_grad", "grad"
```

That's it. No shape array, no stride tracker, no device field. Everything is derived from the `uop`:

```python
@property
def device(self) -> str|tuple[str, ...]: return self.uop.device

@property
def shape(self) -> tuple[sint, ...]: return self.uop.shape

@property
def dtype(self) -> DType: return self.uop.dtype
```

The Tensor is a thin wrapper. The UOp does all the work.

## Creating Tensors

When you write `Tensor([1, 2, 3])`, here's what happens:

```python
def __init__(self, data, device=None, dtype=None, requires_grad=None):
    # 1. Figure out dtype from the data
    if dtype is None:
        dtype = dtypes.default_int  # for [1, 2, 3]

    # 2. Pack the Python list into bytes
    data = _frompy(data, dtype)
    # _frompy creates a UOp.new_buffer on device "PYTHON"
    # and copies the data into it

    # 3. Copy to target device if needed
    if data.device != device:
        data = data.copy_to_device(device)

    self.uop = data
    self.grad = None
    self.requires_grad = requires_grad
```

Different input types take different paths:

| Input | What happens |
|-------|-------------|
| `Tensor(3.14)` | Creates a `UOp.const` — no buffer, just a constant in the graph |
| `Tensor([1,2,3])` | Packs into bytes via `struct.pack`, creates a buffer on `"PYTHON"` device |
| `Tensor(numpy_array)` | Creates a buffer on `"NPY"` device pointing at the numpy data |
| `Tensor(existing_uop)` | Uses the UOp directly |
| `Tensor(pathlib.Path)` | Creates a `DISK` buffer — data stays on disk until needed |

## Lazy Evaluation

The most important thing about Tensor: **nothing computes until you ask for the result.**

```python
a = Tensor([1, 2, 3])   # just builds a graph
b = Tensor([4, 5, 6])   # just builds a graph
c = a + b                # just builds a graph — no addition happens!
c = c * 2               # still just building the graph
print(c.numpy())         # NOW computation happens
```

Every operation calls `_apply_uop`, which creates a new UOp in the graph:

```python
def _apply_uop(self, fxn, *x, extra_args=(), **kwargs):
    srcs = (self,) + x
    new_uop = fxn(*[t.uop for t in srcs], *extra_args, **kwargs)

    # Create the result Tensor (no computation yet!)
    ret = Tensor.__new__(Tensor)
    ret.uop = new_uop
    ret.grad = None

    # Propagate requires_grad
    needs_input_grad = [t.requires_grad for t in srcs]
    ret.requires_grad = True if any(needs_input_grad) else \
                         None if None in needs_input_grad else False
    return ret
```

The `_binop` method shows how binary operations work:

```python
def _binop(self, op, x, reverse):
    lhs, rhs = self._broadcasted(x, reverse)  # handle broadcasting
    return lhs._apply_uop(lambda *u: u[0].alu(op, *u[1:]), rhs)
```

When you write `a + b`, Python calls `a.__add__(b)`, which calls `a._binop(Ops.ADD, b, False)`, which creates a UOp with `Ops.ADD`. No math happens.

## When Computation Happens

Computation is triggered by three methods:

### 1. `.realize()` — Explicit Trigger

```python
c = (a + b) * 2
c.realize()  # forces computation
```

Here's what `realize` does:

```python
def realize(self, *lst, do_update_stats=True):
    # 1. Handle pending assigns (for in-place operations)
    # 2. Check if already realized
    if self.uop.has_buffer_identity():
        return self  # already a concrete buffer, nothing to do

    # 3. Create schedule and run it
    run_schedule(*Tensor.schedule_with_vars(self))
    return self
```

### 2. `.numpy()` / `.item()` / `.data()` — Data Extraction

These call `._buffer()` which calls `.realize()` internally:

```python
def _buffer(self):
    x = self.cast(self.dtype.base).contiguous()
    return cast(Buffer, x.realize().uop.buffer).ensure_allocated()

def item(self):
    assert self.numel() == 1, "must have one element for item"
    return self.data()[(0,) * len(self.shape)]

def numpy(self):
    return self._buffer().numpy().reshape(self.shape)
```

### 3. `.backward()` — Gradient Computation

This triggers realization of the gradient graph (covered in Chapter 30).

## The Schedule

When computation is triggered, `schedule_with_vars` converts the UOp graph into a list of executable items:

```python
def schedule_with_vars(self, *lst):
    # Collect all UOps from all tensors being realized
    big_sink = UOp.sink(*[x.uop for x in (self,) + lst])

    # This is the big function — see Chapter 4 (Scheduling)
    becomes_map, schedule, var_vals = complete_create_schedule_with_vars(big_sink)

    # Update all live tensors to point to realized buffers
    _apply_map_to_tensors(becomes_map, name="buffers")

    return schedule, var_vals
```

After scheduling, the UOp graph is transformed into a list of `ExecItem`s — each one is a GPU kernel (or memory copy) that needs to run.

## The Global Tensor Registry

tinygrad tracks all live tensors:

```python
all_tensors: dict[weakref.ref[Tensor], None] = {}
```

When a tensor is created, it's added:
```python
all_tensors[weakref.ref(self)] = None
```

When it's garbage collected, it's removed:
```python
def __del__(self):
    all_tensors.pop(weakref.ref(self), None)
```

This registry is used by `_apply_map_to_tensors` — after realization, all live tensors need their UOps updated to point to the realized buffers instead of the old lazy graph:

```python
def _apply_map_to_tensors(applied_map, name):
    # Find tensors whose UOps reference anything in applied_map
    scope_tensors = [t for tref in all_tensors
                     if (t := tref()) is not None
                     and t.uop.topovisit(visitor, in_scope)]

    # Substitute old UOps with new ones
    sink = UOp.sink(*[t.uop for t in scope_tensors])
    new_sink = sink.substitute(applied_map)

    # Update each tensor's UOp
    for t, s, ns in zip(scope_tensors, sink.src, new_sink.src):
        if s is ns: continue
        t.uop = ns
```

## `.assign()` — In-Place Operations

In-place operations use `assign`, which creates a pending write:

```python
def assign(self, x):
    # Validate shapes and devices match
    assert self.shape == x.shape
    assert self.device == x.device
    assert self.dtype == x.dtype

    result = self._apply_uop(UOp.assign, x)

    # Track as pending assign — will be realized when the buffer is next read
    _pending_assigns.setdefault(buf_uop, []).append(result.uop)
    return self.replace(result)
```

This is how `Tensor.assign` works with optimizers:

```python
# In Adam optimizer:
self.b1_running *= self.b1  # creates pending assign
self.b2_running *= self.b2  # creates pending assign
# Both are realized when the next forward pass reads these buffers
```

## `.detach()` — Stopping Gradients

`detach` creates a new tensor that doesn't flow gradients:

```python
def detach(self):
    return Tensor(self.uop.detach(), device=self.device, requires_grad=False)
```

The `Ops.DETACH` operation in the UOp graph acts as a barrier — `compute_gradient` won't traverse past it.

## Broadcasting

When operands have different shapes, tinygrad broadcasts them automatically:

```python
a = Tensor.ones(3, 4)    # shape (3, 4)
b = Tensor.ones(4)       # shape (4,)
c = a + b                # b is broadcast to (3, 4)
```

Broadcasting works by:
1. Left-aligning shapes: `(4,)` becomes `(1, 4)`
2. Expanding: `(1, 4)` becomes `(3, 4)` via `Ops.EXPAND`

No data is copied — expanding just changes how the same data is indexed.

## Multiple Tensor Realization

You can realize multiple tensors at once for better fusion:

```python
a = Tensor([1, 2, 3])
b = a + 1
c = a * 2
# Realize both at once — the scheduler can optimize across them
b.realize(c)
```

## PyTorch Comparison

| Feature | PyTorch | tinygrad |
|---------|---------|---------|
| Evaluation | Eager (immediate) | Lazy (deferred) |
| Data structure | `torch.Tensor` (C++) | `Tensor` wrapping `UOp` (Python) |
| Shape tracking | Stored in tensor | Derived from UOp graph |
| Device | Stored in tensor | Derived from UOp graph |
| Autograd | Tape-based | Graph rewriting (Chapter 30) |
| Size | ~3M lines | ~10K lines |

## Exercises

1. **Trace a creation**: Run `DEBUG=4 python -c "from tinygrad import Tensor; t = Tensor([1,2,3]); print(t.numpy())"`. How many kernels are scheduled?

2. **Lazy proof**: Create `a = Tensor([1,2,3]); b = a + a + a + a`. Before calling `.numpy()`, no computation happens. Verify by checking `a.uop.op` — it should still be a buffer, not a computed result.

3. **Multiple realize**: Compare `a.realize(); b.realize()` vs `a.realize(b)`. With `DEBUG=2`, count the kernels. Realizing together may produce fewer kernels due to fusion.

4. **Read the source**: In `tinygrad/tensor.py`, find the `__init__` method. Trace what happens for each input type (int, list, numpy array, UOp).

## Source Code Map

| File | What to read |
|------|-------------|
| `tinygrad/tensor.py:102-175` | `Tensor.__init__` — all the creation paths |
| `tinygrad/tensor.py:179-190` | `_apply_uop` — how operations build the graph |
| `tinygrad/tensor.py:252-292` | `schedule_with_vars` and `realize` — when computation happens |
| `tinygrad/tensor.py:332-401` | `_buffer`, `data`, `item`, `numpy` — getting data out |
