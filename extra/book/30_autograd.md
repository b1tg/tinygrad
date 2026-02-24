# Chapter 30: Autograd — How Gradients Flow Backward

Every PyTorch user knows `loss.backward()`. But how does it actually work? This chapter explains tinygrad's autograd system — how gradients are computed by rewriting the UOp graph in reverse.

## The Big Picture

```
Forward:  x → multiply → add → reduce → loss
Backward: dx ← multiply ← add ← broadcast ← 1.0
```

In the forward pass, you build a computation graph. In the backward pass, you walk that graph in reverse, applying the chain rule at each node to compute gradients.

## `backward()` in 10 Lines

```python
def backward(self, gradient=None):
    # 1. Find all UOps in the forward graph
    all_uops = self.uop.toposort()

    # 2. Find all live Tensors that need gradients
    tensors_need_grad = [t for tref in all_tensors
                         if (t := tref()) is not None
                         and t.uop in all_uops
                         and t.requires_grad]

    # 3. Compute gradients for each target
    for t, g in zip(tensors_need_grad,
                    self.gradient(*tensors_need_grad, gradient=gradient,
                                 materialize_grads=True)):
        if t.grad is None: t.grad = g
        else: t.grad.assign(t.grad + g)  # accumulate
    return self
```

Key points:
- It finds every live `Tensor` with `requires_grad=True` that's in the computation graph
- It calls `self.gradient()` to compute all gradients at once
- Gradients are **accumulated** (not replaced) — this is important for parameter sharing

## `gradient()` — The Core

```python
def gradient(self, *targets, gradient=None, materialize_grads=False):
    if gradient is None:
        gradient = Tensor(1.0, dtype=self.dtype, device=self.device)

    target_uops = [x.uop for x in targets]
    grads = compute_gradient(self.uop, gradient.uop, set(target_uops))
    return [Tensor(grads[x], device=t.device) for t, x in zip(targets, target_uops)]
```

The real work happens in `compute_gradient` from `tinygrad/gradient.py`.

## `compute_gradient` — Reverse-Mode Autodiff

```python
def compute_gradient(root, root_grad, targets):
    # Start: the gradient of the output w.r.t. itself is 1.0
    grads = {root: root_grad}

    # Walk the graph in reverse topological order
    for t0 in reversed(_deepwalk(root, targets)):
        if t0 not in grads: continue

        # Apply the gradient rule for this operation
        lgrads = pm_gradient.rewrite(t0, ctx=grads[t0])

        # Distribute gradients to inputs
        for k, v in zip(t0.src, lgrads):
            if v is None: continue
            if k in grads: grads[k] = grads[k] + v  # accumulate
            else: grads[k] = v

    return grads
```

This is textbook reverse-mode autodiff:
1. Start with `grad_output = 1.0` at the loss
2. Walk backward through the graph
3. At each node, compute local gradients using the chain rule
4. Accumulate gradients when a node has multiple consumers

## The Gradient Rules

The gradient rules are defined as a `PatternMatcher` — the same rewriting engine from Chapter 3:

```python
pm_gradient = PatternMatcher([
    # Addition: gradient passes through unchanged
    (UPat(Ops.ADD), lambda ctx: (ctx, ctx)),

    # Multiplication: grad * other_input
    (UPat(Ops.MUL, name="ret"), lambda ctx, ret: (ret.src[1]*ctx, ret.src[0]*ctx)),

    # Reciprocal: -grad * ret^2
    (UPat(Ops.RECIPROCAL, name="ret"), lambda ctx, ret: (-ctx * ret * ret,)),

    # exp2: grad * ret * ln(2)
    (UPat(Ops.EXP2, name="ret"), lambda ctx, ret: (ret * ctx * math.log(2),)),

    # log2: grad / (x * ln(2))
    (UPat(Ops.LOG2, name="ret"), lambda ctx, ret: (ctx / (ret.src[0] * math.log(2)),)),

    # sin: grad * cos(x)
    (UPat(Ops.SIN, name="ret"),
     lambda ctx, ret: ((math.pi/2 - ret.src[0]).sin() * ctx,)),

    # sqrt: grad / (2 * sqrt(x))
    (UPat(Ops.SQRT, name="ret"), lambda ctx, ret: (ctx / (ret * 2),)),

    # cast: cast the gradient back
    (UPat(Ops.CAST, name="ret"), lambda ctx, ret: (ctx.cast(ret.src[0].dtype),)),

    # comparison ops: no gradient (not differentiable)
    (UPat((Ops.CMPLT, Ops.CMPNE)), lambda: (None, None)),

    # where: route gradient to the selected branch
    (UPat(Ops.WHERE, name="ret"),
     lambda ctx, ret: (None,
                       ret.src[0].where(ctx, ctx.const_like(0)),
                       ret.src[0].where(ctx.const_like(0), ctx))),
])
```

Each rule takes `ctx` (the incoming gradient, i.e., `grad_output`) and returns a tuple of gradients for each input.

### Why `ctx`?

In the pattern matcher, `ctx` is the gradient flowing backward into this node. Think of it as "how much does the final loss change if this node's output changes?"

### Movement Op Gradients

Movement operations have inverse gradients:

```python
# reshape: reshape the gradient back to the input shape
(UPat(Ops.RESHAPE, name="ret"),
 lambda ctx, ret: (ctx.reshape(ret.src[0].shape), None)),

# expand: sum the gradient over expanded dimensions
(UPat(Ops.EXPAND, name="ret"),
 lambda ctx, ret: (ctx.r(Ops.ADD, tuple(i for i, (s, n) in
     enumerate(zip(ret.src[0].shape, ret.shape)) if s != n)), None)),

# pad: shrink the gradient (remove the padded regions)
(UPat(Ops.PAD, name="ret"),
 lambda ctx, ret: (ctx.shrink(...), None, None)),

# permute: permute with inverse permutation
(UPat(Ops.PERMUTE, name="ret"),
 lambda ctx, ret: (ctx.permute(argsort(ret.marg)),)),
```

The pattern: **the gradient of a movement op is the inverse movement op.**

### Reduce Gradients

Reduce operations (sum, max) need special handling:

```python
def reduce_gradient(ctx, ret, op):
    def broadcast_to_input(x):
        return x.reshape(x.shape + (1,) * (len(ret.src[0].shape) - len(x.shape))) \
               .expand(ret.src[0].shape)

    if op == Ops.ADD:
        # sum gradient: broadcast back to input shape
        return (broadcast_to_input(ctx),)

    if op == Ops.MAX:
        # max gradient: only flows to the element(s) that were the max
        mask = ret.src[0].eq(broadcast_to_input(ret)).cast(ctx.dtype)
        count = mask.r(Ops.ADD, ret.arg[1])
        return ((mask / broadcast_to_input(count)) * broadcast_to_input(ctx),)

    if op == Ops.MUL:
        # product gradient: grad * product / each_element
        return (broadcast_to_input(ctx * ret) / ret.src[0],)
```

For `sum`: the gradient is broadcast back (every element contributed equally).
For `max`: only the maximum element(s) get gradient (others contributed nothing).

## `_deepwalk` — Finding the Gradient Path

Not every node in the graph needs gradients. `_deepwalk` finds only the nodes on the path from the root to the targets:

```python
def _deepwalk(root, targets):
    # Top-down: mark nodes that lead to targets
    in_target_path = {}
    for u in root.toposort():
        in_target_path[u] = any(x in targets or in_target_path[x]
                                for x in u.src)

    # Return only nodes on the path, excluding DETACH and ASSIGN
    return list(root.toposort(
        lambda node: node.op not in {Ops.DETACH, Ops.ASSIGN}
                     and in_target_path[node]))
```

`Ops.DETACH` acts as a gradient barrier — this is what `tensor.detach()` creates.

## A Concrete Example

```python
x = Tensor([2.0, 3.0], requires_grad=True)
y = (x * x).sum()
y.backward()
print(x.grad.numpy())  # [4.0, 6.0]
```

Step by step:

```
Forward graph:
  x (BUFFER)
  → x * x (MUL)
  → sum (REDUCE_AXIS, Ops.ADD)
  → y (scalar)

Backward walk (reversed topological order):

1. y: grads[y] = 1.0

2. sum (REDUCE_AXIS, ADD):
   grad rule: broadcast_to_input(ctx) = broadcast(1.0) = [1.0, 1.0]
   grads[x*x] = [1.0, 1.0]

3. x * x (MUL):
   grad rule: (src[1]*ctx, src[0]*ctx) = (x * [1,1], x * [1,1])
   grads[x] += x * 1.0 = [2.0, 3.0]  (from first input)
   grads[x] += x * 1.0 = [2.0, 3.0]  (from second input, accumulated)
   grads[x] = [4.0, 6.0]
```

The gradient of `x²` is `2x` — exactly `[4.0, 6.0]`.

## `requires_grad` Propagation

When you create a tensor from operations on grad-requiring tensors, the result automatically requires grad:

```python
def _apply_uop(self, fxn, *x, **kwargs):
    needs_input_grad = [t.requires_grad for t in srcs]
    ret.requires_grad = True if any(needs_input_grad) else \
                         None if None in needs_input_grad else False
```

Three states:
- `True`: definitely needs gradient
- `False`: definitely doesn't
- `None`: unknown (will become `True` if put in an optimizer)

## Exercises

1. **Manual gradient**: Compute `y = (x * 3 + 2).sum()` with `x = Tensor([1.0, 2.0], requires_grad=True)`. What should `x.grad` be? Verify with `y.backward()`.

2. **Detach**: What happens if you do `y = (x * x.detach()).sum(); y.backward()`? The gradient should be `x` (not `2x`), because one copy of `x` is detached.

3. **Read the rules**: In `tinygrad/gradient.py`, find the gradient rule for `Ops.MAX`. Why does it divide by `count`?

4. **No grad for comparisons**: Why do `CMPLT` and `CMPNE` return `(None, None)`? What would happen if you tried to differentiate through a comparison?

## Source Code Map

| File | What to read |
|------|-------------|
| `tinygrad/gradient.py` | All gradient rules and `compute_gradient` (88 lines total) |
| `tinygrad/tensor.py:1029-1075` | `gradient()` and `backward()` methods |
