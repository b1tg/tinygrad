# 第30章：Autograd — gradient 如何反向传播

每个 PyTorch 用户都知道 `loss.backward()`。但它到底是怎么工作的？本章解释 tinygrad 的 autograd 系统 — gradient 如何通过反向重写 UOp 图来计算。

## 总体概览

```
Forward:  x → multiply → add → reduce → loss
Backward: dx ← multiply ← add ← broadcast ← 1.0
```

在前向传播中，你构建一个计算图。在 backward 传播中，你反向遍历该图，在每个节点应用链式法则来计算 gradient。

## `backward()` 的10行核心代码

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

要点：
- 它找到计算图中所有 `requires_grad=True` 的存活 `Tensor`
- 它调用 `self.gradient()` 一次性计算所有 gradient
- gradient 是**累加**的（而非替换）— 这对参数共享很重要

## `gradient()` — 核心方法

```python
def gradient(self, *targets, gradient=None, materialize_grads=False):
    if gradient is None:
        gradient = Tensor(1.0, dtype=self.dtype, device=self.device)

    target_uops = [x.uop for x in targets]
    grads = compute_gradient(self.uop, gradient.uop, set(target_uops))
    return [Tensor(grads[x], device=t.device) for t, x in zip(targets, target_uops)]
```

真正的工作发生在 `tinygrad/gradient.py` 中的 `compute_gradient`。

## `compute_gradient` — 反向模式自动微分

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

这是教科书式的反向模式自动微分：
1. 从 loss 处以 `grad_output = 1.0` 开始
2. 反向遍历计算图
3. 在每个节点，使用链式法则计算局部 gradient
4. 当一个节点有多个消费者时，累加 gradient

## Gradient 规则

gradient 规则定义为一个 `PatternMatcher` — 与第3章相同的重写引擎：

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

每条规则接收 `ctx`（传入的 gradient，即 `grad_output`）并返回一个元组，包含对每个输入的 gradient。

### 为什么是 `ctx`？

在 PatternMatcher 中，`ctx` 是反向流入该节点的 gradient。可以把它理解为"如果这个节点的输出发生变化，最终的 loss 会变化多少？"

### Movement 操作的 Gradient

Movement 操作的 gradient 是其逆操作：

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

规律是：**movement 操作的 gradient 就是其逆 movement 操作。**

### Reduce 操作的 Gradient

Reduce 操作（sum、max）需要特殊处理：

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

对于 `sum`：gradient 被广播回去（每个元素的贡献相同）。
对于 `max`：只有最大值元素获得 gradient（其他元素没有贡献）。

## `_deepwalk` — 寻找 Gradient 路径

并非图中的每个节点都需要 gradient。`_deepwalk` 只找到从根节点到目标节点路径上的节点：

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

`Ops.DETACH` 充当 gradient 屏障 — 这就是 `tensor.detach()` 所创建的。

## 一个具体的例子

```python
x = Tensor([2.0, 3.0], requires_grad=True)
y = (x * x).sum()
y.backward()
print(x.grad.numpy())  # [4.0, 6.0]
```

逐步分析：

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

`x²` 的 gradient 是 `2x` — 正好是 `[4.0, 6.0]`。

## `requires_grad` 的传播

当你从需要 grad 的张量进行运算创建新张量时，结果会自动需要 grad：

```python
def _apply_uop(self, fxn, *x, **kwargs):
    needs_input_grad = [t.requires_grad for t in srcs]
    ret.requires_grad = True if any(needs_input_grad) else \
                         None if None in needs_input_grad else False
```

三种状态：
- `True`：确定需要 gradient
- `False`：确定不需要
- `None`：未知（如果放入 optimizer 中会变为 `True`）

## 练习

1. **手动计算 gradient**：用 `x = Tensor([1.0, 2.0], requires_grad=True)` 计算 `y = (x * 3 + 2).sum()`。`x.grad` 应该是什么？用 `y.backward()` 验证。

2. **Detach**：如果执行 `y = (x * x.detach()).sum(); y.backward()` 会发生什么？gradient 应该是 `x`（而不是 `2x`），因为 `x` 的一个副本被 detach 了。

3. **阅读规则**：在 `tinygrad/gradient.py` 中，找到 `Ops.MAX` 的 gradient 规则。为什么它要除以 `count`？

4. **比较操作没有 grad**：为什么 `CMPLT` 和 `CMPNE` 返回 `(None, None)`？如果你试图对比较操作求导会发生什么？

## 源代码索引

| 文件 | 阅读内容 |
|------|---------|
| `tinygrad/gradient.py` | 所有 gradient 规则和 `compute_gradient`（共88行） |
| `tinygrad/tensor.py:1029-1075` | `gradient()` 和 `backward()` 方法 |
