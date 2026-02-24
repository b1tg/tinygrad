# Chapter 20: MNIST — Your First Neural Network

MNIST is the "hello world" of deep learning: 70,000 handwritten digit images, each 28x28 pixels, labeled 0-9. In this chapter, you'll build a digit recognizer from scratch in tinygrad.

## What is MNIST?

MNIST is a dataset of handwritten digits. Each image is a 28x28 grayscale image (784 pixels), and each pixel is a value from 0 to 255. The task: given an image, predict which digit (0-9) it shows.

```
 ████
██  ██
    ██
   ██
  ██
 ██
████████
```
This is a "7". Your model needs to learn that this pattern of pixels means "7".

## Loading MNIST in Tinygrad

Tinygrad includes MNIST as a built-in dataset:

```python
from tinygrad.nn.datasets import mnist

X_train, Y_train, X_test, Y_test = mnist()
print(f"Training: {X_train.shape} images, {Y_train.shape} labels")
print(f"Test:     {X_test.shape} images, {Y_test.shape} labels")
# Training: (60000, 1, 28, 28) images, (60000,) labels
# Test:     (10000, 1, 28, 28) images, (10000,) labels
```

The shapes tell us:
- `(60000, 1, 28, 28)` — 60k images, 1 channel (grayscale), 28x28 pixels
- `(60000,)` — 60k labels (integers 0-9)

## The nn Module

Tinygrad's `nn` module provides the standard building blocks. If you've used PyTorch, these are familiar:

### Linear Layer

A linear layer computes `y = x @ W.T + b`:

```python
from tinygrad import nn

lin = nn.Linear(784, 128)   # 784 inputs -> 128 outputs
print(lin.weight.shape)      # (128, 784)
print(lin.bias.shape)        # (128,)
```

Internally, `nn.Linear.__call__` just calls `x.linear(self.weight.T, self.bias)`.

### Conv2d

A convolutional layer slides a small filter over the image:

```python
conv = nn.Conv2d(1, 32, 5)   # 1 input channel, 32 output filters, 5x5 kernel
print(conv.weight.shape)      # (32, 1, 5, 5)
```

This scans a 5x5 window across the image, producing 32 feature maps. Each feature map detects a different pattern (edges, corners, curves).

### BatchNorm

Batch normalization normalizes activations to have zero mean and unit variance, which stabilizes training:

```python
bn = nn.BatchNorm(32)  # normalize 32-channel feature maps
```

## Building the MNIST Model

Here's the full model from `examples/beautiful_mnist.py`:

```python
from typing import Callable
from tinygrad import Tensor, nn

class Model:
    def __init__(self):
        self.layers: list[Callable[[Tensor], Tensor]] = [
            nn.Conv2d(1, 32, 5), Tensor.relu,
            nn.Conv2d(32, 32, 5), Tensor.relu,
            nn.BatchNorm(32), Tensor.max_pool2d,
            nn.Conv2d(32, 64, 3), Tensor.relu,
            nn.Conv2d(64, 64, 3), Tensor.relu,
            nn.BatchNorm(64), Tensor.max_pool2d,
            lambda x: x.flatten(1), nn.Linear(576, 10)]

    def __call__(self, x: Tensor) -> Tensor:
        return x.sequential(self.layers)
```

Let's trace the shapes through the model:

```
Input:          (BS, 1, 28, 28)   # batch of grayscale 28x28 images
Conv2d(1,32,5): (BS, 32, 24, 24)  # 32 feature maps, 24x24 (28-5+1=24)
ReLU:           (BS, 32, 24, 24)  # zero out negatives
Conv2d(32,32,5):(BS, 32, 20, 20)  # 24-5+1=20
ReLU:           (BS, 32, 20, 20)
BatchNorm(32):  (BS, 32, 20, 20)  # normalize
MaxPool2d:      (BS, 32, 10, 10)  # halve spatial dims
Conv2d(32,64,3):(BS, 64, 8, 8)    # 10-3+1=8
ReLU:           (BS, 64, 8, 8)
Conv2d(64,64,3):(BS, 64, 6, 6)    # 8-3+1=6
ReLU:           (BS, 64, 6, 6)
BatchNorm(64):  (BS, 64, 6, 6)
MaxPool2d:      (BS, 64, 3, 3)    # halve again
Flatten:        (BS, 576)         # 64*3*3 = 576
Linear(576,10): (BS, 10)          # 10 class scores
```

The model outputs 10 numbers — one score for each digit. The highest score is the prediction.

### What Does `x.sequential(layers)` Do?

It applies each layer in order:

```python
# x.sequential([f, g, h]) is equivalent to:
x = f(x)
x = g(x)
x = h(x)
```

## The Training Loop

Training has four steps, repeated many times:

1. **Forward pass**: run images through the model to get predictions
2. **Compute loss**: measure how wrong the predictions are
3. **Backward pass**: compute gradients (how to adjust each weight)
4. **Update weights**: nudge weights in the direction that reduces loss

```python
from tinygrad import Tensor, TinyJit, nn, GlobalCounters
from tinygrad.nn.datasets import mnist

# Load data
X_train, Y_train, X_test, Y_test = mnist()

# Create model and optimizer
model = Model()
opt = nn.optim.Adam(nn.state.get_parameters(model))

@TinyJit
@Tensor.train()
def train_step() -> Tensor:
    opt.zero_grad()
    # Random batch of 512 images
    samples = Tensor.randint(512, high=X_train.shape[0])
    # Forward + loss + backward
    loss = model(X_train[samples]).sparse_categorical_crossentropy(Y_train[samples]).backward()
    return loss.realize(*opt.schedule_step())
```

### Understanding Each Line

**`nn.state.get_parameters(model)`** — walks the model's attributes and collects every `Tensor` that has `requires_grad=True` (the learnable weights).

**`nn.optim.Adam(params)`** — creates an Adam optimizer. Adam maintains running averages of gradients and squared gradients to adaptively set learning rates per parameter.

**`@Tensor.train()`** — context manager that sets `Tensor.training = True`. This enables dropout and batch norm's training behavior.

**`@TinyJit`** — caches the compiled computation graph. The first call compiles; subsequent calls replay the same kernels with new data (see Chapter 16).

**`opt.zero_grad()`** — clears old gradients.

**`Tensor.randint(512, high=60000)`** — randomly samples 512 indices from the training set.

**`sparse_categorical_crossentropy`** — the loss function. "Sparse" means labels are integers (0-9), not one-hot vectors. "Categorical cross-entropy" measures the distance between the model's predicted probability distribution and the true label.

**`.backward()`** — computes gradients for all parameters.

**`opt.schedule_step()`** — returns tensors that need to be realized to apply the weight updates.

### Evaluation

```python
@TinyJit
def get_test_acc() -> Tensor:
    return (model(X_test).argmax(axis=1) == Y_test).mean() * 100
```

`model(X_test).argmax(axis=1)` picks the digit with the highest score for each test image. Comparing with `Y_test` gives a boolean tensor, and `.mean() * 100` gives the accuracy percentage.

### The Full Loop

```python
from tinygrad.helpers import trange

for i in (t := trange(70)):
    GlobalCounters.reset()
    loss = train_step()
    if i % 10 == 9:
        test_acc = get_test_acc().item()
    t.set_description(f"loss: {loss.item():6.2f} test_accuracy: {test_acc:5.2f}%")
```

After ~70 steps, you should see >98% accuracy.

## Optimizers Explained

### SGD (Stochastic Gradient Descent)

The simplest optimizer. Updates each weight by:

```
weight = weight - lr * gradient
```

With momentum, it accumulates a velocity:

```
velocity = momentum * velocity + gradient
weight = weight - lr * velocity
```

### Adam

Adam adapts the learning rate for each parameter:

```python
m = b1 * m + (1 - b1) * gradient          # first moment (mean of gradients)
v = b2 * v + (1 - b2) * gradient^2        # second moment (mean of squared gradients)
m_hat = m / (1 - b1^t)                    # bias correction
v_hat = v / (1 - b2^t)
weight = weight - lr * m_hat / (sqrt(v_hat) + eps)
```

Parameters that have been receiving large gradients get smaller updates (stabilization), and parameters with small gradients get larger updates (exploration).

### Muon

Muon is a newer optimizer that applies the Newton-Schulz iteration to approximate the matrix square root inverse of the gradient:

```python
opt = nn.optim.Muon(nn.state.get_parameters(model))
```

All optimizers in tinygrad inherit from `Optimizer` and implement `_step()`.

## Loss Functions

### Cross-Entropy Loss

For classification, you want the model to output high probability for the correct class and low probability for wrong classes.

Cross-entropy loss: `L = -log(p_correct)`

If the model predicts the correct class with 90% probability: `L = -log(0.9) = 0.105` (small loss).
If the model predicts the correct class with 10% probability: `L = -log(0.1) = 2.303` (large loss).

### Softmax

The model outputs raw scores (logits). Softmax converts them to probabilities:

```python
probs = logits.softmax()   # each row sums to 1.0
```

In practice, `sparse_categorical_crossentropy` combines softmax + log + negative indexing for numerical stability.

## What Makes This "Beautiful"?

The `beautiful_mnist.py` example achieves >99% accuracy in 47 lines. Key design choices:

1. **Random sampling** instead of sequential batches — simpler, works well enough
2. **TinyJit** — the training loop compiles once and replays, making it fast
3. **No data augmentation** — keeps it simple
4. **Adam optimizer** — converges reliably without tuning

## Exercises

1. **Run it**: `python examples/beautiful_mnist.py` and observe the accuracy curve.

2. **Change the model**: Replace Conv2d layers with Linear layers (a "fully connected" model). How does accuracy change?

3. **Try Fashion-MNIST**: `FASHION=1 python examples/beautiful_mnist.py`. Fashion-MNIST has the same format but with clothing items instead of digits. Is it harder?

4. **Visualize**: After training, look at what the model predicts on specific test images:
   ```python
   pred = model(X_test[:10]).argmax(axis=1).numpy()
   true = Y_test[:10].numpy()
   print(f"Predicted: {pred}")
   print(f"True:      {true}")
   ```

5. **Inspect parameters**: Count the total number of learnable parameters:
   ```python
   total = sum(p.numel() for p in nn.state.get_parameters(model))
   print(f"Total parameters: {total:,}")
   ```

## Source Code Map

| File | What to read |
|------|-------------|
| `examples/beautiful_mnist.py` | The canonical MNIST example (47 lines) |
| `tinygrad/nn/__init__.py` | `Conv2d`, `Linear`, `BatchNorm`, `Embedding` |
| `tinygrad/nn/optim.py` | `Adam`, `SGD`, `Muon`, `LAMB` optimizers |
| `tinygrad/nn/state.py` | `get_parameters()`, `get_state_dict()`, `load_state_dict()` |
| `tinygrad/nn/datasets.py` | `mnist()` dataset loader |
