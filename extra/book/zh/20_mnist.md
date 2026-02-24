# 第20章：MNIST —— 你的第一个神经网络

MNIST 是深度学习的"hello world"：70,000 张手写数字图像，每张 28x28 像素，标签为 0-9。在本章中，你将使用 tinygrad 从零开始构建一个数字识别器。

## 什么是 MNIST？

MNIST 是一个手写数字数据集。每张图像是 28x28 的灰度图像（784 个像素），每个像素的值从 0 到 255。任务是：给定一张图像，预测它显示的是哪个数字（0-9）。

```
 ████
██  ██
    ██
   ██
  ██
 ██
████████
```
这是一个"7"。你的模型需要学习这种像素模式代表"7"。

## 在 Tinygrad 中加载 MNIST

Tinygrad 内置了 MNIST 数据集：

```python
from tinygrad.nn.datasets import mnist

X_train, Y_train, X_test, Y_test = mnist()
print(f"Training: {X_train.shape} images, {Y_train.shape} labels")
print(f"Test:     {X_test.shape} images, {Y_test.shape} labels")
# Training: (60000, 1, 28, 28) images, (60000,) labels
# Test:     (10000, 1, 28, 28) images, (10000,) labels
```

这些形状告诉我们：
- `(60000, 1, 28, 28)` —— 60k 张图像，1 个通道（灰度），28x28 像素
- `(60000,)` —— 60k 个标签（整数 0-9）

## nn 模块

Tinygrad 的 `nn` 模块提供了标准构建模块。如果你使用过 PyTorch，这些会很熟悉：

### Linear 层

Linear 层计算 `y = x @ W.T + b`：

```python
from tinygrad import nn

lin = nn.Linear(784, 128)   # 784 inputs -> 128 outputs
print(lin.weight.shape)      # (128, 784)
print(lin.bias.shape)        # (128,)
```

在内部，`nn.Linear.__call__` 只是调用了 `x.linear(self.weight.T, self.bias)`。

### Conv2d

卷积层��图像上滑动一个小的滤波器：

```python
conv = nn.Conv2d(1, 32, 5)   # 1 input channel, 32 output filters, 5x5 kernel
print(conv.weight.shape)      # (32, 1, 5, 5)
```

它在图像上扫描一个 5x5 的窗口，产生 32 个特征图。每个特征图检测不同的模式（边缘、角点、曲线）。

### BatchNorm

BatchNorm 将激活值归一化为零均值和单位方差，从而稳定训练过程：

```python
bn = nn.BatchNorm(32)  # normalize 32-channel feature maps
```

## 构建 MNIST 模型

以下是来自 `examples/beautiful_mnist.py` 的完整模型：

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

让我们追踪数据在模型中的形状变化：

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

模型输出 10 个数字 —— 每个数字对应一个分数。最高分数即为预测结果。

### `x.sequential(layers)` 做了什么？

它按顺序应用每一层：

```python
# x.sequential([f, g, h]) is equivalent to:
x = f(x)
x = g(x)
x = h(x)
```

## 训练循环

训练有四个步骤，重复多次：

1. **前向传播**：将图像输入模型以获得预测结果
2. **计算损失**：衡量预测的错误程度
3. **反向传播**：计算梯度（如何调整每个权重）
4. **更新权重**：沿着减少损失的方向微调权重

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

### 逐行理解

**`nn.state.get_parameters(model)`** —— 遍历模型的属性，收集所有 `requires_grad=True` 的 Tensor（即可学习的权重）。

**`nn.optim.Adam(params)`** —— 创建一个 Adam 优化器。Adam 维护梯度和梯度平方的滑动平均值，以自适应地为每个参数设置学习率。

**`@Tensor.train()`** —— 上下文管理器，设置 `Tensor.training = True`。这会启用 dropout 和 BatchNorm 的训练行为。

**`@TinyJit`** —— 缓存编译后的计算图。首次调用时编译；后续调用使用新数据重放相同的内核（见第16章）。

**`opt.zero_grad()`** —— 清除旧的梯度。

**`Tensor.randint(512, high=60000)`** —— 从训练集中随机采样 512 个索引。

**`sparse_categorical_crossentropy`** —— 损失函数。"Sparse"表示标签是整数（0-9），而不是 one-hot 向量。"Categorical cross-entropy"衡量模型预测的概率分布与真实标签之间的距离。

**`.backward()`** —— 为所有参数计算梯度。

**`opt.schedule_step()`** —— 返回需要被 realize 以应用权重更新的张量。

### 评估

```python
@TinyJit
def get_test_acc() -> Tensor:
    return (model(X_test).argmax(axis=1) == Y_test).mean() * 100
```

`model(X_test).argmax(axis=1)` 为每张测试图像选择得分最高的数字。与 `Y_test` 比较得到布尔张量，`.mean() * 100` 给出准确率百分比。

### 完整循环

```python
from tinygrad.helpers import trange

for i in (t := trange(70)):
    GlobalCounters.reset()
    loss = train_step()
    if i % 10 == 9:
        test_acc = get_test_acc().item()
    t.set_description(f"loss: {loss.item():6.2f} test_accuracy: {test_acc:5.2f}%")
```

经过约 70 步后，你应该能看到 >98% 的准确率。

## 优化器详解

### SGD（随机梯度下降）

最简单的优化器。每个权重的更新方式为：

```
weight = weight - lr * gradient
```

加上动量后，它会积累一个速度项：

```
velocity = momentum * velocity + gradient
weight = weight - lr * velocity
```

### Adam

Adam 为每个参数自适应地调整学习率：

```python
m = b1 * m + (1 - b1) * gradient          # first moment (mean of gradients)
v = b2 * v + (1 - b2) * gradient^2        # second moment (mean of squared gradients)
m_hat = m / (1 - b1^t)                    # bias correction
v_hat = v / (1 - b2^t)
weight = weight - lr * m_hat / (sqrt(v_hat) + eps)
```

接收到较大梯度的参数会获得较小的更新（稳定化），而梯度较小的参数会获得较大的更新（探索）。

### Muon

Muon 是一种较新的优化器，它应用 Newton-Schulz 迭代来近似梯度的矩阵平方根逆：

```python
opt = nn.optim.Muon(nn.state.get_parameters(model))
```

tinygrad 中所有优化器都继承自 `Optimizer` 并实现 `_step()` 方法。

## 损失函数

### 交叉熵损失

对于分类任务，你希望模型对正确类别输出高概率，对错误类别输出低概率。

交叉熵损失：`L = -log(p_correct)`

如果模型以 90% 的概率预测正确类别：`L = -log(0.9) = 0.105`（损失较小）。
如果模型以 10% 的概率预测正确类别：`L = -log(0.1) = 2.303`（损失较大）。

### Softmax

模型输出原始分数（logits）。Softmax 将它们转换为概率：

```python
probs = logits.softmax()   # each row sums to 1.0
```

在实际应用中，`sparse_categorical_crossentropy` 将 softmax + log + 负索引组合在一起以保证数值稳定性。

## 什么让它"优雅"？

`beautiful_mnist.py` 示例仅用 47 行代码就实现了 >99% 的准确率。关键设计选择：

1. **随机采样**而非顺序批次 —— 更简单，效果足够好
2. **TinyJit** —— 训练循环只编译一次然后重放，使其运行更快
3. **无数据增强** —— 保持简单
4. **Adam 优化器** —— 无需调参即可可靠收敛

## 练习

1. **运行它**：执行 `python examples/beautiful_mnist.py` 并观察准确率曲线。

2. **修改模型**：将 Conv2d 层替换为 Linear 层（"全连接"模型）。准确率会有什么变化？

3. **尝试 Fashion-MNIST**：执行 `FASHION=1 python examples/beautiful_mnist.py`。Fashion-MNIST 具有相同的格式，但用服装物品替代了数字。它更难吗？

4. **可视化**：训练后，查看模型对特定测试图像的预测：
   ```python
   pred = model(X_test[:10]).argmax(axis=1).numpy()
   true = Y_test[:10].numpy()
   print(f"Predicted: {pred}")
   print(f"True:      {true}")
   ```

5. **检查参数**：统计可学习参数的总数：
   ```python
   total = sum(p.numel() for p in nn.state.get_parameters(model))
   print(f"Total parameters: {total:,}")
   ```

## 源代码索引

| 文件 | 阅读内容 |
|------|---------|
| `examples/beautiful_mnist.py` | 经典的 MNIST 示例（47 行） |
| `tinygrad/nn/__init__.py` | `Conv2d`、`Linear`、`BatchNorm`、`Embedding` |
| `tinygrad/nn/optim.py` | `Adam`、`SGD`、`Muon`、`LAMB` 优化器 |
| `tinygrad/nn/state.py` | `get_parameters()`、`get_state_dict()`、`load_state_dict()` |
| `tinygrad/nn/datasets.py` | `mnist()` 数据集加载器 |
