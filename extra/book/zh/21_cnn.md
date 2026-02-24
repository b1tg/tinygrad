# 第21章：CNN — ResNet、EfficientNet 与图像分类

卷积神经网络（CNN）是计算机视觉的基础。本章将解释 CNN 的工作原理，并详细介绍 tinygrad 中实现的两个生产级模型：ResNet 和 EfficientNet。

## 为什么需要 CNN？

在第20章中，我们使用了 Conv2d 层，但没有解释*为什么*要这样做。以下是关键要点：

一个**全连接**层处理 224x224 的 RGB 图像时，每个输出神经元有 `224 * 224 * 3 = 150,528` 个输入权重。参数太多了，而且忽略了图像的空间结构。

**卷积**层使用一个小型滤波器（例如 3x3）在图像上滑动。这带来了两个特性：

1. **权重共享**：同一个 3x3 滤波器在每个位置都被复用——大幅减少参数数量
2. **平移不变性**：在左上角检测到猫的滤波器，同样能在右下角检测到猫

## tinygrad 中的卷积

```python
from tinygrad import Tensor, nn

# A 3x3 convolution: 3 input channels (RGB), 64 output channels
conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
print(conv.weight.shape)  # (64, 3, 3, 3) = 64 filters, each 3x3x3

x = Tensor.rand(1, 3, 224, 224)  # batch=1, RGB, 224x224
y = conv(x)
print(y.shape)  # (1, 64, 224, 224) — 64 feature maps
```

在底层，`x.conv2d(weight)` 使用了 `_pool` 技巧（第10章）将输入重塑为滑动窗口，然后执行矩阵乘法。

## 现代 CNN 的基本组件

### ReLU

最简单的激活函数：`max(0, x)`。它引入了非线性——没有它的话，堆叠线性层只不过是一个大的线性层。

```python
x = Tensor([-1, 0, 1, 2])
print(x.relu().numpy())  # [0, 0, 1, 2]
```

### 池化

最大池化取每个窗口中的最大值，从而缩减空间维度：

```python
x = Tensor.rand(1, 64, 8, 8)
y = x.max_pool2d(kernel_size=2)  # halves each spatial dimension
print(y.shape)  # (1, 64, 4, 4)
```

### 批归一化（Batch Normalization）

在批次维度上对每个通道进行归一化，使其均值为零、方差为一：

```python
bn = nn.BatchNorm(64)
x = Tensor.rand(4, 64, 8, 8)  # batch of 4
y = bn(x)  # each of the 64 channels is normalized
```

在训练期间，它从当前批次计算均值和方差。在推理期间，它使用训练过程中累积的运行统计量。

### 全局平均池化（Global Average Pooling）

全局平均池化不是将特征图展平（这依赖于输入大小），而是对空间维度取均值：

```python
x = Tensor.rand(1, 512, 7, 7)
y = x.mean([2, 3])  # average over height and width
print(y.shape)  # (1, 512)
```

这使得模型可以接受任意大小的输入（只要对卷积层来说足够大）。

## ResNet：残差网络

### ResNet 解决的问题

更深的网络应该更好，对吧？但实际上，非常深的网络（20层以上）训练效果反而*不如*较浅的网络。这不是因为过拟合，而是因为梯度消失——梯度变得太小，无法更新前面的层。

### 残差连接

ResNet 的核心思想：不直接学习 `H(x)`，而是学习*残差* `F(x) = H(x) - x`，然后加回来：`H(x) = F(x) + x`。

```
Input x ──────────────────────┐
    │                         │ (skip connection)
    ├─→ Conv → BN → ReLU      │
    ├─→ Conv → BN             │
    └─────────────→ + ← ──────┘
                   │
                  ReLU
                   │
                Output
```

如果最优函数接近恒等映射，残差 `F(x)` 就接近于零，这很容易学习。梯度通过跳跃连接直接流动，解决了梯度消失问题。

### BasicBlock（ResNet-18/34）

来自 `extra/models/resnet.py`：

```python
class BasicBlock:
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               padding=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = []
        if stride != 1 or in_planes != planes:
            self.downsample = [
                nn.Conv2d(in_planes, planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            ]

    def __call__(self, x):
        out = self.bn1(self.conv1(x)).relu()
        out = self.bn2(self.conv2(out))
        out = out + x.sequential(self.downsample)  # skip connection!
        return out.relu()
```

当维度发生变化时（步长 > 1 或通道数改变），需要 `downsample` 路径。它使用 1x1 卷积来匹配形状，以便进行加法运算。

### Bottleneck（ResNet-50/101/152）

对于更深的模型，Bottleneck 块使用 1x1 -> 3x3 -> 1x1 的模式来减少计算量：

```python
class Bottleneck:
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        width = planes
        self.conv1 = nn.Conv2d(in_planes, width, kernel_size=1, bias=False)  # reduce channels
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, padding=1,
                               stride=stride, bias=False)              # 3x3 conv
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)  # expand channels

    def __call__(self, x):
        out = self.bn1(self.conv1(x)).relu()     # 256 -> 64
        out = self.bn2(self.conv2(out)).relu()   # 64 -> 64 (3x3 conv)
        out = self.bn3(self.conv3(out))          # 64 -> 256
        return (out + x.sequential(self.downsample)).relu()
```

1x1 卷积计算代价很低（没有空间计算），因此昂贵的 3x3 卷积只需在更少的通道上操作。

### 完整的 ResNet

```python
class ResNet:
    def __init__(self, num, num_classes=1000):
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def forward(self, x):
        out = self.bn1(self.conv1(x)).relu()
        out = out.pad([1,1,1,1]).max_pool2d((3,3), 2)
        out = out.sequential(self.layer1)   # 56x56
        out = out.sequential(self.layer2)   # 28x28
        out = out.sequential(self.layer3)   # 14x14
        out = out.sequential(self.layer4)   # 7x7
        out = out.mean([2, 3])              # global average pool
        return self.fc(out)
```

ResNet 的各个变体仅在块的数量上有所不同：

| 模型 | 块数 | 参数量 | 块类型 |
|-------|--------|-----------|------------|
| ResNet-18  | [2,2,2,2]    | 11M  | BasicBlock |
| ResNet-34  | [3,4,6,3]    | 22M  | BasicBlock |
| ResNet-50  | [3,4,6,3]    | 25M  | Bottleneck |
| ResNet-101 | [3,4,23,3]   | 44M  | Bottleneck |
| ResNet-152 | [3,8,36,3]   | 60M  | Bottleneck |

### 加载预训练权重

```python
from extra.models.resnet import ResNet

model = ResNet(18, num_classes=1000)
model.load_from_pretrained()  # downloads ImageNet weights from PyTorch hub
```

## EfficientNet：高效缩放 CNN

### 核心思想

如何让 CNN 变得更好？你可以：
- 让它更**宽**（更多通道）
- 让它更**深**（更多层）
- 使用更高**分辨率**的输入

EfficientNet 的洞察：使用复合缩放系数同时缩放这三个维度。

### MBConvBlock：基本构建块

EfficientNet 使用移动端反向瓶颈块（来自 MobileNetV2）：

```
Input
  ├─→ 1x1 Conv (expand channels)
  ├─→ Depthwise 3x3/5x5 Conv
  ├─→ Squeeze-and-Excite (channel attention)
  ├─→ 1x1 Conv (project back)
  └─→ + Input (skip connection if shapes match)
```

**深度可分离卷积**对每个通道单独应用一个滤波器（而不是混合通道），使其计算代价大大降低：

```python
# Standard conv: 32 * 32 * 3 * 3 = 9,216 parameters
conv = nn.Conv2d(32, 32, 3, padding=1)

# Depthwise conv: 32 * 1 * 3 * 3 = 288 parameters (32x fewer!)
# In tinygrad: groups=in_channels
x.conv2d(weight, groups=32)
```

**Squeeze-and-Excite** 学习哪些通道更重要：

```python
# Global average pool -> FC -> ReLU -> FC -> Sigmoid -> Scale
squeezed = x.avg_pool2d(kernel_size=x.shape[2:4])  # (B, C, 1, 1)
scale = squeezed.conv2d(se_reduce).swish().conv2d(se_expand).sigmoid()
x = x * scale  # re-weight channels
```

### EfficientNet 变体

EfficientNet-B0 到 B7 仅在宽度和深度乘数上有所不同：

| 模型 | 宽度 | 深度 | 分辨率 | 参数量 |
|-------|-------|-------|-----------|-----------|
| B0 | 1.0 | 1.0 | 224 | 5.3M |
| B1 | 1.0 | 1.1 | 240 | 7.8M |
| B4 | 1.4 | 1.8 | 380 | 19M |
| B7 | 2.0 | 3.1 | 600 | 66M |

## CIFAR-10 训练

CIFAR-10 比 MNIST 更进一步：60,000 张 32x32 彩色图像，分为10个类别（飞���、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车）。

```python
from tinygrad.nn.datasets import cifar

X_train, Y_train, X_test, Y_test = cifar()
print(X_train.shape)  # (50000, 3, 32, 32)
```

`examples/hlb_cifar10.py` 实现了一个快速的 CIFAR-10 训练器，使用自定义的 SpeedyResNet，可以达到 94% 以上的准确率。它使用了激进的数据增强（随机裁剪、翻转）和单周期学习率调度。

## 练习

1. **运行 ResNet 推理**：加载预训练的 ResNet-18 并对图像进行分类：
   ```python
   from extra.models.resnet import ResNet
   model = ResNet(18, num_classes=1000)
   model.load_from_pretrained()
   x = Tensor.rand(1, 3, 224, 224)  # random image
   pred = model(x).argmax().item()
   print(f"Predicted class: {pred}")
   ```

2. **统计参数数量**：比较 ResNet-18 和 ResNet-50 的参数数量。

3. **追踪形状变化**：对于 (1, 3, 224, 224) 的输入，追踪张量在 ResNet-18 每一层中的形状变化。空间维度在哪里缩小了？

4. **阅读 CIFAR 示例**：打开 `examples/hlb_cifar10.py`，找到数据增强的应用位置。它使用了哪些增强方法？

## 源代码索引

| 文件 | 阅读内容 |
|------|-------------|
| `extra/models/resnet.py` | ResNet（BasicBlock、Bottleneck、ResNet 类） |
| `extra/models/efficientnet.py` | EfficientNet（MBConvBlock、复合缩放） |
| `extra/models/convnext.py` | ConvNeXt（现代化的 CNN） |
| `examples/hlb_cifar10.py` | 快速 CIFAR-10 训练 |
| `examples/beautiful_cifar.py` | 简洁的 CIFAR-10 训练 |
| `examples/train_resnet.py` | 在 ImageNet 上训练 ResNet |
