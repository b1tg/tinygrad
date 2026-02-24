# Chapter 21: CNNs — ResNet, EfficientNet & Image Classification

Convolutional neural networks (CNNs) are the foundation of computer vision. This chapter explains how CNNs work and walks through two production models implemented in tinygrad: ResNet and EfficientNet.

## Why CNNs?

In Chapter 20, we used Conv2d layers without explaining *why*. Here's the key insight:

A **fully connected** layer treating a 224x224 RGB image has `224 * 224 * 3 = 150,528` input weights per output neuron. That's too many parameters, and it ignores the spatial structure of images.

A **convolutional** layer uses a small filter (e.g. 3x3) that slides across the image. This gives two properties:

1. **Weight sharing**: the same 3x3 filter is applied everywhere — dramatically fewer parameters
2. **Translation invariance**: a cat detected in the top-left is detected by the same filter in the bottom-right

## Convolution in Tinygrad

```python
from tinygrad import Tensor, nn

# A 3x3 convolution: 3 input channels (RGB), 64 output channels
conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
print(conv.weight.shape)  # (64, 3, 3, 3) = 64 filters, each 3x3x3

x = Tensor.rand(1, 3, 224, 224)  # batch=1, RGB, 224x224
y = conv(x)
print(y.shape)  # (1, 64, 224, 224) — 64 feature maps
```

Under the hood, `x.conv2d(weight)` uses the `_pool` trick (Chapter 10) to reshape the input into sliding windows, then does a matrix multiply.

## Building Blocks of Modern CNNs

### ReLU

The simplest activation function: `max(0, x)`. It introduces non-linearity — without it, stacking linear layers would just be one big linear layer.

```python
x = Tensor([-1, 0, 1, 2])
print(x.relu().numpy())  # [0, 0, 1, 2]
```

### Pooling

Max pooling takes the maximum value in each window, reducing spatial dimensions:

```python
x = Tensor.rand(1, 64, 8, 8)
y = x.max_pool2d(kernel_size=2)  # halves each spatial dimension
print(y.shape)  # (1, 64, 4, 4)
```

### Batch Normalization

Normalizes each channel to have zero mean and unit variance across the batch:

```python
bn = nn.BatchNorm(64)
x = Tensor.rand(4, 64, 8, 8)  # batch of 4
y = bn(x)  # each of the 64 channels is normalized
```

During training, it computes mean/variance from the current batch. During inference, it uses running statistics accumulated during training.

### Global Average Pooling

Instead of flattening a feature map (which depends on input size), global average pooling takes the mean over the spatial dimensions:

```python
x = Tensor.rand(1, 512, 7, 7)
y = x.mean([2, 3])  # average over height and width
print(y.shape)  # (1, 512)
```

This makes the model accept any input size (as long as it's large enough for the conv layers).

## ResNet: Residual Networks

### The Problem ResNet Solves

Deeper networks should be better, right? In practice, very deep networks (20+ layers) train *worse* than shallower ones. Not because of overfitting, but because gradients vanish — they get too small to update early layers.

### The Residual Connection

ResNet's key idea: instead of learning `H(x)` directly, learn the *residual* `F(x) = H(x) - x`, then add back: `H(x) = F(x) + x`.

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

If the optimal function is close to identity, the residual `F(x)` is close to zero, which is easy to learn. Gradients flow directly through the skip connection, solving the vanishing gradient problem.

### BasicBlock (ResNet-18/34)

From `extra/models/resnet.py`:

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

The `downsample` path is needed when dimensions change (stride > 1 or channel count changes). It applies a 1x1 convolution to match shapes for the addition.

### Bottleneck (ResNet-50/101/152)

For deeper models, Bottleneck blocks use a 1x1 → 3x3 → 1x1 pattern to reduce computation:

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

The 1x1 convolutions are cheap (no spatial computation), so the expensive 3x3 conv operates on fewer channels.

### The Full ResNet

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

ResNet variants differ only in the number of blocks:

| Model | Blocks | Parameters | Block Type |
|-------|--------|-----------|------------|
| ResNet-18  | [2,2,2,2]    | 11M  | BasicBlock |
| ResNet-34  | [3,4,6,3]    | 22M  | BasicBlock |
| ResNet-50  | [3,4,6,3]    | 25M  | Bottleneck |
| ResNet-101 | [3,4,23,3]   | 44M  | Bottleneck |
| ResNet-152 | [3,8,36,3]   | 60M  | Bottleneck |

### Loading Pretrained Weights

```python
from extra.models.resnet import ResNet

model = ResNet(18, num_classes=1000)
model.load_from_pretrained()  # downloads ImageNet weights from PyTorch hub
```

## EfficientNet: Scaling CNNs Efficiently

### The Idea

How do you make a CNN better? You can:
- Make it **wider** (more channels)
- Make it **deeper** (more layers)
- Use higher **resolution** inputs

EfficientNet's insight: scale all three dimensions together with a compound scaling coefficient.

### MBConvBlock: The Building Block

EfficientNet uses Mobile Inverted Bottleneck blocks (from MobileNetV2):

```
Input
  ├─→ 1x1 Conv (expand channels)
  ├─→ Depthwise 3x3/5x5 Conv
  ├─→ Squeeze-and-Excite (channel attention)
  ├─→ 1x1 Conv (project back)
  └─→ + Input (skip connection if shapes match)
```

**Depthwise convolution** applies one filter per channel (instead of mixing channels), making it much cheaper:

```python
# Standard conv: 32 * 32 * 3 * 3 = 9,216 parameters
conv = nn.Conv2d(32, 32, 3, padding=1)

# Depthwise conv: 32 * 1 * 3 * 3 = 288 parameters (32x fewer!)
# In tinygrad: groups=in_channels
x.conv2d(weight, groups=32)
```

**Squeeze-and-Excite** learns which channels matter:

```python
# Global average pool -> FC -> ReLU -> FC -> Sigmoid -> Scale
squeezed = x.avg_pool2d(kernel_size=x.shape[2:4])  # (B, C, 1, 1)
scale = squeezed.conv2d(se_reduce).swish().conv2d(se_expand).sigmoid()
x = x * scale  # re-weight channels
```

### EfficientNet Variants

EfficientNet-B0 through B7 differ only in width and depth multipliers:

| Model | Width | Depth | Resolution | Parameters |
|-------|-------|-------|-----------|-----------|
| B0 | 1.0 | 1.0 | 224 | 5.3M |
| B1 | 1.0 | 1.1 | 240 | 7.8M |
| B4 | 1.4 | 1.8 | 380 | 19M |
| B7 | 2.0 | 3.1 | 600 | 66M |

## CIFAR-10 Training

CIFAR-10 is a step up from MNIST: 60,000 32x32 color images in 10 classes (airplane, car, bird, cat, deer, dog, frog, horse, ship, truck).

```python
from tinygrad.nn.datasets import cifar

X_train, Y_train, X_test, Y_test = cifar()
print(X_train.shape)  # (50000, 3, 32, 32)
```

The `examples/hlb_cifar10.py` implements a fast CIFAR-10 trainer using a custom SpeedyResNet, achieving 94%+ accuracy. It uses aggressive data augmentation (random crops, flips) and a one-cycle learning rate schedule.

## Exercises

1. **Run ResNet inference**: Load a pretrained ResNet-18 and classify an image:
   ```python
   from extra.models.resnet import ResNet
   model = ResNet(18, num_classes=1000)
   model.load_from_pretrained()
   x = Tensor.rand(1, 3, 224, 224)  # random image
   pred = model(x).argmax().item()
   print(f"Predicted class: {pred}")
   ```

2. **Count parameters**: Compare parameter counts of ResNet-18 vs ResNet-50.

3. **Trace the shapes**: For a (1, 3, 224, 224) input, trace the tensor shape through every layer of ResNet-18. Where does the spatial dimension shrink?

4. **Read the CIFAR example**: Open `examples/hlb_cifar10.py` and find where data augmentation is applied. What augmentations does it use?

## Source Code Map

| File | What to read |
|------|-------------|
| `extra/models/resnet.py` | ResNet (BasicBlock, Bottleneck, ResNet class) |
| `extra/models/efficientnet.py` | EfficientNet (MBConvBlock, compound scaling) |
| `extra/models/convnext.py` | ConvNeXt (modernized CNN) |
| `examples/hlb_cifar10.py` | Fast CIFAR-10 training |
| `examples/beautiful_cifar.py` | Simple CIFAR-10 training |
| `examples/train_resnet.py` | ResNet training on ImageNet |
