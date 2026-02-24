# 第27章：GAN — 用对抗网络生成图像

生成对抗网络（GAN）让两个神经网络相互对抗：一个 **Generator**（生成器）负责创建假图像，一个 **Discriminator**（判别器）负责区分真假。本章将详细介绍 tinygrad 的 MNIST GAN 实现。

## 对抗博弈

想象一个伪造者（Generator）和一个侦探（Discriminator）：

1. 伪造者制造假钞
2. 侦探检查真钞和假钞，试图区分它们
3. 伪造者根据侦探发现的问题进行改进
4. 随着伪造技术的提高，侦探也不断进步
5. 最终，伪造者制造出与真钞无法区分的假钞

用数学表示：

```
Generator G: random noise z → fake image G(z)
Discriminator D: image x → probability that x is real

D wants to maximize: D(real) should be 1, D(G(z)) should be 0
G wants to minimize: D(G(z)) should be 1 (fool the discriminator)
```

## Generator

Generator 将随机噪声映射为 28x28 的图像：

```python
class LinearGen:
    def __init__(self):
        self.l1 = Tensor.scaled_uniform(128, 256)
        self.l2 = Tensor.scaled_uniform(256, 512)
        self.l3 = Tensor.scaled_uniform(512, 1024)
        self.l4 = Tensor.scaled_uniform(1024, 784)

    def forward(self, x):
        x = x.dot(self.l1).leaky_relu(0.2)
        x = x.dot(self.l2).leaky_relu(0.2)
        x = x.dot(self.l3).leaky_relu(0.2)
        x = x.dot(self.l4).tanh()      # output in [-1, 1]
        return x
```

网络结构：`128 → 256 → 512 → 1024 → 784 (28×28)`。

**Leaky ReLU**（`max(0.2*x, x)`）在 GAN 中比 ReLU 更受青睐，因为它不会在负值时杀死梯度，有助于 Generator 的学习。

**Tanh** 输出映射到 [-1, 1]，与归一化后的图像范围一致。

## Discriminator

Discriminator 将图像分类为真或假：

```python
class LinearDisc:
    def __init__(self):
        self.l1 = Tensor.scaled_uniform(784, 1024)
        self.l2 = Tensor.scaled_uniform(1024, 512)
        self.l3 = Tensor.scaled_uniform(512, 256)
        self.l4 = Tensor.scaled_uniform(256, 2)

    def forward(self, x):
        x = x.dot(self.l1).add(1).leaky_relu(0.2).dropout(0.3)
        x = x.dot(self.l2).leaky_relu(0.2).dropout(0.3)
        x = x.dot(self.l3).leaky_relu(0.2).dropout(0.3)
        x = x.dot(self.l4).log_softmax()
        return x
```

镜像结构：`784 → 1024 → 512 → 256 → 2`。输出为 2 个类别：`[fake_score, real_score]`。

**Dropout**（0.3）在训练时随机将 30% 的激活值置零，防止 Discriminator 过快变得太强。

## 训练

### 训练 Discriminator

向其展示真实图像（标签："真"）和假图像（标签："假"）：

```python
def train_discriminator(optimizer, data_real, data_fake):
    real_labels = make_labels(batch_size, 1)  # class 1 = real
    fake_labels = make_labels(batch_size, 0)  # class 0 = fake

    optimizer.zero_grad()
    loss_real = (discriminator.forward(data_real) * real_labels).mean()
    loss_fake = (discriminator.forward(data_fake) * fake_labels).mean()
    loss_real.backward()
    loss_fake.backward()
    optimizer.step()
    return (loss_real + loss_fake).numpy()
```

### 训练 Generator

生成假图像，并尝试让 Discriminator 将其分类为真：

```python
def train_generator(optimizer, data_fake):
    real_labels = make_labels(batch_size, 1)  # we want D(G(z)) = real!

    optimizer.zero_grad()
    output = discriminator.forward(data_fake)
    loss = (output * real_labels).mean()
    loss.backward()
    optimizer.step()
    return loss.numpy()
```

注意：在训练 Generator 时，Discriminator 的权重是冻结的（我们只调用 `optimizer_g.step()`）。但梯度仍然会通过 Discriminator 流向 Generator。

### 训练循环

```python
generator = LinearGen()
discriminator = LinearDisc()
optim_g = optim.Adam(get_parameters(generator), lr=0.0002, b1=0.5)
optim_d = optim.Adam(get_parameters(discriminator), lr=0.0002, b1=0.5)

for epoch in range(300):
    for _ in range(n_steps):
        # Train discriminator: show real and fake images
        data_real = make_batch(images_real)
        noise = Tensor.randn(batch_size, 128)
        data_fake = generator.forward(noise).detach()  # detach! don't train G here
        train_discriminator(optim_d, data_real, data_fake)

        # Train generator: try to fool discriminator
        noise = Tensor.randn(batch_size, 128)
        data_fake = generator.forward(noise)  # no detach! train G through D
        train_generator(optim_g, data_fake)
```

**`detach()`** 至关重要：训练 Discriminator 时，我们不希望梯度回传到 Generator。训练 Generator 时，我们*需要*梯度通过 Discriminator 流动（以学习如何欺骗它）。

### 训练技巧

- **学习率**：0.0002 是 GAN 的标准值（来自 DCGAN 论文）
- **Beta1 = 0.5**：Adam 的动量从默认的 0.9 降低，以防止振荡
- **平衡**：如果 D 变得太强，G 就无法学习。两者的损失应大致保持平衡

## 运行 GAN

```bash
python examples/mnist_gan.py
```

每 30 个 epoch，程序会将生成的图像网格保存到 `outputs/`。早期 epoch 生成的是噪声；后期 epoch 会生成可识别的数字。

## Diffusion 与 GAN 的对比

GAN 在 Stable Diffusion 出现之前是主流的生成模型。它们的对比如下：

| | GAN | Diffusion |
|--|------|-----------|
| 训练 | 对抗式（难以平衡） | 简单的去噪目标 |
| 质量 | 清晰但可能遗漏模式 | 高质量，覆盖所有模式 |
| 速度 | 推理快（一次前向传播） | 推理慢（多步去噪） |
| 稳定性 | 模式崩塌，训练不稳定 | 训练稳定 |

**模式崩塌（Mode collapse）** 是指 GAN 只学会生成少数几种数字，而非全部 10 种。这是对抗训练的一个根本性局限。

## 练习

1. **运行它**：训练 GAN，观察生成图像在 300 个 epoch 中的改善过程。

2. **模式崩塌**：如果减少 Generator 的容量（更少的隐藏单元），它还能学会生成全部 10 个数字吗？这就是模式崩塌的实际表现。

3. **detach 实验**：如果在训练 Discriminator 时移除假数据上的 `.detach()`，会发生什么？为什么？

4. **损失动态**：绘制训练过程中 Generator 和 Discriminator 的损失曲线。健康的训练曲线是什么样的？

5. **更好的架构**：将全连接 GAN 替换为卷积层（DCGAN）。图像质量会有什么变化？

## 源代码索引

| 文件 | 阅读内容 |
|------|-------------|
| `examples/mnist_gan.py` | 完整的 MNIST GAN 训练代码 |
