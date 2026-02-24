# Chapter 27: GANs — Generating Images with Adversarial Networks

A Generative Adversarial Network (GAN) pits two neural networks against each other: a **generator** that creates fake images and a **discriminator** that tries to tell real from fake. This chapter walks through tinygrad's MNIST GAN.

## The Adversarial Game

Imagine a counterfeiter (generator) and a detective (discriminator):

1. The counterfeiter creates fake banknotes
2. The detective examines both real and fake banknotes, trying to tell them apart
3. The counterfeiter improves based on what the detective catches
4. The detective improves as the counterfeits get better
5. Eventually, the counterfeiter produces banknotes indistinguishable from real ones

In math terms:

```
Generator G: random noise z → fake image G(z)
Discriminator D: image x → probability that x is real

D wants to maximize: D(real) should be 1, D(G(z)) should be 0
G wants to minimize: D(G(z)) should be 1 (fool the discriminator)
```

## The Generator

The generator maps random noise to a 28x28 image:

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

The architecture: `128 → 256 → 512 → 1024 → 784 (28×28)`.

**Leaky ReLU** (`max(0.2*x, x)`) is preferred over ReLU in GANs because it doesn't kill gradients for negative values, which helps the generator learn.

**Tanh** output maps to [-1, 1], matching the normalized image range.

## The Discriminator

The discriminator classifies images as real or fake:

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

Mirror architecture: `784 → 1024 → 512 → 256 → 2`. The output is 2 classes: `[fake_score, real_score]`.

**Dropout** (0.3) randomly zeros 30% of activations during training, preventing the discriminator from becoming too strong too fast.

## Training

### Training the Discriminator

Show it real images (label: "real") and fake images (label: "fake"):

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

### Training the Generator

Generate fake images and try to make the discriminator classify them as real:

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

Note: during generator training, the discriminator's weights are frozen (we only call `optimizer_g.step()`). But gradients still flow through the discriminator to the generator.

### The Training Loop

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

**`detach()`** is critical: when training the discriminator, we don't want gradients flowing back to the generator. When training the generator, we *do* want gradients flowing through the discriminator (to learn what fools it).

### Training Tips

- **Learning rate**: 0.0002 is the standard for GANs (from the DCGAN paper)
- **Beta1 = 0.5**: Adam's momentum is reduced from the default 0.9 to prevent oscillation
- **Balance**: if D becomes too strong, G can't learn. The losses should stay roughly balanced

## Running the GAN

```bash
python examples/mnist_gan.py
```

Every 30 epochs, it saves a grid of generated images to `outputs/`. Early epochs produce noise; later epochs produce recognizable digits.

## Diffusion vs GANs

GANs were the dominant generative model before Stable Diffusion. How do they compare?

| | GANs | Diffusion |
|--|------|-----------|
| Training | Adversarial (tricky to balance) | Simple denoising objective |
| Quality | Sharp but can miss modes | High quality, covers all modes |
| Speed | Fast inference (one forward pass) | Slow inference (many denoising steps) |
| Stability | Mode collapse, training instability | Stable training |

**Mode collapse** is when the GAN only learns to generate a few digit types instead of all 10. This is a fundamental limitation of adversarial training.

## Exercises

1. **Run it**: Train the GAN and watch the generated images improve over 300 epochs.

2. **Mode collapse**: If you reduce the generator's capacity (fewer hidden units), does it still learn all 10 digits? This is mode collapse in action.

3. **Detach experiment**: What happens if you remove `.detach()` from the fake data during discriminator training? Why?

4. **Loss dynamics**: Plot the generator and discriminator losses during training. What does a healthy training curve look like?

5. **Better architecture**: Replace the fully connected GAN with convolutional layers (DCGAN). How does image quality change?

## Source Code Map

| File | What to read |
|------|-------------|
| `examples/mnist_gan.py` | Full GAN training on MNIST |
