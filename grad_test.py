from tinygrad import Tensor, nn
import torch
from tinygrad import dtypes
def round_straight_through(x: Tensor) -> Tensor:
    x_det = x.detach()
    rounded_val = x_det.round()
    # Combine to form output: (rounded_val - x_det) has no grad, x has grad.
    return x + (rounded_val - x_det)
from tinygrad import Tensor

x = Tensor([1, 600.0, 449, 2], dtype=dtypes.float32, requires_grad=True)
print(x.numpy(), (x*2).numpy())
x = x.nround()
print(x.numpy(), (x*2).numpy())


exit(0)

class LinearNet:
  def __init__(self):
    self.l1 = nn.Linear(784, 128)
    self.l2 = nn.Linear(128, 10)
  def __call__(self, x:Tensor) -> Tensor:
    return x.flatten(1).dot(self.l1).relu().dot(self.l2)

model = LinearNet()
optim = nn.optim.Adam([model.l1, model.l2], lr=0.001)

x, y = Tensor.rand(4, 1, 28, 28), Tensor([2,4,3,7])  # replace with real mnist dataloader

with Tensor.train():
  for i in range(10):
    optim.zero_grad()
    loss = model(x).sparse_categorical_crossentropy(y).backward()
    optim.step()
    print(i, loss.item())
if __name__ == "__main__":
    # Device.DEFAULT = "CPU"
    t0 = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    t0.clamp(2).sum().backward()
    t = Tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    # t.sum().backward()
    t.clamp(2).sum().backward()
    print(f"{t0.grad.numpy()=}")
    print(f"{t.grad.numpy()=}")
    print("- "*4)
    for i in range(10):
        with Tensor.train():
            x = Tensor([2.9], requires_grad=True)
            # y = x.round()
            y = x.nround().round()
            # x = x + Tensor.randint(x.shape, low=0, high=2).sub(0.5)
            # x.gradient(x, gradient=x+0.5)
            # x = x.max()
            # y = x.cos().sum()
            y.backward(gradient=Tensor([0.0]))
            # y.backward()

        # print(y.grad.tolist())
        print(f"y = {y.numpy()}, x grad = {x.grad.tolist()}")
        # print(y.grad.tolist())

    print("- * " *10)
    for i in range(10):
        with Tensor.train():
            x = Tensor([2.9], requires_grad=True)
            y = x.round()
            y.backward(gradient=Tensor([0.0]))

        print(f"y = {y.numpy()}, x grad = {x.grad.tolist()}")