
import torch
from typing import Callable
import torch.nn as nn
import numpy as np

from tinygrad.nn import Linear
from tinygrad import Tensor, dtypes
from torchao.float8.config import Float8LinearConfig
np.random.seed(42)
torch.manual_seed(42)
from torchao.prototype.float8nocompile.float8nocompile_linear import (
    Float8LinearNoCompile,
)
B, M, N = 2, 32, 32
B, M, N = 4096, 4096, 4096
np_x = np.random.randn(B, M).astype(np.float32)
np_w = np.random.randn(N, M).astype(np.float32)
np_b = np.random.randn(N).astype(np.float32)
# create model and sample input
m_i = nn.Linear(M, N)
m_i.weight.data = torch.from_numpy(np_w).clone()
m_i.bias.data = torch.from_numpy(np_b).clone()
m = (
    nn.Sequential(
        m_i
    )
    # .half()
    # .cuda()
)
m = m_i
class Model:
  def __init__(self):
    m1 = Linear(M, N)
    m1.weight.assign(Tensor(np_w)).realize()
    m1.bias.assign(Tensor(np_b)).realize()
    self.layers: list[Callable[[Tensor], Tensor]] = [
      m1, Tensor.half,
    ]
  def __call__(self, x:Tensor) -> Tensor: return x.sequential(self.layers)
# m1 = Model()
from tinygrad import Tensor, UOp
from tinygrad.uop.ops import KernelInfo
def backward_gemm(gradient:UOp, kernel:UOp) -> tuple[UOp, UOp]:
  out, a, b = kernel.src
  grad_a = (Tensor(gradient) @ Tensor(b).T).uop
  grad_b = (Tensor(a).T @ Tensor(gradient)).uop
  return (None, grad_a, grad_b)
def k_clamp_back(grads:UOp, kernel:UOp):
  y, x, s = kernel.src
  1/0
  return (None, None, None)

  print("grads xxx: ", Tensor(grads).numpy())
  return (Tensor.ones_like(Tensor(y)).uop, Tensor.ones_like(Tensor(x)).uop, Tensor.ones_like(Tensor(s)).uop)
  return (None, Tensor(grads).uop, Tensor(grads).uop)
  grad_x = (Tensor(grads)+1).uop
  return (None, grad_x, None)
# k_clamp_back =backward_gemm
def k_clamp(y:UOp, x:UOp, s: UOp):
  y = y.flatten()
  x = x.flatten()
  s = s.flatten()
  i = UOp.range(x.size, 0)
  x1 = (x[i]*s.index(UOp.const(dtypes.index, 0))).maximum(UOp.const(x.dtype.base, -448.0)).minimum(UOp.const(x.dtype.base, 448.0))

  return y[i].store(x1).end(i).sink(arg=KernelInfo(name=f"k_clamp{x.size}",opts_to_apply=()))


dtype = dtypes.fp8e4m3

def f0(x: Tensor, scale: Tensor):
  y = (x * scale).clamp(-448.0, 448.0)
  res = y
  res= y.cast(dtype)

  return res
def f1(x: Tensor, scale: Tensor):
  y = Tensor.empty_like(x)
  y = Tensor.custom_kernel(y, x, scale, fxn=k_clamp, grad_fxn=k_clamp_back)[0]
  res = y
  # res= y.cast(dtype)
  return res

x = Tensor(409.0, dtype=dtypes.float)
scale = Tensor(0.5)

print(f0(x, scale).numpy())
print(f1(x, scale).numpy())

exit()

def quantize_to_fp8(x: Tensor, axis=None, dtype=dtypes.fp8e4m3):
  x_abs_max = x.abs().max()
  scale = 448. / (x_abs_max + 1e-8)
  # scale = Tensor([1.0])

  # return x, scale.float().reciprocal()
  y = Tensor.empty_like(x)
  y = Tensor.custom_kernel(y, x, scale, fxn=k_clamp, grad_fxn=k_clamp_back)[0]
  res= y.cast(dtype)
  return res, scale.float().reciprocal()
  y = (x * scale).clamp(-448.0, 448.0)
  res= y.cast(dtype)
  return res, scale.float().reciprocal()

class FP8LinearBert:
  def __init__(self, in_features, out_features, bias=True):
    self.weight = Tensor.empty(out_features, in_features, dtype=dtypes.float32)
    self.bias = Tensor.empty(out_features, dtype=dtypes.float32) if bias else None
  def __call__(self, x:Tensor):
    w1, ws = self.weight, 1.0
    x1, s = x, 1.0
    w1, ws = quantize_to_fp8(self.weight)
    x1, s = quantize_to_fp8(x)
    y = x1.dot(w1.T, dtype=dtypes.float) * ws * s
    if self.bias is not None: y = y + self.bias
    return y.cast(x.dtype)

Linear = FP8LinearBert

m1 = Linear(M, N)
m1.weight.requires_grad = True
m1.weight.assign(Tensor(np_w)).realize()
m1.bias.assign(Tensor(np_b)).realize()
# nn
# m = Float8Linear()
x = torch.randn(B, M, device="cpu", dtype=torch.float)
x1 = Tensor(x.cpu().numpy(), dtype=dtypes.float)
# print("x: ", x.cpu().numpy())
# print("x1: ", x1.numpy())
# optimizer = torch.optim.SGD(m.parameters(), lr=0.1)

# convert specified `torch.nn.Linear` modules to `Float8Linear`
# print("calling convert_to_float8_nocompile_training")
# m=convert_to_float8_nocompile_training(m)
m = Float8LinearNoCompile.from_float(m, config=Float8LinearConfig())
# print("finished convert_to_float8_nocompile_training")

# print("grad: ", m.weight.grad.cpu().numpy())
# print("grad: ", m1.weight.grad.numpy())


with Tensor.train():
  for i in range(1):
      print("-" * 20)
      print(f"step {i}")

      # optimizer.zero_grad()
    #   y = m(x)
      y1 = m1(x1)
      print("after forward")
    #   print("y: ", y.detach().cpu().numpy())
      print("y1: ", y1.numpy())
      # continue

      # 1. 获取 Loss 并打印
    #   loss = y.sum()
    #   loss.backward()
    #   print(f"Loss: {loss.item():.6f}")
    #   print("grad: ", m.weight.grad.cpu().numpy())
    #   print(f"-" * 10)

      # loss1 = y1.sum()
      # loss1.backward()
      y1.sum().backward()
      # print(f"Loss1: {loss1.item():.6f}")
      print("grad1: ", m1.weight.grad.numpy())
      # optimizer.step()