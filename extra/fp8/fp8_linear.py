import torch
import torch.nn.functional as F
from tinygrad import Tensor, dtypes, Context
from tinygrad.dtype import DType
from tinygrad.helpers import getenv
from tinygrad import nn
from tinygrad import Tensor, dtypes, UOp
from tinygrad.uop.ops import KernelInfo, AxisType, Ops
FP8 = getenv("FP8")
Tensor.manual_seed(42)
torch.manual_seed(42)
def k_clamp_back(grads:UOp, kernel:UOp):
  y, x, s = kernel.src
  # 1/0
  return (None, Tensor(grads).uop, None) # through
  return (None, None, None)
  return (None, Tensor.empty_like(Tensor(x)).uop)
def k_clamp_back(grads:UOp, kernel:UOp):
  # print(f"grads xxx: ", Tensor(grads).numpy())
  y, x, s = kernel.src
  # print(f"{y.shape=}, {x.shape=}, {s.shape=}")
  if y.shape == (4194304,):
    y = y.reshape((1024, 4096))
  if x.shape == (4194304,):
    x = x.reshape((1024, 4096))
  if x.shape == (268435456,):
    x = x.reshape((128, 512, 4096))
  if y.shape == (268435456,):
    y = y.reshape((128, 512, 4096))
  ctx = grads
  y = 448.0
  ctx1 = (y>x).where(ctx, (x.eq(y)).where(ctx * 1, 0)) 
  y = -448.0
  ctx2 = (y<x).where(ctx1, (x.eq(y)).where(ctx1 * 1, 0)) 
  return (None, ctx2, None)
  return (None, Tensor(grads).uop, None)
  return (None, None, None)
 
def k_clamp(y:UOp, x:UOp, s: UOp):
  y = y.flatten()
  x = x.flatten()
  # s = s.flatten()
  i = UOp.range(x.size, 0)
  # x1 = (x[i]*s.index(UOp.const(dtypes.index, 0))).maximum(UOp.const(x.dtype.base, -448.0)).minimum(UOp.const(x.dtype.base, 448.0))
  x1 = x[i].maximum(UOp.const(x.dtype.base, -448.0)).minimum(UOp.const(x.dtype.base, 448.0))
  return y[i].store(x1).end(i).sink(arg=KernelInfo(name=f"k_clamp{x.size}"))  

def q_abs_max_kernel(y: UOp, x:UOp):
  B = y.flatten()
  A = x.flatten()
  i = UOp.range(A.shape[0], 0, axis_type=AxisType.REDUCE)
  B = B[0].set(UOp.const(x.dtype.base, 0.0))
  B = B[0].set(B.after(i)[0].maximum((A[i]<0.0).where(A[i]*UOp.const(x.dtype.base, UOp.const(x.dtype.base, -1.0)), A[i])), end=i)
  B = B[0].set(B[0].reciprocal()*448.0)
  return B.sink(arg=KernelInfo(name=f"custom_sumx_{A.shape[0]}"))
def q_abs_max_kernel(y: UOp, x:UOp):
  # c0 = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(1), (), 0)
  # c4 = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(7), (), 1)
  c0 = y
  c4 = x
  c6 = UOp.range(x.size, 0, AxisType.REDUCE)
  c7 = c4.index(c6)
  c13 = (c7<0.0).where(UOp.const(x.dtype.base, -1.0), UOp.const(x.dtype.base, 1.0))
  c14 = (c7!=0.0).where(c13, UOp.const(x.dtype.base, 0.0))
  c15 = c7*c14
  c20 = 448.0*(c15.reduce(c6, arg=Ops.MAX)+1e-08).reciprocal()
  c21 = c0.index(UOp.const(dtypes.index, 0), ptr=True).store(c20)
  ast = c21.sink(arg=KernelInfo(name=f"custom_sumx_{x.shape[0]}"))
  return ast
def q_abs_max_kernel_back(grads:UOp, kernel:UOp):
  y, x = kernel.src
  return (None, None)
  return (None, Tensor.zeros_like(Tensor(x)).uop)
  # return (None, Tensor(grads).uop)
  return (grads, grads)

from tinygrad.helpers import getenv
CUSTOM_CLAMP = getenv("CUSTOM_CLAMP", 0)
CUSTOM_AMAX = getenv("CUSTOM_AMAX", 0)
def quantize_to_fp8(x: Tensor, axis=None, dtype=dtypes.fp8e4m3):
  if CUSTOM_AMAX:
    y = Tensor.empty((), dtype=x.dtype)
    y = Tensor.custom_kernel(y, x, fxn=q_abs_max_kernel, grad_fxn=q_abs_max_kernel_back)[0]
    # scale = 448. / (y + 1e-8)  
    scale = y
  # x_abs_max = x.abs().max()
  else:
    if axis is None:
      x_abs_max = x.abs().max()
    else:
      x_abs_max = x.abs().max(axis=axis, keepdim=True)
    # scale = fp8_max / max_val  
    scale = 448. / (x_abs_max + 1e-8)  
  
  x = x*scale

  if CUSTOM_CLAMP:
    y = Tensor.empty_like(x)
    y = Tensor.custom_kernel(y, x, Tensor(1.0), fxn=k_clamp, grad_fxn=k_clamp_back)[0]
    res= y.cast(dtype)
    return res, scale.float().reciprocal()
  # ---
  # y = (x * scale).clamp(-448.0, 448.0)
  y = x.clamp(-448.0, 448.0)
  res= y.cast(dtype)
  return res, scale.float().reciprocal()
class FP8LinearBert:
  def __init__(self, in_features, out_features, bias=True):
    # (1024, 4096)
    self.weight = Tensor.empty(out_features, in_features, dtype=dtypes.float32)
    # (1024)
    self.bias = Tensor.empty(out_features, dtype=dtypes.float32) if bias else None
  def __call__(self, x:Tensor):
    # return self.kk(x)
    # print(f"{x.shape=}, {self.weight.shape=}")
    # x.shape=(128, 512, 4096), self.weight.shape=(1024, 4096)
    w1, ws = quantize_to_fp8(self.weight)
    x1, s = quantize_to_fp8(x)
    # x1.shape=(192, 512, 1024),w1.T.shape=(1024, 1024)
    # y = x1.dot(w1.T, dtype=dtypes.float) * ws * s
    y = x1.cast(dtypes.float).dot(w1.T.cast(dtypes.float), dtype=dtypes.float) * ws * s
    if self.bias is not None: y = y + self.bias.cast(y.dtype)
    return y.cast(x.dtype)
def main():
  M, N = 4096, 1024
  M, N = 256, 128
  M, N = 16, 8
  if not FP8:
    m = nn.Linear(M, N)
    m.weight.assign(Tensor.rand(N, M))
    m.bias.assign(Tensor.rand(N))
    m.weight.requires_grad = True
    m.bias.requires_grad = True
  else:
    m = FP8LinearBert(M, N)
    m.weight.assign(Tensor.rand(N, M))
    m.bias.assign(Tensor.rand(N))
    m.weight.requires_grad = True
    m.bias.requires_grad = True
  # inp = Tensor.rand((128, 512, M))
  inp = Tensor.rand((4, M))
  inp.requires_grad = True
  out = m(inp)
  print(out.numpy())
  # print(m.weight.numpy())
  # out.sum().backward()
  # print(f"{m.weight.grad.numpy()=}")
  # print(f"{m.weight.grad.mean().numpy()=}")
# [[113.722176  141.76231   447.1139    ... 178.28998   118.2638
#   288.23758  ]
#  [145.38972   214.84967   410.42303   ... 435.85028   431.9217
#    94.99939  ]
#  [429.03537   249.07835   273.05557   ... 426.0181    193.48743
#   185.01964  ]
#  ...
#  [  0.5322039 196.65662   154.92981   ...   3.9781451 406.11868
#    78.86939  ]
#  [281.1676    311.03583   367.92993   ...  37.30223   435.0939
#    52.607132 ]
#  [131.03128   122.33866   199.30325   ...  27.553743  292.16745
#    53.249252 ]]

if __name__ == "__main__":
#   compare_f8_mm_torch()
  main()