from tinygrad import Tensor, dtypes, nn
from tinygrad.helpers import getenv
from tinygrad import Tensor, dtypes, UOp
from tinygrad.uop.ops import KernelInfo, AxisType, Ops
from tinygrad.helpers import prod, argfix, Context
from tinygrad.helpers import Timing
import numpy as np
import time
from tinygrad import Tensor, Device
from tinygrad.helpers import getenv
# Tensor.manual_seed(42)
if getenv("GPUS", 1) > 1:
  # GPUS= devs = ('AMD', 'AMD:1')
  GPUS = tuple(f'{Device.DEFAULT}:{i}' if i else f'{Device.DEFAULT}' for i in range(getenv("GPUS", 1)))
else:
  GPUS= devs = 'AMD'
class LinearBert(nn.Linear):
  def __init__(self, in_features, out_features, bias=True, std=0.02):
    self.weight = Tensor.randn(out_features, in_features, dtype=dtypes.float16)
    self.bias = Tensor.zeros(out_features, dtype=dtypes.float16) if bias else None
  def __call__(self, x:Tensor):
    return x.cast(dtypes.half).contiguous().linear(self.weight.cast(dtypes.half).transpose(), self.bias.cast(dtypes.half) if self.bias is not None else None).contiguous()

def quantize_to_fp8(x: Tensor, axis=None, dtype=dtypes.fp8e4m3):
  x_abs_max = x.abs().max1(axis=axis, keepdim=True)
  scale = 448. / (x_abs_max + 1e-8)  
  x = x*scale
  y = x.nround()
  print(f"q: {x.uop.axis=}, {y.uop.axis=}")
  res= y.cast(dtype)
  return res, scale.float().reciprocal().contiguous()
def k_clamp_back(grads:UOp, kernel:UOp):
  y, x = kernel.src
  return (None, Tensor(grads, grads.device).uop) # through
 
def k_clamp(y:UOp, x:UOp):
  y = y.flatten()
  x = x.flatten()
  # s = s.flatten()
  i = UOp.range(x.size, 0)
  x1 = x[i].maximum(UOp.const(x.dtype.base, -448.0)).minimum(UOp.const(x.dtype.base, 448.0))
  return y[i].store(x1).end(i).sink(arg=KernelInfo(name=f"custom_clamp_{x.size}"))  

def q_abs_max_kernel(y: UOp, x:UOp):
  A = x.flatten()
  B = y.flatten()
  i = UOp.range(A.size, 0, axis_type=AxisType.REDUCE)
  B = B[0].set(UOp.const(x.dtype.base, 0.0))
  B = B[0].set(B.after(i)[0].maximum((A[i]<0.0).where(A[i]*UOp.const(x.dtype.base, UOp.const(x.dtype.base, -1.0)), A[i])), end=i)
  # B = B[0].set(B[0].reciprocal()*448.0)
  return B.sink(arg=KernelInfo(name=f"custom_sumx_{A.shape}"))

def q_abs_max_kernel(y: UOp, x:UOp):
  c26 = UOp.unique_const(dtypes.uint, 1, device='AMD', unique=14).reshape((1,)).expand((262144,)).pad(((262143, 0),)).reshape((1,524287)).expand((262145,524287)).reshape((137439215615,)).shrink(((0, 137438953472),)).reshape((262144,524288)).shrink(((0, 262144),(0, 262144))).reshape((262144,262144,1)).reshape((262144,262144)).permute((1, 0))
  c30 = UOp.const(dtypes.uint, -1, device='AMD').reshape((1,))
  c34 = UOp.new_buffer('AMD', 1, dtypes.uint, 10)
  c35 = c34.forced_reshape((1,))
  c37 = UOp.const(dtypes.uint, 524288, device='AMD').reshape((1,))
  c39 = c35.assign((c35+c37))
  c43 = c26.r(Ops.ADD, (1,)).reshape((262144,))+c30.expand((262144,))+(c39+c37*c30).expand((262144,))
  c51 = UOp.const(dtypes.ulong, 4294967296, device='AMD').reshape((1,)).expand((262144,))
  c56 = UOp.new_buffer('AMD', 2, dtypes.uint, 11)
  c58 = c56.forced_reshape((2,))
  c60 = UOp(Ops.VECTORIZE, dtypes.index.vec(0), ())
  c72 = ((c43+UOp.const(dtypes.uint, 262144, device='AMD').reshape((1,)).expand((262144,))).cast(dtypes.ulong)*c51|c43.cast(dtypes.ulong)).threefry((c58.shrink(((1, 2),)).reshape(()).reshape((1,)).expand((262144,)).cast(dtypes.ulong)*c51|c58.shrink(((0, 1),)).reshape(()).reshape((1,)).expand((262144,)).cast(dtypes.ulong)))
  c75 = UOp.const(dtypes.ulong, 4294967295, device='AMD').reshape((1,)).expand((262144,))
  c106 = ((((c72&c75).cast(dtypes.uint).pad(((0, 262144),))+(c72//c51&c75).cast(dtypes.uint).pad(((262144, 0),)))//UOp.const(dtypes.uint, 512, device='AMD').reshape((1,)).expand((524288,))|UOp.unique_const(dtypes.float, 1.0, device='AMD', unique=15).reshape((1,)).expand((524288,)).bitcast(dtypes.uint)).bitcast(dtypes.float)+UOp.const(dtypes.float, 1.0, device='AMD').reshape((1,)).expand((524288,))*UOp.const(dtypes.float, -1.0, device='AMD').reshape((1,)).expand((524288,))).reshape((8,512,128)).contiguous()
  c107 = c106.cast(dtypes.half)
  c109 = c107.r(Ops.MAX1, (0, 1, 2)).contiguous()
  ast = c109.sink()
  return ast
def q_abs_max_kernel_(y: UOp, x:UOp):
  # c0 = UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(1), (), 0)
  # c3 = UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(8), (), 1)
  c0 = y
  c3 = x
  c5 = UOp.range(2, 0, AxisType.REDUCE)
  c8 = UOp.range(4, 1, AxisType.REDUCE)
  c10 = c3.index((c5*4+c8))
  c16 = (c10<0.0).where(UOp.const(dtypes.half, -1.0), UOp.const(dtypes.half, 1.0))
  c17 = (c10!=0.0).where(c16, UOp.const(dtypes.half, 0.0))
  c18 = c10*c17
  c19 = c18.reduce(c5, c8, arg=Ops.MAX1)
  c20 = c0.index(UOp.const(dtypes.index, 0), ptr=True).store(c19)
  ast = c20.sink()
  return ast
def q_abs_max_kernel_back(grads:UOp, kernel:UOp):
  y, x = kernel.src
  return (None, None)

from tinygrad.helpers import getenv
CUSTOM_CLAMP = getenv("CUSTOM_CLAMP", 0)
CUSTOM_AMAX = getenv("CUSTOM_AMAX", 0)
FP8 = getenv("FP8", 0)
print("== CUSTOM_CLAMP: ", CUSTOM_CLAMP)
print("== CUSTOM_AMAX: ", CUSTOM_AMAX)
def q_amax(x: Tensor, axis=None):
  return x.max1(axis=axis, keepdim=True)
  # return x.abs().max1(axis=axis, keepdim=True)

def q_amax_custom(x: Tensor, axis=None):
  y = Tensor.empty((), dtype=x.dtype)
  y = Tensor.custom_kernel(y, x, fxn=q_abs_max_kernel, grad_fxn=q_abs_max_kernel_back)[0]
  return y

def q_clamp(x: Tensor):
  return x.nround()
def q_clamp_custom(x: Tensor):
  if x.uop.axis == 0:
    y = Tensor(Tensor.empty((x.shape[0]//len(GPUS), *x.shape[1:]), device=GPUS, dtype=x.dtype).uop.multi(0), device=GPUS)
  else:
    y = Tensor(Tensor.empty((x.shape[0], *x.shape[1:]), device=GPUS, dtype=x.dtype).uop, device=GPUS)
  y = Tensor.custom_kernel(y, x,  fxn=k_clamp, grad_fxn=k_clamp_back)[0]
  return y


def quantize_to_fp8_custom(x: Tensor, axis=None, dtype=dtypes.fp8e4m3):
  x_abs_max = x.abs().max1(axis=axis, keepdim=True)
  scale = 448. / (x_abs_max + 1e-8)  
  x = x*scale
  if x.uop.axis == 0:
    y = Tensor(Tensor.empty((x.shape[0], *x.shape[1:]), device=GPUS, dtype=x.dtype).uop.multi(0), device=GPUS)
  else:
    y = Tensor(Tensor.empty((x.shape[0], *x.shape[1:]), device=GPUS, dtype=x.dtype).uop, device=GPUS)
  y = Tensor.custom_kernel(y, x,  fxn=k_clamp, grad_fxn=k_clamp_back)[0]
  print(f"Q: {x.uop.axis=}, {y.uop.axis=}")
  res= y.cast(dtype).contiguous()
  return res, scale.float().reciprocal().contiguous()

if __name__ == "__main__":
  # Tensor.manual_seed(42)
  IN, OUT = 1024, 4096
  BS = 1024
  # BS = 66
  IN, OUT = 128, 256
  BS= 8
  m0 = LinearBert(IN, OUT)
  m0.bias.assign(Tensor.randn(OUT).cast(dtypes.half).realize())
  m0.weight.requires_grad = True
  m0.weight.to_(GPUS)
  x = Tensor.rand((BS, 512, IN)).cast(dtypes.half)
  if isinstance(GPUS, tuple):
    x = x.shard(GPUS, axis=0)
  x.requires_grad = True 
  print("---- qamax --")
  # x =  Tensor([[1.2, 447, 1.2, -448.0], [1.2, 449, 1.2, -448.2], [1.2, 449, 1.2, -448.2]], dtype=dtypes.half)
  x1 = Tensor([[1.2, 447, 1.2, -448.0], [1.2, 449, 1.2, -448.2], [1.2, 449, 1.2, -448.2]], dtype=dtypes.half)
  x.requires_grad = True
  x1.requires_grad = True
  # x.to_(GPUS)
  print(f"{GPUS=}")
  # if GPUS
  if not isinstance(GPUS, str):
    x.shard_(GPUS, axis=0)
    x1.shard_(GPUS, axis=0)
  
  if 0:
    print(f"{x.uop.axis=}")
    y0 = q_clamp(x)
    y1 = q_clamp_custom(x1)

    # print(y0.shape, y0.numpy())
    # print('--')
    # print(y1.shape, y1.numpy())
    # print("+++")

    y0.sum().backward()
    y1.sum().backward()
    print("** grad **")
    print(x.grad.numpy())
    print("----")
    print(x1.grad.numpy())
  
  y0 = q_amax(x)
  # print(y0.numpy(), y1.numpy())
  print(y0.shape, y0.numpy())
  # print('--')
  # y1 = q_amax_custom(x)
  # print(y1.shape, y1.numpy())
  # print("+++")
  # y0.realize()
  # x1.realize()
  # x1.sum().backward()
  # print(x.grad.numpy())
  if 0:
    print("--- activation ---")
    x1, s = quantize_to_fp8(x)
    # x2, s2 = quantize_to_fp8_custom(x)
    # print(s.numpy(), s2.numpy())
    print(x1.numpy())

    # np.testing.assert_allclose(x1.numpy(), x2.numpy())
    print("--- weight ---")
    m1, s = quantize_to_fp8(m0.weight)
    m2, s2 = quantize_to_fp8_custom(m0.weight)
    print(s.numpy(), s2.numpy())
    np.testing.assert_allclose(m1.numpy(), m2.numpy())

  # m0_res_ = m0(x).contiguous().contiguous_backward()
  # m_res_ = m(x).contiguous().contiguous_backward()
