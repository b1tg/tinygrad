
import torch
from tinygrad.dtype import fp8_to_float, float_to_fp8
from tinygrad import dtypes, Tensor
from tinygrad import Tensor, dtypes, UOp
from tinygrad.uop.ops import KernelInfo, AxisType, Ops
from tinygrad.helpers import getenv
expected=[29, 70, 30, 72, 168, 184, 65, 62, 224, 67, 213, 49, 109, 71, 242, 59]
expected=[189, 255, 206, 255, 198, 255, 27, 0, 208, 255, 181, 255, 45, 0, 22, 0]
# expected=[29, 70, 30, 72, 168, 184, 65, 62, 224, 67, 213, 49, 109, 71, 242, 59]
# fp8: expected=[0.1015625, 3.5, 0.109375, 4.0, -0.25, -1.0, 2.25, 1.75, -32.0, 2.75, -13.0, 0.5625, 104.0, 3.75, -160.0, 1.375]
# E          ACTUAL: array([ 5.078125e-02,  1.750000e+00,  5.468750e-02,  2.000000e+00,
# E                -1.250000e-01, -5.000000e-01,  1.125000e+00,  8.750000e-01,
# E                -1.600000e+01,  1.375000e+00, -6.500000e+00,  2.812500e-01,...
# E          DESIRED: array([ 1.015625e-01,  3.500000e+00,  1.093750e-01,  4.000000e+00,
# E                -2.500000e-01, -1.000000e+00,  2.250000e+00,  1.750000e+00,
# E                -3.200000e+01,  2.750000e+00, -1.300000e+01,  5.625000e-01,
# E                 1.040000e+02,  3.750000e+00, -1.600000e+02,  1.375000e+00])
fp8s = []
for f in expected:
    f8 = fp8_to_float(f, dtypes.fp8e4m3)
    # f8_amd = Tensor(f8, dtype=dtypes.uint).bitcast(dtypes.fp8e4m3)
    fp8s.append(f8)
# print(fp8s)
# [0.1015625, 3.5, 0.109375, 4.0, -0.25, -1.0, 2.25, 1.75, -32.0, 2.75, -13.0, 0.5625, 104.0, 3.75, -160.0, 1.375]

fp8s_amd = Tensor(expected, dtype=dtypes.uchar).bitcast(dtypes.fp8e4m3) 
# print(fp8s_amd.numpy())


x = 249.0
# print(torch.tensor(x, dtype=torch.float8_e4m3fnuz).view(torch.uint8).item()) # 128
# print(float_to_fp8(x, dtypes.fp8e4m3fnuz)) # 127
# print(float_to_fp8(x, dtypes.fp8e4m3))

def k_clamp(y:UOp, x:UOp, s: UOp):
  y = y.flatten()
  x = x.flatten()
  # s = s.flatten()
  i = UOp.range(x.size, 0)
  x1 = x[i].maximum(UOp.const(x.dtype.base, -240.0)).minimum(UOp.const(x.dtype.base, 240.0))
  return y[i].store(x1).end(i).sink(arg=KernelInfo(name=f"k_clamp{x.size}"))  

# def clamp(x: Tensor):
#   if not CUSTOM_CLAMP: return x.clamp(-448.0, 448.0)
#   y = Tensor.empty_like(x, dtype=dtypes.fp8e4m3)
#   y = Tensor.custom_kernel(y, x, Tensor(1.0), fxn=k_clamp, grad_fxn=k_clamp_back)[0]
#   res = y
#   # res= y.cast(dtypes.fp8e4m3)
#   return res

# def k_clamp(y:UOp, x:UOp, s: UOp):
#     c0 = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(8), (), 0)
#     c5 = UOp.range(x.size, 0, AxisType.LOOP)
#     c7 = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(8), (), 1)
#     c18 = UOp(Ops.MAX, dtypes.float, ((UOp(Ops.MAX, dtypes.float, (c7.reshape((2,4)).reshape((8,)).index(c5), UOp.const(dtypes.float, -240.0)))*-1.0), (240.0*UOp.const(dtypes.float, -1.0))))*-1.0
#     c20 = c0.reshape((2,4)).reshape((8,)).index(c5).store(c18).end(c5)
#     ast = c20.sink(arg=KernelInfo(name='k_clamp8xx', axis_types=(), dont_use_locals=False, applied_opts=(), opts_to_apply=None))   
#     return ast

CUSTOM_CLAMP = getenv("CUSTOM_CLAMP", 0)
print(f"{CUSTOM_CLAMP=}")
def clamp(x: Tensor):
  return x.nround()
  if not CUSTOM_CLAMP: return x.clamp(-240.0, 240.0)
  y = Tensor.empty_like(x)
  y = Tensor.custom_kernel(y, x, Tensor(1.0), fxn=k_clamp)[0]
  res = y
  # res= y.cast(dtypes.fp8e4m3)
  return res
x = Tensor([[1.0, 241, 30, -240.0], [0.5625, 104.0, 3.75, -1670.0]])
x.requires_grad = True
# y = x.nround()
y = x.max1()
# y = x.sin()
print(y.numpy())
# print(y.sum().numpy())
y.sum().backward()
print(x.grad.numpy())
# [   1.      240.       30.     -240.        0.5625  104.        3.75
#  -160.    ]