import math
from typing import Union

from tinygrad import Tensor, nn, dtypes
from tinygrad import Tensor, dtypes, UOp
from tinygrad.uop.ops import KernelInfo, AxisType, Ops
from tinygrad.helpers import prod, argfix, Context
from tinygrad.nn.state import get_parameters
from extra.models.unet import UNetModel

# rejection sampling truncated randn
def rand_truncn(*shape, dtype=None, truncstds=2, **kwargs) -> Tensor:
  CNT=8
  x = Tensor.randn(*(*shape, CNT), dtype=dtype, **kwargs)
  ctr = Tensor.arange(CNT).reshape((1,) * len(x.shape[:-1]) + (CNT,)).expand(x.shape)
  take = (x.abs() <= truncstds).where(ctr, CNT).min(axis=-1, keepdim=True)  # set to 0 if no good samples
  return (ctr == take).where(x, 0).sum(axis=-1)

# https://github.com/keras-team/keras/blob/v2.15.0/keras/initializers/initializers.py#L1026-L1065
def he_normal(*shape, a: float = 0.00, **kwargs) -> Tensor:
  std = math.sqrt(2.0 / (1 + a ** 2)) / math.sqrt(prod(argfix(*shape)[1:])) / 0.87962566103423978
  return std * rand_truncn(*shape, **kwargs)

# Stable Diffusion v2 training uses default torch gelu, which doesn't use tanh approximation
def gelu_erf(x:Tensor) -> Tensor:
  return 0.5 * x * (1.0 + (x / 1.4142135623730951).erf())

class Conv2dHeNormal(nn.Conv2d):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
    self.in_channels, self.out_channels = in_channels, out_channels  # for testing
    self.weight = he_normal(out_channels, in_channels//groups, *self.kernel_size, a=0.0, dtype=dtypes.float32)
    if bias: self.bias = self.bias.cast(dtypes.float32)
  def __call__(self, x: Tensor):
    return x.conv2d(self.weight.cast(dtypes.default_float), self.bias.cast(dtypes.default_float) if self.bias is not None else None,
                    padding=self.padding, stride=self.stride, dilation=self.dilation, groups=self.groups)

class Linear(nn.Linear):
  def __init__(self, in_features, out_features, bias=True):
    super().__init__(in_features, out_features, bias=bias)
    self.weight = Tensor.normal((out_features, in_features), mean=0.0, std=0.01, dtype=dtypes.float32)
    if bias: self.bias = Tensor.zeros(out_features, dtype=dtypes.float32)
  def __call__(self, x:Tensor):
    return x.linear(self.weight.cast(dtypes.default_float).transpose(), self.bias.cast(dtypes.default_float) if self.bias is not None else None)

class LinearBert(nn.Linear):
  def __init__(self, in_features, out_features, bias=True, std=0.02):
    self.weight = std * rand_truncn(out_features, in_features, dtype=dtypes.float32)
    self.bias = Tensor.zeros(out_features, dtype=dtypes.float32) if bias else None

  def __call__(self, x:Tensor):
    return x.cast(dtypes.default_float).linear(self.weight.cast(dtypes.default_float).transpose(), self.bias.cast(dtypes.default_float) if self.bias is not None else None)


def k_clamp_back(grads:UOp, kernel:UOp):
  y, x = kernel.src
  return (None, None)
  return (None, Tensor(grads).uop) # through
  return (None, Tensor.empty_like(Tensor(x)).uop)
def k_clamp_back1(grads:UOp, kernel:UOp):
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
 
def k_clamp(y:UOp, x:UOp):
  y = y.flatten()
  x = x.flatten()
  # s = s.flatten()
  i = UOp.range(x.size, 0)
  x1 = x[i].maximum(UOp.const(x.dtype.base, -448.0)).minimum(UOp.const(x.dtype.base, 448.0))
  return y[i].store(x1).end(i).sink(arg=KernelInfo(name=f"custom_clamp_{x.size}"))  

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
from tinygrad import Tensor, Device
from tinygrad.helpers import getenv
if getenv("GPUS", 1) > 1:
  # GPUS= devs = ('AMD', 'AMD:1')
  GPUS = tuple(f'{Device.DEFAULT}:{i}' if i else f'{Device.DEFAULT}' for i in range(getenv("GPUS", 1)))
else:
  GPUS= devs = 'AMD'

from tinygrad.helpers import getenv
CUSTOM_CLAMP = getenv("CUSTOM_CLAMP", 0)
CUSTOM_AMAX = getenv("CUSTOM_AMAX", 0)
PO2 = getenv("PO2", 0)
FP8 = getenv("FP8", 0)
print("== CUSTOM_CLAMP: ", CUSTOM_CLAMP)
print("== CUSTOM_AMAX: ", CUSTOM_AMAX)
def quantize_to_fp8(x: Tensor, axis=None, dtype=dtypes.fp8e4m3):
  if CUSTOM_AMAX:
    # y = Tensor.empty((), dtype=x.dtype)
    # y = Tensor.custom_kernel(y, x, fxn=q_abs_max_kernel, grad_fxn=q_abs_max_kernel_back)[0]
    # scale = y
    x_abs_max = x.abs().max1(axis=axis, keepdim=True)
    scale = 448. / (x_abs_max + 1e-8)  
    # assert scale.dtype == dtypes.float16
    # print(f"{x.dtype=}, {x.shape=}, {scale.dtype=}, {scale.shape=}")
  # x_abs_max = x.abs().max()
  else:
    if axis is None:
      x_abs_max = x.abs().max()
    else:
      x_abs_max = x.abs().max(axis=axis, keepdim=True)
    if PO2:
      target_scale = 448.0 / (x_abs_max + 1e-8)
      # log_scale = target_scale.log2().floor()
      scale = target_scale.log2().floor().exp2()
      # scale = 2.0 ** log_scale
      # scale = log
    else:
      scale = 448. / (x_abs_max + 1e-8)  
  
  x = x*scale

  if CUSTOM_CLAMP:
    # y = Tensor.empty_like(x)
    # print(f"x clamp: {x.shape}, {x.device}, {x.uop.axis}")
    old_axis = x.uop.axis
    # y = y.shard(GPUS, axis=x.uop.axis)
    if 0 and old_axis == None:
      # y = y.to(GPUS)
      y = Tensor(Tensor.empty((x.shape[0], *x.shape[1:]), device=GPUS, dtype=x.dtype).uop, device=GPUS)
      y = Tensor.custom_kernel(y, x,  fxn=k_clamp, grad_fxn=k_clamp_back)[0]
      y = y.shard_(GPUS, axis=None)
      pass
      # return x.cast(dtype), 1.0
    elif 0:
      y = Tensor(Tensor.empty((x.shape[0]//2, *x.shape[1:]), device=GPUS, dtype=x.dtype).uop.multi(0), device=GPUS)
      y = Tensor.custom_kernel(y, x,  fxn=k_clamp, grad_fxn=k_clamp_back)[0]
      # return x.cast(dtype), 1.0
      y = y.shard(GPUS, axis=old_axis)
      # y = Tensor(y.shard(GPUS, axis=old_axis).uop.multi(0), device=GPUS)
      # print(f"> y clamp: {y.shape}, {y.device}, {y.uop.axis}")

    y = x.nround()
    # print(f"y clamp: {y.shape}, {y.device}, {y.uop.axis}")
    res= y.cast(dtype).contiguous()
    # res = y
    return res, scale.float().reciprocal().contiguous()
  # ---
  # y = (x * scale).clamp(-448.0, 448.0)
  y = x.clamp(-448.0, 448.0)
  res= y.cast(dtype).contiguous().contiguous_backward()
  return res, scale.float().reciprocal().contiguous().contiguous_backward()
  # ---


def custom_linear(C:UOp, A:UOp, B:UOp, D:UOp) -> UOp:
  # c0 = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(67108864), (), 0)
  c0 = C
  c2 = UOp.range(512, 2, AxisType.LOOP)
  c4 = c2*4096
  c5 = UOp.range(4096, 3, AxisType.LOOP)
  c8 = UOp.range(128, 1, AxisType.LOOP)
  c10 = c8*2097152
  # c13 = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(67108864), (), 1)
  c13 = A
  c14 = UOp.range(4096, 0, AxisType.REDUCE)
  # c18 = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(1048576), (), 2)
  c18 = B
  c22 = c13.index((c4+c14+c10))*c18.index((c5*4096+c14))
  # c24 = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(1024), (), 3)
  c24 = D
  c26 = c22.reduce(c14, arg=Ops.ADD)+c24.index(c5)
  c28 = c0.index((c4+c5+c10), ptr=True).store(c26).end(c8, c2, c5)
  ast = c28.sink()  
  return ast

def custom_linear_backward(gradient:UOp, kernel:UOp) -> tuple[UOp, UOp]:
  return None, None, None, None
  out, a, b, d = kernel.src
  # 1/0
  # return None, Tensor.ones_like(Tensor(a)).uop, Tensor.ones_like(Tensor(b)).uop, Tensor.ones_like(Tensor(d)).uop
  # Tensor(gradient).shape=(128, 512, 1024), Tensor(a).shape=(128, 512, 1024)
  # print(f"{Tensor(gradient).shape=}, {Tensor(a).shape=}, {Tensor(b).shape=}")
  dy =gradient 
  dy2 = Tensor(dy).reshape(-1, dy.shape[-1])
  x2  = Tensor(a).reshape(-1, a.shape[-1])
  grad_b = Tensor.empty_like(Tensor.empty((1024, 4096))).custom_kernel(dy2.T, x2, fxn=custom_gemm)[0].uop
  return (Tensor(out).uop, Tensor(a).uop, grad_b, Tensor(d).uop)
# def custom_x():

# def custom_y():
#   pass
def clamp(x: Tensor):
  if not CUSTOM_CLAMP: return x.clamp(-448.0, 448.0)
  y = Tensor.empty_like(x, dtype=dtypes.fp8e4m3)
  y = Tensor.custom_kernel(y, x, Tensor(1.0), fxn=k_clamp, grad_fxn=k_clamp_back)[0]
  res = y
  # res= y.cast(dtypes.fp8e4m3)
  return res
def fp8_scale(x: Tensor):
  if CUSTOM_AMAX:
    y = Tensor.empty((), dtype=x.dtype)
    scale = Tensor.custom_kernel(y, x, fxn=q_abs_max_kernel, grad_fxn=q_abs_max_kernel_back)[0]
  else:
    x_abs_max = x.abs().max()
    scale = 448. / x_abs_max
  return scale.float()
def quantize_to_fp81(x: Tensor, axis=None, dtype=dtypes.fp8e4m3):
  s = fp8_scale(x)
  return clamp((x*s)).cast(dtype), s.float().reciprocal()
def custom_linear(C:UOp, A:UOp, B:UOp) -> UOp:
  # return UOp.sink().simplify()
  # A = A
  # B = B
  # C = C
  # c0 = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(34603008), (), 0)
  # 1/0
  assert A.shape[1] == 512, f"{A.shape=}"
  c0 = C
  OUT = B.shape[0]
  IN = B.shape[-1]
  c2 = UOp.range(512, 2, AxisType.LOOP)
  c5 = UOp.range(OUT, 3, AxisType.LOOP)
  c8 = UOp.range(C.size//512//OUT, 1, AxisType.LOOP)
  # c13 = UOp(Ops.DEFINE_GLOBAL, dtypes.fp8e4m3.ptr(138412032), (), 1)
  c13 = A
  c16 = UOp.range(IN, 0, AxisType.REDUCE)
  # c22 = UOp(Ops.DEFINE_GLOBAL, dtypes.fp8e4m3.ptr(4194304), (), 2)
  c22 = B
  c27 = (c13.index((c2*IN+c16+c8*IN*512))*c22.index((c5*IN+c16))).cast(dtypes.float)
  c28 = c27.reduce(c16, arg=Ops.ADD)
  c30 = c0.index((c2*OUT+c5+c8*OUT*512), ptr=True).store(c28).end(c8, c2, c5)
  ast = c30.sink(arg=KernelInfo(name=f"custom dot {A.shape}x{B.shape}"))
  return ast
def custom_linear_backward(gradient:UOp, kernel:UOp) -> tuple[UOp, UOp]:
  # 1/0
  # return None, None, None
  out, a, b = kernel.src
  out =out 
  assert a.shape[1] == 512, f"{a.shape=}"
  a2 = Tensor(a).reshape(a.shape[0]*512, a.shape[-1])
  g2 = Tensor(gradient).reshape(gradient.shape[0]*gradient.shape[1], gradient.shape[-1])
  g2, s = quantize_to_fp8(g2) 
  grad_b = (g2.T.dot(a2,dtype=dtypes.float))*s
  grad_b = grad_b.cast(dtypes.float)
  grad_a = (g2.dot(Tensor(b), dtype=dtypes.float)).reshape(a.shape)*s
  return (None, grad_a.uop, grad_b.uop)
def custom_linear_backward_multi(gradient:UOp, kernel:UOp) -> tuple[UOp, UOp]:
  # 1/0
  # return None, None, None
  out, a, b = kernel.src
  out =out 
  # print(f"{out.device}, {a.device}, {b.device}")
  assert a.shape[1] == 512, f"{a.shape=}"
  a2 = Tensor(a, device=GPUS).reshape(a.shape[0]*512, a.shape[-1])
  g2 = Tensor(gradient, device=GPUS).reshape(gradient.shape[0]*gradient.shape[1], gradient.shape[-1])
  # print(f"{a2.device=}, {g2.device=}")
  g2, s = quantize_to_fp8(g2) 
  # print(f"Q: {g2.device=}, {s.device=}")
  grad_b = (g2.T.dot(a2,dtype=dtypes.float))*s
  grad_b = grad_b.cast(dtypes.float)
  grad_a = (g2.dot(Tensor(b, device=GPUS), dtype=dtypes.float)).reshape(a.shape)*s
  # print(f"Q: {grad_a.device=}, {grad_b.device=}")
  return (None, grad_a.uop, grad_b.uop)
def custom_gemm(C:UOp, A:UOp, B:UOp) -> UOp:
  assert A.shape[1] == B.shape[0]
  i, j, k = UOp.range(C.shape[0], 0), UOp.range(C.shape[1], 1), UOp.range(A.shape[1], 2, axis_type=AxisType.REDUCE)
  C = C[i, j].set(0.0)
  C = C[i, j].set(C.after(k)[i, j] + A[i, k] * B[k, j], end=k)
  prog = C.end(i, j)
  return prog.sink(arg=KernelInfo(name=f"custom_gemm_{C.shape[0]}_{C.shape[1]}_{A.shape[1]}", opts_to_apply=()))

class FP8LinearBertBasic:
  def __init__(self, in_features, out_features, bias=True):
    # (1024, 4096)
    self.weight = Tensor.empty(out_features, in_features, dtype=dtypes.float32)
    # (1024)
    self.bias = Tensor.empty(out_features, dtype=dtypes.float32) if bias else None
  # def kk(self, x: Tensor):
  #   y = Tensor.empty((128, 512, 1024))
  #   y.requires_grad = True
  #   res = Tensor.custom_kernel(y, x, self.weight, self.bias, fxn=custom_linear, grad_fxn=custom_linear_backward)[0]
  #   return y
    
  def __call__(self, x:Tensor):
    # return self.kk(x)
    # FP8LinearBertBasic x.shape=(66, 512, 1024), self.weight.shape=(1024, 1024) #QKV
    # FP8LinearBertBasic x.shape=(66, 512, 4096), self.weight.shape=(1024, 4096) # BertOUTPUT
    # FP8LinearBertBasic x.shape=(66, 512, 4096), self.weight.shape=(1024, 4096)
    print(f"FP8LinearBertBasic {x.shape=}, {self.weight.shape=}")
    # x.shape=(128, 512, 4096), self.weight.shape=(1024, 4096)
    # x.shape=(1024, 512, 4096), self.weight.shape=(1024, 4096) GPUS=8, BS=1024

    # FP8LinearBertBasic x.shape=(66, 512, 4096), self.weight.shape=(1024, 4096)
    w1, ws = quantize_to_fp8(self.weight)
    x1, s = quantize_to_fp8(x)
    # w1 = self.weight.cast(dtypes.fp8e4m3)
    # x1 = x.cast(dtypes.fp8e4m3)
    # ws = s = Tensor(1.0)
    assert x1.dtype in dtypes.fp8s
    assert w1.dtype in dtypes.fp8s
    # print(f"{x1.device=}, {x1.uop.axis=} {w1.device=} {w1.uop.axis=}")
    # x1.device=('AMD', 'AMD:1'), x1.uop.axis=0 w1.device=('AMD', 'AMD:1') w1.uop.axis=None
    # x1.shape=(192, 512, 1024),w1.T.shape=(1024, 1024)
    # x1.shape=(24, 512, 1024), w1.shape=(1024, 1024)
    # x = Tensor(Tensor.empty(8, 16, device=devs).uop.multi(0), device=devs)
    devs = GPUS
    if isinstance(GPUS, tuple) and len(GPUS) > 1:
      y = Tensor(Tensor.empty((x.shape[0]//len(GPUS), x.shape[1], self.weight.shape[0]), dtype=dtypes.float, device=devs).uop.multi(0), device=devs) # axis=0
      # y = Tensor(Tensor.empty((x.shape[0], x.shape[1], self.weight.shape[0]), dtype=dtypes.float, device=devs).uop, device=devs) # axis = None
      # print(f"custom {y.shape=}")
      # print(x1.device)
      # print(w1.device)
      # print(f"{y.device=}, {y.shape=}")
      # y.requires_grad = True
      print(y.device, x1.device, w1.device, devs)
      assert y.device == devs
      assert x1.device == devs
      assert w1.device == devs
      y=Tensor.custom_kernel(y, x1, w1, fxn=custom_linear,grad_fxn=custom_linear_backward_multi)[0]
    else:
      y = Tensor.empty((x.shape[0], x.shape[1], self.weight.shape[0]), dtype=dtypes.float)
      print(f"{x1.shape=}, {w1.shape=}, {x1.uop.axis=}, {w1.uop.axis=}")
      y=Tensor.custom_kernel(y, x1, w1, fxn=custom_linear,grad_fxn=custom_linear_backward)[0]

    # oold
    old_shape = y.shape
    # y = y.shard_(devices=devs)
    # y = Tensor(y.shard(GPUS, axis=0).uop.multi(0), device=GPUS)
    # y = y.flatten().reshape(old_shape)
    # y = x1.dot(w1.T, dtype=dtypes.float).contiguous().contiguous_backward()
    print(f"{y.device=}, {y.shape=}, {y.uop.axis}")
    # print(f"{y1.device=}, {y1.shape=}, {y1.uop.axis}")
    # y.device=('AMD', 'AMD:1'), y.shape=(66, 512, 1024), 0
    # assert y.device==('AMD', 'AMD:1')
    # assert y.shape==(128, 512, 1024)
    y = (ws * s).contiguous() * y.contiguous()
    # y = y * ws * s
    if self.bias is not None: y = y + self.bias.cast(y.dtype)
    print(f"{y.device=}, {self.bias.device=}")
    return y.cast(x.dtype)

custom_backward_gemm = False
def backward_gemm(gradient:UOp, kernel:UOp) -> tuple[UOp, UOp]:
  out, a, b = kernel.src
  grad_a = (Tensor(gradient) @ Tensor(b).T).uop
  grad_b = (Tensor(a).T @ Tensor(gradient)).uop
  return (None, grad_a, grad_b)
def backward_gemm_custom(gradient:UOp, kernel:UOp) -> tuple[UOp, UOp]:
  out, a, b = kernel.src
  grad_a = Tensor.empty_like(Tensor(a)).custom_kernel(Tensor(gradient), Tensor(b).T, fxn=custom_gemm)[0].uop
  grad_b = Tensor.empty_like(Tensor(b)).custom_kernel(Tensor(a).T, Tensor(gradient), fxn=custom_gemm)[0].uop
  return (None, grad_a, grad_b)
def custom_gemm(C:UOp, A:UOp, B:UOp) -> UOp:
  assert A.shape[1] == B.shape[0]
  i, j, k = UOp.range(C.shape[0], 0), UOp.range(C.shape[1], 1), UOp.range(A.shape[1], 2, axis_type=AxisType.REDUCE)
  C = C[i, j].set(0.0)
  C = C[i, j].set(C.after(k)[i, j] + A[i, k] * B[k, j], end=k)
  prog = C.end(i, j)
  return prog.sink(arg=KernelInfo(name=f"custom_gemm_{C.shape[0]}_{C.shape[1]}_{A.shape[1]}", opts_to_apply=()))

# class FP8LinearBert:
#   def __init__(self, in_features, out_features, bias=True):
#     self.weight = Tensor.empty(out_features, in_features, dtype=dtypes.float32)
#     self.bias = Tensor.empty(out_features, dtype=dtypes.float32) if bias else None
#   def __call__(self, x:Tensor):
#     # w1, ws = quantize_to_fp8(self.weight)
#     # x1, s = quantize_to_fp8(x)
#     # x1.shape=(192, 512, 1024),w1.T.shape=(1024, 1024)
#     # x1.shape=(24, 512, 1024), w1.shape=(1024, 1024)
#     # y = x1.dot(w1.T, dtype=dtypes.float) * ws * s
#     c = Tensor.empty(N, N)
#     tst = Tensor.custom_kernel(c, a, b, fxn=custom_gemm, grad_fxn=backward_gemm_custom if custom_backward_gemm else backward_gemm)[0]
#     if self.bias is not None: y = y + self.bias.cast(y.dtype)
#     return y.cast(x.dtype)

class FP8LinearBertRow:
  def __init__(self, in_features, out_features, bias=True):
    self.weight = Tensor.empty(out_features, in_features, dtype=dtypes.float32)
    self.bias = Tensor.empty(out_features, dtype=dtypes.float32) if bias else None

  def __call__(self, x: Tensor):
    # x: [Batch, Seq, In]
    # self.weight: [Out, In]
    
    # 1. 动态量化权重 (Per-Channel / Row-wise)
    # axis=1 表示对输出维度的每一行单独计算 Scale
    # w_fp8: [Out, In] (fp8)
    # w_inv_scale: [Out, 1] (float)
    w_fp8, w_inv_scale = quantize_to_fp8(self.weight, axis=1, dtype=dtypes.fp8e4m3)
    
    # 2. 动态量化输入 (Per-Token / Row-wise)
    # axis=-1 表示对每个 Token 向量单独计算 Scale
    # x_fp8: [Batch, Seq, In] (fp8)
    # x_inv_scale: [Batch, Seq, 1] (float)
    x_fp8, x_inv_scale = quantize_to_fp8(x, axis=-1, dtype=dtypes.fp8e4m3)

    x1, w1 = x_fp8, w_fp8
    
    # 3. 执行 FP8 矩阵乘法
    # [Batch, Seq, In] @ [In, Out] -> [Batch, Seq, Out]
    # 注意：这里 y 的结果通常会以 float32 累加 (accumulate)
    assert x_fp8.dtype in dtypes.fp8s
    assert w_fp8.dtype in dtypes.fp8s
    print(f"{x_fp8.shape=}, {w_fp8.T.shape=}")
    # y = x_fp8.dot(w_fp8.T, dtype=dtypes.float).contiguous().contiguous_backward()* x_inv_scale * w_inv_scale.reshape(1, -1)
    devs = GPUS
    if isinstance(GPUS, tuple) and len(GPUS) > 1:
      y = Tensor(Tensor.empty((x1.shape[0]//len(GPUS), x1.shape[1], w1.shape[0]), dtype=dtypes.float, device=devs).uop.multi(0), device=devs) # axis=0
      # y = Tensor(Tensor.empty((x.shape[0], x.shape[1], self.weight.shape[0]), dtype=dtypes.float, device=devs).uop, device=devs) # axis = None
      # print(f"custom {y.shape=}")
      # print(x1.device)
      # print(w1.device)
      # print(f"{y.device=}, {y.shape=}")
      # y.requires_grad = True
      print(y.device, x1.device, w1.device, devs)
      assert y.device == devs
      assert x1.device == devs
      assert w1.device == devs
      y=Tensor.custom_kernel(y, x1, w1, fxn=custom_linear,grad_fxn=custom_linear_backward_multi)[0]
    else:
      y = Tensor.empty((x.shape[0], x.shape[1], self.weight.shape[0]), dtype=dtypes.float)
      print(f"{x1.shape=}, {w1.shape=}, {x1.uop.axis=}, {w1.uop.axis=}")
      y=Tensor.custom_kernel(y, x1, w1, fxn=custom_linear,grad_fxn=custom_linear_backward)[0]

    y = y.contiguous() * (x_inv_scale * w_inv_scale.reshape(1, -1)).contiguous()
    # 4. 反量化 (Dequantize)
    # 此时 y 是 [Batch, Seq, Out]
    # x_inv_scale 是 [Batch, Seq, 1]，可以直接广播乘
    # w_inv_scale 是 [Out, 1]，我们需要 reshape 成 [1, Out] 来匹配 y 的最后一维
    
    if self.bias is not None: y = y + self.bias.cast(y.dtype)
    return y.cast(x.dtype)


"""

"""
def quantize_group(x: Tensor, axis: int):
    """
    常规量化：
    1. 计算 Scale
    2. 模拟 FP8 (Round + Clamp)
    3. 返回 Quantized Tensor 和 InvScale
    """
    max_val = x.abs().max(axis=axis, keepdim=True)
    scale = 448.0 / (max_val + 1e-8)
    # 模拟量化带来的精度损失
    x_quant = (x * scale).clamp(-448.0, 448.0)
    return x_quant, 1.0 / scale

# ==========================================
# 2. Grouped (512粒度) Linear - 无循环版
# ==========================================
# class GroupedFP8LinearNoLoop:
class FP8LinearBertBLOCK:
    def __init__(self, in_features, out_features, group_size=512, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        
        # 检查维度
        assert in_features % group_size == 0, "In_features 必须能被 group_size 整除"
        self.num_groups = in_features // group_size
        
        # 初始化权重
        self.weight = Tensor.kaiming_uniform(out_features, in_features)
        self.bias = Tensor.zeros(out_features) if bias else None
        
    def __call__(self, x: Tensor):
        # x: [Batch, Seq, In] -> [1024, 512, 4096]
        # w: [Out, In]        -> [1024, 4096]
        
        B, S, IN = x.shape
        OUT = self.out_features
        G = self.num_groups     # 8
        BLK = self.group_size   # 512
        
        # -----------------------------------------------------------
        # 第一步：Reshape 成组 (把 4096 拆成 8 x 512)
        # -----------------------------------------------------------
        
        # x: [B, S, 8, 512]
        x_g = x.reshape(B, S, G, BLK)
        
        # w: [Out, 8, 512]
        w_g = self.weight.reshape(OUT, G, BLK)
        
        # -----------------------------------------------------------
        # 第二步：分组量化 (axis=-1 即对 512 这个维度求 Scale)
        # -----------------------------------------------------------
        
        # x_q: [B, S, 8, 512], x_scale: [B, S, 8, 1]
        x_q, x_scale = quantize_to_fp8(x_g, axis=-1)
        
        # w_q: [Out, 8, 512], w_scale: [Out, 8, 1]
        w_q, w_scale = quantize_to_fp8(w_g, axis=-1)
        
        # -----------------------------------------------------------
        # 第三步：准备 Batched MatMul (核心技巧)
        # 目标：利用 Groups 作为 Batch 维度进行并行乘法
        # -----------------------------------------------------------
        
        # 1. 处理 X
        # 我们把 (B, S) 合并，并将 G 移到最前面
        # [B, S, G, 512] -> Permute [G, B, S, 512] -> Reshape [G, B*S, 512]
        x_batch = x_q.permute(2, 0, 1, 3).reshape(G, B*S, BLK)
        
        # 2. 处理 W
        # [Out, G, 512] -> Permute [G, Out, 512] -> 转置最后两维 [G, 512, Out]
        # 这样才能做矩阵乘法: (..., 512) @ (..., 512, Out)
        w_batch = w_q.permute(1, 0, 2).transpose(1, 2) # Shape: [G, 512, Out]
        
        # -----------------------------------------------------------
        # 第四步：执行 Batched Matrix Multiplication
        # -----------------------------------------------------------
        
        # Tinygrad/Numpy 会自动广播首位维度 G
        # [G, B*S, 512] @ [G, 512, Out] -> [G, B*S, Out]
        # 这里是一次性算出所有组的中间结果 (Int/FP8 模拟)
        print(f"{x_batch.shape=}, {w_batch.shape=}")
        # print(f"{x_batch.dtype=}, {w_batch.dtype=}")
        assert x_batch.dtype  in dtypes.fp8s
        assert w_batch.dtype  in dtypes.fp8s
        # x_batch.shape=(8, 524288, 512), w_batch.shape=(8, 512, 1024)
        y_batch_int = x_batch.dot(w_batch, dtype=dtypes.float32).contiguous().contiguous_backward()
        
        # -----------------------------------------------------------
        # 第五步：反量化 (Dequantize)
        # -----------------------------------------------------------
        
        # 我们需要把 Scale 也调整成 [G, B*S, Out] 的形状来做点乘
        
        # 1. Input Scale: [B, S, 8, 1] -> [8, B*S, 1]
        x_scale_batch = x_scale.permute(2, 0, 1, 3).reshape(G, B*S, 1)
        
        # 2. Weight Scale: [Out, 8, 1] -> [8, Out, 1] -> [8, 1, Out]
        w_scale_batch = w_scale.permute(1, 0, 2).transpose(1, 2)
        
        # 3. Apply Scales
        # [G, B*S, Out] * [G, B*S, 1] * [8, 1, Out] -> [G, B*S, Out]
        y_batch_float = y_batch_int * x_scale_batch * w_scale_batch
        
        # -----------------------------------------------------------
        # 第六步：规约 (Sum over Groups) 与 还原形状
        # -----------------------------------------------------------
        
        # 对 G 维度求和 (相当于公式里的 Σ)
        # [G, B*S, Out] -> [B*S, Out]
        y_merged = y_batch_float.sum(axis=0)
        
        # 还原回 [B, S, Out]
        y_out = y_merged.reshape(B, S, OUT)
        
        if self.bias is not None:
            y_out = y_out + self.bias
            
        return y_out
      
FP8LinearBert = FP8LinearBertBasic
#   FP8LinearBert = FP8LinearBertBLOCK
# if FP8 == 3:
if FP8 == 2:
  FP8LinearBert = FP8LinearBertRow

class EmbeddingBert(nn.Embedding):
  def __init__(self, vocab_size:int, embed_size:int, std=0.02):
    self.vocab_sz, self.embed_sz = vocab_size, embed_size
    self.weight = std * rand_truncn(vocab_size, embed_size, dtype=dtypes.float32)

  def __call__(self, idx:Tensor) -> Tensor:
    if idx.numel() == 0: return Tensor.empty(idx.shape+(self.embed_sz,), dtype=self.weight.dtype, device=self.weight.device)
    arange_shp, weight_shp, big_shp = (1, 1, self.vocab_sz, 1), (1, 1, self.vocab_sz, self.embed_sz), idx.shape+(self.vocab_sz, self.embed_sz,)
    if not hasattr(self, 'arange'): self.arange = Tensor.arange(self.vocab_sz, requires_grad=False, device=self.weight.device).reshape(arange_shp)
    arange, idx, vals = self.arange.expand(big_shp), idx.reshape(idx.shape+(1, 1,)).expand(big_shp), self.weight.cast(dtypes.default_float).reshape(weight_shp).expand(big_shp)
    return (arange == idx).where(vals, 0).sum(2, dtype=vals.dtype)

class LayerNormBert:
  def __init__(self, normalized_shape:Union[int, tuple[int, ...]], eps:float=1e-12, elementwise_affine:bool=True):
    self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
    self.axis, self.eps, self.elementwise_affine = tuple(-1-i for i in range(len(self.normalized_shape))), eps, elementwise_affine
    self.weight, self.bias = (Tensor.ones(*self.normalized_shape, dtype=dtypes.float32), Tensor.zeros(*self.normalized_shape, dtype=dtypes.float32)) if elementwise_affine else (None, None)

  def __call__(self, x:Tensor):
    assert self.normalized_shape == x.shape[-len(self.normalized_shape):], f"last dimensions of {x.shape} must match {self.normalized_shape}"
    xn = x.cast(dtypes.float32).layernorm(eps=self.eps, axis=self.axis).cast(x.dtype)
    if not self.elementwise_affine: return xn
    return (xn * self.weight.cast(dtypes.default_float) + self.bias.cast(dtypes.default_float))

class FrozenBatchNorm2dRetinaNet(nn.BatchNorm2d):
  def __init__(self, sz:int, eps=1e-5, affine=True, track_running_stats=True, momentum=0.1):
    self.eps, self.track_running_stats, self.momentum = eps, track_running_stats, momentum

    self.weight = Tensor.ones(sz, dtype=dtypes.float32, requires_grad=False) if affine else None
    self.bias = Tensor.zeros(sz, dtype=dtypes.float32, requires_grad=False) if affine else None

    if track_running_stats: self.running_mean, self.running_var = Tensor.zeros(sz, dtype=dtypes.float32, requires_grad=False), Tensor.ones(sz, dtype=dtypes.float32, requires_grad=False)
    self.num_batches_tracked = Tensor.zeros(1, dtype=dtypes.long, requires_grad=False)

  def __call__(self, x:Tensor) -> Tensor:
    batch_mean, batch_var = super().calc_stats(x.cast(dtypes.float32))
    if self.track_running_stats and Tensor.training:
      self.running_mean.assign((1-self.momentum) * self.running_mean + self.momentum * batch_mean.detach().cast(self.running_mean.dtype))
      self.running_var.assign((1-self.momentum) * self.running_var + self.momentum * x.numel()/(x.numel()-x.shape[1]) * batch_var.detach().cast(self.running_var.dtype))
      self.num_batches_tracked += 1
    return x.cast(dtypes.float32).batchnorm(self.weight, self.bias, batch_mean, batch_var.add(self.eps).rsqrt()).cast(x.dtype)

class Conv2dNormalRetinaNet(nn.Conv2d):
  def __init__(self, in_channels:int, out_channels:int, kernel_size:int|tuple[int, ...],
               stride:int=1, padding:int|tuple[int, ...]|str=0, dilation:int=1, groups:int=1,
               bias:bool=True, prior_prob:float|None=None):
    super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
    self.weight = Tensor.normal(*self.weight.shape, std=0.01, dtype=dtypes.float32)
    if bias:
      if prior_prob:
        prior_prob = Tensor(prior_prob, device=self.bias.device, dtype=dtypes.float32).expand(*self.bias.shape)
        self.bias = -(((1 - prior_prob) / prior_prob).log())
      else: self.bias = Tensor.zeros_like(self.bias, dtype=dtypes.float32)

  def __call__(self, x:Tensor) -> Tensor:
    return x.conv2d(self.weight.cast(dtypes.default_float), self.bias.cast(dtypes.default_float) if self.bias is not None else None,
                    groups=self.groups, stride=self.stride, padding=self.padding)

class Conv2dKaimingUniformRetinaNet(nn.Conv2d):
  def __init__(self, in_channels:int, out_channels:int, kernel_size:int|tuple[int, ...],
               stride:int=1, padding:int|tuple[int, ...]|str=0, dilation:int=1, groups:int=1,
               bias:bool=True):
    super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
    self.weight = Tensor.kaiming_uniform(*self.weight.shape, a=1, dtype=dtypes.float32)
    if bias: self.bias = Tensor.zeros_like(self.bias, dtype=dtypes.float32)

  def __call__(self, x:Tensor) -> Tensor:
    return x.conv2d(self.weight.cast(dtypes.default_float), self.bias.cast(dtypes.default_float) if self.bias is not None else None,
                    groups=self.groups, stride=self.stride, padding=self.padding)

class Conv2dRetinaNet(nn.Conv2d):
  def __init__(self, in_channels:int, out_channels:int, kernel_size:int|tuple[int, ...],
               stride:int=1, padding:int|tuple[int, ...]|str=0, dilation:int=1, groups:int=1,
               bias:bool=True):
    super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
    scale = 1 / math.sqrt(in_channels * prod(self.kernel_size))
    self.weight = Tensor.uniform(out_channels, in_channels//groups, *self.kernel_size, low=-scale, high=scale, dtype=dtypes.float32)
    self.bias: Tensor|None = Tensor.uniform(out_channels, low=-scale, high=scale, dtype=dtypes.float32) if bias else None

  def __call__(self, x:Tensor) -> Tensor:
    return x.conv2d(self.weight.cast(dtypes.default_float), self.bias.cast(dtypes.default_float) if self.bias is not None else None,
                    groups=self.groups, stride=self.stride, dilation=self.dilation, padding=self.padding)

# copy torch AMP: isolate mixed precision to just the below autocast ops, instead of using dtypes.default_float which affects all new Tensors
class AutocastLinear(nn.Linear):
  cast_dtype=dtypes.bfloat16 # enable monkeypatching of the mixed precision dtype
  def __call__(self, x:Tensor) -> Tensor:
    dtype = type(self).cast_dtype
    return x.cast(dtype).linear(self.weight.cast(dtype).transpose(), self.bias.cast(dtype) if self.bias is not None else None)

class AutocastConv2d(nn.Conv2d):
  cast_dtype=dtypes.bfloat16
  def __call__(self, x:Tensor) -> Tensor:
    dtype = type(self).cast_dtype
    return x.cast(dtype).conv2d(self.weight.cast(dtype), self.bias.cast(dtype), self.groups, self.stride, self.dilation, self.padding)

# copy torch AMP: upcast to float32 before GroupNorm and LayerNorm
class AutocastGroupNorm(nn.GroupNorm):
  def __call__(self, x:Tensor) -> Tensor:
    return super().__call__(x.cast(dtypes.float32))

class AutocastLayerNorm(nn.LayerNorm):
  def __call__(self, x:Tensor) -> Tensor:
    return super().__call__(x.cast(dtypes.float32))

def zero_module(module):
  for p in get_parameters(module): p.assign(Tensor.zeros_like(p).contiguous())

# Stable Diffusion mlperf reference doesn't call scaled_dot_product_attention
# copy torch AMP: upcast to float32 before softmax on CUDA
def attn_f32_softmax(q:Tensor, k:Tensor, v:Tensor) -> Tensor:
  return (q.matmul(k.transpose(-2,-1), dtype=dtypes.float32) / math.sqrt(q.shape[-1])).softmax(-1).cast(q.dtype) @ v

def init_stable_diffusion(version:str, pretrained:str, devices:list[str]):
  from examples.stable_diffusion import StableDiffusion
  from tinygrad.nn.state import safe_load, safe_save, load_state_dict, get_state_dict
  from tempfile import TemporaryDirectory
  model = StableDiffusion(version=version, pretrained=pretrained)
  unet:UNetModel = model.model.diffusion_model

  # this prevents extra consumption of memory, enabling much larger BS
  Tensor.realize(*get_parameters(unet))
  with TemporaryDirectory(prefix="unet_init") as tmp:
    safe_save(get_state_dict(unet), init_fn:=f"{tmp}/init_model.safetensors")
    load_state_dict(unet, safe_load(init_fn))

  sqrt_alphas_cumprod = model.alphas_cumprod.sqrt().realize()
  sqrt_one_minus_alphas_cumprod = (1 - model.alphas_cumprod).sqrt().realize()

  if len(devices) > 1:
    to_move = [sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod]
    if version == "v2-mlperf-train": to_move += get_parameters(unet) + get_parameters(model.cond_stage_model)
    for p in to_move:
      p.to_(devices)
    with Context(BEAM=0):
      Tensor.realize(*to_move)

  return model, unet, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod
