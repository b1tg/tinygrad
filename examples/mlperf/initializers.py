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
  y, x, s = kernel.src
  return (None, Tensor(grads).uop, None) # through
  return (None, None, None)
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
 
def k_clamp(y:UOp, x:UOp, s: UOp):
  y = y.flatten()
  x = x.flatten()
  # s = s.flatten()
  i = UOp.range(x.size, 0)
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
print("== CUSTOM_CLAMP: ", CUSTOM_CLAMP)
print("== CUSTOM_AMAX: ", CUSTOM_AMAX)
def quantize_to_fp8(x: Tensor, axis=None, dtype=dtypes.fp8e4m3):
  if CUSTOM_AMAX:
    # y = Tensor.empty((), dtype=x.dtype)
    # y = Tensor.custom_kernel(y, x, fxn=q_abs_max_kernel, grad_fxn=q_abs_max_kernel_back)[0]
    # scale = 448. / (y + 1e-8)  
    x_abs_max = x.abs().max1()
    scale = 448. / (x_abs_max + 1e-8)  
    # scale = y
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
    # y = Tensor.empty_like(x)
    # y = Tensor.custom_kernel(y, x, Tensor(1.0), fxn=k_clamp, grad_fxn=k_clamp_back)[0]
    y = x.nround()
    res= y.cast(dtype)
    # res = y
    return res, scale.float().reciprocal()
  # ---
  # y = (x * scale).clamp(-448.0, 448.0)
  y = x.clamp(-448.0, 448.0)
  res= y.cast(dtype)
  return res, scale.float().reciprocal()
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
class FP8LinearBert:
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
    # print(f"{x.shape=}, {self.weight.shape=}")
    # x.shape=(128, 512, 4096), self.weight.shape=(1024, 4096)
    w1, ws = quantize_to_fp8(self.weight)
    x1, s = quantize_to_fp8(x)
    # x1.shape=(192, 512, 1024),w1.T.shape=(1024, 1024)
    # x1.shape=(24, 512, 1024), w1.shape=(1024, 1024)
    y = x1.dot(w1.T, dtype=dtypes.float) * ws * s
    if self.bias is not None: y = y + self.bias.cast(y.dtype)
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

class FP8LinearBertAA:
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
    
    # 3. 执行 FP8 矩阵乘法
    # [Batch, Seq, In] @ [In, Out] -> [Batch, Seq, Out]
    # 注意：这里 y 的结果通常会以 float32 累加 (accumulate)
    y = x_fp8.dot(w_fp8.T, dtype=dtypes.float)* x_inv_scale * w_inv_scale.reshape(1, -1)
    
    # 4. 反量化 (Dequantize)
    # 此时 y 是 [Batch, Seq, Out]
    # x_inv_scale 是 [Batch, Seq, 1]，可以直接广播乘
    # w_inv_scale 是 [Out, 1]，我们需要 reshape 成 [1, Out] 来匹配 y 的最后一维
    
    if self.bias is not None: y = y + self.bias.cast(y.dtype)
    return y.cast(x.dtype)

class EmbeddingBert(nn.Embedding):
  def __init__(self, vocab_size:int, embed_size:int, std=0.02):
    self.vocab_sz, self.embed_sz = vocab_size, embed_size
    self.weight = std * rand_truncn(vocab_size, embed_size, dtype=dtypes.float32)

  def __call__(self, idx:Tensor) -> Tensor:
    if idx.numel() == 0: return Tensor.empty(idx.shape+(self.embed_sz,), dtype=self.weight.dtype, device=self.weight.device)
    arange_shp, weight_shp, big_shp = (1, 1, self.vocab_sz, 1), (1, 1, self.vocab_sz, self.embed_sz), idx.shape+(self.vocab_sz, self.embed_sz,)
    if not hasattr(self, 'arange'): self.arange = Tensor.arange(self.vocab_sz, requires_grad=False, device=self.weight.device).reshape(arange_shp)
    arange, idx, vals = self.arange.expand(big_shp), idx.reshape(idx.shape+(1, 1,)).expand(big_shp), self.weight.cast(dtypes.default_float).reshape(weight_shp).expand(big_shp)
    # TODO: contiguous() here because the embedding dropout creates different asts on each device, and search becomes very slow.
    # Should fix with fixing random ast on multi device, and fuse arange to make embedding fast.
    return (arange == idx).mul(vals).sum(2, dtype=vals.dtype).contiguous()

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
