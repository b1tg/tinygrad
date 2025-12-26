import math
from typing import Union

from tinygrad import Tensor, nn, dtypes, UOp, Device
from tinygrad.uop.ops import KernelInfo, AxisType, Ops
from tinygrad.helpers import prod, argfix, getenv, Context
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
    return x.cast(dtypes.float16).linear(self.weight.cast(dtypes.float16).transpose(), self.bias.cast(dtypes.float16) if self.bias is not None else None)

def k_clamp_back(grads:UOp, kernel:UOp):
  return (None, grads)
  return (None, Tensor(grads, grads.device).uop)

def k_clamp(y:UOp, x:UOp):
  y = y.flatten()
  x = x.flatten()
  i = UOp.range(x.size, 0)
  x1 = x[i].maximum(UOp.const(x.dtype.base, -448.0)).minimum(UOp.const(x.dtype.base, 448.0))
  return y[i].store(x1).end(i).sink(arg=KernelInfo(name=f"custom_clamp_{x.size}"))  

# GPUS =  tuple(f"{Device.DEFAULT}:{i}" for i in range(getenv("GPUS", 1)))
if getenv("GPUS", 1) > 1:
  # GPUS = tuple(f'{Device.DEFAULT}:{i}' if i else f'{Device.DEFAULT}' for i in range(getenv("GPUS", 1)))
  GPUS = tuple(f"{Device.DEFAULT}:{i}" for i in range(getenv("GPUS", 1)))
else:
  GPUS = Device.DEFAULT
FP8 = getenv("FP8", 0)

def quantize_to_fp8(x: Tensor, axis=None, dtype=dtypes.fp8e4m3):
  x_abs_max = x.abs().max(axis=axis, keepdim=True).detach()
  scale = 448. / (x_abs_max + 1e-8)  
  x = x*scale
  if x.uop.axis == 0:
    y = Tensor(Tensor.empty((x.shape[0]//len(GPUS), *x.shape[1:]), device=GPUS, dtype=x.dtype).uop.multi(0), device=GPUS)
  else:
    y = Tensor(Tensor.empty((x.shape[0], *x.shape[1:]), device=GPUS, dtype=x.dtype).uop, device=GPUS)
  y = Tensor.custom_kernel(y, x,  fxn=k_clamp, grad_fxn=k_clamp_back)[0]
  res = y.cast(dtype).contiguous()
  return res, scale.float().reciprocal()
def quantize_to_fp8(x: Tensor, axis=None, dtype=dtypes.fp8e4m3):
    # 使用 detach() 替代 max1()
    x_abs_max = x.abs().max(axis=axis, keepdim=True).detach()
    scale = 448. / (x_abs_max + 1e-8)
    # scale = Tensor(1.0, x.device)
    
    x_scaled = x * scale
    # return x_scaled.cast(dtype).contiguous().contiguous_backward(), scale.float().reciprocal().contiguous().contiguous_backward()

    
    # 使用 STE clamp 替代 custom kernel
    x_det = x_scaled.detach()
    x_clamped = x_det.maximum(-448.0).minimum(448.0)
    # x_clamped = x_det.clamp(-448.0, 448.0)
    x_clamped_ste = x_scaled + (x_clamped - x_det)
    
    res = x_clamped_ste.cast(dtype).contiguous().contiguous_backward()
    return res, scale.float().reciprocal().contiguous().contiguous_backward()
def custom_linear(C:UOp, A:UOp, B:UOp) -> UOp:
  SEQ = A.shape[1]
  OUT = B.shape[0]
  IN = B.shape[-1]
  c2 = UOp.range(SEQ, 2, AxisType.LOOP)
  c5 = UOp.range(OUT, 3, AxisType.LOOP)
  c8 = UOp.range(C.size//SEQ//OUT, 1, AxisType.LOOP)
  c16 = UOp.range(IN, 0, AxisType.REDUCE)
  c27 = (A.index((c2*IN+c16+c8*IN*SEQ))*B.index((c5*IN+c16))).cast(dtypes.float)
  c28 = c27.reduce(c16, arg=Ops.ADD)
  c30 = C.index((c2*OUT+c5+c8*OUT*SEQ), ptr=True).store(c28).end(c8, c2, c5)
  return c30.sink(arg=KernelInfo(name=f"custom dot {A.shape}x{B.shape}"))

def custom_linear_backward(gradient:UOp, kernel:UOp) -> tuple[UOp, UOp]:
  out, a, b = kernel.src
  print(f"{a.shape=}, {b.shape=}, {gradient.shape=}")
  print(f"{a.dtype=}, {b.dtype=}, {gradient.dtype=}")

  # a.shape=(66, 512, 4096), b.shape=(1024, 4096), gradient.shape=(66, 512, 1024)
  a2 = Tensor(a, device=a.device).reshape(a.shape[0]*a.shape[1], a.shape[-1])
  g2 = Tensor(gradient, device=gradient.device).reshape(gradient.shape[0]*gradient.shape[1], gradient.shape[-1])
  # if 1 or g2.shape[-1] < 10240:
  g2, s = quantize_to_fp8(g2) 
  # print(f"{a2.shape=}, {g2.shape=}, {b.shape=}")
  grad_b = (g2.T.dot(a2,dtype=dtypes.float)).contiguous()*s
  grad_b = grad_b.cast(dtypes.float)
  grad_a = (g2.dot(Tensor(b, device=b.device), dtype=dtypes.float)).contiguous().reshape(a.shape)*s
  return (None, grad_a.uop, grad_b.uop)
  # else:
  #   # g2,s = g2, 1
  #   grad_b = (g2.T.dot(a2.cast(dtypes.half)))
  #   grad_b = grad_b.cast(dtypes.float)
  #   grad_a = (g2.dot(Tensor(b.cast(dtypes.half), device=b.device))).reshape(a.shape)
  #   return (None, grad_a.uop, grad_b.uop)


def custom_linear_backward(gradient: UOp, kernel: UOp) -> tuple[UOp, UOp]:
    out, a, b = kernel.src
    
    g_tensor = Tensor(gradient, device=gradient.device)
    a_tensor = Tensor(a, device=a.device)
    b_tensor = Tensor(b, device=b.device)
    
    g_quantized, scale = quantize_to_fp8(g_tensor)
    
    # 确保scale是标量
    scale = scale.reshape(())  # reshape to scalar
    
    # grad_a
    grad_a = ((g_quantized @ b_tensor) * scale).cast(dtypes.float)
    
    # grad_b
    flatten_size = 1
    for dim in gradient.shape[:-1]:
        flatten_size *= dim
    
    grad_b = ((g_quantized.reshape(flatten_size, -1).T @ 
               a_tensor.reshape(flatten_size, -1)) * scale).cast(dtypes.float)
    
    return (None, grad_a.uop, grad_b.uop)

# DEPRECATED: This implementation will be removed in the future.
# Use extra.fp8.FP8Linear instead for better shape handling and drop-in compatibility.
class FP8LinearBert:
  def __init__(self, in_features, out_features, bias=True):
    self.weight = Tensor.empty(out_features, in_features, dtype=dtypes.float32)
    self.bias = Tensor.empty(out_features, dtype=dtypes.float32) if bias else None
  def __call__(self, x:Tensor):
    # return x.cast(dtypes.default_float).linear(self.weight.cast(dtypes.default_float).transpose(), self.bias.cast(dtypes.default_float) if self.bias is not None else None)

    w1, ws = quantize_to_fp8(self.weight)
    x1, s = quantize_to_fp8(x)
    if isinstance(GPUS, (tuple, list)) and len(GPUS) > 1:
      y = Tensor(Tensor.empty((x.shape[0]//len(GPUS), x.shape[1], self.weight.shape[0]), dtype=dtypes.float, device=GPUS).uop.multi(0), device=GPUS)
      y = Tensor.custom_kernel(y, x1, w1, fxn=custom_linear,grad_fxn=custom_linear_backward)[0]
    else:
      y = Tensor.empty((x.shape[0], x.shape[1], self.weight.shape[0]), dtype=dtypes.float)
      y = Tensor.custom_kernel(y, x1, w1, fxn=custom_linear,grad_fxn=custom_linear_backward)[0]
    # y = x1.dot(w1.T, dtype=dtypes.float).cast(dtypes.default_float)
    y = (ws * s).contiguous() * y.contiguous()
    if self.bias is not None: y = y.cast(dtypes.default_float) + self.bias.cast(dtypes.default_float)
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
