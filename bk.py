from tinygrad import Tensor, dtypes, nn
from tinygrad.helpers import getenv
from tinygrad import Tensor, dtypes, UOp
from tinygrad.uop.ops import KernelInfo, AxisType, Ops
from tinygrad.helpers import prod, argfix, Context
from tinygrad.helpers import Timing
import numpy as np
Tensor.manual_seed(42)
CUSTOM_CLAMP = getenv("CUSTOM_CLAMP", 0)
CUSTOM_AMAX = getenv("CUSTOM_AMAX", 0)
PO2 = getenv("PO2", 0)
FP8 = getenv("FP8", 0)
def quantize_to_fp8(x: Tensor, axis=None, dtype=dtypes.fp8e4m3):
  if CUSTOM_AMAX:
    # y = Tensor.empty((), dtype=x.dtype)
    # y = Tensor.custom_kernel(y, x, fxn=q_abs_max_kernel, grad_fxn=q_abs_max_kernel_back)[0]
    # scale = 448. / (y + 1e-8)  
    x_abs_max = x.abs().max1(axis=axis, keepdim=True)
    scale = 448. / (x_abs_max + 1e-8)  
    # scale = y
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
    # y = Tensor.empty_like(x).contiguous().contiguous_backward()
    # y = Tensor.custom_kernel(y, x, Tensor(1.0), fxn=k_clamp, grad_fxn=k_clamp_back)[0]
    y = x.nround()
    res= y.cast(dtype).contiguous()
    # res = y
    return res, scale.float().reciprocal().contiguous()
  # ---
  # y = (x * scale).clamp(-448.0, 448.0)
  y = x.clamp(-448.0, 448.0)
  res= y.cast(dtype).contiguous().contiguous_backward()
  return res, scale.float().reciprocal().contiguous().contiguous_backward()
  # ---
# rejection sampling truncated randn
def rand_truncn(*shape, dtype=None, truncstds=2, **kwargs) -> Tensor:
  CNT=8
  x = Tensor.randn(*(*shape, CNT), dtype=dtype, **kwargs)
  ctr = Tensor.arange(CNT).reshape((1,) * len(x.shape[:-1]) + (CNT,)).expand(x.shape)
  take = (x.abs() <= truncstds).where(ctr, CNT).min(axis=-1, keepdim=True)  # set to 0 if no good samples
  return (ctr == take).where(x, 0).sum(axis=-1)
class LinearBert(nn.Linear):
  def __init__(self, in_features, out_features, bias=True, std=0.02):
    self.weight = Tensor.randn(out_features, in_features, dtype=dtypes.float16)
    self.bias = Tensor.zeros(out_features, dtype=dtypes.float16) if bias else None

  def __call__(self, x:Tensor):
    return x.cast(dtypes.half).contiguous().linear(self.weight.cast(dtypes.half).transpose(), self.bias.cast(dtypes.half) if self.bias is not None else None).contiguous()
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
 

def custom_linear(C:UOp, A:UOp, B:UOp) -> UOp:
  # c0 = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(34603008), (), 0)
  c0 = C
  c2 = UOp.range(512, 2, AxisType.LOOP)
  c5 = UOp.range(1024, 3, AxisType.LOOP)
  c8 = UOp.range(C.size//512//1024, 1, AxisType.LOOP)
  # c13 = UOp(Ops.DEFINE_GLOBAL, dtypes.fp8e4m3.ptr(138412032), (), 1)
  c13 = A
  c16 = UOp.range(4096, 0, AxisType.REDUCE)
  # c22 = UOp(Ops.DEFINE_GLOBAL, dtypes.fp8e4m3.ptr(4194304), (), 2)
  c22 = B
  c27 = (c13.index((c2*4096+c16+c8*2097152))*c22.index((c5*4096+c16))).cast(dtypes.float)
  c28 = c27.reduce(c16, arg=Ops.ADD)
  c30 = c0.index((c2*1024+c5+c8*524288), ptr=True).store(c28).end(c8, c2, c5)
  ast = c30.sink(arg=KernelInfo(name=f"cust0m dot"))
  return ast
def custom_linear_backward(gradient:UOp, kernel:UOp) -> tuple[UOp, UOp]:
  out, a, b = kernel.src
  # grad_a = (Tensor(gradient) @ Tensor(b).cast(dtypes.float)).uop
  # 
# a.shape=(66, 512, 4096), a.dtype=dtypes.fp8e4m3
# b.shape=(1024, 4096), b.dtype=dtypes.fp8e4m3
# Tensor(gradient).shape=(66, 512, 1024), Tensor(gradient).dtype=dtypes.float
  # grad_b = (Tensor(a).cast(dtypes.float).reshape() @ Tensor(gradient)).uop
  a2 = Tensor(a).reshape(a.shape[0]*512, 4096).contiguous()        # (33792, 4096)
  g2 = Tensor(gradient).reshape(a.shape[0]*512, 1024)  # (33792, 1024)
  g2, s = quantize_to_fp8(g2) 
  grad_b = (g2.T @ a2)*s                  # (1024, 4096)
  grad_b = grad_b.cast(dtypes.float)
  grad_a = (g2 @ Tensor(b)).reshape(a.shape)*s
  return (None, grad_a.uop, grad_b.uop)
  out, a, b = kernel.src
  # 1/0
  # return None, Tensor.ones_like(Tensor(a)).uop, Tensor.ones_like(Tensor(b)).uop, Tensor.ones_like(Tensor(d)).uop
  # Tensor(gradient).shape=(128, 512, 1024), Tensor(a).shape=(128, 512, 1024)
  # print(f"{Tensor(gradient).shape=}, {Tensor(a).shape=}, {Tensor(b).shape=}")
  dy =gradient 
# a.shape=(66, 512, 4096), a.dtype=dtypes.fp8e4m3
# b.shape=(1024, 4096), b.dtype=dtypes.fp8e4m3
# Tensor(gradient).shape=(66, 512, 1024), Tensor(gradient).dtype=dtypes.float
  # print(f"{a.shape=}, {a.dtype=}")
  # print(f"{b.shape=}, {b.dtype=}")
  # print(f"{Tensor(gradient).shape=}, {Tensor(gradient).dtype=}")
  dy2 = Tensor(dy).reshape(-1, dy.shape[-1])
  # dy3, s = quantize_to_fp8(dy2)
  # dy2.cas

  x2  = Tensor(a).reshape(-1, a.shape[-1]).cast(dtypes.float)
  grad_b = dy2.T @ x2

  return (None, Tensor(a).uop, grad_b.uop)
def custom_linear(C:UOp, A:UOp, B:UOp) -> UOp:
  # c0 = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(34603008), (), 0)
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
  out, a, b = kernel.src
  assert a.shape[1] == 512, f"{a.shape=}"
  a2 = Tensor(a).reshape(a.shape[0]*512, a.shape[-1]).contiguous()
  g2 = Tensor(gradient).reshape(gradient.shape[0]*gradient.shape[1], gradient.shape[-1])
  g2, s = quantize_to_fp8(g2) 
  grad_b = (g2.T.dot(a2,dtype=dtypes.float))*s
  grad_b = grad_b.cast(dtypes.float)
  grad_a = (g2.dot(Tensor(b), dtype=dtypes.float)).reshape(a.shape)*s
  return (None, grad_a.uop, grad_b.uop)
class FP8LinearBertBasic:
  def __init__(self, in_features, out_features, bias=True):
    # (1024, 4096)
    self.weight = Tensor.empty(out_features, in_features, dtype=dtypes.float16)
    # (1024)
    self.bias = Tensor.empty(out_features, dtype=dtypes.float16) if bias else None
  def __call__(self, x:Tensor):
    # return self.kk(x).contiguous().contiguous_backward()
    # print(f"FP8LinearBertBasic {x.shape=}, {self.weight.shape=}")
    # x.shape=(128, 512, 4096), self.weight.shape=(1024, 4096)
    # x.shape=(1024, 512, 4096), self.weight.shape=(1024, 4096) GPUS=8, BS=1024
    w1, ws = quantize_to_fp8(self.weight)
    x1, s = quantize_to_fp8(x)
    assert x1.dtype in dtypes.fp8s
    assert w1.dtype in dtypes.fp8s
    # x1.shape=(192, 512, 1024),w1.T.shape=(1024, 1024)
    # x1.shape=(24, 512, 1024), w1.shape=(1024, 1024)
    # y = Tensor.empty((66, 512, 1024), dtype=dtypes.float)
    y = Tensor.empty((x.shape[0], x.shape[1], self.weight.shape[0]), dtype=dtypes.float)
    devs = ('AMD:0', 'AMD:1')
    y = Tensor(Tensor.empty((x.shape[0], x.shape[1], self.weight.shape[0]), dtype=dtypes.float, device=devs).uop.multi(-1), device=devs)
    # y.requires_grad = True
    # assert x. 
    if getenv("CUSTOM", 1) == 0:
      y = x1.dot(w1.T, dtype=dtypes.float).contiguous().contiguous_backward()
      # ; print("dot dot")
    else:
      y = Tensor.custom_kernel(y, x1, w1, fxn=custom_linear,grad_fxn=custom_linear_backward )[0].contiguous().contiguous_backward()
      # ; print("--- CUSTOM!!")

    y = (ws * s).contiguous() * y.contiguous()
    if self.bias is not None: y = y + self.bias.cast(y.dtype)
    return y.cast(x.dtype)
# Max relative difference among violations: 20336.
#  ACTUAL: array([[[ -30.8   ,  -72.    ,   19.47  , ...,  -55.88  ,  -20.84  ,
#            39.34  ],
#         [ -27.77  ,  -77.94  ,  -19.64  , ...,  -59.16  ,  -61.94  ,...
#  DESIRED: array([[[ -31.64  ,  -73.1   ,   20.86  , ...,  -53.6   ,  -17.8   ,
#            40.25  ],
#         [ -27.8   ,  -79.06  ,  -17.97  , ...,  -56.72  ,  -60.2   ,...
if __name__ == "__main__":
  IN = 4096
  OUT = 1024
  # FP8LinearBertBasic x.shape=(66, 512, 4096), self.weight.shape=(1024, 4096)
  # back, a.shape=(66, 512, 4096), b.shape=(1024, 4096), gradient.shape=(66, 512, 1024)


  IN, OUT = 1024, 4096
  IN, OUT = 1024, 1024
  BS = 1024
  IN = getenv("IN", IN)
  OUT = getenv("OUT", OUT)
  BS = getenv("BS", BS)
  print(f"{IN=}, {OUT=}, {BS=}")
  print("----")

  # IN, OUT = 512, 512

  # FP8LinearBertBasic x.shape=(66, 512, 1024), self.weight.shape=(4096, 1024)
  # back, a.shape=(66, 512, 1024), b.shape=(4096, 1024), gradient.shape=(66, 512, 4096)

  m0 = LinearBert(IN, OUT)
  m0.bias.assign(Tensor.randn(OUT).cast(dtypes.half).realize())
  m0.weight.requires_grad = True
  if FP8:
    m = FP8LinearBertBasic(IN, OUT)
  else:
    m = LinearBert(IN, OUT)
  m.weight.requires_grad=True
  m.weight.assign(m0.weight)
  m.bias.assign(m0.bias) 
  

  exit()
  if 1:
    for i in range(1):
      x = Tensor.rand((BS, 512, IN)).cast(dtypes.half)
      x.requires_grad = True
      m0_res = m0(x).contiguous().contiguous_backward().realize()
      m_res = m(x).contiguous().contiguous_backward().realize()
      y0 = m0(x).contiguous().contiguous_backward().relu().sum().backward().realize()
      y = m(x).contiguous().contiguous_backward().relu().sum().backward().realize()
      (m0.weight.grad.realize())
      (m.weight.grad.realize())

  x = Tensor.rand((BS, 512, IN)).cast(dtypes.half)
  x.requires_grad = True
  with Timing(f"{m0.__class__} use:"):
      m0_res = m0(x).contiguous().contiguous_backward().realize()
      y0 = m0(x).contiguous().contiguous_backward().relu().sum().backward().realize()
      # print(m0.weight.grad.numpy())
      m0.weight.grad.realize()
  x = Tensor.rand((BS, 512, IN)).cast(dtypes.half)
  x.requires_grad = True
  # x.realize()
  with Timing(f"{m.__class__} use:"):
      m_res = m(x).contiguous().contiguous_backward().realize()
      y = m(x).contiguous().contiguous_backward().relu().sum().backward().realize()
      # print(m.weight.grad.numpy())
      m.weight.grad.realize()
  print("---")
      
  exit()
  from tinygrad.helpers import getenv
  if getenv("SHOW")>0:
    print(f"{m0_res.numpy()=}")
    print("---------------")
    print(f"{m_res.numpy()=}")
  # np.testing.assert_allclose(m_res.numpy(), m0_res.numpy(), rtol=1e-1, atol=1e-1)
  print("ok")
  # exit()
  y0 = m0(x).contiguous().contiguous_backward().relu().sum().backward().realize()
  y = m(x).contiguous().contiguous_backward().relu().sum().backward().realize()
  print(m0.weight.grad.numpy())
  print("---")
  print(m.weight.grad.numpy())
  # print(x.grad.numpy())