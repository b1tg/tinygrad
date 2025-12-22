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
  res= y.cast(dtype).contiguous()
  return res, scale.float().reciprocal().contiguous()
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
class FP8LinearBertRow:
  def __init__(self, in_features, out_features, bias=True):
    self.weight = Tensor.empty(out_features, in_features, dtype=dtypes.float16)
    self.bias = Tensor.empty(out_features, dtype=dtypes.float16) if bias else None

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
    assert x_fp8.dtype in dtypes.fp8s
    assert w_fp8.dtype in dtypes.fp8s
    print(f"{x_fp8.shape=}, {w_fp8.T.shape=}")
    # y = x_fp8.dot(w_fp8.T, dtype=dtypes.float).contiguous().contiguous_backward()* x_inv_scale * w_inv_scale.reshape(1, -1)
    y = Tensor.empty((x.shape[0], x.shape[1], self.weight.shape[0]), dtype=dtypes.float)
    # print(f"{x1.shape=}, {w1.shape=}, {x1.uop.axis=}, {w1.uop.axis=}")
    y=Tensor.custom_kernel(y, x_fp8, w_fp8, fxn=custom_linear,grad_fxn=custom_linear_backward)[0]
    
    # 4. 反量化 (Dequantize)
    # 此时 y 是 [Batch, Seq, Out]
    # x_inv_scale 是 [Batch, Seq, 1]，可以直接广播乘
    # w_inv_scale 是 [Out, 1]，我们需要 reshape 成 [1, Out] 来匹配 y 的最后一维
    
    if self.bias is not None: y = y + self.bias.cast(y.dtype)
    return y.cast(x.dtype)
def cosine_similarity(x1, x2):
  num = (x1 * x2).sum(axis=-1)
  den = (x1**2).sum(axis=-1).sqrt() * (x2**2).sum(axis=-1).sqrt()
  return num / den
def benchmark():
  M = 0
  M1 = 0
  ts0 = []
  ts = []
  t1s = []
  # ** init **
  for i in range(1000):
    m0 = LinearBert(IN, OUT)
    m0.bias.assign(Tensor.randn(OUT).cast(dtypes.half).realize())
    m0.weight.requires_grad = True

    m = FP8LinearBertRow(IN, OUT)
    m.weight.requires_grad=True
    m.weight.assign(m0.weight)
    m.bias.assign(m0.bias) 

    m1 = FP8LinearBertBasic(IN, OUT)
    m1.weight.requires_grad=True
    m1.weight.assign(m0.weight)
    m1.bias.assign(m0.bias) 

    x = Tensor.rand((BS, 512, IN)).cast(dtypes.half)
    x.requires_grad = True 

    m0_res_ = m0(x).contiguous().contiguous_backward()
    m_res_ = m(x).contiguous().contiguous_backward()
    m1_res_ = m1(x).contiguous().contiguous_backward()
    t0 = time.perf_counter()
    m0_res = m0_res_.numpy()
    ts0.append(time.perf_counter()-t0)
    t0 = time.perf_counter()
    m_res = m_res_.numpy()
    ts.append(time.perf_counter()-t0)
    t0 = time.perf_counter()
    m1_res = m1_res_.numpy()
    t1s.append(time.perf_counter()-t0)

    # print("------m0--------")
    # print(f"{m0_res[0][0]=}")
    # print("------m--------")
    # print(f"{m_res[0][0]=}")
    # print("------m1---------")
    # print(f"{m1_res[0][0]=}")

    #np.testing.assert_allclose(m_res, m0_res, rtol=1e-1, atol=1e-1)
    # print(cosine_similarity(m0_res_, m_res_).numpy())
    # print(cosine_similarity(m0_res_, m1_res_).numpy())

    mse0 = ((m0_res_ - m_res_)**2).mean().numpy()
    mse1 = ((m0_res_ - m1_res_)**2).mean().numpy()

    # print(f"MSE (越小越好) - m_res_: {mse0}")
    # print(f"MSE (越小越好) - m1_res_: {mse1}")
    if mse0 > mse1:
      M1 += 1 
    if mse0 < mse1:
      M += 1 
  print(sum(ts0)/len(ts0))
  print(sum(ts)/len(ts))
  print(sum(t1s)/len(t1s))
  print(f"{M=}, {M1=}")

  # 1024, 4096, 1024 *100
  # 1.060070407189196
  # 1.0604561431065667
  # 1.0602086302207316
  # M=63, M1=34

  # 128, 128, 6 * 1000
  # 0.0032716629666974767
  # 0.003251897276728414
  # 0.002772005642298609
  # M=727, M1=270

if __name__ == "__main__":
  Tensor.manual_seed(42)
  IN, OUT = 1024, 4096
  BS = 1024
  BS = 66
  # IN, OUT = 128, 128
  # BS= 6
  FP8 = getenv("FP8", 0)
  m0 = LinearBert(IN, OUT)
  m0.bias.assign(Tensor.randn(OUT).cast(dtypes.half).realize())
  m0.weight.requires_grad = True

  m = FP8LinearBertRow(IN, OUT)
  m.weight.requires_grad=True
  m.weight.assign(m0.weight)
  m.bias.assign(m0.bias) 

  # m1 = FP8LinearBertBasic(IN, OUT)
  # m1.weight.requires_grad=True
  # m1.weight.assign(m0.weight)
  # m1.bias.assign(m0.bias) 

  x = Tensor.rand((BS, 512, IN)).cast(dtypes.half)
  x.requires_grad = True 

  # m0_res_ = m0(x).contiguous().contiguous_backward()
  m_res_ = m(x).contiguous().contiguous_backward()
  # m1_res_ = m1(x).contiguous().contiguous_backward()
  # t0 = time.perf_counter()
  # m0_res = m0_res_.numpy()
  # ts0.append(time.perf_counter()-t0)
  # t0 = time.perf_counter()
  m_res = m_res_.numpy()
  # ts.append(time.perf_counter()-t0)
  # t0 = time.perf_counter()
  # m1_res = m1_res_.numpy()
  # t1s.append(time.perf_counter()-t0)