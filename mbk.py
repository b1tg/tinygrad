from tinygrad import Tensor, dtypes, nn
from tinygrad.helpers import getenv
from tinygrad import Tensor, dtypes, UOp
from tinygrad.uop.ops import KernelInfo, AxisType, Ops
from tinygrad.helpers import prod, argfix, Context
from tinygrad.helpers import Timing
import numpy as np
Tensor.manual_seed(42)
GPUS = ('AMD', 'AMD:1')
GPUS= ('CPU', 'CPU:1')

def custom_linear(C:UOp, A:UOp, B:UOp) -> UOp:
  return UOp.sink().simplify()
def custom_linear_backward(gradient:UOp, kernel:UOp) -> tuple[UOp, UOp]:
  out, a, b = kernel.src
  return None, None, None
class FP8LinearBertBasic:
  def __init__(self, in_features, out_features, bias=True):
    self.weight = Tensor.rand(out_features, in_features, dtype=dtypes.float16)
    self.bias = Tensor.rand(out_features, dtype=dtypes.float16) if bias else None
  def __call__(self, x:Tensor):
    devs = GPUS
    y = Tensor(Tensor.empty((x.shape[0], x.shape[1], self.weight.shape[0]), dtype=dtypes.float, device=devs).uop.multi(0), device=devs)
    if getenv("CUSTOM", 1) == 0:
      y = x.dot(self.weight.T, dtype=dtypes.float).contiguous().contiguous_backward()
    else:
      y = Tensor.custom_kernel(y, x, self.weight, fxn=custom_linear,grad_fxn=custom_linear_backward )[0].contiguous().contiguous_backward()
    # if self.bias is not None: y = y + self.bias.cast(y.dtype)
    return y.cast(x.dtype)
  
if __name__ == "__main__":
  IN = 4096
  OUT = 1024
  BS = 66
  X = 512
  IN = 8
  OUT = 4
  BS = 2
  X = 1
  m = FP8LinearBertBasic(IN, OUT)
  m.weight.requires_grad = True
  m.weight.to_(GPUS)
  m.bias.to_(GPUS)
  x = Tensor.rand((BS, X, IN)).cast(dtypes.half)
  x.shard_(GPUS, axis=0)
  print(f"{x.device=}, {m.weight.device=}")
  print(m(x).numpy())