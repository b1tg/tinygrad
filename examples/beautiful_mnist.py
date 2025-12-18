# model based off https://medium.com/data-science/going-beyond-99-mnist-handwritten-digits-recognition-cfff96337392
from typing import Callable
from tinygrad import Tensor, TinyJit, nn, GlobalCounters
from tinygrad.helpers import getenv, colored, trange
from tinygrad.nn.datasets import mnist
from tinygrad import Tensor, dtypes, UOp
from tinygrad.uop.ops import KernelInfo, AxisType, Ops

def quantize_to_fp8(x: Tensor, dtype=dtypes.fp8e4m3):
  fp8_min = -448.0 if dtype == dtypes.fp8e4m3 else -57344.0
  fp8_max = 448.0 if dtype == dtypes.fp8e4m3 else 57344.0
  x_abs_max = x.abs().max()
  print(f"{x_abs_max.item()=}")
  scale = fp8_max / (x_abs_max + 1e-8)
  # x_scl_sat = (x * scale).clamp(fp8_min, fp8_max)
  x_scl_sat = (x * scale).nround()
  res= x_scl_sat.cast(dtype)
  return res, scale.float().reciprocal()

def k_clamp_back(grads:UOp, kernel:UOp):
  y, x = kernel.src
  return (None, None)
  return (None, Tensor(grads).uop)
  return (None, Tensor.empty_like(Tensor(x)).uop)
def k_clamp(y:UOp, x:UOp):
  y = y.flatten()
  x = x.flatten()
  i = UOp.range(x.size, 0)
  x1 = x[i].maximum(UOp.const(x.dtype.base, -448.0)).minimum(UOp.const(x.dtype.base, 448.0))
  return y[i].store(x1).end(i).sink(arg=KernelInfo(name=f"k_clamp{x.size}"))
def q_abs_max_kernel(y: UOp, x:UOp):
  B = y.flatten()
  A = x.flatten()
  i = UOp.range(A.shape[0], 0, axis_type=AxisType.REDUCE)
  B = B[0].set(UOp.const(x.dtype.base, 0.0))
  B = B[0].set(B.after(i)[0].maximum((A[i]<0.0).where(A[i]*UOp.const(x.dtype.base, UOp.const(x.dtype.base, -1.0)), A[i])), end=i)
  return B.sink(arg=KernelInfo(name=f"custom_sumx_{A.shape[0]}"))
def q_abs_max_kernel_back(grads:UOp, kernel:UOp):
  y, x = kernel.src
  return (None, None)
  return (None, Tensor.empty_like(Tensor(x)).uop)
  return (None, Tensor(grads).uop)

def quantize_to_fp8(x: Tensor, dtype=dtypes.fp8e4m3, pre_scale=None):
  fp8_min = -448.0 if dtype == dtypes.fp8e4m3 else -57344.0
  fp8_max = 448.0 if dtype == dtypes.fp8e4m3 else 57344.0
  # fp8_max = fp8_max*0.9
  # fp8_min = fp8_min*0.9
  # x_abs_max = x.abs().max()
  # print(f"{x_abs_max.numpy()=}")
  # x_abs_max = stats_amax(x) # 1626.88 ms AMD,  7.57 loss
  # scale = x_abs_max /fp8_max
  # scale = fp8_max  / x_abs_max
  y = Tensor.empty((), dtype=x.dtype)
  y = Tensor.custom_kernel(y, x, fxn=q_abs_max_kernel, grad_fxn=q_abs_max_kernel_back)[0]
  scale = fp8_max / y

  # print(f"quantize_to_fp8: {scale.numpy()=}")
  # x_scl_sat = (x * scale).clamp(fp8_min, fp8_max)

  xs = (x * scale)
  y = Tensor.empty_like(xs)
  y = Tensor.custom_kernel(y, xs, fxn=k_clamp, grad_fxn=k_clamp_back)[0]
  x_scl_sat = y


  # x_scl_sat = (x * scale).nround()
  res= x_scl_sat.cast(dtype)
  # print("q:", x.numpy(), res.numpy())
  return res, scale.float().reciprocal()
class FP8LinearBert:
  def __init__(self, in_features, out_features, bias=True, ste=True):
    self.weight = Tensor.empty(out_features, in_features, dtype=dtypes.float32)
    self.bias = Tensor.empty(out_features, dtype=dtypes.float32) if bias else None
  def __call__(self, x:Tensor):
    w1, ws = quantize_to_fp8(self.weight)
    x1, s = quantize_to_fp8(x)
    # print(x1.shape, w1.T.shape)
    # (512, 576) (576, 10)
    # y = x.dot(w1.T.cast(dtypes.float)).cast(dtypes.float) * ws
    w1 = w1.cast(self.weight.dtype)
    x1 = x1.cast(x.dtype)
    # print(x1.numpy())

    # w1, ws = self.weight, 1.0
    # x1, s = x, 1.0
    y = x1.dot(w1.T, dtype=dtypes.float) * ws * s
    if self.bias is not None: y = y + self.bias.cast(y.dtype)
    return y.cast(x.dtype)

class Model:
  def __init__(self):
    self.conv1 = nn.Conv2d(1, 32, 3, 1)
    self.conv2 = nn.Conv2d(32, 64, 3, 1)
    # self.dropout1 = nn.Dropout(0.25)
    # self.dropout2 = nn.Dropout(0.5)
    if getenv("FP8",0):
        self.fc1 = FP8LinearBert(9216, 128)
        self.fc2 = FP8LinearBert(128, 16)
    else:
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 16)
    self.fc3 = nn.Linear(16, 10)
  def __call__(self, x:Tensor):
      """FWD"""
      x = self.conv1(x)
      x = x.relu()
      x = self.conv2(x)
      x = x.relu()
      x = x.max_pool2d(x)
      x = x.dropout(0.25)
      x = x.flatten(1)
      x = self.fc1(x)
      x = x.relu()
      x = x.dropout(0.5)
      x = self.fc2(x)
      x = self.fc3(x)
      # output = F.log_softmax(x, dim=1)
      output = x.log_softmax(axis=1)
      return output

  #   self.layers: list[Callable[[Tensor], Tensor]] = [
  #     nn.Conv2d(1, 32, 3, 1), Tensor.relu,
  #     nn.Conv2d(32, 64, 3, 1), Tensor.relu,
  #     nn.BatchNorm(32), Tensor.max_pool2d,
  #     nn.Conv2d(32, 64, 3), Tensor.relu,
  #     nn.Conv2d(64, 64, 3), Tensor.relu,
  #     nn.BatchNorm(64), Tensor.max_pool2d,
  #     lambda x: x.flatten(1), FP8LinearBert(576, 10) if getenv("FP8", 0) else nn.Linear(576, 10) ]

  # def __call__(self, x:Tensor) -> Tensor: return x.sequential(self.layers)

if __name__ == "__main__":
  X_train, Y_train, X_test, Y_test = mnist(fashion=getenv("FASHION"))

  model = Model()
  opt = (nn.optim.Muon if getenv("MUON") else nn.optim.SGD if getenv("SGD") else nn.optim.Adam)(nn.state.get_parameters(model))

  @TinyJit
  @Tensor.train()
  def train_step() -> Tensor:
    opt.zero_grad()
    samples = Tensor.randint(getenv("BS", 512), high=X_train.shape[0])
    loss = model(X_train[samples]).sparse_categorical_crossentropy(Y_train[samples]).backward()
    return loss.realize(*opt.schedule_step())

  @TinyJit
  def get_test_acc() -> Tensor: return (model(X_test).argmax(axis=1) == Y_test).mean()*100

  test_acc = float('nan')
  for i in (t:=trange(getenv("STEPS", 70))):
    GlobalCounters.reset()   # NOTE: this makes it nice for DEBUG=2 timing
    loss = train_step()
    if i%10 == 9: test_acc = get_test_acc().item()
    t.set_description(f"loss: {loss.item():6.2f} test_accuracy: {test_acc:5.2f}%")

  # verify eval acc
  if target := getenv("TARGET_EVAL_ACC_PCT", 0.0):
    if test_acc >= target and test_acc != 100.0: print(colored(f"{test_acc=} >= {target}", "green"))
    else: raise ValueError(colored(f"{test_acc=} < {target}", "red"))
