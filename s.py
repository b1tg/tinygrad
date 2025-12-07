

from tinygrad import Tensor, nn, dtypes
# from examples.mlperf.initializers import FP8LinearBert
def quantize_to_fp8(x: Tensor, dtype=dtypes.fp8e4m3, pre_scale=None):
  fp8_min = -448.0 if dtype == dtypes.fp8e4m3 else -57344.0
  fp8_max = 448.0 if dtype == dtypes.fp8e4m3 else 57344.0
  # fp8_max = fp8_max*0.9
  # fp8_min = fp8_min*0.9
  x_abs_max = x.abs().max()
  # print(f"{x_abs_max.numpy()=}")
  # x_abs_max = stats_amax(x) # 1626.88 ms AMD,  7.57 loss
  # scale = x_abs_max /fp8_max
  scale = fp8_max  / x_abs_max

  # print(f"quantize_to_fp8: {scale.numpy()=}")
  x_scl_sat = (x * scale).clamp(fp8_min, fp8_max)
#   x_scl_sat = (x * scale).nround()
  res= x_scl_sat.cast(dtype)
  # print("q:", x.numpy(), res.numpy())
  return res, scale.float().reciprocal()
class LinearNet:
  def __init__(self):
    self.l1 = Tensor.kaiming_uniform(784, 128)
    self.l2 = Tensor.kaiming_uniform(128, 10)
  def __call__(self, x:Tensor) -> Tensor:
    x1, w0 = quantize_to_fp8(x)
    l1, w1 = quantize_to_fp8(self.l1)
    # print("x1: ", x1.shape, x1.flatten(1).shape)
    # print("l1: ", l1.shape)
#     x1:  (4, 1, 28, 28) (4, 784)
# l1:  (784, 128)
    # l2, w2 = quantize_to_fp8(self.l2)
    return (x1.flatten(1).dot(l1,dtype=dtypes.float)*w0*w1).relu().dot(self.l2)

model = LinearNet()
optim = nn.optim.Adam([model.l1, model.l2], lr=0.111)

x, y = Tensor.rand(4, 1, 28, 28), Tensor([2,4,3,7])  # replace with real mnist dataloader

with Tensor.train():
  for i in range(20):
    optim.zero_grad()
    loss = model(x).sparse_categorical_crossentropy(y).backward()
    optim.step()
    print(i, loss.item())