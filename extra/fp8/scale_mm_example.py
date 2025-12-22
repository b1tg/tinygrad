
# https://gist.github.com/malfet/7874d96b99670c3da83cbb779ab770c6


import torch
import torch.nn.functional as F
from tinygrad import Tensor, dtypes, Context
from tinygrad.dtype import DType
from tinygrad.helpers import getenv

def to_float8_torch(x, dtype=torch.float8_e4m3fn):
  finfo = torch.finfo(dtype)
  # Calculate the scale as dtype max divided by absmax
  scale = finfo.max / x.abs().max().clamp(min=1e-12)
  # scale and clamp the tensor to bring it to
  # the representative range of float8 data type
  # (as default cast is unsaturated)
  x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
  # Return both float8 data and the inverse scale (as float),
  # as both required as inputs to torch._scaled_mm
  return x_scl_sat.to(dtype), scale.float().reciprocal()

def to_float8(x: Tensor, dtype=dtypes.fp8e4m3):
  fp8_min = -448.0 if dtype == dtypes.fp8e4m3 else -57344.0
  fp8_max = 448.0 if dtype == dtypes.fp8e4m3 else 57344.0
  scale = fp8_max / x.abs().max()
  x_scl_sat = (x * scale).clamp(fp8_min, fp8_max)
  return x_scl_sat.cast(dtype), scale.float().reciprocal()  
def _scaled_mm(x: Tensor, w: Tensor, out_dtype: DType=dtypes.float16, scale_a: Tensor=Tensor(1.0), scale_b: Tensor=Tensor(1.0)):
  # TODO: check BEAM
  if (not getenv("BEAM")) or x.shape[-1] < 1024:
    y1 = x.cast(dtypes.float).dot(w.cast(dtypes.float), dtype=dtypes.float32) * scale_a * scale_b
  else:
    y1 = x.dot(w, dtype=dtypes.float32) * scale_a * scale_b
  return y1.cast(out_dtype)
def compare_f8_mm_torch(size=(16, 16), dtype=torch.float8_e4m3fn) -> None:
  # create test inputs
  # Note: cuBLASLt float8 matmul requires column major
  #        for the second argument
  torch.manual_seed(42)
  x = torch.randn (size, dtype=torch.float16)
  w = torch.randn (size, dtype=torch.float16).t()

  # do a scaled cast to float8 on the inputs
  x_f8, x_inv_s = to_float8_torch(x, dtype=dtype)
  w_f8, w_inv_s = to_float8_torch(w)

  x1 = Tensor(x.numpy())
  w1 = Tensor(w.numpy())
  x1_f8, x1_inv_s = to_float8(x1)
  w1_f8, w1_inv_s = to_float8(w1)

  print(f"{x.numpy()=}")
  print(f"{x1.numpy()=}")
  print(f"{x_f8.float().numpy()=}")
  print(f"{x1_f8.numpy()=}")

  print(f"{x_inv_s.numpy()=}")
  print(f"{w_inv_s.numpy()=}")
  print(f"{x1_inv_s.numpy()=}")
  print(f"{w1_inv_s.numpy()=}")

  # y1 = x1_f8.dot(w1_f8, dtype=dtypes.float32) * x1_inv_s * w1_inv_s
  y1 = _scaled_mm(x1_f8, w1_f8, dtypes.float16, x1_inv_s, w1_inv_s)

  # perform the float8 matmul
  y = torch._scaled_mm(x_f8, w_f8, out_dtype=torch.float16,
                            scale_a=x_inv_s , scale_b=w_inv_s)
  y1 = torch.tensor(y1.numpy())
  print(f"{y.numpy()=}, {y.dtype=}")
  print(f"{y1.numpy()=} {y1.dtype=}")
  # compare output of float8 matmul to the fp16 baseline
  cos_sim = F.cosine_similarity(torch.mm(x, w).reshape(-1),
                                y.reshape(-1), dim=0)
  cos_sim1 = F.cosine_similarity(torch.mm(x, w).reshape(-1),
                                y1.reshape(-1), dim=0)
  # Cosine similarity between scaled mm and reference
  # should be close to 1.0
  print(f"--- compare_f8_mm_torch {size} ---")
  print(f'cos_sim {cos_sim.item():.4f}')
  print(f'cos_sim1 {cos_sim1.item():.4f}')



if __name__ == "__main__":
  compare_f8_mm_torch()
  # compare_f8_mm_torch((1024, 1024))