from tinygrad import Tensor, dtypes
dtypes.default_float = dtypes.float16
from tinygrad.dtype import to_dtype
from tinygrad.helpers import getenv

if __name__ == "__main__":
  # matmuls in bert layers
  BS = getenv("BS", 96//6)
  BS = 128
  acc_dtype = to_dtype(getenv("ACC_DTYPE", "float"))
  if getenv("FP8",0):
    acc_dtype = to_dtype("float")
    dtypes.default_float = dtypes.fp8e4m3
  # x.shape=(1024, 512, 4096), self.weight.shape=(1024, 4096)
  tensors = [
    # (Tensor.empty(BS, 512, 1024), Tensor.empty(1024, 1024).T),                                          # linear to get qkv
    (Tensor.empty(128, 512, 4096), Tensor.empty(1024, 4096).T),                                          # linear to get qkv
    # (Tensor.empty(BS, 512, 16, 64).permute(0,2,1,3), Tensor.empty(BS, 512, 16, 64).permute(0,2,3,1)),   # q@k
    # (Tensor.empty(BS, 16, 512, 512), Tensor.empty(BS, 512, 16, 64).permute(0,2,1,3)),                   # qk@v
  ]
  for t0, t1 in tensors:
    # print(f"{t0.shape=}, {t0.uop.st.is_expanded()=}, {t1.shape=}, {t1.uop.st.is_expanded()=}")
    print(f"{t0.shape=},{t1.shape=}, ")
    for _ in range(5):
      # t0.dot(t1, dtype=acc_dtype).realize()
      r = t0.dot(t1, dtype=acc_dtype) * 0.9
      r.cast(dtypes.float16).realize()
