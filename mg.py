from tinygrad import Tensor, dtypes
dtypes.default_float = dtypes.float16
from tinygrad.dtype import to_dtype
from tinygrad.helpers import getenv

# 选择你要使用的多个 GPU
DEVICES = ["AMD:0", "AMD:1",  "AMD:2",  "AMD:3"]   # 改这里即可

if __name__ == "__main__":
  BS = getenv("BS", 96//6)
  acc_dtype = to_dtype(getenv("ACC_DTYPE", "half"))

  tensors = [
    (Tensor.empty(BS, 512, 1024), Tensor.empty(1024, 1024).T),                                          # qkv
    (Tensor.empty(BS, 512, 16, 64).permute(0,2,1,3), Tensor.empty(BS, 512, 16, 64).permute(0,2,3,1)),   # q@k
    (Tensor.empty(BS, 16, 512, 512), Tensor.empty(BS, 512, 16, 64).permute(0,2,1,3)),                   # qk@v
  ]

  for t0, t1 in tensors:
    print(f"{t0.shape=}, {t1.shape=}")

    # --- 🔥关键：把 t0 shard 在 batch 维度到多个 GPU ---
    t0_sharded = t0.shard(DEVICES, axis=0)

    # --- 🔥关键：将 t1 broadcast 复制到所有 GPU ---
    t1_replicated = t1.shard(DEVICES, axis=None)

    # --- 🔥多 GPU matmul ---
    for _ in range(20):
      (t0_sharded.dot(t1_replicated, dtype=acc_dtype)).realize()
