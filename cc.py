import math
import numpy as np
from tinygrad import Tensor, dtypes

ACT_TILE = 128
WT_O = 128
WT_I = 128
FP8_MAX = 448.0

def quantize_blockwise(x, block_size, dtype):
    last = x.shape[-1]
    assert last % block_size == 0
    groups = last // block_size

    x_rb = x.reshape(*x.shape[:-1], groups, block_size)
    x_absmax = x_rb.abs().max(axis=-1, keepdim=True)

    scale = FP8_MAX / (x_absmax + 1e-8)
    x_scaled = x_rb * scale
    x_clamped = x_scaled.clamp(-FP8_MAX, FP8_MAX)

    x_fp8 = x_clamped.cast(dtype).reshape(*x.shape)
    inv_scale = scale.squeeze(-1)**-1   # shape (..., groups)

    return x_fp8, inv_scale


def quantize_weight_block(W, to, ti, dtype):
    O, I = W.shape
    assert O % to == 0
    assert I % ti == 0

    BO = O // to
    BI = I // ti

    Wb = W.reshape(BO, to, BI, ti)
    W_absmax = Wb.abs().max(axis=(1,3), keepdim=True)   # (BO,1,BI,1)

    scale = FP8_MAX / (W_absmax + 1e-8)
    W_scaled = Wb * scale
    W_clamped = W_scaled.clamp(-FP8_MAX, FP8_MAX)

    W_fp8 = W_clamped.cast(dtype).reshape(O, I)
    inv_scale = scale.squeeze(1).squeeze(-1)**-1      # (BO, BI)

    return W_fp8, inv_scale


class FP8LinearDeepSeek:
    def __init__(self, in_features, out_features, bias=True, dtype=dtypes.fp8e4m3):
        self.inf = in_features
        self.outf = out_features
        self.dtype = dtype

        self.weight = Tensor.randn(out_features, in_features)
        self.bias = Tensor.randn(out_features) if bias else None

    def __call__(self, x:Tensor):
        B,S,C = x.shape

        # 1) activation FGQ: 1×128 tiles
        x_fp8, x_inv = quantize_blockwise(x, ACT_TILE, self.dtype)
        # x_inv: (B,S,32)

        # 2) weight FGQ: 128×128 blocks
        W_fp8, W_inv = quantize_weight_block(self.weight,
                                             WT_O, WT_I,
                                             self.dtype)
        # W_inv: (8, 32)

        # 3) FP8 GEMM
        xf = x_fp8.reshape(B*S, C)
        y = xf.dot(W_fp8.T, dtype=dtypes.float32)   # (B*S,1024)

        # 4) Apply correct FGQ scaling per output channel
        # -----------------------------------------------------
        # For each out channel o:
        #  out_tile = o // 128   (0..7)
        #  For the dot product:
        #     sum over in tiles k (0..31)
        #     each tile contributes x_inv[b,s,k] * W_inv[out_tile,k]
        #
        # So we precompute scale_per_out[b*s, o]
        # -----------------------------------------------------

        x_inv_flat = x_inv.reshape(B*S, C//ACT_TILE)     # (BS, 32)

        # build output scale matrix:
        # For each out channel:
        #   o → o_tile = o//128
        o_tiles = np.arange(self.outf)//WT_O     # shape (1024,)
        o_tiles = Tensor(o_tiles, dtype=dtypes.int32)

        W_inv_T = (W_inv.T)                # (32,8)

        sc = x_inv_flat @ W_inv_T[:, o_tiles]    # shape (BS,1024)

        y = y * sc

        y = y.reshape(B,S,self.outf)

        if self.bias is not None:
            y = y + self.bias.cast(y.dtype)
        return y

def quantize_to_fp8(x: Tensor, axis=None, dtype=dtypes.fp8e4m3):
  if axis is None:
      x_abs_max = x.abs().max()
  else:
      x_abs_max = x.abs().max(axis=axis, keepdim=True)
  # scale = fp8_max / max_val  
  scale = 448. / (x_abs_max + 1e-8)  
  
  x = x*scale

 
  y = x.clamp(-448.0, 448.0)
  res= y.cast(dtype)
  return res, scale.float().reciprocal()

class FP8LinearBert:
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
# --------------------------------------------------------
# TEST
# --------------------------------------------------------

def test_fp8_linear():
    B,S,C = 1, 512, 4096
    O = 1024
    print("Creating test...")

    x = Tensor.randn(B,S,C)
    W = Tensor.randn(O,C)
    b = Tensor.randn(O)

    # fp32 baseline
    y32 = x.reshape(B*S,C).dot(W.T) + b
    y32 = y32.reshape(B,S,O)
    print("1")

    # FP8 FGQ layer
    lin = FP8LinearDeepSeek(C,O)
    lin.weight = W
    lin.bias = b
    print("2")

    y8 = lin(x)
    print("3")

    print(y8.numpy())
    print("---")
    print(y32.numpy())
    diff = (y8 - y32).abs()
    print("Max error:", diff.max().numpy())
    print("Mean error:", diff.mean().numpy())
    print("Rel error:", (diff/(y32.abs()+1e-6)).mean().numpy())


if __name__ == "__main__":
    test_fp8_linear()
