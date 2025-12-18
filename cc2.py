# 修正版：正确实现 DeepSeek FGQ 的 tinygrad 验证例子
import numpy as np
from tinygrad import Tensor, dtypes

ACT_TILE = 128
WT_O = 128
WT_I = 128
FP8_MAX = 448.0

def quantize_blockwise(x: Tensor, block_size: int, dtype):
    last = x.shape[-1]
    assert last % block_size == 0
    groups = last // block_size
    # reshape to (..., groups, block_size)
    x_rb = x.reshape(*x.shape[:-1], groups, block_size)
    x_absmax = x_rb.abs().max(axis=-1, keepdim=True)  # (..., groups, 1)
    scale = FP8_MAX / (x_absmax + 1e-8)                # (..., groups, 1)
    x_scaled = x_rb * scale
    x_clamped = x_scaled.clamp(-FP8_MAX, FP8_MAX)
    x_fp8 = x_clamped.cast(dtype).reshape(*x.shape)
    inv_scale = scale.squeeze(-1) ** -1                # (..., groups)
    return x_fp8, inv_scale

def quantize_weight_block(W: Tensor, to: int, ti: int, dtype):
    O, I = W.shape
    assert O % to == 0 and I % ti == 0
    BO = O // to
    BI = I // ti
    Wb = W.reshape(BO, to, BI, ti)                     # (BO, to, BI, ti)
    W_absmax = Wb.abs().max(axis=(1,3), keepdim=True)  # (BO,1,BI,1)
    scale = FP8_MAX / (W_absmax + 1e-8)                # (BO,1,BI,1)
    W_scaled = Wb * scale
    W_clamped = W_scaled.clamp(-FP8_MAX, FP8_MAX)
    W_fp8 = W_clamped.cast(dtype).reshape(O, I)
    inv_scale = scale.squeeze(1).squeeze(-1) ** -1     # (BO, BI) Tensor
    return W_fp8, inv_scale

class FP8LinearDeepSeek_Correct:
    def __init__(self, in_features, out_features, bias=True, dtype=dtypes.fp8e4m3):
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype
        self.weight = Tensor.randn(out_features, in_features)
        self.bias = Tensor.randn(out_features) if bias else None

    def __call__(self, x: Tensor):
        B, S, C = x.shape                 # e.g. (1024,512,4096)
        # 1) activation FGQ (per token, per 128)
        x_fp8, x_inv = quantize_blockwise(x, ACT_TILE, self.dtype)   # x_inv: (B,S,32)
        # 2) weight FGQ (128x128 blocks)
        W_fp8, W_inv = quantize_weight_block(self.weight, WT_O, WT_I, self.dtype)  # W_inv: (8,32)
        # 3) accumulate partials per in-tile k
        BS = B * S
        xf = x_fp8.reshape(BS, C)        # (BS,4096)
        # prepare W blocks: for each k, Wk = W_fp8[:, k*128:(k+1)*128]  -> shape (Out,128)
        num_in_tiles = C // ACT_TILE     # 32
        # Prepare W_inv_per_out: expand W_inv (8,32) -> (Out,32) mapping each out channel to its out_tile row
        W_inv_np = W_inv.numpy()         # (8,32)
        # repeat each row WT_O times along out dimension (BO rows -> BO*WT_O == Out)
        W_inv_per_out_np = np.repeat(W_inv_np, WT_O, axis=0)   # (Out,32)
        W_inv_per_out = Tensor(W_inv_per_out_np)               # (Out,32) as Tensor

        # x_inv flatten: (B,S,32) -> (BS,32)
        x_inv_flat = x_inv.reshape(BS, num_in_tiles)           # Tensor (BS,32)

        # accumulate result y (BS, Out)
        y_acc = None
        # Loop over k tiles (can be vectorized but loop is clear and safe)
        for k in range(num_in_tiles):
            ks = k * ACT_TILE
            ke = ks + ACT_TILE
            # partial contribution using FP8 stored values:
            # xf[:, ks:ke] shape (BS, 128)
            # W_fp8[:, ks:ke] shape (Out, 128)
            # partial_k = xf_k dot Wk.T  -> (BS, Out)
            xf_k = xf[:, ks:ke]                                # (BS,128)
            Wk = W_fp8[:, ks:ke]                               # (Out,128)
            partial_k = xf_k.dot(Wk.T, dtype=dtypes.float32)   # (BS,Out)  (accumulate in fp32)

            # get scale factors: per-batch * per-out
            # x_inv_flat[:, k] -> (BS,)  -> (BS,1)
            xinv_k = x_inv_flat[:, k].reshape(BS, 1)           # (BS,1)
            # W_inv_per_out[:, k] -> (Out,) -> (1,Out)
            winv_k = W_inv_per_out[:, k].reshape(1, self.out_features)  # (1,Out)

            # multiply partial by outer product -> (BS,Out)
            scaled_partial = partial_k * xinv_k * winv_k

            if y_acc is None:
                y_acc = scaled_partial
            else:
                y_acc = y_acc + scaled_partial

        # reshape back to (B,S,Out)
        y = y_acc.reshape(B, S, self.out_features)
        if self.bias is not None:
            y = y + self.bias.cast(y.dtype)
        return y

# ------------------ Test harness ------------------

def test_fp8_linear_correct():
    B,S,C = 64, 8, 4096   # 用小一点的尺寸先测试，运行更快；把 B,S 改回 1024,512 做完整测试
    O = 1024
    print("Building random inputs...")
    x = Tensor.randn(B, S, C)
    W = Tensor.randn(O, C)
    b = Tensor.randn(O)
    # FP32 baseline
    y_fp32 = x.reshape(B*S, C).dot(W.T) + b
    y_fp32 = y_fp32.reshape(B, S, O)
    # FGQ FP8 layer (corrected)
    layer = FP8LinearDeepSeek_Correct(C, O)
    layer.weight = W
    layer.bias = b
    y_fp8 = layer(x)
    diff = (y_fp8 - y_fp32)
    print(y_fp8.numpy())
    print("---")
    print(y_fp32.numpy())
    # print("Max abs err:", diff.abs().max().numpy())
    # print("Mean abs err:", diff.abs().mean().numpy())
    # print("Mean rel err:", (diff.abs()/(y_fp32.abs()+1e-6)).mean().numpy())

if __name__:
    test_fp8_linear_correct()
