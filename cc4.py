import numpy as np
from tinygrad import Tensor, dtypes, nn
from tinygrad.helpers import Timing
from  examples.mlperf.initializers import FP8LinearBert, LinearBert, FP8LinearBertRow, FP8LinearBert

# ==========================================
# 1. 模拟 FP8 量化函数 (保持不变)
# ==========================================
def simulate_fp8_quant(x: Tensor, group_shape: tuple, axis: tuple):
    """
    模拟量化过程：Reshape -> Max Scale -> Round & Clamp -> Dequant Scale
    """
    x_grouped = x.reshape(group_shape)
    max_val = x_grouped.abs().max(axis=axis, keepdim=True)
    scale = 448.0 / (max_val + 1e-8)
    
    # 注意：加上 .round() 是为了模拟离散化带来的精度损失
    x_quant = (x_grouped * scale).round().clamp(-448.0, 448.0)
    
    scale_inv = 1.0 / scale
    return x_quant, scale_inv

# ==========================================
# 2. DeepSeek 风格的 Linear Layer (支持 Bias)
# ==========================================
class DeepSeekFP8Linear:
    def __init__(self, in_features, out_features, bias=True, block_size=128):
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        
        # 1. 初始化权重
        self.weight = Tensor.kaiming_uniform(out_features, in_features)
        
        # 2. 初始化 Bias (新增)
        if bias:
            # 通常 Bias 初始化为 0 或很小的值
            self.bias = Tensor.zeros(out_features)
        else:
            self.bias = None
        
    def __call__(self, x: Tensor):
        # x shape: [Batch, Seq, In]
        B, S, IN = x.shape
        OUT = self.out_features
        BLOCK = self.block_size
        
        # 检查维度是否对齐
        if IN % BLOCK != 0 or OUT % BLOCK != 0:
            raise ValueError(f"Dimensions must be divisible by block_size {BLOCK}")
        
        num_groups = IN // BLOCK 
        output = Tensor.zeros(B, S, OUT)
        
        # === 核心：Split-K 累加 (矩阵乘法部分) ===
        for g in range(num_groups):
            start_idx = g * BLOCK
            end_idx = (g + 1) * BLOCK
            
            # 切片
            x_slice = x[:, :, start_idx:end_idx]      # [B, S, 128]
            w_slice = self.weight[:, start_idx:end_idx] # [Out, 128]
            
            # --- Activation 量化 (Per-Tile: 1x128) ---
            x_q, x_scale_inv = simulate_fp8_quant(
                x_slice, 
                group_shape=(B, S, BLOCK), 
                axis=-1 
            )
            
            # --- Weight 量化 (Per-Block: 128x128) ---
            out_blocks = OUT // BLOCK
            w_q_grouped, w_scale_inv_grouped = simulate_fp8_quant(
                w_slice,
                group_shape=(out_blocks, BLOCK, BLOCK),
                axis=(1, 2)
            )
            
            # 调整形状以进行乘法
            w_q = w_q_grouped.reshape(OUT, BLOCK)
            w_scale_inv = w_scale_inv_grouped.expand(out_blocks, BLOCK, 1).reshape(OUT, 1)

            # 模拟 FP8 整数矩阵乘法
            partial_sum_int = x_q.dot(w_q.T)
            
            # 反量化并累加到 FP32
            partial_sum_float = partial_sum_int * x_scale_inv * w_scale_inv.T
            output = output + partial_sum_float

        # === Bias 处理 (新增) ===
        # Bias 加法发生在所有矩阵乘法累加完成之后
        if self.bias is not None:
            # self.bias shape: [Out]
            # output shape:    [B, S, Out]
            # Tinygrad/Numpy 会自动广播最后维度
            output = output + self.bias

        return output

# ==========================================
# 3. 运行对比测试
# ==========================================
if __name__ == "__main__":
    # 配置
    REAL_BS = 1024
    TEST_BS = 1024    
    SEQ = 512
    IN_DIM = 4096
    OUT_DIM = 1024
    
    print(f"Input Shape: {TEST_BS}x{SEQ}x{IN_DIM} (Bias=True)")
    
    x = Tensor.randn(TEST_BS, SEQ, IN_DIM)
    
    # 1. 实例化 DeepSeek Layer (bias=True)
    ds_layer = FP8LinearBert(IN_DIM, OUT_DIM, bias=True)
    
    # 为了测试效果，我们给 bias 赋一些非零的随机值
    ds_layer.bias.assign(Tensor.randn(OUT_DIM))
    # ds_layer.weight.assign(Tensor.randn(OUT_DIM))
    
    # 2. 实例化标准 Linear (bias=True)
    std_layer = LinearBert(IN_DIM, OUT_DIM, bias=True)
    
    # === 关键：同步权重和 Bias ===
    std_layer.weight.assign(ds_layer.weight)
    std_layer.bias.assign(ds_layer.bias) 
    
    print("开始计算...")
    
    # --- 运行 DeepSeek FP8 ---
    with Timing("DeepSeek FP8 Forward: "):
        y_ds = ds_layer(x).realize()
        
    # --- 运行 Standard FP32 ---
    with Timing("Standard FP32 Forward: "):
        y_std = std_layer(x).realize()
    from tinygrad.helpers import DEBUG
    if DEBUG>2:
      print(f"{y_ds.numpy()=}")
      print("---------------")
      print(f"{y_std.numpy()=}")
          
    # 1. 均方误差 (MSE)
    diff = (y_ds - y_std)
    mse = (diff * diff).mean().numpy()
    
    # 2. 余弦相似度
    y_ds_flat = y_ds.reshape(-1)
    y_std_flat = y_std.reshape(-1)
    # cos_sim = (y_ds_flat * y_std_flat).sum() / (y_ds_flat.norm() * y_std_flat.norm())
    
    print("\n=== 结果对比 ===")
    print(f"MSE Loss: {mse:.6f}")
    # print(f"Cosine Similarity: {cos_sim.numpy():.6f}")