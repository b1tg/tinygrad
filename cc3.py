import numpy as np
from tinygrad import Tensor, dtypes, nn
from tinygrad.helpers import Timing

# ==========================================
# 1. 模拟 FP8 量化函数
# ==========================================
def simulate_fp8_quant(x: Tensor, group_shape: tuple, axis: tuple):
    """
    模拟量化过程：
    1. Reshape 成 group 形式
    2. 计算 Max Abs Scale
    3. Clamp 到 FP8 范围 (E4M3 约为 +/- 448)
    4. 返回量化后的整数模拟值和反量化 Scale
    """
    # 1. Reshape 用于计算 Scale
    x_grouped = x.reshape(group_shape)
    
    # 2. 计算 Scale (Scale = 448 / max_abs)
    # axis 是在 grouped shape 中计算 max 的维度
    max_val = x_grouped.abs().max(axis=axis, keepdim=True)
    scale = 448.0 / (max_val + 1e-8)
    
    # 3. 量化 (Quantize) -> 模拟变为整数
    # x_grouped * scale 也就是把数值拉伸到 [-448, 448]
    x_quant = (x_grouped * scale).round().clamp(-448.0, 448.0)
    # x_quant = (x_grouped * scale).clamp(-448.0, 448.0)
    
    # 4. 准备反量化 Scale (Inv Scale)
    scale_inv = 1.0 / scale
    
    return x_quant, scale_inv

# ==========================================
# 2. DeepSeek 风格的 Linear Layer
# ==========================================
class DeepSeekFP8Linear:
    def __init__(self, in_features, out_features, block_size=128):
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        
        # 初始化权重 (FP32)
        # 实际训练中，权重会以 FP8 存储，这里为了演示保留 FP32 源
        self.weight = Tensor.kaiming_uniform(out_features, in_features)
        
    def __call__(self, x: Tensor):
        print(f"{x.shape=}, {self.weight.shape=}")
        # x shape: [Batch, Seq, In]
        B, S, IN = x.shape
        OUT = self.out_features
        BLOCK = self.block_size
        assert IN % BLOCK == 0, "Input dimension must be divisible by block size"
        assert OUT % BLOCK == 0, "Output dimension must be divisible by block size"
        
        num_groups = IN // BLOCK # 4096 // 128 = 32
        
        # 初始化累加器 (FP32 Accumulation)
        output = Tensor.zeros(B, S, OUT)
        
        # === 核心逻辑：分块计算 (Split-K) ===
        # DeepSeek 方法需要在内积维度(Inner Dim)上应用不同的 Scale。
        # 最内存友好的方式是循环遍历 Group，计算 partial result 并累加。
        # 这样避免了分配巨大的中间 Tensor (Batch, Seq, Num_Groups, Out)。
        
        for g in range(num_groups):
            # 1. 切片 (Slicing) - 取出当前的 128 个通道
            start_idx = g * BLOCK
            end_idx = (g + 1) * BLOCK
            
            # x_slice: [Batch, Seq, 128]
            x_slice = x[:, :, start_idx:end_idx]
            # w_slice: [Out, 128]
            w_slice = self.weight[:, start_idx:end_idx]
            
            # 2. Activation 量化 (Tile-wise: 1x128)
            # 对每个 Token 的当前 128 维度计算 Scale
            # Group Shape 实际上就是 Slice 本身，axis=-1
            x_q, x_scale_inv = simulate_fp8_quant(
                x_slice, 
                group_shape=(B, S, BLOCK), 
                axis=-1 
            ) # x_scale_inv: [B, S, 1]
            
            # 3. Weight 量化 (Block-wise: 128x128)
            # 需要将 Out 维度也切分成 128 的块来计算独立的 Scale
            # W_slice shape: [Out, 128] -> Reshape [Out//128, 128, 128]
            out_blocks = OUT // BLOCK
            w_q_grouped, w_scale_inv_grouped = simulate_fp8_quant(
                w_slice,
                group_shape=(out_blocks, BLOCK, BLOCK),
                axis=(1, 2) # 对整个 128x128 块求一个 Scale
            )
            # w_q_grouped: [Out/128, 128, 128]
            # w_scale_inv_grouped: [Out/128, 1, 1]
            
            # 还原 w_q 的形状用于矩阵乘法: [Out, 128]
            w_q = w_q_grouped.reshape(OUT, BLOCK)
            
            # 扩展 w_scale 以匹配输出维度: [Out, 1] (每个输出元素对应的 Scale)
            # 这里需要注意广播：Scale 是以 128 为单位跳变的
            w_scale_inv = w_scale_inv_grouped.expand(out_blocks, BLOCK, 1).reshape(OUT, 1)

            # 4. 执行 FP8 矩阵乘法 (Simulated)
            # [Batch, Seq, 128] @ [128, Out] -> [Batch, Seq, Out]
            # 这里的 x_q 和 w_q 实际上存储的是 [-448, 448] 的整数模拟值
            partial_sum_int = x_q.dot(w_q.T, dtype=dtypes.float32)
            
            # 5. 反量化并累加 (Dequantize & Accumulate)
            # Formula: Y += (X_int @ W_int.T) * Scale_X * Scale_W
            # Scale_X: [B, S, 1], Scale_W: [Out, 1] -> Transpose to [1, Out]
            partial_sum_float = partial_sum_int * x_scale_inv * w_scale_inv.T
            
            output = output + partial_sum_float

        return output

# ==========================================
# 3. 运行对比测试
# ==========================================
if __name__ == "__main__":
    # 定义维度
    # 注意：为了演示能在普通 GPU/CPU 运行，我减小了 Batch Size。
    # 如果你有 H100 或者要在集群运行，可以改回 1024
    REAL_BS = 1024
    TEST_BS = 2      # 仅用于演示运行，防止 OOM
    TEST_BS = 1024
    SEQ = 512
    IN_DIM = 4096
    OUT_DIM = 1024
    
    print(f"创建 Tensors... (Input Shape: {TEST_BS}x{SEQ}x{IN_DIM})")
    
    # 随机输入和权重
    x = Tensor.randn(TEST_BS, SEQ, IN_DIM)
    
    # 1. 实例化 DeepSeek 风格 FP8 Linear
    ds_layer = DeepSeekFP8Linear(IN_DIM, OUT_DIM)
    
    # 2. 实例化标准 FP32 Linear (作为基准 Ground Truth)
    # 我们手动把 ds_layer 的权重复制过来，保证权重一致
    std_layer = nn.Linear(IN_DIM, OUT_DIM, bias=False)
    std_layer.weight.assign(ds_layer.weight) # 复制权重
    
    print("开始计算...")
    
    # --- 运行 DeepSeek FP8 ---
    with Timing("DeepSeek FP8 Forward: "):
        y_ds = ds_layer(x).realize()
        
    # --- 运行 Standard FP32 ---
    with Timing("Standard FP32 Forward: "):
        y_std = std_layer(x).realize()
        
    # --- 验证结果 ---
    # 由于量化不仅引入了精度损失，还引入了 Clipping，结果不会完全一致。
    # 我们检查 Cosine Similarity 和 相对误差。
    
    # 1. 计算均方误差 (MSE)
    diff = (y_ds - y_std)
    mse = (diff * diff).mean().numpy()
    # mse.realize
    
    # 2. 计算余弦相似度 (Cosine Similarity) - 验证方向一致性
    # Flatten to [N, D]
    # print(y_ds.numpy())
    # print("---")
    # print(y_std.numpy())
    # y_ds_flat = y_ds.reshape(-1)
    # y_std_flat = y_std.reshape(-1)
    # cos_sim = (y_ds_flat * y_std_flat).sum() / (y_ds_flat.normal() * y_std_flat.normal())
    
    # print("\n=== 结果对比 ===")
    # print(f"Input Shape: ({TEST_BS}, {SEQ}, {IN_DIM})")
    # print(f"Output Shape: {y_ds.shape}")
    # print(f"MSE Loss (Quantization Error): {mse:.6f}")
    # print(f"Cosine Similarity: {cos_sim.numpy():.6f}")
    
    # print("\n如果 Cosine Similarity > 0.99，说明细粒度量化逻辑正确且精度损失在可接受范围内。")