import math

# --- 配置：清晰定义四种 FP8 格式的所有差异 ---
# 键格式为: <规范>_<位宽>
CONFIG = {
    # OCP (Open Compute Project) 规范
    "ocp_e4m3": {
        "exp_bits": 4, "mantissa_bits": 3, "bias": 7,
        "max_val": 448.0, "has_inf": False, "has_signed_zero": True,
        "nan": 0x7F, "max_norm": 0x7E
    },
    "ocp_e5m2": {
        "exp_bits": 5, "mantissa_bits": 2, "bias": 15,
        "max_val": 57344.0, "has_inf": True, "has_signed_zero": True,
        "nan": 0x7F, "inf": 0x7C, "max_norm": 0x7B
    },
    # FNUZ (Finite No-unsigned-inf, signed Zero) 规范
    "fnuz_e4m3": {
        "exp_bits": 4, "mantissa_bits": 3, "bias": 8,
        "max_val": 240.0, "has_inf": False, "has_signed_zero": False,
        "nan": 0x80, "max_norm": 0x7E # Note: FNUZ e4m3's max_norm bits are same as OCP
    },
    "fnuz_e5m2": {
        "exp_bits": 5, "mantissa_bits": 2, "bias": 16,
        "max_val": 57344.0, "has_inf": False, "has_signed_zero": False,
        "nan": 0x80, "max_norm": 0x7B # Note: FNUZ e5m2's max_norm bits are same as OCP
    },
}

def float_to_fp8(x: float, dtype: str) -> int:
    """
    将浮点数转换为 FP8 格式。

    Args:
        x (float): 输入的浮点数。
        dtype (str): 目标 FP8 格式，必须是 'ocp_e4m3', 'ocp_e5m2', 'fnuz_e4m3', 'fnuz_e5m2' 之一。

    Returns:
        int: 转换后的 8 位整数表示。
    """
    if dtype not in CONFIG:
        raise ValueError(f"Unsupported dtype: {dtype}. Must be one of {list(CONFIG.keys())}")

    cfg = CONFIG[dtype]
    wm = cfg["mantissa_bits"]
    
    # 1. 处理特殊值
    if math.isnan(x):
        return cfg["nan"]
    if math.isinf(x):
        return cfg["nan"] if not cfg["has_inf"] else (cfg["inf"] if x > 0 else cfg["inf"] | 0x80)
    if x == 0.0:
        return 0 if not cfg["has_signed_zero"] else (0 if x > 0 else 0x80)

    # 2. 分解符号、尾数和指数
    sign_bit = 0 if x > 0 else 0x80
    abs_x = abs(x)
    mantissa, exponent = math.frexp(abs_x)

    # 3. 计算目标 FP8 指数
    fp8_exp = exponent + cfg["bias"] - 1
    max_exp_val = (1 << cfg["exp_bits"]) - 1

    # 4. 根据指数范围处理不同情况
    if abs_x > cfg["max_val"]:
        # 上溢：钳位到最大规格化数或返回 NaN/Inf
        if cfg["has_inf"]: return sign_bit | cfg["inf"]
        else: return sign_bit | cfg["max_norm"]
        
    elif fp8_exp >= max_exp_val:
        # 指数上溢
        if cfg["has_inf"]: return sign_bit | cfg["inf"]
        else: return sign_bit | cfg["max_norm"]

    elif fp8_exp > 0:
        # 规格化数
        fp8_mant = round((mantissa - 0.5) * (1 << wm))
        if fp8_mant == (1 << wm): # 处理舍入进位
            fp8_mant = 0
            fp8_exp += 1
            if fp8_exp >= max_exp_val:
                if cfg["has_inf"]: return sign_bit | cfg["inf"]
                else: return sign_bit | cfg["max_norm"]
        return sign_bit | (fp8_exp << wm) | fp8_mant
        
    else: # fp8_exp <= 0, 非规格化数
        fp8_mant = round(mantissa * (1 << wm) * (2 ** fp8_exp))
        return sign_bit | fp8_mant

# --- 测试和验证 ---
if __name__ == '__main__':
    test_vals = [65528, 0.0, -0.0, 1.0, -1.0, 500, -500, float('inf'), -float('inf'), float('nan')]
    
    print("--- OCP e4m3 ---")
    for v in test_vals:
        print(f"{v!r:>12} -> {float_to_fp8(v, 'ocp_e4m3'):#04x}")
    # Expected: inf -> NaN (0x7F), -0 -> 0x80, 500 -> clamped to max (0x7E)

    print("\n--- OCP e5m2 ---")
    for v in test_vals:
        print(f"{v!r:>12} -> {float_to_fp8(v, 'ocp_e5m2'):#04x}")
    # Expected: inf -> Inf (0x7C), -inf -> 0xFC, -0 -> 0x80, 500 -> 0x59

    print("\n--- FNUZ e4m3 ---")
    for v in test_vals:
        print(f"{v!r:>12} -> {float_to_fp8(v, 'fnuz_e4m3'):#04x}")
    # Expected: inf -> NaN (0x80), -0 -> 0, 500 -> clamped to max (0x7E)

    print("\n--- FNUZ e5m2 ---")
    for v in test_vals:
        print(f"{v!r:>12} -> {float_to_fp8(v, 'fnuz_e5m2'):#04x}")
    # Expected: inf -> NaN (0x80), -inf -> NaN (0x80), -0 -> 0, 500 -> 0x59