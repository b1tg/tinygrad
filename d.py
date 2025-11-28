from tinygrad import Tensor, dtypes
from tinygrad.dtype import float_to_fp8, fp8_to_float, truncate
a_dtype = dtypes.uchar
b_dtype = dtypes.fp8e5m2
# b_dtype = dtypes.fp8e4m3

# a, b = Tensor([1,2,3,4], dtype=a_dtype), Tensor([1,2,3,4], dtype=b_dtype)
# print(a.numpy())
# print(b.numpy())
# c = a + b
# print(c.numpy())
# a = Tensor([0x38], dtype=dtypes.uchar)
# a = Tensor([0xff, 255], dtype=dtypes.uchar)
# # b = a.bitcast(dtypes.fp8e4m3)
# b = a.bitcast(dtypes.fp8e5m2)
# print(b.numpy())


# a = Tensor([124], dtype=dtypes.uchar)
# b = a.bitcast(dtypes.fp8e5m2)
# print(b.numpy())


# print(fp8_to_float(127, dtypes.fp8e5m2))
# print(fp8_to_float(-4, dtypes.fp8e4m3))
# a = Tensor([251,   0, 255,   4,   2, 251, 255,   9,   5, 247], dtype=dtypes.uint8)
# a = Tensor([251,], dtype=dtypes.float)
# b = a.cast(dtypes.fp8e4m3)
# print(b.numpy(), b.bitcast(dtypes.uchar).numpy())

# _test_cast a.numpy()=array([65528,     6, 65527,     4,     6,     4,     7,     3, 65532,
#            1], dtype=uint16), a.dtype=dtypes.ushort, expected=[448.0, 6.0, 448.0, 4.0, 6.0, 4.0, 7.0, 3.0, 448.0, 1.0], target_dtype=dtypes.fp8e4m3
a = Tensor([65528,     6, 65527,     4,     6,     4,     7,     3, 65532, 1], dtype=dtypes.uint16)

b = a.cast(dtypes.fp8e4m3)
print(b.numpy())
print(truncate[dtypes.fp8e4m3](65528))
print(float_to_fp8(65528, dtypes.fp8e4m3))

import math
import struct

# 定义数据类型和配置
class DType: pass
class dtypes:
    fp8e4m3 = DType()
    fp8e5m2 = DType()

# 简化的配置，增加了 EXPONENT_BITS 以便计算
CONFIG = {
    dtypes.fp8e4m3: {
        "EXPONENT_BITS": 4, "SIGNIFICAND_BITS": 3, "EXP_BIAS": 7,
        "MAXNORM": 0x7E, "INF_VALUE": 0x7F
    },
    dtypes.fp8e5m2: {
        "EXPONENT_BITS": 5, "SIGNIFICAND_BITS": 2, "EXP_BIAS": 15,
        "MAXNORM": 0x7B, "INF_VALUE": 0x7C
    }
}

def float_to_fp8_simple(x: float, dtype: DType) -> int:
    """一个更简单、更易读的 float 到 FP8 转换实现。"""
    cfg = CONFIG[dtype]
    
    # 1. 处理特殊值
    if math.isnan(x):
        # e4m3 没有专门的 Inf，NaN 的最高位代表符号
        return 0x7F if math.copysign(1, x) > 0 else 0xFF if dtype == dtypes.fp8e4m3 \
               else cfg["INF_VALUE"] | (1 << (cfg["SIGNIFICAND_BITS"] - 1)) # e5m2 NaN
    if math.isinf(x):
        return cfg["INF_VALUE"] if x > 0 else cfg["INF_VALUE"] | 0x80
    if x == 0.0:
        return 0

    # 2. 分解符号、尾数和指数
    sign_bit = 0 if x > 0 else 0x80
    abs_x = abs(x)
    
    # math.frexp(x) 返回，其中 x = m * 2**e 且 0.5 <= abs(m) < 1
    # 这为我们提供了规格化的尾数和以2为底的指数
    mantissa, exponent = math.frexp(abs_x)
    
    # 3. 计算目标 FP8 指数
    # FP8 的指数是 (frexp的指数) + (FP8偏置) - 1
    # 减 1 是因为 frexp 的尾数范围是 [0.5, 1)，即 1.xxxx * 2**(-1)
    fp8_exp = exponent + cfg["EXP_BIAS"] - 1
    max_exp_val = (1 << cfg["EXPONENT_BITS"]) - 1 # 全1指数，用于Inf/NaN
    print(f"{max_exp_val=:x}, {cfg["EXPONENT_BITS"]=}, {fp8_exp=}")

    # 4. 根据指数范围处理不同情况
    if fp8_exp >= max_exp_val:
        # 上溢：钳位到最大值或返回无穷大
        if dtype == dtypes.fp8e4m3:
            return sign_bit | cfg["MAXNORM"] # e4m3 钳位
        else:
            return sign_bit | cfg["INF_VALUE"] # e5m2 返回 Inf
            
    elif fp8_exp > 0:
        # 规格化数
        # 尾数部分是，我们取其整数部分
        # 使用 round() 实现“舍入到最接近的偶数”
        fp8_mant = round((mantissa - 0.5) * (1 << cfg["SIGNIFICAND_BITS"]))
        
        # 处理舍入导致的进位 (例如 1.111 -> 10.000)
        if fp8_mant == (1 << cfg["SIGNIFICAND_BITS"]):
            fp8_mant = 0
            fp8_exp += 1
            # 再次检查是否上溢
            if fp8_exp >= max_exp_val:
                if dtype == dtypes.fp8e4m3: return sign_bit | cfg["MAXNORM"]
                else: return sign_bit | cfg["INF_VALUE"]
        
        return sign_bit | (fp8_exp << cfg["SIGNIFICAND_BITS"]) | fp8_mant
        
    else: # fp8_exp <= 0
        # 非规格化数
        # 将尾数右移，指数为0
        # 值为 mantissa * 2**exponent
        # FP8非规格化值为 mantissa / 2**N * 2**(1-bias)
        # -> mantissa = x * 2**N * 2**(bias-1)
        # -> mantissa = (m * 2**e) * 2**N * 2**(bias-1)
        # -> mantissa = m * 2**(e + bias - 1) * 2**N
        # -> mantissa = m * 2**fp8_exp * 2**N
        fp8_mant = round(mantissa * (1 << cfg["SIGNIFICAND_BITS"]) * (1 << fp8_exp))
        return sign_bit | fp8_mant


print(float_to_fp8_simple(65528, dtypes.fp8e4m3))