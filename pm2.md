
# 希望你能在tinygrad中实现 amd cstyle后端支持进行fp8 matmul计算时能融合反量化参数(scale)

- 最终目的是减少FP8Linear的反量化的额外开销
- c函数是：__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4
- 目前触发tensorcore amd_cdna_1616128 时能用到这个指令，不过现在的实现是把scale值都设成了1.0， 利用不了matmul完使用scale参数的能力
    - 触发方式 DEBUG=5 N=4096 AMD_LLVM=0  PYTHONPATH=. CNT=1    TC=1 AMD=1 PYTHONPATH=. FP8E4M3=1 HALF=0 python extra/gemm/simple_matmul.py
- scale值貌似是特殊格式E8M0（• The scale is an E8M0 exponent with a bias of 127）

# amd rdna4 文档中的描述

7.1.5. 8-bit and Smaller Matrix Operations and Layouts
There are two MFMA instructions which can independently select FP4, FP6 or FP6 for the A and B matrices:
V_MFMA_F32_16x16x128_F8F6F4 V_MFMA_F32_32x32x64_F8F6F4 A & B Matrix 16x128
F8: 8 VGPRs
F6: 6 VGPRs
F4: 4 VGPRs
32x64
F8: 8 VGPRs
F6: 6 VGPRs
F4: 4 VGPRs
C & D Matrices Notes
16x16 F32
4 VGPRs
If either matrix = F8 → 32 cycles
Else → 16 cycles
32x32 F32
16 VGPRs
If either matrix = F8 → 64 cycles
Else → 32 cycles
Rules for the F8F6F4 MFMA instructions:
Control Matrix Format Behavior
supports mixed types (i.e., any combination of the formats defined).
CBSZ[2:0] defines the matrix A format, BLGP[2:0] defines the matrix B format. Matrix op
BLGP[2:0] /
CBSZ[2:0]
3’b000 E4M3 (FP8)
3’b001 E5M2 (BF8)
3’b010 E2M3 (FP6)
3’b011 E3M2 (BF6)
3’b100 E2M1 (FP4)
Denorm Control Ignores Denorm Control from MODE and keep Input/Output Denorms.
Clamp Supported and uses the FP16_OVFL bit.
normalized to +/-INF.
dropped.
If set, F32 Result on overflow is clamped to +/- MAX, otherwise the overflow result is
If set, I32 Result is clamped to +/-MAX on overflow/underflow, otherwise the carry out bits are
Round Mode ignores Round Mode from MODE and forces it to RNE.
Imod/Omod Not Supported
Exceptions Not Supported
Execution Mask ignores exec mask from MODE and forces it to 1 for all threads
Operand
Alignment/Sources
Src0/1/2/VDST if VGPR need to be even aligned.
Src0/1 can be only VGPR/ACC_VGPR.
SRC2 can be VGPR/ACC_VGPR/Constant
7.1. Matrix fused-multiply-add (MFMA) 50 of 600
CDNA4 Instruction Set Architecture
Control Behavior
Scale Format is E8M0.
ABID[0] = 1’b1 : Must be set for V_MFMA_SCALE_F32_16X16X128_F8F6F4 and
V_MFMA_SCALE_F32_32X32X64_F8F6F4 instructions.
ABID[0] = 1’b0 : forces all scales into the ALU as 1.0f (exponent = 0x7f Biased – MFMA Runs
without scale source).
Hardware adjusts this scale value in its calculation: d_exp = (a0_exp+b0_exp) + (a1_exp+
b1_exp) + … + c_exp + scale_a + scale_b.


7.2.1. MFMA with Block Exponent Scaling
Scale values are set for MFMA with 4-dword instructions that combine a "Load-Scale factors" and MFMA
functions into one instruction:
V_MFMA_SCALE_F32_16X16X128_F8F6F4, V_MFMA_SCALE_F32_32X32X64_F8F6F4.
The scale value is used just for one instruction and does not carry forward into non-"scale" MFMA ops.
The 4-DWORD instruction is constructed in a manner that looks like two back-to-back VOP3P’s, where the first
holds has the constant 0xD3AC across what is normally the ENCODING through OPCODE fields, and the second
VOP3P has OP = V_MFMA_SCALE_F32_16X16X128_F8F6F4 or V_MFMA_SCALE_F32_32X32X64_F8F6F4.
Operands of Load-Scale (first 2 DWORDs of "SCALE" ops):
ENCODING 0xCC35 in bits [31:16]
SRC0 Matrix A scale
{OP_SEL_HI [0], OP_SEL[0]} defines which part of scale is used by the Matrix A of MFMA instruction.
SRC1 Matrix B scale
{OP_SEL_HI [1], OP_SEL[1]} defines which part of scale is used by the Matrix B of MFMA instruction.
Scale for F4/6/8 matrix (2-bit OPSEL codes):
00: Src[7:0] Lane 0-63 is the scale to be used
01: Src[15:8] Lane 0-63 is the scale to be used
10: Src[23:16] Lane 0-63 is the scale to be used
11: Src[31:24] Lane 0-63 is the scale to be used
Scale values (SRC0 and SRC1) can be either VGPRs or Inline constants (floats, using only the exponent portion).
For the V_MFMA_F32_16x16x128_F8F6F4 op, the K dimension is 128. There is one scale value for every 32 K-
dimension values: 128/32 = 4 scale values per matrix row. The M and N dimensions are 16, so there are 16 rows.
This means in total the matrix needs 16 * 4 = 64 8-bit scale values. This comes from one-quarter of one VGPR
across 64 lanes.
See the next section for the list of MFMA operations which support SCALE.
7.2. Block Scaled Matrices 56 of 600
CDNA4 Instruction Set Architecture
Scale data layout for 16x16 Output Matrices (K=128):
Lane 0        Lane 1 …       Lane 15       Lane 16 …       Lane 32 …       Lane 63
M=0, K=0..31  M=1, K=0..31 … M=15, K=0..31 M=0, K=32..63 … M=0, K=64..95 … M=15, K=96..127

Scale data layout for 32x32 Output Matrices (K=64):
Lane 0       Lane 1 …       Lane 15       Lane 16 …       Lane 32 …       Lane 63
M=0, K=0..31 M=1, K=0..31 … M=15, K=0..31 M=16, K=0..31 … M=0, K=32..63 … M=31, K=32..63

6.7.1. Packed Convert
All convert opcodes operating on FP6/BF6/FP4 data must use VGPR sources for any operand slots providing
more than 32-bits of data.
4-bit CVT_SCALE_PK_FP4_F32
CVT_SCALE_SR_PK_FP4_F32
CVT_SCALE_PK_F32_FP4
6-bit CVT_SCALE_PK_FP6_F32
CVT_SCALE_PK_BF6_F32
CVT_SCALE_SR_PK_FP6_F32
CVT_SCALE_SR_PK_BF6_F32
CVT_SCALE_PK_F32_FP6
CVT_SCALE_PK_F32_BF6
CVT_SCALE_PK_FP4_F16
CVT_SCALE_PK_FP4_BF16
CVT_SCALE_SR_PK_FP4_F16
CVT_SCALE_SR_PK_FP4_BF16
CVT_SCALE_PK_F16_FP4
CVT_SCALE_PK_BF16_FP4
CVT_SCALE_PK_FP6_F16
CVT_SCALE_PK_FP6_FB16
CVT_SCALE_PK_BF6_F16
CVT_SCALE_PK_BF6_BF16
CVT_SCALE_SR_PK_FP6_F16
CVT_SCALE_SR_PK_FP6_BF16
CVT_SCALE_SR_PK_BF6_F16
CVT_SCALE_SR_PK_BF6_BF16
CVT_SCALE_PK_F16_FP6
CVT_SCALE_PK_F16_BF6
CVT_SCALE_PK_BF16_FP6
CVT_SCALE_PK_BF16_BF6
16-bit 8-bit
CVT_SCALE_PK_FP8_F32
CVT_SCALE_PK_BF8_F32
CVT_SCALE_SR_FP8_F32
CVT_SCALE_SR_BF8_F32
CVT_SCALE_PK_F32_FP8
CVT_SCALE_PK_F32_BF8
CVT_SCALE_F32_FP8
CVT_SCALE_F32_BF8
CVT_SCALE_PK_FP8_F16
CVT_SCALE_PK_BF8_F16
CVT_SCALE_PK_FP8_BF16
CVT_SCALE_PK_BF8_BF16
CVT_SCALE_SR_FP8_F16
CVT_SCALE_SR_BF8_F16
CVT_SCALE_SR_FP8_BF16
CVT_SCALE_SR_BF8_BF16
CVT_SCALE_PK_F16_FP8
CVT_SCALE_PK_F16_BF8
CVT_SCALE_F16_FP8
CVT_SCALE_F16_BF8
Integer 8-bit
6.7. Packed Math 39 of 600
CDNA4 Instruction Set Architecture
CVT_PK_F16_F32
CVT_PK_BF16_F32
CVT_F32_BF16
ASHR_PK_I8_I32
ASHR_PK_U8_I32
Convert instructions with SCALE add an 8-bit exponent bias (E8M0, bias of 127) to each F4/F6/F8 value. Each
exponent bias is shared by a block of 32 values along the K dimension.
For example, conversion from FP32 to FP6 (16x16x128):
• Source data is in VGPRs 0..31, with K=0..31 for M=0 in lane0, M=1 in lane1 etc up to M=15 in lane 15; then
K=32..63 in lanes 16..31; K=64..96 in lanes 32..48; and K=96..127 in lanes 48..63.
• Result data is in VGPRs 0..5, with K and M distributed similarly (lanes0..15 has K=0..31 and M=0..15).
• Exponent biases: the VGPR holds one set of exponent biases in bits [30:23] (typical float32 exponent
position).

# 代码示例

hip c 代码参考：
template <typename AType_, typename BType_, WGAttrCtlEnum Ctrl_ = WGAttrCtlEnum::Default_>
struct WarpGemmAttributeMfmaImpl_f32_16x16x128_f8f6f4
{
    static constexpr WGAttrCtlEnum Ctrl = Ctrl_;
    using ADataType                     = AType_;
    using BDataType                     = BType_;
    using CDataType                     = float;

    using AVecType = ext_vector_t<ADataType, 32 / numeric_traits<ADataType>::PackedSize>;
    using BVecType = ext_vector_t<BDataType, 32 / numeric_traits<BDataType>::PackedSize>;
    using CVecType = ext_vector_t<CDataType, 4>;

    static constexpr index_t kM = 16;
    static constexpr index_t kN = 16;
    static constexpr index_t kK = 128;

    static constexpr index_t kAMBlock = 1;
    static constexpr index_t kBNBlock = 1;

    static constexpr index_t kAMLane     = 16;
    static constexpr index_t kBNLane     = 16;
    static constexpr index_t kABKLane    = 4;
    static constexpr index_t kABKPerLane = 32;

    static constexpr index_t kCMLane     = 4;
    static constexpr index_t kCNLane     = 16;
    static constexpr index_t kCM0PerLane = 1;
    static constexpr index_t kCM1PerLane = 4;

    // c_vec += a_vec * b_vec
    template <index_t opselA, index_t opselB, bool post_nop_ = false>
    CK_TILE_DEVICE void operator()(CVecType& c_vec,
                                   const AVecType& a_vec,
                                   const int32_t& a_scale,
                                   const BVecType& b_vec,
                                   const int32_t& b_scale,
                                   bool_constant<post_nop_> = {}) const
    {
#if defined(__gfx950__)
        auto dtype2conf = [](auto dtype) {
            if constexpr(std::is_same_v<decltype(dtype), fp8_t>)
                return make_tuple(number<0>{}, int32x8_t{});
            else if constexpr(std::is_same_v<decltype(dtype), bf8_t>)
                return make_tuple(number<1>{}, int32x8_t{});
            // else if e2m3 => make_tuple(number<2>{}, int32x6_t{})
            // else if e3m2 => make_tuple(number<3>{}, int32x6_t{})
            else if constexpr(std::is_same_v<decltype(dtype), pk_fp4_t>)
                return make_tuple(number<4>{}, int32x4_t{});
            else
                static_assert(false, "Unsupported data type for mfma scale");
        };
        auto dtype2code = [&](auto dtype) { return dtype2conf(dtype)(number<0>{}); };
        auto dtype2vec  = [&](auto dtype) { return dtype2conf(dtype)(number<1>{}); };
        auto arg256     = [&](auto x) {
            if constexpr(sizeof(x) == 16)
                return int32x8_t{x[0], x[1], x[2], x[3], 0, 0, 0, 0};
            else if constexpr(sizeof(x) == 24)
                return int32x8_t{x[0], x[1], x[2], x[3], x[4], x[5], 0, 0};
            else if constexpr(sizeof(x) == 32)
                return x;
            else
                static_assert(false, "Unexpected vector size for mfma scale");
        };

        auto arg_a         = bit_cast<decltype(dtype2vec(ADataType{}))>(a_vec);
        auto arg_b         = bit_cast<decltype(dtype2vec(BDataType{}))>(b_vec);
        constexpr int cbsz = decltype(dtype2code(ADataType{}))::value;
        constexpr int blgp = decltype(dtype2code(BDataType{}))::value;
        c_vec              = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            arg256(arg_a), arg256(arg_b), c_vec, cbsz, blgp, opselA, a_scale, opselB, b_scale);
#else
        ck_tile::ignore = c_vec;
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = a_scale;
        ck_tile::ignore = b_scale;
#endif
    }

    // c_vec = a_vec * b_vec
    template <index_t opselA, index_t opselB>
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec,
                                       const int32_t& a_scale,
                                       const BVecType& b_vec,
                                       const int32_t& b_scale) const
    {
        CVecType c_vec{0.f};
        operator()<opselA, opselB>(c_vec, a_vec, a_scale, b_vec, b_scale);
        return c_vec;
    }

    // c_vec += a_vec * b_vec
    template <bool post_nop_ = false>
    CK_TILE_DEVICE void operator()(CVecType& c_vec,
                                   const AVecType& a_vec,
                                   const BVecType& b_vec,
                                   bool_constant<post_nop_> = {}) const
    {
        operator()<0, 0>(c_vec, a_vec, 0, b_vec, 0);
    }

    // c_vec = a_vec * b_vec
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
        return operator()<0, 0>(a_vec, 0, b_vec, 0);
    }
};
