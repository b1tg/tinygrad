
# 20260205 23:30

当前状态（简要）：
• - 新增文件包括UOps助手的移植及测试运行器
  - 运行了性能测试，达到102 TFLOPS，tinygrad内核达143 TFLOPS
  - 核心代码未做修改，仍有待改进的性能优化项
  - 距离165 TFLOPS的目标还有差距
  - 已把 /home/b1tg/data/tinygrad1 的 wmma_uop_helpers.py 移植到本仓库并适配当前 UOp API（改成 Ops.PARAM，并用 __ldmatrix_a/b），未改 tinygrad 核心。
  - 新增了独立可跑的 UOps 测试程序 extra/gemm/nv_wmma_uop_matmul.py（默认 FP32 累加，支持 VARIANT=fp16/fp32）。
  - 4090 上 FP32 累加的 UOps 版本目前约 102 TFLOPS（N=4096）；仍低于 tinygrad 现有生成核的 ~143 TFLOPS，也未接近 165+。
  - 运行过的命令：
    PYTHONPATH=. CNT=2 N=4096 VARIANT=fp32 .venv/bin/python extra/gemm/nv_wmma_uop_matmul.py

  如果要继续冲 165+，下一步需要定位与 extra/gemm/max_kernels/nv.fp16_fp32_fp32.max.cu 的差距（PTX/SASS 或 pipeline/ldmatrix 负载与 barrier 时序），或允许极小的 renderer 支持（不改核心逻辑）。
(.venv) (cuda_uop_matmul) ~/tinygrad$ PYTHONPATH=. CNT=2 N=4096 VARIANT=fp32 .venv/bin/python extra/gemm/nv_wmma_uop_matmul.py
  16777216   1350.66 us, would be 101757.18 GFLOPS matmul, 74.53 GB/s
(.venv) (cuda_uop_matmul) ~/tinygrad$ python extra/gemm/nv_uop_matmul.py
*** NV         3 nv_uop_gemm                                    arg  3 mem   0.10 GB tm    961.09us/     0.96ms (    143 TFLOPS  105|8973   GB/s) 
*** NV         4 nv_uop_gemm                                    arg  3 mem   0.10 GB tm    963.55us/     1.92ms (    143 TFLOPS  104|8950   GB/s) 
*** NV         5 nv_uop_gemm                                    arg  3 mem   0.10 GB tm    965.25us/     2.89ms (    142 TFLOPS  104|8934   GB/s) 
*** NV         6 nv_uop_gemm                                    arg  3 mem   0.10 GB tm    962.75us/     3.85ms (    143 TFLOPS  105|8957   GB/s) 
*** NV         7 nv_uop_gemm                                    arg  3 mem   0.10 GB tm    962.72us/     4.82ms (    143 TFLOPS  105|8957   GB/s) 
REAL TFLOPS 143.00
*** NV         1 r_64_32_32_4_2_2_4_4_256_2                     arg  3 mem   0.13 GB tm   1435.39us/     1.44ms (  95750 GFLOPS   70|4512   GB/s) ['__matmul__']
mean squared error 0.0
(.venv) (cuda_uop_matmul) ~/tinygrad$ 


# 20260205 init

finish the bounty: 165+ TFLOP GEMM (目标是达到torch的速度) with kernel on 4090, FP16 with FP32 acc. amd_uop_matmul style
- amd_uop_matmul style的意思是模仿extra/gemm/amd_uop_matmul.py，自己构建tinygrad uops来实现，最后运行的kernel不能是依赖于.cu或者除了tinygrad之外的第三份库（除了作为速度对比测试）
- 最终需要的是一个像extra/gemm/amd_uop_matmul.py一样单独可以运行测试的程序
- always test and run command by yourself
- manually built UOp kernel like amd_uop_matmul without relying on opts_to_apply or auto scheduling



for reference: tinygrad beam got 132 TFLOP currently
参考simple_matmul.py生成的132 TFLOPS kernel，这是tinygrad生成的kernel，你没有理由比这个慢

(.venv) (master) ~/tinygrad$ DEBUG=5 BEAM=3 N=4096 CNT=2  HALF=1 SHOULD_USE_TC=1 python extra/gemm/simple_matmul.py
loading libc from /lib/x86_64-linux-gnu/libc.so.6
loading hsa failed: not found on system
loading iokit failed: not found on system
loading corefoundation failed: not found on system
loading libusb from /lib/x86_64-linux-gnu/libusb-1.0.so.0
loading nvrtc from /usr/local/cuda/targets/x86_64-linux/lib/libnvrtc.so
loading nvjitlink from /usr/local/cuda/targets/x86_64-linux/lib/libnvJitLink.so
loading mesa from /usr/lib/libtinymesa_cpu.so
opened device NV from pid:1193540
opened device NPY from pid:1193540
scheduled    1 kernels in    26.08 ms | CACHE MISS 1329eee9 | 53 uops in cache
*** NV         1 copy   33.55M,      NV <- NPY                  arg  2 mem   0.07 GB tm   4586.45us/     4.59ms (      0 GFLOPS    7|7      GB/s) 
scheduled    1 kernels in     0.26 ms |  cache hit 1329eee9 | 54 uops in cache
*** NV         2 copy   33.55M,      NV <- NPY                  arg  2 mem   0.10 GB tm   5294.67us/     9.88ms (      0 GFLOPS    6|6      GB/s) 
scheduled    1 kernels in    16.39 ms | CACHE MISS 8f2fa1f5 | 145 uops in cache
c0 = UOp(Ops.PARAM, dtypes.half.ptr(16777216), (), 0)
c2 = UOp.range(4096, 1, AxisType.LOOP)
c3 = c2*UOp.const(dtypes.index, 4096)
c4 = UOp.range(4096, 2, AxisType.LOOP)
c7 = UOp(Ops.PARAM, dtypes.half.ptr(16777216), (), 1)
c8 = UOp.range(4096, 0, AxisType.REDUCE)
c11 = UOp(Ops.PARAM, dtypes.half.ptr(16777216), (), 2)
c16 = (c7.index((c3+c8))*c11.index((c8*UOp.const(dtypes.index, 4096)+c4))).cast(dtypes.float)
c18 = c16.reduce(c8, arg=Ops.ADD).cast(dtypes.half)
c20 = c0.index((c3+c4), ptr=True).store(c18).end(c2, c4)
ast = c20.sink(arg=KernelInfo(name='test', axis_types=(), dont_use_locals=False, applied_opts=(), opts_to_apply=None, estimates=None))
TC(0): [(1, 4096)] [(2, 4096)] [(0, 4096)]
(Opt(op=OptOps.TC, axis=0, arg=(-1, 0, 1)), Opt(op=OptOps.UPCAST, axis=0, arg=4), Opt(op=OptOps.UPCAST, axis=1, arg=4), Opt(op=OptOps.UPCAST, axis=1, arg=2), Opt(op=OptOps.LOCAL, axis=0, arg=2), Opt(op=OptOps.LOCAL, axis=1, arg=2), Opt(op=OptOps.UNROLL, axis=0, arg=4))
#define INFINITY (__int_as_float(0x7f800000))
#define NAN (__int_as_float(0x7fffffff))
template <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f = v; return u.t; }
#include <cuda_fp16.h>
struct __align__(8) half4 { half x, y, z, w; }; __device__ half4 make_half4(half x, half y, half z, half w) { half4 r={x, y, z, w}; return r; }
struct __align__(16) half8 { half x, y, z, w, a, b, c, d; }; __device__ half8 make_half8(half x, half y, half z, half w, half a, half b, half c, half d) { half8 r={x, y, z, w, a, b, c, d}; return r; }
__device__ float4 __WMMA_8_16_16_half_float(half8 a, half4 b, float4 c){
  int *a_pk = (int *)(&a), *b_pk = (int *)(&b), *c_pk = (int *)(&c);
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7},"
      "{%8, %9}, {%0, %1, %2, %3};"
    : "+r"(c_pk[0]), "+r"(c_pk[1]), "+r"(c_pk[2]), "+r"(c_pk[3])
    : "r"(a_pk[0]), "r"(a_pk[1]), "r"(a_pk[2]), "r"(a_pk[3]), "r"(b_pk[0]), "r"(b_pk[1]));
  return c;
}
extern "C" __global__ void __launch_bounds__(128) r_32_32_32_2_2_2_2_4_4_2_64_2_4(half* data0_16777216, half* data1_16777216, half* data2_16777216) {
  float acc0[128];
  int gidx0 = blockIdx.x; /* 32 */
  int gidx1 = blockIdx.y; /* 32 */
  int lidx0 = threadIdx.x; /* 32 */
  int lidx1 = threadIdx.y; /* 2 */
  int lidx2 = threadIdx.z; /* 2 */
  int alu0 = ((gidx0<<7)+(lidx2<<6));
  int alu1 = (lidx0>>4);
  int alu2 = ((lidx0>>3)&1);
  int alu3 = ((lidx0>>2)&1);
  int alu4 = ((gidx1<<19)+(lidx1<<18)+(alu1<<14)+(alu2<<13)+(alu3<<12));
  int alu5 = (lidx0&1);
  int alu6 = (alu5<<1);
  int alu7 = ((lidx0>>1)&1);
  int alu8 = (alu7<<2);
  *(acc0+0) = 0.0f;
  *(acc0+1) = 0.0f;
  *(acc0+2) = 0.0f;
  *(acc0+3) = 0.0f;
  *(acc0+4) = 0.0f;
  *(acc0+5) = 0.0f;
  *(acc0+6) = 0.0f;
  *(acc0+7) = 0.0f;
  *(acc0+8) = 0.0f;
  *(acc0+9) = 0.0f;
  *(acc0+10) = 0.0f;
  *(acc0+11) = 0.0f;
  *(acc0+12) = 0.0f;
  *(acc0+13) = 0.0f;
  *(acc0+14) = 0.0f;
  *(acc0+15) = 0.0f;
  *(acc0+16) = 0.0f;
  *(acc0+17) = 0.0f;
  *(acc0+18) = 0.0f;
  *(acc0+19) = 0.0f;
  *(acc0+20) = 0.0f;
  *(acc0+21) = 0.0f;
  *(acc0+22) = 0.0f;
  *(acc0+23) = 0.0f;
  *(acc0+24) = 0.0f;
  *(acc0+25) = 0.0f;
  *(acc0+26) = 0.0f;
  *(acc0+27) = 0.0f;
  *(acc0+28) = 0.0f;
  *(acc0+29) = 0.0f;
  *(acc0+30) = 0.0f;
  *(acc0+31) = 0.0f;
  *(acc0+32) = 0.0f;
  *(acc0+33) = 0.0f;
  *(acc0+34) = 0.0f;
  *(acc0+35) = 0.0f;
  *(acc0+36) = 0.0f;
  *(acc0+37) = 0.0f;
  *(acc0+38) = 0.0f;
  *(acc0+39) = 0.0f;
  *(acc0+40) = 0.0f;
  *(acc0+41) = 0.0f;
  *(acc0+42) = 0.0f;
  *(acc0+43) = 0.0f;
  *(acc0+44) = 0.0f;
  *(acc0+45) = 0.0f;
  *(acc0+46) = 0.0f;
  *(acc0+47) = 0.0f;
  *(acc0+48) = 0.0f;
  *(acc0+49) = 0.0f;
  *(acc0+50) = 0.0f;
  *(acc0+51) = 0.0f;
  *(acc0+52) = 0.0f;
  *(acc0+53) = 0.0f;
  *(acc0+54) = 0.0f;
  *(acc0+55) = 0.0f;
  *(acc0+56) = 0.0f;
  *(acc0+57) = 0.0f;
  *(acc0+58) = 0.0f;
  *(acc0+59) = 0.0f;
  *(acc0+60) = 0.0f;
  *(acc0+61) = 0.0f;
  *(acc0+62) = 0.0f;
  *(acc0+63) = 0.0f;
  *(acc0+64) = 0.0f;
  *(acc0+65) = 0.0f;
  *(acc0+66) = 0.0f;
  *(acc0+67) = 0.0f;
  *(acc0+68) = 0.0f;
  *(acc0+69) = 0.0f;
  *(acc0+70) = 0.0f;
  *(acc0+71) = 0.0f;
  *(acc0+72) = 0.0f;
  *(acc0+73) = 0.0f;
  *(acc0+74) = 0.0f;
  *(acc0+75) = 0.0f;
  *(acc0+76) = 0.0f;
  *(acc0+77) = 0.0f;
  *(acc0+78) = 0.0f;
  *(acc0+79) = 0.0f;
  *(acc0+80) = 0.0f;
  *(acc0+81) = 0.0f;
  *(acc0+82) = 0.0f;
  *(acc0+83) = 0.0f;
  *(acc0+84) = 0.0f;
  *(acc0+85) = 0.0f;
  *(acc0+86) = 0.0f;
  *(acc0+87) = 0.0f;
  *(acc0+88) = 0.0f;
  *(acc0+89) = 0.0f;
  *(acc0+90) = 0.0f;
  *(acc0+91) = 0.0f;
  *(acc0+92) = 0.0f;
  *(acc0+93) = 0.0f;
  *(acc0+94) = 0.0f;
  *(acc0+95) = 0.0f;
  *(acc0+96) = 0.0f;
  *(acc0+97) = 0.0f;
  *(acc0+98) = 0.0f;
  *(acc0+99) = 0.0f;
  *(acc0+100) = 0.0f;
  *(acc0+101) = 0.0f;
  *(acc0+102) = 0.0f;
  *(acc0+103) = 0.0f;
  *(acc0+104) = 0.0f;
  *(acc0+105) = 0.0f;
  *(acc0+106) = 0.0f;
  *(acc0+107) = 0.0f;
  *(acc0+108) = 0.0f;
  *(acc0+109) = 0.0f;
  *(acc0+110) = 0.0f;
  *(acc0+111) = 0.0f;
  *(acc0+112) = 0.0f;
  *(acc0+113) = 0.0f;
  *(acc0+114) = 0.0f;
  *(acc0+115) = 0.0f;
  *(acc0+116) = 0.0f;
  *(acc0+117) = 0.0f;
  *(acc0+118) = 0.0f;
  *(acc0+119) = 0.0f;
  *(acc0+120) = 0.0f;
  *(acc0+121) = 0.0f;
  *(acc0+122) = 0.0f;
  *(acc0+123) = 0.0f;
  *(acc0+124) = 0.0f;
  *(acc0+125) = 0.0f;
  *(acc0+126) = 0.0f;
  *(acc0+127) = 0.0f;
  for (int Ridx0 = 0; Ridx0 < 64; Ridx0++) {
    int alu137 = (alu0+(alu1<<2)+(alu2<<1)+alu3+(alu7<<14)+(Ridx0<<18)+(alu5<<13));
    half val0 = (*(data2_16777216+(alu137+8)));
    half val1 = (*(data2_16777216+(alu137+16)));
    half val2 = (*(data2_16777216+(alu137+24)));
    half val3 = (*(data2_16777216+(alu137+32)));
    half val4 = (*(data2_16777216+(alu137+40)));
    half val5 = (*(data2_16777216+(alu137+48)));
    half val6 = (*(data2_16777216+(alu137+56)));
    half val7 = (*(data2_16777216+(alu137+4096)));
    half val8 = (*(data2_16777216+(alu137+4104)));
    half val9 = (*(data2_16777216+(alu137+4112)));
    half val10 = (*(data2_16777216+(alu137+4120)));
    half val11 = (*(data2_16777216+(alu137+4128)));
    half val12 = (*(data2_16777216+(alu137+4136)));
    half val13 = (*(data2_16777216+(alu137+4144)));
    half val14 = (*(data2_16777216+(alu137+4152)));
    half val15 = (*(data2_16777216+(alu137+32768)));
    half val16 = (*(data2_16777216+(alu137+32776)));
    half val17 = (*(data2_16777216+(alu137+32784)));
    half val18 = (*(data2_16777216+(alu137+32792)));
    half val19 = (*(data2_16777216+(alu137+32800)));
    half val20 = (*(data2_16777216+(alu137+32808)));
    half val21 = (*(data2_16777216+(alu137+32816)));
    half val22 = (*(data2_16777216+(alu137+32824)));
    half val23 = (*(data2_16777216+(alu137+36864)));
    half val24 = (*(data2_16777216+(alu137+36872)));
    half val25 = (*(data2_16777216+(alu137+36880)));
    half val26 = (*(data2_16777216+(alu137+36888)));
    half val27 = (*(data2_16777216+(alu137+36896)));
    half val28 = (*(data2_16777216+(alu137+36904)));
    half val29 = (*(data2_16777216+(alu137+36912)));
    half val30 = (*(data2_16777216+(alu137+36920)));
    half val31 = (*(data2_16777216+(alu137+65536)));
    half val32 = (*(data2_16777216+(alu137+65544)));
    half val33 = (*(data2_16777216+(alu137+65552)));
    half val34 = (*(data2_16777216+(alu137+65560)));
    half val35 = (*(data2_16777216+(alu137+65568)));
    half val36 = (*(data2_16777216+(alu137+65576)));
    half val37 = (*(data2_16777216+(alu137+65584)));
    half val38 = (*(data2_16777216+(alu137+65592)));
    half val39 = (*(data2_16777216+(alu137+69632)));
    half val40 = (*(data2_16777216+(alu137+69640)));
    half val41 = (*(data2_16777216+(alu137+69648)));
    half val42 = (*(data2_16777216+(alu137+69656)));
    half val43 = (*(data2_16777216+(alu137+69664)));
    half val44 = (*(data2_16777216+(alu137+69672)));
    half val45 = (*(data2_16777216+(alu137+69680)));
    half val46 = (*(data2_16777216+(alu137+69688)));
    half val47 = (*(data2_16777216+(alu137+98304)));
    half val48 = (*(data2_16777216+(alu137+98312)));
    half val49 = (*(data2_16777216+(alu137+98320)));
    half val50 = (*(data2_16777216+(alu137+98328)));
    half val51 = (*(data2_16777216+(alu137+98336)));
    half val52 = (*(data2_16777216+(alu137+98344)));
    half val53 = (*(data2_16777216+(alu137+98352)));
    half val54 = (*(data2_16777216+(alu137+98360)));
    half val55 = (*(data2_16777216+(alu137+102400)));
    half val56 = (*(data2_16777216+(alu137+102408)));
    half val57 = (*(data2_16777216+(alu137+102416)));
    half val58 = (*(data2_16777216+(alu137+102424)));
    half val59 = (*(data2_16777216+(alu137+102432)));
    half val60 = (*(data2_16777216+(alu137+102440)));
    half val61 = (*(data2_16777216+(alu137+102448)));
    half val62 = (*(data2_16777216+(alu137+102456)));
    half val63 = (*(data2_16777216+(alu137+131072)));
    half val64 = (*(data2_16777216+(alu137+131080)));
    half val65 = (*(data2_16777216+(alu137+131088)));
    half val66 = (*(data2_16777216+(alu137+131096)));
    half val67 = (*(data2_16777216+(alu137+131104)));
    half val68 = (*(data2_16777216+(alu137+131112)));
    half val69 = (*(data2_16777216+(alu137+131120)));
    half val70 = (*(data2_16777216+(alu137+131128)));
    half val71 = (*(data2_16777216+(alu137+135168)));
    half val72 = (*(data2_16777216+(alu137+135176)));
    half val73 = (*(data2_16777216+(alu137+135184)));
    half val74 = (*(data2_16777216+(alu137+135192)));
    half val75 = (*(data2_16777216+(alu137+135200)));
    half val76 = (*(data2_16777216+(alu137+135208)));
    half val77 = (*(data2_16777216+(alu137+135216)));
    half val78 = (*(data2_16777216+(alu137+135224)));
    half val79 = (*(data2_16777216+(alu137+163840)));
    half val80 = (*(data2_16777216+(alu137+163848)));
    half val81 = (*(data2_16777216+(alu137+163856)));
    half val82 = (*(data2_16777216+(alu137+163864)));
    half val83 = (*(data2_16777216+(alu137+163872)));
    half val84 = (*(data2_16777216+(alu137+163880)));
    half val85 = (*(data2_16777216+(alu137+163888)));
    half val86 = (*(data2_16777216+(alu137+163896)));
    half val87 = (*(data2_16777216+(alu137+167936)));
    half val88 = (*(data2_16777216+(alu137+167944)));
    half val89 = (*(data2_16777216+(alu137+167952)));
    half val90 = (*(data2_16777216+(alu137+167960)));
    half val91 = (*(data2_16777216+(alu137+167968)));
    half val92 = (*(data2_16777216+(alu137+167976)));
    half val93 = (*(data2_16777216+(alu137+167984)));
    half val94 = (*(data2_16777216+(alu137+167992)));
    half val95 = (*(data2_16777216+(alu137+196608)));
    half val96 = (*(data2_16777216+(alu137+196616)));
    half val97 = (*(data2_16777216+(alu137+196624)));
    half val98 = (*(data2_16777216+(alu137+196632)));
    half val99 = (*(data2_16777216+(alu137+196640)));
    half val100 = (*(data2_16777216+(alu137+196648)));
    half val101 = (*(data2_16777216+(alu137+196656)));
    half val102 = (*(data2_16777216+(alu137+196664)));
    half val103 = (*(data2_16777216+(alu137+200704)));
    half val104 = (*(data2_16777216+(alu137+200712)));
    half val105 = (*(data2_16777216+(alu137+200720)));
    half val106 = (*(data2_16777216+(alu137+200728)));
    half val107 = (*(data2_16777216+(alu137+200736)));
    half val108 = (*(data2_16777216+(alu137+200744)));
    half val109 = (*(data2_16777216+(alu137+200752)));
    half val110 = (*(data2_16777216+(alu137+200760)));
    half val111 = (*(data2_16777216+(alu137+229376)));
    half val112 = (*(data2_16777216+(alu137+229384)));
    half val113 = (*(data2_16777216+(alu137+229392)));
    half val114 = (*(data2_16777216+(alu137+229400)));
    half val115 = (*(data2_16777216+(alu137+229408)));
    half val116 = (*(data2_16777216+(alu137+229416)));
    half val117 = (*(data2_16777216+(alu137+229424)));
    half val118 = (*(data2_16777216+(alu137+229432)));
    half val119 = (*(data2_16777216+(alu137+233472)));
    half val120 = (*(data2_16777216+(alu137+233480)));
    half val121 = (*(data2_16777216+(alu137+233488)));
    half val122 = (*(data2_16777216+(alu137+233496)));
    half val123 = (*(data2_16777216+(alu137+233504)));
    half val124 = (*(data2_16777216+(alu137+233512)));
    half val125 = (*(data2_16777216+(alu137+233520)));
    half val126 = (*(data2_16777216+(alu137+233528)));
    half val127 = (*(data2_16777216+alu137));
    int alu138 = (alu4+alu8+(Ridx0<<6)+alu6);
    half2 val128 = (*((half2*)((data1_16777216+(alu138+8)))));
    half2 val129 = (*((half2*)((data1_16777216+(alu138+16)))));
    half2 val130 = (*((half2*)((data1_16777216+(alu138+24)))));
    half2 val131 = (*((half2*)((data1_16777216+(alu138+32)))));
    half2 val132 = (*((half2*)((data1_16777216+(alu138+40)))));
    half2 val133 = (*((half2*)((data1_16777216+(alu138+48)))));
    half2 val134 = (*((half2*)((data1_16777216+(alu138+56)))));
    half2 val135 = (*((half2*)((data1_16777216+(alu138+32768)))));
    half2 val136 = (*((half2*)((data1_16777216+(alu138+32776)))));
    half2 val137 = (*((half2*)((data1_16777216+(alu138+32784)))));
    half2 val138 = (*((half2*)((data1_16777216+(alu138+32792)))));
    half2 val139 = (*((half2*)((data1_16777216+(alu138+32800)))));
    half2 val140 = (*((half2*)((data1_16777216+(alu138+32808)))));
    half2 val141 = (*((half2*)((data1_16777216+(alu138+32816)))));
    half2 val142 = (*((half2*)((data1_16777216+(alu138+32824)))));
    half2 val143 = (*((half2*)((data1_16777216+(alu138+65536)))));
    half2 val144 = (*((half2*)((data1_16777216+(alu138+65544)))));
    half2 val145 = (*((half2*)((data1_16777216+(alu138+65552)))));
    half2 val146 = (*((half2*)((data1_16777216+(alu138+65560)))));
    half2 val147 = (*((half2*)((data1_16777216+(alu138+65568)))));
    half2 val148 = (*((half2*)((data1_16777216+(alu138+65576)))));
    half2 val149 = (*((half2*)((data1_16777216+(alu138+65584)))));
    half2 val150 = (*((half2*)((data1_16777216+(alu138+65592)))));
    half2 val151 = (*((half2*)((data1_16777216+(alu138+98304)))));
    half2 val152 = (*((half2*)((data1_16777216+(alu138+98312)))));
    half2 val153 = (*((half2*)((data1_16777216+(alu138+98320)))));
    half2 val154 = (*((half2*)((data1_16777216+(alu138+98328)))));
    half2 val155 = (*((half2*)((data1_16777216+(alu138+98336)))));
    half2 val156 = (*((half2*)((data1_16777216+(alu138+98344)))));
    half2 val157 = (*((half2*)((data1_16777216+(alu138+98352)))));
    half2 val158 = (*((half2*)((data1_16777216+(alu138+98360)))));
    half2 val159 = (*((half2*)((data1_16777216+(alu138+131072)))));
    half2 val160 = (*((half2*)((data1_16777216+(alu138+131080)))));
    half2 val161 = (*((half2*)((data1_16777216+(alu138+131088)))));
    half2 val162 = (*((half2*)((data1_16777216+(alu138+131096)))));
    half2 val163 = (*((half2*)((data1_16777216+(alu138+131104)))));
    half2 val164 = (*((half2*)((data1_16777216+(alu138+131112)))));
    half2 val165 = (*((half2*)((data1_16777216+(alu138+131120)))));
    half2 val166 = (*((half2*)((data1_16777216+(alu138+131128)))));
    half2 val167 = (*((half2*)((data1_16777216+(alu138+163840)))));
    half2 val168 = (*((half2*)((data1_16777216+(alu138+163848)))));
    half2 val169 = (*((half2*)((data1_16777216+(alu138+163856)))));
    half2 val170 = (*((half2*)((data1_16777216+(alu138+163864)))));
    half2 val171 = (*((half2*)((data1_16777216+(alu138+163872)))));
    half2 val172 = (*((half2*)((data1_16777216+(alu138+163880)))));
    half2 val173 = (*((half2*)((data1_16777216+(alu138+163888)))));
    half2 val174 = (*((half2*)((data1_16777216+(alu138+163896)))));
    half2 val175 = (*((half2*)((data1_16777216+(alu138+196608)))));
    half2 val176 = (*((half2*)((data1_16777216+(alu138+196616)))));
    half2 val177 = (*((half2*)((data1_16777216+(alu138+196624)))));
    half2 val178 = (*((half2*)((data1_16777216+(alu138+196632)))));
    half2 val179 = (*((half2*)((data1_16777216+(alu138+196640)))));
    half2 val180 = (*((half2*)((data1_16777216+(alu138+196648)))));
    half2 val181 = (*((half2*)((data1_16777216+(alu138+196656)))));
    half2 val182 = (*((half2*)((data1_16777216+(alu138+196664)))));
    half2 val183 = (*((half2*)((data1_16777216+(alu138+229376)))));
    half2 val184 = (*((half2*)((data1_16777216+(alu138+229384)))));
    half2 val185 = (*((half2*)((data1_16777216+(alu138+229392)))));
    half2 val186 = (*((half2*)((data1_16777216+(alu138+229400)))));
    half2 val187 = (*((half2*)((data1_16777216+(alu138+229408)))));
    half2 val188 = (*((half2*)((data1_16777216+(alu138+229416)))));
    half2 val189 = (*((half2*)((data1_16777216+(alu138+229424)))));
    half2 val190 = (*((half2*)((data1_16777216+(alu138+229432)))));
    half2 val191 = (*((half2*)((data1_16777216+alu138))));
    half4 cast0 = make_half4(val0,val8,val16,val24);
    half4 cast1 = make_half4(val32,val40,val48,val56);
    half4 cast2 = make_half4(val64,val72,val80,val88);
    half4 cast3 = make_half4(val96,val104,val112,val120);
    half8 cast4 = make_half8(val143.x,val143.y,val151.x,val151.y,val144.x,val144.y,val152.x,val152.y);
    half8 cast5 = make_half8(val145.x,val145.y,val153.x,val153.y,val146.x,val146.y,val154.x,val154.y);
    half8 cast6 = make_half8(val147.x,val147.y,val155.x,val155.y,val148.x,val148.y,val156.x,val156.y);
    half8 cast7 = make_half8(val149.x,val149.y,val157.x,val157.y,val150.x,val150.y,val158.x,val158.y);
    float4 wmma0 = __WMMA_8_16_16_half_float(cast7, cast3, make_float4((*(acc0+40)),(*(acc0+41)),(*(acc0+42)),(*(acc0+43))));
    float4 wmma1 = __WMMA_8_16_16_half_float(cast6, cast2, wmma0);
    float4 wmma2 = __WMMA_8_16_16_half_float(cast5, cast1, wmma1);
    float4 wmma3 = __WMMA_8_16_16_half_float(cast4, cast0, wmma2);
    half4 cast8 = make_half4(val1,val9,val17,val25);
    half4 cast9 = make_half4(val33,val41,val49,val57);
    half4 cast10 = make_half4(val65,val73,val81,val89);
    half4 cast11 = make_half4(val97,val105,val113,val121);
    float4 wmma4 = __WMMA_8_16_16_half_float(cast7, cast11, make_float4((*(acc0+48)),(*(acc0+49)),(*(acc0+50)),(*(acc0+51))));
    float4 wmma5 = __WMMA_8_16_16_half_float(cast6, cast10, wmma4);
    float4 wmma6 = __WMMA_8_16_16_half_float(cast5, cast9, wmma5);
    float4 wmma7 = __WMMA_8_16_16_half_float(cast4, cast8, wmma6);
    half4 cast12 = make_half4(val2,val10,val18,val26);
    half4 cast13 = make_half4(val34,val42,val50,val58);
    half4 cast14 = make_half4(val66,val74,val82,val90);
    half4 cast15 = make_half4(val98,val106,val114,val122);
    float4 wmma8 = __WMMA_8_16_16_half_float(cast7, cast15, make_float4((*(acc0+56)),(*(acc0+57)),(*(acc0+58)),(*(acc0+59))));
    float4 wmma9 = __WMMA_8_16_16_half_float(cast6, cast14, wmma8);
    float4 wmma10 = __WMMA_8_16_16_half_float(cast5, cast13, wmma9);
    float4 wmma11 = __WMMA_8_16_16_half_float(cast4, cast12, wmma10);
    half4 cast16 = make_half4(val3,val11,val19,val27);
    half4 cast17 = make_half4(val35,val43,val51,val59);
    half4 cast18 = make_half4(val67,val75,val83,val91);
    half4 cast19 = make_half4(val99,val107,val115,val123);
    float4 wmma12 = __WMMA_8_16_16_half_float(cast7, cast19, make_float4((*(acc0+36)),(*(acc0+37)),(*(acc0+38)),(*(acc0+39))));
    float4 wmma13 = __WMMA_8_16_16_half_float(cast6, cast18, wmma12);
    float4 wmma14 = __WMMA_8_16_16_half_float(cast5, cast17, wmma13);
    float4 wmma15 = __WMMA_8_16_16_half_float(cast4, cast16, wmma14);
    half4 cast20 = make_half4(val4,val12,val20,val28);
    half4 cast21 = make_half4(val36,val44,val52,val60);
    half4 cast22 = make_half4(val68,val76,val84,val92);
    half4 cast23 = make_half4(val100,val108,val116,val124);
    float4 wmma16 = __WMMA_8_16_16_half_float(cast7, cast23, make_float4((*(acc0+44)),(*(acc0+45)),(*(acc0+46)),(*(acc0+47))));
    float4 wmma17 = __WMMA_8_16_16_half_float(cast6, cast22, wmma16);
    float4 wmma18 = __WMMA_8_16_16_half_float(cast5, cast21, wmma17);
    float4 wmma19 = __WMMA_8_16_16_half_float(cast4, cast20, wmma18);
    half4 cast24 = make_half4(val5,val13,val21,val29);
    half4 cast25 = make_half4(val37,val45,val53,val61);
    half4 cast26 = make_half4(val69,val77,val85,val93);
    half4 cast27 = make_half4(val101,val109,val117,val125);
    float4 wmma20 = __WMMA_8_16_16_half_float(cast7, cast27, make_float4((*(acc0+52)),(*(acc0+53)),(*(acc0+54)),(*(acc0+55))));
    float4 wmma21 = __WMMA_8_16_16_half_float(cast6, cast26, wmma20);
    float4 wmma22 = __WMMA_8_16_16_half_float(cast5, cast25, wmma21);
    float4 wmma23 = __WMMA_8_16_16_half_float(cast4, cast24, wmma22);
    half4 cast28 = make_half4(val6,val14,val22,val30);
    half4 cast29 = make_half4(val38,val46,val54,val62);
    half4 cast30 = make_half4(val70,val78,val86,val94);
    half4 cast31 = make_half4(val102,val110,val118,val126);
    float4 wmma24 = __WMMA_8_16_16_half_float(cast7, cast31, make_float4((*(acc0+60)),(*(acc0+61)),(*(acc0+62)),(*(acc0+63))));
    float4 wmma25 = __WMMA_8_16_16_half_float(cast6, cast30, wmma24);
    float4 wmma26 = __WMMA_8_16_16_half_float(cast5, cast29, wmma25);
    float4 wmma27 = __WMMA_8_16_16_half_float(cast4, cast28, wmma26);
    half4 cast32 = make_half4(val31,val39,val47,val55);
    half4 cast33 = make_half4(val63,val71,val79,val87);
    half4 cast34 = make_half4(val95,val103,val111,val119);
    half4 cast35 = make_half4(val127,val7,val15,val23);
    float4 wmma28 = __WMMA_8_16_16_half_float(cast7, cast34, make_float4((*(acc0+32)),(*(acc0+33)),(*(acc0+34)),(*(acc0+35))));
    float4 wmma29 = __WMMA_8_16_16_half_float(cast6, cast33, wmma28);
    float4 wmma30 = __WMMA_8_16_16_half_float(cast5, cast32, wmma29);
    float4 wmma31 = __WMMA_8_16_16_half_float(cast4, cast35, wmma30);
    half8 cast36 = make_half8(val159.x,val159.y,val167.x,val167.y,val160.x,val160.y,val168.x,val168.y);
    half8 cast37 = make_half8(val161.x,val161.y,val169.x,val169.y,val162.x,val162.y,val170.x,val170.y);
    half8 cast38 = make_half8(val163.x,val163.y,val171.x,val171.y,val164.x,val164.y,val172.x,val172.y);
    half8 cast39 = make_half8(val165.x,val165.y,val173.x,val173.y,val166.x,val166.y,val174.x,val174.y);
    float4 wmma32 = __WMMA_8_16_16_half_float(cast39, cast3, make_float4((*(acc0+72)),(*(acc0+73)),(*(acc0+74)),(*(acc0+75))));
    float4 wmma33 = __WMMA_8_16_16_half_float(cast38, cast2, wmma32);
    float4 wmma34 = __WMMA_8_16_16_half_float(cast37, cast1, wmma33);
    float4 wmma35 = __WMMA_8_16_16_half_float(cast36, cast0, wmma34);
    float4 wmma36 = __WMMA_8_16_16_half_float(cast39, cast11, make_float4((*(acc0+80)),(*(acc0+81)),(*(acc0+82)),(*(acc0+83))));
    float4 wmma37 = __WMMA_8_16_16_half_float(cast38, cast10, wmma36);
    float4 wmma38 = __WMMA_8_16_16_half_float(cast37, cast9, wmma37);
    float4 wmma39 = __WMMA_8_16_16_half_float(cast36, cast8, wmma38);
    float4 wmma40 = __WMMA_8_16_16_half_float(cast39, cast15, make_float4((*(acc0+88)),(*(acc0+89)),(*(acc0+90)),(*(acc0+91))));
    float4 wmma41 = __WMMA_8_16_16_half_float(cast38, cast14, wmma40);
    float4 wmma42 = __WMMA_8_16_16_half_float(cast37, cast13, wmma41);
    float4 wmma43 = __WMMA_8_16_16_half_float(cast36, cast12, wmma42);
    float4 wmma44 = __WMMA_8_16_16_half_float(cast39, cast19, make_float4((*(acc0+68)),(*(acc0+69)),(*(acc0+70)),(*(acc0+71))));
    float4 wmma45 = __WMMA_8_16_16_half_float(cast38, cast18, wmma44);
    float4 wmma46 = __WMMA_8_16_16_half_float(cast37, cast17, wmma45);
    float4 wmma47 = __WMMA_8_16_16_half_float(cast36, cast16, wmma46);
    float4 wmma48 = __WMMA_8_16_16_half_float(cast39, cast23, make_float4((*(acc0+76)),(*(acc0+77)),(*(acc0+78)),(*(acc0+79))));
    float4 wmma49 = __WMMA_8_16_16_half_float(cast38, cast22, wmma48);
    float4 wmma50 = __WMMA_8_16_16_half_float(cast37, cast21, wmma49);
    float4 wmma51 = __WMMA_8_16_16_half_float(cast36, cast20, wmma50);
    float4 wmma52 = __WMMA_8_16_16_half_float(cast39, cast27, make_float4((*(acc0+84)),(*(acc0+85)),(*(acc0+86)),(*(acc0+87))));
    float4 wmma53 = __WMMA_8_16_16_half_float(cast38, cast26, wmma52);
    float4 wmma54 = __WMMA_8_16_16_half_float(cast37, cast25, wmma53);
    float4 wmma55 = __WMMA_8_16_16_half_float(cast36, cast24, wmma54);
    float4 wmma56 = __WMMA_8_16_16_half_float(cast39, cast31, make_float4((*(acc0+92)),(*(acc0+93)),(*(acc0+94)),(*(acc0+95))));
    float4 wmma57 = __WMMA_8_16_16_half_float(cast38, cast30, wmma56);
    float4 wmma58 = __WMMA_8_16_16_half_float(cast37, cast29, wmma57);
    float4 wmma59 = __WMMA_8_16_16_half_float(cast36, cast28, wmma58);
    float4 wmma60 = __WMMA_8_16_16_half_float(cast39, cast34, make_float4((*(acc0+64)),(*(acc0+65)),(*(acc0+66)),(*(acc0+67))));
    float4 wmma61 = __WMMA_8_16_16_half_float(cast38, cast33, wmma60);
    float4 wmma62 = __WMMA_8_16_16_half_float(cast37, cast32, wmma61);
    float4 wmma63 = __WMMA_8_16_16_half_float(cast36, cast35, wmma62);
    half8 cast40 = make_half8(val175.x,val175.y,val183.x,val183.y,val176.x,val176.y,val184.x,val184.y);
    half8 cast41 = make_half8(val177.x,val177.y,val185.x,val185.y,val178.x,val178.y,val186.x,val186.y);
    half8 cast42 = make_half8(val179.x,val179.y,val187.x,val187.y,val180.x,val180.y,val188.x,val188.y);
    half8 cast43 = make_half8(val181.x,val181.y,val189.x,val189.y,val182.x,val182.y,val190.x,val190.y);
    float4 wmma64 = __WMMA_8_16_16_half_float(cast43, cast3, make_float4((*(acc0+104)),(*(acc0+105)),(*(acc0+106)),(*(acc0+107))));
    float4 wmma65 = __WMMA_8_16_16_half_float(cast42, cast2, wmma64);
    float4 wmma66 = __WMMA_8_16_16_half_float(cast41, cast1, wmma65);
    float4 wmma67 = __WMMA_8_16_16_half_float(cast40, cast0, wmma66);
    float4 wmma68 = __WMMA_8_16_16_half_float(cast43, cast11, make_float4((*(acc0+112)),(*(acc0+113)),(*(acc0+114)),(*(acc0+115))));
    float4 wmma69 = __WMMA_8_16_16_half_float(cast42, cast10, wmma68);
    float4 wmma70 = __WMMA_8_16_16_half_float(cast41, cast9, wmma69);
    float4 wmma71 = __WMMA_8_16_16_half_float(cast40, cast8, wmma70);
    float4 wmma72 = __WMMA_8_16_16_half_float(cast43, cast15, make_float4((*(acc0+120)),(*(acc0+121)),(*(acc0+122)),(*(acc0+123))));
    float4 wmma73 = __WMMA_8_16_16_half_float(cast42, cast14, wmma72);
    float4 wmma74 = __WMMA_8_16_16_half_float(cast41, cast13, wmma73);
    float4 wmma75 = __WMMA_8_16_16_half_float(cast40, cast12, wmma74);
    float4 wmma76 = __WMMA_8_16_16_half_float(cast43, cast19, make_float4((*(acc0+100)),(*(acc0+101)),(*(acc0+102)),(*(acc0+103))));
    float4 wmma77 = __WMMA_8_16_16_half_float(cast42, cast18, wmma76);
    float4 wmma78 = __WMMA_8_16_16_half_float(cast41, cast17, wmma77);
    float4 wmma79 = __WMMA_8_16_16_half_float(cast40, cast16, wmma78);
    float4 wmma80 = __WMMA_8_16_16_half_float(cast43, cast23, make_float4((*(acc0+108)),(*(acc0+109)),(*(acc0+110)),(*(acc0+111))));
    float4 wmma81 = __WMMA_8_16_16_half_float(cast42, cast22, wmma80);
    float4 wmma82 = __WMMA_8_16_16_half_float(cast41, cast21, wmma81);
    float4 wmma83 = __WMMA_8_16_16_half_float(cast40, cast20, wmma82);
    float4 wmma84 = __WMMA_8_16_16_half_float(cast43, cast27, make_float4((*(acc0+116)),(*(acc0+117)),(*(acc0+118)),(*(acc0+119))));
    float4 wmma85 = __WMMA_8_16_16_half_float(cast42, cast26, wmma84);
    float4 wmma86 = __WMMA_8_16_16_half_float(cast41, cast25, wmma85);
    float4 wmma87 = __WMMA_8_16_16_half_float(cast40, cast24, wmma86);
    float4 wmma88 = __WMMA_8_16_16_half_float(cast43, cast31, make_float4((*(acc0+124)),(*(acc0+125)),(*(acc0+126)),(*(acc0+127))));
    float4 wmma89 = __WMMA_8_16_16_half_float(cast42, cast30, wmma88);
    float4 wmma90 = __WMMA_8_16_16_half_float(cast41, cast29, wmma89);
    float4 wmma91 = __WMMA_8_16_16_half_float(cast40, cast28, wmma90);
    float4 wmma92 = __WMMA_8_16_16_half_float(cast43, cast34, make_float4((*(acc0+96)),(*(acc0+97)),(*(acc0+98)),(*(acc0+99))));
    float4 wmma93 = __WMMA_8_16_16_half_float(cast42, cast33, wmma92);
    float4 wmma94 = __WMMA_8_16_16_half_float(cast41, cast32, wmma93);
    float4 wmma95 = __WMMA_8_16_16_half_float(cast40, cast35, wmma94);
    half8 cast44 = make_half8(val129.x,val129.y,val137.x,val137.y,val130.x,val130.y,val138.x,val138.y);
    half8 cast45 = make_half8(val131.x,val131.y,val139.x,val139.y,val132.x,val132.y,val140.x,val140.y);
    half8 cast46 = make_half8(val133.x,val133.y,val141.x,val141.y,val134.x,val134.y,val142.x,val142.y);
    half8 cast47 = make_half8(val191.x,val191.y,val135.x,val135.y,val128.x,val128.y,val136.x,val136.y);
    float4 wmma96 = __WMMA_8_16_16_half_float(cast46, cast3, make_float4((*(acc0+8)),(*(acc0+9)),(*(acc0+10)),(*(acc0+11))));
    float4 wmma97 = __WMMA_8_16_16_half_float(cast45, cast2, wmma96);
    float4 wmma98 = __WMMA_8_16_16_half_float(cast44, cast1, wmma97);
    float4 wmma99 = __WMMA_8_16_16_half_float(cast47, cast0, wmma98);
    float4 wmma100 = __WMMA_8_16_16_half_float(cast46, cast11, make_float4((*(acc0+16)),(*(acc0+17)),(*(acc0+18)),(*(acc0+19))));
    float4 wmma101 = __WMMA_8_16_16_half_float(cast45, cast10, wmma100);
    float4 wmma102 = __WMMA_8_16_16_half_float(cast44, cast9, wmma101);
    float4 wmma103 = __WMMA_8_16_16_half_float(cast47, cast8, wmma102);
    float4 wmma104 = __WMMA_8_16_16_half_float(cast46, cast15, make_float4((*(acc0+24)),(*(acc0+25)),(*(acc0+26)),(*(acc0+27))));
    float4 wmma105 = __WMMA_8_16_16_half_float(cast45, cast14, wmma104);
    float4 wmma106 = __WMMA_8_16_16_half_float(cast44, cast13, wmma105);
    float4 wmma107 = __WMMA_8_16_16_half_float(cast47, cast12, wmma106);
    float4 wmma108 = __WMMA_8_16_16_half_float(cast46, cast19, make_float4((*(acc0+4)),(*(acc0+5)),(*(acc0+6)),(*(acc0+7))));
    float4 wmma109 = __WMMA_8_16_16_half_float(cast45, cast18, wmma108);
    float4 wmma110 = __WMMA_8_16_16_half_float(cast44, cast17, wmma109);
    float4 wmma111 = __WMMA_8_16_16_half_float(cast47, cast16, wmma110);
    float4 wmma112 = __WMMA_8_16_16_half_float(cast46, cast23, make_float4((*(acc0+12)),(*(acc0+13)),(*(acc0+14)),(*(acc0+15))));
    float4 wmma113 = __WMMA_8_16_16_half_float(cast45, cast22, wmma112);
    float4 wmma114 = __WMMA_8_16_16_half_float(cast44, cast21, wmma113);
    float4 wmma115 = __WMMA_8_16_16_half_float(cast47, cast20, wmma114);
    float4 wmma116 = __WMMA_8_16_16_half_float(cast46, cast27, make_float4((*(acc0+20)),(*(acc0+21)),(*(acc0+22)),(*(acc0+23))));
    float4 wmma117 = __WMMA_8_16_16_half_float(cast45, cast26, wmma116);
    float4 wmma118 = __WMMA_8_16_16_half_float(cast44, cast25, wmma117);
    float4 wmma119 = __WMMA_8_16_16_half_float(cast47, cast24, wmma118);
    float4 wmma120 = __WMMA_8_16_16_half_float(cast46, cast31, make_float4((*(acc0+28)),(*(acc0+29)),(*(acc0+30)),(*(acc0+31))));
    float4 wmma121 = __WMMA_8_16_16_half_float(cast45, cast30, wmma120);
    float4 wmma122 = __WMMA_8_16_16_half_float(cast44, cast29, wmma121);
    float4 wmma123 = __WMMA_8_16_16_half_float(cast47, cast28, wmma122);
    float4 wmma124 = __WMMA_8_16_16_half_float(cast46, cast34, make_float4((*(acc0+0)),(*(acc0+1)),(*(acc0+2)),(*(acc0+3))));
    float4 wmma125 = __WMMA_8_16_16_half_float(cast45, cast33, wmma124);
    float4 wmma126 = __WMMA_8_16_16_half_float(cast44, cast32, wmma125);
    float4 wmma127 = __WMMA_8_16_16_half_float(cast47, cast35, wmma126);
    *(acc0+0) = wmma127.x;
    *(acc0+1) = wmma127.y;
    *(acc0+2) = wmma127.z;
    *(acc0+3) = wmma127.w;
    *(acc0+4) = wmma111.x;
    *(acc0+5) = wmma111.y;
    *(acc0+6) = wmma111.z;
    *(acc0+7) = wmma111.w;
    *(acc0+8) = wmma99.x;
    *(acc0+9) = wmma99.y;
    *(acc0+10) = wmma99.z;
    *(acc0+11) = wmma99.w;
    *(acc0+12) = wmma115.x;
    *(acc0+13) = wmma115.y;
    *(acc0+14) = wmma115.z;
    *(acc0+15) = wmma115.w;
    *(acc0+16) = wmma103.x;
    *(acc0+17) = wmma103.y;
    *(acc0+18) = wmma103.z;
    *(acc0+19) = wmma103.w;
    *(acc0+20) = wmma119.x;
    *(acc0+21) = wmma119.y;
    *(acc0+22) = wmma119.z;
    *(acc0+23) = wmma119.w;
    *(acc0+24) = wmma107.x;
    *(acc0+25) = wmma107.y;
    *(acc0+26) = wmma107.z;
    *(acc0+27) = wmma107.w;
    *(acc0+28) = wmma123.x;
    *(acc0+29) = wmma123.y;
    *(acc0+30) = wmma123.z;
    *(acc0+31) = wmma123.w;
    *(acc0+32) = wmma31.x;
    *(acc0+33) = wmma31.y;
    *(acc0+34) = wmma31.z;
    *(acc0+35) = wmma31.w;
    *(acc0+36) = wmma15.x;
    *(acc0+37) = wmma15.y;
    *(acc0+38) = wmma15.z;
    *(acc0+39) = wmma15.w;
    *(acc0+40) = wmma3.x;
    *(acc0+41) = wmma3.y;
    *(acc0+42) = wmma3.z;
    *(acc0+43) = wmma3.w;
    *(acc0+44) = wmma19.x;
    *(acc0+45) = wmma19.y;
    *(acc0+46) = wmma19.z;
    *(acc0+47) = wmma19.w;
    *(acc0+48) = wmma7.x;
    *(acc0+49) = wmma7.y;
    *(acc0+50) = wmma7.z;
    *(acc0+51) = wmma7.w;
    *(acc0+52) = wmma23.x;
    *(acc0+53) = wmma23.y;
    *(acc0+54) = wmma23.z;
    *(acc0+55) = wmma23.w;
    *(acc0+56) = wmma11.x;
    *(acc0+57) = wmma11.y;
    *(acc0+58) = wmma11.z;
    *(acc0+59) = wmma11.w;
    *(acc0+60) = wmma27.x;
    *(acc0+61) = wmma27.y;
    *(acc0+62) = wmma27.z;
    *(acc0+63) = wmma27.w;
    *(acc0+64) = wmma63.x;
    *(acc0+65) = wmma63.y;
    *(acc0+66) = wmma63.z;
    *(acc0+67) = wmma63.w;
    *(acc0+68) = wmma47.x;
    *(acc0+69) = wmma47.y;
    *(acc0+70) = wmma47.z;
    *(acc0+71) = wmma47.w;
    *(acc0+72) = wmma35.x;
    *(acc0+73) = wmma35.y;
    *(acc0+74) = wmma35.z;
    *(acc0+75) = wmma35.w;
    *(acc0+76) = wmma51.x;
    *(acc0+77) = wmma51.y;
    *(acc0+78) = wmma51.z;
    *(acc0+79) = wmma51.w;
    *(acc0+80) = wmma39.x;
    *(acc0+81) = wmma39.y;
    *(acc0+82) = wmma39.z;
    *(acc0+83) = wmma39.w;
    *(acc0+84) = wmma55.x;
    *(acc0+85) = wmma55.y;
    *(acc0+86) = wmma55.z;
    *(acc0+87) = wmma55.w;
    *(acc0+88) = wmma43.x;
    *(acc0+89) = wmma43.y;
    *(acc0+90) = wmma43.z;
    *(acc0+91) = wmma43.w;
    *(acc0+92) = wmma59.x;
    *(acc0+93) = wmma59.y;
    *(acc0+94) = wmma59.z;
    *(acc0+95) = wmma59.w;
    *(acc0+96) = wmma95.x;
    *(acc0+97) = wmma95.y;
    *(acc0+98) = wmma95.z;
    *(acc0+99) = wmma95.w;
    *(acc0+100) = wmma79.x;
    *(acc0+101) = wmma79.y;
    *(acc0+102) = wmma79.z;
    *(acc0+103) = wmma79.w;
    *(acc0+104) = wmma67.x;
    *(acc0+105) = wmma67.y;
    *(acc0+106) = wmma67.z;
    *(acc0+107) = wmma67.w;
    *(acc0+108) = wmma83.x;
    *(acc0+109) = wmma83.y;
    *(acc0+110) = wmma83.z;
    *(acc0+111) = wmma83.w;
    *(acc0+112) = wmma71.x;
    *(acc0+113) = wmma71.y;
    *(acc0+114) = wmma71.z;
    *(acc0+115) = wmma71.w;
    *(acc0+116) = wmma87.x;
    *(acc0+117) = wmma87.y;
    *(acc0+118) = wmma87.z;
    *(acc0+119) = wmma87.w;
    *(acc0+120) = wmma75.x;
    *(acc0+121) = wmma75.y;
    *(acc0+122) = wmma75.z;
    *(acc0+123) = wmma75.w;
    *(acc0+124) = wmma91.x;
    *(acc0+125) = wmma91.y;
    *(acc0+126) = wmma91.z;
    *(acc0+127) = wmma91.w;
  }
  int alu268 = (alu4+alu0+alu8+alu6);
  *((half2*)((data0_16777216+(alu268+8)))) = make_half2(((half)((*(acc0+8)))),((half)((*(acc0+9)))));
  *((half2*)((data0_16777216+(alu268+16)))) = make_half2(((half)((*(acc0+16)))),((half)((*(acc0+17)))));
  *((half2*)((data0_16777216+(alu268+24)))) = make_half2(((half)((*(acc0+24)))),((half)((*(acc0+25)))));
  *((half2*)((data0_16777216+(alu268+32)))) = make_half2(((half)((*(acc0+4)))),((half)((*(acc0+5)))));
  *((half2*)((data0_16777216+(alu268+40)))) = make_half2(((half)((*(acc0+12)))),((half)((*(acc0+13)))));
  *((half2*)((data0_16777216+(alu268+48)))) = make_half2(((half)((*(acc0+20)))),((half)((*(acc0+21)))));
  *((half2*)((data0_16777216+(alu268+56)))) = make_half2(((half)((*(acc0+28)))),((half)((*(acc0+29)))));
  *((half2*)((data0_16777216+(alu268+32768)))) = make_half2(((half)((*(acc0+2)))),((half)((*(acc0+3)))));
  *((half2*)((data0_16777216+(alu268+32776)))) = make_half2(((half)((*(acc0+10)))),((half)((*(acc0+11)))));
  *((half2*)((data0_16777216+(alu268+32784)))) = make_half2(((half)((*(acc0+18)))),((half)((*(acc0+19)))));
  *((half2*)((data0_16777216+(alu268+32792)))) = make_half2(((half)((*(acc0+26)))),((half)((*(acc0+27)))));
  *((half2*)((data0_16777216+(alu268+32800)))) = make_half2(((half)((*(acc0+6)))),((half)((*(acc0+7)))));
  *((half2*)((data0_16777216+(alu268+32808)))) = make_half2(((half)((*(acc0+14)))),((half)((*(acc0+15)))));
  *((half2*)((data0_16777216+(alu268+32816)))) = make_half2(((half)((*(acc0+22)))),((half)((*(acc0+23)))));
  *((half2*)((data0_16777216+(alu268+32824)))) = make_half2(((half)((*(acc0+30)))),((half)((*(acc0+31)))));
  *((half2*)((data0_16777216+(alu268+65536)))) = make_half2(((half)((*(acc0+32)))),((half)((*(acc0+33)))));
  *((half2*)((data0_16777216+(alu268+65544)))) = make_half2(((half)((*(acc0+40)))),((half)((*(acc0+41)))));
  *((half2*)((data0_16777216+(alu268+65552)))) = make_half2(((half)((*(acc0+48)))),((half)((*(acc0+49)))));
  *((half2*)((data0_16777216+(alu268+65560)))) = make_half2(((half)((*(acc0+56)))),((half)((*(acc0+57)))));
  *((half2*)((data0_16777216+(alu268+65568)))) = make_half2(((half)((*(acc0+36)))),((half)((*(acc0+37)))));
  *((half2*)((data0_16777216+(alu268+65576)))) = make_half2(((half)((*(acc0+44)))),((half)((*(acc0+45)))));
  *((half2*)((data0_16777216+(alu268+65584)))) = make_half2(((half)((*(acc0+52)))),((half)((*(acc0+53)))));
  *((half2*)((data0_16777216+(alu268+65592)))) = make_half2(((half)((*(acc0+60)))),((half)((*(acc0+61)))));
  *((half2*)((data0_16777216+(alu268+98304)))) = make_half2(((half)((*(acc0+34)))),((half)((*(acc0+35)))));
  *((half2*)((data0_16777216+(alu268+98312)))) = make_half2(((half)((*(acc0+42)))),((half)((*(acc0+43)))));
  *((half2*)((data0_16777216+(alu268+98320)))) = make_half2(((half)((*(acc0+50)))),((half)((*(acc0+51)))));
  *((half2*)((data0_16777216+(alu268+98328)))) = make_half2(((half)((*(acc0+58)))),((half)((*(acc0+59)))));
  *((half2*)((data0_16777216+(alu268+98336)))) = make_half2(((half)((*(acc0+38)))),((half)((*(acc0+39)))));
  *((half2*)((data0_16777216+(alu268+98344)))) = make_half2(((half)((*(acc0+46)))),((half)((*(acc0+47)))));
  *((half2*)((data0_16777216+(alu268+98352)))) = make_half2(((half)((*(acc0+54)))),((half)((*(acc0+55)))));
  *((half2*)((data0_16777216+(alu268+98360)))) = make_half2(((half)((*(acc0+62)))),((half)((*(acc0+63)))));
  *((half2*)((data0_16777216+(alu268+131072)))) = make_half2(((half)((*(acc0+64)))),((half)((*(acc0+65)))));
  *((half2*)((data0_16777216+(alu268+131080)))) = make_half2(((half)((*(acc0+72)))),((half)((*(acc0+73)))));
  *((half2*)((data0_16777216+(alu268+131088)))) = make_half2(((half)((*(acc0+80)))),((half)((*(acc0+81)))));
  *((half2*)((data0_16777216+(alu268+131096)))) = make_half2(((half)((*(acc0+88)))),((half)((*(acc0+89)))));
  *((half2*)((data0_16777216+(alu268+131104)))) = make_half2(((half)((*(acc0+68)))),((half)((*(acc0+69)))));
  *((half2*)((data0_16777216+(alu268+131112)))) = make_half2(((half)((*(acc0+76)))),((half)((*(acc0+77)))));
  *((half2*)((data0_16777216+(alu268+131120)))) = make_half2(((half)((*(acc0+84)))),((half)((*(acc0+85)))));
  *((half2*)((data0_16777216+(alu268+131128)))) = make_half2(((half)((*(acc0+92)))),((half)((*(acc0+93)))));
  *((half2*)((data0_16777216+(alu268+163840)))) = make_half2(((half)((*(acc0+66)))),((half)((*(acc0+67)))));
  *((half2*)((data0_16777216+(alu268+163848)))) = make_half2(((half)((*(acc0+74)))),((half)((*(acc0+75)))));
  *((half2*)((data0_16777216+(alu268+163856)))) = make_half2(((half)((*(acc0+82)))),((half)((*(acc0+83)))));
  *((half2*)((data0_16777216+(alu268+163864)))) = make_half2(((half)((*(acc0+90)))),((half)((*(acc0+91)))));
  *((half2*)((data0_16777216+(alu268+163872)))) = make_half2(((half)((*(acc0+70)))),((half)((*(acc0+71)))));
  *((half2*)((data0_16777216+(alu268+163880)))) = make_half2(((half)((*(acc0+78)))),((half)((*(acc0+79)))));
  *((half2*)((data0_16777216+(alu268+163888)))) = make_half2(((half)((*(acc0+86)))),((half)((*(acc0+87)))));
  *((half2*)((data0_16777216+(alu268+163896)))) = make_half2(((half)((*(acc0+94)))),((half)((*(acc0+95)))));
  *((half2*)((data0_16777216+(alu268+196608)))) = make_half2(((half)((*(acc0+96)))),((half)((*(acc0+97)))));
  *((half2*)((data0_16777216+(alu268+196616)))) = make_half2(((half)((*(acc0+104)))),((half)((*(acc0+105)))));
  *((half2*)((data0_16777216+(alu268+196624)))) = make_half2(((half)((*(acc0+112)))),((half)((*(acc0+113)))));
  *((half2*)((data0_16777216+(alu268+196632)))) = make_half2(((half)((*(acc0+120)))),((half)((*(acc0+121)))));
  *((half2*)((data0_16777216+(alu268+196640)))) = make_half2(((half)((*(acc0+100)))),((half)((*(acc0+101)))));
  *((half2*)((data0_16777216+(alu268+196648)))) = make_half2(((half)((*(acc0+108)))),((half)((*(acc0+109)))));
  *((half2*)((data0_16777216+(alu268+196656)))) = make_half2(((half)((*(acc0+116)))),((half)((*(acc0+117)))));
  *((half2*)((data0_16777216+(alu268+196664)))) = make_half2(((half)((*(acc0+124)))),((half)((*(acc0+125)))));
  *((half2*)((data0_16777216+(alu268+229376)))) = make_half2(((half)((*(acc0+98)))),((half)((*(acc0+99)))));
  *((half2*)((data0_16777216+(alu268+229384)))) = make_half2(((half)((*(acc0+106)))),((half)((*(acc0+107)))));
  *((half2*)((data0_16777216+(alu268+229392)))) = make_half2(((half)((*(acc0+114)))),((half)((*(acc0+115)))));
  *((half2*)((data0_16777216+(alu268+229400)))) = make_half2(((half)((*(acc0+122)))),((half)((*(acc0+123)))));
  *((half2*)((data0_16777216+(alu268+229408)))) = make_half2(((half)((*(acc0+102)))),((half)((*(acc0+103)))));
  *((half2*)((data0_16777216+(alu268+229416)))) = make_half2(((half)((*(acc0+110)))),((half)((*(acc0+111)))));
  *((half2*)((data0_16777216+(alu268+229424)))) = make_half2(((half)((*(acc0+118)))),((half)((*(acc0+119)))));
  *((half2*)((data0_16777216+(alu268+229432)))) = make_half2(((half)((*(acc0+126)))),((half)((*(acc0+127)))));
  *((half2*)((data0_16777216+alu268))) = make_half2(((half)((*(acc0+0)))),((half)((*(acc0+1)))));
}
*** NV         3 r_32_32_32_2_2_2_2_4_4_2_64_2_4                arg  3 mem   0.10 GB tm   1037.18us/    10.92ms (    133 TFLOPS   97|4173   GB/s) ['matmul']
scheduled    1 kernels in     0.36 ms |  cache hit 8f2fa1f5 | 2923 uops in cache
*** NV         4 r_32_32_32_2_2_2_2_4_4_2_64_2_4                arg  3 mem   0.13 GB tm   1038.21us/    11.96ms (    132 TFLOPS   97|4169   GB/s) ['matmul']
scheduled    1 kernels in     0.32 ms |  cache hit 8f2fa1f5 | 2923 uops in cache