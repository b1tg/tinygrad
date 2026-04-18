# MUSA / MTT S4000 知识卡

本文档汇总为 tinygrad 适配 MUSA 过程中调研到的所有事实，作为后续工作的参考。

所有数据测自 **MTT S4000 + MUSA SDK 3.1.0 + driver 2.7.0** (Ubuntu 22.04，2026-04-18)。

---

## 硬件规格（来自 `musaInfo`）

```
Name:                             MTT S4000
compute capability (major.minor): 2.2         → arch flag mp_22
multiProcessorCount:              56
maxThreadsPerMultiProcessor:      6144
maxThreadsPerBlock:               1024
maxThreadsDim.x/y/z:              1024/1024/1024
maxGridSize.x/y/z:                2147483647/2147483647/2147483647
warpSize:                         128          ← 注意！非 32 / 64
sharedMemPerBlock:                72 KB
sharedMemPerMultiprocessor:       72 KB
regsPerBlock:                     262144
totalGlobalMem:                   47.91 GB
totalConstMem:                    8192
memoryBusWidth:                   256 bit
memoryClockRate:                  8012 MHz
clockRate:                        1679.99 MHz
l2CacheSize:                      24 MB
concurrentKernels:                1
isMultiGpuBoard:                  1
```

`warpSize=128` 与 NVIDIA (32) / AMD (64) 都不同，BEAM/tensor_cores 的 hardcoded 假设需要注意。

## SDK 安装布局

```
/usr/local/musa                     (symlink → musa-3.1.0)
├── bin/
│   ├── mcc                         (clang 14 fork；mcc --version → clang version 14.0.0)
│   ├── musaInfo                    (设备能力查询，类似 deviceQuery)
│   ├── muMemcpyTest
│   ├── musa_*_version              (各组件版本)
│   ├── musify_version              (CUDA→MUSA 翻译工具版本)
│   └── clang / llvm-* / mlir-*     (完整 LLVM 工具链)
├── include/
│   ├── musa.h                      (driver API, 19459 行; mu* 前缀)
│   ├── musa_runtime_api.h          (runtime API, musa* 前缀)
│   ├── musa_runtime.h              (主入口)
│   ├── musa_fp16.h / musa_fp16_mtgpu.h / musa_bf16.h
│   ├── musa_fp4/6/8.h              (更低精度)
│   ├── musaTypedefs.h / driver_types.h
│   ├── mublas.h / mudnn.h / mccl.h / mufft.h / murand.h / musparse-functions.h
│   ├── mma.h                       (warp-level matrix ops 声明)
│   ├── cub/ cooperative_groups.h / thrust/   (CUDA 风格模板库移植)
│   └── mupti*.h                    (profiling/tracing)
└── lib/
    ├── libmusa.so.1.0.0            (driver API, 4.4 MB; 265 T-symbols 以 mu* 开头)
    ├── libmusart.so.1.0.0          (runtime API, 222 KB; thin wrapper)
    ├── libmublas.so / libmudnn.so.2.7 / libmccl.so.2.11
    ├── libmufft.so / libmurand.so / libmusolver.so / libmusparse.so
    ├── libmupp*.so                 (performance primitives, NPP 的 MUSA 版)
    └── libLLVM*.a / libclang*.so   (clang/LLVM 静态库供 mcc 用)
```

环境变量默认：`LD_LIBRARY_PATH=/usr/local/musa/lib`，`PATH=/usr/local/musa/bin:...`

## MUSA 软件生态（官方，非 tinygrad 相关）

| 组件 | 对应 CUDA | 说明 |
|---|---|---|
| **mcc** | nvcc | clang 14 fork 编译器；支持 `-x musa -mtgpu` |
| **libmusa** | libcuda | driver API，dlopen 目标 |
| **libmusart** | libcudart | runtime API，thin wrapper |
| **muBLAS** | cuBLAS | GEMM |
| **muDNN** | cuDNN | DNN ops |
| **MCCL** | NCCL | 集合通信 |
| **muFFT / muRAND / muSparse / muSolver** | cuFFT / … | 数学库 |
| **muPP** | NPP | 性能原语 |
| **muTriton** | Triton (port) | tile DSL |
| **Musify** | n/a | 源码 CUDA→MUSA 翻译工具 |
| **mupti** | CUPTI | profiling callback |

## API 命名规则

**完全 1:1 镜像** libcuda。把前缀 `cu` 换成 `mu`，类型前缀 `CU` 换成 `MU`，运行时前缀 `cuda` 换成 `musa`。`_v2` 后缀也保留。

### Driver API（`libmusa.so` / `musa.h`）

| CUDA | MUSA | 用途 |
|---|---|---|
| `cuInit` | `muInit` | 初始化 |
| `cuDeviceGet/Count/Attribute/Name/TotalMem/ComputeCapability` | `muDevice*` | 设备查询 |
| `cuDeviceCanAccessPeer` | `muDeviceCanAccessPeer` | P2P 探测 |
| `cuCtxCreate_v2 / SetCurrent / Synchronize / Destroy_v2 / EnablePeerAccess` | `muCtx*` | 上下文 |
| `cuModuleLoadData / GetFunction / Unload` | `muModule*` | fatbin 加载 |
| `cuMemAlloc_v2 / Free_v2 / HostAlloc / FreeHost` | `muMem*` | 显存/锁页 |
| `cuMemcpyHtoD_v2 / DtoH_v2 / DtoD_v2 / *Async_v2` | `muMemcpy*` | 拷贝 |
| `cuLaunchKernel` | `muLaunchKernel` | kernel 启动 |
| `cuStreamCreate / Destroy / Synchronize / WaitEvent` | `muStream*` | stream |
| `cuEventCreate / Record / Synchronize / ElapsedTime / Destroy_v2` | `muEvent*` | 事件计时 |
| `cuFuncSetAttribute` | `muFuncSetAttribute` | 动态 smem 等 |
| `cuGetErrorString / cuGetErrorName` | `muGetErrorString / muGetErrorName` | 错误码→字符串 |
| `cuGraph*` | `muGraph*` | command graph（见符号表，tinygrad 未用） |

全部 265 个 driver 导出符号在 `libmusa.so`，autogen 的 `musa.py` 覆盖。

### Runtime API（`libmusart.so` / `musa_runtime_api.h`）

`cudaMalloc → musaMalloc`，`cudaMemcpy → musaMemcpy`，`cudaLaunchKernel → musaLaunchKernel` 等。tinygrad 目前只用 driver API，runtime API 未绑定。

## 关键编译器 macros（`mcc -dM -E`）

```
__MUSACC__            (空定义)       device 编译时
__MUSA_ARCH__         220            mp_22 的 ISA 版本
__MUSA__              1              MUSA 语言
__device__            __location__(device)
MUSART_DEVICE         __device__
```

`__MUSA_ARCH__` 版本号与 arch flag 对应关系推测：
- `mp_22` → `__MUSA_ARCH__ == 220`
- `mp_31` → 应为 `310`（未验证，S4000 之后的 Quyin 芯片）
- `__MUSA_ARCH__ >= 800` guard 在 musa_fp16.h 中出现，表明有未来 arch 预留

## `mcc` 命令行

### 设备代码编译（tinygrad 使用）
```bash
mcc -x musa -mtgpu --offload-arch=mp_22 -O2 --cuda-device-only -o out.fatbin src.mu
```

- `-x musa` — 语言方言
- `-mtgpu` — 启用 MTT GPU target
- `--offload-arch=mp_22` — 目标架构（canonical flag；llama.cpp PR 的 `-fmusa-rdc`/`--cuda-gpu-arch` 在此 SDK 不可用）
- `-O2` — 优化级别
- `--cuda-device-only` — 只出 device 端 fatbin，不生成 host stub
- 输出：fatbin 格式（`file` 报 "data"），可被 `muModuleLoadData` 直接加载

### 有用的调试 flags
- `-dM -E` 打印所有预定义 macro
- `mcc -v ...` 打印完整调用链（clang driver → cc1 → mtgpu-link）
- `mcc --help` / `mcc --help-hidden` 列完整选项
- 错误默认到 stderr，tinygrad 的 `system()` 把 stderr 并进 stdout

### 不可用 flags
- `-fmusa-rdc` (llama.cpp PR 用的，此 SDK 报 unknown argument)
- PTX 相关（`--gpu-architecture=sm_*`）— MUSA 不用 PTX 中间表示

## 半精度 intrinsic 可用性矩阵（mp_22）

| 函数 | 声明 | 定义（body） | 状态 |
|---|---|---|---|
| `htrunc` | ✓ (fp16.h:1104) | ✗ | 不可用 |
| `hsqrt` | ✓ (fp16.h:3178) | ✓ (fp16_mtgpu.h) | 仅 fp16 OK |
| `hrcp`  | ✓ (fp16.h:3209) | ✓ (fp16_mtgpu.h) | 仅 fp16 OK |
| `hlog2` | ✓ (fp16.h:3239) | ✓ (fp16_mtgpu.h) | 仅 fp16 OK |
| `hexp2` | **✗ (`__MUSA_ARCH__ >= 800` guarded out)** | ✗ | 不可用 |
| `hsin`  | **✗ (同上 guard)** | ✗ | 不可用 |

因此 MUSARenderer 统一用 fp32 桥接这 6 个 op，牺牲一点精度/性能换正确性。

## 数据类型

| CUDA 类型 | MUSA 类型 | 头文件 |
|---|---|---|
| `half` / `__half` | `half` / `__half` | `musa_fp16.h` |
| `nv_bfloat16` | `__mt_bfloat16` | `musa_bf16.h` |
| `__nv_fp8_e4m3` | 待查 | `musa_fp8.h` |
| `__nv_fp8_e5m2` | 待查 | `musa_fp8.h` |

MUSARenderer 里 `type_map = {dtypes.bfloat16: "__mt_bfloat16"}`，half 名字相同不用覆盖。

## 进程模型 & 启动语法

MUSA kernel 使用 **CUDA 完全相同的 `<<<grid, block>>>` 和 `__global__/__device__` 语法**。`blockIdx/threadIdx/blockDim/gridDim` 都有。`__syncthreads()` / `__shfl_sync` / `__shared__ __align__` 都一样。

这意味着 tinygrad 的 `CUDARenderer` 输出的 kernel 文本 **未经修改即可被 mcc 编译**，只需改 `#include` 头名 + bf16 类型名。

## 启动 vector add（已验证）

```bash
cat > vadd.mu <<EOF
extern "C" __global__ void vadd(float *a, float *b, float *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}
EOF
mcc -x musa -mtgpu --offload-arch=mp_22 -O2 --cuda-device-only -o vadd.fatbin vadd.mu
python extra/musa/probe.py    # expects [0, 11, 22, ..., 165]
```

完整 ctypes 链路：`dlopen(libmusa.so)` → `muInit(0)` → `muDeviceGet` → `muCtxCreate_v2` → open("vadd.fatbin") → `muModuleLoadData(img)` → `muModuleGetFunction("vadd")` → `muMemAlloc_v2` ×3 → `muMemcpyHtoD_v2` ×2 → `muLaunchKernel(fn, 1,1,1, N,1,1, 0, NULL, args, NULL)` → `muCtxSynchronize` → `muMemcpyDtoH_v2`。

## BEAM 已知问题

`JITBEAM=2` 在大模型（Qwen3.5-9B+）上会触发硬件级 fault，`muLaunchKernel` 调用直接 SIGSEGV 整个 Python 进程。search.py 的 `try/except RuntimeError` 捕不了 signal。诊断：BEAM 探索的某些 opt 组合超出 mp_22 资源限制（register / smem / local size），驱动抛硬件异常。

已知安全范围：
- 小模型（0.6B）JITBEAM=2 稳定，实测提速 55%（31 → 48 tok/s）
- 中型/大型模型不开 BEAM

未来根治方案：`_time_program` 改为 `multiprocessing.Process` 子进程隔离，SIGSEGV 只杀 worker。

## 参考资料

- 官方 SDK 下载：https://developer.mthreads.com/sdk/download/musa
- MTT S4000 产品页：https://www.mthreads.com/product/S4000
- MTT S4000 文档中心：https://docs.mthreads.com/en/s4000/s4000-doc-online/
- llama.cpp MUSA PR #8383（API 宏替换范例）：https://github.com/ggml-org/llama.cpp/pull/8383
- torch_musa（PyTorch 扩展）：https://github.com/MooreThreads/torch_musa
- tilelang_musa（DSL → MUSA C）：https://github.com/MooreThreads/tilelang_musa
- MATE (MUSA AI Tensor Engine)：https://github.com/MooreThreads/mate — 开源 "CUTLASS 等价物"，只支持 mp_31

## 生态路线观察（2026-04-18，4 框架实测）

| 项目 | GEMM 路径 | 能提供给 tinygrad 的参考 |
|---|---|---|
| **torch_musa** | 100% → `mublasSgemm` / `mudnn::MatMul` | 无，薄 wrapper；example kernel 是 naive |
| **vllm-musa** | 源码级 Musify 翻译 + torch_musa/muDNN 兜底 | 无高性能 kernel；全靠 muDNN |
| **tilelang-musa** | DSL → MUSA C tile templates | `dequantize_gemm/` 示例只覆盖 FP4/MXFP4/W4A8/FP16xInt4，**无 Q4_K/K-quant** |
| **MATE** | **256 个预编译 ELF blob** (`mubin/mp31/gemm/*.cpp`) | 证实 Moore Threads 自己也不做 codegen，fast path = 手写 SASS blob |

**Moore Threads 自己都 100% 放弃 codegen 路线**。muDNN 内部 = MATE 在 mp_31 上暴露的那种 hand-assembled blob，只是 mp_22 版本闭源。tinygrad 在 mp_22 codegen 天花板 ~30% muDNN 是结构性的——IR 没有 muDNN 用的原语（warp shuffle、`ldmatrix` TC fragment、`cp.async`、swizzled smem）。

## muDNN C++ API（`<mudnn.h>` / `<mudnn_math.h>`）

核心接口：

```cpp
namespace musa::dnn;

class Handle {
  Handle(int device_id);
  // Handle 持有 musaStream_t，可通过 SetStream() 设置
};

enum class Tensor::Type { QINT4, QINT8, INT8, INT16, INT32, INT64,
                          UINT8, UINT16, UINT32, UINT64,
                          HALF, BFLOAT16, FLOAT, DOUBLE, BOOL };

class Tensor {
  Status SetAddr(const void* addr);
  Status SetType(Type t);
  Status SetNdInfo(int ndims, const int64_t* dim);
  // Format: NCW/NWC/NCHW/NHWC/… 对 matmul 可忽略
};

class MatMul {
  Status SetTranspose(bool left, bool right);
  Status SetAlpha(double); Status SetBeta(double);
  Status Run(Handle& h, Tensor& c, const Tensor& a, const Tensor& b,
             const MemoryMaintainer& = nullptr);
  Status RunWithBiasAdd(Handle&, Tensor& d, const Tensor& a, const Tensor& b,
                         const Tensor& c, const Tensor& bias, ...);
};
```

我们写的 `extra/musa/mudnn_wrapper.cc` 把 MatMul 封成 `extern "C" int mudnn_tg_matmul(handle, a, b, c, M, N, K, dtype_code, ta, tb)`，供 `runtime/support/mudnn.py` ctypes 调用。
