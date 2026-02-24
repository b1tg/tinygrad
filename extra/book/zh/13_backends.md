# 第13章：后端与添加加速器

tinygrad 支持多种硬件后端：NVIDIA GPU、AMD GPU、Apple Metal、CPU 等。本章介绍后端的结构以及添加新后端所需的工作。

## 后端接口

每个后端都实现了 `tinygrad/device.py` 中定义的一小组抽象：

```python
# 后端必须提供的四个组件：
class Compiler:
    def compile(self, src: str) -> bytes: ...  # source code -> binary

class Allocator:
    def _alloc(self, size: int, options) -> Any: ...  # allocate GPU memory
    def _free(self, buf) -> None: ...                  # free GPU memory
    def _copyin(self, buf, src: memoryview) -> None: ...  # CPU -> GPU
    def _copyout(self, dest: memoryview, buf) -> None: ... # GPU -> CPU

class Runtime:
    def __call__(self, *bufs, global_size, local_size): ...  # launch kernel

class Device:
    compiler: Compiler
    allocator: Allocator
    runtime: type  # Runtime class
    renderer: Renderer
```

就这些。只要实现这四个组件，tinygrad 就能在你的硬件上运行。

## 现有后端

| 后端 | 文件 | 工作原理 |
|------|------|----------|
| **METAL** | `ops_metal.py` | Apple Metal API，使用 MSL 着色器 |
| **CUDA** | `ops_cuda.py` | NVIDIA CUDA，通过 NVRTC 编译内核 |
| **NV** | `ops_nv.py` | 直接访问 NVIDIA 驱动（无需 CUDA SDK） |
| **AMD** | `ops_amd.py` | 直接访问 AMD KFD 驱动（无需 ROCm） |
| **HIP** | `ops_hip.py` | AMD HIP 运行时 |
| **CPU** | `ops_cpu.py` | LLVM IR 编译为本地代码 |
| **PYTHON** | `ops_python.py` | 纯 Python 解释器（参考实现） |
| **OPENCL** | `ops_cl.py` | OpenCL，适用于任何 GPU |
| **WEBGPU** | `ops_webgpu.py` | WebGPU（在浏览器中运行） |
| **QCOM** | `ops_qcom.py` | Qualcomm Adreno GPU |
| **DISK** | `ops_disk.py` | 内存映射文件 |

### "原始"后端（NV、AMD）

最有趣的后端是 NV 和 AMD。与 CUDA 或 HIP 不同，它们直接与内核驱动通信——不需要运行时库：

```python
# CUDA 路径: Python -> CUDA Runtime -> NVIDIA Driver -> GPU
# NV 路径:   Python -> NVIDIA Driver -> GPU (no CUDA needed!)

# HIP 路径:  Python -> HIP Runtime -> AMD Driver -> GPU
# AMD 路径:  Python -> KFD Driver -> GPU (no ROCm needed!)
```

tinygrad 自行实现了命令提交协议（AMD 的 PM4 数据包、NV 的 push buffer）、内存管理和内核调度。

## 渲染器

每个后端都有一个关联的**渲染器**，将线性化的 UOp 列表转换为源代码：

```python
# C 风格渲染器（共享大部分代码）：
MetalRenderer    # Apple Metal Shading Language
CUDARenderer     # NVIDIA CUDA C++
OpenCLRenderer   # OpenCL C
HIPRenderer      # AMD HIP C++

# 汇编渲染器：
PTXRenderer      # NVIDIA PTX assembly
AMDRenderer      # AMD RDNA assembly (raw machine code)
LLVMRenderer     # LLVM IR (for CPU)
WGSLRenderer     # WebGPU Shading Language
```

C 风格渲染器共享 `tinygrad/renderer/cstyle.py` 中的公共基类。主要区别在于：
- 函数签名格式
- 线程索引变量（`gid.x` vs `blockIdx.x`）
- 向量类型名称（`float4` vs `make_float4`）
- 可用的内置函数

## 添加新加速器

要为新硬件设备添加支持，你需要：

### 第1步：选择渲染目标

选择最接近的现有渲染器。如果你的硬件支持 OpenCL，使用 `OpenCLRenderer`。如果它有类 C 的着色语言，扩展 `CStyleRenderer`。

### 第2步：实现 Compiler

```python
class MyCompiler(Compiler):
    def compile(self, src: str) -> bytes:
        # Call your hardware's shader compiler
        # Return the compiled binary blob
        return my_compile(src)
```

### 第3步：实现 Allocator

```python
class MyAllocator(Allocator):
    def _alloc(self, size: int, options=None):
        # Allocate GPU memory
        return my_malloc(size)

    def _free(self, buf):
        my_free(buf)

    def _copyin(self, buf, src: memoryview):
        # Copy from CPU (src) to GPU (buf)
        my_memcpy_to_device(buf, src)

    def _copyout(self, dest: memoryview, buf):
        # Copy from GPU (buf) to CPU (dest)
        my_memcpy_from_device(dest, buf)
```

### 第4步：实现 Runtime

```python
class MyProgram:
    def __init__(self, name: str, lib: bytes):
        # Load the compiled program
        self.module = my_load_program(lib)

    def __call__(self, *bufs, global_size, local_size, wait=False):
        # Launch the kernel
        my_dispatch(self.module, bufs, global_size, local_size)
```

### 第5步：注册设备

```python
class MyDevice(CompiledDevice):
    def __init__(self, device: str):
        self.renderer = CStyleRenderer(...)
        self.compiler = MyCompiler()
        self.allocator = MyAllocator()
        self.runtime = MyProgram
        super().__init__(device)
```

### 第6步：测试

```bash
MY_DEVICE=1 python -c "
from tinygrad import Tensor, Device
Device.DEFAULT = 'MY_DEVICE'
print((Tensor.ones(4) + Tensor.ones(4)).numpy())
"
```

## PYTHON 后端

最简单的后端是 `PYTHON`——一个纯 Python 解释器，直接执行 UOps：

```python
# From ops_python.py (simplified):
def exec_uop(uop):
    if uop.op is Ops.CONST: return uop.arg
    if uop.op is Ops.ADD: return exec_uop(uop.src[0]) + exec_uop(uop.src[1])
    if uop.op is Ops.MUL: return exec_uop(uop.src[0]) * exec_uop(uop.src[1])
    if uop.op is Ops.LOAD: return memory[uop.src[0]][uop.src[1]]
    if uop.op is Ops.STORE: memory[uop.src[0]][uop.src[1]] = exec_uop(uop.src[2])
    # ... etc
```

这对于作为参考实现和测试非常有用。

## 练习

1. **列出设备**：运行 `python -c "from tinygrad import Device; print(Device.DEFAULT)"` 查看你的默认设备。

2. **尝试不同后端**：在不同后端上运行相同的内核并比较输出：
   ```bash
   METAL=1 DEBUG=4 python -c "from tinygrad import Tensor; (Tensor.ones(4)+Tensor.ones(4)).realize()"
   CPU=1 DEBUG=4 python -c "from tinygrad import Tensor; (Tensor.ones(4)+Tensor.ones(4)).realize()"
   ```

3. **阅读后端代码**：打开 `tinygrad/runtime/ops_metal.py`，找到 Compiler、Allocator 和 Runtime 类。

## 源代码导航

| 文件 | 阅读内容 |
|------|----------|
| `tinygrad/device.py` | `Device`、`Compiler`、`Allocator`、`Buffer` 基类 |
| `tinygrad/runtime/ops_metal.py` | Apple Metal 后端 |
| `tinygrad/runtime/ops_cuda.py` | NVIDIA CUDA 后端 |
| `tinygrad/runtime/ops_nv.py` | 原始 NVIDIA 驱动后端 |
| `tinygrad/runtime/ops_amd.py` | 原始 AMD 驱动后端 |
| `tinygrad/runtime/ops_cpu.py` | CPU (LLVM) 后端 |
| `tinygrad/runtime/ops_python.py` | Python 参考解释器 |
| `tinygrad/renderer/cstyle.py` | 共享的 C 风格渲染器 |
