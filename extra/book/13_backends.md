# Chapter 13: Backends & Adding an Accelerator

Tinygrad supports many hardware backends: NVIDIA GPUs, AMD GPUs, Apple Metal, CPUs, and more. This chapter explains how backends are structured and what it takes to add a new one.

## The Backend Interface

Every backend implements a small set of abstractions defined in `tinygrad/device.py`:

```python
# The four things a backend must provide:
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

That's it. If you implement these four things, tinygrad can run on your hardware.

## Existing Backends

| Backend | File | How it works |
|---------|------|-------------|
| **METAL** | `ops_metal.py` | Apple Metal API, shaders in MSL |
| **CUDA** | `ops_cuda.py` | NVIDIA CUDA, kernels via NVRTC |
| **NV** | `ops_nv.py` | Raw NVIDIA driver (no CUDA SDK) |
| **AMD** | `ops_amd.py` | Raw AMD KFD driver (no ROCm) |
| **HIP** | `ops_hip.py` | AMD HIP runtime |
| **CPU** | `ops_cpu.py` | LLVM IR compiled to native code |
| **PYTHON** | `ops_python.py` | Pure Python interpreter (reference) |
| **OPENCL** | `ops_cl.py` | OpenCL for any GPU |
| **WEBGPU** | `ops_webgpu.py` | WebGPU (runs in browser) |
| **QCOM** | `ops_qcom.py` | Qualcomm Adreno GPUs |
| **DISK** | `ops_disk.py` | Memory-mapped files |

### The "Raw" Backends (NV, AMD)

The most interesting backends are NV and AMD. Unlike CUDA or HIP, these talk directly to the kernel driver — no runtime library needed:

```python
# CUDA path: Python -> CUDA Runtime -> NVIDIA Driver -> GPU
# NV path:   Python -> NVIDIA Driver -> GPU (no CUDA needed!)

# HIP path:  Python -> HIP Runtime -> AMD Driver -> GPU
# AMD path:  Python -> KFD Driver -> GPU (no ROCm needed!)
```

Tinygrad includes its own implementations of the command submission protocols (PM4 packets for AMD, push buffers for NV), memory management, and kernel dispatch.

## Renderers

Each backend has an associated **renderer** that turns the linearized UOp list into source code:

```python
# C-style renderers (share most code):
MetalRenderer    # Apple Metal Shading Language
CUDARenderer     # NVIDIA CUDA C++
OpenCLRenderer   # OpenCL C
HIPRenderer      # AMD HIP C++

# Assembly renderers:
PTXRenderer      # NVIDIA PTX assembly
AMDRenderer      # AMD RDNA assembly (raw machine code)
LLVMRenderer     # LLVM IR (for CPU)
WGSLRenderer     # WebGPU Shading Language
```

The C-style renderers share a common base in `tinygrad/renderer/cstyle.py`. The differences are mainly:
- Function signature format
- Thread index variables (`gid.x` vs `blockIdx.x`)
- Vector type names (`float4` vs `make_float4`)
- Available intrinsics

## Adding a New Accelerator

To add support for a new hardware device, you need:

### Step 1: Choose a rendering target

Pick the closest existing renderer. If your hardware runs OpenCL, use `OpenCLRenderer`. If it has a C-like shading language, extend `CStyleRenderer`.

### Step 2: Implement the Compiler

```python
class MyCompiler(Compiler):
    def compile(self, src: str) -> bytes:
        # Call your hardware's shader compiler
        # Return the compiled binary blob
        return my_compile(src)
```

### Step 3: Implement the Allocator

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

### Step 4: Implement the Runtime

```python
class MyProgram:
    def __init__(self, name: str, lib: bytes):
        # Load the compiled program
        self.module = my_load_program(lib)

    def __call__(self, *bufs, global_size, local_size, wait=False):
        # Launch the kernel
        my_dispatch(self.module, bufs, global_size, local_size)
```

### Step 5: Register the Device

```python
class MyDevice(CompiledDevice):
    def __init__(self, device: str):
        self.renderer = CStyleRenderer(...)
        self.compiler = MyCompiler()
        self.allocator = MyAllocator()
        self.runtime = MyProgram
        super().__init__(device)
```

### Step 6: Test

```bash
MY_DEVICE=1 python -c "
from tinygrad import Tensor, Device
Device.DEFAULT = 'MY_DEVICE'
print((Tensor.ones(4) + Tensor.ones(4)).numpy())
"
```

## The PYTHON Backend

The simplest backend is `PYTHON` — a pure Python interpreter that executes UOps directly:

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

This is useful as a reference implementation and for testing.

## Exercises

1. **List devices**: Run `python -c "from tinygrad import Device; print(Device.DEFAULT)"` to see your default device.

2. **Try different backends**: Run the same kernel on different backends and compare output:
   ```bash
   METAL=1 DEBUG=4 python -c "from tinygrad import Tensor; (Tensor.ones(4)+Tensor.ones(4)).realize()"
   CPU=1 DEBUG=4 python -c "from tinygrad import Tensor; (Tensor.ones(4)+Tensor.ones(4)).realize()"
   ```

3. **Read a backend**: Open `tinygrad/runtime/ops_metal.py` and find the Compiler, Allocator, and Runtime classes.

## Source Code Map

| File | What to read |
|------|-------------|
| `tinygrad/device.py` | `Device`, `Compiler`, `Allocator`, `Buffer` base classes |
| `tinygrad/runtime/ops_metal.py` | Apple Metal backend |
| `tinygrad/runtime/ops_cuda.py` | NVIDIA CUDA backend |
| `tinygrad/runtime/ops_nv.py` | Raw NVIDIA driver backend |
| `tinygrad/runtime/ops_amd.py` | Raw AMD driver backend |
| `tinygrad/runtime/ops_cpu.py` | CPU (LLVM) backend |
| `tinygrad/runtime/ops_python.py` | Python reference interpreter |
| `tinygrad/renderer/cstyle.py` | Shared C-style renderer |
