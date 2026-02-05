import os
import numpy as np
from dataclasses import replace
from tinygrad import Tensor, dtypes
from tinygrad.helpers import getenv, flat_mv
from tinygrad.engine.realize import ExecItem, CompiledRunner
from tinygrad.device import Device
from tinygrad.runtime.ops_cuda import CUDAAllocator, CUDADevice, CUDAProgram, CUDACompiler
from tinygrad.renderer import ProgramSpec
from tinygrad.renderer.cstyle import CUDARenderer
from extra.gemm.wmma_uop_helpers import build_wmma_uops, build_wmma_uops_list, build_wmma_uops_sink, MAX_FP16_VARIANT, MAX_FP32_VARIANT, WmmaUOpBuilder

os.environ.setdefault("NV_UOP_LDMATRIX", "1")
os.environ.setdefault("NV_UOP_CP_ASYNC", "1")

N = getenv("N", 4096)
M = getenv("M", N)
K = getenv("K", N)
CNT = getenv("CNT", 10)
VERIFY = getenv("VERIFY", 1)
VARIANT = getenv("VARIANT", "fp32")

if VARIANT == "fp16":
  variant = MAX_FP16_VARIANT
else:
  variant = MAX_FP32_VARIANT

dtype_in = dtypes.half
dtype_out = dtypes.float if variant.acc_dtype == "float" else dtypes.half

assert M % variant.M_TILE == 0
assert N % variant.N_TILE == 0
assert K % variant.K_TILE == 0

FLOPS = M * N * K * 2
BW = 2 * ((M*K) + (K*N) + (M*N))

def test_matmul_execitem():
  """Test using ExecItem like nv_uop_matmul.py format - with CUDA device."""
  # Create builder and get UOps
  builder = WmmaUOpBuilder(M, N, K, variant)
  sink = builder.build()
  uops = builder.uops

  # Render to source code
  device = CUDADevice("cuda:0")
  compiler = CUDACompiler(device.arch)
  renderer = CUDARenderer(device.arch)
  allocator = CUDAAllocator(device)
  src = renderer.render(uops)

  # Compile and create CUDA program directly
  smem = variant.smem_bytes if variant.smem_bytes > 49152 else 0
  prog = CUDAProgram(device, f"nv_{variant.name}_uop", compiler.compile(src), smem)

  # Allocate buffers using CUDA allocator
  a_buf = allocator.alloc(M * K * dtype_in.itemsize)
  b_buf = allocator.alloc(K * N * dtype_in.itemsize)
  c_buf = allocator.alloc(M * N * dtype_out.itemsize)

  # Initialize input data
  rng = np.random.default_rng()
  na = rng.random((M, K), dtype=np.float32).astype(np.float16) - 0.5
  nb = rng.random((K, N), dtype=np.float32).astype(np.float16) - 0.5
  allocator._copyin(a_buf, memoryview(bytearray(na)))
  allocator._copyin(b_buf, memoryview(bytearray(nb)))

  # Benchmark
  tms = []
  for _ in range(CNT):
    tms.append(prog(c_buf, a_buf, b_buf, global_size=(M // variant.M_TILE, N // variant.N_TILE, 1),
      local_size=(variant.block_threads, 1, 1), wait=True))

  print(f"ExecItem: {M*N:10d} {min(tms)*1e6:9.2f} us, would be {FLOPS*1e-9/min(tms):9.2f} GFLOPS matmul")

  if VERIFY:
    out = np.empty(M * N, dtype=np.float32 if dtype_out == dtypes.float else np.float16)
    allocator._copyout(flat_mv(out.data), c_buf)
    comp = na.astype(np.float32) @ nb.astype(np.float32)
    res = out.reshape(M, N).astype(np.float32)
    np.testing.assert_allclose(res, comp, atol=5e-3, rtol=1e-3)

def test_matmul_direct():
  """Test using CUDAProgram directly (original approach)."""
  device = CUDADevice("cuda:0")
  compiler = CUDACompiler(device.arch)
  allocator = CUDAAllocator(device)

  a = allocator.alloc(M * K * dtype_in.itemsize)
  b = allocator.alloc(K * N * dtype_in.itemsize)
  c = allocator.alloc(M * N * dtype_out.itemsize)

  na = np.random.default_rng().normal(scale=1.0, size=(M, K)).astype(np.float32).astype(np.float16)
  nb = np.random.default_rng().normal(scale=1.0, size=(K, N)).astype(np.float32).astype(np.float16)
  allocator._copyin(a, memoryview(bytearray(na)))
  allocator._copyin(b, memoryview(bytearray(nb)))

  uops = build_wmma_uops_list(M, N, K, variant)
  renderer = CUDARenderer(device.arch)
  src = renderer.render(uops)
  if getenv("DEBUG", 0) > 1:
    print(src)

  smem = variant.smem_bytes if variant.smem_bytes > 49152 else 0
  prog = CUDAProgram(device, f"nv_{variant.name}_uop", compiler.compile(src), smem)

  tms = []
  for _ in range(CNT):
    tms.append(prog(c, a, b, global_size=(M // variant.M_TILE, N // variant.N_TILE, 1),
      local_size=(variant.block_threads, 1, 1), wait=True))

  print(f"Direct: {M*N:10d} {min(tms)*1e6:9.2f} us, would be {FLOPS*1e-9/min(tms):9.2f} GFLOPS matmul, {BW*1e-9/min(tms):.2f} GB/s")

  if VERIFY:
    out = np.empty(M * N, dtype=np.float32 if dtype_out == dtypes.float else np.float16)
    allocator._copyout(flat_mv(out.data), c)
    comp = na.astype(np.float32) @ nb.astype(np.float32)
    res = out.reshape(M, N).astype(np.float32)
    np.testing.assert_allclose(res, comp, atol=5e-3, rtol=1e-3)

if __name__ == "__main__":
  import sys
  if len(sys.argv) > 1 and sys.argv[1] == "execitem":
    test_matmul_execitem()
  else:
    test_matmul_direct()
