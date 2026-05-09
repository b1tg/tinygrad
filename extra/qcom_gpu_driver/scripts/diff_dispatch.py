#!/usr/bin/env python3
"""Run the SAME kernel via CL and QCOM dispatch, compare results.
Tests that llvm-qcom binary + QCOM register setup produces correct output.

Usage: source extra/cl_android.sh
  OPENCL_PATH=/system/vendor/lib64/libOpenCL.so python3 extra/qcom_gpu_driver/scripts/diff_dispatch.py
"""
import os, ctypes, array
os.environ.setdefault("LIBC_PATH", "/system/lib64/libc.so")
os.environ.setdefault("LLVM_QCOM_PATH", "/vendor/lib64/libllvm-qcom.so")
os.environ.setdefault("OPENCL_PATH", "/system/vendor/lib64/libOpenCL.so")
os.environ.setdefault("SPEC", "0")

import numpy as np
np.random.seed(42)

src = b"""__kernel void r_test(__global float* out, __global float* a, __global float* b) {
  int gid0 = get_group_id(0);
  int lid0 = get_local_id(0);
  int lid1 = get_local_id(1);
  __local float shmem[128];
  int idx = gid0 * 128 + lid0 * 8 + lid1;
  shmem[lid0 * 8 + lid1] = a[idx] * b[idx];
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid1 == 0) {
    float sum = 0.0f;
    for (int i = 0; i < 8; i++) sum += shmem[lid0 * 8 + i];
    out[gid0 * 16 + lid0] = sum;
  }
}"""

N = 256
a_np = np.random.randn(N).astype(np.float32)
b_np = np.random.randn(N).astype(np.float32)

# CL
os.environ["DEV"] = "CL"
from tinygrad import Device
from tinygrad.runtime.autogen import opencl as cl
from tinygrad.runtime.ops_cl import checked, to_char_p_p, BP_CB

cl_dev = Device["CL"]
program = checked(cl.clCreateProgramWithSource(cl_dev.context, 1, to_char_p_p([src]), None, s:=ctypes.c_int32()), s)
cl.clBuildProgram(program, 1, cl_dev.device_id, None, BP_CB(), None)
kernel = checked(cl.clCreateKernel(program, b"r_test", s:=ctypes.c_int32()), s)
buf_out = checked(cl.clCreateBuffer(cl_dev.context, cl.CL_MEM_READ_WRITE, 128, None, s:=ctypes.c_int32()), s)
buf_a = checked(cl.clCreateBuffer(cl_dev.context, cl.CL_MEM_READ_WRITE | cl.CL_MEM_COPY_HOST_PTR, N*4, a_np.ctypes.data_as(ctypes.c_void_p), s:=ctypes.c_int32()), s)
buf_b = checked(cl.clCreateBuffer(cl_dev.context, cl.CL_MEM_READ_WRITE | cl.CL_MEM_COPY_HOST_PTR, N*4, b_np.ctypes.data_as(ctypes.c_void_p), s:=ctypes.c_int32()), s)
for i, buf in enumerate([buf_out, buf_a, buf_b]): cl.clSetKernelArg(kernel, i, 8, ctypes.byref(buf))
cl.clEnqueueNDRangeKernel(cl_dev.queue, kernel, 2, None, (ctypes.c_size_t*3)(32,8,1), (ctypes.c_size_t*3)(16,8,1), 0, None, None)
cl_result = (ctypes.c_float * 32)()
cl.clEnqueueReadBuffer(cl_dev.queue, buf_out, True, 0, 128, cl_result, 0, None, None)
cl_result = np.array(list(cl_result))

# QCOM
os.environ["DEV"] = "QCOM"
Device._opened_devices.clear()
from tinygrad.runtime.ops_qcom import QCOMProgram
from tinygrad.helpers import to_mv
qcom_dev = Device["QCOM"]
lib = qcom_dev.compiler.compile(src.decode())
buf_out_q = qcom_dev._gpu_alloc(128, uncached=True, fill_zeroes=True)
buf_a_q = qcom_dev._gpu_alloc(N*4, uncached=True)
buf_b_q = qcom_dev._gpu_alloc(N*4, uncached=True)
to_mv(buf_a_q.va_addr, N*4).cast('f')[:] = array.array('f', a_np.tolist())
to_mv(buf_b_q.va_addr, N*4).cast('f')[:] = array.array('f', b_np.tolist())
prg = QCOMProgram(qcom_dev, "r_test", lib, buf_dtypes=[[(0, None)], [(0, None)], [(0, None)]])
prg(buf_out_q, buf_a_q, buf_b_q, global_size=(2,1,1), local_size=(16,8,1), wait=True)
qcom_result = np.array(list(to_mv(buf_out_q.va_addr, 128).cast('f')[:32]))

print(f"CL:   {cl_result[:6]}")
print(f"QCOM: {qcom_result[:6]}")
print(f"diff: {np.abs(cl_result - qcom_result).max():.6f}  {'PASS' if np.abs(cl_result - qcom_result).max() < 1e-4 else 'FAIL'}")
