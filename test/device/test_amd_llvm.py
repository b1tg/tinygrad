import unittest
import numpy as np
from tinygrad import Device
from tinygrad.device import CompileError
from tinygrad.helpers import flat_mv
if Device.DEFAULT=="AMD":
  from tinygrad.runtime.ops_amd import AMDAllocator, AMDDevice, AMDProgram
  from tinygrad.runtime.support.compiler_amd import AMDLLVMCompiler

@unittest.skipUnless(Device.DEFAULT == "AMD", "Runs only on AMD")
class TestAMDLLVM(unittest.TestCase):
  def test_speed(self):
    src = """
define amdgpu_kernel void @test_wmma_f32_16x16x128_f8f6f4___v16i32_fp6___v16i32_fp8(ptr addrspace(1) %out) {
  %res = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(i32 2, <16 x i32> %A, i32 0, <16 x i32> %B, i16 0, <8 x float> %C)
  store <8 x float> %res, ptr addrspace(1) %out
  ret void

"""
    src = """
; 一个最小 amdgpu kernel，调用该 intrinsic 并写回结果
define amdgpu_kernel void @test_mfma_scale(ptr addrspace(1) %out) {
entry:
  ; 初始化输入寄存器
  %a = shufflevector <8 x i32> undef, <8 x i32> undef, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  %b = shufflevector <8 x i32> undef, <8 x i32> undef, <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  %c = shufflevector <4 x float> undef, <4 x float> undef, <4 x i32> <i32 0, i32 0, i32 0, i32 0>

  ; 设置 scale 参数
  %scale0 = add i32 0, 1
  %scale1 = add i32 0, 1

  ; 调用 intrinsic
  %result = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(
    <8 x i32> %a, <8 x i32> %b, <4 x float> %c,
    i32 0,   ; cbsz
    i32 0,   ; blgp
    i32 0,   ; abid (通常为 0)
    i32 %scale0,  ; scale0
    i32 0,        ; op_sel
    i32 %scale1   ; scale1
  )

  ; 写回结果
  store <4 x float> %result, ptr addrspace(1) %out, align 16
  ret void
}

"""
    device = AMDDevice()
    compiler = AMDLLVMCompiler("gfx950")
    obj = compiler.compile(src)
    allocator = AMDAllocator(device)
    a = allocator.alloc(1*8)
    prog = AMDProgram(device, "test", obj)
    prog(a, wait=True)
    na = np.empty(1, np.uint64)
    # allocator._copyout(flat_mv(na.data), a)
    # assert na == [0x1234567800000005]
  def test_compiler(self):
    src = '''
; https://github.com/llvm/llvm-project/blob/main/llvm/test/CodeGen/AMDGPU/imm.ll
define amdgpu_kernel void @i64_imm_inline_lo(ptr addrspace(1) %out) {
entry:
  store i64 1311768464867721221, ptr addrspace(1) %out ; 0x1234567800000005
  ret void
}
    '''
    device = AMDDevice()
    compiler = AMDLLVMCompiler("gfx1100")
    obj = compiler.compile(src)
    allocator = AMDAllocator(device)
    a = allocator.alloc(1*8)
    prog = AMDProgram(device, "test", obj)
    prog(a, wait=True)
    na = np.empty(1, np.uint64)
    allocator._copyout(flat_mv(na.data), a)
    assert na == [0x1234567800000005]

  def test_compiler_diag_error(self):
    src = """
@local_temp0 = internal unnamed_addr addrspace(3) global [{N} x float*] undef, align 16
define amdgpu_kernel void @test(float* noalias align 32 %data0, half* noalias align 32 %data1, float* noalias align 32 %data2) #0
{{
  %local_temp0 = addrspacecast [{N} x float*] addrspace(3)* @local_temp0 to [{N} x float*]*
  %v178 = getelementptr inbounds float, float* %local_temp0, i32 1
  %v133 = getelementptr inbounds float, float* %data2, i32 1
  %v134 = load float, float* %v133
  store float %v134, float* %v178
  ret void
}}
"""
    compiler = AMDLLVMCompiler("gfx1100")
    compiler.compile(src.format(N=65536//8))
    with self.assertRaises(CompileError):
      # llvm diagnostic: <unknown>:0:0: local memory (65544) exceeds limit (65536) in function 'test'
      compiler.compile(src.format(N=65536//8+1))


if __name__ == '__main__':
  unittest.main()
