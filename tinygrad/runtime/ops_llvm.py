from tinygrad.device import Compiled, MallocAllocator, CPUProgram
from tinygrad.renderer.llvmir import LLVMRenderer
from tinygrad.runtime.support.comipler_llvm import HostLLVMCompiler

class LLVMDevice(Compiled):
  def __init__(self, device:str):
    super().__init__(device, MallocAllocator, LLVMRenderer(), HostLLVMCompiler(), CPUProgram)
