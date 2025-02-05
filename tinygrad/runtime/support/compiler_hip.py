import ctypes, subprocess, tempfile
import tinygrad.runtime.autogen.comgr as comgr
import tinygrad.runtime.autogen.llvm as llvm
from tinygrad.device import Compiler, CompileError

def check(status):
  if status != 0:
    comgr.amd_comgr_status_string(status, ctypes.byref(status_str := ctypes.POINTER(ctypes.c_char)()))
    raise RuntimeError(f"comgr fail {status}, {ctypes.string_at(status_str).decode()}")

def _get_comgr_data(data_set, data_type):
  check(comgr.amd_comgr_action_data_get_data(data_set, data_type, 0, ctypes.byref(data_exec := comgr.amd_comgr_data_t())))
  check(comgr.amd_comgr_get_data(data_exec, ctypes.byref(sz := ctypes.c_uint64()), None))
  check(comgr.amd_comgr_get_data(data_exec, ctypes.byref(sz), (dat := ctypes.create_string_buffer(sz.value))))
  check(comgr.amd_comgr_release_data(data_exec))
  return bytes(dat)

# AMD_COMGR_SAVE_TEMPS=1 AMD_COMGR_REDIRECT_LOGS=stdout AMD_COMGR_EMIT_VERBOSE_LOGS=1
def compile_hip_comgr(prg:str, arch="gfx1100", asm=False) -> bytes:
  check(comgr.amd_comgr_create_action_info(ctypes.byref(action_info := comgr.amd_comgr_action_info_t())))
  check(comgr.amd_comgr_action_info_set_language(action_info, comgr.AMD_COMGR_LANGUAGE_HIP))
  check(comgr.amd_comgr_action_info_set_isa_name(action_info, b"amdgcn-amd-amdhsa--" + arch.encode()))
  check(comgr.amd_comgr_action_info_set_logging(action_info, True))

  check(comgr.amd_comgr_create_data_set(ctypes.byref(data_set_src := comgr.amd_comgr_data_set_t())))
  check(comgr.amd_comgr_create_data_set(ctypes.byref(data_set_bc := comgr.amd_comgr_data_set_t())))
  check(comgr.amd_comgr_create_data_set(ctypes.byref(data_set_reloc := comgr.amd_comgr_data_set_t())))
  check(comgr.amd_comgr_create_data_set(ctypes.byref(data_set_exec := comgr.amd_comgr_data_set_t())))

  check(comgr.amd_comgr_create_data(comgr.AMD_COMGR_DATA_KIND_SOURCE, ctypes.byref(data_src := comgr.amd_comgr_data_t())))
  check(comgr.amd_comgr_set_data(data_src, len(rprg := prg.encode()), rprg))

  if asm:
    check(comgr.amd_comgr_set_data_name(data_src, b"<null>.s"))
    check(comgr.amd_comgr_data_set_add(data_set_src, data_src))
    status = comgr.amd_comgr_do_action(comgr.AMD_COMGR_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE, action_info, data_set_src, data_set_reloc)
    if status != 0:
      print(_get_comgr_data(data_set_reloc, comgr.AMD_COMGR_DATA_KIND_LOG).decode())
      raise RuntimeError("assemble failed")
  else:
    check(comgr.amd_comgr_set_data_name(data_src, b"<null>"))
    check(comgr.amd_comgr_data_set_add(data_set_src, data_src))
    # -include hiprtc_runtime.h was removed
    check(comgr.amd_comgr_action_info_set_options(action_info, f"-O3 -mcumode --hip-version=6.0.32830 -DHIP_VERSION_MAJOR=6 -DHIP_VERSION_MINOR=0 -DHIP_VERSION_PATCH=32830 -D__HIPCC_RTC__ -std=c++14 -nogpuinc -Wno-gnu-line-marker -Wno-missing-prototypes --offload-arch={arch} -I/opt/rocm/include -Xclang -disable-llvm-passes".encode())) # noqa: E501
    status = comgr.amd_comgr_do_action(comgr.AMD_COMGR_ACTION_COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC, action_info, data_set_src, data_set_bc)
    if status != 0:
      print(_get_comgr_data(data_set_bc, comgr.AMD_COMGR_DATA_KIND_LOG).decode())
      raise RuntimeError("compile failed")
    # print("==== IR === ")
    data_relo = None
    # print(bc[:10])
    # print("==== IR end === ")
    # print("=== IR -> relo ===")
    # check(comgr.amd_comgr_action_info_set_options(action_info, b"-O3 -mllvm -amdgpu-internalize-symbols"))
    # check(comgr.amd_comgr_do_action(comgr.AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE, action_info, data_set_bc, data_set_reloc))
    # print("--- use llvm ---")
    bc = _get_comgr_data(data_set_bc, comgr.AMD_COMGR_DATA_KIND_BC)
    relo = ir_to_relo(bc, arch)
    check(comgr.amd_comgr_create_data(comgr.AMD_COMGR_DATA_KIND_RELOCATABLE, ctypes.byref(data_relo := comgr.amd_comgr_data_t())))
    check(comgr.amd_comgr_set_data(data_relo, len(relo), relo))
    check(comgr.amd_comgr_set_data_name(data_relo, b"DO"))
    check(comgr.amd_comgr_data_set_add(data_set_reloc, data_relo))
    # print("=== IR -> relo end ===")
  check(comgr.amd_comgr_action_info_set_options(action_info, b""))
  # relo -> ELF
  check(comgr.amd_comgr_do_action(comgr.AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE, action_info, data_set_reloc, data_set_exec))
  ret = _get_comgr_data(data_set_exec, comgr.AMD_COMGR_DATA_KIND_EXECUTABLE)
  check(comgr.amd_comgr_release_data(data_src))
  if data_relo: check(comgr.amd_comgr_release_data(data_relo))
  for x in [data_set_src, data_set_bc, data_set_reloc, data_set_exec]: check(comgr.amd_comgr_destroy_data_set(x))
  check(comgr.amd_comgr_destroy_action_info(action_info))
  return ret

def cerr(): return ctypes.pointer(ctypes.pointer(ctypes.c_char()))
def expect(x, err, ret=None):
  if x: raise RuntimeError(llvm.string_cast(err.contents) if not isinstance(err, str) else err)
  return ret
def ir_to_relo(src, arch="gfx1100"):
  llvm.LLVMInitializeAMDGPUTarget()
  llvm.LLVMInitializeAMDGPUTargetInfo()
  llvm.LLVMInitializeAMDGPUTargetMC()
  llvm.LLVMInitializeAMDGPUDisassembler()
  llvm.LLVMInitializeAMDGPUAsmParser()
  llvm.LLVMInitializeAMDGPUAsmPrinter()
  triple=b"amdgcn-amd-amdhsa"
  features = b'+cumode'
  target = expect(llvm.LLVMGetTargetFromTriple(triple, ctypes.pointer(tgt:=llvm.LLVMTargetRef()), err:=cerr()), err, tgt)
  target_machine = llvm.LLVMCreateTargetMachine(target, triple, arch.encode(), features, llvm.LLVMCodeGenLevelAggressive, llvm.LLVMRelocPIC,
                                                llvm.LLVMCodeModelDefault)
  src_buf = llvm.LLVMCreateMemoryBufferWithMemoryRangeCopy(ctypes.create_string_buffer(src_bytes:=src), len(src_bytes), b'src')
  mod = expect(llvm.LLVMParseIRInContext(llvm.LLVMGetGlobalContext(), src_buf, ctypes.pointer(m:=llvm.LLVMModuleRef()), err:=cerr()), err, m)
  expect(llvm.LLVMVerifyModule(mod, llvm.LLVMReturnStatusAction, err:=cerr()), err)
  # ==== pass 1
  passes = b'default<O3>'
  pbo = llvm.LLVMCreatePassBuilderOptions()
  llvm.LLVMPassBuilderOptionsSetLoopUnrolling(pbo, True)
  llvm.LLVMPassBuilderOptionsSetLoopVectorization(pbo, True)
  llvm.LLVMPassBuilderOptionsSetSLPVectorization(pbo, True)
  llvm.LLVMPassBuilderOptionsSetVerifyEach(pbo, True)
  expect(llvm.LLVMRunPasses(mod, passes, target_machine, pbo), 'failed to run passes')
  # ==== pass 1 end
  # ==== pass 2
  # optimizer = llvm.LLVMCreatePassManager()
  # pmb = llvm.LLVMPassManagerBuilderCreate()
  # llvm.LLVMPassManagerBuilderSetOptLevel(pmb, 3)
  # llvm.LLVMPassManagerBuilderSetSizeLevel(pmb, 0)
  # llvm.LLVMPassManagerBuilderPopulateModulePassManager(pmb, optimizer)
  # llvm.LLVMRunPassManager(optimizer, mod)
  # ==== pass 2 end
  obj_buf = expect(llvm.LLVMTargetMachineEmitToMemoryBuffer(target_machine, mod, llvm.LLVMObjectFile, err:=cerr(),
                                                            ctypes.pointer(buf:=llvm.LLVMMemoryBufferRef())), err, buf)
  obj = ctypes.string_at(llvm.LLVMGetBufferStart(obj_buf), llvm.LLVMGetBufferSize(obj_buf))
  llvm.LLVMDisposeModule(mod)
  llvm.LLVMDisposeMemoryBuffer(obj_buf)
  return obj
def compile_hip(prg:str, arch="gfx1100", asm=False) -> bytes:
  args = ["-x", "hip", f"--offload-arch={arch}", "-O3", "-S", "-emit-llvm", "--cuda-device-only", "-", "-o", "-"]
  obj = subprocess.check_output(['/opt/rocm/llvm/bin/clang', *args], input=prg.encode('utf-8'))
  with tempfile.NamedTemporaryFile(delete=True) as f:
    relo = ir_to_relo(obj, arch)
    f.write(relo)
    f.flush()
    args = [f.name, "--no-undefined", "-shared", "-o", "-"]
    obj = subprocess.check_output(['/opt/rocm/llvm/bin/ld.lld', *args])
    return obj

class AMDCompiler(Compiler):
  def __init__(self, arch:str):
    self.arch = arch
    super().__init__(f"compile_hip_{self.arch}")
  def compile(self, src:str) -> bytes:
    # self.arch = "gfx803"
    try: return compile_hip_comgr(src, self.arch)
    except RuntimeError as e: raise CompileError(e) from e
  def disassemble(self, lib:bytes):
    asm = subprocess.check_output(["/opt/rocm/llvm/bin/llvm-objdump", '-d', '-'], input=lib)
    print('\n'.join([x for x in asm.decode('utf-8').split("\n") if 's_code_end' not in x]))
