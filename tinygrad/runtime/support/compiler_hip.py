import ctypes, subprocess, tempfile
import tinygrad.runtime.autogen.comgr as comgr
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
    check(comgr.amd_comgr_action_info_set_options(action_info, b"-O3 -mllvm -amdgpu-internalize-symbols"))
    check(comgr.amd_comgr_do_action(comgr.AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE, action_info, data_set_bc, data_set_reloc))

  check(comgr.amd_comgr_action_info_set_options(action_info, b""))
  check(comgr.amd_comgr_do_action(comgr.AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE, action_info, data_set_reloc, data_set_exec))
  ret = _get_comgr_data(data_set_exec, comgr.AMD_COMGR_DATA_KIND_EXECUTABLE)
  check(comgr.amd_comgr_release_data(data_src))
  for x in [data_set_src, data_set_bc, data_set_reloc, data_set_exec]: check(comgr.amd_comgr_destroy_data_set(x))
  check(comgr.amd_comgr_destroy_action_info(action_info))
  return ret

def compile_hip_sh(prg:str, arch="gfx1100", asm=False) -> bytes:
  with open("/tmp/1.c", "w") as f:
    f.write(prg)

  subprocess.run(['/home/b1tg/tinygrad/1.sh'], input=prg.encode('utf-8'), check=True)
  with open("/tmp/comgr-24faa3/output/a.so", "rb") as f:
    return f.read()
def compile_hip(prg:str, arch="gfx1100", asm=False) -> bytes:
  # print("[*] stage 1")
  args = [
"-cc1", "-triple", "amdgcn-amd-amdhsa", "-aux-triple", "x86_64-unknown-linux-gnu", "-E",  "-clear-ast-before-backend", "-disable-llvm-verifier", "-discard-value-names", "-main-file-name", "<null>", "-mrelocation-model", "pic", "-pic-level", "2", "-fhalf-no-semantic-interposition", "-mframe-pointer=none", "-fno-rounding-math", "-mconstructor-aliases", "-aux-target-cpu", "x86-64", "-fcuda-is-device", "-mllvm", "-amdgpu-internalize-symbols", "-fcuda-allow-variadic-functions", "-fvisibility=hidden", "-fapply-global-visibility-to-externs", "-mlink-builtin-bitcode", "/opt/rocm/amdgcn/bitcode/hip.bc", "-mlink-builtin-bitcode", "/opt/rocm/amdgcn/bitcode/ocml.bc", "-mlink-builtin-bitcode", "/opt/rocm/amdgcn/bitcode/ockl.bc", "-mlink-builtin-bitcode", "/opt/rocm/amdgcn/bitcode/oclc_daz_opt_off.bc", "-mlink-builtin-bitcode", "/opt/rocm/amdgcn/bitcode/oclc_unsafe_math_off.bc", "-mlink-builtin-bitcode", "/opt/rocm/amdgcn/bitcode/oclc_finite_only_off.bc", "-mlink-builtin-bitcode", "/opt/rocm/amdgcn/bitcode/oclc_correctly_rounded_sqrt_on.bc", "-mlink-builtin-bitcode", "/opt/rocm/amdgcn/bitcode/oclc_wavefrontsize64_off.bc", "-mlink-builtin-bitcode", f"/opt/rocm/amdgcn/bitcode/oclc_isa_version_{arch[3:]}.bc", "-mlink-builtin-bitcode", "/opt/rocm/amdgcn/bitcode/oclc_abi_version_500.bc", "-target-cpu", arch, "-target-feature", "+cumode", "-debugger-tuning=gdb", "-fdebug-compilation-dir=/home/b1tg/tinygrad", "-resource-dir", "/opt/rocm-6.2.4/llvm/lib/clang/18", "-internal-isystem", "/opt/rocm-6.2.4/llvm/lib/clang/18/include/cuda_wrappers", "-isystem", "/opt/rocm-6.2.4/include", "-isystem", "/opt/rocm-6.2.4/hip/include", "-I", "/tmp/comgr-6c7c96/include", "-D", "HIP_VERSION_MAJOR=6", "-D", "HIP_VERSION_MINOR=0", "-D", "HIP_VERSION_PATCH=32830", "-D", "__HIPCC_RTC__", "-I", "/opt/rocm/include", "-internal-isystem", "/usr/lib/gcc/x86_64-linux-gnu/12/../../../../include/c++/12", "-internal-isystem", "/usr/lib/gcc/x86_64-linux-gnu/12/../../../../include/x86_64-linux-gnu/c++/12", "-internal-isystem", "/usr/lib/gcc/x86_64-linux-gnu/12/../../../../include/c++/12/backward", "-internal-isystem", "/usr/lib/gcc/x86_64-linux-gnu/12/../../../../include/c++/12", "-internal-isystem", "/usr/lib/gcc/x86_64-linux-gnu/12/../../../../include/x86_64-linux-gnu/c++/12", "-internal-isystem", "/usr/lib/gcc/x86_64-linux-gnu/12/../../../../include/c++/12/backward", "-internal-isystem", "/opt/rocm-6.2.4/llvm/lib/clang/18/include", "-internal-isystem", "/usr/local/include", "-internal-isystem", "/usr/lib/gcc/x86_64-linux-gnu/12/../../../../x86_64-linux-gnu/include", "-internal-externc-isystem", "/usr/include/x86_64-linux-gnu", "-internal-externc-isystem", "/include", "-internal-externc-isystem", "/usr/include", "-internal-isystem", "/opt/rocm-6.2.4/llvm/lib/clang/18/include", "-internal-isystem", "/usr/local/include", "-internal-isystem", "/usr/lib/gcc/x86_64-linux-gnu/12/../../../../x86_64-linux-gnu/include", "-internal-externc-isystem", "/usr/include/x86_64-linux-gnu", "-internal-externc-isystem", "/include", "-internal-externc-isystem", "/usr/include", "-O3", "-Wno-gnu-line-marker", "-Wno-missing-prototypes", "-std=c++14", "-fdeprecated-macro", "-fno-autolink", "-ferror-limit", "19", "-fhip-new-launch-api", "-fgnuc-version=4.2.1", "-fcxx-exceptions", "-fexceptions", "-fcolor-diagnostics", "-vectorize-loops", "-vectorize-slp",  "-disable-llvm-passes", "-cuid=d4ddbcdbb98e61b8", "-fcuda-allow-variadic-functions", "-faddrsig", "-D__GCC_HAVE_DWARF2_CFI_ASM=1", "-o", "-", "-x", "hip", "-" #noqa:E501
  ]
  obj = subprocess.check_output(['/opt/rocm/llvm/bin/clang', *args], input=prg.encode('utf-8'))
  args = [
"-cc1", "-triple", "amdgcn-amd-amdhsa", "-aux-triple", "x86_64-unknown-linux-gnu", "-emit-llvm-bc", "-emit-llvm-uselists", "-save-temps=/tmp/comgr-6c7c96/output", "-clear-ast-before-backend", "-disable-llvm-verifier", "-discard-value-names", "-main-file-name", "<null>", "-mrelocation-model", "pic", "-pic-level", "2", "-fhalf-no-semantic-interposition", "-mframe-pointer=none", "-fno-rounding-math", "-mconstructor-aliases", "-aux-target-cpu", "x86-64", "-fcuda-is-device", "-mllvm", "-amdgpu-internalize-symbols", "-fcuda-allow-variadic-functions", "-fvisibility=hidden", "-fapply-global-visibility-to-externs", "-mlink-builtin-bitcode", "/opt/rocm/amdgcn/bitcode/hip.bc", "-mlink-builtin-bitcode", "/opt/rocm/amdgcn/bitcode/ocml.bc", "-mlink-builtin-bitcode", "/opt/rocm/amdgcn/bitcode/ockl.bc", "-mlink-builtin-bitcode", "/opt/rocm/amdgcn/bitcode/oclc_daz_opt_off.bc", "-mlink-builtin-bitcode", "/opt/rocm/amdgcn/bitcode/oclc_unsafe_math_off.bc", "-mlink-builtin-bitcode", "/opt/rocm/amdgcn/bitcode/oclc_finite_only_off.bc", "-mlink-builtin-bitcode", "/opt/rocm/amdgcn/bitcode/oclc_correctly_rounded_sqrt_on.bc", "-mlink-builtin-bitcode", "/opt/rocm/amdgcn/bitcode/oclc_wavefrontsize64_off.bc", "-mlink-builtin-bitcode", f"/opt/rocm/amdgcn/bitcode/oclc_isa_version_{arch[3:]}.bc", "-mlink-builtin-bitcode", "/opt/rocm/amdgcn/bitcode/oclc_abi_version_500.bc", "-target-cpu", arch, "-target-feature", "+cumode", "-debugger-tuning=gdb", "-fdebug-compilation-dir=/home/b1tg/tinygrad", "-resource-dir", "/opt/rocm-6.2.4/llvm/lib/clang/18", "-O3", "-Wno-gnu-line-marker", "-Wno-missing-prototypes", "-std=c++14", "-fdeprecated-macro", "-fno-autolink", "-ferror-limit", "19", "-fhip-new-launch-api", "-fgnuc-version=4.2.1", "-fcxx-exceptions", "-fexceptions", "-fcolor-diagnostics", "-vectorize-loops", "-vectorize-slp",  "-disable-llvm-passes", "-disable-llvm-passes", "-cuid=d4ddbcdbb98e61b8", "-fcuda-allow-variadic-functions", "-faddrsig", "-D__GCC_HAVE_DWARF2_CFI_ASM=1", "-o", "-", "-x", "hip-cpp-output", "-" #noqa:E501
  ]
  obj = subprocess.check_output(['/opt/rocm/llvm/bin/clang', *args], input=obj)

  args = [
"-cc1", "-triple", "amdgcn-amd-amdhsa", "-aux-triple", "x86_64-unknown-linux-gnu", "-emit-llvm-bc", "-emit-llvm-uselists", "-save-temps=/tmp/comgr-6c7c96/output", "-clear-ast-before-backend", "-disable-llvm-verifier", "-discard-value-names", "-main-file-name", "<null>", "-mrelocation-model", "pic", "-pic-level", "2", "-fhalf-no-semantic-interposition", "-mframe-pointer=none", "-fno-rounding-math", "-mconstructor-aliases", "-aux-target-cpu", "x86-64", "-fcuda-is-device", "-mllvm", "-amdgpu-internalize-symbols", "-fcuda-allow-variadic-functions", "-fvisibility=hidden", "-fapply-global-visibility-to-externs", "-mlink-builtin-bitcode", "/opt/rocm/amdgcn/bitcode/hip.bc", "-mlink-builtin-bitcode", "/opt/rocm/amdgcn/bitcode/ocml.bc", "-mlink-builtin-bitcode", "/opt/rocm/amdgcn/bitcode/ockl.bc", "-mlink-builtin-bitcode", "/opt/rocm/amdgcn/bitcode/oclc_daz_opt_off.bc", "-mlink-builtin-bitcode", "/opt/rocm/amdgcn/bitcode/oclc_unsafe_math_off.bc", "-mlink-builtin-bitcode", "/opt/rocm/amdgcn/bitcode/oclc_finite_only_off.bc", "-mlink-builtin-bitcode", "/opt/rocm/amdgcn/bitcode/oclc_correctly_rounded_sqrt_on.bc", "-mlink-builtin-bitcode", "/opt/rocm/amdgcn/bitcode/oclc_wavefrontsize64_off.bc", "-mlink-builtin-bitcode", f"/opt/rocm/amdgcn/bitcode/oclc_isa_version_{arch[3:]}.bc", "-mlink-builtin-bitcode", "/opt/rocm/amdgcn/bitcode/oclc_abi_version_500.bc", "-target-cpu", arch, "-target-feature", "+cumode", "-debugger-tuning=gdb", "-fdebug-compilation-dir=/home/b1tg/tinygrad", "-resource-dir", "/opt/rocm-6.2.4/llvm/lib/clang/18", "-O3", "-Wno-gnu-line-marker", "-Wno-missing-prototypes", "-std=c++14", "-fno-autolink", "-ferror-limit", "19", "-fhip-new-launch-api", "-fgnuc-version=4.2.1", "-fcolor-diagnostics", "-vectorize-loops", "-vectorize-slp",  "-disable-llvm-passes", "-cuid=d4ddbcdbb98e61b8", "-fcuda-allow-variadic-functions", "-faddrsig", "-o", "-", "-x", "ir", "-" #noqa:E501
  ]
  obj = subprocess.check_output(['/opt/rocm/llvm/bin/clang', *args], input=obj)
  # print("[*] stage 2")
  args = [
"-cc1", "-triple", "amdgcn-amd-amdhsa", "-S", "-clear-ast-before-backend", "-disable-llvm-verifier", "-discard-value-names", "-main-file-name", "<null>.bc", "-mrelocation-model", "pic", "-pic-level", "2", "-fhalf-no-semantic-interposition", "-mframe-pointer=none", "-ffp-contract=on", "-fno-rounding-math", "-mconstructor-aliases", "-fvisibility=hidden", "-fapply-global-visibility-to-externs", "-target-cpu", arch, "-debugger-tuning=gdb", "-fdebug-compilation-dir=/home/b1tg/tinygrad", "-resource-dir", "/opt/rocm-6.2.4/llvm/lib/clang/18", "-O3", "-ferror-limit", "19", "-nogpulib", "-fcolor-diagnostics", "-vectorize-loops", "-vectorize-slp", "-mllvm", "-amdgpu-internalize-symbols", "-mllvm", "-amdgpu-internalize-symbols", "-faddrsig", "-o", "-", "-x", "ir", "-" #noqa:E501
  ]
  obj = subprocess.check_output(['/opt/rocm/llvm/bin/clang', *args], input=obj)

  with tempfile.NamedTemporaryFile(delete=True) as relo_file:
    args = [
  "-cc1as", "-triple", "amdgcn-amd-amdhsa", "-filetype", "obj", "-main-file-name", "<null>.bc", "-target-cpu", arch, "-fdebug-compilation-dir=/home/b1tg/tinygrad", "-dwarf-version=5", "-mrelocation-model", "pic", "-mllvm", "-amdgpu-internalize-symbols", "-mllvm", "-amdgpu-internalize-symbols", "-o", relo_file.name, "-" #noqa:E501
    ]
    subprocess.run(['/opt/rocm/llvm/bin/clang', *args], input=obj, check=True)

    # print("[*] stage 3")
    args = [
  "--no-undefined", "-shared", "--enable-new-dtags", relo_file.name, f"-plugin-opt=mcpu={arch}", "-o", "-"
    ]
    obj = subprocess.check_output(['/opt/rocm/llvm/bin/ld.lld', *args])
    return obj
def compile_hip_0202(prg:str, arch="gfx1100", asm=False) -> bytes:
  args = ["-x", "hip", f"--offload-arch={arch}", "-O3", "-S", "-emit-llvm", "--cuda-device-only", "-", "-o", "-"]
  obj = subprocess.check_output(['/opt/rocm/llvm/bin/clang', *args], input=prg.encode('utf-8'))
  with tempfile.NamedTemporaryFile(delete=True) as relo_file:
    args = ["-mtriple=amdgcn-amd-amdhsa", f"-mcpu={arch}", "-O3", "-filetype=obj", "-mattr=+cumode", "-", "-o", relo_file.name]
    subprocess.run(['/opt/rocm/llvm/bin/llc', *args], input=obj, check=True)
    args = [relo_file.name, "--no-undefined", "-shared", "-o", "-"]
    obj = subprocess.check_output(['/opt/rocm/llvm/bin/ld.lld', *args])
    return obj

def compile_hip_old(prg:str, arch="gfx1100", asm=False) -> bytes:
  args = ["-cc1", "-triple", "amdgcn-amd-amdhsa", "-aux-triple", "x86_64-unknown-linux-gnu",
    "-emit-llvm-bc", "-emit-llvm-uselists", "-clear-ast-before-backend", "-disable-llvm-verifier", "-discard-value-names",
    "-mrelocation-model", "pic", "-pic-level", "2", "-fhalf-no-semantic-interposition", "-mframe-pointer=none",
    "-fdenormal-fp-math-f32=preserve-sign,preserve-sign", "-fno-rounding-math", "-mconstructor-aliases", "-aux-target-cpu", "x86-64",
    "-fcuda-is-device", "-mllvm", "-amdgpu-internalize-symbols", "-fcuda-allow-variadic-functions", "-fvisibility=hidden",
    "-fapply-global-visibility-to-externs",
    "-mlink-builtin-bitcode", "/opt/rocm/amdgcn/bitcode/hip.bc",
    "-mlink-builtin-bitcode", "/opt/rocm/amdgcn/bitcode/ocml.bc",
    "-mlink-builtin-bitcode", "/opt/rocm/amdgcn/bitcode/ockl.bc",
    "-mlink-builtin-bitcode", "/opt/rocm/amdgcn/bitcode/oclc_daz_opt_off.bc",
    "-mlink-builtin-bitcode", "/opt/rocm/amdgcn/bitcode/oclc_correctly_rounded_sqrt_on.bc",
    "-mlink-builtin-bitcode", "/opt/rocm/amdgcn/bitcode/oclc_unsafe_math_off.bc",
    "-mlink-builtin-bitcode", "/opt/rocm/amdgcn/bitcode/oclc_finite_only_off.bc",
    "-mlink-builtin-bitcode", "/opt/rocm/amdgcn/bitcode/oclc_wavefrontsize64_off.bc",
    "-mlink-builtin-bitcode", f"/opt/rocm/amdgcn/bitcode/oclc_isa_version_{arch[3:]}.bc",
    "-mlink-builtin-bitcode", "/opt/rocm/amdgcn/bitcode/oclc_abi_version_500.bc",
    "-target-cpu", arch, "-target-feature", "+cumode", "-debugger-tuning=gdb",
    "-D", "HIP_VERSION_MAJOR=6", "-D", "HIP_VERSION_MINOR=0", "-D", "HIP_VERSION_PATCH=32830", "-D", "__HIPCC_RTC__", "-I", "/opt/rocm/include",
    "-O3", "-Wno-gnu-line-marker", "-Wno-missing-prototypes", "-std=c++14", "-fdeprecated-macro", "-fno-autolink",
    "-fhip-new-launch-api", "-fgnuc-version=4.2.1", "-fcxx-exceptions", "-fexceptions", "-fcolor-diagnostics", "-vectorize-loops",
    # ""
    "-vectorize-slp", "-disable-llvm-passes", "-fcuda-allow-variadic-functions", "-faddrsig",
    "-D__GCC_HAVE_DWARF2_CFI_ASM=1", "-o", "-", "-x", "hip", "-"]
  bc_obj = subprocess.check_output(['/opt/rocm/llvm/bin/clang', *args], input=prg.encode('utf-8'))
  with tempfile.NamedTemporaryFile(delete=True) as relo_file:
    args = [
      "-cc1",
      "-triple", "amdgcn-amd-amdhsa", "-emit-obj", "-clear-ast-before-backend", "-disable-llvm-verifier",
      "-discard-value-names", "-mrelocation-model", "pic", "-pic-level", "2", "-fhalf-no-semantic-interposition",
      "-mframe-pointer=none", "-fdenormal-fp-math-f32=preserve-sign,preserve-sign", "-ffp-contract=on", "-fno-rounding-math",
      "-mconstructor-aliases", "-fvisibility=hidden", "-fapply-global-visibility-to-externs", "-target-cpu", arch,
      "-debugger-tuning=gdb",  "-O3", "-nogpulib", "-fcolor-diagnostics", "-vectorize-loops", "-vectorize-slp", "-mllvm",
      "-amdgpu-internalize-symbols", "-mllvm", "-amdgpu-internalize-symbols", "-faddrsig", "-o", relo_file.name, "-x", "ir", "-"
    ]
    subprocess.run(['/opt/rocm/llvm/bin/clang', *args], input=bc_obj, check=True)
    args = [relo_file.name, "--no-undefined", "-shared", "-o", "-"]
    obj = subprocess.check_output(['/opt/rocm/llvm/bin/ld.lld', *args])
    return obj

class AMDCompiler(Compiler):
  def __init__(self, arch:str):
    self.arch = arch
    super().__init__(f"compile_hip_{self.arch}")
  def compile(self, src:str) -> bytes:
    import os
    try: return compile_hip_0202(src, self.arch) if os.getenv("NEW") else compile_hip_comgr(src, self.arch)
    except RuntimeError as e: raise CompileError(e) from e
  def disassemble(self, lib:bytes):
    asm = subprocess.check_output(["/opt/rocm/llvm/bin/llvm-objdump", '-d', '-'], input=lib)
    print('\n'.join([x for x in asm.decode('utf-8').split("\n") if 's_code_end' not in x]))
