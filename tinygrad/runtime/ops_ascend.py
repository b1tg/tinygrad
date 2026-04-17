from __future__ import annotations
import ctypes, functools, hashlib, os, pathlib, struct, subprocess, tempfile, time
from tinygrad.helpers import DEBUG, getenv, mv_address, suppress_finalizing
from tinygrad.device import Compiled, Compiler, CompileError, BufferSpec, LRUAllocator
from tinygrad.renderer.cstyle import AscendRenderer

ASCEND_HOME = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")
CANN_VER = getenv("ASCEND_CANN_VER", "8.1.RC1")
BISHENG = f"{ASCEND_HOME}/compiler/ccec_compiler/bin/bisheng"
LLD = f"{ASCEND_HOME}/compiler/ccec_compiler/bin/ld.lld"
TIKCFW = os.path.realpath(f"{ASCEND_HOME}/../{CANN_VER}/aarch64-linux/tikcpp/tikcfw")
DEVLIB = os.path.realpath(f"{ASCEND_HOME}/../{CANN_VER}/aarch64-linux/devlib")
for p in ("/usr/local/Ascend/driver/lib64/driver", "/usr/local/Ascend/driver/lib64", f"{ASCEND_HOME}/lib64"):
  if p not in os.environ.get("LD_LIBRARY_PATH", ""): os.environ["LD_LIBRARY_PATH"] = p + ":" + os.environ.get("LD_LIBRARY_PATH", "")

acl = ctypes.CDLL("libascendcl.so")
_acl_protos = [
  ("aclInit", [ctypes.c_char_p], ctypes.c_int), ("aclFinalize", [], ctypes.c_int),
  ("aclrtSetDevice", [ctypes.c_int], ctypes.c_int), ("aclrtResetDevice", [ctypes.c_int], ctypes.c_int),
  ("aclrtCreateStream", [ctypes.POINTER(ctypes.c_void_p)], ctypes.c_int),
  ("aclrtDestroyStream", [ctypes.c_void_p], ctypes.c_int),
  ("aclrtSynchronizeStream", [ctypes.c_void_p], ctypes.c_int),
  ("aclrtMalloc", [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_int], ctypes.c_int),
  ("aclrtMallocHost", [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t], ctypes.c_int),
  ("aclrtFree", [ctypes.c_void_p], ctypes.c_int), ("aclrtFreeHost", [ctypes.c_void_p], ctypes.c_int),
  ("aclrtMemcpy", [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int], ctypes.c_int),
  ("aclGetRecentErrMsg", [], ctypes.c_char_p),
]
for fn, at, rt in _acl_protos:
  f = getattr(acl, fn); f.argtypes, f.restype = at, rt

ACL_MEM_MALLOC_HUGE_FIRST = 0
H2D, D2H, D2D = 1, 2, 3

ASC_WRAP_SRC = r"""
#include <stdint.h>
#include <stddef.h>
extern "C" {
uint32_t RegisterAscendBinary(const char *fileBuf, size_t fileSize, uint32_t type, void **handle);
uint32_t LaunchAscendKernel(void *handle, const uint64_t key, const uint32_t blockDim, void **args,
                            uint32_t size, const void *stream);
uint32_t UnregisterAscendBinary(void *hdl);

int32_t asc_register(const char *buf, size_t sz, uint32_t type, void **hdl) { return (int32_t)RegisterAscendBinary(buf, sz, type, hdl); }
int32_t asc_launch(void *hdl, uint64_t key, uint32_t blk, void **args, uint32_t sz, const void *stm) { return (int32_t)LaunchAscendKernel(hdl, key, blk, args, sz, stm); }
int32_t asc_unregister(void *hdl) { return (int32_t)UnregisterAscendBinary(hdl); }
}
"""

def _build_wrap_so():
  cache = pathlib.Path.home() / ".cache" / "tinygrad" / "ascend"
  cache.mkdir(parents=True, exist_ok=True)
  so_path = cache / "asc_wrap.so"
  if so_path.exists(): return str(so_path)
  with tempfile.NamedTemporaryFile(suffix=".cc", delete=False, mode="w") as cf:
    cf.write(ASC_WRAP_SRC); cf.flush(); cpath = cf.name
  cmd = ["g++", "-shared", "-fPIC", "-o", str(so_path), cpath,
         "-Wl,--whole-archive", f"{ASCEND_HOME}/lib64/libascendc_runtime.a", "-Wl,--no-whole-archive",
         f"-L{ASCEND_HOME}/lib64", f"-L{DEVLIB}", "-lascendcl", "-lruntime", "-lmsprofiler", "-lerror_manager"]
  r = subprocess.run(cmd, capture_output=True)
  if r.returncode != 0: raise RuntimeError(f"asc_wrap build failed: {r.stderr.decode()}")
  os.unlink(cpath)
  return str(so_path)

wrap = ctypes.CDLL(_build_wrap_so())
wrap.asc_register.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p)]
wrap.asc_register.restype = ctypes.c_int
wrap.asc_launch.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p]
wrap.asc_launch.restype = ctypes.c_int
wrap.asc_unregister.argtypes = [ctypes.c_void_p]
wrap.asc_unregister.restype = ctypes.c_int

def check(rc:int, what:str="acl"):
  if rc != 0:
    msg = acl.aclGetRecentErrMsg(); msg = msg.decode() if msg else ""
    raise RuntimeError(f"ACL {what} rc={rc} {msg}")

# Tiling key INT32_MAX means "generic" — matches <kernel>_2147483647 naming
TILING_KEY_GENERIC = 0x7FFFFFFF

class AscendCompiler(Compiler):
  ARCH_MAP = {"Ascend910B": "dav-c220-vec", "Ascend910B2": "dav-c220-vec", "Ascend910B3": "dav-c220-vec", "Ascend910B4": "dav-c220-vec",
              "Ascend310P": "dav-m200-vec", "Ascend310B": "dav-m300"}
  def __init__(self, arch:str):
    self.arch = arch; self.cce_arch = self.ARCH_MAP.get(arch, "dav-c220-vec")
    super().__init__(f"compile_ascend_{self.cce_arch}_v2")

  def compile(self, src:str) -> bytes:
    with tempfile.TemporaryDirectory() as d:
      srcp, obj, rel, final = f"{d}/k.cce", f"{d}/k.o", f"{d}/k_r.o", f"{d}/k_final.o"
      with open(srcp, "w") as f: f.write(src)
      cmd = [BISHENG, "-c", f"--cce-aicore-arch={self.cce_arch}", "--cce-aicore-only",
             "-fcce-kernel-type-section", "--cce-auto-sync", "-O2", "-std=c++17",
             f"-I{TIKCFW}", f"-I{TIKCFW}/interface", f"-I{TIKCFW}/impl", f"-I{ASCEND_HOME}/include",
             "-DTILING_KEY_VAR=0", "-x", "cce", srcp, "-o", obj]
      r = subprocess.run(cmd, capture_output=True)
      if r.returncode != 0: raise CompileError(f"bisheng: {r.stderr.decode()[-500:]}\nSRC:\n{src[:600]}")
      r = subprocess.run([LLD, "-r", "-Ttext=0", obj, "-o", rel], capture_output=True)
      if r.returncode != 0: raise CompileError(f"lld pass1: {r.stderr.decode()}")
      r = subprocess.run([LLD, "-Ttext=0", rel, "-static", "-o", final], capture_output=True)
      if r.returncode != 0 and not os.path.exists(final): raise CompileError(f"lld pass2: {r.stderr.decode()}")
      with open(final, "rb") as f: return f.read()

class AscendProgram:
  def __init__(self, dev:AscendDevice, name:str, lib:bytes, smem:int=0, **kwargs):
    self.dev, self.name, self.lib = dev, name, lib
    # name is tinygrad's function name. We renamed in source to f"{name}_2147483647"; runtime kernel lookup uses same.
    self._hdl = ctypes.c_void_p()
    # type=1 for AIV (vector core)
    rc = wrap.asc_register(lib, len(lib), 1, ctypes.byref(self._hdl))
    if rc != 0:
      msg = acl.aclGetRecentErrMsg(); raise RuntimeError(f"asc_register failed rc={rc} {msg.decode() if msg else ''}")

  @suppress_finalizing
  def __del__(self):
    if getattr(self, "_hdl", None) and self._hdl.value: wrap.asc_unregister(self._hdl)

  def __call__(self, *args, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int,...]=(), wait=False, **kw):
    # AscendC ABI: args is a packed struct [ptr,ptr,...,int,pad,int,pad,...,overflow_ptr]
    # Each ptr is 8 bytes (8-aligned); each int is 4 bytes + 4 pad to 8-align next field;
    # trailing __ascendc_overflow slot is 8 bytes (device ptr to 8-byte status buffer).
    if not hasattr(self, "_overflow_buf"):
      self._overflow_buf = ctypes.c_void_p()
      check(acl.aclrtMalloc(ctypes.byref(self._overflow_buf), 8, ACL_MEM_MALLOC_HUGE_FIRST), "MallocOverflow")
    buf = bytearray()
    for a in args:
      v = a.value if isinstance(a, ctypes.c_void_p) else int(a)
      buf += struct.pack("Q", v & 0xFFFFFFFFFFFFFFFF)
    for v in vals:
      buf += struct.pack("i", int(v)) + b"\x00\x00\x00\x00"
    buf += struct.pack("Q", self._overflow_buf.value & 0xFFFFFFFFFFFFFFFF)
    args_size = len(buf)
    packed = (ctypes.c_ubyte * args_size).from_buffer(buf)
    block_num = max(1, global_size[0] * global_size[1] * global_size[2])
    if wait: t0 = time.perf_counter()
    rc = wrap.asc_launch(self._hdl, TILING_KEY_GENERIC, block_num, packed, args_size, self.dev.stream)
    if rc != 0:
      msg = acl.aclGetRecentErrMsg(); raise RuntimeError(f"asc_launch {self.name} rc={rc} {msg.decode() if msg else ''}")
    if wait:
      check(acl.aclrtSynchronizeStream(self.dev.stream), f"sync({self.name})")
      return time.perf_counter() - t0

class AscendAllocator(LRUAllocator['AscendDevice']):
  def _alloc(self, size:int, options:BufferSpec):
    p = ctypes.c_void_p()
    if options.host: check(acl.aclrtMallocHost(ctypes.byref(p), size), "MallocHost")
    else: check(acl.aclrtMalloc(ctypes.byref(p), size, ACL_MEM_MALLOC_HUGE_FIRST), "Malloc")
    return p
  @suppress_finalizing
  def _free(self, opaque, options:BufferSpec):
    if options.host: acl.aclrtFreeHost(opaque)
    else: acl.aclrtFree(opaque)
  def _copyin(self, dest, src:memoryview): check(acl.aclrtMemcpy(dest, len(src), mv_address(src), len(src), H2D), "H2D")
  def _copyout(self, dest:memoryview, src):
    self.dev.synchronize()
    check(acl.aclrtMemcpy(mv_address(dest), len(dest), src, len(dest), D2H), "D2H")
  def _transfer(self, dest, src, sz:int, src_dev, dest_dev): check(acl.aclrtMemcpy(dest, sz, src, sz, D2D), "D2D")
  def _offset(self, buf, size:int, offset:int): return ctypes.c_void_p(buf.value + offset)

class AscendDevice(Compiled):
  _initialized = False
  devices: list[AscendDevice] = []

  def __init__(self, device:str):
    if not AscendDevice._initialized:
      check(acl.aclInit(None), "aclInit"); AscendDevice._initialized = True
    self.device_id = int(device.split(":")[1]) if ":" in device else 0
    check(acl.aclrtSetDevice(self.device_id), "SetDevice")
    self.stream = ctypes.c_void_p()
    check(acl.aclrtCreateStream(ctypes.byref(self.stream)), "CreateStream")
    AscendDevice.devices.append(self)
    arch = getenv("ASCEND_ARCH_NAME", "Ascend910B2")
    super().__init__(device, AscendAllocator(self), [AscendRenderer], functools.partial(AscendProgram, self), None, arch=arch)

  def synchronize(self): check(acl.aclrtSynchronizeStream(self.stream), "Sync")

  def finalize(self):
    if getattr(self, "stream", None) and self.stream.value: acl.aclrtDestroyStream(self.stream)
    acl.aclrtResetDevice(self.device_id)
