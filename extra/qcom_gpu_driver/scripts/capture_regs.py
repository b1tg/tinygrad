#!/usr/bin/env python3
"""Capture GPU register writes for a kernel dispatch via KGSL ioctl hook.
Follows indirect buffers (CP_INDIRECT_BUFFER_PFE) to capture full CL driver state.

Usage: source extra/cl_android.sh
  DEV=QCOM python3 extra/qcom_gpu_driver/scripts/capture_regs.py        # capture QCOM dispatch
  OPENCL_PATH=/system/vendor/lib64/libOpenCL.so DEV=CL python3 extra/qcom_gpu_driver/scripts/capture_regs.py  # capture CL dispatch
"""
import os, sys, struct, ctypes, ctypes.util, mmap, json
os.environ.setdefault("LIBC_PATH", "/system/lib64/libc.so")
os.environ.setdefault("LLVM_QCOM_PATH", "/vendor/lib64/libllvm-qcom.so")
os.environ.setdefault("SPEC", "0")

from tinygrad.helpers import to_mv
from extra.qcom_gpu_driver import msm_kgsl
import pathlib, re

def ioctls_from_header():
  hdr = (pathlib.Path(__file__).parent.parent / "msm_kgsl.h").read_text().replace("\\\n", "")
  pattern = r'#define\s+(IOCTL_KGSL_[A-Z0-9_]+)\s+_IOWR?\(KGSL_IOC_TYPE,\s+(0x[0-9a-fA-F]+),\s+struct\s([A-Za-z0-9_]+)\)'
  return {int(nr, 0x10):(name, getattr(msm_kgsl, "struct_"+sname)) for name, nr, sname in re.findall(pattern, hdr, re.MULTILINE)}

nrs = ioctls_from_header()
mmaped = {}
all_submissions = []

NAMES = {
  0xa9b0: "CNTL_0", 0xa9b1: "CNTL_1", 0xa9b3: "OBJ_FIRST_EXEC",
  0xa9b6: "PVT_MEM_PARAM", 0xa9b9: "PVT_MEM_SIZE", 0xa9ba: "TSIZE", 0xa9bb: "CONFIG",
  0xa9bc: "INSTR_SIZE", 0xa9bd: "STACK_OFF", 0xa9be: "HYSTERESIS",
  0xa9c2: "WIE_CNTL_0", 0xa9c3: "WIE_CNTL_1", 0xa9c5: "VGS_CNTL",
  0xa9c8: "REG_PROG_ID_0", 0xa9cd: "CONST_CONFIG",
  0xa9d4: "NDRANGE_0", 0xa9db: "WGE_CNTL", 0xa9dc: "KERNEL_GROUP_X",
  0xa9dd: "KERNEL_GROUP_Y", 0xa9de: "KERNEL_GROUP_Z", 0xa9df: "NDRANGE_7",
  0xaa00: "USIZE", 0xab00: "MODE_CNTL", 0xab1f: "UPDATE_CNTL",
  0xb309: "TPL1_MODE_CNTL", 0xb600: "TPL1_DBG_ECO",
}

def get_mem(addr, vlen):
  for k,v in mmaped.items():
    if k <= addr and addr < k+len(v): return v[addr-k:addr-k+vlen]
  return bytes(to_mv(addr, vlen))

def parse_cmd_buf(dat, depth=0):
  regs, cmds = {}, []
  ptr = 0
  while ptr < len(dat):
    w = struct.unpack("I", dat[ptr:ptr+4])[0]
    if (w>>24) == 0x70:
      op, sz = ((w>>16)&0x7F), w&0x3FFF
      vals = list(struct.unpack("I"*sz, dat[ptr+4:ptr+4+4*sz])) if sz > 0 else []
      cmds.append((op, vals))
      if op == 0x3f and len(vals) >= 3:  # follow indirect buffers
        try:
          ib_regs, ib_cmds = parse_cmd_buf(get_mem(vals[0] | (vals[1] << 32), vals[2] * 4), depth+1)
          for k, v in ib_regs.items(): regs.setdefault(k, v)  # preamble defaults, don't override per-kernel
        except: pass
      ptr += 4*sz
    elif (w>>28) == 0x4:
      off, sz = ((w>>8)&0x7FFFF), w&0x7F
      vals = struct.unpack("I"*sz, dat[ptr+4:ptr+4+4*sz])
      for i, v in enumerate(vals): regs[off+i] = v  # per-kernel overrides preamble
      ptr += 4*sz
    ptr += 4
  return regs, cmds

@ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_ulong, ctypes.c_void_p)
def hooked_ioctl(fd, request, argp):
  ret = libc.syscall(0x1d, ctypes.c_int(fd), ctypes.c_ulong(request), ctypes.c_void_p(argp))
  itype, nr = (request>>8)&0xFF, request&0xFF
  if nr in nrs and itype == 9:
    name, stype = nrs[nr]
    s = ctypes.cast(ctypes.c_void_p(argp), ctypes.POINTER(stype)).contents
    if name == "IOCTL_KGSL_GPUOBJ_INFO":
      mmaped[s.gpuaddr] = mmap.mmap(fd, s.size, offset=s.id*0x1000)
    if name == "IOCTL_KGSL_GPU_COMMAND":
      for i in range(s.numcmds):
        cmd = ctypes.cast(ctypes.c_void_p(s.cmdlist+ctypes.sizeof(msm_kgsl.struct_kgsl_command_object)*i),
                          ctypes.POINTER(msm_kgsl.struct_kgsl_command_object)).contents
        regs, cmds = parse_cmd_buf(get_mem(cmd.gpuaddr, cmd.size))
        all_submissions.append(regs)
  return ret

libc = ctypes.CDLL(ctypes.util.find_library("libc"))
tramp = b"\x70\x00\x00\x10\x10\x02\x40\xf9\x00\x02\x1f\xd6"
tramp += struct.pack("Q", ctypes.cast(ctypes.byref(hooked_ioctl), ctypes.POINTER(ctypes.c_ulong)).contents.value)
ioctl_addr = ctypes.cast(ctypes.byref(libc.ioctl), ctypes.POINTER(ctypes.c_ulong))
libc.mprotect(ctypes.c_ulong((ioctl_addr.contents.value//0x1000)*0x1000), 0x2000, 7)
libc.memcpy(ioctl_addr.contents, ctypes.create_string_buffer(tramp), len(tramp))

# Run a kernel
from tinygrad import Tensor
a = Tensor.ones(16, 16).contiguous()
b = Tensor.eye(16).contiguous()
c = (a + b).numpy()
backend = os.environ.get("DEV", "QCOM")
print(f"{backend}: {len(all_submissions)} submissions, result sum={c.sum():.0f}")

# Print compute registers from kernel dispatch submissions
for si, regs in enumerate(all_submissions):
  if 0xa9d4 not in regs and 0xb990 not in regs: continue
  print(f"\n  Submission {si}:")
  for addr in sorted(regs.keys()):
    if addr < 0xa900 or addr > 0xb700: continue
    name = NAMES.get(addr, f"0x{addr:04x}")
    print(f"    {name:20s} (0x{addr:04x}) = 0x{regs[addr]:08x}")
