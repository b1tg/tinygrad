#!/usr/bin/env python3
"""Probe QCOM GPU info: chip_id, llvm-qcom availability, compile test.
Usage: python3 extra/qcom_gpu_driver/scripts/probe_gpu.py
"""
import os, ctypes, ctypes.util
os.environ.setdefault("LIBC_PATH", "/system/lib64/libc.so")

print("=== KGSL GPU Info ===")
try:
  from tinygrad.runtime.autogen import kgsl
  from tinygrad.runtime.support.hcq import FileIOInterface
  fd = FileIOInterface('/dev/kgsl-3d0', os.O_RDWR)
  info = kgsl.struct_kgsl_devinfo()
  kgsl.IOCTL_KGSL_DEVICE_GETPROPERTY(fd, type=kgsl.KGSL_PROP_DEVICE_INFO, value=ctypes.addressof(info), sizebytes=ctypes.sizeof(info))
  gid = ((info.chip_id >> 24) & 0xFF, (info.chip_id >> 16) & 0xFF, (info.chip_id >> 8) & 0xFF)
  print(f"  chip_id: 0x{info.chip_id:08x}  parsed: a{gid[0]}{gid[1]}{gid[2]}  gmem: {info.gmem_sizebytes//1024} KB")
except Exception as e:
  print(f"  FAILED: {e}")

print("\n=== llvm-qcom ===")
for path in ["/vendor/lib64/libllvm-qcom.so", "/system/vendor/lib64/libllvm-qcom.so"]:
  if os.path.exists(path): print(f"  FOUND: {path} ({os.path.getsize(path)//1024//1024} MB)")

print("\n=== sysfs ===")
for f in ["gpu_model", "devfreq/max_freq", "devfreq/cur_freq"]:
  try:
    with open(f"/sys/class/kgsl/kgsl-3d0/{f}") as fp: print(f"  {f}: {fp.read().strip()}")
  except: pass
