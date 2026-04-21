from __future__ import annotations
import ctypes, functools
from tinygrad.helpers import DEBUG, mv_address, suppress_finalizing
from tinygrad.device import Compiled, BufferSpec, LRUAllocator
from tinygrad.renderer.cstyle import MUSARenderer
from tinygrad.runtime.autogen import musa
from tinygrad.runtime.support.c import init_c_struct_t, init_c_var

def check(status):
  if status != 0:
    p = ctypes.POINTER(ctypes.c_char)()
    musa.muGetErrorString(status, ctypes.byref(p))
    raise RuntimeError(f"MUSA Error {status}, {ctypes.string_at(p).decode() if p else ''}")

def encode_args(args, vals) -> tuple[ctypes.Structure, ctypes.Array]:
  c_args = init_c_struct_t(len(args) * 8 + len(vals) * 4, tuple([(f'f{i}', musa.MUdeviceptr, i*8) for i in range(len(args))] +
                                                                [(f'v{i}', ctypes.c_int, len(args)*8 + i*4) for i in range(len(vals))]))(*args, *vals)
  vargs = (ctypes.c_void_p * 5)(ctypes.c_void_p(1), ctypes.cast(ctypes.byref(c_args), ctypes.c_void_p), ctypes.c_void_p(2),
                                ctypes.cast(ctypes.pointer(ctypes.c_size_t(ctypes.sizeof(c_args))), ctypes.c_void_p), ctypes.c_void_p(0))
  return c_args, vargs

def mu_time_execution(cb, enable=False) -> float|None:
  if not enable: return cb()
  evs = [init_c_var(musa.MUevent, lambda x: musa.muEventCreate(ctypes.byref(x), 0)) for _ in range(2)]
  musa.muEventRecord(evs[0], None)
  cb()
  musa.muEventRecord(evs[1], None)
  check(musa.muEventSynchronize(evs[1]))
  musa.muEventElapsedTime(ctypes.byref(ret := ctypes.c_float()), evs[0], evs[1])
  for ev in evs: musa.muEventDestroy_v2(ev)
  return ret.value * 1e-3

class MUSAProgram:
  def __init__(self, dev:MUSADevice, name:str, lib:bytes, smem:int=0, **kwargs):
    self.dev, self.name, self.lib, self.smem = dev, name, lib, smem
    if DEBUG >= 5: print(f"MUSA fatbin for {name}: {len(lib)} bytes")
    check(musa.muCtxSetCurrent(self.dev.context))
    self.module = musa.MUmodule()
    status = musa.muModuleLoadData(ctypes.byref(self.module), lib)
    if status != 0:
      del self.module
      raise RuntimeError(f"muModuleLoadData failed: {status}")
    check(musa.muModuleGetFunction(ctypes.byref(prg := musa.MUfunction()), self.module, name.encode("utf-8")))
    self.prg = prg
    if self.smem > 0: check(musa.muFuncSetAttribute(self.prg, musa.MU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, self.smem))
    # Query mp_22 per-function limits so we can reject launches that the driver would crash on instead of merely error-coding.
    mt = ctypes.c_int()
    check(musa.muFuncGetAttribute(ctypes.byref(mt), musa.MU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, self.prg))
    self.max_threads = mt.value

  @suppress_finalizing
  def __del__(self): check(musa.muModuleUnload(self.module))

  def __call__(self, *args, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int,...]=(), wait=False, **kw):
    check(musa.muCtxSetCurrent(self.dev.context))
    if (lt:=local_size[0]*local_size[1]*local_size[2]) > self.max_threads:
      raise RuntimeError(f"local_size {local_size} product={lt} exceeds function max_threads={self.max_threads} on {self.name}")
    if not hasattr(self, "vargs"): self.c_args, self.vargs = encode_args(args, vals)
    else:
      for i in range(len(args)): self.c_args.__setattr__(f'f{i}', args[i])
      for i in range(len(vals)): self.c_args.__setattr__(f'v{i}', vals[i])
    return mu_time_execution(lambda: check(musa.muLaunchKernel(self.prg, *global_size, *local_size, self.smem, None, None, self.vargs)), enable=wait)

class MUSAAllocator(LRUAllocator['MUSADevice']):
  def _alloc(self, size, options:BufferSpec):
    check(musa.muCtxSetCurrent(self.dev.context))
    if options.external_ptr: return musa.MUdeviceptr(options.external_ptr)
    if options.host: return init_c_var(ctypes.c_void_p, lambda x: check(musa.muMemHostAlloc(ctypes.byref(x), size, 0x01)))
    return init_c_var(musa.MUdeviceptr, lambda x: check(musa.muMemAlloc_v2(ctypes.byref(x), size)))
  @suppress_finalizing
  def _free(self, opaque, options:BufferSpec):
    if options.external_ptr: return
    if options.host: check(musa.muMemFreeHost(opaque))
    else: check(musa.muMemFree_v2(opaque))
  def _copyin(self, dest, src:memoryview):
    check(musa.muCtxSetCurrent(self.dev.context))
    host_mem = self.alloc(len(src), BufferSpec(host=True))
    self.dev.pending_copyin.append((host_mem, len(src), BufferSpec(host=True)))
    ctypes.memmove(host_mem, mv_address(src), len(src))
    check(musa.muMemcpyHtoDAsync_v2(dest, host_mem, len(src), None))
  def _copyout(self, dest:memoryview, src):
    MUSADevice.synchronize_system()
    check(musa.muCtxSetCurrent(self.dev.context))
    check(musa.muMemcpyDtoH_v2(mv_address(dest), src, len(dest)))
  def _transfer(self, dest, src, sz:int, src_dev, dest_dev):
    check(musa.muCtxSetCurrent(src_dev.context))
    check(musa.muEventCreate(ctypes.byref(sync_event := musa.MUevent()), 0))
    check(musa.muMemcpyDtoDAsync_v2(dest, src, sz, None))
    check(musa.muEventRecord(sync_event, None))
    check(musa.muCtxSetCurrent(dest_dev.context))
    check(musa.muStreamWaitEvent(None, sync_event, 0))
  def _offset(self, buf, size:int, offset:int): return musa.MUdeviceptr(buf.value + offset)

class MUSADevice(Compiled):
  devices: list[MUSADevice] = []
  peer_access = False

  def __init__(self, device:str):
    device_id = int(device.split(":")[1]) if ":" in device else 0
    check(musa.muInit(0))
    self.mu_device = init_c_var(musa.MUdevice, lambda x: check(musa.muDeviceGet(ctypes.byref(x), device_id)))
    self.context = init_c_var(musa.MUcontext, lambda x: check(musa.muCtxCreate_v2(ctypes.byref(x), 0, self.mu_device)))
    check(musa.muDeviceComputeCapability(ctypes.byref(major := ctypes.c_int()), ctypes.byref(minor := ctypes.c_int()), device_id))

    for dev in MUSADevice.devices:
      check(musa.muDeviceCanAccessPeer(ctypes.byref(val := ctypes.c_int()), self.mu_device, dev.mu_device))
      if val.value != 1: continue
      check(musa.muCtxSetCurrent(dev.context))
      check(musa.muCtxEnablePeerAccess(self.context, 0))
      check(musa.muCtxSetCurrent(self.context))
      check(musa.muCtxEnablePeerAccess(dev.context, 0))
      MUSADevice.peer_access = True

    self.pending_copyin: list[tuple[int, int, BufferSpec|None]] = []
    MUSADevice.devices.append(self)

    # MUSAGraph disabled: MUSA SDK 3.1.0 does not support muGraphExecUpdate / muGraphExec*SetParams (all return 801).
    # Re-instantiating the graph per call is slower than direct launches. Revisit when driver supports exec-level updates.
    super().__init__(device, MUSAAllocator(self), [MUSARenderer], functools.partial(MUSAProgram, self),
                     None, arch=f"mp_{major.value}{minor.value}")

  def synchronize(self):
    check(musa.muCtxSetCurrent(self.context))
    check(musa.muCtxSynchronize())
    for opaque,sz,options in self.pending_copyin: self.allocator.free(opaque, sz, options)
    self.pending_copyin.clear()

  @staticmethod
  def synchronize_system():
    for d in MUSADevice.devices: d.synchronize()
