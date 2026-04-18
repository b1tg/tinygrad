"""Minimal ctypes binding for muDNN MatMul via the wrapper .so in extra/musa/.
Used when MUDNN=1 env var is set to short-circuit matmul kernels on MUSA backend.
This is a debug / benchmark path and is expected to be replaced by native codegen later."""
import ctypes, os, pathlib, subprocess
from tinygrad.dtype import dtypes, DType

_WRAPPER_SRC = (pathlib.Path(__file__).parents[3] / "extra/musa/mudnn_wrapper.cc").resolve()
_WRAPPER_SO  = _WRAPPER_SRC.with_suffix(".so")

def _build_wrapper():
  if _WRAPPER_SO.exists() and _WRAPPER_SO.stat().st_mtime >= _WRAPPER_SRC.stat().st_mtime: return
  subprocess.run(["mcc","-x","musa","-mtgpu","--offload-arch=mp_22","-O2","-fPIC","-shared",
                  "-I/usr/local/musa/include","-L/usr/local/musa/lib","-lmudnn","-lmusart","-lmusa",
                  str(_WRAPPER_SRC),"-o",str(_WRAPPER_SO)], check=True)

_lib = None
def _load():
  global _lib
  if _lib is not None: return _lib
  _build_wrapper()
  _lib = ctypes.CDLL(str(_WRAPPER_SO))
  _lib.mudnn_tg_create_handle.restype = ctypes.c_void_p
  _lib.mudnn_tg_create_handle.argtypes = [ctypes.c_int]
  _lib.mudnn_tg_destroy_handle.argtypes = [ctypes.c_void_p]
  _lib.mudnn_tg_matmul.restype = ctypes.c_int
  _lib.mudnn_tg_matmul.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                    ctypes.c_long, ctypes.c_long, ctypes.c_long, ctypes.c_int, ctypes.c_int, ctypes.c_int]
  return _lib

_handles: dict[int, ctypes.c_void_p] = {}
def get_handle(device_id:int=0) -> ctypes.c_void_p:
  if device_id not in _handles:
    _handles[device_id] = ctypes.c_void_p(_load().mudnn_tg_create_handle(device_id))
  return _handles[device_id]

_DTYPE_CODE = {dtypes.half: 0, dtypes.bfloat16: 1, dtypes.float: 2}
def matmul(a_ptr:int, b_ptr:int, c_ptr:int, M:int, N:int, K:int, dtype:DType, ta:bool=False, tb:bool=False, device_id:int=0) -> int:
  code = _DTYPE_CODE.get(dtype.scalar())
  if code is None: raise ValueError(f"muDNN matmul: unsupported dtype {dtype}")
  return _load().mudnn_tg_matmul(get_handle(device_id), a_ptr, b_ptr, c_ptr, M, N, K, code, int(ta), int(tb))
