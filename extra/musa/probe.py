import ctypes
musa = ctypes.CDLL("/usr/local/musa/lib/libmusa.so")

def chk(s, fn=""):
  if s != 0:
    p = ctypes.POINTER(ctypes.c_char)()
    musa.muGetErrorString(s, ctypes.byref(p))
    raise RuntimeError(f"{fn} -> {s}: {ctypes.string_at(p).decode() if p else ''}")

chk(musa.muInit(0), "muInit")
dev = ctypes.c_int()
chk(musa.muDeviceGet(ctypes.byref(dev), 0), "muDeviceGet")
print(f"device={dev.value}")
name = (ctypes.c_char * 256)()
chk(musa.muDeviceGetName(name, 256, dev), "muDeviceGetName")
print(f"name={name.value.decode()}")

ctx = ctypes.c_void_p()
chk(musa.muCtxCreate_v2(ctypes.byref(ctx), 0, dev), "muCtxCreate_v2")
print(f"ctx={hex(ctx.value)}")

with open("vadd.fatbin","rb") as f: img = f.read()
mod = ctypes.c_void_p()
chk(musa.muModuleLoadData(ctypes.byref(mod), img), "muModuleLoadData")
print(f"module={hex(mod.value)}")

fn = ctypes.c_void_p()
chk(musa.muModuleGetFunction(ctypes.byref(fn), mod, b"vadd"), "muModuleGetFunction")
print(f"func={hex(fn.value)}")

# alloc buffers
N = 16
sz = N*4
da, db, dc = ctypes.c_uint64(), ctypes.c_uint64(), ctypes.c_uint64()
chk(musa.muMemAlloc_v2(ctypes.byref(da), sz), "muMemAlloc da")
chk(musa.muMemAlloc_v2(ctypes.byref(db), sz), "muMemAlloc db")
chk(musa.muMemAlloc_v2(ctypes.byref(dc), sz), "muMemAlloc dc")
print(f"da={hex(da.value)} db={hex(db.value)} dc={hex(dc.value)}")

ha = (ctypes.c_float * N)(*[float(i) for i in range(N)])
hb = (ctypes.c_float * N)(*[float(i*10) for i in range(N)])
hc = (ctypes.c_float * N)()
chk(musa.muMemcpyHtoD_v2(da, ha, sz), "HtoD a")
chk(musa.muMemcpyHtoD_v2(db, hb, sz), "HtoD b")

# launch: vadd<<<1, N>>>(da, db, dc, N)
n_int = ctypes.c_int(N)
args = (ctypes.c_void_p * 4)(
    ctypes.cast(ctypes.pointer(da), ctypes.c_void_p),
    ctypes.cast(ctypes.pointer(db), ctypes.c_void_p),
    ctypes.cast(ctypes.pointer(dc), ctypes.c_void_p),
    ctypes.cast(ctypes.pointer(n_int), ctypes.c_void_p),
)
musa.muLaunchKernel.restype = ctypes.c_int
chk(musa.muLaunchKernel(fn, 1, 1, 1, N, 1, 1, 0, None, args, None), "muLaunchKernel")
chk(musa.muCtxSynchronize(), "muCtxSynchronize")

chk(musa.muMemcpyDtoH_v2(hc, dc, sz), "DtoH c")
print("result:", list(hc))
