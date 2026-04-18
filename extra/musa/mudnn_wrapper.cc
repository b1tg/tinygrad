// extern "C" wrapper for muDNN MatMul so tinygrad can call it via ctypes.
// build: mcc -x musa -mtgpu -O2 -fPIC -shared -lmudnn -lmusart -lmusa mudnn_wrapper.cc -o libmudnn_tg.so
#include <mudnn.h>
#include <musa.h>
#include <musa_runtime.h>
#include <cstdio>

using namespace musa::dnn;

static Tensor::Type dtype_from_code(int code) {
  switch (code) {
    case 0: return Tensor::Type::HALF;
    case 1: return Tensor::Type::BFLOAT16;
    case 2: return Tensor::Type::FLOAT;
    default: return Tensor::Type::HALF;
  }
}

extern "C" {

// returns heap-allocated Handle*; caller keeps it for lifetime of app
void* mudnn_tg_create_handle(int device_id) {
  Handle* h = new Handle(device_id);
  return h;
}

void mudnn_tg_destroy_handle(void* h) { delete static_cast<Handle*>(h); }

// c = a @ b, shapes A=[M,K] B=[K,N] C=[M,N]; dtype_code: 0=half 1=bf16 2=float
// ta/tb: 0 = not transposed; if tb=1, B is [N,K] (common for weight matrices)
// Returns 0 on success, non-zero on failure.
int mudnn_tg_matmul(void* handle, void* a_dev, void* b_dev, void* c_dev,
                    long M, long N, long K, int dtype_code, int ta, int tb) {
  Handle& h = *static_cast<Handle*>(handle);
  Tensor::Type t = dtype_from_code(dtype_code);

  Tensor A, B, C;
  A.SetType(t); B.SetType(t); C.SetType(t);
  A.SetAddr(a_dev); B.SetAddr(b_dev); C.SetAddr(c_dev);

  // By default A is [M,K], B is [K,N] (tb=0), C is [M,N]
  long a_dims[2] = { ta ? K : M, ta ? M : K };
  long b_dims[2] = { tb ? N : K, tb ? K : N };
  long c_dims[2] = { M, N };
  A.SetNdInfo(2, a_dims);
  B.SetNdInfo(2, b_dims);
  C.SetNdInfo(2, c_dims);

  MatMul mm;
  mm.SetTranspose(ta != 0, tb != 0);
  auto st = mm.Run(h, C, A, B);
  return static_cast<int>(st);
}

}  // extern "C"
