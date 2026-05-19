import functools

from tinygrad import Context, Device, Tensor, dtypes
from tinygrad.renderer import Estimates
from tinygrad.uop.ops import KernelInfo, Ops, UOp

_DIM, _HIDDEN, _TOPK = 7168, 2048, 8
_Q4_BLOCK, _Q4_BLOCK_BYTES = 32, 18
_Q8_BLOCK_BYTES = 34
_ROUTER_EXPERTS, _VOCAB, _OUT_GROUP = 384, 163840, 32

@Context(ALLOW_DEVICE_USAGE=1)
def _arch(device:str) -> str: return Device[device].renderer.target.arch

_ROUTER_TOPK_HIP_SRC = r"""
#include <hip/hip_runtime.h>
#include <stdint.h>

constexpr int N = 384;
constexpr int K = 8;
constexpr int THREADS = 512;

extern "C" __global__ __launch_bounds__(THREADS) void kimi_router_topk(
    int* __restrict__ sel,
    float* __restrict__ probs_out,
    const float* __restrict__ logits,
    const _Float16* __restrict__ bias) {
  __shared__ float probs[N];
  __shared__ float scores[THREADS];
  __shared__ float best_s[THREADS];
  __shared__ int best_i[THREADS];
  __shared__ float norm;
  int tid = threadIdx.x;
  if (tid < N) {
    float x = logits[tid];
    float p = 1.0f / (1.0f + exp2f(-1.4426950408889634f * x));
    probs[tid] = p;
    scores[tid] = p + float(bias[tid]);
  } else scores[tid] = -3.4028234663852886e38f;
  __syncthreads();
  if (tid == 0) norm = 0.0f;
  #pragma unroll
  for (int k = 0; k < K; k++) {
    best_s[tid] = scores[tid];
    best_i[tid] = tid;
    __syncthreads();
    for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        float s = best_s[tid + stride];
        int i = best_i[tid + stride];
        if (s > best_s[tid] || (s == best_s[tid] && i < best_i[tid])) {
          best_s[tid] = s;
          best_i[tid] = i;
        }
      }
      __syncthreads();
    }
    int bi = best_i[0];
    if (tid == 0) {
      int outk = K - 1 - k;
      sel[outk] = bi;
      probs_out[outk] = probs[bi];
      norm += probs_out[outk];
    }
    if (tid == bi) scores[tid] = -3.4028234663852886e38f;
    __syncthreads();
  }
  if (tid < K) probs_out[tid] /= norm;
}
"""

@functools.cache
def _compiled_router_topk(arch:str) -> bytes:
  from tinygrad.runtime.support.compiler_amd import HIPCCCompiler
  return HIPCCCompiler(arch).compile_cached(_ROUTER_TOPK_HIP_SRC)

def _router_topk_kernel(sel:UOp, probs:UOp, logits:UOp, bias:UOp, device:str, arch:str) -> UOp:
  assert sel.numel() == _TOPK and probs.numel() == _TOPK and logits.numel() == _ROUTER_EXPERTS and bias.numel() == _ROUTER_EXPERTS
  estimates = Estimates(ops=_ROUTER_EXPERTS * 8 + _TOPK * _ROUTER_EXPERTS, mem=(_ROUTER_EXPERTS * 2 + _TOPK * 2) * 4)
  sink = UOp.sink(
    UOp.special(1, "gidx0"), UOp.special(512, "lidx0"), sel, probs, logits, bias,
    arg=KernelInfo(name="kimi_router_topk", estimates=estimates))
  return UOp(Ops.PROGRAM, src=(
    sink, UOp(Ops.DEVICE, arg=device), UOp(Ops.LINEAR, src=(*sink.src, sink)),
    UOp(Ops.SOURCE, arg=_ROUTER_TOPK_HIP_SRC), UOp(Ops.BINARY, arg=_compiled_router_topk(arch))))

def kimi_router_topk(logits:Tensor, bias:Tensor) -> tuple[Tensor, Tensor]:
  assert logits.numel() == _ROUTER_EXPERTS and bias.numel() == _ROUTER_EXPERTS
  sel = Tensor.empty(_TOPK, dtype=dtypes.int32, device=logits.device)
  probs = Tensor.empty(_TOPK, dtype=dtypes.float32, device=logits.device)
  sel, probs, *_ = Tensor.custom_kernel(
    sel, probs, logits.reshape(-1), bias.reshape(-1), fxn=functools.partial(_router_topk_kernel, device=logits.device, arch=_arch(logits.device)))
  return sel.reshape(*logits.shape[:-1], _TOPK), probs.reshape(*logits.shape[:-1], _TOPK)

_GATE_UP_Q4_0_HIP_SRC = r"""
#include <hip/hip_runtime.h>
#include <stdint.h>

constexpr int DIM = 7168;
constexpr int HIDDEN = 2048;
constexpr int TOPK = 8;
constexpr int THREADS = 32;
constexpr int Q4_BLOCK_BYTES = 18;

extern "C" __global__ __launch_bounds__(THREADS) void kimi_gate_up_q4_0(
    float* __restrict__ z,
    const float* __restrict__ x,
    const int* __restrict__ sel,
    const unsigned char* __restrict__ gate_w,
    const unsigned char* __restrict__ up_w) {
  int tid = threadIdx.x;
  int idx = blockIdx.x;
  int k = idx / HIDDEN;
  int row = idx - k * HIDDEN;
  int expert = sel[k];

  float gacc = 0.0f, uacc = 0.0f;
  for (int block = tid; block < DIM / 32; block += THREADS) {
    size_t base = ((size_t(expert) * HIDDEN + row) * (DIM / 32) + block) * Q4_BLOCK_BYTES;
    const unsigned char* gb = gate_w + base;
    const unsigned char* ub = up_w + base;
    float gs = float(*reinterpret_cast<const _Float16*>(gb));
    float us = float(*reinterpret_cast<const _Float16*>(ub));
    float gsum = 0.0f, usum = 0.0f;
    #pragma unroll
    for (int j = 0; j < 16; j++) {
      unsigned char gq = gb[2 + j], uq = ub[2 + j];
      float x0 = x[block * 32 + j];
      float x1 = x[block * 32 + j + 16];
      gsum += float((_Float16)(float(int(gq & 15) - 8) * gs)) * x0 + float((_Float16)(float(int(gq >> 4) - 8) * gs)) * x1;
      usum += float((_Float16)(float(int(uq & 15) - 8) * us)) * x0 + float((_Float16)(float(int(uq >> 4) - 8) * us)) * x1;
    }
    gacc += gsum;
    uacc += usum;
  }
  #pragma unroll
  for (int delta = 16; delta > 0; delta >>= 1) {
    gacc += __shfl_down(gacc, delta, 32);
    uacc += __shfl_down(uacc, delta, 32);
  }
  if (tid == 0) z[idx] = (gacc / (1.0f + exp2f(-1.4426950408889634f * gacc))) * uacc;
}
"""

@functools.cache
def _compiled_gate_up(arch:str) -> bytes:
  from tinygrad.runtime.support.compiler_amd import HIPCCCompiler
  return HIPCCCompiler(arch).compile_cached(_GATE_UP_Q4_0_HIP_SRC)

def _gate_up_kernel(z:UOp, x:UOp, sel:UOp, gate_w:UOp, up_w:UOp, device:str, arch:str) -> UOp:
  assert z.numel() == _TOPK * _HIDDEN, f"kimi_gate_up_q4_0 expects {_TOPK * _HIDDEN} outputs, got {z.numel()}"
  assert x.numel() == _DIM, f"kimi_gate_up_q4_0 expects dim={_DIM}, got {x.numel()}"
  assert sel.numel() == _TOPK, f"kimi_gate_up_q4_0 expects topk={_TOPK}, got {sel.numel()}"
  mem = _DIM * 4 + _TOPK * _HIDDEN * 4 + 2 * _TOPK * _HIDDEN * (_DIM // _Q4_BLOCK) * _Q4_BLOCK_BYTES
  ops = _TOPK * _HIDDEN * _DIM * 4 + _TOPK * _HIDDEN * 4
  sink = UOp.sink(
    UOp.special(_TOPK * _HIDDEN, "gidx0"), UOp.special(32, "lidx0"), z, x, sel, gate_w, up_w,
    arg=KernelInfo(name="kimi_gate_up_q4_0", estimates=Estimates(ops=ops, mem=mem)))
  return UOp(Ops.PROGRAM, src=(
    sink, UOp(Ops.DEVICE, arg=device), UOp(Ops.LINEAR, src=(*sink.src, sink)),
    UOp(Ops.SOURCE, arg=_GATE_UP_Q4_0_HIP_SRC), UOp(Ops.BINARY, arg=_compiled_gate_up(arch))))

def kimi_gate_up_q4_0(x:Tensor, sel:Tensor, gate_w:Tensor, up_w:Tensor) -> Tensor:
  assert x.numel() == _DIM and sel.numel() == _TOPK
  z = Tensor.empty(_TOPK, _HIDDEN, dtype=dtypes.float32, device=x.device)
  z, *_ = Tensor.custom_kernel(
    z, x.reshape(-1), sel.reshape(-1), gate_w.reshape(-1), up_w.reshape(-1),
    fxn=functools.partial(_gate_up_kernel, device=x.device, arch=_arch(x.device)))
  return z.reshape(*sel.shape, _HIDDEN)

_QUANT_Q8_0_HIP_SRC = r"""
#include <hip/hip_runtime.h>
#include <stdint.h>

constexpr int DIM = 7168;
constexpr int THREADS = 32;
constexpr int Q8_BLOCK_BYTES = 34;

extern "C" __global__ __launch_bounds__(THREADS) void kimi_quant_q8_0(
    unsigned char* __restrict__ q8,
    const float* __restrict__ x) {
  int tid = threadIdx.x;
  int block = blockIdx.x;
  float v = x[block * 32 + tid];
  float m = fabsf(v);
  #pragma unroll
  for (int delta = 16; delta > 0; delta >>= 1) m = fmaxf(m, __shfl_down(m, delta, 32));
  m = __shfl(m, 0, 32);
  float d = m / 127.0f;
  int q = d > 0.0f ? int(rintf(v / d)) : 0;
  q = q < -128 ? -128 : (q > 127 ? 127 : q);
  int base = block * Q8_BLOCK_BYTES;
  if (tid == 0) *reinterpret_cast<_Float16*>(q8 + base) = (_Float16)d;
  q8[base + 2 + tid] = (unsigned char)(int8_t)q;
}
"""

@functools.cache
def _compiled_quant_q8(arch:str) -> bytes:
  from tinygrad.runtime.support.compiler_amd import HIPCCCompiler
  return HIPCCCompiler(arch).compile_cached(_QUANT_Q8_0_HIP_SRC)

def _quant_q8_kernel(q8:UOp, x:UOp, device:str, arch:str) -> UOp:
  assert q8.numel() == (_DIM // _Q4_BLOCK) * _Q8_BLOCK_BYTES and x.numel() == _DIM
  sink = UOp.sink(
    UOp.special(_DIM // _Q4_BLOCK, "gidx0"), UOp.special(32, "lidx0"), q8, x,
    arg=KernelInfo(name="kimi_quant_q8_0", estimates=Estimates(ops=_DIM * 10, mem=_DIM * 4 + (_DIM // _Q4_BLOCK) * _Q8_BLOCK_BYTES)))
  return UOp(Ops.PROGRAM, src=(
    sink, UOp(Ops.DEVICE, arg=device), UOp(Ops.LINEAR, src=(*sink.src, sink)),
    UOp(Ops.SOURCE, arg=_QUANT_Q8_0_HIP_SRC), UOp(Ops.BINARY, arg=_compiled_quant_q8(arch))))

def kimi_quant_q8_0(x:Tensor) -> Tensor:
  assert x.numel() == _DIM
  q8 = Tensor.empty((_DIM // _Q4_BLOCK) * _Q8_BLOCK_BYTES, dtype=dtypes.uint8, device=x.device)
  q8, *_ = Tensor.custom_kernel(q8, x.reshape(-1), fxn=functools.partial(_quant_q8_kernel, device=x.device, arch=_arch(x.device)))
  return q8

_OUTPUT_ARGMAX_Q8_0_STAGE1_HIP_SRC = r"""
#include <hip/hip_runtime.h>
#include <stdint.h>

constexpr int DIM = 7168;
constexpr int VOCAB = 163840;
constexpr int GROUP = 32;
constexpr int THREADS = 1024;
constexpr int Q8_BLOCK_BYTES = 34;

__device__ __forceinline__ int pack4_i8(int a, int b, int c, int d) {
  return (a & 255) | ((b & 255) << 8) | ((c & 255) << 16) | ((d & 255) << 24);
}

extern "C" __global__ __launch_bounds__(THREADS) void kimi_output_argmax_q8_0_stage1(
    float* __restrict__ vals,
    int* __restrict__ idxs,
    const unsigned char* __restrict__ xq,
    const unsigned char* __restrict__ w) {
  __shared__ float svals[GROUP];
  __shared__ unsigned char xqs[(DIM / 32) * Q8_BLOCK_BYTES];
  int tid = threadIdx.x;
  int lane = tid & 31;
  int row_in = tid >> 5;
  int row = blockIdx.x * GROUP + row_in;
  for (int i = tid; i < (DIM / 32) * Q8_BLOCK_BYTES; i += THREADS) xqs[i] = xq[i];
  __syncthreads();
  float acc = 0.0f;
  for (int block = lane; block < DIM / 32; block += 32) {
    int xbase = block * Q8_BLOCK_BYTES;
    size_t wbase = (size_t(row) * (DIM / 32) + block) * Q8_BLOCK_BYTES;
    const unsigned char* xb = xqs + xbase;
    const unsigned char* wb = w + wbase;
    float xs = float(*reinterpret_cast<const _Float16*>(xb));
    float ws = float(*reinterpret_cast<const _Float16*>(wb));
    const int8_t* xqi = reinterpret_cast<const int8_t*>(xb + 2);
    const int8_t* wqi = reinterpret_cast<const int8_t*>(wb + 2);
    int dot = 0;
    #pragma unroll
    for (int j = 0; j < 32; j += 4) {
      int xp = pack4_i8(int(xqi[j + 0]), int(xqi[j + 1]), int(xqi[j + 2]), int(xqi[j + 3]));
      dot = __builtin_amdgcn_sdot4(pack4_i8(int(wqi[j + 0]), int(wqi[j + 1]), int(wqi[j + 2]), int(wqi[j + 3])), xp, dot, false);
    }
    acc += float(dot) * xs * ws;
  }
  #pragma unroll
  for (int delta = 16; delta > 0; delta >>= 1) acc += __shfl_down(acc, delta, 32);
  if (lane == 0) svals[row_in] = acc;
  __syncthreads();
  if (row_in == 0) {
    float v = svals[lane];
    int idx = blockIdx.x * GROUP + lane;
    #pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
      float ov = __shfl_down(v, delta, 32);
      int oi = __shfl_down(idx, delta, 32);
      if (ov > v || (ov == v && oi < idx)) {
        v = ov;
        idx = oi;
      }
    }
    if (lane == 0) {
      vals[blockIdx.x] = v;
      idxs[blockIdx.x] = idx;
    }
  }
}
"""

_OUTPUT_ARGMAX_Q8_0_STAGE2_HIP_SRC = r"""
#include <hip/hip_runtime.h>
#include <stdint.h>

constexpr int VOCAB = 163840;
constexpr int GROUP = 32;
constexpr int GROUPS = VOCAB / GROUP;
constexpr int THREADS = 1024;

extern "C" __global__ __launch_bounds__(THREADS) void kimi_output_argmax_q8_0_stage2(
    int* __restrict__ out,
    const float* __restrict__ vals,
    const int* __restrict__ idxs) {
  __shared__ float sv[THREADS];
  __shared__ int si[THREADS];
  int tid = threadIdx.x;
  float best = -3.4028234663852886e38f;
  int best_i = 0;
  for (int i = tid; i < GROUPS; i += THREADS) {
    float v = vals[i];
    int idx = idxs[i];
    if (v > best || (v == best && idx < best_i)) {
      best = v;
      best_i = idx;
    }
  }
  sv[tid] = best;
  si[tid] = best_i;
  __syncthreads();
  for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      float v = sv[tid + stride];
      int idx = si[tid + stride];
      if (v > sv[tid] || (v == sv[tid] && idx < si[tid])) {
        sv[tid] = v;
        si[tid] = idx;
      }
    }
    __syncthreads();
  }
  if (tid == 0) out[0] = si[0];
}
"""

@functools.cache
def _compiled_output_argmax_stage1(arch:str) -> bytes:
  from tinygrad.runtime.support.compiler_amd import HIPCCCompiler
  return HIPCCCompiler(arch).compile_cached(_OUTPUT_ARGMAX_Q8_0_STAGE1_HIP_SRC)

@functools.cache
def _compiled_output_argmax_stage2(arch:str) -> bytes:
  from tinygrad.runtime.support.compiler_amd import HIPCCCompiler
  return HIPCCCompiler(arch).compile_cached(_OUTPUT_ARGMAX_Q8_0_STAGE2_HIP_SRC)

def _output_argmax_stage1_kernel(vals:UOp, idxs:UOp, xq:UOp, w:UOp, device:str, arch:str) -> UOp:
  assert vals.numel() == _VOCAB // _OUT_GROUP and idxs.numel() == _VOCAB // _OUT_GROUP
  assert xq.numel() == (_DIM // _Q4_BLOCK) * _Q8_BLOCK_BYTES and w.numel() == _VOCAB * (_DIM // _Q4_BLOCK) * _Q8_BLOCK_BYTES
  mem = _VOCAB * (_DIM // _Q4_BLOCK) * _Q8_BLOCK_BYTES + (_DIM // _Q4_BLOCK) * _Q8_BLOCK_BYTES + (_VOCAB // _OUT_GROUP) * 8
  sink = UOp.sink(
    UOp.special(_VOCAB // _OUT_GROUP, "gidx0"), UOp.special(1024, "lidx0"), vals, idxs, xq, w,
    arg=KernelInfo(name="kimi_output_argmax_q8_0_stage1", estimates=Estimates(ops=_VOCAB * _DIM * 2, mem=mem)))
  return UOp(Ops.PROGRAM, src=(
    sink, UOp(Ops.DEVICE, arg=device), UOp(Ops.LINEAR, src=(*sink.src, sink)),
    UOp(Ops.SOURCE, arg=_OUTPUT_ARGMAX_Q8_0_STAGE1_HIP_SRC), UOp(Ops.BINARY, arg=_compiled_output_argmax_stage1(arch))))

def _output_argmax_stage2_kernel(out:UOp, vals:UOp, idxs:UOp, device:str, arch:str) -> UOp:
  assert out.numel() == 1 and vals.numel() == _VOCAB // _OUT_GROUP and idxs.numel() == _VOCAB // _OUT_GROUP
  sink = UOp.sink(
    UOp.special(1, "gidx0"), UOp.special(1024, "lidx0"), out, vals, idxs,
    arg=KernelInfo(name="kimi_output_argmax_q8_0_stage2", estimates=Estimates(ops=_VOCAB // _OUT_GROUP, mem=(_VOCAB // _OUT_GROUP) * 8)))
  return UOp(Ops.PROGRAM, src=(
    sink, UOp(Ops.DEVICE, arg=device), UOp(Ops.LINEAR, src=(*sink.src, sink)),
    UOp(Ops.SOURCE, arg=_OUTPUT_ARGMAX_Q8_0_STAGE2_HIP_SRC), UOp(Ops.BINARY, arg=_compiled_output_argmax_stage2(arch))))

def kimi_output_argmax_q8_0(x:Tensor, w:Tensor) -> Tensor:
  assert x.numel() == _DIM and w.numel() == _VOCAB * (_DIM // _Q4_BLOCK) * _Q8_BLOCK_BYTES
  xq = kimi_quant_q8_0(x)
  vals = Tensor.empty(_VOCAB // _OUT_GROUP, dtype=dtypes.float32, device=x.device)
  idxs = Tensor.empty(_VOCAB // _OUT_GROUP, dtype=dtypes.int32, device=x.device)
  vals, idxs, *_ = Tensor.custom_kernel(vals, idxs, xq.reshape(-1), w.reshape(-1),
    fxn=functools.partial(_output_argmax_stage1_kernel, device=x.device, arch=_arch(x.device)))
  out = Tensor.empty(1, dtype=dtypes.int32, device=x.device)
  out, *_ = Tensor.custom_kernel(out, vals, idxs, fxn=functools.partial(_output_argmax_stage2_kernel, device=x.device, arch=_arch(x.device)))
  return out.reshape(*x.shape[:-2], 1)

_GATE_UP_Q4_Q8_0_HIP_SRC = r"""
#include <hip/hip_runtime.h>
#include <stdint.h>

constexpr int DIM = 7168;
constexpr int HIDDEN = 2048;
constexpr int TOPK = 8;
constexpr int THREADS = 32;
constexpr int Q4_BLOCK_BYTES = 18;
constexpr int Q8_BLOCK_BYTES = 34;

__device__ __forceinline__ int pack4_i8(int a, int b, int c, int d) {
  return (a & 255) | ((b & 255) << 8) | ((c & 255) << 16) | ((d & 255) << 24);
}

extern "C" __global__ __launch_bounds__(THREADS) void kimi_gate_up_q4_q8_0(
    float* __restrict__ z,
    const unsigned char* __restrict__ xq,
    const int* __restrict__ sel,
    const unsigned char* __restrict__ gate_w,
    const unsigned char* __restrict__ up_w) {
  int tid = threadIdx.x;
  int idx = blockIdx.x;
  int k = idx / HIDDEN;
  int row = idx - k * HIDDEN;
  int expert = sel[k];
  float gacc = 0.0f, uacc = 0.0f;
  for (int block = tid; block < DIM / 32; block += THREADS) {
    size_t wbase = ((size_t(expert) * HIDDEN + row) * (DIM / 32) + block) * Q4_BLOCK_BYTES;
    int xbase = block * Q8_BLOCK_BYTES;
    const unsigned char* gb = gate_w + wbase;
    const unsigned char* ub = up_w + wbase;
    const unsigned char* xb = xq + xbase;
    float gs = float(*reinterpret_cast<const _Float16*>(gb));
    float us = float(*reinterpret_cast<const _Float16*>(ub));
    float xs = float(*reinterpret_cast<const _Float16*>(xb));
    const int8_t* xqi = reinterpret_cast<const int8_t*>(xb + 2);
    int gdot = 0, udot = 0;
    #pragma unroll
    for (int j = 0; j < 16; j += 4) {
      unsigned char g0 = gb[2 + j + 0], g1 = gb[2 + j + 1], g2 = gb[2 + j + 2], g3 = gb[2 + j + 3];
      unsigned char u0 = ub[2 + j + 0], u1 = ub[2 + j + 1], u2 = ub[2 + j + 2], u3 = ub[2 + j + 3];
      int xl = pack4_i8(int(xqi[j + 0]), int(xqi[j + 1]), int(xqi[j + 2]), int(xqi[j + 3]));
      int xh = pack4_i8(int(xqi[j + 16]), int(xqi[j + 17]), int(xqi[j + 18]), int(xqi[j + 19]));
      gdot = __builtin_amdgcn_sdot4(pack4_i8(int(g0 & 15) - 8, int(g1 & 15) - 8, int(g2 & 15) - 8, int(g3 & 15) - 8), xl, gdot, false);
      udot = __builtin_amdgcn_sdot4(pack4_i8(int(u0 & 15) - 8, int(u1 & 15) - 8, int(u2 & 15) - 8, int(u3 & 15) - 8), xl, udot, false);
      gdot = __builtin_amdgcn_sdot4(pack4_i8(int(g0 >> 4) - 8, int(g1 >> 4) - 8, int(g2 >> 4) - 8, int(g3 >> 4) - 8), xh, gdot, false);
      udot = __builtin_amdgcn_sdot4(pack4_i8(int(u0 >> 4) - 8, int(u1 >> 4) - 8, int(u2 >> 4) - 8, int(u3 >> 4) - 8), xh, udot, false);
    }
    gacc += float(gdot) * gs * xs;
    uacc += float(udot) * us * xs;
  }
  #pragma unroll
  for (int delta = 16; delta > 0; delta >>= 1) {
    gacc += __shfl_down(gacc, delta, 32);
    uacc += __shfl_down(uacc, delta, 32);
  }
  if (tid == 0) z[idx] = (gacc / (1.0f + exp2f(-1.4426950408889634f * gacc))) * uacc;
}
"""

@functools.cache
def _compiled_gate_up_q8(arch:str) -> bytes:
  from tinygrad.runtime.support.compiler_amd import HIPCCCompiler
  return HIPCCCompiler(arch).compile_cached(_GATE_UP_Q4_Q8_0_HIP_SRC)

def _gate_up_q8_kernel(z:UOp, xq:UOp, sel:UOp, gate_w:UOp, up_w:UOp, device:str, arch:str) -> UOp:
  assert z.numel() == _TOPK * _HIDDEN and xq.numel() == (_DIM // _Q4_BLOCK) * _Q8_BLOCK_BYTES and sel.numel() == _TOPK
  mem = _TOPK * _HIDDEN * 4 + (_DIM // _Q4_BLOCK) * _Q8_BLOCK_BYTES + 2 * _TOPK * _HIDDEN * (_DIM // _Q4_BLOCK) * _Q4_BLOCK_BYTES
  ops = _TOPK * _HIDDEN * _DIM * 4
  sink = UOp.sink(
    UOp.special(_TOPK * _HIDDEN, "gidx0"), UOp.special(32, "lidx0"), z, xq, sel, gate_w, up_w,
    arg=KernelInfo(name="kimi_gate_up_q4_q8_0", estimates=Estimates(ops=ops, mem=mem)))
  return UOp(Ops.PROGRAM, src=(
    sink, UOp(Ops.DEVICE, arg=device), UOp(Ops.LINEAR, src=(*sink.src, sink)),
    UOp(Ops.SOURCE, arg=_GATE_UP_Q4_Q8_0_HIP_SRC), UOp(Ops.BINARY, arg=_compiled_gate_up_q8(arch))))

def kimi_gate_up_q4_q8_0(x:Tensor, sel:Tensor, gate_w:Tensor, up_w:Tensor) -> Tensor:
  assert x.numel() == _DIM and sel.numel() == _TOPK
  z = Tensor.empty(_TOPK, _HIDDEN, dtype=dtypes.float32, device=x.device)
  z, *_ = Tensor.custom_kernel(
    z, kimi_quant_q8_0(x), sel.reshape(-1), gate_w.reshape(-1), up_w.reshape(-1),
    fxn=functools.partial(_gate_up_q8_kernel, device=x.device, arch=_arch(x.device)))
  return z.reshape(*sel.shape, _HIDDEN)

_GATE_UP_Q4_Q8_TO_Q8_0_HIP_SRC = r"""
#include <hip/hip_runtime.h>
#include <stdint.h>

constexpr int DIM = 7168;
constexpr int HIDDEN = 2048;
constexpr int TOPK = 8;
constexpr int THREADS = 1024;
constexpr int Q4_BLOCK_BYTES = 18;
constexpr int Q8_BLOCK_BYTES = 34;
constexpr int XQ_BLOCKS = DIM / 32;

__device__ __forceinline__ int pack4_i8(int a, int b, int c, int d) {
  return (a & 255) | ((b & 255) << 8) | ((c & 255) << 16) | ((d & 255) << 24);
}
__device__ __forceinline__ int q4sign(uint32_t r) {
  uint32_t s = r & 0x08080808u;
  return int(r | (s << 1) | (s << 2) | (s << 3) | (s << 4));
}
__device__ __forceinline__ int q4lo(uint32_t p) { return q4sign((p & 0x0f0f0f0fu) ^ 0x08080808u); }
__device__ __forceinline__ int q4hi(uint32_t p) { return q4sign(((p >> 4) & 0x0f0f0f0fu) ^ 0x08080808u); }

extern "C" __global__ __launch_bounds__(THREADS) void kimi_gate_up_q4_q8_to_q8_0(
    unsigned char* __restrict__ zq,
    const unsigned char* __restrict__ xq,
    const int* __restrict__ sel,
    const unsigned char* __restrict__ gate_w,
    const unsigned char* __restrict__ up_w) {
  __shared__ int xps[XQ_BLOCKS * 8];
  __shared__ float xss[XQ_BLOCKS];
  __shared__ float vals[32];
  int tid = threadIdx.x;
  for (int i = tid; i < XQ_BLOCKS * 8; i += THREADS) {
    int block = i >> 3;
    int j = (i & 7) << 2;
    const int8_t* xqi = reinterpret_cast<const int8_t*>(xq + block * Q8_BLOCK_BYTES + 2);
    xps[i] = pack4_i8(int(xqi[j + 0]), int(xqi[j + 1]), int(xqi[j + 2]), int(xqi[j + 3]));
  }
  for (int i = tid; i < XQ_BLOCKS; i += THREADS) xss[i] = float(*reinterpret_cast<const _Float16*>(xq + i * Q8_BLOCK_BYTES));
  __syncthreads();
  int lane = tid & 31;
  int r = tid >> 5;
  int k = blockIdx.x / (HIDDEN / 32);
  int hblock = blockIdx.x - k * (HIDDEN / 32);
  int row = hblock * 32 + r;
  int expert = sel[k];
  float gacc = 0.0f, uacc = 0.0f;
  for (int block = lane; block < DIM / 32; block += 32) {
    size_t wbase = ((size_t(expert) * HIDDEN + row) * (DIM / 32) + block) * Q4_BLOCK_BYTES;
    const unsigned char* gb = gate_w + wbase;
    const unsigned char* ub = up_w + wbase;
    float gs = float(*reinterpret_cast<const _Float16*>(gb));
    float us = float(*reinterpret_cast<const _Float16*>(ub));
    float xs = xss[block];
    int gdot = 0, udot = 0;
    #pragma unroll
    for (int j = 0; j < 16; j += 4) {
      uint32_t gp = *reinterpret_cast<const uint32_t*>(gb + 2 + j);
      uint32_t up = *reinterpret_cast<const uint32_t*>(ub + 2 + j);
      int xl = xps[block * 8 + (j >> 2)];
      int xh = xps[block * 8 + ((j + 16) >> 2)];
      gdot = __builtin_amdgcn_sdot4(q4lo(gp), xl, gdot, false);
      udot = __builtin_amdgcn_sdot4(q4lo(up), xl, udot, false);
      gdot = __builtin_amdgcn_sdot4(q4hi(gp), xh, gdot, false);
      udot = __builtin_amdgcn_sdot4(q4hi(up), xh, udot, false);
    }
    gacc += float(gdot) * gs * xs;
    uacc += float(udot) * us * xs;
  }
  #pragma unroll
  for (int delta = 16; delta > 0; delta >>= 1) {
    gacc += __shfl_down(gacc, delta, 32);
    uacc += __shfl_down(uacc, delta, 32);
  }
  if (lane == 0) vals[r] = (gacc / (1.0f + exp2f(-1.4426950408889634f * gacc))) * uacc;
  __syncthreads();
  if (r == 0) {
    float v = vals[lane];
    float m = fabsf(v);
    #pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) m = fmaxf(m, __shfl_down(m, delta, 32));
    m = __shfl(m, 0, 32);
    float d = m / 127.0f;
    int q = d > 0.0f ? int(rintf(v / d)) : 0;
    q = q < -128 ? -128 : (q > 127 ? 127 : q);
    int base = (k * (HIDDEN / 32) + hblock) * Q8_BLOCK_BYTES;
    if (lane == 0) *reinterpret_cast<_Float16*>(zq + base) = (_Float16)d;
    zq[base + 2 + lane] = (unsigned char)(int8_t)q;
  }
}
"""

@functools.cache
def _compiled_gate_up_to_q8(arch:str) -> bytes:
  from tinygrad.runtime.support.compiler_amd import HIPCCCompiler
  return HIPCCCompiler(arch).compile_cached(_GATE_UP_Q4_Q8_TO_Q8_0_HIP_SRC)

def _gate_up_to_q8_kernel(zq:UOp, xq:UOp, sel:UOp, gate_w:UOp, up_w:UOp, device:str, arch:str) -> UOp:
  assert zq.numel() == _TOPK * (_HIDDEN // _Q4_BLOCK) * _Q8_BLOCK_BYTES and xq.numel() == (_DIM // _Q4_BLOCK) * _Q8_BLOCK_BYTES
  assert sel.numel() == _TOPK
  mem = _TOPK * (_HIDDEN // _Q4_BLOCK) * _Q8_BLOCK_BYTES + (_DIM // _Q4_BLOCK) * _Q8_BLOCK_BYTES + \
    2 * _TOPK * _HIDDEN * (_DIM // _Q4_BLOCK) * _Q4_BLOCK_BYTES
  ops = _TOPK * _HIDDEN * _DIM * 4
  sink = UOp.sink(
    UOp.special(_TOPK * (_HIDDEN // _Q4_BLOCK), "gidx0"), UOp.special(1024, "lidx0"), zq, xq, sel, gate_w, up_w,
    arg=KernelInfo(name="kimi_gate_up_q4_q8_to_q8_0", estimates=Estimates(ops=ops, mem=mem)))
  return UOp(Ops.PROGRAM, src=(
    sink, UOp(Ops.DEVICE, arg=device), UOp(Ops.LINEAR, src=(*sink.src, sink)),
    UOp(Ops.SOURCE, arg=_GATE_UP_Q4_Q8_TO_Q8_0_HIP_SRC), UOp(Ops.BINARY, arg=_compiled_gate_up_to_q8(arch))))

def kimi_gate_up_q4_q8_to_q8_from_q8_0(xq:Tensor, sel:Tensor, gate_w:Tensor, up_w:Tensor) -> Tensor:
  assert xq.numel() == (_DIM // _Q4_BLOCK) * _Q8_BLOCK_BYTES and sel.numel() == _TOPK
  zq = Tensor.empty(_TOPK * (_HIDDEN // _Q4_BLOCK) * _Q8_BLOCK_BYTES, dtype=dtypes.uint8, device=xq.device)
  zq, *_ = Tensor.custom_kernel(
    zq, xq.reshape(-1), sel.reshape(-1), gate_w.reshape(-1), up_w.reshape(-1),
    fxn=functools.partial(_gate_up_to_q8_kernel, device=xq.device, arch=_arch(xq.device)))
  return zq

def kimi_gate_up_q4_q8_to_q8_0(x:Tensor, sel:Tensor, gate_w:Tensor, up_w:Tensor) -> Tensor:
  assert x.numel() == _DIM and sel.numel() == _TOPK
  return kimi_gate_up_q4_q8_to_q8_from_q8_0(kimi_quant_q8_0(x), sel, gate_w, up_w)

_SHARED_GATE_UP_Q8_0_HIP_SRC = r"""
#include <hip/hip_runtime.h>
#include <stdint.h>

constexpr int DIM = 7168;
constexpr int HIDDEN = 2048;
constexpr int THREADS = 32;
constexpr int Q8_BLOCK_BYTES = 34;

__device__ __forceinline__ int pack4_i8(int a, int b, int c, int d) {
  return (a & 255) | ((b & 255) << 8) | ((c & 255) << 16) | ((d & 255) << 24);
}

extern "C" __global__ __launch_bounds__(THREADS) void kimi_shared_gate_up_q8_0(
    float* __restrict__ z,
    const unsigned char* __restrict__ xq,
    const unsigned char* __restrict__ gate_w,
    const unsigned char* __restrict__ up_w) {
  int tid = threadIdx.x;
  int row = blockIdx.x;
  float gacc = 0.0f, uacc = 0.0f;
  for (int block = tid; block < DIM / 32; block += THREADS) {
    int xbase = block * Q8_BLOCK_BYTES;
    size_t wbase = (size_t(row) * (DIM / 32) + block) * Q8_BLOCK_BYTES;
    const unsigned char* xb = xq + xbase;
    const unsigned char* gb = gate_w + wbase;
    const unsigned char* ub = up_w + wbase;
    float xs = float(*reinterpret_cast<const _Float16*>(xb));
    float gs = float(*reinterpret_cast<const _Float16*>(gb));
    float us = float(*reinterpret_cast<const _Float16*>(ub));
    const int8_t* xqi = reinterpret_cast<const int8_t*>(xb + 2);
    const int8_t* gqi = reinterpret_cast<const int8_t*>(gb + 2);
    const int8_t* uqi = reinterpret_cast<const int8_t*>(ub + 2);
    int gdot = 0, udot = 0;
    #pragma unroll
    for (int j = 0; j < 32; j += 4) {
      int xp = pack4_i8(int(xqi[j + 0]), int(xqi[j + 1]), int(xqi[j + 2]), int(xqi[j + 3]));
      gdot = __builtin_amdgcn_sdot4(pack4_i8(int(gqi[j + 0]), int(gqi[j + 1]), int(gqi[j + 2]), int(gqi[j + 3])), xp, gdot, false);
      udot = __builtin_amdgcn_sdot4(pack4_i8(int(uqi[j + 0]), int(uqi[j + 1]), int(uqi[j + 2]), int(uqi[j + 3])), xp, udot, false);
    }
    gacc += float(gdot) * gs * xs;
    uacc += float(udot) * us * xs;
  }
  #pragma unroll
  for (int delta = 16; delta > 0; delta >>= 1) {
    gacc += __shfl_down(gacc, delta, 32);
    uacc += __shfl_down(uacc, delta, 32);
  }
  if (tid == 0) z[row] = (gacc / (1.0f + exp2f(-1.4426950408889634f * gacc))) * uacc;
}
"""

@functools.cache
def _compiled_shared_gate_up_q8(arch:str) -> bytes:
  from tinygrad.runtime.support.compiler_amd import HIPCCCompiler
  return HIPCCCompiler(arch).compile_cached(_SHARED_GATE_UP_Q8_0_HIP_SRC)

def _shared_gate_up_q8_kernel(z:UOp, xq:UOp, gate_w:UOp, up_w:UOp, device:str, arch:str) -> UOp:
  assert z.numel() == _HIDDEN and xq.numel() == (_DIM // _Q4_BLOCK) * _Q8_BLOCK_BYTES
  assert gate_w.numel() == _HIDDEN * (_DIM // _Q4_BLOCK) * _Q8_BLOCK_BYTES and up_w.numel() == _HIDDEN * (_DIM // _Q4_BLOCK) * _Q8_BLOCK_BYTES
  mem = _HIDDEN * 4 + (_DIM // _Q4_BLOCK) * _Q8_BLOCK_BYTES + 2 * _HIDDEN * (_DIM // _Q4_BLOCK) * _Q8_BLOCK_BYTES
  sink = UOp.sink(
    UOp.special(_HIDDEN, "gidx0"), UOp.special(32, "lidx0"), z, xq, gate_w, up_w,
    arg=KernelInfo(name="kimi_shared_gate_up_q8_0", estimates=Estimates(ops=_HIDDEN * _DIM * 4, mem=mem)))
  return UOp(Ops.PROGRAM, src=(
    sink, UOp(Ops.DEVICE, arg=device), UOp(Ops.LINEAR, src=(*sink.src, sink)),
    UOp(Ops.SOURCE, arg=_SHARED_GATE_UP_Q8_0_HIP_SRC), UOp(Ops.BINARY, arg=_compiled_shared_gate_up_q8(arch))))

def kimi_shared_gate_up_q8_from_q8_0(xq:Tensor, gate_w:Tensor, up_w:Tensor) -> Tensor:
  assert xq.numel() == (_DIM // _Q4_BLOCK) * _Q8_BLOCK_BYTES
  z = Tensor.empty(_HIDDEN, dtype=dtypes.float32, device=xq.device)
  z, *_ = Tensor.custom_kernel(
    z, xq.reshape(-1), gate_w.reshape(-1), up_w.reshape(-1),
    fxn=functools.partial(_shared_gate_up_q8_kernel, device=xq.device, arch=_arch(xq.device)))
  return z

_DOWN_REDUCE_Q4_0_HIP_SRC = r"""
#include <hip/hip_runtime.h>
#include <stdint.h>

constexpr int DIM = 7168;
constexpr int HIDDEN = 2048;
constexpr int TOPK = 8;
constexpr int THREADS = 32;
constexpr int Q4_BLOCK_BYTES = 18;

extern "C" __global__ __launch_bounds__(THREADS) void kimi_down_reduce_q4_0(
    float* __restrict__ z,
    const float* __restrict__ gate_up,
    const int* __restrict__ sel,
    const float* __restrict__ probs,
    const unsigned char* __restrict__ down_w) {
  int tid = threadIdx.x;
  int row = blockIdx.x;
  float acc = 0.0f;
  #pragma unroll
  for (int k = 0; k < TOPK; k++) {
    int expert = sel[k];
    float sum = 0.0f;
    for (int block = tid; block < HIDDEN / 32; block += THREADS) {
      size_t base = ((size_t(expert) * DIM + row) * (HIDDEN / 32) + block) * Q4_BLOCK_BYTES;
      const unsigned char* wb = down_w + base;
      float ws = float(*reinterpret_cast<const _Float16*>(wb));
      #pragma unroll
      for (int j = 0; j < 16; j++) {
        unsigned char wq = wb[2 + j];
        float x0 = gate_up[k * HIDDEN + block * 32 + j];
        float x1 = gate_up[k * HIDDEN + block * 32 + j + 16];
        sum += float((_Float16)(float(int(wq & 15) - 8) * ws)) * x0 + float((_Float16)(float(int(wq >> 4) - 8) * ws)) * x1;
      }
    }
    #pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) sum += __shfl_down(sum, delta, 32);
    if (tid == 0) acc += sum * probs[k];
  }
  if (tid == 0) z[row] = acc;
}
"""

@functools.cache
def _compiled_down_reduce(arch:str) -> bytes:
  from tinygrad.runtime.support.compiler_amd import HIPCCCompiler
  return HIPCCCompiler(arch).compile_cached(_DOWN_REDUCE_Q4_0_HIP_SRC)

def _down_reduce_kernel(z:UOp, gate_up:UOp, sel:UOp, probs:UOp, down_w:UOp, device:str, arch:str) -> UOp:
  assert z.numel() == _DIM, f"kimi_down_reduce_q4_0 expects dim={_DIM}, got {z.numel()}"
  assert gate_up.numel() == _TOPK * _HIDDEN, f"kimi_down_reduce_q4_0 expects {_TOPK * _HIDDEN} gate_up, got {gate_up.numel()}"
  assert sel.numel() == _TOPK and probs.numel() == _TOPK
  mem = _TOPK * _HIDDEN * 4 + _TOPK * 8 + _DIM * 4 + _TOPK * _DIM * (_HIDDEN // _Q4_BLOCK) * _Q4_BLOCK_BYTES
  ops = _TOPK * _DIM * _HIDDEN * 2 + _TOPK * _DIM * 2
  sink = UOp.sink(
    UOp.special(_DIM, "gidx0"), UOp.special(32, "lidx0"), z, gate_up, sel, probs, down_w,
    arg=KernelInfo(name="kimi_down_reduce_q4_0", estimates=Estimates(ops=ops, mem=mem)))
  return UOp(Ops.PROGRAM, src=(
    sink, UOp(Ops.DEVICE, arg=device), UOp(Ops.LINEAR, src=(*sink.src, sink)),
    UOp(Ops.SOURCE, arg=_DOWN_REDUCE_Q4_0_HIP_SRC), UOp(Ops.BINARY, arg=_compiled_down_reduce(arch))))

def kimi_down_reduce_q4_0(gate_up:Tensor, sel:Tensor, probs:Tensor, down_w:Tensor) -> Tensor:
  assert gate_up.numel() == _TOPK * _HIDDEN and sel.numel() == _TOPK and probs.numel() == _TOPK
  z = Tensor.empty(_DIM, dtype=dtypes.float32, device=gate_up.device)
  z, *_ = Tensor.custom_kernel(
    z, gate_up.reshape(-1), sel.reshape(-1), probs.reshape(-1), down_w.reshape(-1),
    fxn=functools.partial(_down_reduce_kernel, device=gate_up.device, arch=_arch(gate_up.device)))
  return z.reshape(*probs.shape[:-1], _DIM)

_QUANT_GATE_UP_Q8_0_HIP_SRC = r"""
#include <hip/hip_runtime.h>
#include <stdint.h>

constexpr int HIDDEN = 2048;
constexpr int TOPK = 8;
constexpr int THREADS = 32;
constexpr int Q8_BLOCK_BYTES = 34;

extern "C" __global__ __launch_bounds__(THREADS) void kimi_quant_gate_up_q8_0(
    unsigned char* __restrict__ q8,
    const float* __restrict__ x) {
  int tid = threadIdx.x;
  int block = blockIdx.x;
  float v = x[block * 32 + tid];
  float m = fabsf(v);
  #pragma unroll
  for (int delta = 16; delta > 0; delta >>= 1) m = fmaxf(m, __shfl_down(m, delta, 32));
  m = __shfl(m, 0, 32);
  float d = m / 127.0f;
  int q = d > 0.0f ? int(rintf(v / d)) : 0;
  q = q < -128 ? -128 : (q > 127 ? 127 : q);
  int base = block * Q8_BLOCK_BYTES;
  if (tid == 0) *reinterpret_cast<_Float16*>(q8 + base) = (_Float16)d;
  q8[base + 2 + tid] = (unsigned char)(int8_t)q;
}
"""

@functools.cache
def _compiled_quant_gate_up_q8(arch:str) -> bytes:
  from tinygrad.runtime.support.compiler_amd import HIPCCCompiler
  return HIPCCCompiler(arch).compile_cached(_QUANT_GATE_UP_Q8_0_HIP_SRC)

def _quant_gate_up_q8_kernel(q8:UOp, x:UOp, device:str, arch:str) -> UOp:
  assert q8.numel() == _TOPK * (_HIDDEN // _Q4_BLOCK) * _Q8_BLOCK_BYTES and x.numel() == _TOPK * _HIDDEN
  sink = UOp.sink(
    UOp.special(_TOPK * (_HIDDEN // _Q4_BLOCK), "gidx0"), UOp.special(32, "lidx0"), q8, x,
    arg=KernelInfo(name="kimi_quant_gate_up_q8_0", estimates=Estimates(
      ops=_TOPK * _HIDDEN * 10, mem=_TOPK * _HIDDEN * 4 + _TOPK * (_HIDDEN // _Q4_BLOCK) * _Q8_BLOCK_BYTES)))
  return UOp(Ops.PROGRAM, src=(
    sink, UOp(Ops.DEVICE, arg=device), UOp(Ops.LINEAR, src=(*sink.src, sink)),
    UOp(Ops.SOURCE, arg=_QUANT_GATE_UP_Q8_0_HIP_SRC), UOp(Ops.BINARY, arg=_compiled_quant_gate_up_q8(arch))))

def kimi_quant_gate_up_q8_0(gate_up:Tensor) -> Tensor:
  assert gate_up.numel() == _TOPK * _HIDDEN
  q8 = Tensor.empty(_TOPK * (_HIDDEN // _Q4_BLOCK) * _Q8_BLOCK_BYTES, dtype=dtypes.uint8, device=gate_up.device)
  q8, *_ = Tensor.custom_kernel(
    q8, gate_up.reshape(-1), fxn=functools.partial(_quant_gate_up_q8_kernel, device=gate_up.device, arch=_arch(gate_up.device)))
  return q8

_DOWN_REDUCE_Q4_Q8_0_HIP_SRC = r"""
#include <hip/hip_runtime.h>
#include <stdint.h>

constexpr int DIM = 7168;
constexpr int HIDDEN = 2048;
constexpr int TOPK = 8;
constexpr int THREADS = 512;
constexpr int ROWS = 8;
constexpr int WARPS_PER_ROW = 2;
constexpr int Q4_BLOCK_BYTES = 18;
constexpr int Q8_BLOCK_BYTES = 34;
constexpr int XQ_BLOCKS = TOPK * (HIDDEN / 32);

__device__ __forceinline__ int pack4_i8(int a, int b, int c, int d) {
  return (a & 255) | ((b & 255) << 8) | ((c & 255) << 16) | ((d & 255) << 24);
}

extern "C" __global__ __launch_bounds__(THREADS) void kimi_down_reduce_q4_q8_0(
    float* __restrict__ z,
    const unsigned char* __restrict__ gate_up_q8,
    const int* __restrict__ sel,
    const float* __restrict__ probs,
    const unsigned char* __restrict__ down_w) {
  __shared__ int xps[XQ_BLOCKS * 8];
  __shared__ float xss[XQ_BLOCKS];
  __shared__ float partial[ROWS * WARPS_PER_ROW];
  int tid = threadIdx.x;
  for (int i = tid; i < XQ_BLOCKS * 8; i += THREADS) {
    int block = i >> 3;
    int j = (i & 7) << 2;
    const int8_t* xqi = reinterpret_cast<const int8_t*>(gate_up_q8 + block * Q8_BLOCK_BYTES + 2);
    xps[i] = pack4_i8(int(xqi[j + 0]), int(xqi[j + 1]), int(xqi[j + 2]), int(xqi[j + 3]));
  }
  for (int i = tid; i < XQ_BLOCKS; i += THREADS) xss[i] = float(*reinterpret_cast<const _Float16*>(gate_up_q8 + i * Q8_BLOCK_BYTES));
  __syncthreads();
  int lane = tid & 31;
  int warp_in_block = tid >> 5;
  int row_in = warp_in_block / WARPS_PER_ROW;
  int warp = warp_in_block - row_in * WARPS_PER_ROW;
  int row = blockIdx.x * ROWS + row_in;
  float acc = 0.0f;
  #pragma unroll
  for (int k = 0; k < TOPK; k++) {
    int expert = sel[k];
    float sum = 0.0f;
    for (int block = warp * 32 + lane; block < HIDDEN / 32; block += 32 * WARPS_PER_ROW) {
      size_t wbase = ((size_t(expert) * DIM + row) * (HIDDEN / 32) + block) * Q4_BLOCK_BYTES;
      int xidx = k * (HIDDEN / 32) + block;
      const unsigned char* wb = down_w + wbase;
      float ws = float(*reinterpret_cast<const _Float16*>(wb));
      float xs = xss[xidx];
      int dot = 0;
      #pragma unroll
      for (int j = 0; j < 16; j += 4) {
        unsigned char w0 = wb[2 + j + 0], w1 = wb[2 + j + 1], w2 = wb[2 + j + 2], w3 = wb[2 + j + 3];
        int xl = xps[xidx * 8 + (j >> 2)];
        int xh = xps[xidx * 8 + ((j + 16) >> 2)];
        dot = __builtin_amdgcn_sdot4(pack4_i8(int(w0 & 15) - 8, int(w1 & 15) - 8, int(w2 & 15) - 8, int(w3 & 15) - 8), xl, dot, false);
        dot = __builtin_amdgcn_sdot4(pack4_i8(int(w0 >> 4) - 8, int(w1 >> 4) - 8, int(w2 >> 4) - 8, int(w3 >> 4) - 8), xh, dot, false);
      }
      sum += float(dot) * ws * xs;
    }
    #pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) sum += __shfl_down(sum, delta, 32);
    if (lane == 0) partial[row_in * WARPS_PER_ROW + warp] = sum;
    __syncthreads();
    if (warp == 0 && lane == 0) {
      float total = 0.0f;
      #pragma unroll
      for (int w = 0; w < WARPS_PER_ROW; w++) total += partial[row_in * WARPS_PER_ROW + w];
      acc += total * probs[k];
    }
    __syncthreads();
  }
  if (warp == 0 && lane == 0) z[row] = acc;
}
"""

@functools.cache
def _compiled_down_reduce_q8(arch:str) -> bytes:
  from tinygrad.runtime.support.compiler_amd import HIPCCCompiler
  return HIPCCCompiler(arch).compile_cached(_DOWN_REDUCE_Q4_Q8_0_HIP_SRC)

def _down_reduce_q8_kernel(z:UOp, gate_up_q8:UOp, sel:UOp, probs:UOp, down_w:UOp, device:str, arch:str) -> UOp:
  assert z.numel() == _DIM and gate_up_q8.numel() == _TOPK * (_HIDDEN // _Q4_BLOCK) * _Q8_BLOCK_BYTES and sel.numel() == _TOPK
  mem = _TOPK * (_HIDDEN // _Q4_BLOCK) * _Q8_BLOCK_BYTES + _TOPK * 8 + _DIM * 4 + _TOPK * _DIM * (_HIDDEN // _Q4_BLOCK) * _Q4_BLOCK_BYTES
  ops = _TOPK * _DIM * _HIDDEN * 2 + _TOPK * _DIM * 2
  sink = UOp.sink(
    UOp.special(_DIM // 8, "gidx0"), UOp.special(512, "lidx0"), z, gate_up_q8, sel, probs, down_w,
    arg=KernelInfo(name="kimi_down_reduce_q4_q8_0", estimates=Estimates(ops=ops, mem=mem)))
  return UOp(Ops.PROGRAM, src=(
    sink, UOp(Ops.DEVICE, arg=device), UOp(Ops.LINEAR, src=(*sink.src, sink)),
    UOp(Ops.SOURCE, arg=_DOWN_REDUCE_Q4_Q8_0_HIP_SRC), UOp(Ops.BINARY, arg=_compiled_down_reduce_q8(arch))))

def kimi_down_reduce_q4_q8_from_q8_0(gate_up_q8:Tensor, sel:Tensor, probs:Tensor, down_w:Tensor) -> Tensor:
  assert gate_up_q8.numel() == _TOPK * (_HIDDEN // _Q4_BLOCK) * _Q8_BLOCK_BYTES and sel.numel() == _TOPK and probs.numel() == _TOPK
  z = Tensor.empty(_DIM, dtype=dtypes.float32, device=gate_up_q8.device)
  z, *_ = Tensor.custom_kernel(
    z, gate_up_q8.reshape(-1), sel.reshape(-1), probs.reshape(-1), down_w.reshape(-1),
    fxn=functools.partial(_down_reduce_q8_kernel, device=gate_up_q8.device, arch=_arch(gate_up_q8.device)))
  return z.reshape(*probs.shape[:-1], _DIM)

def kimi_down_reduce_q4_q8_0(gate_up:Tensor, sel:Tensor, probs:Tensor, down_w:Tensor) -> Tensor:
  assert gate_up.numel() == _TOPK * _HIDDEN and sel.numel() == _TOPK and probs.numel() == _TOPK
  return kimi_down_reduce_q4_q8_from_q8_0(kimi_quant_gate_up_q8_0(gate_up), sel, probs, down_w)
