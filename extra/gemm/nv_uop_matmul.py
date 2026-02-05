import numpy as np
from tinygrad import Tensor, Device, Context, GlobalCounters, dtypes
from tinygrad.engine.realize import ExecItem, get_runner
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import getenv
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, UOp

N = getenv("N", 4096)
M = getenv("M", N)
K = getenv("K", N)
run_count = getenv("CNT", 5)

WARP_SIZE = 32
WMMA_M, WMMA_N, WMMA_K = 16, 8, 16

BLOCK_M = getenv("NV_UOP_BM", 128)
BLOCK_N = getenv("NV_UOP_BN", 128)
BLOCK_K = getenv("NV_UOP_BK", 16)
PAD_A = getenv("NV_UOP_PAD_A", 0)
PAD_B = getenv("NV_UOP_PAD_B", 0)
SWIZZLE_A = getenv("NV_UOP_SWIZZLE_A", 0)
SWIZZLE_B = getenv("NV_UOP_SWIZZLE_B", 0)
B_COALESCE = getenv("NV_UOP_B_COALESCE", 0)
PIPE = getenv("NV_UOP_PIPE", 0)
UNROLL_K = getenv("NV_UOP_UNROLL_K", 0)
WARP_SWIZZLE = getenv("NV_UOP_WARP_SWIZZLE", 0)
LDMATRIX = getenv("NV_UOP_LDMATRIX", 0)
LDMATRIX_SWIZZLE = getenv("NV_UOP_LDMATRIX_SWIZZLE", 0)
CP_ASYNC = getenv("NV_UOP_CP_ASYNC", 0)
CP_WAIT = getenv("NV_UOP_CP_WAIT", 0)

WARPS_M = getenv("NV_UOP_WM", 2)
WARPS_N = getenv("NV_UOP_WN", 2)
WARPS_PER_BLOCK = WARPS_M * WARPS_N

THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE

WARP_TILE_M = BLOCK_M // WARPS_M
WARP_TILE_N = BLOCK_N // WARPS_N
TILES_M = WARP_TILE_M // WMMA_M
TILES_N = WARP_TILE_N // WMMA_N

assert THREADS_PER_BLOCK <= 1024
assert BLOCK_M % (WARPS_M * WMMA_M) == 0
assert BLOCK_N % (WARPS_N * WMMA_N) == 0
assert BLOCK_K % WMMA_K == 0
if SWIZZLE_A or SWIZZLE_B:
  assert (BLOCK_K & (BLOCK_K - 1)) == 0, "swizzle requires power-of-two BLOCK_K"
if LDMATRIX:
  assert BLOCK_M == 64 and BLOCK_N == 128 and BLOCK_K == 64, "ldmatrix path is tuned for 64x128x64 tiles"
  assert WARPS_M == 2 and WARPS_N == 2, "ldmatrix path assumes 2x2 warps"
  assert SWIZZLE_A == 0 and SWIZZLE_B == 0, "ldmatrix path assumes unswizzled A/B"
  assert B_COALESCE == 0 and PIPE == 0, "ldmatrix path does not support B coalesce or pipelining"
if LDMATRIX_SWIZZLE:
  assert LDMATRIX, "ldmatrix swizzle requires LDMATRIX"
if CP_ASYNC:
  assert LDMATRIX, "cp.async path requires LDMATRIX"

WMMA_ARG = (
  "WMMA_8_16_16_half_float",
  (WMMA_N, WMMA_M, WMMA_K),
  dtypes.half,
  dtypes.float,
  "CUDA",
  WARP_SIZE,
  (((0, 2), (1, 2), (2, 2)), ((3, 2), (4, 2)), ((5, 2), (6, 2))),
  (),
)

def _swizzle_k(row: UOp, col: UOp, swizzle: int) -> UOp:
  if swizzle == 0: return col
  mask = (1 << swizzle) - 1
  return col ^ (((row & mask) << swizzle) & (BLOCK_K - 1))

def _copy_flat(dst: UOp, src: UOp, tid: UOp, rng: int, inner: int = 8, swizzle: int = 0) -> UOp:
  cp_i = UOp.range(dst.size // (THREADS_PER_BLOCK * inner), rng)
  cp_inner = UOp.range(inner, rng + 1, AxisType.UPCAST)
  idx = cp_i * THREADS_PER_BLOCK * inner + tid * inner + cp_inner
  idx_dst = idx
  if swizzle:
    row = idx // BLOCK_K
    col = idx - row * BLOCK_K
    idx_dst = row * BLOCK_K + _swizzle_k(row, col, swizzle)
  return dst[idx_dst].store(src[idx]).end(cp_i, cp_inner)

def _copy_flat_transpose(dst: UOp, src: UOp, tid: UOp, rng: int, inner: int = 8, swizzle: int = 0) -> UOp:
  cp_i = UOp.range(dst.size // (THREADS_PER_BLOCK * inner), rng)
  cp_inner = UOp.range(inner, rng + 1, AxisType.UPCAST)
  idx = cp_i * THREADS_PER_BLOCK * inner + tid * inner + cp_inner
  row = idx // BLOCK_N
  col = idx - row * BLOCK_N
  if swizzle:
    idx_dst = col * BLOCK_K + _swizzle_k(col, row, swizzle)
  else:
    idx_dst = col * BLOCK_K + row
  return dst[idx_dst].store(src[idx]).end(cp_i, cp_inner)

def _copy_ldmatrix_a_swizzle(dst: UOp, src: UOp, tid: UOp, rng: int) -> UOp:
  row_chunk = UOp.range(4, rng)
  inner = UOp.range(8, rng + 1, AxisType.UPCAST)
  tid_div8 = tid >> 3
  tid_mod8 = tid & 7
  row = tid_div8 + row_chunk * 16
  col_src = (tid_mod8 << 3) + inner
  col_dst = (((tid << 3) ^ tid) & UOp.const(dtypes.index, 56)) + inner
  idx = row * BLOCK_K + col_src
  idx_dst = row * BLOCK_K + col_dst
  return dst[idx_dst].store(src[idx]).end(row_chunk, inner)

def _copy_ldmatrix_b_swizzle(dst: UOp, src: UOp, tid: UOp, rng: int) -> UOp:
  row_chunk = UOp.range(8, rng)
  inner = UOp.range(8, rng + 1, AxisType.UPCAST)
  tid_div16 = tid >> 4
  tid_mod16 = tid & 15
  row = tid_div16 + row_chunk * 8
  col_src = (tid_mod16 << 3) + inner
  col_dst = ((tid_div16 << 3) ^ (tid_mod16 << 3)) + inner
  idx = row * BLOCK_N + col_src
  idx_dst = row * BLOCK_N + col_dst
  return dst[idx_dst].store(src[idx]).end(row_chunk, inner)

def _custom_nv_uop_gemm(C: UOp, A: UOp, B: UOp) -> UOp:
  M, K = A.shape
  K2, N = B.shape
  assert K == K2

  gidx0 = UOp.special(M // BLOCK_M, "gidx0")
  gidx1 = UOp.special(N // BLOCK_N, "gidx1")
  lane = UOp.special(WARP_SIZE, "lidx0")
  warp = UOp.special(WARPS_PER_BLOCK, "lidx1")
  tid = lane + warp * WARP_SIZE

  k_tiles = K // BLOCK_K

  if (WARPS_N & (WARPS_N - 1)) == 0:
    warp_n = warp & (WARPS_N - 1)
    warp_m = warp >> int(WARPS_N.bit_length() - 1)
  else:
    warp_n = warp % WARPS_N
    warp_m = warp // WARPS_N
  if WARP_SWIZZLE:
    if (WARPS_M & (WARPS_M - 1)) == 0:
      warp_m = warp & (WARPS_M - 1)
      warp_n = warp >> int(WARPS_M.bit_length() - 1)
    else:
      warp_m = warp % WARPS_M
      warp_n = warp // WARPS_M
  lane_row = lane >> 2
  lane_k2 = lane & 3
  inner = getenv("NV_UOP_INNER", 8)

  def _wmma_compute(As: UOp, Bs: UOp, k_outer_axis: UOp, k_inner_axis: UOp, Ar: UOp, Br: UOp,
                    acc_dep: UOp | None = None, end_k_inner: bool = True) -> UOp:
    a_loads, b_loads = [], []
    if LDMATRIX:
      assert TILES_M == 2 and TILES_N == 8
      lane_mod8 = lane & 7
      lane_div16 = lane >> 4
      if LDMATRIX_SWIZZLE:
        lane_mod16 = lane & 15
        k_phase = k_inner_axis * UOp.const(dtypes.index, 2)
        a_base = (warp_m * UOp.const(dtypes.index, 16) + lane_mod16) * UOp.const(dtypes.index, BLOCK_K)
        a_base = a_base + (((lane_div16 + k_phase) ^ lane_mod8) * UOp.const(dtypes.index, 8))
        b_base = lane_mod16 * UOp.const(dtypes.index, BLOCK_N)
        b_phase = warp_n * UOp.const(dtypes.index, 2) + lane_div16
        b_k_off = k_inner_axis * UOp.const(dtypes.index, 16 * BLOCK_N)
      else:
        lane_div8 = lane >> 3
        lane_div8_mod2 = lane_div8 & 1
        k_off = k_inner_axis * UOp.const(dtypes.index, 16)
        a_base = warp_m * UOp.const(dtypes.index, 16 * BLOCK_K)
        a_base = a_base + lane_mod8 * UOp.const(dtypes.index, BLOCK_K)
        a_base = a_base + lane_div8_mod2 * UOp.const(dtypes.index, BLOCK_K * 8)
        a_base = a_base + lane_div16 * UOp.const(dtypes.index, 8) + k_off
        b_base = warp_n * UOp.const(dtypes.index, 16)
        b_base = b_base + lane_mod8 * UOp.const(dtypes.index, BLOCK_N)
        b_base = b_base + lane_div8_mod2 * UOp.const(dtypes.index, BLOCK_N * 8)
        b_base = b_base + lane_div16 * UOp.const(dtypes.index, 8) + k_off * UOp.const(dtypes.index, BLOCK_N)
      As_flat = As.reshape(BLOCK_M * BLOCK_K)
      Bs_flat = Bs.reshape(BLOCK_K * BLOCK_N)
      for wm in range(TILES_M):
        a_ptr = As_flat.index(a_base + UOp.const(dtypes.index, wm * 32 * BLOCK_K), ptr=True)
        a_vec = UOp(Ops.CUSTOMI, dtypes.half.vec(8), (a_ptr,), arg="__ldmatrix_a({0})")
        a_loads.append(Ar[wm].store(a_vec))
      for pair in range(TILES_N // 2):
        if LDMATRIX_SWIZZLE:
          b_phase_off = b_phase + UOp.const(dtypes.index, pair * 4)
          b_ptr = Bs_flat.index(b_base + (((b_phase_off ^ lane_mod8) * UOp.const(dtypes.index, 8))) + b_k_off, ptr=True)
        else:
          b_ptr = Bs_flat.index(b_base + UOp.const(dtypes.index, pair * 32), ptr=True)
        b_pack = UOp(Ops.CUSTOMI, dtypes.half.vec(8), (b_ptr,), arg="__ldmatrix_b({0})")
        b_lo = UOp.vectorize(b_pack.gep(0), b_pack.gep(1), b_pack.gep(2), b_pack.gep(3))
        b_hi = UOp.vectorize(b_pack.gep(4), b_pack.gep(5), b_pack.gep(6), b_pack.gep(7))
        b_loads.append(Br[pair * 2].store(b_lo))
        b_loads.append(Br[pair * 2 + 1].store(b_hi))
    else:
      for wm in range(TILES_M):
        base_row = warp_m * WARP_TILE_M + wm * WMMA_M
        row_off = ((a_rng & 3) >> 1) * 8
        k_off = (lane_k2 << 1) + (a_rng & 1) + ((a_rng >> 2) << 3)
        a_row = base_row + lane_row + row_off
        a_col = _swizzle_k(a_row, k_inner_axis * WMMA_K + k_off, SWIZZLE_A)
        a_vec = As[a_row, a_col].contract(a_rng)
        a_loads.append(Ar[wm].store(a_vec))
      for wn in range(TILES_N):
        base_col = warp_n * WARP_TILE_N + wn * WMMA_N
        k_off = (lane_k2 << 1) + (b_rng & 1) + ((b_rng >> 1) << 3)
        b_row = base_col + lane_row
        b_col = _swizzle_k(b_row, k_inner_axis * WMMA_K + k_off, SWIZZLE_B)
        b_vec = Bs[b_row, b_col].contract(b_rng)
        b_loads.append(Br[wn].store(b_vec))

    load_group = UOp.group(*a_loads, *b_loads)
    Ar, Br = Ar.after(load_group), Br.after(load_group)

    dep_srcs = [k_outer_axis, load_group]
    if end_k_inner: dep_srcs.insert(1, k_inner_axis)
    if acc_dep is not None: dep_srcs.append(acc_dep)

    acc_ops = []
    for wm in range(TILES_M):
      for wn in range(TILES_N):
        acc_load = UOp.vectorize(*[acc.after(*dep_srcs)[wm, wn, i] for i in range(4)])
        out = UOp(Ops.WMMA, dtypes.float.vec(4), (Ar[wm], Br[wn], acc_load), arg=WMMA_ARG)
        acc_ops.append(UOp.group(*[acc[wm, wn, i].store(out.gep(i)) for i in range(4)]))

    acc_store = UOp.group(*acc_ops)
    return acc_store.end(k_inner_axis) if end_k_inner else acc_store

  Cb = C.reshape(M // BLOCK_M, BLOCK_M, N // BLOCK_N, BLOCK_N)[gidx0, :, gidx1, :]
  a_rng = UOp.range(8, 400, AxisType.UPCAST)
  b_rng = UOp.range(4, 401, AxisType.UPCAST)

  acc = UOp.placeholder((TILES_M, TILES_N, 4), dtypes.float, slot=0, addrspace=AddrSpace.REG)
  init_l = UOp.range(acc.size, 10)
  acc = acc.after(acc.flatten()[init_l].store(UOp.const(dtypes.float, 0.0)).end(init_l))

  if PIPE:
    assert k_tiles % 2 == 0, "NV_UOP_PIPE requires even K tiles"
    k_pair = UOp.range(k_tiles // 2, 0, AxisType.REDUCE)
    k0 = k_pair * 2
    k1 = k0 + 1

    Ab0 = A.reshape(M // BLOCK_M, BLOCK_M, k_tiles, BLOCK_K)[gidx0, :, k0, :]
    Ab1 = A.reshape(M // BLOCK_M, BLOCK_M, k_tiles, BLOCK_K)[gidx0, :, k1, :]
    Bb0 = B.reshape(k_tiles, BLOCK_K, N // BLOCK_N, BLOCK_N)[k0, :, gidx1, :]
    Bb1 = B.reshape(k_tiles, BLOCK_K, N // BLOCK_N, BLOCK_N)[k1, :, gidx1, :]

    As0 = UOp.placeholder((BLOCK_M, BLOCK_K + PAD_A), dtypes.half, slot=0, addrspace=AddrSpace.LOCAL).shrink_to((BLOCK_M, BLOCK_K))
    Bs0 = UOp.placeholder((BLOCK_N, BLOCK_K + PAD_B), dtypes.half, slot=1, addrspace=AddrSpace.LOCAL).shrink_to((BLOCK_N, BLOCK_K))
    As1 = UOp.placeholder((BLOCK_M, BLOCK_K + PAD_A), dtypes.half, slot=2, addrspace=AddrSpace.LOCAL).shrink_to((BLOCK_M, BLOCK_K))
    Bs1 = UOp.placeholder((BLOCK_N, BLOCK_K + PAD_B), dtypes.half, slot=3, addrspace=AddrSpace.LOCAL).shrink_to((BLOCK_N, BLOCK_K))

    As0_flat = As0.reshape(BLOCK_M * BLOCK_K)
    Ab0_flat = Ab0.reshape(BLOCK_M * BLOCK_K)
    Bs0_flat = Bs0.reshape(BLOCK_N * BLOCK_K)
    Bb0_flat = Bb0.reshape(BLOCK_K * BLOCK_N)
    As1_flat = As1.reshape(BLOCK_M * BLOCK_K)
    Ab1_flat = Ab1.reshape(BLOCK_M * BLOCK_K)
    Bs1_flat = Bs1.reshape(BLOCK_N * BLOCK_K)
    Bb1_flat = Bb1.reshape(BLOCK_K * BLOCK_N)

    As0_store = _copy_flat(As0_flat, Ab0_flat, tid, 100, inner=inner, swizzle=SWIZZLE_A)
    As1_store = _copy_flat(As1_flat, Ab1_flat, tid, 110, inner=inner, swizzle=SWIZZLE_A)
    if B_COALESCE:
      Bs0_store = _copy_flat_transpose(Bs0_flat, Bb0_flat, tid, 200, inner=inner, swizzle=SWIZZLE_B)
      Bs1_store = _copy_flat_transpose(Bs1_flat, Bb1_flat, tid, 210, inner=inner, swizzle=SWIZZLE_B)
    else:
      Bb0_flat = Bb0.permute((1, 0)).reshape(BLOCK_N * BLOCK_K)
      Bb1_flat = Bb1.permute((1, 0)).reshape(BLOCK_N * BLOCK_K)
      Bs0_store = _copy_flat(Bs0_flat, Bb0_flat, tid, 200, inner=inner, swizzle=SWIZZLE_B)
      Bs1_store = _copy_flat(Bs1_flat, Bb1_flat, tid, 210, inner=inner, swizzle=SWIZZLE_B)

    barrier0 = UOp.barrier(As0_store, Bs0_store, As1_store, Bs1_store)
    As0, Bs0 = As0.after(barrier0), Bs0.after(barrier0)
    As1, Bs1 = As1.after(barrier0), Bs1.after(barrier0)

    k_inner0 = UOp.range(BLOCK_K // WMMA_K, 300, AxisType.REDUCE)
    k_inner1 = UOp.range(BLOCK_K // WMMA_K, 301, AxisType.REDUCE)
    Ar0 = UOp.placeholder((TILES_M,), dtypes.half.vec(8), slot=2, addrspace=AddrSpace.REG)
    Br0 = UOp.placeholder((TILES_N,), dtypes.half.vec(4), slot=3, addrspace=AddrSpace.REG)
    Ar1 = UOp.placeholder((TILES_M,), dtypes.half.vec(8), slot=4, addrspace=AddrSpace.REG)
    Br1 = UOp.placeholder((TILES_N,), dtypes.half.vec(4), slot=5, addrspace=AddrSpace.REG)

    acc_store0 = _wmma_compute(As0, Bs0, k_pair, k_inner0, Ar0, Br0)
    acc_store1 = _wmma_compute(As1, Bs1, k_pair, k_inner1, Ar1, Br1, acc_dep=acc_store0)
    acc = acc.after(acc_store1.barrier().end(k_pair))
  else:
    k_outer = UOp.range(k_tiles, 0, AxisType.REDUCE)
    Ab = A.reshape(M // BLOCK_M, BLOCK_M, k_tiles, BLOCK_K)[gidx0, :, k_outer, :]
    Bb = B.reshape(k_tiles, BLOCK_K, N // BLOCK_N, BLOCK_N)[k_outer, :, gidx1, :]

    if LDMATRIX and CP_ASYNC:
      assert (k_tiles & 1) == 0, "cp.async path requires even K tiles"

      As0 = UOp.placeholder((BLOCK_M, BLOCK_K + PAD_A), dtypes.half, slot=0, addrspace=AddrSpace.LOCAL).shrink_to((BLOCK_M, BLOCK_K))
      Bs0 = UOp.placeholder((BLOCK_K, BLOCK_N + PAD_B), dtypes.half, slot=1, addrspace=AddrSpace.LOCAL).shrink_to((BLOCK_K, BLOCK_N))
      As1 = UOp.placeholder((BLOCK_M, BLOCK_K + PAD_A), dtypes.half, slot=2, addrspace=AddrSpace.LOCAL).shrink_to((BLOCK_M, BLOCK_K))
      Bs1 = UOp.placeholder((BLOCK_K, BLOCK_N + PAD_B), dtypes.half, slot=3, addrspace=AddrSpace.LOCAL).shrink_to((BLOCK_K, BLOCK_N))

      As0_flat = As0.reshape(BLOCK_M * BLOCK_K)
      Bs0_flat = Bs0.reshape(BLOCK_K * BLOCK_N)
      As1_flat = As1.reshape(BLOCK_M * BLOCK_K)
      Bs1_flat = Bs1.reshape(BLOCK_K * BLOCK_N)

      def _cp_async_tile(As_flat: UOp, Bs_flat: UOp, k_tile: UOp, wait: bool) -> UOp:
        tid_div8 = tid >> 3
        tid_mod8 = tid & 7
        tid_div16 = tid >> 4
        tid_mod16 = tid & 15
        if LDMATRIX_SWIZZLE:
          store_smem_a_off = tid_div8 * UOp.const(dtypes.index, 64)
          store_smem_a_off = store_smem_a_off + (((tid * UOp.const(dtypes.index, 8)) ^ tid) & UOp.const(dtypes.index, 56))
          store_smem_b_off = tid_div16 * UOp.const(dtypes.index, 128)
          store_smem_b_off = store_smem_b_off + ((tid_div16 * UOp.const(dtypes.index, 8)) ^ (tid_mod16 * UOp.const(dtypes.index, 8)))
        else:
          store_smem_a_off = tid_mod8 * UOp.const(dtypes.index, 8) + tid_div8 * UOp.const(dtypes.index, 64)
          store_smem_b_off = tid_mod16 * UOp.const(dtypes.index, 8) + tid_div16 * UOp.const(dtypes.index, 128)
        cp_args, cp_lines = [], []
        for ra in range(4):
          row_a = tid_div8 + UOp.const(dtypes.index, ra * 16)
          dst_a_off = store_smem_a_off + UOp.const(dtypes.index, ra * 16 * BLOCK_K)
          g_row_a = gidx0 * UOp.const(dtypes.index, BLOCK_M) + row_a
          g_col_a = k_tile * UOp.const(dtypes.index, BLOCK_K) + tid_mod8 * UOp.const(dtypes.index, 8)
          cp_lines.append(f"__pipeline_memcpy_async({{{len(cp_args)}}}, {{{len(cp_args) + 1}}}, 16);")
          cp_args += [As_flat.index(dst_a_off, ptr=True), A.index(g_row_a, g_col_a, ptr=True)]
        for rb in range(8):
          row_b = tid_div16 + UOp.const(dtypes.index, rb * 8)
          dst_b_off = store_smem_b_off + UOp.const(dtypes.index, rb * 8 * BLOCK_N)
          g_row_b = k_tile * UOp.const(dtypes.index, BLOCK_K) + row_b
          g_col_b = gidx1 * UOp.const(dtypes.index, BLOCK_N) + tid_mod16 * UOp.const(dtypes.index, 8)
          cp_lines.append(f"__pipeline_memcpy_async({{{len(cp_args)}}}, {{{len(cp_args) + 1}}}, 16);")
          cp_args += [Bs_flat.index(dst_b_off, ptr=True), B.index(g_row_b, g_col_b, ptr=True)]
        cp_lines.append("__pipeline_commit();")
        if wait: cp_lines.append("__pipeline_wait_prior(0);")
        return UOp(Ops.CUSTOM, dtypes.void, tuple(cp_args), arg=" ".join(cp_lines))

      k_pair = UOp.range(k_tiles // 2, 0, AxisType.REDUCE)
      k0 = k_pair * 2
      k1 = k0 + 1

      cp0 = _cp_async_tile(As0_flat, Bs0_flat, k0, wait=True)
      cp1 = _cp_async_tile(As1_flat, Bs1_flat, k1, wait=False)
      barrier0 = UOp.barrier(cp0)
      As0, Bs0 = As0.after(barrier0), Bs0.after(barrier0)

      Ar0 = UOp.placeholder((TILES_M,), dtypes.half.vec(8), slot=2, addrspace=AddrSpace.REG)
      Br0 = UOp.placeholder((TILES_N,), dtypes.half.vec(4), slot=3, addrspace=AddrSpace.REG)
      Ar1 = UOp.placeholder((TILES_M,), dtypes.half.vec(8), slot=4, addrspace=AddrSpace.REG)
      Br1 = UOp.placeholder((TILES_N,), dtypes.half.vec(4), slot=5, addrspace=AddrSpace.REG)
      k_inner0 = UOp.range(BLOCK_K // WMMA_K, 300, AxisType.REDUCE)
      k_inner1 = UOp.range(BLOCK_K // WMMA_K, 301, AxisType.REDUCE)
      acc_store0 = _wmma_compute(As0, Bs0, k_pair, k_inner0, Ar0, Br0, acc_dep=cp1)
      bs1_dep = Bs1.after(cp1, acc_store0)
      wait1_ptr = bs1_dep.index(UOp.const(dtypes.index, 0), UOp.const(dtypes.index, 0), ptr=True)
      wait1 = UOp(Ops.CUSTOM, dtypes.void, (wait1_ptr,), arg=f"__pipeline_wait_prior({CP_WAIT});")
      barrier1 = UOp.barrier(wait1)
      As1, Bs1 = As1.after(barrier1), Bs1.after(barrier1)
      acc_store1 = _wmma_compute(As1, Bs1, k_pair, k_inner1, Ar1, Br1, acc_dep=acc_store0)
      acc = acc.after(acc_store1.barrier().end(k_pair))
    else:
      if LDMATRIX:
        As = UOp.placeholder((BLOCK_M, BLOCK_K + PAD_A), dtypes.half, slot=0, addrspace=AddrSpace.LOCAL).shrink_to((BLOCK_M, BLOCK_K))
        Bs = UOp.placeholder((BLOCK_K, BLOCK_N + PAD_B), dtypes.half, slot=1, addrspace=AddrSpace.LOCAL).shrink_to((BLOCK_K, BLOCK_N))
        As_flat = As.reshape(BLOCK_M * BLOCK_K)
        Bs_flat = Bs.reshape(BLOCK_K * BLOCK_N)
        Ab_flat = Ab.reshape(BLOCK_M * BLOCK_K)
        Bb_flat = Bb.reshape(BLOCK_K * BLOCK_N)
        if LDMATRIX_SWIZZLE:
          As_store = _copy_ldmatrix_a_swizzle(As_flat, Ab_flat, tid, 100)
          Bs_store = _copy_ldmatrix_b_swizzle(Bs_flat, Bb_flat, tid, 200)
        else:
          As_store = _copy_flat(As_flat, Ab_flat, tid, 100, inner=inner, swizzle=SWIZZLE_A)
          Bs_store = _copy_flat(Bs_flat, Bb_flat, tid, 200, inner=inner, swizzle=SWIZZLE_B)
        barrier0 = UOp.barrier(As_store, Bs_store)
      else:
        As = UOp.placeholder((BLOCK_M, BLOCK_K + PAD_A), dtypes.half, slot=0, addrspace=AddrSpace.LOCAL).shrink_to((BLOCK_M, BLOCK_K))
        Bs = UOp.placeholder((BLOCK_N, BLOCK_K + PAD_B), dtypes.half, slot=1, addrspace=AddrSpace.LOCAL).shrink_to((BLOCK_N, BLOCK_K))
        As_flat = As.reshape(BLOCK_M * BLOCK_K)
        Ab_flat = Ab.reshape(BLOCK_M * BLOCK_K)
        Bs_flat = Bs.reshape(BLOCK_N * BLOCK_K)
        Bb_flat = Bb.reshape(BLOCK_K * BLOCK_N)
        As_store = _copy_flat(As_flat, Ab_flat, tid, 100, inner=inner, swizzle=SWIZZLE_A)
        if B_COALESCE:
          Bs_store = _copy_flat_transpose(Bs_flat, Bb_flat, tid, 200, inner=inner, swizzle=SWIZZLE_B)
        else:
          Bb_flat = Bb.permute((1, 0)).reshape(BLOCK_N * BLOCK_K)
          Bs_store = _copy_flat(Bs_flat, Bb_flat, tid, 200, inner=inner, swizzle=SWIZZLE_B)
        barrier0 = UOp.barrier(As_store, Bs_store)
      As, Bs = As.after(barrier0), Bs.after(barrier0)

      Ar = UOp.placeholder((TILES_M,), dtypes.half.vec(8), slot=2, addrspace=AddrSpace.REG)
      Br = UOp.placeholder((TILES_N,), dtypes.half.vec(4), slot=3, addrspace=AddrSpace.REG)
      if UNROLL_K and not LDMATRIX:
        acc_dep = None
        for step in range(BLOCK_K // WMMA_K):
          k_inner_val = UOp.const(dtypes.index, step)
          acc_dep = _wmma_compute(As, Bs, k_outer, k_inner_val, Ar, Br, acc_dep=acc_dep, end_k_inner=False)
        acc = acc.after(acc_dep.barrier().end(k_outer))
      else:
        k_inner = UOp.range(BLOCK_K // WMMA_K, 300, AxisType.REDUCE)
        acc_store = _wmma_compute(As, Bs, k_outer, k_inner, Ar, Br)
        acc = acc.after(acc_store.barrier().end(k_outer))

  store_ops = []
  use_half2_store = C.dtype.base == dtypes.half
  if LDMATRIX:
    n_const = UOp.const(dtypes.index, N)
    wg_c_off = gidx0 * UOp.const(dtypes.index, BLOCK_M) * n_const
    wg_c_off = wg_c_off + gidx1 * UOp.const(dtypes.index, BLOCK_N)
    wg_c_off = wg_c_off + warp_m * UOp.const(dtypes.index, 16) * n_const + warp_n * UOp.const(dtypes.index, 16)
    thread_c_off = (lane & 3) * UOp.const(dtypes.index, 2)
    thread_c_off = thread_c_off + ((lane >> 2) & 7) * n_const
    C_flat = C.reshape(M * N)
    for wm in range(TILES_M):
      wm_off = wg_c_off + UOp.const(dtypes.index, wm * 32) * n_const
      for wn in range(TILES_N):
        col_group = ((wn // 2) * 4 + (wn & 1)) * 8
        base = wm_off + thread_c_off + UOp.const(dtypes.index, col_group)
        val0 = acc[wm, wn, 0]
        val1 = acc[wm, wn, 1]
        val2 = acc[wm, wn, 2]
        val3 = acc[wm, wn, 3]
        if use_half2_store:
          ptr0 = C_flat.index(base, ptr=True)
          ptr1 = C_flat.index(base + UOp.const(dtypes.index, 8) * n_const, ptr=True)
          store_ops.append(ptr0.store(UOp.vectorize(val0.cast(dtypes.half), val1.cast(dtypes.half))))
          store_ops.append(ptr1.store(UOp.vectorize(val2.cast(dtypes.half), val3.cast(dtypes.half))))
        else:
          store_ops.append(C_flat.index(base, ptr=True).store(val0))
          store_ops.append(C_flat.index(base + UOp.const(dtypes.index, 1), ptr=True).store(val1))
          store_ops.append(C_flat.index(base + UOp.const(dtypes.index, 8) * n_const, ptr=True).store(val2))
          store_ops.append(C_flat.index(base + UOp.const(dtypes.index, 8) * n_const + UOp.const(dtypes.index, 1), ptr=True).store(val3))
  else:
    for wm in range(TILES_M):
      base_row = warp_m * WARP_TILE_M + wm * WMMA_M
      for wn in range(TILES_N):
        base_col = warp_n * WARP_TILE_N + wn * WMMA_N
        if use_half2_store:
          for elem_base in range(0, 4, 2):
            row = lane_row + ((elem_base >> 1) << 3)
            col = (lane_k2 << 1)
            val0 = acc[wm, wn, elem_base].cast(dtypes.half)
            val1 = acc[wm, wn, elem_base + 1].cast(dtypes.half)
            global_row = gidx0 * BLOCK_M + base_row + row
            global_col = gidx1 * BLOCK_N + base_col + col
            ptr = C.index(global_row, global_col, ptr=True)
            store_ops.append(ptr.store(UOp.vectorize(val0, val1)))
        else:
          for elem in range(4):
            row = lane_row + ((elem >> 1) << 3)
            col = (lane_k2 << 1) + (elem & 1)
            val = acc[wm, wn, elem]
            store_ops.append(Cb[base_row + row, base_col + col].store(val))

  sink = UOp.group(*store_ops).sink(arg=KernelInfo(name="nv_uop_gemm", opts_to_apply=()))
  return sink if LDMATRIX else sink.simplify()

def custom_gemm_bw(gradient: UOp, kernel: UOp):
  out, a, b = kernel.src
  a_t, b_t, g_t = Tensor(a, device=a.device), Tensor(b, device=a.device), Tensor(gradient, device=a.device)
  grad_a = (g_t @ b_t.T).uop
  a_t = a_t.transpose(-2, -1).reshape(*a_t.shape[:-1], 1, a_t.shape[-1])
  g_t = g_t.reshape(*g_t.shape[:-2], 1, *g_t.shape[-2:]).transpose(-1, -2)
  grad_b = (a_t * g_t).sum((-1, 0)).uop
  return (None, grad_a, grad_b)

def can_use_nv_uop_gemm(a: Tensor, b: Tensor, dtype) -> bool:
  if a.dtype != dtypes.half or b.dtype != dtypes.half: return False
  if a.ndim != 2 or b.ndim != 2: return False
  if a.shape[1] != b.shape[0]: return False
  if dtype is not None and dtype != dtypes.float: return False
  if not a.uop.is_contiguous() or not b.uop.is_contiguous(): return False
  M, K = a.shape
  N = b.shape[1]
  return (M % BLOCK_M) == 0 and (N % BLOCK_N) == 0 and (K % BLOCK_K) == 0

def nv_uop_gemm(a: Tensor, b: Tensor, dtype) -> Tensor:
  assert can_use_nv_uop_gemm(a, b, dtype)
  out_dtype = dtypes.float if dtype is not None else dtypes.half
  out = Tensor.empty(a.shape[0], b.shape[1], dtype=out_dtype, device=a.device)
  return Tensor.custom_kernel(out, a, b, fxn=_custom_nv_uop_gemm, grad_fxn=custom_gemm_bw)[0]

def hand_spec_kernel(dtype_out=dtypes.half) -> UOp:
  c = UOp.placeholder((M, N), dtype_out, slot=0)
  a = UOp.placeholder((M, K), dtypes.half, slot=1)
  b = UOp.placeholder((K, N), dtypes.half, slot=2)
  return _custom_nv_uop_gemm(c, a, b)

def test_matmul(sink: UOp, dtype_in=dtypes.half, dtype_out=dtypes.half, M=M, N=N, K=K):
  rng = np.random.default_rng()
  a = Tensor(rng.random((M, K), dtype=np.float32)-0.5, dtype=dtype_in)
  b = Tensor(rng.random((K, N), dtype=np.float32)-0.5, dtype=dtype_in)
  hc = Tensor.empty(M, N, dtype=dtype_out)
  Tensor.realize(a, b, hc)

  ei = ExecItem(sink, [t.uop.buffer for t in [hc, a, b]], prg=get_runner(Device.DEFAULT, sink))

  ets = []
  with Context(DEBUG=2):
    for _ in range(run_count):
      ets.append(ei.run(wait=True))
  print(f"REAL TFLOPS {M * N * K * 2 / min(ets) * 1e-12:.2f}")

  if getenv("VERIFY", 1):
    GlobalCounters.reset()
    with Context(DEBUG=2):
      tc = (a @ b).realize()
    with Context(DEBUG=0):
      err = (hc - tc).square().mean().item()
    print(f"mean squared error {err}")
    if err > 1e-04:
      raise RuntimeError("matmul is wrong!")

if __name__ == "__main__":
  out_dtype = dtypes.float if getenv("OUT_FLOAT", 0) else dtypes.half
  test_matmul(hand_spec_kernel(out_dtype), dtype_in=dtypes.half, dtype_out=out_dtype, N=N)
