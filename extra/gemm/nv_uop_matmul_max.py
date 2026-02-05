"""High-performance WMMA GEMM kernel using UOps targeting 165+ TFLOPS on RTX 4090.

Based on the reference kernel in extra/gemm/max_kernels/nv.fp16_fp32_fp32.max.cu
"""
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

# Tile sizes matching the max kernel
BLOCK_M = 64
BLOCK_N = 128
BLOCK_K = 64
MMA_K = 16  # K elements per WMMA operation

THREADS_PER_BLOCK = 128
WARP_SIZE = 32
WARPS_M = 2
WARPS_N = 2

# WMMA: m16n8k16 produces float4 output
# M_FRAGS = 2 (rows per warp: 2 * 16 = 32, total 64 with 2 warps)
# N_FRAGS = 8 (cols per warp: 8 * 8 = 64, total 128 with 2 warps)
M_FRAGS = 2
N_FRAGS = 8

# Shared memory layout: double buffered
# Each stage: A = 64*64 = 4096 half (8KB), B = 64*128 = 8192 half (16KB)
# Total: 2 * (8KB + 16KB) = 48KB
SMEM_A_STAGE = BLOCK_M * BLOCK_K * 2  # bytes
SMEM_B_STAGE = BLOCK_K * BLOCK_N * 2  # bytes
SMEM_TOTAL = 2 * (SMEM_A_STAGE + SMEM_B_STAGE)

WMMA_ARG = ("WMMA_8_16_16_half_float", (8, 16, 16), dtypes.half, dtypes.float, "CUDA", 32, (((0, 8),), ((0, 4),), ((0, 4),)), ())

def ci(v: int) -> UOp:
  """Create index constant."""
  return UOp.const(dtypes.index, v)

def _custom_nv_uop_gemm_max(C: UOp, A: UOp, B: UOp) -> UOp:
  """Build a high-performance WMMA GEMM kernel."""
  M_dim, K_dim = A.shape
  _, N_dim = B.shape
  assert M_dim % BLOCK_M == 0 and N_dim % BLOCK_N == 0 and K_dim % BLOCK_K == 0

  # Grid and thread indices
  gidx0 = UOp.special(M_dim // BLOCK_M, "gidx0")  # M tiles
  gidx1 = UOp.special(N_dim // BLOCK_N, "gidx1")  # N tiles
  threads = UOp.special(THREADS_PER_BLOCK, "lidx0")

  # Warp decomposition
  wg_m = threads // ci(64)  # 0 or 1
  wg_n = (threads // ci(32)) % ci(2)  # 0 or 1
  lane = threads % ci(32)

  # Index helpers
  tid_div8 = threads // ci(8)
  tid_mod8 = threads % ci(8)
  tid_div16 = threads // ci(16)
  tid_mod16 = threads % ci(16)

  # Shared memory with double buffering
  As_0 = UOp.placeholder((BLOCK_M, BLOCK_K), dtypes.half, slot=0, addrspace=AddrSpace.LOCAL)
  As_1 = UOp.placeholder((BLOCK_M, BLOCK_K), dtypes.half, slot=1, addrspace=AddrSpace.LOCAL)
  Bs_0 = UOp.placeholder((BLOCK_K, BLOCK_N), dtypes.half, slot=2, addrspace=AddrSpace.LOCAL)
  Bs_1 = UOp.placeholder((BLOCK_K, BLOCK_N), dtypes.half, slot=3, addrspace=AddrSpace.LOCAL)

  # Accumulator and fragment registers
  acc = UOp.placeholder((M_FRAGS, N_FRAGS, 4), dtypes.float, slot=4, addrspace=AddrSpace.REG)
  init_l = UOp.range(acc.size, 10)
  acc = acc.after(acc.flatten()[init_l].store(UOp.const(dtypes.float, 0.0)).end(init_l))

  Ar = UOp.placeholder((M_FRAGS,), dtypes.half.vec(8), slot=5, addrspace=AddrSpace.REG)
  Br = UOp.placeholder((N_FRAGS,), dtypes.half.vec(4), slot=6, addrspace=AddrSpace.REG)

  # Swizzled SMEM store offsets
  store_a_off = tid_div8 * ci(64) + (((threads * ci(8)) ^ threads) & ci(56))
  store_b_off = tid_div16 * ci(128) + (((tid_div16 % ci(8)) * ci(8)) ^ (tid_mod16 * ci(8)))

  # Global offsets
  g_a_base = gidx0 * ci(BLOCK_M * K_dim) + tid_mod8 * ci(8) + tid_div8 * ci(K_dim)
  g_b_base = gidx1 * ci(BLOCK_N) + tid_mod16 * ci(8) + tid_div16 * ci(N_dim)

  # Swizzled ldmatrix offsets
  ld_a_row = (wg_m * ci(16) + tid_mod16) * ci(64)
  ld_a_phase = tid_div16 % ci(2)
  ld_b_row = tid_mod16 * ci(128)
  ld_b_phase = wg_n * ci(2) + (tid_div16 % ci(2))

  K_TILES = K_dim // BLOCK_K

  def _cp_async_tile(As_flat: UOp, Bs_flat: UOp, k_tile: UOp) -> list[UOp]:
    """Emit cp.async prefetch for one K tile."""
    ops = []
    g_a_off = g_a_base + k_tile * ci(BLOCK_K)
    g_b_off = g_b_base + k_tile * ci(BLOCK_K * N_dim)

    for r in range(4):  # A: 4 rows per thread
      s_off = store_a_off + ci(r * 16 * BLOCK_K)
      g_off = g_a_off + ci(r * 16 * K_dim)
      ops.append(UOp(Ops.CUSTOM, dtypes.void,
                     (As_flat.index(s_off, ptr=True), A.index(g_off, ptr=True)),
                     arg="__pipeline_memcpy_async({0}, {1}, 16);"))

    for r in range(8):  # B: 8 rows per thread
      s_off = store_b_off + ci(r * 8 * BLOCK_N)
      g_off = g_b_off + ci(r * 8 * N_dim)
      ops.append(UOp(Ops.CUSTOM, dtypes.void,
                     (Bs_flat.index(s_off, ptr=True), B.index(g_off, ptr=True)),
                     arg="__pipeline_memcpy_async({0}, {1}, 16);"))

    return ops

  def _commit() -> UOp:
    return UOp(Ops.CUSTOM, dtypes.void, (), arg="__pipeline_commit();")

  def _wait(n: int) -> UOp:
    return UOp(Ops.CUSTOM, dtypes.void, (), arg=f"__pipeline_wait_prior({n});")

  def _wmma_block(As: UOp, Bs: UOp, k_idx: int, acc_in: UOp) -> list[UOp]:
    """Emit ldmatrix + WMMA for one 16-element K block."""
    As_flat = As.reshape(BLOCK_M * BLOCK_K)
    Bs_flat = Bs.reshape(BLOCK_K * BLOCK_N)
    load_ops, wmma_ops = [], []

    # Load A fragments
    k_phase = ci(k_idx * 2)
    a_off_0 = ld_a_row + ((ld_a_phase + k_phase) ^ (threads % ci(8))) * ci(8)
    a_off_1 = a_off_0 + ci(32 * 64)

    a_ld_0 = UOp(Ops.CUSTOMI, dtypes.half.vec(8), (As_flat.index(a_off_0, ptr=True),), arg="__ldmatrix_a({0})")
    a_ld_1 = UOp(Ops.CUSTOMI, dtypes.half.vec(8), (As_flat.index(a_off_1, ptr=True),), arg="__ldmatrix_a({0})")
    load_ops.append(Ar[0].store(a_ld_0))
    load_ops.append(Ar[1].store(a_ld_1))

    # Load B fragments (4 pairs -> 8 half4 vectors)
    for pair in range(4):
      b_phase_off = ld_b_phase + ci(pair * 4)
      b_off = ld_b_row + ci(k_idx * 16 * BLOCK_N) + ((b_phase_off ^ (threads % ci(8))) * ci(8))
      b_pack = UOp(Ops.CUSTOMI, dtypes.half.vec(8), (Bs_flat.index(b_off, ptr=True),), arg="__ldmatrix_b({0})")
      b_lo = UOp.vectorize(b_pack.gep(0), b_pack.gep(1), b_pack.gep(2), b_pack.gep(3))
      b_hi = UOp.vectorize(b_pack.gep(4), b_pack.gep(5), b_pack.gep(6), b_pack.gep(7))
      load_ops.append(Br[pair * 2].store(b_lo))
      load_ops.append(Br[pair * 2 + 1].store(b_hi))

    load_group = UOp.group(*load_ops)
    Ar_after = Ar.after(load_group)
    Br_after = Br.after(load_group)

    # WMMA ops: M_FRAGS x N_FRAGS
    for am in range(M_FRAGS):
      a_val = Ar_after[am]
      for bn in range(N_FRAGS):
        b_val = Br_after[bn]
        acc_vec = UOp.vectorize(*[acc_in[am, bn, i] for i in range(4)])
        out = UOp(Ops.WMMA, dtypes.float.vec(4), (a_val, b_val, acc_vec), arg=WMMA_ARG)
        for i in range(4):
          wmma_ops.append(acc_in[am, bn, i].store(out.gep(i)))

    return load_ops + wmma_ops

  # Main pipeline
  As_0_flat = As_0.reshape(BLOCK_M * BLOCK_K)
  As_1_flat = As_1.reshape(BLOCK_M * BLOCK_K)
  Bs_0_flat = Bs_0.reshape(BLOCK_K * BLOCK_N)
  Bs_1_flat = Bs_1.reshape(BLOCK_K * BLOCK_N)

  # Prefetch first tile
  cp_ops_0 = _cp_async_tile(As_0_flat, Bs_0_flat, ci(0))
  commit_0 = _commit()
  barrier_0 = UOp.barrier(*cp_ops_0, commit_0)

  # Prefetch second tile
  cp_ops_1 = _cp_async_tile(As_1_flat, Bs_1_flat, ci(1))
  commit_1 = _commit()
  wait_0 = _wait(0)
  barrier_1 = UOp.barrier(*cp_ops_1, commit_1, wait_0, barrier_0)

  As_0 = As_0.after(barrier_1)
  As_1 = As_1.after(barrier_1)
  Bs_0 = Bs_0.after(barrier_1)
  Bs_1 = Bs_1.after(barrier_1)

  # K loop
  k_outer = UOp.range(K_TILES, 0, AxisType.REDUCE)

  # Select buffers based on parity
  is_odd = (k_outer % ci(2)).ne(ci(0))
  As_cur = UOp(Ops.WHERE, As_0.dtype, (is_odd, As_0, As_1))
  Bs_cur = UOp(Ops.WHERE, Bs_0.dtype, (is_odd, Bs_0, Bs_1))

  # 4 K inner blocks per outer iteration
  all_ops = []
  acc_cur = acc
  for k_idx in range(4):
    ops = _wmma_block(As_cur, Bs_cur, k_idx, acc_cur)
    all_ops.extend(ops)
    acc_cur = acc_cur.after(*ops)

  sync = UOp.barrier(*all_ops)

  # Conditional prefetch
  need_pf = k_outer < ci(K_TILES - 2)
  next_k = k_outer + ci(2)
  cp_ops_next = _cp_async_tile(
    UOp(Ops.WHERE, As_0_flat.dtype, (is_odd, As_0_flat, As_1_flat)),
    UOp(Ops.WHERE, Bs_0_flat.dtype, (is_odd, Bs_0_flat, Bs_1_flat)),
    next_k
  )
  if_pf = UOp(Ops.IF, dtypes.void, (need_pf,))
  end_pf = UOp(Ops.END, dtypes.void, (if_pf, *cp_ops_next))
  commit_next = _commit()

  # Wait for next tile
  need_wait = k_outer < ci(K_TILES - 1)
  wait_1 = _wait(1)
  wait_barrier = UOp.barrier(wait_1)
  if_wait = UOp(Ops.IF, dtypes.void, (need_wait,))
  end_wait = UOp(Ops.END, dtypes.void, (if_wait, wait_barrier))

  k_loop = UOp.group(sync, end_pf, commit_next, end_wait, *all_ops).end(k_outer)
  acc = acc.after(k_loop)

  # Epilogue - wait and write
  wait_final = _wait(0)
  final_barrier = UOp.barrier(wait_final)
  acc = acc.after(final_barrier)

  # Store results to global memory
  store_ops = []
  lane_row = lane // ci(4)
  lane_col = (lane % ci(4)) * ci(2)

  c_base = gidx0 * ci(BLOCK_M * N_dim) + gidx1 * ci(BLOCK_N)
  c_base = c_base + wg_m * ci(16 * N_dim) + wg_n * ci(16)
  c_base = c_base + lane_row * ci(N_dim) + lane_col

  C_flat = C.reshape(M_dim * N_dim)
  for am in range(M_FRAGS):
    am_off = ci(am * 32 * N_dim)
    for bn in range(N_FRAGS):
      # bn layout: 0,1,4,5,8,9,12,13
      bn_col = ((bn // 2) * 4 + (bn % 2)) * 8
      bn_off = ci(bn_col)
      for elem in range(4):
        elem_row = (elem // 2) * 8
        elem_col = elem % 2
        elem_off = ci(elem_row * N_dim + elem_col)
        val = acc[am, bn, elem]
        c_idx = c_base + am_off + bn_off + elem_off
        store_ops.append(C_flat.index(c_idx, ptr=True).store(val))

  store_group = UOp.group(*store_ops)
  return store_group.sink(arg=KernelInfo(name="nv_uop_gemm_max", opts_to_apply=()))


def hand_spec_kernel(dtype_out=dtypes.float) -> UOp:
  c = UOp.placeholder((M, N), dtype_out, slot=0)
  a = UOp.placeholder((M, K), dtypes.half, slot=1)
  b = UOp.placeholder((K, N), dtypes.half, slot=2)
  return _custom_nv_uop_gemm_max(c, a, b)


def test_matmul(sink: UOp, dtype_in=dtypes.half, dtype_out=dtypes.float, M=M, N=N, K=K):
  rng = np.random.default_rng()
  a = Tensor(rng.random((M, K), dtype=np.float32) - 0.5, dtype=dtype_in)
  b = Tensor(rng.random((K, N), dtype=np.float32) - 0.5, dtype=dtype_in)
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
  test_matmul(hand_spec_kernel(), dtype_in=dtypes.half, dtype_out=dtypes.float)
