"""
Side-by-side MUSA perf comparison: torch_musa (vendor muDNN/muBLAS) vs tinygrad-MUSA.
Focus on single-user LLM decode shapes:
  * GEMV  : (d,d) @ (d,1)  -- the decode hot path
  * GEMM  : (d,d) @ (d,d)  -- prefill / TC-eligible
  * Stream: raw copy / bandwidth ceiling

Run: python extra/musa/bench_vs_torch.py
Env : DEV=MUSA assumed for tinygrad path. Set TINY=0 to skip, TORCH=0 to skip.
"""
import os, time, math
os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")

SIZES = [1024, 2048, 4096, 8192]
DTYPES = ["float16", "bfloat16"]
N_WARM = 3
N_ITER = 20

def hw_peak_gbs():
  # MTT S4000 spec: 768 GB/s (48 GB HBM2-like), 128 TC
  return 768.0

def time_min(fn, warm=N_WARM, iters=N_ITER):
  for _ in range(warm): fn()
  tms = []
  for _ in range(iters):
    st = time.perf_counter()
    fn()
    tms.append(time.perf_counter() - st)
  return min(tms)

def bench_torch():
  try:
    import torch_musa  # noqa: F401
    import torch
    dev = "musa"
  except ImportError:
    print("torch_musa not installed, skipping"); return
  print(f"\n=== torch_musa {torch.__version__} on {dev} ===")
  for dt_name in DTYPES:
    dt = getattr(torch, dt_name)
    print(f"\n-- dtype={dt_name} --")
    print(f"{'op':8s} {'shape':22s} {'time_us':>10s} {'GFLOPS':>10s} {'GB/s':>10s} {'util':>6s}")
    for d in SIZES:
      # GEMV
      A = torch.randn(d, d, dtype=dt, device=dev)
      x = torch.randn(d, 1, dtype=dt, device=dev)
      def _gemv():
        y = A @ x
        torch.musa.synchronize()
      tm = time_min(_gemv)
      bytes_moved = d*d*A.element_size() + d*x.element_size()
      gbs = bytes_moved/tm/1e9
      flops = 2*d*d
      print(f"{'gemv':8s} {f'({d},{d})@({d},1)':22s} {tm*1e6:>10.1f} {flops/tm/1e9:>10.1f} {gbs:>10.1f} {gbs/hw_peak_gbs()*100:>5.1f}%")
      # GEMM
      B = torch.randn(d, d, dtype=dt, device=dev)
      def _gemm():
        C = A @ B
        torch.musa.synchronize()
      tm = time_min(_gemm, iters=10)
      flops = 2*d*d*d
      print(f"{'gemm':8s} {f'({d},{d})@({d},{d})':22s} {tm*1e6:>10.1f} {flops/tm/1e9:>10.1f} {'-':>10s} {'-':>6s}")

def bench_tinygrad():
  if os.environ.get("DEV") != "MUSA":
    print("set DEV=MUSA for tinygrad path"); return
  try:
    from tinygrad import Tensor, dtypes, Device, TinyJit
    from tinygrad.helpers import Context, getenv
  except ImportError:
    print("tinygrad not importable"); return
  print(f"\n=== tinygrad on {Device.DEFAULT} (BEAM={getenv('BEAM',0)}) ===")
  for dt_name in DTYPES:
    dt = getattr(dtypes, dt_name)
    print(f"\n-- dtype={dt_name} --")
    print(f"{'op':8s} {'shape':22s} {'time_us':>10s} {'GFLOPS':>10s} {'GB/s':>10s} {'util':>6s}")
    for d in SIZES:
      A = Tensor.randn(d, d, dtype=dt).realize()
      x = Tensor.randn(d, 1, dtype=dt).realize()
      @TinyJit
      def _gemv(A_, x_): return (A_ @ x_).realize()
      # warm (2x needed for JIT capture)
      for _ in range(3): _gemv(A, x).numpy()
      tms = []
      for _ in range(N_ITER):
        st = time.perf_counter()
        r = _gemv(A, x)
        r.numpy()  # force D2H sync
        tms.append(time.perf_counter()-st)
      tm = min(tms)
      elem_size = 2 if dt_name in ("float16","bfloat16") else 4
      bytes_moved = d*d*elem_size + d*elem_size
      gbs = bytes_moved/tm/1e9
      flops = 2*d*d
      print(f"{'gemv':8s} {f'({d},{d})@({d},1)':22s} {tm*1e6:>10.1f} {flops/tm/1e9:>10.1f} {gbs:>10.1f} {gbs/hw_peak_gbs()*100:>5.1f}%")

      B = Tensor.randn(d, d, dtype=dt).realize()
      @TinyJit
      def _gemm(A_, B_): return (A_ @ B_).realize()
      for _ in range(3): _gemm(A, B).numpy()
      tms = []
      for _ in range(10):
        st = time.perf_counter()
        r = _gemm(A, B)
        r.numpy()
        tms.append(time.perf_counter()-st)
      tm = min(tms)
      flops = 2*d*d*d
      print(f"{'gemm':8s} {f'({d},{d})@({d},{d})':22s} {tm*1e6:>10.1f} {flops/tm/1e9:>10.1f} {'-':>10s} {'-':>6s}")

if __name__ == "__main__":
  print(f"MTT S4000 spec: peak HBM BW={hw_peak_gbs()} GB/s, 128 TCs, 48 GB mem\n")
  print("Decode-shape GEMV bytes_moved ≈ weight size → GB/s is the key metric.\n")
  if os.environ.get("TORCH","1") == "1": bench_torch()
  if os.environ.get("TINY","1") == "1": bench_tinygrad()
