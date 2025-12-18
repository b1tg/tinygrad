# benchmark_linear.py
# Scientific & reproducible Linear benchmark for tinygrad
# Compare FP16 Linear vs FP8/custom Linear

import time
import numpy as np
from tinygrad import Tensor, dtypes, nn
from tinygrad.helpers import getenv

Tensor.manual_seed(42)

# ----------------------------
# Config
# ----------------------------
IN  = getenv("IN", 1024)
OUT = getenv("OUT", 1024)
BS  = getenv("BS", 1024)
SEQ = getenv("SEQ", 512)

ITERS  = getenv("ITERS", 5)
WARMUP = getenv("WARMUP", 2)

FP8    = getenv("FP8", 0)

print("====== BENCH CONFIG ======")
print(f"IN={IN}, OUT={OUT}, BS={BS}, SEQ={SEQ}")
print(f"FP8={FP8}, ITERS={ITERS}, WARMUP={WARMUP}")
print("==========================")

from bk import FP8LinearBertBasic as LinearFP8 
from bk import LinearBert as LinearFP16
# ----------------------------
# Models
# ----------------------------
# ----------------------------
# Benchmark utils
# ----------------------------
def benchmark(fn, iters=50, warmup=10, name=""):
  # warmup
  for _ in range(warmup):
    fn()

  fn()  # sync

  t0 = time.perf_counter()
  for _ in range(iters):
    fn()
  t1 = time.perf_counter()

  ms = (t1 - t0) * 1000 / iters
  print(f"{name:<30}: {ms:7.3f} ms")
  return ms

# ----------------------------
# Prepare inputs
# ----------------------------
x = Tensor.rand((BS, SEQ, IN), dtype=dtypes.float16)
x.requires_grad = True
grad = Tensor.ones((BS, SEQ, OUT), dtype=dtypes.float16)

m_fp16 = LinearFP16(IN, OUT)
m_fp8  = LinearFP8(IN, OUT)

# copy weights to ensure fairness
m_fp8.weight.assign(m_fp16.weight)
m_fp8.bias.assign(m_fp16.bias)


m_fp16.weight.requires_grad=True
m_fp8.weight.requires_grad=True

# ----------------------------
# Forward benchmark
# ----------------------------
# print("\n--- FORWARD ---")
if 0:
  benchmark(lambda: m_fp16(x).contiguous().contiguous_backward().realize(),
            iters=ITERS, warmup=WARMUP,
            name="FP16 Linear forward")

  benchmark(lambda: m_fp8(x).contiguous().contiguous_backward().realize(),
            iters=ITERS, warmup=WARMUP,
            name="FP8  Linear forward")

# ----------------------------
# Backward benchmark
# ----------------------------
# print("\n--- BACKWARD (dW) ---")

def fp16_backward():
  y = m_fp16(x).contiguous().contiguous_backward()
  y.backward(grad).realize()
  return m_fp16.weight.grad.realize()

def fp8_backward():
  y = m_fp8(x).contiguous().contiguous_backward()
  y.backward(grad).realize()
  return m_fp8.weight.grad.realize()
if 0:
  benchmark(fp16_backward,
            iters=ITERS, warmup=WARMUP,
            name="FP16 Linear backward")

  benchmark(fp8_backward,
            iters=ITERS, warmup=WARMUP,
            name="FP8  Linear backward")

# ----------------------------
# Train step benchmark
# ----------------------------
print("\n--- TRAIN STEP ---")

def fp16_train():
  y = m_fp16(x).contiguous().contiguous_backward()
  loss = y.square().mean()
  loss.backward()
  m_fp16.weight.grad.realize()
  x.grad.realize()

def fp8_train():
  y = m_fp8(x).contiguous().contiguous_backward()
  loss = y.square().mean()
  loss.backward()
  m_fp8.weight.grad.realize()
  x.grad.realize()

# for i in range(2):
#   fp16_train()
# print("----")
# for i in range(2):
#   fp8_train()
benchmark(fp16_train,
          iters=ITERS, warmup=WARMUP,
          name="FP16 Linear train")

benchmark(fp8_train,
          iters=ITERS, warmup=WARMUP,
          name="FP8  Linear train")

# ----------------------------
# FLOPs report
# ----------------------------
# flops = BS * SEQ * IN * OUT * 2
# print("\n--- THEORETICAL FLOPs ---")
# print(f"{flops/1e12:.3f} TFLOPs per forward")
