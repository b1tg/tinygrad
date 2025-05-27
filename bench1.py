
from tinygrad import Tensor, Device
from tinygrad.frontend.onnx import onnx_load, OnnxRunner
# from extra.onnx import onnx_load, OnnxRunner
from extra.onnx_old import OnnxRunner as OnnxRunnerOLD
import pathlib
import time
import onnx
N=10
import sys
m = "./model.onnx"
if len(sys.argv) == 2:
   m = sys.argv[1]
fm = Tensor(pathlib.Path(m)).to(Device.DEFAULT)
print(f"to device ok {Device.DEFAULT}")
sts = []
for i in range(N):
  print(f"---- {i} ----")
  st = time.perf_counter()
  x = onnx_load(fm)
  print(f"\t{i}: onnx_load use {(time.perf_counter()-st)*1000:6.2f} ms")
  print("--- run")
  run_onnx = OnnxRunner(x)
  print(f"\t{i}: run use {(time.perf_counter()-st)*1000:6.2f} ms")
  sts.append(time.perf_counter()-st)
print(f"onnx_load use {(sum(sts)/N)*1000:6.2f} ms")


sts = []
for i in range(N):
  print(f"---- {i} ----")
  st = time.perf_counter()
  x = onnx.load(m)
  run_onnx = OnnxRunnerOLD(x)
  sts.append(time.perf_counter()-st)
print(f"onnx.load use {(sum(sts)/N)*1000:6.2f} ms")

