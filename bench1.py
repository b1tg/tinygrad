
from tinygrad import Tensor, Device
from tinygrad.frontend.onnx import onnx_load, OnnxRunner
# from extra.onnx import onnx_load, OnnxRunner
from extra.onnx_old import OnnxRunner as OnnxRunnerOLD
import pathlib
import time
import onnx
from tinygrad.helpers import getenv
N=getenv("N", 10)
import sys
m = "./model.onnx"
if len(sys.argv) == 2:
   m = sys.argv[1]
fm = Tensor(pathlib.Path(m)).to(Device.DEFAULT)
print(f"to device {Device.DEFAULT}")
sts = []
for i in range(N):
  print(f"---- {i} ----")
  st = time.perf_counter()
  x = onnx_load(fm)
  run_onnx = OnnxRunner(x)
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

