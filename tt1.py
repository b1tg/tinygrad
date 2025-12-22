from tinygrad import Tensor, Device
from tinygrad.helpers import getenv
import fcntl, os
GPUS = tuple(f'{Device.DEFAULT}:{i}' for i in range(getenv("GPUS", 2)))
N = getenv("N", 4096)
BS = getenv("BS", 512)
class TinyModel:
  def __init__(self):
    self.w1 = Tensor.empty(N, N)
  def __call__(self, x:Tensor) -> Tensor:
    return x.dot(self.w1).relu()
if __name__ == "__main__":
  model = TinyModel()
  model.w1.to_(GPUS) 
  step = 0
  x = Tensor.randn(BS, N).shard_(GPUS, axis=0)
  while True:
    out = model(x)
    (out - x).mean().realize()
    if step % 10 == 0:
      print(f"\rStep {step}", end="")
    if step % 100==0:
      fd = os.open('/dev/kfd', os.O_RDWR)
      print("ioctl 0")
      # kfd.AMDKFD_IOC_RUNTIME_ENABLE(KFDIface.kfd, mode_mask=0)
      fcntl.ioctl(fd, 0xc0104b25, bytearray(12))
      print("ioctl 1")
    step += 1
