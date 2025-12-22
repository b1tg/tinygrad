from tinygrad import Tensor, dtypes, nn
from tinygrad.helpers import getenv
from tinygrad import Tensor, dtypes, UOp
from tinygrad.uop.ops import KernelInfo, AxisType, Ops
from tinygrad.helpers import prod, argfix, Context
from tinygrad.helpers import Timing
import numpy as np
import time
from tinygrad import Tensor, Device
from tinygrad.helpers import getenv



x1 = Tensor([[1.2, 447, 1.2, -448.0], [1.2, -459, 1.2, -448.2], [1.2, 449, 1.2, -448.2]])
x1.requires_grad = True
y1 = x1.abs().max().detach()
y1.sum().backward()
print(x1.grad.numpy())
# [[ 0.  0.  0. -0.]
#  [ 0. -1.  0. -0.]
#  [ 0.  0.  0. -0.]]


x1 = Tensor([[1.2, 447, 1.2, -448.0], [1.2, -459, 1.2, -448.2], [1.2, 449, 1.2, -448.2]])
x1.requires_grad = True
y1 = x1.abs().max1()
y1.sum().backward()
print(x1.grad.numpy())
# [[0. 0. 0. 0.]
#  [0. 0. 0. 0.]
#  [0. 0. 0. 0.]]