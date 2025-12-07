from tinygrad import Tensor, dtypes, UOp
from tinygrad.uop.ops import KernelInfo, AxisType, Ops
import torch
def backward_gemm(gradient:UOp, kernel:UOp) -> tuple[UOp, UOp]:
  out, a, b = kernel.src
  grad_a = (Tensor(gradient) @ Tensor(b).T).uop
  grad_b = (Tensor(a).T @ Tensor(gradient)).uop
  return (None, grad_a, grad_b)
def k_clamp_back(grads:UOp, kernel:UOp):
  # print(f"grads xxx: ", Tensor(grads).numpy())
  y, x, s = kernel.src
  ctx = grads
  y = 448.0
  ctx1 = (y>x).where(ctx, (x.eq(y)).where(ctx * 1, 0))
  y = -448.0
  ctx2 = (y<x).where(ctx1, (x.eq(y)).where(ctx1 * 1, 0))
  return (None, ctx2, None)
  return (None, Tensor(grads).uop, None)
  return (None, None, None)

  return (Tensor.ones_like(Tensor(y)).uop, Tensor.ones_like(Tensor(x)).uop, Tensor.ones_like(Tensor(s)).uop)
  grad_x = (Tensor(grads)+1).uop
  return (None, grad_x, None)
# k_clamp_back =backward_gemm
def k_clamp(y:UOp, x:UOp, s: UOp):
  y = y.flatten()
  x = x.flatten()
  s = s.flatten()
  i = UOp.range(x.size, 0)
  x1 = (x[i]*s.index(UOp.const(dtypes.index, 0))).maximum(UOp.const(x.dtype.base, -448.0)).minimum(UOp.const(x.dtype.base, 448.0))
  # x1 = (x[i]*s.index(UOp.const(dtypes.index, 0)))
  return y[i].store(x1).end(i).sink(arg=KernelInfo(name=f"k_clamp{x.size}",opts_to_apply=()))

def k_clamp(y:UOp, x:UOp, s: UOp):
  # c0 = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(2), (), 0)
  c0 = y
  c2 = UOp.range(x.size, 0, AxisType.LOOP)
  # c4 = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(2), (), 1)
  c4 = x
  c15 = (UOp(Ops.MAX, dtypes.float, ((UOp(Ops.MAX, dtypes.float, ((c4.index(c2)*1.0), UOp.const(dtypes.float, -448.0)))*-1.0), UOp.const(dtypes.float, -448.0)))*-1.0).cast(dtypes.fp8e4m3).cast(dtypes.float)
  c17 = c0.index(c2, ptr=True).store(c15).end(c2)
  ast = c17.sink()
  return ast

dtype = dtypes.fp8e4m3

def f0(x: Tensor, scale: Tensor):
  y = (x * scale).clamp(-448.0, 448.0)
  res = y
  res= y.cast(dtype)

  return res
def f1(x: Tensor, scale: Tensor):
  y = Tensor.empty_like(x)
  y = Tensor.custom_kernel(y, x, scale, fxn=k_clamp, grad_fxn=k_clamp_back)[0]
  res = y
  # res= y.cast(dtype)
  return res

def q_abs_max_kernel(y: UOp, x:UOp):
  B = y.flatten()
  A = x.flatten()
  i = UOp.range(A.shape[0], 0, axis_type=AxisType.REDUCE)
  B = B[0].set(UOp.const(x.dtype.base, 0.0))
  B = B[0].set(B.after(i)[0].maximum((A[i]<0.0).where(A[i]*UOp.const(x.dtype.base, UOp.const(x.dtype.base, -1.0)), A[i])), end=i)
  B = B[0].set(B[0].reciprocal()*448.0)
  return B.sink(arg=KernelInfo(name=f"custom_sumx_{A.shape[0]}"))

def q_abs_max_kernel(y: UOp, x:UOp):
  # c0 = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(1), (), 0)
  # c4 = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(7), (), 1)
  c0 = y
  c4 = x
  c6 = UOp.range(x.size, 0, AxisType.REDUCE)
  c7 = c4.index(c6)
  c13 = (c7<0.0).where(UOp.const(dtypes.float, -1.0), UOp.const(dtypes.float, 1.0))
  c14 = (c7!=0.0).where(c13, UOp.const(dtypes.float, 0.0))
  c15 = c7*c14
  c20 = 448.0*(c15.reduce(c6, arg=Ops.MAX)+1e-08).reciprocal()
  c21 = c0.index(UOp.const(dtypes.index, 0), ptr=True).store(c20)
  ast = c21.sink()
  return ast

def q_abs_max_kernel_back(grads:UOp, kernel:UOp):
  y, x = kernel.src
  # 1/0
  # print(f"{Tensor(grads).numpy()=}")
  # return (None, Tensor(grads).uop)
  return (None, None)
  # print(Tensor(grads).numpy())
  # return (None, Tensor.ones_like(Tensor(x)).uop)
  return (grads, grads)
x = Tensor([409.0, 1508.0, 2.0, 3, -555, 448.4, 448.0, 1]).reshape((2,4))
x.requires_grad = True
y = Tensor.empty((1,), dtype=x.dtype)
y.requires_grad = True
y = Tensor.custom_kernel(y, x, fxn=q_abs_max_kernel, grad_fxn=q_abs_max_kernel_back)[0]
# y = Tensor.custom_kernel(y, x, fxn=q_abs_max_kernel)[0]

# x_abs_max = x.abs().max()
# y = 448. / (x_abs_max + 1e-8)
# print(f"{y.numpy()=}")
x.mul(y).sum().backward()
print(x.grad.numpy())

exit()
print("--- torch ---")
x = torch.Tensor([409.0, 1508.0, 2.0, 3, -555, 448.4, 448.0])
x.requires_grad = True
x.clamp(-448.0, 448.0).sum().backward()
print(x.grad.numpy())
import torch

x = torch.tensor([1.0], requires_grad=True)
y = torch.tensor([1.0], requires_grad=True)

z = torch.max(x, y)
z.backward()

print(x.grad, y.grad)

print("-- end of torch ---")

x = Tensor([409.0, 1508.0, 2.0, 3, -555, 448.4, 448.0], dtype=dtypes.float, requires_grad=True)
scale = Tensor(1.0, requires_grad=True)

y0 = f0(x, scale)
print(y0.numpy(), y0.dtype)

y0.sum().backward()
print("grad:", x.grad.numpy())
print("----")
x = Tensor([409.0, 1508.0, 2.0, 3, -555, 448.4, 448.0], dtype=dtypes.float, requires_grad=True)
scale = Tensor(1.0, requires_grad=True)
y1 = f1(x, scale)
# print(y1.numpy(), y1.dtype)
y1.sum().backward()
print("grad:", x.grad.numpy())

exit()