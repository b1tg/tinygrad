from tinygrad import Tensor

# 测试1: max() 会产生梯度
print("Test 1: Using max() - gradients flow back")
x1 = Tensor([[1.2, 447, 1.2, -448.0], [1.2, -459, 1.2, -448.2], [1.2, 449, 1.2, -448.2]])
x1.requires_grad = True
y1 = x1.abs().max()
scale1 = 448.0 / (y1 + 1e-8)
loss1 = (x1 * scale1).sum()
loss1.backward()
print("x1.grad (has gradients from scale):")
print(x1.grad.numpy())
print()

# 测试2: max().detach() 不产生梯度
print("Test 2: Using max().detach() - no gradients from scale")
x2 = Tensor([[1.2, 447, 1.2, -448.0], [1.2, -459, 1.2, -448.2], [1.2, 449, 1.2, -448.2]])
x2.requires_grad = True
y2 = x2.abs().max().detach()  # detach here!
scale2 = 448.0 / (y2 + 1e-8)
loss2 = (x2 * scale2).sum()
loss2.backward()
print("x2.grad (scale detached, only x's direct gradient):")
print(x2.grad.numpy())
print()

# 测试3: 对比 max1() (你的临时方案)
print("Test 3: Using max1() - your temporary hack")
x3 = Tensor([[1.2, 447, 1.2, -448.0], [1.2, -459, 1.2, -448.2], [1.2, 449, 1.2, -448.2]])
x3.requires_grad = True
y3 = x3.abs().max1()
scale3 = 448.0 / (y3 + 1e-8)
loss3 = (x3 * scale3).sum()
loss3.backward()
print("x3.grad (using max1):")
print(x3.grad.numpy())
