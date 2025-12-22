from tinygrad import Tensor

print("=" * 60)
print("理解 STE (Straight-Through Estimator) 的工作原理")
print("=" * 60)

# 设置输入
x = Tensor([3.5], requires_grad=True)
print(f"\n输入: x = {x.numpy()}, requires_grad=True")

# Step 1: detach
x_det = x.detach()
print(f"\nStep 1: x_det = x.detach()")
print(f"  x_det.numpy() = {x_det.numpy()}")
print(f"  x_det.requires_grad = {x_det.requires_grad}")
print(f"  关键: x_det 的值来自 x，但从计算图中分离了")

# Step 2: 对 detached 值应用操作
clamped = x_det.minimum(2.0)  # clamp 到最大值 2.0
print(f"\nStep 2: clamped = x_det.minimum(2.0)")
print(f"  clamped.numpy() = {clamped.numpy()}")
print(f"  clamped.requires_grad = {clamped.requires_grad}")
print(f"  关键: clamped 是从 x_det 计算的，所以它也没有梯度!")

# Step 3: 构造 STE 输出
output = x + (clamped - x_det)
print(f"\nStep 3: output = x + (clamped - x_det)")
print(f"  output.numpy() = {output.numpy()}")
print(f"  output.requires_grad = {output.requires_grad}")

# 前向计算分析
print(f"\n前向传播分析:")
print(f"  x = {x.numpy()[0]}")
print(f"  x_det = {x_det.numpy()[0]}")
print(f"  clamped = {clamped.numpy()[0]}")
print(f"  clamped - x_det = {clamped.numpy()[0]} - {x_det.numpy()[0]} = {(clamped - x_det).numpy()[0]}")
print(f"  output = x + (clamped - x_det) = {x.numpy()[0]} + {(clamped - x_det).numpy()[0]} = {output.numpy()[0]}")
print(f"  ✓ 前向结果 = clamped 的值!")

# 反向传播
output.sum().backward()

print(f"\n反向传播分析:")
print(f"  ∂output/∂x = ?")
print(f"  output = x + (clamped - x_det)")
print(f"           ↓     ↓")
print(f"           ↓     └─ (常量) 因为 clamped 和 x_det 都是 detached")
print(f"           └─ 有梯度")
print(f"  ")
print(f"  ∂output/∂x = ∂x/∂x + ∂(clamped - x_det)/∂x")
print(f"             = 1     +         0")
print(f"             = 1")
print(f"  ")
print(f"  实际计算: x.grad = {x.grad.numpy()}")
print(f"  ✓ 梯度完全传递，不受 clamp 影响!")

print("\n" + "=" * 60)
print("关键理解")
print("=" * 60)
print("1. x_det = x.detach()")
print("   → x_det 是'常量'（从梯度角度）")
print("")
print("2. clamped = x_det.minimum(2.0)")
print("   → clamped 也是'常量'（因为从常量计算）")
print("   → 即使 clamped 在计算图中，但它不会产生梯度回传到 x")
print("")
print("3. output = x + (clamped - x_det)")
print("   → (clamped - x_det) 整体是常量")
print("   → 相当于 output = x + 常量")
print("   → ∂output/∂x = 1")
print("")
print("结论: 前向=clamped值，反向=梯度直通!")

# 对比普通 clamp
print("\n" + "=" * 60)
print("对比: 普通 clamp (没有 STE)")
print("=" * 60)
x2 = Tensor([3.5], requires_grad=True)
y2 = x2.minimum(2.0)
y2.sum().backward()
print(f"x2 = {x2.numpy()}, y2 = {y2.numpy()}")
print(f"x2.grad = {x2.grad.numpy()}")
print(f"普通 clamp: 梯度被截断为 0 (因为超出范围)")
