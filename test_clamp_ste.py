from tinygrad import Tensor

def clamp_ste(x: Tensor, min_val, max_val) -> Tensor:
    """Clamp with Straight-Through Estimator - gradient passes through unchanged"""
    x_det = x.detach()
    clamped = x_det.maximum(min_val).minimum(max_val)
    return x + (clamped - x_det)

print("Test 1: Standard clamp - gradient is 0 when out of bounds")
x1 = Tensor([0.5, 1.5, 2.5, 3.5], requires_grad=True)
y1 = x1.maximum(-2.0).minimum(2.0)  # clamp to [-2, 2]
y1.sum().backward()
print(f"x1: {x1.numpy()}")
print(f"y1 (clamped): {y1.numpy()}")
print(f"x1.grad: {x1.grad.numpy()}")
print("  ^ Notice: gradient is 0 for x=3.5 (out of bounds)\n")

print("Test 2: STE clamp - gradient passes through even when out of bounds")
x2 = Tensor([0.5, 1.5, 2.5, 3.5], requires_grad=True)
y2 = clamp_ste(x2, -2.0, 2.0)
y2.sum().backward()
print(f"x2: {x2.numpy()}")
print(f"y2 (clamped with STE): {y2.numpy()}")
print(f"x2.grad: {x2.grad.numpy()}")
print("  ^ Notice: all gradients are 1.0 (straight through!)\n")

print("Test 3: Apply to FP8 quantization scenario")
x3 = Tensor([100.0, 200.0, 500.0, 600.0], requires_grad=True)
scale = 448.0 / 600.0  # max value is 600

# Quantization with STE
x_scaled = x3 * scale
x_clamped = clamp_ste(x_scaled, -448.0, 448.0)
loss = x_clamped.sum()
loss.backward()

print(f"x3: {x3.numpy()}")
print(f"x_scaled: {x_scaled.numpy()}")
print(f"x_clamped: {x_clamped.numpy()}")
print(f"x3.grad: {x3.grad.numpy()}")
print(f"  ^ All gradients = {scale:.4f} (scale value, no clamp effect)")
