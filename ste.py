import torch
from tinygrad import Tensor
def round_straight_through(x: Tensor) -> Tensor:
    x_det = x.detach()
    # rounded_val = x_det.round()
    rounded_val = x_det.clamp(1)
    # Combine to form output: (rounded_val - x_det) has no grad, x has grad.
    return x + (rounded_val - x_det)
class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # In the forward pass, perform the non-differentiable operation
        # For example, a simple rounding operation
        # output = torch.round(input)
        output = input.clamp(1)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # In the backward pass, pass the gradient straight through
        # This means the gradient of the input is equal to the gradient of the output
        grad_input = grad_output.clone()
        return grad_input

# Create an instance of the custom function
ste_op = STEFunction.apply

# Example usage:
x = torch.tensor([0.2, 0.7, 1.3, 1.8], requires_grad=True)
# y = ste_op(x) # Apply the straight-through estimator
y = x.clamp(1)
# loss = (y - torch.tensor([0., 1., 1., 2.])).pow(2).sum()
loss = y.sum()
loss.backward()

print("Input tensor:", x)
print("Output of STE:", y)
print("Gradient of input:", x.grad)
print("---- ")

x = Tensor([0.2, 0.7, 1.3, 1.8], requires_grad=True)
y= round_straight_through(x)
# y = x.clamp(1)
loss = y.sum()
loss.backward()
print(y.numpy())
print(x.grad.numpy())