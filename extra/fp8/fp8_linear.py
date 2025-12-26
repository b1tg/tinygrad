"""
FP8 Linear Layer

A drop-in replacement for nn.Linear that uses FP8 quantization for both
activations and weights. Supports both 2D and 3D inputs with automatic
shape handling.

Usage:
    from extra.fp8 import FP8Linear

    # Works with 2D inputs (like nn.Linear)
    layer = FP8Linear(512, 256)
    x = Tensor.randn(32, 512)
    y = layer(x)  # (32, 256)

    # Also works with 3D inputs (sequence models)
    x = Tensor.randn(32, 128, 512)
    y = layer(x)  # (32, 128, 256)

Note:
    FP8 quantization introduces ~10% error compared to FP32/FP16 due to
    reduced precision (3 mantissa bits). This is acceptable for training
    large models but may not be suitable for all tasks.
"""

from tinygrad import Tensor, dtypes, UOp, Device
from tinygrad.uop.ops import KernelInfo, AxisType, Ops
from tinygrad.helpers import getenv


# ============================================================================
# Quantization Utilities
# ============================================================================

def quantize_to_fp8(x: Tensor, axis=None, dtype=dtypes.fp8e4m3):
  """
  Quantize a tensor to FP8 format using dynamic scaling.

  Uses straight-through estimator (STE) for gradients - forward pass uses
  quantized values, backward pass flows through as if no quantization.

  Args:
      x: Input tensor
      axis: Axis for per-channel quantization (None = per-tensor)
      dtype: FP8 dtype (fp8e4m3 or fp8e5m2)

  Returns:
      Tuple of (quantized_tensor, reciprocal_scale)
      reciprocal_scale is 1/scale for efficient descaling
  """
  # Compute dynamic scale from absolute maximum
  x_abs_max = x.abs().max(axis=axis, keepdim=True).detach()
  scale = 448. / (x_abs_max + 1e-8)  # 448 = max value for FP8E4M3

  # Scale input
  x_scaled = x * scale

  # STE clamp: forward uses clamped, backward uses original
  x_det = x_scaled.detach()
  x_clamped = x_det.maximum(-448.0).minimum(448.0)
  x_clamped_ste = x_scaled + (x_clamped - x_det)
  # x_clamped_ste = x_clamped

  # Cast to FP8 and ensure contiguous
  res = x_clamped_ste.cast(dtype).contiguous()
  return res, scale.float().reciprocal().contiguous()

def custom_linear(C: UOp, A: UOp, B: UOp) -> UOp:
  """
  Custom matmul kernel: C[batch, seq, out] = A[batch, seq, in] @ B[out, in]

  Manually implements matrix multiplication using nested loops.
  Can be faster than standard .dot() for certain shapes on specific hardware.
  """
  SEQ = A.shape[1]
  OUT = B.shape[0]
  IN = B.shape[-1]

  # Loop iterators
  c2 = UOp.range(SEQ, 2, AxisType.LOOP)        # Sequence dimension
  c5 = UOp.range(OUT, 3, AxisType.LOOP)        # Output features
  c8 = UOp.range(C.size//SEQ//OUT, 1, AxisType.LOOP)  # Batch dimension
  c16 = UOp.range(IN, 0, AxisType.REDUCE)      # Reduction (input features)

  # Compute: C[batch, seq, out] = sum(A[batch, seq, in] * B[out, in])
  c27 = (A.index((c2*IN+c16+c8*IN*SEQ)) * B.index((c5*IN+c16))).cast(dtypes.float)
  c28 = c27.reduce(c16, arg=Ops.ADD)
  c30 = C.index((c2*OUT+c5+c8*OUT*SEQ), ptr=True).store(c28).end(c8, c2, c5)
  return c30.sink(arg=KernelInfo(name=f"custom dot {A.shape}x{B.shape}"))


def custom_linear_backward(gradient: UOp, kernel: UOp) -> tuple[UOp, UOp]:
  """
  Custom backward pass for FP8 linear layer.

  Quantizes gradient to FP8 to reduce memory bandwidth during backward pass.
  Computes gradients for both input (grad_a) and weight (grad_b).

  Args:
      gradient: Gradient from upstream (shape: batch, seq, out_features)
      kernel: The forward kernel UOp with src = (out, a, b) where:
              a = FP8 activations (batch, seq, in_features)
              b = FP8 weights (out_features, in_features)

  Returns:
      (None, grad_a, grad_b) where:
              grad_a: gradient w.r.t. input activations
              grad_b: gradient w.r.t. weights
  """
  out, a, b = kernel.src

  # Reshape gradient and activations to 2D for matmul
  # gradient: (batch, seq, out_features) -> (batch*seq, out_features)
  # a: (batch, seq, in_features) -> (batch*seq, in_features)
  a_tensor = Tensor(a, device=a.device)
  g_tensor = Tensor(gradient, device=gradient.device)
  b_tensor = Tensor(b, device=b.device)

  a_2d = a_tensor.reshape(a_tensor.shape[0] * a_tensor.shape[1], a_tensor.shape[-1])
  g_2d = g_tensor.reshape(g_tensor.shape[0] * g_tensor.shape[1], g_tensor.shape[-1])

  # Quantize gradient to FP8 for bandwidth savings
  g_quantized, scale = quantize_to_fp8(g_2d)

  # Compute weight gradient: grad_b = g.T @ a
  # (out_features, batch*seq) @ (batch*seq, in_features) = (out_features, in_features)
  grad_b = (g_quantized.T.dot(a_2d, dtype=dtypes.float)).contiguous() * scale
  grad_b = grad_b.cast(dtypes.float)

  # Compute input gradient: grad_a = g @ b
  # (batch*seq, out_features) @ (out_features, in_features) = (batch*seq, in_features)
  grad_a = (g_quantized.dot(b_tensor, dtype=dtypes.float)).contiguous().reshape(a_tensor.shape) * scale

  return (None, grad_a.uop, grad_b.uop)

def custom_linear_backward(gradient: UOp, kernel: UOp) -> tuple[UOp, UOp]:
    out, a, b = kernel.src
    
    g_tensor = Tensor(gradient, device=gradient.device)
    a_tensor = Tensor(a, device=a.device)
    b_tensor = Tensor(b, device=b.device)
    
    g_quantized, scale = quantize_to_fp8(g_tensor)
    scale = scale.reshape(())  # 确保是标量
    # grad_a: 不涉及reduce，保持维度
    grad_a = ((g_quantized.dot(b_tensor, dtype=dtypes.float)) * scale).cast(dtypes.float).contiguous()
    
    # grad_b: 处理2D和多维输入
    if len(g_tensor.shape) == 2:
        # 2D输入: (batch, out_features) 和 (batch, in_features)
        g_flat = g_quantized
        a_flat = a_tensor
    else:
        # 多维输入: flatten除了最后一维的所有维度
        # (dim0, dim1, ..., dimN, out_features) -> (dim0*dim1*...*dimN, out_features)
        batch_seq = 1
        for dim in g_tensor.shape[:-1]:
            batch_seq *= dim
        g_flat = g_quantized.reshape(batch_seq, g_tensor.shape[-1])
        a_flat = a_tensor.reshape(batch_seq, a_tensor.shape[-1])
    
    grad_b = ((g_flat.T.dot(a_flat, dtype=dtypes.float)) * scale).cast(dtypes.float).contiguous()
    
    return (None, grad_a.uop, grad_b.uop)

class FP8Linear:
  """
  FP8 quantized linear layer - drop-in replacement for nn.Linear.

  Quantizes both weights and activations to FP8 format, providing ~2x speedup
  over FP16 on supported hardware with ~10% accuracy loss.

  Args:
      in_features: Size of input features
      out_features: Size of output features
      bias: Whether to include bias term (default: True)
      use_custom_kernel: Use custom matmul kernel instead of .dot() (default: False)

  Shape:
      - Input: (*, in_features) or (batch, seq, in_features)
      - Output: (*, out_features) or (batch, seq, out_features)

  Examples:
      # 2D input (standard linear layer usage)
      >>> layer = FP8Linear(512, 256)
      >>> x = Tensor.randn(32, 512)
      >>> y = layer(x)  # (32, 256)

      # 3D input (sequence model usage)
      >>> x = Tensor.randn(32, 128, 512)
      >>> y = layer(x)  # (32, 128, 256)
  """

  def __init__(self, in_features: int, out_features: int, bias: bool = True,
               use_custom_kernel: bool = True):
    self.in_features = in_features
    self.out_features = out_features
    self.use_custom_kernel = use_custom_kernel or getenv("FP8_CUSTOM_KERNEL", 0)

    self.weight = Tensor.empty(out_features, in_features, dtype=dtypes.float32)
    self.bias = Tensor.empty(out_features, dtype=dtypes.float32) if bias else None

    # Multi-GPU support
    if getenv("GPUS", 1) > 1:
      self.GPUS = tuple(f"{Device.DEFAULT}:{i}" for i in range(getenv("GPUS", 1)))
    else:
      self.GPUS = Device.DEFAULT

  def __call__(self, x: Tensor) -> Tensor:
    """
    Forward pass with automatic shape handling.

    Supports both 2D (batch, features) and 3D (batch, seq, features) inputs.
    Internally normalizes to 3D for processing, then restores original shape.
    """
    # Save original shape info
    original_ndim = len(x.shape)

    # Validate and normalize to 3D
    if original_ndim == 2:
      # (batch, in_features) -> (batch, 1, in_features)
      batch, in_feat = x.shape
      assert in_feat == self.in_features, f"Input size {in_feat} doesn't match layer size {self.in_features}"
      x = x.reshape(batch, 1, in_feat)
    elif original_ndim == 3:
      # Already 3D, just validate
      batch, seq, in_feat = x.shape
      assert in_feat == self.in_features, f"Input size {in_feat} doesn't match layer size {self.in_features}"
    else:
      raise ValueError(f"FP8Linear only supports 2D or 3D inputs, got {original_ndim}D: {x.shape}")

    # At this point x is always 3D: (batch, seq, in_features)
    batch, seq, _ = x.shape

    # Quantize weight and input to FP8
    w_fp8, w_scale = quantize_to_fp8(self.weight)
    x_fp8, x_scale = quantize_to_fp8(x)

    # Compute output
    if self.use_custom_kernel:
      # Use custom kernel (faster on some hardware)
      if isinstance(self.GPUS, (tuple, list)) and len(self.GPUS) > 1:
        # Multi-GPU case
        y = Tensor(Tensor.empty((batch//len(self.GPUS), seq, self.out_features),
                                dtype=dtypes.float, device=self.GPUS).uop.multi(0),
                   device=self.GPUS)
        y = Tensor.custom_kernel(y, x_fp8, w_fp8,
                                  fxn=custom_linear,
                                  grad_fxn=custom_linear_backward)[0]
      else:
        # Single GPU case
        y = Tensor.empty((batch, seq, self.out_features), dtype=dtypes.float)
        y = Tensor.custom_kernel(y, x_fp8, w_fp8,
                                  fxn=custom_linear,
                                  grad_fxn=custom_linear_backward)[0]
    else:
      # Use standard matmul (more portable, slightly slower)
      # y = x_fp8 @ w_fp8.T (in float)
      y = x_fp8.cast(dtypes.float).dot(w_fp8.T.cast(dtypes.float), dtype=dtypes.float)

    # Descale output: y_float = y_fp8 * (1/w_scale) * (1/x_scale)
    # y = (w_scale * x_scale).contiguous() * y.contiguous()
    y = (y*w_scale * x_scale).contiguous()

    # Add bias if present
    if self.bias is not None:
      y = y.cast(dtypes.default_float) + self.bias.cast(dtypes.default_float)

    # Restore original shape
    if original_ndim == 2:
      # (batch, 1, out_features) -> (batch, out_features)
      y = y.reshape(batch, self.out_features)

    # Cast to input dtype and return
    return y.cast(x.dtype) if original_ndim == 3 else y.cast(dtypes.default_float)


FP8LinearBert = FP8Linear
