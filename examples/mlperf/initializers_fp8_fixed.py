"""FP8LinearBert with fixed scale quantization for reduced overhead"""

from tinygrad import Tensor, dtypes, UOp
from tinygrad.uop.ops import KernelInfo, AxisType, Ops
from examples.mlperf.initializers import GPUS

# Fixed scale quantization (no abs().max() computation)
def quantize_to_fp8_fixed(x: Tensor, scale=448.0, dtype=dtypes.fp8e4m3):
    """Fixed scale quantization - much faster than dynamic

    Assumes input is roughly normalized (mean ~0, std ~1)
    For BERT with LayerNorm, this is usually true
    """
    x_scaled = (x * scale).clamp(-448.0, 448.0)
    res = x_scaled.cast(dtype)
    inv_scale = Tensor([1.0 / scale], dtype=dtypes.float, device=x.device)
    return res.contiguous().contiguous_backward(), inv_scale.contiguous().contiguous_backward()


# Per-channel weight quantization
def quantize_weight_per_channel(w: Tensor, dtype=dtypes.fp8e4m3):
    """Per-channel quantization for weights

    w shape: (out_features, in_features)
    Compute one scale per output channel
    """
    w_abs_max = w.abs().max(axis=1, keepdim=True).detach()
    scale = 448. / (w_abs_max + 1e-8)
    w_scaled = (w * scale).clamp(-448.0, 448.0)
    res = w_scaled.cast(dtype)
    return res, scale.reciprocal()


def custom_linear(C:UOp, A:UOp, B:UOp) -> UOp:
    SEQ = A.shape[1]
    OUT = B.shape[0]
    IN = B.shape[-1]
    c2 = UOp.range(SEQ, 2, AxisType.LOOP)
    c5 = UOp.range(OUT, 3, AxisType.LOOP)
    c8 = UOp.range(C.size//SEQ//OUT, 1, AxisType.LOOP)
    c16 = UOp.range(IN, 0, AxisType.REDUCE)
    c27 = (A.index((c2*IN+c16+c8*IN*SEQ))*B.index((c5*IN+c16))).cast(dtypes.float)
    c28 = c27.reduce(c16, arg=Ops.ADD)
    c30 = C.index((c2*OUT+c5+c8*OUT*SEQ), ptr=True).store(c28).end(c8, c2, c5)
    return c30.sink(arg=KernelInfo(name=f"custom dot {A.shape}x{B.shape}"))


def custom_linear_backward(gradient:UOp, kernel:UOp) -> tuple[UOp, UOp]:
    out, a, b = kernel.src
    a2 = Tensor(a, device=a.device).reshape(a.shape[0]*a.shape[1], a.shape[-1])
    g2 = Tensor(gradient, device=gradient.device).reshape(gradient.shape[0]*gradient.shape[1], gradient.shape[-1])

    # Use fixed scale for gradient quantization
    g2, s = quantize_to_fp8_fixed(g2, scale=448.0)

    grad_b = (g2.T.dot(a2, dtype=dtypes.float)).contiguous() * s
    grad_b = grad_b.cast(dtypes.float)
    grad_a = (g2.dot(Tensor(b, device=b.device), dtype=dtypes.float)).contiguous().reshape(a.shape) * s
    return (None, grad_a.uop, grad_b.uop)


class FP8LinearBertFixed:
    """FP8Linear with fixed scale for activations and per-channel for weights"""

    def __init__(self, in_features, out_features, bias=True,
                 input_scale=448.0, weight_per_channel=False):
        self.weight = Tensor.empty(out_features, in_features, dtype=dtypes.float32)
        self.bias = Tensor.empty(out_features, dtype=dtypes.float32) if bias else None
        self.input_scale = input_scale
        self.weight_per_channel = weight_per_channel

        # Cache for quantized weights
        self._w_fp8_cache = None
        self._w_scale_cache = None
        self._weight_version = 0

    def __call__(self, x: Tensor):
        # Quantize input with fixed scale (fast!)
        x1, s_x = quantize_to_fp8_fixed(x, scale=self.input_scale)

        # Quantize weight (cache or recompute)
        # In training, weight changes every step, but we could still cache for inference
        if self.weight_per_channel:
            w1, s_w = quantize_weight_per_channel(self.weight)
        else:
            w1, s_w = quantize_to_fp8_fixed(self.weight, scale=448.0)

        # Custom matmul
        if isinstance(GPUS, (tuple, list)) and len(GPUS) > 1:
            y = Tensor(Tensor.empty((x.shape[0]//len(GPUS), x.shape[1], self.weight.shape[0]),
                                   dtype=dtypes.float, device=GPUS).uop.multi(0), device=GPUS)
            y = Tensor.custom_kernel(y, x1, w1, fxn=custom_linear, grad_fxn=custom_linear_backward)[0]
        else:
            y = Tensor.empty((x.shape[0], x.shape[1], self.weight.shape[0]), dtype=dtypes.float)
            y = Tensor.custom_kernel(y, x1, w1, fxn=custom_linear, grad_fxn=custom_linear_backward)[0]

        # Scale and add bias
        if self.weight_per_channel:
            # Per-channel: broadcast scale across sequence dimension
            y = y * (s_w.reshape(1, 1, -1) * s_x)
        else:
            y = y * (s_w * s_x)

        y = y.contiguous()
        if self.bias is not None:
            y = y + self.bias.cast(y.dtype)

        return y.cast(x.dtype)


# For drop-in replacement in helpers.py
FP8LinearBert = FP8LinearBertFixed
