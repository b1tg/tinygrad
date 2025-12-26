"""FP8 Quantization Module

Provides FP8 quantized layers for training and inference with reduced
precision and improved performance.
"""

from extra.fp8.fp8_linear import FP8Linear, FP8LinearBert, quantize_to_fp8

__all__ = ['FP8Linear', 'FP8LinearBert', 'quantize_to_fp8']
