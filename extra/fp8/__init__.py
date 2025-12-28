"""FP8 Quantization Module

Provides FP8 quantized layers for training and inference with reduced
precision and improved performance.

Available classes:
- FP8Linear: Basic FP8 quantized linear layer
- FP8LinearBert: Alias for FP8Linear (BERT compatibility)
- FP8LinearCached: FP8Linear with weight caching (2.52x speedup with gradient accumulation)
- FP8Optimizer: Optimizer wrapper with automatic cache invalidation

Functions:
- quantize_to_fp8: Quantize tensor to FP8 format
- invalidate_all_fp8_caches: Manually invalidate all FP8 weight caches in a model
"""

from extra.fp8.fp8_linear import FP8Linear, FP8LinearBert, quantize_to_fp8
from extra.fp8.fp8_linear_cached import FP8LinearCached
from extra.fp8.optimizer_utils import FP8Optimizer, invalidate_all_fp8_caches

__all__ = [
    'FP8Linear',
    'FP8LinearBert',
    'FP8LinearCached',
    'quantize_to_fp8',
    'FP8Optimizer',
    'invalidate_all_fp8_caches',
]
