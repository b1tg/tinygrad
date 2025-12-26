#!/usr/bin/env python3
"""
FP8Linear Usage Examples

Demonstrates how to use FP8Linear as a drop-in replacement for nn.Linear
in various models (BERT, Llama, custom architectures).
"""

from tinygrad import Tensor
from tinygrad.nn import Linear
from extra.fp8 import FP8Linear


def example_basic_usage():
  """Basic FP8Linear usage - drop-in replacement for nn.Linear"""
  print("="*60)
  print("Example 1: Basic Usage")
  print("="*60)

  # Create FP8 linear layer
  layer = FP8Linear(512, 256)
  layer.weight.assign(Tensor.randn(256, 512) * 0.02)
  layer.bias.assign(Tensor.zeros(256))

  # Works with 2D inputs (batch, features)
  x_2d = Tensor.randn(32, 512)
  y_2d = layer(x_2d)
  print(f"2D input: {x_2d.shape} -> {y_2d.shape}")

  # Also works with 3D inputs (batch, seq, features)
  x_3d = Tensor.randn(32, 128, 512)
  y_3d = layer(x_3d)
  print(f"3D input: {x_3d.shape} -> {y_3d.shape}")

  # Supports gradients
  layer.weight.requires_grad = True
  x_3d.requires_grad = True
  y_3d = layer(x_3d)
  y_3d.sum().backward()
  print(f"Gradient shapes: weight={layer.weight.grad.shape}, input={x_3d.grad.shape}")
  print()


def example_bert_integration():
  """Using FP8Linear with BERT models"""
  print("="*60)
  print("Example 2: BERT Integration")
  print("="*60)

  # Method 1: Direct monkeypatch
  from extra.models import bert
  bert.QuantLinear = FP8Linear

  print("BERT integration: Set bert.QuantLinear = FP8Linear")
  print("Now BERT models will use FP8 quantization for linear layers")

  # Method 2: Using environment variable (MLPerf style)
  print("\nAlternatively, use FP8=1 environment variable:")
  print("  FP8=1 python examples/mlperf/model_train.py ...")
  print()


def example_llama_integration():
  """Using FP8Linear with Llama models"""
  print("="*60)
  print("Example 3: Llama Integration")
  print("="*60)

  # Llama models accept `linear` parameter in constructor
  print("Llama integration via constructor parameter:")
  print()
  print("  from extra.fp8 import FP8Linear")
  print("  from extra.models.llama import Transformer")
  print()
  print("  model = Transformer(")
  print("    dim=4096,")
  print("    n_layers=32,")
  print("    linear=FP8Linear,  # Use FP8 quantization")
  print("    ...)")
  print()


def example_custom_kernel():
  """Using custom kernel for better performance"""
  print("="*60)
  print("Example 4: Custom Kernel (Advanced)")
  print("="*60)

  # Standard ops (default, more portable)
  layer_standard = FP8Linear(512, 256, use_custom_kernel=False)
  print("Standard ops: FP8Linear(512, 256, use_custom_kernel=False)")

  # Custom kernel (faster on some hardware)
  layer_custom = FP8Linear(512, 256, use_custom_kernel=True)
  print("Custom kernel: FP8Linear(512, 256, use_custom_kernel=True)")

  # Or use environment variable
  print("\nAlternatively, use FP8_CUSTOM_KERNEL=1 environment variable:")
  print("  FP8_CUSTOM_KERNEL=1 python your_script.py")
  print()


def example_comparison():
  """Compare FP8 vs normal Linear"""
  print("="*60)
  print("Example 5: FP8 vs Normal Linear Comparison")
  print("="*60)

  in_features, out_features = 512, 256
  batch_size = 32

  # Create both layers with same weights
  fp8_layer = FP8Linear(in_features, out_features)
  normal_layer = Linear(in_features, out_features)

  weight = Tensor.randn(out_features, in_features) * 0.02
  fp8_layer.weight.assign(weight.detach())
  normal_layer.weight.assign(weight.detach())

  bias = Tensor.zeros(out_features)
  fp8_layer.bias.assign(bias.detach())
  normal_layer.bias.assign(bias.detach())

  # Forward pass
  x = Tensor.randn(batch_size, in_features)
  y_fp8 = fp8_layer(x).numpy()
  y_normal = normal_layer(x).numpy()

  # Compare
  import numpy as np
  abs_diff = np.abs(y_fp8 - y_normal)
  print(f"Output shape: {y_fp8.shape}")
  print(f"Max absolute difference: {abs_diff.max():.4f}")
  print(f"Mean absolute difference: {abs_diff.mean():.4f}")
  print(f"Relative error: {(abs_diff / (np.abs(y_normal) + 1e-8)).max():.4f}")
  print()
  print("Note: FP8 has reduced precision (~3 mantissa bits) compared to FP32")
  print("Expected error range: 1-5% for typical values")
  print()


def example_performance_notes():
  """Performance characteristics and when to use FP8"""
  print("="*60)
  print("Performance Notes")
  print("="*60)
  print()
  print("When to use FP8Linear:")
  print("  - Large models (BERT-large, Llama-8B/405B)")
  print("  - Training with mixed precision")
  print("  - Memory-constrained environments")
  print("  - Batch size > 16, seq length > 128")
  print()
  print("Expected speedup:")
  print("  - vs FP32: 1.5-2.0x on most GPUs")
  print("  - vs FP16: 1.2-1.5x on GPUs with FP8 support")
  print()
  print("Trade-offs:")
  print("  - ~1-5% accuracy loss (tolerable for large models)")
  print("  - Not recommended for small models or tasks requiring high precision")
  print("  - Quantization overhead: ~10% of forward time (negligible for large matmuls)")
  print()
  print("Benchmark:")
  print("  python test/external/external_benchmark_fp8_linear.py")
  print()


if __name__ == "__main__":
  example_basic_usage()
  example_bert_integration()
  example_llama_integration()
  example_custom_kernel()
  example_comparison()
  example_performance_notes()
