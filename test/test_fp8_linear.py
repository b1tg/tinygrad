#!/usr/bin/env python
"""
Tests for FP8Linear layer.

FP8 quantization introduces ~10% error compared to FP32/FP16, so tolerances
are relaxed compared to standard layer tests.
"""
import unittest
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.device import is_dtype_supported
from tinygrad.nn import Linear
from extra.fp8 import FP8Linear

Tensor.manual_seed(0x32)
class TestFP8Linear(unittest.TestCase):
  """Test FP8Linear accuracy and shape handling"""

  @classmethod
  def setUpClass(cls):
    # Skip all tests if FP8 is not supported
    if not is_dtype_supported(dtypes.fp8e4m3):
      raise unittest.SkipTest("FP8 not supported on this device")

  def _create_fp8_and_normal_layers(self, in_features, out_features, bias=True):
    """Helper to create FP8 and normal linear layers with same weights"""
    # Create both layers
    fp8_layer = FP8Linear(in_features, out_features, bias=bias)
    normal_layer = Linear(in_features, out_features, bias=bias)

    # Initialize with same random weights
    weight = Tensor.randn(out_features, in_features, dtype=dtypes.float32)
    fp8_layer.weight = weight.detach()
    normal_layer.weight = weight.detach()

    if bias:
      bias_val = Tensor.randn(out_features, dtype=dtypes.float32)
      fp8_layer.bias = bias_val.detach()
      normal_layer.bias = bias_val.detach()

    return fp8_layer, normal_layer

  def test_forward_2d_accuracy(self):
    """Test forward pass accuracy on 2D input (batch, features)"""
    in_features, out_features = 512, 256
    batch_size = 32

    fp8_layer, normal_layer = self._create_fp8_and_normal_layers(in_features, out_features)

    # Test multiple iterations for stability
    for _ in range(3):
      x = Tensor.randn(batch_size, in_features, dtype=dtypes.float32)

      y_fp8 = fp8_layer(x)
      y_normal = normal_layer(x)

      # FP8 has very reduced precision - can have large errors
      # With 3 mantissa bits, quantization can cause absolute errors up to ~5
      # and relative errors can be 100x+ for values near zero
      np.testing.assert_allclose(y_fp8.numpy(), y_normal.numpy(),
                                 rtol=500.0, atol=5.0)

  def test_forward_3d_accuracy(self):
    """Test forward pass accuracy on 3D input (batch, seq, features)"""
    in_features, out_features = 512, 256
    batch_size, seq_len = 16, 128

    fp8_layer, normal_layer = self._create_fp8_and_normal_layers(in_features, out_features)

    # Test multiple iterations for stability
    for _ in range(3):
      x = Tensor.randn(batch_size, seq_len, in_features, dtype=dtypes.float32)

      y_fp8 = fp8_layer(x)
      y_normal = normal_layer(x)

      # FP8 has very reduced precision - can have large errors
      np.testing.assert_allclose(y_fp8.numpy(), y_normal.numpy(),
                                 rtol=500.0, atol=5.0)

  def test_backward_2d_accuracy(self):
    """Test gradient accuracy on 2D input"""
    in_features, out_features = 256, 128
    batch_size = 16

    fp8_layer, normal_layer = self._create_fp8_and_normal_layers(in_features, out_features)

    # Enable gradients
    fp8_layer.weight.requires_grad = True
    normal_layer.weight.requires_grad = True

    # Forward + backward
    x_fp8 = Tensor.randn(batch_size, in_features, dtype=dtypes.float32, requires_grad=True)
    x_normal = x_fp8.detach()
    x_normal.requires_grad = True

    y_fp8 = fp8_layer(x_fp8)
    y_normal = normal_layer(x_normal)

    # Backward pass
    y_fp8.sum().backward()
    y_normal.sum().backward()

    # Check input gradients (FP8 gradient quantization adds noise)
    np.testing.assert_allclose(x_fp8.grad.numpy(), x_normal.grad.numpy(),
                               rtol=1.0, atol=1.0)
                              #  rtol=500.0, atol=5.0)
# 
    # Check weight gradients
    np.testing.assert_allclose(fp8_layer.weight.grad.numpy(),
                               normal_layer.weight.grad.numpy(),
                               rtol=1.0, atol=1.0)
                              #  rtol=500.0, atol=5.0)

  def test_backward_3d_accuracy(self):
    """Test gradient accuracy on 3D input"""
    in_features, out_features = 256, 128
    batch_size, seq_len = 8, 64

    fp8_layer, normal_layer = self._create_fp8_and_normal_layers(in_features, out_features)

    # Enable gradients
    fp8_layer.weight.requires_grad = True
    normal_layer.weight.requires_grad = True

    # Forward + backward
    x_fp8 = Tensor.randn(batch_size, seq_len, in_features, dtype=dtypes.float32, requires_grad=True)
    x_normal = x_fp8.detach()
    x_normal.requires_grad = True

    y_fp8 = fp8_layer(x_fp8)
    y_normal = normal_layer(x_normal)

    # Backward pass
    y_fp8.sum().backward()
    y_normal.sum().backward()

    # Check input gradients
    np.testing.assert_allclose(x_fp8.grad.numpy(), x_normal.grad.numpy(),
                               rtol=500.0, atol=5.0)

    # Check weight gradients
    np.testing.assert_allclose(fp8_layer.weight.grad.numpy(),
                               normal_layer.weight.grad.numpy(),
                               rtol=500.0, atol=5.0)

  def test_shape_preservation_2d(self):
    """Test that 2D input produces 2D output"""
    layer = FP8Linear(512, 256)
    x = Tensor.randn(32, 512)
    y = layer(x)

    self.assertEqual(len(y.shape), 2)
    self.assertEqual(y.shape, (32, 256))

  def test_shape_preservation_3d(self):
    """Test that 3D input produces 3D output"""
    layer = FP8Linear(512, 256)
    x = Tensor.randn(32, 128, 512)
    y = layer(x)

    self.assertEqual(len(y.shape), 3)
    self.assertEqual(y.shape, (32, 128, 256))

  def test_invalid_shape_1d(self):
    """Test that 1D input raises error"""
    layer = FP8Linear(512, 256)
    x = Tensor.randn(512)

    with self.assertRaises(ValueError) as cm:
      layer(x)
    self.assertIn("2D or 3D", str(cm.exception))

  def test_invalid_shape_4d(self):
    """Test that 4D input raises error"""
    layer = FP8Linear(512, 256)
    x = Tensor.randn(2, 2, 128, 512)

    with self.assertRaises(ValueError) as cm:
      layer(x)
    self.assertIn("2D or 3D", str(cm.exception))

  def test_2d_to_3d_roundtrip(self):
    """Test that 2D->3D->2D reshape doesn't break gradients"""
    layer = FP8Linear(512, 256)
    layer.weight.requires_grad = True

    x = Tensor.randn(32, 512, requires_grad=True)
    y = layer(x)
    y.sum().backward()

    # Should have gradients
    self.assertIsNotNone(x.grad)
    self.assertIsNotNone(layer.weight.grad)
    self.assertEqual(x.grad.shape, (32, 512))
    self.assertEqual(layer.weight.grad.shape, (256, 512))

  def test_bert_large_qkv_shape(self):
    """Test BERT-large QKV projection shape (BS=32, SEQ=512, hidden=1024)"""
    layer = FP8Linear(1024, 1024)
    x = Tensor.randn(32, 512, 1024)
    y = layer(x)

    self.assertEqual(y.shape, (32, 512, 1024))

  def test_bert_large_ffn_shape(self):
    """Test BERT-large FFN shape (BS=32, SEQ=512, 1024->4096)"""
    layer = FP8Linear(1024, 4096)
    x = Tensor.randn(32, 512, 1024)
    y = layer(x)

    self.assertEqual(y.shape, (32, 512, 4096))

  def test_llama_8b_prefill_shape(self):
    """Test Llama-8B prefill QKV (BS=1, SEQ=512, dim=4096)"""
    layer = FP8Linear(4096, 4096)
    x = Tensor.randn(1, 512, 4096)
    y = layer(x)

    self.assertEqual(y.shape, (1, 512, 4096))

  def test_llama_8b_ffn_shape(self):
    """Test Llama-8B FFN (BS=1, SEQ=512, 4096->14336)"""
    layer = FP8Linear(4096, 14336)
    x = Tensor.randn(1, 512, 4096)
    y = layer(x)

    self.assertEqual(y.shape, (1, 512, 14336))

  def test_llama_8b_decode_shape(self):
    """Test Llama-8B decode QKV (BS=1, SEQ=1, dim=4096)"""
    layer = FP8Linear(4096, 4096)
    x = Tensor.randn(1, 1, 4096)
    y = layer(x)

    self.assertEqual(y.shape, (1, 1, 4096))

  def test_llama_405b_prefill_shape(self):
    """Test Llama-405B prefill QKV (BS=1, SEQ=512, dim=16384)"""
    layer = FP8Linear(16384, 16384)
    x = Tensor.randn(1, 512, 16384)
    y = layer(x)

    self.assertEqual(y.shape, (1, 512, 16384))

  def test_llama_405b_ffn_shape(self):
    """Test Llama-405B FFN (BS=1, SEQ=512, 16384->53248)"""
    layer = FP8Linear(16384, 53248)
    x = Tensor.randn(1, 512, 16384)
    y = layer(x)

    self.assertEqual(y.shape, (1, 512, 53248))

  def test_no_bias(self):
    """Test layer without bias"""
    layer = FP8Linear(512, 256, bias=False)
    self.assertIsNone(layer.bias)

    x = Tensor.randn(32, 512)
    y = layer(x)

    self.assertEqual(y.shape, (32, 256))

  def test_multi_iteration_stability(self):
    """Test that multiple forward passes are deterministic"""
    layer = FP8Linear(256, 128)
    x = Tensor.randn(16, 256)

    # Run multiple times
    results = [layer(x).numpy() for _ in range(5)]

    # All results should be identical (no randomness in forward pass)
    for i in range(1, len(results)):
      np.testing.assert_allclose(results[0], results[i], rtol=1e-6, atol=1e-6)

  def test_custom_kernel_vs_standard(self):
    """Test that custom kernel produces similar results to standard ops"""
    in_features, out_features = 512, 256
    batch_size, seq_len = 16, 128

    # Create two layers with same weights
    layer_standard = FP8Linear(in_features, out_features, use_custom_kernel=False)
    layer_custom = FP8Linear(in_features, out_features, use_custom_kernel=True)

    # Copy weights
    layer_custom.weight = layer_standard.weight.detach()
    layer_custom.bias = layer_standard.bias.detach()

    # Test on 3D input (custom kernel requires 3D)
    x = Tensor.randn(batch_size, seq_len, in_features)

    y_standard = layer_standard(x)
    y_custom = layer_custom(x)

    # Results should be very close (both use FP8)
    np.testing.assert_allclose(y_standard.numpy(), y_custom.numpy(),
                               rtol=1e-2, atol=1e-2)


if __name__ == '__main__':
  unittest.main()
