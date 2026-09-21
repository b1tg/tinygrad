import unittest
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.quantize import pack_fp4, unpack_fp4, quantize_mxfp4, dequant_mxfp4, quantize_nvfp4, dequant_nvfp4, MXFP4_BLOCK, NVFP4_BLOCK

class TestPackFp4(unittest.TestCase):
  def test_layout(self):
    # low nibble is the even element
    vals = [1.0, 2.0, 3.0, 4.0]  # nibbles 2, 4, 5, 6
    q = pack_fp4(Tensor(vals, dtype=dtypes.fp4e2m1))
    self.assertEqual(q.dtype, dtypes.uint8)
    np.testing.assert_equal(q.numpy(), [2 | (4 << 4), 5 | (6 << 4)])

  def test_roundtrip_all_patterns(self):
    q = Tensor(list(range(16)), dtype=dtypes.uint8)
    vals = unpack_fp4(q)
    self.assertEqual(vals.dtype, dtypes.fp4e2m1)
    np.testing.assert_equal(pack_fp4(vals).numpy(), list(range(16)))

  def test_odd_last_dim(self):
    with self.assertRaises(ValueError): pack_fp4(Tensor([1.0, 2.0, 3.0], dtype=dtypes.fp4e2m1))

class TestMXFP4(unittest.TestCase):
  def test_exact_block(self):
    # amax == 6.0 -> E8M0 exponent 0 (scale 1.0), so representable values survive exactly
    vals = [6.0, 4.0, 3.0, 2.0, 1.5, 1.0, 0.5, 0.0, -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, -0.0]
    x = Tensor(vals * 2, dtype=dtypes.float32).reshape(1, MXFP4_BLOCK)
    q, scale = quantize_mxfp4(x)
    self.assertEqual((q.dtype, scale.dtype), (dtypes.uint8, dtypes.uint8))
    self.assertEqual((q.shape, scale.shape), ((1, MXFP4_BLOCK // 2), (1, 1)))
    np.testing.assert_equal(scale.numpy().reshape(-1), [127])
    np.testing.assert_allclose(dequant_mxfp4(q, scale).numpy(), x.numpy(), atol=0)

  def test_power_of_two_scale(self):
    # amax == 2.0 -> exponent -1 -> E8M0 126, scale 0.5; values scale by 2 into fp4 range
    x = Tensor([[2.0, 1.0, 0.5, 0.0] * (MXFP4_BLOCK // 4)], dtype=dtypes.float32)
    q, scale = quantize_mxfp4(x)
    np.testing.assert_equal(scale.numpy().reshape(-1), [126])
    np.testing.assert_allclose(dequant_mxfp4(q, scale).numpy(), x.numpy(), atol=0)

  def test_zero_block(self):
    q, scale = quantize_mxfp4(Tensor.zeros(2, MXFP4_BLOCK))
    np.testing.assert_equal(scale.numpy().reshape(-1), [127, 127])
    np.testing.assert_equal(dequant_mxfp4(q, scale).numpy(), np.zeros((2, MXFP4_BLOCK)))

  def test_error_bound(self):
    np.random.seed(0)
    x = Tensor((np.random.randn(8, 4 * MXFP4_BLOCK) * 3).astype(np.float32))
    dq = dequant_mxfp4(*quantize_mxfp4(x))
    self.assertEqual(dq.shape, x.shape)
    # every value is within half a fp4 ulp at its block scale
    self.assertLess((x - dq).abs().max().item(), 1.5)

class TestNVFP4(unittest.TestCase):
  def test_roundtrip(self):
    np.random.seed(1)
    x = Tensor((np.random.randn(4, 4 * NVFP4_BLOCK) * 2).astype(np.float32))
    q, block_scale, global_scale = quantize_nvfp4(x)
    self.assertEqual((q.dtype, block_scale.dtype), (dtypes.uint8, dtypes.uint8))
    self.assertEqual(q.shape, (4, 2 * NVFP4_BLOCK))
    self.assertEqual(block_scale.shape, (4, 4))
    dq = dequant_nvfp4(q, block_scale, global_scale)
    self.assertEqual(dq.shape, x.shape)
    self.assertLess((x - dq).abs().max().item(), 0.9)

if __name__ == '__main__':
  unittest.main()
