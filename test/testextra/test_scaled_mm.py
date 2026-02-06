#!/usr/bin/env python
import unittest
import numpy as np
from tinygrad import Tensor, dtypes, _scaled_mm, Device
from tinygrad.device import is_dtype_supported

TEST_DEVICE = Device.DEFAULT
CPU_LIKE = TEST_DEVICE.split(":")[0] in {"CPU", "PYTHON", "NPY", "DISK", "TINYFS"}
print(TEST_DEVICE, CPU_LIKE)

@unittest.skipUnless(is_dtype_supported(dtypes.fp8e4m3) and is_dtype_supported(dtypes.fp8e5m2), f"no fp8 on {TEST_DEVICE}")
class TestScaledMM(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    # Some environments expose CUDA in capability checks but cannot initialize the driver.
    try:
      Tensor([1.0], device=TEST_DEVICE, dtype=dtypes.float32).realize()
    except Exception as e:
      raise unittest.SkipTest(f"{TEST_DEVICE} backend not runnable: {e}")

  def setUp(self):
    Tensor.manual_seed(0)

  def _fp8_tensor(self, shape, dtype):
    x = Tensor.randn(*shape, device=TEST_DEVICE, dtype=dtypes.float32) * 0.5
    return x.cast(dtype)

  def _ref(self, a, b, scale_a, scale_b, bias=None, out_dtype=None):
    acc = a.cast(dtypes.float32).matmul(b.cast(dtypes.float32), dtype=dtypes.float32)
    scale = scale_a.cast(dtypes.float32) * scale_b.cast(dtypes.float32)
    out = acc * scale
    if bias is not None: out = out + bias
    return out.cast(out_dtype if out_dtype is not None else a.dtype)

  def test_tensorwise_same_dtype(self):
    a = self._fp8_tensor((4, 6), dtypes.fp8e4m3)
    b = self._fp8_tensor((6, 5), dtypes.fp8e4m3)
    scale_a = Tensor([0.5], device=TEST_DEVICE, dtype=dtypes.float32)
    scale_b = Tensor([2.0], device=TEST_DEVICE, dtype=dtypes.float32)
    out = _scaled_mm(a, b, scale_a, scale_b)
    ref = self._ref(a, b, scale_a, scale_b)
    np.testing.assert_allclose(out.numpy(), ref.numpy(), rtol=0, atol=0)
    self.assertEqual(out.dtype, a.dtype)

  def test_tensorwise_mixed_dtype_default_out(self):
    a = self._fp8_tensor((3, 4), dtypes.fp8e4m3)
    b = self._fp8_tensor((4, 2), dtypes.fp8e5m2)
    scale_a = Tensor([1.25], device=TEST_DEVICE, dtype=dtypes.float32)
    scale_b = Tensor([0.75], device=TEST_DEVICE, dtype=dtypes.float32)
    out = _scaled_mm(a, b, scale_a, scale_b)
    ref = self._ref(a, b, scale_a, scale_b)
    np.testing.assert_allclose(out.numpy(), ref.numpy(), rtol=0, atol=0)
    self.assertEqual(out.dtype, a.dtype)

  def test_out_dtype(self):
    a = self._fp8_tensor((5, 7), dtypes.fp8e4m3)
    b = self._fp8_tensor((7, 3), dtypes.fp8e4m3)
    scale_a = Tensor([1.0], device=TEST_DEVICE, dtype=dtypes.float32)
    scale_b = Tensor([1.0], device=TEST_DEVICE, dtype=dtypes.float32)
    for out_dtype in (dtypes.float16, dtypes.float32, dtypes.int32, dtypes.fp8e5m2):
      out = _scaled_mm(a, b, scale_a, scale_b, out_dtype=out_dtype)
      ref = self._ref(a, b, scale_a, scale_b, out_dtype=out_dtype)
      np.testing.assert_allclose(out.numpy(), ref.numpy(), rtol=0, atol=0)
      self.assertEqual(out.dtype, out_dtype)

  def test_bias(self):
    a = self._fp8_tensor((4, 4), dtypes.fp8e4m3)
    b = self._fp8_tensor((4, 4), dtypes.fp8e4m3)
    scale_a = Tensor([1.0], device=TEST_DEVICE, dtype=dtypes.float32)
    scale_b = Tensor([1.0], device=TEST_DEVICE, dtype=dtypes.float32)
    bias = Tensor.randn(4, device=TEST_DEVICE, dtype=dtypes.float32)
    out = _scaled_mm(a, b, scale_a, scale_b, bias=bias, out_dtype=dtypes.float32)
    ref = self._ref(a, b, scale_a, scale_b, bias=bias, out_dtype=dtypes.float32)
    np.testing.assert_allclose(out.numpy(), ref.numpy(), rtol=0, atol=0)

  def test_scale_result_ignored(self):
    a = self._fp8_tensor((2, 3), dtypes.fp8e4m3)
    b = self._fp8_tensor((3, 2), dtypes.fp8e4m3)
    scale_a = Tensor([0.5], device=TEST_DEVICE, dtype=dtypes.float32)
    scale_b = Tensor([2.0], device=TEST_DEVICE, dtype=dtypes.float32)
    scale_result = Tensor([4.0], device=TEST_DEVICE, dtype=dtypes.float32)
    out = _scaled_mm(a, b, scale_a, scale_b, scale_result=scale_result, out_dtype=dtypes.float32)
    ref = self._ref(a, b, scale_a, scale_b, out_dtype=dtypes.float32)
    np.testing.assert_allclose(out.numpy(), ref.numpy(), rtol=0, atol=0)

  def test_scale_result_validation_cpu_like(self):
    if not CPU_LIKE: self.skipTest("CPU-like only")
    a = self._fp8_tensor((4, 4), dtypes.fp8e4m3)
    b = self._fp8_tensor((4, 4), dtypes.fp8e4m3)
    scale = Tensor([1.0], device=TEST_DEVICE, dtype=dtypes.float32)
    bad_shape = Tensor([1.0, 2.0], device=TEST_DEVICE, dtype=dtypes.float32)
    bad_dtype = Tensor([1], device=TEST_DEVICE, dtype=dtypes.int32)
    with self.assertRaises(RuntimeError):
      _scaled_mm(a, b, scale, scale, scale_result=bad_shape)
    with self.assertRaises(RuntimeError):
      _scaled_mm(a, b, scale, scale, scale_result=bad_dtype)

  def test_scaling_mode_by_backend(self):
    a = self._fp8_tensor((4, 4), dtypes.fp8e4m3)
    b = self._fp8_tensor((4, 4), dtypes.fp8e4m3)
    scale_a = Tensor.ones(4, 1, device=TEST_DEVICE, dtype=dtypes.float32)
    scale_b = Tensor.ones(1, 4, device=TEST_DEVICE, dtype=dtypes.float32)
    if CPU_LIKE:
      with self.assertRaises(RuntimeError):
        _scaled_mm(a, b, scale_a, scale_b)
    else:
      out = _scaled_mm(a, b, scale_a, scale_b, out_dtype=dtypes.float32)
      ref = self._ref(a, b, scale_a, scale_b, out_dtype=dtypes.float32)
      np.testing.assert_allclose(out.numpy(), ref.numpy(), rtol=0, atol=0)

  def test_shape_error(self):
    a = self._fp8_tensor((4, 5), dtypes.fp8e4m3)
    b = self._fp8_tensor((4, 5), dtypes.fp8e4m3)
    scale = Tensor([1.0], device=TEST_DEVICE, dtype=dtypes.float32)
    with self.assertRaises(RuntimeError):
      _scaled_mm(a, b, scale, scale)

  def test_dtype_error(self):
    a = Tensor.randn(4, 4, device=TEST_DEVICE, dtype=dtypes.float16)
    b = self._fp8_tensor((4, 4), dtypes.fp8e4m3)
    scale = Tensor([1.0], device=TEST_DEVICE, dtype=dtypes.float32)
    with self.assertRaises(RuntimeError):
      _scaled_mm(a, b, scale, scale)

if __name__ == "__main__":
  unittest.main()
