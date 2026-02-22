import unittest
from tinygrad import Tensor, dtypes


class TestTensorAudio(unittest.TestCase):
  def test_hann_window(self):
    win = Tensor.hann_window(4, dtype=dtypes.float32).tolist()
    expected = [0.0, 0.5, 1.0, 0.5]
    for got, exp in zip(win, expected):
      self.assertAlmostEqual(float(got), exp, places=6)

  def test_unfold(self):
    x = Tensor.arange(8)
    y = x.unfold(0, 3, 2)
    self.assertEqual(y.shape, (3, 3))
    self.assertEqual(y.tolist(), [[0, 1, 2], [2, 3, 4], [4, 5, 6]])

  def test_stft_impulse(self):
    x = Tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=dtypes.float32)
    y = x.stft(4, hop_length=4, center=False, return_complex=True)
    self.assertEqual(y.shape, (3, 2, 2))
    mag = (y[:, :, 0].square() + y[:, :, 1].square()).numpy().tolist()
    self.assertEqual(len(mag), 3)
    for row in mag:
      self.assertAlmostEqual(float(row[0]), 1.0, places=5)
      self.assertAlmostEqual(float(row[1]), 0.0, places=5)


if __name__ == "__main__":
  unittest.main()
