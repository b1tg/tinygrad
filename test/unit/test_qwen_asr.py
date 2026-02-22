import importlib.util, pathlib, unittest
import math, random
from tinygrad import Tensor, dtypes


def load_qwen_module():
  root = pathlib.Path(__file__).resolve().parents[2]
  mod_path = root / "extra" / "qwen-asr" / "transcribe.py"
  spec = importlib.util.spec_from_file_location("qwen_asr_transcribe", mod_path)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


class TestQwenAsrHelpers(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.m = load_qwen_module()

  def test_bytes_to_unicode_is_bijective(self):
    table = self.m.bytes_to_unicode()
    self.assertEqual(len(table), 256)
    self.assertEqual(len(set(table.values())), 256)

  def test_parse_asr_text(self):
    self.assertEqual(self.m.parse_asr_text("lang en<asr_text>Hello"), "Hello")
    self.assertEqual(self.m.parse_asr_text("language English Hello"), "Hello")
    self.assertEqual(self.m.parse_asr_text("No marker"), "No marker")

  def test_mel_filters_shape(self):
    fb = self.m.compute_mel_filters()
    self.assertEqual(fb.shape, (201, 128))
    self.assertGreater(float(fb.sum().item()), 0.0)
    self.assertFalse(math.isnan(float(fb.min().item())))
    self.assertFalse(math.isnan(float(fb.max().item())))

  def test_mel_spectrogram_shape(self):
    random.seed(0)
    audio = Tensor([random.uniform(-1.0, 1.0) for _ in range(16000)], dtype=dtypes.float32)
    mel = self.m.compute_mel_spectrogram(audio, self.m.compute_mel_filters())
    self.assertEqual(mel.shape[0], 128)
    self.assertGreater(mel.shape[1], 0)
    self.assertFalse(math.isnan(float(mel.min().item())))
    self.assertFalse(math.isnan(float(mel.max().item())))

if __name__ == "__main__":
  unittest.main()
