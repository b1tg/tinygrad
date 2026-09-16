import unittest
import numpy as np
from tinygrad import Tensor, UOp, nn, dtypes
from tinygrad.llm.model import Transformer, TransformerConfig, SSMConfig, GatedDeltaNetBlock


def model():
  cfg = TransformerConfig(num_blocks=2, dim=32, hidden_dim=64, n_heads=1, n_kv_heads=1, norm_eps=1e-6,
    vocab_size=64, head_dim=128, rope_theta=10000, rope_dim=16, v_head_dim=128, max_context=64,
    ssm=SSMConfig(4, 32, 1, 1, 32), ssm_layers=(True, False), attn_output_gate=True, mtp_layers=1)
  m = Transformer(cfg)
  for b in m.blk:
    if isinstance(b, GatedDeltaNetBlock):
      b.ssm_conv1d['weight'] = Tensor.randn(*b.ssm_conv1d['weight'].shape)*0.2
      b.ssm_a = -Tensor.ones(*b.ssm_a.shape)
  for p in nn.state.get_parameters(m): p.replace(p.contiguous())
  Tensor.realize(*nn.state.get_parameters(m))
  for b in m.blk:
    if isinstance(b, GatedDeltaNetBlock):
      b._init_state(Tensor.empty(1, 1, 32))
      b.state_history = Tensor.empty(8, *b.recurrent_state.shape).realize()
      b.conv_history = Tensor.empty(8, *b.conv_state.shape).realize()
  return m


class TestMTP(unittest.TestCase):
  def test_verify_and_restore(self):
    Tensor.manual_seed(17)
    m = model()
    sp = UOp.variable('start_pos', 0, 63)
    tokens = [3, 5, 7, 9, 11]
    ref = []
    for i,tok in enumerate(tokens):
      out, h = m._mtp_target(Tensor([[tok]], dtype=dtypes.int32), sp.bind(i))
      Tensor.realize(out, h)
      ref.append(h.numpy())
    _, h = m._mtp_target(Tensor([tokens[:4]], dtype=dtypes.int32), sp.bind(0), verify=True)
    np.testing.assert_allclose(h.numpy(), np.concatenate(ref[:4], axis=1), atol=2e-3, rtol=2e-3)
    for accepted in range(4):
      Tensor.realize(*m._mtp_restore(Tensor([accepted], dtype=dtypes.int32)))
      _, h = m._mtp_target(Tensor([[tokens[accepted+1]]], dtype=dtypes.int32), sp.bind(accepted+1))
      np.testing.assert_allclose(h.numpy(), ref[accepted+1], atol=2e-3, rtol=2e-3)

  def test_greedy_generation(self):
    Tensor.manual_seed(21)
    m = model()
    from itertools import islice
    expected = list(islice(m.generate([3, 5, 7], mtp=0), 10))
    for count in (1, 2, 7):
      actual = list(islice(m.generate([3, 5, 7], mtp=count), 10))
      self.assertEqual(actual, expected)
      self.assertGreater(m.mtp_stats['rounds'], 0)
      self.assertEqual(list(islice(m.generate([3, 5, 7], mtp=count), 10)), expected)

  def test_prefix_reuse(self):
    from itertools import islice
    Tensor.manual_seed(23)
    m = model()
    t = [3, 5, 7, 11, 13]
    list(islice(m.generate(t, mtp=2), 8))          # t extended in place with the generated tokens
    t2 = t + [17, 19]
    # the next turn must resume from the cached prefix instead of re-prefilling the whole prompt
    prefix = m.get_start_pos(t2)
    self.assertGreater(prefix, 0)
    self.assertEqual(t2[:prefix], m._cached_tokens)
    out = list(islice(m.generate(t2.copy(), mtp=2), 8))
    self.assertEqual(len(out), 8)
    # the resumed turn extends the cache again (ready for the next turn to hit)
    self.assertGreater(len(m._cached_tokens), prefix)

  def test_context_limit(self):
    Tensor.manual_seed(22)
    m = model()
    prompt = [3, 5, 7]*19+[3]
    expected = list(m.generate(prompt.copy(), mtp=0))
    actual = list(m.generate(prompt.copy(), mtp=2))
    self.assertEqual(len(actual), 6)
    self.assertEqual(actual, expected)
    self.assertEqual(list(m.generate([3]*64, mtp=2)), [])
    with self.assertRaisesRegex(ValueError, 'nonempty'): next(m.generate([], mtp=2))

  def test_unsupported_sampling(self):
    m = model()
    with self.assertRaisesRegex(ValueError, 'greedy'): next(m.generate([1], temperature=0.5, mtp=1))
    for count in (-1, 8):
      with self.assertRaisesRegex(ValueError, 'draft count'): next(m.generate([1], mtp=count))
    m.mtp = []
    with self.assertRaisesRegex(ValueError, 'MTP weights'): next(m.generate([1], mtp=1))


if __name__ == '__main__': unittest.main()
