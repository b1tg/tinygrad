import unittest
import numpy as np
from tinygrad import Tensor, UOp, nn, dtypes, TinyJit
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
  return m


class TestMTP(unittest.TestCase):
  def test_attention_fallback_causal(self):
    from tinygrad.llm.kernels.amd import flash_attention
    # An unaligned cache takes the fallback path, which must still mask future verification tokens.
    cache = Tensor.arange(65).reshape(1, 1, 65, 1).expand(1, 1, 65, 32)
    cache = Tensor.stack(Tensor.zeros_like(cache), cache).contiguous()
    for tokens in (1, 3):
      q = Tensor.zeros(1, 1, tokens, 32)
      for end in (5, 65):
        expected = np.broadcast_to(np.arange(end-tokens, end).reshape(1, 1, tokens, 1)/2, q.shape)
        np.testing.assert_allclose(flash_attention(q, cache, end).numpy(), expected, atol=1e-5)
        n = UOp.variable('toks', 1, 3).bind(tokens)
        symbolic = Tensor.zeros(1, 1, 3, 32)[:, :, :n]
        out = flash_attention(symbolic, cache, end).pad_to((1, 1, 3, 32))
        np.testing.assert_allclose(out.numpy()[:, :, :tokens], expected, atol=1e-5)

  def test_verify_and_restore(self):
    Tensor.manual_seed(17)
    m = model()
    for b in m.blk:
      if isinstance(b, GatedDeltaNetBlock):
        b._init_state(Tensor.empty(1, 1, 32))
        b.state_history = Tensor.empty(8, *b.recurrent_state.shape).realize()
        b.conv_history = Tensor.empty(8, *b.conv_state.shape).realize()
    sp = UOp.variable('start_pos', 0, 63)
    tokens = [3, 5, 7, 9, 11]
    states = [state for b in m.blk if isinstance(b, GatedDeltaNetBlock) for state in (b.recurrent_state, b.conv_state)]
    ref, snapshots = [], []
    for i,tok in enumerate(tokens):
      out, h = m._mtp_target(Tensor([[tok]], dtype=dtypes.int32), sp.bind(i))
      Tensor.realize(out, h)
      ref.append(h.numpy())
      snapshots.append([state.numpy().copy() for state in states])
    _, h = m._mtp_target(Tensor([tokens[:4]], dtype=dtypes.int32), sp.bind(0), verify=True)
    np.testing.assert_allclose(h.numpy(), np.concatenate(ref[:4], axis=1), atol=2e-3, rtol=2e-3)
    for accepted in range(4):
      restored = m._mtp_restore(Tensor([accepted], dtype=dtypes.int32))
      self.assertEqual(len(restored), len(states))
      for actual, expected in zip(restored, snapshots[accepted]):
        np.testing.assert_allclose(actual.numpy(), expected, atol=2e-3, rtol=2e-3)

  def test_greedy_generation(self):
    Tensor.manual_seed(21)
    m = model()
    from itertools import islice
    expected = list(islice(m.generate([3, 5, 7], mtp=0), 10))
    for count in (1, 2, 7, 1):
      actual = list(islice(m.generate([3, 5, 7], mtp=count), 10))
      self.assertEqual(actual, expected)
      self.assertGreater(m.mtp_stats['rounds'], 0)
      self.assertEqual(list(islice(m.generate([3, 5, 7], mtp=count), 10)), expected)
    for count in (0, 2): self.assertEqual(list(islice(m.generate([3, 5, 7], mtp=count), 10)), expected)

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
    m._cached_tokens = []
    self.assertEqual(list(islice(m.generate(t2.copy(), mtp=2), 8)), out)

  def test_symbolic_prefill(self):
    Tensor.manual_seed(23)
    m = model()
    pending, previous = Tensor.zeros(1, 1, dtype=dtypes.int32).realize(), Tensor.zeros(1, 1, 32).realize()
    tokens = [3, 5, 7, 11, 13, 17]
    sp, nt = UOp.variable('start_pos', 0, 63), UOp.variable('toks', 1, 3)
    for pos, tok in enumerate(tokens): m._mtp_prefill(Tensor([[tok]], dtype=dtypes.int32), sp.bind(pos), pending, previous).realize()
    expected_hidden, expected_cache = previous.numpy(), m.mtp[0].cache_kv.numpy()[:, :, :, :6]
    run = TinyJit(m._mtp_prefill)
    for chunks in ((3, 2, 1), (2, 3, 1)):
      pos = 0
      for n in chunks:
        t = Tensor([tokens[pos:pos+n]], dtype=dtypes.int32).pad_to((1, 3))[:, :nt.bind(n)].contiguous()
        run(t, sp.bind(pos), pending, previous).realize()
        pos += n
      np.testing.assert_allclose(previous.numpy(), expected_hidden, atol=2e-3, rtol=2e-3)
      np.testing.assert_allclose(m.mtp[0].cache_kv.numpy()[:, :, :, :6], expected_cache, atol=2e-3, rtol=2e-3)

  def test_stop_mid_round(self):
    from unittest.mock import patch
    m, tokens = model(), [3, 5, 7]
    # A verifier accepting both drafts advances state past the first token yielded to the consumer.
    with patch.object(m, '_mtp_round', lambda *a, **kw: (Tensor([[11, 13, 17]]), Tensor([2]))):
      gen = m.generate(tokens, mtp=2)
      next(gen)
      self.assertEqual(m.get_start_pos(tokens), 3)
      self.assertEqual(next(gen), 11)
      self.assertEqual(m.get_start_pos(tokens), 0)
      self.assertEqual([next(gen), next(gen)], [13, 17])
      self.assertEqual(m._cached_tokens, tokens[:-1])
      gen.close()

  def test_context_limit(self):
    Tensor.manual_seed(22)
    m = model()
    prompt = [3, 5, 7]*19+[3]
    expected = list(m.generate(prompt.copy(), mtp=0))
    actual = list(m.generate(prompt.copy(), mtp=2))
    self.assertEqual(len(actual), 6)
    self.assertEqual(actual, expected)
    self.assertEqual(list(m.generate([3]*64, mtp=2)), [])
    self.assertEqual(list(m.generate([3]*65, mtp=2)), [])
    with self.assertRaisesRegex(ValueError, 'nonempty'): next(m.generate([], mtp=2))

  def test_unsupported_sampling(self):
    m = model()
    with self.assertRaisesRegex(ValueError, 'greedy'): next(m.generate([1], temperature=0.5, mtp=1))
    for count in (-1, 8):
      with self.assertRaisesRegex(ValueError, 'draft count'): next(m.generate([1], mtp=count))
    m.mtp = []
    with self.assertRaisesRegex(ValueError, 'MTP weights'): next(m.generate([1], mtp=1))


if __name__ == '__main__': unittest.main()
