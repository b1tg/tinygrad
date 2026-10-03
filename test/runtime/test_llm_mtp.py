import gc, unittest
from itertools import islice
from unittest.mock import patch
import numpy as np
from tinygrad import Tensor, UOp, nn, dtypes
from tinygrad.llm.model import Transformer, TransformerConfig, SSMConfig, GatedDeltaNetBlock


def model(recurrent=True):
  cfg = TransformerConfig(num_blocks=2, dim=32, hidden_dim=64, n_heads=1, n_kv_heads=1, norm_eps=1e-6,
    vocab_size=64, head_dim=128, rope_theta=10000, rope_dim=16, v_head_dim=128, max_context=64,
    ssm=SSMConfig(4, 32, 1, 1, 32) if recurrent else None, ssm_layers=(True, False) if recurrent else (),
    qk_norm=128, attn_output_gate=True, mtp_layers=1)
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
        b._prepare_history(Tensor.empty(1, 1, 32), 8)
        self.assertEqual(b.state_history.dtype, b.recurrent_state.dtype)
        self.assertEqual(b.conv_history.dtype, b.conv_state.dtype)
    sp = UOp.variable('start_pos', 0, 63)
    tokens = [3, 5, 7, 9, 11]
    states = [state for b in m.blk if isinstance(b, GatedDeltaNetBlock) for state in (b.recurrent_state, b.conv_state)]
    ref, snapshots = [], []
    for i,tok in enumerate(tokens):
      h = m.output_norm(m._hidden(Tensor([[tok]], dtype=dtypes.int32), sp.bind(i)))
      ref.append(h.numpy())
      snapshots.append([state.numpy().copy() for state in states])
    h = m.output_norm(m._hidden(Tensor([tokens[:4]], dtype=dtypes.int32), sp.bind(0), save_state=True))
    np.testing.assert_allclose(h.numpy(), np.concatenate(ref[:4], axis=1), atol=2e-3, rtol=2e-3)
    # The last snapshot must preserve every bit of the live FP32 state, including low mantissa bits.
    last = [state.numpy().copy() for state in states]
    for state, store, expected in zip(states, m._mtp_restore(Tensor([3], dtype=dtypes.int32)), last):
      np.testing.assert_array_equal(Tensor(state.uop.after(store)).numpy(), expected)
    for accepted in range(4):
      restored = m._mtp_restore(Tensor([accepted], dtype=dtypes.int32))
      self.assertEqual(len(restored), len(states))
      for state, store, expected in zip(states, restored, snapshots[accepted]):
        np.testing.assert_allclose(Tensor(state.uop.after(store)).numpy(), expected, atol=2e-3, rtol=2e-3)

  def test_greedy_generation(self):
    Tensor.manual_seed(21)
    m = model()
    expected = list(islice(m.generate([3, 5, 7], mtp=0), 10))
    for count in (1, 2, 7, 1):
      with patch.object(m, '_mtp_round', wraps=m._mtp_round) as run:
        actual = list(islice(m.generate([3, 5, 7], mtp=count), 10))
        self.assertTrue(run.called)
      self.assertEqual(actual, expected)
      self.assertEqual(list(islice(m.generate([3, 5, 7], mtp=count), 10)), expected)
    for count in (0, 2): self.assertEqual(list(islice(m.generate([3, 5, 7], mtp=count), 10)), expected)

  def test_attention_only_target(self):
    Tensor.manual_seed(23)
    m = model(recurrent=False)
    self.assertFalse(m.has_recurrent_block)
    prompt = [3, 5, 7]
    expected = list(islice(m.generate(prompt.copy(), mtp=0), 8))
    tokens = prompt.copy()
    gen = m.generate(tokens, mtp=2)
    self.assertEqual(list(islice(gen, 8)), expected)
    gen.close()
    self.assertEqual(m.get_start_pos(tokens+[17]), len(tokens)-1)
    # The draft's previous hidden belongs to the full prefix, even for a pure KV-cache target.
    self.assertEqual(m.get_start_pos(prompt), 0)
    extended = tokens+[17, 19]
    resumed = list(islice(m.generate(extended.copy(), mtp=2), 8))
    m._cached_tokens = []
    self.assertEqual(list(islice(m.generate(extended.copy(), mtp=2), 8)), resumed)

  def test_stop_mid_round(self):
    Tensor.manual_seed(23)
    m = model()
    # Make every draft accepted while retaining nontrivial recurrent and convolution states.
    m.output.weight.replace(Tensor.zeros_like(m.output.weight).contiguous().realize())
    def prefill(prompt):
      gen = m.generate(prompt.copy(), mtp=2)
      next(gen)
      gen.close()
      return [s.numpy().copy() for b in m.blk if isinstance(b, GatedDeltaNetBlock) for s in (b.recurrent_state, b.conv_state)] \
        + [m._mtp_inputs[1].numpy().copy()]
    for offset, length in enumerate((2, 3, 4, 2)):
      tokens = [3, 5, 7+offset]
      gen = m.generate(tokens, mtp=2)
      list(islice(gen, length))
      # Explicit close and garbage collection must never submit or synchronize GPU work.
      with patch.object(Tensor, 'realize', side_effect=AssertionError('GPU work during finalization')), \
           patch.object(Tensor, 'item', side_effect=AssertionError('GPU sync during finalization')):
        if offset % 2:
          del gen
          gc.collect()
        else: gen.close()
      extended = tokens+[17, 19]
      resumed = prefill(extended)
      m._cached_tokens = []
      for actual, expected in zip(prefill(extended), resumed):
        np.testing.assert_allclose(actual, expected, atol=2e-3, rtol=2e-3)

  def test_reject_separate_mtp_weights(self):
    kv = {'general.architecture':'qwen35', 'tokenizer.ggml.tokens':['']*64}
    kv.update({f'qwen35.{k}':v for k,v in {
      'block_count':3, 'nextn_predict_layers':1, 'context_length':64, 'embedding_length':32, 'feed_forward_length':64,
      'attention.head_count':1, 'attention.head_count_kv':1, 'attention.key_length':128, 'attention.value_length':128,
      'attention.layer_norm_rms_epsilon':1e-6, 'rope.freq_base':10000, 'rope.dimension_count':16, 'full_attention_interval':2,
      'ssm.conv_kernel':4, 'ssm.state_size':32, 'ssm.group_count':1, 'ssm.time_step_rank':1, 'ssm.inner_size':32}.items()})
    weights = {k.replace('mtp.0.', 'blk.2.'):v for k,v in nn.state.get_state_dict(model()).items()}
    with patch('tinygrad.llm.model.gguf_load', return_value=(kv, weights)):
      loaded, _ = Transformer.from_gguf('test.gguf')
      self.assertEqual(len(loaded.mtp), 1)
    for name in ('embed_tokens', 'shared_head_head'):
      with self.subTest(name=name):
        separate = {**weights, f'blk.2.nextn.{name}.weight':Tensor.zeros(64, 32)}
        with patch('tinygrad.llm.model.gguf_load', return_value=(kv, separate)):
          with self.assertRaisesRegex(AssertionError, f'separate MTP {name}'): Transformer.from_gguf('test.gguf')

  def test_context_limit(self):
    Tensor.manual_seed(22)
    m = model()
    prompt = [3, 5, 7]*19+[3]
    expected = list(m.generate(prompt.copy(), mtp=0))
    actual = list(m.generate(prompt.copy(), mtp=2))
    self.assertEqual(len(actual), 6)
    self.assertEqual(actual, expected)
    self.assertEqual(set(m.mtp_round_jit), {2})  # no new verification graphs at the context tail
    self.assertEqual(list(m.generate([3]*64, mtp=2)), [])
    self.assertEqual(list(m.generate([3]*65, mtp=2)), [])
    with self.assertRaisesRegex(ValueError, 'nonempty'): next(m.generate([], mtp=2))
    m.output.weight.replace(Tensor.zeros_like(m.output.weight).contiguous().realize())
    m.mtp_round_jit.clear()
    self.assertEqual(len(list(m.generate(prompt.copy(), mtp=2))), 6)

  def test_unsupported_sampling(self):
    m = model()
    with self.assertRaisesRegex(ValueError, 'greedy'): next(m.generate([1], temperature=0.5, mtp=1))
    for count in (-1, 8):
      with self.assertRaisesRegex(ValueError, 'draft count'): next(m.generate([1], mtp=count))
    m.mtp = []
    with self.assertRaisesRegex(ValueError, 'MTP weights'): next(m.generate([1], mtp=1))


if __name__ == '__main__': unittest.main()
