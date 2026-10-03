import gc, unittest
from itertools import islice
from unittest.mock import patch
import numpy as np
from tinygrad import Tensor, UOp, nn, dtypes
from tinygrad.llm.model import Transformer, TransformerConfig, SSMConfig, GatedDeltaNetBlock


def model(recurrent=True, mtp_tokens=2):
  cfg = TransformerConfig(num_blocks=2, dim=32, hidden_dim=64, n_heads=1, n_kv_heads=1, norm_eps=1e-6,
    vocab_size=64, head_dim=128, rope_theta=10000, rope_dim=16, v_head_dim=128, max_context=64,
    ssm=SSMConfig(4, 32, 1, 1, 32) if recurrent else None, ssm_layers=(True, False) if recurrent else (),
    qk_norm=128, attn_output_gate=True, mtp_tokens=mtp_tokens)
  m = Transformer(cfg)
  for b in m.blk:
    if isinstance(b, GatedDeltaNetBlock):
      b.ssm_conv1d['weight'] = Tensor.randn(*b.ssm_conv1d['weight'].shape)*0.2
      b.ssm_a = -Tensor.ones(*b.ssm_a.shape)
  for p in nn.state.get_parameters(m): p.replace(p.contiguous())
  Tensor.realize(*nn.state.get_parameters(m))
  return m

def pair(recurrent=True, mtp_tokens=2):
  # an MTP model and a target-only reference with the same weights
  m, ref = model(recurrent, mtp_tokens), model(recurrent, 0)
  nn.state.load_state_dict(ref, nn.state.get_state_dict(m), verbose=False)
  return m, ref

def states(m): return [s for b in m.blk if isinstance(b, GatedDeltaNetBlock) for s in (b.recurrent_state, b.conv_state)]


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
    m = model(mtp_tokens=3)
    sp = UOp.variable('start_pos', 0, 63)
    tokens = [3, 5, 7, 9, 11]
    ref, snapshots = [], []
    for i,tok in enumerate(tokens):
      h = m.output_norm(m._hidden(Tensor([[tok]], dtype=dtypes.int32), sp.bind(i)))
      ref.append(h.numpy())
      snapshots.append([state.numpy().copy() for state in states(m)])
    h = m.output_norm(m._hidden(Tensor([tokens[:4]], dtype=dtypes.int32), sp.bind(0), save_state=True))
    np.testing.assert_allclose(h.numpy(), np.concatenate(ref[:4], axis=1), atol=2e-3, rtol=2e-3)
    def restore(i): return [t.numpy().copy() for b in m.blk for t in b._restore_state(Tensor([i], dtype=dtypes.int32))]
    # The last snapshot must preserve every bit of the live FP32 state, including low mantissa bits.
    last = [state.numpy().copy() for state in states(m)]
    for actual, expected in zip(restore(3), last): np.testing.assert_array_equal(actual, expected)
    for accepted in range(4):
      for actual, expected in zip(restore(accepted), snapshots[accepted]):
        np.testing.assert_allclose(actual, expected, atol=2e-3, rtol=2e-3)

  def test_greedy_generation(self):
    Tensor.manual_seed(21)
    for count in (1, 2, 7):
      m, ref = pair(mtp_tokens=count)
      expected = list(islice(ref.generate([3, 5, 7]), 10))
      self.assertEqual(list(islice(m.generate([3, 5, 7]), 10)), expected)
      self.assertGreater(m.mtp_jit.cnt, 0)
      self.assertEqual(list(islice(m.generate([3, 5, 7]), 10)), expected)

  def test_attention_only_target(self):
    Tensor.manual_seed(23)
    m, ref = pair(recurrent=False)
    self.assertFalse(m.has_recurrent_block)
    prompt = [3, 5, 7]
    expected = list(islice(ref.generate(prompt.copy()), 8))
    tokens = prompt.copy()
    gen = m.generate(tokens)
    self.assertEqual(list(islice(gen, 8)), expected)
    gen.close()
    self.assertEqual(m.get_start_pos(tokens+[17]), len(tokens)-1)
    # The draft's previous hidden belongs to the full prefix, even for a pure KV-cache target.
    self.assertEqual(m.get_start_pos(prompt), 0)
    extended = tokens+[17, 19]
    resumed = list(islice(m.generate(extended.copy()), 8))
    m._cached_tokens = []
    self.assertEqual(list(islice(m.generate(extended.copy()), 8)), resumed)

  def test_stop_mid_round(self):
    Tensor.manual_seed(23)
    m = model()
    # Make every draft accepted while retaining nontrivial recurrent and convolution states.
    m.output.weight.replace(Tensor.zeros_like(m.output.weight).contiguous().realize())
    def prefill(prompt):
      gen = m.generate(prompt.copy())
      next(gen)
      gen.close()
      return [s.numpy().copy() for s in [*states(m), m.mtp.previous]]
    for offset, length in enumerate((2, 3, 4, 2)):
      tokens = [3, 5, 7+offset]
      gen = m.generate(tokens)
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

  def test_from_gguf(self):
    kv = {'general.architecture':'qwen35', 'tokenizer.ggml.tokens':['']*64}
    kv.update({f'qwen35.{k}':v for k,v in {
      'block_count':3, 'nextn_predict_layers':1, 'context_length':64, 'embedding_length':32, 'feed_forward_length':64,
      'attention.head_count':1, 'attention.head_count_kv':1, 'attention.key_length':128, 'attention.value_length':128,
      'attention.layer_norm_rms_epsilon':1e-6, 'rope.freq_base':10000, 'rope.dimension_count':16, 'full_attention_interval':2,
      'ssm.conv_kernel':4, 'ssm.state_size':32, 'ssm.group_count':1, 'ssm.time_step_rank':1, 'ssm.inner_size':32}.items()})
    weights = {k.replace('mtp.', 'blk.2.'):v for k,v in nn.state.get_state_dict(model()).items()}
    for mtp_tokens in (0, 2):
      with patch('tinygrad.llm.model.gguf_load', return_value=(kv, dict(weights))):
        self.assertEqual(Transformer.from_gguf('test.gguf', mtp_tokens=mtp_tokens)[0].mtp is not None, bool(mtp_tokens))
    separate = {**weights, 'blk.2.nextn.embed_tokens.weight':Tensor.zeros(64, 32)}
    with patch('tinygrad.llm.model.gguf_load', return_value=(kv, separate)):
      with self.assertRaisesRegex(AssertionError, 'separate MTP'): Transformer.from_gguf('test.gguf', mtp_tokens=2)

  def test_context_limit(self):
    Tensor.manual_seed(22)
    m, ref = pair()
    prompt = [3, 5, 7]*19+[3]
    expected = list(ref.generate(prompt.copy()))
    self.assertEqual(len(expected), 6)
    self.assertEqual(list(m.generate(prompt.copy())), expected)
    self.assertEqual(list(m.generate([3]*64)), [])
    # every draft accepted: the last round is cut at the context end
    m = model()
    m.output.weight.replace(Tensor.zeros_like(m.output.weight).contiguous().realize())
    self.assertEqual(len(list(m.generate(prompt.copy()))), 6)

  def test_sampling_skips_drafts(self):
    m = model()
    self.assertEqual(len(list(islice(m.generate([1], temperature=0.5), 4))), 4)
    self.assertEqual(m.mtp_jit.cnt, 0)
    with self.assertRaises(AssertionError): model(mtp_tokens=8)


if __name__ == '__main__': unittest.main()
