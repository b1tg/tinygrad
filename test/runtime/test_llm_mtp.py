import gc, unittest
from itertools import islice
from unittest.mock import patch
import numpy as np
from tinygrad import Tensor, UOp, nn, dtypes, TinyJit
from tinygrad.llm.model import Transformer, TransformerConfig, SSMConfig, GatedDeltaNetBlock, sample_logits


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

def prefill(m, tokens, temperature=0.0):
  gen = m.generate(tokens, temperature=temperature)
  next(gen)
  gen.close()

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
    def restore(i): return [t.numpy().copy() for b in m.blk for t in b._restore_state(Tensor([i], dtype=dtypes.int32))]
    for length in (4, 2):  # a shorter capture must overwrite the valid prefix without reading stale history
      h = m.output_norm(m._hidden(Tensor([tokens[:length]], dtype=dtypes.int32), sp.bind(0), save_state=True))
      np.testing.assert_allclose(h.numpy(), np.concatenate(ref[:length], axis=1), atol=2e-3, rtol=2e-3)
      # The last snapshot must preserve every bit of the live FP32 state, including low mantissa bits.
      last = [state.numpy().copy() for state in states(m)]
      for actual, expected in zip(restore(length-1), last): np.testing.assert_array_equal(actual, expected)
      for accepted in range(length):
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

  def test_greedy_ties(self):
    Tensor.manual_seed(123)
    for m in pair():
      m.output.weight.replace(Tensor.zeros_like(m.output.weight).contiguous().realize())
      # Exercise JIT replay and switching between greedy decoding and sampling with tied logits.
      for temperature in (0.0, 0.0, 0.5, 0.0):
        tokens = list(islice(m.generate([3, 5, 7], temperature=temperature), 10))
        if temperature == 0: self.assertEqual(tokens, [0]*10)
        else: self.assertNotEqual(tokens, [0]*10)

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
    def snapshot(prompt):
      prefill(m, prompt.copy())
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
      resumed = snapshot(extended)
      m._cached_tokens = []
      for actual, expected in zip(snapshot(extended), resumed):
        np.testing.assert_allclose(actual, expected, atol=2e-3, rtol=2e-3)

  def test_acceptance_boundaries(self):
    Tensor.manual_seed(23)
    m, ref = pair()
    for net in (m, ref): net.output.weight.replace(Tensor.zeros_like(net.output.weight).contiguous().realize())
    # Force draft mismatches before verification so both acceptance and state restoration run inside the real round.
    for accepted, delivered in ((0, 1), (1, 1), (1, 2), (2, 1), (2, 2), (2, 3)):
      with self.subTest(accepted=accepted, delivered=delivered):
        verify, output = TinyJit(m._mtp_round), m.output
        def limited_verify(sp, temperature):
          drafts = iter([0]*accepted + [1]*(2-accepted))
          def logits(h):
            if h.shape[1] != 1: return output(h)
            return (Tensor.arange(output.out_features).to(h.device) == next(drafts)).float().reshape(1, 1, -1)
          with patch.object(m, 'output', side_effect=logits): return verify(sp, temperature)
        with patch.object(m, 'mtp_jit', side_effect=limited_verify):
          tokens = [3, 5, 7]
          gen = m.generate(tokens)
          list(islice(gen, 1 + delivered))
          gen.close()
          self.assertEqual(m._pending_restore_index, delivered-1 if delivered <= accepted else None)
          # Resume with one sampled token, restoring the stopped round before another draft round.
          with patch.object(m, 'mtp_jit', side_effect=AssertionError('drafts before consuming the prompt')):
            prefill(m, tokens, temperature=0.5)
          prefill(ref, tokens[:-1])
          for actual, state in zip(states(m), states(ref)):
            np.testing.assert_allclose(actual.numpy(), state.numpy(), atol=2e-3, rtol=2e-3)
          self.assertIsNone(m._pending_restore_index)
          # Continue another verification round from the accepted prefix.
          gen = m.generate(tokens)
          list(islice(gen, accepted+2))
          gen.close()
          prefill(ref, tokens[:-1])
          for actual, state in zip(states(m), states(ref)):
            np.testing.assert_allclose(actual.numpy(), state.numpy(), atol=2e-3, rtol=2e-3)

  def test_new_prompt_and_context_end(self):
    Tensor.manual_seed(23)
    m = model()
    m.output.weight.replace(Tensor.zeros_like(m.output.weight).contiguous().realize())
    tokens = [3, 5, 7]*19+[3, 5]
    gen = m.generate(tokens)
    list(islice(gen, 2))
    gen.close()
    self.assertEqual(m._pending_restore_index, 0)
    # An empty generation must not discard the restore needed by a later continuation.
    self.assertEqual(list(m.generate([3]*64)), [])
    self.assertEqual(m._pending_restore_index, 0)
    with patch.object(m, 'mtp_jit', side_effect=AssertionError('round does not fit')):
      self.assertEqual(len(list(m.generate(tokens+[17, 19]))), 1)
    resumed = [s.numpy().copy() for s in [*states(m), m.mtp.previous]]
    m._cached_tokens = []
    prefill(m, tokens+[17, 19])
    for actual, expected in zip([*states(m), m.mtp.previous], resumed):
      np.testing.assert_allclose(actual.numpy(), expected, atol=2e-3, rtol=2e-3)
    gen = m.generate([3, 5, 7])
    list(islice(gen, 2))
    gen.close()
    self.assertEqual(m._pending_restore_index, 0)
    with patch.object(m, '_mtp_restore', side_effect=AssertionError('restore for unrelated prompt')):
      prefill(m, [11, 13, 17])
    self.assertIsNone(m._pending_restore_index)

  def test_failed_forward_invalidates_cache(self):
    m = model()
    prefill(m, [3, 5, 7])
    with patch.object(Transformer, '__call__', side_effect=ValueError('test failure')):
      with self.assertRaisesRegex(ValueError, 'test failure'): next(m.generate([11, 13, 17]))
    self.assertEqual(m._cached_tokens, [])
    prefill(m, [11, 13, 17])

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

  def test_sampling(self):
    Tensor.manual_seed(42)
    m, ref = pair()
    tokens = [3, 5, 7]
    for temperature in (0.7, 1.2, 0.0):
      gen = m.generate(tokens, temperature=temperature)
      self.assertEqual(len(list(islice(gen, 10))), 10)
      gen.close()
      # Consume the last delivered token and restore any stopped round before comparing recurrent state.
      prefill(m, tokens, temperature)
      prefill(ref, tokens[:-1])
      for actual, state in zip(states(m), states(ref)):
        np.testing.assert_allclose(actual.numpy(), state.numpy(), atol=2e-3, rtol=2e-3)
    self.assertGreater(m.mtp_jit.cnt, 2)

  def test_sampling_distribution(self):
    Tensor.manual_seed(42)
    logits = Tensor([0.0, 1.0, 2.0]).reshape(1, 1, 3).expand(1, 16384, 3).contiguous().realize()
    sample = TinyJit(lambda t: sample_logits(logits, t).realize())
    previous = None
    for temperature in (1.0, 1.0, 0.5, 2.0, 0.0):
      actual = sample(Tensor([temperature])).numpy().flatten()
      if temperature == 0: np.testing.assert_array_equal(actual, 2)
      else:
        p = np.exp(np.array([0.0, 1.0, 2.0])/temperature)
        np.testing.assert_allclose(np.bincount(actual, minlength=3)/len(actual), p/p.sum(), atol=0.015)
        if previous is not None: self.assertFalse(np.array_equal(actual, previous))
      previous = actual

  def test_sample_and_match_distribution(self):
    Tensor.manual_seed(42)
    m = model(recurrent=False)
    p = np.array([0.6, 0.3, 0.1])
    logits = Tensor(np.concatenate((np.log(p), np.full(61, -np.inf))).astype(np.float32)).realize()
    accepted, tokens, verify = [], [], m.mtp_jit
    def record(sp, temperature):
      out = verify(sp, temperature)
      accepted.append(int(out.numpy()[0, 0]))
      return out
    # Greedy drafts always propose token 0; emitted tokens must still follow the target distribution.
    with patch.object(m, 'output', side_effect=lambda h: logits.reshape(1, 1, 64).expand(h.shape[0], h.shape[1], 64)), \
         patch.object(m, 'mtp_jit', side_effect=record):
      for _ in range(8): tokens.extend(islice(m.generate([3, 5, 7], temperature=1.0), 32))
    np.testing.assert_allclose(np.bincount(tokens, minlength=3)/len(tokens), p, atol=0.1)
    self.assertEqual(set(accepted), {0, 1, 2})

  def test_mtp_temperature_replay(self):
    m = model()
    logits = Tensor([0., 1., 2.] + [-float('inf')]*61).realize()
    # Fix the Gumbel noise to [2, 0, 0, ...], so only temperature can change the target's choice.
    uniform = Tensor(np.exp(-np.exp(-np.array([2.] + [0.]*63))).astype(np.float32)).realize()
    with patch.object(m, 'output', side_effect=lambda h: logits.reshape(1, 1, 64).expand(h.shape[0], h.shape[1], 64)), \
         patch.object(Tensor, 'rand_like', side_effect=lambda x: uniform.reshape(1, 1, 64).expand(x.shape)):
      prefill(m, [3, 5, 7])
      # The first two calls warm up/capture; the remaining calls change temperature on the same JIT replay.
      for temperature, token in ((0.25, 2), (4.0, 0), (0.0, 2), (4.0, 0), (0.25, 2)):
        result = m.mtp_jit(UOp.variable('start_pos', 0, 63).bind(3), Tensor([temperature])).numpy()[0]
        np.testing.assert_array_equal(result, [2 if token == 2 else 0, token, token, token])
    self.assertEqual(m.mtp_jit.cnt, 5)

  def test_invalid_mtp_tokens(self):
    for count in (-2, -1, 8):
      with self.subTest(mtp_tokens=count), self.assertRaisesRegex(AssertionError, "MTP needs 1..7 draft tokens"):
        model(mtp_tokens=count)


if __name__ == '__main__': unittest.main()
