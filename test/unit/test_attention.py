import unittest
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.llm.model import (
  GatedDeltaNetBlock, KDAConfig, KimiDeltaAttentionBlock, MLATransformerBlock, SSMConfig, Transformer, TransformerBlock, TransformerConfig,
  apply_rope as apply_rope_new, precompute_freqs_cis, pairwise_topk,
)

def apply_rope(x:Tensor, start_pos:int):
  B, H, T, Hd = x.shape
  precompute_freqs_cis.cache_clear()
  freqs_cis = precompute_freqs_cis(Hd, start_pos+T)[start_pos:start_pos+T]
  return apply_rope_new(x, freqs_cis)

class TestAttention(unittest.TestCase):
  def test_apply_rope(self):
    x = Tensor.randn(1, 2, 4, 8, dtype=dtypes.float32)
    result = apply_rope(x, 0)
    self.assertEqual(result.shape, x.shape)
    self.assertEqual(result.dtype, x.dtype)
    self.assertGreater((result - apply_rope(x, 5)).abs().max().item(), 1e-6)
    with self.assertRaises(AssertionError): apply_rope(Tensor.randn(1, 1, 4, 7, dtype=dtypes.float32), 0)

  def test_partial_rope_in_attention(self):
    dim, rope_dim, seqlen = 8, 4, 3
    config = TransformerConfig(num_blocks=1, dim=dim, hidden_dim=16, n_heads=1, n_kv_heads=1,
                               norm_eps=1e-5, vocab_size=32, head_dim=dim, rope_theta=10000.0,
                               rope_dim=rope_dim, v_head_dim=dim, max_context=8)
    block = TransformerBlock(config)

    x = Tensor.randn(1, seqlen, dim, dtype=dtypes.float32)
    x_norm = block.attn_norm(x)
    k = block.attn_k(x_norm).reshape(1, seqlen, 1, dim).transpose(1, 2)

    precompute_freqs_cis.cache_clear()
    block.cache_kv = Tensor.empty(2, 1, 1, config.max_context, max(dim, config.v_head_dim), device=x.device)
    block.freqs_cis = precompute_freqs_cis(rope_dim, config.max_context, config.rope_theta)
    block._attention(x_norm, 0).realize()

    expected = apply_rope_new(k[..., :rope_dim], block.freqs_cis[:seqlen]).cat(k[..., rope_dim:], dim=-1)
    np.testing.assert_allclose(block.cache_kv[0, :, :, :seqlen, :].numpy(), expected.numpy(), rtol=1e-5, atol=1e-5)

class TestGatedDeltaNetBlock(unittest.TestCase):
  def _tensor_linspace(self, start:float, stop:float, shape:tuple[int, ...]) -> Tensor:
    return Tensor.linspace(start, stop, int(np.prod(shape)), dtype=dtypes.float32).reshape(*shape)

  def _make_config(self, **kwargs):
    return TransformerConfig(**({"num_blocks":1, "dim":4, "hidden_dim":8, "n_heads":1, "n_kv_heads":1,
                                 "norm_eps":1e-5, "vocab_size":32, "head_dim":4, "rope_theta":10000.0,
                                 "rope_dim":4, "v_head_dim":4, "max_context":4, "full_attention_interval":2,
                                 "ssm":SSMConfig(conv_kernel=2, state_size=2, group_count=1, time_step_rank=1, inner_size=2)} | kwargs))

  def _make_block(self, config:TransformerConfig) -> GatedDeltaNetBlock:
    block = GatedDeltaNetBlock(config, config.ssm)
    block.attn_norm.weight = self._tensor_linspace(0.8, 1.2, (config.dim,))
    block.attn_qkv.weight = self._tensor_linspace(-0.15, 0.2, (block.conv_channels, config.dim))
    block.attn_gate.weight = self._tensor_linspace(-0.1, 0.15, (config.ssm.inner_size, config.dim))
    block.ssm_alpha.weight = self._tensor_linspace(-0.08, 0.12, (block.num_v_heads, config.dim))
    block.ssm_beta.weight = self._tensor_linspace(-0.12, 0.07, (block.num_v_heads, config.dim))
    block.ssm_conv1d["weight"] = self._tensor_linspace(-0.05, 0.05, (block.conv_channels, block.ssm_conv_kernel))
    block.ssm_dt["bias"] = self._tensor_linspace(-0.1, 0.1, (block.num_v_heads,))
    block.ssm_a = self._tensor_linspace(-0.1, -0.05, (block.num_v_heads,))
    block.ssm_norm.weight = self._tensor_linspace(0.9, 1.1, (block.head_v_dim,))
    block.ssm_out.weight = self._tensor_linspace(-0.2, 0.18, (config.dim, config.ssm.inner_size))
    return block

  def _run_attention(self, block:GatedDeltaNetBlock, x:Tensor, start_pos:int):
    x_norm = block.attn_norm(x)
    block._init_state(x_norm)
    return block._attention(x_norm, start_pos).realize().numpy()

  def _cache_views(self, block:GatedDeltaNetBlock) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(block, 'conv_state'):
      return block.conv_state.numpy(), block.recurrent_state.numpy()
    else:
      conv_flat = (block.ssm_conv_kernel - 1) * block.conv_channels
      cache = block.delta_cache.numpy()
      conv_state = cache[:, :conv_flat].reshape(cache.shape[0], block.ssm_conv_kernel - 1, block.conv_channels)
      recurrent_state = cache[:, conv_flat:].reshape(cache.shape[0], block.num_v_heads, block.head_v_dim, block.head_v_dim)
      return conv_state, recurrent_state

  def _linear_np(self, x:np.ndarray, weight:np.ndarray) -> np.ndarray:
    return x.astype(np.float32) @ weight.T.astype(np.float32)

  def _rms_norm_np(self, x:np.ndarray, weight:np.ndarray, eps:float) -> np.ndarray:
    x_float = x.astype(np.float32)
    return (x_float / np.sqrt((x_float * x_float).mean(axis=-1, keepdims=True) + eps)) * weight.astype(np.float32)

  def _normalize_np(self, x:np.ndarray, eps:float=1e-12) -> np.ndarray:
    return x / np.maximum(np.sqrt((x * x).sum(axis=-1, keepdims=True)), eps)

  def _softplus_np(self, x:np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

  def _silu_np(self, x:np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))

  def _naive_attention(self, block:GatedDeltaNetBlock, x:Tensor):
    x_np = x.numpy().astype(np.float32)
    B, T, _ = x_np.shape
    conv_state = np.zeros((B, block.ssm_conv_kernel - 1, block.conv_channels), dtype=np.float32)
    recurrent_state = np.zeros((B, block.num_v_heads, block.head_v_dim, block.head_v_dim), dtype=np.float32)
    conv_weight = block.ssm_conv1d["weight"].numpy().astype(np.float32).T[None, :, :]
    qkv_weight = block.attn_qkv.weight.numpy().astype(np.float32)
    gate_weight = block.attn_gate.weight.numpy().astype(np.float32)
    alpha_weight = block.ssm_alpha.weight.numpy().astype(np.float32)
    beta_weight = block.ssm_beta.weight.numpy().astype(np.float32)
    out_weight = block.ssm_out.weight.numpy().astype(np.float32)
    dt_bias = block.ssm_dt["bias"].numpy().astype(np.float32)
    ssm_a = block.ssm_a.numpy().astype(np.float32)
    attn_norm_weight = block.attn_norm.weight.numpy().astype(np.float32)
    ssm_norm_weight = block.ssm_norm.weight.numpy().astype(np.float32)
    outputs, conv_states, recurrent_states = [], [], []

    for t in range(T):
      x_norm = self._rms_norm_np(x_np[:, t:t+1, :], attn_norm_weight, block.attn_norm.eps)
      x_half = x_norm.astype(np.float16)
      out_gate = self._linear_np(x_half, gate_weight).reshape(B, 1, block.num_v_heads, block.head_v_dim)
      beta = 1.0 / (1.0 + np.exp(-self._linear_np(x_half, beta_weight))).reshape(B, block.num_v_heads, 1, 1)
      alpha = np.exp((self._softplus_np(self._linear_np(x_half, alpha_weight) + dt_bias)).reshape(B, block.num_v_heads, 1, 1) *
                     ssm_a.reshape(1, block.num_v_heads, 1, 1))
      conv_window = np.concatenate([conv_state, self._linear_np(x_half, qkv_weight)], axis=1)
      conv_out = self._silu_np((conv_window * conv_weight).sum(axis=1))
      q, k, v = np.split(conv_out, [block.q_dim, 2 * block.q_dim], axis=-1)
      q = self._normalize_np(q.reshape(B, block.num_k_heads, block.head_k_dim))
      k = self._normalize_np(k.reshape(B, block.num_k_heads, block.head_k_dim))
      v = v.reshape(B, block.num_v_heads, block.head_v_dim)
      if block.num_v_heads != block.num_k_heads:
        k_repeat = block.num_v_heads // block.num_k_heads
        q = np.repeat(q[:, None, :, :], k_repeat, axis=1).reshape(B, block.num_v_heads, block.head_k_dim)
        k = np.repeat(k[:, None, :, :], k_repeat, axis=1).reshape(B, block.num_v_heads, block.head_k_dim)
      q, k, v = (q * (block.head_k_dim ** -0.5))[..., None], k[..., None], v[..., None]
      recurrent_state = recurrent_state * alpha
      recurrent_state = recurrent_state + np.matmul((v - np.matmul(recurrent_state, k)) * beta, np.swapaxes(k, -1, -2))
      core_attn_out = np.matmul(recurrent_state, q).squeeze(-1).reshape(B, 1, block.num_v_heads, block.head_v_dim)
      core_attn_out = self._rms_norm_np(core_attn_out, ssm_norm_weight, block.ssm_norm.eps)
      out = self._linear_np((core_attn_out * self._silu_np(out_gate)).reshape(B, 1, -1).astype(np.float16), out_weight)
      conv_state = conv_window[:, 1:, :]
      outputs.append(out)
      conv_states.append(conv_state.copy())
      recurrent_states.append(recurrent_state.copy())

    return outputs, conv_states, recurrent_states

  def test_gatedeltanet_reference_and_reset(self):
    config = self._make_config(max_context=3)
    block = self._make_block(config)
    x = Tensor.linspace(-1.0, 1.0, 3 * config.dim, dtype=dtypes.float32).reshape(1, 3, config.dim)

    expected_outs, expected_conv, expected_recurrent = self._naive_attention(block, x)

    for step in range(x.shape[1]):
      out = self._run_attention(block, x[:, step:step+1], step)
      conv_state, recurrent_state = self._cache_views(block)
      np.testing.assert_allclose(out, expected_outs[step], rtol=1e-3, atol=1e-3,
                                 err_msg=f"GatedDeltaNet output mismatch at step {step}")
      np.testing.assert_allclose(conv_state, expected_conv[step], rtol=1e-3, atol=1e-3,
                                 err_msg=f"GatedDeltaNet conv cache mismatch at step {step}")
      np.testing.assert_allclose(recurrent_state, expected_recurrent[step], rtol=1e-3, atol=1e-3,
                                 err_msg=f"GatedDeltaNet recurrent cache mismatch at step {step}")

    warmup = Tensor.linspace(-0.5, 0.5, 2 * config.dim, dtype=dtypes.float32).reshape(1, 2, config.dim)
    prompt = Tensor.linspace(0.75, -0.75, 2 * config.dim, dtype=dtypes.float32).reshape(1, 2, config.dim)

    for i in range(warmup.shape[1]): self._run_attention(block, warmup[:, i:i+1], i)
    Tensor.realize(*block._state_reset_ops())
    expected_outs, expected_conv, expected_recurrent = self._naive_attention(block, prompt)

    for step in range(prompt.shape[1]):
      out = self._run_attention(block, prompt[:, step:step+1], step)
      conv_state, recurrent_state = self._cache_views(block)
      np.testing.assert_allclose(out, expected_outs[step], rtol=1e-3, atol=1e-3,
                                 err_msg=f"GatedDeltaNet reset output mismatch at step {step}")
      np.testing.assert_allclose(conv_state, expected_conv[step], rtol=1e-3, atol=1e-3,
                                 err_msg=f"GatedDeltaNet reset conv cache mismatch at step {step}")
      np.testing.assert_allclose(recurrent_state, expected_recurrent[step], rtol=1e-3, atol=1e-3,
                                 err_msg=f"GatedDeltaNet reset recurrent cache mismatch at step {step}")

class TestKimiDeltaAttentionBlock(unittest.TestCase):
  def _linspace(self, start:float, stop:float, shape:tuple[int, ...]) -> Tensor:
    return Tensor.linspace(start, stop, int(np.prod(shape)), dtype=dtypes.float32).reshape(*shape)

  def _make_block(self) -> KimiDeltaAttentionBlock:
    config = TransformerConfig(num_blocks=1, dim=4, hidden_dim=8, n_heads=2, n_kv_heads=1,
      norm_eps=1e-5, vocab_size=16, head_dim=2, rope_theta=10000.0, rope_dim=1, v_head_dim=2, max_context=4,
      kda=KDAConfig(conv_kernel=2, head_dim=2), recurrent_layers=(True,), use_rope=False)
    block = KimiDeltaAttentionBlock(config, config.kda)
    for i, proj in enumerate((block.attn_q, block.attn_k, block.attn_v)):
      proj.weight = self._linspace(-0.2 + i*0.03, 0.18 + i*0.03, proj.weight.shape)
    for i, conv in enumerate((block.ssm_conv1d_q, block.ssm_conv1d_k, block.ssm_conv1d_v)):
      conv["weight"] = self._linspace(-0.1 + i*0.02, 0.09 + i*0.02, conv["weight"].shape)
    block.ssm_f_a.weight = self._linspace(-0.12, 0.11, block.ssm_f_a.weight.shape)
    block.ssm_f_b.weight = self._linspace(-0.08, 0.1, block.ssm_f_b.weight.shape)
    block.ssm_beta.weight = self._linspace(-0.15, 0.12, block.ssm_beta.weight.shape)
    block.ssm_a = Tensor([-0.3, -0.7], dtype=dtypes.float32).reshape(2, 1)
    block.ssm_dt["bias"] = self._linspace(-0.2, 0.15, block.ssm_dt["bias"].shape)
    block.ssm_g_a.weight = self._linspace(-0.09, 0.13, block.ssm_g_a.weight.shape)
    block.ssm_g_b.weight = self._linspace(-0.11, 0.07, block.ssm_g_b.weight.shape)
    block.ssm_norm.weight = self._linspace(0.8, 1.2, block.ssm_norm.weight.shape)
    block.attn_output.weight = self._linspace(-0.14, 0.16, block.attn_output.weight.shape)
    return block

  @staticmethod
  def _linear(x:np.ndarray, weight:np.ndarray) -> np.ndarray: return x @ weight.T
  @staticmethod
  def _silu(x:np.ndarray) -> np.ndarray: return x / (1.0 + np.exp(-x))
  @staticmethod
  def _sigmoid(x:np.ndarray) -> np.ndarray: return 1.0 / (1.0 + np.exp(-x))
  @staticmethod
  def _softplus(x:np.ndarray) -> np.ndarray: return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

  def test_recurrent_reference_and_reset(self):
    block = self._make_block()
    x = np.linspace(-0.8, 0.9, 12, dtype=np.float32).reshape(1, 3, 4)
    state = np.zeros((1, block.num_heads, block.head_dim, block.head_dim), dtype=np.float32)
    conv_state = np.zeros((1, block.conv_kernel-1, block.conv_channels), dtype=np.float32)
    proj_weights = [p.weight.numpy() for p in (block.attn_q, block.attn_k, block.attn_v)]
    conv_weights = np.concatenate([c["weight"].numpy().squeeze(1) for c in
                                   (block.ssm_conv1d_q, block.ssm_conv1d_k, block.ssm_conv1d_v)], axis=0).T

    block._init_state(Tensor(x[:, :1]))
    for step in range(x.shape[1]):
      xh = x[:, step:step+1].astype(np.float16)
      projected = np.concatenate([self._linear(xh, w) for w in proj_weights], axis=-1)
      conv_window = np.concatenate((conv_state, projected), axis=1)
      conv_out = self._silu((conv_window * conv_weights[None]).sum(axis=1))
      q, k, v = (part.reshape(1, block.num_heads, block.head_dim) for part in
                 np.split(conv_out, [block.inner_size, 2*block.inner_size], axis=-1))
      q = q / np.sqrt((q*q).sum(axis=-1, keepdims=True)) * block.head_dim**-0.5
      k = k / np.sqrt((k*k).sum(axis=-1, keepdims=True))
      g = self._linear(self._linear(xh, block.ssm_f_a.weight.numpy()), block.ssm_f_b.weight.numpy())
      g = self._softplus(g + block.ssm_dt["bias"].numpy()).reshape(1, block.num_heads, block.head_dim)
      g = g * block.ssm_a.numpy().reshape(1, block.num_heads, 1)
      beta = self._sigmoid(self._linear(xh, block.ssm_beta.weight.numpy())).reshape(1, block.num_heads, 1, 1)
      q, k, v = q[..., None], k[..., None], v[..., None]
      state *= np.exp(g)[:, :, None, :]
      state += np.matmul((v - np.matmul(state, k)) * beta, np.swapaxes(k, -1, -2))
      core = np.matmul(state, q).squeeze(-1).reshape(1, 1, block.num_heads, block.head_dim)
      core = core / np.sqrt((core*core).mean(axis=-1, keepdims=True) + block.ssm_norm.eps) * block.ssm_norm.weight.numpy()
      gate = self._linear(self._linear(xh, block.ssm_g_a.weight.numpy()), block.ssm_g_b.weight.numpy())
      expected = self._linear((core * self._sigmoid(gate.reshape(core.shape))).reshape(1, 1, -1), block.attn_output.weight.numpy())
      conv_state = conv_window[:, 1:, :]

      actual = block._attention(Tensor(x[:, step:step+1]), step).realize().numpy()
      np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-3)
      np.testing.assert_allclose(block.conv_state.numpy(), conv_state, rtol=2e-3, atol=2e-3)
      np.testing.assert_allclose(block.recurrent_state.numpy(), state, rtol=2e-3, atol=2e-3)

    Tensor.realize(*block._state_reset_ops())
    np.testing.assert_equal(block.conv_state.numpy(), 0)
    np.testing.assert_equal(block.recurrent_state.numpy(), 0)

  def test_transformer_mixed_layout(self):
    config = TransformerConfig(num_blocks=2, dim=4, hidden_dim=8, n_heads=2, n_kv_heads=1,
      norm_eps=1e-5, vocab_size=16, head_dim=2, rope_theta=10000.0, rope_dim=1, v_head_dim=2, max_context=4,
      kv_lora_rank=2, kda=KDAConfig(conv_kernel=2, head_dim=2), recurrent_layers=(True, False), use_rope=False)
    model = Transformer(config)
    self.assertIsInstance(model.blk[0], KimiDeltaAttentionBlock)
    self.assertIsInstance(model.blk[1], MLATransformerBlock)
    model.blk[1]._init_state(Tensor.zeros(1, 1, config.dim))
    self.assertFalse(hasattr(model.blk[1], "freqs_cis"))

class TestPairwiseTopk(unittest.TestCase):
  def test_basic_topk(self):
    x = Tensor([[[1.0, 3.0, 2.0, 5.0, 4.0]]])
    vals, sel = pairwise_topk(x, 3)
    np.testing.assert_allclose(vals.numpy(), [[[3.0, 4.0, 5.0]]])
    np.testing.assert_equal(sel.numpy(), [[[1, 4, 3]]])

  def test_duplicates(self):
    x = Tensor([[[5.0, 5.0, 3.0, 5.0]]])
    vals, sel = pairwise_topk(x, 2)
    np.testing.assert_allclose(vals.numpy(), [[[5.0, 5.0]]])
    np.testing.assert_equal(sel.numpy(), [[[1, 0]]])

  def test_matches_numpy(self):
    np.random.seed(42)
    data = np.random.randn(4, 2, 16).astype(np.float32)
    vals, sel = pairwise_topk(Tensor(data), 5)
    for b in range(4):
      for t in range(2):
        expected = set(np.argsort(-data[b, t])[:5].tolist())
        self.assertEqual(set(sel.numpy()[b, t].tolist()), expected)
        np.testing.assert_allclose(vals.numpy()[b, t], data[b, t][sel.numpy()[b, t]])

if __name__ == '__main__':
  unittest.main()
