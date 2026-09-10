import unittest
import numpy as np
from tinygrad import Tensor, nn
from tinygrad.llm.model import Transformer, TransformerConfig, apply_rope, MLATransformerBlock, precompute_freqs_cis, _shard_policy, YaRNConfig

class TestMLA(unittest.TestCase):
  def _make_config(self, **kwargs):
    return TransformerConfig(**{
      "num_blocks": 1, "dim": 64, "hidden_dim": 128, "n_heads": 4, "n_kv_heads": 1,
      "norm_eps": 1e-5, "vocab_size": 100, "head_dim": 16, "rope_theta": 10000.0, "rope_dim": 8, "max_context": 32,
      "kv_lora_rank": 16, "v_head_dim": 8,
    } | kwargs)

  def test_kimi_yarn(self):
    import math
    yarn = YaRNConfig(factor=64.0, original_context=4096)
    dim, theta = 64, 50000.0
    low = max(math.floor(dim * math.log(4096 / (32 * 2 * math.pi)) / (2 * math.log(theta))), 0)
    high = min(math.ceil(dim * math.log(4096 / (2 * math.pi)) / (2 * math.log(theta))), dim-1)
    ramp = np.clip((np.arange(dim//2) - low) / (high-low), 0, 1)
    inv = theta ** (-np.arange(0, dim, 2) / dim)
    angles = np.arange(128)[:, None] * (inv * (1-ramp) + inv / 64 * ramp)
    expected = np.concatenate((np.cos(angles), np.sin(angles)), axis=-1)
    actual = precompute_freqs_cis(dim, 128, theta, yarn=yarn).numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
    self.assertAlmostEqual(yarn.attention_factor, (1 + 0.1 * math.log(64)) ** 2)
    # YaRN changes positions even below the original context length; it preserves position zero's magnitude.
    np.testing.assert_equal(actual[0], np.concatenate((np.ones(dim//2), np.zeros(dim//2))))
    self.assertGreater(np.max(np.abs(actual[1] - precompute_freqs_cis(dim, 128, theta).numpy()[1])), 0.001)

  def test_mla_yarn_attention_matches_naive(self):
    self.test_mla_attention_matches_naive(YaRNConfig(64.0, 4096))

  def test_mla_attention_matches_naive(self, yarn=None):
    config = self._make_config(max_context=16, yarn=yarn)

    block = MLATransformerBlock(config)
    c = config
    B, T = 1, 4
    q_nope_head_dim = c.head_dim - c.rope_dim

    x = Tensor.randn(B, T, c.dim)
    x_norm = block.attn_norm(x)

    # --- Our absorbed implementation ---
    q = block.attn_q(x_norm).reshape(B, T, c.n_heads, c.head_dim).transpose(1, 2)
    q_nope, q_rope = q[..., :q_nope_head_dim], q[..., q_nope_head_dim:]
    freqs = precompute_freqs_cis(c.rope_dim, 16, c.rope_theta, yarn=yarn)
    q_rope = apply_rope(q_rope, freqs[0:T])

    kv_a = block.attn_kv_a_mqa(x_norm)
    c_kv = block.attn_kv_a_norm(kv_a[..., :c.kv_lora_rank])
    k_rope = kv_a[..., c.kv_lora_rank:].reshape(B, T, 1, c.rope_dim).transpose(1, 2)
    k_rope = apply_rope(k_rope, freqs[0:T])

    # --- Naive (non-absorbed): expand K and V, do standard attention ---
    k_nope_naive = c_kv.unsqueeze(1) @ block.attn_k_b["weight"]  # (B, H, T, nope)
    k_naive = k_nope_naive.cat(k_rope.expand(-1, c.n_heads, -1, -1), dim=-1)  # (B, H, T, nope+rope)
    v_naive = c_kv.unsqueeze(1) @ block.attn_v_b["weight"].transpose(-1, -2)  # (B, H, T, v_dim)

    q_naive = q_nope.cat(q_rope, dim=-1)
    scale = (yarn.attention_factor if yarn else 1.0) / c.head_dim ** 0.5
    scores_naive = (q_naive @ k_naive.transpose(-1, -2)) * scale
    # causal mask
    mask = Tensor.full((1, 1, T, T), float("-inf")).triu(1)
    attn_naive = (scores_naive + mask).softmax(-1) @ v_naive  # (B, H, T, v_dim)
    out_naive = block.attn_output(attn_naive.transpose(1, 2).reshape(B, T, -1))

    # --- Absorbed: q_nope @ wk_b^T, then dot with compressed kv ---
    q_nope_abs = q_nope @ block.attn_k_b["weight"].transpose(-1, -2)  # (B, H, T, lora)
    q_abs = q_nope_abs.cat(q_rope, dim=-1)  # (B, H, T, lora+rope)
    k_abs = c_kv.reshape(B, 1, T, c.kv_lora_rank).cat(k_rope.reshape(B, 1, T, c.rope_dim), dim=-1)
    scores_abs = (q_abs @ k_abs.transpose(-1, -2)) * scale
    attn_abs = (scores_abs + mask).softmax(-1)
    # attn @ v_compressed @ wv_b
    v_compressed = c_kv.reshape(B, 1, T, c.kv_lora_rank)
    attn_abs_out = (attn_abs @ v_compressed) @ block.attn_v_b["weight"].transpose(-1, -2)
    out_abs = block.attn_output(attn_abs_out.transpose(1, 2).reshape(B, T, -1))

    # Compare
    naive_np = out_naive.realize().numpy()
    abs_np = out_abs.realize().numpy()
    np.testing.assert_allclose(naive_np, abs_np, atol=1e-4, rtol=1e-4,
      err_msg="Absorbed MLA should match naive MLA")
    block._init_state(x_norm)
    np.testing.assert_allclose(block._attention(x_norm, 0).numpy(), naive_np, atol=1e-4, rtol=1e-4)

  def test_shared_expert_gate_optional(self):
    model = Transformer(self._make_config(num_experts=4, num_experts_per_tok=2, shared_expert_dim=32, shared_expert_gate=False))
    self.assertNotIn('blk.0.ffn_gate_inp_shexp.weight', nn.state.get_state_dict(model))
    out = model.blk[0]._feed_forward(Tensor.randn(1, 4, model.blk[0].config.dim))
    self.assertEqual(out.shape, (1, 4, model.blk[0].config.dim))

  def test_sharded_mla_moe_matches_single_device(self):
    config = self._make_config(num_experts=4, num_experts_per_tok=2, shared_expert_dim=32, q_lora_rank=16)
    reference, sharded = Transformer(config), Transformer(config)
    devices = ("CPU", "CPU:1", "CPU:2", "CPU:3")
    for name, weight in nn.state.get_state_dict(reference).items():
      weight.replace(Tensor.randn(*weight.shape, device="CPU") * 0.1)
      weight.realize()
      spec = _shard_policy(name)
      nn.state.get_state_dict(sharded)[name].replace(weight.shard(devices, axis=spec if isinstance(spec, int) else None))
    x = Tensor.randn(1, 3, config.dim, device="CPU").realize()
    for pos, inp in ((0, x), (3, x[:, :1])):
      expected = reference.blk[0](inp, pos).numpy()
      actual = sharded.blk[0](inp.to(devices), pos).numpy()
      np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)

  def test_sharded_generation_jit(self):
    reference, sharded = Transformer(self._make_config()), Transformer(self._make_config())
    devices = ("CPU", "CPU:1", "CPU:2", "CPU:3")
    sharded.devices = devices
    for name, weight in nn.state.get_state_dict(reference).items():
      weight.replace(Tensor.randn(*weight.shape, device="CPU") * 0.1)
      weight.realize()
      spec = _shard_policy(name)
      value = weight.shard(devices, axis=spec if isinstance(spec, int) else None) if spec is not None else weight
      nn.state.get_state_dict(sharded)[name].replace(value)
    expected = [token for _, token in zip(range(3), reference.generate([1]))]
    actual = [token for _, token in zip(range(3), sharded.generate([1]))]
    self.assertEqual(actual, expected)

if __name__ == "__main__": unittest.main()
