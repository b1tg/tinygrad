import unittest
import numpy as np
from tinygrad import Tensor, UOp, nn
from tinygrad.llm.model import (_kda_log_decay, AttentionIndexer, IndexerConfig, Transformer, TransformerConfig, apply_rope,
                                bitonic_topk, gather_rows, MLATransformerBlock, precompute_freqs_cis)

class TestMLA(unittest.TestCase):
  def test_glm_bounded_decay_uses_converted_a(self):
    # GGUF stores -exp(A_log), so the GLM safe gate must negate it instead of exponentiating it again.
    gate = Tensor([-1.0, 0.0, 1.0])
    converted_a = Tensor([-4.0])
    expected = -5.0 / (1.0 + np.exp(-(gate.numpy() * 4.0)))
    np.testing.assert_allclose(_kda_log_decay(gate, converted_a, -5.0).numpy(), expected, rtol=1e-6, atol=1e-6)

  def _make_config(self, **kwargs):
    return TransformerConfig(**{
      "num_blocks": 1, "dim": 64, "hidden_dim": 128, "n_heads": 4, "n_kv_heads": 1,
      "norm_eps": 1e-5, "vocab_size": 100, "head_dim": 16, "rope_theta": 10000.0, "rope_dim": 8, "max_context": 32,
      "kv_lora_rank": 16, "v_head_dim": 8,
    } | kwargs)

  def test_mla_attention_matches_naive(self):
    config = self._make_config(max_context=16)

    block = MLATransformerBlock(config)
    c = config
    B, T = 1, 4
    q_nope_head_dim = c.head_dim - c.rope_dim

    x = Tensor.randn(B, T, c.dim)
    x_norm = block.attn_norm(x)

    # --- Our absorbed implementation ---
    q = block.attn_q(x_norm).reshape(B, T, c.n_heads, c.head_dim).transpose(1, 2)
    q_nope, q_rope = q[..., :q_nope_head_dim], q[..., q_nope_head_dim:]
    freqs = precompute_freqs_cis(c.rope_dim, 16, c.rope_theta)
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
    scale = 1.0 / c.head_dim ** 0.5
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

  def test_shared_expert_gate_optional(self):
    model = Transformer(self._make_config(num_experts=4, num_experts_per_tok=2, shared_expert_dim=32, shared_expert_gate=False))
    self.assertNotIn('blk.0.ffn_gate_inp_shexp.weight', nn.state.get_state_dict(model))
    out = model.blk[0]._feed_forward(Tensor.randn(1, 4, model.blk[0].config.dim))
    self.assertEqual(out.shape, (1, 4, model.blk[0].config.dim))

class TestGLMIndexer(unittest.TestCase):
  @staticmethod
  def _config(**kwargs):
    return TransformerConfig(**{
      "num_blocks": 1, "dim": 16, "hidden_dim": 32, "n_heads": 2, "n_kv_heads": 1,
      "norm_eps": 1e-5, "vocab_size": 32, "head_dim": 8, "rope_theta": 10000.0, "rope_dim": 0,
      "v_head_dim": 4, "max_context": 8, "q_lora_rank": 6, "kv_lora_rank": 5,
      "indexer": IndexerConfig(top_k=4, head_dim=4, n_heads=2, kpool=2),
    } | kwargs)

  def test_bitonic_topk_preserves_indices(self):
    x = np.array([[[3, 1, 4, 1, 5, 9, 2], [8, 5, 9, 7, 9, 3, 2]],
                  [[6, 5, 3, 5, 8, 9, 7], [9, 3, 2, 3, 8, 4, 6]]], dtype=np.float32)
    values, indices = bitonic_topk(Tensor(x), 4)
    expected_indices = np.argsort(-x, axis=-1, kind="stable")[..., :4]
    np.testing.assert_equal(indices.numpy(), expected_indices)
    np.testing.assert_equal(values.numpy(), np.take_along_axis(x, expected_indices, axis=-1))

  def test_gather_rows(self):
    src = np.arange(2*5*3, dtype=np.float32).reshape(2, 5, 3)
    idx = np.array([[[4, 1], [0, 3]], [[2, 2], [4, 0]]], dtype=np.int32)
    expected = np.stack([src[b][idx[b]] for b in range(2)])
    np.testing.assert_equal(gather_rows(Tensor(src).realize(), Tensor(idx).realize()).numpy(), expected)

  def test_kpool_indexer_matches_numpy(self):
    rng = np.random.default_rng(7)
    config, B, T = self._config(), 1, 8
    indexer = AttentionIndexer(config)
    indexer.attn_q_b.weight = Tensor(rng.normal(size=indexer.attn_q_b.weight.shape).astype(np.float32))
    indexer.attn_k.weight = Tensor(rng.normal(size=indexer.attn_k.weight.shape).astype(np.float32))
    indexer.k_norm.weight = Tensor(rng.normal(size=indexer.k_norm.weight.shape).astype(np.float32))
    indexer.k_norm.bias = Tensor(rng.normal(size=indexer.k_norm.bias.shape).astype(np.float32))
    indexer.proj.weight = Tensor(rng.normal(size=indexer.proj.weight.shape).astype(np.float32))
    indexer.compressor_gate.weight = Tensor(rng.normal(size=indexer.compressor_gate.weight.shape).astype(np.float32))
    indexer.compressor_ape = Tensor(rng.normal(size=indexer.compressor_ape.shape).astype(np.float32))
    hidden = rng.normal(size=(B, T, config.dim)).astype(np.float32)
    q_resid = rng.normal(size=(B, T, config.q_lora_rank)).astype(np.float32)
    actual = indexer(Tensor(hidden), Tensor(q_resid), 0).numpy()[0]

    ic = config.indexer
    assert ic is not None
    q = (q_resid @ indexer.attn_q_b.weight.numpy().T).reshape(B, T, ic.n_heads, ic.head_dim)[0]
    keys = hidden @ indexer.attn_k.weight.numpy().T
    keys = (keys - keys.mean(-1, keepdims=True)) / np.sqrt(keys.var(-1, keepdims=True) + 1e-6)
    keys = keys * indexer.k_norm.weight.numpy() + indexer.k_norm.bias.numpy()
    gates = hidden @ indexer.compressor_gate.weight.numpy().T
    weights = (hidden @ indexer.proj.weight.numpy().T)[0] * ic.n_heads**-0.5
    pool_keys = []
    for p in range(T // ic.kpool):
      sl = slice(p*ic.kpool, (p+1)*ic.kpool)
      logits = gates[0, sl] + indexer.compressor_ape.numpy()
      probabilities = np.exp(logits - logits.max(0, keepdims=True))
      probabilities /= probabilities.sum(0, keepdims=True)
      pool_keys.append((probabilities * keys[0, sl]).sum(0))
    pool_keys = np.stack(pool_keys)

    expected = []
    select_k = ic.top_k // ic.kpool
    for pos in range(T):
      visible_pools = (pos+1) // ic.kpool
      scores = np.maximum(0, np.einsum("hd,pd->hp", q[pos], pool_keys) * ic.head_dim**-0.5)
      scores = np.einsum("h,hp->p", weights[pos], scores)
      selected = np.argsort(-scores[:visible_pools], kind="stable")[:select_k]
      raw = np.concatenate([np.arange(p*ic.kpool, (p+1)*ic.kpool) for p in selected]) if len(selected) else np.array([], dtype=int)
      raw = np.pad(raw, (0, ic.top_k-len(raw)), constant_values=-1)
      tail_count = (pos+1) % ic.kpool
      tail = np.arange(pos+1-tail_count, pos+1) if tail_count else np.array([], dtype=int)
      tail = np.pad(tail, (0, ic.kpool-1-len(tail)), constant_values=-1)
      expected.append(np.concatenate((raw, tail)))
    np.testing.assert_equal(actual, np.stack(expected).astype(np.int32))

  def test_sparse_mla_equals_dense_before_topk_limit(self):
    sparse = MLATransformerBlock(self._config(max_context=8))
    dense = MLATransformerBlock(self._config(max_context=8, indexer=None))
    sparse_state = nn.state.get_state_dict(sparse)
    for name, param in nn.state.get_state_dict(dense).items(): param.replace(sparse_state[name])
    hidden = Tensor.randn(1, 4, sparse.config.dim).realize()
    sparse._init_state(hidden)
    dense._init_state(hidden)
    np.testing.assert_allclose(sparse._attention(hidden, 0).numpy(), dense._attention(hidden, 0).numpy(), rtol=2e-3, atol=2e-3)

  def test_indexer_state_dict_names_match_gguf_mapping(self):
    state = nn.state.get_state_dict(MLATransformerBlock(self._config()))
    for name in ("indexer.attn_k.weight", "indexer.attn_q_b.weight", "indexer.k_norm.bias", "indexer.k_norm.weight",
                 "indexer.proj.weight", "indexer.compressor_ape", "indexer.compressor_gate.weight"):
      self.assertIn(name, state)

  def test_symbolic_prefill(self):
    block = MLATransformerBlock(self._config())
    toks = UOp.variable("indexer_test_toks", 1, 4).bind(3)
    start = UOp.variable("indexer_test_start", 0, 7).bind(0)
    out = block(Tensor.randn(1, 4, block.config.dim)[:, :toks], start).realize()
    self.assertEqual(out.max_shape, (1, 4, block.config.dim))
