import unittest
import numpy as np

from tinygrad import Tensor, dtypes, nn
from tinygrad.llm.gguf import ggml_data_to_tensor
from tinygrad.llm.model import PLEEmbedding, Gemma4Config, Transformer, TransformerConfig

def config(**kwargs):
  gemma4 = Gemma4Config(hidden_dims=(6, 8, 10, 12), sliding_layers=(True, False, True, False), sliding_window=3,
    swa_head_dim=2, swa_rope_theta=10000.0, shared_kv_layers=2, per_layer_embed_dim=2, final_logit_softcap=2.0)
  args = dict(num_blocks=4, dim=8, hidden_dim=12, n_heads=2, n_kv_heads=1, norm_eps=1e-6, vocab_size=16,
    head_dim=4, rope_theta=1000000.0, rope_dim=4, v_head_dim=4, max_context=8, gemma4=gemma4)
  return TransformerConfig(**(args | kwargs))

class TestGemma4(unittest.TestCase):
  def test_state_dict_and_shared_kv_layout(self):
    model = Transformer(config())
    state = nn.state.get_state_dict(model)
    self.assertEqual(state["blk.0.attn_q.weight"].shape, (4, 8))
    self.assertEqual(state["blk.1.attn_q.weight"].shape, (8, 8))
    self.assertEqual(state["blk.0.ffn_gate.weight"].shape, (6, 8))
    self.assertEqual(state["blk.3.ffn_gate.weight"].shape, (12, 8))
    self.assertNotIn("blk.0.rope_freqs.weight", state)
    self.assertIn("blk.1.rope_freqs.weight", state)
    self.assertEqual([b.store_shared_kv for b in model.blk], [True, True, False, False])
    self.assertEqual([b.shared_kv for b in model.blk], [False, False, True, True])
    self.assertEqual(state["per_layer_token_embd.weight"].shape, (16, 8))

  def test_q5_k_gather(self):
    raw = np.arange(3*176, dtype=np.uint8).reshape(3, 176)
    raw[:, :4] = np.frombuffer(np.array([1.0, 0.25], dtype=np.float16).tobytes(), dtype=np.uint8)
    decoded = ggml_data_to_tensor(Tensor(raw.flatten()), 3*256, 13).reshape(3, 256)
    embedding = PLEEmbedding(3, 256)
    embedding.weight = decoded
    out = embedding(Tensor([[2, 0]], dtype=dtypes.int64))
    np.testing.assert_equal(out.numpy(), decoded.numpy()[[2, 0]][None])

  def test_forward_uses_both_shared_caches(self):
    model = Transformer(config())
    for name, param in nn.state.get_state_dict(model).items():
      value = 1.0 if "norm.weight" in name or "layer_output_scale.weight" in name else 0.01
      param.assign(Tensor.full(param.shape, value, dtype=param.dtype)).realize()
    out = model.forward(Tensor([[1, 2]], dtype=dtypes.int32), 0, Tensor([0.0]))
    self.assertEqual(out.shape, (1, 1))
    self.assertTrue(hasattr(model.blk[0], "cache_kv"))
    self.assertTrue(hasattr(model.blk[1], "cache_kv"))
    self.assertFalse(hasattr(model.blk[2], "cache_kv"))
    self.assertFalse(hasattr(model.blk[3], "cache_kv"))

if __name__ == "__main__":
  unittest.main()
