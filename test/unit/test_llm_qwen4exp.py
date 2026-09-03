import unittest
import numpy as np

from tinygrad import Tensor, dtypes, nn
from tinygrad.llm.gguf import ggml_data_to_tensor
from tinygrad.llm.model import PLEEmbedding, GatedResidual, QSAIndexer, SSMConfig, Transformer, TransformerConfig, precompute_freqs_cis

def config(**kwargs):
  args = dict(num_blocks=2, dim=8, hidden_dim=4, n_heads=2, n_kv_heads=1, norm_eps=1e-6, vocab_size=16,
    head_dim=4, rope_theta=10000, rope_dim=2, v_head_dim=4, max_context=8, num_experts=4, num_experts_per_tok=2,
    norm_topk_prob=True, shared_expert_dim=4, ssm=SSMConfig(2, 2, 1, 2, 4), ssm_layers=(True, False), attn_output_gate=True,
    hyper_connection_count=2, hyper_connection_low_rank=2, indexer_heads=1, indexer_head_dim=4, indexer_top_k=4,
    indexer_compress_ratio=2, ple_layers=(0,), ple_ngram_size=3, ple_heads_per_ngram=1, ple_conv_kernel=2,
    ple_eos_token_id=0, ple_row_dim=2, ple_vocab_size=8, ple_layer_multipliers=(3, 5, 7), ple_head_offsets=(0, 4),
    ple_head_vocab_sizes=(4, 4))
  return TransformerConfig(**(args | kwargs))

class TestQwen4Exp(unittest.TestCase):
  def test_state_dict(self):
    state = nn.state.get_state_dict(Transformer(config()))
    self.assertIn("blk.0.ple.embedding.weight", state)
    self.assertIn("blk.1.indexer.q_proj.weight", state)
    self.assertIn("output_hc.up.weight", state)
    self.assertNotIn("output_norm.weight", state)

  def test_iq4_nl_gather(self):
    raw = np.arange(8*18, dtype=np.uint8).reshape(8, 18)
    raw[:, :2] = np.frombuffer(np.float16(1).tobytes(), dtype=np.uint8)
    decoded = ggml_data_to_tensor(Tensor(raw.flatten()), 8*32, 20).reshape(8, 32)
    embedding = PLEEmbedding(8, 32)
    embedding.weight = decoded
    out = embedding(Tensor([[7, 1]], dtype=dtypes.int64))
    np.testing.assert_equal(out.numpy(), decoded.numpy()[[7, 1]][None])

  def test_gated_residual(self):
    layer = GatedResidual(config(ple_layers=()))
    layer.norm.weight.assign(Tensor.ones(16)).realize()
    down = np.arange(32, dtype=np.float32).reshape(2, 16) / 50
    up = np.arange(32, dtype=np.float32).reshape(16, 2) / 40
    inject = np.arange(32, dtype=np.float32).reshape(2, 16) / 30
    layer.down.weight.assign(Tensor(down)).realize()
    layer.up.weight.assign(Tensor(up)).realize()
    layer.inject.weight.assign(Tensor(inject)).realize()
    x = np.arange(16, dtype=np.float32).reshape(1, 1, 16) / 10 + 0.1
    mixed, residual, injection = layer(Tensor(x))
    xn = x.reshape(1, 1, 2, 8)
    xn = (xn / np.sqrt(np.mean(xn*xn, axis=-1, keepdims=True) + 1e-6)).reshape(1, 1, 16)
    silu = lambda z: z / (1 + np.exp(-z))
    sigmoid = lambda z: 1 / (1 + np.exp(-z))
    mix = sigmoid(silu(xn @ down.T / 2) @ up.T).reshape(1, 1, 2, 8)
    np.testing.assert_allclose(mixed.numpy(), (mix*xn.reshape(1, 1, 2, 8)).mean(-2), rtol=1e-5, atol=1e-5)
    np.testing.assert_equal(residual.numpy(), x)
    np.testing.assert_allclose(injection.numpy(), 2*sigmoid(xn @ inject.T / 2), rtol=1e-5, atol=1e-5)

  def test_qsa_mask(self):
    c = config(ple_layers=(), indexer_top_k=2)
    indexer = QSAIndexer(c)
    proj = Tensor.eye(4).pad(((0, 0), (0, 4)))
    indexer.q_proj.weight.assign(proj).realize()
    indexer.k_proj.weight.assign(proj).realize()
    indexer.q_norm.weight.assign(Tensor.ones(4)).realize()
    indexer.k_norm.weight.assign(Tensor.ones(4)).realize()
    x = Tensor(np.arange(48, dtype=np.float32).reshape(1, 6, 8) / 10 + 0.1)
    indexer._init_state(x)
    mask = indexer(x, 0, precompute_freqs_cis(2, 8, 10000)).numpy()[0]
    self.assertEqual(mask.shape, (6, 6))
    self.assertFalse(np.triu(mask, 1).any())
    self.assertEqual(mask[-1].sum(), 2)

  def test_incremental_forward_resets(self):
    model = Transformer(config())
    tokens, temperature = Tensor([[1]], dtype=dtypes.int32), Tensor([0.0])
    first = model.forward(tokens, 0, temperature).numpy()
    model.forward(Tensor([[2]], dtype=dtypes.int32), 1, temperature).realize()
    np.testing.assert_equal(model.forward(tokens, 0, temperature).numpy(), first)

  def test_gated_delta_sigmoid_output_gate(self):
    block = Transformer(config(ple_layers=(), ssm_output_gate_sigmoid=True)).blk[0]
    block.ssm_conv1d["weight"].assign(Tensor.ones(*block.ssm_conv1d["weight"].shape)).realize()
    block.attn_gate.weight.assign(Tensor.full(block.attn_gate.weight.shape, 0.25)).realize()
    x = Tensor(np.arange(8, dtype=np.float32).reshape(1, 1, 8) / 10 + 0.1)
    block._init_state(x)
    xh = x.half()
    gate = block.attn_gate(xh).reshape(1, 1, block.num_v_heads, block.head_v_dim)
    beta = block.ssm_beta(xh).sigmoid().reshape(1, 1, block.num_v_heads)
    conv = (block.attn_qkv(xh) * block.ssm_conv1d["weight"][:, -1]).silu()
    q, k, v = conv.split([block.q_dim, block.q_dim, block.conv_channels - 2*block.q_dim], dim=-1)
    q, k = (z.reshape(1, 1, block.num_k_heads, block.head_k_dim).normalize(dim=-1, eps=1e-6)
            .repeat(1, 1, block.num_v_heads//block.num_k_heads, 1) for z in (q, k))
    v = v.reshape(1, 1, block.num_v_heads, block.head_v_dim)
    core = beta.unsqueeze(-1) * v * ((q * block.head_k_dim**-0.5) * k).sum(-1, keepdim=True)
    expected = block.ssm_out((block.ssm_norm(core) * gate.sigmoid()).half().reshape(1, 1, -1)).numpy()
    np.testing.assert_allclose(block._attention(x, 0).numpy(), expected, rtol=2e-3, atol=2e-3)

if __name__ == "__main__": unittest.main()
