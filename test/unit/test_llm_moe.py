import unittest
import numpy as np
import os
from dataclasses import replace
from tinygrad import Tensor, dtypes
from tinygrad.helpers import getenv
from tinygrad.llm.gguf import ggml_data_to_tensor, _layer_shard_device, _layer_shard_splits
from tinygrad.llm.model import ExpertWeights, TransformerBlock, TransformerConfig

def _moe_config(dim=8, hidden=16, n_heads=2, num_experts=4, num_experts_per_tok=2):
  return TransformerConfig(
    num_blocks=1, dim=dim, hidden_dim=hidden, n_heads=n_heads, n_kv_heads=n_heads,
    norm_eps=1e-5, vocab_size=100, head_dim=dim//n_heads, rope_theta=10000,
    rope_dim=dim//n_heads, v_head_dim=dim//n_heads, max_context=16,
    num_experts=num_experts, num_experts_per_tok=num_experts_per_tok)

class TestMoEFeedForward(unittest.TestCase):
  def test_layer_shard_splits(self):
    old = os.environ.get("LLM_SHARD_SPLITS")
    try:
      os.environ.pop("LLM_SHARD_SPLITS", None)
      getenv.cache_clear()
      self.assertEqual(_layer_shard_splits(61, 4), (0, 15, 30, 45, 61))
      os.environ["LLM_SHARD_SPLITS"] = "0,14,30,46,61"
      getenv.cache_clear()
      splits = _layer_shard_splits(61, 4)
      self.assertEqual(splits, (0, 14, 30, 46, 61))
      self.assertEqual([_layer_shard_device(i, ("A", "B", "C", "D"), splits) for i in (0, 13, 14, 45, 60)], ["A", "A", "B", "C", "D"])
      os.environ["LLM_SHARD_SPLITS"] = "0,14,61"
      getenv.cache_clear()
      with self.assertRaises(ValueError): _layer_shard_splits(61, 4)
    finally:
      if old is None: os.environ.pop("LLM_SHARD_SPLITS", None)
      else: os.environ["LLM_SHARD_SPLITS"] = old
      getenv.cache_clear()

  def test_raw_expert_gemv_q4_0_q8_0(self):
    def f16(x): return np.array([x], dtype=np.float16).view(np.uint8).tolist()
    for qt in (2, 8):
      ne, hidden, dim = 3, 2, 32
      raw = []
      for e in range(ne):
        for h in range(hidden):
          if qt == 2:
            raw += f16(0.5)
            qs = [((i+e+h)&15) for i in range(dim)]
            raw += [qs[i] | (qs[i+16]<<4) for i in range(16)]
          else:
            raw += f16(0.25)
            raw += np.array([((i+e+h)%17)-8 for i in range(dim)], dtype=np.int8).view(np.uint8).tolist()
      rawt = Tensor(raw, dtype=dtypes.uint8)
      ew = ExpertWeights(ne, dim, hidden)
      ew.weight, ew._raw_weight = ggml_data_to_tensor(rawt, ne*hidden*dim, qt).reshape(ne, hidden, dim), (rawt, qt)
      sel, x = Tensor([[[0, 2]]], dtype=dtypes.int32), Tensor(np.arange(dim, dtype=np.float32).reshape(1, 1, 1, dim))
      ref = (x.unsqueeze(-2) @ ew.weight[sel].transpose(-1, -2)).contiguous().squeeze(-2)
      old = os.environ.get("CODEGEN_EXPERT")
      os.environ["CODEGEN_EXPERT"] = "1"
      getenv.cache_clear()
      try: np.testing.assert_allclose(ew(sel, x).numpy(), ref.numpy(), rtol=1e-5, atol=1e-5)
      finally:
        if old is None: os.environ.pop("CODEGEN_EXPERT", None)
        else: os.environ["CODEGEN_EXPERT"] = old
        getenv.cache_clear()

  def test_moe_feed_forward(self):
    dim, hidden, n_heads = 8, 16, 2
    num_experts, k = 4, 2

    block = TransformerBlock(_moe_config(dim, hidden, n_heads, num_experts, k))

    # set up weights: gate scales by (expert_id+1), up/down are identity-ish, router picks experts 0,2
    block.ffn_gate_exps.weight = Tensor.stack(*[Tensor.eye(hidden, dim) * (i + 1) for i in range(num_experts)])
    block.ffn_up_exps.weight = Tensor.stack(*[Tensor.eye(hidden, dim) for _ in range(num_experts)])
    block.ffn_down_exps.weight = Tensor.stack(*[Tensor.eye(dim, hidden) for _ in range(num_experts)])
    block.ffn_gate_inp.weight = Tensor([[1, 0, 1, 0]] * dim).T  # router strongly prefers experts 0 and 2
    block.ffn_norm.weight = Tensor.ones(dim)  # identity norm

    # input of ones -> after norm still ~ones -> experts 0,2 selected -> weighted sum of silu outputs
    h = Tensor.ones(1, 1, dim)
    out = block._feed_forward(block.ffn_norm(h))

    # expected moe_output ≈ avg(silu(1), silu(3))
    expected = (Tensor([1.0]).silu().item() + Tensor([3.0]).silu().item()) / 2
    np.testing.assert_allclose(out.numpy()[0, 0, 0], expected, rtol=1e-2)

  def test_moe_feed_forward_batched(self):
    dim, hidden, n_heads = 8, 16, 2
    num_experts, k = 4, 2

    block = TransformerBlock(_moe_config(dim, hidden, n_heads, num_experts, k))

    # same setup as BS=1 test
    block.ffn_gate_exps.weight = Tensor.stack(*[Tensor.eye(hidden, dim) * (i + 1) for i in range(num_experts)])
    block.ffn_up_exps.weight = Tensor.stack(*[Tensor.eye(hidden, dim) for _ in range(num_experts)])
    block.ffn_down_exps.weight = Tensor.stack(*[Tensor.eye(dim, hidden) for _ in range(num_experts)])
    block.ffn_gate_inp.weight = Tensor([[1, 0, 1, 0]] * dim).T
    block.ffn_norm.weight = Tensor.ones(dim)

    # test with BS=2, T=3
    h = Tensor.ones(2, 3, dim)
    out = block._feed_forward(block.ffn_norm(h))

    # all outputs should match the BS=1 expected value
    expected = (Tensor([1.0]).silu().item() + Tensor([3.0]).silu().item()) / 2
    np.testing.assert_allclose(out.numpy(), expected, rtol=1e-2)

  def test_moe_feed_forward_norm_topk_prob(self):
    dim, hidden, n_heads = 8, 16, 2
    num_experts, k = 4, 2

    block = TransformerBlock(replace(_moe_config(dim, hidden, n_heads, num_experts, k), norm_topk_prob=True))

    block.ffn_gate_exps.weight = Tensor.stack(*[Tensor.eye(hidden, dim) * (i + 1) for i in range(num_experts)])
    block.ffn_up_exps.weight = Tensor.stack(*[Tensor.eye(hidden, dim) for _ in range(num_experts)])
    block.ffn_down_exps.weight = Tensor.stack(*[Tensor.eye(dim, hidden) for _ in range(num_experts)])
    block.ffn_gate_inp.weight = Tensor([[0.1, 0, 0.1, 0]] * dim).T  # equal top-2 experts, but only ~69% mass before renorm
    block.ffn_norm.weight = Tensor.ones(dim)

    h = Tensor.ones(1, 1, dim)
    out = block._feed_forward(block.ffn_norm(h))

    expected = (Tensor([1.0]).silu().item() + Tensor([3.0]).silu().item()) / 2
    np.testing.assert_allclose(out.numpy()[0, 0, 0], expected, rtol=1e-2)

  def test_moe_feed_forward_shared_expert(self):
    dim, hidden, n_heads = 8, 16, 2
    num_experts, k = 4, 2

    block = TransformerBlock(replace(_moe_config(dim, hidden, n_heads, num_experts, k), shared_expert_dim=dim))

    block.ffn_gate_exps.weight = Tensor.stack(*[Tensor.eye(hidden, dim) * (i + 1) for i in range(num_experts)])
    block.ffn_up_exps.weight = Tensor.stack(*[Tensor.eye(hidden, dim) for _ in range(num_experts)])
    block.ffn_down_exps.weight = Tensor.stack(*[Tensor.eye(dim, hidden) for _ in range(num_experts)])
    block.ffn_gate_inp.weight = Tensor([[1, 0, 1, 0]] * dim).T
    block.ffn_gate_shexp.weight = Tensor.eye(dim) * 2
    block.ffn_up_shexp.weight = Tensor.eye(dim)
    block.ffn_down_shexp.weight = Tensor.eye(dim)
    block.ffn_gate_inp_shexp["weight"] = Tensor.zeros(dim)
    block.ffn_norm.weight = Tensor.ones(dim)

    h = Tensor.ones(1, 1, dim)
    out = block._feed_forward(block.ffn_norm(h))

    moe_expected = (Tensor([1.0]).silu().item() + Tensor([3.0]).silu().item()) / 2
    shared_expected = Tensor([2.0]).silu().item() * 0.5
    expected = moe_expected + shared_expected
    np.testing.assert_allclose(out.numpy(), expected, rtol=1e-2)

if __name__ == '__main__':
  unittest.main()
