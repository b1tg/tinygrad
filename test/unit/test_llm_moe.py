import os, unittest
import numpy as np
from dataclasses import replace
from tinygrad import Tensor, dtypes, getenv
from tinygrad.llm.gguf import ggml_data_to_tensor
from tinygrad.llm.model import ExpertWeights, TransformerBlock, TransformerConfig

def _moe_config(dim=8, hidden=16, n_heads=2, num_experts=4, num_experts_per_tok=2):
  return TransformerConfig(
    num_blocks=1, dim=dim, hidden_dim=hidden, n_heads=n_heads, n_kv_heads=n_heads,
    norm_eps=1e-5, vocab_size=100, head_dim=dim//n_heads, rope_theta=10000,
    rope_dim=dim//n_heads, v_head_dim=dim//n_heads, max_context=16,
    num_experts=num_experts, num_experts_per_tok=num_experts_per_tok)

class TestMoEFeedForward(unittest.TestCase):
  def test_q4_0_expert_weights(self):
    num_experts, hidden, dim, k = 4, 8, 32, 2
    raw = Tensor([x for e in range(num_experts) for h in range(hidden) for x in [0, 60] + [
      ((i+h+e)&15) | (((15-i+h+e)&15)<<4) for i in range(16)]], dtype=dtypes.uint8).contiguous().realize()
    w = ggml_data_to_tensor(raw, num_experts*hidden*dim, 2).reshape(num_experts, hidden, dim).cast(dtypes.float16)
    sel = Tensor([0, 3], dtype=dtypes.int32).reshape(1, 1, k)
    x = Tensor([127] + list(range(-15, 16)), dtype=dtypes.float16).reshape(1, 1, 1, dim)
    ref, opt = ExpertWeights(num_experts, dim, hidden), ExpertWeights(num_experts, dim, hidden)
    ref.weight.replace(w)
    opt.weight.replace(w)
    opt.weight._ggml_qtype, opt.weight._ggml_raw = 2, raw.reshape(num_experts, hidden, dim//32, 18)
    old, old_dot4 = os.environ.get("Q4_EXPERT"), os.environ.get("Q4_DOT4")
    os.environ["Q4_EXPERT"], os.environ["Q4_DOT4"] = "1", "0"
    getenv.cache_clear()
    try: np.testing.assert_allclose(ref(sel, x).numpy(), opt(sel, x).numpy(), rtol=1e-3, atol=1e-2)
    finally:
      if old is None: os.environ.pop("Q4_EXPERT")
      else: os.environ["Q4_EXPERT"] = old
      if old_dot4 is None: os.environ.pop("Q4_DOT4")
      else: os.environ["Q4_DOT4"] = old_dot4
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
