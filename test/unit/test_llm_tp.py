import unittest
from dataclasses import replace
import numpy as np

from tinygrad import Tensor, nn
from tinygrad.llm.model import (gathered_argmax, _place_model, _predequant_category, _tp_policy, MLATransformerBlock, Transformer,
                                TransformerBlock, TransformerConfig)


def _config(num_blocks:int=1, dim:int=8, hidden:int=16, vocab_size:int=16, num_experts:int=4) -> TransformerConfig:
  return TransformerConfig(
    num_blocks=num_blocks, dim=dim, hidden_dim=hidden, n_heads=2, n_kv_heads=2,
    norm_eps=1e-5, vocab_size=vocab_size, head_dim=dim//2, rope_theta=10000,
    rope_dim=dim//2, v_head_dim=dim//2, max_context=16,
    num_experts=num_experts, num_experts_per_tok=2)


def _tensor_parallel(model, devices:tuple[str, ...]):
  placements = {}
  for name,target in nn.state.get_state_dict(model).items():
    if (spec := _tp_policy(name)) is not None:
      placements[name] = Tensor.empty(*target.shape, dtype=target.dtype, device=devices[0]).shard(devices, None if spec == "replicate" else spec)
  _place_model(model, placements, devices)

class TestLLMTensorParallel(unittest.TestCase):
  devices = ("CPU:0", "CPU:1")

  def test_predequant_categories(self):
    self.assertEqual(_predequant_category("blk.0.attn_q_b.weight"), "attention")
    self.assertEqual(_predequant_category("blk.0.ffn_gate_inp.weight"), "router")
    self.assertEqual(_predequant_category("output.weight"), "output")
    self.assertIsNone(_predequant_category("blk.0.ffn_down_exps.weight"))

  def test_moe_weight_layout(self):
    config = _config()
    Tensor.manual_seed(0)
    gate = Tensor.randn(config.num_experts, config.hidden_dim, config.dim).contiguous().realize()
    up = Tensor.randn(config.num_experts, config.hidden_dim, config.dim).contiguous().realize()
    down = Tensor.randn(config.num_experts, config.dim, config.hidden_dim).contiguous().realize()
    router = Tensor.randn(config.num_experts, config.dim).contiguous().realize()
    x = Tensor.randn(1, 1, config.dim).contiguous().realize()

    reference = TransformerBlock(config)
    reference.ffn_gate_exps.weight = gate
    reference.ffn_up_exps.weight = up
    reference.ffn_down_exps.weight = down
    reference.ffn_gate_inp.weight = router

    sharded_model = Transformer(config)
    _tensor_parallel(sharded_model, self.devices)
    sharded = sharded_model.blk[0]
    sharded.ffn_gate_exps.weight.replace(gate.shard(self.devices, axis=1))
    sharded.ffn_up_exps.weight.replace(up.shard(self.devices, axis=1))
    sharded.ffn_down_exps.weight.replace(down.shard(self.devices, axis=2))
    sharded.ffn_gate_inp.weight.replace(router.shard(self.devices))

    expected = reference._feed_forward(x).numpy()
    sharded_out = sharded._feed_forward(x.shard(self.devices))
    actual = sharded_out.to("CPU").numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

  def test_vocab_sharded_output(self):
    config = _config(num_blocks=0, dim=4, hidden=8, vocab_size=8, num_experts=0)
    embedding = Tensor.arange(config.vocab_size*config.dim).reshape(config.vocab_size, config.dim).float().contiguous().realize()
    output = Tensor.arange(config.vocab_size*config.dim).reshape(config.vocab_size, config.dim).float().contiguous().realize()
    norm = Tensor.ones(config.dim).contiguous().realize()
    tokens,temperature = Tensor([[1]]),Tensor([0.0])

    reference = Transformer(config)
    reference.token_embd.weight,reference.output.weight,reference.output_norm.weight = embedding,output,norm
    expected = reference.forward(tokens, 0, temperature).item()

    sharded = Transformer(config)
    _tensor_parallel(sharded, self.devices)
    sharded.token_embd.weight = embedding
    sharded.output.weight.replace(output.shard(self.devices, axis=0))
    sharded.output_norm.weight.replace(norm.shard(self.devices))
    actual = sharded.forward(tokens, 0, temperature).item()
    self.assertEqual(actual, expected)

  def test_gathered_argmax(self):
    values = Tensor([[0, 9, 2, 3, 4, 9, 6, 7], [8, 1, 2, 3, 4, 5, 6, 10]]).float().shard(self.devices, axis=1).realize()
    np.testing.assert_array_equal(gathered_argmax(values).numpy(), values.argmax(-1, keepdim=True).to(self.devices[0]).numpy())

  def test_mla_block(self):
    config = replace(_config(num_experts=0), q_lora_rank=4, kv_lora_rank=4, rope_dim=2, v_head_dim=2)
    Tensor.manual_seed(1)
    reference = MLATransformerBlock(config)
    sharded_model = Transformer(config)
    _tensor_parallel(sharded_model, self.devices)
    sharded = sharded_model.blk[0]

    sharded_state = nn.state.get_state_dict(sharded)
    for name,ref in nn.state.get_state_dict(reference).items():
      value = Tensor.randn(*ref.shape).contiguous().realize()
      ref.replace(value)
      target = sharded_state[name]
      target.replace(value.shard(self.devices, axis=target.uop.axis) if isinstance(target.device, tuple) else value)

    x = Tensor.randn(1, 1, config.dim).contiguous().realize()
    expected = reference(x, 0).realize().numpy()
    actual = sharded(x.shard(self.devices), 0).to("CPU").realize().numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)

  def test_generate_jit(self):
    config = replace(_config(), q_lora_rank=4, kv_lora_rank=4, rope_dim=2, v_head_dim=2, max_context=8)
    Tensor.manual_seed(2)
    model = Transformer(config)
    _tensor_parallel(model, self.devices)
    for name,param in nn.state.get_state_dict(model).items():
      value = Tensor.randn(*param.shape).contiguous().realize()
      param.replace(value.shard(self.devices, axis=param.uop.axis) if isinstance(param.device, tuple) else value)

    gen = model.generate([1, 2, 3], chunk_size=4, temperature=0.0)
    self.assertIsInstance(next(gen), int)
    self.assertIsInstance(next(gen), int)
    gen.close()
    gen = model.generate([4, 5], chunk_size=4, temperature=0.0)
    self.assertIsInstance(next(gen), int)
    gen.close()

if __name__ == "__main__":
  unittest.main()
