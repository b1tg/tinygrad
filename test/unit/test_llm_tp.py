import unittest
import numpy as np
from tinygrad import Tensor, UOp, nn
from tinygrad.llm.model import Transformer, TransformerConfig, TransformerBlock, GatedDeltaNetBlock, SSMConfig
from tinygrad.llm.tp import TensorParallelBlock, partition, split_linear, shard_model
from tinygrad.llm.kernels.amd import Linear, QUANT_SIZES
from tinygrad.llm.gguf import ggml_data_to_tensor

class TestTensorParallel(unittest.TestCase):
  def test_group_partition(self):
    x = Tensor.arange(24).reshape(12, 2)
    shards = [partition(x, 0, i, 2, (4, 4, 4)).numpy() for i in range(2)]
    np.testing.assert_array_equal(shards[0], x.numpy()[[0, 1, 4, 5, 8, 9]])
    np.testing.assert_array_equal(shards[1], x.numpy()[[2, 3, 6, 7, 10, 11]])

  def test_packed_weights(self):
    for typ, size in QUANT_SIZES.items():
      raw = Tensor(np.random.default_rng(0).integers(0, 255, 16*size+4, dtype=np.uint8)).realize()[4:]
      source = Linear(1024, 4, bias=False)
      source.set_quantized(ggml_data_to_tensor(raw, 4*1024, typ).reshape(4, 1024))
      for axis in (0, 1):
        for groups in (None, (2, 2) if axis == 0 else (512, 512)):
          for rank in (0, 1):
            target = Linear(512 if axis == 1 else 1024, 2 if axis == 0 else 4, bias=False)
            split_linear(source, target, str(raw.device), axis, rank, 2, groups)
            actual = target.weight.bitcast('uint8').reshape(target.out_features, target.in_features//256, size).numpy()
            full = raw.numpy().reshape(4, 4, size)
            parts = [full] if groups is None else np.split(full, 2, axis=axis)
            expected = np.concatenate([np.array_split(p, 2, axis=axis)[rank] for p in parts], axis=axis)
            np.testing.assert_array_equal(actual, expected)

  def test_model_jit(self):
    Tensor.manual_seed(42)
    config = TransformerConfig(num_blocks=2, dim=32, hidden_dim=64, n_heads=4, n_kv_heads=2, norm_eps=1e-5,
      vocab_size=64, head_dim=8, v_head_dim=8, rope_theta=10000, rope_dim=8, max_context=16)
    model, tp = Transformer(config), Transformer(config)
    nn.state.load_state_dict(tp, nn.state.get_state_dict(model), verbose=False)
    shard_model(tp, 2)
    for pos, token in enumerate((3, 7, 5, 11, 1)):
      start = UOp.variable('start_pos', 0, 15).bind(pos)
      x, temp = Tensor([[token]]), Tensor([0.0])
      np.testing.assert_array_equal(tp(x, start, temp).numpy(), model(x, start, temp).numpy())

  def _check(self, recurrent, kernel=False):
    Tensor.manual_seed(42)
    config = TransformerConfig(num_blocks=1, dim=32, hidden_dim=64, n_heads=4, n_kv_heads=2, norm_eps=1e-5,
      vocab_size=32, head_dim=8, v_head_dim=8, rope_theta=10000, rope_dim=8, max_context=16,
      qk_norm=8, attn_output_gate=True, ssm=SSMConfig(4, 8, 4, 12, 96) if recurrent else None)
    if kernel:
      from dataclasses import replace
      config = replace(config, dim=256, hidden_dim=512, head_dim=256, v_head_dim=256, qk_norm=256, rope_dim=64,
                       n_heads=12, n_kv_heads=2, max_context=4096, ssm=SSMConfig(4, 128, 4, 12, 1536))
    block = GatedDeltaNetBlock(config, config.ssm) if recurrent else TransformerBlock(config)
    for name, weight in nn.state.get_state_dict(block).items():
      if name == 'ssm_a': weight.replace(-Tensor.rand(*weight.shape))
      elif 'norm' not in name: weight.replace(Tensor.randn(*weight.shape)*0.05)
      weight.realize()
    dev = str(block.attn_norm.weight.device).split(':')[0]
    tp = TensorParallelBlock(block, (dev, dev+':1'))
    for pos, tokens in ((0, 3), (3, 1), (4, 1), (0, 2)):
      x = Tensor.randn(1, tokens, config.dim).realize()
      expected = block(x, pos).numpy()
      actual = tp(tuple(x.to(d) for d in tp.devices), pos)
      Tensor.realize(*actual)
      for shard in actual: np.testing.assert_allclose(shard.numpy(), expected, atol=2e-4, rtol=2e-4)

  def test_attention(self): self._check(False)
  def test_gated_delta(self): self._check(True)

  def test_gated_delta_kernel(self):
    from tinygrad.llm.kernels.amd import amd_custom_kernels_supported
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    self._check(True, kernel=True)

  def test_attention_kernel(self):
    from tinygrad.llm.kernels.amd import amd_custom_kernels_supported
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    self._check(False, kernel=True)

if __name__ == '__main__': unittest.main()
