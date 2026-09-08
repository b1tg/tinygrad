import unittest
import numpy as np
from tinygrad import Tensor, Device, TinyJit, UOp
from tinygrad.nn.state import get_state_dict, load_state_dict
from tinygrad.llm.model import Transformer, TransformerConfig, SSMConfig
from tinygrad.llm.shard import ShardedBlock, _linear, shard_model
from tinygrad.llm.kernels.amd import Linear, ExpertWeights, amd_custom_kernels_supported
from tinygrad.llm.gguf import ggml_data_to_tensor
from test.helpers import needs_second_gpu

class TestLLMShard(unittest.TestCase):
  @needs_second_gpu
  def setUp(self):
    self.devices = (Device.DEFAULT, Device.canonicalize(f'{Device.DEFAULT}:1'))

  def test_blocks_and_recurrent_reset(self):
    c = TransformerConfig(num_blocks=2, dim=256, hidden_dim=512, n_heads=4, n_kv_heads=2, norm_eps=1e-6, vocab_size=512,
      head_dim=64, rope_theta=10000, rope_dim=64, v_head_dim=64, max_context=32, num_experts=4, num_experts_per_tok=2,
      norm_topk_prob=True, shared_expert_dim=256, attn_output_gate=True, ssm=SSMConfig(4,64,2,4,256), ssm_layers=(True,False))
    model = Transformer(c)
    for name,t in get_state_dict(model).items():
      t.replace(Tensor.randn(*t.shape)*0.03 if 'norm' not in name else Tensor.ones(*t.shape)).realize()
    for block in model.blk:
      sharded = ShardedBlock(block, self.devices)
      def run(x, pos): return sharded(x, pos).to(self.devices[0]).realize()
      jit = TinyJit(run)
      for pos, count in ((0,3), (3,1), (4,1), (5,1), (0,1)):
        x = Tensor.randn(1,count,c.dim).realize()
        sp = UOp.variable('start_pos', 0, 31).bind(pos)
        expected = block(x, sp).numpy()
        # Separate prefill and decode JIT input shapes, as in Transformer.
        actual = (jit(x, sp) if count == 1 else run(x, sp)).numpy()
        np.testing.assert_allclose(actual, expected, atol=3e-4, rtol=3e-4)

  def test_packed_q8_rows_and_columns(self):
    if not amd_custom_kernels_supported(Device.DEFAULT): self.skipTest('RDNA3 required')
    rng = np.random.default_rng(42)
    no, ni = 256, 512
    packed = rng.integers(0, 256, (no*ni//32,34), dtype=np.uint8)
    packed[:, :2] = np.array([0.01], dtype=np.float16).view(np.uint8)
    raw = Tensor(np.pad(packed.flatten(), (2,0))).realize()[2:]
    layer = Linear(ni, no, bias=False)
    layer.weight = ggml_data_to_tensor(raw, ni*no, 8).reshape(no, ni).half()
    for axis in (0,1):
      local = [_linear(layer, d, [(i*no//2,(i+1)*no//2)] if axis == 0 else None,
                       (i*ni//2,(i+1)*ni//2) if axis == 1 else None) for i,d in enumerate(self.devices)]
      for tokens in (1,3):
        x = Tensor.randn(1,tokens,ni).realize()
        outputs = [l((x if axis == 0 else x[...,i*ni//2:(i+1)*ni//2]).to(d)).to(self.devices[0])
                   for i,(l,d) in enumerate(zip(local,self.devices))]
        actual = Tensor.cat(*outputs, dim=-1) if axis == 0 else outputs[0]+outputs[1]
        np.testing.assert_allclose(actual.numpy(), layer(x).numpy(), atol=2e-3, rtol=2e-3)
        self.assertTrue(all(l._q8_0_weight is not None for l in local))

  def test_model_greedy(self):
    c = TransformerConfig(num_blocks=2, dim=256, hidden_dim=512, n_heads=4, n_kv_heads=2, norm_eps=1e-6, vocab_size=512,
      head_dim=64, rope_theta=10000, rope_dim=64, v_head_dim=64, max_context=64)
    model, parallel = Transformer(c), Transformer(c)
    load_state_dict(parallel, get_state_dict(model), verbose=False, realize=False)
    shard_model(parallel, self.devices)
    temp = Tensor([0.0]).realize()
    for pos, tokens in ((0,[1,2,3]), (3,[4]), (4,[5]), (5,[6]), (0,[1])):
      x = Tensor([tokens], dtype='int32').realize()
      sp = UOp.variable('start_pos', 0, 63).bind(pos)
      np.testing.assert_array_equal(parallel(x,sp,temp).numpy(), model(x,sp,temp).numpy())

  def test_packed_expert_rows_and_columns(self):
    if not amd_custom_kernels_supported(Device.DEFAULT): self.skipTest('RDNA3 required')
    rng = np.random.default_rng(42)
    ne, no, ni = 4, 64, 512
    for typ, size in ((12,144), (13,176), (14,210), (23,136)):
      packed = rng.integers(0, 256, (ne*no*ni//256,size), dtype=np.uint8)
      packed[:, -2:] = np.array([0.001], dtype=np.float16).view(np.uint8)
      if typ != 14: packed[:, :2] = np.array([0.001], dtype=np.float16).view(np.uint8)
      if typ in (12,13): packed[:, 2:4] = np.array([0.0002], dtype=np.float16).view(np.uint8)
      raw = Tensor(np.pad(packed.flatten(), (4,0))).realize()[4:]
      weight = ggml_data_to_tensor(raw, ne*no*ni, typ).reshape(ne,no,ni).half()
      for axis in (0,1):
        layer = ExpertWeights(ne, ni, no)
        layer.weight = weight
        local = [_linear(layer, d, [(i*no//2,(i+1)*no//2)] if axis == 0 else None,
                         (i*ni//2,(i+1)*ni//2) if axis == 1 else None) for i,d in enumerate(self.devices)]
        sel = Tensor([[[3,1]]], dtype='int32').realize()
        x = Tensor.randn(1,1,2,ni).realize()
        outputs = [l(sel.to(d), (x if axis == 0 else x[...,i*ni//2:(i+1)*ni//2]).to(d)).to(self.devices[0])
                   for i,(l,d) in enumerate(zip(local,self.devices))]
        actual = Tensor.cat(*outputs, dim=-1) if axis == 0 else outputs[0]+outputs[1]
        np.testing.assert_allclose(actual.numpy(), layer(sel,x).numpy(), atol=1e-4, rtol=1e-4)
        self.assertTrue(all(l.ggml_type == typ for l in local))

if __name__ == '__main__': unittest.main()
