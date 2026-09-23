import unittest, tempfile, pathlib
from unittest.mock import patch
import numpy as np
from tinygrad import Tensor, UOp, nn, Device, TinyJit
from tinygrad.llm.model import Transformer, TransformerConfig, TransformerBlock, GatedDeltaNetBlock, SSMConfig
from tinygrad.llm.tp import _local_shard, shard_config, load_sharded, replicate
from tinygrad.llm.kernels.amd import Linear, QUANT_SIZES
from tinygrad.llm.gguf import gguf_load
from test.unit import test_gguf

class TestTensorParallel(unittest.TestCase):
  def test_group_partition(self):
    x = Tensor(np.arange(24).reshape(12, 2), device="CPU").realize()
    local = _local_shard(x, ("CPU", "CPU:1"), 0, (4, 4, 4)).realize()
    shards = [Tensor(local.uop.mselect(i)).numpy() for i in range(2)]
    np.testing.assert_array_equal(shards[0], x.numpy()[[0, 1, 4, 5, 8, 9]])
    np.testing.assert_array_equal(shards[1], x.numpy()[[2, 3, 6, 7, 10, 11]])

  def test_file_backed_packed_weights(self):
    devices = ('CPU:1', 'CPU:2') if Device.DEFAULT == 'CPU' else (Device.DEFAULT, Device.DEFAULT+':1')
    for typ,size in QUANT_SIZES.items():
      packed = np.random.default_rng(0).integers(0, 255, (8, 4, size), dtype=np.uint8)
      with tempfile.TemporaryDirectory() as folder:
        path = pathlib.Path(folder)/'weights.gguf'
        path.write_bytes(test_gguf.TestGGUF._build_gguf([('weight', (8, 1024), typ, packed.tobytes())], []))
        for axis in (0, 1):
          for grouped in (False, True):
            config = TransformerConfig(num_blocks=1, dim=1024, hidden_dim=8, n_heads=4, n_kv_heads=2, norm_eps=1e-5,
              vocab_size=64, head_dim=8, v_head_dim=8, rope_theta=10000, rope_dim=8, ssm_layers=(grouped,),
              ssm=SSMConfig(4, 1, 2, 4, 4 if axis == 0 else 1024) if grouped else None)
            key = ('attn_qkv' if axis == 0 else 'ssm_out') if grouped else ('ffn_gate' if axis == 0 else 'ffn_down')
            _, state = gguf_load(path, device=lambda _: 'CPU')
            target = Linear(512 if axis == 1 else 1024, 4 if axis == 0 else 8, bias=False)
            with patch('tinygrad.llm.tp.amd_custom_kernels_supported', return_value=True):
              load_sharded({'blk':[{key:target}]}, {'blk.0.'+key+'.weight':state['weight'].half()}, config, devices)
            self.assertEqual(target.ggml_type, typ)
            pieces = np.split(packed, 4 if axis == 0 else 2, axis=axis) if grouped else (packed,)
            for rank in (0, 1):
              local = Tensor(target.weight.uop.mselect(rank))
              self.assertEqual(local.uop.buf_uop.dtype, target.weight.dtype)
              expected = np.concatenate([np.split(p, 2, axis=axis)[rank] for p in pieces], axis=axis)
              np.testing.assert_array_equal(local.bitcast('uint8').numpy(), expected.flatten())

  def test_file_backed_model(self):
    Tensor.manual_seed(42)
    config = TransformerConfig(num_blocks=2, dim=32, hidden_dim=64, n_heads=4, n_kv_heads=2, norm_eps=1e-5,
      vocab_size=64, head_dim=8, v_head_dim=8, rope_theta=10000, rope_dim=8, max_context=16)
    devices = (Device.DEFAULT, Device.DEFAULT+":1")
    model, tp = Transformer(config), Transformer(config, devices)
    for weight in nn.state.get_parameters(model): weight.replace(weight.half().realize())
    with tempfile.TemporaryDirectory() as folder:
      path = pathlib.Path(folder)/'model.gguf'
      tensors = [(name, weight.shape, 0, weight.numpy().astype(np.float32).tobytes()) for name,weight in nn.state.get_state_dict(model).items()]
      path.write_bytes(test_gguf.TestGGUF._build_gguf(tensors, []))
      _, index = gguf_load(path, device=lambda _: "CPU")
      load_sharded(tp, {k:v.half() for k,v in index.items()}, config, devices)
      for pos, token in enumerate((3, 7, 5, 11, 1)):
        start = UOp.variable('start_pos', 0, 15).bind(pos)
        x, temp = Tensor([[token]]), Tensor([0.0])
        np.testing.assert_array_equal(tp(x, start, temp).numpy(), model(x, start, temp).numpy())

  def test_gguf_streaming_model(self):
    from gguf import GGUFWriter
    Tensor.manual_seed(42)
    config = TransformerConfig(num_blocks=1, dim=32, hidden_dim=64, n_heads=4, n_kv_heads=2, norm_eps=1e-5,
      vocab_size=64, head_dim=8, v_head_dim=8, rope_theta=10000, rope_dim=8, max_context=16)
    model = Transformer(config)
    with tempfile.TemporaryDirectory() as folder:
      path = pathlib.Path(folder)/'model.gguf'
      writer = GGUFWriter(path, 'qwen3')
      for key,value in {'context_length':16, 'embedding_length':32, 'feed_forward_length':64, 'block_count':1,
                        'attention.head_count':4, 'attention.head_count_kv':2}.items(): writer.add_uint32('qwen3.'+key, value)
      writer.add_float32('qwen3.rope.freq_base', 10000)
      writer.add_float32('qwen3.attention.layer_norm_rms_epsilon', 1e-5)
      writer.add_array('tokenizer.ggml.tokens', [str(i) for i in range(64)])
      for name,weight in nn.state.get_state_dict(model).items(): writer.add_tensor(name, weight.numpy())
      writer.write_header_to_file()
      writer.write_kv_data_to_file()
      writer.write_tensors_to_file()
      writer.close()
      single,_ = Transformer.from_gguf(path, 16)
      parallel,_ = Transformer.from_gguf(path, 16, shard=2)
      for pos,token in enumerate((3, 7, 5, 11, 1)):
        start = UOp.variable('start_pos', 0, 15).bind(pos)
        x, temp = Tensor([[token]]), Tensor([0.0])
        np.testing.assert_array_equal(parallel(x, start, temp).numpy(), single(x, start, temp).numpy())

  def _check(self, recurrent, kernel=False):
    Tensor.manual_seed(42)
    config = TransformerConfig(num_blocks=1, dim=32, hidden_dim=64, n_heads=4, n_kv_heads=2, norm_eps=1e-5,
      vocab_size=32, head_dim=8, v_head_dim=8, rope_theta=10000, rope_dim=8, max_context=16, ssm_layers=(recurrent,),
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
    devices = (Device.DEFAULT, Device.DEFAULT+':1')
    local = shard_config(config, devices)
    tp = GatedDeltaNetBlock(local, local.ssm) if recurrent else TransformerBlock(local)
    weights = {}
    for name,weight in nn.state.get_state_dict(block).items():
      weight.replace(weight.half().realize())
      weights['blk.0.'+name] = weight
    load_sharded({'blk':[tp]}, weights, config, devices)
    ref_decode, tp_decode = TinyJit(block), TinyJit(tp)
    for pos, tokens in ((0, 3), (3, 1), (4, 1), (5, 1), (6, 1), (0, 2)):
      start = UOp.variable("start_pos", 0, config.max_context-1).bind(pos)
      if tokens > 1:
        width = 32 if kernel else 8
        x = Tensor.randn(1, width, config.dim).realize()[:, :UOp.variable("toks", 1, width).bind(tokens)]
        expected = block(x, start).pad_to((1, width, config.dim)).numpy()[:, :tokens]
        actual = tp(replicate(x, devices), start).pad_to((1, width, config.dim)).realize()
      else:
        x = Tensor.randn(1, 1, config.dim).realize()
        expected = ref_decode(x, start).numpy()
        actual = tp_decode(replicate(x, devices), start).realize()
      for rank in range(len(devices)):
        np.testing.assert_allclose(Tensor(actual.uop.mselect(rank)).numpy()[:, :tokens], expected, atol=2e-4, rtol=2e-4)

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
