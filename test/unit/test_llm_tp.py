import unittest, tempfile, pathlib
from unittest.mock import patch
from dataclasses import replace
import numpy as np
from tinygrad import Tensor, UOp, nn, Device, TinyJit
from tinygrad.helpers import get_child
from tinygrad.llm.model import Transformer, TransformerConfig, TransformerBlock, GatedDeltaNetBlock, SSMConfig
from tinygrad.llm.tp import gguf_sharder, replicate
from tinygrad.llm.gguf import gguf_load, ggml_data_to_tensor, GGUFQuantizedTensor
from tinygrad.llm.kernels.amd import Linear, QUANT_SIZES, amd_custom_kernels_supported
from test.unit import test_gguf

def install(model, state):
  for name,target in nn.state.get_state_dict(model).items():
    value = state[name]
    if isinstance(value, GGUFQuantizedTensor):
      module = get_child(model, name.rsplit('.', 1)[0])
      module.weight, module.ggml_type = value.data, value.ggml_type
    else: target.replace(value.half())

class TestTensorParallel(unittest.TestCase):
  def test_packed_layout(self):
    devices = ('CPU:1', 'CPU:2') if Device.DEFAULT == 'CPU' else (Device.DEFAULT, Device.DEFAULT+':1')
    for typ,size in QUANT_SIZES.items():
      packed = np.random.default_rng(0).integers(0,255,(8,4,size),dtype=np.uint8)
      for key,axis in (('ffn_gate',0),('ffn_down',1),('attn_qkv',None)):
        with tempfile.TemporaryDirectory() as folder:
          path=pathlib.Path(folder)/'w.gguf'
          path.write_bytes(test_gguf.TestGGUF._build_gguf([(key+'.weight',(8,1024),typ,packed.tobytes())], [('general.architecture','qwen35')]))
          with patch('tinygrad.llm.tp.amd_custom_kernels_supported', return_value=True):
            _,state=gguf_load(path,loader=gguf_sharder(devices))
          weight=state[key+'.weight']
          self.assertIsInstance(weight,GGUFQuantizedTensor)
          self.assertEqual(weight.shape,(8,1024))
          self.assertEqual(weight.data.uop.axis,axis)
          for rank in (0,1):
            local=Tensor(weight.data.uop.unsharded_base.mselect(rank)).bitcast('uint8').numpy()
            expected=packed if axis is None else np.split(packed,2,axis=axis)[rank]
            np.testing.assert_array_equal(local.flatten(),expected.flatten())

  def test_quantized_matmuls(self):
    if not amd_custom_kernels_supported(Device.DEFAULT): self.skipTest('RDNA3 required')
    devices=(Device.DEFAULT,Device.DEFAULT+':1')
    rng=np.random.default_rng(42)
    for typ,size in QUANT_SIZES.items():
      raw=rng.integers(0,256,(64*4,size),dtype=np.uint8)
      raw[:, -2:]=np.array([.001],np.float16).view(np.uint8)
      if typ!=14:raw[:, :2]=np.array([.001],np.float16).view(np.uint8)
      if typ in (12,13):raw[:, 2:4]=np.array([.0002],np.float16).view(np.uint8)
      weight=ggml_data_to_tensor(Tensor(raw.flatten(),device='CPU'),64*1024,typ).reshape(64,1024).numpy()
      for axis in (0,1):
        key='ffn_gate' if axis==0 else 'ffn_down'
        with tempfile.TemporaryDirectory() as folder:
          path=pathlib.Path(folder)/'w.gguf'
          path.write_bytes(test_gguf.TestGGUF._build_gguf([(key+'.weight',(64,1024),typ,raw.tobytes())],[('general.architecture','qwen35')]))
          _,state=gguf_load(path,loader=gguf_sharder(devices))
          layer=Linear(1024,64,bias=False)
          layer.weight,layer.ggml_type=state[key+'.weight'].data,typ
        for tokens in (1,32):
          values=rng.normal(size=(1,tokens,1024)).astype(np.float32)
          x=Tensor(values).realize().shard(devices,axis=2 if axis==1 else None)
          actual=layer(x).to(Device.DEFAULT).numpy()
          if tokens==32 and typ!=14: expected=values.astype(np.float16).astype(np.float32)@weight.astype(np.float16).astype(np.float32).T
          else:
            grouped=values.reshape(1,tokens,-1,32)
            scale=np.maximum(np.abs(grouped).max(-1,keepdims=True)/127,1e-8)
            expected=(np.rint(grouped/scale)*scale).reshape(values.shape)@weight.T
          np.testing.assert_allclose(actual,expected,atol=2e-2,rtol=3e-3,err_msg=f'{typ=} {axis=} {tokens=}')

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
    config=TransformerConfig(num_blocks=1,dim=32,hidden_dim=64,n_heads=4,n_kv_heads=2,norm_eps=1e-5,vocab_size=32,
      head_dim=8,v_head_dim=8,rope_theta=10000,rope_dim=8,max_context=16,qk_norm=8,attn_output_gate=True,
      ssm=SSMConfig(4,8,4,12,96) if recurrent else None,ssm_layers=(recurrent,))
    if kernel: config=replace(config,dim=256,hidden_dim=512,head_dim=256,v_head_dim=256,qk_norm=256,rope_dim=64,
                               n_heads=12,n_kv_heads=2,max_context=4096,ssm=SSMConfig(4,128,4,12,1536))
    devices=(Device.DEFAULT,Device.DEFAULT+':1')
    reference=GatedDeltaNetBlock(config,config.ssm) if recurrent else TransformerBlock(config)
    parallel=GatedDeltaNetBlock(config,config.ssm) if recurrent else TransformerBlock(config)
    for name,weight in nn.state.get_state_dict(reference).items():
      value=-Tensor.rand(*weight.shape) if name=='ssm_a' else weight if 'norm' in name else Tensor.randn(*weight.shape)*.05
      weight.replace(value.half().realize())
    with tempfile.TemporaryDirectory() as folder:
      path=pathlib.Path(folder)/'block.gguf'
      path.write_bytes(test_gguf.TestGGUF._build_gguf([(name,tuple(w.shape),1,w.numpy().tobytes())
        for name,w in nn.state.get_state_dict(reference).items()],[('general.architecture','qwen35')]))
      _,state=gguf_load(path,loader=gguf_sharder(devices))
      install(parallel,state)
    self.assertEqual(parallel.config,reference.config)
    for name,value in nn.state.get_state_dict(parallel).items():
      np.testing.assert_array_equal(value.to(Device.DEFAULT).numpy(),nn.state.get_state_dict(reference)[name].numpy(),err_msg=name)
    ref_jit,tp_jit=TinyJit(reference),TinyJit(parallel)
    for pos,tokens in ((0,3),(3,1),(4,1),(5,1),(6,1),(0,2)):
      width=32 if kernel else 8
      start=UOp.variable('start_pos',0,config.max_context-1).bind(pos)
      x=Tensor.randn(1,width if tokens>1 else 1,config.dim).realize()
      if tokens>1:x=x[:,:UOp.variable('toks',1,width).bind(tokens)]
      expected=(reference(x,start) if tokens>1 else ref_jit(x,start)).pad_to((1,width if tokens>1 else 1,config.dim)).numpy()[:,:tokens]
      out=parallel(replicate(x,devices),start) if tokens>1 else tp_jit(replicate(x,devices),start)
      out=out.pad_to((1,width if tokens>1 else 1,config.dim)).realize()
      self.assertIsNone(out.uop.axis)
      for rank in (0,1):
        np.testing.assert_allclose(Tensor(out.uop.mselect(rank)).numpy()[:,:tokens],expected,rtol=3e-4,atol=3e-4,err_msg=f"{pos=} {tokens=} {rank=}")

  def test_attention(self): self._check(False)
  def test_gated_delta(self): self._check(True)
  def test_attention_kernel(self):
    if not amd_custom_kernels_supported(Device.DEFAULT): self.skipTest('RDNA3 required')
    self._check(False,True)
  def test_gated_delta_kernel(self):
    if not amd_custom_kernels_supported(Device.DEFAULT): self.skipTest('RDNA3 required')
    self._check(True,True)

if __name__=='__main__': unittest.main()
