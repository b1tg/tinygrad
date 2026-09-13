import unittest, pathlib, tempfile, itertools
import numpy as np
from tinygrad import Tensor, Device, TinyJit, UOp, Context
from tinygrad.device import MultiBuffer
from tinygrad.nn.state import get_state_dict, load_state_dict
from tinygrad.llm.model import Transformer, TransformerConfig, SSMConfig, IndexerConfig
from tinygrad.llm.shard import ShardedBlock, _linear, _fused_gate_up_exps, shard_model
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

  def test_kda_blocks_and_reset(self):
    c = TransformerConfig(num_blocks=1, dim=256, hidden_dim=512, n_heads=4, n_kv_heads=4, norm_eps=1e-6, vocab_size=512,
      head_dim=64, rope_theta=10000, rope_dim=64, v_head_dim=64, max_context=32, num_experts=4, num_experts_per_tok=2,
      norm_topk_prob=True, ssm=SSMConfig(4,64,4,4,256,kda=True), ssm_layers=(True,))
    block = Transformer(c).blk[0]
    for name,t in get_state_dict(block).items():
      t.replace(Tensor.randn(*t.shape)*0.03 if 'norm' not in name else Tensor.ones(*t.shape)).realize()
    parallel = ShardedBlock(block, self.devices)
    jit = TinyJit(lambda x,pos: parallel(x,pos).to(self.devices[0]).realize())
    for pos,count in ((0,3),(3,1),(4,1),(5,1),(0,1)):
      x,sp = Tensor.randn(1,count,c.dim).realize(), UOp.variable('start_pos',0,31).bind(pos)
      expected = block(x,sp).numpy()
      actual = (jit(x,sp) if count == 1 else parallel(x,sp).to(self.devices[0])).numpy()
      np.testing.assert_allclose(actual,expected,atol=3e-4,rtol=3e-4)

  def test_mla_blocks(self):
    c = TransformerConfig(num_blocks=1, dim=256, hidden_dim=512, n_heads=4, n_kv_heads=1, norm_eps=1e-6, vocab_size=512,
      head_dim=64, rope_theta=10000, rope_dim=32, v_head_dim=32, max_context=32, num_experts=4, num_experts_per_tok=2,
      norm_topk_prob=True, shared_expert_dim=256, kv_lora_rank=64, q_lora_rank=0)
    model = Transformer(c)
    for name,t in get_state_dict(model).items():
      t.replace(Tensor.randn(*t.shape)*0.03 if 'norm' not in name else Tensor.ones(*t.shape)).realize()
    block = model.blk[0]
    sharded = ShardedBlock(block, self.devices)
    def run(x, pos): return sharded(x, pos).to(self.devices[0]).realize()
    jit = TinyJit(run)
    for pos, count in ((0,3), (3,1), (4,1), (0,1)):
      x = Tensor.randn(1,count,c.dim).realize()
      sp = UOp.variable('start_pos', 0, 31).bind(pos)
      expected = block(x, sp).numpy()
      actual = (jit(x, sp) if count == 1 else run(x, sp)).numpy()
      np.testing.assert_allclose(actual, expected, atol=3e-4, rtol=3e-4)

  def test_moe_gather_ffn(self):
    # gather_ffn shards experts on their output axis and all-gathers the hidden activation before down
    c = TransformerConfig(num_blocks=2, dim=256, hidden_dim=512, n_heads=4, n_kv_heads=2, norm_eps=1e-6, vocab_size=512,
      head_dim=64, rope_theta=10000, rope_dim=64, v_head_dim=64, max_context=32, num_experts=4, num_experts_per_tok=2,
      norm_topk_prob=True, shared_expert_dim=256)
    model = Transformer(c)
    for name,t in get_state_dict(model).items():
      t.replace(Tensor.randn(*t.shape)*0.03 if 'norm' not in name else Tensor.ones(*t.shape)).realize()
    for block in model.blk:
      sharded = ShardedBlock(block, self.devices, gather_ffn=True)
      def run(x, pos, sharded=sharded): return sharded(x, pos).to(self.devices[0]).realize()
      jit = TinyJit(run)
      for pos, count in ((0,3), (3,1), (4,1), (5,1), (0,1)):
        x = Tensor.randn(1, count, c.dim).realize()
        sp = UOp.variable('start_pos', 0, 31).bind(pos)
        expected = block(x, sp).numpy()
        actual = (jit(x, sp) if count == 1 else run(x, sp)).numpy()
        np.testing.assert_allclose(actual, expected, atol=2e-3, rtol=2e-3)

  def test_glm_hyperconnection_indexer_block(self):
    # glm5next: MLA + KPool indexer + hyper-connections + MoE + KDA. hc makes the stream (B,T,hc,D).
    c = TransformerConfig(num_blocks=1, dim=64, hidden_dim=128, n_heads=4, n_kv_heads=1, norm_eps=1e-6, vocab_size=512,
      head_dim=16, rope_theta=10000, rope_dim=0, v_head_dim=8, max_context=16, num_experts=4, num_experts_per_tok=2,
      norm_topk_prob=True, shared_expert_dim=64, q_lora_rank=16, kv_lora_rank=8,
      indexer=IndexerConfig(top_k=4, head_dim=4, n_heads=2, kpool=2), hc_mult=2, hc_eps=1e-6, hc_sinkhorn_iters=3)
    for gather_ffn in (False, True):
      model = Transformer(c)
      for name,t in get_state_dict(model).items():
        t.replace(Tensor.randn(*t.shape)*0.03 if 'norm' not in name else Tensor.ones(*t.shape)).realize()
      block = model.blk[0]
      sharded = ShardedBlock(block, self.devices, gather_ffn=gather_ffn)
      def run(x, pos, sharded=sharded): return sharded(x, pos).to(self.devices[0]).realize()
      jit = TinyJit(run)
      for pos, count in ((0,3), (3,1), (4,1), (0,1)):
        x = Tensor.randn(1, count, c.dim).realize()
        xh = x.unsqueeze(2).expand(1, count, c.hc_mult, c.dim).contiguous()
        sp = UOp.variable('start_pos', 0, 15).bind(pos)
        expected = block(xh, sp).numpy()
        actual = (jit(xh, sp) if count == 1 else run(xh, sp)).numpy()
        np.testing.assert_allclose(actual, expected, atol=2e-3, rtol=2e-3)

  def test_hyperconnection_combine_kernel(self):
    # the fused softmax+sinkhorn kernel must match the reference tensor-op computation
    import functools
    from tinygrad.llm.model import _hc_combine_kernel
    rng = np.random.default_rng(0)
    for hc, iters, eps in ((2,3,1e-6), (4,20,1e-6), (4,1,1e-3), (3,5,1e-4)):
      with self.subTest(hc=hc, iters=iters):
        logits_np = rng.normal(size=(2, 5, hc, hc)).astype(np.float32)
        logits = Tensor(logits_np).realize()
        out = Tensor.custom_kernel(Tensor.empty(*logits.shape, dtype='float32', device=logits.device), logits.contiguous(),
                                   fxn=functools.partial(_hc_combine_kernel, hc=hc, iters=iters, eps=eps))[0].numpy()
        m = np.exp(logits_np - logits_np.max(-1, keepdims=True))
        m = m / m.sum(-1, keepdims=True) + eps
        m = m / (m.sum(-2, keepdims=True) + eps)
        for _ in range(1, iters):
          m = m / (m.sum(-1, keepdims=True) + eps)
          m = m / (m.sum(-2, keepdims=True) + eps)
        np.testing.assert_allclose(out, m, rtol=1e-5, atol=1e-6)

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

  def test_fused_experts_do_not_allocate_separate_weights(self):
    if not amd_custom_kernels_supported(Device.DEFAULT): self.skipTest('RDNA3 required')
    from unittest.mock import patch
    c = TransformerConfig(num_blocks=1, dim=256, hidden_dim=512, n_heads=4, n_kv_heads=2, norm_eps=1e-6, vocab_size=512,
      head_dim=64, rope_theta=10000, rope_dim=64, v_head_dim=64, max_context=32, num_experts=4, num_experts_per_tok=2,
      norm_topk_prob=True)
    block = Transformer(c).blk[0]
    rng = np.random.default_rng(42)
    for name,t in get_state_dict(block).items():
      t.replace(Tensor.randn(*t.shape)*0.03 if 'norm' not in name else Tensor.ones(*t.shape)).realize()
    for layer in (block.ffn_gate_exps, block.ffn_up_exps):
      packed = rng.integers(0,256,(4*512*256//256,136),dtype=np.uint8)
      packed[:,:2] = np.array([0.001],dtype=np.float16).view(np.uint8)
      raw = Tensor(np.pad(packed.flatten(),(4,0))).realize()[4:]
      layer.weight = ggml_data_to_tensor(raw,4*512*256,23).reshape(4,512,256).half()
    with patch('tinygrad.llm.shard._linear', wraps=_linear) as linear:
      parallel = ShardedBlock(block,self.devices)
    for call in linear.call_args_list:
      self.assertIsNot(call.args[0],block.ffn_gate_exps)
      self.assertIsNot(call.args[0],block.ffn_up_exps)
    for local in parallel.blocks:
      self.assertTrue(hasattr(local,'ffn_gateup_exps'))
      self.assertFalse(hasattr(local,'ffn_gate_exps'))
      self.assertFalse(hasattr(local,'ffn_up_exps'))
    jit = TinyJit(lambda x,pos: parallel(x,pos).to(self.devices[0]).realize())
    for pos,count in ((0,3),(3,1),(4,1),(5,1)):
      x,sp = Tensor.randn(1,count,c.dim).realize(), UOp.variable('start_pos',0,31).bind(pos)
      expected = block(x,sp).numpy()
      actual = (jit(x,sp) if count == 1 else parallel(x,sp).to(self.devices[0])).numpy()
      np.testing.assert_allclose(actual,expected,atol=3e-4,rtol=3e-4)

  def test_staged_packed_shards_release_cpu_caches(self):
    if not amd_custom_kernels_supported(Device.DEFAULT): self.skipTest('RDNA3 required')
    ne, no, ni = 2, 32, 512
    packed = np.random.default_rng(42).integers(0, 256, (ne, no, ni//256, 144), dtype=np.uint8)
    layer = ExpertWeights(ne, ni, no)
    raw = Tensor(packed.flatten(), device='CPU').realize()
    layer.weight = ggml_data_to_tensor(raw, ne*no*ni, 12).reshape(ne, no, ni).half()
    for i, device in enumerate(self.devices):
      pending = []
      local = _linear(layer, device, cols=(i*256, (i+1)*256), pending=pending)
      self.assertTrue(pending)
      pending[0].realize(*pending[1:])
      np.testing.assert_equal(local.weight.bitcast('uint8').numpy(), packed[:, :, i:i+1].flatten())
      # Later ranks used to inherit the source layer's cached CPU backing via copy.copy(layer).
      self.assertTrue(all(t.device == device for t in get_state_dict(local).values()))

  def test_packed_expert_rows_and_columns(self):
    if not amd_custom_kernels_supported(Device.DEFAULT): self.skipTest('RDNA3 required')
    rng = np.random.default_rng(42)
    ne, no, ni = 4, 64, 512
    for typ, size in ((12,144), (13,176), (14,210), (23,136), (17,74), (18,98)):
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

  def _disk_tensor(self, blob: bytes, prefix: bytes, name: str) -> tuple[Tensor, pathlib.Path]:
    # TIP: `Tensor(path)` keeps the file on the DISK device; slicing it gives the same SHRINK views the streamed loader builds
    p = pathlib.Path(tempfile.mkdtemp()) / name
    p.write_bytes(prefix + blob)
    return Tensor(p)[len(prefix):], p

  def test_packed_disk_expert_rows(self):
    # gather_ffn reads only this rank's byte spans out of the packed DISK tensor (output rows are contiguous quant blocks)
    if not amd_custom_kernels_supported(Device.DEFAULT): self.skipTest('RDNA3 required')
    rng, prefix, ne, no, ni = np.random.default_rng(42), b'GGUF'*64, 4, 64, 512
    for typ, size in ((12,144), (23,136), (17,74), (18,98)):
      packed = rng.integers(0, 256, (ne*no*ni//256, size), dtype=np.uint8)
      packed[:, -2:] = np.array([0.001], dtype=np.float16).view(np.uint8)
      packed[:, :2] = np.array([0.001], dtype=np.float16).view(np.uint8)
      if typ == 12: packed[:, 2:4] = np.array([0.0002], dtype=np.float16).view(np.uint8)
      disk, p = self._disk_tensor(packed.tobytes(), prefix, f'{typ}.bin')
      try:
        dl = ExpertWeights(ne, ni, no)
        dl.weight = ggml_data_to_tensor(disk, ne*no*ni, typ).reshape(ne,no,ni).half()
        raw = Tensor(np.pad(packed.flatten(), (4,0))).to(Device.DEFAULT).realize()[4:]
        ref = ExpertWeights(ne, ni, no)
        ref.weight = ggml_data_to_tensor(raw, ne*no*ni, typ).reshape(ne,no,ni).half()
        self.assertEqual(_linear(ref, self.devices[0], [(0, no//2)]).ggml_type, typ)
        self.assertEqual(_linear(dl, self.devices[0], [(0, no//2)]).ggml_type, typ)
        sel, x = Tensor([[[3,1]]], dtype='int32').realize(), Tensor.randn(1,1,2,ni).realize()
        actual = Tensor.cat(*[_linear(dl, d, [(i*no//2,(i+1)*no//2)])(sel.to(d), x.to(d)).to(self.devices[0])
                              for i,d in enumerate(self.devices)], dim=-1)
        np.testing.assert_allclose(actual.numpy(), ref(sel, x).numpy(), atol=1e-4, rtol=1e-4)
      finally: p.unlink()

  def test_packed_disk_fused_gate_up(self):
    if not amd_custom_kernels_supported(Device.DEFAULT): self.skipTest('RDNA3 required')
    rng, prefix, E, no, ni, typ, size = np.random.default_rng(7), b'GGUF'*64, 4, 64, 512, 23, 136
    def make() -> np.ndarray:
      pk = rng.integers(0, 256, (E*no*ni//256, size), dtype=np.uint8)
      pk[:, -2:] = np.array([0.001], dtype=np.float16).view(np.uint8)
      pk[:, :2] = np.array([0.001], dtype=np.float16).view(np.uint8)
      return pk
    pk_g, pk_u = make(), make()
    disk, p = self._disk_tensor(pk_g.tobytes() + pk_u.tobytes(), prefix, 'fused.bin')
    try:
      gate, up = ExpertWeights(E, ni, no), ExpertWeights(E, ni, no)
      gate.weight = ggml_data_to_tensor(disk[:pk_g.nbytes], E*no*ni, typ).reshape(E,no,ni).half()
      up.weight = ggml_data_to_tensor(disk[pk_g.nbytes:], E*no*ni, typ).reshape(E,no,ni).half()
      rg, ru = ExpertWeights(E, ni, no), ExpertWeights(E, ni, no)
      rg.weight = ggml_data_to_tensor(Tensor(np.pad(pk_g.flatten(), (4,0))).to(Device.DEFAULT).realize()[4:], E*no*ni, typ).reshape(E,no,ni).half()
      ru.weight = ggml_data_to_tensor(Tensor(np.pad(pk_u.flatten(), (4,0))).to(Device.DEFAULT).realize()[4:], E*no*ni, typ).reshape(E,no,ni).half()
      sel, x = Tensor([[[3,1]]], dtype='int32').realize(), Tensor.randn(1,1,2,ni).realize()
      outs = []
      for rank, d in enumerate(self.devices):
        fused = _fused_gate_up_exps(gate, up, d, *[(0, no//2), (no//2, no)][rank])
        assert fused is not None
        outs.append(fused(sel.to(d), x.to(d)).to(self.devices[0]))
      np.testing.assert_allclose(Tensor.cat(outs[0][...,:no//2], outs[1][...,:no//2], dim=-1).numpy(), rg(sel, x).numpy(), atol=1e-4, rtol=1e-4)
      np.testing.assert_allclose(Tensor.cat(outs[0][...,no//2:], outs[1][...,no//2:], dim=-1).numpy(), ru(sel, x).numpy(), atol=1e-4, rtol=1e-4)
    finally: p.unlink()

  def test_packed_disk_q8_linear(self):
    # Q8_0 is streamed too: rows go through the disk spans, an in_features split uses the shared cached CPU copy
    if not amd_custom_kernels_supported(Device.DEFAULT): self.skipTest('RDNA3 required')
    rng, prefix, no, ni = np.random.default_rng(3), b'GGUF'*64, 256, 512
    packed = rng.integers(0, 256, (no*ni//32, 34), dtype=np.uint8)
    packed[:, :2] = np.array([0.001], dtype=np.float16).view(np.uint8)
    disk, p = self._disk_tensor(packed.tobytes(), prefix, 'q8.bin')
    try:
      dl = Linear(ni, no, bias=False)
      dl.weight = ggml_data_to_tensor(disk, no*ni, 8).reshape(no, ni).half()
      raw = Tensor(np.pad(packed.flatten(), (2,0))).to(Device.DEFAULT).realize()[2:]
      ref = Linear(ni, no, bias=False)
      ref.weight = ggml_data_to_tensor(raw, no*ni, 8).reshape(no, ni).half()
      x = Tensor.randn(1,3,ni).realize()
      local = [_linear(dl, d, [(i*no//2,(i+1)*no//2)]) for i,d in enumerate(self.devices)]
      self.assertTrue(all(l._q8_0_weight is not None for l in local))
      actual = Tensor.cat(*[l(x.to(d)).to(self.devices[0]) for l,d in zip(local,self.devices)], dim=-1)
      np.testing.assert_allclose(actual.numpy(), ref(x).numpy(), atol=1e-4, rtol=1e-4)
      local = [_linear(dl, d, None, (i*ni//2,(i+1)*ni//2)) for i,d in enumerate(self.devices)]
      self.assertTrue(all(l._q8_0_weight is not None for l in local))
      outputs = [l(x[...,i*ni//2:(i+1)*ni//2].to(d)).to(self.devices[0]) for i,(l,d) in enumerate(zip(local,self.devices))]
      np.testing.assert_allclose((outputs[0]+outputs[1]).numpy(), ref(x).numpy(), atol=1e-4, rtol=1e-4)
    finally: p.unlink()

class TestEmbeddingMemory(unittest.TestCase):
  def test_vocab_split_storage_and_lookup(self):
    # Match the five-way layout without requiring five physical GPUs in the regression test.
    devices = tuple(f'CPU:{i}' if i else 'CPU' for i in range(5))
    with Context(DEV='CPU'):
      c = TransformerConfig(num_blocks=0, dim=16, hidden_dim=32, n_heads=2, n_kv_heads=2, norm_eps=1e-5,
        vocab_size=60, head_dim=8, rope_theta=10000, rope_dim=8, v_head_dim=8, max_context=16)
      model = Transformer(c)
      values = np.arange(60*16, dtype=np.float32).reshape(60,16)
      model.token_embd.weight = Tensor(values).realize()
      shard_model(model, devices)
      storage = model.token_embd.weight.uop.buffer
      self.assertIsInstance(storage, MultiBuffer)
      self.assertEqual(sum(b.nbytes for b in storage.bufs), values.nbytes)
      self.assertEqual(max(b.nbytes for b in storage.bufs), values.nbytes//5)
      ids = np.array([[0,11,12,23,24], [35,36,47,48,59]], dtype=np.int32)
      np.testing.assert_equal(model.token_embd(Tensor(ids).to(devices)).numpy(), values[ids])

  def test_generate_matches_unsharded(self):
    with Context(DEV='CPU'):
      c = TransformerConfig(num_blocks=1, dim=40, hidden_dim=80, n_heads=5, n_kv_heads=5, norm_eps=1e-5,
        vocab_size=180, head_dim=8, rope_theta=10000, rope_dim=8, v_head_dim=8, max_context=32)
      model, parallel = Transformer(c), Transformer(c)
      Tensor.realize(*get_state_dict(model).values())
      load_state_dict(parallel, get_state_dict(model), verbose=False, realize=False)
      shard_model(parallel, tuple(f'CPU:{i}' if i else 'CPU' for i in range(5)))
      for prompt in ([1,2,3,4,5], [31,32,33]):
        expected = list(itertools.islice(model.generate(list(prompt)), 5))
        self.assertEqual(list(itertools.islice(parallel.generate(list(prompt)), 5)), expected)

if __name__ == '__main__': unittest.main()
