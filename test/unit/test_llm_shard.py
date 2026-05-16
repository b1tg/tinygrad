import unittest
import struct
from tinygrad import Tensor, dtypes, nn
from tinygrad.llm.model import ExpertWeights, SSMConfig, Transformer, TransformerConfig, _expert_ffn_q4k, shard_weights

def cfg(**kw):
  return TransformerConfig(**{
    "num_blocks": 1, "dim": 8, "hidden_dim": 16, "n_heads": 2, "n_kv_heads": 2, "norm_eps": 1e-5,
    "vocab_size": 32, "head_dim": 4, "rope_theta": 10000.0, "rope_dim": 4, "v_head_dim": 4, "max_context": 8,
  } | kw)

class TestLLMShard(unittest.TestCase):
  def test_expert_weight_tensor_shard_matches_unsharded(self):
    devs = ("CPU:0", "CPU:1")
    w = Tensor.arange(4*16*8, device="CPU").reshape(4, 16, 8).float()
    sel = Tensor([[[1, 3]]], dtype="int32", device="CPU")
    x = Tensor.arange(8, device="CPU").reshape(1, 1, 1, 8).float()
    ref = ExpertWeights(4, 8, 16)
    ref.weight.replace(w)
    expected = ref(sel, x).realize()
    for axis in (1, 2):
      sharded = ExpertWeights(4, 8, 16)
      sharded.weight.replace(w.shard(devs, axis=axis))
      self.assertEqual(float((sharded(sel, x).to("CPU") - expected).abs().max().item()), 0.0)

  def test_expert_weight_q4k_raw_tensor_shard(self):
    def block(q):
      b = bytearray(144)
      b[0:2], b[2:4], b[4:8], b[8:12], b[12:16] = struct.pack("<e", 1.0), struct.pack("<e", 0.0), bytes([1]*4), bytes(4), bytes([1]*4)
      b[16:144] = bytes([q | (q << 4)] * 128)
      return bytes(b)
    devs = ("CPU:0", "CPU:1")
    raw0 = Tensor(block(1), dtype=dtypes.uint8, device="CPU").to(devs[0]).realize()
    raw1 = Tensor(block(2), dtype=dtypes.uint8, device="CPU").to(devs[1]).realize()
    sharded = ExpertWeights(1, 256, 2)
    sharded.weight.replace(Tensor.zeros(1, 2, 256).shard(devs, axis=1))
    sharded.weight._gguf_raw = Tensor(raw0.uop.mstack(raw1.uop).multi(0))
    sharded.weight._gguf_type = 12
    out = sharded(Tensor([[[0]]], dtype="int32"), Tensor.ones(1, 1, 1, 256)).to("CPU")
    self.assertEqual(out.tolist(), [[[[256.0, 512.0]]]])

  def test_expert_weight_q5k_raw_tensor_shard(self):
    def block(q, high=False):
      b = bytearray(176)
      b[0:2], b[2:4], b[4:8], b[8:12], b[12:16] = struct.pack("<e", 1.0), struct.pack("<e", 0.0), bytes([1]*4), bytes(4), bytes([1]*4)
      b[16:48], b[48:176] = bytes([0xff if high else 0] * 32), bytes([q | (q << 4)] * 128)
      return bytes(b)
    devs = ("CPU:0", "CPU:1")
    raw0 = Tensor(block(1), dtype=dtypes.uint8, device="CPU").to(devs[0]).realize()
    raw1 = Tensor(block(1, True), dtype=dtypes.uint8, device="CPU").to(devs[1]).realize()
    sharded = ExpertWeights(1, 256, 2)
    sharded.weight.replace(Tensor.zeros(1, 2, 256).shard(devs, axis=1))
    sharded.weight._gguf_raw = Tensor(raw0.uop.mstack(raw1.uop).multi(0))
    sharded.weight._gguf_type = 13
    out = sharded(Tensor([[[0]]], dtype="int32"), Tensor.ones(1, 1, 1, 256)).to("CPU")
    self.assertEqual(out.tolist(), [[[[256.0, 4352.0]]]])

  def test_expert_weight_iq4xs_raw_tensor_shard(self):
    def block(q):
      b = bytearray(136)
      b[0:2], b[2:4], b[4:8], b[8:136] = struct.pack("<e", 1.0), struct.pack("<H", 0xAAAA), bytes([0x11]*4), bytes([q | (q << 4)] * 128)
      return bytes(b)
    devs = ("CPU:0", "CPU:1")
    raw0 = Tensor(block(8), dtype=dtypes.uint8, device="CPU").to(devs[0]).realize()
    raw1 = Tensor(block(9), dtype=dtypes.uint8, device="CPU").to(devs[1]).realize()
    sharded = ExpertWeights(1, 256, 2)
    sharded.weight.replace(Tensor.zeros(1, 2, 256).shard(devs, axis=1))
    sharded.weight._gguf_raw = Tensor(raw0.uop.mstack(raw1.uop).multi(0))
    sharded.weight._gguf_type = 23
    out = sharded(Tensor([[[0]]], dtype="int32"), Tensor.ones(1, 1, 1, 256)).to("CPU")
    self.assertEqual(out.tolist(), [[[[256.0, 3328.0]]]])

  def test_expert_weight_q4k_fused_tensor_shard(self):
    def block(q):
      b = bytearray(144)
      b[0:2], b[2:4], b[4:8], b[8:12], b[12:16] = struct.pack("<e", 1.0), struct.pack("<e", 0.0), bytes([1]*4), bytes(4), bytes([1]*4)
      b[16:144] = bytes([q | (q << 4)] * 128)
      return bytes(b)
    devs, sel, x = ("CPU:0", "CPU:1"), Tensor([[[0]]], dtype="int32"), Tensor.ones(1, 1, 1, 256)
    gate, up = ExpertWeights(1, 256, 2), ExpertWeights(1, 256, 2)
    for ew, q0, q1 in ((gate, 1, 2), (up, 3, 4)):
      raw0 = Tensor(block(q0), dtype=dtypes.uint8, device="CPU").to(devs[0]).realize()
      raw1 = Tensor(block(q1), dtype=dtypes.uint8, device="CPU").to(devs[1]).realize()
      ew.weight.replace(Tensor.zeros(1, 2, 256).shard(devs, axis=1))
      ew.weight._gguf_raw = Tensor(raw0.uop.mstack(raw1.uop).multi(0))
      ew.weight._gguf_type = 12
    expected = (gate(sel, x).silu() * up(sel, x)).to("CPU").realize()
    self.assertLess(float((_expert_ffn_q4k(gate, up, sel, x).to("CPU") - expected).abs().max().item()), 1e-3)

  def test_expert_weight_q4k_raw_input_shard(self):
    def block(q):
      b = bytearray(144)
      b[0:2], b[2:4], b[4:8], b[8:12], b[12:16] = struct.pack("<e", 1.0), struct.pack("<e", 0.0), bytes([1]*4), bytes(4), bytes([1]*4)
      b[16:144] = bytes([q | (q << 4)] * 128)
      return bytes(b)
    devs = ("CPU:0", "CPU:1")
    raw0 = Tensor(block(1), dtype=dtypes.uint8, device="CPU").to(devs[0]).realize()
    raw1 = Tensor(block(2), dtype=dtypes.uint8, device="CPU").to(devs[1]).realize()
    sharded = ExpertWeights(1, 512, 1)
    sharded.weight.replace(Tensor.zeros(1, 1, 512).shard(devs, axis=2))
    sharded.weight._gguf_raw = Tensor(raw0.uop.mstack(raw1.uop).multi(0))
    sharded.weight._gguf_type = 12
    out = sharded(Tensor([[[0]]], dtype="int32"), Tensor.ones(1, 1, 1, 512)).to("CPU")
    self.assertEqual(out.tolist(), [[[[768.0]]]])

  def test_dense_shard_axes(self):
    devs = ("NULL:1", "NULL:2")
    model = Transformer(cfg())
    shard_weights(model, devs)
    sd = nn.state.get_state_dict(model)
    self.assertEqual((sd["token_embd.weight"].device, sd["token_embd.weight"].uop.axis), (devs[0], None))
    self.assertEqual((sd["output.weight"].device, sd["output.weight"].uop.axis), (devs, 0))
    self.assertEqual((sd["blk.0.attn_q.weight"].device, sd["blk.0.attn_q.weight"].uop.axis), (devs, 0))
    for k in ("blk.0.attn_k.weight", "blk.0.attn_v.weight"):
      self.assertEqual((sd[k].device, sd[k].uop.axis), (devs, None))
    self.assertEqual((sd["blk.0.attn_output.weight"].device, sd["blk.0.attn_output.weight"].uop.axis), (devs, 1))
    self.assertEqual((sd["blk.0.ffn_down.weight"].device, sd["blk.0.ffn_down.weight"].uop.axis), (devs, 1))
    for k in ("blk.0.ffn_gate.weight", "blk.0.ffn_up.weight"): self.assertEqual((sd[k].device, sd[k].uop.axis), (devs, 0))
    self.assertEqual((sd["blk.0.attn_norm.weight"].device, sd["blk.0.attn_norm.weight"].uop.axis), (devs, None))

  def test_qk_norm_stays_layerwise(self):
    devs = ("NULL:1", "NULL:2")
    model = Transformer(cfg(qk_norm=4))
    shard_weights(model, devs)
    sd = nn.state.get_state_dict(model)
    for k in ("blk.0.attn_q_norm.weight", "blk.0.attn_k_norm.weight"):
      self.assertEqual((sd[k].device, sd[k].uop.axis), (devs, None))

  def test_mla_shard_axes(self):
    devs = ("NULL:1", "NULL:2")
    model = Transformer(cfg(kv_lora_rank=4, rope_dim=2))
    shard_weights(model, devs)
    sd = nn.state.get_state_dict(model)
    for k in ("blk.0.attn_q.weight", "blk.0.attn_k_b.weight", "blk.0.attn_v_b.weight"):
      self.assertEqual((sd[k].device, sd[k].uop.axis), (devs, 0))
    self.assertEqual((sd["blk.0.attn_output.weight"].device, sd["blk.0.attn_output.weight"].uop.axis), (devs, 1))
    for k in ("blk.0.attn_kv_a_mqa.weight", "blk.0.attn_kv_a_norm.weight", "blk.0.attn_norm.weight"):
      self.assertEqual((sd[k].device, sd[k].uop.axis), (devs, None))

  def test_mla_q_lora_shard_axes(self):
    devs = ("NULL:1", "NULL:2")
    model = Transformer(cfg(kv_lora_rank=4, q_lora_rank=2, rope_dim=2))
    shard_weights(model, devs)
    sd = nn.state.get_state_dict(model)
    self.assertEqual((sd["blk.0.attn_q_a.weight"].device, sd["blk.0.attn_q_a.weight"].uop.axis), (devs, None))
    self.assertEqual((sd["blk.0.attn_q_b.weight"].device, sd["blk.0.attn_q_b.weight"].uop.axis), (devs, 0))

  def test_moe_shard_axes(self):
    devs = ("NULL:1", "NULL:2")
    model = Transformer(cfg(num_experts=4, num_experts_per_tok=2, shared_expert_dim=16))
    shard_weights(model, devs)
    sd = nn.state.get_state_dict(model)
    for k in ("blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight"):
      self.assertEqual((sd[k].device, sd[k].uop.axis), (devs, 1))
    self.assertEqual((sd["blk.0.ffn_down_exps.weight"].device, sd["blk.0.ffn_down_exps.weight"].uop.axis), (devs, 2))
    self.assertEqual((sd["blk.0.ffn_down_shexp.weight"].device, sd["blk.0.ffn_down_shexp.weight"].uop.axis), (devs, 1))
    self.assertEqual((sd["blk.0.ffn_gate_inp.weight"].device, sd["blk.0.ffn_gate_inp.weight"].uop.axis), (devs, None))
    for k in ("blk.0.ffn_gate_shexp.weight", "blk.0.ffn_up_shexp.weight"):
      self.assertEqual((sd[k].device, sd[k].uop.axis), (devs, 0))

  def test_moe_shard_axes_without_experts(self):
    devs = ("NULL:1", "NULL:2")
    model = Transformer(cfg(num_experts=4, num_experts_per_tok=2, shared_expert_dim=16))
    shard_weights(model, devs, expert_axis=None)
    sd = nn.state.get_state_dict(model)
    for k in ("blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight", "blk.0.ffn_down_exps.weight"):
      self.assertEqual((sd[k].device, sd[k].uop.axis), (devs, None))
    self.assertEqual((sd["blk.0.ffn_down_shexp.weight"].device, sd["blk.0.ffn_down_shexp.weight"].uop.axis), (devs, 1))

  def test_ssm_shard_axes(self):
    devs = ("NULL:1", "NULL:2")
    model = Transformer(cfg(ssm=SSMConfig(conv_kernel=2, state_size=4, group_count=2, time_step_rank=2, inner_size=8), full_attention_interval=2))
    shard_weights(model, devs)
    sd = nn.state.get_state_dict(model)
    for k in ("blk.0.attn_gate.weight", "blk.0.ssm_out.weight", "blk.0.ssm_alpha.weight", "blk.0.ssm_beta.weight"):
      self.assertEqual((sd[k].device, sd[k].uop.axis), (devs[0], None))
    for k in ("blk.0.attn_qkv.weight", "blk.0.ssm_conv1d.weight"):
      self.assertEqual((sd[k].device, sd[k].uop.axis), (devs[0], None))

if __name__ == "__main__":
  unittest.main()
