import unittest
from unittest.mock import patch
import numpy as np
from tinygrad import Tensor, TinyJit, UOp, dtypes, nn, function
from tinygrad.llm.kernels.amd import Linear, amd_custom_kernels_supported, q8_quantize, flash_attention, gated_delta_prefill
from tinygrad.llm.kernels.amd import hc_prepare, dsa_decode, amd_topk
from tinygrad.llm.gguf import ggml_data_to_tensor
from tinygrad.llm.model import ExpertWeights

class TestTopK(unittest.TestCase):
  def setUp(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("AMD custom kernels required")

  def test_stable_masked_topk(self):
    rng = np.random.default_rng(53)
    for n, k in ((1, 1), (7, 4), (288, 8), (1024, 512), (2048, 512), (4096, 512)):
      with self.subTest(n=n, k=k):
        data = rng.integers(-8, 9, (2, 3, n)).astype(np.float32)
        data[0, 1] = -np.inf
        data[1, 1] = np.finfo(np.float32).min
        values, indices = amd_topk(Tensor(data), k)
        expected = np.argsort(-data, axis=-1, kind="stable")[..., :k]
        np.testing.assert_array_equal(indices.numpy(), expected)
        np.testing.assert_array_equal(values.numpy(), np.take_along_axis(data, expected, axis=-1))

  def test_router_order(self):
    from tinygrad.llm.model import pairwise_topk
    data = np.random.default_rng(54).integers(-4, 5, (2, 1, 288)).astype(np.float32)
    values, indices = pairwise_topk(Tensor(data), 8)
    expected = np.argsort(-data, axis=-1, kind="stable")[..., :8][..., ::-1]
    np.testing.assert_array_equal(indices.numpy(), expected)
    np.testing.assert_array_equal(values.numpy(), np.take_along_axis(data, expected, axis=-1))

  def test_jit_symbolic_rows(self):
    @TinyJit
    def run(x, rows):
      values, indices = amd_topk(x[:, :rows], 8)
      values, indices = values.pad_to(values.max_shape).contiguous(), indices.pad_to(indices.max_shape).contiguous()
      Tensor.realize(values, indices)
      return values, indices
    rng = np.random.default_rng(55)
    for count in (1, 3, 2, 4, 1):
      data = rng.standard_normal((1, 4, 288)).astype(np.float32)
      rows = UOp.variable("topk_rows", 1, 4).bind(count)
      values, indices = run(Tensor(data).realize(), rows)
      expected = np.argsort(-data[:, :count], axis=-1, kind="stable")[..., :8]
      np.testing.assert_array_equal(indices.numpy()[:, :count], expected)
      np.testing.assert_array_equal(values.numpy()[:, :count], np.take_along_axis(data[:, :count], expected, axis=-1))

class TestExpertGateUp(unittest.TestCase):
  def setUp(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("AMD custom kernels required")

  @staticmethod
  def make_block(width, limit, formats=(21, 21)):
    from tinygrad.llm.model import FFNBlock, TransformerConfig
    block = FFNBlock(TransformerConfig(num_blocks=1, dim=width, hidden_dim=16, n_heads=1, n_kv_heads=1,
      norm_eps=1e-5, vocab_size=1, head_dim=1, rope_theta=1., rope_dim=0, v_head_dim=1,
      num_experts=3, num_experts_per_tok=2, swiglu_limit=limit))
    rng = np.random.default_rng(31)
    for weight, typ in zip((block.ffn_gate_exps, block.ffn_up_exps), formats):
      size = {21:110, 23:136}[typ]
      packed = rng.integers(0, 256, (3*16*width//256, size), dtype=np.uint8)
      packed[:, :2] = np.array([0.001], dtype=np.float16).view(np.uint8)
      raw = Tensor(np.pad(packed.flatten(), (4, 0))).realize()[4:]
      weight.weight = ggml_data_to_tensor(raw, 3*16*width, typ).reshape(3, 16, width).half()
    return block

  @staticmethod
  def reference(block, sel, x): return block._swiglu(block.ffn_gate_exps(sel, x), block.ffn_up_exps(sel, x)).contiguous()

  def test_jit_clipping_and_routing(self):
    rng = np.random.default_rng(32)
    for width in (256, 768, 1024, 4096):
      for limit in (0., 10.):
        with self.subTest(width=width, limit=limit):
          block = self.make_block(width, limit)
          run = TinyJit(lambda sel, x: block._expert_swiglu(sel, x).realize())
          for magnitude in (0.1, 1., 20., 0.):
            sel = Tensor(rng.integers(0, 3, (2, 1, 2), dtype=np.int32)).realize()
            x = Tensor((rng.normal(size=(2, 1, 1, width))*magnitude).astype(np.float32)).realize()
            expected = self.reference(block, sel, x).numpy()
            np.testing.assert_allclose(run(sel, x).numpy(), expected, rtol=1e-4, atol=1e-4)
          self.assertEqual(block.ffn_gate_exps.ggml_type, 21)
          self.assertEqual(block.ffn_up_exps.ggml_type, 21)

  def test_prefill_and_mixed_formats(self):
    selected = Tensor(np.array([[[2, 0], [1, 1]]], dtype=np.int32)).realize()
    x = Tensor(np.random.default_rng(33).normal(size=(1, 2, 1, 256)).astype(np.float32)).realize()
    with patch("tinygrad.llm.model.expert_gate_up_silu", side_effect=AssertionError("unexpected fused gate/up")):
      for formats, tokens in (((21, 21), 2), ((21, 23), 1)):
        block = self.make_block(256, 10., formats)
        sel, inp = selected[:, :tokens], x[:, :tokens]
        expected = self.reference(block, sel, inp).numpy()
        np.testing.assert_array_equal(block._expert_swiglu(sel, inp).numpy(), expected)
      block = self.make_block(256, 10.)
      tokens = UOp.variable("gate_up_prefill_tokens", 1, 2).bind(1)
      sel, inp = selected[:, :tokens], x[:, :tokens]
      np.testing.assert_array_equal(block._expert_swiglu(sel, inp)[:, :1].numpy(), self.reference(block, sel, inp)[:, :1].numpy())

class TestQ6Experts(unittest.TestCase):
  def setUp(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("AMD custom kernels required")

  @staticmethod
  def make_weight(width, dtype=dtypes.half):
    packed = np.random.default_rng(14).integers(0, 256, (3*16*width//256, 210), dtype=np.uint8)
    packed[:, 208:] = np.array([0.001], dtype=np.float16).view(np.uint8)
    raw = Tensor(np.pad(packed.flatten(), (4, 0))).realize()[4:]
    weight = ExpertWeights(3, width, 16)
    weight.weight = ggml_data_to_tensor(raw, 3*16*width, 14).reshape(3, 16, width).cast(dtype)
    low = ((packed[:, :128].reshape(-1, 2, 1, 64) >> np.array([0, 4])[None, None, :, None]) & 15).reshape(-1, 256)
    high = ((packed[:, 128:192].reshape(-1, 2, 1, 32) >> np.array([0, 2, 4, 6])[None, None, :, None]) & 3).reshape(-1, 256)
    scales = np.repeat(packed[:, 192:208].view(np.int8), 16, axis=1).astype(np.float32)
    d = packed[:, 208:].copy().view(np.float16).astype(np.float32)
    reference = (d*((low | (high << 4))-32).astype(np.float32)*scales).reshape(3, 16, width)
    # The inline GPU dequantization/matmul retains fp32 intermediates; materializing fp16 weights separately changes its rounding.
    return weight, reference

  def test_decode_jit_routes_and_activations(self):
    rng = np.random.default_rng(15)
    for width in (256, 2048):
      for shared in (True, False):
        with self.subTest(width=width, shared=shared):
          weight, reference = self.make_weight(width)
          run = TinyJit(lambda sel, x: weight(sel, x).realize())
          for _ in range(4):
            selected = rng.integers(0, 3, (2, 1, 2), dtype=np.int32)
            x = rng.normal(size=(2, 1, 1 if shared else 2, width)).astype(np.float32)
            expected = np.matmul(x[..., None, :], reference[selected].swapaxes(-1, -2)).squeeze(-2)
            actual = run(Tensor(selected).realize(), Tensor(x).realize()).numpy()
            np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-4)
          self.assertEqual(weight.ggml_type, 14)
          self.assertEqual(weight.weight.dtype, dtypes.half)
          self.assertEqual(weight._packed_q6.dtype, dtypes.uint8)
          self.assertEqual(weight._packed_q6.nbytes(), 3*16*width//256*210)

  def test_prefill_and_float_weights_keep_generic_path(self):
    rng = np.random.default_rng(16)
    selected = np.array([[[2, 0], [1, 1]]], dtype=np.int32)
    x = rng.normal(size=(1, 2, 2, 256)).astype(np.float32)
    for dtype in (dtypes.half, dtypes.float32):
      weight, reference = self.make_weight(256, dtype)
      original = weight.weight
      expected = np.matmul(x[..., None, :], reference[selected].swapaxes(-1, -2)).squeeze(-2)
      with patch("tinygrad.llm.model.expert_q6_f16_linear", side_effect=AssertionError("unexpected fused decode")):
        np.testing.assert_allclose(weight(Tensor(selected), Tensor(x)).numpy(), expected, rtol=2e-4, atol=2e-4)
        count = UOp.variable("q6_prefill_tokens", 1, 2).bind(1)
        actual = weight(Tensor(selected)[:, :count], Tensor(x)[:, :count])[:, :1].numpy()
        np.testing.assert_allclose(actual, expected[:, :1], rtol=2e-4, atol=2e-4)
        if dtype == dtypes.float32:
          np.testing.assert_allclose(weight(Tensor(selected[:, :1]), Tensor(x[:, :1])).numpy(), expected[:, :1], rtol=2e-4, atol=2e-4)
        else:
          self.assertEqual(weight(Tensor(selected[:, :1]), Tensor(x[:, :1]).half()).realize().dtype, dtypes.half)
      self.assertIs(weight.weight, original)

class TestQ8Grouped(unittest.TestCase):
  def setUp(self):
    from tinygrad.llm.kernels.amd import amd_wmma_kernels_supported
    device = Tensor.empty(1).device
    if not amd_custom_kernels_supported(device) or amd_wmma_kernels_supported(device): self.skipTest("CDNA custom kernels required")

  @staticmethod
  def make_linear(width, outputs):
    packed = np.random.default_rng(8).integers(0, 256, (outputs*width//32, 34), dtype=np.uint8)
    packed[:, :2] = np.array([0.001], dtype=np.float16).view(np.uint8)
    raw = Tensor(np.pad(packed.flatten(), (4, 0))).realize()[4:]
    linear = Linear(width, outputs, bias=False)
    linear.weight = ggml_data_to_tensor(raw, outputs*width, 8).reshape(outputs, width).half()
    reference = (packed[:, 2:].view(np.int8).astype(np.float32) * packed[:, :2].copy().view(np.float16).astype(np.float32))
    return linear, reference.reshape(outputs, width)

  def test_chunk_boundaries_and_jit(self):
    rng = np.random.default_rng(9)
    for width, outputs in ((1024, 2048), (1568, 2052), (4096, 2048), (8192, 2048), (16384, 2048)):
      with self.subTest(width=width, outputs=outputs):
        linear, weights = self.make_linear(width, outputs)
        run = TinyJit(lambda x: linear(x).realize())
        for step in range(4):
          x = rng.normal(size=(1, width)).astype(np.float32) if step != 2 else np.zeros((1, width), dtype=np.float32)
          groups = x.reshape(1, width//32, 32)
          scale = np.maximum(np.abs(groups).max(-1, keepdims=True)/127, 1e-8)
          quantized = (np.clip(np.rint(groups/scale), -127, 127)*scale).reshape(1, width)
          expected = quantized @ weights.T
          np.testing.assert_allclose(run(Tensor(x).realize()).numpy(), expected, rtol=2e-4, atol=2e-4)
        self.assertEqual(linear.ggml_type, 8)

  def test_prefill_and_small_shapes_use_original_path(self):
    with patch("tinygrad.llm.kernels.amd._decode_linear_grouped", side_effect=AssertionError("unexpected grouped decode")):
      for width, outputs, tokens in ((4096, 2048, 2), (16384, 24, 1), (128, 8192, 1), (4096, 2050, 1)):
        linear, _ = self.make_linear(width, outputs)
        out = linear(Tensor.zeros(tokens, width)).numpy()
        np.testing.assert_array_equal(out, np.zeros((tokens, outputs)))
      linear, _ = self.make_linear(4096, 2048)
      count = UOp.variable("q8_prefill_tokens", 1, 2).bind(1)
      np.testing.assert_array_equal(linear(Tensor.zeros(2, 4096)[:count])[:1].numpy(), np.zeros((1, 2048)))

class TestQ8Quantize(unittest.TestCase):
  def test_iq3s_signed_table_exhaustive(self):
    from tinygrad.llm.kernels.amd import iq3s_grid_lut
    base = iq3s_grid_lut("CPU").numpy().view(np.int8).reshape(512, 4).astype(np.int16)
    signs = np.where((np.arange(16)[:, None] >> np.arange(4)) & 1, -1, 1)
    expected = (base[:, None, :] * signs[None]).astype(np.int8)
    table = iq3s_grid_lut("CPU", signed=True)
    words = table.numpy()
    magnitudes, masks = words[:512, None], words[None, 512:]
    packed = (magnitudes ^ masks)+(masks & 0x01010101)
    np.testing.assert_array_equal(packed.view(np.int8).reshape(512, 16, 4), expected)
    self.assertTrue(np.all(words[:512].view(np.uint8) != 0))
    self.assertEqual(table.nbytes(), 2112)

  def test_quant_weights_share_storage(self):
    for ggml_type, type_size in ((12, 144), (13, 176), (14, 210), (23, 136)):
      with self.subTest(ggml_type=ggml_type):
        packed = np.arange(type_size + 4, dtype=np.uint8)
        raw = Tensor(packed, device="CPU").realize()[4:]
        decoded = ggml_data_to_tensor(raw, 256, ggml_type).reshape(1, 256)
        linear = Linear(256, 1, bias=False)
        linear.set_quantized(decoded)
        linear.weight.realize()
        self.assertEqual(linear.weight.dtype, dtypes.uint16 if ggml_type == 14 else dtypes.uint32)
        self.assertEqual(linear.weight.nbytes(), type_size)
        np.testing.assert_array_equal(linear.weight.bitcast(dtypes.uint8).numpy(), packed[4:])
        raw.assign(raw.full_like(1)).realize()
        np.testing.assert_array_equal(linear.weight.bitcast(dtypes.uint8).numpy(), np.ones(type_size, dtype=np.uint8))

  def test_values_and_scales(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    x = np.linspace(-3.1, 2.7, 64, dtype=np.float32).reshape(2, 32)
    quant, scale, gsum = q8_quantize(Tensor(x), 2, 32)
    scale_np = np.maximum(np.max(np.abs(x), axis=-1, keepdims=True) / 127, 1e-8)
    expected = np.clip(np.rint(x / scale_np), -127, 127).astype(np.int8)
    np.testing.assert_array_equal(quant.bitcast(dtypes.int8).reshape(2, 32).numpy(), expected)
    np.testing.assert_allclose(scale.numpy(), scale_np, rtol=1e-6)
    # xsum holds the two per-16 sums per 32-wide group
    np.testing.assert_array_equal(gsum.numpy().reshape(2, 2), expected.reshape(2, 2, 16).sum(-1).astype(np.float32))

  def test_quantize_rounding_ties(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    values = np.array([-127,127]+[i+0.5 for i in range(-15,15)],dtype=np.float32)
    quant,_,_ = q8_quantize(Tensor(values),1,32)
    np.testing.assert_array_equal(quant.bitcast(dtypes.int8).reshape(32).numpy(),np.rint(values).astype(np.int8))

  def test_q6_linear_compiles_in_function(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    rng = np.random.default_rng(42)
    packed = rng.integers(0, 256, 210, dtype=np.uint8)
    packed[-2:] = np.array([0.01], dtype=np.float16).view(np.uint8)
    raw = Tensor(np.pad(packed, (2, 0))).contiguous().realize()[2:]
    decoded = ggml_data_to_tensor(raw, 256, 14).reshape(1, 256)
    linear = Linear(256, 1, bias=False)
    nn.state.load_state_dict(linear, {"weight":decoded}, verbose=False, realize=False)
    @function(allow_implicit=True)
    def run(x:Tensor): return linear(x)
    self.assertTrue(np.isfinite(run(Tensor.randn(1, 256)).realize().item()))
    self.assertEqual(linear.weight.nbytes(), 210)
    self.assertEqual(linear.weight.dtype, dtypes.uint16)

  def test_q4_k_linear(self): self._test_quant_linear(12, 144)
  def test_iq4_linear(self): self._test_quant_linear(23, 136)
  def test_q5_linear(self): self._test_quant_linear(13, 176)
  def test_q6_linear_odd_blocks(self): self._test_quant_linear(14, 210, in_features=768, out_features=3, token_counts=(1, 3, 16))

  def test_quant_linear_partial_output_tile(self):
    # Cover a sub-tile output, a trailing tile, and IQ4's larger-output tile selection.
    for typ, size, outputs, tokens in ((12, 144, 16, 16), (12, 144, 48, 32), (13, 176, 48, 16), (23, 136, 4112, 32)):
      with self.subTest(ggml_type=typ, out_features=outputs):
        self._test_quant_linear(typ, size, in_features=256, out_features=outputs, token_counts=(tokens,))

  def test_quant_linear_preserves_rope_permutation(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    rng = np.random.default_rng(42)
    for typ, size in ((12, 144), (13, 176), (14, 210), (23, 136)):
      with self.subTest(ggml_type=typ):
        packed = rng.integers(0, 256, (16, size), dtype=np.uint8)
        packed[:, -2:] = np.array([0.001], dtype=np.float16).view(np.uint8)
        if typ != 14: packed[:, :2] = np.array([0.001], dtype=np.float16).view(np.uint8)
        if typ in (12, 13): packed[:, 2:4] = np.array([0.0002], dtype=np.float16).view(np.uint8)
        raw = Tensor(np.pad(packed.flatten(), (4, 0))).contiguous().realize()[4:]
        decoded = ggml_data_to_tensor(raw, 16*256, typ).reshape(16, 256).half()
        original = decoded.numpy()
        x = rng.normal(size=(3, 256)).astype(np.float16)
        for prefix in (None, 0, 4):
          with self.subTest(prefix=prefix):
            w = decoded.reshape(2, 8, 256)
            if prefix is None:
              weight = w.rearrange("n (h two) d -> n (two h) d", two=2)
            else:
              weight = w[:, :prefix].cat(w[:, prefix:].rearrange("n (h two) d -> n (two h) d", two=2), dim=1)
            start = prefix or 0
            rows = np.arange(16).reshape(2, 8)
            order = np.concatenate((rows[:, :start], rows[:, start:].reshape(2, -1, 2).transpose(0, 2, 1).reshape(2, -1)), axis=1)
            linear = Linear(256, 16, bias=False)
            linear.weight = weight.reshape(16, 256)
            np.testing.assert_allclose(linear(Tensor(x)).numpy(), x.astype(np.float32) @ original[order.flatten()].astype(np.float32).T,
                                       rtol=3e-3, atol=2e-2)
            self.assertIsNone(linear.ggml_type)

  def test_quant_linear_rejects_unaligned_rows_and_integer_casts(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    for width in (128, 256):
      with self.subTest(width=width):
        packed = np.zeros((2*width//256, 136), dtype=np.uint8)
        packed[:, :2] = np.array([0.001], dtype=np.float16).view(np.uint8)
        packed[:, 8:] = np.arange(128, dtype=np.uint8)
        raw = Tensor(np.pad(packed.flatten(), (4, 0))).realize()[4:]
        weight = ggml_data_to_tensor(raw, 2*width, 23).reshape(2, width)
        if width == 256: weight = weight.int().float()
        expected = weight.numpy().sum(-1)[None]
        linear = Linear(width, 2, bias=False)
        linear.weight = weight
        np.testing.assert_allclose(linear(Tensor.ones(1, width)).numpy(), expected, rtol=1e-3, atol=1e-3)
        self.assertIsNone(linear.ggml_type)

  def test_dense_gemv_preserves_integer_casts(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    linear = Linear(128, 1)
    linear.weight = Tensor.full((1, 128), 0.75).contiguous().realize().int().float()
    linear.bias = Tensor.full((1,), 0.75).contiguous().realize().int().float()
    np.testing.assert_array_equal(linear(Tensor.ones(1, 128)).numpy(), 0)

  def test_dense_gemv_float32_range(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    linear = Linear(128, 1, bias=False)
    linear.weight = Tensor.full((1, 128), 1/128, dtype=dtypes.float32).realize()
    np.testing.assert_array_equal(linear(Tensor.full((1, 128), 65536, dtype=dtypes.float32)).numpy(), 65536)

  def test_gated_delta_state_and_precision(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    for case in ("view", "reset", "half"):
      with self.subTest(case=case):
        q = Tensor.full((1, 1, 1, 32), 256 if case == "half" else 1, dtype=dtypes.half if case == "half" else dtypes.float32)
        state = Tensor.full((1, 1, 32, 4), int(case == "reset"), dtype=dtypes.float32).contiguous().realize().transpose(-1, -2)
        if case != "view": state = state.contiguous().realize()
        start = Tensor(UOp.variable("start_pos", 0, 10).bind(0)) if case == "reset" else None
        beta = Tensor.full((1, 1, 1), 1/2097152 if case == "half" else 1, dtype=dtypes.float32)
        if case != "reset":
          message = "recurrent state must be contiguous" if case == "view" else "recurrent Q/K must be float32"
          with self.assertRaisesRegex(AssertionError, message):
            gated_delta_prefill(q, q, Tensor.ones(1, 1, 1, 4), beta, Tensor.ones(1, 1, 1), state, start)
          continue
        out = gated_delta_prefill(q, q, Tensor.ones(1, 1, 1, 4), beta, Tensor.ones(1, 1, 1), state, start)
        np.testing.assert_array_equal(out.numpy(), 32)
        np.testing.assert_array_equal(state.numpy(), 1)

  def test_dense_gemv_bias(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    rng = np.random.default_rng(42)
    w, bias = rng.normal(size=(32, 128)).astype(np.float16), rng.normal(size=32).astype(np.float16)
    linear = Linear(128, 32)
    linear.weight, linear.bias = Tensor(w), Tensor(bias)
    for tokens in (1, 3):
      with self.subTest(tokens=tokens):
        x = rng.normal(size=(tokens, 128)).astype(np.float16)
        np.testing.assert_allclose(linear(Tensor(x)).numpy(), x.astype(np.float32) @ w.astype(np.float32).T + bias, rtol=2e-3, atol=2e-3)

  def _test_quant_linear(self, ggml_type, block_bytes, in_features=2048, out_features=64, token_counts=(1, 3, 32, 64, 128)):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    rng = np.random.default_rng(42)
    packed = rng.integers(0, 256, (out_features*in_features//256, block_bytes), dtype=np.uint8)
    if ggml_type == 14: packed[:, -2:] = np.array([0.001], dtype=np.float16).view(np.uint8)
    else: packed[:, :2] = np.array([0.001], dtype=np.float16).view(np.uint8)
    if ggml_type in (12, 13): packed[:, 2:4] = np.array([0.0002], dtype=np.float16).view(np.uint8)
    raw = Tensor(np.pad(packed.flatten(), (4, 0))).contiguous().realize()[4:]
    decoded = ggml_data_to_tensor(raw, out_features*in_features, ggml_type).reshape(out_features, in_features)
    weight = decoded.numpy()
    linear = Linear(in_features, out_features, bias=False)
    linear.weight = decoded
    for tokens in token_counts:
      with self.subTest(tokens=tokens):
        x = rng.normal(size=(tokens, in_features)).astype(np.float32 if tokens == 3 else np.float16)
        reference_x = x.astype(np.float32)
        if tokens < 16 or ggml_type == 14:
          grouped = reference_x.reshape(tokens, -1, 32)
          scale = np.maximum(np.abs(grouped).max(-1, keepdims=True) / 127, 1e-8)
          reference_x = (np.clip(np.rint(grouped/scale), -127, 127)*scale).reshape(tokens, in_features)
        reference_w = weight if tokens < 16 or ggml_type == 14 else weight.astype(np.float16).astype(np.float32)
        np.testing.assert_allclose(linear(Tensor(x)).numpy(), reference_x @ reference_w.T, rtol=3e-3, atol=2e-2)
    self.assertEqual(linear.ggml_type, ggml_type)

  def test_q6_linear_multiple_tokens(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    rng = np.random.default_rng(42)
    in_features, blocks = 2048, 16*2048//256
    packed = rng.integers(0, 256, blocks*210, dtype=np.uint8)
    for i in range(blocks): packed[i*210+208:i*210+210] = np.array([0.01], dtype=np.float16).view(np.uint8)
    raw = Tensor(np.pad(packed, (4, 0))).contiguous().realize()[4:]
    decoded = ggml_data_to_tensor(raw, 16*in_features, 14).reshape(16, in_features)
    weight = decoded.numpy()
    linear = Linear(in_features, 16, bias=False)
    nn.state.load_state_dict(linear, {"weight":decoded}, verbose=False, realize=False)
    x = rng.normal(size=(3, in_features)).astype(np.float32)
    scale = np.maximum(np.abs(x).reshape(3, in_features//32, 32).max(-1, keepdims=True) / 127, 1e-8)
    xq = np.clip(np.rint(x.reshape(3, in_features//32, 32) / scale), -127, 127) * scale
    np.testing.assert_allclose(linear(Tensor(x)).numpy(), xq.reshape(3, in_features) @ weight.T, rtol=2e-3, atol=2e-2)
    self.assertEqual(linear.ggml_type, 14)

    # symbolic token counts take the padded kernel path and give the same results
    generic = Linear(in_features, 16, bias=False)
    nn.state.load_state_dict(generic, {"weight":decoded}, verbose=False, realize=False)
    sym = Tensor(np.concatenate([x, np.zeros((1, in_features), np.float32)])).contiguous()[:UOp.variable("tokens", 1, 4).bind(3)]
    np.testing.assert_allclose(generic(sym)[:3].numpy(), xq.reshape(3, in_features) @ weight.T, rtol=2e-3, atol=2e-2)
    self.assertTrue(generic.use_custom_quant)
    self.assertEqual(generic.ggml_type, 14)

  def test_attention_fallback_shapes(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    for tokens, capacity, dim in ((1, 65, 64), (32, 64, 32), (32, 64, 384), (32, 64, 512)):
      with self.subTest(tokens=tokens, capacity=capacity, dim=dim):
        valid = 33
        cache = np.full((2, 1, 1, capacity, dim), np.nan, dtype=np.float16)
        cache[0, :, :, :valid] = 0
        cache[1, :, :, :valid] = np.arange(valid)[:, None]
        q = Tensor.zeros(1, 2, tokens, dim, dtype=dtypes.half)
        expected = np.broadcast_to(np.arange(valid-tokens, valid)[None, None, :, None]/2, q.shape)
        np.testing.assert_allclose(flash_attention(q, Tensor(cache), valid).numpy(), expected, rtol=1e-3, atol=1e-3)

  def test_attention_uses_physical_cache_length(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    q, k, v = Tensor.zeros(1, 2, 1, 32), Tensor.randn(1, 1, 1, 32), Tensor.randn(1, 1, 1, 32)
    cache = Tensor.empty(2, 1, 1, 256, 32, dtype=dtypes.half).contiguous()
    assigned = Tensor(cache.uop.after(cache[:, :, :, 0:1, :].uop.store(Tensor.stack(k, v).cast(dtypes.half).uop)))
    out = flash_attention(q, assigned, 1).realize()
    np.testing.assert_allclose(out.numpy(), v.expand(1, 2, 1, 32).numpy(), rtol=2e-2, atol=2e-2)

  def test_flash_attention_decode_symbolic_gqa(self):
    with patch.object(Tensor, "scaled_dot_product_attention", side_effect=AssertionError("expected custom decode")):
      self._test_flash_decode(8, 2, 256, 128, 37, symbolic=True)

  def test_flash_attention_decode_gqa_tail(self): self._test_flash_decode(3, 1, 192, 64, 37)

  def test_flash_attention_decode_gqa_output_layout(self): self._test_flash_decode(4, 1, 128, 256, 3)
  def test_flash_attention_decode_large_gqa_group(self): self._test_flash_decode(8, 1, 256, 256, 73)

  def _test_flash_decode(self, heads, kv_heads, dim, n, valid, symbolic=False):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    rng = np.random.default_rng(42)
    q = rng.normal(size=(1, heads, 1, dim)).astype(np.float16)
    cache = rng.normal(size=(2, 1, kv_heads, n, dim)).astype(np.float16)
    k, v = (np.repeat(c[0, :, :valid].astype(np.float32), heads//kv_heads, axis=0) for c in cache)
    scores = q[0].astype(np.float32) @ k.transpose(0, 2, 1) / np.sqrt(dim)
    probs = np.exp(scores - scores.max(-1, keepdims=True))
    expected = (probs / probs.sum(-1, keepdims=True)) @ v
    cache_tensor = Tensor(cache)
    if symbolic:
      start_pos = UOp.variable("start_pos", 0, n-1).bind(valid-1)
      valid = start_pos + 1
      cache_tensor = Tensor(cache_tensor.realize().uop.after(Tensor(start_pos).uop))
    np.testing.assert_allclose(flash_attention(Tensor(q), cache_tensor, valid).numpy(), expected[None], rtol=2e-3, atol=2e-3)

  def test_prefill_attention_nonfinite_cache_tail(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    rng = np.random.default_rng(42)
    q = Tensor.zeros(1, 2, 32, 128, dtype=dtypes.half)
    values = rng.normal(size=(33, 128)).astype(np.float16)
    expected = np.stack([values[:i+2].astype(np.float32).mean(0) for i in range(32)])[None, None].repeat(2, axis=1)
    for tail in (np.nan, np.inf, -np.inf):
      with self.subTest(tail=tail):
        cache = np.full((2, 1, 1, 64, 128), tail, dtype=np.float16)
        cache[0, :, :, :33] = 0
        cache[1, :, :, :33] = values
        valid = UOp.variable("valid_end", 32, 64).bind(33)
        cache_tensor = Tensor(cache).realize()
        assigned = Tensor(cache_tensor.uop.after(Tensor(valid).uop))
        out = flash_attention(q, assigned, valid)
        np.testing.assert_allclose(out.numpy(), expected, rtol=2e-3, atol=2e-3)

  def test_flash_attention_decode_beyond_256_chunks(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    n = 257 * 64
    q = Tensor.zeros(1, 1, 1, 32, dtype=dtypes.half).realize()
    k = Tensor.zeros(1, 1, n, 32, dtype=dtypes.half)
    v = Tensor.zeros(1, 1, n-64, 32, dtype=dtypes.half).cat(Tensor.ones(1, 1, 64, 32, dtype=dtypes.half), dim=2)
    cache = Tensor.stack(k, v).contiguous().realize()
    for valid, expected in ((1, 0), (n, 1/257)):
      with self.subTest(valid=valid):
        valid_kv_len = UOp.variable("valid_kv_len", 1, n).bind(valid)
        assigned = Tensor(cache.uop.after(Tensor(valid_kv_len).uop))
        np.testing.assert_allclose(flash_attention(q, assigned, valid_kv_len).numpy(), expected, rtol=2e-3, atol=2e-4)

  def test_flash_attention_decode_long_context_random(self):
    self._test_flash_decode(8, 2, 128, 257*64, 257*64-13)  # past 256 chunks, with a ragged tail

  def test_flash_attention_decode_chunk_round_accumulator_range(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    valid_kv_len, max_kv_len = 6749, 6784  # three chunk rounds, with a ragged tail
    q = Tensor.zeros(1, 8, 1, 32, dtype=dtypes.half).realize()
    cache = Tensor.stack(Tensor.zeros(1, 1, max_kv_len, 32, dtype=dtypes.half),
                         Tensor.full((1, 1, max_kv_len, 32), 5500, dtype=dtypes.half)).contiguous().realize()
    np.testing.assert_allclose(flash_attention(q, cache, valid_kv_len).numpy(), 5500, rtol=2e-3, atol=2e-3)

  def test_prefill_attention_unaligned_start(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    rng = np.random.default_rng(42)
    start_pos = 1718
    q = Tensor.zeros(1, 8, 32, 128)
    old_kv = rng.normal(size=(2, 1, 1, start_pos, 128)).astype(np.float32)
    new_kv = rng.normal(size=(2, 1, 1, 32, 128)).astype(np.float32)
    cache = Tensor.zeros(2, 1, 1, 2048, 128, dtype=dtypes.half).contiguous()
    Tensor.realize(cache[:, :, :, :start_pos].assign(Tensor(old_kv).cast(dtypes.half)))
    sp = UOp.variable("start_pos", 0, 2047).bind(start_pos)
    assigned = Tensor(cache.uop.after(cache[:, :, :, sp:sp+32, :].uop.store(Tensor(new_kv).cast(dtypes.half).uop)))
    out = flash_attention(q, assigned, sp+32).realize()
    values = np.concatenate([old_kv[1, 0, 0], new_kv[1, 0, 0]]).astype(np.float16).astype(np.float32)
    expected = np.stack([values[:start_pos+i+1].mean(0) for i in range(32)])[None, None].repeat(8, axis=1)
    np.testing.assert_allclose(out.numpy(), expected, rtol=2e-3, atol=2e-3)

  def test_q4_0_linear(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    rng = np.random.default_rng(42)
    in_features, out_features = 2048, 16
    blocks = out_features * in_features // 32
    packed = rng.integers(0, 256, blocks * 18, dtype=np.uint8)
    for i in range(blocks): packed[i*18:i*18+2] = np.array([0.01], dtype=np.float16).view(np.uint8)
    raw = Tensor(np.pad(packed, (4, 0))).contiguous().realize()[4:]
    decoded = ggml_data_to_tensor(raw, out_features * in_features, 2).reshape(out_features, in_features)
    weight = decoded.numpy()
    linear = Linear(in_features, out_features, bias=False)
    nn.state.load_state_dict(linear, {"weight": decoded}, verbose=False, realize=False)
    x = rng.normal(size=(3, in_features)).astype(np.float32)
    scale = np.maximum(np.abs(x).reshape(3, in_features//32, 32).max(-1, keepdims=True) / 127, 1e-8)
    xq = np.clip(np.rint(x.reshape(3, in_features//32, 32) / scale), -127, 127) * scale
    np.testing.assert_allclose(linear(Tensor(x)).numpy(), xq.reshape(3, in_features) @ weight.T, rtol=2e-3, atol=2e-2)
    self.assertEqual(linear.ggml_type, 2)
    self.assertEqual(linear.weight.dtype, dtypes.uint8)
    x_pre = rng.normal(size=(16, in_features)).astype(np.float32)
    np.testing.assert_allclose(linear(Tensor(x_pre)).numpy(), x_pre @ weight.T, rtol=1e-2, atol=1e-1)

  def test_q8_0_linear(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    rng = np.random.default_rng(42)
    in_features, out_features = 2048, 16
    blocks = out_features * in_features // 32
    packed = rng.integers(0, 256, blocks * 34, dtype=np.uint8)
    for i in range(blocks): packed[i*34:i*34+2] = np.array([0.01], dtype=np.float16).view(np.uint8)
    raw = Tensor(np.pad(packed, (4, 0))).contiguous().realize()[4:]
    decoded = ggml_data_to_tensor(raw, out_features * in_features, 8).reshape(out_features, in_features)
    weight = decoded.numpy()
    linear = Linear(in_features, out_features, bias=False)
    nn.state.load_state_dict(linear, {"weight": decoded}, verbose=False, realize=False)
    x = rng.normal(size=(3, in_features)).astype(np.float32)
    scale = np.maximum(np.abs(x).reshape(3, in_features//32, 32).max(-1, keepdims=True) / 127, 1e-8)
    xq = np.clip(np.rint(x.reshape(3, in_features//32, 32) / scale), -127, 127) * scale
    np.testing.assert_allclose(linear(Tensor(x)).numpy(), xq.reshape(3, in_features) @ weight.T, rtol=2e-3, atol=2e-2)
    self.assertEqual(linear.ggml_type, 8)
    self.assertEqual(linear.weight.dtype, dtypes.uint8)
    x_pre = rng.normal(size=(16, in_features)).astype(np.float32)
    np.testing.assert_allclose(linear(Tensor(x_pre)).numpy(), x_pre @ weight.T, rtol=1e-2, atol=1e-1)

  def test_q5_k_linear(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    rng = np.random.default_rng(42)
    in_features, out_features = 2048, 16
    blocks = out_features * in_features // 256
    packed = rng.integers(0, 256, blocks * 176, dtype=np.uint8)
    for i in range(blocks): packed[i*176:i*176+4] = np.array([0.01, 0.002], dtype=np.float16).view(np.uint8)
    raw = Tensor(np.pad(packed, (4, 0))).contiguous().realize()[4:]
    decoded = ggml_data_to_tensor(raw, out_features * in_features, 13).reshape(out_features, in_features)
    weight = decoded.numpy()
    linear = Linear(in_features, out_features, bias=False)
    nn.state.load_state_dict(linear, {"weight": decoded}, verbose=False, realize=False)
    x = rng.normal(size=(3, in_features)).astype(np.float32)
    scale = np.maximum(np.abs(x).reshape(3, in_features//32, 32).max(-1, keepdims=True) / 127, 1e-8)
    xq = np.clip(np.rint(x.reshape(3, in_features//32, 32) / scale), -127, 127) * scale
    np.testing.assert_allclose(linear(Tensor(x)).numpy(), xq.reshape(3, in_features) @ weight.T, rtol=2e-3, atol=2e-2)
    self.assertEqual(linear.ggml_type, 13)
    x_pre = rng.normal(size=(16, in_features)).astype(np.float32)
    np.testing.assert_allclose(linear(Tensor(x_pre)).numpy(), x_pre @ weight.T, rtol=1e-2, atol=1e-1)

  def test_q6_k_wmma_prefill(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    rng = np.random.default_rng(42)
    in_features, out_features = 2048, 16
    blocks = out_features * in_features // 256
    packed = rng.integers(0, 256, blocks * 210, dtype=np.uint8)
    for i in range(blocks): packed[i*210+208:i*210+210] = np.array([0.01], dtype=np.float16).view(np.uint8)
    raw = Tensor(np.pad(packed, (4, 0))).contiguous().realize()[4:]
    decoded = ggml_data_to_tensor(raw, out_features * in_features, 14).reshape(out_features, in_features)
    weight = decoded.numpy()
    linear = Linear(in_features, out_features, bias=False)
    nn.state.load_state_dict(linear, {"weight": decoded}, verbose=False, realize=False)
    x_pre = rng.normal(size=(16, in_features)).astype(np.float32)
    np.testing.assert_allclose(linear(Tensor(x_pre)).numpy(), x_pre @ weight.T, rtol=2e-2, atol=4e-1)
    self.assertEqual(linear.ggml_type, 14)

  def test_iq3_xxs_linear(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    rng = np.random.default_rng(42)
    in_features, out_features = 2048, 16
    blocks = out_features * in_features // 256
    packed = rng.integers(0, 256, blocks * 98, dtype=np.uint8)
    for i in range(blocks): packed[i*98:i*98+2] = np.array([0.01], dtype=np.float16).view(np.uint8)
    raw = Tensor(np.pad(packed, (4, 0))).contiguous().realize()[4:]
    decoded = ggml_data_to_tensor(raw, out_features * in_features, 18).reshape(out_features, in_features)
    weight = decoded.numpy()
    linear = Linear(in_features, out_features, bias=False)
    nn.state.load_state_dict(linear, {"weight": decoded}, verbose=False, realize=False)
    x = rng.normal(size=(3, in_features)).astype(np.float32)
    scale = np.maximum(np.abs(x).reshape(3, in_features//32, 32).max(-1, keepdims=True) / 127, 1e-8)
    xq = np.clip(np.rint(x.reshape(3, in_features//32, 32) / scale), -127, 127) * scale
    np.testing.assert_allclose(linear(Tensor(x)).numpy(), xq.reshape(3, in_features) @ weight.T, rtol=2e-3, atol=2e-2)
    self.assertEqual(linear.ggml_type, 18)
    self.assertEqual(linear.weight.dtype, dtypes.uint8)
    x_pre = rng.normal(size=(16, in_features)).astype(np.float32)
    np.testing.assert_allclose(linear(Tensor(x_pre)).numpy(), x_pre @ weight.T, rtol=2e-2, atol=4e-1)

  def test_iq2_xs_linear(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    rng = np.random.default_rng(42)
    in_features, out_features = 2048, 16
    blocks = out_features * in_features // 256
    packed = rng.integers(0, 256, blocks * 74, dtype=np.uint8)
    for i in range(blocks): packed[i*74:i*74+2] = np.array([0.01], dtype=np.float16).view(np.uint8)
    raw = Tensor(np.pad(packed, (4, 0))).contiguous().realize()[4:]
    decoded = ggml_data_to_tensor(raw, out_features * in_features, 17).reshape(out_features, in_features)
    weight = decoded.numpy()
    linear = Linear(in_features, out_features, bias=False)
    nn.state.load_state_dict(linear, {"weight": decoded}, verbose=False, realize=False)
    x = rng.normal(size=(3, in_features)).astype(np.float32)
    scale = np.maximum(np.abs(x).reshape(3, in_features//32, 32).max(-1, keepdims=True) / 127, 1e-8)
    xq = np.clip(np.rint(x.reshape(3, in_features//32, 32) / scale), -127, 127) * scale
    np.testing.assert_allclose(linear(Tensor(x)).numpy(), xq.reshape(3, in_features) @ weight.T, rtol=2e-3, atol=2e-2)
    self.assertEqual(linear.ggml_type, 17)
    self.assertEqual(linear.weight.dtype, dtypes.uint8)
    x_pre = rng.normal(size=(16, in_features)).astype(np.float32)
    np.testing.assert_allclose(linear(Tensor(x_pre)).numpy(), x_pre @ weight.T, rtol=2e-2, atol=4e-1)

  def test_fused_routed_iq_experts(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    rng = np.random.default_rng(42)
    experts, in_features, out_features = 3, 256, 16
    selected_np = np.array([[[0, 2], [1, 0]]], dtype=np.int32)
    selected = Tensor(selected_np)

    for ggml_type, block_bytes in ((17, 74), (18, 98), (21, 110), (23, 136)):
      blocks = experts * out_features * in_features // 256
      packed = rng.integers(0, 256, blocks * block_bytes, dtype=np.uint8)
      for i in range(blocks): packed[i*block_bytes:i*block_bytes+2] = np.array([0.01], dtype=np.float16).view(np.uint8)
      raw = Tensor(np.pad(packed, (4, 0))).contiguous().realize()[4:]
      decoded = ggml_data_to_tensor(raw, experts * out_features * in_features, ggml_type).reshape(experts, out_features, in_features)
      reference = ggml_data_to_tensor(raw, experts * out_features * in_features, ggml_type).reshape(experts, out_features, in_features).numpy()
      weight = ExpertWeights(experts, in_features, out_features)
      weight.weight = decoded

      for shared_input in (True, False):
        with self.subTest(ggml_type=ggml_type, shared_input=shared_input):
          x_shape = (1, 2, 1 if shared_input else 2, in_features)
          x = rng.normal(size=x_shape).astype(np.float32)
          flat = x.reshape(-1, in_features//32, 32)
          scale = np.maximum(np.abs(flat).max(-1, keepdims=True) / 127, 1e-8)
          xq = (np.clip(np.rint(flat / scale), -127, 127) * scale).reshape(x_shape)
          expected = np.matmul(xq[..., None, :], reference[selected_np].swapaxes(-1, -2)).squeeze(-2)
          np.testing.assert_allclose(weight(selected, Tensor(x)).numpy(), expected, rtol=3e-3, atol=3e-2)
          self.assertEqual(weight.ggml_type, ggml_type)
          self.assertEqual(weight.weight.dtype, dtypes.uint32 if ggml_type == 23 else dtypes.uint8)

  def test_fused_routed_iq_experts_symbolic_tokens(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    rng = np.random.default_rng(7)
    experts, in_features, out_features, block_bytes = 3, 256, 16, 74
    blocks = experts * out_features * in_features // 256
    packed = rng.integers(0, 256, blocks * block_bytes, dtype=np.uint8)
    for i in range(blocks): packed[i*block_bytes:i*block_bytes+2] = np.array([0.01], dtype=np.float16).view(np.uint8)
    raw = Tensor(np.pad(packed, (4, 0))).contiguous().realize()[4:]
    decoded = ggml_data_to_tensor(raw, experts * out_features * in_features, 17).reshape(experts, out_features, in_features)
    reference = ggml_data_to_tensor(raw, experts * out_features * in_features, 17).reshape(experts, out_features, in_features).numpy()
    weight = ExpertWeights(experts, in_features, out_features)
    weight.weight = decoded
    selected_np = np.array([[[0, 2], [1, 0]]], dtype=np.int32)
    x = rng.normal(size=(1, 2, 1, in_features)).astype(np.float32)
    flat = x.reshape(-1, in_features//32, 32)
    scale = np.maximum(np.abs(flat).max(-1, keepdims=True) / 127, 1e-8)
    xq = (np.clip(np.rint(flat / scale), -127, 127) * scale).reshape(x.shape)
    expected = np.matmul(xq[..., None, :], reference[selected_np].swapaxes(-1, -2)).squeeze(-2)[:, :1]
    tokens = UOp.variable("expert_tokens", 1, 2).bind(1)
    out = weight(Tensor(selected_np)[:, :tokens], Tensor(x)[:, :tokens])
    np.testing.assert_allclose(out[:, :1].numpy(), expected, rtol=3e-3, atol=3e-2)
    self.assertEqual(out.shape, (1, tokens, 2, out_features))

  def test_hc_prepare_jit(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("custom kernels required")
    @TinyJit
    def run(mixes, base, scale):
      result = hc_prepare(mixes, base, scale, 4, 1e-6, 20)
      Tensor.realize(*result)
      return result
    rng = np.random.default_rng(7)
    for magnitude in (1, 1, 20, 0, 0.01):
      mixes = Tensor(rng.normal(size=(1, 1, 24)).astype(np.float32)*magnitude).realize()
      base = Tensor(rng.normal(size=24).astype(np.float32)).realize()
      scale = Tensor(rng.normal(size=3).astype(np.float32)).realize()
      got = run(mixes, base, scale)
      expected = hc_prepare(mixes, base, scale, 4, 1e-6, 20)
      for actual, reference in zip(got, expected):
        np.testing.assert_allclose(actual.numpy(), reference.numpy(), rtol=1e-5, atol=1e-6)
      comb = got[2].numpy()
      self.assertTrue(np.isfinite(comb).all())
      np.testing.assert_allclose(comb.sum(-2), 1, rtol=1e-5, atol=1e-5)

  def test_hc_prepare(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("custom kernels required")
    rng = np.random.default_rng(42)
    hc, iters, eps = 4, 20, 1e-6
    width = (2 + hc) * hc
    base = Tensor(rng.normal(size=(width,)).astype(np.float32))
    scale = Tensor(rng.normal(size=(3,)).astype(np.float32))
    for B, T in ((1, 1), (1, 8), (2, 3)):
      with self.subTest(B=B, T=T):
        mixes = Tensor(rng.normal(size=(B, T, width)).astype(np.float32))
        pre, post, comb = hc_prepare(mixes, base, scale, hc, eps, iters)
        pre_ref = (mixes[..., :hc] * scale[0] + base[:hc]).sigmoid() + eps
        post_ref = (mixes[..., hc:2*hc] * scale[1] + base[hc:2*hc]).sigmoid() * 2
        comb_ref = (mixes[..., 2*hc:] * scale[2] + base[2*hc:]).reshape(B, T, hc, hc).softmax(-1) + eps
        comb_ref = comb_ref / (comb_ref.sum(-2, keepdim=True) + eps)
        for _ in range(1, iters):
          comb_ref = comb_ref / (comb_ref.sum(-1, keepdim=True) + eps)
          comb_ref = comb_ref / (comb_ref.sum(-2, keepdim=True) + eps)
        np.testing.assert_allclose(pre.numpy(), pre_ref.numpy(), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(post.numpy(), post_ref.numpy(), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(comb.numpy(), comb_ref.numpy(), rtol=1e-4, atol=1e-5)

  def test_dsa_decode(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("custom kernels required")
    rng = np.random.default_rng(42)
    B, H, D, LORA, N, K = 1, 8, 576, 512, 512, 2051
    q = Tensor(rng.normal(size=(B, H, D)).astype(np.float16))
    cached = Tensor(rng.normal(size=(B, N, D)).astype(np.float16))
    idx = rng.integers(0, N, (B, K)).astype(np.int32)
    idx[:, ::7] = -1  # sparse selections carry invalid entries
    scale = 0.25
    got = dsa_decode(q, cached, Tensor(idx), scale, LORA)
    # float64 reference
    q64, c64 = q.numpy().astype(np.float64), cached.numpy().astype(np.float64)
    sel = np.where(idx < 0, 0, idx)
    rows = c64[0][sel[0]]
    sc = np.einsum('hd,kd->hk', q64[0], rows) * scale
    sc[:, idx[0] < 0] = -np.inf
    p = np.exp(sc - sc.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    want = p @ rows[:, :LORA]
    np.testing.assert_allclose(got.numpy(), want[None], rtol=1e-3, atol=1e-3)

  def test_dsa_decode_effective_selection(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("custom kernels required")
    rng = np.random.default_rng(42)
    B, H, D, LORA, N, K = 2, 4, 160, 128, 512, 259
    q_np = rng.normal(size=(B, H, D)).astype(np.float16)
    cache_np = rng.normal(size=(B, N, D)).astype(np.float16)
    q = Tensor(q_np).realize()
    @TinyJit
    def run(cached, indices, valid):
      return dsa_decode(q, cached, indices, 0.25, LORA, valid_tokens=valid).realize()
    for length in (1, 3, 4, 127, 128, 131, 256, 259, 511, 1):
      with self.subTest(length=length):
        idx = np.full((B, K), -1, dtype=np.int32)
        for b in range(B):
          pools = rng.permutation(length//4)[:64]
          prefix = (pools[:, None]*4 + np.arange(4)).flatten()
          idx[b, :len(prefix)] = prefix
          idx[b, 256:256+length%4] = np.arange(length//4*4, length)
        cached = cache_np.copy()
        cached[:, length:] = np.nan  # skipped indices must not access unwritten rows
        want = []
        for b in range(B):
          rows = cached[b, idx[b, idx[b] >= 0]].astype(np.float64)
          scores = q_np[b].astype(np.float64) @ rows.T * 0.25
          p = np.exp(scores - scores.max(-1, keepdims=True))
          want.append((p / p.sum(-1, keepdims=True)) @ rows[:, :LORA])
        valid = UOp.variable("dsa_valid_tokens", 1, N).bind(length)
        got = run(Tensor(cached).realize(), Tensor(idx).realize(), valid).numpy()
        np.testing.assert_allclose(got, np.array(want), rtol=1e-3, atol=1e-3)

  def test_dsa_decode_sparse_tail_unwritten_cache(self):
    # early decode steps select almost only invalid entries, and the cache past the cursor is uninitialized
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("custom kernels required")
    rng = np.random.default_rng(42)
    B, H, D, LORA, N, K = 1, 8, 576, 512, 512, 2051
    q = Tensor(rng.normal(size=(B, H, D)).astype(np.float16))
    cache_buf = Tensor.empty(B, N, D, dtype=dtypes.half)
    new = Tensor(rng.normal(size=(B, 2, D)).astype(np.float16))
    cached = Tensor(cache_buf.uop.after(cache_buf[:, 0:2].uop.store(new.uop)))
    idx = np.full((B, K), -1, dtype=np.int32)
    idx[0, -3:] = [0, 1, -1]
    got = dsa_decode(q, cached, Tensor(idx), 0.25, LORA).numpy()
    rows = cached.numpy()[0][[0, 1]]
    sc = (q.numpy()[0].astype(np.float64) @ rows.astype(np.float64).T) * 0.25
    p = np.exp(sc - sc.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    want = p @ rows[:, :LORA].astype(np.float64)
    np.testing.assert_allclose(got, want[None], rtol=1e-3, atol=1e-3)

class TestKDAConvNorm(unittest.TestCase):
  def setUp(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("AMD custom kernels required")

  def test_matches_generic_conv(self):
    from tinygrad.llm.kernels.amd import kda_conv_norm
    B, H, D = 1, 4, 128
    C = 3*H*D
    rng = np.random.default_rng(5)
    qkv_np = rng.normal(size=(B, 1, C)).astype(np.float32)
    st_np = rng.normal(size=(B, 3, C)).astype(np.float32)
    wt_np = rng.normal(size=(C, 4)).astype(np.float16)
    eps, q_scale = 1e-6, D**-0.5
    for pos in (0, 7):
      with self.subTest(pos=pos):
        state = Tensor(st_np.copy()).realize()
        q, k, v, _ = kda_conv_norm(Tensor(qkv_np.copy()), state, Tensor(wt_np.copy()), pos, H, D, eps, q_scale)
        Tensor.realize(q, k, v)  # the kernel runs once per schedule; realize all outputs together (in-place state write included)
        win = np.concatenate([np.zeros((B, 3, C), np.float32) if pos == 0 else st_np, qkv_np], axis=1)
        conv = sum(win[:, i] * wt_np[:, i].astype(np.float32) for i in range(3)) + win[:, 3] * wt_np[:, 3].astype(np.float32)
        conv = conv / (1 + np.exp(-conv))
        np.testing.assert_allclose(state.numpy(), np.stack([win[:, 1], win[:, 2], win[:, 3]], axis=1), rtol=1e-6, atol=1e-6)
        def heads(sl): return conv[0, sl].reshape(1, H, D)
        def normed(a, s=1.0): return a / np.maximum(np.sqrt((a**2).sum(-1, keepdims=True)), eps) * s
        np.testing.assert_allclose(q.numpy().reshape(B, H, D), normed(heads(slice(0, H*D)), q_scale), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(k.numpy().reshape(B, H, D), normed(heads(slice(H*D, 2*H*D))), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(v.numpy().reshape(B, H, D), heads(slice(2*H*D, C)), rtol=1e-5, atol=1e-6)

class TestHCMixes(unittest.TestCase):
  def setUp(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("AMD custom kernels required")

  def test_matches_normed_linear(self):
    from tinygrad.llm.kernels.amd import hc_mixes
    width, outputs = 16384, 24
    linear, weights = TestQ8Grouped.make_linear(width, outputs)
    linear.set_quantized(linear.weight)
    self.assertEqual(linear.ggml_type, 8)
    x = Tensor(np.random.default_rng(7).normal(size=(1, 1, 4, 4096)).astype(np.float32)).realize()
    got = hc_mixes(x, linear.weight, outputs, 1e-5).numpy()
    flat = x.numpy().reshape(1, -1)
    normed = flat / np.sqrt((flat**2).mean(-1, keepdims=True) + 1e-5)
    expected = linear(Tensor(normed).realize()).numpy()  # the unfused path: q8 quantize + packed linear
    np.testing.assert_allclose(got.reshape(1, outputs), expected, rtol=5e-3, atol=2e-3)

  def test_fuse_q8_linears(self):
    from tinygrad.llm.model import fuse_q8_linears
    l1, _ = TestQ8Grouped.make_linear(1024, 512)
    l2, _ = TestQ8Grouped.make_linear(1024, 256)
    fused = fuse_q8_linears(l1, l2)
    self.assertIsNotNone(fused)
    x = Tensor(np.random.default_rng(3).normal(size=(1, 1, 1024)).astype(np.float32)).realize()
    np.testing.assert_allclose(fused(x).numpy(), np.concatenate([l1(x).numpy(), l2(x).numpy()], axis=-1), rtol=1e-4, atol=1e-4)

class TestFusedDecodeKernels(unittest.TestCase):
  def setUp(self):
    if not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("AMD custom kernels required")

  def test_gumbel_argmax(self):
    from tinygrad.llm.kernels.amd import gumbel_argmax
    rng = np.random.default_rng(12)
    logits = Tensor(rng.normal(size=(1, 1024)).astype(np.float32)).realize()
    got = gumbel_argmax(logits, Tensor([0.7])).numpy()
    self.assertEqual(got.shape, (1, 1))
    # temperature near zero is argmax of logits
    got0 = gumbel_argmax(logits, Tensor([0.0])).numpy()
    np.testing.assert_array_equal(got0, logits.numpy().argmax(-1, keepdims=True))

  def test_router_probs(self):
    from tinygrad.llm.kernels.amd import router_probs
    rng = np.random.default_rng(13)
    logits = rng.normal(size=(3, 32)).astype(np.float32)
    sel = rng.integers(0, 32, (3, 4)).astype(np.int32)
    got = router_probs(Tensor(logits).realize(), Tensor(sel).realize(), 2.5).numpy()
    p = 1/(1+np.exp(-np.take_along_axis(logits, sel, axis=-1)))
    np.testing.assert_allclose(got, p/p.sum(-1, keepdims=True)*2.5, rtol=1e-5, atol=1e-6)

  def test_hc_mix(self):
    from tinygrad.llm.kernels.amd import hc_mix
    rng = np.random.default_rng(14)
    B, T, HC, D = 1, 2, 4, 256
    x = Tensor(rng.normal(size=(B, T, D)).astype(np.float32)).realize()
    res = Tensor(rng.normal(size=(B, T, HC, D)).astype(np.float32)).realize()
    post = Tensor(rng.normal(size=(B, T, HC)).astype(np.float32)).realize()
    comb = Tensor(rng.uniform(size=(B, T, HC, HC)).astype(np.float32)).realize()
    x_np, res_np, post_np, comb_np = x.numpy(), res.numpy(), post.numpy(), comb.numpy()
    got = hc_mix(x, res, post, comb).numpy()
    exp = post_np[..., None]*x_np[:, :, None, :] + np.einsum('btij,btid->btjd', comb_np, res_np)
    np.testing.assert_allclose(got, exp, rtol=1e-5, atol=1e-5)

  def test_kda_gates_and_out(self):
    from tinygrad.llm.kernels.amd import kda_gates, kda_out
    rng = np.random.default_rng(15)
    B, H, K, lb = 1, 4, 32, -5.0
    beta_raw = rng.normal(size=(B, H)).astype(np.float32)
    alpha_raw = rng.normal(size=(B, H, K)).astype(np.float32)
    dt = rng.normal(size=H*K).astype(np.float32)
    a_log = rng.uniform(0.1, 1, size=H).astype(np.float32)
    beta, alpha = kda_gates(Tensor(beta_raw).realize(), Tensor(alpha_raw).realize(), Tensor(dt).realize(), Tensor(a_log).realize(), lb, H, K)
    Tensor.realize(beta, alpha)
    exp_beta = 1/(1+np.exp(-beta_raw))
    exp_alpha = np.exp(lb / (1+np.exp((alpha_raw.reshape(B, H*K) + dt).reshape(B, H, K) * a_log[None, :, None])))
    np.testing.assert_allclose(beta.numpy(), exp_beta.reshape(B, H, 1), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(alpha.numpy(), exp_alpha.reshape(B, H, 1, K), rtol=1e-4, atol=1e-6)
    core = rng.normal(size=(B, H, 1, K)).astype(np.float32)
    gate = rng.normal(size=(B, 1, H, K)).astype(np.float32)
    w = rng.normal(size=K).astype(np.float32)
    z = kda_out(Tensor(core).realize(), Tensor(gate).realize(), Tensor(w).realize(), 1e-5, dtypes.float32).numpy()
    nf = np.sqrt((core**2).mean(-1, keepdims=True) + 1e-5)
    exp = core/nf * w[None, None, None, :] * (1/(1+np.exp(-gate.transpose(0, 2, 1, 3))))
    np.testing.assert_allclose(z, exp, rtol=1e-4, atol=1e-5)

if __name__ == "__main__": unittest.main()
