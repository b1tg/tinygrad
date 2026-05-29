import unittest, numpy as np

from tinygrad import Tensor, dtypes
from tinygrad.codegen import to_program
from tinygrad.helpers import Target
from tinygrad.renderer.cstyle import HIPRenderer, SUDOT4_ARG
from tinygrad.uop.ops import Ops
from tinygrad.llm.quant import q8_0_matvec, q8_0_pack_weight_u32

class TestLLMQuant(unittest.TestCase):
  def test_q8_0_matvec_matches_reference(self):
    rng = np.random.default_rng(0)
    M, K = 5, 64
    qs = rng.integers(-50, 50, size=(M, K), dtype=np.int8)
    d = rng.random((M, K//32)).astype(np.float16)
    blocks = np.zeros((M, K//32, 34), dtype=np.uint8)
    blocks[:, :, :2] = d.view(np.uint8).reshape(M, K//32, 2)
    blocks[:, :, 2:] = qs.reshape(M, K//32, 32).view(np.uint8)
    x = rng.normal(size=(1, K)).astype(np.float32)

    out = q8_0_matvec(Tensor(x), Tensor(blocks, dtype=dtypes.uint8)).numpy()
    xb = x.reshape(1, K//32, 32)
    xd = np.max(np.abs(xb), axis=-1) / 127.0
    xq = np.round(xb / np.maximum(xd[..., None], 1e-12)).clip(-127, 127).astype(np.int8)
    ref = ((qs.reshape(M, K//32, 32)[None].astype(np.int32) * xq[:, None].astype(np.int32)).sum(axis=-1) *
           d[None].astype(np.float32) * xd[:, None]).sum(axis=-1)
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)

  def test_q8_0_pack_weight_u32_matches_raw(self):
    rng = np.random.default_rng(1)
    M, K = 7, 64
    blocks = Tensor(rng.integers(0, 255, size=(M, K//32, 34), dtype=np.uint8), dtype=dtypes.uint8)
    x = Tensor(rng.normal(size=(1, K)).astype(np.float32))
    np.testing.assert_allclose(q8_0_matvec(x, q8_0_pack_weight_u32(blocks)).numpy(), q8_0_matvec(x, blocks).numpy(), rtol=1e-5, atol=1e-5)

  def test_q8_0_matvec_hip_renders_sudot4(self):
    blocks = Tensor(np.zeros((1, 1, 34), dtype=np.uint8), dtype=dtypes.uint8)
    x = Tensor(np.zeros((1, 32), dtype=np.float32))
    prg = to_program(q8_0_matvec(x, blocks).contiguous().schedule_linear().src[-1].src[0],
                     HIPRenderer(Target("AMD", arch="gfx1100")))
    self.assertTrue(any(u.op is Ops.WMMA and u.arg == SUDOT4_ARG for u in prg.src[0].toposort()))
    src = next(u.arg for u in prg.src if u.op is Ops.SOURCE)
    self.assertIn("__builtin_amdgcn_sudot4", src)

if __name__ == "__main__":
  unittest.main()
