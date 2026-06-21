import unittest
import numpy as np
from tinygrad import Tensor, Variable, GlobalCounters

class TestTensorVariable(unittest.TestCase):
  def test_add_tvar(self):
    vv = Variable("a", 0, 10).bind(1)
    ret = (Tensor(vv) + 3).item()
    assert ret == 4

  def test_inner_tvar_node(self):
    vv = Variable("w", 0, 10).bind(2)
    ret = Tensor(vv * 4).item()
    assert ret == 8

  def test_inner_tvar_mul(self):
    vv = Variable("w", 0, 10).bind(2)
    assert (Tensor(3) * vv).item() == 6

  def test_inner_tvar_mul_node(self):
    vv = Variable("w", 0, 10).bind(2)
    assert (Tensor(3) * (vv * 4)).item() == 24

  def test_symbolic_mean(self):
    vv = Variable("a", 1, 10).bind(2)
    t = Tensor.ones(2, 10).contiguous()[:, :vv]
    ret = t.mean().item()
    assert ret == 1

  def test_symbolic_mean_2d(self):
    vv = Variable("a", 1, 10).bind(2)
    vv2 = Variable("b", 1, 10).bind(2)
    t = Tensor.ones(10, 10).contiguous()[:vv2, :vv]
    ret = t.mean().item()
    assert ret == 1

  def test_symbolic_mean_2d_axis_1(self):
    vv = Variable("a", 1, 10).bind(2)
    vv2 = Variable("b", 1, 10).bind(2)
    t = Tensor.ones(10, 10).contiguous()[:vv2, :vv]
    ret = t.mean(axis=1)[:2].reshape(2, 1).numpy()
    assert np.all(ret == 1)

  def test_symbolic_mean_2d_add(self):
    add_term = Variable("c", 0, 10).bind(1)
    vv = Variable("a", 1, 10).bind(1)
    vv2 = Variable("b", 1, 10).bind(1)
    t = Tensor.ones(20, 20).contiguous()[:vv2+add_term, :vv+add_term]
    ret = t.mean().item()
    assert ret == 1

  def test_symbolic_var(self):
    vv = Variable("a", 1, 10).bind(2)
    t = Tensor.ones(2, 10).contiguous()[:, :vv]
    ret = t.var().item()
    assert ret == 0

  def test_symbolic_pad(self):
    vv = Variable("a", 1, 10).bind(2)
    t = Tensor.ones(2, 2).contiguous()
    t = t.pad([vv, vv, vv, vv]).mean()
    ones = 4
    zeros = 6+6+4+4+6+6
    self.assertAlmostEqual(t.item(), ones/(ones+zeros))

  def test_symbolic_arange(self):
    vv = Variable("a", 1, 10)
    ret = Tensor.arange(0, vv.bind(4))
    self.assertListEqual(ret[:4].tolist(), [0,1,2,3])

  def test_symbolic_arange_sym_start(self):
    vv = Variable("a", 1, 6)
    ret = Tensor.arange(vv.bind(4), 7)
    self.assertListEqual(ret[:3].tolist(), [4,5,6])

  def test_symbolic_arange_sym_step(self):
    vv = Variable("step", 1, 3)
    ret = Tensor.arange(0, 10, vv.bind(2))
    self.assertListEqual(ret[:5].tolist(), [0,2,4,6,8])

  def test_symbolic_arange_two_vars(self):
    begin = Variable("b", 1, 5)
    end = Variable("e", 6, 10)
    ret = Tensor.arange(begin.bind(4), end.bind(7))
    self.assertListEqual(ret[:3].tolist(), [4,5,6])

  def test_symbolic_arange_three_vars(self):
    begin = Variable("b", 0, 5)
    end = Variable("e", 10, 20)
    step = Variable("s", 1, 3)
    ret = Tensor.arange(begin.bind(2), end.bind(14), step.bind(3))
    self.assertListEqual(ret[:4].tolist(), [2,5,8,11])

  def test_symbolic_full(self):
    vv = Variable("x", 1, 10).bind(5)
    t = Tensor.full((3,), vv)
    self.assertListEqual(t.tolist(), [5,5,5])

  def test_variable_empty(self):
    v = Variable("i", 1, 10)
    # TODO: Tensor creation from unbound variable should assert
    # with self.assertRaises(AssertionError): t = Tensor.empty(3, v)
    vb = v.bind(3)
    t = Tensor.empty(3, vb)
    assert t.uop.base.buffer.size == 30
    assert t.uop.shape == (3, vb)

  def test_symbolic_chunk(self):
    # chunk should work when split dimension is concrete, even if other dims are symbolic
    vv = Variable("a", 1, 10).bind(4)
    t = Tensor.ones(10, 8).contiguous()[:vv, :]  # shape (vv, 8)
    chunks = t.chunk(2, dim=-1)  # split along concrete dim 8
    assert len(chunks) == 2
    assert chunks[0].shape[1] == 4
    assert chunks[1].shape[1] == 4
    # verify the values by shrinking to concrete shape first
    np.testing.assert_equal(chunks[0].shrink(((0, 4), (0, 4))).numpy(), np.ones((4, 4)))
    np.testing.assert_equal(chunks[1].shrink(((0, 4), (0, 4))).numpy(), np.ones((4, 4)))

  def test_symbolic_split(self):
    # split should work when split dimension is concrete, even if other dims are symbolic
    vv = Variable("a", 1, 10).bind(3)
    t = Tensor.arange(30).reshape(10, 3).contiguous()[:, :vv]  # shape (10, vv)
    splits = t.split(5, dim=0)  # split along concrete dim 10
    assert len(splits) == 2
    assert splits[0].shape[0] == 5
    assert splits[1].shape[0] == 5
    # verify the values by shrinking to concrete shape first
    np.testing.assert_equal(splits[0].shrink(((0, 5), (0, 3))).numpy(), np.arange(30).reshape(10, 3)[:5, :3])
    np.testing.assert_equal(splits[1].shrink(((0, 5), (0, 3))).numpy(), np.arange(30).reshape(10, 3)[5:, :3])

  def test_symbolic_chunk_error_on_symbolic_dim(self):
    # chunk should fail when trying to split along a symbolic dimension
    vv = Variable("a", 1, 10).bind(4)
    t = Tensor.ones(10, 8).contiguous()[:vv, :]  # shape (vv, 8)
    with self.assertRaises(AssertionError):
      t.chunk(2, dim=0)  # can't split along symbolic dim

  def test_symbolic_var_sum(self, var_name="u"):
    t = Variable("t", 1, 10).bind(4)
    v = Variable(var_name, 1, 5).bind(1)
    mask = (Tensor.full((1, 1, t, v+t), 1) + 1).contiguous()
    mask.shrink(((0, 1), (0, 1), (0, 4), (0, 4))).numpy()
  def test_symbolic_var_sum_alt_name(self): self.test_symbolic_var_sum("s")

  def test_symbolic_triu(self):
    t = Variable("t", 1, 10).bind(4)
    for start_pos in (0, 1, 3):
      var_start_pos = Variable("start_pos", 0, 5).bind(start_pos)
      mask = Tensor.full((1, 1, t, var_start_pos+t), float("-inf")).triu(var_start_pos+1)
      out = mask.shrink(((0, 1), (0, 1), (0, 4), (0, start_pos+4))).numpy()
      expected = np.triu(np.full((1, 1, 4, start_pos+4), float("-inf")), k=start_pos+1)
      np.testing.assert_equal(out, expected)

  def test_symbolic_tril(self):
    t = Variable("t", 1, 10).bind(4)
    for start_pos in (0, 1, 3):
      var_start_pos = Variable("start_pos", 0, 5).bind(start_pos)
      mask = Tensor.full((1, 1, t, var_start_pos+t), float("-inf")).tril(var_start_pos+1)
      out = mask.shrink(((0, 1), (0, 1), (0, 4), (0, start_pos+4))).numpy()
      expected = np.tril(np.full((1, 1, 4, start_pos+4), float("-inf")), k=start_pos+1)
      np.testing.assert_equal(out, expected)

class TestSymbolicCumsum(unittest.TestCase):
  # arange builds on cumsum; the basic cumsum (_cumalu) is O(n^2) and the O(n) two-stage (_split_cumalu) used to
  # bail to it for symbolic shapes, so any symbolic-length arange/cumsum ran O(n^2) at runtime. These guard correctness + O(n).
  def test_symbolic_cumsum(self):
    for n in (5, 256, 513, 600, 1000):  # incl. non-multiples of the 256 split (exercises the back-pad + [:s] recovery)
      v = Variable("n", 1, 2000).bind(n)
      got = Tensor.arange(2000).float().contiguous()[:v].cumsum().pad(((0, 2000-v),)).numpy()
      np.testing.assert_allclose(got[:n], np.cumsum(np.arange(n)), rtol=1e-4)

  def test_symbolic_arange(self):
    for n in (5, 256, 600):
      v = Variable("n", 1, 2000).bind(n)
      np.testing.assert_equal(Tensor.arange(v).pad(((0, 2000-v),)).numpy()[:n], np.arange(n))

  def test_symbolic_cumsum_is_linear(self):
    def ops(n):
      v = Variable("n", 1, 16384).bind(n)  # keep vmax < 65535 (WGSL dispatch limit) so the symbolic realize compiles on WEBGPU
      base = Tensor.ones(16384).contiguous().realize()
      GlobalCounters.reset()
      base[:v].cumsum().realize()
      return GlobalCounters.global_ops
    o1, o8 = ops(1024), ops(8192)  # 8x length: O(n) is ~8x, O(n^2) would be ~64x
    if o1 == 0: self.skipTest("backend doesn't report op estimates (e.g. CPU:X86 asm)")  # can't measure flops here
    self.assertLess(o8/o1, 16, f"symbolic cumsum ops grew {o8/o1:.1f}x for 8x length (quadratic?)")

if __name__ == '__main__':
  unittest.main()
