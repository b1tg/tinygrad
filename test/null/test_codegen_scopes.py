import unittest
from tinygrad import UOp, dtypes
from tinygrad.uop.ops import AxisType, Ops
from tinygrad.codegen.gpudims import get_grouped_dims
from tinygrad.codegen.late.linearizer import linearize


class TestCodegenScopes(unittest.TestCase):
  def test_singleton_grouped_dimension(self):
    indices = get_grouped_dims('gidx', (32, 1), (65535,))
    self.assertTrue(all(isinstance(index, UOp) for index in indices))
    self.assertEqual(indices[1].vmin, 0)
    self.assertEqual(indices[1].vmax, 0)

  def test_runtime_single_iteration_scope(self):
    index = UOp.special(32, 'lidx0')*4
    loop = UOp.range(UOp.variable('n', 0, 1), 0, AxisType.REDUCE)
    buf = UOp.placeholder((128,), dtypes.float32, slot=0)
    end = buf[index+loop].store(1).end(loop)
    sink = buf[index].store(buf[index].after(end)).sink()
    ordered = linearize(sink)
    self.assertLess(ordered.index(index), ordered.index(loop))
    self.assertLess(ordered.index(loop), next(i for i,u in enumerate(ordered) if u.op is Ops.END))


if __name__ == '__main__': unittest.main()
