import unittest
from tinygrad import Device, Tensor, dtypes, Variable
from tinygrad.codegen.opt import Opt, OptOps, KernelOptError
from tinygrad.codegen.opt.search import get_kernel_actions
from tinygrad.codegen.opt.postrange import Scheduler

# TODO: write a clean version of this
from test.backend.test_linearizer import helper_linearizer_opt, replace_opts, to_program

class TestKernelOpts(unittest.TestCase):
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "test requires locals")
  def test_group_symbolic_reduce(self):
    # GROUP a symbolic reduce axis by a non-dividing amount: rounds up + gates the tail (master raises KernelOptError)
    A, B = Tensor.rand(8, 512).realize(), Tensor.rand(512, 8).realize()
    vi = Variable("i", 1, 512).bind(333)
    ast = (A[:, :vi] @ B[:vi, :]).linear_with_vars()[0].src[-1].src[0]
    to_program(replace_opts(ast, [Opt(OptOps.GROUP, 0, 3)]), renderer=Device[Device.DEFAULT].renderer)

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "test requires locals")
  def test_no_group_on_symbolic_output(self):
    # GROUP/GROUPTOP split a reduce for parallelism; a symbolic output grid (batched prefill) already saturates the GPU, so BEAM must not offer it
    ren = Device[Device.DEFAULT].renderer
    def groups(ast):
      s = Scheduler(ast, ren); s.convert_loop_to_global()  # BEAM globalizes output loops before enumerating actions
      return [o for sc in get_kernel_actions(s, include_0=False).values() for o in sc.applied_opts if o.op in (OptOps.GROUP, OptOps.GROUPTOP)]
    kv, toks = Variable("kv", 1, 1024).bind(1000), Variable("toks", 1, 512).bind(32)
    v = Tensor.rand(8, 1, 1024, 128, dtype=dtypes.half).realize()
    # prefill: symbolic query dim in the output -> no GROUP candidates
    wp = Tensor.rand(8, 2, 512, 1024, dtype=dtypes.half).realize()
    pre = (wp[:, :, :toks, :kv].unsqueeze(-1) * v[:, :, None, :kv, :]).sum(3).linear_with_vars()[0].src[-1].src[0]
    self.assertEqual(groups(pre), [])
    # decode: fixed-size output -> GROUP still offered
    wd = Tensor.rand(8, 2, 1024, dtype=dtypes.half).realize()
    dec = (wd[:, :, :kv].unsqueeze(-1) * v[:, :, :kv, :]).sum(2).linear_with_vars()[0].src[-1].src[0]
    self.assertNotEqual(groups(dec), [])

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "test requires locals")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_shared, "test requires shared")
  def test_local_and_grouped_reduce(self):
    N = 128
    Tensor.manual_seed(1882)
    a = Tensor.rand(4, 4, N, N)
    b = Tensor.rand(4, 4, N)
    r = (b.sqrt() + ((a+1).sum(axis=3).exp()))
    helper_linearizer_opt(r, [
      [Opt(OptOps.LOCAL, 0, 2)],
      [Opt(OptOps.LOCAL, 0, 8)],
      [Opt(OptOps.LOCAL, 0, 16)], # Checking how it works with locals
      [Opt(OptOps.GROUPTOP, 0, 2)],
      [Opt(OptOps.GROUPTOP, 0, 32)],
      [Opt(OptOps.GROUPTOP, 0, 64)], # Checking how it works with grouped reduce
      [Opt(OptOps.LOCAL, 0, 2), Opt(OptOps.GROUPTOP, 0, 2)],
      [Opt(OptOps.LOCAL, 0, 16), Opt(OptOps.GROUPTOP, 0, 16)],
      [Opt(OptOps.LOCAL, 0, 32), Opt(OptOps.GROUPTOP, 0, 2)],
      # Checking how it works with locals + grouped reduce
      [Opt(OptOps.LOCAL, 0, 2), Opt(OptOps.GROUPTOP, 0, 64)],
      # Checking how it works with locals + grouped reduce + upcasts
      [Opt(OptOps.LOCAL, 0, 2), Opt(OptOps.GROUPTOP, 0, 2), Opt(OptOps.UPCAST, 0, 8), Opt(OptOps.UNROLL, 1, 4)],
      # many local + many group
      [Opt(OptOps.GROUP, 0, 2)] * 4,
      [Opt(OptOps.LOCAL, 0, 2)] * 4,
      [Opt(OptOps.LOCAL, 0, 2), Opt(OptOps.GROUP, 0, 2)] * 4,
    ])

  def test_upcasts(self):
    N = 16
    Tensor.manual_seed(1772)
    a = Tensor.rand(N, N)
    b = Tensor.rand(N, N)
    r = (a+b).sqrt() * ((a+1).exp())
    helper_linearizer_opt(r, [
      [Opt(OptOps.UPCAST, 0, 2)],
      [Opt(OptOps.UPCAST, 0, 4)],
      [Opt(OptOps.UPCAST, 0, 8)], # Checking how it works with upcasts
    ])

  def test_full_upcast(self):
    Tensor.manual_seed(1772)
    a = Tensor.rand(4)
    b = Tensor.rand(4)
    r = (a+b).sqrt() * ((a+1).exp())
    helper_linearizer_opt(r, [
      [Opt(OptOps.UPCAST, 0, 4)], # Checking how it works with upcasts
    ])

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "test requires locals")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_shared, "test requires shared")
  def test_matmul(self):
    N = 128
    Tensor.manual_seed(1552)
    a = Tensor.rand(N, N)
    b = Tensor.rand(N, N)
    r = a@b
    helper_linearizer_opt(r, [
      [Opt(OptOps.UPCAST, 0, 2)],
      [Opt(OptOps.UPCAST, 0, 4), Opt(OptOps.UPCAST, 1, 4)], # Checking how it works with upcasts
      [Opt(OptOps.LOCAL, 0, 2)],
      [Opt(OptOps.LOCAL, 1, 32)],
      [Opt(OptOps.LOCAL, 0, 4), Opt(OptOps.LOCAL, 1, 4)],
      [Opt(OptOps.LOCAL, 0, 4), Opt(OptOps.LOCAL, 1, 32)],
      [Opt(OptOps.LOCAL, 0, 16), Opt(OptOps.LOCAL, 1, 8)], # Checking how it works with locals
      [Opt(OptOps.GROUPTOP, 0, 2)],
      [Opt(OptOps.GROUPTOP, 0, 32)],
      [Opt(OptOps.GROUPTOP, 0, 32), Opt(OptOps.UNROLL, 0, 4)], # Checking how it works with grouped_reduce
      [Opt(OptOps.LOCAL, 0, 2), Opt(OptOps.LOCAL, 1, 2), Opt(OptOps.GROUPTOP, 0, 32)],
      [Opt(OptOps.LOCAL, 0, 8), Opt(OptOps.GROUPTOP, 0, 32)],
      [Opt(OptOps.LOCAL, 0, 4), Opt(OptOps.LOCAL, 0, 8), Opt(OptOps.GROUPTOP, 0, 4)], # Checking how it works with local+grouped_reduce
      # Checking all together
      [Opt(OptOps.LOCAL, 0, 4), Opt(OptOps.LOCAL, 0, 4), Opt(OptOps.GROUPTOP, 0, 8), Opt(OptOps.UNROLL, 0, 4), Opt(OptOps.UPCAST, 0, 4),
       Opt(OptOps.UPCAST, 1, 2)],
      # Full global upcast + local
      [Opt(OptOps.LOCAL, 0, 4), Opt(OptOps.LOCAL, 0, 4), Opt(OptOps.GROUPTOP, 0, 8), Opt(OptOps.UNROLL, 0, 4), Opt(OptOps.UPCAST, 0, 8)],
    ])

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "test requires locals")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_shared, "test requires shared")
  def test_double_reduce(self):
    N = 128
    Tensor.manual_seed(1552)
    a = Tensor.rand(8, N, 8, N)
    r = a.sum(axis=(1,3))
    helper_linearizer_opt(r, [
      # openCL / DEV=CL is 256 max threads
      [Opt(OptOps.GROUPTOP, 0, 2)], [Opt(OptOps.GROUPTOP, 0, 32)],
      [Opt(OptOps.GROUPTOP, 1, 2)], [Opt(OptOps.GROUPTOP, 1, 32)], # Checking how it works with 1 grouped_reduce.
      [Opt(OptOps.GROUPTOP, 0, 2), Opt(OptOps.GROUPTOP, 1, 2)],
      [Opt(OptOps.GROUPTOP, 0, 16), Opt(OptOps.GROUPTOP, 1, 2)],
      [Opt(OptOps.GROUPTOP, 0, 4), Opt(OptOps.GROUPTOP, 1, 64)], # Checking how it works with 2 grouped_reduces.
      [Opt(OptOps.GROUPTOP, 0, 16), Opt(OptOps.GROUPTOP, 1, 2), Opt(OptOps.UNROLL, 0, 4)],
      [Opt(OptOps.GROUPTOP, 0, 2), Opt(OptOps.GROUPTOP, 1, 32), Opt(OptOps.UNROLL, 2, 4)], # Checking how it works with 2 grouped_reduces + upcasts.
      [Opt(OptOps.LOCAL, 0, 4), Opt(OptOps.LOCAL, 1, 4), Opt(OptOps.GROUPTOP, 0, 4), Opt(OptOps.GROUPTOP, 1, 4)],
      # Checking how it works with 2 grouped_reduces + upcasts + locals.
      [Opt(OptOps.LOCAL, 0, 4), Opt(OptOps.LOCAL, 1, 4), Opt(OptOps.GROUPTOP, 0, 2), Opt(OptOps.GROUPTOP, 1, 32), Opt(OptOps.UNROLL, 1, 4)],
      [Opt(OptOps.LOCAL, 0, 2), Opt(OptOps.LOCAL, 1, 2), Opt(OptOps.GROUPTOP, 0, 8), Opt(OptOps.GROUPTOP, 1, 4), Opt(OptOps.UPCAST, 0, 2)],
      [Opt(OptOps.LOCAL, 0, 2), Opt(OptOps.LOCAL, 1, 2), Opt(OptOps.GROUPTOP, 0, 8), Opt(OptOps.GROUPTOP, 1, 4), Opt(OptOps.UPCAST, 0, 2),
       Opt(OptOps.UNROLL, 0, 4), Opt(OptOps.UNROLL, 1, 4)], # Checking how it works with 2 grouped_reduces + upcasts + locals.
      [Opt(OptOps.LOCAL, 0, 4), Opt(OptOps.LOCAL, 1, 4), Opt(OptOps.GROUPTOP, 0, 4), Opt(OptOps.GROUPTOP, 1, 4), Opt(OptOps.UPCAST, 0, 2),
       Opt(OptOps.UPCAST, 0, 2)], # No globals
    ])

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.tensor_cores, "test requires tensor cores")
  @unittest.skipUnless(any(tc.dtype_in == tc.dtype_out == dtypes.half for tc in Device[Device.DEFAULT].renderer.tensor_cores),
                      "test requires tensor cores with accumulation in half") # testing with half suffices.
  def test_tensor_core_opts(self):
    N = 128
    Tensor.manual_seed(1552)
    a, b = Tensor.rand(N, N, dtype=dtypes.half), Tensor.rand(N, N, dtype=dtypes.half)
    r = a.matmul(b, dtype=dtypes.half)
    atol, rtol = 0.25, 0.01
    helper_linearizer_opt(r, [
      [],
      [Opt(OptOps.UPCAST, 0, 4)],
      [Opt(OptOps.UPCAST, 1, 4)],
      [Opt(OptOps.UPCAST, 0, 4), Opt(OptOps.UPCAST, 1, 4)], # check upcasts
      [Opt(OptOps.UNROLL, 0, 2)], # check unroll
      [Opt(OptOps.UPCAST, 0, 4), Opt(OptOps.UNROLL, 0, 2)], # check combo of unroll and local
      [Opt(OptOps.UPCAST, 0, 4), Opt(OptOps.UPCAST, 1, 4), Opt(OptOps.UNROLL, 0, 2)],
      [Opt(OptOps.UPCAST, 0, 4), Opt(OptOps.UPCAST, 1, 4), Opt(OptOps.UNROLL, 0, 4)],
      [Opt(OptOps.UPCAST, 1, 4), Opt(OptOps.UPCAST, 0, 4)], # check permutations
      [Opt(OptOps.UNROLL, 0, 2), Opt(OptOps.UPCAST, 0, 4)],
      [Opt(OptOps.UPCAST, 0, 4), Opt(OptOps.UNROLL, 0, 2), Opt(OptOps.UPCAST, 1, 4)],
      [Opt(OptOps.UNROLL, 0, 2), Opt(OptOps.UPCAST, 1, 4), Opt(OptOps.UPCAST, 0, 4), Opt(OptOps.UNROLL, 0, 4)],
    ], apply_tc=True, atol=atol, rtol=rtol)

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.tensor_cores, "test requires tensor cores")
  @unittest.skipUnless(any(tc.dtype_in == tc.dtype_out == dtypes.half for tc in Device[Device.DEFAULT].renderer.tensor_cores),
                      "test requires tensor cores with accumulation in half") # testing with half suffices.
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "test requires locals")
  def test_tensor_core_opts_locals(self):
    N = 128
    Tensor.manual_seed(1552)
    a, b = Tensor.rand(N, N, dtype=dtypes.half), Tensor.rand(N, N, dtype=dtypes.half)
    r = a.matmul(b, dtype=dtypes.half)
    atol, rtol = 0.25, 0.01
    helper_linearizer_opt(r, [
      [Opt(OptOps.UNROLL, 0, 0)], # check full unroll of reduce with locals
      [Opt(OptOps.LOCAL, 0, 4)], # check local
      [Opt(OptOps.UPCAST, 0, 4), Opt(OptOps.UPCAST, 1, 4), Opt(OptOps.UNROLL, 0, 4), Opt(OptOps.LOCAL, 0, 2)],
      [Opt(OptOps.LOCAL, 0, 2), Opt(OptOps.UPCAST, 1, 4), Opt(OptOps.UNROLL, 0, 2), Opt(OptOps.UPCAST, 0, 4)],
    ], apply_tc=True, atol=atol, rtol=rtol)

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.tensor_cores, "test requires tensor cores")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "test requires locals")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_shared, "test requires shared memory")
  @unittest.skipUnless(any(tc.dtype_in == tc.dtype_out == dtypes.half for tc in Device[Device.DEFAULT].renderer.tensor_cores),
                      "test requires tensor cores with accumulation in half") # testing with half suffices.
  # NOTE: the METAL test is broken, likely due to a compiler bug. passes on CI with -O0 and with default opt level locally on M3
  @unittest.skipIf(Device.DEFAULT == "METAL", "broken for METAL")
  @unittest.skip("feature was removed")
  def test_tensor_core_opts_group(self):
    N = 128
    Tensor.manual_seed(1552)
    a, b = Tensor.rand(N, N, dtype=dtypes.half), Tensor.rand(N, N, dtype=dtypes.half)
    r = a.matmul(b, dtype=dtypes.half)
    atol, rtol = 0.25, 0.01
    helper_linearizer_opt(r, [
      [Opt(OptOps.GROUP, 0, 2)],
      [Opt(OptOps.GROUPTOP, 0, 4)],
      [Opt(OptOps.UPCAST, 0, 4), Opt(OptOps.GROUP, 0, 2)],
      [Opt(OptOps.LOCAL, 0, 4), Opt(OptOps.GROUP, 0, 2)],
      [Opt(OptOps.UNROLL, 0, 4), Opt(OptOps.GROUP, 0, 2)],
      [Opt(OptOps.UPCAST, 0, 2), Opt(OptOps.LOCAL, 0, 2), Opt(OptOps.GROUP, 0, 2)],
      [Opt(OptOps.LOCAL, 0, 2), Opt(OptOps.GROUPTOP, 0, 8), Opt(OptOps.UNROLL, 0, 2), Opt(OptOps.UPCAST, 1, 2)],
    ], apply_tc=True, atol=atol, rtol=rtol)

  def test_padto_matmul(self):
    N = 17
    Tensor.manual_seed(289)
    a = Tensor.rand(N, N)
    b = Tensor.rand(N, N)
    helper_linearizer_opt(a@b, [
      [Opt(OptOps.PADTO, 0, 32)],
      [Opt(OptOps.PADTO, 1, 32)],
      [Opt(OptOps.PADTO, 2, 32)],
      [Opt(OptOps.PADTO, 0, 32), Opt(OptOps.PADTO, 1, 32)],
      [Opt(OptOps.PADTO, 0, 32), Opt(OptOps.PADTO, 1, 32), Opt(OptOps.PADTO, 2, 32)],
      # can optimize further post PADTO
      [Opt(OptOps.PADTO, 0, 32), Opt(OptOps.PADTO, 1, 32), Opt(OptOps.UPCAST, 0, 2), Opt(OptOps.UPCAST, 1, 2),],
    ])

  def test_padto_upcasted_not_ok(self):
    N = 4
    a = Tensor.rand(N, N)
    b = Tensor.rand(N, N)
    helper_linearizer_opt(a@b, [
      [Opt(OptOps.UPCAST, 0, 0)],
      [Opt(OptOps.UPCAST, 1, 0)],
      [Opt(OptOps.UNROLL, 0, 0)],
      [Opt(OptOps.PADTO, 0, 8)],
      [Opt(OptOps.PADTO, 1, 8)],
      [Opt(OptOps.PADTO, 2, 8)],
    ])
    with self.assertRaises(KernelOptError):
      helper_linearizer_opt(a@b, [[Opt(OptOps.UPCAST, 0, 0), Opt(OptOps.PADTO, 1, 8)]])
    with self.assertRaises(KernelOptError):
      helper_linearizer_opt(a@b, [[Opt(OptOps.UPCAST, 1, 0), Opt(OptOps.PADTO, 1, 8)]])
    with self.assertRaises(KernelOptError):
      helper_linearizer_opt(a@b, [[Opt(OptOps.UNROLL, 0, 0), Opt(OptOps.PADTO, 2, 8)]])

  def test_padto_sum_ok(self):
    N = 18
    # NOTE: this setup prevents 17 * 17 contiguous merged into one dimension
    a = Tensor.rand(N, N).realize().shrink(((0, 17), (0, 17))) * 100
    b = (Tensor.rand(N, N) < 0.5).realize().shrink(((0, 17), (0, 17)))

    helper_linearizer_opt(a.sum(0), [
      [Opt(OptOps.PADTO, 0, 32)],
      [Opt(OptOps.PADTO, 0, 32), Opt(OptOps.UPCAST, 0, 8),],
    ])
    helper_linearizer_opt(a.sum(1), [
      [Opt(OptOps.PADTO, 0, 32)],
      [Opt(OptOps.PADTO, 0, 32), Opt(OptOps.UPCAST, 0, 8),],
    ])

    for axis in (0, 1):
      helper_linearizer_opt(a.sum(), [[Opt(OptOps.PADTO, axis, 32)],])
      helper_linearizer_opt(a.sum(0), [[Opt(OptOps.PADTO, axis, 32)],])
      helper_linearizer_opt(b.sum(), [[Opt(OptOps.PADTO, axis, 32)],])
      helper_linearizer_opt(b.sum(0), [[Opt(OptOps.PADTO, axis, 32)],])
      helper_linearizer_opt(b.sum(dtype=dtypes.bool), [[Opt(OptOps.PADTO, axis, 32)],])
      # TODO: why?
      if Device.DEFAULT != "WEBGPU":
        helper_linearizer_opt(b.sum(0, dtype=dtypes.bool), [[Opt(OptOps.PADTO, axis, 32)],])
        helper_linearizer_opt(b.sum(1, dtype=dtypes.bool), [[Opt(OptOps.PADTO, axis, 32)],])

    # having unsafe ops after sum is fine
    helper_linearizer_opt(a.sum().exp(), [[Opt(OptOps.PADTO, 0, 32)],])
    helper_linearizer_opt(a.sum(0).exp(), [[Opt(OptOps.PADTO, 1, 32)],])

  def test_padto_sum(self):
    N = 18
    # NOTE: this setup prevents 17 * 17 contiguous merged into one dimension
    a = Tensor.rand(N, N).shrink(((0, 17), (0, 17))).exp()
    helper_linearizer_opt(a.exp().sum(), [[Opt(OptOps.PADTO, 0, 32)],])
    helper_linearizer_opt(a.exp().sum(0), [[Opt(OptOps.PADTO, 1, 32)],])

    b = a < 1
    helper_linearizer_opt(b.sum(), [[Opt(OptOps.PADTO, 0, 32)],])
    helper_linearizer_opt(b.sum(0), [[Opt(OptOps.PADTO, 1, 32)],])

  def test_padto_max(self):
    N = 18
    # NOTE: this setup prevents 17 * 17 contiguous merged into one axis
    a = -Tensor.rand(N, N).shrink(((0, 17), (0, 17))) * 100

    helper_linearizer_opt(a.max(0), [
      [Opt(OptOps.PADTO, 0, 32)],
      [Opt(OptOps.PADTO, 0, 32), Opt(OptOps.UPCAST, 0, 8),],
    ])
    helper_linearizer_opt(a.max(1), [
      [Opt(OptOps.PADTO, 0, 32)],
      [Opt(OptOps.PADTO, 0, 32), Opt(OptOps.UPCAST, 0, 8),],
    ])

    helper_linearizer_opt(a.max(), [[Opt(OptOps.PADTO, 0, 32)],])
    helper_linearizer_opt(a.max(0), [[Opt(OptOps.PADTO, 1, 32)],])

  def test_padto_where(self):
    Tensor.manual_seed(0)
    N = 17
    a = (Tensor.randn(N, N).realize().max(axis=0, keepdim=True) > 1).where(1, 0)
    helper_linearizer_opt(a.max(0), [
      [Opt(OptOps.PADTO, 0, 32)],
      [Opt(OptOps.PADTO, 0, 32), Opt(OptOps.UPCAST, 0, 8),],
    ])

  def test_padto_where_multioutput(self):
    Tensor.manual_seed(0)
    N = 17
    r = Tensor.randn(N, N).realize().max(axis=0, keepdim=True) > 1
    a0 = r.where(1, 0)
    a1 = r.where(2, 0)
    helper_linearizer_opt([a0.max(0), a1.max(0)], [
      [Opt(OptOps.PADTO, 0, 32)],
      [Opt(OptOps.PADTO, 0, 32), Opt(OptOps.UPCAST, 0, 8),],
    ])

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "test requires locals")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_shared, "test requires shared")
  def test_color_shapes_with_local(self):
    N = 32
    Tensor.manual_seed(1552)
    a = Tensor.rand(N, N)
    b = Tensor.rand(N, N)
    r = a@b
    opts_shapes = [
      ([Opt(OptOps.LOCAL, 0, 2)], [("blue",16),("blue",32),("cyan",2),("red",32)]),
      ([Opt(OptOps.LOCAL, 0, 2),Opt(OptOps.GROUP, 0, 2)], [("blue",16),("blue",32),("cyan",2),("green",2),("red",16)]),
      # check to ensure local_dims are stable for full UNROLL of the first reduce
      ([Opt(OptOps.LOCAL, 0, 2),Opt(OptOps.UNROLL, 0, 0)], [("blue",16),("blue",32),("cyan",2),("magenta",32)]),
      ([Opt(OptOps.UNROLL, 0, 0),Opt(OptOps.LOCAL, 0, 2)], [("blue",16),("blue",32),("cyan",2),("magenta",32)]),
      # check behavior for full UNROLL on an existing GROUP
      ([Opt(OptOps.LOCAL, 0, 2),Opt(OptOps.GROUP, 0, 0),Opt(OptOps.UNROLL, 0, 2)], [("blue",16),("blue",32),("cyan",2),("green",16),("magenta",2)]),
      ([Opt(OptOps.LOCAL, 0, 2),Opt(OptOps.GROUP, 0, 0),Opt(OptOps.UNROLL, 0, 0)], [("blue",16),("blue",32),("cyan",2),("magenta",32)]),
      ([Opt(OptOps.GROUP, 0, 0),Opt(OptOps.LOCAL, 0, 2),Opt(OptOps.UNROLL, 0, 0)], [("blue",16),("blue",32),("cyan",2),("magenta",32)]),
      ([Opt(OptOps.GROUP, 0, 2),Opt(OptOps.UNROLL, 0, 0)], [("blue",32),("blue",32),("red",16),("magenta",2)]),
    ]
    helper_linearizer_opt(r, [x[0] for x in opts_shapes], color_sizes=[x[1] for x in opts_shapes])

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "test requires locals")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_shared, "test requires shared")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.supports_float4, "test requires float4")
  def test_arange_opts(self):
    a = Tensor.arange(128).clone()
    # NOTE: arange no longer has reduce ops available for opt
    helper_linearizer_opt(a, [
      #[Opt(OptOps.GROUP, 0, 32)],
      #[Opt(OptOps.GROUPTOP, 0, 32)],
      [Opt(op=OptOps.LOCAL, axis=0, arg=8)],
      [Opt(op=OptOps.LOCAL, axis=0, arg=8), Opt(op=OptOps.UPCAST, axis=0, arg=0)],
      #[Opt(op=OptOps.LOCAL, axis=0, arg=8), Opt(op=OptOps.UPCAST, axis=0, arg=0), Opt(op=OptOps.GROUP, axis=0, arg=8)],
      #[Opt(op=OptOps.LOCAL, axis=0, arg=8), Opt(op=OptOps.UPCAST, axis=0, arg=0), Opt(op=OptOps.GROUP, axis=0, arg=8), Opt(op=OptOps.UNROLL, axis=1, arg=4)], # noqa: E501
    ])

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_threads, "test requires threads")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.global_max is not None and
                       Device[Device.DEFAULT].renderer.global_max[0] > 1, "test requires multicore")
  def test_thread_opts(self):
    a = Tensor.rand(4, 4, 4, 4)
    b = Tensor.rand(4, 4, 4)
    r = (b.sqrt() + ((a+1).sum(axis=3).exp()))
    helper_linearizer_opt(r, [
      [Opt(OptOps.THREAD, 0, 2)],
      [Opt(OptOps.UPCAST, 0, 2), Opt(OptOps.THREAD, 0, 2)],
      [Opt(OptOps.UPCAST, 0, 2), Opt(OptOps.THREAD, 0, 2), Opt(OptOps.UNROLL, 0, 2)],
    ] + [[Opt(OptOps.THREAD, 0, 4)] if Device[Device.DEFAULT].renderer.global_max[0] >= 4 else []]
      + [[Opt(OptOps.THREAD, 0, 8)] if Device[Device.DEFAULT].renderer.global_max[0] >= 8 else []])

  def test_double_sum_group(self):
    a = Tensor.rand(4, 4, 4)
    r = a.sum((1, 2)).sum()
    with self.assertRaises(KernelOptError):
      helper_linearizer_opt(r, [[Opt(OptOps.GROUPTOP, 0, 16)],])
    r = a.sum((1, 2)).sum()
    with self.assertRaises(KernelOptError):
      helper_linearizer_opt(r, [[Opt(OptOps.UNROLL, 1, 4), Opt(OptOps.GROUPTOP, 0, 16)],])
    r = a.sum((1, 2)).sum()
    with self.assertRaises(KernelOptError):
      helper_linearizer_opt(r, [[Opt(OptOps.GROUPTOP, 1, 4), Opt(OptOps.GROUPTOP, 0, 16)],])

if __name__ == '__main__':
  unittest.main()
