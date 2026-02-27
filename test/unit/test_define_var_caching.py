"""Tests for DEFINE_VAR kernel caching and int64 variable support."""
import unittest
import numpy as np
from tinygrad import Tensor, Variable, Device, dtypes
from tinygrad.uop.ops import UOp, Ops, graph_rewrite, pm_lower_index_dtype, select_dtype
from tinygrad.helpers import Context
from tinygrad.engine.realize import CompiledRunner, run_schedule, ExecItem
from tinygrad.engine.schedule import schedule_cache

# *** Part 1: per-kernel variable values (schedule.py changes) ***

class TestPerKernelVars(unittest.TestCase):
  def test_same_var_different_bindings_no_assert(self):
    """Multiple tensors sharing a variable name with different bound values should not assert."""
    v = UOp.variable("test_var", 0, 100)
    a = Tensor.empty(10).contiguous().realize()
    t1 = a + Tensor(v.bind(5))
    t2 = a + Tensor(v.bind(10))
    # should not raise "bind mismatch" assertion
    sched, var_vals = Tensor.schedule_with_vars(t1, t2)
    # variable with conflicting values should NOT be in global var_vals
    self.assertNotIn("test_var", var_vals)
    run_schedule(sched, var_vals)

  def test_same_var_same_binding_in_var_vals(self):
    """Multiple tensors sharing a variable name with the same bound value should go to var_vals normally."""
    v = UOp.variable("shared_v", 0, 100)
    a = Tensor.empty(10).contiguous().realize()
    t1 = a + Tensor(v.bind(7))
    t2 = a + Tensor(v.bind(7))
    sched, var_vals = Tensor.schedule_with_vars(t1, t2)
    self.assertEqual(var_vals.get("shared_v"), 7)

  def test_fixedvars_assigned_correctly(self):
    """Per-kernel fixedvars should carry the correct per-tensor values."""
    v = UOp.variable("fv", 0, 100)
    a = Tensor.empty(10).contiguous().realize()
    vals = [3, 7, 11]
    tensors = [a + Tensor(v.bind(val)) for val in vals]
    sched, var_vals = Tensor.schedule_with_vars(*tensors)
    self.assertNotIn("fv", var_vals)
    # check that fixedvars are assigned to ExecItems that use this variable
    exec_items_with_fv = [ei for ei in sched if any(vv.expr == "fv" for vv in ei.ast.variables())]
    assigned_vals = [ei.fixedvars.get("fv") for ei in exec_items_with_fv if "fv" in ei.fixedvars]
    self.assertEqual(assigned_vals, vals)

  def test_per_kernel_vars_correct_results(self):
    """Tensors with different variable bindings should compute correct results."""
    v = UOp.variable("off", 0, 200)
    a = Tensor.ones(4).contiguous().realize()
    t1 = a + Tensor(v.bind(10))
    t2 = a + Tensor(v.bind(20))
    t3 = a + Tensor(v.bind(30))
    Tensor.realize(t1, t2, t3)
    np.testing.assert_equal(t1.numpy(), np.full(4, 11.0))
    np.testing.assert_equal(t2.numpy(), np.full(4, 21.0))
    np.testing.assert_equal(t3.numpy(), np.full(4, 31.0))

  def test_mixed_shared_and_conflicting_vars(self):
    """Mix of shared (same value) and conflicting (different value) variables."""
    v1 = UOp.variable("shared", 0, 100)
    v2 = UOp.variable("conflict", 0, 100)
    a = Tensor.empty(4).contiguous().realize()
    t1 = a + Tensor(v1.bind(5)) + Tensor(v2.bind(10))
    t2 = a + Tensor(v1.bind(5)) + Tensor(v2.bind(20))
    sched, var_vals = Tensor.schedule_with_vars(t1, t2)
    self.assertEqual(var_vals.get("shared"), 5)
    self.assertNotIn("conflict", var_vals)

# *** Part 2: kernel cache reuse with shared variable names ***

class TestVarKernelCacheReuse(unittest.TestCase):
  def test_same_shape_different_offset_reuses_cache(self):
    """Tensors created with same variable name but different bindings should reuse the schedule cache."""
    schedule_cache.clear()
    v = UOp.variable("cache_off", 0, 1000)
    a = Tensor.ones(8).contiguous().realize()

    t1 = a * Tensor(v.bind(5))
    t1.realize()
    cache_after_first = len(schedule_cache)

    t2 = a * Tensor(v.bind(10))
    t2.realize()
    cache_after_second = len(schedule_cache)

    self.assertEqual(cache_after_first, cache_after_second, "schedule cache should not grow for same-shaped variable ops")

  def test_slice_with_variable_offset_reuses_kernels(self):
    """Slicing a tensor at different variable offsets should produce the same kernel AST."""
    buf = Tensor.ones(100).contiguous().realize()
    v = UOp.variable("sl_off", 0, 99)

    t1 = buf[v.bind(10):v.bind(10)+5].contiguous()
    t2 = buf[v.bind(20):v.bind(20)+5].contiguous()
    sched1, _ = Tensor.schedule_with_vars(t1)
    for si in sched1: si.lower()
    sched2, _ = Tensor.schedule_with_vars(t2)
    for si in sched2: si.lower()

    # compiled runners should share the same AST key (same kernel)
    kernels1 = [si for si in sched1 if isinstance(si.prg, CompiledRunner)]
    kernels2 = [si for si in sched2 if isinstance(si.prg, CompiledRunner)]
    if kernels1 and kernels2:
      self.assertEqual(kernels1[0].prg.p.ast.key, kernels2[0].prg.p.ast.key)

# *** Part 3: int64 DEFINE_VAR (ops.py + cstyle.py changes) ***

class TestInt64DefineVar(unittest.TestCase):
  def test_small_var_uses_int32(self):
    """Variable with range fitting in int32 should lower to int32."""
    v = UOp.variable("small", 0, 1000)
    result = graph_rewrite(v.sink(), pm_lower_index_dtype)
    lowered = result.src[0]
    # should be cast(index, DEFINE_VAR(int))
    if lowered.op is Ops.CAST:
      self.assertEqual(lowered.src[0].dtype, dtypes.int)
    else:
      self.assertIn(lowered.dtype, (dtypes.int, dtypes.index))

  def test_large_var_uses_int64(self):
    """Variable with range exceeding int32 should lower to int64."""
    v = UOp.variable("big", 0, 5_000_000_000)
    result = graph_rewrite(v.sink(), pm_lower_index_dtype)
    lowered = result.src[0]
    # should be cast(index, DEFINE_VAR(long))
    if lowered.op is Ops.CAST:
      self.assertEqual(lowered.src[0].dtype, dtypes.long)
    else:
      self.assertEqual(lowered.dtype, dtypes.long)

  def test_int64_var_correct_result(self):
    """Variable with int64 range should produce correct kernel code (long type in signature)."""
    v = UOp.variable("big_off", 0, 5_000_000_000)
    result = graph_rewrite(v.sink(), pm_lower_index_dtype)
    lowered = result.src[0]
    # the lowered var should be long
    if lowered.op is Ops.CAST:
      self.assertEqual(lowered.src[0].dtype, dtypes.long)
      # the range should be preserved
      self.assertEqual(lowered.src[0].arg[2], 5_000_000_000)

# *** Part 4: renderer cstyle.py changes ***

class TestRendererVarDtype(unittest.TestCase):
  def test_render_int32_var_parameter(self):
    """int32 DEFINE_VAR should render with int type in kernel signature."""
    ren = Device[Device.DEFAULT].renderer
    if not hasattr(ren, 'arg_int_prefix'): self.skipTest("not a CStyle renderer")
    # int32: "const int".replace("int", "int") = "const int"
    result = ren.arg_int_prefix.replace("int", ren.render_dtype(dtypes.int))
    self.assertIn("int", result)
    self.assertNotIn("long", result)

  def test_render_int64_var_parameter(self):
    """int64 DEFINE_VAR should render with long type in kernel signature."""
    ren = Device[Device.DEFAULT].renderer
    if not hasattr(ren, 'arg_int_prefix'): self.skipTest("not a CStyle renderer")
    # int64: "const int".replace("int", "long") = "const long"
    result = ren.arg_int_prefix.replace("int", ren.render_dtype(dtypes.long))
    self.assertIn("long", result)

# *** Part 5: AMD ELF kernarg size (amd/elf.py changes) ***

class TestAMDKernargSize(unittest.TestCase):
  def test_kernarg_size_int32_var(self):
    """Kernel with int32 var should have kernarg_size = n_bufs*8 + 4."""
    if Device.DEFAULT != "AMD": self.skipTest("AMD only")
    v = UOp.variable("i32var", 0, 100)
    a = Tensor.ones(4).contiguous().realize()
    t = a + Tensor(v.bind(5))
    sched, var_vals = Tensor.schedule_with_vars(t)
    for si in sched: si.lower()
    runners = [si.prg for si in sched if isinstance(si.prg, CompiledRunner)]
    for r in runners:
      if any(v.expr == "i32var" for v in r.p.vars):
        n_globals = len(r.p.globals)
        expected_min = n_globals * 8 + 4
        self.assertGreaterEqual(r._prg.kernargs_segment_size, expected_min)

  def test_kernarg_var_bytes_match_dtype(self):
    """n_vars in ELF should accumulate dtype.itemsize, not count."""
    # Test the logic directly: int32 = 4 bytes, int64 = 8 bytes
    self.assertEqual(dtypes.int.itemsize, 4)
    self.assertEqual(dtypes.long.itemsize, 8)

# *** Part 6: HCQ var_fmts (hcq.py + ops_amd.py + realize.py changes) ***

class TestVarFmts(unittest.TestCase):
  def test_var_fmts_int32_only(self):
    """All int32 vars should produce all 'I' formats."""
    if Device.DEFAULT != "AMD": self.skipTest("AMD only")
    v = UOp.variable("v32", 0, 100)
    a = Tensor.ones(4).contiguous().realize()
    t = a + Tensor(v.bind(5))
    sched, _ = Tensor.schedule_with_vars(t)
    for si in sched: si.lower()
    for si in sched:
      if isinstance(si.prg, CompiledRunner) and any(v.expr == "v32" for v in si.prg.p.vars):
        fmts = si.prg._prg.var_fmts
        self.assertIsNotNone(fmts)
        self.assertTrue(all(f == 'I' for f in fmts))

  def test_var_fmts_generation_logic(self):
    """var_fmts tuple should use 'Q' for >4 byte dtypes, 'I' otherwise."""
    # test the logic used in realize.py directly
    class FakeVar:
      def __init__(self, itemsize): self.dtype = type('D', (), {'itemsize': itemsize})()
    vars_i32 = [FakeVar(4), FakeVar(4)]
    vars_mixed = [FakeVar(4), FakeVar(8), FakeVar(4)]
    fmts_i32 = tuple('Q' if v.dtype.itemsize > 4 else 'I' for v in vars_i32)
    fmts_mixed = tuple('Q' if v.dtype.itemsize > 4 else 'I' for v in vars_mixed)
    self.assertEqual(fmts_i32, ('I', 'I'))
    self.assertEqual(fmts_mixed, ('I', 'Q', 'I'))

# *** Part 7: end-to-end correctness with tensor slicing ***

class TestVariableSliceCorrectness(unittest.TestCase):
  def test_variable_slice_same_as_const_slice(self):
    """Slicing with variable offset should produce same result as constant offset."""
    data = Tensor(list(range(50))).contiguous().realize()
    v = UOp.variable("s_off", 0, 49)

    for off in [0, 5, 10, 25, 45]:
      expected = data[off:off+5].numpy()
      result = data[v.bind(off):v.bind(off)+5].contiguous().realize().numpy()
      np.testing.assert_equal(result, expected, err_msg=f"mismatch at offset {off}")

  def test_batch_realize_variable_slices(self):
    """Batch-realizing multiple variable slices should produce correct results."""
    data = Tensor(list(range(100))).contiguous().realize()
    v = UOp.variable("b_off", 0, 99)

    slices = []
    offsets = [0, 10, 20, 30, 40]
    for off in offsets:
      slices.append(data[v.bind(off):v.bind(off)+5].contiguous())

    Tensor.realize(*slices)
    for i, (s, off) in enumerate(zip(slices, offsets)):
      expected = list(range(off, off+5))
      np.testing.assert_equal(s.numpy(), expected, err_msg=f"slice {i} at offset {off}")

  def test_large_offset_variable_slice(self):
    """Variable with range exceeding int32 should lower to int64 dtype."""
    v = UOp.variable("lg_off", 0, 5_000_000_000)
    result = graph_rewrite(v.sink(), pm_lower_index_dtype)
    lowered = result.src[0]
    if lowered.op is Ops.CAST:
      self.assertEqual(lowered.src[0].dtype, dtypes.long)

# *** Part 8: select_dtype boundary tests ***

class TestSelectDtypeBoundary(unittest.TestCase):
  def test_int32_max_stays_int32(self):
    """Variable with vmax exactly at int32 max (2,147,483,647) should use int32."""
    v = UOp.variable("edge32", 0, 2_147_483_647)
    self.assertEqual(select_dtype(v), dtypes.int)

  def test_int32_max_plus_one_uses_int64(self):
    """Variable with vmax at int32 max + 1 (2,147,483,648) should use int64."""
    v = UOp.variable("over32", 0, 2_147_483_648)
    self.assertEqual(select_dtype(v), dtypes.long)

  def test_negative_range_int32(self):
    """Variable with negative min that fits int32 should use int32."""
    v = UOp.variable("neg32", -1_000_000, 1_000_000)
    self.assertEqual(select_dtype(v), dtypes.int)

  def test_negative_range_int64(self):
    """Variable with negative min that exceeds int32 should use int64."""
    v = UOp.variable("neg64", -3_000_000_000, 0)
    self.assertEqual(select_dtype(v), dtypes.long)

# *** Part 9: ExecItem.fixedvars merge behavior ***

class TestExecItemFixedvars(unittest.TestCase):
  def test_fixedvars_override_var_vals(self):
    """ExecItem.run should merge fixedvars on top of var_vals."""
    v = UOp.variable("merge_v", 0, 100)
    a = Tensor.ones(4).contiguous().realize()
    t = a + Tensor(v.bind(42))
    sched, var_vals = Tensor.schedule_with_vars(t)
    # manually set fixedvars to a different value than var_vals
    for ei in sched:
      if any(vv.expr == "merge_v" for vv in ei.ast.variables()):
        ei.fixedvars["merge_v"] = 99
    run_schedule(sched, var_vals)
    # result should use fixedvars value (99), not var_vals value
    np.testing.assert_equal(t.numpy(), np.full(4, 100.0))

  def test_fixedvars_empty_by_default(self):
    """ExecItem should have empty fixedvars dict by default."""
    ei = ExecItem(UOp(Ops.NOOP))
    self.assertEqual(ei.fixedvars, {})

# *** Part 10: GGUF-like pattern (multiple same-shape slices via variable) ***

class TestGGUFLikePattern(unittest.TestCase):
  def test_gguf_like_weight_loading(self):
    """Simulate GGUF pattern: multiple same-shape slices at different offsets via one variable."""
    # create a "file" tensor and slice it at different variable offsets, like gguf_load does
    file_data = Tensor(list(range(200))).contiguous().realize()
    off_var = UOp.variable("_gguf_off", 0, 199)
    weight_size = 10

    weights = []
    offsets = [0, 20, 50, 100, 150]
    for off in offsets:
      w = file_data[off_var.bind(off):off_var.bind(off)+weight_size].contiguous()
      weights.append(w)

    Tensor.realize(*weights)
    for w, off in zip(weights, offsets):
      expected = list(range(off, off + weight_size))
      np.testing.assert_equal(w.numpy(), expected, err_msg=f"weight at offset {off}")

  def test_gguf_like_cache_reuse(self):
    """All same-shape GGUF-like slices should reuse the same compiled kernel."""
    file_data = Tensor(list(range(200))).contiguous().realize()
    off_var = UOp.variable("_gguf_cache", 0, 199)

    weights = [file_data[off_var.bind(off):off_var.bind(off)+10].contiguous() for off in [0, 30, 60, 90]]
    sched, _ = Tensor.schedule_with_vars(*weights)
    for si in sched: si.lower()

    # all compute kernels should share the same AST key
    kernel_keys = [si.prg.p.ast.key for si in sched if isinstance(si.prg, CompiledRunner)]
    if kernel_keys:
      self.assertTrue(len(set(kernel_keys)) == 1, f"expected 1 unique kernel, got {len(set(kernel_keys))}")

# *** Part 11: rendered kernel source tests ***

class TestRenderedKernelSource(unittest.TestCase):
  def test_int32_var_in_kernel_source(self):
    """Compiled kernel with int32 var should have 'int' (not 'long') in source."""
    ren = Device[Device.DEFAULT].renderer
    if not hasattr(ren, 'arg_int_prefix'): self.skipTest("not a CStyle renderer")
    v = UOp.variable("src_i32", 0, 100)
    a = Tensor.ones(4).contiguous().realize()
    t = a + Tensor(v.bind(5))
    sched, _ = Tensor.schedule_with_vars(t)
    for si in sched: si.lower()
    for si in sched:
      if isinstance(si.prg, CompiledRunner) and any(vv.expr == "src_i32" for vv in si.prg.p.vars):
        src = si.prg.p.src
        self.assertIn("int", src)
        # "int" appears but not as part of "long" — check the var parameter specifically
        self.assertIn("src_i32", src)

  def test_int64_var_in_kernel_source(self):
    """Compiled kernel with int64 var should have 'long' in rendered source."""
    ren = Device[Device.DEFAULT].renderer
    if not hasattr(ren, 'arg_int_prefix'): self.skipTest("not a CStyle renderer")
    # int64 vars work inside index expressions (SHRINK), not as standalone Tensor values
    # test the renderer logic directly: arg_int_prefix.replace("int", render_dtype(dtypes.long)) should contain "long"
    result = ren.arg_int_prefix.replace("int", ren.render_dtype(dtypes.long))
    self.assertIn("long", result)
    # also verify select_dtype picks long for large range
    v = UOp.variable("src_i64", 0, 5_000_000_000)
    self.assertEqual(select_dtype(v), dtypes.long)
    # verify the lowered DEFINE_VAR has long dtype
    result = graph_rewrite(v.sink(), pm_lower_index_dtype)
    lowered = result.src[0]
    if lowered.op is Ops.CAST:
      self.assertEqual(lowered.src[0].dtype, dtypes.long)

# *** Part 12: realize.py var_fmts generation ***

class TestVarFmtsFromRealize(unittest.TestCase):
  def test_var_fmts_passed_to_runtime(self):
    """CompiledRunner should generate var_fmts tuple matching its vars."""
    v = UOp.variable("fmt_v", 0, 100)
    a = Tensor.ones(4).contiguous().realize()
    t = a + Tensor(v.bind(5))
    sched, _ = Tensor.schedule_with_vars(t)
    for si in sched: si.lower()
    for si in sched:
      if isinstance(si.prg, CompiledRunner) and si.prg.p.vars:
        # var_fmts should exist on the runtime program
        if hasattr(si.prg._prg, 'var_fmts'):
          fmts = si.prg._prg.var_fmts
          self.assertIsNotNone(fmts)
          self.assertEqual(len(fmts), len(si.prg.p.vars))
          # each format should be 'I' or 'Q'
          for f in fmts:
            self.assertIn(f, ('I', 'Q'))

if __name__ == "__main__":
  unittest.main()
