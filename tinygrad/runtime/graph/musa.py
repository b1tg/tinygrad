import ctypes
from typing import Any, cast
import tinygrad.runtime.autogen.musa as musa
from tinygrad.helpers import dedup
from tinygrad.runtime.support.c import init_c_var
from tinygrad.device import Buffer, Device
from tinygrad.runtime.ops_musa import MUSADevice, check, encode_args, mu_time_execution
from tinygrad.engine.realize import BufferXfer, CompiledRunner
from tinygrad.engine.jit import MultiGraphRunner, GraphException

class MUSAGraph(MultiGraphRunner):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    if not all(isinstance(ji.prg, (CompiledRunner, BufferXfer)) for ji in self.jit_cache): raise GraphException

    self.jc_idx_with_updatable_bufs = dedup([x[0] for x in self.input_replace.keys()])
    self.updatable_nodes: dict[int, tuple[Any, Any, Any, bool]] = {}

    self.graph = init_c_var(musa.MUgraph, lambda x: check(musa.muGraphCreate(ctypes.byref(x), 0)))

    for j,ji in enumerate(self.jit_cache):
      if isinstance(ji.prg, CompiledRunner):
        global_size, local_size = ji.prg.p.launch_dims({v: 0 for v in self.vars})
        new_node = musa.MUgraphNode()
        deps = self._access_resources([x.base for x in ji.bufs if x is not None], ji.prg.p.outs, new_dependency=new_node)
        c_deps = (musa.MUgraphNode*len(deps))(*deps) if deps else None
        c_args, vargs = encode_args([cast(Buffer, x)._buf for x in ji.bufs], [ji.fixedvars.get(x.expr, 0) for x in ji.prg.p.vars])
        kern_params = musa.MUSA_KERNEL_NODE_PARAMS(ji.prg._prg.prg, *global_size, *local_size, 0,
                                                    ctypes.cast(0, ctypes.POINTER(ctypes.c_void_p)), vargs)
        check(musa.muGraphAddKernelNode(ctypes.byref(new_node), self.graph, c_deps, len(deps), ctypes.byref(kern_params)))
        if j in self.launch_dims_replace or j in self.var_vals_replace or j in self.jc_idx_with_updatable_bufs:
          self.updatable_nodes[j] = (new_node, kern_params, c_args, False)
      elif isinstance(ji.prg, BufferXfer):
        dest, src = [cast(Buffer, x) for x in ji.bufs[0:2]]
        src_dev = cast(MUSADevice, Device[src.device])
        node_from = musa.MUgraphNode()
        deps = self._access_resources(bufs=[dest.base, src.base], write=[0], new_dependency=node_from)
        c_deps = (musa.MUgraphNode*len(deps))(*deps) if deps else None
        cp_params = musa.MUSA_MEMCPY3D_v2(srcMemoryType=musa.MU_MEMORYTYPE_DEVICE, srcDevice=src._buf, srcPitch=src.nbytes, srcHeight=1,
                                          dstMemoryType=musa.MU_MEMORYTYPE_DEVICE, dstDevice=dest._buf, dstPitch=dest.nbytes, dstHeight=1,
                                          WidthInBytes=dest.nbytes, Height=1, Depth=1)
        check(musa.muGraphAddMemcpyNode(ctypes.byref(node_from), self.graph, c_deps, len(deps), ctypes.byref(cp_params), src_dev.context))
        if j in self.jc_idx_with_updatable_bufs: self.updatable_nodes[j] = (node_from, cp_params, src_dev.context, True)

    self.instance = init_c_var(musa.MUgraphExec, lambda x: check(musa.muGraphInstantiate_v2(ctypes.byref(x), self.graph, None, None, 0)))

  def __call__(self, input_buffers: list[Buffer], var_vals: dict[str, int], wait=False) -> float|None:
    for (j,i),input_idx in self.input_replace.items():
      if not self.updatable_nodes[j][3]: setattr(self.updatable_nodes[j][2], f'f{i}', input_buffers[input_idx]._buf)
      else:
        if i == 0: self.updatable_nodes[j][1].destDevice = input_buffers[input_idx]._buf
        elif i == 1: self.updatable_nodes[j][1].srcDevice = input_buffers[input_idx]._buf

    for j, i, v in self.updated_vars(var_vals): setattr(self.updatable_nodes[j][2], f'v{i}', v)

    for j, global_dims, local_dims in self.updated_launch_dims(var_vals):
      node = self.updatable_nodes[j][1]
      node.blockDimX, node.blockDimY, node.blockDimZ, node.gridDimX, node.gridDimY, node.gridDimZ = *local_dims, *global_dims # type: ignore[misc]

    # MUSA driver doesn't support muGraphExec*SetParams nor muGraphExecUpdate (both 801). Re-instantiate each call.
    for node, c_node_params, c_args, is_copy in self.updatable_nodes.values():
      if not is_copy: check(musa.muGraphKernelNodeSetParams(node, ctypes.byref(c_node_params)))
      else: check(musa.muGraphMemcpyNodeSetParams(node, ctypes.byref(c_node_params)))
    if self.updatable_nodes:
      check(musa.muGraphExecDestroy(self.instance))
      self.instance = init_c_var(musa.MUgraphExec, lambda x: check(musa.muGraphInstantiate_v2(ctypes.byref(x), self.graph, None, None, 0)))

    return mu_time_execution(lambda: check(musa.muGraphLaunch(self.instance, None)), enable=wait)

  def __del__(self):
    if hasattr(self, 'graph'): check(musa.muGraphDestroy(self.graph))
    if hasattr(self, 'instance'): check(musa.muGraphExecDestroy(self.instance))
