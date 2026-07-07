import time, inspect
from collections import deque
from tinygrad.uop.ops import UOp, Ops, GroupOp, UOpMetaClass, track_rewrites, graph_rewrite, gate_kernel_sink, KernelInfo
from tinygrad.uop.spec import type_verify, spec_tensor
from tinygrad.helpers import DEBUG, cpu_profile, TracingKey, SPEC, pluralize, SCACHE, BASEDIR, partition

# **** schedule linearizer

# unwrap VIEW/CAST/etc to find the actual data source (kernel output, buffer, or multi-device op)
def _unwrap_src(s: UOp) -> UOp:
  while len(s.src) and s.op not in {Ops.AFTER, Ops.BUFFER, Ops.PARAM, Ops.MSELECT, Ops.MSTACK, Ops.BIND}: s = s.src[0]
  return s

def _split_after(after: UOp, cache:dict|None=None) -> tuple[tuple[UOp, ...], tuple[UOp, ...]]:
  if cache is not None and (ret:=cache.get(after)) is not None: return ret
  # deps are scheduling-only: movement wrappers on them carry no meaning, strip to base
  srcs = [s.base if s.op in GroupOp.Movement else s for s in after.src[1:]]
  kernels, remaining = partition(srcs, lambda s: s.op in {Ops.CALL, Ops.END})
  deps, remaining = partition(remaining, lambda s: s.op is Ops.AFTER)
  if invalid := [s for s in remaining if s.op is not Ops.STORE]:
    raise AssertionError(f"AFTER source should be CALL, END, STORE, or AFTER, not {invalid[0].op}")
  # nested AFTER with no producer at this level (a pure scheduling pin): the producer kernels live in the inner
  # AFTER. write chains have their own store per level and data-dep on the inner, so they never recurse here
  if after.src[0].op is Ops.AFTER and not any(k.op in {Ops.CALL, Ops.END} and _touches(k, after.buf_uop) for k in kernels):
    kernels = list(kernels) + list(_split_after(after.src[0], cache)[0])
  ret = (tuple(kernels), tuple(deps))
  if cache is not None: cache[after] = ret
  return ret

# a kernel in an AFTER that doesn't touch the AFTER's buffer is an explicit .after() scheduling pin:
# it must still run, but the buffer's producers are additionally ordered after it
def _touches(k:UOp, buf:UOp) -> bool:
  call = k.src[0] if k.op is Ops.END else k
  return any((su:=_unwrap_src(s)) is buf or (su.op is Ops.AFTER and su.buf_uop is buf) for s in call.src[1:])

def create_schedule(sched_sink:UOp) -> UOp:
  with cpu_profile(TracingKey("toposort sched_sink")):
    # build kernel dependency graph: edges from producer kernel to consumer kernels
    children: dict[UOp, list[UOp]] = {}
    in_degree: dict[UOp, int] = {}
    # per-AFTER caches: _split_after recursion and _touches walks are quadratic on long assign chains without them
    sa_cache: dict[UOp, tuple[tuple[UOp, ...], tuple[UOp, ...]]] = {}
    split_cache: dict[UOp, tuple[tuple[UOp, ...], tuple[UOp, ...]]] = {}
    def split_touching(s:UOp) -> tuple[tuple[UOp, ...], tuple[UOp, ...]]:
      if s not in split_cache:
        ks = _split_after(s, sa_cache)[0]
        split_cache[s] = (ks, ks if len(ks) <= 1 else tuple(t for t in ks if _touches(t, s.buf_uop)))
      return split_cache[s]
    for u in sched_sink.toposort(gate_kernel_sink):
      if u.op is not Ops.AFTER: continue
      kernels, after_deps = _split_after(u, sa_cache)
      pins = tuple(k for k in kernels if k not in split_touching(u)[1]) if len(kernels) > 1 else ()
      for k in kernels:
        in_degree.setdefault(k, 0)
        if k.op is Ops.END: assert k.src[0].op is Ops.CALL, f"END src[0] should be KERNEL, not {k.src[0].op}"
        kernel_deps = k.src[0].src[1:] if k.op is Ops.END else k.src[1:]
        if k not in pins:  # order this buffer's producers after the pins
          for p in pins:
            # allreduce pins retarget to the collective's local input, same as the after_deps path below
            targets = _split_after(tt, sa_cache)[0] if p.op is Ops.CALL and p.arg.name == 'allreduce' \
              and (tt:=_unwrap_src(p.src[2])).op is Ops.AFTER else (p,)
            for t2 in targets:
              children.setdefault(t2, []).append(k)
              in_degree[k] += 1
        for si, s in enumerate(kernel_deps + after_deps):
          is_pin_dep = si >= len(kernel_deps)
          match (s := _unwrap_src(s)).op:
            case Ops.AFTER:
              # pins riding on this AFTER (kernels that don't write its buffer) aren't its producers: the pin
              # ordering (pin -> producer) already sequences them, and edges from them here reverse pin2-style
              # pins into cycles. wait only on the kernels that actually write the buffer
              for t in split_touching(s)[1]:
                # an explicit pin on an allreduce output means "after its local input" (the CALL's partial arg):
                # data deps already order readers after the full collective, while overlapping the copies needs this.
                # SEMANTIC CHOICE: any .after() pin on a sharded-reduce tensor therefore guarantees "local partial
                # done", NOT "collective done" — if you ever need the latter, read the value instead of pinning
                targets = _split_after(tt, sa_cache)[0] if is_pin_dep and t.op is Ops.CALL and t.arg.name == 'allreduce' \
                  and (tt:=_unwrap_src(t.src[2])).op is Ops.AFTER else (t,)
                for t2 in targets:
                  children.setdefault(t2, []).append(k)
                  in_degree[k] += 1
            case Ops.MSELECT | Ops.MSTACK:
              for ss in s.src:
                if ss.op is Ops.MSELECT: ss = ss.src[0]
                ss = _unwrap_src(ss)
                if ss.op not in {Ops.BUFFER, Ops.PARAM}:
                  assert ss.op is Ops.AFTER, f"ss.op is not AFTER, it's {ss.op}"
                  for t in _split_after(ss, sa_cache)[0]:
                    children.setdefault(t, []).append(k)
                    in_degree[k] += 1
            case Ops.BUFFER | Ops.PARAM | Ops.BIND:
              pass  # BUFFER/PARAM is already realized, BIND is a bound variable (not a buffer dependency)
            case _:
              raise RuntimeError(f"input to kernel must be AFTER, BUFFER, PARAM, MSELECT, MSTACK, or BIND, not {s.op}")

  with cpu_profile(TracingKey("linearize schedule")):
    queue: deque[UOp] = deque(k for k,v in in_degree.items() if v == 0)
    linearized: list[UOp] = []
    while len(queue):
      rk = queue.popleft()
      if rk.op is Ops.LINEAR:
        linearized.extend(rk.src)
      else:
        k = rk.src[0] if rk.op is Ops.END else rk
        assert k.op is Ops.CALL, f"unexpected op in queue: {k.op}"
        buf_uops = tuple(_unwrap_src(s).buf_uop for s in k.src[1:] if s.op is not Ops.BIND)
        linearized.append(k.src[0].call(*buf_uops, metadata=k.arg.metadata))
      for x in children.get(rk, []):
        in_degree[x] -= 1
        if in_degree[x] == 0: queue.append(x)
    if stuck := [k for k,v in in_degree.items() if v > 0]:
      if DEBUG >= 1:
        stuck_set = set(stuck)
        def desc(k):
          kk = k.src[0] if k.op is Ops.END else k
          sizes = [s.max_numel() if (s:=_unwrap_src(ss)).op in {Ops.BUFFER, Ops.PARAM} and s._shape is not None else str(s.op)
                   for ss in (kk.src[1:] if kk.op is Ops.CALL else ())]
          return f"{str(k.op).replace('Ops.','')}(bufs={sizes})"
        for k in stuck:
          waiting = [t for t, kids in children.items() if k in kids and t in stuck_set]
          print(f"STUCK {desc(k)} indeg={in_degree[k]} waits_on_stuck={[desc(w) for w in waiting]}")
      raise RuntimeError(f"schedule dependency cycle: {len(stuck)} kernels never became ready (check .after() pin directions)")
  return UOp(Ops.LINEAR, src=tuple(linearized))

from tinygrad.schedule.memory import memory_plan_rewrite
from tinygrad.engine.realize import capturing, pm_flatten_linear
from tinygrad.schedule.rangeify import get_kernel_graph
from tinygrad.helpers import CAPTURING
from tinygrad.uop.ops import PatternMatcher, UPat, ParamArg
from tinygrad.dtype import AddrSpace

def create_new_buffer(ctx:tuple[dict[UOp, UOp], tuple[UOp, ...]], b:UOp):
  if (ret:=ctx[0].get(b, None)) is None: ctx[0][b] = ret = UOp.new_buffer(b.device, b.max_numel(), b.dtype)
  return ret

pm_post_sched_cache = PatternMatcher([
  (UPat(Ops.PARAM, name="x"), lambda ctx,x: ctx[1][x.arg.slot]),
  # create new BUFFERs
  (UPat(Ops.BUFFER, src=(UPat(),), name="b"), lambda ctx,b:
   create_new_buffer(ctx, b) if isinstance(b.arg, ParamArg) and b.addrspace is AddrSpace.GLOBAL else None),
])

pm_resolve_linear_call = PatternMatcher([
  # call LINEAR is resolved here
  (UPat(Ops.CALL, src=(UPat(Ops.LINEAR),), name="linear_call", allow_any_len=True), lambda linear_call:
   graph_rewrite(linear_call.src[0], pm_post_sched_cache, ctx=({}, linear_call.src[1:]), walk=True, name="params to buffers")),
])+pm_flatten_linear

schedule_cache: dict[bytes, UOp] = {}
# ctx is just for DEBUG on inner
def lower_sink_to_linear(function:UOp) -> UOp|None:
  st = time.perf_counter()
  if isinstance(function.arg, KernelInfo): return None
  cache_key = function.key
  if not SCACHE or (sc_ret:=schedule_cache.get(cache_key, None)) is None:
    if SPEC: type_verify(function, spec_tensor)
    # support recursive CALLs
    linear = create_schedule(get_kernel_graph(function))
    if SCACHE: schedule_cache[cache_key] = linear
  else:
    # schedule cache hit
    linear = sc_ret
  if (DEBUG >= 1 and len(linear.src) > 1) or DEBUG >= 3:
    for frm in inspect.stack():
      if frm.filename == "<string>": continue
      if frm.filename.startswith(str(BASEDIR / "apps")): break
      if not frm.filename.startswith(str(BASEDIR)) and not frm.filename.endswith("/contextlib.py"): break
    else:
      frm = None
    print(f"scheduled {len(linear.src):5d} kernels in {(time.perf_counter()-st)*1000:8.2f} ms"+\
          f" | {' cache hit' if SCACHE and sc_ret is not None else 'CACHE MISS'} {cache_key.hex()[:8]}"+\
          f" | {len(UOpMetaClass.ucache):7d} uops in cache"+("" if frm is None else f" | {frm.filename}:{frm.lineno}"))
  return linear

pm_schedule = PatternMatcher([
  (UPat(Ops.SINK, name="function"), lower_sink_to_linear),
])

@track_rewrites(lambda _,ret: f"Schedule {pluralize('Kernel', len(ret[0].src))}")
def create_linear_with_vars(big_sink:UOp) -> tuple[UOp, dict[str, int]]:
  # big_sink srcs are all the Tensors
  linear_call = graph_rewrite(big_sink, pm_schedule, name="schedule to linear", enter_calls=True)

  # this recursively resolves the linear_call and allocates buffers
  linear = graph_rewrite(linear_call, pm_resolve_linear_call, name="resolve linear call")

  # vars used in the schedule
  used_vars = set().union(*[{v.expr for v in si.src[0].variables()} for si in linear.src])
  # get var_vals
  var_vals: dict[str, int] = {}
  for b in big_sink.src[1:]:
    if b.op is Ops.BIND:
      nm = b.src[0].expr
      if nm not in used_vars: continue
      val = b.src[1].arg
      if var_vals.get(nm, val) != val: raise RuntimeError(f"bind mismatch on {nm}, {var_vals[nm]} != {val}")
      var_vals[nm] = val

  # jit captures this schedule, no need to execute.
  if len(capturing) and CAPTURING:
    capturing[0].add_linear(linear, var_vals)
    return UOp(Ops.LINEAR, src=()), var_vals

  held_bufs = ({b for b in linear_call.src[1:] if b.op is Ops.BUFFER} if linear_call.op is Ops.CALL else set())
  return memory_plan_rewrite(linear, held_bufs), var_vals
