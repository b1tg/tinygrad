from __future__ import annotations
import functools

from tinygrad import Tensor, UOp, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import prod
from tinygrad.uop.ops import AxisType, KernelInfo, Ops

Q4_0_BLOCK_SIZE, Q4_0_BLOCK_BYTES = 32, 18
Q8_0_BLOCK_SIZE, Q8_0_BLOCK_BYTES = 32, 34


def _index_sint(value:int|UOp) -> int|UOp:
  if isinstance(value, int): return value
  value, _ = value.unbind_all()
  variables = [u for u in value.toposort() if u.op is Ops.PARAM and u.addrspace is AddrSpace.ALU]
  return value.substitute({v:UOp.variable(v.expr, v.vmin, v.vmax, dtype=dtypes.int, multiple_of=v.arg.multiple_of) for v in variables})


def _amd_sdot4(a:UOp, b:UOp, c:UOp) -> UOp:
  return UOp(Ops.CUSTOMI, dtypes.int32, (a.int(), b.int(), c), arg="__builtin_amdgcn_sdot4({0}, {1}, {2}, false)")


def _amd_fdot2(a:tuple[UOp, UOp], b:tuple[UOp, UOp], c:UOp) -> UOp:
  return UOp(Ops.CUSTOMI, dtypes.float32, (*a, *b, c),
             arg="__builtin_amdgcn_fdot2((_Float16 __attribute__((ext_vector_type(2)))){{{0}, {1}}}, "
                 "(_Float16 __attribute__((ext_vector_type(2)))){{{2}, {3}}}, {4}, false)")


def _amd_warp_sum(value:UOp, lane:UOp) -> UOp:
  for offset in (16, 8, 4, 2, 1):
    other = UOp(Ops.CUSTOM, dtypes.float32, (((lane ^ offset)*4).int(), value),
                arg="__builtin_bit_cast(float, __builtin_amdgcn_ds_bpermute({0}, __builtin_bit_cast(int, {1})))")
    value = value + other
  return value


def _q8_0_block_dot(raw_words:UOp, x:UOp, token:UOp, output:UOp, block:UOp, blocks:int) -> UOp:
  in_features = blocks*Q8_0_BLOCK_SIZE
  flat_block = output*blocks + block
  word_base, odd_block = flat_block*Q8_0_BLOCK_BYTES//4, (flat_block & 1).eq(1)
  scale_word = raw_words[word_base]
  scale_bits = odd_block.where(scale_word >> 16, scale_word & 0xffff).cast(dtypes.uint16)
  scale = scale_bits.bitcast(dtypes.float16).float()
  dot = UOp.const(0, dtypes.float32)
  for word_idx in range(8):
    aligned = raw_words[word_base+1+word_idx]
    packed = odd_block.where(aligned, (raw_words[word_base+word_idx] >> 16) | (aligned << 16))
    for byte_idx in range(4):
      quant = ((packed >> (byte_idx*8)) & 255).cast(dtypes.uint8).bitcast(dtypes.int8).float()
      dot = dot + quant*x.flatten()[token*in_features+block*Q8_0_BLOCK_SIZE+word_idx*4+byte_idx].float()
  return dot*scale


def _q8_0_block_dot_f16(raw_words:UOp, x:UOp, token:UOp, output:UOp, block:UOp, blocks:int) -> UOp:
  in_features = blocks*Q8_0_BLOCK_SIZE
  flat_block = output*blocks + block
  word_base, odd_block = flat_block*Q8_0_BLOCK_BYTES//4, (flat_block & 1).eq(1)
  scale_word = raw_words[word_base]
  scale_bits = odd_block.where(scale_word >> 16, scale_word & 0xffff).cast(dtypes.uint16)
  scale = scale_bits.bitcast(dtypes.float16)
  dot = UOp.const(0, dtypes.float32)
  for word_idx in range(8):
    aligned = raw_words[word_base+1+word_idx]
    packed = odd_block.where(aligned, (raw_words[word_base+word_idx] >> 16) | (aligned << 16))
    for pair_idx in range(2):
      weights, inputs = [], []
      for elem_idx in range(2):
        byte_idx = pair_idx*2+elem_idx
        quant = ((packed >> (byte_idx*8)) & 255).cast(dtypes.uint8).bitcast(dtypes.int8).cast(dtypes.float16)
        weights.append(quant*scale)
        inputs.append(x.flatten()[token*in_features+block*Q8_0_BLOCK_SIZE+word_idx*4+byte_idx].load().cast(dtypes.float16))
      dot = _amd_fdot2((weights[0], weights[1]), (inputs[0], inputs[1]), dot)
  return dot


def _q8_0_block_dot_half(raw_words:UOp, x:UOp, token:UOp, output:UOp, block:UOp, blocks:int) -> UOp:
  in_features = blocks*Q8_0_BLOCK_SIZE
  flat_block = output*blocks + block
  word_base, odd_block = flat_block*Q8_0_BLOCK_BYTES//4, (flat_block & 1).eq(1)
  scale_word = raw_words[word_base]
  scale_bits = odd_block.where(scale_word >> 16, scale_word & 0xffff).cast(dtypes.uint16)
  scale = scale_bits.bitcast(dtypes.float16)
  dot = UOp.const(0, dtypes.float32)
  for word_idx in range(8):
    aligned = raw_words[word_base+1+word_idx]
    packed = odd_block.where(aligned, (raw_words[word_base+word_idx] >> 16) | (aligned << 16))
    for byte_idx in range(4):
      quant = ((packed >> (byte_idx*8)) & 255).cast(dtypes.uint8).bitcast(dtypes.int8).cast(dtypes.float16)
      weight = (quant*scale).float()
      value = x.flatten()[token*in_features+block*Q8_0_BLOCK_SIZE+word_idx*4+byte_idx].load().float()
      dot = dot + weight*value
  return dot


def _q8_0_rmsnorm_block_dot(raw_words:UOp, x:UOp, norm_weight:UOp, denominator:UOp,
                             output:UOp, block:UOp, blocks:int) -> UOp:
  flat_block = output*blocks + block
  word_base, odd_block = flat_block*Q8_0_BLOCK_BYTES//4, (flat_block & 1).eq(1)
  scale_word = raw_words[word_base]
  scale_bits = odd_block.where(scale_word >> 16, scale_word & 0xffff).cast(dtypes.uint16)
  scale = scale_bits.bitcast(dtypes.float16)
  dot = UOp.const(0, dtypes.float32)
  for word_idx in range(8):
    aligned = raw_words[word_base+1+word_idx]
    packed = odd_block.where(aligned, (raw_words[word_base+word_idx] >> 16) | (aligned << 16))
    for byte_idx in range(4):
      idx = block*Q8_0_BLOCK_SIZE+word_idx*4+byte_idx
      quant = ((packed >> (byte_idx*8)) & 255).cast(dtypes.uint8).bitcast(dtypes.int8).float()
      weight = (scale.float()*quant).cast(dtypes.float16).float()
      value = x.flatten()[idx].load().float() * denominator.flatten()[0].load().reciprocal() * \
              norm_weight.flatten()[idx].load().cast(dtypes.float16).float()
      dot = dot + value*weight
  return dot


@functools.cache
def _q8_0_gemv_exact(out:UOp, raw_words:UOp, x:UOp, tokens:int|UOp, out_features:int, in_features:int) -> UOp:
  blocks = in_features//Q8_0_BLOCK_SIZE
  assert in_features % Q8_0_BLOCK_SIZE == 0 and blocks % 32 == 0
  work = UOp.range(tokens*out_features, 0, AxisType.GLOBAL)
  token, output = work//out_features, work%out_features
  lane, subgroup_count = UOp.range(32, 1, AxisType.LOCAL), blocks//32
  subgroup = UOp.range(subgroup_count, 2, AxisType.LOCAL)
  block = lane*subgroup_count + subgroup

  flat_block = output*blocks + block
  word_base, odd_block = flat_block*Q8_0_BLOCK_BYTES//4, (flat_block & 1).eq(1)
  scale_word = raw_words[word_base]
  scale_bits = odd_block.where(scale_word >> 16, scale_word & 0xffff).cast(dtypes.uint16)
  scale = scale_bits.bitcast(dtypes.float16)
  partial = UOp.const(0, dtypes.float32)
  for word_idx in range(8):
    aligned = raw_words[word_base+1+word_idx]
    packed = odd_block.where(aligned, (raw_words[word_base+word_idx] >> 16) | (aligned << 16))
    for byte_idx in range(4):
      quant = ((packed >> (byte_idx*8)) & 255).cast(dtypes.uint8).bitcast(dtypes.int8).float()
      weight = (scale.float()*quant).cast(dtypes.float16)
      value = x.flatten()[token*in_features+block*Q8_0_BLOCK_SIZE+word_idx*4+byte_idx].load()
      partial = partial + (value*weight).cast(dtypes.float16).float()

  scratch = UOp.placeholder((blocks,), dtypes.float32, slot=0, addrspace=AddrSpace.LOCAL)
  ready = scratch[block].store(partial).barrier()
  reduce_lane = UOp.range(32, 3, AxisType.REDUCE)
  reduce_subgroup = UOp.range(subgroup_count, 4, AxisType.REDUCE)
  acc = UOp.placeholder((), dtypes.float32, slot=0, addrspace=AddrSpace.REG)
  acc_idx = acc.after(acc.store(acc.const_like(0)), reduce_lane, reduce_subgroup)
  update = acc_idx.store(acc_idx + scratch.after(ready)[reduce_lane*subgroup_count+reduce_subgroup]).end(reduce_subgroup).end(reduce_lane)
  result = acc.after(update).cast(out.dtype)
  leader = lane.eq(0) & subgroup.eq(0)
  return out.flatten()[(token*out_features+output).valid(leader)].store(result).end(subgroup, lane, work).sink(
    arg=KernelInfo(name="q8_0_gemv_exact", opts_to_apply=()))


@functools.cache
def _q8_0_gemv(out:UOp, raw_words:UOp, x:UOp, tokens:int|UOp, out_features:int, in_features:int,
                is_amd:bool, mode:str) -> UOp:
  output_tile = 4
  blocks = in_features//Q8_0_BLOCK_SIZE
  assert in_features % Q8_0_BLOCK_SIZE == 0 and blocks % 32 == 0
  assert out_features % output_tile == 0
  assert raw_words.dtype == dtypes.uint32 and raw_words.numel()*4 == out_features*blocks*Q8_0_BLOCK_BYTES
  if not is_amd:
    work = UOp.range(tokens*out_features, 0, AxisType.WEAK)
    token, output = work//out_features, work%out_features
    block = UOp.range(blocks, 1, AxisType.REDUCE)
    acc = UOp.placeholder((), dtypes.float32, slot=0, addrspace=AddrSpace.REG)
    acc = acc.after(acc.after(work).store(acc.const_like(0)))
    update = acc.store(acc.after(block) + _q8_0_block_dot(raw_words, x, token, output, block, blocks)).end(block)
    return out.flatten()[token*out_features+output].store(acc.after(update).cast(out.dtype)).end(work).sink(
      arg=KernelInfo(name="q8_0_gemv", opts_to_apply=()))

  output_groups = out_features//output_tile
  work = UOp.range(tokens*output_groups, 0, AxisType.GLOBAL)
  token, output_base = work//output_groups, (work%output_groups)*output_tile
  lane = UOp.range(32, 1, AxisType.LOCAL)
  chunk = UOp.range(blocks//32, 2, AxisType.REDUCE)
  block = chunk*32 + lane
  block_dot = {"float":_q8_0_block_dot, "half":_q8_0_block_dot_half, "fdot2":_q8_0_block_dot_f16}[mode]
  values = UOp.stack(*(block_dot(raw_words, x, token, output_base+i, block, blocks) for i in range(output_tile)))
  acc = UOp.placeholder((output_tile,), dtypes.float32, slot=0, addrspace=AddrSpace.REG)
  acc = acc.after(acc.store(acc.const_like(0)))
  update = acc.store(acc.after(chunk) + values).end(chunk)
  stores = [out.flatten()[(token*out_features+output_base+i).valid(lane.eq(0))].store(
              _amd_warp_sum(acc.after(update)[i].load(), lane).cast(out.dtype))
            for i in range(output_tile)]
  return UOp.group(*stores).end(lane, work).sink(arg=KernelInfo(name=f"q8_0_gemv_{mode}", opts_to_apply=()))


@functools.cache
def _q8_0_rmsnorm_gemv(out:UOp, raw_words:UOp, x:UOp, norm_weight:UOp, denominator:UOp,
                        out_features:int, in_features:int, is_amd:bool) -> UOp:
  output_tile, blocks = 2, in_features//Q8_0_BLOCK_SIZE
  assert in_features % Q8_0_BLOCK_SIZE == 0 and blocks % 32 == 0
  assert out_features % output_tile == 0
  assert raw_words.dtype == dtypes.uint32 and raw_words.numel()*4 == out_features*blocks*Q8_0_BLOCK_BYTES
  if not is_amd:
    output = UOp.range(out_features, 0, AxisType.WEAK)
    block = UOp.range(blocks, 1, AxisType.REDUCE)
    value = _q8_0_rmsnorm_block_dot(raw_words, x, norm_weight, denominator, output, block, blocks).reduce(block, arg=(Ops.ADD, 0))
    return out.flatten()[output].store(value.cast(out.dtype)).end(output).sink(
      arg=KernelInfo(name="q8_0_rmsnorm_gemv", opts_to_apply=()))
  work = UOp.range(out_features//output_tile, 0, AxisType.GLOBAL)
  output_base = work*output_tile
  lane = UOp.range(32, 1, AxisType.LOCAL)
  chunk = UOp.range(blocks//32, 2, AxisType.REDUCE)
  block = chunk*32 + lane
  values = UOp.stack(*(_q8_0_rmsnorm_block_dot(raw_words, x, norm_weight, denominator, output_base+i, block, blocks)
                       for i in range(output_tile)))
  acc = UOp.placeholder((output_tile,), dtypes.float32, slot=0, addrspace=AddrSpace.REG)
  acc = acc.after(acc.store(acc.const_like(0)))
  update = acc.store(acc.after(chunk) + values).end(chunk)
  stores = [out.flatten()[(output_base+i).valid(lane.eq(0))].store(
              _amd_warp_sum(acc.after(update)[i].load(), lane).cast(out.dtype)) for i in range(output_tile)]
  return UOp.group(*stores).end(lane, work).sink(arg=KernelInfo(name="q8_0_rmsnorm_gemv", opts_to_apply=()))


def q8_0_linear(raw_words:Tensor, x:Tensor, shape:tuple[int, ...], mode:str="exact") -> Tensor:
  assert len(shape) == 2 and shape[1] == x.shape[-1]
  out_features, in_features = shape
  tokens = _index_sint(prod(x.shape[:-1]))
  max_output_shape = (*x.uop.max_shard_shape[:-1], out_features)
  output = Tensor.invalids(*max_output_shape, dtype=x.dtype, device=x.device)
  device = x.device[0] if isinstance(x.device, tuple) else x.device
  is_amd = isinstance(device, str) and device.split(":")[0] == "AMD"
  srcs = (output.uop, raw_words.uop, x.uop)
  params = (UOp.placeholder_like(srcs[0], slot=0), UOp.placeholder_like(srcs[1], slot=1),
            UOp.placeholder(x.uop.max_shard_shape, x.dtype, slot=2))
  kernel = _q8_0_gemv_exact(*params, tokens=tokens, out_features=out_features, in_features=in_features) if is_amd and mode == "exact" else \
    _q8_0_gemv(*params, tokens=tokens, out_features=out_features, in_features=in_features,
                is_amd=is_amd, mode=mode if is_amd else "float")
  call = kernel.call(*srcs)
  return Tensor(srcs[0].after(call)).shrink_to((*x.shape[:-1], out_features))


def q8_0_rmsnorm_linear(raw_words:Tensor, x:Tensor, norm_weight:Tensor, denominator:Tensor, shape:tuple[int, ...]) -> Tensor:
  assert len(shape) == 2 and shape[1] == x.shape[-1] == norm_weight.shape[0]
  assert prod(x.shape[:-1]) == 1 and prod(x.uop.max_shape[:-1]) == 1
  out_features, in_features = shape
  output = Tensor.invalids(*x.shape[:-1], out_features, dtype=x.dtype, device=x.device)
  output, raw_words, x, norm_weight, denominator = (output.uop, raw_words.uop, x.uop, norm_weight.uop, denominator.uop)
  params = (UOp.placeholder_like(output, slot=0), UOp.placeholder_like(raw_words, slot=1),
            UOp.placeholder_like(x, slot=2), UOp.placeholder_like(norm_weight, slot=3), UOp.placeholder_like(denominator, slot=4))
  device = x.device[0] if isinstance(x.device, tuple) else x.device
  is_amd = isinstance(device, str) and device.split(":")[0] == "AMD"
  kernel = _q8_0_rmsnorm_gemv(*params, out_features=out_features, in_features=in_features, is_amd=is_amd)
  call = kernel.call(output, raw_words, x, norm_weight, denominator)
  return Tensor(output.after(call))


@functools.cache
def _q8_quantize(q:UOp, scale:UOp, x:UOp, tokens:int|UOp, in_features:int) -> UOp:
  physical_tokens, blocks, _ = q.shape
  assert isinstance(physical_tokens, int) and isinstance(blocks, int) and in_features == blocks*Q4_0_BLOCK_SIZE
  x = x.reshape(physical_tokens, blocks, Q4_0_BLOCK_SIZE)
  group = UOp.range(tokens*blocks, 0, AxisType.GLOBAL)
  token, block = group//blocks, group%blocks
  lane = UOp.range(Q4_0_BLOCK_SIZE, 1, AxisType.LOCAL)
  value = x[token, block, lane].float()

  scratch = UOp.placeholder((Q4_0_BLOCK_SIZE,), dtypes.float32, slot=0, addrspace=AddrSpace.LOCAL)
  update = scratch[lane].store(value.abs()).barrier()
  for stride in (16, 8, 4, 2, 1):
    current = scratch.after(update)
    update = current[lane.valid(lane < stride)].store(current[lane].maximum(current[lane+stride])).barrier()
  group_scale = (scratch.after(update)[0] / 127).maximum(1e-8)

  word_lane = lane.minimum(7)
  word = UOp.const(0, dtypes.uint32)
  for i in range(4):
    quant = (x[token, block, word_lane*4+i].float() / group_scale).round().clip(-127, 127).cast(dtypes.int8)
    word = word | quant.cast(dtypes.uint8).cast(dtypes.uint32).lshift(i*8)
  stores = (q[token, block, lane.valid(lane < 8)].store(word), scale[token, block.valid(lane.eq(0))].store(group_scale))
  return UOp.group(*stores).end(lane, group).sink(arg=KernelInfo(name="q8_quantize", opts_to_apply=()))


def q8_quantize(x:Tensor, tokens:int|UOp, max_tokens:int, in_features:int) -> tuple[Tensor, Tensor]:
  blocks = in_features//Q4_0_BLOCK_SIZE
  q = Tensor.empty(max_tokens, blocks, 8, dtype=dtypes.uint32, device=x.device)
  scale = Tensor.empty(max_tokens, blocks, dtype=dtypes.float32, device=x.device)
  srcs = (q.uop, scale.uop, x.uop)
  params = (UOp.placeholder_like(srcs[0], slot=0), UOp.placeholder_like(srcs[1], slot=1),
            UOp.placeholder(x.uop.max_shard_shape, x.dtype, slot=2))
  call = _q8_quantize(*params, tokens=tokens, in_features=in_features).call(*srcs)
  return Tensor(srcs[0].after(call)), Tensor(srcs[1].after(call))


@functools.cache
def _q4_0_expert_partial(out:UOp, raw:UOp, sel:UOp, x:UOp, tokens:int|UOp, blocks_per_partial:int) -> UOp:
  physical_tokens, topk, out_features, chunks = out.shape
  experts, raw_out_features, blocks, block_bytes = raw.shape
  assert out_features == raw_out_features and block_bytes == Q4_0_BLOCK_BYTES
  assert all(isinstance(v, int) for v in (physical_tokens, topk, out_features, chunks, experts, blocks))
  assert blocks == chunks * blocks_per_partial

  in_features = blocks * Q4_0_BLOCK_SIZE
  input_slots = prod(x.shape)//(physical_tokens*in_features)
  assert input_slots in (1, topk)
  x, sel = x.reshape(physical_tokens, input_slots, in_features), sel.reshape(physical_tokens, topk)
  is_amd = isinstance(out.device, str) and out.device.split(":")[0] == "AMD"
  work_axis = AxisType.GLOBAL if is_amd else AxisType.WEAK
  work = UOp.range(tokens*topk*out_features*chunks, 0, work_axis)
  chunk, output = work % chunks, (work // chunks) % out_features
  slot, token = (work // (chunks*out_features)) % topk, work // (chunks*out_features*topk)
  expert = sel[token, slot]

  lane = UOp.range(blocks_per_partial, 1, AxisType.LOCAL if is_amd else AxisType.WEAK)
  block = chunk*blocks_per_partial + lane
  scale_bits = raw[expert, output, block, 0].cast(dtypes.uint16) | raw[expert, output, block, 1].cast(dtypes.uint16).lshift(8)
  scale = scale_bits.bitcast(dtypes.float16).float()
  dot = UOp.const(0, dtypes.float32)
  for i in range(Q4_0_BLOCK_SIZE):
    packed = raw[expert, output, block, 2 + i%16]
    quant = ((packed >> (4*(i//16))) & 15).cast(dtypes.float32) - 8
    dot = dot + quant * x[token, slot if input_slots > 1 else 0, block*Q4_0_BLOCK_SIZE+i].float()
  if not is_amd:
    acc = UOp.placeholder((), dtypes.float32, slot=0, addrspace=AddrSpace.REG)
    acc = acc.after(acc.after(work).store(acc.const_like(0)))
    update = acc.store(acc.after(lane) + dot*scale).end(lane)
    return out.flatten()[work].store(acc.after(update).cast(out.dtype)).end(work).sink(
      arg=KernelInfo(name="q4_0_expert_partial", opts_to_apply=()))
  scratch = UOp.placeholder((blocks_per_partial,), dtypes.float32, slot=0, addrspace=AddrSpace.LOCAL)
  update = scratch[lane].store(dot*scale).barrier()
  for stride in (blocks_per_partial >> i for i in range(1, blocks_per_partial.bit_length())):
    current = scratch.after(update)
    update = current[lane.valid(lane < stride)].store(current[lane] + current[lane+stride]).barrier()
  result = scratch.after(update)[0]
  return out.flatten()[work.valid(lane.eq(0))].store(result.cast(out.dtype)).end(lane, work).sink(
    arg=KernelInfo(name="q4_0_expert_partial", opts_to_apply=()))


@functools.cache
def _q4_0_expert_q8(out:UOp, raw_words:UOp, sel:UOp, xq:UOp, xd:UOp, tokens:int|UOp,
                    experts:int, blocks:int, output_tile:int, input_slots:int) -> UOp:
  physical_tokens, topk, out_features = out.shape
  assert all(isinstance(v, int) for v in (physical_tokens, topk, out_features, experts, blocks))
  assert out_features % output_tile == 0
  assert raw_words.dtype == dtypes.uint32 and raw_words.numel()*4 == experts*out_features*blocks*Q4_0_BLOCK_BYTES

  assert input_slots in (1, topk)
  xq, xd = xq.reshape(physical_tokens*input_slots, blocks, 8), xd.reshape(physical_tokens*input_slots, blocks)
  sel = sel.reshape(physical_tokens, topk)
  output_groups = out_features//output_tile
  work = UOp.range(tokens*topk*output_groups, 0, AxisType.GLOBAL)
  output_base, slot, token = (work%output_groups)*output_tile, (work//output_groups)%topk, work//(output_groups*topk)
  expert = sel[token, slot]
  xrow = token*input_slots + (slot if input_slots > 1 else 0)
  lane = UOp.range(32, 1, AxisType.LOCAL)
  chunk = UOp.range((blocks+31)//32, 2, AxisType.REDUCE)
  block = (lane + chunk*32).valid(lane + chunk*32 < blocks)
  xwords = tuple(xq[xrow, block, word] for word in range(8))
  xscale = xd[xrow, block]

  values = []
  for output_offset in range(output_tile):
    output = output_base + output_offset
    flat_block = (expert*out_features + output)*blocks + block
    word_base, odd_block = flat_block*Q4_0_BLOCK_BYTES//4, (flat_block & 1).eq(1)
    raw = tuple(raw_words[word_base+i] for i in range(5))
    scale_word = raw[0]
    scale_bits = odd_block.where(scale_word >> 16, scale_word & 0xffff).cast(dtypes.uint16)
    scale = scale_bits.bitcast(dtypes.float16).float() * xscale
    dot = UOp.const(0, dtypes.int32)
    for word in range(4):
      aligned = raw[1+word]
      packed = odd_block.where(aligned, (raw[word] >> 16) | (raw[word+1] << 16))
      low = ((packed & 0x0f0f0f0f) + 0x78787878) ^ 0x80808080
      high = (((packed >> 4) & 0x0f0f0f0f) + 0x78787878) ^ 0x80808080
      dot = _amd_sdot4(low, xwords[word], dot)
      dot = _amd_sdot4(high, xwords[word+4], dot)
    values.append(dot.float()*scale)

  acc = UOp.placeholder((output_tile,), dtypes.float32, slot=0, addrspace=AddrSpace.REG)
  acc = acc.after(acc.store(acc.const_like(0)))
  update = acc.store(acc.after(chunk) + UOp.stack(*values)).end(chunk)
  acc = acc.after(update)
  stores = [out[token, slot, (output_base+i).valid(lane.eq(0))].store(_amd_warp_sum(acc[i].load(), lane).cast(out.dtype))
            for i in range(output_tile)]
  return UOp.group(*stores).end(lane, work).sink(arg=KernelInfo(name="q4_0_expert_q8", opts_to_apply=()))


def q4_0_expert_linear(raw:Tensor, sel:Tensor, x:Tensor, packed_shape:tuple[int, int, int]|None=None,
                       physical_output:bool=False) -> Tensor:
  if raw.dtype == dtypes.uint8:
    assert len(raw.shape) == 4 and raw.shape[-1] == Q4_0_BLOCK_BYTES and all(isinstance(v, int) for v in raw.shape)
    experts, out_features, blocks, _ = (int(v) for v in raw.shape)
    in_features, raw_words = blocks*Q4_0_BLOCK_SIZE, raw.flatten().bitcast(dtypes.uint32)
  else:
    assert raw.dtype == dtypes.uint32 and packed_shape is not None
    experts, out_features, in_features = packed_shape
    assert in_features % Q4_0_BLOCK_SIZE == 0
    blocks, raw_words = in_features//Q4_0_BLOCK_SIZE, raw.flatten()
    assert prod(raw.uop.max_shard_shape)*4 == experts*out_features*blocks*Q4_0_BLOCK_BYTES
  tokens, max_sel_shape = _index_sint(prod(sel.shape[:-1])), sel.uop.max_shape
  max_tokens, topk = prod(max_sel_shape[:-1]), int(max_sel_shape[-1])
  assert prod(x.uop.max_shard_shape) % (max_tokens*in_features) == 0
  input_slots = prod(x.uop.max_shard_shape)//(max_tokens*in_features)
  assert input_slots in (1, topk), f"expected 1 or {topk} input slots, got {input_slots}"
  raw_device = raw.device[0] if isinstance(raw.device, tuple) else raw.device
  if isinstance(raw_device, str) and raw_device.split(":")[0] == "AMD":
    rows = _index_sint(prod(x.shape[:-1]))
    quant, scale = q8_quantize(x, rows, max_tokens*input_slots, in_features)
    output = Tensor.empty(max_tokens, topk, out_features, dtype=x.dtype, device=raw.device)
    output = Tensor.custom_kernel(output, raw_words, sel.pad_to(max_sel_shape), quant, scale, fxn=functools.partial(
      _q4_0_expert_q8, tokens=tokens, experts=experts, blocks=blocks, output_tile=1, input_slots=input_slots))[0]
  else:
    blocks_per_partial = next(v for v in (32, 16, 8, 4, 2, 1) if blocks % v == 0)
    chunks = blocks // blocks_per_partial
    partial = Tensor.empty(max_tokens, topk, out_features, chunks, dtype=x.dtype, device=raw.device)
    partial = Tensor.custom_kernel(partial, raw, sel.pad_to(max_sel_shape), x.pad_to(x.uop.max_shape), fxn=functools.partial(
      _q4_0_expert_partial, tokens=tokens, blocks_per_partial=blocks_per_partial))[0]
    output = partial.sum(-1)
  output = output.reshape(*max_sel_shape, out_features)
  return output if physical_output else output.shrink_to((*sel.shape, out_features))
