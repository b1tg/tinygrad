import functools, io, pathlib, re, struct
from typing import Any, Callable, cast

from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
from tinygrad.helpers import prod, round_up
from tinygrad.nn.state import TensorIO
from tinygrad.uop.ops import Ops

# ggml packs each iq grid entry as N bytes (N=4 for uint32 grids, N=8 for uint64 grids) in a single word. See ggml-common.h.
@functools.lru_cache(None)
def _ggml_iq_grid(device: str, grid: tuple[int, ...], grid_shape: tuple[int, int]) -> Tensor:
  values = [float((w >> (8*i)) & 0xFF) for w in grid for i in range(grid_shape[1])]
  return Tensor(values, dtype=dtypes.float32, device=device).reshape(grid_shape)

# native types {ggml_type: dtype}
_GGML_NATIVE = {0: dtypes.float32, 1: dtypes.float16, 24: dtypes.int8, 25: dtypes.int16,
                26: dtypes.int32, 27: dtypes.int64, 28: dtypes.float64, 30: dtypes.bfloat16}

# quant types {ggml_type: (number of elements, number of bytes)}
_GGML_QUANT = {2:(32,18), 3:(32,20), 6:(32,22), 7:(32,24), 8:(32,34),
               12:(256,144), 13:(256,176), 14:(256,210), 17:(256,74), 18:(256,98), 21:(256,110), 22:(256,82), 23:(256,136), 39:(32,17), 41:(128,18)}
_TensorInfo = tuple[str, tuple[int, ...], int, int]

class GGUFQuantizedTensor:
  """Packed GGUF weights retained for custom kernels or selected-expert decoding."""
  def __init__(self, data:tuple[Tensor, ...], shape:tuple[int, ...], ggml_type:int, axis:int|None, offsets:tuple[int, ...]):
    self.data, self.shape, self.ggml_type, self.axis, self.offsets = data, shape, ggml_type, axis, offsets

  @property
  def device(self) -> str|tuple[str, ...]:
    return cast(str, self.data[0].device) if len(self.data) == 1 else cast(tuple[str, ...], tuple(x.device for x in self.data))

  def numel(self) -> int: return prod(self.shape)

  def decode(self, sel:Tensor) -> Tensor:
    nelements, _ = _GGML_QUANT[self.ggml_type]
    shard_size = self.shape[-1] // len(self.data) if self.axis == len(self.shape)-1 else self.shape[-1]
    out: list[Tensor] = []
    for raw, offset in zip(self.data, self.offsets):
      if raw.dtype == dtypes.uint32:
        assert self.ggml_type in (2, 8)
        local_shape = self.shape if self.axis is None else tuple(s//len(self.data) if i == self.axis else s for i,s in enumerate(self.shape))
        raw = raw.bitcast(dtypes.uint8).reshape(*local_shape[:-1], local_shape[-1]//nelements, _GGML_QUANT[self.ggml_type][1])
      selected = raw[sel.to(raw.device)].contiguous()
      local_elems = int(selected.shape[-2]) * nelements
      decoded = ggml_data_to_tensor(selected.reshape(-1), cast(int, prod(selected.shape[:-2])) * local_elems,
                                    self.ggml_type, block_contiguous=False).reshape(*selected.shape[:-2], local_elems)
      out.append(decoded[..., offset:offset+shard_size])
    if len(out) == 1: return out[0]
    assert self.axis is not None
    return Tensor(out[0].uop.mstack(*[x.uop for x in out[1:]]).unshard(self.axis + len(sel.shape) - 1))

  def q4_0_expert_linear(self, sel:Tensor, x:Tensor) -> Tensor:
    assert self.ggml_type == 2 and (len(self.data) == 1 or self.axis in (1, 2))
    from tinygrad.llm.kernels.amd import q4_0_expert_linear
    local_out_features = self.shape[1]//len(self.data) if self.axis == 1 else self.shape[1]
    local_in_features = self.shape[-1]//len(self.data) if self.axis == 2 else self.shape[-1]
    packed_shape = (self.shape[0], local_out_features, local_in_features)
    if len(self.data) > 1:
      words = tuple(raw.uop.flatten().bitcast(dtypes.uint32) for raw in self.data)
      raw = Tensor(words[0].mstack(*words[1:]).unshard(0))
      output = q4_0_expert_linear(raw, sel, x, packed_shape, physical_output=True)
      if self.axis == 2: return Tensor(output.uop.allreduce(Ops.ADD, self.device)).shrink_to((*sel.shape, local_out_features))
      return Tensor(output.uop.unshard(len(output.shape)-1)).shrink_to((*sel.shape, self.shape[1]))
    outputs = []
    for i,raw in enumerate(self.data):
      local_sel = Tensor(sel.uop.mselect(i)) if isinstance(sel.device, tuple) else sel.to(raw.device)
      local_x = Tensor(x.uop.mselect(i)) if isinstance(x.device, tuple) else x.to(raw.device)
      outputs.append(q4_0_expert_linear(raw, local_sel, local_x, packed_shape, physical_output=self.axis == 2))
    if len(outputs) == 1: return outputs[0]
    assert self.axis is not None
    if self.axis == 2:
      stacked = outputs[0].uop.mstack(*[out.uop for out in outputs[1:]])
      from tinygrad.schedule.allreduce import handle_allreduce
      reduced = handle_allreduce(stacked, stacked.allreduce(Ops.ADD, self.device))
      assert reduced is not None
      return Tensor(reduced).shrink_to((*sel.shape, local_out_features))
    stacked = outputs[0].uop.mstack(*[out.uop for out in outputs[1:]])
    return Tensor(stacked.unshard(self.axis + len(sel.shape) - 1))

  def q8_0_linear(self, x:Tensor) -> Tensor:
    assert self.ggml_type == 8 and self.axis is None and len(self.shape) == 2
    from tinygrad.llm.kernels.amd import q8_0_linear
    raw_words = self.data[0] if len(self.data) == 1 else Tensor(self.data[0].uop.mstack(*[raw.uop for raw in self.data[1:]]))
    return q8_0_linear(raw_words, x, self.shape)

  def q8_0_rmsnorm_linear(self, x:Tensor, norm_weight:Tensor, denominator:Tensor) -> Tensor:
    assert self.ggml_type == 8 and self.axis is None and len(self.shape) == 2
    from tinygrad.llm.kernels.amd import q8_0_rmsnorm_linear
    raw_words = self.data[0] if len(self.data) == 1 else Tensor(self.data[0].uop.mstack(*[raw.uop for raw in self.data[1:]]))
    return q8_0_rmsnorm_linear(raw_words, x, norm_weight, denominator, self.shape)

def ggml_data_to_tensor(t: Tensor, n: int, ggml_type: int, block_contiguous: bool=True) -> Tensor:
  """
  Converts ggml tensor data to a tinygrad tensor.

  Supported native types: float32 (id: 0), float16 (id: 1), int8 (id: 24),
  int16 (id: 25), int32 (id: 26), int64 (id: 27), float64 (id: 28), bfloat16 (id: 30)
  Supported quantized types: Q4_0 (id: 2), Q4_1 (id: 3), Q5_0 (id: 6),
  Q5_1 (id: 7), Q8_0 (id: 8), Q4_K (id: 12), Q5_K (id: 13),
  Q6_K (id: 14), IQ2_XS (id: 17), IQ3_XXS (id: 18), IQ3_S (id: 21), IQ2_S (id: 22), IQ4_XS (id: 23), MXFP4 (id: 39), Q1_0 (id: 41)
  """
  # https://github.com/ggerganov/ggml/blob/323951f1bdcdfbd5b5ff3a9a7c3770e63b1a560e/include/ggml.h#L356

  if (dtype := _GGML_NATIVE.get(ggml_type)) is not None:
    raw = t[:dtype.itemsize * n]
    return (raw.contiguous() if block_contiguous else raw).bitcast(dtype)

  def q_to_uint8(t: Tensor, b: int) -> Tensor:
    # TODO: rewrite with arange?
    shift_tensor, bitmask = Tensor.const(tuple(2**(i*b) for i in range(8//b)), t.dtype), 0xff >> (8 - b)
    return t.unsqueeze(-1).div(shift_tensor, rounding_mode="trunc").bitwise_and(bitmask).transpose(-1, -2).flatten(-2)

  if (nelements_nbytes := _GGML_QUANT.get(ggml_type)) is not None:
    from tinygrad.runtime.autogen import ggml_common as _ggml
    blocks = t[:(n//nelements_nbytes[0])*nelements_nbytes[1]].reshape((-1, nelements_nbytes[1]))
    if block_contiguous: blocks = blocks.contiguous()
    if ggml_type == 2: return (q_to_uint8(blocks[:,2:], 4).bitcast(dtypes.int8) - 8) * blocks[:,:2].bitcast(dtypes.float16).cast(dtypes.float32)
    if ggml_type == 3:
      d, m = (blocks[:,s:s+2].bitcast(dtypes.float16).cast(dtypes.float32) for s in [ 0, 2 ])
      return q_to_uint8(blocks[:,4:], 4).bitcast(dtypes.int8) * d + m
    if ggml_type in (6, 7):
      d = blocks[:,:2].bitcast(dtypes.float16).cast(dtypes.float32)
      qh_off = 2 if ggml_type == 6 else 4
      qh = q_to_uint8(blocks[:,qh_off:qh_off+4], 1).reshape((-1, 8, 4)).transpose(-1, -2).flatten(-2).bitcast(dtypes.int8)
      q = q_to_uint8(blocks[:,qh_off+4:], 4).bitcast(dtypes.int8) + qh * 16
      return q * d + (blocks[:,2:4].bitcast(dtypes.float16).cast(dtypes.float32) if ggml_type == 7 else -16 * d)
    if ggml_type == 8: return blocks[:,:2].bitcast(dtypes.float16).cast(dtypes.float32) * blocks[:,2:].bitcast(dtypes.int8)
     # Q4_K: 256 elements per 144-byte block (d:2, dmin:2, scales:12, qs:128)
     # Q5_K: 256 elements per 176-byte block (d:2, dmin:2, scales:12, qh:32, qs:128)
    if ggml_type in (12, 13):
      d, dmin = (blocks[:,i:i+2].bitcast(dtypes.float16).cast(dtypes.float32).unsqueeze(-1) for i in [0, 2])
      s = blocks[:,4:16]  # 12 bytes: 6-bit scales[0-3], 6-bit mins[0-3], high bits[4-7]
      sc = s[:,0:4].bitwise_and(63).cat(s[:,8:12].bitwise_and(0xF).bitwise_or(s[:,0:4].rshift(6).lshift(4)), dim=-1)
      mn = s[:,4:8].bitwise_and(63).cat(s[:,8:12].rshift(4).bitwise_or(s[:,4:8].rshift(6).lshift(4)), dim=-1)
      qs_off = 48 if ggml_type == 13 else 16
      q = Tensor.stack((qs:=blocks[:,qs_off:qs_off+128].reshape(-1,4,32)).bitwise_and(0xF), qs.rshift(4), dim=2).reshape(-1,8,32)
      if ggml_type == 13: q = q + q_to_uint8(blocks[:,16:48], 1).reshape(-1, 8, 32) * 16
      return (d * sc.unsqueeze(-1) * q - dmin * mn.unsqueeze(-1)).flatten(-2)
    if ggml_type == 14:
      xl, xh = q_to_uint8(blocks[:,:128].reshape((-1, 2, 64)), 4), q_to_uint8(blocks[:,128:192].reshape((-1, 2, 32)), 2).lshift(4)
      scales = blocks[:,192:208].bitcast(dtypes.int8).unsqueeze(-1).expand((-1, 16, 16)).reshape((-1, 256))
      d = blocks[:,-2:].bitcast(dtypes.float16).cast(dtypes.float32)
      return d * (xl.bitwise_or(xh).bitcast(dtypes.int8) - 32).flatten(-2) * scales
    if ggml_type == 17:
      d = blocks[:, :2].bitcast(dtypes.float16).cast(dtypes.float32).reshape((-1, 1, 1, 1))
      qs = blocks[:, 2:66].bitcast(dtypes.uint16).reshape((-1, 8, 4))
      scales = Tensor.stack((s:=blocks[:, 66:74]).bitwise_and(0xF), s.rshift(4), dim=2).cast(dtypes.float32)
      scales = (d * (scales + 0.5).unsqueeze(-1) * 0.25).expand((-1, 8, 2, 2)).reshape((-1, 8, 4, 1))
      grid = _ggml_iq_grid(t.device, _ggml.iq2xs_grid, (512, 8))[qs.bitwise_and(511).cast(dtypes.int32)]
      sign_masks = Tensor(list(_ggml.ksigns_iq2xs), dtype=dtypes.uint8, device=t.device)[qs.rshift(9).cast(dtypes.int32)]
      signs = (q_to_uint8(sign_masks.unsqueeze(-1), 1) == 0).where(1.0, -1.0)
      return (scales * grid * signs).flatten(-3)
    if ggml_type == 18:
      d = blocks[:, :2].bitcast(dtypes.float16).cast(dtypes.float32).reshape((-1, 1, 1, 1))
      scale_words = blocks[:, 66:98].bitcast(dtypes.uint32)
      db = d * (scale_words.rshift(28).cast(dtypes.float32) + 0.5).reshape((-1, 8, 1, 1)) * 0.5
      sign_idx = scale_words.unsqueeze(-1).rshift(Tensor.const((0, 7, 14, 21), dtypes.uint32)).bitwise_and(0x7F).reshape((-1, 32)).cast(dtypes.int32)
      even_signs = Tensor([i | (0x80 if i.bit_count() % 2 else 0) for i in range(128)], dtype=dtypes.uint8, device=t.device)
      signs = (q_to_uint8(even_signs[sign_idx].reshape((-1, 32, 1)), 1) == 0).where(1.0, -1.0).reshape((-1, 8, 4, 8))
      grid = _ggml_iq_grid(t.device, _ggml.iq3xxs_grid, (256, 4))[blocks[:, 2:66]].reshape((-1, 8, 4, 8))
      return (db * grid * signs).flatten(-3)
    if ggml_type == 21:
      d = blocks[:, :2].bitcast(dtypes.float16).cast(dtypes.float32).reshape((-1, 1, 1, 1))
      scales = (1 + 2 * q_to_uint8(blocks[:, 106:110].reshape((-1, 4, 1)), 4).reshape((-1, 8))).cast(dtypes.float32).reshape((-1, 8, 1, 1))
      qh = q_to_uint8(blocks[:, 66:74].reshape((-1, 8, 1)), 1).reshape((-1, 64)).cast(dtypes.uint16)
      signs = (q_to_uint8(blocks[:, 74:106].reshape((-1, 32, 1)), 1).reshape((-1, 256)) == 0).where(1.0, -1.0).reshape((-1, 8, 4, 8))
      q = blocks[:, 2:66].cast(dtypes.uint16) + qh.lshift(8)
      return (d * scales * _ggml_iq_grid(t.device, _ggml.iq3s_grid, (512, 4))[q].reshape((-1, 8, 4, 8)) * signs).flatten(-3)
    if ggml_type == 22:
      d = blocks[:, :2].bitcast(dtypes.float16).cast(dtypes.float32).reshape((-1, 1, 1, 1))
      db = d * (q_to_uint8(blocks[:, 74:82].reshape((-1, 8, 1)), 4).reshape((-1, 16)).cast(dtypes.float32) + 0.5).reshape((-1, 16, 1, 1)) * 0.25
      signs = (q_to_uint8(blocks[:, 34:66].reshape((-1, 32, 1)), 1) == 0).where(1.0, -1.0).reshape((-1, 16, 2, 8))
      qh = q_to_uint8(blocks[:, 66:74].reshape((-1, 8, 1)), 2).reshape((-1, 32)).cast(dtypes.uint16)
      q = blocks[:, 2:34].cast(dtypes.uint16) + qh.lshift(8)
      return (db * _ggml_iq_grid(t.device, _ggml.iq2s_grid, (1024, 8))[q].reshape((-1, 16, 2, 8)) * signs).flatten(-3)
    if ggml_type == 23:
      d = blocks[:, :2].bitcast(dtypes.float16).cast(dtypes.float32).reshape((-1, 1, 1))
      scale_shifts = Tensor.const((0, 2, 4, 6, 8, 10, 12, 14), dtypes.uint16)
      iq4_xs_lut = Tensor(list(_ggml.kvalues_iq4nl), dtype=dtypes.float32, device=t.device)
      scales_l = Tensor.stack((sl:=blocks[:, 4:8]).bitwise_and(0xF), sl.rshift(4), dim=2).reshape((-1, 8))
      scales_h = blocks[:, 2:4].bitcast(dtypes.uint16).unsqueeze(-1).rshift(scale_shifts).bitwise_and(0x03).reshape((-1, 8)).cast(dtypes.uint8)
      scales = (scales_l.bitwise_or(scales_h.lshift(4)).bitcast(dtypes.int8) - 32).cast(dtypes.float32).reshape((-1, 8, 1))
      q = (qs:=blocks[:, 8:].reshape((-1, 8, 16))).bitwise_and(0xF).cat(qs.rshift(4), dim=2)
      return (d * scales * iq4_xs_lut[q]).flatten(-2)
    if ggml_type == 39:
      e = blocks[:, 0].cast(dtypes.uint32)
      small_bits = Tensor([0x00200000, 0x00400000], dtype=dtypes.uint32, device=t.device)[e.clip(0, 1).cast(dtypes.int32)] # e = 0 or e = 1 case
      d = (e < 2).where(small_bits, (e - 1) * 0x00800000).bitcast(dtypes.float32).unsqueeze(-1)
      codes = q_to_uint8(blocks[:, 1:17], 4)
      fp4_lut = Tensor([0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0,
                       -0.0,-1.0,-2.0,-3.0,-4.0,-6.0,-8.0,-12.0],
                      dtype=dtypes.float32, device=t.device)
      fp4_val = fp4_lut[codes]
      return (fp4_val * d).flatten(-2)[:n]
    if ggml_type == 41:
      d = blocks[:,:2].bitcast(dtypes.float16)
      bits = q_to_uint8(blocks[:,2:], 1).reshape(-1, 8, 16).transpose(-1, -2).flatten(-2).bitcast(dtypes.int8)
      return d * (bits * 2 - 1)
  raise ValueError(f"GGML type '{ggml_type}' is not supported!")

def _read_unpack(fmt: str, n: int, r:io.BufferedIOBase): return struct.unpack(fmt, r.read(n))[0]
def read_str(r:io.BufferedIOBase): return str(r.read(read_uint64(r)), "utf-8")
def read_arr(r:io.BufferedIOBase):
  item_reader, n = readers[read_int32(r)], read_uint64(r)
  return [item_reader(r) for _ in range(n)]

readers: dict[int, Callable[[io.BufferedIOBase], Any]] = { 8: read_str, 9: read_arr,
  **{ t: functools.partial(_read_unpack, "<"+f, nb) for t,f,nb in \
    [ (0,"c",1), (1,"b",1), (2,"H",2), (3,"h",2), (4,"I",4), (5,"i",4), (6,"f",4), (7,"?",1), (10,"Q",8), (11,"q",8), (12,"d",8) ] } }
read_uint32, read_int32, read_uint64, read_int64 = readers[4], readers[5], readers[10], readers[11]

def _ggml_nbytes(n:int, typ:int) -> int:
  return n * _GGML_NATIVE[typ].itemsize if typ in _GGML_NATIVE else (n // _GGML_QUANT[typ][0]) * _GGML_QUANT[typ][1]

def _quantized_tensor(tensor:Tensor, data_start:int, t_info:_TensorInfo,
                      devices:tuple[str, ...]|None=None, axis:int|None=None) -> GGUFQuantizedTensor:
  name, dims, typ, off = t_info
  shape, start = tuple(reversed(dims)), data_start + off
  nelements, nbytes = _GGML_QUANT[typ]
  assert shape[-1] % nelements == 0, f"{name}: last dim {shape[-1]} does not divide block size {nelements}"
  nblocks = shape[-1] // nelements
  if devices is None:
    raw = tensor[start:start + prod(shape[:-1])*nblocks*nbytes].reshape(*shape[:-1], nblocks, nbytes).contiguous().realize()
    if typ == 8: raw = raw.flatten().bitcast(dtypes.uint32)
    return GGUFQuantizedTensor((raw,), shape, typ, None, (0,))

  if axis is None:
    assert typ == 8, f"{name}: only Q8_0 custom weights may be replicated"
    raw = tensor[start:start + prod(shape[:-1])*nblocks*nbytes].to("CPU").realize().contiguous().flatten().bitcast(dtypes.uint32)
    raws = tuple(raw.to(device) for device in devices)
    Tensor.realize(*raws)
    return GGUFQuantizedTensor(raws, shape, typ, None, (0,)*len(devices))
  assert shape[axis] % len(devices) == 0, f"{name}: axis {axis} size {shape[axis]} does not divide {len(devices)} devices"
  blocks = tensor[start:start + prod(shape[:-1])*nblocks*nbytes].to("CPU").realize().reshape(*shape[:-1], nblocks, nbytes)
  if axis == len(shape)-1:
    per_dev, raws, offsets = shape[-1] // len(devices), [], []
    for i, device in enumerate(devices):
      elem_start, elem_end = i*per_dev, (i+1)*per_dev
      block_start, block_end = elem_start//nelements, (elem_end+nelements-1)//nelements
      raw = blocks[..., block_start:block_end, :].contiguous()
      if typ == 2 and elem_start % nelements == 0 and elem_end % nelements == 0: raw = raw.flatten().bitcast(dtypes.uint32)
      raws.append(raw.to(device))
      offsets.append(elem_start-block_start*nelements)
  else:
    per_dev = shape[axis] // len(devices)
    raws = [blocks[tuple(slice(i*per_dev, (i+1)*per_dev) if a == axis else slice(None) for a in range(len(shape)+1))]
            .contiguous().to(device) for i, device in enumerate(devices)]
    offsets = [0] * len(devices)
  Tensor.realize(*raws)
  return GGUFQuantizedTensor(tuple(raws), shape, typ, axis, tuple(offsets))

def _shard_tensor(tensor:Tensor, data_start:int, t_info:tuple[str, tuple[int, ...], int, int], devices:tuple[str, ...], axis:int) -> Tensor:
  name, dims, typ, off = t_info
  shape,ndev,start = tuple(reversed(dims)),len(devices),data_start+off
  assert shape[axis] % ndev == 0, f"{name}: axis {axis} size {shape[axis]} does not divide {ndev} devices"
  shard_shape = (*shape[:axis], shape[axis]//ndev, *shape[axis+1:])
  if axis == 0:
    shard_nbytes = _ggml_nbytes(prod(shard_shape), typ)
    raws = [tensor[start+i*shard_nbytes:start+(i+1)*shard_nbytes].to(d).contiguous() for i,d in enumerate(devices)]
  else:
    block_elems,block_nbytes = _GGML_QUANT[typ] if typ in _GGML_QUANT else (1, _GGML_NATIVE[typ].itemsize)
    assert shape[-1] % block_elems == 0, f"{name}: last dim {shape[-1]} does not divide block size {block_elems}"
    blocks_per_row = shape[-1]//block_elems
    packed = tensor[start:start+_ggml_nbytes(prod(shape), typ)].to("CPU").realize().reshape(*shape[:-1], blocks_per_row*block_nbytes)
    if axis == len(shape)-1 and blocks_per_row % ndev:
      per_dev, decoded_shards = shape[-1]//ndev, []
      for i,d in enumerate(devices):
        elem_start, elem_end = i*per_dev, (i+1)*per_dev
        block_start, block_end = elem_start//block_elems, (elem_end+block_elems-1)//block_elems
        raw = packed[..., block_start*block_nbytes:block_end*block_nbytes].contiguous().to(d).realize()
        local_elems = (block_end-block_start)*block_elems
        decoded = ggml_data_to_tensor(raw, prod(shape[:-1])*local_elems, typ, block_contiguous=False).reshape(*shape[:-1], local_elems)
        decoded_shards.append(decoded[..., elem_start-block_start*block_elems:elem_end-block_start*block_elems])
      return Tensor(decoded_shards[0].uop.mstack(*[x.uop for x in decoded_shards[1:]]).unshard(axis))
    raws = [x.contiguous().to(d) for x,d in zip(packed.chunk(ndev, dim=axis), devices)]
  Tensor.realize(*raws)
  packed = Tensor(raws[0].uop.mstack(*[r.uop for r in raws[1:]]))
  decoded = ggml_data_to_tensor(packed, prod(shard_shape), typ, block_contiguous=False).reshape(*shard_shape)
  return Tensor(decoded.uop.unshard(axis))

def _shard_tensor_fused(tensor:Tensor, data_start:int, t_a:_TensorInfo, t_b:_TensorInfo, devices:tuple[str, ...]) -> Tensor:
  name, dims, typ, off_a = t_a
  shape, ndev = tuple(reversed(dims)), len(devices)
  nelements, nbytes = _GGML_QUANT[typ] if typ in _GGML_QUANT else (1, _GGML_NATIVE[typ].itemsize)
  assert shape[-1] % nelements == 0, f"{name}: last dim {shape[-1]} does not divide block size {nelements}"
  assert shape[1] % ndev == 0, f"{name}: axis 1 size {shape[1]} does not divide {ndev} devices"
  row_nbytes, count = shape[-1]//nelements*nbytes, shape[1]//ndev
  blocks = [tensor[data_start+off:data_start+off+prod(shape[:-1])*row_nbytes].to("CPU").realize().reshape(*shape[:-1], row_nbytes)
            for off in (off_a, t_b[3])]
  raws = [blocks[0][:, i*count:(i+1)*count].cat(blocks[1][:, i*count:(i+1)*count], dim=1).contiguous().to(device)
          for i,device in enumerate(devices)]
  Tensor.realize(*raws)
  shard_shape = (shape[0], 2*count, *shape[2:])
  packed = Tensor(raws[0].uop.mstack(*[raw.uop for raw in raws[1:]]))
  decoded = ggml_data_to_tensor(packed, prod(shard_shape), typ, block_contiguous=False).reshape(*shard_shape)
  return Tensor(decoded.uop.unshard(1))

def _quantized_tensor_fused(tensor:Tensor, data_start:int, t_a:_TensorInfo, t_b:_TensorInfo,
                            devices:tuple[str, ...]) -> GGUFQuantizedTensor:
  name, dims, typ, _ = t_a
  shape, ndev = tuple(reversed(dims)), len(devices)
  nelements, nbytes = _GGML_QUANT[typ]
  assert shape[-1] % nelements == 0, f"{name}: last dim {shape[-1]} does not divide block size {nelements}"
  assert shape[1] % ndev == 0, f"{name}: axis 1 size {shape[1]} does not divide {ndev} devices"
  nblocks, count = shape[-1]//nelements, shape[1]//ndev
  blocks = [tensor[data_start+t[3]:data_start+t[3]+prod(shape[:-1])*nblocks*nbytes].to("CPU").realize()
            .reshape(*shape[:-1], nblocks, nbytes) for t in (t_a, t_b)]
  raws = []
  for i,device in enumerate(devices):
    raw = blocks[0][:, i*count:(i+1)*count].cat(blocks[1][:, i*count:(i+1)*count], dim=1).contiguous()
    if typ == 2:
      assert nblocks*nbytes % 4 == 0, f"{name}: Q4_0 row bytes must divide 4"
      raw = raw.flatten().bitcast(dtypes.uint32)
    raws.append(raw.to(device))
  Tensor.realize(*raws)
  return GGUFQuantizedTensor(tuple(raws), (shape[0], 2*shape[1], *shape[2:]), typ, 1, (0,)*ndev)

def _gguf_parse(tensor:Tensor, devices:tuple[str, ...]|None=None,
                shard_policy:Callable[[str], int|str|None]|None=None,
                fuse_policy:Callable[[str], tuple[str, str]|None]|None=None,
                raw_policy:Callable[[str, int], bool]|None=None,
                tensor_filter:Callable[[str], bool]|None=None) -> tuple[dict, dict[str, Tensor|GGUFQuantizedTensor]]:
  if devices is None: tensor = tensor.to(None).realize()
  r = io.BufferedReader(TensorIO(tensor), 1_000_000)
  magic, version, n_tensors, n_kv = r.read(4), read_int32(r), read_int64(r), read_int64(r)
  if magic != b"GGUF" or version not in [2, 3]: raise ValueError("Invalid GGUF format!")

  kv_data = {}
  for _ in range(n_kv):
    k, typ = read_str(r), read_int32(r)
    kv_data[k] = readers[typ](r)

  t_infos = [ (read_str(r), tuple(read_uint64(r) for _ in range(read_uint32(r))), read_int32(r), read_uint64(r)) for _ in range(n_tensors) ]
  alignment, pos = kv_data.get("general.alignment", 32), r.tell()
  data_start = round_up(pos, alignment)

  if devices is None:
    state_dict = {}
    for t_info in t_infos:
      name, dims, typ, off = t_info
      if tensor_filter is not None and not tensor_filter(name): continue
      state_dict[name] = (_quantized_tensor(tensor, data_start, t_info) if raw_policy is not None and raw_policy(name, typ) else
                          ggml_data_to_tensor(tensor[data_start+off:], prod(dims), typ).reshape(*reversed(dims)))
    return kv_data, state_dict
  state_dict = {}
  infos_by_name = {t[0]: t for t in t_infos}
  fuse_plan: dict[str, tuple[_TensorInfo, str, bool]] = {}
  fused_inputs: set[str] = set()
  if fuse_policy is not None:
    for t_info in t_infos:
      name,dims,typ,_ = t_info
      if tensor_filter is not None and not tensor_filter(name): continue
      if (pair := fuse_policy(name)) is None or (partner := infos_by_name.get(pair[0])) is None: continue
      if tensor_filter is not None and not tensor_filter(pair[0]): continue
      spec = shard_policy(name) if shard_policy is not None else None
      keep_quantized = raw_policy is not None and raw_policy(name, typ)
      partner_quantized = raw_policy is not None and raw_policy(pair[0], partner[2])
      if partner[1] == dims and partner[2] == typ and spec == 1 and shard_policy is not None and shard_policy(pair[0]) == 1 \
          and keep_quantized == partner_quantized:
        fuse_plan[name] = (partner, pair[1], keep_quantized)
        fused_inputs.add(pair[0])
  for t_info in t_infos:
    name,dims,typ,off = t_info
    if tensor_filter is not None and not tensor_filter(name): continue
    if name in fused_inputs: continue
    spec = shard_policy(name) if shard_policy is not None else None
    keep_quantized = raw_policy is not None and raw_policy(name, typ)
    if (plan := fuse_plan.get(name)) is not None:
      partner, fused_name, keep_quantized = plan
      state_dict[fused_name] = (_quantized_tensor_fused(tensor, data_start, t_info, partner, devices)
                                if keep_quantized else
                                _shard_tensor_fused(tensor, data_start, t_info, partner, devices))
      continue
    if keep_quantized:
      assert isinstance(spec, int) or spec == "replicate"
      state_dict[name] = _quantized_tensor(tensor, data_start, t_info, devices, spec if isinstance(spec, int) else None)
    elif isinstance(spec, int):
      state_dict[name] = _shard_tensor(tensor, data_start, t_info, devices, spec)
    else:
      n = prod(dims)
      raw = tensor[data_start+off:data_start+off+_ggml_nbytes(n, typ)].to(devices if spec == "replicate" else devices[0]).realize()
      state_dict[name] = ggml_data_to_tensor(raw, n, typ).reshape(*reversed(dims))
  return kv_data, state_dict

def _gguf_split_paths(path: pathlib.Path, kv: dict) -> list[pathlib.Path]:
  if (total := kv.get('split.count', 1)) <= 1: return [path]
  if kv.get('split.no', 0) != 0: raise ValueError(f"multi-part GGUF must be loaded from the first split, got split.no={kv['split.no']}")
  if not (m := re.match(r"^(.*)-00001-of-\d{5}\.gguf$", str(path))): raise ValueError(f"first split path must end with -00001-of-NNNNN.gguf: {path}")
  return [pathlib.Path(f"{m.group(1)}-{i:05d}-of-{total:05d}.gguf") for i in range(1, total+1)]

def gguf_load(fn:Tensor|str|pathlib.Path, devices:tuple[str, ...]|None=None,
              shard_policy:Callable[[str], int|str|None]|None=None,
              fuse_policy:Callable[[str], tuple[str, str]|None]|None=None,
              raw_policy:Callable[[str, int], bool]|None=None, split_limit:int|None=None,
              tensor_filter:Callable[[str], bool]|None=None) -> tuple[dict, dict[str, Tensor|GGUFQuantizedTensor]]:
  """
  Loads a .gguf file, returning the `kv_data` and `state_dict`. Multi-part splits are auto-merged when loaded by path.

  ```python
  import pathlib
  from tinygrad import Device, Tensor
  from tinygrad.llm.gguf import gguf_load

  gguf_tensor = Tensor(pathlib.Path("Meta-Llama-3-8B-Instruct.Q4_0.gguf")).to(Device.DEFAULT)
  kv_data, state_dict = gguf_load(gguf_tensor)
  ```

  NOTE: The provided tensor must be on a device that supports execution.
  """
  if split_limit is not None and split_limit < 1: raise ValueError(f"split_limit must be positive, got {split_limit}")
  kv, sd = _gguf_parse(fn if isinstance(fn, Tensor) else Tensor(pathlib.Path(fn)), devices, shard_policy, fuse_policy, raw_policy, tensor_filter)
  if kv.get('split.count', 1) <= 1: return kv, sd
  if isinstance(fn, Tensor): raise ValueError("multi-part GGUF requires a path argument (got Tensor)")
  paths = _gguf_split_paths(pathlib.Path(fn), kv)
  if split_limit is not None: paths = paths[:split_limit]
  for pp in paths[1:]: sd.update(_gguf_parse(Tensor(pp), devices, shard_policy, fuse_policy, raw_policy, tensor_filter)[1])
  return kv, sd
