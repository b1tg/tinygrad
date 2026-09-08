import functools, io, pathlib, re, struct
from typing import Any, Callable

from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
from tinygrad.helpers import prod, round_up
from tinygrad.nn.state import TensorIO
from tinygrad.uop.ops import UOp

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
               10:(256,84), 11:(256,110), 12:(256,144), 13:(256,176), 14:(256,210), 16:(256,66), 17:(256,74),
               18:(256,98), 19:(256,50), 20:(32,18), 21:(256,110), 22:(256,82), 23:(256,136), 29:(256,56),
               39:(32,17), 41:(128,18)}

def ggml_data_to_tensor(t: Tensor, n: int|UOp, ggml_type: int) -> Tensor:
  """
  Converts ggml tensor data to a tinygrad tensor.

  Supported native types: float32 (id: 0), float16 (id: 1), int8 (id: 24),
  int16 (id: 25), int32 (id: 26), int64 (id: 27), float64 (id: 28), bfloat16 (id: 30)
  Supported quantized types: Q4_0 (id: 2), Q4_1 (id: 3), Q5_0 (id: 6),
  Q5_1 (id: 7), Q8_0 (id: 8), Q4_K (id: 12), Q5_K (id: 13),
  Q6_K (id: 14), IQ3_XXS (id: 18), IQ3_S (id: 21), IQ2_S (id: 22), IQ4_XS (id: 23), MXFP4 (id: 39), Q1_0 (id: 41)
  """
  # https://github.com/ggerganov/ggml/blob/323951f1bdcdfbd5b5ff3a9a7c3770e63b1a560e/include/ggml.h#L356

  if (dtype := _GGML_NATIVE.get(ggml_type)) is not None:
    return t[:dtype.itemsize * n].contiguous().bitcast(dtype)

  def q_to_uint8(t: Tensor, b: int) -> Tensor:
    # TODO: rewrite with arange?
    shift_tensor, bitmask = Tensor.const(tuple(2**(i*b) for i in range(8//b)), t.dtype), 0xff >> (8 - b)
    return t.unsqueeze(-1).div(shift_tensor, rounding_mode="trunc").bitwise_and(bitmask).transpose(-1, -2).flatten(-2)

  if (nelements_nbytes := _GGML_QUANT.get(ggml_type)) is not None:
    from tinygrad.runtime.autogen import ggml_common as _ggml
    blocks = t[:(n//nelements_nbytes[0])*nelements_nbytes[1]].reshape((-1, nelements_nbytes[1])).contiguous()
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
    if ggml_type == 10:
      scales, qs = blocks[:, :16], blocks[:, 16:80]
      d, dmin = blocks[:, 80:82].bitcast(dtypes.float16).cast(dtypes.float32), blocks[:, 82:84].bitcast(dtypes.float16).cast(dtypes.float32)
      dl = (d * scales.bitwise_and(0xF)).unsqueeze(-1)
      ml = (dmin * scales.rshift(4)).unsqueeze(-1)
      q = q_to_uint8(qs.reshape(-1, 2, 32), 2).reshape(-1, 16, 16).cast(dtypes.float32)
      return (dl * q - ml).flatten(-2)
    if ggml_type == 11:
      hmask, qs, packed_scales = blocks[:, :32], blocks[:, 32:96], blocks[:, 96:108]
      low = q_to_uint8(packed_scales[:, :8], 4).bitwise_and(0xF)
      high = q_to_uint8(packed_scales[:, 8:12], 2).bitwise_and(0x3)
      scales = low.bitwise_or(high.lshift(4)).cast(dtypes.int16) - 32
      d = blocks[:, 108:110].bitcast(dtypes.float16).cast(dtypes.float32)
      dl = (d * scales).unsqueeze(-1)
      ql = q_to_uint8(qs.reshape(-1, 2, 32), 2).reshape(-1, 16, 16).cast(dtypes.int16)
      qh = q_to_uint8(hmask, 1).reshape(-1, 16, 16).bitwise_xor(1).cast(dtypes.int16)
      return (dl * (ql - qh * 4)).flatten(-2)
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
    if ggml_type == 16:
      d = blocks[:, :2].bitcast(dtypes.float16).cast(dtypes.float32).reshape((-1, 1, 1, 1))
      words = blocks[:, 2:66].bitcast(dtypes.uint32).reshape((-1, 8, 2))
      db = d * (words[..., 1].rshift(28).cast(dtypes.float32) + 0.5).reshape((-1, 8, 1, 1)) * 0.25
      sign_idx = words[..., 1].unsqueeze(-1).rshift(Tensor.const((0, 7, 14, 21), dtypes.uint32)).bitwise_and(0x7F)
      sign_masks = Tensor(list(_ggml.ksigns_iq2xs), dtype=dtypes.uint8, device=t.device)[sign_idx.cast(dtypes.int32)]
      signs = (q_to_uint8(sign_masks.unsqueeze(-1), 1) == 0).where(1.0, -1.0)
      grid_idx = words[..., 0].contiguous().bitcast(dtypes.uint8).reshape((-1, 8, 4))
      grid = _ggml_iq_grid(t.device, _ggml.iq2xxs_grid, (256, 8))[grid_idx.cast(dtypes.int32)]
      return (db * grid * signs).flatten(-3)
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
    if ggml_type == 19:
      d = blocks[:, :2].bitcast(dtypes.float16).cast(dtypes.float32)
      qh = blocks[:, 34:50].bitcast(dtypes.uint16)
      dl = (d * (qh.rshift(12).bitwise_and(7) * 2 + 1)).reshape((-1, 8, 1, 1))
      delta = (qh.bitwise_and(0x8000) == 0).where(0.125, -0.125).reshape((-1, 8, 1, 1))
      high = qh.unsqueeze(-1).rshift(Tensor.const((0, 3, 6, 9), dtypes.uint16)).bitwise_and(7).reshape((-1, 32))
      indices = blocks[:, 2:34].cast(dtypes.uint16).bitwise_or(high.lshift(8)).cast(dtypes.int32)
      raw_grid = _ggml_iq_grid(t.device, _ggml.iq1s_grid, (2048, 8))
      signed_grid = (raw_grid < 128).where(raw_grid, raw_grid - 256)
      grid = signed_grid[indices].reshape((-1, 8, 4, 8))
      return (dl * (grid + delta)).flatten(-3)
    if ggml_type == 20:
      d = blocks[:, :2].bitcast(dtypes.float16).cast(dtypes.float32)
      q = q_to_uint8(blocks[:, 2:18], 4).cast(dtypes.int32)
      lut = Tensor(list(_ggml.kvalues_iq4nl), dtype=dtypes.float32, device=t.device)
      return d * lut[q]
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
    if ggml_type == 29:
      scale_words = blocks[:, 48:56].bitcast(dtypes.uint16).reshape((-1, 4))
      scale_parts = scale_words.bitwise_and(0xF000).rshift(Tensor.const((12, 8, 4, 0), dtypes.uint16))
      d = scale_parts.sum(-1).cast(dtypes.uint16).bitcast(dtypes.float16).cast(dtypes.float32).reshape((-1, 1))
      scales = scale_words.unsqueeze(-1).rshift(Tensor.const((0, 3, 6, 9), dtypes.uint16)).bitwise_and(7).reshape((-1, 16))
      dl = (d * (scales * 2 + 1)).reshape((-1, 8, 2, 1, 1))
      qh = blocks[:, 32:48].unsqueeze(-1).rshift(Tensor.const((0, 4), dtypes.uint8))
      indices = blocks[:, :32].cast(dtypes.uint16).bitwise_or(
        qh.bitwise_and(7).cast(dtypes.uint16).lshift(8).reshape((-1, 32))).cast(dtypes.int32)
      delta = (qh.bitwise_and(8) == 0).where(0.125, -0.125).reshape((-1, 8, 2, 2, 1))
      raw_grid = _ggml_iq_grid(t.device, _ggml.iq1s_grid, (2048, 8))
      signed_grid = (raw_grid < 128).where(raw_grid, raw_grid - 256)
      grid = signed_grid[indices].reshape((-1, 8, 2, 2, 8))
      return (dl * (grid + delta)).flatten(-4)
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

TensorInfo = tuple[str, tuple[int, ...], int, int]

def _ggml_nbytes(n:int, typ:int) -> int:
  if (dtype := _GGML_NATIVE.get(typ)) is not None: return n * dtype.itemsize
  nelements, nbytes = _GGML_QUANT[typ]
  return n // nelements * nbytes

def _gguf_info(tensor:Tensor) -> tuple[dict, int, list[TensorInfo]]:
  r = io.BufferedReader(TensorIO(tensor), 1_000_000)
  magic, version, n_tensors, n_kv = r.read(4), read_int32(r), read_int64(r), read_int64(r)
  if magic != b"GGUF" or version not in [2, 3]: raise ValueError("Invalid GGUF format!")

  kv_data = {}
  for _ in range(n_kv):
    k, typ = read_str(r), read_int32(r)
    kv_data[k] = readers[typ](r)

  t_infos:list[TensorInfo] = [
    (read_str(r), tuple(read_uint64(r) for _ in range(read_uint32(r))), read_int32(r), read_uint64(r)) for _ in range(n_tensors)]
  alignment, pos = kv_data.get("general.alignment", 32), r.tell()
  return kv_data, round_up(pos, alignment), t_infos

def _pipeline_device_map(t_infos:list[TensorInfo], devices:tuple[str, ...], block_count:int) -> dict[str, str]:
  """Balance consecutive transformer blocks by packed bytes, with embeddings first and output weights last."""
  if not devices: raise ValueError("pipeline placement requires at least one device")
  if block_count < len(devices): raise ValueError(f"cannot place {block_count} blocks on {len(devices)} non-empty pipeline stages")
  sizes, first_extra, last_extra = [0] * block_count, 0, 0
  for name, dims, typ, _ in t_infos:
    if (m := re.match(r"^blk\.(\d+)\.", name)):
      if (layer := int(m.group(1))) >= block_count: continue
      sizes[layer] += _ggml_nbytes(prod(dims), typ)
    elif name.startswith("token_embd."):
      first_extra += _ggml_nbytes(prod(dims), typ)
    elif name.startswith("output.") or name.startswith("output_norm."):
      last_extra += _ggml_nbytes(prod(dims), typ)

  prefix = [0]
  for size in sizes: prefix.append(prefix[-1] + size)
  inf = sum(sizes) + first_extra + last_extra + 1
  dp, previous = [[inf] * (block_count+1) for _ in devices], [[-1] * (block_count+1) for _ in devices]
  for end in range(1, block_count+1): dp[0][end] = first_extra + prefix[end]
  for stage in range(1, len(devices)):
    for end in range(stage+1, block_count+1):
      extra = last_extra if stage == len(devices)-1 else 0
      for start in range(stage, end):
        score = max(dp[stage-1][start], prefix[end] - prefix[start] + extra)
        if score < dp[stage][end]: dp[stage][end], previous[stage][end] = score, start

  cuts, end = [block_count], block_count
  for stage in range(len(devices)-1, 0, -1):
    end = previous[stage][end]
    cuts.append(end)
  cuts = [0] + list(reversed(cuts))

  device_map:dict[str, str] = {}
  for name, _, _, _ in t_infos:
    if (m := re.match(r"^blk\.(\d+)\.", name)):
      layer = int(m.group(1))
      if layer >= block_count: continue
      stage = next(i for i, stop in enumerate(cuts[1:]) if layer < stop)
      device_map[name] = devices[stage]
    elif name.startswith("output.") or name.startswith("output_norm."): device_map[name] = devices[-1]
    else: device_map[name] = devices[0]
  return device_map

def _gguf_tensors(tensor:Tensor, data_start:int, t_infos:list[TensorInfo], device_map:dict[str, str]|None=None) -> dict[str, Tensor]:
  if device_map is None: tensor = tensor.to(None).realize()
  state_dict:dict[str, Tensor] = {}
  for name, dims, typ, off in t_infos:
    if device_map is not None and name not in device_map: continue
    n = prod(dims)
    raw = tensor[data_start + off:data_start + off + _ggml_nbytes(n, typ)]
    if device_map is not None: raw = raw.to(device_map[name]).realize()
    state_dict[name] = ggml_data_to_tensor(raw, n, typ).reshape(*reversed(dims))
  return state_dict
def _gguf_parse(tensor: Tensor) -> tuple[dict, dict[str, Tensor]]:
  # TODO: remove the need for copy to default device
  tensor = tensor.to(None).realize()
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

  state_dict = {name: ggml_data_to_tensor(tensor[data_start + off:], prod(dims), typ).reshape(*reversed(dims)) for name, dims, typ, off in t_infos}
  return kv_data, state_dict

def _gguf_split_paths(path: pathlib.Path, kv: dict) -> list[pathlib.Path]:
  if (total := kv.get('split.count', 1)) <= 1: return [path]
  if kv.get('split.no', 0) != 0: raise ValueError(f"multi-part GGUF must be loaded from the first split, got split.no={kv['split.no']}")
  if not (m := re.match(r"^(.*)-00001-of-\d{5}\.gguf$", str(path))): raise ValueError(f"first split path must end with -00001-of-NNNNN.gguf: {path}")
  return [pathlib.Path(f"{m.group(1)}-{i:05d}-of-{total:05d}.gguf") for i in range(1, total+1)]

def gguf_load(fn:Tensor|str|pathlib.Path, devices:tuple[str, ...]|None=None,
              block_count:int|None=None) -> tuple[dict, dict[str, Tensor]]:
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
  tensor = fn if isinstance(fn, Tensor) else Tensor(pathlib.Path(fn))
  kv, data_start, t_infos = _gguf_info(tensor)
  if kv.get('split.count', 1) > 1 and isinstance(fn, Tensor):
    raise ValueError("multi-part GGUF requires a path argument (got Tensor)")

  parts = [(tensor, data_start, t_infos)]
  if not isinstance(fn, Tensor):
    for split_path in _gguf_split_paths(pathlib.Path(fn), kv)[1:]:
      part = Tensor(split_path)
      _, part_data_start, part_infos = _gguf_info(part)
      parts.append((part, part_data_start, part_infos))

  all_infos = [info for _, _, infos in parts for info in infos]
  device_map = None
  if devices is not None:
    arch = kv['general.architecture']
    used_blocks = block_count if block_count is not None else \
      kv[f'{arch}.block_count'] - kv.get(f'{arch}.nextn_predict_layers', 0)
    device_map = _pipeline_device_map(all_infos, devices, used_blocks)
  state_dict:dict[str, Tensor] = {}
  for part, part_data_start, part_infos in parts:
    state_dict.update(_gguf_tensors(part, part_data_start, part_infos, device_map))
  return kv, state_dict
