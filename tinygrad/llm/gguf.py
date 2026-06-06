import functools, io, pathlib, re, struct
from typing import Any, Callable

from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
from tinygrad.helpers import prod, round_up
from tinygrad.nn.state import TensorIO

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
               12:(256,144), 13:(256,176), 14:(256,210), 18:(256,98), 21:(256,110), 22:(256,82), 23:(256,136), 39:(32,17), 41:(128,18)}

def ggml_data_to_tensor(t: Tensor, n: int, ggml_type: int) -> Tensor:
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
    shift_tensor, bitmask = Tensor.stack(*[ Tensor(2**(i*b), device=t.device, dtype=t.dtype) for i in range(8//b) ]), 0xff >> (8 - b)
    return t.unsqueeze(-1).expand((*t.shape,8//b)).div(shift_tensor, rounding_mode="trunc").bitwise_and(bitmask).transpose(-1, -2).flatten(-2)

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
      d = blocks[:,-2:].bitcast(dtypes.float16).cast(dtypes.float32).expand((-1, 256))
      return d * (xl.bitwise_or(xh).bitcast(dtypes.int8) - 32).flatten(-2) * scales
    if ggml_type == 18:
      d = blocks[:, :2].bitcast(dtypes.float16).cast(dtypes.float32).reshape((-1, 1, 1, 1))
      scale_words = blocks[:, 66:98].bitcast(dtypes.uint32)
      db = d * (scale_words.rshift(28).cast(dtypes.float32) + 0.5).reshape((-1, 8, 1, 1)) * 0.5
      sign_idx = scale_words.unsqueeze(-1).rshift(
        Tensor([0, 7, 14, 21], device=t.device, dtype=dtypes.uint32)).bitwise_and(0x7F).reshape((-1, 32)).cast(dtypes.int32)
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
      scale_shifts = Tensor([0, 2, 4, 6, 8, 10, 12, 14], device=t.device, dtype=dtypes.uint16)
      iq4_xs_lut = Tensor(list(_ggml.kvalues_iq4nl), dtype=dtypes.float32, device=t.device)
      scales_l = Tensor.stack((sl:=blocks[:, 4:8]).bitwise_and(0xF), sl.rshift(4), dim=2).reshape((-1, 8))
      scales_h = blocks[:, 2:4].bitcast(dtypes.uint16).unsqueeze(-1).rshift(scale_shifts).bitwise_and(0x03).reshape((-1, 8)).cast(dtypes.uint8)
      scales = (scales_l.bitwise_or(scales_h.lshift(4)).bitcast(dtypes.int8) - 32).cast(dtypes.float32).reshape((-1, 8, 1))
      q = (qs:=blocks[:, 8:].reshape((-1, 8, 16))).bitwise_and(0xF).cat(qs.rshift(4), dim=2)
      return (d * scales * iq4_xs_lut[q]).flatten(-2)
    if ggml_type == 39:
      e = blocks[:, 0].cast(dtypes.uint32)
      small_bits = Tensor([0x00200000, 0x00400000], dtype=dtypes.uint32, device=t.device)[e.clip(0, 1).cast(dtypes.int32)] # e = 0 or e = 1 case
      d = (e < 2).where(small_bits, ((e - 1) * 0x00800000).cast(dtypes.uint32)).bitcast(dtypes.float32).unsqueeze(-1)
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

def _shard_tensor(tensor:Tensor, data_start:int, name:str, dims:tuple[int, ...], typ:int, off:int,
                  devices:tuple[str, ...], axis:int) -> dict[str, Tensor]:
  shape = tuple(reversed(dims))
  if shape[axis] % len(devices) != 0: raise RuntimeError(f"axis size {shape[axis]} does not divide {len(devices)} devices")
  parts, per = [], shape[axis] // len(devices)
  returns_local_weights = name.endswith("_exps.weight")
  if axis == 0:
    row_elems = prod(shape[1:])
    if typ in _GGML_QUANT and row_elems % _GGML_QUANT[typ][0] != 0:
      raise RuntimeError(f"quantized tensor {name} row size {row_elems} does not divide {_GGML_QUANT[typ][0]}")
    part_nbytes = _ggml_nbytes(row_elems, typ)
    raws = [tensor[data_start + off + i*per*part_nbytes:data_start + off + (i+1)*per*part_nbytes].to(d) for i,d in enumerate(devices)]
    Tensor.realize(*raws)
    if returns_local_weights or typ not in _GGML_QUANT:
      parts = [ggml_data_to_tensor(raw, per*row_elems, typ).reshape(per, *shape[1:]) for raw in raws]
    else:
      raw = Tensor(raws[0].uop.mstack(*[r.uop for r in raws[1:]]))
      t = ggml_data_to_tensor(raw, per*row_elems, typ).reshape(per, *shape[1:])
      return {name: Tensor(t.uop.multi(axis))}
  elif typ in _GGML_QUANT:
    qblock, block_bytes = _GGML_QUANT[typ]
    if shape[-1] % qblock != 0: raise RuntimeError(f"quantized tensor {name} last dim {shape[-1]} does not divide {qblock}")
    raw_shape = (*shape[:-1], shape[-1] // qblock, block_bytes)
    raw_view = tensor[data_start + off:data_start + off + _ggml_nbytes(prod(shape), typ)].to("CPU").realize().reshape(*raw_shape)
    raws, part_shapes = [], []
    for i,d in enumerate(devices):
      if axis == len(shape) - 1 and per % qblock != 0: raise RuntimeError(f"quantized shard axis size {per} does not divide {qblock}")
      part_shape = (*shape[:axis], per, *shape[axis+1:])
      raw_axis, raw_per = (len(shape)-1, per // qblock) if axis == len(shape)-1 else (axis, per)
      slc = [slice(None)] * len(raw_shape)
      slc[raw_axis] = slice(i*raw_per, (i+1)*raw_per)
      raws.append(raw_view[tuple(slc)].contiguous().reshape(-1).to(d))
      part_shapes.append(part_shape)
    Tensor.realize(*raws)
    if returns_local_weights:
      parts = [ggml_data_to_tensor(raw, prod(part_shape), typ).reshape(*part_shape) for raw,part_shape in zip(raws, part_shapes)]
    else:
      assert all(s == part_shapes[0] for s in part_shapes)
      raw = Tensor(raws[0].uop.mstack(*[r.uop for r in raws[1:]]))
      t = ggml_data_to_tensor(raw, prod(part_shapes[0]), typ).reshape(*part_shapes[0])
      return {name: Tensor(t.uop.multi(axis))}
  elif axis != 0: raise RuntimeError(f"native tensor {name} only supports axis0 shard")
  if not name.endswith("_exps.weight"): return {name: Tensor(parts[0].uop.mstack(*[p.uop for p in parts[1:]]).multi(axis))}
  return {f"{name[:-6]}weights.{i}":p for i,p in enumerate(parts)}

def _tp_axis(name:str) -> int|None:
  key = name.split(".", 2)[-1] if name.startswith("blk.") else name
  if key in ("attn_k.weight", "attn_v.weight", "attn_k.bias", "attn_v.bias"): return 0
  if key in ("attn_q.weight", "attn_q_b.weight", "attn_k_b.weight", "attn_v_b.weight"): return 0
  if key == "attn_output.weight": return 1
  if key in ("ffn_gate.weight", "ffn_up.weight"): return 0
  if key == "ffn_down.weight": return 1
  if key in ("ffn_gate_shexp.weight", "ffn_up_shexp.weight"): return 0
  if key == "ffn_down_shexp.weight": return 1
  if key in ("ffn_gate_exps.weight", "ffn_up_exps.weight"): return 1
  if key == "ffn_down_exps.weight": return 2
  return None

def _replicated_weight(name:str) -> str|None:
  key = name.split(".", 2)[-1] if name.startswith("blk.") else name
  return key if key in ("attn_norm.weight", "attn_q_a.weight", "attn_q_a_norm.weight",
                        "attn_kv_a_mqa.weight", "attn_kv_a_norm.weight") else None

def _gguf_parse(tensor: Tensor, devices:tuple[str, ...]|None=None) -> tuple[dict, dict[str, Tensor]]:
  # TODO: remove the need for copy to default device
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

  state_dict = {}
  for name, dims, typ, off in t_infos:
    if devices is not None and _replicated_weight(name) is not None:
      n = prod(dims)
      raw = tensor[data_start + off:data_start + off + _ggml_nbytes(n, typ)].to("CPU").realize()
      weight = ggml_data_to_tensor(raw, n, typ).reshape(*reversed(dims))
      state_dict[name] = weight.to(devices).realize()
    elif devices is not None and (axis := _tp_axis(name)) is not None:
      state_dict.update(_shard_tensor(tensor, data_start, name, dims, typ, off, devices, axis))
    else:
      n = prod(dims)
      raw = tensor[data_start + off:data_start + off + _ggml_nbytes(n, typ)].to(devices[0] if devices is not None else None).realize()
      state_dict[name] = ggml_data_to_tensor(raw, n, typ).reshape(*reversed(dims))
  return kv_data, state_dict

def _gguf_split_paths(path: pathlib.Path, kv: dict) -> list[pathlib.Path]:
  if (total := kv.get('split.count', 1)) <= 1: return [path]
  if kv.get('split.no', 0) != 0: raise ValueError(f"multi-part GGUF must be loaded from the first split, got split.no={kv['split.no']}")
  if not (m := re.match(r"^(.*)-00001-of-\d{5}\.gguf$", str(path))): raise ValueError(f"first split path must end with -00001-of-NNNNN.gguf: {path}")
  return [pathlib.Path(f"{m.group(1)}-{i:05d}-of-{total:05d}.gguf") for i in range(1, total+1)]

def gguf_load(fn: Tensor|str|pathlib.Path, devices:tuple[str, ...]|None=None) -> tuple[dict, dict[str, Tensor]]:
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
  kv, sd = _gguf_parse(fn if isinstance(fn, Tensor) else Tensor(pathlib.Path(fn)), devices)
  if kv.get('split.count', 1) <= 1: return kv, sd
  if isinstance(fn, Tensor): raise ValueError("multi-part GGUF requires a path argument (got Tensor)")
  for pp in _gguf_split_paths(pathlib.Path(fn), kv)[1:]: sd.update(_gguf_parse(Tensor(pp), devices)[1])
  return kv, sd
