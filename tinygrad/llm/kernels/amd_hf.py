from __future__ import annotations
import functools
from typing import cast
from tinygrad import Tensor, UOp, Device, Context
from tinygrad.dtype import DType, dtypes, float_to_fp8, fp8_to_float
from tinygrad.uop.ops import AxisType, KernelInfo
from tinygrad.llm.kernels.amd import warp_reduce, _amd_load, fp8_matmul, amd_custom_kernels_supported

# kernels for the hf checkpoint formats: fp8 weights with a scale per 128x128 block, mxfp4 experts (e2m1 pairs + e8m0 scale per 32)
FP8_BLOCK, FP4_GROUP, ROWS = 128, 32, 4

def _fp8_to_float(word:UOp, byte:int) -> UOp:
  return UOp.native(f"__builtin_amdgcn_cvt_f32_fp8({{0}}, {byte})", shape=word.shape, dtype=dtypes.float, fmt=True).call(word, inline=True)

@functools.cache
def _fp8_quant_kernel(q:UOp, scale:UOp, x:UOp, tokens:int, in_features:int, limit:float) -> UOp:
  # the activations in fp8 with a scale per 128 (aiter group_fp8_quant), a lane packs four
  blocks = in_features // FP8_BLOCK
  token_block, lane = UOp.range(tokens*blocks, 0, AxisType.GLOBAL), UOp.range(32, -1, AxisType.WARP)
  token, block = token_block // blocks, token_block % blocks
  vals = [x.reshape(tokens, blocks, 32, 4)[token, block, lane, i].load().float() for i in range(4)]
  amax = warp_reduce(functools.reduce(UOp.maximum, (v.abs() for v in vals)), maximum=True, full_wave=True)
  s = amax.maximum(1e-10) / limit
  qv = [(v / s).clip(-limit, limit) for v in vals]
  def pack(hi:str, *src:UOp) -> UOp:
    code = f"__builtin_amdgcn_cvt_pk_fp8_f32({{0}}, {{1}}, {{2}}, {hi})"
    return UOp.native(code, shape=s.shape, dtype=dtypes.int32, fmt=True).call(*src, inline=True)
  word = pack("true", qv[2], qv[3], pack("false", qv[0], qv[1], UOp.const(0, dtypes.int32)))
  stores = (q.reshape(tokens, blocks, 32)[token, block, lane].store(word.bitcast(dtypes.uint32)), scale[token, block.valid(lane.eq(0))].store(s))
  return UOp.group(*stores).end(token_block, lane).sink(arg=KernelInfo(name="fp8_quant", opts_to_apply=()))

@functools.cache
def _fp8_gemv_kernel(out:UOp, w:UOp, q:UOp, xs:UOp, ws:UOp, tokens:int, out_features:int, in_features:int, groups:int) -> UOp:
  # one wave per (token, output row), a lane takes 16 weights at a time; the rows of group g read the activations of group g
  row_block, wave = UOp.range(tokens*out_features//ROWS, 0, AxisType.GLOBAL), UOp.range(ROWS, 1, AxisType.LOCAL)
  lane = UOp.range(32, -1, AxisType.WARP)
  token, row = (row_block*ROWS + wave) // out_features, (row_block*ROWS + wave) % out_features
  xrow = token*groups + row // (out_features//groups)
  w, q = w.reshape(out_features, in_features//16, 4), q.reshape(tokens*groups, in_features//16, 4)
  acc = UOp.const(0, dtypes.float)
  for i in range(in_features // 512):
    chunk = lane + i*32
    wv, xv = _amd_load(w[row, chunk, 0], 4, stream=True), _amd_load(q[xrow, chunk, 0], 4)
    dot = sum((_fp8_to_float(wv[j], b) * _fp8_to_float(xv[j], b) for j in range(4) for b in range(4)), UOp.const(0, dtypes.float))
    acc = acc + dot * xs[xrow, chunk // 8] * ws[row // FP8_BLOCK, chunk // 8]
  stores = out[token, row.valid(lane.eq(0))].store(warp_reduce(acc, full_wave=True))
  return stores.end(row_block, wave, lane).sink(arg=KernelInfo(name="fp8_gemv", opts_to_apply=()))

def fp8_linear(x:Tensor, weight:Tensor, scale:Tensor, block_size:tuple[int, int]=(FP8_BLOCK, FP8_BLOCK), groups:int=1) -> Tensor:
  # x: (..., groups*in), weight: (out, in) fp8, scale: (out/128, in/128) -> (..., out); the weight rows of group g see the inputs of group g
  out_features, in_features = cast(tuple[int, int], weight.shape)
  tokens = x.numel() // (groups*in_features)
  if not (amd_custom_kernels_supported(x.device) and isinstance(tokens, int) and tokens <= 8 and block_size == (FP8_BLOCK, FP8_BLOCK) and
          in_features % 512 == 0 and out_features % (groups*ROWS) == 0):
    out = fp8_matmul(x.unflatten(-1, (groups, 1, in_features)), weight.reshape(groups, -1, in_features), scale.reshape(groups, -1, scale.shape[-1]),
                     block_size)
    return out.flatten(-3)
  out = Tensor(_fp8_gemv(x.reshape(tokens*groups, in_features).contiguous().uop, weight.uop, scale.uop, groups))
  return out.reshape(*x.shape[:-1], out_features).cast(x.dtype)

# a custom kernel's output is a new buffer, so these are cached on their inputs: linears that share an input share its quantization
@functools.lru_cache(maxsize=64)
def _fp8_quant(x:UOp, limit:float) -> tuple[UOp, UOp]:
  rows, in_features = cast(tuple[int, int], x.shape)
  q = Tensor.empty(rows, in_features//4, dtype=dtypes.uint32, device=x.device)
  s = Tensor.empty(rows, in_features//FP8_BLOCK, dtype=dtypes.float32, device=x.device)
  fxn = functools.partial(_fp8_quant_kernel, tokens=rows, in_features=in_features, limit=limit)
  q, s = Tensor.custom_kernel(q, s, Tensor(x), fxn=fxn)[:2]
  return q.uop, s.uop

@functools.lru_cache(maxsize=64)
def _fp8_gemv(x:UOp, weight:UOp, scale:UOp, groups:int) -> UOp:
  (rows, in_features), out_features = cast(tuple[int, int], x.shape), cast(int, weight.shape[0])
  q, s = _fp8_quant(x, 224.0 if weight.dtype == dtypes.fp8e4m3fnuz else 448.0)
  out = Tensor.empty(rows // groups, out_features, dtype=dtypes.float32, device=x.device)
  fxn = functools.partial(_fp8_gemv_kernel, tokens=rows // groups, out_features=out_features, in_features=in_features, groups=groups)
  return Tensor.custom_kernel(out, Tensor(weight).bitcast(dtypes.uint32).contiguous(), Tensor(q), Tensor(s), Tensor(scale).float().contiguous(),
                              fxn=fxn)[0].uop

@functools.cache
def _bf16_gemv_kernel(out:UOp, w:UOp, x:UOp, tokens:int, out_features:int, in_features:int) -> UOp:
  # one wave per (token, output row), a lane takes 8 weights (4 words) at a time
  row_block, wave = UOp.range(tokens*out_features//ROWS, 0, AxisType.GLOBAL), UOp.range(ROWS, 1, AxisType.LOCAL)
  lane = UOp.range(32, -1, AxisType.WARP)
  token, row = (row_block*ROWS + wave) // out_features, (row_block*ROWS + wave) % out_features
  w, x = w.reshape(out_features, in_features//8, 4), x.reshape(tokens, in_features//8, 8)
  acc = UOp.const(0, dtypes.float)
  for i in range(in_features // 256):
    chunk = lane + i*32
    words, xlo, xhi = _amd_load(w[row, chunk, 0], 4, stream=True), _amd_load(x[token, chunk, 0], 4), _amd_load(x[token, chunk, 4], 4)
    xv = [xlo[j] for j in range(4)] + [xhi[j] for j in range(4)]
    # a word holds the bf16 pair (even, odd) in its (low, high) half
    acc = acc + sum(((words[j] << 16).bitcast(dtypes.float) * xv[2*j] + (words[j] & 0xffff0000).bitcast(dtypes.float) * xv[2*j+1] for j in range(4)),
                    UOp.const(0, dtypes.float))
  stores = out[token, row.valid(lane.eq(0))].store(warp_reduce(acc, full_wave=True))
  return stores.end(row_block, wave, lane).sink(arg=KernelInfo(name="bf16_gemv", opts_to_apply=()))

def bf16_linear(x:Tensor, weight:Tensor) -> Tensor|None:
  # decode with a bf16 weight (out, in), None when the kernel does not apply
  out_features, in_features = cast(tuple[int, int], weight.shape)
  tokens = x.numel() // in_features
  if not (amd_custom_kernels_supported(x.device) and isinstance(tokens, int) and tokens <= 8 and in_features % 256 == 0 and out_features % ROWS == 0):
    return None
  out = Tensor.empty(tokens, out_features, dtype=dtypes.float32, device=x.device)
  fxn = functools.partial(_bf16_gemv_kernel, tokens=tokens, out_features=out_features, in_features=in_features)
  out = Tensor.custom_kernel(out, weight.bitcast(dtypes.uint32).contiguous(), x.reshape(tokens, in_features).float().contiguous(), fxn=fxn)[0]
  return out.reshape(*x.shape[:-1], out_features)

def _fp8_dtype(device:str) -> DType|None:
  from tinygrad.renderer.cstyle import amd_fp8s
  with Context(ALLOW_DEVICE_USAGE=1): fp8s = amd_fp8s(getattr(Device[device].renderer, "target").arch)
  return fp8s[0] if fp8s else None

def _byte_perm(a:UOp|int, b:UOp|int, selectors:UOp|int) -> UOp:
  srcs = [x if isinstance(x, UOp) else UOp.const(x, dtypes.uint32) for x in (a, b, selectors)]
  return UOp.native("__builtin_amdgcn_perm({}, {}, {})", shape=srcs[2].shape, dtype=dtypes.uint32, fmt=True).call(*srcs, inline=True)

def _fp4_to_fp8(nibbles:UOp, lut:tuple[int, ...]) -> UOp:
  # four nibbles, one a byte, to their fp8 bytes: a 16 entry byte table in 4 words
  low, high = _byte_perm(lut[1], lut[0], nibbles), _byte_perm(lut[3], lut[2], nibbles & 0x07070707)
  return _byte_perm(high, low, ((nibbles & 0x08080808) >> 1) | 0x03020100)

@functools.cache
def _mxfp4_moe_kernel(out:UOp, codes:UOp, scales:UOp, x:UOp, sel:UOp, out_features:int, in_features:int, x_share:int,
                      lut:tuple[int, ...], rows:int=4) -> UOp:
  # out rows are (token, slot) x output feature: the slot's expert weight rows, x_share slots share an activation row
  # a wave takes `rows` output rows sharing its activation loads, a lane one scale group: 16 bytes of codes, the low nibble is the even weight
  slots, groups = out.shape[0], in_features // FP4_GROUP
  row_block, wave = UOp.range(slots*out_features//(ROWS*rows), 0, AxisType.GLOBAL), UOp.range(ROWS, 1, AxisType.LOCAL)
  lane = UOp.range(32, -1, AxisType.WARP)
  slot, row = (row_block*ROWS*rows + wave*rows) // out_features, (row_block*ROWS*rows + wave*rows) % out_features
  erows = [sel[slot].cast(dtypes.weakint)*out_features + row + r for r in range(rows)]
  codes, scales, x = codes.reshape(-1, groups, 4), scales.reshape(-1, groups), x.reshape(-1, groups, FP4_GROUP)
  accs = [UOp.const(0, dtypes.float)] * rows
  for i in range(groups // 32):
    group = lane + i*32
    xv = []
    for j in range(8):
      vec = _amd_load(x[slot // x_share, group, j*4], 4)
      xv += [vec[k] for k in range(4)]
    for r, erow in enumerate(erows):
      words = _amd_load(codes[erow, group, 0], 4, stream=True)
      # the even and the odd weights of a word as fp8 bytes
      fp8 = [_fp4_to_fp8((words[j] >> shift) & 0x0f0f0f0f, lut) for j in range(4) for shift in (0, 4)]
      dot = sum((_fp8_to_float(fp8[2*(k//8) + k%2], k%8//2) * xv[k] for k in range(32)), UOp.const(0, dtypes.float))
      accs[r] = accs[r] + dot * (scales[erow, group].cast(dtypes.uint32) << 23).bitcast(dtypes.float)  # 2**(e-127)
  stores = [out[slot, (row + r).valid(lane.eq(0))].store(warp_reduce(acc, full_wave=True)) for r, acc in enumerate(accs)]
  return UOp.group(*stores).end(row_block, wave, lane).sink(arg=KernelInfo(name="mxfp4_moe", opts_to_apply=()))

def mxfp4_moe(codes:Tensor, scales:Tensor, sel:Tensor, x:Tensor) -> Tensor:
  # codes: (E, out, in/2) uint8, scales: (E, out, in/32) e8m0, sel: (B, T, k), x: (B, T, 1|k, in) -> (B, T, k, out)
  if not (amd_custom_kernels_supported(x.device) and isinstance(x.device, str) and isinstance(sel.shape[1], int) and (fp8:=_fp8_dtype(x.device))):
    return (x.unsqueeze(-2) @ mxfp4_dequant(codes[sel], scales[sel]).transpose(-1, -2)).squeeze(-2)
  table = [float_to_fp8(v, fp8) for v in (0, 0.5, 1, 1.5, 2, 3, 4, 6)]
  table += [float_to_fp8(-fp8_to_float(b, fp8), fp8) if b else 0 for b in table]
  B, T, k = cast(tuple[int, int, int], sel.shape)
  out_features, in_features, x_slots = int(codes.shape[1]), int(x.shape[-1]), int(x.shape[-2])
  out = Tensor.empty(B*T*k, out_features, dtype=dtypes.float32, device=x.device)
  fxn = functools.partial(_mxfp4_moe_kernel, out_features=out_features, in_features=in_features, x_share=k//x_slots,
                          lut=tuple(sum(b << (8*i) for i, b in enumerate(table[j:j+4])) for j in range(0, 16, 4)))
  srcs = (codes.bitcast(dtypes.uint32).contiguous(), scales.contiguous(), x.float().contiguous(), sel.cast(dtypes.int32).flatten())
  out = Tensor.custom_kernel(out, *srcs, fxn=fxn)[0]
  return out.reshape(B, T, k, out_features)

def mxfp4_dequant(codes:Tensor, scales:Tensor) -> Tensor:
  # the reference: (..., out, in/2) codes and (..., out, in/32) scales -> (..., out, in) float
  lut = Tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6], device=codes.device)
  nibbles = codes.unsqueeze(-1).rshift(Tensor.const((0, 4), dtypes.uint8)).bitwise_and(0xF).flatten(-2)
  return (lut[nibbles.cast(dtypes.int32)].unflatten(-1, (-1, FP4_GROUP)) * (scales.float() - 127).exp2().unsqueeze(-1)).flatten(-2)
