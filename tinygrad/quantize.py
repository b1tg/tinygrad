"""Packed fp4 and block-scale quantization.

The core `fp4e2m1` dtype (see `tinygrad/dtype.py`) stores one 4-bit value per byte. Real
hardware and model formats instead pack two fp4 values per byte and share a scale across a
block of elements. This module builds that representation out of plain tensor ops:

* `pack_fp4` / `unpack_fp4`: two fp4 values per `uint8` (low nibble is the even element).
* `quantize_mxfp4` / `dequant_mxfp4`: OCP MXFP4, one E8M0 power-of-two scale per 32 elements.
* `quantize_nvfp4` / `dequant_nvfp4`: NVFP4, one E4M3 scale per 16 elements plus a per-tensor
  fp32 scale (`x ~= q * block_scale * global_scale`).

Rounding goes through the native `fp4e2m1`/`fp8e4m3` dtypes, so run quantize/dequantize on a
backend that supports them (PYTHON is the reference) for exact results.
"""
from tinygrad.dtype import dtypes
from tinygrad.tensor import Tensor
from tinygrad.uop.ops import UOp

MXFP4_BLOCK, NVFP4_BLOCK = 32, 16
FP4_MAX, E4M3_MAX = 6.0, 448.0

def pack_fp4(x:Tensor) -> Tensor:
  """Pack pairs of `fp4e2m1` values along the last axis into `uint8` (two per byte)."""
  if x.dtype != dtypes.fp4e2m1: raise ValueError(f"pack_fp4 expects fp4e2m1, got {x.dtype}")
  if (n:=x.shape[-1]) % 2: raise ValueError(f"pack_fp4 needs an even last dim, got {n}")
  nib = x.bitcast(dtypes.uint8)
  return nib[..., 0::2].bitwise_or(nib[..., 1::2].lshift(4))

def unpack_fp4(q:Tensor) -> Tensor:
  """Unpack `uint8` (two fp4 per byte) into `fp4e2m1` along the last axis."""
  if q.dtype != dtypes.uint8: raise ValueError(f"unpack_fp4 expects uint8, got {q.dtype}")
  lo, hi = q.bitwise_and(0xF), q.rshift(4)
  return lo.unsqueeze(-1).cat(hi.unsqueeze(-1), dim=-1).flatten(-2).bitcast(dtypes.fp4e2m1)

def _e8m0(amax:Tensor) -> tuple[Tensor, Tensor]:
  """E8M0 encode non-negative per-block `amax`: returns (uint8 exponent, float32 scale)."""
  # round the exponent to nearest (the hardware uses the same +0x200000 bias), then shift by
  # 2 so the block max maps near 4 (the E2M1 element max exponent)
  bits = amax.float().bitcast(dtypes.uint32)
  exp = (bits.add(0x200000).bitwise_and(0xFF800000).rshift(23).bitwise_and(0xFF).cast(dtypes.int32) - 129).clip(-127, 127)
  exp = (amax == 0).where(0, exp)  # zero block gets scale 1.0
  return (exp + 127).cast(dtypes.uint8), exp.cast(dtypes.float32).exp2()

def _e8m0_float(scale:Tensor) -> Tensor: return (scale.cast(dtypes.int32) - 127).cast(dtypes.float32).exp2()

def _blockify(x:Tensor, block:int) -> Tensor:
  if (n:=x.shape[-1]) % block: raise ValueError(f"last dim {n} is not a multiple of block {block}")
  return x.reshape(*x.shape[:-1], -1, block)

def _unblock(s:Tensor, block:int, shape:tuple[int|UOp, ...]) -> Tensor:
  return s.unsqueeze(-1).expand(*s.shape, block).reshape(shape)

def quantize_mxfp4(x:Tensor, block:int=MXFP4_BLOCK) -> tuple[Tensor, Tensor]:
  """Quantize to MXFP4: returns packed `uint8` and one E8M0 `uint8` scale per `block` elements."""
  x = x.float()
  scale_u8, scale = _e8m0(_blockify(x, block).abs().max(axis=-1))
  scaled = (_blockify(x, block) / scale.unsqueeze(-1)).reshape(x.shape)
  return pack_fp4(scaled.cast(dtypes.fp4e2m1)), scale_u8

def dequant_mxfp4(q:Tensor, scale_u8:Tensor, block:int=MXFP4_BLOCK) -> Tensor:
  """Inverse of `quantize_mxfp4`, returning float32. `scale_u8` holds the E8M0 exponents."""
  vals = unpack_fp4(q).float()
  return vals * _unblock(_e8m0_float(scale_u8), block, vals.shape)

def quantize_nvfp4(x:Tensor, block:int=NVFP4_BLOCK) -> tuple[Tensor, Tensor, Tensor]:
  """Quantize to NVFP4: returns packed `uint8`, per-`block` E4M3 scales (`uint8`) and a per-tensor fp32 scale."""
  x = x.float()
  amax = x.abs().max().clip(1e-12)
  global_scale = amax / (E4M3_MAX * FP4_MAX)
  block_scale = (_blockify(x, block).abs().max(axis=-1) / (global_scale * FP4_MAX)).clip(0, E4M3_MAX).cast(dtypes.fp8e4m3)
  scaled = (_blockify(x, block) / (block_scale.float().unsqueeze(-1) * global_scale)).reshape(x.shape)
  return pack_fp4(scaled.cast(dtypes.fp4e2m1)), block_scale.bitcast(dtypes.uint8), global_scale

def dequant_nvfp4(q:Tensor, block_scale_u8:Tensor, global_scale:Tensor, block:int=NVFP4_BLOCK) -> Tensor:
  """Inverse of `quantize_nvfp4`, returning float32."""
  vals = unpack_fp4(q).float()
  bs = block_scale_u8.bitcast(dtypes.fp8e4m3).float()
  return vals * _unblock(bs, block, vals.shape) * global_scale
