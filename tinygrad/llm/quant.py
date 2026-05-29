from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes

QK8_0 = 32
Q8_0_BLOCK_BYTES = 34

def dot4_i8_packed(a:Tensor, b:Tensor, acc:Tensor|None=None) -> Tensor:
  ret = Tensor.zeros(a.shape, dtype=dtypes.int32, device=a.device) if acc is None else acc
  for shift in (0, 8, 16, 24):
    av = ((a if shift == 0 else a.rshift(shift)).bitwise_and(255)).cast(dtypes.int8).cast(dtypes.int32)
    bv = ((b if shift == 0 else b.rshift(shift)).bitwise_and(255)).cast(dtypes.int8).cast(dtypes.int32)
    ret = ret + av * bv
  return ret

def pack_i8x4(x:Tensor) -> Tensor:
  return x.bitcast(dtypes.uint8).reshape(*x.shape[:-1], x.shape[-1]//4, 4).bitcast(dtypes.uint32).squeeze(-1)

def pack_u8x4(x:Tensor) -> Tensor:
  return x.reshape(*x.shape[:-1], x.shape[-1]//4, 4).bitcast(dtypes.uint32).squeeze(-1)

def quantize_q8_blocks(x:Tensor) -> tuple[Tensor, Tensor]:
  xb = x.reshape(*x.shape[:-1], x.shape[-1]//QK8_0, QK8_0)
  scale = xb.abs().max(axis=-1).float() / 127.0
  q = (xb / scale.maximum(1e-12).unsqueeze(-1)).round().clip(-127, 127).cast(dtypes.int8)
  return pack_i8x4(q), scale

def q8_0_weight_blocks(raw:Tensor, out_features:int, in_features:int) -> Tensor:
  return raw.reshape(out_features, in_features//QK8_0, Q8_0_BLOCK_BYTES)

def q8_0_pack_weight_u32(blocks:Tensor) -> Tensor:
  # 9 uint32 words per Q8_0 block: scale bits in the low half of word 0,
  # followed by 8 aligned i8x4 payload words.
  wd = blocks[..., :2].pad(tuple((0, 0) for _ in blocks.shape[:-1]) + ((0, 2),)).bitcast(dtypes.uint32).squeeze(-1)
  return wd.unsqueeze(-1).cat(pack_u8x4(blocks[..., 2:]), dim=-1)

def q8_0_matvec(x:Tensor, blocks:Tensor) -> Tensor:
  # blocks shape: (out_features, in_features//32, 34), GGML Q8_0 layout: fp16 d + int8 qs[32].
  xq, xd = quantize_q8_blocks(x)
  return q8_0_matvec_prequant(xq, xd, blocks)

def q8_0_matvec_prequant(xq:Tensor, xd:Tensor, blocks:Tensor) -> Tensor:
  if blocks.dtype == dtypes.uint32 and blocks.shape[-1] == 9:
    wd = blocks[..., 0].bitcast(dtypes.uint16).reshape(*blocks.shape[:-1], 2)[..., 0].bitcast(dtypes.float16).cast(dtypes.float32)
    wq = blocks[..., 1:]
  else:
    wd = blocks[..., :2].bitcast(dtypes.float16).cast(dtypes.float32).squeeze(-1)
    wq = pack_u8x4(blocks[..., 2:])
  dots = dot4_i8_packed(wq.unsqueeze(0), xq.unsqueeze(-3)).sum(axis=-1, dtype=dtypes.int32).float()
  return (dots * wd.unsqueeze(0) * xd.unsqueeze(-2)).sum(axis=-1)

def q8_0_matvec_many(x:Tensor, blocks:list[Tensor]) -> list[Tensor]:
  xq, xd = quantize_q8_blocks(x)
  return [q8_0_matvec_prequant(xq, xd, w) for w in blocks]
