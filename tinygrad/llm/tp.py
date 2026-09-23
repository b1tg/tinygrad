"""GGUF placement policy: logical model dimensions stay global."""
from tinygrad import Tensor, UOp, dtypes
from tinygrad.helpers import prod
from tinygrad.llm.gguf import ggml_data_to_tensor, GGUFQuantizedTensor, GGUFLoader, _GGML_NATIVE, _GGML_QUANT
from tinygrad.llm.kernels.amd import QUANT_SIZES, amd_custom_kernels_supported

# Unlisted weights are replicated. QKV/conv stay whole so Q/K repetition retains its global ordering.
_TP_LAYOUT = {"attn_q.weight":0, "attn_k.weight":0, "attn_v.weight":0, "attn_q.bias":0, "attn_k.bias":0, "attn_v.bias":0,
              "attn_output.weight":1, "ffn_gate.weight":0, "ffn_up.weight":0, "ffn_down.weight":1, "output.weight":0,
              "attn_gate.weight":0, "ssm_alpha.weight":0, "ssm_beta.weight":0, "ssm_a":0, "ssm_dt.bias":0, "ssm_out.weight":1}

def replicate(x:Tensor, devices:tuple[str, ...]) -> Tensor:
  return x.pad_to(x.max_shape).to(devices).shrink(tuple((0, s) for s in x.shape))

def gguf_sharder(devices:tuple[str, ...]) -> GGUFLoader:
  assert len(devices) > 1 and len(set(devices)) == len(devices), "TP requires distinct devices"
  packed_kernels = all(amd_custom_kernels_supported(d) for d in devices)
  def load(kv:dict, name:str, raw:Tensor, shape:tuple[int, ...], typ:int) -> Tensor|GGUFQuantizedTensor:
    arch = kv['general.architecture']
    assert not kv.get(f'{arch}.expert_count', 0) and not kv.get(f'{arch}.attention.kv_lora_rank', 0) and arch != 'kimi-linear'
    assert all(kv.get(f'{arch}.attention.{k}', len(devices)) % len(devices) == 0 for k in ('head_count', 'head_count_kv')), 'uneven heads'
    key = name.split('.', 2)[-1] if name.startswith('blk.') else name
    if key == 'token_embd.weight': return ggml_data_to_tensor(raw.to(devices[0]), prod(shape), typ).reshape(shape)
    axis = _TP_LAYOUT.get(key)
    if axis is not None: assert shape[axis] % len(devices) == 0, f"{name}: uneven TP split"
    block, size = _GGML_QUANT[typ] if typ in _GGML_QUANT else (1, _GGML_NATIVE[typ].itemsize)
    packed = packed_kernels and typ in QUANT_SIZES and len(shape) == 2 and shape[-1] % 256 == 0 \
      and not (arch == 'llama' and key in ('attn_q.weight', 'attn_k.weight'))
    word = dtypes.uint16 if typ == 14 else dtypes.uint32
    storage = raw.reshape(*shape[:-1], shape[-1]//block, size)
    if axis == len(shape)-1:
      assert shape[-1] % (block*len(devices)) == 0, f'{name}: shard crosses a quantization block'
      storage = storage.to('CPU').realize()
    pieces = storage.chunk(len(devices), dim=axis) if axis is not None else (storage,)*len(devices)
    shards = [p.contiguous().flatten().bitcast(word if packed else dtypes.uint8).to(d).clone().realize() for p,d in zip(pieces, devices)]
    local = tuple(s//len(devices) if i == axis else s for i,s in enumerate(shape))
    data = Tensor(UOp.mstack(*(p.uop for p in shards)))
    if packed:
      data = data.reshape(local[0], local[1]//block, -1)
      if axis is not None: data = Tensor(data.uop.unshard(axis))
      return GGUFQuantizedTensor(data, shape, typ)
    decoded = ggml_data_to_tensor(data, prod(local), typ).reshape(local).clone().realize()
    return (Tensor(decoded.uop.unshard(axis)) if axis is not None else decoded).contiguous().realize()
  return load
