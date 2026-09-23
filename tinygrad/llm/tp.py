"""Tensor-parallel configuration, weight placement, and collective operations."""
from __future__ import annotations
from dataclasses import replace
from typing import TYPE_CHECKING
from tinygrad import Tensor, UOp, nn, dtypes, getenv
from tinygrad.helpers import get_child, tqdm, prod
from tinygrad.llm.kernels.amd import Linear, QUANT_SIZES, amd_custom_kernels_supported
from tinygrad.llm.gguf import GGUFWeight
from tinygrad.uop.ops import Ops
if TYPE_CHECKING:
  from tinygrad.llm.model import TransformerConfig


def replicate(x:Tensor, devices:tuple[str, ...]) -> Tensor:
  # Device copies must have static sizes; restore the logical prefill length after broadcasting.
  return x.pad_to(x.max_shape).to(devices).shrink(tuple((0, s) for s in x.shape))


def sum_shards(x:Tensor) -> Tensor:
  if not isinstance(x.device, tuple): return x
  summed = Tensor(x.pad_to(x.max_shape).contiguous().uop.allreduce(Ops.ADD, x.device))
  return summed.shrink(tuple((0, s) for s in x.shape))


def shard_config(config:TransformerConfig, devices:tuple[str, ...]) -> TransformerConfig:
  count = len(devices)
  assert count > 1 and len(set(devices)) == count, "tensor parallelism requires distinct devices"
  if config.num_experts or config.kv_lora_rank or (config.ssm is not None and config.ssm.kda):
    raise ValueError("tensor parallelism supports dense attention and Qwen Gated DeltaNet blocks")
  dims = (config.n_heads, config.n_kv_heads, config.hidden_dim, config.dense_hidden_dim, config.vocab_size)
  assert all(v % count == 0 for v in dims), "uneven TP split"
  ssm = config.ssm
  if ssm is not None:
    assert all(v % count == 0 for v in (ssm.group_count, ssm.time_step_rank, ssm.inner_size)), "uneven SSM split"
    ssm = replace(ssm, group_count=ssm.group_count//count, time_step_rank=ssm.time_step_rank//count, inner_size=ssm.inner_size//count)
  return replace(config, n_heads=config.n_heads//count, n_kv_heads=config.n_kv_heads//count,
                 hidden_dim=config.hidden_dim//count, dense_hidden_dim=config.dense_hidden_dim//count, ssm=ssm)


def shard_layout(name:str, module, shape:tuple[int, ...], config:TransformerConfig) -> tuple[int|None, tuple[int, ...]|None]:
  key = name.split('.', 2)[-1] if name.startswith('blk.') else name
  axis = (1 if key in ('attn_output.weight', 'ffn_down.weight', 'ssm_out.weight') else 0) if isinstance(module, Linear) else None
  if key in ('ssm_conv1d.weight', 'ssm_dt.bias', 'ssm_a'): axis = 0
  groups = None
  if config.ssm is not None and name.startswith('blk.') and config.ssm_layers[int(name.split('.')[1])]:
    ssm = config.ssm
    repeats = ssm.time_step_rank//ssm.group_count
    if key in ('attn_qkv.weight', 'ssm_conv1d.weight'):
      groups = (ssm.group_count*ssm.state_size,)*2 + (ssm.inner_size//repeats,)*repeats
    elif key in ('attn_gate.weight', 'ssm_out.weight', 'ssm_alpha.weight', 'ssm_beta.weight', 'ssm_dt.bias', 'ssm_a'):
      assert axis is not None
      groups = (shape[axis]//repeats,)*repeats
  return axis, groups


def partition(t:Tensor, axis:int, rank:int, count:int, groups:tuple[int, ...]|None=None) -> Tensor:
  parts = t.split(groups, dim=axis) if groups is not None else (t,)
  assert all(p.shape[axis] % count == 0 for p in parts), f"cannot split {t.shape} into {count} shards along {axis}"
  return Tensor.cat(*(p.chunk(count, dim=axis)[rank] for p in parts), dim=axis).contiguous()


def load_linear(weight:GGUFWeight, target:Linear, device:str, axis:int, rank:int, count:int,
                groups:tuple[int, ...]|None=None, rope:tuple[int, int]|None=None):
  out_features, in_features = weight.shape
  if weight.ggml_type in QUANT_SIZES and amd_custom_kernels_supported(device) and in_features % 256 == 0 and rope is None:
    block_bytes = QUANT_SIZES[weight.ggml_type]
    word = dtypes.uint16 if weight.ggml_type == 14 else dtypes.uint32
    if axis == 0:
      # Each row range is contiguous in the GGUF. Transfer it before concatenating grouped projections.
      offset, parts = 0, []
      row_bytes = in_features//256*block_bytes
      for rows in groups or (out_features,):
        assert rows % count == 0
        first, last = offset + rank*(rows//count), offset + (rank+1)*(rows//count)
        parts.append(weight.data[first*row_bytes:last*row_bytes].bitcast(word).to(device).realize())
        offset += rows
      assert offset == out_features
      raw = Tensor.cat(*parts).contiguous().realize()
    else:
      assert in_features % (256*count) == 0, "TP must preserve GGML blocks"
      if groups is not None:
        assert all(g % 256 == 0 for g in groups)
        groups = tuple(g//256 for g in groups)
      # A column shard is strided in the file. Stage only this weight, never the whole GGUF.
      packed = weight.data.to("CPU").realize().reshape(out_features, in_features//256, block_bytes)
      raw = partition(packed, 1, rank, count, groups).flatten().bitcast(word).to(device).realize()
    target.weight, target.ggml_type = raw.contiguous().realize(), weight.ggml_type
  else:
    decoded = partition(weight.decode("CPU"), axis, rank, count, groups).to(device).cast(target.weight.dtype)
    if rope is not None:
      head_dim, prefix = rope
      w = decoded.reshape(-1, head_dim, in_features)
      decoded = w[:, :prefix].cat(w[:, prefix:].rearrange("n (h two) d -> n (two h) d", two=2), dim=1).reshape(decoded.shape)
    target.weight = decoded.contiguous().realize()
  assert prod(target.weight.shape) == (target.in_features*target.out_features if target.ggml_type is None else
                                       target.in_features*target.out_features//256*QUANT_SIZES[target.ggml_type]//target.weight.dtype.itemsize)


def load_sharded(model, weights:dict[str, GGUFWeight], config:TransformerConfig, devices:tuple[str, ...], arch:str=""):
  if arch in ('qwen35', 'qwen35moe', 'glm4moe'): weights = {k.replace('post_attention_norm', 'ffn_norm'):v for k,v in weights.items()}
  if 'output.weight' not in weights and 'token_embd.weight' in weights: weights['output.weight'] = weights['token_embd.weight']
  for name,target in tqdm(nn.state.get_state_dict(model).items(), desc="sharding"):
    weight = weights[name]
    module = get_child(model, name.rsplit('.', 1)[0])
    dtype = dtypes.half if getenv("HALF", 1) else weight.dtype
    if name == 'token_embd.weight':
      target.replace(weight.decode(devices[0]).cast(dtype).contiguous().realize())
      continue
    axis, groups = shard_layout(name, module, weight.shape, config)
    if isinstance(module, Linear) and name.endswith('.weight'):
      key = name.rsplit('.', 2)[-2] if name.startswith('blk.') else name.rsplit('.', 1)[0]
      rope = (config.head_dim, config.head_dim-config.rope_dim if key == 'attn_q' else 0) \
        if arch == 'llama' and key in ('attn_q', 'attn_k') else None
      assert axis is not None
      shards = []
      for rank,device in enumerate(devices):
        local = Linear(module.in_features, module.out_features, bias=False)
        local.weight = local.weight.cast(dtype)
        load_linear(weight, local, device, axis, rank, len(devices), groups, rope)
        shards.append(local.weight)
      module.ggml_type = local.ggml_type
      module.weight = Tensor(UOp.mstack(*(w.uop for w in shards))).contiguous().realize()
    else:
      value = weight.decode("CPU").cast(dtype)
      shards = [value.to(d) if axis is None else partition(value, axis, i, len(devices), groups).to(d) for i,d in enumerate(devices)]
      placed = Tensor(UOp.mstack(*(w.contiguous().realize().uop for w in shards))).contiguous().realize()
      target.replace(placed)
