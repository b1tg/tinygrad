"""Tensor-parallel configuration and placement; model math stays in model.py."""
from __future__ import annotations
from dataclasses import replace
from typing import TYPE_CHECKING
from tinygrad import Tensor, UOp, nn
from tinygrad.helpers import get_child
from tinygrad.llm.kernels.amd import Linear, amd_custom_kernels_supported
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


def _local_shard(t:Tensor, devices:tuple[str, ...], axis:int, groups:tuple[int, ...]=()) -> Tensor:
  parts = t.split(groups, dim=axis) if groups else (t,)
  assert all(p.shape[axis] % len(devices) == 0 for p in parts), "uneven TP split"
  shards = []
  for rank,device in enumerate(devices):
    local = Tensor.cat(*(p.chunk(len(devices), dim=axis)[rank].contiguous().to(device) for p in parts), dim=axis)
    shards.append(local.contiguous().realize())
  return Tensor(UOp.mstack(*(p.uop for p in shards)))


def load_sharded(model, state:dict[str, Tensor], config:TransformerConfig, devices:tuple[str, ...]):
  for name,target in nn.state.get_state_dict(model).items():
    value = state.pop(name)
    module, key = get_child(model, name.rsplit('.', 1)[0]), name.split('.', 2)[-1] if name.startswith('blk.') else name
    if name == 'token_embd.weight':
      target.replace(value.to(devices[0]))
      continue
    axis = (1 if key in ('attn_output.weight', 'ffn_down.weight', 'ssm_out.weight') else 0) if isinstance(module, Linear) else None
    if key in ('ssm_conv1d.weight', 'ssm_dt.bias', 'ssm_a'): axis = 0
    groups:tuple[int, ...] = ()
    if (ssm:=config.ssm) is not None and name.startswith('blk.') and config.ssm_layers[int(name.split('.')[1])]:
      repeats = ssm.time_step_rank//ssm.group_count
      if key in ('attn_qkv.weight', 'ssm_conv1d.weight'): groups = (ssm.group_count*ssm.state_size,)*2 + (ssm.inner_size//repeats,)*repeats
      elif key in ('attn_gate.weight', 'ssm_out.weight', 'ssm_alpha.weight', 'ssm_beta.weight', 'ssm_dt.bias', 'ssm_a'):
        assert axis is not None
        groups = (int(value.shape[axis])//repeats,)*repeats
    if isinstance(module, Linear) and name.endswith('.weight') and all(amd_custom_kernels_supported(d) for d in devices):
      out_features, in_features = map(int, value.shape)
      packed = Linear(in_features, out_features, bias=False)
      packed.set_quantized(value)
      if packed.ggml_type is not None:
        assert axis is not None
        if axis == 1: assert in_features % (256*len(devices)) == 0 and all(g % (256*len(devices)) == 0 for g in groups)
        raw = packed.weight
        if raw.device in devices: raw = raw.clone()  # tied embedding/output: keep packed words materialized on the source GPU
        raw = raw.realize().reshape(out_features, in_features//256, -1)
        module.ggml_type = packed.ggml_type
        module.weight = _local_shard(raw, devices, axis, tuple(g//256 for g in groups) if axis == 1 else groups).flatten().realize()
        continue
    placed = value.to(devices).contiguous() if axis is None else _local_shard(value, devices, axis, groups)
    assert placed.shape == target.shape, f"{name}: {placed.shape} != {target.shape}"
    target.replace(placed.realize())
