"""Tensor parallel blocks with local packed weights and two reductions per transformer block."""
from dataclasses import replace
from tinygrad import Tensor, Device, UOp, nn, function
from tinygrad.llm.kernels.amd import Linear, QUANT_SIZES, amd_custom_kernels_supported
from tinygrad.llm.model import TransformerBlock, GatedDeltaNetBlock
from tinygrad.helpers import tqdm


def partition(t:Tensor, axis:int, rank:int, count:int, groups:tuple[int, ...]|None=None) -> Tensor:
  parts = t.split(groups, dim=axis) if groups is not None else (t,)
  assert all(p.shape[axis] % count == 0 for p in parts), f"cannot split {t.shape} into {count} shards along {axis}"
  return Tensor.cat(*(p.chunk(count, dim=axis)[rank] for p in parts), dim=axis).contiguous()


def split_linear(source:Linear, target:Linear, device:str, axis:int, rank:int, count:int, groups:tuple[int, ...]|None=None):
  if amd_custom_kernels_supported(device) and source.ggml_type is None: source.set_quantized(source.weight)
  if source.ggml_type is not None:
    assert source.in_features % (256*count if axis == 1 else 256) == 0, "TP must preserve GGML blocks"
    words = QUANT_SIZES[source.ggml_type] // source.weight.dtype.itemsize
    raw = source.weight.realize().reshape(source.out_features, source.in_features//256, words)
    if axis == 1 and groups is not None:
      assert all(g % 256 == 0 for g in groups), "TP must preserve GGML blocks"
      groups = tuple(g//256 for g in groups)
    target.weight = partition(raw, axis, rank, count, groups).flatten().to(device).contiguous().realize()
    target.ggml_type = source.ggml_type
  else:
    target.weight = partition(source.weight, axis, rank, count, groups).to(device).contiguous().realize()
  if source.bias is not None:
    target.bias = (partition(source.bias, 0, rank, count, groups) if axis == 0 else source.bias/count).to(device).contiguous().realize()


class TensorParallelBlock:
  def __init__(self, source, devices:tuple[str, ...]):
    self.devices, self.parts = devices, []
    config, count = source.config, len(devices)
    if config.num_experts or config.kv_lora_rank or (config.ssm is not None and config.ssm.kda):
      raise ValueError("tensor parallel currently supports dense attention and Qwen Gated DeltaNet blocks")
    assert config.n_heads % count == config.n_kv_heads % count == config.hidden_dim % count == 0
    local = replace(config, n_heads=config.n_heads//count, n_kv_heads=config.n_kv_heads//count, hidden_dim=config.hidden_dim//count)
    ssm = config.ssm
    recurrent = isinstance(source, GatedDeltaNetBlock)
    if recurrent:
      assert ssm is not None and ssm.group_count % count == 0
      local = replace(local, ssm=replace(ssm, group_count=ssm.group_count//count, time_step_rank=ssm.time_step_rank//count,
                                        inner_size=ssm.inner_size//count))
      # q/k heads repeat across the value heads. Preserve that ordering within each shard.
      repeats = ssm.time_step_rank//ssm.group_count
      value_groups = (ssm.inner_size//repeats,)*repeats
      head_groups = (ssm.group_count,)*repeats
      qkv_groups = (ssm.group_count*ssm.state_size,)*2 + value_groups
    for rank, device in enumerate(devices):
      block = GatedDeltaNetBlock(local, local.ssm) if recurrent else TransformerBlock(local)
      for name, module in vars(source).items():
        target = getattr(block, name)
        if isinstance(module, Linear):
          axis = 1 if name in ('ffn_down', 'attn_output', 'ssm_out') else 0
          groups = None
          if recurrent:
            if name == 'attn_qkv': groups = qkv_groups
            elif name in ('attn_gate', 'ssm_out'): groups = value_groups
            elif name in ('ssm_alpha', 'ssm_beta'): groups = head_groups
          split_linear(module, target, device, axis, rank, count, groups)
        elif isinstance(module, nn.RMSNorm) and module.weight is not None: target.weight = module.weight.to(device).contiguous().realize()
        elif recurrent and name in ('ssm_a', 'ssm_dt', 'ssm_conv1d'):
          tensor = module if isinstance(module, Tensor) else module['bias' if name == 'ssm_dt' else 'weight']
          value = partition(tensor, 0, rank, count, qkv_groups if name == 'ssm_conv1d' else head_groups).to(device).realize()
          if isinstance(module, Tensor): setattr(block, name, value)
          else: target['bias' if name == 'ssm_dt' else 'weight'] = value
      self.parts.append(block)

  def _reusable_prefix_len(self, prefix_len:int, cached_len:int) -> int: return prefix_len

  def __call__(self, xs:tuple[Tensor, ...], start_pos):
    for block, x in zip(self.parts, xs): block._init_state(x)
    @function(precompile=True, allow_implicit=True)
    def run(start_pos:int|UOp, *xs:Tensor):
      attn = [b._attention(b.attn_norm(x), start_pos).contiguous() for b,x in zip(self.parts, xs)]
      hs = [(x + sum(a.to(d) for a in attn)).contiguous() for x,d in zip(xs, self.devices)]
      ffn = [b._feed_forward(b.ffn_norm(h)).contiguous() for b,h in zip(self.parts, hs)]
      return tuple((h + sum(f.to(d) for f in ffn)).contiguous() for h,d in zip(hs, self.devices))
    return run(start_pos, *xs)


def shard_model(model, count:int):
  devices = tuple(Device.canonicalize(f'{Device.DEFAULT}:{i}') for i in range(count))
  model.tp_blocks = [TensorParallelBlock(block, devices) for block in tqdm(model.blk, desc="sharding")]
  model.blk = []
  outputs = []
  for rank, device in enumerate(devices):
    layer = Linear(model.output.in_features, model.output.out_features//count, bias=False)
    split_linear(model.output, layer, device, 0, rank, count)
    outputs.append(layer)
  model.output, model.tp_outputs = outputs[0], outputs[1:]
  model.token_embd.weight = model.token_embd.weight.to(devices[0]).contiguous().realize()
  model.output_norm.weight = model.output_norm.weight.to(devices[0]).contiguous().realize()
  model.devices = devices
