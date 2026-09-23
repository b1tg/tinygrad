"""Tensor parallel blocks with local packed weights and two reductions per transformer block."""
from dataclasses import replace
from tinygrad import Tensor, Device, UOp, nn, function, dtypes
from tinygrad.llm.kernels.amd import Linear, QUANT_SIZES, amd_custom_kernels_supported
from tinygrad.llm.model import TransformerBlock, GatedDeltaNetBlock
from tinygrad.helpers import tqdm, prod
from tinygrad.llm.gguf import GGUFWeight


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


class TensorParallelBlock:
  def __init__(self, source, devices:tuple[str, ...], weights:dict[str, GGUFWeight]|None=None, arch:str=""):
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
          if weights is None: split_linear(module, target, device, axis, rank, count, groups)
          else:
            rope = (config.head_dim, config.head_dim-config.rope_dim if name == 'attn_q' else 0) \
              if arch == 'llama' and name in ('attn_q', 'attn_k') else None
            target.weight = target.weight.cast(module.weight.dtype)
            load_linear(weights[name+'.weight'], target, device, axis, rank, count, groups, rope)
            if module.bias is not None:
              bias = weights[name+'.bias'].decode("CPU")
              target.bias = (partition(bias, 0, rank, count, groups) if axis == 0 else bias/count).to(device).cast(module.bias.dtype).realize()
        elif isinstance(module, nn.RMSNorm) and module.weight is not None:
          target.weight = (module.weight.to(device) if weights is None else weights[name+'.weight'].decode(device).cast(module.weight.dtype)) \
            .contiguous().realize()
        elif recurrent and name in ('ssm_a', 'ssm_dt', 'ssm_conv1d'):
          tensor = module if isinstance(module, Tensor) else module['bias' if name == 'ssm_dt' else 'weight']
          if weights is not None:
            key = name if isinstance(module, Tensor) else name + ('.bias' if name == 'ssm_dt' else '.weight')
            tensor = weights[key].decode("CPU").cast(tensor.dtype)
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


def shard_model(model, count:int, weights:dict[str, GGUFWeight]|None=None, arch:str=""):
  devices = tuple(Device.canonicalize(f'{Device.DEFAULT}:{i}') for i in range(count))
  if weights is not None:
    if arch in ('qwen35', 'qwen35moe', 'glm4moe'): weights = {k.replace('post_attention_norm', 'ffn_norm'):v for k,v in weights.items()}
    if 'output.weight' not in weights: weights['output.weight'] = weights['token_embd.weight']
  model.tp_blocks = []
  for i in tqdm(range(len(model.blk)), desc="sharding"):
    prefix = f'blk.{i}.'
    local_weights = {k[len(prefix):]:v for k,v in weights.items() if k.startswith(prefix)} if weights is not None else None
    model.tp_blocks.append(TensorParallelBlock(model.blk.pop(0), devices, local_weights, arch))
  outputs = []
  for rank, device in enumerate(devices):
    layer = Linear(model.output.in_features, model.output.out_features//count, bias=False)
    if weights is None: split_linear(model.output, layer, device, 0, rank, count)
    else:
      layer.weight = layer.weight.cast(model.output.weight.dtype)
      load_linear(weights['output.weight'], layer, device, 0, rank, count)
    outputs.append(layer)
  model.output, model.tp_outputs = outputs[0], outputs[1:]
  for name in ('token_embd', 'output_norm'):
    module = getattr(model, name)
    module.weight = (module.weight.to(devices[0]) if weights is None else weights[name+'.weight'].decode(devices[0]).cast(module.weight.dtype)) \
      .contiguous().realize()
  model.devices = devices
