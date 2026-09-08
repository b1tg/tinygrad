"""Tensor parallel LLM blocks. Packed GGML blocks are partitioned before dequantization."""
import copy
from dataclasses import replace
from tinygrad import Tensor, UOp, function, Device
from tinygrad.llm.gguf import ggml_data_to_tensor
from tinygrad.llm.kernels.amd import Linear, ExpertWeights, packed_q8_0_weight, packed_ggml_weight, amd_custom_kernels_supported
from tinygrad.llm.model import Transformer, FFNBlock, TransformerBlock, GatedDeltaNetBlock


def _replicate(obj, device):
  if isinstance(obj, Tensor): return obj.to(device).contiguous().realize()
  if isinstance(obj, dict): return {k:_replicate(v, device) for k,v in obj.items()}
  if hasattr(obj, '__dict__'):
    ret = copy.copy(obj)
    for k,v in vars(obj).items(): setattr(ret, k, _replicate(v, device))
    return ret
  return obj


def _linear(layer, device, rows=None, cols=None):
  if cols is not None and isinstance(cols[0], int): cols = [cols]
  ret = copy.copy(layer)
  ret._fused_q8_0_weight = None
  ret._q8_0_tried = True
  ret._q8_0_weight = None
  ni, no = layer.in_features, layer.out_features
  ne = layer.num_experts if isinstance(layer, ExpertWeights) else 1
  ret.in_features = ni if cols is None else sum(e-s for s,e in cols)
  ret.out_features = no if rows is None else sum(e-s for s,e in rows)
  supported = amd_custom_kernels_supported(device)
  q8 = packed_q8_0_weight(layer.weight, ne*ni*no) if supported and isinstance(layer, Linear) else None
  packed = packed_ggml_weight(layer.weight, ne*ni*no) if supported and ni % 256 == 0 else None
  if q8 is not None or packed is not None:
    if q8 is not None: typ, raw, block = 8, q8, 32
    else:
      assert packed is not None
      typ, raw, block = *packed, 256
    assert cols is None or all(c % block == 0 for r in cols for c in r), 'shards must contain whole quantization blocks'
    w = raw.reshape(ne, no, ni//block, -1)
    if rows is not None: w = Tensor.cat(*(w[:, s:e] for s,e in rows), dim=1)
    if cols is not None: w = Tensor.cat(*(w[:, :, s//block:e//block] for s,e in cols), dim=2)
    raw = w.contiguous().reshape(-1).to(device).realize()
    if typ == 8:
      ret._q8_0_weight = raw
      ret.weight = ggml_data_to_tensor(raw.bitcast('uint8'), ret.in_features*ret.out_features, 8).reshape(
        ret.out_features, ret.in_features).cast(layer.weight.dtype)
    else:
      ret.ggml_type, ret.weight = typ, raw
  else:
    w = layer.weight.reshape(ne, no, ni)
    if rows is not None: w = Tensor.cat(*(w[:, s:e] for s,e in rows), dim=1)
    if cols is not None: w = Tensor.cat(*(w[:, :, s:e] for s,e in cols), dim=2)
    ret.weight = w.reshape(*((ne,) if isinstance(layer, ExpertWeights) else ()), ret.out_features, ret.in_features).to(device).contiguous().realize()
  if isinstance(layer, Linear) and layer.bias is not None:
    bias = layer.bias if rows is None else Tensor.cat(*(layer.bias[s:e] for s,e in rows))
    ret.bias = (bias if cols is None else bias * (ret.in_features / ni)).to(device).contiguous().realize()
  return ret


def _part(size, rank, count):
  assert size % count == 0, f'{size} is not divisible by {count}'
  return size*rank//count, size*(rank+1)//count


def _sum(xs):
  xs = [x.contiguous() for x in xs]
  return [sum((x.to(y.device) for x in xs[1:]), xs[0].to(y.device)) for y in xs]


class ShardedBlock(FFNBlock):
  def __init__(self, block, devices):
    assert not any(hasattr(block, k) for k in ('cache_kv', 'conv_state')), 'shard before initializing caches'
    assert type(block) in (TransformerBlock, GatedDeltaNetBlock), 'tensor parallel MLA is not supported'
    assert not block.config.ssm or not block.config.ssm.kda, 'tensor parallel KDA is not supported'
    self.devices, self.blocks = devices, []
    n = len(devices)
    for rank, device in enumerate(devices):
      local = copy.copy(block)
      local.config = replace(block.config, hidden_dim=block.config.hidden_dim//n, shared_expert_dim=block.config.shared_expert_dim//n)
      rows, cols = {}, {}
      for name in ('ffn_gate_exps', 'ffn_up_exps', 'ffn_gate', 'ffn_up', 'ffn_gate_shexp', 'ffn_up_shexp'):
        if hasattr(block, name): rows[name] = [_part(getattr(block, name).out_features, rank, n)]
      for name in ('ffn_down_exps', 'ffn_down', 'ffn_down_shexp'):
        if hasattr(block, name): cols[name] = _part(getattr(block, name).in_features, rank, n)
      if isinstance(block, TransformerBlock):
        assert block.config.qk_norm in (0, block.config.head_dim), 'tensor parallel requires per-head Q/K normalization'
        _part(block.config.n_heads, rank, n)
        _part(block.config.n_kv_heads, rank, n)
        local.config = replace(local.config, n_heads=block.config.n_heads//n, n_kv_heads=block.config.n_kv_heads//n)
        for name in ('attn_q', 'attn_k', 'attn_v'): rows[name] = [_part(getattr(block, name).out_features, rank, n)]
        cols['attn_output'] = _part(block.attn_output.in_features, rank, n)
      else:
        # Keep every repeated V-head group with its Q/K heads, avoiding duplicated Q/K projections.
        ks, ke = _part(block.num_k_heads, rank, n)
        heads = [(g*block.num_k_heads+ks, g*block.num_k_heads+ke) for g in range(block.num_v_heads//block.num_k_heads)]
        vr = [(s*block.head_v_dim,e*block.head_v_dim) for s,e in heads]
        qr = (ks*block.head_k_dim,ke*block.head_k_dim)
        rows['attn_qkv'] = [qr, (qr[0]+block.q_dim,qr[1]+block.q_dim)] + [(s+2*block.q_dim,e+2*block.q_dim) for s,e in vr]
        for name in ('ssm_alpha', 'ssm_beta'): rows[name] = heads
        rows['attn_gate'], cols['ssm_out'] = vr, vr
        local.num_k_heads, local.num_v_heads = ke-ks, block.num_v_heads//n
        local.q_dim = (ke-ks)*block.head_k_dim
        local.conv_channels = block.conv_channels//n
      for name, value in vars(block).items():
        if name == 'config': continue
        if isinstance(value, (Linear, ExpertWeights)): setattr(local, name, _linear(value, device, rows.get(name), cols.get(name)))
        elif isinstance(value, (Tensor, dict)) or hasattr(value, 'weight'): setattr(local, name, _replicate(value, device))
      if isinstance(block, GatedDeltaNetBlock):
        local.ssm_conv1d = {'weight':Tensor.cat(*(block.ssm_conv1d['weight'][s:e] for s,e in rows['attn_qkv'])).to(device).contiguous().realize()}
        local.ssm_dt = {'bias':Tensor.cat(*(block.ssm_dt['bias'][s:e] for s,e in heads)).to(device).contiguous().realize()}
        local.ssm_a = Tensor.cat(*(block.ssm_a[s:e] for s,e in heads)).to(device).contiguous().realize()
      self.blocks.append(local)

  def _reusable_prefix_len(self, prefix_len, cached_len):
    return min(b._reusable_prefix_len(prefix_len, cached_len) for b in self.blocks)

  def __call__(self, x, start_pos):
    xs = [Tensor(x.uop.mselect(i)) if isinstance(x.device, tuple) else x.to(d) for i,d in enumerate(self.devices)]
    for b,t in zip(self.blocks, xs): b._init_state(t)
    @function(precompile=True, allow_implicit=True)
    def run(*xs):
      attn = [b._attention(b.attn_norm(t), start_pos) for b,t in zip(self.blocks, xs)]
      hs = [t+a for t,a in zip(xs, _sum(attn))]
      ffn = [b._feed_forward(b.ffn_norm(h)) for b,h in zip(self.blocks, hs)]
      return tuple((h+f).contiguous() for h,f in zip(hs, _sum(ffn)))
    return Tensor(UOp.mstack(*(t.uop for t in run(*xs))))


class ShardedOutput(Linear):
  def __init__(self, layer, devices):
    self.layers = [_linear(layer, d, [_part(layer.out_features, i, len(devices))]) for i,d in enumerate(devices)]

  def __call__(self, x):
    x = x.contiguous()
    return Tensor(UOp.mstack(*(layer(Tensor(x.uop.mselect(i))).uop for i,layer in enumerate(self.layers))).unshard(len(x.shape)-1))


def shard_model(model:Transformer, devices:tuple[str, ...]) -> Transformer:
  """Partition heads/FFN channels across devices, retaining packed weights and per-device caches. Call before warmup."""
  devices = tuple(Device.canonicalize(d) for d in devices)
  assert len(devices) > 1 and len(set(devices)) == len(devices)
  assert not model._cached_tokens, 'shard before generating tokens'
  model.blk = [ShardedBlock(b, devices) for b in model.blk]
  model.output = ShardedOutput(model.output, devices)
  assert model.output_norm.weight is not None
  model.output_norm.weight.shard_(devices)
  model.token_embd.weight.to_(devices[0]).realize()
  return model
