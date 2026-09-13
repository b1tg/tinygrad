"""Tensor parallel LLM blocks. Packed GGML blocks are partitioned before dequantization."""
import copy, dataclasses, functools
from typing import cast
from dataclasses import replace
from tinygrad import Tensor, UOp, function, Device, dtypes, nn
from tinygrad.device import Buffer
from tinygrad.llm.gguf import ggml_data_to_tensor
from tinygrad.llm.kernels.amd import Linear, ExpertWeights, packed_q8_0_weight, packed_ggml_weight, amd_custom_kernels_supported
from tinygrad.llm.model import Transformer, FFNBlock, TransformerBlock, GatedDeltaNetBlock, MLATransformerBlock


def _from_disk(t:Tensor) -> Tensor:
  # streamed GGUF tensors live on the DISK device; realize the (small) per-tensor slice on CPU before decoding
  if isinstance(t.device, str) and t.device.startswith('DISK'): return t.to('CPU').realize()
  return t


def _stage(t:Tensor, pending:list[Tensor]|None=None) -> Tensor:
  if pending is None: return t.realize()
  pending.append(t)
  return t


def _replicate(obj, device, pending:list[Tensor]|None=None):
  if isinstance(obj, Tensor): return _stage(_from_disk(obj).to(device).contiguous(), pending)
  if isinstance(obj, dict): return {k:_replicate(v, device, pending) for k,v in obj.items()}
  if dataclasses.is_dataclass(obj) and not isinstance(obj, type): return obj  # frozen configs are shared
  if hasattr(obj, '__dict__'):
    ret = copy.copy(obj)
    for k,v in vars(obj).items(): setattr(ret, k, _replicate(v, device, pending))
    return ret
  return obj


def _on_disk(t:Tensor) -> bool:
  return isinstance(t.device, str) and t.device.startswith('DISK')


def _packed_of(layer, ne:int, ni:int, no:int, supported:bool) -> tuple[Tensor|None, tuple[int, Tensor]|None]:
  # recovering the packed backing from the decoded weight walks (and rebuilds) the whole decode graph, so do it once
  # per layer and share the result across ranks
  cache = layer.__dict__.setdefault('_packed_cache', {})
  key = (supported, ne*ni*no)
  if key not in cache:
    cache[key] = (packed_q8_0_weight(layer.weight, ne*ni*no) if supported and isinstance(layer, Linear) else None,
                  packed_ggml_weight(layer.weight, ne*ni*no) if supported and ni % 256 == 0 else None)
  return cache[key]


def _cached_cpu(full:Tensor, layer) -> Tensor:
  # a packed tensor sliced on in_features cannot be read as contiguous disk spans, so materialize it once on CPU and
  # share that copy across ranks instead of re-reading the whole tensor from disk for every device.
  cache = layer.__dict__.setdefault('_disk_cpu', {})
  if 'full' not in cache: cache['full'] = full.to('CPU').realize()
  return cache['full']


def _packed_rows(raw:Tensor, ne:int, no:int, rows:list[tuple[int,int]]|None, device, pending:list[Tensor]|None=None) -> Tensor:
  """Slice the output axis of a packed tensor, returning `(ne, nrows, row_elems)` on `device`.

  On DISK, only this rank's byte spans are read (each output row is a contiguous run of quant blocks), so the n
  ranks together read the packed tensor exactly once instead of materializing the whole thing on every rank."""
  row_elems = raw.numel() // ne // no
  sel = rows or [(0, no)]
  if _on_disk(raw):
    row_bytes = row_elems * raw.dtype.itemsize
    buf = cast(Buffer, raw.uop.buffer)
    buf.get_buf(buf.device)
    mv = buf.as_memoryview(allow_zero_copy=True)
    chunks = [bytes(mv[(e*no+s)*row_bytes:(e*no+en)*row_bytes]) for e in range(ne) for s, en in sel]
    nrows = sum(en-s for s, en in sel)
    # stage on PYTHON so each shard makes a single H2D copy straight to its own device (not through the default one)
    return Tensor(b''.join(chunks), dtype=dtypes.uint8, device='PYTHON').to(device).contiguous().bitcast(raw.dtype).reshape(ne, nrows, row_elems)
  full = raw.reshape(ne, no, row_elems)
  w = full[:, sel[0][0]:sel[0][1]] if len(sel) == 1 else Tensor.cat(*(full[:, s:e] for s, e in sel), dim=1)
  return _stage(w.contiguous().to(device), pending)


def _linear(layer, device, rows=None, cols=None, pending:list[Tensor]|None=None):
  if cols is not None and isinstance(cols[0], int): cols = [cols]
  ret = copy.copy(layer)
  # Loading caches belong to the source layer, not to its persistent GPU shards.
  for name in ('_packed_cache', '_disk_cpu'): ret.__dict__.pop(name, None)
  ret._fused_q8_0_weight = None
  ret._q8_0_tried = True
  ret._q8_0_weight = None
  ni, no = layer.in_features, layer.out_features
  ne = layer.num_experts if isinstance(layer, ExpertWeights) else 1
  ret.in_features = ni if cols is None else sum(e-s for s,e in cols)
  ret.out_features = no if rows is None else sum(e-s for s,e in rows)
  supported = amd_custom_kernels_supported(device)
  q8, packed = _packed_of(layer, ne, ni, no, supported)
  # a sharded in_features split must keep whole quantization blocks, else fall back to the unpacked fp16 path
  if cols is not None:
    if q8 is not None and any(c % 32 for r in cols for c in r): q8 = None
    if packed is not None and any(c % 256 for r in cols for c in r): packed = None
  if q8 is not None or packed is not None:
    if q8 is not None: typ, raw, block = 8, q8, 32
    else:
      assert packed is not None
      typ, raw, block = *packed, 256
    if cols is None:
      # output-axis split: every output row is a contiguous run of quant blocks, so read only this rank's spans
      raw = _stage(_packed_rows(raw, ne, no, rows, device, pending).reshape(-1), pending)
    else:
      # in_features split cannot be expressed as contiguous disk spans. A strided slice of a DISK tensor also lowers
      # to a kernel the DISK device can't render, so materialize the packed tensor on CPU once (shared across ranks)
      # and slice it there.
      full = raw.reshape(ne, no, ni//block, -1)
      if _on_disk(full): full = _cached_cpu(full, layer)
      if rows is not None: w = (lambda s,e: full[:, s:e])(*rows[0]) if len(rows) == 1 else Tensor.cat(*(full[:, s:e] for s,e in rows), dim=1)
      else: w = full
      w = (lambda s,e: w[:, :, s//block:e//block])(*cols[0]) if len(cols) == 1 else \
        Tensor.cat(*(w[:, :, s//block:e//block] for s,e in cols), dim=2)
      raw = _stage(w.contiguous().reshape(-1).to(device), pending)
    if typ == 8:
      ret._q8_0_weight = raw
      ret.weight = ggml_data_to_tensor(raw.bitcast('uint8'), ret.in_features*ret.out_features, 8).reshape(
        ret.out_features, ret.in_features).cast(layer.weight.dtype)
    else:
      ret.ggml_type, ret.weight = typ, raw
  else:
    w = _from_disk(layer.weight).reshape(ne, no, ni)
    if rows is not None: w = Tensor.cat(*(w[:, s:e] for s,e in rows), dim=1)
    if cols is not None: w = Tensor.cat(*(w[:, :, s:e] for s,e in cols), dim=2)
    ret.ggml_type = None
    ret.weight = _stage(w.reshape(*((ne,) if isinstance(layer, ExpertWeights) else ()),
                                  ret.out_features, ret.in_features).to(device).contiguous(), pending)
  if isinstance(layer, Linear) and layer.bias is not None:
    bias = _from_disk(layer.bias) if rows is None else Tensor.cat(*(layer.bias[s:e] for s,e in rows))
    ret.bias = _stage((bias if cols is None else bias * (ret.in_features / ni)).to(device).contiguous(), pending)
  return ret


def _part(size, rank, count, gran=1):
  # uneven contiguous split into `count` chunks, each rounded to a multiple of `gran` (needed to keep packed
  # quant blocks whole when slicing in_features). The first `rem` ranks get one extra granule.
  assert size % gran == 0, f'{size} is not divisible by {gran}'
  granules, rem = divmod(size // gran, count)
  start = rank*granules + min(rank, rem)
  end = start + granules + (1 if rank < rem else 0)
  return start*gran, end*gran


def _hidden_part(size, rank, n):
  # split a hidden/feature axis so every local slice is a whole 256-weight quant block when possible,
  # but only if there are enough blocks to give every rank a non-empty slice.
  # NOTE: this forces the fullest rank to hold ceil(size/256/n)*256/`size` of the axis. For GLM-5.3's
  # hidden=2048 on 5 ranks that is 512/2048 = 25% of every IQ2_XS/IQ3_XXS expert matrix (~24.75 GB of
  # ~99 GB), which does not fit a 24 GB card. That is why GLM uses `gather_ffn=True`, which shards the
  # experts on their (block-free) output axis and all-gathers the hidden activation before the down
  # projection, giving an even ~20 GB per rank.
  return _part(size, rank, n, 256 if size % 256 == 0 and size // 256 >= n else 1)


def _fused_gate_up_exps(gate:ExpertWeights, up:ExpertWeights, device, gs:int, ge:int, pending:list[Tensor]|None=None) -> ExpertWeights|None:
  """Interleave the gate/up packed weights per expert so one mul_mat_id call covers both (bigger grid, one launch)."""
  if gate.in_features != up.in_features or gate.out_features != up.out_features: return None
  if gate.in_features % 256: return None
  E, ni, no = gate.num_experts, gate.in_features, gate.out_features
  pg, pu = _packed_of(gate, E, ni, no, True)[1], _packed_of(up, E, ni, no, True)[1]
  if pg is None or pu is None or pg[0] != pu[0]: return None
  ggml_type, raw_g, raw_u = pg[0], pg[1], pu[1]
  # output-axis split: read only this rank's byte spans from DISK (the ranks together cover each tensor once)
  wg, wu = _packed_rows(raw_g, E, no, [(gs, ge)], device, pending), _packed_rows(raw_u, E, no, [(gs, ge)], device, pending)
  fused = ExpertWeights(E, ni, ge-gs + ge-gs)
  fused.ggml_type = ggml_type
  fused.weight = _stage(Tensor.cat(wg, wu, dim=1).contiguous().reshape(-1), pending)
  return fused


class ShardedBlock(FFNBlock):
  def __init__(self, block, devices, gather_ffn:bool=False):
    assert not any(hasattr(block, k) for k in ('cache_kv', 'cache_k', 'conv_state')), 'shard before initializing caches'
    assert type(block) in (TransformerBlock, GatedDeltaNetBlock, MLATransformerBlock), 'unsupported block for tensor parallel'
    self.devices, self.blocks, self.gather_ffn = devices, [], gather_ffn
    n = len(devices)
    # `gather_ffn` shards the MLP/experts on their *output* axis (always a multiple of 1 quant block-free), then
    # all-gathers the hidden activation before the down projection. This keeps the 256-element quant blocks intact
    # while balancing the weights evenly (a hidden-axis split is forced to 256-aligned and overloads one rank).
    for rank, device in enumerate(devices):
      pending: list[Tensor] = []
      local = copy.copy(block)
      hidden_part = _part if gather_ffn else _hidden_part
      lh0, lh1 = hidden_part(block.config.hidden_dim, rank, n) if block.config.hidden_dim else (0, 0)
      local.config = replace(block.config, hidden_dim=lh1-lh0)
      if block.config.shared_expert_dim:
        ls0, ls1 = hidden_part(block.config.shared_expert_dim, rank, n)
        local.config = replace(local.config, shared_expert_dim=ls1-ls0)
      rows, cols = {}, {}
      for name in ('ffn_gate_exps', 'ffn_up_exps', 'ffn_gate', 'ffn_up', 'ffn_gate_shexp', 'ffn_up_shexp'):
        if hasattr(block, name): rows[name] = [hidden_part(getattr(block, name).out_features, rank, n)]
      for name in ('ffn_down_exps', 'ffn_down', 'ffn_down_shexp'):
        if not hasattr(block, name): continue
        if gather_ffn: rows[name] = [_part(getattr(block, name).out_features, rank, n)]
        else: cols[name] = _hidden_part(getattr(block, name).in_features, rank, n)
      if isinstance(block, TransformerBlock):
        assert block.config.qk_norm in (0, block.config.head_dim), 'tensor parallel requires per-head Q/K normalization'
        hs, he = _part(block.config.n_heads, rank, n)
        khs, khe = _part(block.config.n_kv_heads, rank, n)
        local.config = replace(local.config, n_heads=he-hs, n_kv_heads=khe-khs)
        q_mul = 2 if block.config.attn_output_gate else 1
        rows['attn_q'] = [(hs*block.config.head_dim*q_mul, he*block.config.head_dim*q_mul)]
        for name in ('attn_k', 'attn_v'): rows[name] = [(khs*block.config.head_dim, khe*block.config.head_dim)]
        cols['attn_output'] = (hs*block.config.head_dim, he*block.config.head_dim)
      elif isinstance(block, MLATransformerBlock):
        # MLA: heads are independent, but the KV latent (n_kv_heads=1) is shared across all heads and must stay
        # replicated. Shard Q and the per-head K_B/V_B by head, and attn_output by head on its input side.
        hs, he = _part(block.config.n_heads, rank, n)
        local.config = replace(local.config, n_heads=he-hs)
        q_name = 'attn_q_b' if block.config.q_lora_rank > 0 else 'attn_q'
        rows[q_name] = [(hs*block.config.head_dim, he*block.config.head_dim)]
        cols['attn_output'] = (hs*block.config.v_head_dim, he*block.config.v_head_dim)
      else:
        # Keep every repeated V-head group with its Q/K heads, avoiding duplicated Q/K projections.
        ks, ke = _part(block.num_k_heads, rank, n)
        ratio = block.num_v_heads // block.num_k_heads
        heads = [(g*block.num_k_heads+ks, g*block.num_k_heads+ke) for g in range(ratio)]
        vr = [(s*block.head_v_dim,e*block.head_v_dim) for s,e in heads]
        qr = (ks*block.head_k_dim,ke*block.head_k_dim)
        qkv_rows = [qr, (qr[0]+block.q_dim,qr[1]+block.q_dim)] + [(s+2*block.q_dim,e+2*block.q_dim) for s,e in vr]
        if hasattr(block, 'attn_q'):
          rows['attn_q'], rows['attn_k'], rows['attn_v'] = [qr], [qr], vr
        else: rows['attn_qkv'] = qkv_rows
        for name in ('ssm_alpha', 'ssm_beta'): rows[name] = heads
        rows['attn_gate'], cols['ssm_out'] = vr, vr
        if hasattr(block, 'ssm_g_a'):
          rows['ssm_g_b'], rows['ssm_f_b'] = vr, vr
        local.num_k_heads = ke-ks
        local.num_v_heads = (ke-ks)*ratio
        local.q_dim = (ke-ks)*block.head_k_dim
        local.conv_channels = local.num_v_heads*block.head_v_dim + 2*local.q_dim
      fused = None
      if hasattr(block, 'ffn_gate_exps'):
        gs, ge = rows['ffn_gate_exps'][0]
        fused = _fused_gate_up_exps(block.ffn_gate_exps, block.ffn_up_exps, device, gs, ge, pending)
      if fused is not None:
        local.ffn_gateup_exps = fused
        del local.ffn_gate_exps, local.ffn_up_exps
      for name, value in vars(block).items():
        if fused is not None and name in ('ffn_gate_exps', 'ffn_up_exps'): continue
        if name == 'config': continue
        # These weights are sliced below; do not first upload a full replica that will immediately be discarded.
        if isinstance(block, MLATransformerBlock) and name in ('attn_k_b', 'attn_v_b'): continue
        if isinstance(block, GatedDeltaNetBlock) and name in ('ssm_conv1d', 'ssm_dt', 'ssm_a'): continue
        if isinstance(value, (Linear, ExpertWeights)): setattr(local, name, _linear(value, device, rows.get(name), cols.get(name), pending))
        elif name in ('indexer', 'hc_attn', 'hc_ffn'): setattr(local, name, _replicate(value, device, pending))
        elif isinstance(value, (Tensor, dict)) or hasattr(value, 'weight'): setattr(local, name, _replicate(value, device, pending))
      if isinstance(block, MLATransformerBlock):
        # per-head MLA weights (plain tensors, not Linear): shard the leading head axis
        hs, he = _part(block.config.n_heads, rank, n)
        local.attn_k_b = {"weight": _stage(block.attn_k_b["weight"][hs:he].to(device).contiguous(), pending)}
        local.attn_v_b = {"weight": _stage(block.attn_v_b["weight"][hs:he].to(device).contiguous(), pending)}
      if isinstance(block, GatedDeltaNetBlock):
        local.ssm_conv1d = {'weight':_stage(Tensor.cat(*(block.ssm_conv1d['weight'][s:e] for s,e in qkv_rows)).to(device).contiguous(), pending)}
        dt_rows = vr if hasattr(block, 'ssm_g_a') else heads
        local.ssm_dt = {'bias':_stage(Tensor.cat(*(block.ssm_dt['bias'][s:e] for s,e in dt_rows)).to(device).contiguous(), pending)}
        local.ssm_a = _stage(Tensor.cat(*(block.ssm_a[s:e] for s,e in heads)).to(device).contiguous(), pending)
      # Realize one rank of one block together: each realization otherwise walks every live tensor in the model.
      # This also bounds temporary packed gate/up buffers to one rank instead of retaining them for the entire model.
      if pending: pending[0].realize(*pending[1:])
      self.blocks.append(local)
    # the load caches (whole CPU copies of in_features-split tensors, recovered packed views) are only needed while
    # producing the shards; drop them so they don't outlive the model load
    for value in vars(block).values():
      if isinstance(value, (Linear, ExpertWeights)):
        value.__dict__.pop('_disk_cpu', None)
        value.__dict__.pop('_packed_cache', None)

  def _reusable_prefix_len(self, prefix_len, cached_len):
    return min(b._reusable_prefix_len(prefix_len, cached_len) for b in self.blocks)

  def __call__(self, x, start_pos):
    xs = [Tensor(x.uop.mselect(i)) if isinstance(x.device, tuple) else x.to(d) for i,d in enumerate(self.devices)]
    for b,t in zip(self.blocks, xs): b._init_state(t)
    def down_local(acts, residuals, xs):
      # gather mode: gate/up are sharded on the hidden axis, so all-gather the activation before down. Without
      # gather the activation is already full and down is sharded on in_features, giving full-dim partial sums.
      if self.gather_ffn:
        gathered = [dict(a) for a in acts]
        for key in FFNBlock._FFN_HIDDEN_KEYS:
          if key not in acts[0]: continue
          for i, t in enumerate(xs):
            gathered[i][key] = Tensor.cat(*(a[key].to(t.device) for a in acts), dim=-1)
        acts = gathered
      return [b._ffn_down(residuals[i], acts[i]).contiguous() for i, b in enumerate(self.blocks)]

    def combine(ffn, xs):
      # gather mode: down is sharded on out_features, so all-gather the dim slices instead of summing partials
      if self.gather_ffn: return [Tensor.cat(*(f.to(t.device) for f in ffn), dim=-1).contiguous() for t in xs]
      return [functools.reduce(lambda acc, f: acc + f.to(t.device), ffn[1:], ffn[0].to(t.device)) for t in xs]

    @function(precompile=True, allow_implicit=True)
    def run(*xs):
      attn = [b._attention(b.attn_norm(t), start_pos).contiguous() for b,t in zip(self.blocks, xs)]
      # residual add fused into the all-reduce: one elementwise kernel instead of sum + add
      hs = [functools.reduce(lambda acc, a: acc + a.to(t.device), attn, t) for t in xs]
      ffn_in = [b.ffn_norm(h) for b,h in zip(self.blocks, hs)]
      ffn_act = [b._ffn_gate_up(x) for b,x in zip(self.blocks, ffn_in)]
      ffn_full = combine(down_local(ffn_act, ffn_in, xs), xs)
      return tuple((h + f).contiguous() for h, f in zip(hs, ffn_full))

    @function(precompile=True, allow_implicit=True)
    def run_hc(*xs):
      # hyper-connections keep a (B,T,hc,D) residual stream that is replicated on every rank. Only the
      # head-split attention/FFN outputs need an all-reduce before each mix.
      attn, residual, posts, combs = [], [], [], []
      for b, t in zip(self.blocks, xs):
        residual.append(t)
        h, post, comb = b.hc_attn.prepare(t)
        attn.append(b._attention(b.attn_norm(h), start_pos).contiguous())
        posts.append(post)
        combs.append(comb)
      ffn_in, ffn_act, residual2, posts2, combs2 = [], [], [], [], []
      for i, (b, t) in enumerate(zip(self.blocks, xs)):
        attn_full = functools.reduce(lambda acc, a: acc + a.to(t.device), attn[1:], attn[0].to(t.device))
        x1 = b.hc_attn.mix(attn_full, residual[i], posts[i], combs[i])
        h2, post2, comb2 = b.hc_ffn.prepare(x1)
        ffn_in.append(b.ffn_norm(h2))
        ffn_act.append(b._ffn_gate_up(ffn_in[-1]))
        residual2.append(x1)
        posts2.append(post2)
        combs2.append(comb2)
      ffn_full = combine(down_local(ffn_act, ffn_in, xs), xs)
      return tuple(self.blocks[i].hc_ffn.mix(ffn_full[i], residual2[i], posts2[i], combs2[i]).contiguous()
                   for i in range(len(xs)))

    out = run_hc(*xs) if getattr(self.blocks[0].config, 'hc_mult', 0) else run(*xs)
    return Tensor(UOp.mstack(*(t.uop for t in out)))


class ShardedEmbedding(nn.Embedding):
  def __init__(self, layer:nn.Embedding, devices:tuple[str, ...]):
    self.weight = layer.weight.shard(devices, axis=0).realize()

  def __call__(self, idx:Tensor) -> Tensor:
    # Vocab-parallel lookup with a symbolic token dimension currently misindexes its reduction. Use a fixed-size
    # lookup buffer, then discard the padded tokens before attention/recurrent state updates.
    shape = (*idx.shape, self.weight.shape[1])
    idx = idx.to(self.weight.device)
    return super().__call__(idx.pad_to(idx.max_shape)).contiguous().shrink(tuple((0, s) for s in shape))


class ShardedOutput(Linear):
  def __init__(self, layer, devices):
    self.layers = [_linear(layer, d, [_part(layer.out_features, i, len(devices))]) for i,d in enumerate(devices)]

  def __call__(self, x):
    x = x.contiguous()
    return Tensor(UOp.mstack(*(layer(Tensor(x.uop.mselect(i))).uop for i,layer in enumerate(self.layers))).unshard(len(x.shape)-1))


def shard_model(model:Transformer, devices:tuple[str, ...], gather_ffn:bool=False) -> Transformer:
  """Partition heads/FFN channels across devices, retaining packed weights and per-device caches. Call before warmup."""
  devices = tuple(Device.canonicalize(d) for d in devices)
  assert len(devices) > 1 and len(set(devices)) == len(devices)
  assert not model._cached_tokens, 'shard before generating tokens'
  # Release each source block as soon as its shards are ready, including its CPU decode graphs and packed views.
  for i in range(len(model.blk)): model.blk[i] = ShardedBlock(model.blk[i], devices, gather_ffn=gather_ffn)
  model.output = ShardedOutput(model.output, devices)
  assert model.output_norm.weight is not None
  model.output_norm.weight = _from_disk(model.output_norm.weight)
  model.output_norm.weight.shard_(devices)
  model.token_embd.weight = _from_disk(model.token_embd.weight)
  # The full embedding table otherwise overloads rank 0 when long-context caches are allocated during warmup.
  if model.token_embd.weight.shape[0] % len(devices) == 0: model.token_embd = ShardedEmbedding(model.token_embd, devices)
  else: model.token_embd.weight.to_(devices[0]).realize()
  return model
