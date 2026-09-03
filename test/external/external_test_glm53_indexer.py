#!/usr/bin/env python3
"""Real-weight GLM-5.3 Flash indexer parity test.

This intentionally lives outside the default pytest collection. It selectively
loads one indexer layer from a split GGUF and compares the first genuinely sparse
query against an independent NumPy implementation.
"""
from __future__ import annotations

import argparse, pathlib, time
import numpy as np

from tinygrad import Device, Tensor, dtypes, nn
from tinygrad.llm.gguf import _gguf_info, _gguf_split_paths, _gguf_tensors
from tinygrad.llm.model import AttentionIndexer, IndexerConfig, TransformerConfig


def load_indexer_weights(path:pathlib.Path, layer:int, device:str) -> tuple[dict, dict[str, Tensor]]:
  first = Tensor(path)
  kv, _, _ = _gguf_info(first)
  prefix = f"blk.{layer}."
  wanted = {
    prefix + "indexer.attn_k.weight",
    prefix + "indexer.attn_q_b.weight",
    prefix + "indexer.k_norm.bias",
    prefix + "indexer.k_norm.weight",
    prefix + "indexer.proj.weight",
    prefix + "indexer_compressor_ape.weight",
    prefix + "indexer_compressor_gate.weight",
  }
  state:dict[str, Tensor] = {}
  for split_path in _gguf_split_paths(path, kv):
    disk = Tensor(split_path)
    _, data_start, infos = _gguf_info(disk)
    selected = [info for info in infos if info[0] in wanted]
    if selected: state.update(_gguf_tensors(disk, data_start, selected, {info[0]:device for info in selected}))
  missing = wanted - state.keys()
  if missing: raise RuntimeError(f"missing real indexer tensors: {sorted(missing)}")
  renamed = {}
  for name, value in state.items():
    suffix = name.removeprefix(prefix)
    suffix = suffix.removeprefix("indexer.")
    suffix = suffix.replace("indexer_compressor_ape.weight", "compressor_ape")
    suffix = suffix.replace("indexer_compressor_gate.weight", "compressor_gate.weight")
    renamed[suffix] = value.cast(dtypes.half).contiguous()
  return kv, renamed


def selection_reference(keys:np.ndarray, gates:np.ndarray, q:np.ndarray, head_weights:np.ndarray, ape:np.ndarray,
                        ic:IndexerConfig, position:int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
  # Match the real fp16 cache boundary before KPool compression.
  keys, gates = keys.astype(np.float16), gates.astype(np.float16)
  token_count = len(keys)
  padded_len = ((token_count + ic.kpool - 1) // ic.kpool) * ic.kpool
  keys = np.pad(keys, ((0, padded_len-token_count), (0, 0))).reshape(-1, ic.kpool, ic.head_dim)
  gates = np.pad(gates, ((0, padded_len-token_count), (0, 0))).reshape(-1, ic.kpool, ic.head_dim)
  valid = np.arange(padded_len).reshape(-1, ic.kpool) < token_count
  logits = np.where(valid[..., None], gates.astype(np.float32) + ape.astype(np.float32)[None], -1e30)
  probabilities = np.exp(logits - logits.max(1, keepdims=True))
  probabilities = (probabilities / probabilities.sum(1, keepdims=True)).astype(np.float16)
  products = probabilities * keys * valid[..., None]
  pool_keys = products.sum(1, dtype=np.float32)

  scores = np.maximum(0, np.einsum("hd,pd->hp", q, pool_keys, dtype=np.float32) * ic.head_dim**-0.5)
  index_scores = np.einsum("h,hp->p", head_weights, scores, dtype=np.float32)
  pool_ends = np.arange(len(pool_keys)) * ic.kpool + (ic.kpool-1)
  candidates = valid.all(1) & (pool_ends <= position)
  index_scores = np.where(candidates, index_scores, np.finfo(np.float32).min)

  select_k = ic.top_k // ic.kpool
  selected = np.argsort(-index_scores, kind="stable")[:select_k]
  selected_tokens = (selected[:, None] * ic.kpool + np.arange(ic.kpool)).reshape(-1)
  tail_count = (position+1) % ic.kpool
  tail = np.full(ic.kpool-1, -1, dtype=np.int32)
  if tail_count: tail[:tail_count] = np.arange(position+1-tail_count, position+1)
  expected = np.concatenate((selected_tokens, tail)).astype(np.int32)
  finite = np.sort(index_scores[candidates])[::-1]
  cutoff_gap = float(finite[select_k-1] - finite[select_k])
  return expected, selected.astype(np.int32), index_scores, cutoff_gap


def numpy_reference(hidden:np.ndarray, q_resid:np.ndarray, weights:dict[str, np.ndarray], ic:IndexerConfig,
                    position:int) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, float], tuple[np.ndarray, ...]]:
  # All model parameters were cast to fp16 by the GGUF model loader. Matmuls accumulate in fp32.
  w = {name:value.astype(np.float32) for name, value in weights.items()}
  keys = hidden @ w["attn_k.weight"].T
  keys = (keys - keys.mean(-1, keepdims=True)) / np.sqrt(((keys-keys.mean(-1, keepdims=True))**2).mean(-1, keepdims=True) + 1e-6)
  keys = keys * w["k_norm.weight"] + w["k_norm.bias"]
  gates = hidden @ w["compressor_gate.weight"].T
  q = (q_resid @ w["attn_q_b.weight"].T).reshape(ic.n_heads, ic.head_dim)
  head_weights = (hidden[position] @ w["proj.weight"].T) * ic.n_heads**-0.5
  return selection_reference(keys, gates, q, head_weights, w["compressor_ape"], ic, position), (keys, gates, q, head_weights)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("gguf", type=pathlib.Path, help="first file of the split GGUF")
  parser.add_argument("--layer", type=int, default=3)
  parser.add_argument("--device", default=Device.DEFAULT)
  parser.add_argument("--position", type=int, help="query position (default: first position that drops a complete pool)")
  args = parser.parse_args()
  started = time.perf_counter()

  kv, loaded = load_indexer_weights(args.gguf, args.layer, args.device)
  arch = kv["general.architecture"]
  ic = IndexerConfig(top_k=kv[f"{arch}.attention.indexer.top_k"], head_dim=kv[f"{arch}.attention.indexer.key_length"],
                     n_heads=kv[f"{arch}.attention.indexer.head_count"], kpool=kv[f"{arch}.attention.indexer.kpool"])
  dim, q_lora_rank = loaded["attn_k.weight"].shape[1], loaded["attn_q_b.weight"].shape[1]
  position = args.position if args.position is not None else ic.top_k + ic.kpool
  candidate_count = (position+1) // ic.kpool
  if candidate_count <= ic.top_k // ic.kpool: raise ValueError("position must produce more complete pools than the indexer retains")
  config = TransformerConfig(num_blocks=1, dim=dim, hidden_dim=1, n_heads=1, n_kv_heads=1, norm_eps=1e-5,
    vocab_size=1, head_dim=1, rope_theta=1.0, rope_dim=0, v_head_dim=1, max_context=position+1,
    q_lora_rank=q_lora_rank, indexer=ic)
  indexer = AttentionIndexer(config)
  params = nn.state.get_state_dict(indexer)
  for name, value in loaded.items(): params[name].replace(value)
  Tensor.realize(*params.values())
  weights = {name:value.numpy() for name, value in params.items()}
  print(f"loaded real blk.{args.layer} weights: dim={dim}, q_lora_rank={q_lora_rank}, heads={ic.n_heads}, "
        f"head_dim={ic.head_dim}, kpool={ic.kpool}, top_k={ic.top_k}")

  rng = np.random.default_rng(53)
  hidden = (rng.standard_normal((position+1, dim), dtype=np.float32) * 0.1).astype(np.float32)
  q_resid = (rng.standard_normal(q_lora_rank, dtype=np.float32) * 0.1).astype(np.float32)

  # Fill history using the real projections, then run the sparse query.
  hidden_tensor = Tensor(hidden[None], device=args.device)
  projected_keys = indexer.k_norm(indexer.attn_k(hidden_tensor)).realize()
  projected_gates = indexer.compressor_gate(hidden_tensor).realize()
  projected_q = indexer.attn_q_b(Tensor(q_resid[None, None], device=args.device)).reshape(ic.n_heads, ic.head_dim).realize()
  projected_head_weights = (indexer.proj(hidden_tensor[:, position:position+1]) * ic.n_heads**-0.5).realize()
  packed = projected_keys[:, :position].cat(projected_gates[:, :position],
    Tensor.ones(1, position, 1, device=args.device), dim=-1).cast(dtypes.half)
  current = hidden_tensor[:, position:position+1]
  indexer._init_state(current)
  indexer.cache[:, :position].assign(packed).realize()
  actual = indexer(current, Tensor(q_resid[None, None], device=args.device), position).realize().numpy()[0, 0]

  (expected, expected_pools, scores, cutoff_gap), numpy_projected = numpy_reference(hidden, q_resid, weights, ic, position)
  device_projected = (projected_keys.numpy()[0], projected_gates.numpy()[0], projected_q.numpy(), projected_head_weights.numpy()[0, 0])
  device_expected, device_pools, _, device_cutoff_gap = selection_reference(*device_projected, weights["compressor_ape"], ic, position)
  actual_rows = actual[:ic.top_k].reshape(-1, ic.kpool)
  if not np.all(actual_rows == actual_rows[:, :1] + np.arange(ic.kpool)):
    raise AssertionError("selected token indices do not expand to complete contiguous KPool rows")
  actual_pools = actual_rows[:, 0] // ic.kpool
  np.testing.assert_array_equal(np.sort(actual_pools), np.sort(device_pools),
                                err_msg="selected pool set differs from independent reference given identical projections")
  np.testing.assert_array_equal(actual[ic.top_k:], device_expected[ic.top_k:], err_msg="incomplete current-pool tail differs")
  exact_order = np.array_equal(actual, device_expected)
  reference_overlap = len(set(actual_pools.tolist()) & set(expected_pools.tolist()))
  projection_errors = [float(np.max(np.abs(a.astype(np.float32)-b.astype(np.float32))))
                       for a,b in zip(device_projected, numpy_projected)]
  dropped_actual = sorted(set(range(candidate_count)) - set(actual_pools.tolist()))
  dropped_expected = sorted(set(range(candidate_count)) - set(device_pools.tolist()))
  if dropped_actual != dropped_expected: raise AssertionError(f"wrong dropped pool: actual={dropped_actual}, expected={dropped_expected}")

  print(f"PASS position={position}: candidates={candidate_count} selected_pools={len(actual_pools)} selected_tokens={ic.top_k}")
  print(f"PASS dropped_pools={len(dropped_actual)} first_dropped={dropped_actual[0]} tail={actual[ic.top_k:].tolist()} "
        f"exact_reference_order={exact_order}")
  print(f"full NumPy overlap={reference_overlap}/{len(actual_pools)} projection max_abs_error={projection_errors}")
  print(f"cutoff gaps: device_projection={device_cutoff_gap:.8g} full_numpy={cutoff_gap:.8g}, elapsed={time.perf_counter()-started:.2f}s")


if __name__ == "__main__": main()
