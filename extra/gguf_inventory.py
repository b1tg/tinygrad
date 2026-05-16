#!/usr/bin/env python3
import collections, pathlib, re, struct, sys

NATIVE = {0:("F32",4), 1:("F16",2), 24:("I8",1), 25:("I16",2), 26:("I32",4), 27:("I64",8), 28:("F64",8), 30:("BF16",2)}
QUANT = {2:("Q4_0",32,18), 3:("Q4_1",32,20), 6:("Q5_0",32,22), 7:("Q5_1",32,24), 8:("Q8_0",32,34),
         10:("Q2_K",256,84), 11:("Q3_K",256,110), 12:("Q4_K",256,144), 13:("Q5_K",256,176), 14:("Q6_K",256,210),
         15:("Q8_K",256,292), 18:("IQ3_XXS",256,98), 20:("IQ4_NL",32,18), 21:("IQ3_S",256,110), 22:("IQ2_S",256,82),
         23:("IQ4_XS",256,136), 39:("MXFP4",32,17), 41:("Q1_0",128,18)}

def nbytes(dims, typ):
  n = 1
  for d in dims: n *= d
  return n * NATIVE[typ][1] if typ in NATIVE else (n // QUANT[typ][1]) * QUANT[typ][2]

def tname(typ): return NATIVE.get(typ, QUANT.get(typ, (f"TYPE{typ}",)))[0]

def cat(name):
  if "_exps.weight" in name: return "experts"
  if "shexp" in name: return "shared_expert"
  if "ffn_" in name: return "ffn"
  if "attn" in name: return "attn"
  if name.startswith("blk."): return "blk_other"
  return "other"

def shard_axis(name, ndim, kv):
  arch = kv.get("general.architecture", "")
  attention = not kv.get(f"{arch}.ssm.conv_kernel", 0)
  if name.endswith(".bias"): return 0 if attention and any(x in name for x in (".attn_q.", ".attn_k.", ".attn_v.")) else None
  if ndim <= 1 or "norm" in name or "scale" in name: return None
  if name == "output.weight": return 0
  if attention and ".attn_q.weight" in name: return 0
  if attention and ".attn_q_b.weight" in name: return 0
  if attention and (".attn_k_b.weight" in name or ".attn_v_b.weight" in name): return 0
  if attention and ".attn_output.weight" in name: return -1
  if ".attn_q_a.weight" in name or ".attn_kv_a_mqa.weight" in name: return None
  if ".ffn_gate_inp.weight" in name: return None
  if "_exps.weight" in name: return -1 if ".ffn_down_exps.weight" in name else 1
  if ".ffn_down" in name: return -1
  if ".ffn_gate" in name or ".ffn_up" in name: return 0
  return None

class R:
  def __init__(self, p): self.f = open(p, "rb")
  def rd(self, fmt): return struct.unpack(fmt, self.f.read(struct.calcsize(fmt)))[0]
  def s(self): return self.f.read(self.rd("<Q")).decode("utf-8", "replace")
  def skip(self, typ):
    if typ in (0,1,7): self.f.read(1)
    elif typ in (2,3): self.f.read(2)
    elif typ in (4,5,6): self.f.read(4)
    elif typ in (10,11,12): self.f.read(8)
    elif typ == 8: self.f.read(self.rd("<Q"))
    elif typ == 9:
      et, n = self.rd("<I"), self.rd("<Q")
      for _ in range(n): self.skip(et)
    else: raise ValueError(f"bad kv type {typ}")
  def val(self, typ, key):
    if typ == 0: return self.rd("<B")
    if typ == 1: return self.rd("<b")
    if typ == 2: return self.rd("<H")
    if typ == 3: return self.rd("<h")
    if typ == 4: return self.rd("<I")
    if typ == 5: return self.rd("<i")
    if typ == 6: return self.rd("<f")
    if typ == 7: return bool(self.rd("<?"))
    if typ == 8: return self.f.read(self.rd("<Q")).decode("utf-8", "replace")
    if typ == 10: return self.rd("<Q")
    if typ == 11: return self.rd("<q")
    if typ == 9:
      et, n = self.rd("<I"), self.rd("<Q")
      if key.startswith("tokenizer."):
        for _ in range(n): self.skip(et)
        return f"array[{n}]"
      out = []
      for i in range(n):
        if i < 16: out.append(self.val(et, key))
        else: self.skip(et)
      if n > 16: out.append(f"...({n})")
      return out
    self.skip(typ)
    return None

def split_paths(p, kv):
  total = kv.get("split.count", 1)
  if total <= 1: return [p]
  m = re.match(r"^(.*)-00001-of-\d{5}\.gguf$", str(p))
  if not m: raise SystemExit("first split path must end with -00001-of-NNNNN.gguf")
  return [pathlib.Path(f"{m.group(1)}-{i:05d}-of-{total:05d}.gguf") for i in range(1, total+1)]

def parse(p, base_kv=None):
  r = R(p)
  if r.f.read(4) != b"GGUF": raise ValueError(p)
  _, nt, nk = r.rd("<I"), r.rd("<Q"), r.rd("<Q")
  kv = {}
  for _ in range(nk):
    k, typ = r.s(), r.rd("<I")
    kv[k] = r.val(typ, k)
  meta = kv if base_kv is None else {**base_kv, **kv}
  infos = []
  for _ in range(nt):
    name = r.s()
    nd = r.rd("<I")
    dims = tuple(r.rd("<Q") for _ in range(nd))
    typ = r.rd("<I")
    r.rd("<Q")
    shape = tuple(reversed(dims))
    infos.append((name, shape, typ, nbytes(dims, typ), shard_axis(name, len(dims), meta)))
  return kv, infos

p = pathlib.Path(sys.argv[1])
kv, infos0 = parse(p)
infos = list(infos0)
for pp in split_paths(p, kv)[1:]: infos += parse(pp, kv)[1]
arch = kv.get("general.architecture")
print("model", kv.get("general.name"), "arch", arch, "splits", kv.get("split.count", 1), "tensors", len(infos))
print("block_count", kv.get(f"{arch}.block_count"), "ctx", kv.get(f"{arch}.context_length"))
stats, examples = collections.Counter(), {}
for name, shape, typ, nb, axis in infos:
  key = (cat(name), tname(typ), axis)
  stats[key] += nb
  examples.setdefault(key, (name, shape))
print("\nBY category/type/shard_axis")
for (c,t,a), nb in sorted(stats.items(), key=lambda x:-x[1]):
  ex, sh = examples[(c,t,a)]
  print(f"{nb/1e9:9.2f} GB  {c:14s} {t:8s} axis={str(a):4s}  ex={ex} shape={sh}")
print("\nBY tensor type")
bytype = collections.Counter()
for _,_,typ,nb,_ in infos: bytype[tname(typ)] += nb
for t, nb in bytype.most_common(): print(f"{nb/1e9:9.2f} GB  {t}")
if len(sys.argv) > 2:
  ndev = int(sys.argv[2])
  print(f"\nraw bytes / {ndev} devices lower bound: {sum(nb for *_,nb,_ in infos)/ndev/1e9:.2f} GB/device")
