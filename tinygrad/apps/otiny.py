import argparse
import gc
import json
import os
import pwd
import sys
from pathlib import Path

from tinygrad import Tensor
from tinygrad.apps.llm import SimpleTokenizer, Transformer
from tinygrad.helpers import fetch

HOST = "registry.ollama.ai"
NS = "library"
MANIFEST_MT = "application/vnd.docker.distribution.manifest.v2+json"
MODEL_MT = "application/vnd.ollama.image.model"
UA = "tinygrad-otiny"


def parse_ref(ref: str) -> tuple[str, str, str]:
  tail = ref.rsplit("/", 1)[-1]
  base, tag = ref.rsplit(":", 1) if ":" in tail else (ref, "latest")
  if "/" in base:
    namespace, model = base.split("/", 1)
  else:
    namespace, model = NS, base
  return namespace, model, tag


def daemon_root() -> Path | None:
  try:
    return Path(pwd.getpwnam("ollama").pw_dir) / ".ollama" / "models"
  except KeyError:
    return None


def root_candidates() -> list[Path]:
  if env_root := os.getenv("OLLAMA_MODELS"):
    return [Path(env_root).expanduser()]
  out: list[Path] = []
  if droot := daemon_root():
    out.append(droot)
  out.append(Path.home() / ".ollama" / "models")
  uniq: list[Path] = []
  for p in out:
    if p not in uniq:
      uniq.append(p)
  return uniq


def manifest_path(root: Path, namespace: str, model: str, tag: str) -> Path:
  return root / "manifests" / HOST / namespace / model / tag


def blob_path(root: Path, digest: str) -> Path:
  return root / "blobs" / digest.lower().replace(":", "-", 1)


def find_existing_root(namespace: str, model: str, tag: str) -> Path | None:
  for root in root_candidates():
    if manifest_path(root, namespace, model, tag).is_file():
      return root
  return None


def find_writable_root() -> Path:
  for root in root_candidates():
    try:
      (root / "manifests").mkdir(parents=True, exist_ok=True)
      return root
    except PermissionError:
      continue
  return root_candidates()[-1]


def maybe_warn_daemon_access(selected_root: Path) -> None:
  if os.getenv("OLLAMA_MODELS"):
    return
  droot = daemon_root()
  if droot is None or droot == selected_root or not droot.exists():
    return
  if os.access(droot, os.R_OK | os.X_OK):
    return
  gid = pwd.getpwnam("ollama").pw_gid
  if gid not in os.getgroups() and gid != os.getgid():
    print(f"note: ollama store is {droot}. this shell is not in group 'ollama'; run `newgrp ollama` or relogin.", file=sys.stderr)
  else:
    print(f"note: ollama store is {droot} but is not accessible; using {selected_root}", file=sys.stderr)


def list_models() -> None:
  seen: set[str] = set()
  rows: list[tuple[str, Path]] = []
  inaccessible: list[Path] = []
  for root in root_candidates():
    mroot = root / "manifests" / HOST
    try:
      if not mroot.is_dir():
        continue
      for ns_dir in sorted(mroot.iterdir()):
        if not ns_dir.is_dir():
          continue
        for model_dir in sorted(ns_dir.iterdir()):
          if not model_dir.is_dir():
            continue
          for tag_file in sorted(model_dir.iterdir()):
            if not tag_file.is_file():
              continue
            name = f"{model_dir.name}:{tag_file.name}"
            if name in seen:
              continue
            seen.add(name)
            rows.append((name, root))
    except PermissionError:
      inaccessible.append(root)

  if not rows:
    print("no models found")
  else:
    print(f"{'NAME':<40} STORE")
    for name, root in rows:
      print(f"{name:<40} {root}")

  for root in inaccessible:
    print(f"note: cannot access model store: {root}", file=sys.stderr)


def pull(ref: str, root: Path | None = None) -> Path:
  namespace, model, tag = parse_ref(ref)
  root = root if root is not None else (find_existing_root(namespace, model, tag) or find_writable_root())
  maybe_warn_daemon_access(root)
  root.mkdir(parents=True, exist_ok=True)
  mp = manifest_path(root, namespace, model, tag)
  mp.parent.mkdir(parents=True, exist_ok=True)
  name = f"{HOST}/{namespace}/{model}:{tag}"
  print(f"pulling manifest {name}")
  fetch(f"https://{HOST}/v2/{namespace}/{model}/manifests/{tag}", name=mp, headers={"Accept": MANIFEST_MT, "User-Agent": UA}, allow_caching=False)
  mp.chmod(0o644)
  manifest = json.loads(mp.read_text())
  for layer in [manifest["config"], *manifest["layers"]]:
    bp = blob_path(root, layer["digest"])
    bp.parent.mkdir(parents=True, exist_ok=True)
    size = int(layer["size"])
    short = layer["digest"][:19]
    if bp.is_file() and bp.stat().st_size == size:
      print(f"using existing blob {short}")
      continue
    print(f"pulling {layer.get('mediaType', 'unknown')} {short} ({size:,} bytes)")
    fetch(f"https://{HOST}/v2/{namespace}/{model}/blobs/{layer['digest']}", name=bp, headers={"User-Agent": UA}, allow_caching=False)
    bp.chmod(0o644)
  print(f"wrote manifest to {mp}")
  return root


def run(ref: str) -> None:
  namespace, model, tag = parse_ref(ref)
  root = find_existing_root(namespace, model, tag) or find_writable_root()
  maybe_warn_daemon_access(root)
  name = f"{HOST}/{namespace}/{model}:{tag}"
  bp = None
  for _ in range(2):
    mp = manifest_path(root, namespace, model, tag)
    mp.parent.mkdir(parents=True, exist_ok=True)
    if not mp.is_file():
      print(f"model not found locally, pulling {name}")
      root = pull(ref, root=root)
      mp = manifest_path(root, namespace, model, tag)
    manifest = json.loads(mp.read_text())
    layer = next(x for x in manifest["layers"] if x.get("mediaType") == MODEL_MT)
    bp = blob_path(root, layer["digest"])
    bp.parent.mkdir(parents=True, exist_ok=True)
    if bp.is_file():
      break
    print(f"model blob missing, pulling {name}")
    root = pull(ref, root=root)
  if bp is None or not bp.is_file():
    raise FileNotFoundError(f"missing model blob for {name}")

  print(f"loading {name} from {bp}")
  raw = Tensor(bp)
  model_obj, kv = Transformer.from_gguf(raw, max_context=4096)
  del raw
  gc.collect()
  tok = SimpleTokenizer.from_gguf_kv(kv)
  bos = kv.get("tokenizer.ggml.bos_token_id") if kv.get("tokenizer.ggml.add_bos_token", True) else None
  eos = kv["tokenizer.ggml.eos_token_id"]
  ids = [bos] if bos is not None else []
  while True:
    try:
      q = input(">>> ").strip()
    except (EOFError, KeyboardInterrupt):
      print()
      return
    if not q:
      continue
    start = max(len(ids) - 1, 0)
    ids += tok.role("user") + tok.encode(q) + tok.end_turn(eos)
    ids += tok.encode("<|im_start|>assistant\n<think>\n\n</think>\n\n") if tok.preset == "qwen35" else tok.role("assistant")
    for tid in model_obj.generate(ids, start):
      if tid == eos:
        print("\n")
        break
      sys.stdout.write(tok.decode([tid]))
      sys.stdout.flush()


def main() -> int:
  ap = argparse.ArgumentParser("otiny")
  sub = ap.add_subparsers(dest="cmd", required=True)
  sub.add_parser("pull").add_argument("model")
  sub.add_parser("run").add_argument("model")
  sub.add_parser("ls")
  args = ap.parse_args()
  try:
    if args.cmd == "ls":
      list_models()
    elif args.cmd == "pull":
      pull(args.model)
    else:
      run(args.model)
  except Exception as e:
    print(f"error: {e}", file=sys.stderr)
    return 1
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
