#!/usr/bin/env python3
"""Benchmark tinygrad qwen-asr vs extra/antirez-qwen-asr/qwen_asr."""

from __future__ import annotations
import argparse, json, os, pathlib, re, statistics, subprocess, sys, time


def run_cmd(cmd: list[str], cwd: pathlib.Path | None = None) -> tuple[int, str, str, float]:
  st = time.perf_counter()
  proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
  return proc.returncode, proc.stdout, proc.stderr, time.perf_counter() - st


def parse_tinygrad_timings(stderr: str) -> dict:
  out = {}
  for line in stderr.splitlines():
    line = line.strip()
    if not (line.startswith("{") and line.endswith("}")):
      continue
    try:
      obj = json.loads(line)
      if "wall_seconds" in obj:
        out = obj
    except json.JSONDecodeError:
      pass
  return out


def parse_antirez_realtime(stderr: str) -> dict[str, float]:
  out: dict[str, float] = {}
  for line in stderr.splitlines():
    m = re.search(r"Audio:\s*([0-9.]+)\s*s processed in\s*([0-9.]+)\s*s\s*\(([0-9.]+)x realtime\)", line)
    if m:
      out = {
        "audio_seconds": float(m.group(1)),
        "wall_seconds": float(m.group(2)),
        "realtime_x": float(m.group(3)),
      }
  return out


def summarize(name: str, walls: list[float], extra: dict | None = None) -> None:
  mean_s = statistics.mean(walls)
  p50_s = statistics.median(walls)
  print(f"{name:8s} mean={mean_s:.3f}s p50={p50_s:.3f}s min={min(walls):.3f}s max={max(walls):.3f}s")
  if extra:
    kept = {k: extra[k] for k in sorted(extra.keys()) if isinstance(extra[k], (int, float))}
    print(f"{name:8s} details={json.dumps(kept, sort_keys=True)}")


def main() -> None:
  parser = argparse.ArgumentParser(description="Benchmark tinygrad qwen-asr vs antirez qwen_asr")
  parser.add_argument("--model-dir", required=True)
  parser.add_argument("--audio", required=True)
  parser.add_argument("--runs", type=int, default=3)
  parser.add_argument("--warmup", type=int, default=1)
  parser.add_argument("--python", default=sys.executable)
  parser.add_argument("--tinygrad-script", default="extra/qwen-asr/transcribe.py")
  parser.add_argument("--tinygrad-args", default="")
  parser.add_argument("--tinygrad-jit", action="store_true")
  parser.add_argument("--antirez-bin", default="extra/antirez-qwen-asr/qwen_asr")
  parser.add_argument("--antirez-args", default="--silent")
  args = parser.parse_args()

  root = pathlib.Path(__file__).resolve().parents[2]
  tiny_script = root / args.tinygrad_script
  antirez_bin = root / args.antirez_bin

  if not tiny_script.exists():
    raise FileNotFoundError(f"missing tinygrad script: {tiny_script}")
  if not antirez_bin.exists():
    raise FileNotFoundError(f"missing antirez binary: {antirez_bin}")

  tiny_cmd = [args.python, str(tiny_script), args.model_dir, args.audio, "--silent", "--timings-json"] + args.tinygrad_args.split()
  if args.tinygrad_jit:
    tiny_cmd.append("--jit")
  antirez_cmd = [str(antirez_bin), "-d", args.model_dir, "-i", args.audio] + args.antirez_args.split()

  print("Commands:")
  print(" tinygrad:", " ".join(tiny_cmd))
  print(" antirez :", " ".join(antirez_cmd))

  for _ in range(args.warmup):
    rc, _, _, _ = run_cmd(tiny_cmd, cwd=root)
    if rc != 0:
      raise RuntimeError("tinygrad warmup failed")
    rc, _, _, _ = run_cmd(antirez_cmd, cwd=root)
    if rc != 0:
      raise RuntimeError("antirez warmup failed")

  tiny_walls: list[float] = []
  anti_walls: list[float] = []
  tiny_meta: dict | None = None
  anti_meta: dict | None = None

  for _ in range(args.runs):
    rc, out, err, wall = run_cmd(tiny_cmd, cwd=root)
    if rc != 0:
      print(out)
      print(err, file=sys.stderr)
      raise RuntimeError("tinygrad benchmark run failed")
    tiny_walls.append(wall)
    meta = parse_tinygrad_timings(err)
    if meta:
      tiny_meta = meta

    rc, out, err, wall = run_cmd(antirez_cmd, cwd=root)
    if rc != 0:
      print(out)
      print(err, file=sys.stderr)
      raise RuntimeError("antirez benchmark run failed")
    anti_walls.append(wall)
    meta = parse_antirez_realtime(err)
    if meta:
      anti_meta = meta

  print("\nResults")
  summarize("tinygrad", tiny_walls, tiny_meta)
  summarize("antirez", anti_walls, anti_meta)
  print(f"speedup antirez/tinygrad (mean wall): {statistics.mean(tiny_walls) / statistics.mean(anti_walls):.2f}x")


if __name__ == "__main__":
  main()
