#!/usr/bin/env python3
"""Download model + test WAV, run transcription, verify output."""
import subprocess, sys, pathlib
from tinygrad.helpers import fetch

MODEL_URL = "https://huggingface.co/m1el/nemotron-speech-streaming-0.6B-gguf/resolve/main/nemotron-speech-streaming-0.6B-v0.1.Q8_0.gguf"
WAV_URL = "https://www.voiptroubleshooter.com/open_speech/american/OSR_us_000_0010_8k.wav"

if __name__ == "__main__":
  model_path = fetch(MODEL_URL, "nemotron-0.6b-q8.gguf", subdir="nemotron-asr")
  wav_path = fetch(WAV_URL, "test_speech.wav", subdir="nemotron-asr")
  print(f"model: {model_path}")
  print(f"wav:   {wav_path}")

  script = pathlib.Path(__file__).parent / "transcribe.py"
  result = subprocess.run([sys.executable, str(script), str(model_path), str(wav_path)], capture_output=True, text=True)
  print(result.stdout)
  if result.stderr: print(result.stderr, file=sys.stderr)

  output = result.stdout.strip().split("\n")[-1].lower()
  assert "the" in output, f"transcription looks wrong: {output!r}"
  print("PASS")
