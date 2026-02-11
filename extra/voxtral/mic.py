"""Microphone capture via ffmpeg subprocess + silence cancellation.

macOS: ffmpeg -f avfoundation -i ":default" -ar 16000 -ac 1 -f s16le -
Linux: ffmpeg -f pulse -i default -ar 16000 -ac 1 -f s16le -
"""
import sys, struct, subprocess, threading, collections, math

SAMPLE_RATE = 16000

class MicCapture:
  """ffmpeg-based microphone capture producing float32 samples at 16kHz."""
  def __init__(self):
    self.process: subprocess.Popen | None = None
    self._thread: threading.Thread | None = None
    self._buf: collections.deque[float] = collections.deque(maxlen=SAMPLE_RATE * 10)
    self._lock = threading.Lock()
    self._running = False

  def start(self):
    if sys.platform == "darwin":
      cmd = ["ffmpeg", "-f", "avfoundation", "-i", ":default", "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "s16le", "-"]
    elif sys.platform == "linux":
      cmd = ["ffmpeg", "-f", "pulse", "-i", "default", "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "s16le", "-"]
    else:
      raise RuntimeError(f"Unsupported platform for mic capture: {sys.platform}")
    self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    self._running = True
    self._thread = threading.Thread(target=self._read_loop, daemon=True)
    self._thread.start()
    print("Listening (Ctrl+C to stop)...", file=sys.stderr)

  def _read_loop(self):
    assert self.process and self.process.stdout
    CHUNK_BYTES = 3200 * 2  # 200ms at 16kHz, 2 bytes/sample
    while self._running:
      raw = self.process.stdout.read(CHUNK_BYTES)
      if not raw: break
      n_samples = len(raw) // 2
      samples = struct.unpack(f"<{n_samples}h", raw[:n_samples * 2])
      with self._lock:
        for s in samples:
          self._buf.append(s / 32768.0)

  def read(self, max_samples: int = 4800) -> list[float]:
    with self._lock:
      n = min(len(self._buf), max_samples)
      return [self._buf.popleft() for _ in range(n)]

  def available(self) -> int:
    with self._lock:
      return len(self._buf)

  def stop(self):
    self._running = False
    if self.process:
      self.process.terminate()
      self.process.wait()
      self.process = None
    if self._thread:
      self._thread.join(timeout=2.0)
      self._thread = None


class SilenceCanceller:
  """Silence cancellation matching C main.c logic.

  10ms windows, RMS threshold 0.002, 600ms silence pass-through.
  """
  WINDOW = 160          # 10ms at 16kHz
  THRESHOLD = 0.002     # RMS ~-54 dBFS
  PASS_WINDOWS = 60     # 600ms pass-through

  def __init__(self):
    self.silence_count = 0
    self.was_skipping = False

  def process(self, samples: list[float]) -> tuple[list[float], bool]:
    """Returns (filtered_samples, should_flush). should_flush=True on entering prolonged silence."""
    output: list[float] = []
    should_flush = False
    off = 0
    while off + self.WINDOW <= len(samples):
      window = samples[off:off + self.WINDOW]
      rms = math.sqrt(sum(v * v for v in window) / self.WINDOW)
      if rms > self.THRESHOLD:
        if self.was_skipping: self.was_skipping = False
        output.extend(window)
        self.silence_count = 0
      else:
        self.silence_count += 1
        if self.silence_count <= self.PASS_WINDOWS:
          output.extend(window)
        elif not self.was_skipping:
          self.was_skipping = True
          should_flush = True
      off += self.WINDOW
    if off < len(samples):
      output.extend(samples[off:])
    return output, should_flush
