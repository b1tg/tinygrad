import math, struct, time, threading
from queue import Queue, Full
from tinygrad import Tensor, dtypes


class LiveAudioInput:
  def __init__(self, sample_rate, gain=1.0, pyaudio_chunk=1024, debug_level=0):
    self.sample_rate = sample_rate
    self.gain = gain
    self.pyaudio_chunk = pyaudio_chunk
    self.debug_level = debug_level
    self.backend = None
    self.sd = None
    self.sd_stream = None
    self.sd_overflows = 0
    self.pa = None
    self.stream = None

  def __enter__(self):
    sd_err = None
    try:
      import sounddevice as sd
      self.backend, self.sd = "sounddevice", sd
      self.sd_stream = sd.InputStream(samplerate=self.sample_rate, channels=1, dtype="float32", blocksize=self.pyaudio_chunk)
      self.sd_stream.start()
      return self
    except ImportError:
      pass
    except Exception as e:
      sd_err = e

    try:
      import pyaudio
    except ImportError as e:
      if sd_err is not None:
        raise RuntimeError(f"sounddevice init failed ({sd_err}) and pyaudio is not installed") from e
      raise RuntimeError("live mode requires `sounddevice` or `pyaudio` (pip install sounddevice or pyaudio)") from e
    self.backend = "pyaudio"
    self.pa = pyaudio.PyAudio()
    self.stream = self.pa.open(format=pyaudio.paInt16, channels=1, rate=self.sample_rate, input=True, frames_per_buffer=self.pyaudio_chunk)
    return self

  def __exit__(self, exc_type, exc, tb):
    if self.sd_stream is not None:
      self.sd_stream.stop()
      self.sd_stream.close()
    if self.stream is not None:
      self.stream.stop_stream()
      self.stream.close()
    if self.pa is not None:
      self.pa.terminate()

  def record_raw(self, seconds):
    n_frames = max(1, int(seconds * self.sample_rate))
    if self.backend == "sounddevice":
      samples = []
      while len(samples) < n_frames:
        frame, overflowed = self.sd_stream.read(n_frames - len(samples))
        if overflowed:
          self.sd_overflows += 1
          if self.debug_level >= 3 and (self.sd_overflows <= 3 or self.sd_overflows % 20 == 0):
            print(f"[audio] sounddevice overflow x{self.sd_overflows}")
        samples.extend(frame[:, 0].tolist())
      if self.gain != 1.0:
        samples = [s * self.gain for s in samples]
      return samples

    pcm = bytearray()
    while len(pcm) < n_frames * 2:
      rem = n_frames - len(pcm) // 2
      pcm.extend(self.stream.read(min(rem, self.pyaudio_chunk), exception_on_overflow=False))
    samples = struct.unpack(f"<{n_frames}h", pcm[:n_frames*2])
    gain = self.gain / 32768.0
    return [s * gain for s in samples]


def add_live_args(parser):
  parser.add_argument("--chunk-seconds", type=float, default=1.0, help="seconds per live chunk")
  parser.add_argument("--mic-gain", type=float, default=2.0, help="multiply microphone waveform by this value before decoding")
  parser.add_argument("--live-debug-level", type=int, default=0, help="0: transcript only, 1: segment events, 2: +timing, 3: +VAD/queue")
  parser.add_argument("--vad-threshold", type=float, default=0.010, help="RMS threshold for speech detection in segment mode")
  parser.add_argument("--vad-ratio", type=float, default=1.4, help="relative threshold multiplier over noise floor in segment mode")
  parser.add_argument("--vad-preroll-seconds", type=float, default=1.0, help="seconds prepended when speech starts in segment mode")
  parser.add_argument("--vad-end-seconds", type=float, default=1.5, help="flush segment after this many seconds of non-speech")
  parser.add_argument("--segment-max-seconds", type=float, default=8.0, help="force segment flush at this duration even without silence")
  parser.add_argument("--segment-overlap-seconds", type=float, default=1.0, help="carry this much context after force flush in segment mode")
  parser.add_argument("--live-queue-chunks", type=int, default=256, help="max buffered live chunks while decode is running")
  parser.add_argument("--max-chunks", type=int, default=0, help="stop after N chunks in live mode (0 means run until Ctrl+C)")


def validate_live_args(parser, args):
  if args.chunk_seconds <= 0: parser.error("--chunk-seconds must be > 0")
  if args.mic_gain <= 0: parser.error("--mic-gain must be > 0")
  if args.vad_threshold < 0: parser.error("--vad-threshold must be >= 0")
  if args.vad_ratio <= 1.0: parser.error("--vad-ratio must be > 1.0")
  if args.vad_preroll_seconds < 0: parser.error("--vad-preroll-seconds must be >= 0")
  if args.vad_end_seconds <= 0: parser.error("--vad-end-seconds must be > 0")
  if args.segment_max_seconds <= 0: parser.error("--segment-max-seconds must be > 0")
  if args.segment_overlap_seconds < 0: parser.error("--segment-overlap-seconds must be >= 0")
  if args.live_queue_chunks <= 0: parser.error("--live-queue-chunks must be > 0")
  if args.live_debug_level < 0 or args.live_debug_level > 3: parser.error("--live-debug-level must be in [0, 3]")
  if args.max_chunks < 0: parser.error("--max-chunks must be >= 0")


def _suffix_prefix_overlap(a, b):
  max_len = min(len(a), len(b))
  for n in range(max_len, 0, -1):
    if a[-n:] == b[:n]:
      return n
  return 0


def _build_recorder(recorder_factory, sample_rate, gain, live_debug_level):
  try:
    return recorder_factory(sample_rate, gain=gain, debug_level=live_debug_level)
  except TypeError:
    return recorder_factory(sample_rate, gain=gain)


def _live_metrics_str(recorder, backlog, queue_chunks, queue_highwater, queue_full_events):
  util = (100.0 * backlog) / max(1, queue_chunks)
  hi_util = (100.0 * queue_highwater) / max(1, queue_chunks)
  overflows = getattr(recorder, "sd_overflows", 0)
  return f"queue={backlog}/{queue_chunks}({util:.0f}%) hi={queue_highwater}({hi_util:.0f}%) full={queue_full_events} ovf={overflows}"


def _decode_with_live_feedback(decode_fn, live_debug_level):
  heartbeat_stop = threading.Event()
  heartbeat_thr = None
  if live_debug_level == 0:
    print("...", flush=True)
    def _heartbeat():
      while not heartbeat_stop.wait(1.5):
        print("...", flush=True)
    heartbeat_thr = threading.Thread(target=_heartbeat, daemon=True, name="live-decode-heartbeat")
    heartbeat_thr.start()
  t0 = time.perf_counter()
  try:
    out = decode_fn()
  finally:
    t_decode = time.perf_counter() - t0
    heartbeat_stop.set()
    if heartbeat_thr is not None:
      heartbeat_thr.join(timeout=0.1)
  return out, t_decode


def _chunk_rms(chunk_audio):
  return (chunk_audio.square().mean().sqrt().item()) if chunk_audio.shape[0] > 0 else 0.0


def _live_chunk_iterator(recorder, chunk_seconds, max_chunks=0, queue_chunks=128):
  chunk_queue = Queue(maxsize=max(1, queue_chunks))
  stop_event = threading.Event()
  stats = {"queue_hi": 0, "queue_full": 0}

  def _queue_put(item):
    while not stop_event.is_set():
      try:
        chunk_queue.put(item, timeout=0.1)
        stats["queue_hi"] = max(stats["queue_hi"], chunk_queue.qsize())
        return
      except Full:
        stats["queue_full"] += 1

  def _producer():
    try:
      i = 0
      while not stop_event.is_set() and (max_chunks == 0 or i < max_chunks):
        i += 1
        t0 = time.perf_counter()
        if hasattr(recorder, "record_raw"):
          chunk_data = recorder.record_raw(chunk_seconds)
        else:
          chunk_data = recorder.record(chunk_seconds)
          if isinstance(chunk_data, Tensor): chunk_data = chunk_data.tolist()
        t_record = time.perf_counter() - t0
        _queue_put((i, chunk_data, t_record))
    except BaseException as e:
      _queue_put(e)
    finally:
      _queue_put(None)

  producer = threading.Thread(target=_producer, daemon=True, name="live-audio-capture")
  producer.start()
  try:
    while True:
      item = chunk_queue.get()
      if item is None: break
      if isinstance(item, BaseException): raise item
      yield item, chunk_queue.qsize(), stats["queue_hi"], stats["queue_full"]
  finally:
    stop_event.set()
    producer.join(timeout=0.2)


def transcribe_live_segmented(decode_audio_fn, sample_rate, chunk_seconds, max_chunks=0, mic_gain=1.0,
                              vad_threshold=0.015, vad_end_seconds=1.0, segment_max_seconds=20.0,
                              segment_overlap_seconds=1.0, vad_ratio=1.8, vad_preroll_seconds=1.0,
                              recorder_factory=LiveAudioInput, live_queue_chunks=128, live_debug_level=0):
  vad_end_chunks = max(2, math.ceil(vad_end_seconds / chunk_seconds))
  segment_max_samples = int(segment_max_seconds * sample_rate)
  segment_overlap_samples = max(0, int(segment_overlap_seconds * sample_rate))
  preroll_chunks = max(1, math.ceil(vad_preroll_seconds / chunk_seconds))

  with _build_recorder(recorder_factory, sample_rate, mic_gain, live_debug_level) as recorder:
    if live_debug_level >= 1:
      print(f"live input ready ({recorder.backend}, segmented mode, {chunk_seconds:.2f}s chunk, vad={vad_threshold:.4f}, "
            f"end_silence={vad_end_seconds:.2f}s, max_seg={segment_max_seconds:.2f}s, overlap={segment_overlap_seconds:.2f}s, "
            f"preroll={vad_preroll_seconds:.2f}s, queue={live_queue_chunks}, Ctrl+C to stop)")

    seg_audio, silence_chunks, seg_idx = None, 0, 0
    transcript = ""
    speech_chunks, noise_rms, preroll = 0, None, []
    empty_retry, max_empty_retry = 0, 2
    min_emit_samples = int(0.5 * sample_rate)

    for (i, chunk_data, t_record), backlog, queue_hi, queue_full in _live_chunk_iterator(recorder, chunk_seconds, max_chunks, live_queue_chunks):
      if live_debug_level >= 3: print(f"\nprocessing chunk {i}...")
      chunk_audio = Tensor(chunk_data, dtype=dtypes.float32).realize()
      rms = _chunk_rms(chunk_audio)
      rel_threshold = max(vad_threshold * 0.5, noise_rms * vad_ratio) if noise_rms is not None else 0.0
      rel_active = noise_rms is not None and rms >= rel_threshold and rms >= noise_rms * 1.2
      is_speech = (rms >= vad_threshold) or rel_active
      if live_debug_level >= 3:
        noise_str = f"{noise_rms:.4f}" if noise_rms is not None else "None"
        metrics = _live_metrics_str(recorder, backlog, live_queue_chunks, queue_hi, queue_full)
        print(f"[vad] rms={rms:.4f} abs={vad_threshold:.4f} rel={rel_threshold:.4f} noise={noise_str} "
              f"speech={is_speech} record={t_record:.2f}s {metrics}")

      preroll.append(chunk_audio)
      if len(preroll) > preroll_chunks: preroll.pop(0)
      if seg_audio is None and not is_speech:
        noise_rms = rms if noise_rms is None else (noise_rms * 0.90 + rms * 0.10)
        continue

      if seg_audio is not None:
        seg_audio = seg_audio.cat(chunk_audio).realize()
      else:
        seg_audio = preroll[0]
        for c in preroll[1:]:
          seg_audio = seg_audio.cat(c).realize()
        speech_chunks = 0

      if is_speech:
        silence_chunks, speech_chunks = 0, speech_chunks + 1
      elif seg_audio is not None:
        silence_chunks += 1

      force_flush = seg_audio is not None and seg_audio.shape[0] >= segment_max_samples
      end_flush = max_chunks and i == max_chunks and seg_audio is not None
      should_flush = (seg_audio is not None and silence_chunks >= vad_end_chunks) or force_flush or end_flush
      if not should_flush: continue

      tail_silence = int(min(silence_chunks, vad_end_chunks) * chunk_seconds * sample_rate)
      if force_flush: tail_silence = 0
      use_audio = seg_audio[:-tail_silence] if tail_silence > 0 and seg_audio.shape[0] > tail_silence else seg_audio
      if speech_chunks > 0 and use_audio.shape[0] >= min_emit_samples:
        seg_idx += 1
        if live_debug_level >= 1:
          print(f"transcribing segment {seg_idx}... ({use_audio.shape[0]/sample_rate:.2f}s)")
        t1 = time.perf_counter()
        text, _ = _decode_with_live_feedback(lambda: decode_audio_fn(use_audio).strip(), live_debug_level)
        if not text:
          peak = use_audio.abs().max().item()
          if peak > 1e-6:
            boost = min(3.0, 0.75 / peak)
            retry_audio = (use_audio * boost).maximum(-1.0).minimum(1.0).realize()
            text, _ = _decode_with_live_feedback(lambda: decode_audio_fn(retry_audio).strip(), live_debug_level)
        t_decode = time.perf_counter() - t1
        if text:
          empty_retry = 0
          if transcript:
            overlap = _suffix_prefix_overlap(transcript, text)
            text = text[overlap:].strip() if overlap else text
          if text:
            transcript = (transcript + " " + text).strip() if transcript else text
          if live_debug_level >= 1:
            print(f"[segment {seg_idx}] {text}")
            print(f"[full] {transcript}")
          elif text:
            print(text)
        else:
          if live_debug_level >= 1: print(f"[segment {seg_idx}] (empty)")
          if empty_retry < max_empty_retry and (force_flush or silence_chunks >= vad_end_chunks):
            empty_retry += 1
            seg_idx -= 1
            keep = max(segment_max_samples // 2, int(4 * sample_rate))
            seg_audio = seg_audio[-keep:].realize() if seg_audio.shape[0] > keep else seg_audio
            silence_chunks = 0
            speech_chunks = max(1, speech_chunks)
            if live_debug_level >= 1:
              print(f"[segment retry] keep {seg_audio.shape[0]/sample_rate:.2f}s and continue (retry {empty_retry}/{max_empty_retry})")
            if live_debug_level >= 2: print(f"[timing] decode {t_decode:.2f}s")
            continue
        if live_debug_level >= 2: print(f"[timing] decode {t_decode:.2f}s")

      if force_flush and segment_overlap_samples > 0 and seg_audio is not None:
        keep = min(segment_overlap_samples, seg_audio.shape[0])
        seg_audio = seg_audio[-keep:].realize() if keep > 0 else None
        silence_chunks, speech_chunks = 0, 1 if seg_audio is not None else 0
      else:
        seg_audio, silence_chunks, speech_chunks = None, 0, 0
