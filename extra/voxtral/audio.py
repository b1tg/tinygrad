"""Audio loading and mel spectrogram computation for Voxtral — no torch dependency."""
import math, struct, wave
from tinygrad import Tensor

from extra.voxtral import SAMPLE_RATE, NUM_MEL_BINS, HOP_LENGTH, WINDOW_SIZE, GLOBAL_LOG_MEL_MAX
from extra.voxtral import RAW_AUDIO_LENGTH_PER_TOK, N_LEFT_PAD_TOKENS, N_RIGHT_PAD_TOKENS

# ============================================================================
# WAV loading (stdlib only)
# ============================================================================

def load_wav(path: str) -> tuple[list[float], int]:
  """Load a WAV file and return (samples_float32, sample_rate). Mono output."""
  with wave.open(path, 'rb') as wf:
    sr = wf.getframerate()
    n_channels = wf.getnchannels()
    sampwidth = wf.getsampwidth()
    n_frames = wf.getnframes()
    raw = wf.readframes(n_frames)

  if sampwidth == 2:
    fmt = f"<{n_frames * n_channels}h"
    samples = struct.unpack(fmt, raw)
    scale = 1.0 / 32768.0
  elif sampwidth == 4:
    fmt = f"<{n_frames * n_channels}i"
    samples = struct.unpack(fmt, raw)
    scale = 1.0 / 2147483648.0
  elif sampwidth == 1:
    samples = [b - 128 for b in raw]
    scale = 1.0 / 128.0
  else:
    raise ValueError(f"Unsupported sample width: {sampwidth}")

  float_samples = [s * scale for s in samples]

  # Convert to mono by averaging channels
  if n_channels > 1:
    mono = []
    for i in range(0, len(float_samples), n_channels):
      mono.append(sum(float_samples[i:i+n_channels]) / n_channels)
    float_samples = mono

  return float_samples, sr

def resample_linear(samples: list[float], src_sr: int, dst_sr: int) -> list[float]:
  """Simple linear interpolation resampler."""
  if src_sr == dst_sr: return samples
  ratio = src_sr / dst_sr
  n_out = int(len(samples) * dst_sr / src_sr)
  out = []
  for i in range(n_out):
    src_pos = i * ratio
    idx = int(src_pos)
    frac = src_pos - idx
    if idx + 1 < len(samples):
      out.append(samples[idx] * (1 - frac) + samples[idx + 1] * frac)
    else:
      out.append(samples[min(idx, len(samples) - 1)])
  return out

# ============================================================================
# Audio padding
# ============================================================================

def pad_audio_streaming(audio: list[float]) -> list[float]:
  """Pad audio for offline streaming mode (matching mistral_common AudioEncoder.pad)."""
  mult_of = RAW_AUDIO_LENGTH_PER_TOK
  n_samples = len(audio)
  align_pad = (mult_of - (n_samples % mult_of)) % mult_of
  right_pad = align_pad + N_RIGHT_PAD_TOKENS * mult_of
  left_pad = N_LEFT_PAD_TOKENS * mult_of
  return [0.0] * left_pad + audio + [0.0] * right_pad

# ============================================================================
# Mel filter bank (Slaney-style, matching mistral_common/audio.py)
# ============================================================================

def _hertz_to_mel(freq: float) -> float:
  min_log_hertz = 1000.0
  min_log_mel = 15.0
  logstep = 27.0 / math.log(6.4)
  mels = 3.0 * freq / 200.0
  if freq >= min_log_hertz:
    mels = min_log_mel + math.log(freq / min_log_hertz) * logstep
  return mels

def _mel_to_hertz(mel: float) -> float:
  min_log_hertz = 1000.0
  min_log_mel = 15.0
  logstep = math.log(6.4) / 27.0
  freq = 200.0 * mel / 3.0
  if mel >= min_log_mel:
    freq = min_log_hertz * math.exp(logstep * (mel - min_log_mel))
  return freq

def compute_mel_filters() -> list[list[float]]:
  """Returns mel filter bank as [num_frequency_bins, num_mel_bins] list."""
  num_freq_bins = 1 + WINDOW_SIZE // 2  # 201
  fft_freqs = [i * (SAMPLE_RATE // 2) / (num_freq_bins - 1) for i in range(num_freq_bins)]
  mel_min = _hertz_to_mel(0.0)
  mel_max = _hertz_to_mel(8000.0)
  mel_freqs = [mel_min + i * (mel_max - mel_min) / (NUM_MEL_BINS + 1) for i in range(NUM_MEL_BINS + 2)]
  filter_freqs = [_mel_to_hertz(m) for m in mel_freqs]
  filter_diff = [filter_freqs[i+1] - filter_freqs[i] for i in range(len(filter_freqs)-1)]
  fb = []
  for f_idx in range(num_freq_bins):
    row = []
    for m in range(NUM_MEL_BINS):
      down_slope = (fft_freqs[f_idx] - filter_freqs[m]) / filter_diff[m]
      up_slope = (filter_freqs[m+2] - fft_freqs[f_idx]) / filter_diff[m+1]
      val = max(0.0, min(down_slope, up_slope))
      enorm = 2.0 / (filter_freqs[m+2] - filter_freqs[m])
      row.append(val * enorm)
    fb.append(row)
  return fb

# ============================================================================
# STFT via DFT matrix (matching torch.stft with center=True, default behavior)
# ============================================================================

def _hann_window(size: int) -> list[float]:
  return [0.5 * (1 - math.cos(2 * math.pi * n / size)) for n in range(size)]

def _build_dft_matrix(n_fft: int):
  """Build DFT cos/sin matrices for n_fft//2+1 frequency bins. Returns flat lists for Tensor construction."""
  n_freq = n_fft // 2 + 1
  cos_data, sin_data = [], []
  for k in range(n_freq):
    for n in range(n_fft):
      angle = 2 * math.pi * k * n / n_fft
      cos_data.append(math.cos(angle))
      sin_data.append(-math.sin(angle))
  return cos_data, sin_data, n_freq

def compute_mel_spectrogram(audio: Tensor, mel_filters_list: list[list[float]]) -> Tensor:
  """Compute mel spectrogram matching torch.stft(center=True) + mel filterbank.
  audio: 1D Tensor of float32 samples.
  mel_filters_list: [n_freq, n_mel] precomputed filter bank.
  Returns: [n_mel, n_frames] Tensor.
  """
  n_fft = WINDOW_SIZE  # 400
  hop = HOP_LENGTH     # 160
  n_freq = n_fft // 2 + 1  # 201

  # Center padding (matching torch.stft center=True, pad_mode='reflect')
  pad_len = n_fft // 2  # 200
  # Use simple zero padding instead of reflect (close enough for edge regions which are zero-padded audio anyway)
  audio = audio.pad(((pad_len, pad_len),))

  n_samples = audio.shape[0]
  n_frames = 1 + (n_samples - n_fft) // hop

  # Extract frames using tensor indexing: frame_indices[i, j] = i*hop + j
  frame_offsets = Tensor.arange(n_frames).unsqueeze(1) * hop  # [n_frames, 1]
  sample_indices = Tensor.arange(n_fft).unsqueeze(0)           # [1, n_fft]
  indices = (frame_offsets + sample_indices).flatten()           # [n_frames * n_fft]

  # Gather and reshape to [n_frames, n_fft]
  frames = audio[indices].reshape(n_frames, n_fft)

  # Apply Hann window
  window = Tensor(_hann_window(n_fft))
  frames = frames * window

  # DFT via matrix multiply
  cos_data, sin_data, _ = _build_dft_matrix(n_fft)
  dft_cos = Tensor(cos_data).reshape(n_freq, n_fft)
  dft_sin = Tensor(sin_data).reshape(n_freq, n_fft)

  real = frames @ dft_cos.T   # [n_frames, n_freq]
  imag = frames @ dft_sin.T   # [n_frames, n_freq]
  magnitudes = real.square() + imag.square()

  # Drop last frame (matching torch stft[..., :-1])
  magnitudes = magnitudes[:-1]  # [n_frames-1, n_freq]

  # Mel filterbank
  mel_filters_t = Tensor(mel_filters_list)  # [n_freq, n_mel]
  mel_spec = magnitudes @ mel_filters_t     # [n_frames-1, n_mel]
  mel_spec = mel_spec.T                     # [n_mel, n_frames-1]

  # Log scale
  log_spec = mel_spec.clamp(min_=1e-10).log2() * math.log10(2)  # log10 via log2
  log_spec = log_spec.maximum(Tensor.full(log_spec.shape, GLOBAL_LOG_MEL_MAX - 8.0))
  log_spec = (log_spec + 4.0) / 4.0
  return log_spec

# ============================================================================
# Incremental mel spectrogram (matching C vox_mel_ctx_t)
# ============================================================================

class IncrementalMel:
  """Compute mel frames incrementally as audio samples arrive.

  Each frame t needs samples[t*HOP_LENGTH : t*HOP_LENGTH + WINDOW_SIZE].
  Frames are computed in pure Python (per-frame DFT on 400 samples is cheap).
  """
  def __init__(self, left_pad_samples: int = 0):
    self.mel_filters = compute_mel_filters()  # [n_freq, n_mel]
    self._dft_cos, self._dft_sin, self._n_freq = _build_dft_matrix(WINDOW_SIZE)
    self._window = _hann_window(WINDOW_SIZE)
    # Sample buffer: center-pad (200) + left_pad zeros
    self.left_pad = WINDOW_SIZE // 2 + left_pad_samples
    self.samples: list[float] = [0.0] * self.left_pad
    self.mel_frames: list[list[float]] = []

  def feed(self, new_samples: list[float]) -> int:
    """Append samples, compute available mel frames. Returns new frame count."""
    self.samples.extend(new_samples)
    return self._compute_available()

  def finish(self, right_pad_samples: int = 0) -> int:
    """Finalize: add right + reflect padding, compute remaining, drop last frame."""
    # Right padding zeros
    self.samples.extend([0.0] * right_pad_samples)
    # Reflect padding (200 samples from end of real content)
    real_end = len(self.samples) - right_pad_samples
    pad_len = WINDOW_SIZE // 2
    for i in range(pad_len):
      src = real_end - 2 - i
      self.samples.append(self.samples[src] if src >= 0 else 0.0)
    self._compute_available()
    # Drop last frame (vLLM convention: stft[..., :-1])
    if self.mel_frames:
      self.mel_frames.pop()
    return len(self.mel_frames)

  @property
  def n_frames(self) -> int:
    return len(self.mel_frames)

  def get_mel_tensor(self, start: int = 0, count: int | None = None) -> Tensor:
    """Return mel frames as Tensor [NUM_MEL_BINS, count]."""
    end = start + count if count is not None else len(self.mel_frames)
    data = self.mel_frames[start:end]
    return Tensor(data).T  # [n_mel, n_frames]

  def _compute_available(self) -> int:
    """Compute all mel frames whose window fits in current samples."""
    n_fft = WINDOW_SIZE
    n_freq = self._n_freq
    new_count = 0
    while True:
      t = len(self.mel_frames)
      start = t * HOP_LENGTH
      if start + n_fft > len(self.samples):
        break
      # Windowed frame
      windowed = [self.samples[start + i] * self._window[i] for i in range(n_fft)]
      # DFT -> power spectrum
      power = []
      for k in range(n_freq):
        off = k * n_fft
        re = sum(windowed[n] * self._dft_cos[off + n] for n in range(n_fft))
        im = sum(windowed[n] * self._dft_sin[off + n] for n in range(n_fft))
        power.append(re * re + im * im)
      # Mel filterbank + log scale
      mel_row = []
      for m in range(NUM_MEL_BINS):
        s = sum(self.mel_filters[f][m] * power[f] for f in range(n_freq))
        val = math.log10(max(s, 1e-10))
        val = max(val, GLOBAL_LOG_MEL_MAX - 8.0)
        mel_row.append((val + 4.0) / 4.0)
      self.mel_frames.append(mel_row)
      new_count += 1
    return new_count
