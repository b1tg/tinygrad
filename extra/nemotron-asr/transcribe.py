#!/usr/bin/env python3
"""Nemotron ASR - tinygrad implementation of NVIDIA's streaming conformer-transducer model.

Usage:
  python extra/nemotron-asr/transcribe.py model.gguf test.wav
  python extra/nemotron-asr/transcribe.py model.nemo test.wav
"""
import sys, math, struct, tarfile, tempfile, os
from tinygrad import Tensor, nn, dtypes
from tinygrad.nn.state import torch_load, load_state_dict

# ============================================================================
# Constants
# ============================================================================
SAMPLE_RATE = 16000
N_FFT = 512
N_WINDOW = 400
HOP_LENGTH = 160
N_MELS = 128
PREEMPH = 0.97
LOG_ZERO_GUARD = 2**-24

D_MODEL = 1024
N_HEADS = 8
D_HEAD = 128
D_FF = 4096
N_LAYERS = 24
KERNEL_SIZE = 9
VOCAB_SIZE = 1025
DECODER_DIM = 640
JOINT_DIM = 640
BLANK_TOKEN = 1024

# ============================================================================
# Preprocessing (tinygrad - matches NeMo's AudioToMelSpectrogramPreprocessor)
# ============================================================================
def preprocess(audio_float, filterbank, window):
  # audio_float: Tensor [N], filterbank: Tensor [n_mels, n_bins], window: Tensor [N_WINDOW]
  # pre-emphasis
  audio = audio_float[1:] - PREEMPH * audio_float[:-1]
  audio = audio_float[:1].cat(audio)

  # center-pad with n_fft/2 on each side
  pad = N_FFT // 2
  audio = audio.pad(((pad, pad),))

  # pad window from 400 to 512 (center)
  p = (N_FFT - N_WINDOW) // 2
  win = window.pad(((p, N_FFT - N_WINDOW - p),))

  # STFT via _pool for frame extraction
  n_samples = audio.shape[0]
  n_frames = 1 + (n_samples - N_FFT) // HOP_LENGTH
  n_bins = N_FFT // 2 + 1

  # build DFT matrix [n_bins, N_FFT]
  k = Tensor.arange(n_bins).reshape(n_bins, 1)  # [n_bins, 1]
  n = Tensor.arange(N_FFT).reshape(1, N_FFT)    # [1, N_FFT]
  angle = -2.0 * math.pi * k * n / N_FFT         # [n_bins, N_FFT]
  dft_real = angle.cos()  # [n_bins, N_FFT]
  dft_imag = angle.sin()  # [n_bins, N_FFT]

  # extract frames [n_frames, N_FFT] via _pool
  frames = audio.reshape(1, 1, -1)._pool((N_FFT,), stride=(HOP_LENGTH,)).reshape(-1, N_FFT)[:n_frames]
  frames = frames * win  # apply window

  # DFT: [n_frames, N_FFT] @ [N_FFT, n_bins] -> [n_frames, n_bins]
  real = frames @ dft_real.T
  imag = frames @ dft_imag.T
  spec = real * real + imag * imag  # power spectrum [n_frames, n_bins]

  # mel filterbank + log
  mel = spec @ filterbank.T  # [n_frames, n_mels]
  mel = (mel + LOG_ZERO_GUARD).log()
  return mel  # [n_frames, 128]

# ============================================================================
# Positional embeddings (sinusoidal, NeMo convention: descending order)
# ============================================================================
def compute_pos_emb(max_len, d_model):
  total_len = 2 * max_len - 1
  positions = Tensor.arange(max_len - 1, -(max_len), -1, dtype=dtypes.float32)
  dims = Tensor.arange(0, d_model, 2, dtype=dtypes.float32)
  div_terms = (-dims * math.log(10000.0) / d_model).exp()
  angles = positions.reshape(-1, 1) * div_terms.reshape(1, -1)  # [total_len, d_model/2]
  pos_emb = Tensor.zeros(total_len, d_model)
  # interleave sin/cos
  sin_vals = angles.sin()
  cos_vals = angles.cos()
  pos_emb = sin_vals.unsqueeze(-1).cat(cos_vals.unsqueeze(-1), dim=-1).reshape(total_len, d_model)
  return pos_emb

# ============================================================================
# Model classes (attribute names match NeMo checkpoint keys for load_state_dict)
# ============================================================================
def causal_pad_2d(x, kernel_size, stride):
  pl, pr = kernel_size - 1, stride - 1
  pt, pb = kernel_size - 1, stride - 1
  return x.pad(((0,0), (0,0), (pt, pb), (pl, pr)))

class ConvSubsampling:
  def __init__(self):
    self.conv = [
      nn.Conv2d(1, 256, 3, stride=2, padding=0),      # 0
      None,                                             # 1 (ReLU)
      nn.Conv2d(256, 256, 3, stride=2, padding=0, groups=256),  # 2 (depthwise)
      nn.Conv2d(256, 256, 1, stride=1, padding=0),     # 3 (pointwise)
      None,                                             # 4 (ReLU)
      nn.Conv2d(256, 256, 3, stride=2, padding=0, groups=256),  # 5 (depthwise)
      nn.Conv2d(256, 256, 1, stride=1, padding=0),     # 6 (pointwise)
    ]
    self.out = nn.Linear(256 * 17, D_MODEL)  # 17 = ceil(128/8) for 128 mels

  def __call__(self, x):
    # x: [B, 1, time, n_mels]
    x = causal_pad_2d(x, 3, 2); x = self.conv[0](x).relu()
    x = causal_pad_2d(x, 3, 2); x = self.conv[2](x)
    x = self.conv[3](x).relu()
    x = causal_pad_2d(x, 3, 2); x = self.conv[5](x)
    x = self.conv[6](x).relu()
    # x: [B, 256, H, W] where H=time/8, W=17
    B, C, H, W = x.shape
    x = x.permute(0, 2, 1, 3).reshape(B, H, C * W)  # [B, time/8, 256*17]
    return self.out(x)  # [B, time/8, d_model]

class FFN:
  def __init__(self):
    self.linear1 = nn.Linear(D_MODEL, D_FF, bias=False)
    self.linear2 = nn.Linear(D_FF, D_MODEL, bias=False)

  def __call__(self, x):
    return self.linear2(self.linear1(x).silu())

class RelPosMHA:
  def __init__(self):
    self.linear_q = nn.Linear(D_MODEL, D_MODEL, bias=False)
    self.linear_k = nn.Linear(D_MODEL, D_MODEL, bias=False)
    self.linear_v = nn.Linear(D_MODEL, D_MODEL, bias=False)
    self.linear_pos = nn.Linear(D_MODEL, D_MODEL, bias=False)
    self.linear_out = nn.Linear(D_MODEL, D_MODEL, bias=False)
    self.pos_bias_u = Tensor.zeros(N_HEADS, D_HEAD)
    self.pos_bias_v = Tensor.zeros(N_HEADS, D_HEAD)

  def __call__(self, x, pos_emb):
    B, T, D = x.shape
    q = self.linear_q(x).reshape(B, T, N_HEADS, D_HEAD).permute(0, 2, 1, 3)
    k = self.linear_k(x).reshape(B, T, N_HEADS, D_HEAD).permute(0, 2, 1, 3)
    v = self.linear_v(x).reshape(B, T, N_HEADS, D_HEAD).permute(0, 2, 1, 3)
    pos = self.linear_pos(pos_emb).reshape(-1, N_HEADS, D_HEAD).permute(1, 0, 2)  # [H, P, d]

    q_u = q + self.pos_bias_u.reshape(1, N_HEADS, 1, D_HEAD)
    q_v = q + self.pos_bias_v.reshape(1, N_HEADS, 1, D_HEAD)

    content_attn = q_u @ k.permute(0, 1, 3, 2)  # [B, H, T, T]
    pos_attn = q_v @ pos.permute(0, 2, 1)  # [B, H, T, P]
    pos_attn = self._rel_shift(pos_attn)

    scale = 1.0 / math.sqrt(D_HEAD)
    attn = (content_attn + pos_attn) * scale
    attn = attn.softmax(-1)

    ctx = (attn @ v).permute(0, 2, 1, 3).reshape(B, T, D)
    return self.linear_out(ctx)

  @staticmethod
  def _rel_shift(x):
    B, H, T, P = x.shape
    x = x.pad(((0,0), (0,0), (0,0), (1,0)))  # [B, H, T, P+1]
    x = x.reshape(B, H, P+1, T)
    x = x[:, :, 1:, :].contiguous()  # [B, H, P, T]
    x = x.reshape(B, H, T, P)
    return x[:, :, :, :T]  # [B, H, T, T]

class _Conv1dWeight:
  def __init__(self, out_ch, in_ch, kernel):
    self.weight = Tensor.zeros(out_ch, in_ch, kernel)

class ConvModule:
  def __init__(self):
    self.pointwise_conv1 = _Conv1dWeight(2*D_MODEL, D_MODEL, 1)
    self.depthwise_conv = _Conv1dWeight(D_MODEL, 1, KERNEL_SIZE)
    self.batch_norm = nn.LayerNorm(D_MODEL)
    self.pointwise_conv2 = _Conv1dWeight(D_MODEL, D_MODEL, 1)

  def __call__(self, x):
    # x: [B, T, D]
    x = x.permute(0, 2, 1).unsqueeze(2)  # [B, D, 1, T]

    # pointwise conv1 (conv1d as conv2d with H=1)
    w = self.pointwise_conv1.weight.unsqueeze(2)  # [2D, D, 1, 1]
    x = x.conv2d(w)  # [B, 2D, 1, T]

    # GLU
    a, b = x.chunk(2, dim=1)
    x = a * b.sigmoid()

    # causal depthwise conv1d: pad left by kernel-1
    x = x.pad(((0,0), (0,0), (0,0), (KERNEL_SIZE-1, 0)))
    w = self.depthwise_conv.weight.unsqueeze(2)  # [D, 1, 1, K]
    x = x.conv2d(w, groups=D_MODEL)  # [B, D, 1, T]

    x = x.squeeze(2).permute(0, 2, 1)  # [B, T, D]
    x = self.batch_norm(x).silu()
    x = x.permute(0, 2, 1).unsqueeze(2)  # [B, D, 1, T]

    # pointwise conv2
    w = self.pointwise_conv2.weight.unsqueeze(2)  # [D, D, 1, 1]
    x = x.conv2d(w)
    return x.squeeze(2).permute(0, 2, 1)  # [B, T, D]

class ConformerLayer:
  def __init__(self):
    self.norm_feed_forward1 = nn.LayerNorm(D_MODEL)
    self.feed_forward1 = FFN()
    self.norm_self_att = nn.LayerNorm(D_MODEL)
    self.self_attn = RelPosMHA()
    self.norm_conv = nn.LayerNorm(D_MODEL)
    self.conv = ConvModule()
    self.norm_feed_forward2 = nn.LayerNorm(D_MODEL)
    self.feed_forward2 = FFN()
    self.norm_out = nn.LayerNorm(D_MODEL)

  def __call__(self, x, pos_emb):
    x = x + self.feed_forward1(self.norm_feed_forward1(x)) * 0.5
    x = x + self.self_attn(self.norm_self_att(x), pos_emb)
    x = x + self.conv(self.norm_conv(x))
    x = x + self.feed_forward2(self.norm_feed_forward2(x)) * 0.5
    return self.norm_out(x)

class Encoder:
  def __init__(self):
    self.pre_encode = ConvSubsampling()
    self.layers = [ConformerLayer() for _ in range(N_LAYERS)]

  def __call__(self, mel):
    # mel: [B, T, 128]
    x = mel.reshape(mel.shape[0], 1, mel.shape[1], mel.shape[2])  # [B, 1, T, 128]
    x = self.pre_encode(x)  # [B, T/8, D]
    T = x.shape[1]
    pos_emb = compute_pos_emb(T, D_MODEL)
    for layer in self.layers:
      x = layer(x, pos_emb)
    return x

class LSTMWeights:
  def __init__(self):
    self.weight_ih_l0 = Tensor.zeros(4*DECODER_DIM, DECODER_DIM)
    self.weight_hh_l0 = Tensor.zeros(4*DECODER_DIM, DECODER_DIM)
    self.bias_ih_l0 = Tensor.zeros(4*DECODER_DIM)
    self.bias_hh_l0 = Tensor.zeros(4*DECODER_DIM)
    self.weight_ih_l1 = Tensor.zeros(4*DECODER_DIM, DECODER_DIM)
    self.weight_hh_l1 = Tensor.zeros(4*DECODER_DIM, DECODER_DIM)
    self.bias_ih_l1 = Tensor.zeros(4*DECODER_DIM)
    self.bias_hh_l1 = Tensor.zeros(4*DECODER_DIM)

class DecRNN:
  def __init__(self):
    self.lstm = LSTMWeights()

class Prediction:
  def __init__(self):
    self.embed = nn.Embedding(VOCAB_SIZE, DECODER_DIM)
    self.dec_rnn = DecRNN()

class Decoder:
  def __init__(self):
    self.prediction = Prediction()

class JointNetwork:
  def __init__(self):
    self.enc = nn.Linear(D_MODEL, JOINT_DIM)
    self.pred = nn.Linear(DECODER_DIM, JOINT_DIM)
    self.joint_net = [None, None, nn.Linear(JOINT_DIM, VOCAB_SIZE)]

class Featurizer:
  def __init__(self):
    self.fb = Tensor.zeros(N_MELS, N_FFT // 2 + 1)
    self.window = Tensor.zeros(N_WINDOW)

class PreprocessorWeights:
  def __init__(self):
    self.featurizer = Featurizer()

class NemotronASR:
  def __init__(self):
    self.encoder = Encoder()
    self.decoder = Decoder()
    self.joint = JointNetwork()
    self.preprocessor = PreprocessorWeights()

# ============================================================================
# Greedy RNN-T decoding (numpy - sequential per-step, too slow for tinygrad graph overhead)
# ============================================================================
def _sigmoid(x):
  import numpy as np
  return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))

def _lstm_cell(x, h, c, w_ih, w_hh, b_ih, b_hh):
  import numpy as np
  gates = x @ w_ih.T + b_ih + h @ w_hh.T + b_hh
  hs = h.shape[-1]
  i = _sigmoid(gates[..., :hs])
  f = _sigmoid(gates[..., hs:2*hs])
  g = np.tanh(gates[..., 2*hs:3*hs])
  o = _sigmoid(gates[..., 3*hs:])
  c_new = f * c + i * g
  h_new = o * np.tanh(c_new)
  return h_new, c_new

def greedy_decode(encoder_out, model, vocab):
  import numpy as np
  encoder_out_np = encoder_out.numpy()
  embed_w = model.decoder.prediction.embed.weight.numpy()
  lstm = model.decoder.prediction.dec_rnn.lstm
  w_ih = [lstm.weight_ih_l0.numpy(), lstm.weight_ih_l1.numpy()]
  w_hh = [lstm.weight_hh_l0.numpy(), lstm.weight_hh_l1.numpy()]
  b_ih = [lstm.bias_ih_l0.numpy(), lstm.bias_ih_l1.numpy()]
  b_hh = [lstm.bias_hh_l0.numpy(), lstm.bias_hh_l1.numpy()]
  enc_w = model.joint.enc.weight.numpy()
  enc_b = model.joint.enc.bias.numpy()
  dec_w = model.joint.pred.weight.numpy()
  dec_b = model.joint.pred.bias.numpy()
  out_w = model.joint.joint_net[2].weight.numpy()
  out_b = model.joint.joint_net[2].bias.numpy()

  time_steps = encoder_out_np.shape[1]
  h = [np.zeros(DECODER_DIM, dtype=np.float32), np.zeros(DECODER_DIM, dtype=np.float32)]
  c = [np.zeros(DECODER_DIM, dtype=np.float32), np.zeros(DECODER_DIM, dtype=np.float32)]
  prev_token = BLANK_TOKEN
  tokens = []

  for t in range(time_steps):
    enc_frame = encoder_out_np[0, t]
    enc_proj = enc_frame @ enc_w.T + enc_b
    for _sym in range(10):
      emb = embed_w[prev_token]
      h0, c0 = _lstm_cell(emb, h[0], c[0], w_ih[0], w_hh[0], b_ih[0], b_hh[0])
      h1, c1 = _lstm_cell(h0, h[1], c[1], w_ih[1], w_hh[1], b_ih[1], b_hh[1])
      dec_proj = h1 @ dec_w.T + dec_b
      logits = np.maximum(enc_proj + dec_proj, 0) @ out_w.T + out_b
      best = int(logits.argmax())
      if best == BLANK_TOKEN:
        break
      tokens.append(best)
      prev_token = best
      h[0], c[0], h[1], c[1] = h0, c0, h1, c1

  return tokens_to_text(tokens, vocab)

def tokens_to_text(token_ids, vocab):
  text = ""
  for tid in token_ids:
    if tid < 0 or tid >= len(vocab): continue
    piece = vocab[tid]
    if piece.startswith("\u2581"):  # SentencePiece word boundary
      text += " " + piece[1:]
    else:
      text += piece
  return text.strip()

# ============================================================================
# Weight loading
# ============================================================================
def load_nemo(path):
  import yaml
  with tarfile.open(path) as tar:
    config = yaml.safe_load(tar.extractfile("./model_config.yaml"))
    vocab = config['joint']['vocabulary']
    with tempfile.NamedTemporaryFile(suffix='.ckpt', delete=False) as tmp:
      tmp.write(tar.extractfile("./model_weights.ckpt").read())
      tmp_path = tmp.name
  try:
    weights = torch_load(tmp_path)
  finally:
    os.unlink(tmp_path)
  return weights, vocab

def load_gguf(path):
  """Load weights and vocab from GGUF format."""
  import numpy as np
  with open(path, 'rb') as f:
    magic = f.read(4)
    assert magic == b'GGUF', f"not a GGUF file: {magic}"
    version, = struct.unpack('<I', f.read(4))
    n_tensors, = struct.unpack('<q', f.read(8))
    n_kv, = struct.unpack('<q', f.read(8))

    def read_string():
      length, = struct.unpack('<Q', f.read(8))
      return f.read(length)

    # read KV pairs
    metadata = {}
    for _ in range(n_kv):
      key = read_string().decode('utf-8')
      vtype, = struct.unpack('<i', f.read(4))
      if vtype == 4: val, = struct.unpack('<I', f.read(4))  # uint32
      elif vtype == 5: val, = struct.unpack('<i', f.read(4))  # int32
      elif vtype == 6: val, = struct.unpack('<f', f.read(4))  # float32
      elif vtype == 8: val = read_string()  # string (bytes)
      else: raise ValueError(f"unknown GGUF kv type {vtype} for key {key}")
      metadata[key] = val

    # extract vocab from metadata
    vocab_data = metadata.get('tokenizer.vocab', b'')
    vocab_size = metadata.get('nemo.vocab_size', VOCAB_SIZE)
    vocab = []
    for i in range(vocab_size):
      entry = vocab_data[i*8:(i+1)*8]
      s = entry.split(b'\x00')[0].decode('utf-8', errors='replace')
      vocab.append(s)

    # read tensor infos
    GGML_TYPE_F32 = 0
    GGML_TYPE_F16 = 1
    GGML_TYPE_Q8_0 = 8
    GGML_TYPE_Q4_0 = 2
    tensor_infos = []
    for _ in range(n_tensors):
      name = read_string().decode('utf-8')
      n_dims, = struct.unpack('<I', f.read(4))
      dims = [struct.unpack('<q', f.read(8))[0] for _ in range(n_dims)]
      ttype, = struct.unpack('<i', f.read(4))
      offset, = struct.unpack('<Q', f.read(8))
      tensor_infos.append((name, dims, ttype, offset))

    # align to 32 bytes for data section
    pos = f.tell()
    aligned = (pos + 31) // 32 * 32
    f.seek(aligned)
    data_start = f.tell()

    # load tensors
    weights = {}
    for name, dims, ttype, offset in tensor_infos:
      # dims are in GGUF order (reversed from numpy)
      shape = list(reversed(dims))
      n_elements = 1
      for d in shape: n_elements *= d
      f.seek(data_start + offset)

      if ttype == GGML_TYPE_F32:
        data = np.frombuffer(f.read(n_elements * 4), dtype=np.float32).reshape(shape)
      elif ttype == GGML_TYPE_F16:
        data = np.frombuffer(f.read(n_elements * 2), dtype=np.float16).reshape(shape).astype(np.float32)
      elif ttype == GGML_TYPE_Q8_0:
        n_blocks = (n_elements + 31) // 32
        raw = f.read(n_blocks * 34)
        block_dt = np.dtype([('scale', np.float16), ('quants', np.int8, 32)])
        blocks = np.frombuffer(raw, dtype=block_dt)
        scales = blocks['scale'].astype(np.float32)
        quants = blocks['quants'].astype(np.float32)
        data = (quants * scales[:, None]).reshape(-1)[:n_elements].reshape(shape)
      elif ttype == GGML_TYPE_Q4_0:
        n_blocks = (n_elements + 31) // 32
        raw = f.read(n_blocks * 18)
        block_dt = np.dtype([('scale', np.float16), ('quants', np.uint8, 16)])
        blocks = np.frombuffer(raw, dtype=block_dt)
        scales = blocks['scale'].astype(np.float32)
        packed = blocks['quants']
        low = (packed & 0x0F).astype(np.int8) - 8
        high = ((packed >> 4) & 0x0F).astype(np.int8) - 8
        quants = np.concatenate([low, high], axis=1).astype(np.float32)
        data = (quants * scales[:, None]).reshape(-1)[:n_elements].reshape(shape)
      else:
        print(f"warning: skipping tensor {name} with unsupported type {ttype}")
        continue

      # handle conv weights stored as 2D in GGUF (need to restore 3D for our model)
      if 'pointwise_conv1.weight' in name and len(shape) == 2:
        data = data[:, :, np.newaxis]  # [out, in] -> [out, in, 1]
      elif 'pointwise_conv2.weight' in name and len(shape) == 2:
        data = data[:, :, np.newaxis]
      elif 'depthwise_conv.weight' in name and len(shape) == 2:
        # GGUF stores as [kernel, d_model] (transposed), restore to [d_model, 1, kernel]
        data = data.T[:, np.newaxis, :]
      elif name == 'preprocessor.featurizer.fb' and len(shape) == 3:
        data = data.squeeze(0)  # [1, n_mels, n_bins] -> [n_mels, n_bins]

      weights[name] = Tensor(data)

  return weights, vocab

def build_model(weights, vocab):
  model = NemotronASR()
  load_state_dict(model, weights, strict=False)
  return model

# ============================================================================
# Main
# ============================================================================
def load_wav(path):
  """Load WAV file as Tensor at 16kHz."""
  import wave
  import numpy as np
  with wave.open(path, 'rb') as wf:
    sr = wf.getframerate()
    nch = wf.getnchannels()
    sw = wf.getsampwidth()
    data = wf.readframes(wf.getnframes())
  if sw == 2:
    audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
  elif sw == 4:
    audio = np.frombuffer(data, dtype=np.int32).astype(np.float32) / 2147483648.0
  else:
    raise ValueError(f"unsupported sample width {sw}")
  if nch > 1:
    audio = audio.reshape(-1, nch)[:, 0]
  if sr != SAMPLE_RATE:
    ratio = SAMPLE_RATE / sr
    new_len = int(len(audio) * ratio)
    indices = np.arange(new_len) / ratio
    audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
  return Tensor(audio)

def transcribe(model, vocab, audio_path):
  audio = load_wav(audio_path)
  print(f"audio: {audio.shape[0]} samples, {audio.shape[0]/SAMPLE_RATE:.2f}s")

  filterbank = model.preprocessor.featurizer.fb
  window = model.preprocessor.featurizer.window
  mel = preprocess(audio, filterbank, window)
  print(f"mel: {mel.shape}")

  mel_t = mel.reshape(1, mel.shape[0], mel.shape[1])  # [1, T, 128]
  encoder_out = model.encoder(mel_t)
  print(f"encoder: {encoder_out.shape}")

  text = greedy_decode(encoder_out, model, vocab)
  return text

if __name__ == "__main__":
  if len(sys.argv) < 3:
    print("usage: python extra/nemotron-asr/transcribe.py <model.nemo|model.gguf> <audio.wav>")
    sys.exit(1)

  model_path = sys.argv[1]
  audio_path = sys.argv[2]

  if model_path.endswith('.gguf'):
    weights, vocab = load_gguf(model_path)
  else:
    weights, vocab = load_nemo(model_path)

  model = build_model(weights, vocab)
  text = transcribe(model, vocab, audio_path)
  print(text)
