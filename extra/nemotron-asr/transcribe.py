#!/usr/bin/env python3
import argparse, math, os, pathlib, struct, sys, tarfile, tempfile
from types import SimpleNamespace as NS
from tinygrad import Tensor, TinyJit, dtypes, nn
from tinygrad.nn.state import gguf_load, load_state_dict, torch_load
try:
  from live import transcribe_live_segmented, add_live_args, validate_live_args
except ModuleNotFoundError:
  sys.path.append(str(pathlib.Path(__file__).resolve().parent))
  from live import transcribe_live_segmented, add_live_args, validate_live_args

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

def preprocess(audio_float, filterbank, window):
  audio = audio_float[1:] - PREEMPH * audio_float[:-1]
  audio = audio_float[:1].cat(audio)
  audio = audio.pad(((N_FFT // 2, N_FFT // 2),))
  pad = (N_FFT - N_WINDOW) // 2
  win = window.pad(((pad, N_FFT - N_WINDOW - pad),))
  n_samples = audio.shape[0]
  n_frames = 1 + (n_samples - N_FFT) // HOP_LENGTH
  n_bins = N_FFT // 2 + 1
  k = Tensor.arange(n_bins).reshape(n_bins, 1)
  n = Tensor.arange(N_FFT).reshape(1, N_FFT)
  angle = -2.0 * math.pi * k * n / N_FFT
  frames = audio.reshape(1, 1, -1)._pool((N_FFT,), stride=(HOP_LENGTH,)).reshape(-1, N_FFT)[:n_frames]
  frames = frames * win
  real, imag = frames @ angle.cos().T, frames @ angle.sin().T
  return (real * real + imag * imag) @ filterbank.T + LOG_ZERO_GUARD


def compute_pos_emb(max_len, d_model):
  total_len = 2 * max_len - 1
  positions = Tensor.arange(max_len - 1, -(max_len), -1, dtype=dtypes.float32)
  dims = Tensor.arange(0, d_model, 2, dtype=dtypes.float32)
  div_terms = (-dims * math.log(10000.0) / d_model).exp()
  angles = positions.reshape(-1, 1) * div_terms.reshape(1, -1)
  return angles.sin().unsqueeze(-1).cat(angles.cos().unsqueeze(-1), dim=-1).reshape(total_len, d_model)


def causal_pad_2d(x, kernel_size, stride):
  pl, pr = kernel_size - 1, stride - 1
  pt, pb = kernel_size - 1, stride - 1
  return x.pad(((0,0), (0,0), (pt, pb), (pl, pr)))

class ConvSubsampling:
  def __init__(self):
    self.conv = [
      nn.Conv2d(1, 256, 3, stride=2, padding=0),
      None,
      nn.Conv2d(256, 256, 3, stride=2, padding=0, groups=256),
      nn.Conv2d(256, 256, 1, stride=1, padding=0),
      None,
      nn.Conv2d(256, 256, 3, stride=2, padding=0, groups=256),
      nn.Conv2d(256, 256, 1, stride=1, padding=0),
    ]
    self.out = nn.Linear(256 * 17, D_MODEL)

  def __call__(self, x):
    x = causal_pad_2d(x, 3, 2); x = self.conv[0](x).relu()
    x = causal_pad_2d(x, 3, 2); x = self.conv[2](x)
    x = self.conv[3](x).relu()
    x = causal_pad_2d(x, 3, 2); x = self.conv[5](x)
    x = self.conv[6](x).relu()
    B, C, H, W = x.shape
    return self.out(x.permute(0, 2, 1, 3).reshape(B, H, C * W))

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
    pos = self.linear_pos(pos_emb).reshape(-1, N_HEADS, D_HEAD).permute(1, 0, 2)

    q_u = q + self.pos_bias_u.reshape(1, N_HEADS, 1, D_HEAD)
    q_v = q + self.pos_bias_v.reshape(1, N_HEADS, 1, D_HEAD)

    content_attn = q_u @ k.permute(0, 1, 3, 2)
    pos_attn = q_v @ pos.permute(0, 2, 1)
    pos_attn = self._rel_shift(pos_attn)
    attn = (content_attn + pos_attn) * (1.0 / math.sqrt(D_HEAD))
    attn = attn.softmax(-1)

    return self.linear_out((attn @ v).permute(0, 2, 1, 3).reshape(B, T, D))

  @staticmethod
  def _rel_shift(x):
    B, H, T, P = x.shape
    x = x.pad(((0, 0), (0, 0), (0, 0), (1, 0)))
    x = x.reshape(B, H, P+1, T)
    x = x[:, :, 1:, :].contiguous()
    x = x.reshape(B, H, T, P)
    return x[:, :, :, :T]

class ConvModule:
  def __init__(self):
    self.pointwise_conv1 = NS(weight=Tensor.zeros(2*D_MODEL, D_MODEL, 1))
    self.depthwise_conv = NS(weight=Tensor.zeros(D_MODEL, 1, KERNEL_SIZE))
    self.batch_norm = nn.LayerNorm(D_MODEL)
    self.pointwise_conv2 = NS(weight=Tensor.zeros(D_MODEL, D_MODEL, 1))

  def __call__(self, x):
    x = x.permute(0, 2, 1).unsqueeze(2)
    x = x.conv2d(self.pointwise_conv1.weight.unsqueeze(2))
    a, b = x.chunk(2, dim=1)
    x = a * b.sigmoid()
    x = x.pad(((0, 0), (0, 0), (0, 0), (KERNEL_SIZE-1, 0)))
    x = x.conv2d(self.depthwise_conv.weight.unsqueeze(2), groups=D_MODEL)
    x = x.squeeze(2).permute(0, 2, 1)
    x = self.batch_norm(x).silu()
    x = x.permute(0, 2, 1).unsqueeze(2).conv2d(self.pointwise_conv2.weight.unsqueeze(2))
    return x.squeeze(2).permute(0, 2, 1)

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
    x = self.pre_encode(mel.reshape(mel.shape[0], 1, mel.shape[1], mel.shape[2]))
    T = x.shape[1]
    pos_emb = compute_pos_emb(T, D_MODEL)
    for layer in self.layers:
      x = layer(x, pos_emb)
    return x

class NemotronASR:
  def __init__(self):
    self.encoder = Encoder()
    lstm = NS(
      weight_ih_l0=Tensor.zeros(4*DECODER_DIM, DECODER_DIM), weight_hh_l0=Tensor.zeros(4*DECODER_DIM, DECODER_DIM),
      bias_ih_l0=Tensor.zeros(4*DECODER_DIM), bias_hh_l0=Tensor.zeros(4*DECODER_DIM),
      weight_ih_l1=Tensor.zeros(4*DECODER_DIM, DECODER_DIM), weight_hh_l1=Tensor.zeros(4*DECODER_DIM, DECODER_DIM),
      bias_ih_l1=Tensor.zeros(4*DECODER_DIM), bias_hh_l1=Tensor.zeros(4*DECODER_DIM),
    )
    self.decoder = NS(prediction=NS(embed=nn.Embedding(VOCAB_SIZE, DECODER_DIM), dec_rnn=NS(lstm=lstm)))
    self.joint = NS(
      enc=nn.Linear(D_MODEL, JOINT_DIM),
      pred=nn.Linear(DECODER_DIM, JOINT_DIM),
      joint_net=[None, None, nn.Linear(JOINT_DIM, VOCAB_SIZE)],
    )
    self.preprocessor = NS(featurizer=NS(fb=Tensor.zeros(N_MELS, N_FFT // 2 + 1), window=Tensor.zeros(N_WINDOW)))


def _lstm_cell(x, h, c, w_ih, w_hh, b_ih, b_hh):
  gates = x @ w_ih.T + b_ih + h @ w_hh.T + b_hh
  hs = h.shape[-1]
  i, f, g, o = gates[..., :hs].sigmoid(), gates[..., hs:2*hs].sigmoid(), gates[..., 2*hs:3*hs].tanh(), gates[..., 3*hs:].sigmoid()
  c_new = f * c + i * g
  return o * c_new.tanh(), c_new

def _make_decoder_step(model):
  lstm = model.decoder.prediction.dec_rnn.lstm
  wl = [(lstm.weight_ih_l0, lstm.weight_hh_l0, lstm.bias_ih_l0, lstm.bias_hh_l0),
        (lstm.weight_ih_l1, lstm.weight_hh_l1, lstm.bias_ih_l1, lstm.bias_hh_l1)]
  dec_w, dec_b = model.joint.pred.weight, model.joint.pred.bias
  out_w, out_b = model.joint.joint_net[2].weight, model.joint.joint_net[2].bias
  @TinyJit
  def step(emb, h0, c0, h1, c1, enc_proj):
    h0n, c0n = _lstm_cell(emb, h0, c0, *wl[0])
    h1n, c1n = _lstm_cell(h0n, h1, c1, *wl[1])
    logits = (enc_proj + h1n @ dec_w.T + dec_b).relu() @ out_w.T + out_b
    best = logits.argmax(-1, keepdim=True)
    return best.realize(), h0n.realize(), c0n.realize(), h1n.realize(), c1n.realize()
  return step

def greedy_decode(encoder_out, model, vocab):
  step_fn = _make_decoder_step(model)
  embed_w = model.decoder.prediction.embed.weight
  enc_proj_all = (encoder_out[0] @ model.joint.enc.weight.T + model.joint.enc.bias).realize()

  time_steps = encoder_out.shape[1]
  h0, c0 = Tensor.zeros(DECODER_DIM).contiguous().realize(), Tensor.zeros(DECODER_DIM).contiguous().realize()
  h1, c1 = Tensor.zeros(DECODER_DIM).contiguous().realize(), Tensor.zeros(DECODER_DIM).contiguous().realize()
  enc_proj = Tensor.zeros(JOINT_DIM).contiguous().realize()
  emb = Tensor.zeros(DECODER_DIM).contiguous().realize()
  prev_token, tokens = BLANK_TOKEN, []

  for t in range(time_steps):
    enc_proj.assign(enc_proj_all[t:t+1].reshape(JOINT_DIM)).realize()
    for _ in range(10):
      emb.assign(embed_w[prev_token:prev_token+1].reshape(DECODER_DIM)).realize()
      best_t, h0n, c0n, h1n, c1n = step_fn(emb, h0, c0, h1, c1, enc_proj)
      best = best_t.item()
      if best == BLANK_TOKEN: break
      tokens.append(best)
      prev_token = best
      h0.assign(h0n).realize(); c0.assign(c0n).realize()
      h1.assign(h1n).realize(); c1.assign(c1n).realize()

  return tokens_to_text(tokens, vocab)

def tokens_to_text(token_ids, vocab):
  return "".join((" " + vocab[tid][1:]) if vocab[tid].startswith("\u2581") else vocab[tid]
                 for tid in token_ids if 0 <= tid < len(vocab)).strip()

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

def load_gguf_weights(path):
  from tinygrad import Device
  kv_data, state_dict = gguf_load(Tensor(pathlib.Path(path)).to(Device.DEFAULT))
  vocab_bytes = kv_data.get('tokenizer.vocab', '').encode('utf-8')
  vocab_size = kv_data.get('nemo.vocab_size', VOCAB_SIZE)
  vocab = [vocab_bytes[i*8:(i+1)*8].split(b'\x00')[0].decode('utf-8') for i in range(vocab_size)]

  weights = {}
  for name, t in state_dict.items():
    if 'pointwise_conv1.weight' in name and t.ndim == 2: t = t.unsqueeze(-1)
    elif 'pointwise_conv2.weight' in name and t.ndim == 2: t = t.unsqueeze(-1)
    elif 'depthwise_conv.weight' in name and t.ndim == 2: t = t.T.unsqueeze(1)
    elif name == 'preprocessor.featurizer.fb' and t.ndim == 3: t = t[0]
    weights[name] = t

  return weights, vocab

def load_model(model_path):
  weights, vocab = load_gguf_weights(model_path) if model_path.endswith(".gguf") else load_nemo(model_path)
  model = NemotronASR()
  load_state_dict(model, weights, strict=False)
  return model, vocab

def load_wav(path):
  import wave
  with wave.open(path, 'rb') as wf:
    sr, nch, sw = wf.getframerate(), wf.getnchannels(), wf.getsampwidth()
    raw = wf.readframes(wf.getnframes())
  n = len(raw) // sw // nch
  fmt = {2: f'<{n*nch}h', 4: f'<{n*nch}i'}[sw]
  scale = {2: 32768.0, 4: 2147483648.0}[sw]
  audio = Tensor(list(struct.unpack(fmt, raw)), dtype=dtypes.float32) / scale
  if nch > 1: audio = audio.reshape(-1, nch)[:, 0]
  if sr != SAMPLE_RATE:
    ratio = SAMPLE_RATE / sr
    new_len = int(audio.shape[0] * ratio)
    idx = Tensor.arange(new_len, dtype=dtypes.float32) / ratio
    idx_f = idx.cast(dtypes.int32)
    idx_c = (idx_f + 1).minimum(audio.shape[0] - 1)
    frac = idx - idx_f.cast(dtypes.float32)
    audio = audio[idx_f] * (1 - frac) + audio[idx_c] * frac
  return audio

def transcribe_audio(model, vocab, audio):
  f = model.preprocessor.featurizer
  mel_t = preprocess(audio, f.fb, f.window).log().reshape(1, -1, N_MELS)
  encoder_out = model.encoder(mel_t)
  return greedy_decode(encoder_out, model, vocab)

def main(argv):
  parser = argparse.ArgumentParser(description="Nemotron ASR tinygrad transcription")
  parser.add_argument("model_path", help="path to model.nemo or model.gguf")
  parser.add_argument("audio_path", nargs="?", help="path to input wav (omit when using --live)")
  parser.add_argument("--live", action="store_true", help="capture microphone input and transcribe continuously")
  add_live_args(parser)
  args = parser.parse_args(argv)
  if args.live and args.audio_path is not None: parser.error("audio_path cannot be used with --live")
  if not args.live and args.audio_path is None: parser.error("audio_path is required unless --live is set")
  validate_live_args(parser, args)
  model, vocab = load_model(args.model_path)
  if not args.live:
    print(transcribe_audio(model, vocab, load_wav(args.audio_path)))
    return
  try:
    transcribe_live_segmented(lambda audio: transcribe_audio(model, vocab, audio),
                              SAMPLE_RATE, args.chunk_seconds, args.max_chunks, args.mic_gain,
                              args.vad_threshold, args.vad_end_seconds, args.segment_max_seconds,
                              args.segment_overlap_seconds, args.vad_ratio, args.vad_preroll_seconds,
                              live_queue_chunks=args.live_queue_chunks, live_debug_level=args.live_debug_level)
  except KeyboardInterrupt:
    if args.live_debug_level >= 1: print("\nlive input stopped")
  except RuntimeError as e:
    print(f"error: {e}")
    sys.exit(1)


if __name__ == "__main__":
  main(sys.argv[1:])
