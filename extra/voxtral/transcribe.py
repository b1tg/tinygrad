#!/usr/bin/env python3
"""Voxtral Realtime 4B transcription — tinygrad implementation.

Usage:
  python -m extra.voxtral.transcribe <model_dir> <audio.wav>

Example:
  python -m extra.voxtral.transcribe voxtral-model extra/voxtral.c/samples/test_speech.wav
"""
import sys, time

def main():
  if len(sys.argv) < 3:
    print(f"Usage: {sys.argv[0]} <model_dir> <audio.wav>", file=sys.stderr)
    sys.exit(1)

  model_dir = sys.argv[1]
  wav_path = sys.argv[2]

  from tinygrad import Tensor
  from extra.voxtral import (load_voxtral, compute_time_embedding,
                              TOKEN_BOS, TOKEN_EOS, TOKEN_STREAMING_PAD, N_LEFT_PAD_TOKENS, N_DELAY_TOKENS,
                              DEC_DIM, SAMPLE_RATE)
  from extra.voxtral.audio import load_wav, resample_linear, pad_audio_streaming, compute_mel_spectrogram, compute_mel_filters
  from extra.voxtral.tokenizer import load_tokenizer

  t0 = time.time()

  # Load audio
  audio, sr = load_wav(wav_path)
  if sr != SAMPLE_RATE:
    print(f"Resampling from {sr} to {SAMPLE_RATE} Hz", file=sys.stderr)
    audio = resample_linear(audio, sr, SAMPLE_RATE)
  print(f"Audio: {len(audio)} samples ({len(audio)/SAMPLE_RATE:.1f}s)", file=sys.stderr)

  # Pad audio
  prompt_ids = [TOKEN_BOS] + [TOKEN_STREAMING_PAD] * (N_LEFT_PAD_TOKENS + N_DELAY_TOKENS)
  padded = pad_audio_streaming(audio)
  print(f"Audio padded: {len(padded)} samples ({len(padded)/SAMPLE_RATE:.1f}s)", file=sys.stderr)

  # Mel spectrogram
  mel_filters = compute_mel_filters()
  audio_tensor = Tensor(padded)
  mel = compute_mel_spectrogram(audio_tensor, mel_filters)
  print(f"Mel: {mel.shape[1]} frames", file=sys.stderr)

  # Truncate if odd number of frames (conv stride=2)
  if mel.shape[1] % 2 != 0:
    mel = mel[:, 1:]
    print(f"Mel truncated to {mel.shape[1]} frames", file=sys.stderr)

  t_mel = time.time()
  print(f"Mel spectrogram: {t_mel - t0:.2f}s", file=sys.stderr)

  # Load model
  n_audio_estimate = mel.shape[1] // 8 + 64
  encoder, adapter, decoder = load_voxtral(model_dir, max_context=max(n_audio_estimate, 256))

  t_load = time.time()
  print(f"Model load: {t_load - t_mel:.2f}s", file=sys.stderr)

  # Encoder
  print("Running encoder...", file=sys.stderr)
  Tensor.no_grad = True
  enc_out = encoder(mel)
  print(f"Encoder output: {enc_out.shape}", file=sys.stderr)

  t_enc = time.time()
  print(f"Encoder: {t_enc - t_load:.2f}s", file=sys.stderr)

  # Adapter
  adapter_out = adapter(enc_out)
  print(f"Adapter output: {adapter_out.shape}", file=sys.stderr)

  # Time conditioning
  t_cond = compute_time_embedding(float(N_DELAY_TOKENS), DEC_DIM)
  t_ada = time.time()

  # Decode schedule
  n_audio = adapter_out.shape[0]
  L = len(prompt_ids)

  prefix_text_embeds = decoder.embed_tokens(prompt_ids)
  prefix_embeds = adapter_out[:L] + prefix_text_embeds

  print(f"  audio_tokens={n_audio}, prefix_tokens={L}", file=sys.stderr)

  # Prefill all prefix tokens except the last one
  print("Running decoder prefill...", file=sys.stderr)
  if L > 1:
    _ = decoder.prefill(prefix_embeds[:-1], t_cond)

  # First decode: last prefix token (no JIT yet - different shape on first call)
  token = decoder.decode_token(prefix_embeds[-1], pos=L - 1, t_cond=t_cond, use_jit=False)
  generated = [token]
  t_prefill = time.time()
  print(f"Prefill: {t_prefill - t_ada:.2f}s", file=sys.stderr)
  print(f"  Token 1 (after prefix): {token}", file=sys.stderr)

  # Generate within audio span (JIT kicks in after warmup)
  print("Running decoder decode...", file=sys.stderr)
  for pos in range(L, n_audio):
    if token == TOKEN_EOS: break
    embed = adapter_out[pos] + decoder.embed_token(token)
    token = decoder.decode_token(embed, pos=pos, t_cond=t_cond, use_jit=True)
    generated.append(token)
    if len(generated) <= 5 or len(generated) % 20 == 0:
      print(f"  Token {len(generated)} (pos={pos}): {token}", file=sys.stderr)

  t_decode = time.time()
  n_tokens = len(generated)
  print(f"Decode: {t_decode - t_prefill:.2f}s ({n_tokens} tokens, {(t_decode-t_prefill)/n_tokens*1000:.0f} ms/tok)", file=sys.stderr)
  print(f"Total: {t_decode - t0:.2f}s", file=sys.stderr)

  # Remove EOS
  if generated and generated[-1] == TOKEN_EOS:
    generated = generated[:-1]

  # Decode text
  decode = load_tokenizer(model_dir)
  text = decode(generated).strip()
  print(text)

if __name__ == "__main__":
  main()
