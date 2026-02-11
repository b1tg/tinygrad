#!/usr/bin/env python3
"""Voxtral Realtime 4B transcription — tinygrad implementation.

Usage:
  python -m extra.voxtral.transcribe <model_dir> <audio.wav>

Example:
  python -m extra.voxtral.transcribe voxtral-model extra/voxtral.c/samples/test_speech.wav
"""
import sys

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

  # Load model
  encoder, adapter, decoder = load_voxtral(model_dir)

  # Encoder
  print("Running encoder...", file=sys.stderr)
  Tensor.no_grad = True
  enc_out = encoder(mel)
  print(f"Encoder output: {enc_out.shape}", file=sys.stderr)

  # Adapter
  print("Running adapter...", file=sys.stderr)
  adapter_out = adapter(enc_out)
  print(f"Adapter output: {adapter_out.shape}", file=sys.stderr)

  # Time conditioning
  t_cond = compute_time_embedding(float(N_DELAY_TOKENS), DEC_DIM)
  print(f"Time conditioning: t={N_DELAY_TOKENS}", file=sys.stderr)

  # Decode schedule
  n_audio = adapter_out.shape[0]
  L = len(prompt_ids)

  prefix_text_embeds = decoder.embed_tokens(prompt_ids)
  prefix_embeds = adapter_out[:L] + prefix_text_embeds

  print(f"  audio_tokens={n_audio}, prefix_tokens={L}", file=sys.stderr)

  # Prefill
  print("Running decoder prefill...", file=sys.stderr)
  if L > 1:
    _ = decoder.prefill(prefix_embeds[:-1], t_cond)
  logits = decoder.forward_one(prefix_embeds[-1], pos=L - 1, t_cond=t_cond)
  token = int(logits.argmax().item())
  generated = [token]
  print(f"  Token 1 (after prefix): {token}", file=sys.stderr)

  # Generate within audio span
  print("Running decoder decode...", file=sys.stderr)
  for pos in range(L, n_audio):
    if token == TOKEN_EOS: break
    embed = adapter_out[pos] + decoder.embed_token(token)
    logits = decoder.forward_one(embed, pos=pos, t_cond=t_cond)
    token = int(logits.argmax().item())
    generated.append(token)
    if len(generated) <= 5 or len(generated) % 20 == 0:
      print(f"  Token {len(generated)} (pos={pos}): {token}", file=sys.stderr)

  print(f"Generated {len(generated)} tokens", file=sys.stderr)

  # Remove EOS
  if generated and generated[-1] == TOKEN_EOS:
    generated = generated[:-1]

  # Decode text
  decode = load_tokenizer(model_dir)
  text = decode(generated).strip()
  print(text)

if __name__ == "__main__":
  main()
