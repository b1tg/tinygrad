# BERT-Large FP8 Matmul Benchmark Results

## Configuration

| Parameter | Value |
|-----------|-------|
| Device | AMD (MI300X, gfx942) |
| BERT hidden | 1024 |
| BERT intermediate | 4096 |
| Attention heads | 16 |
| Sequence length | 512 |
| Batch size (per GPU) | 8 |
| Tokens per batch | 4096 |
| BEAM | 3 |
| Iterations | 10 (warmup: 3) |

## FP8 vs Half Comparison

### Forward Pass (per layer, x24 layers)

| Operation | M | K | N | FP8 (ms) | FP8 TFLOPS | Half (ms) | Half TFLOPS | Speedup |
|-----------|---:|---:|---:|---:|---:|---:|---:|---:|
| QKV projection (x3) | 4096 | 1024 | 1024 | 0.88 | 9.76 | 0.90 | 9.54 | 1.02x |
| Attention output proj | 4096 | 1024 | 1024 | 0.88 | 9.75 | 0.89 | 9.66 | 1.01x |
| FFN up-projection | 4096 | 1024 | 4096 | 0.96 | 35.68 | 0.99 | 34.83 | 1.03x |
| FFN down-projection | 4096 | 4096 | 1024 | 0.97 | 35.37 | 1.03 | 33.46 | 1.06x |

### Backward Pass: grad_input (per layer, x24 layers)

| Operation | M | K | N | FP8 (ms) | FP8 TFLOPS | Half (ms) | Half TFLOPS | Speedup |
|-----------|---:|---:|---:|---:|---:|---:|---:|---:|
| QKV grad_input (x3) | 4096 | 1024 | 1024 | 0.87 | 9.88 | 0.88 | 9.74 | 1.01x |
| Attn out grad_input | 4096 | 1024 | 1024 | 0.87 | 9.85 | 0.88 | 9.79 | 1.01x |
| FFN up grad_input | 4096 | 4096 | 1024 | 0.95 | 36.08 | 1.04 | 32.91 | 1.09x |
| FFN down grad_input | 4096 | 1024 | 4096 | 0.95 | 36.26 | 0.97 | 35.55 | 1.02x |

### Backward Pass: grad_weight (per layer, x24 layers)

| Operation | M | K | N | FP8 (ms) | FP8 TFLOPS | Half (ms) | Half TFLOPS | Speedup |
|-----------|---:|---:|---:|---:|---:|---:|---:|---:|
| QKV grad_weight (x3) | 1024 | 4096 | 1024 | 0.91 | 9.46 | 0.92 | 9.29 | 1.01x |
| Attn out grad_weight | 1024 | 4096 | 1024 | 0.92 | 9.38 | 0.90 | 9.53 | 0.98x |
| FFN up grad_weight | 4096 | 4096 | 1024 | 0.97 | 35.44 | 1.07 | 32.19 | 1.10x |
| FFN down grad_weight | 1024 | 4096 | 4096 | 0.96 | 35.79 | 1.07 | 32.23 | 1.11x |

## Summary

| Metric | FP8 (fp8e4m3fnuz) | Half (fp16) |
|--------|---:|---:|
| Accumulation dtype | float32 | default (fp16) |
| 1024x1024 projections (avg TFLOPS) | 9.71 | 9.59 |
| 4096-dim FFN ops (avg TFLOPS) | 35.60 | 33.60 |
| Best single-op TFLOPS | 36.26 | 35.55 |

## Observations (Raw Matmul)

- **FP8 is ~1-11% faster** than half across all shapes with BEAM=3.
- The largest gains are in **FFN grad_weight** shapes (+10-11%), where the larger matrix dimensions benefit most from FP8's 2x theoretical throughput advantage.
- **1024x1024 projections** show minimal difference (~1%) — these are memory-bandwidth bound at this size, so the compute dtype matters less.
- **FFN layers** (4096 dimension) see consistent improvement: FP8 averages ~35.6 TFLOPS vs half's ~33.6 TFLOPS.
- The speedup is modest because MI300X's FP8 tensor core throughput is only 2x FP16, and these BERT shapes are not large enough to be fully compute-bound.

---

## End-to-End: FP8Linear vs nn.Linear (Half)

Benchmarks the actual layer implementations including quantization (abs-max, scaling, clamping, cast) and custom kernel overhead in FP8Linear.

### Forward Only (per layer, x24 layers)

| Operation | Shape | Half (ms) | Half TFLOPS | FP8Linear (ms) | FP8Linear TFLOPS | Speedup |
|-----------|-------|---:|---:|---:|---:|---:|
| QKV projection | 1024 -> 1024 | 2.22 | 3.87 | 7.38 | 1.16 | 0.30x |
| Attention output | 1024 -> 1024 | 2.23 | 3.85 | 7.34 | 1.17 | 0.30x |
| FFN up-projection | 1024 -> 4096 | 2.90 | 11.83 | 7.40 | 4.65 | 0.39x |
| FFN down-projection | 4096 -> 1024 | 3.37 | 10.19 | 7.46 | 4.61 | 0.45x |
| **Total (per layer)** | | **10.73** | | **29.58** | | **0.36x** |

### Forward + Backward (per layer, x24 layers)

| Operation | Shape | Half (ms) | Half TFLOPS | FP8Linear (ms) | FP8Linear TFLOPS | Speedup |
|-----------|-------|---:|---:|---:|---:|---:|
| QKV projection | 1024 -> 1024 | 3.35 | 7.70 | 13.73 | 1.88 | 0.24x |
| Attention output | 1024 -> 1024 | 3.32 | 7.77 | 13.87 | 1.86 | 0.24x |
| FFN up-projection | 1024 -> 4096 | 4.03 | 25.57 | 14.08 | 7.32 | 0.29x |
| FFN down-projection | 4096 -> 1024 | 4.45 | 23.15 | 14.21 | 7.26 | 0.31x |
| **Total (per layer)** | | **15.15** | | **55.88** | | **0.27x** |

### Observations (End-to-End)

- **FP8Linear is ~3-4x slower** than nn.Linear(half) when including quantization overhead.
- The quantization cost (abs-max reduction, scaling, clamping, fp8 cast) dominates — each FP8Linear call runs multiple extra kernels beyond the matmul itself.
- The backward pass is even worse (0.24-0.31x) because FP8Linear quantizes gradients too, adding more overhead on top of the custom backward kernels.
- The raw matmul speedup (1-11%) is completely negated by quantization overhead at these BERT shapes.
- Larger batch sizes or fusing quantization into the matmul kernel could help close the gap.

## Reproduce

```bash
# Raw matmul: FP8 (auto-detects fp8e4m3fnuz on MI300X)
BEAM=3 python extra/gemm/bert_fp8_matmul.py --backward --cnt 10 --warmup 3

# Raw matmul: Half precision baseline
BEAM=3 HALF=1 python extra/gemm/bert_fp8_matmul.py --backward --cnt 10 --warmup 3

# End-to-end: FP8Linear vs nn.Linear (forward only)
BEAM=3 PYTHONPATH=. python extra/gemm/bert_fp8_linear_bench.py --cnt 10 --warmup 3

# End-to-end: FP8Linear vs nn.Linear (forward + backward)
BEAM=3 PYTHONPATH=. python extra/gemm/bert_fp8_linear_bench.py --backward --cnt 10 --warmup 3
```
