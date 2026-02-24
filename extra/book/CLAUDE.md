# CLAUDE.md — What I Learned Building This Book

## tinygrad Architecture (10,000-foot view)

The entire framework is a pipeline: **Tensor → UOp graph → Schedule → Rangeify → Codegen → GPU kernel**.

- `Tensor` is a thin wrapper around a `UOp`. Shape, dtype, device are all derived from the UOp — the Tensor itself only stores `uop`, `requires_grad`, `grad`.
- Everything is lazy. Operations build a UOp DAG. Nothing executes until `.realize()`, `.numpy()`, or `.item()`.
- The UOp is the universal IR — a node with `(op, dtype, src, arg)`. Every transformation in tinygrad is a graph rewrite on UOps using `PatternMatcher`.

## The Key Insight: Rangeify

Shapes become loops. Movement ops (reshape, permute, expand, shrink, pad, flip) don't move data — they transform how loop variables map to buffer indices. A chain like `transpose → reshape → sum` becomes a single kernel with index arithmetic (`r0//3`, `r0%2`, etc.). This is why tinygrad can be 10K lines and still competitive.

## Scheduling & Fusion

Fusion is the most important optimization. The decision happens in `remove_bufferize` in `tinygrad/schedule/rangeify.py`. An intermediate buffer is removed (operations fused) unless:
1. Too many input buffers (>3)
2. A reduce reads from the buffer (would re-read for every reduction step)
3. User explicitly requested `.contiguous()`

Element-wise chains always fuse. Element-wise + reduce fuses. Reduce + reduce does NOT fuse.

## Autograd

Gradients are computed by graph rewriting — `pm_gradient` is a PatternMatcher that maps each op to its gradient rule. `compute_gradient` walks the graph in reverse topological order, applying chain rule at each node. The gradient of a movement op is the inverse movement op (expand → sum, reshape → reshape back, permute → inverse permute).

## Dtype System

Priority-based promotion lattice following JAX rules. Exotic types (bfloat16, fp8e4m3, fp8e5m2) have `fmt=None` — Python's struct can't handle them, so conversion is manual. Accumulation dtypes prevent overflow (uint8 sums in uint32).

## Buffer & Memory

`Buffer` is a handle to device memory. `LRUAllocator` caches freed buffers by size for reuse — GPU allocation is expensive. The memory planner (`tinygrad/engine/memory.py`) analyzes buffer lifetimes and reuses non-overlapping ones to reduce peak memory.

## Book Structure

34 chapters across 7 parts, ~8200 lines total:

| Part | Chapters | Focus |
|------|----------|-------|
| 1: Foundations | 01-03 | UOp, PatternMatcher |
| 2: Pipeline | 04-08 | Schedule, Rangeify, ShapeTracker, Codegen, BEAM |
| 3: Tricks | 09-11 | Matmul, Conv, Tensor Cores |
| 4: Hardware | 12-15 | AMD Emulator, Backends, Multi-GPU, Profiling |
| 5: Advanced | 16-19 | JIT, VIZ, Symbolic, PTX |
| 6: Models | 20-28 | MNIST through RL (9 models from first principles) |
| 7: Deep Dives | 29-34 | Tensor class, Autograd, Dtype, Buffer, Fusion, End-to-End |

## Writing Approach

Target audience: CS/ML new grads who know basic matmul and PyTorch. Each chapter:
- Explains the concept from first principles before showing tinygrad code
- Uses real, runnable code examples against the current codebase
- Includes a "Source Code Map" table pointing to exact files and line numbers
- Ends with exercises that build understanding incrementally
- Keeps technical terms in English even in the Chinese translation

## Chinese Translation

All 36 files (34 chapters + README + SUMMARY) translated to `zh/`. Rules:
- Code blocks preserved exactly as-is
- Technical terms kept in English (Tensor, UOp, kernel, realize, etc.)
- Prose, headings, exercises translated to Simplified Chinese

## Build System

`build.py` — pure Python, no external dependencies beyond what tinygrad already has (`markdown-it-py`, `pygments`). Generates:
- Multi-page HTML with prev/next navigation, syntax highlighting, dark mode
- EPUB 3.0 with proper OPF manifest, XHTML chapters, CSS
- Both English and Chinese (`--lang zh`)

Key gotcha: `markdown-it-py`'s `gfm-like` preset enables linkify by default, which requires an extra package. Must pass `linkify=False` and call `md.disable("linkify")`.

## File Layout

```
extra/book/
├── README.md              # English table of contents
├── SUMMARY.md             # mdbook-compatible chapter list
├── CLAUDE.md              # this file
├── book.toml              # mdbook config (if mdbook is available)
├── build.py               # build script (HTML + EPUB)
├── build.sh               # shell wrapper
├── 01_introduction.md     # chapters 01-34
├── ...
├── 34_endtoend.md
└── zh/                    # Chinese translation (mirrors English)
    ├── README.md
    ├── SUMMARY.md
    ├── 01_introduction.md
    ├── ...
    └── 34_endtoend.md
```
