# Chapter 17: VIZ — The Graph Visualizer

When debugging tinygrad, you often need to see the UOp graph — what nodes exist, how they connect, and how each rewrite pass transforms them. VIZ is tinygrad's built-in interactive graph visualizer.

## Quick Start

```bash
VIZ=1 python -c "
from tinygrad import Tensor
(Tensor.ones(4, 4) + Tensor.ones(4, 4)).realize()
"
```

This opens a web browser with an interactive visualization of the UOp graph at each stage of compilation. You can:
- Navigate between rewrite passes
- Click on nodes to see their details
- See before/after for each pattern match
- Zoom and pan the graph

## What VIZ Shows

VIZ captures the UOp graph at each `graph_rewrite` pass. For a simple `a + b`:

```
Pass 1: "earliest rewrites"     - high-level tensor graph
Pass 2: "rangeify"              - movement ops -> ranges
Pass 3: "symbolic+debuf"        - symbolic simplification
Pass 4: "bufferize to store"    - add store instructions
Pass 5: "split kernels"         - split into individual kernels
Pass 6: "early movement ops"    - lower movement ops
Pass 7: "symbolic"              - more simplification
Pass 8: "apply_opts"            - BEAM/heuristic optimization
Pass 9: "expander"              - unroll loops
Pass 10: "add gpudims"          - assign thread dimensions
Pass 11: "devectorize"          - lower vectors
Pass 12: "decompositions"       - decompose ops
Pass 13: "final rewrite"        - cleanup
Pass 14: "add control flow"     - insert loops/conditions
```

You can step through each pass and see exactly how the graph transforms.

## Reading the Graph

Nodes are colored by their operation type:
- **Green**: Memory operations (LOAD, STORE)
- **Blue**: Math operations (ADD, MUL, etc.)
- **Red**: Control flow (RANGE, END, IF)
- **Gray**: Constants and parameters

Edges show data dependencies — arrows point from inputs to outputs.

## VIZ for Debugging

VIZ is particularly useful when:

1. **A kernel produces wrong results**: Step through the passes to find where the graph diverges from what you expect.

2. **A pattern match isn't firing**: VIZ shows which patterns matched and what they replaced. If your pattern didn't match, you can see why.

3. **Understanding optimization**: See what BEAM search or heuristics do to the graph structure.

```bash
# See optimization applied to a matmul
VIZ=1 python -c "
from tinygrad import Tensor
(Tensor.ones(64, 64) @ Tensor.ones(64, 64)).realize()
"
```

## The Implementation

VIZ works by hooking into `graph_rewrite`. When `VIZ=1`, each `graph_rewrite` call records the before/after state of the UOp graph. These snapshots are served via a local web server using dagre (a JavaScript graph layout library) for visualization.

The visualizer code lives in `tinygrad/viz/`.

## Exercises

1. **Explore a matmul**: Run `VIZ=1` on a 4x4 matmul. In the "rangeify" pass, find the RANGE nodes and trace how they map to the reduction loop.

2. **Compare passes**: Look at the graph before and after "apply_opts". What changed?

3. **Debug a bug**: If you ever write a custom pattern matcher, use VIZ to verify it's matching and replacing correctly.

## Source Code Map

| File | What to read |
|------|-------------|
| `tinygrad/viz/` | Visualizer implementation |
| `tinygrad/uop/ops.py` | `graph_rewrite()` with VIZ hooks |
| `tinygrad/helpers.py` | `VIZ` environment variable |
