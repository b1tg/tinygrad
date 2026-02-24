# Chapter 3: Pattern Matcher — The Graph Rewriting Engine

Tinygrad's entire compilation pipeline is built on one mechanism: **pattern matching and graph rewriting**. Instead of writing a traditional compiler with separate passes for parsing, optimization, and code generation, tinygrad expresses everything as "find this pattern in the UOp graph, replace it with that."

## The Core Idea

A pattern matcher takes two things:
1. A **pattern** — what to look for in the graph
2. A **rewrite function** — what to replace it with

```python
from tinygrad.uop.ops import UOp, Ops, PatternMatcher, UPat
from tinygrad.dtype import dtypes

# Pattern: find ADD(CONST(0), x) -> replace with x
# "Adding zero to anything is just the thing itself"
pm = PatternMatcher([
    (UPat(Ops.ADD, src=(UPat(Ops.CONST, arg=0), UPat.var("x"))),
     lambda x: x),
])

# Build a graph: 0 + 42
zero = UOp(Ops.CONST, dtypes.int, arg=0)
val = UOp(Ops.CONST, dtypes.int, arg=42)
expr = UOp(Ops.ADD, dtypes.int, src=(zero, val))

# Apply the pattern matcher
result = pm.rewrite(expr)
print(result.op, result.arg)  # Ops.CONST 42
```

The pattern `UPat(Ops.ADD, src=(UPat(Ops.CONST, arg=0), UPat.var("x")))` says: "match any ADD node whose first source is a CONST with value 0, and call the second source `x`." The rewrite function `lambda x: x` says: "replace the whole match with just `x`."

## UPat: The Pattern Language

`UPat` is tinygrad's pattern DSL. Here are the key forms:

```python
from tinygrad.uop.ops import UPat, Ops

# Match a specific op
UPat(Ops.ADD)                    # any ADD node

# Match with a named variable (captured for the rewrite function)
UPat(Ops.ADD, name="a")         # any ADD node, call it "a"

# Match any node
UPat.var("x")                    # any node, call it "x"

# Match a constant
UPat.cvar("c")                   # any CONST node, call it "c"

# Match with specific sources
UPat(Ops.ADD, src=(              # ADD where:
    UPat.var("x"),               #   first source is anything
    UPat(Ops.CONST, name="c"),   #   second source is a CONST
))

# Match multiple ops
UPat((Ops.ADD, Ops.MUL), name="op")  # ADD or MUL

# Match with a dtype constraint
UPat(Ops.CONST, dtype=dtypes.float)  # float CONST only
```

## graph_rewrite: Applying Patterns to a Graph

`graph_rewrite` applies a PatternMatcher to an entire UOp graph, repeatedly rewriting until no more patterns match:

```python
from tinygrad.uop.ops import UOp, Ops, PatternMatcher, UPat, graph_rewrite
from tinygrad.dtype import dtypes

# Constant folding: replace ADD(CONST, CONST) with CONST
def fold_add(a, b):
    return UOp(Ops.CONST, dtypes.int, arg=a.arg + b.arg)

pm = PatternMatcher([
    (UPat(Ops.ADD, src=(UPat(Ops.CONST, name="a"), UPat(Ops.CONST, name="b"))), fold_add),
])

# Build: (1 + 2) + (3 + 4)
c1 = UOp(Ops.CONST, dtypes.int, arg=1)
c2 = UOp(Ops.CONST, dtypes.int, arg=2)
c3 = UOp(Ops.CONST, dtypes.int, arg=3)
c4 = UOp(Ops.CONST, dtypes.int, arg=4)
expr = UOp(Ops.ADD, dtypes.int, src=(
    UOp(Ops.ADD, dtypes.int, src=(c1, c2)),
    UOp(Ops.ADD, dtypes.int, src=(c3, c4)),
))

# Apply repeatedly until fixed point
result = graph_rewrite(expr, pm)
print(result.op, result.arg)  # Ops.CONST 10
```

The `graph_rewrite` function walks the graph bottom-up (by default), trying every pattern at each node, until no more rewrites are possible (a fixed point).

## How the Renderer Works

Code generation in tinygrad is a PatternMatcher. The renderer walks the linearized UOp list and matches each node to a code emission rule:

```python
# Simplified from tinygrad/renderer/cstyle.py
# These patterns turn UOps into C code strings:

# CONST -> literal value
(UPat(Ops.CONST, name="u"), lambda u: f"{u.arg}f")

# ADD -> infix +
(UPat(Ops.ADD, name="u"), lambda u: f"({u.src[0].rendered}+{u.src[1].rendered})")

# LOAD -> pointer dereference
(UPat(Ops.LOAD, name="u"), lambda u: f"*(data+{u.src[1].rendered})")

# STORE -> pointer write
(UPat(Ops.STORE, name="u"), lambda u: f"*(data+{u.src[1].rendered}) = {u.src[2].rendered};")
```

This is a massive simplification, but the actual renderer (`tinygrad/renderer/cstyle.py`) works on exactly this principle — ~100 pattern rules that turn UOps into C/CUDA/Metal code strings.

## The Compilation Pipeline as Pattern Matchers

The entire tinygrad compiler is a sequence of pattern matcher passes. In `tinygrad/codegen/__init__.py:full_rewrite_to_sink()`:

```
Pass 1:  pm_mops + pm_syntactic_sugar    # movement op rewrites
Pass 2:  pm_load_collapse                # merge loads
Pass 3:  pm_split_ranges                 # split range expressions
Pass 4:  symbolic                        # simplify math expressions
Pass 5:  pm_simplify_ranges              # simplify range bounds
Pass 6:  apply_opts                      # BEAM search / heuristic opts
Pass 7:  expander                        # unroll loops, expand vectors
Pass 8:  pm_add_buffers_local            # add local memory buffers
Pass 9:  pm_reduce                       # lower reductions
Pass 10: pm_add_gpudims                  # assign GPU thread dimensions
Pass 11: pm_add_loads                    # add load instructions
Pass 12: devectorize                     # lower vector ops
Pass 13: decompositions                  # decompose complex ops
Pass 14: pm_final_rewrite                # final cleanup
Pass 15: pm_add_control_flow             # add loops and conditionals
```

Each pass is a `graph_rewrite(ast, some_pattern_matcher)`. The UOp graph goes in at the top as high-level tensor operations and comes out at the bottom as a flat list of GPU instructions.

## Writing Your Own Patterns

You can combine PatternMatchers with `+`:

```python
from tinygrad.uop.ops import PatternMatcher, UPat, Ops

pm1 = PatternMatcher([
    (UPat(Ops.ADD, src=(UPat(Ops.CONST, arg=0), UPat.var("x"))), lambda x: x),  # x + 0 = x
])
pm2 = PatternMatcher([
    (UPat(Ops.MUL, src=(UPat(Ops.CONST, arg=1), UPat.var("x"))), lambda x: x),  # x * 1 = x
])

combined = pm1 + pm2  # matches both patterns
```

## Returning None Means "No Match"

If a rewrite function returns `None`, the pattern is treated as not matching:

```python
pm = PatternMatcher([
    (UPat(Ops.ADD, src=(UPat(Ops.CONST, name="c"), UPat.var("x"))),
     lambda c, x: x if c.arg == 0 else None),  # only match when const is 0
])
```

This is useful for conditional rewrites — you pattern-match on structure, then check values in the function.

## Context Parameters

Some pattern matchers need shared state. You can pass a `ctx` object:

```python
from tinygrad.uop.ops import PatternMatcher, UPat, Ops, graph_rewrite

def count_adds(ctx, a):
    ctx.append(a)
    return None  # don't rewrite, just collect

pm = PatternMatcher([
    (UPat(Ops.ADD, name="a"), count_adds),
])

counts = []
graph_rewrite(some_graph, pm, ctx=counts)
print(f"Found {len(counts)} ADD nodes")
```

## Real Example: The Symbolic Simplifier

Tinygrad's symbolic math simplifier is a PatternMatcher. Here are some rules from `tinygrad/uop/symbolic.py`:

```python
# x + 0 = x
(UPat(Ops.ADD, src=(UPat.var("x"), UPat(Ops.CONST, arg=0))), lambda x: x)

# x * 0 = 0
(UPat(Ops.MUL, src=(UPat.var(), UPat(Ops.CONST, arg=0, name="z"))), lambda z: z)

# x * 1 = x
(UPat(Ops.MUL, src=(UPat.var("x"), UPat(Ops.CONST, arg=1))), lambda x: x)

# (x + c1) + c2 = x + (c1 + c2)  — constant folding through association
(UPat(Ops.ADD, src=(UPat(Ops.ADD, src=(UPat.var("x"), UPat.cvar("c1"))), UPat.cvar("c2"))),
 lambda x, c1, c2: x + UOp.const(c1.dtype, c1.arg + c2.arg))
```

These rules fire automatically during compilation, simplifying index expressions like `(r0 * 4 + 0)` to `(r0 * 4)`.

## Bottom-Up vs Top-Down

`graph_rewrite` supports two traversal orders:

- **`bottom_up=False`** (default): Process nodes from inputs toward outputs. Good for lowering passes where you want to rewrite parents after children are rewritten.
- **`bottom_up=True`**: Process nodes from outputs toward inputs. Good for structural rewrites where context from parents matters.

## Exercises

1. **Write a simplifier**: Create a PatternMatcher that simplifies `x - x` to `0` for integer UOps.

2. **Constant folder**: Extend the fold_add example to handle MUL as well. Test with `(2 * 3) + (4 * 5)`.

3. **Count patterns**: Write a PatternMatcher that counts how many LOAD, STORE, and ADD operations appear in a kernel AST. Test with `DEBUG=5` output.

4. **Read the renderer**: Open `tinygrad/renderer/cstyle.py` and find the PatternMatcher that renders `Ops.ADD`. What does the Metal renderer emit for `a + b`?

## Source Code Map

| File | What to read |
|------|-------------|
| `tinygrad/uop/ops.py` | `PatternMatcher` class, `graph_rewrite()` function |
| `tinygrad/uop/upat.py` | `UPat` — the pattern matching DSL |
| `tinygrad/uop/symbolic.py` | Symbolic math simplifier (real patterns) |
| `tinygrad/uop/decompositions.py` | Op decomposition patterns |
| `tinygrad/renderer/cstyle.py` | Code renderer as PatternMatcher |
| `tinygrad/codegen/__init__.py` | `full_rewrite_to_sink()` — the full pipeline |
