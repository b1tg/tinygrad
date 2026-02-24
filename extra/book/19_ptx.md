# Chapter 19: Low-Level PTX & LOP3

This chapter dives into the lowest level of tinygrad's code generation: NVIDIA PTX assembly and the LOP3 instruction. This is advanced material for readers who want to understand how tinygrad generates hardware-specific code.

## What is PTX?

PTX (Parallel Thread eXecution) is NVIDIA's virtual ISA — an assembly language for NVIDIA GPUs. While CUDA C++ is compiled to PTX by nvcc/nvrtc, tinygrad can generate PTX directly:

```ptx
.version 7.8
.target sm_89
.address_size 64

.visible .entry E_4(
    .param .u64 data0,
    .param .u64 data1,
    .param .u64 data2
) {
    .reg .u32 %r<10>;
    .reg .f32 %f<10>;
    .reg .u64 %rd<10>;

    mov.u32 %r0, %ctaid.x;          // block index
    ld.param.u64 %rd0, [data1];      // load pointer
    mad.wide.u32 %rd1, %r0, 4, %rd0; // address = ptr + idx*4
    ld.global.f32 %f0, [%rd1];       // load float

    ld.param.u64 %rd2, [data2];
    mad.wide.u32 %rd3, %r0, 4, %rd2;
    ld.global.f32 %f1, [%rd3];

    add.f32 %f2, %f0, %f1;          // add

    ld.param.u64 %rd4, [data0];
    mad.wide.u32 %rd5, %r0, 4, %rd4;
    st.global.f32 [%rd5], %f2;       // store result
}
```

Tinygrad's PTX renderer (`tinygrad/renderer/ptx.py`) generates this from the UOp list.

## Why PTX Instead of CUDA C++?

1. **No nvcc dependency**: PTX can be loaded directly by the NVIDIA driver, no compiler SDK needed
2. **More control**: Direct access to hardware features like WMMA, shared memory barriers, warp shuffles
3. **Smaller overhead**: No C++ parsing/compilation step
4. **Predictable output**: What you write is (approximately) what executes

## The LOP3 Instruction

LOP3 is an NVIDIA-specific instruction that computes an **arbitrary 3-input boolean function** in a single cycle. It's used for bitwise operations.

### The Idea

Any boolean function of 3 inputs (a, b, c) can be encoded as an 8-bit truth table:

```
Inputs:  a=0,b=0,c=0 | a=0,b=0,c=1 | a=0,b=1,c=0 | ... | a=1,b=1,c=1
Output:  bit 0        | bit 1        | bit 2        | ... | bit 7
```

There are 256 possible 3-input boolean functions (2^8). LOP3 takes the truth table as an immediate operand:

```ptx
lop3.b32 %r0, %r1, %r2, %r3, 0xCA;  // %r0 = LOP3(%r1, %r2, %r3, 0xCA)
```

The byte `0xCA` encodes: `(a & b) | (~a & c)` — which is actually a 2:1 multiplexer!

### Common LOP3 Encodings

| Function | Truth table | Hex |
|----------|-------------|-----|
| `a & b & c` | `10000000` | `0x80` |
| `a \| b \| c` | `11111110` | `0xFE` |
| `a ^ b ^ c` | `10010110` | `0x96` |
| `(a & b) \| c` | `11101010` | `0xEA` |
| `a ? b : c` | `11001010` | `0xCA` |
| `a & ~b` | `00001100` | `0x0C` |

### Why It Matters

Without LOP3, computing `(a & b) | (~a & c)` requires 4 instructions:
```ptx
not.b32 %t0, %r1;           // ~a
and.b32 %t1, %r1, %r2;      // a & b
and.b32 %t2, %t0, %r3;      // ~a & c
or.b32  %r0, %t1, %t2;      // (a & b) | (~a & c)
```

With LOP3, it's 1 instruction:
```ptx
lop3.b32 %r0, %r1, %r2, %r3, 0xCA;
```

4x fewer instructions, 4x fewer register reads, potentially 4x faster for bitwise-heavy code.

### SASS vs PTX

LOP3 exists in both PTX and SASS (the actual machine code). In SASS, it's even more flexible — it can combine with register moves and predication. The NVIDIA assembler translates PTX `lop3.b32` to the appropriate SASS instruction.

## How Tinygrad Uses LOP3

Tinygrad's PTX renderer detects chains of bitwise operations and fuses them into LOP3:

```python
# Before: three separate instructions
# AND(a, b) -> XOR(result, c) -> OR(result, d)

# After: fused into LOP3 where possible
# lop3.b32(a, b, c, truth_table)
```

The truth table is computed by evaluating the boolean function for all 8 input combinations.

## PTX Renderer Architecture

The PTX renderer in `tinygrad/renderer/ptx.py` differs from C-style renderers:

1. **Register allocation**: PTX uses virtual registers (`%r0`, `%f0`, `%rd0`), which the NVIDIA assembler maps to physical registers
2. **Explicit types**: Every operation specifies its type (`.f32`, `.u32`, `.b32`)
3. **Memory model**: Explicit `ld.global`, `st.global`, `ld.shared` for different address spaces
4. **Barrier instructions**: `bar.sync` for warp synchronization
5. **WMMA**: Direct access to tensor core instructions

## Exercises

1. **Compute a truth table**: For the function `(a | b) & c`, compute the 8-bit truth table. What LOP3 immediate value would you use?

2. **Read the renderer**: Open `tinygrad/renderer/ptx.py` and find how `Ops.ADD` is rendered for float32. What PTX instruction does it emit?

3. **Compare outputs**: Run `DEBUG=4` on a simple kernel using both C-style and PTX renderers (if you have NVIDIA hardware).

## Source Code Map

| File | What to read |
|------|-------------|
| `tinygrad/renderer/ptx.py` | PTX renderer — generates NVIDIA assembly |
| `tinygrad/renderer/cstyle.py` | C-style renderer for comparison |
| `tinygrad/runtime/ops_nv.py` | Raw NVIDIA driver (loads PTX directly) |
| `tinygrad/runtime/ops_cuda.py` | CUDA runtime (also loads PTX via nvrtc) |
