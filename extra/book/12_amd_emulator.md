# Chapter 12: The AMD GPU Emulator

This chapter teaches you how tinygrad emulates an AMD GPU entirely in software. You'll understand GPU hardware architecture at the instruction level, and see how tinygrad uses its own compiler infrastructure to build a GPU emulator that runs on any machine — including macOS laptops with no AMD hardware.

## Why an Emulator?

Tinygrad supports AMD GPUs directly — no ROCm, no HIP runtime, just raw kernel dispatch through Linux's KFD driver. But how do you test AMD GPU code when:

- Your CI runs on macOS (Apple Silicon)?
- You don't have an AMD GPU?
- You want to debug kernel execution instruction-by-instruction?

The answer: **emulate the entire GPU**. Tinygrad has two AMD emulators:

1. **Python emulator** (`test/mockgpu/amd/emu.py`): Compiles each GPU instruction to a tinygrad CPU kernel. The default.
2. **Rust emulator** (`extra/remu/`): Direct interpreter. Faster, used in CI.

Both emulate RDNA3 (and RDNA4/CDNA) instruction sets. Let's focus on the Python emulator — it's a beautiful example of self-hosting: tinygrad uses itself to emulate the hardware it runs on.

## Try It Right Now

You don't need an AMD GPU. Run this:

```bash
MOCKGPU=1 AMD=1 PYTHON_REMU=1 python -c "
from tinygrad import Tensor, Device
Device.DEFAULT = 'AMD'
a = Tensor([1.0, 2.0, 3.0, 4.0])
b = Tensor([5.0, 6.0, 7.0, 8.0])
print((a + b).numpy())
"
```

Output: `[ 6.  8. 10. 12.]`

What just happened? Tinygrad:
1. Compiled the add operation to HIP C++ code
2. Compiled that to RDNA3 machine code (via LLVM)
3. Emulated every RDNA3 instruction on your CPU
4. Wrote the results back to a buffer you can read with `.numpy()`

No GPU was involved.

## GPU Architecture in 5 Minutes

Before understanding the emulator, you need to know how a GPU actually works. Here's the minimum:

### Waves (Wavefronts)

A GPU doesn't run one thread at a time. It runs **32 threads in lockstep** — this group is called a **wave** (NVIDIA calls it a "warp"). All 32 threads execute the same instruction simultaneously, but on different data.

```
Wave = 32 threads executing the same instruction
       Thread 0: ADD v0, v1, v2  -> v0[lane0] = v1[lane0] + v2[lane0]
       Thread 1: ADD v0, v1, v2  -> v0[lane1] = v1[lane1] + v2[lane1]
       ...
       Thread 31: ADD v0, v1, v2 -> v0[lane31] = v1[lane31] + v2[lane31]
```

### Registers

Each wave has two types of registers:

**SGPRs (Scalar General Purpose Registers)**: 128 registers, shared across all 32 threads. Used for loop counters, addresses, constants — values that are the same for all threads.

**VGPRs (Vector General Purpose Registers)**: 256 registers, each containing 32 values (one per thread/lane). Used for per-thread computation.

```
SGPRs (shared):     s0=0x1000, s1=0x0000, s2=42, ...
VGPRs (per-lane):   v0 = [0, 1, 2, 3, ..., 31]   <- thread IDs
                    v1 = [_, _, _, _, ..., _]
                    v2 = [_, _, _, _, ..., _]
```

### The EXEC Mask

What if you have an `if` statement and only some threads take the branch? The **EXEC mask** is a 32-bit value where each bit controls whether a thread participates:

```
EXEC = 0b11110000_00000000_00001111_11111111
       ^ threads 28-31 active    ^ threads 0-11 active
         threads 12-27 inactive
```

When a vector instruction executes, it only writes results for threads whose EXEC bit is set.

### LDS (Local Data Share)

Shared memory within a workgroup. All waves in a workgroup can read/write the same LDS. Used for communication between threads (e.g., in reductions).

### Instruction Formats

AMD RDNA3 has ~15 instruction formats:

| Format | Example | Description |
|--------|---------|-------------|
| SOP2 | `s_add_u32 s0, s1, s2` | Scalar ALU, 2 sources |
| SOP1 | `s_mov_b32 s0, s1` | Scalar ALU, 1 source |
| SOPP | `s_branch`, `s_endpgm` | Flow control |
| SMEM | `s_load_b64 s[0:1], s[2:3]` | Scalar memory load |
| VOP2 | `v_add_f32 v0, v1, v2` | Vector ALU, 2 sources |
| VOP3 | `v_fma_f32 v0, v1, v2, v3` | Vector ALU, 3 sources |
| VOPC | `v_cmp_eq_f32 vcc, v0, v1` | Vector compare |
| DS | `ds_store_b32 v0, v1` | LDS operations |
| GLOBAL | `global_load_b32 v0, v[1:2]` | Global memory access |

Each format has a different binary encoding. The emulator must decode all of them.

## How the Python Emulator Works

### The Key Insight: Compile GPU Instructions to CPU Kernels

The Python emulator doesn't interpret instructions one by one in a Python loop (that would be way too slow). Instead, it **compiles each GPU instruction into a tinygrad UOp kernel** and runs it on the CPU backend.

This is a form of **dynamic binary translation**: GPU machine code -> UOp IR -> CPU machine code.

### Architecture Overview

```
RDNA3 machine code bytes
       |
       v
  decode_inst()           # Parse bytes into Inst object
       |
       v
  _get_runner(bytes, arch) # Compile instruction to CPU kernel
       |                   # Uses _INST_HANDLERS dispatch table
       v                   # Builds UOp graph via _Ctx
  get_runner('CPU', sink)  # Tinygrad compiles UOps to Clang C
       |
       v
  Cached CPU function      # Called with register buffer pointers
```

### WaveState: The Emulator's Register File

The `WaveState` class holds the complete state of one wavefront:

```python
from test.mockgpu.amd.emu import WaveState, EXEC_LO, SGPR_COUNT, VGPR_SIZE

ws = WaveState(32)  # 32-lane wave

# SGPRs: 260 x uint32
# Slots 0-127:   actual SGPRs
# Slots 128-255: inline constants (0, 1, 2, ..., 64, -1, -2, ..., -16, 0.5, 1.0, ...)
# Slots 256-259: special (PC_LO, PC_HI, SCC, SCRATCH_STRIDE)
print(f"SGPR buffer: {SGPR_COUNT} uint32s")

# VGPRs: 256 * 32 = 8192 x uint32
# Layout: vgpr[reg_num * 32 + lane_id]
print(f"VGPR buffer: {VGPR_SIZE} uint32s (256 regs x 32 lanes)")

# Check initial state
print(f"PC: {ws.pc}")                                    # 0
print(f"EXEC: {ws._read_sgpr(EXEC_LO.offset):#010x}")   # 0xffffffff (all lanes active)
print(f"Inline const[129] (=1): {ws._read_sgpr(129)}")   # 1
print(f"Inline const[193] (=-1): {ws._read_sgpr(193):#010x}")  # 0xffffffff
```

The key insight is that SGPRs and VGPRs are stored as flat `Buffer` objects — regular tinygrad CPU buffers. When a compiled instruction runs, it reads and writes these buffers directly.

### _Ctx: Building UOp Kernels for Instructions

The `_Ctx` class is the heart of the emulator. It defines five buffer PARAMs that every compiled instruction receives:

```python
class _Ctx:
    sgpr = UOp(Ops.PARAM, dtypes.uint32.ptr(260), arg=0)      # Scalar registers
    vgpr = UOp(Ops.PARAM, dtypes.uint32.ptr(8192), arg=1)     # Vector registers
    vmem = UOp(Ops.PARAM, dtypes.uint32.ptr(1<<46), arg=2)    # Host memory (!)
    lds  = UOp(Ops.PARAM, dtypes.uint32.ptr(16384), arg=3)    # Local data share
    scratch = UOp(Ops.PARAM, dtypes.uint8.ptr(1<<30), arg=4)  # Per-lane scratch
```

Notice `vmem` — parameter 2 maps to **all of host memory** (starting at virtual address 0). This is how the emulator accesses tensor data: GPU global memory loads become direct reads from the host process's address space.

### The EXEC Mask as a RANGE Loop

For vector instructions (VOP1/VOP2/VOP3), the emulator creates a `RANGE` loop over 32 lanes:

```python
# Inside the emulator's instruction compiler:
lane = ctx.range(32)  # UOp.range(32, ...)

# Read VGPR: index = reg_num * 32 + lane
v0 = ctx.vgpr.index(0 * 32 + lane, ptr=True).load()
v1 = ctx.vgpr.index(1 * 32 + lane, ptr=True).load()

# Compute
result = v0 + v1

# Write VGPR only if lane is active (EXEC mask check)
exec_mask = ctx.sgpr.index(EXEC_LO, ptr=True).load()
active = ((exec_mask >> lane.cast(dtypes.uint32)) & 1).ne(0)
ctx.vgpr.index(2 * 32 + lane, active).store(result)
```

This UOp graph gets compiled by tinygrad's CPU backend into efficient C code with a loop over 32 lanes. The EXEC mask check becomes a conditional store.

### Pcode: AMD's Official Instruction Semantics

How does the emulator know what `V_ADD_F32` does? It uses AMD's own pseudocode from the ISA reference manuals. These are stored in autogenerated files:

```python
from test.mockgpu.amd.emu import get_pcode
from tinygrad.runtime.autogen.amd.rdna3.enum import VOP2Op, SOP2Op, VOP3Op

print(get_pcode(VOP2Op.V_ADD_F32_E32))
# Output: D0.f32 = S0.f32 + S1.f32

print(get_pcode(VOP2Op.V_MUL_F32_E32))
# Output: D0.f32 = S0.f32 * S1.f32

print(get_pcode(SOP2Op.S_ADD_U32))
# Output: tmp = 64'U(S0.u32) + 64'U(S1.u32);
#         SCC = tmp >= 0x100000000ULL ? 1'1U : 1'0U;
#         D0.u32 = tmp.u32

print(get_pcode(VOP3Op.V_FMA_F32))
# Output: D0.f32 = fma(S0.f32, S1.f32, S2.f32)
```

The `parse_pcode()` function in `test/mockgpu/amd/pcode.py` tokenizes this pseudocode and converts it to UOp expressions. So `D0.f32 = S0.f32 + S1.f32` becomes `UOp(Ops.ADD, dtypes.float32, (src0, src1))`.

This is remarkably elegant: instead of hardcoding hundreds of instruction semantics, the emulator derives them from AMD's own documentation.

### Canonical Caching

Compiling every instruction to a CPU kernel sounds expensive. The emulator avoids redundant compilation with a canonical cache:

```python
# Two different v_add_f32 instructions with different registers:
# v_add_f32 v0, v1, v2
# v_add_f32 v5, v6, v7
#
# They have different register fields but identical semantics.
# The emulator masks out the dynamic fields (register numbers)
# and uses (base_bits, mask, size) as the cache key.
#
# Result: only ONE compilation, the register numbers are
# extracted dynamically at runtime via ctx.inst_field().
```

The `canonical_mask()` method computes which bits of an instruction are "static" (opcode, format) vs "dynamic" (register numbers, offsets). Instructions with the same static bits share a compiled runner.

## The Execution Loop

When a kernel is dispatched, `run_asm()` orchestrates the full execution:

```python
def run_asm(lib, lib_sz, gx, gy, gz, lx, ly, lz, args_ptr, ...):
    # lib = pointer to RDNA3 machine code in host memory
    # gx,gy,gz = grid dimensions (number of workgroups)
    # lx,ly,lz = block dimensions (threads per workgroup)

    for gidz in range(gz):
      for gidy in range(gy):
        for gidx in range(gx):
          # Initialize all waves for this workgroup
          waves = []
          for wave_start in range(0, lx*ly*lz, 32):
            ws = WaveState(min(32, total_threads - wave_start))
            ws.pc = lib  # point PC at the kernel code
            # Set up thread IDs in v0, workgroup IDs in SGPRs
            waves.append(ws)

          # Execute with barrier synchronization
          for wi, ws in enumerate(waves):
            while ws.pc != ENDPGM:
              # Compile and run one instruction
              fxn, globals_list, is_barrier, inst = _ensure_compiled(ws.pc)
              fxn(*[c_bufs[g] for g in globals_list])

              if is_barrier:
                break  # pause until all waves hit barrier
```

Key points:
- Each workgroup gets fresh LDS (zeroed between workgroups)
- Waves within a workgroup synchronize at `s_barrier`
- The `_ensure_compiled()` function lazily compiles instructions on first encounter
- DAZ+FTZ (Denormals Are Zero, Flush To Zero) is enabled during execution to match GPU float behavior

## The MockGPU Driver

The emulator doesn't just emulate instructions — it emulates the entire AMD driver stack. `test/mockgpu/amd/amdgpu.py` intercepts:

- **PM4 packets**: The command format AMD GPUs use. When tinygrad submits a `PACKET3_DISPATCH_DIRECT`, the mock GPU intercepts it and calls `run_asm()`.
- **Memory management**: Buffer allocations become regular CPU allocations.
- **KFD ioctls**: The Linux kernel interface for AMD GPUs is intercepted in `amddriver.py`.

This means the entire tinygrad AMD backend (`tinygrad/runtime/ops_amd.py`) runs unmodified — it thinks it's talking to a real GPU.

## Running a Kernel Step by Step

Let's trace what happens when you add two tensors on the emulated GPU:

```bash
DEBUG=4 MOCKGPU=1 AMD=1 PYTHON_REMU=1 python -c "
from tinygrad import Tensor, Device
Device.DEFAULT = 'AMD'
a = Tensor([1.0, 2.0, 3.0, 4.0])
b = Tensor([5.0, 6.0, 7.0, 8.0])
c = (a + b).numpy()
print(c)
"
```

With `DEBUG=4`, you'll see the generated HIP C++ kernel:

```c
extern "C" __attribute__((global))
void __attribute__((amdgpu_flat_work_group_size(1, 1)))
E_4(float* data0_4, float* data1_4, float* data2_4) {
  float4 val0 = (*((float4*)((data1_4+0))));
  float4 val1 = (*((float4*)((data2_4+0))));
  *((float4*)((data0_4+0))) = float4(
    (val0.x+val1.x), (val0.y+val1.y),
    (val0.z+val1.z), (val0.w+val1.w));
}
```

This HIP code is compiled to RDNA3 machine code by LLVM. The emulator then:

1. **Decodes** each instruction from the binary
2. **Compiles** it to a UOp graph (only on first encounter)
3. **Executes** the compiled function on CPU, passing SGPR/VGPR buffers
4. **Advances** the PC to the next instruction
5. **Repeats** until `s_endpgm`

With `DEBUG=3`, you can see each instruction being compiled:

```
[emu] PC=0: s_load_b64(...)       # load kernel arguments
[emu] PC=8: s_waitcnt(...)        # wait for memory
[emu] PC=12: global_load_b128(...) # load 4 floats from data1
[emu] PC=20: global_load_b128(...) # load 4 floats from data2
[emu] PC=28: s_waitcnt(...)        # wait for loads
[emu] PC=32: v_add_f32(...)        # add element 0
[emu] PC=36: v_add_f32(...)        # add element 1
[emu] PC=40: v_add_f32(...)        # add element 2
[emu] PC=44: v_add_f32(...)        # add element 3
[emu] PC=48: global_store_b128(...)# store 4 results
[emu] PC=56: s_endpgm              # done
```

## Understanding the Instruction Compilation Pipeline

Let's look at how a single `v_add_f32` instruction is compiled. The handler is `_compile_vop12`:

```
1. Decode instruction bytes -> VOP2 object with op=V_ADD_F32_E32
2. Look up pcode: "D0.f32 = S0.f32 + S1.f32"
3. Read source operands:
   - S0 = ctx.rsrc(inst.src0, lane)     # may be SGPR or VGPR
   - S1 = ctx.rvgpr(inst.vsrc1, lane)   # always VGPR for VOP2
4. Parse pcode to UOp: result = S0.bitcast(float32) + S1.bitcast(float32)
5. Write destination:
   - ctx.wvgpr(inst.vdst, lane, result, exec_mask)
6. Create SINK with all stores
7. Call get_runner('CPU', sink) -> compiled C function
```

The compiled function signature is essentially:

```c
void compiled_v_add_f32(uint32_t* sgpr, uint32_t* vgpr, uint32_t* vmem, ...) {
    uint32_t exec = sgpr[EXEC_LO];
    for (int lane = 0; lane < 32; lane++) {
        if (!((exec >> lane) & 1)) continue;
        float s0 = *(float*)&vgpr[src0 * 32 + lane];
        float s1 = *(float*)&vgpr[src1 * 32 + lane];
        *(float*)&vgpr[dst * 32 + lane] = s0 + s1;
    }
}
```

## The Rust Emulator (Remu)

The Rust emulator at `extra/remu/` takes a simpler approach: direct interpretation.

```rust
// Simplified from extra/remu/src/thread.rs
fn interpret(&mut self, inst: Instruction) {
    match inst {
        Instruction::VOP2 { op, vdst, vsrc, src } => {
            for lane in 0..32 {
                if !self.exec.read(lane) { continue; }
                let s0 = self.read_src(src, lane);
                let s1 = self.vgprs[vsrc][lane];
                self.vgprs[vdst][lane] = match op {
                    VOP2Op::V_ADD_F32 => f32_add(s0, s1),
                    VOP2Op::V_MUL_F32 => f32_mul(s0, s1),
                    // ... hundreds more
                };
            }
        }
        // ... other formats
    }
}
```

It's faster but less maintainable — every new instruction requires manually implementing its semantics. The Python emulator derives semantics from pcode automatically.

The test `test/amd/test_compare_emulators.py` runs real tinygrad kernels through both emulators instruction-by-instruction, comparing all register state to ensure they agree.

## SQTT Tracing

When `PROFILE=1`, the Python emulator generates AMD Shader Thread Trace (SQTT) packets. These are the same binary format that real AMD GPUs produce for their performance profiling tools. This means tinygrad's profiling infrastructure works identically whether you're on real hardware or the emulator.

```python
# The emulator generates SQTT packets for each executed instruction:
def emit(wave_id, inst, branch_taken):
    # Classify instruction type (SALU, VALU, SMEM, LDS, GLOBAL, etc.)
    # Emit corresponding SQTT packet with timing delta
    _emit_nibbles(nibbles, INST, delta=1, wave=wave_id & 0x1F, op=inst_op)
```

## Key Design Decisions

### 1. Self-Hosting Compilation

The Python emulator compiles GPU instructions using tinygrad's own CPU backend. This is circular in the best way — tinygrad uses itself to emulate the hardware it generates code for. It also means improvements to tinygrad's code generation automatically improve the emulator.

### 2. Pcode-Driven Semantics

By parsing AMD's official pseudocode, the emulator stays correct without manually implementing hundreds of instruction variants. When AMD adds a new instruction, you just need the pcode string.

### 3. DAZ+FTZ Floating Point

GPUs don't handle denormalized floats the same way CPUs do. The emulator sets the CPU's `MXCSR` (x86) or `FPCR` (ARM64) register to match GPU behavior during execution, then restores it afterward.

### 4. Zero-Based Virtual Memory

The `vmem` parameter is a Buffer with `external_ptr=0`, meaning index 0 maps to physical address 0. GPU global memory loads use the raw virtual addresses from kernel arguments as indices into this buffer. Since the emulator runs in the same process, these addresses are valid host pointers.

## Hands-On Exercises

### Exercise 1: Run the Emulator

```bash
MOCKGPU=1 AMD=1 PYTHON_REMU=1 python -c "
from tinygrad import Tensor, Device
Device.DEFAULT = 'AMD'
print((Tensor.ones(4,4) @ Tensor.ones(4,4)).numpy())
"
```

Verify you get a 4x4 matrix of 4.0s.

### Exercise 2: See the Instructions

```bash
DEBUG=3 MOCKGPU=1 AMD=1 PYTHON_REMU=1 python -c "
from tinygrad import Tensor, Device
Device.DEFAULT = 'AMD'
print(Tensor([1.0, 2.0]).sum().item())
"
```

Count how many instructions the emulator compiles. How many are scalar vs vector?

### Exercise 3: Inspect Pcode

```python
from test.mockgpu.amd.emu import get_pcode
from tinygrad.runtime.autogen.amd.rdna3.enum import VOP2Op, VOP3Op

# What does v_cndmask_b32 do?
print(get_pcode(VOP2Op.V_CNDMASK_B32_E32))

# What about v_fma_f32?
print(get_pcode(VOP3Op.V_FMA_F32))
```

### Exercise 4: Create a WaveState

```python
from test.mockgpu.amd.emu import WaveState, EXEC_LO

ws = WaveState(32)
# Write a value to VGPR v5, lane 0
ws._write_vgpr(5, 0, 42)
print(f"v5[lane0] = {ws._read_vgpr(5, 0)}")

# Disable lane 0 in EXEC mask
exec_val = ws._read_sgpr(EXEC_LO.offset)
ws._write_sgpr(EXEC_LO.offset, exec_val & ~1)
print(f"EXEC = {ws._read_sgpr(EXEC_LO.offset):#010x}")
# Now lane 0 won't participate in vector instructions
```

## Source Code Map

| File | What to read |
|------|-------------|
| `test/mockgpu/amd/emu.py` | Main emulator: `WaveState`, `_Ctx`, instruction handlers, `run_asm` |
| `test/mockgpu/amd/emu.py:402` | `_Ctx` class — the UOp builder for instruction compilation |
| `test/mockgpu/amd/emu.py:1382` | `WaveState` — wavefront register state |
| `test/mockgpu/amd/emu.py:1445` | `run_asm()` — the main execution loop |
| `test/mockgpu/amd/pcode.py` | Pseudocode tokenizer and parser |
| `test/mockgpu/amd/amdgpu.py` | MockGPU driver — PM4 packet handling |
| `test/mockgpu/amd/amddriver.py` | KFD ioctl interception |
| `tinygrad/renderer/amd/` | Instruction encoding/decoding, DSL, SQTT |
| `tinygrad/runtime/autogen/amd/rdna3/` | Autogenerated instruction classes, opcodes, pcode strings |
| `extra/remu/src/` | Rust emulator source |
| `test/amd/test_compare_emulators.py` | Cross-validation between Python and Rust emulators |
