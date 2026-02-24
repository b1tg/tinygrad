# 第12章：AMD GPU 模拟器

本章将教你 tinygrad 如何完全用软件模拟一个 AMD GPU。你将在指令级别理解 GPU 硬件架构，并看到 tinygrad 如何利用自身的编译器基础设施来构建一个可以在任何机器上运行的 GPU 模拟器——包括没有 AMD 硬件的 macOS 笔记本电脑。

## 为什么需要模拟器？

Tinygrad 直接支持 AMD GPU——不需要 ROCm，不需要 HIP 运行时，只需要通过 Linux 的 KFD 驱动进行原始的内核调度。但是，当遇到以下情况时，你如何测试 AMD GPU 代码？

- 你的 CI 运行在 macOS (Apple Silicon) 上？
- 你没有 AMD GPU？
- 你想要逐指令调试内核执行？

答案是：**模拟整个 GPU**。Tinygrad 有两个 AMD 模拟器：

1. **Python 模拟器** (`test/mockgpu/amd/emu.py`)：将每条 GPU 指令编译为 tinygrad CPU 内核。这是默认选项。
2. **Rust 模拟器** (`extra/remu/`)：直接解释执行。速度更快，用于 CI。

两者都模拟 RDNA3（以及 RDNA4/CDNA）指令集。让我们聚焦于 Python 模拟器——它是自举（self-hosting）的一个精彩示例：tinygrad 使用自身来模拟它所运行的硬件。

## 立即试一试

你不需要 AMD GPU。运行以下命令：

```bash
MOCKGPU=1 AMD=1 PYTHON_REMU=1 python -c "
from tinygrad import Tensor, Device
Device.DEFAULT = 'AMD'
a = Tensor([1.0, 2.0, 3.0, 4.0])
b = Tensor([5.0, 6.0, 7.0, 8.0])
print((a + b).numpy())
"
```

输出：`[ 6.  8. 10. 12.]`

刚才发生了什么？Tinygrad：
1. 将加法运算编译为 HIP C++ 代码
2. 将其编译为 RDNA3 机器码（通过 LLVM）
3. 在你的 CPU 上模拟执行了每条 RDNA3 指令
4. 将结果写回到一个你可以通过 `.numpy()` 读取的缓冲区中

整个过程没有涉及任何 GPU。

## 5 分钟了解 GPU 架构

在理解模拟器之前，你需要知道 GPU 实际上是如何工作的。以下是最基本的内容：

### 波前（Wavefronts）

GPU 不是一次运行一个线程。它以**32 个线程同步执行**——这个组被称为**波前（wave）**（NVIDIA 称之为 "warp"）。所有 32 个线程同时执行相同的指令，但处理不同的数据。

```
波前 = 32 个线程执行相同指令
       线程 0:  ADD v0, v1, v2  -> v0[lane0] = v1[lane0] + v2[lane0]
       线程 1:  ADD v0, v1, v2  -> v0[lane1] = v1[lane1] + v2[lane1]
       ...
       线程 31: ADD v0, v1, v2 -> v0[lane31] = v1[lane31] + v2[lane31]
```

### 寄存器

每个波前有两种类型的寄存器：

**SGPR（标量通用寄存器）**：128 个寄存器，在所有 32 个线程之间共享。用于循环计数器、地址、常量——即所有线程相同的值。

**VGPR（向量通用寄存器）**：256 个寄存器，每个包含 32 个值（每个线程/lane 一个）。用于逐线程的计算。

```
SGPRs（共享）：     s0=0x1000, s1=0x0000, s2=42, ...
VGPRs（逐 lane）：  v0 = [0, 1, 2, 3, ..., 31]   <- 线程 ID
                    v1 = [_, _, _, _, ..., _]
                    v2 = [_, _, _, _, ..., _]
```

### EXEC 掩码

如果你有一个 `if` 语句，但只有部分线程进入该分支怎么办？**EXEC 掩码**是一个 32 位值，每个位控制对应线程是否参与：

```
EXEC = 0b11110000_00000000_00001111_11111111
       ^ 线程 28-31 活跃        ^ 线程 0-11 活跃
         线程 12-27 不活跃
```

当一条向量指令执行时，它只为 EXEC 位被设置的线程写入结果。

### LDS（本地数据共享）

工作组内的共享内存。同一工作组中的所有波前可以读写相同的 LDS。用于线程间通信（例如，在归约操作中）。

### 指���格式

AMD RDNA3 有约 15 种指令格式：

| 格式 | 示例 | 描述 |
|--------|---------|-------------|
| SOP2 | `s_add_u32 s0, s1, s2` | 标量 ALU，2 个源操作数 |
| SOP1 | `s_mov_b32 s0, s1` | 标量 ALU，1 个源操作数 |
| SOPP | `s_branch`, `s_endpgm` | 流程控制 |
| SMEM | `s_load_b64 s[0:1], s[2:3]` | 标量内存加载 |
| VOP2 | `v_add_f32 v0, v1, v2` | 向量 ALU，2 个源操作数 |
| VOP3 | `v_fma_f32 v0, v1, v2, v3` | 向量 ALU，3 个源操作数 |
| VOPC | `v_cmp_eq_f32 vcc, v0, v1` | 向量比较 |
| DS | `ds_store_b32 v0, v1` | LDS 操作 |
| GLOBAL | `global_load_b32 v0, v[1:2]` | 全局内存访问 |

每种格式有不同的二进制编码。模拟器必须能够解码所有格式。

## Python 模拟器的工作原理

### 核心思想：将 GPU 指令编译为 CPU 内核

Python 模拟器不会在 Python 循环中逐条解释指令（那样太慢了）。相反，它**将每条 GPU 指令编译为一个 tinygrad UOp 内核**，然后在 CPU 后端上运行。

这是一种**动态二进制翻译**：GPU 机器码 -> UOp IR -> CPU 机器码。

### 架构概览

```
RDNA3 机器码字节
       |
       v
  decode_inst()           # 将字节解析为 Inst 对象
       |
       v
  _get_runner(bytes, arch) # 将指令编译为 CPU 内核
       |                   # 使用 _INST_HANDLERS 调度表
       v                   # 通过 _Ctx 构建 UOp 图
  get_runner('CPU', sink)  # tinygrad 将 UOps 编译为 Clang C
       |
       v
  缓存的 CPU 函数           # 使用寄存器缓冲区指针调用
```

### WaveState：模拟器的寄存器文件

`WaveState` 类保存一个波前的完整状态：

```python
from test.mockgpu.amd.emu import WaveState, EXEC_LO, SGPR_COUNT, VGPR_SIZE

ws = WaveState(32)  # 32 lane 的波前

# SGPRs: 260 x uint32
# 槽位 0-127:   实际的 SGPR
# 槽位 128-255: 内联常量 (0, 1, 2, ..., 64, -1, -2, ..., -16, 0.5, 1.0, ...)
# 槽位 256-259: 特殊用途 (PC_LO, PC_HI, SCC, SCRATCH_STRIDE)
print(f"SGPR 缓冲区: {SGPR_COUNT} 个 uint32")

# VGPRs: 256 * 32 = 8192 x uint32
# 布局: vgpr[reg_num * 32 + lane_id]
print(f"VGPR 缓冲区: {VGPR_SIZE} 个 uint32 (256 个寄存器 x 32 lanes)")

# 检查初始状态
print(f"PC: {ws.pc}")                                    # 0
print(f"EXEC: {ws._read_sgpr(EXEC_LO.offset):#010x}")   # 0xffffffff（所有 lane 活跃）
print(f"内联常量[129] (=1): {ws._read_sgpr(129)}")       # 1
print(f"内联常量[193] (=-1): {ws._read_sgpr(193):#010x}")  # 0xffffffff
```

关键点在于 SGPR 和 VGPR 存储为扁平的 `Buffer` 对象——普通的 tinygrad CPU 缓冲区。当编译后的指令运行时，它直接读写这些缓冲区。

### _Ctx：为指令构建 UOp 内核

`_Ctx` 类是模拟器的核心。它定义了每条编译指令接收的五个缓冲区 PARAM：

```python
class _Ctx:
    sgpr = UOp(Ops.PARAM, dtypes.uint32.ptr(260), arg=0)      # 标量寄存器
    vgpr = UOp(Ops.PARAM, dtypes.uint32.ptr(8192), arg=1)     # 向量寄存器
    vmem = UOp(Ops.PARAM, dtypes.uint32.ptr(1<<46), arg=2)    # 宿主内存（！）
    lds  = UOp(Ops.PARAM, dtypes.uint32.ptr(16384), arg=3)    # 本地数据共享
    scratch = UOp(Ops.PARAM, dtypes.uint8.ptr(1<<30), arg=4)  # 逐 lane 暂存区
```

注意 `vmem`——参数 2 映射到**整个宿主内存**（从虚拟地址 0 开始）。这就是模拟器访问张量数据的方式：GPU 全局内存加载变成了对宿主进程地址空间的直接读取。

### 将 EXEC 掩码实现为 RANGE 循环

对于向量指令（VOP1/VOP2/VOP3），模拟器创建一个遍历 32 个 lane 的 `RANGE` 循环：

```python
# 在模拟器的指令编译器内部：
lane = ctx.range(32)  # UOp.range(32, ...)

# 读取 VGPR：索引 = reg_num * 32 + lane
v0 = ctx.vgpr.index(0 * 32 + lane, ptr=True).load()
v1 = ctx.vgpr.index(1 * 32 + lane, ptr=True).load()

# 计算
result = v0 + v1

# 仅当 lane 活跃时写入 VGPR（EXEC 掩码检查）
exec_mask = ctx.sgpr.index(EXEC_LO, ptr=True).load()
active = ((exec_mask >> lane.cast(dtypes.uint32)) & 1).ne(0)
ctx.vgpr.index(2 * 32 + lane, active).store(result)
```

这个 UOp 图由 tinygrad 的 CPU 后端编译为高效的 C 代码，其中包含一个遍历 32 个 lane 的循环。EXEC 掩码检查变成了条件存储。

### Pcode：AMD 官方指令语义

模拟器如何知道 `V_ADD_F32` 做什么？它使用 AMD 在 ISA 参考手册中给出的官方伪代码。这些存储在自动生成的文件中：

```python
from test.mockgpu.amd.emu import get_pcode
from tinygrad.runtime.autogen.amd.rdna3.enum import VOP2Op, SOP2Op, VOP3Op

print(get_pcode(VOP2Op.V_ADD_F32_E32))
# 输出: D0.f32 = S0.f32 + S1.f32

print(get_pcode(VOP2Op.V_MUL_F32_E32))
# 输出: D0.f32 = S0.f32 * S1.f32

print(get_pcode(SOP2Op.S_ADD_U32))
# 输出: tmp = 64'U(S0.u32) + 64'U(S1.u32);
#       SCC = tmp >= 0x100000000ULL ? 1'1U : 1'0U;
#       D0.u32 = tmp.u32

print(get_pcode(VOP3Op.V_FMA_F32))
# 输出: D0.f32 = fma(S0.f32, S1.f32, S2.f32)
```

`test/mockgpu/amd/pcode.py` 中的 `parse_pcode()` 函数对这些伪代码进行词法分析，并将其转换为 UOp 表达式。因此 `D0.f32 = S0.f32 + S1.f32` 变成 `UOp(Ops.ADD, dtypes.float32, (src0, src1))`。

这种设计非常优雅：模拟器不需要硬编码数百条指令的语义，而是从 AMD 自己的文档中推导出来的。

### 规范化缓存

将每条指令编译为 CPU 内核听起来开销很大。模拟器通过规范化缓存来避免冗余编译：

```python
# 两条使用不同寄存器的 v_add_f32 指令：
# v_add_f32 v0, v1, v2
# v_add_f32 v5, v6, v7
#
# 它们有不同的寄存器字段，但语义完全相同。
# 模拟器屏蔽掉动态字段（寄存器编号），
# 并使用 (base_bits, mask, size) 作为缓存键。
#
# 结果：只需要编译一次，寄存器编号在运行时
# 通过 ctx.inst_field() 动态提取。
```

`canonical_mask()` 方法计算指令中哪些位是"静态的"（操作码、格式）以及哪些是"动态的"（寄存器编号、偏移量）。具有相同静态位的指令共享同一个编译后的执行器。

## 执行循环

当一个内核被调度时，`run_asm()` 负责协调完整的执行流程：

```python
def run_asm(lib, lib_sz, gx, gy, gz, lx, ly, lz, args_ptr, ...):
    # lib = 宿主内存中 RDNA3 机器码的指针
    # gx,gy,gz = 网格维度（工作组数量）
    # lx,ly,lz = 块维度（每个工作组的线程数）

    for gidz in range(gz):
      for gidy in range(gy):
        for gidx in range(gx):
          # 为此工作组初始化所有波前
          waves = []
          for wave_start in range(0, lx*ly*lz, 32):
            ws = WaveState(min(32, total_threads - wave_start))
            ws.pc = lib  # 将 PC 指向内核代码
            # 在 v0 中设置线程 ID，在 SGPR 中设置工作组 ID
            waves.append(ws)

          # 使用屏障同步执行
          for wi, ws in enumerate(waves):
            while ws.pc != ENDPGM:
              # 编译并运行一条指令
              fxn, globals_list, is_barrier, inst = _ensure_compiled(ws.pc)
              fxn(*[c_bufs[g] for g in globals_list])

              if is_barrier:
                break  # 暂停，直到所有波前到达屏障
```

关键要点：
- 每个工作组获得全新的 LDS（在工作组之间清零）
- 同一工作组内的波前在 `s_barrier` 处同步
- `_ensure_compiled()` 函数在首次遇到指令时进行惰性编译
- 执行期间启用 DAZ+FTZ（非规格化数当作零、刷新为零），以匹配 GPU 的浮点行为

## MockGPU 驱动

模拟器不仅仅模拟指令——它模拟了整个 AMD 驱动程序栈。`test/mockgpu/amd/amdgpu.py` 拦截：

- **PM4 数据包**：AMD GPU 使用的命令格式。当 tinygrad 提交一个 `PACKET3_DISPATCH_DIRECT` 时，模拟 GPU 会拦截它并调用 `run_asm()`。
- **内存管理**：缓冲区分配变成普通的 CPU 分配。
- **KFD ioctl**：AMD GPU 的 Linux 内核接口在 `amddriver.py` 中被拦截。

这意味着整个 tinygrad AMD 后端（`tinygrad/runtime/ops_amd.py`）可以不做任何修改地运行——它以为自己在与真正的 GPU 通信。

## 逐步运行一个内核

让我们追踪在模拟 GPU 上将两个张量相加时发生了什么：

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

设置 `DEBUG=4` 后，你会看到生成的 HIP C++ 内核：

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

这段 HIP 代码由 LLVM 编译为 RDNA3 机器码。然后模拟器：

1. **解码**二进制中的每条指令
2. **编译**为 UOp 图（仅在首次遇到时）
3. **执行**编译后的函数在 CPU 上运行，传入 SGPR/VGPR 缓冲区
4. **推进** PC 到下一条指令
5. **重复**直到 `s_endpgm`

使用 `DEBUG=3`，你可以看到每条指令被编译的过程：

```
[emu] PC=0: s_load_b64(...)       # 加载内核参数
[emu] PC=8: s_waitcnt(...)        # 等待内存操作
[emu] PC=12: global_load_b128(...) # 从 data1 加载 4 个浮点数
[emu] PC=20: global_load_b128(...) # 从 data2 加载 4 个浮点数
[emu] PC=28: s_waitcnt(...)        # 等待加载完成
[emu] PC=32: v_add_f32(...)        # 加法：元素 0
[emu] PC=36: v_add_f32(...)        # 加法：元素 1
[emu] PC=40: v_add_f32(...)        # 加法：元素 2
[emu] PC=44: v_add_f32(...)        # 加法：元素 3
[emu] PC=48: global_store_b128(...)# 存储 4 个结果
[emu] PC=56: s_endpgm              # 完成
```

## 理解指令编译管道

让我们看看一条 `v_add_f32` 指令是如何被编译的。处理函数是 `_compile_vop12`：

```
1. 解码指令字节 -> 带有 op=V_ADD_F32_E32 的 VOP2 对象
2. 查找 pcode："D0.f32 = S0.f32 + S1.f32"
3. 读取源操作数：
   - S0 = ctx.rsrc(inst.src0, lane)     # 可能是 SGPR 或 VGPR
   - S1 = ctx.rvgpr(inst.vsrc1, lane)   # 对于 VOP2 始终是 VGPR
4. 将 pcode 解析为 UOp：result = S0.bitcast(float32) + S1.bitcast(float32)
5. 写入目标：
   - ctx.wvgpr(inst.vdst, lane, result, exec_mask)
6. 创建包含所有存储操作的 SINK
7. 调用 get_runner('CPU', sink) -> 编译后的 C 函数
```

编译后的函数签名本质上是：

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

## Rust 模拟器 (Remu)

位于 `extra/remu/` 的 Rust 模拟器采用了更简单的方法：直接解释执行。

```rust
// 简化自 extra/remu/src/thread.rs
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
                    // ... 还有数百条指令
                };
            }
        }
        // ... 其他格式
    }
}
```

它更快，但更难维护——每条新指令都需要手动实现其语义。Python 模拟器则从 pcode 自动推导语义。

`test/amd/test_compare_emulators.py` 中的测试会通过两个模拟器逐指令运行真实的 tinygrad 内核，比较所有寄存器状态以确保它们一致。

## SQTT 追踪

当设置 `PROFILE=1` 时，Python 模拟器会生成 AMD Shader Thread Trace (SQTT) 数据包。这些使用与真实 AMD GPU 性能分析工具完全相同的二进制格式。这意味着无论你是在真实硬件上还是在模拟器上，tinygrad 的性能分析基础设施的工作方式都完全相同。

```python
# 模拟器为每条执行的指令生成 SQTT 数据包：
def emit(wave_id, inst, branch_taken):
    # 分类指令类型（SALU、VALU、SMEM、LDS、GLOBAL 等）
    # 发出带有时间增量的相应 SQTT 数据包
    _emit_nibbles(nibbles, INST, delta=1, wave=wave_id & 0x1F, op=inst_op)
```

## 关键设计决策

### 1. 自举编译

Python 模拟器使用 tinygrad 自身的 CPU 后端来编译 GPU 指令。这是一种最好的循环引用——tinygrad 使用自己来模拟它为之生成代码的硬件。这也意味着 tinygrad 代码生成的改进会自动提升模拟器的性能。

### 2. Pcode 驱动的语义

通过解析 AMD 官方伪代码，模拟器无需手动实现数百种指令变体就能保持正确性。当 AMD 添加新指令时，你只需要提供 pcode 字符串。

### 3. DAZ+FTZ 浮点处理

GPU 处理非规格化浮点数的方式与 CPU 不同。模拟器在执行期间设置 CPU 的 `MXCSR`（x86）或 `FPCR`（ARM64）寄存器以匹配 GPU 行为，然后在执行完毕后恢复。

### 4. 基于零的虚拟内存

`vmem` 参数是一个 `external_ptr=0` 的 Buffer，意味着索引 0 映射到物理地址 0。GPU 全局内存加载使用内核参数中的原始虚拟地址作为该缓冲区的索引。由于模拟器运行在同一进程中，这些地址都是有效的宿主指针。

## 动手练习

### 练习 1：运行模拟器

```bash
MOCKGPU=1 AMD=1 PYTHON_REMU=1 python -c "
from tinygrad import Tensor, Device
Device.DEFAULT = 'AMD'
print((Tensor.ones(4,4) @ Tensor.ones(4,4)).numpy())
"
```

验证你得到的是一个所有元素都为 4.0 的 4x4 矩阵。

### 练习 2：查看指令

```bash
DEBUG=3 MOCKGPU=1 AMD=1 PYTHON_REMU=1 python -c "
from tinygrad import Tensor, Device
Device.DEFAULT = 'AMD'
print(Tensor([1.0, 2.0]).sum().item())
"
```

数一下模拟器编译了多少条指令。其中有多少是标量指令，多少是向量指令？

### 练习 3：查看 Pcode

```python
from test.mockgpu.amd.emu import get_pcode
from tinygrad.runtime.autogen.amd.rdna3.enum import VOP2Op, VOP3Op

# v_cndmask_b32 做了什么？
print(get_pcode(VOP2Op.V_CNDMASK_B32_E32))

# v_fma_f32 呢？
print(get_pcode(VOP3Op.V_FMA_F32))
```

### 练习 4：创建一个 WaveState

```python
from test.mockgpu.amd.emu import WaveState, EXEC_LO

ws = WaveState(32)
# 向 VGPR v5 的 lane 0 写入一个值
ws._write_vgpr(5, 0, 42)
print(f"v5[lane0] = {ws._read_vgpr(5, 0)}")

# 禁用 EXEC 掩码中的 lane 0
exec_val = ws._read_sgpr(EXEC_LO.offset)
ws._write_sgpr(EXEC_LO.offset, exec_val & ~1)
print(f"EXEC = {ws._read_sgpr(EXEC_LO.offset):#010x}")
# 现在 lane 0 将不会参与向量指令
```

## 源代码地图

| 文件 | 阅读内容 |
|------|-------------|
| `test/mockgpu/amd/emu.py` | 主模拟器：`WaveState`、`_Ctx`、指令处理函数、`run_asm` |
| `test/mockgpu/amd/emu.py:402` | `_Ctx` 类——用于指令编译的 UOp 构建器 |
| `test/mockgpu/amd/emu.py:1382` | `WaveState`——波前寄存器状态 |
| `test/mockgpu/amd/emu.py:1445` | `run_asm()`——主执行循环 |
| `test/mockgpu/amd/pcode.py` | 伪代码词法分析器和解析器 |
| `test/mockgpu/amd/amdgpu.py` | MockGPU 驱动——PM4 数据包处理 |
| `test/mockgpu/amd/amddriver.py` | KFD ioctl 拦截 |
| `tinygrad/renderer/amd/` | 指令编码/解码、DSL、SQTT |
| `tinygrad/runtime/autogen/amd/rdna3/` | 自动生成的指令类、操作码、pcode 字符串 |
| `extra/remu/src/` | Rust 模拟器源代码 |
| `test/amd/test_compare_emulators.py` | Python 和 Rust 模拟器之间的交叉验证 |
