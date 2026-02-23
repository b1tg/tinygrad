# RDNA3 emulator v2 - compiles pcode to UOps executed via tinygrad CPU backend
# Each instruction is compiled to a kernel that operates on buffers:
#   arg=0: sgpr - sgpr[0-127], inline constants[128-255], PC_LO=256, PC_HI=257, SCC=258, SCRATCH_STRIDE=259, SCRATCH_BASE=260
#   arg=1: vgpr - vgpr[reg * 32 + lane]
#   arg=2: vmem - base address 0, INDEX offsets directly to host memory
#   arg=3: lds - local data share
#   arg=4: scratch - per-lane scratch memory
from __future__ import annotations
import ctypes, functools, re, platform, subprocess, tempfile, struct
from typing import Callable

# Set/restore DAZ+FTZ (denormals-are-zero + flush-to-zero) to match RDNA3 default float mode
# x86: MXCSR bits DAZ(6)+FTZ(15), ARM64: FPCR bit FZ(24)
# Only applied during emulator execution, restored afterward to avoid breaking hypothesis tests
@functools.cache
def _get_ftz_lib():
  machine = platform.machine()
  if machine in ('x86_64', 'AMD64'):
    src = b'''
unsigned int get_fpcr(void){unsigned int m;__asm__ __volatile__("stmxcsr %0":"=m"(m));return m;}
void set_fpcr(unsigned int m){__asm__ __volatile__("ldmxcsr %0"::"m"(m));}
'''
    ftz_bits = 0x8040  # DAZ (bit 6) + FTZ (bit 15)
  elif machine in ('arm64', 'aarch64'):
    src = b'''
unsigned int get_fpcr(void){unsigned long long v;__asm__ __volatile__("mrs %0,fpcr":"=r"(v));return(unsigned int)v;}
void set_fpcr(unsigned int m){unsigned long long v=m;__asm__ __volatile__("msr fpcr,%0"::"r"(v));}
'''
    ftz_bits = 1 << 24  # FZ (bit 24)
  else: return None, 0
  try:
    with tempfile.NamedTemporaryFile(suffix='.so', delete=False) as f:
      subprocess.check_output(['clang', '-shared', '-O2', '-x', 'c', '-', '-o', f.name], input=src)
      lib = ctypes.CDLL(f.name)
      lib.get_fpcr.restype = ctypes.c_uint32
      lib.set_fpcr.argtypes = [ctypes.c_uint32]
      return lib, ftz_bits
  except Exception: return None, 0

class _MXCSRContext:
  """Context manager to set DAZ+FTZ during emulator execution and restore afterward."""
  __slots__ = ('_saved',)
  def __enter__(self):
    lib, ftz_bits = _get_ftz_lib()
    if lib is None: return self
    self._saved = lib.get_fpcr()
    lib.set_fpcr(self._saved | ftz_bits)
    return self
  def __exit__(self, *args):
    lib, _ = _get_ftz_lib()
    if lib is None or not hasattr(self, '_saved'): return
    lib.set_fpcr(self._saved)

from tinygrad.uop.ops import UOp, Ops, KernelInfo, AxisType
from tinygrad.dtype import dtypes
from tinygrad.device import Buffer, BufferSpec
from tinygrad.runtime.autogen import hsa
from tinygrad.helpers import Context, DEBUG, PROFILE, colored, getenv
from tinygrad.engine.realize import get_runner

from tinygrad.renderer.amd import decode_inst
from tinygrad.runtime.autogen.amd.rdna3.str_pcode import PCODE as PCODE_RDNA3
from tinygrad.runtime.autogen.amd.rdna4.str_pcode import PCODE as PCODE_RDNA4
from tinygrad.runtime.autogen.amd.cdna.str_pcode import PCODE as PCODE_CDNA
from tinygrad.runtime.autogen.amd.rdna3 import ins as ir3
from tinygrad.runtime.autogen.amd.rdna4 import ins as ir4
from tinygrad.runtime.autogen.amd.cdna import ins as irc
from tinygrad.runtime.autogen.amd.cdna.operands import OPERANDS as CDNA_OPERANDS
from tinygrad.renderer.amd.dsl import VCC_LO, EXEC_LO, SCC, ttmp
from tinygrad.runtime.autogen.amd.common import Fmt, OpType
from test.mockgpu.amd.pcode import parse_block, _FUNCS

MASK32 = 0xFFFFFFFF

# ═══════════════════════════════════════════════════════════════════════════════
# SQTT TRACE COLLECTION
# ═══════════════════════════════════════════════════════════════════════════════

# Global trace storage: populated by run_asm as raw SQTT blobs, consumed by amdgpu.py
sqtt_traces: list[bytes] = []

# Encoder primitives
from tinygrad.renderer.amd.sqtt import _build_decode_tables, PACKET_TYPES_RDNA3, LAYOUT_HEADER, WAVESTART, WAVEEND, INST, IMMEDIATE, VALUINST, InstOp

_NIB_COUNTS: dict = {cls: nc for _, (cls, nc, *_) in _build_decode_tables(PACKET_TYPES_RDNA3)[0].items()}

def _encode_raw(pkt_cls, **kwargs) -> tuple[int, int]:
  raw = pkt_cls.encoding.default
  for k, v in kwargs.items(): raw = pkt_cls.__dict__[k].set(raw, v)
  return raw, _NIB_COUNTS[pkt_cls]

def _emit_nibbles(nibbles: list[int], pkt_cls, **kwargs):
  raw, nc = _encode_raw(pkt_cls, **kwargs)
  for i in range(nc): nibbles.append((raw >> (i * 4)) & 0xF)

def _nibbles_to_bytes(nibbles: list[int]) -> bytes:
  result = bytearray()
  for i in range(0, len(nibbles), 2): result.append(nibbles[i] | ((nibbles[i + 1] if i + 1 < len(nibbles) else 0) << 4))
  return bytes(result)

def _init_sqtt_encoder():
  """Initialize and return SQTT encoder state. Called once per dispatch with tracing enabled."""
  from tinygrad.runtime.autogen.amd.rdna3.enum import SOPPOp as SOPPOp3
  from tinygrad.runtime.autogen.amd.rdna4.enum import SOPPOp as SOPPOp4
  import re

  _SOPP = (ir3.SOPP, ir4.SOPP, irc.SOPP)
  _SMEM = (ir3.SMEM, ir4.SMEM, irc.SMEM)
  _VALU = (ir3.VOP1, ir3.VOP2, ir3.VOP3, ir3.VOP3P, ir3.VOPC, ir3.VOPD, ir3.VOP3SD, ir3.VOP3_SDST, ir3.VOP1_SDST,
           ir4.VOP1, ir4.VOP2, ir4.VOP3, ir4.VOP3P, ir4.VOPC, ir4.VOPD, ir4.VOP3SD, ir4.VOP3_SDST, ir4.VOP1_SDST,
           irc.VOP1, irc.VOP2, irc.VOP3, irc.VOP3P, irc.VOPC, irc.VOP3SD, irc.VOP3_SDST)
  _DS = (ir3.DS, ir4.DS, irc.DS)
  _GLOBAL = (ir3.GLOBAL, ir4.VGLOBAL, irc.GLOBAL)
  _FLAT = (ir3.FLAT, ir4.VFLAT, irc.FLAT)
  _SCRATCH = (ir3.SCRATCH, ir4.VSCRATCH, irc.SCRATCH)

  # SOPP classification sets
  _SOPP_SKIP = {SOPPOp3.S_ENDPGM.value, SOPPOp3.S_ENDPGM_SAVED.value, SOPPOp3.S_ENDPGM_ORDERED_PS_DONE.value,
                SOPPOp3.S_DELAY_ALU.value}
  _SOPP_IMMEDIATE = {SOPPOp3.S_NOP.value, SOPPOp3.S_CLAUSE.value, SOPPOp3.S_WAITCNT.value, SOPPOp3.S_WAITCNT_DEPCTR.value,
                     SOPPOp3.S_WAIT_IDLE.value, SOPPOp3.S_WAIT_EVENT.value, SOPPOp3.S_SLEEP.value,
                     SOPPOp3.S_SET_INST_PREFETCH_DISTANCE.value}
  for _op in (SOPPOp4.S_WAIT_ALU, SOPPOp4.S_WAIT_LOADCNT, SOPPOp4.S_WAIT_STORECNT, SOPPOp4.S_WAIT_SAMPLECNT,
              SOPPOp4.S_WAIT_BVHCNT, SOPPOp4.S_WAIT_EXPCNT, SOPPOp4.S_WAIT_DSCNT, SOPPOp4.S_WAIT_KMCNT,
              SOPPOp4.S_WAIT_LOADCNT_DSCNT, SOPPOp4.S_WAIT_STORECNT_DSCNT):
    _SOPP_IMMEDIATE.add(_op.value)
  _SOPP_BARRIER = {SOPPOp3.S_BARRIER.value}
  if hasattr(SOPPOp4, 'S_BARRIER_WAIT'): _SOPP_BARRIER.add(SOPPOp4.S_BARRIER_WAIT.value)
  if hasattr(SOPPOp4, 'S_BARRIER_LEAVE'): _SOPP_BARRIER.add(SOPPOp4.S_BARRIER_LEAVE.value)
  _SOPP_BRANCH = {SOPPOp3.S_BRANCH.value, SOPPOp3.S_CBRANCH_SCC0.value, SOPPOp3.S_CBRANCH_SCC1.value,
                  SOPPOp3.S_CBRANCH_VCCZ.value, SOPPOp3.S_CBRANCH_VCCNZ.value,
                  SOPPOp3.S_CBRANCH_EXECZ.value, SOPPOp3.S_CBRANCH_EXECNZ.value}

  # VALU sub-classification patterns
  _VALU_TRANS_RE = re.compile(r'V_(EXP|LOG|RCP|RSQ|SQRT|SIN|COS|CEIL|FLOOR|TRUNC|RNDNE|FRACT|FREXP)_')
  _VALU_64_SHIFT_RE = re.compile(r'V_(LSHLREV|LSHRREV|ASHRREV)_(B|I)64')
  _VALU_MAD64_RE = re.compile(r'V_MAD_(U|I)64')
  _VALU_64_RE = re.compile(r'V_\w+_F64')

  def _valu_op(op_name: str) -> InstOp|None:
    if 'CMPX' in op_name: return InstOp.VALU_CMPX
    if _VALU_64_SHIFT_RE.search(op_name): return InstOp.VALU_64_SHIFT
    if _VALU_MAD64_RE.search(op_name): return InstOp.VALU_MAD64
    if _VALU_64_RE.search(op_name): return InstOp.VALU_64
    if _VALU_TRANS_RE.search(op_name): return InstOp.VALU_TRANS
    return None

  def _mem_op(t, op_name: str) -> InstOp:
    is_store = "STORE" in op_name
    if issubclass(t, _DS): return InstOp.LDS_STORE if is_store else InstOp.LDS_LOAD
    if issubclass(t, _GLOBAL): return InstOp.GLOBAL_STORE if is_store else InstOp.GLOBAL_LOAD
    if issubclass(t, _FLAT): return InstOp.FLAT_STORE if is_store else InstOp.FLAT_LOAD
    if issubclass(t, _SCRATCH): return InstOp.FLAT_STORE if is_store else InstOp.FLAT_LOAD
    return InstOp.SALU

  nibbles: list[int] = []
  started: set[int] = set()
  _emit_nibbles(nibbles, LAYOUT_HEADER, layout=3, sel_a=6)

  def emit(wave_id: int, inst, branch_taken: bool|None):
    """Emit an SQTT packet for one executed instruction."""
    w = wave_id & 0x1F
    if wave_id not in started:
      _emit_nibbles(nibbles, WAVESTART, delta=1, simd=0, cu_lo=0, wave=w, id7=wave_id)
      started.add(wave_id)
    inst_type, inst_op, op_name = type(inst), inst.op.value if hasattr(inst, 'op') else 0, inst.op.name if hasattr(inst, 'op') else ""
    if issubclass(inst_type, _SOPP):
      if inst_op in _SOPP_SKIP: return
      elif inst_op in _SOPP_IMMEDIATE: _emit_nibbles(nibbles, IMMEDIATE, delta=1, wave=w)
      elif inst_op in _SOPP_BARRIER: _emit_nibbles(nibbles, INST, delta=1, wave=w, op=InstOp.BARRIER)
      elif inst_op in _SOPP_BRANCH:
        _emit_nibbles(nibbles, INST, delta=1, wave=w, op=InstOp.JUMP if branch_taken else InstOp.JUMP_NO)
      else: _emit_nibbles(nibbles, INST, delta=1, wave=w, op=InstOp.SALU)
    elif issubclass(inst_type, _VALU):
      op = _valu_op(op_name)
      if op is None: _emit_nibbles(nibbles, VALUINST, delta=1, wave=w)
      else: _emit_nibbles(nibbles, INST, delta=1, wave=w, op=op)
    elif issubclass(inst_type, _SMEM): _emit_nibbles(nibbles, INST, delta=1, wave=w, op=InstOp.SMEM)
    else: _emit_nibbles(nibbles, INST, delta=1, wave=w, op=_mem_op(inst_type, op_name))

  def finish(wave_id: int):
    """Emit WAVEEND for a completed wave."""
    if wave_id in started: _emit_nibbles(nibbles, WAVEEND, delta=1, simd=0, cu_lo=0, wave=wave_id & 0x1F)

  def finalize() -> bytes:
    """Pad and return the encoded SQTT blob."""
    while len(nibbles) % 2 != 0: nibbles.append(0)
    nibbles.extend([0] * 32)
    while len(nibbles) % 64 != 0: nibbles.append(0)
    return _nibbles_to_bytes(nibbles)

  return emit, finish, finalize

def _c(val, dtype=dtypes.uint32): return UOp.const(dtype, val)

def _u64(lo: UOp, hi: UOp) -> UOp:
  """Combine two 32-bit UOps into a 64-bit UOp."""
  return lo.cast(dtypes.uint64) | (hi.cast(dtypes.uint64) << UOp.const(dtypes.uint64, 32))

def _split64(val: UOp) -> tuple[UOp, UOp]:
  """Split a 64-bit value into (lo, hi) 32-bit values."""
  v64 = val.bitcast(dtypes.uint64) if val.dtype == dtypes.float64 else val.cast(dtypes.uint64) if val.dtype != dtypes.uint64 else val
  return v64.cast(dtypes.uint32), (v64 >> UOp.const(dtypes.uint64, 32)).cast(dtypes.uint32)

_SRC_MOD_TYPES = {16: (dtypes.uint16, dtypes.half, 0x7FFF), 32: (dtypes.uint32, dtypes.float32, 0x7FFFFFFF),
                  64: (dtypes.uint64, dtypes.float64, 0x7FFFFFFFFFFFFFFF)}
def _apply_src_mods(val: UOp, mod_bit: int, abs_bits: int, neg_bits: int, bits: int = 32) -> UOp:
  """Apply abs/neg modifiers to source value based on bit width (16, 32, or 64)."""
  if not (abs_bits & (1 << mod_bit)) and not (neg_bits & (1 << mod_bit)): return val
  ut, ft, mask = _SRC_MOD_TYPES[bits]
  fv = val.cast(ut).bitcast(ft) if bits == 16 else val.bitcast(ft) if val.dtype == ut else val
  if abs_bits & (1 << mod_bit): fv = (fv.bitcast(ut) & UOp.const(ut, mask)).bitcast(ft)
  if neg_bits & (1 << mod_bit): fv = fv.neg()
  return fv.bitcast(ut).cast(dtypes.uint32) if bits == 16 else fv.bitcast(ut)

# Map VOPD ops to VOP2 ops for pcode lookup (both RDNA3 and RDNA4)
VOPD_TO_VOP2 = {
  ir3.VOPDOp.V_DUAL_FMAC_F32: ir3.VOP2Op.V_FMAC_F32_E32, ir3.VOPDOp.V_DUAL_MUL_F32: ir3.VOP2Op.V_MUL_F32_E32,
  ir3.VOPDOp.V_DUAL_ADD_F32: ir3.VOP2Op.V_ADD_F32_E32, ir3.VOPDOp.V_DUAL_SUB_F32: ir3.VOP2Op.V_SUB_F32_E32,
  ir3.VOPDOp.V_DUAL_SUBREV_F32: ir3.VOP2Op.V_SUBREV_F32_E32, ir3.VOPDOp.V_DUAL_MAX_F32: ir3.VOP2Op.V_MAX_F32_E32,
  ir3.VOPDOp.V_DUAL_MIN_F32: ir3.VOP2Op.V_MIN_F32_E32, ir3.VOPDOp.V_DUAL_ADD_NC_U32: ir3.VOP2Op.V_ADD_NC_U32_E32,
  ir3.VOPDOp.V_DUAL_LSHLREV_B32: ir3.VOP2Op.V_LSHLREV_B32_E32, ir3.VOPDOp.V_DUAL_AND_B32: ir3.VOP2Op.V_AND_B32_E32,
  ir3.VOPDOp.V_DUAL_MOV_B32: ir3.VOP1Op.V_MOV_B32_E32, ir3.VOPDOp.V_DUAL_CNDMASK_B32: ir3.VOP2Op.V_CNDMASK_B32_E32,
  ir3.VOPDOp.V_DUAL_FMAAK_F32: ir3.VOP2Op.V_FMAAK_F32_E32, ir3.VOPDOp.V_DUAL_FMAMK_F32: ir3.VOP2Op.V_FMAMK_F32_E32,
  # RDNA4 mappings (same VOP1/VOP2 targets, RDNA4 uses _NUM_ suffix for min/max)
  ir4.VOPDOp.V_DUAL_FMAC_F32: ir3.VOP2Op.V_FMAC_F32_E32, ir4.VOPDOp.V_DUAL_MUL_F32: ir3.VOP2Op.V_MUL_F32_E32,
  ir4.VOPDOp.V_DUAL_ADD_F32: ir3.VOP2Op.V_ADD_F32_E32, ir4.VOPDOp.V_DUAL_SUB_F32: ir3.VOP2Op.V_SUB_F32_E32,
  ir4.VOPDOp.V_DUAL_SUBREV_F32: ir3.VOP2Op.V_SUBREV_F32_E32, ir4.VOPDOp.V_DUAL_MAX_NUM_F32: ir3.VOP2Op.V_MAX_F32_E32,
  ir4.VOPDOp.V_DUAL_MIN_NUM_F32: ir3.VOP2Op.V_MIN_F32_E32, ir4.VOPDOp.V_DUAL_ADD_NC_U32: ir3.VOP2Op.V_ADD_NC_U32_E32,
  ir4.VOPDOp.V_DUAL_LSHLREV_B32: ir3.VOP2Op.V_LSHLREV_B32_E32, ir4.VOPDOp.V_DUAL_AND_B32: ir3.VOP2Op.V_AND_B32_E32,
  ir4.VOPDOp.V_DUAL_MOV_B32: ir3.VOP1Op.V_MOV_B32_E32, ir4.VOPDOp.V_DUAL_CNDMASK_B32: ir3.VOP2Op.V_CNDMASK_B32_E32,
  ir4.VOPDOp.V_DUAL_FMAAK_F32: ir3.VOP2Op.V_FMAAK_F32_E32, ir4.VOPDOp.V_DUAL_FMAMK_F32: ir3.VOP2Op.V_FMAMK_F32_E32,
}
WAVE_SIZE = int(getenv("MOCKGPU_WAVE_SIZE", 64 if getenv("MOCKGPU_ARCH", "") == "cdna4" else 32))
ENFORCE_MEM_RANGES = bool(getenv("MOCKGPU_ENFORCE_MEM_RANGES", 0))
# Special registers stored after inline constants (256-260)
PC_LO_IDX, PC_HI_IDX, SCRATCH_STRIDE_IDX, SCRATCH_BASE_IDX = 256, 257, 259, 260
# SGPR buffer: 0-127 = SGPRs, 128-255 = inline constants, 256-260 = special registers
SGPR_COUNT, VGPR_SIZE = 261, 256 * WAVE_SIZE
# Sentinel PC value for s_endpgm
ENDPGM_PC = 0xFFFFFFFFFFFFFFFF

def _op_name(inst) -> str:
  if hasattr(inst, 'opx'): return f"{inst.opx.name}_{inst.opy.name}"  # VOPD has opx/opy not op
  return inst.op.name if hasattr(inst.op, 'name') else str(inst.op)

def _to_u32(val: UOp) -> UOp:
  if val.dtype == dtypes.uint32: return val
  if val.dtype.itemsize == 4: return val.bitcast(dtypes.uint32)  # same size: bitcast (float32->uint32)
  return val.cast(dtypes.uint32)  # different size: cast (bool, int16, etc)
def _lane_active(exec_mask: UOp, lane: UOp) -> UOp:
  return ((exec_mask.cast(dtypes.uint64) >> lane.cast(dtypes.uint64)) & _c(1, dtypes.uint64)).ne(_c(0, dtypes.uint64))
def _hi16(v: UOp) -> UOp: return (v >> _c(16)) & _c(0xFFFF)
def _cond(cond, if_true, if_false):
  """Select between values based on condition (works with UOp or bool)."""
  return cond.where(if_true, if_false) if isinstance(cond, UOp) else if_true if cond else if_false
def _cond_hi16(cond, val: UOp) -> UOp: return _cond(cond, _hi16(val), val)
def _apply_opsel(val: UOp, sel_bit: int, opsel: int) -> UOp: return _hi16(val) if opsel & (1 << sel_bit) else val

def _apply_sdwa_sel(val: UOp, sel: int) -> UOp:
  """Apply SDWA src/dst selection: 0-3=BYTE_0..3, 4=WORD_0, 5=WORD_1, 6=DWORD."""
  if sel == 6: return val  # DWORD - no selection
  if sel <= 3: return (val >> _c(sel * 8)) & _c(0xFF)  # BYTE_N
  if sel == 4: return val & _c(0xFFFF)  # WORD_0
  return (val >> _c(16)) & _c(0xFFFF)  # WORD_1

def _apply_sdwa_sext(val: UOp, sel: int, sext: int) -> UOp:
  """Apply SDWA sign extension for byte/word selections."""
  if not sext or sel == 6: return val
  if sel <= 3: return val.cast(dtypes.uint8).cast(dtypes.int8).cast(dtypes.int32).bitcast(dtypes.uint32)
  return val.cast(dtypes.uint16).cast(dtypes.int16).cast(dtypes.int32).bitcast(dtypes.uint32)

def _apply_sdwa_dst(old: UOp, val: UOp, dst_sel: int, dst_unused: int) -> UOp:
  """
  Apply SDWA destination selection/unused policy.
  dst_sel: BYTE_0..BYTE_3/WORD_0/WORD_1/DWORD (0..6)
  dst_unused: 0=PAD, 1=SEXT, 2/3=PRESERVE/RESERVED (treated as preserve)
  """
  result = _to_u32(val)
  if dst_sel == 6: return result

  width, shift = (8, dst_sel * 8) if dst_sel <= 3 else (16, 0 if dst_sel == 4 else 16)
  keep_mask = _c((((1 << width) - 1) << shift) & MASK32)
  write_bits = (result & _c((1 << width) - 1)) << _c(shift)

  if dst_unused in (2, 3): base = old & (keep_mask ^ _c(MASK32))
  elif dst_unused == 1:
    sign = ((result >> _c(width - 1)) & _c(1)).ne(_c(0))
    base = sign.where(keep_mask ^ _c(MASK32), _c(0))
  else: base = _c(0)
  return base | write_bits

def _set_lane_bit(old: UOp, lane: UOp, val: UOp, exec_mask: UOp) -> UOp:
  """Set/clear a single bit in a mask based on lane index, respecting exec mask."""
  old_u64 = old.cast(dtypes.uint64)
  mask = _c(1, dtypes.uint64) << lane.cast(dtypes.uint64)
  new_bit = _to_u32(val).cast(dtypes.uint64) << lane.cast(dtypes.uint64)
  cleared = old_u64 & (mask ^ _c((1 << WAVE_SIZE) - 1, dtypes.uint64))
  return _lane_active(exec_mask, lane).where(cleared | new_bit, old_u64)

def _val_to_u32(val: UOp) -> UOp:
  """Convert any value to uint32 for storage (bitcast floats, cast ints)."""
  if val.dtype == dtypes.uint32: return val
  if val.dtype == dtypes.float32: return val.bitcast(dtypes.uint32)
  if val.dtype == dtypes.half: return val.bitcast(dtypes.uint16).cast(dtypes.uint32)
  if val.dtype in (dtypes.uint16, dtypes.int16): return val.cast(dtypes.uint32)
  return val.cast(dtypes.uint32)

_pcode_fixes = {
  'V_DIV_FMAS_F32': ('D0.f32 = 2.0F ** 32 * fma(S0.f32, S1.f32, S2.f32)',
    'D0.f32 = (exponent(S2.f32) > 127) ? (2.0F ** 64 * fma(S0.f32, S1.f32, S2.f32)) : (2.0F ** -64 * fma(S0.f32, S1.f32, S2.f32))'),
  'V_DIV_FMAS_F64': ('D0.f64 = 2.0 ** 64 * fma(S0.f64, S1.f64, S2.f64)',
    'D0.f64 = (exponent(S2.f64) > 1023) ? (2.0 ** 128 * fma(S0.f64, S1.f64, S2.f64)) : (2.0 ** -128 * fma(S0.f64, S1.f64, S2.f64))'),
  'V_DIV_FIXUP_F32': ('D0.f32 = sign_out ? -abs(S0.f32) : abs(S0.f32)',
    'D0.f32 = isNAN(S0.f32) ? (sign_out ? -INF.f32 : +INF.f32) : (sign_out ? -abs(S0.f32) : abs(S0.f32))'),
  'V_DIV_FIXUP_F64': ('D0.f64 = sign_out ? -abs(S0.f64) : abs(S0.f64)',
    'D0.f64 = isNAN(S0.f64) ? (sign_out ? -INF : +INF) : (sign_out ? -abs(S0.f64) : abs(S0.f64))'),
  'V_TRIG_PREOP_F64': ("result = 64'F((1201'B(2.0 / PI)[1200 : 0] << shift.u32) & 1201'0x1fffffffffffff)", "result = trig_preop_result(shift)"),
}

def _get_pcode_dict(op) -> dict:
  """Return the PCODE dictionary for the given opcode based on its architecture."""
  return PCODE_CDNA if 'cdna' in type(op).__module__ else PCODE_RDNA4 if 'rdna4' in type(op).__module__ else PCODE_RDNA3

# Pcode parser
@functools.cache
def get_pcode(op) -> str:
  op_name = op.name
  pcode_dict = _get_pcode_dict(op)
  pcode = pcode_dict.get(op)
  if pcode is None and op_name.endswith('_E64'):
    # Some CDNA E64 aliases share E32 semantics in the pcode tables.
    alias_name = op_name[:-4] + '_E32'
    pcode = next((v for k, v in pcode_dict.items() if k.name == alias_name), None)
  if pcode is None: raise KeyError(op)
  fix_name = op_name if op_name in _pcode_fixes else (op_name[:-4] if op_name.endswith('_E64') and op_name[:-4] in _pcode_fixes else None)
  if fix_name is not None: pcode = pcode.replace(*_pcode_fixes[fix_name])
  if 'V_DIV_SCALE' in op_name:
    dt, exp_lim, ldexp_val = ('f32', '23', '64') if 'F32' in op_name else ('f64', '52', '128')
    for old, new in [(f'S2.{dt} / S1.{dt} == DENORM.{dt}', f'divWouldBeDenorm(S2.{dt}, S1.{dt})'), (f"1.0 / 64'F(S1.{dt}) == DENORM.f64", '0'),
                     (f'1.0 / S1.{dt} == DENORM.{dt}', '0'), (f'S1.{dt} == DENORM.{dt}', f'isDENORM(S1.{dt})'),
                     (f'D0.{dt} = NAN.{dt}', f'VCC = 0x1LL;\nD0.{dt} = NAN.{dt}'),
                     (f'elsif isDENORM(S1.{dt}) then\nD0.{dt} = ldexp(S0.{dt}, {ldexp_val})', f'elsif 1 == 0 then\nD0.{dt} = S0.{dt}'),
                      (f'elsif exponent(S2.{dt}) <= {exp_lim} then\n// Numerator is tiny\n'
                       f'D0.{dt} = ldexp(S0.{dt}, {ldexp_val})',
                       f'elsif exponent(S2.{dt}) <= {exp_lim} then\nVCC = 0x1LL;\n'
                       f'D0.{dt} = ldexp(S0.{dt}, {ldexp_val})'),
                      (f'elsif divWouldBeDenorm(S2.{dt}, S1.{dt}) then\nVCC = 0x1LL;\n'
                       f'if S0.{dt} == S2.{dt} then\n// Only scale the numerator\n'
                       f'D0.{dt} = ldexp(S0.{dt}, {ldexp_val})\nendif',
                       f'elsif divWouldBeDenorm(S2.{dt}, S1.{dt}) then\n'
                       f'VCC = 0x1LL;\nD0.{dt} = S0.{dt}'),
                      (f'D0.{dt} = ldexp(S0.{dt}, {ldexp_val})\nendif\nelsif',
                       f'D0.{dt} = ldexp(S0.{dt}, {ldexp_val})\nelse\n'
                       f'D0.{dt} = S0.{dt}\nendif\nelsif')]:
      pcode = pcode.replace(old, new)
    lines = pcode.rstrip().split('\n')
    for i in range(len(lines) - 1, -1, -1):
      if lines[i].strip() == 'endif':
        lines.insert(i, f'else\nD0.{dt} = S0.{dt}')
        break
    pcode = '\n'.join(lines) + f';\nif isDENORM(S1.{dt}) then\nD0.{dt} = NAN.{dt}\nendif'
    pcode = pcode.replace('VCC = 0x0LL', 'VCC.u64[laneId] = 0').replace('VCC = 0x1LL', 'VCC.u64[laneId] = 1')
  return pcode

def parse_pcode(pcode: str, srcs: dict[str, UOp] | None = None) -> tuple[dict, list[tuple[str, UOp]]]:
  env: dict = srcs.copy() if srcs else {}
  env.setdefault('_wave_size', _c(WAVE_SIZE))
  assigns: list[tuple[str, UOp]] = []
  raw_lines = [l.strip().rstrip(';') for l in pcode.split('\n') if l.strip() and not l.strip().startswith('//')]
  # TODO: pcode.py should tokenize full pcode string instead of line-by-line, then this hack can be removed
  lines: list[str] = []
  for l in raw_lines:
    if lines and lines[-1].endswith('&&'): lines[-1] = lines[-1] + ' ' + l
    else: lines.append(l)
  _, final, _ = parse_block(lines, 0, env, assigns=assigns)
  sliced = set(d.split('[')[0] for d, _ in assigns if '[' in d)
  for var, val in final.items():
    if var in ['D0', 'S0', 'SCC', 'VCC', 'EXEC', 'PC', 'RETURN_DATA', 'VDATA'] and isinstance(val, UOp):
      if var in sliced and not any(re.match(rf'{var}\.\w+\s*=', l) for l in lines): continue
      for l in lines:
        if (m := re.match(rf'{var}\.(\w+(?:\[\w+\])?)', l)):
          assigns.append((f'{var}.{m.group(1)}', val))
          break
      else: assigns.append((var, val))
  return env, assigns

def _write_64bit(val: UOp, wfn, reg_or_addr, is_mem: bool, *args) -> list[UOp]:
  """Write a 64-bit value as two 32-bit writes. args passed to wfn after reg/addr and lo/hi value."""
  lo, hi = _split64(val)
  incr = 4 if is_mem else 1  # 4 bytes for memory addresses, 1 for register indices
  return [wfn(reg_or_addr, lo, *args), wfn(reg_or_addr + (UOp.const(reg_or_addr.dtype, incr) if isinstance(reg_or_addr, UOp) else incr), hi, *args)]

def _write_val(bits: int, val: UOp, wfn, reg_or_addr, *args, is_mem: bool = False) -> list[UOp]:
  """Write value, splitting 64-bit if needed. bits=64 for 64-bit writes, otherwise 32-bit."""
  return _write_64bit(val, wfn, reg_or_addr, is_mem, *args) if bits == 64 else [wfn(reg_or_addr, _to_u32(val), *args)]

def _mem_store(mem: UOp, addr: UOp, val: UOp, active: UOp, addr_bits: int = 32, data_bits: int = 32) -> list[UOp]:
  """Conditional memory store with sub-word support. Returns list of store UOps."""
  adt = dtypes.uint64 if addr_bits == 64 else dtypes.uint32
  word_addr = addr >> UOp.const(adt, 2)
  idx = mem.index(word_addr.cast(dtypes.int), active)
  if data_bits == 32: return [idx.store(active.where(_to_u32(val), idx))]
  # Sub-word store: read-modify-write with mask
  byte_pos = addr.cast(dtypes.uint32) & _c(3)
  byte_shift = byte_pos * _c(8)
  val_u32, size_mask = val.cast(dtypes.uint32), _c(0xFF if data_bits == 8 else 0xFFFF)
  mask = size_mask << byte_shift
  new_word = (idx & (mask ^ _c(0xFFFFFFFF))) | ((val_u32 & size_mask) << byte_shift)
  if data_bits == 8: return [idx.store(active.where(new_word, idx))]
  # 16-bit cross-word case: byte_pos == 3 means value spans two words
  is_cross = byte_pos.eq(_c(3))
  cross_word0 = (idx & _c(0x00FFFFFF)) | ((val_u32 & _c(0xFF)) << _c(24))
  store0 = idx.store(active.where(is_cross.where(cross_word0, new_word), idx))
  next_idx = mem.index((word_addr + UOp.const(adt, 1)).cast(dtypes.int), active & is_cross)
  cross_word1 = (next_idx & _c(0xFFFFFF00)) | ((val_u32 >> _c(8)) & _c(0xFF))
  return [store0, next_idx.store((active & is_cross).where(cross_word1, next_idx))]

def _mem_store_bytes(mem: UOp, addr: UOp, val: UOp, active: UOp, data_bits: int = 32) -> list[UOp]:
  """Store to byte-addressable memory (scratch). addr is byte offset, mem is uint8 buffer."""
  stores = []
  val_u32 = val.cast(dtypes.uint32) if val.dtype != dtypes.uint32 else val
  for i in range(data_bits // 8):
    byte_val = (val_u32 >> UOp.const(dtypes.uint32, i * 8)) & UOp.const(dtypes.uint32, 0xFF)
    stores.append(mem.index((addr + UOp.const(dtypes.uint64, i)).cast(dtypes.int), active).store(byte_val.cast(dtypes.uint8)))
  return stores

def _collect_data_slices(assigns: list[tuple[str, UOp]], data_prefix: str, pcode_vars: dict | None = None, op_name: str = "") -> dict[int, UOp]:
  """Collect bit slices from assigns into {dword_idx: value} dict."""
  slices = {}
  for dest, val in assigns:
    if dest.startswith(f'{data_prefix}['):
      if (m := re.match(rf'{data_prefix}\[(\d+)\s*:\s*(\d+)\]', dest)):
        hi_bit, low_bit = int(m.group(1)), int(m.group(2))
        dword_idx = low_bit // 32
        # D16 loads preserve bits - use final value from pcode_vars which has hi bits preserved
        if pcode_vars and 'D16' in op_name and dword_idx == 0 and hi_bit < 32:
          slices[0] = _to_u32(pcode_vars.get(data_prefix, val))
        else: slices[dword_idx] = _to_u32(val)
    elif dest.startswith(data_prefix): slices[0] = _to_u32(val)
  return slices

# ═══════════════════════════════════════════════════════════════════════════════
# INSTRUCTION COMPILER - converts decoded instruction to UOp SINK
# ═══════════════════════════════════════════════════════════════════════════════

class _Ctx:
  """Context for instruction compilation - holds buffers and helpers."""
  __slots__ = ('inst_size', 'dyn_fields', '_axis_id')
  sgpr = UOp(Ops.PARAM, dtypes.uint32.ptr(SGPR_COUNT), arg=0)
  vgpr = UOp(Ops.PARAM, dtypes.uint32.ptr(VGPR_SIZE), arg=1)
  vmem = UOp(Ops.PARAM, dtypes.uint32.ptr(1 << 46), arg=2)
  lds = UOp(Ops.PARAM, dtypes.uint32.ptr(16384), arg=3)
  scratch = UOp(Ops.PARAM, dtypes.uint8.ptr(1 << 30), arg=4)

  def __init__(self, inst_size: int):
    self.inst_size, self._axis_id = inst_size, 0
    self.dyn_fields: list[tuple[int, int]] = []  # (lo, hi) of fields read dynamically

  def range(self, n: int = WAVE_SIZE) -> UOp:
    """Create a lane range UOp with unique axis ID."""
    self._axis_id += 1
    return UOp.range(n, self._axis_id, AxisType.LOOP, dtype=dtypes.int)

  def unroll_lanes(self, get_lane_bit, exec_mask: UOp, apply_exec: bool = True) -> UOp:
    """Combine lane bits into a wave-sized mask using RANGE+REDUCE."""
    lane = self.range()
    bit = get_lane_bit(lane).cast(dtypes.uint64) << lane.cast(dtypes.uint64)
    result = bit.reduce(lane, arg=Ops.ADD)
    return result & exec_mask if apply_exec else result

  def inst_word(self, dword_idx: int) -> UOp:
    """Read instruction dword from vmem at PC + dword_idx*4."""
    pc = self.rpc()
    addr = pc if dword_idx == 0 else pc + UOp.const(dtypes.uint64, dword_idx * 4)
    return self.vmem.index((addr >> UOp.const(dtypes.uint64, 2)).cast(dtypes.int), ptr=True).load()

  def inst_field(self, field) -> UOp:
    """Extract field bits from instruction encoding. Tracks field for canonical key computation."""
    lo, hi = field.lo, field.hi
    self.dyn_fields.append((lo, hi))
    dword_idx = lo // 32
    lo_in_dword = lo % 32
    hi_in_dword = hi % 32
    word = self.inst_word(dword_idx)
    if lo // 32 == hi // 32:  # Same dword
      mask = (1 << (hi - lo + 1)) - 1
      shifted = word if lo_in_dword == 0 else word >> UOp.const(dtypes.uint32, lo_in_dword)
      return shifted & UOp.const(dtypes.uint32, mask)
    else:  # Spans two dwords
      lo_bits = 32 - lo_in_dword
      lo_mask = (1 << lo_bits) - 1
      hi_mask = (1 << (hi_in_dword + 1)) - 1
      lo_part = (word >> UOp.const(dtypes.uint32, lo_in_dword)) & UOp.const(dtypes.uint32, lo_mask)
      hi_part = self.inst_word(dword_idx + 1) & UOp.const(dtypes.uint32, hi_mask)
      return lo_part | (hi_part << UOp.const(dtypes.uint32, lo_bits))

  def inst_field_signed(self, field) -> UOp:
    """Extract field and sign-extend based on field width."""
    val = self.inst_field(field)
    width = field.hi - field.lo + 1
    sign_bit = 1 << (width - 1)
    return (val.cast(dtypes.int) ^ _c(sign_bit, dtypes.int)) - _c(sign_bit, dtypes.int)

  def canonical_mask(self, inst_bytes: bytes) -> tuple[int, int, int]:
    """Compute canonical (base, mask, size) for cache lookup.
    base = instruction bits with dynamic fields zeroed
    mask = bitmask with 1s for static bits, 0s for dynamic bits
    size = instruction size in bytes"""
    size = self.inst_size
    base = int.from_bytes(inst_bytes[:size], 'little')
    mask = (1 << (size * 8)) - 1  # all 1s initially
    for lo, hi in self.dyn_fields:
      field_mask = ((1 << (hi - lo + 1)) - 1) << lo
      base &= ~field_mask  # zero dynamic bits in base
      mask &= ~field_mask  # zero dynamic bits in mask
    return base, mask, size

  # Dynamic register access (takes UOp index instead of int)
  def rsgpr_dyn(self, reg: UOp, valid: UOp | None = None) -> UOp:
    """Read SGPR with dynamic register index."""
    if valid is not None: return self.sgpr.index(reg.cast(dtypes.int), valid, ptr=True).load()
    return self.sgpr.index(reg.cast(dtypes.int), ptr=True).load()

  def wsgpr_dyn(self, reg: UOp, val: UOp) -> UOp:
    """Write SGPR with dynamic register index. Writes to NULL (124) are discarded."""
    return self.sgpr.index(reg.cast(dtypes.int), reg.ne(_c(124))).store(val.cast(dtypes.uint32))

  def rmask_dyn(self, reg: UOp) -> UOp:
    lo = self.rsgpr_dyn(reg).cast(dtypes.uint64)
    if WAVE_SIZE == 64:
      hi = self.rsgpr_dyn(reg + _c(1)).cast(dtypes.uint64)
      return lo | (hi << _c(32, dtypes.uint64))
    return lo

  def wmask_dyn(self, reg: UOp, val: UOp) -> list[UOp]:
    mask = val.cast(dtypes.uint64)
    stores = [self.wsgpr_dyn(reg, mask.cast(dtypes.uint32))]
    if WAVE_SIZE == 64:
      stores.append(self.sgpr.index((reg + _c(1)).cast(dtypes.int), reg.ne(_c(124))).store((mask >> _c(32, dtypes.uint64)).cast(dtypes.uint32)))
    return stores

  def rexec(self) -> UOp: return self.rmask_dyn(_c(EXEC_LO.offset))
  def rvcc(self, reg: int = VCC_LO.offset) -> UOp: return self.rmask_dyn(_c(reg))

  def rvgpr_dyn(self, reg: UOp, lane: UOp, valid: UOp | None = None) -> UOp:
    """Read VGPR with dynamic register index."""
    idx = reg.cast(dtypes.int) * _c(WAVE_SIZE, dtypes.int) + lane.cast(dtypes.int)
    return self.vgpr.index(idx, valid, ptr=True).load() if valid is not None else self.vgpr.index(idx, ptr=True).load()

  def wvgpr_dyn(self, reg: UOp, lane: UOp, val: UOp, exec_mask: UOp, after: UOp | None = None) -> UOp:
    """Write VGPR with dynamic register index."""
    buf = self.vgpr.after(after) if after is not None else self.vgpr
    offset = reg.cast(dtypes.int) * _c(WAVE_SIZE, dtypes.int) + lane.cast(dtypes.int)
    return buf.index(offset, _lane_active(exec_mask, lane)).store(val.cast(dtypes.uint32))

  def rsrc_dyn(self, off: UOp, lane: UOp | None, bits: int = 32, literal: UOp | None = None, is_f64: bool = False, do_cast: bool = True) -> UOp:
    """Read source operand with dynamic offset. Handles SGPR/inline constants (<256), VGPR (>=256).
    If lane is None, only scalar access is supported (off must be < 256).
    is_f64: True for F64 operations where 64-bit literals go in high 32 bits."""
    is_float_const = (off >= _c(240)) & (off <= _c(248))
    is_vgpr = off >= _c(256)
    is_sgpr = is_vgpr.ne(True)
    sgpr_lo = self.rsgpr_dyn(off, is_sgpr)

    if lane is not None:
      vgpr_reg = off - _c(256)
      vgpr_lo = self.rvgpr_dyn(vgpr_reg, lane, is_vgpr)
      vgpr_val = _u64(vgpr_lo, self.rvgpr_dyn(vgpr_reg + _c(1), lane, is_vgpr)) if bits == 64 else vgpr_lo

    if bits == 64:
      sgpr_hi = self.rsgpr_dyn(off + _c(1), is_sgpr)
      sgpr_val = _u64(sgpr_lo, sgpr_hi)
      # Integer inline constants: sign-extend 32-bit value from buffer to 64-bit
      # Float constants: cast F32 to F64
      int_inline = sgpr_lo.cast(dtypes.int32).cast(dtypes.int64)
      float_inline = sgpr_lo.bitcast(dtypes.float32).cast(dtypes.float64)
      # compute inline
      inline = is_float_const.where(float_inline.bitcast(dtypes.uint64), int_inline.bitcast(dtypes.uint64))
      # Literal handling: F64 VOP puts literal in high 32 bits; B64/I64/U64 VOP and SOP zero-extend
      if literal is not None:
        lit_val = literal.cast(dtypes.uint64) << UOp.const(dtypes.uint64, 32) if is_f64 else literal.cast(dtypes.uint64)
        inline = off.eq(_c(255)).where(lit_val, inline)
      scalar_val = (off < _c(128)).where(sgpr_val, inline)
    else:
      scalar_val = sgpr_lo
      if literal is not None: scalar_val = off.eq(_c(255)).where(literal, scalar_val)
      if bits == 16 and do_cast:  # Float constants: cast F32 to F16
        scalar_val = is_float_const.where(scalar_val.bitcast(dtypes.float32).cast(dtypes.half).bitcast(dtypes.uint16).cast(dtypes.uint32), scalar_val)

    return is_vgpr.where(vgpr_val, scalar_val) if lane is not None else scalar_val

  def rpc(self) -> UOp:
    """Read PC as 64-bit byte address."""
    # Index at PC_LO, then cast to uint64 ptr and load
    return self.sgpr.index(_c(PC_LO_IDX, dtypes.int), ptr=True).cast(dtypes.uint64.ptr(SGPR_COUNT // 2)).load()

  def inc_pc(self) -> list[UOp]:
    """Increment PC by instruction size in bytes. Returns [store]."""
    new_pc = self.rpc() + UOp.const(dtypes.uint64, self.inst_size)
    return [self.sgpr.index(_c(PC_LO_IDX, dtypes.int), ptr=True).cast(dtypes.uint64.ptr(SGPR_COUNT // 2)).store(new_pc)]

  def scalar_stores(self, assigns: list[tuple[str, UOp]], sdst_reg: UOp, sdst_size: int = 1) -> list[UOp]:
    """Generate stores for scalar assigns with dynamic destination register (D0, SCC, EXEC, VCC)."""
    stores: list[UOp] = []
    for dest, val in assigns:
      if dest.startswith('D0'):
        if sdst_size == 2:
          lo, hi = _split64(val)
          stores.extend([self.wsgpr_dyn(sdst_reg, lo), self.wsgpr_dyn(sdst_reg + _c(1), hi)])
        else: stores.append(self.wsgpr_dyn(sdst_reg, _val_to_u32(val)))
      elif dest.startswith('SCC'): stores.append(self.wsgpr_dyn(_c(SCC.offset), _to_u32(val)))
      elif dest.startswith('EXEC'): stores.extend(self.wmask_dyn(_c(EXEC_LO.offset), val.cast(dtypes.uint64)))
      elif dest.startswith('VCC'): stores.extend(self.wmask_dyn(_c(VCC_LO.offset), val.cast(dtypes.uint64)))
    return stores

  def compile_sop_pcode(self, op, srcs: dict[str, UOp], sdst_reg: UOp, sdst_size: int) -> UOp:
    """Compile a scalar instruction with dynamic destination register."""
    pcode = get_pcode(op)
    srcs.update({'VCC': self.rvcc(), 'EXEC': self.rexec(), 'SCC': self.rsgpr_dyn(_c(SCC.offset))})
    if 'D0' not in srcs: srcs['D0'] = self.rsgpr_dyn(sdst_reg)  # D0 is current dest value for read-modify-write ops
    _, assigns = parse_pcode(pcode, srcs)
    return UOp.sink(*self.scalar_stores(assigns, sdst_reg, sdst_size), *self.inc_pc())

  def compile_lane_pcode(self, op, inst) -> UOp:
    """Compile cross-lane ops (READLANE/WRITELANE/PERMLANE) using pcode parser."""
    pcode = get_pcode(op)
    op_name = op.name if hasattr(op, 'name') else str(op)
    src0_off, vdst_off = self.inst_field(type(inst).src0), self.inst_field(type(inst).vdst)
    src0_reg = (src0_off >= _c(256)).where(src0_off - _c(256), _c(0))  # VGPR index or 0
    src1_off = self.inst_field(type(inst).src1) if hasattr(type(inst), 'src1') else None
    src2_off = self.inst_field(type(inst).src2) if hasattr(type(inst), 'src2') else None
    exec_lo, exec_mask = self.rsgpr_dyn(_c(EXEC_LO.offset)), self.rexec()
    srcs = {
      'SRC0': src0_reg, 'VDST': vdst_off, 'EXEC_LO': exec_lo, 'EXEC': exec_mask, '_vgpr': self.vgpr,
      'S0': self.rsrc_dyn(src0_off, _c(0, dtypes.int)) if 'WRITELANE' in op_name else src0_reg,
      'S1': self.rsrc_dyn(src1_off, _c(0, dtypes.int)) if src1_off is not None else _c(0),
      'S2': self.rsrc_dyn(src2_off, _c(0, dtypes.int)) if src2_off is not None else _c(0),
    }
    _, assigns = parse_pcode(pcode, srcs)
    stores = []
    for dest, val in assigns:
      if dest.startswith('D0'): stores.append(self.wsgpr_dyn(vdst_off, val.cast(dtypes.uint32)))
      elif dest.startswith('VGPR['): stores.append(self.vgpr.index(val[0].cast(dtypes.int)).store(val[1].cast(dtypes.uint32)))
    return UOp.sink(*stores, *self.inc_pc())

  def compile_vop_pcode(self, op, srcs: dict[str, UOp], lane: UOp, vdst_reg: UOp, exec_mask: UOp,
                        opsel_dst_hi: bool | UOp = False, sdst_reg: int | None = None, clmp: int = 0,
                        src0_off: UOp | None = None, sdwa_dst_sel: int | None = None, sdwa_dst_unused: int = 0) -> UOp:
    """Compile VOP instruction. Returns sink with stores and inc_pc."""
    pcode = get_pcode(op)
    vcc_reg = sdst_reg if sdst_reg is not None else VCC_LO.offset
    if 'VCC' not in srcs: srcs['VCC'] = self.rvcc(vcc_reg)
    srcs.update({'EXEC': exec_mask, 'SCC': self.rsgpr_dyn(_c(SCC.offset)), 'laneId': lane, 'VDST': vdst_reg,
                 'ROUND_MODE': _c(0), 'ROUND_TOWARD_ZERO': _c(0), 'ROUND_NEAREST_EVEN': _c(0)})  # rounding mode constants
    _, assigns = parse_pcode(pcode, srcs)

    # For integer ops with clamp, compute overflow using wide arithmetic
    # NOTE: MUL_LO ops don't saturate - they always return the low bits
    int_saturate = None
    if clmp and any(p in op.name for p in ('_NC_U', '_MAD_U', '_NC_I', '_MAD_I')):
      is_signed, is_16bit = '_I' in op.name and '_U' not in op.name, '16' in op.name
      if not (is_16bit and is_signed):  # Skip 16-bit signed ops due to codegen issues
        s0, s1, s2 = srcs.get('S0'), srcs.get('S1'), srcs.get('S2')
        if s0 is not None and s1 is not None:
          narrow_dt = dtypes.uint16 if is_16bit else (dtypes.int32 if is_signed else dtypes.uint32)
          wide_dt = dtypes.int32 if is_16bit else dtypes.int64
          narrow_max, narrow_min = (0xFFFF, 0) if is_16bit else ((0x7FFFFFFF, -0x80000000) if is_signed else (0xFFFFFFFF, 0))
          def to_wide(x): return (x.bitcast(narrow_dt) if x.dtype.itemsize == narrow_dt.itemsize else x.cast(narrow_dt)).cast(wide_dt)
          is_sub, is_mad = 'SUB' in op.name, 'MAD' in op.name
          full = (to_wide(s0) * to_wide(s1) + to_wide(s2)) if is_mad and s2 is not None else \
                 (to_wide(s1) - to_wide(s0)) if is_sub and 'SUBREV' in op.name else \
                 (to_wide(s0) - to_wide(s1)) if is_sub else (to_wide(s0) + to_wide(s1))
          int_saturate = full.clamp(narrow_min, narrow_max).cast(narrow_dt)

    raw_stores: list = []
    vcc_val, exec_val = None, None
    for dest, val in assigns:
      if 'D0' in dest and '[laneId]' in dest:
        vcc_val = val
      elif dest.startswith('D0'):
        if (slice_match := re.match(r'D0\[(\d+)\s*:\s*(\d+)\]', dest)):
          hi_bit, lo_bit = int(slice_match.group(1)), int(slice_match.group(2))
          if hi_bit != 31 or lo_bit != 0:
            width = hi_bit - lo_bit + 1
            raw_mask = ((1 << width) - 1) if width < 32 else 0xFFFFFFFF
            val_bits = val.bitcast(dtypes.uint16).cast(dtypes.uint32) if val.dtype == dtypes.half else \
                       val.cast(dtypes.uint32) if val.dtype in (dtypes.uint16, dtypes.int16) else \
                       val.cast(dtypes.uint32) & UOp.const(dtypes.uint32, raw_mask)
            # Aligned 32-bit slices map directly to destination dwords (e.g. D0[63:32] -> vdst+1).
            if width == 32 and lo_bit % 32 == 0:
              raw_stores.append(('vgpr', self.wvgpr_dyn(vdst_reg + _c(lo_bit // 32), lane, val_bits, exec_mask)))
              continue
            # General (sub-dword) slice write; defer merge per target dword.
            dword_idx, lo_local = lo_bit // 32, lo_bit % 32
            if lo_local + width <= 32:
              raw_stores.append(('vgpr_slice', (dword_idx, lo_local, width, val_bits)))
            else:
              # Split cross-dword slices.
              lo_w = 32 - lo_local
              hi_w = width - lo_w
              raw_stores.append(('vgpr_slice', (dword_idx, lo_local, lo_w, val_bits)))
              raw_stores.append(('vgpr_slice', (dword_idx + 1, 0, hi_w, val_bits >> UOp.const(dtypes.uint32, lo_w))))
            continue
        # For integer ops with clamp, use pre-computed saturated value; for floats, clamp to [0,1]
        if int_saturate is not None: val = int_saturate
        elif clmp and val.dtype in (dtypes.float32, dtypes.half, dtypes.float64):
          val = val.maximum(UOp.const(val.dtype, 0.0)).minimum(UOp.const(val.dtype, 1.0))
        if val.dtype in (dtypes.uint64, dtypes.int64, dtypes.float64):
          lo, hi = _split64(val)
          raw_stores.extend([('vgpr', self.wvgpr_dyn(vdst_reg, lane, lo, exec_mask)),
                             ('vgpr', self.wvgpr_dyn(vdst_reg + _c(1), lane, hi, exec_mask))])
        else:
          if val.dtype in (dtypes.half, dtypes.uint16, dtypes.int16):
            result, old_val = _val_to_u32(val), self.rvgpr_dyn(vdst_reg, lane)
            hi_result = (old_val & UOp.const(dtypes.uint32, 0xFFFF)) | (result << UOp.const(dtypes.uint32, 16))
            lo_result = (old_val & UOp.const(dtypes.uint32, 0xFFFF0000)) | (result & UOp.const(dtypes.uint32, 0xFFFF))
            result = opsel_dst_hi.where(hi_result, lo_result) if isinstance(opsel_dst_hi, UOp) else hi_result if opsel_dst_hi else lo_result
          else: result = _val_to_u32(val)
          if sdwa_dst_sel is not None:
            result = _apply_sdwa_dst(self.rvgpr_dyn(vdst_reg, lane), result, sdwa_dst_sel, sdwa_dst_unused)
          raw_stores.append(('vgpr', self.wvgpr_dyn(vdst_reg, lane, result, exec_mask)))
      elif dest.startswith('S0') and src0_off is not None:
        # Write back to src0 VGPR (e.g. v_swap_b32). src0_off is raw encoding (256+ = VGPR)
        src0_vgpr = src0_off - _c(256)
        raw_stores.append(('vgpr_s0', self.wvgpr_dyn(src0_vgpr, lane, _val_to_u32(val), exec_mask)))
      elif dest.startswith('VCC'): vcc_val = val
      elif dest.startswith('EXEC'): exec_val = val
      elif dest.startswith('SCC'): raw_stores.append(('scc', self.wsgpr_dyn(_c(SCC.offset), _to_u32(val))))

    stores, lane_stores, scalar_stores = [], [s for t, s in raw_stores if t in ('vgpr', 'vgpr_s0')], [s for t, s in raw_stores if t == 'scc']
    slice_stores = [s for t, s in raw_stores if t == 'vgpr_slice']
    if slice_stores:
      by_dword: dict[int, list[tuple[int, int, UOp]]] = {}
      for dword_idx, lo_bit, width, val_bits in slice_stores:
        by_dword.setdefault(dword_idx, []).append((lo_bit, width, val_bits))
      for dword_idx, slices in by_dword.items():
        result = self.rvgpr_dyn(vdst_reg + _c(dword_idx), lane)
        for lo_bit, width, val_bits in slices:
          value_mask = 0xFFFFFFFF if width == 32 else (1 << width) - 1
          masked = val_bits if width == 32 else (val_bits & UOp.const(dtypes.uint32, value_mask))
          shifted = masked if lo_bit == 0 else (masked << UOp.const(dtypes.uint32, lo_bit))
          mask = UOp.const(dtypes.uint32, (value_mask << lo_bit) & MASK32)
          result = (result & (mask ^ UOp.const(dtypes.uint32, 0xFFFFFFFF))) | shifted
        lane_stores.append(self.wvgpr_dyn(vdst_reg + _c(dword_idx), lane, result, exec_mask))
    if lane_stores: stores.append(UOp.sink(*lane_stores).end(lane))
    for mask_val, reg in [(vcc_val, vcc_reg), (exec_val, EXEC_LO.offset)]:
      if mask_val is None: continue
      def get_bit(l, v=mask_val): return (_to_u32(v.substitute({lane: l})) & _c(1)).cast(dtypes.uint64)
      stores.extend(self.wmask_dyn(_c(reg), self.unroll_lanes(get_bit, exec_mask, apply_exec=False)))
    stores.extend(scalar_stores)
    return UOp.sink(*stores, *self.inc_pc())

# ═══════════════════════════════════════════════════════════════════════════════
# INSTRUCTION HANDLERS
# ═══════════════════════════════════════════════════════════════════════════════

def _compile_sopp(inst: ir3.SOPP | ir4.SOPP, ctx: _Ctx) -> UOp:
  simm16 = ctx.inst_field_signed(type(inst).simm16).cast(dtypes.int16)
  if inst.op in (ir3.SOPPOp.S_ENDPGM, ir4.SOPPOp.S_ENDPGM, irc.SOPPOp.S_ENDPGM):
    return UOp.sink(ctx.wsgpr_dyn(_c(PC_LO_IDX), UOp.const(dtypes.uint32, 0xFFFFFFFF)),
                          ctx.wsgpr_dyn(_c(PC_HI_IDX), UOp.const(dtypes.uint32, 0xFFFFFFFF)))
  # S_BARRIER: advance PC past the barrier instruction. The execution loop detects barriers before executing and handles synchronization.
  barrier_ops = {ir3.SOPPOp.S_BARRIER, irc.SOPPOp.S_BARRIER}
  if hasattr(ir4.SOPPOp, 'S_BARRIER_WAIT'): barrier_ops.add(ir4.SOPPOp.S_BARRIER_WAIT)
  if inst.op in barrier_ops: return UOp.sink(*ctx.inc_pc())
  # S_NOP and S_WAITCNT are no-ops in emulator (no pipeline/cache to wait on)
  if inst.op in (ir3.SOPPOp.S_NOP, ir4.SOPPOp.S_NOP, irc.SOPPOp.S_NOP, irc.SOPPOp.S_WAITCNT): return UOp.sink(*ctx.inc_pc())
  # NOTE: we ignore SOPPs without PCODE
  if inst.op in _get_pcode_dict(inst.op):
    pcode = get_pcode(inst.op)
    pc_bytes = ctx.rpc()  # PC is already 64-bit byte address
    vcc, exec_mask = ctx.rvcc(), ctx.rexec()
    srcs = {'PC': pc_bytes.cast(dtypes.int64), 'SIMM16': simm16, 'SCC': ctx.rsgpr_dyn(_c(SCC.offset)), 'VCC': vcc,
            'VCCZ': vcc.eq(UOp.const(dtypes.uint64, 0)).cast(dtypes.uint32), 'EXECZ': exec_mask.eq(UOp.const(dtypes.uint64, 0)).cast(dtypes.uint32)}
    for dest, val in parse_pcode(pcode, srcs)[1]:
      if dest == 'PC' or dest.startswith('PC.'):
        lo, hi = _split64(val.cast(dtypes.uint64))
        return UOp.sink(ctx.wsgpr_dyn(_c(PC_LO_IDX), lo), ctx.wsgpr_dyn(_c(PC_HI_IDX), hi))
  return UOp.sink(*ctx.inc_pc())

def _compile_smem(inst: ir3.SMEM | ir4.SMEM, ctx: _Ctx) -> UOp:
  # Cache invalidation instructions are no-ops in the emulator (we don't model caches)
  if '_INV' in inst.op.name: return UOp.sink(*ctx.inc_pc())
  # Dynamic sbase field (bits 5:0) - SGPR pair, field value * 2 = register offset
  sbase = ctx.inst_field(type(inst).sbase) * _c(2)
  # Dynamic sdata field (bits 12:6) - destination SGPR
  sdata_reg = ctx.inst_field(type(inst).sdata)
  # RDNA4 uses 'ioffset', RDNA3 uses 'offset' - use type(inst) to get correct field
  offset_field = type(inst).ioffset if hasattr(type(inst), 'ioffset') else type(inst).offset  # type: ignore[union-attr]
  offset = ctx.inst_field_signed(offset_field)  # signed immediate
  # Dynamic soffset field - SGPR for additional offset (NULL=124 reads as 0, CDNA soffset_en=0 means no soffset)
  soffset_val = _c(0).cast(dtypes.uint64)
  if not (isinstance(inst, irc.SMEM) and not inst.soffset_en):
    soffset_val = ctx.rsgpr_dyn(ctx.inst_field(type(inst).soffset)).cast(dtypes.uint64)
  addr = _u64(ctx.rsgpr_dyn(sbase), ctx.rsgpr_dyn(sbase + _c(1))) + offset.cast(dtypes.uint64) + soffset_val
  # S_LOAD_(DTYPE) series: B32/DWORD=1, B64/DWORDX2=2, U8=0.25, I8=-0.25, etc.
  op_name = _op_name(inst)
  assert (op_name).startswith('S_LOAD_'), f"unexpected SMEM op: {op_name}"
  part = op_name.rsplit('_', 1)[1]  # B32, DWORD, DWORDX2, U8, I8, etc.
  nval = int(part.removeprefix('DWORD').removeprefix('X') or '1') if 'DWORD' in part else int(part[1:]) / 32 * (-1 if part[0] == 'I' else 1)
  ndwords = max(1, int(abs(nval)))
  dword_base = addr >> UOp.const(dtypes.uint64, 2)
  vals = [ctx.vmem.index((dword_base + UOp.const(dtypes.uint64, i)).cast(dtypes.int)) for i in range(ndwords)]
  if abs(nval) < 1:
    nbits = int(abs(nval) * 32)
    byte_off = (addr & UOp.const(dtypes.uint64, 3)).cast(dtypes.uint32) * UOp.const(dtypes.uint32, 8)
    extracted = (vals[0] >> byte_off) & UOp.const(dtypes.uint32, (1 << nbits) - 1)
    vals[0] = extracted.cast({8: dtypes.int8, 16: dtypes.int16}[nbits]).cast(dtypes.int32).bitcast(dtypes.uint32) if nval < 0 else extracted
  stores = [ctx.wsgpr_dyn(sdata_reg + _c(i), vals[i]) for i in range(ndwords)]
  return UOp.sink(*stores, *ctx.inc_pc())

def _compile_sop(inst: ir3.SOP1|ir3.SOP2|ir3.SOPC|ir3.SOPK|ir4.SOP1|ir4.SOP2|ir4.SOPC|ir4.SOPK|irc.SOP1|irc.SOP2|irc.SOPC|irc.SOPK, ctx: _Ctx) -> UOp:
  bits = inst.canonical_op_bits
  literal = ctx.inst_field(type(inst).literal) if hasattr(type(inst), 'literal') else None  # type: ignore[union-attr]

  if isinstance(inst, (ir3.SOPK, ir4.SOPK, irc.SOPK)):
    sdst_off = ctx.inst_field(type(inst).sdst)
    simm16 = ctx.inst_field(type(inst).simm16)
    # Sign-extend simm16
    simm16_sext = simm16.cast(dtypes.int16).cast(dtypes.int32)
    # RDNA4 pcodes use S0.i16 for the immediate (e.g., S_MULK_I32), RDNA3 uses S0 for the register (e.g., S_CMPK_*)
    # CDNA pcode uses S0 for the immediate in MOVK/MULK/ADDK/CMOVK, but S0 = register for CMPK/SETREG
    op_name = _op_name(inst)
    if isinstance(inst, ir4.SOPK): s0 = simm16
    elif isinstance(inst, irc.SOPK) and 'CMPK' not in op_name and 'SETREG' not in op_name: s0 = simm16_sext
    else: s0 = ctx.rsgpr_dyn(sdst_off)
    srcs = {'S0': s0, 'S1': simm16_sext, 'SIMM16': simm16_sext, 'D0': ctx.rsgpr_dyn(sdst_off)}
    dst_off, dst_size = sdst_off, 1
  elif isinstance(inst, (ir3.SOP1, ir4.SOP1, irc.SOP1)):
    sdst_off = ctx.inst_field(type(inst).sdst)
    ssrc0_off = ctx.inst_field(type(inst).ssrc0)
    srcs = {'S0': ctx.rsrc_dyn(ssrc0_off, None, bits['s0'], literal)}
    dst_off, dst_size = sdst_off, bits['d'] // 32
  elif isinstance(inst, (ir3.SOP2, ir4.SOP2, irc.SOP2)):
    sdst_off = ctx.inst_field(type(inst).sdst)
    ssrc0_off = ctx.inst_field(type(inst).ssrc0)
    ssrc1_off = ctx.inst_field(type(inst).ssrc1)
    srcs = {'S0': ctx.rsrc_dyn(ssrc0_off, None, bits['s0'], literal),
            'S1': ctx.rsrc_dyn(ssrc1_off, None, bits['s1'], literal)}
    if literal is not None: srcs['SIMM32'] = literal
    dst_off, dst_size = sdst_off, bits['d'] // 32
  elif isinstance(inst, (ir3.SOPC, ir4.SOPC, irc.SOPC)):
    ssrc0_off = ctx.inst_field(type(inst).ssrc0)
    ssrc1_off = ctx.inst_field(type(inst).ssrc1)
    srcs = {'S0': ctx.rsrc_dyn(ssrc0_off, None, bits['s0'], literal),
            'S1': ctx.rsrc_dyn(ssrc1_off, None, bits['s1'], literal)}
    dst_off, dst_size = _c(0), 0  # SOPC writes to SCC, not sdst
  else:
    raise RuntimeError(f"unknown SOP type: {type(inst).__name__}")

  return ctx.compile_sop_pcode(inst.op, srcs, dst_off, dst_size)

def _compile_vop12(inst: ir3.VOP1 | ir3.VOP1_SDST | ir3.VOP2 | ir4.VOP1 | ir4.VOP1_SDST | ir4.VOP2 | irc.VOP1 | irc.VOP2, ctx: _Ctx) -> UOp:
  op_name = _op_name(inst)
  if op_name in ('V_READFIRSTLANE_B32_E32', 'V_PERMLANE64_B32_E32'): return ctx.compile_lane_pcode(inst.op, inst)
  is_sdwa = isinstance(inst, (irc.VOP1_SDWA, irc.VOP2_SDWA, irc.VOP2_SDWA_SDST))
  has_sdwa_dst = isinstance(inst, (irc.VOP1_SDWA, irc.VOP2_SDWA))
  lane, exec_mask, bits = ctx.range(), ctx.rexec(), inst.canonical_op_bits
  literal = ctx.inst_field(type(inst).literal) if hasattr(type(inst), 'literal') else None  # type: ignore[union-attr]
  vdst_reg = ctx.inst_field(type(inst).vdst)
  write_hi_half = bits['d'] == 16 and (vdst_reg >= _c(128))
  if isinstance(write_hi_half, UOp): vdst_reg = write_hi_half.where(vdst_reg - _c(128), vdst_reg)
  elif write_hi_half: vdst_reg -= 128
  if isinstance(inst, (ir3.VOP1, ir4.VOP1, irc.VOP1)):
    # Handle VOP1 hi-half source operand (src0 >= v[128] for 16-bit ops)
    if is_sdwa:
      vsrc0_reg = ctx.inst_field(type(inst).vsrc0)  # type: ignore[union-attr]
      s0 = _apply_sdwa_sel(ctx.rvgpr_dyn(vsrc0_reg, lane), inst.src0_sel)
      s0 = _apply_sdwa_sext(s0, inst.src0_sel, inst.src0_sext)
    else:
      src0_off = ctx.inst_field(type(inst).src0)
      s0 = ctx.rsrc_dyn(src0_off, lane, bits['s0'], literal)
      if bits['s0'] == 16:
        src0_hi = src0_off >= _c(384)
        # Only compute hi-half when src0_off >= 384, use guarded index to prevent OOB access
        src0_reg = src0_hi.where(src0_off - _c(384), _c(0))
        s0 = src0_hi.where(_hi16(ctx.rvgpr_dyn(src0_reg, lane)), s0)
    d0 = _cond_hi16(write_hi_half, ctx.rvgpr_dyn(vdst_reg, lane))
    srcs = {'S0': s0, 'D0': d0}
    src0_off = src0_off if not is_sdwa else _c(256) + ctx.inst_field(type(inst).vsrc0)  # type: ignore[union-attr]
  else:
    vsrc1_reg = ctx.inst_field(type(inst).vsrc1)
    vsrc1_hi = bits['s0'] == 16 and (vsrc1_reg >= _c(128))
    vsrc1_actual = _cond(vsrc1_hi, vsrc1_reg - _c(128), vsrc1_reg)
    # 64 bit
    s1 = _u64(ctx.rvgpr_dyn(vsrc1_actual, lane), ctx.rvgpr_dyn(vsrc1_actual + _c(1), lane)) if bits.get('s1', 32) == 64 \
      else _cond_hi16(vsrc1_hi, ctx.rvgpr_dyn(vsrc1_actual, lane))
    d0 = _cond_hi16(write_hi_half, ctx.rvgpr_dyn(vdst_reg, lane))  # FMAC/FMAMK hi-half dest needs hi-half accumulator
    if is_sdwa:
      vsrc0_reg = ctx.inst_field(type(inst).vsrc0)  # type: ignore[union-attr]
      s0 = _apply_sdwa_sel(ctx.rvgpr_dyn(vsrc0_reg, lane), inst.src0_sel)
      s1 = _apply_sdwa_sel(s1, inst.src1_sel)
      s0 = _apply_sdwa_sext(s0, inst.src0_sel, inst.src0_sext)
      s1 = _apply_sdwa_sext(s1, inst.src1_sel, inst.src1_sext)
      src0_off = _c(256) + vsrc0_reg
    else:
      # Handle VOP2 hi-half src0 operand (src0 >= v[128] for 16-bit ops)
      src0_off = ctx.inst_field(type(inst).src0)
      s0 = ctx.rsrc_dyn(src0_off, lane, bits['s0'], literal)
      if bits['s0'] == 16:
        src0_hi = src0_off >= _c(384)
        # Only compute hi-half when src0_off >= 384, use guarded index to prevent OOB access
        src0_reg = src0_hi.where(src0_off - _c(384), _c(0))
        s0 = src0_hi.where(_hi16(ctx.rvgpr_dyn(src0_reg, lane)), s0)
    srcs = {'S0': s0, 'S1': s1, 'D0': d0}
    # FMAAK_(DTYPE)_E32 series
    if 'V_FMAA' in _op_name(inst) or 'V_FMAM' in _op_name(inst):
      assert literal is not None
      srcs['SIMM32'] = literal
  return ctx.compile_vop_pcode(inst.op, srcs, lane, vdst_reg, exec_mask, opsel_dst_hi=write_hi_half, src0_off=src0_off,
                               sdwa_dst_sel=getattr(inst, 'dst_sel', None) if has_sdwa_dst else None,
                               sdwa_dst_unused=getattr(inst, 'dst_unused', 0) if has_sdwa_dst else 0)

def _compile_vopc(inst: ir3.VOPC|ir3.VOP3|ir4.VOPC|ir4.VOP3|irc.VOPC|irc.VOP3, ctx: _Ctx,
                  opsel: int = 0, abs_bits: int = 0, neg_bits: int = 0) -> UOp:
  exec_mask, op_name, bits = ctx.rexec(), _op_name(inst), inst.canonical_op_bits
  is_cmpx, is_vopc = 'CMPX' in op_name, hasattr(inst, 'vsrc1')  # is_vopc: e32 vs e64
  is_sdwa = isinstance(inst, irc.VOPC_SDWA_SDST)
  is_cdna = 'cdna' in type(inst).__module__

  # Handle both VOPC (vsrc1) and VOP3 (src1) instruction formats - read operands dynamically
  if is_vopc:
    if is_sdwa:
      vsrc0_reg = ctx.inst_field(type(inst).vsrc0)  # type: ignore[union-attr]
      src0_off = _c(256) + vsrc0_reg  # VGPR offset
      sdwa_sdst = ctx.inst_field(type(inst).sdst)  # type: ignore[union-attr]
    else:
      src0_off = ctx.inst_field(type(inst).src0)
    vsrc1_off = ctx.inst_field(type(inst).vsrc1)  # type: ignore[union-attr]
    # For 16-bit ops, vsrc1 >= 128 means hi-half of v[vsrc1-128]
    if bits['s0'] == 16:
      vsrc1_hi = vsrc1_off >= _c(128)
      src1_off = _c(256) + vsrc1_hi.where(vsrc1_off - _c(128), vsrc1_off)
    else:
      vsrc1_hi = False
      src1_off = _c(256) + vsrc1_off
  else:
    src0_off = ctx.inst_field(type(inst).src0)
    src1_off = ctx.inst_field(type(inst).src1)  # type: ignore[union-attr]
    dst_off = ctx.inst_field(type(inst).vdst)  # type: ignore[union-attr]
    vsrc1_hi = False
  literal = ctx.inst_field(type(inst).literal) if hasattr(type(inst), 'literal') else None  # type: ignore[union-attr]

  is_float, is_f64, pcode = any(x in op_name for x in ('_F32', '_F64', '_F16')), '_F64' in op_name, get_pcode(inst.op)
  def get_cmp_bit(lane) -> UOp:
    lc = lane.cast(dtypes.int) if isinstance(lane, UOp) else _c(lane, dtypes.int)
    s0 = ctx.rsrc_dyn(src0_off, lc, bits['s0'], literal, is_f64)
    s1 = _cond_hi16(vsrc1_hi, ctx.rsrc_dyn(src1_off, lc, bits['s1'], literal, is_f64)) if bits['s0'] == 16 \
      else ctx.rsrc_dyn(src1_off, lc, bits['s1'], literal, is_f64)
    if is_sdwa:
      s0, s1 = _apply_sdwa_sel(s0, inst.src0_sel), _apply_sdwa_sel(s1, inst.src1_sel)
      s0, s1 = _apply_sdwa_sext(s0, inst.src0_sel, inst.src0_sext), _apply_sdwa_sext(s1, inst.src1_sel, inst.src1_sext)
    if bits['s0'] == 16 and opsel: s0, s1 = _apply_opsel(s0, 0, opsel), _apply_opsel(s1, 1, opsel)
    if is_float:
      s0 = _apply_src_mods(s0, 0, abs_bits, neg_bits, bits['s0'])
      s1 = _apply_src_mods(s1, 1, abs_bits, neg_bits, bits['s1'])
    for dest, val in parse_pcode(pcode, {'S0': s0, 'S1': s1, 'laneId': lc})[1]:
      if '[laneId]' in dest and ('D0' in dest or 'EXEC' in dest): return val.cast(dtypes.uint64)
    return _c(0)

  new_bits = ctx.unroll_lanes(get_cmp_bit, exec_mask, apply_exec=False)
  # Both VOPC and VOP3 clear inactive lane bits (hardware verified)
  new_result = new_bits & exec_mask
  cmp_mask_bits = 64 if is_cdna else 32
  def _mask_stores(base_reg: UOp, mask: UOp, width_bits: int) -> list[UOp]:
    return ctx.wmask_dyn(base_reg, mask.cast(dtypes.uint64)) if width_bits == 64 else [ctx.wsgpr_dyn(base_reg, _to_u32(mask))]

  # CMPX e32: writes EXEC only; CMPX e64: writes both EXEC and SDST; non-CMPX: writes dst only
  if is_cmpx:
    stores = _mask_stores(_c(EXEC_LO.offset), new_result, cmp_mask_bits)
    if not is_vopc: stores += _mask_stores(dst_off, new_result, cmp_mask_bits)
  else:
    if is_sdwa: stores = _mask_stores(sdwa_sdst, new_result, cmp_mask_bits)
    elif not is_vopc: stores = _mask_stores(dst_off, new_result, cmp_mask_bits)
    else: stores = _mask_stores(_c(VCC_LO.offset), new_result, cmp_mask_bits)
  return UOp.sink(*stores, *ctx.inc_pc())

def _compile_vop3(inst: ir3.VOP3 | ir4.VOP3 | irc.VOP3, ctx: _Ctx) -> UOp:
  exec_mask = ctx.rexec()
  bits = inst.canonical_op_bits
  opsel, op_name = getattr(inst, 'opsel', 0) or 0, _op_name(inst)

  # Lane operations
  if op_name in ('V_READLANE_B32', 'V_READFIRSTLANE_B32', 'V_READFIRSTLANE_B32_E64', 'V_WRITELANE_B32'):
    return ctx.compile_lane_pcode(inst.op, inst)

  # V_PERMLANE16_B32 / V_PERMLANEX16_B32: cross-lane swizzle via pcode
  if 'PERMLANE16' in op_name or 'PERMLANEX16' in op_name:
    return ctx.compile_lane_pcode(inst.op, inst)

  # VOP3 VOPC (v_cmp_*_e64) - delegate to unified VOPC handler
  if 'V_CMP' in op_name or 'V_CMPX' in op_name:
    return _compile_vopc(inst, ctx, opsel=opsel, abs_bits=getattr(inst, 'abs', 0) or 0, neg_bits=getattr(inst, 'neg', 0) or 0)

  # VOP3 specific fields
  vdst_reg = ctx.inst_field(type(inst).vdst)
  literal = ctx.inst_field(type(inst).literal) if hasattr(type(inst), 'literal') else None  # type: ignore[union-attr]
  abs_bits, neg_bits = getattr(inst, 'abs', 0) or 0, getattr(inst, 'neg', 0) or 0

  # VOP3_SDST: v_s_* instructions goes to SGPR
  if 'V_S_' in op_name:
    src0 = _apply_src_mods(ctx.rsrc_dyn(ctx.inst_field(type(inst).src0), _c(0, dtypes.int), bits['s0'], literal), 0, abs_bits, neg_bits, bits['s0'])
    srcs = {'S0': src0, 'EXEC': exec_mask, 'SCC': ctx.rsgpr_dyn(_c(SCC.offset)), 'laneId': _c(0, dtypes.int),
            'ROUND_MODE': _c(0), 'ROUND_TOWARD_ZERO': _c(0), 'OPSEL': _c(opsel), 'NEG': _c(neg_bits), 'ABS': _c(abs_bits),
            'OMOD': _c(getattr(inst, 'omod', 0) or 0)}
    _, assigns = parse_pcode(get_pcode(inst.op), srcs)
    stores = [ctx.wsgpr_dyn(vdst_reg, _val_to_u32(val)) for dest, val in assigns if dest.startswith('D0')]
    return UOp.sink(*stores, *ctx.inc_pc())

  # Regular VOP3 - read operands dynamically
  lane = ctx.range()
  ops = inst.canonical_operands
  src0 = ctx.rsrc_dyn(ctx.inst_field(type(inst).src0), lane, bits['s0'], literal, 's0' in ops and ops['s0'][0] == Fmt.FMT_NUM_F64)
  src1 = ctx.rsrc_dyn(ctx.inst_field(type(inst).src1), lane, bits['s1'], literal, 's1' in ops and ops['s1'][0] == Fmt.FMT_NUM_F64)
  src2 = ctx.rsrc_dyn(ctx.inst_field(type(inst).src2), lane, bits['s2'], literal, 's2' in ops and ops['s2'][0] == Fmt.FMT_NUM_F64)

  # V_BITOP3_B16/V_BITOP3_B32: 3-input truth table using repurposed omod/abs/neg fields
  if 'BITOP3' in op_name:
    omod = getattr(inst, 'omod', 0) or 0
    ttbl = (omod << 6) | (abs_bits << 3) | neg_bits
    width = 16 if 'B16' in op_name else 32
    mask = _c((1 << width) - 1)
    s0, s1, s2 = src0 & mask, src1 & mask, src2 & mask
    ns0, ns1, ns2 = s0 ^ mask, s1 ^ mask, s2 ^ mask
    result = _c(0)
    minterms = [(ns0, ns1, ns2), (ns0, ns1, s2), (ns0, s1, ns2), (ns0, s1, s2),
                (s0, ns1, ns2), (s0, ns1, s2), (s0, s1, ns2), (s0, s1, s2)]
    for i, (a, b, c) in enumerate(minterms):
      if ttbl & (1 << i): result = result | (a & b & c)
    result = result & mask
    store = ctx.wvgpr_dyn(vdst_reg, lane, _to_u32(result), exec_mask)
    return UOp.sink(UOp.group(store).end(lane), *ctx.inc_pc())

  if bits['s0'] == 16:
    src0 = _apply_opsel(src0, 0, opsel)
    src1 = _apply_opsel(src1, 1, opsel)
    src2 = _apply_opsel(src2, 2, opsel)
  src0 = _apply_src_mods(src0, 0, abs_bits, neg_bits, bits['s0'])
  src1 = _apply_src_mods(src1, 1, abs_bits, neg_bits, bits['s1'])
  src2 = _apply_src_mods(src2, 2, abs_bits, neg_bits, bits['s2'])
  srcs = {'S0': src0, 'S1': src1, 'S2': src2, 'OPSEL': _c(opsel), 'NEG': _c(neg_bits), 'ABS': _c(abs_bits),
          'OMOD': _c(getattr(inst, 'omod', 0) or 0)}
  #irx_CNDMASK series
  if 'CNDMASK' in op_name and src2 is not None: srcs['VCC'] = src2
  # FMAC instructions need D0 (accumulator) from destination register
  if 'FMAC' in op_name: srcs['D0'] = ctx.rvgpr_dyn(vdst_reg, lane)
  opsel_dst_hi = bool(opsel & 0b1000) and bits['d'] == 16
  return ctx.compile_vop_pcode(inst.op, srcs, lane, vdst_reg, exec_mask, opsel_dst_hi=opsel_dst_hi, clmp=getattr(inst, 'clmp', 0))

def _compile_vop3sd(inst: ir3.VOP3SD | ir4.VOP3SD | irc.VOP3SD, ctx: _Ctx) -> UOp:
  exec_mask = ctx.rexec()
  bits, pcode, ops = inst.canonical_op_bits, get_pcode(inst.op), inst.canonical_operands

  # Read operands dynamically from instruction encoding
  vdst_reg, sdst_off = ctx.inst_field(type(inst).vdst), ctx.inst_field(type(inst).sdst)
  src0_off, src1_off, src2_off = ctx.inst_field(type(inst).src0), ctx.inst_field(type(inst).src1), ctx.inst_field(type(inst).src2)
  literal = ctx.inst_field(type(inst).literal) if hasattr(type(inst), 'literal') else None  # type: ignore[union-attr]

  has_carry_in = 's2' in ops and ops['s2'][2] == OpType.OPR_SREG
  vcc_in_off = src2_off if has_carry_in else sdst_off

  def load_srcs(lane_uop):
    ret = {'VCC': ctx.rmask_dyn(vcc_in_off), 'EXEC': exec_mask, 'SCC': ctx.rsgpr_dyn(_c(SCC.offset)), 'laneId': lane_uop}
    ret['S0'] = ctx.rsrc_dyn(src0_off, lane_uop, bits['s0'], literal, ops['s0'][0] == Fmt.FMT_NUM_F64)
    ret['S1'] = ctx.rsrc_dyn(src1_off, lane_uop, bits['s1'], literal, ops['s1'][0] == Fmt.FMT_NUM_F64)
    if 's2' in ops: ret['S2'] = ctx.rsrc_dyn(src2_off, lane_uop, bits['s2'], literal, ops['s2'][0] == Fmt.FMT_NUM_F64)
    return ret

  lane = ctx.range()
  srcs = load_srcs(lane)
  _, assigns = parse_pcode(pcode, srcs)

  has_per_lane_vcc = any('[laneId]' in dest for dest, _ in assigns if dest.startswith('VCC') or dest.startswith('D0.u64'))
  clmp = getattr(inst, 'clmp', 0)
  if has_per_lane_vcc:
    # VCC computation: RANGE+REDUCE gets axis ID first (lower ID = runs first)
    # This ensures VCC reads source values BEFORE VGPR stores modify them
    def get_vcc_bit(lane_uop) -> UOp:
      vcc_bit = _c(0)
      for dest, val in parse_pcode(pcode, load_srcs(lane_uop))[1]:
        if dest.startswith('VCC') or (dest.startswith('D0.u64') and '[laneId]' in dest): vcc_bit = val.cast(dtypes.uint64)
      return vcc_bit
    final_vcc = ctx.unroll_lanes(get_vcc_bit, exec_mask)
    # VGPR stores: RANGE gets axis ID second (higher ID = runs after VCC loop)
    lane3 = ctx.range()
    d0_val, vcc_per_lane = None, None
    for dest, val in parse_pcode(pcode, load_srcs(lane3))[1]:
      if dest.startswith('D0') and '[laneId]' not in dest: d0_val = val
      if dest.startswith('VCC') or (dest.startswith('D0.u64') and '[laneId]' in dest): vcc_per_lane = val
    vgpr_stores = []
    if d0_val is not None:
      # Apply clamp using carry/borrow bit: ADD overflow->0xFFFFFFFF, SUB underflow->0
      if clmp and vcc_per_lane is not None:
        is_sub = 'SUB' in inst.op.name
        sat_val = _c(0) if is_sub else _c(0xFFFFFFFF)
        d0_val = vcc_per_lane.cast(dtypes.bool).where(sat_val, d0_val.cast(dtypes.uint32))
      if d0_val.dtype in (dtypes.uint64, dtypes.int64, dtypes.float64):
        lo, hi = _split64(d0_val)
        vgpr_stores.extend([ctx.wvgpr_dyn(vdst_reg, lane3, lo, exec_mask), ctx.wvgpr_dyn(vdst_reg + _c(1), lane3, hi, exec_mask)])
      else:
        d0_u32 = d0_val.bitcast(dtypes.uint32) if d0_val.dtype in (dtypes.float32, dtypes.half) else d0_val.cast(dtypes.uint32)
        vgpr_stores.append(ctx.wvgpr_dyn(vdst_reg, lane3, d0_u32, exec_mask))
    # Write carry output (wsgpr_dyn handles NULL register 124)
    return UOp.sink(*ctx.wmask_dyn(sdst_off, final_vcc), UOp.group(*vgpr_stores).end(lane3), *ctx.inc_pc())
  else:
    return ctx.compile_vop_pcode(inst.op, srcs, lane, vdst_reg, exec_mask, sdst_reg=inst.sdst.offset)

def _compile_wmma(inst: ir3.VOP3P | ir4.VOP3P | irc.VOP3P, ctx: _Ctx) -> UOp:
  op_name = _op_name(inst)
  exec_mask = ctx.rexec()
  vdst_reg = ctx.inst_field(type(inst).vdst)
  src0_r = ctx.inst_field(type(inst).src0) - _c(256)
  src1_r = ctx.inst_field(type(inst).src1) - _c(256)
  src2_r = ctx.inst_field(type(inst).src2) - _c(256)
  is_f16_output = 'F16_16X16X16_F16' in op_name or 'BF16_16X16X16_BF16' in op_name  # F16/BF16 output vs F32 output
  is_bf16 = 'BF16' in op_name
  cvt = _FUNCS['bf16_to_f32'] if is_bf16 else _FUNCS['f16_to_f32']
  is_rdna4 = isinstance(inst, ir4.VOP3P)
  # read 16x16 F16/BF16 matrix from VGPRs → flat f32 array[row*16+k]
  def read_f16_val(src, lane, vgpr, half):
    v = ctx.rvgpr_dyn(src + _c(vgpr), UOp.const(dtypes.int, lane))
    return cvt((v >> UOp.const(dtypes.uint32, 16)) if half else (v & UOp.const(dtypes.uint32, 0xFFFF)))

  # RDNA3: 16 lanes × 8 VGPRs × 2 halves, k maps linearly
  # RDNA4: 32 lanes × 4 VGPRs × 2 halves, k bits are scrambled (k[2] goes to lane bit 4)
  def read_f16_mat(src):
  # (row, k) → (lane, vgpr, half)
    def ab_map(i, k):
      elem, lane = ((k & 3) | ((k >> 1) & 4), i + ((k >> 2) & 1) * 16) if is_rdna4 else (k, i)
      return lane, elem // 2, elem % 2
    return [read_f16_val(src, *ab_map(row, k)) for row in range(16) for k in range(16)]
  mat_a, mat_b = read_f16_mat(src0_r), read_f16_mat(src1_r)
  # (row, col) -> (lane, vgpr)
  def d_map(m, n):
    lane_bit, vgpr = (m >> 3, m & 7) if is_rdna4 else (m & 1, m >> 1)
    return n + lane_bit * 16, vgpr
  if is_f16_output:
    # read accumulator C with f16 layout: for RDNA4, pairs of f32 vgprs pack into one f16 vgpr
    # for RDNA3, same layout as f32 but only lo 16 bits used
    mat_c = [read_f16_val(src2_r, *((lane, vgpr // 2, vgpr % 2) if is_rdna4 else (lane, vgpr, 0)))
             for m in range(16) for n in range(16) for lane, vgpr in [d_map(m, n)]]
    mat_d = [sum(mat_a[r*16+k] * mat_b[c*16+k] for k in range(16)) + mat_c[r*16+c] for r in range(16) for c in range(16)]
    def f32_to_f16_bits(v: UOp) -> UOp: return v.cast(dtypes.half).bitcast(dtypes.uint16).cast(dtypes.uint32)
    def f32_to_bf16_bits(v: UOp) -> UOp: return (v.bitcast(dtypes.uint32) >> UOp.const(dtypes.uint32, 16)) & UOp.const(dtypes.uint32, 0xFFFF)
    out_cvt = f32_to_bf16_bits if is_bf16 else f32_to_f16_bits
    if is_rdna4:  # pack 2 f16 per VGPR: adjacent m values share (lane, vgpr) since vgpr=m&7, half=m&1
      stores = [ctx.wvgpr_dyn(vdst_reg + _c(d_map(m, n)[1] // 2), UOp.const(dtypes.int, d_map(m, n)[0]),
                out_cvt(mat_d[m*16+n]) | (out_cvt(mat_d[(m+1)*16+n]) << UOp.const(dtypes.uint32, 16)), exec_mask)
                for n in range(16) for m in range(0, 16, 2)]
    else:  # (rdna3) 1 f16 per VGPR (lo half only)
      stores = [ctx.wvgpr_dyn(vdst_reg + _c(d_map(m, n)[1]), UOp.const(dtypes.int, d_map(m, n)[0]), out_cvt(mat_d[m*16+n]), exec_mask)
                for m in range(16) for n in range(16)]
  else: # f32
    mat_c = [ctx.rvgpr_dyn(src2_r + _c(d_map(m, n)[1]), UOp.const(dtypes.int, d_map(m, n)[0])).bitcast(dtypes.float32)
             for m in range(16) for n in range(16)]
    mat_d = [sum(mat_a[r*16+k] * mat_b[c*16+k] for k in range(16)) + mat_c[r*16+c] for r in range(16) for c in range(16)]
    stores = [ctx.wvgpr_dyn(vdst_reg + _c(d_map(m, n)[1]), UOp.const(dtypes.int, d_map(m, n)[0]), mat_d[m*16+n].bitcast(dtypes.uint32), exec_mask)
              for m in range(16) for n in range(16)]
  return UOp.sink(*stores, *ctx.inc_pc())

def _compile_cdna_mfma(inst: irc.VOP3P, ctx: _Ctx) -> UOp | None:
  # CDNA MFMA is executed by a Python fallback in run_asm to avoid huge UOp kernels.
  del inst, ctx
  return None

def _compile_vop3p(inst: ir3.VOP3P | ir4.VOP3P | irc.VOP3P, ctx: _Ctx) -> UOp:
  op_name = _op_name(inst)
  if 'WMMA' in op_name and ('16X16X16_F16' in op_name or '16X16X16_BF16' in op_name): return _compile_wmma(inst, ctx)
  if op_name.startswith('V_MFMA_') and (mfma := _compile_cdna_mfma(inst, ctx)) is not None: return mfma

  lane = ctx.range()
  exec_mask = ctx.rexec()
  vdst_reg = ctx.inst_field(type(inst).vdst)
  if op_name in ('V_ACCVGPR_READ', 'V_ACCVGPR_WRITE'):
    src0 = ctx.rsrc_dyn(ctx.inst_field(type(inst).src0), lane)
    return UOp.sink(UOp.group(ctx.wvgpr_dyn(vdst_reg, lane, _to_u32(src0), exec_mask)).end(lane), *ctx.inc_pc())
  is_pk_mov_b32 = 'PK_MOV_B32' in op_name
  is_pk_f32 = 'PK' in op_name and 'F32' in op_name and 'MOV' not in op_name  # CDNA packed F32 ops
  do_cast = any(x in op_name for x in ('F16', 'F32', 'BF16')) and 'IU' not in op_name and not is_pk_f32 and not is_pk_mov_b32
  src0 = ctx.rsrc_dyn(ctx.inst_field(type(inst).src0), lane, 16, do_cast=do_cast)
  src1 = ctx.rsrc_dyn(ctx.inst_field(type(inst).src1), lane, 16, do_cast=do_cast)
  src2 = ctx.rsrc_dyn(ctx.inst_field(type(inst).src2), lane, 16, do_cast=do_cast)
  opsel, opsel_hi = getattr(inst, 'opsel', 0) or 0, getattr(inst, 'opsel_hi', 3) if getattr(inst, 'opsel_hi', 3) is not None else 3
  opsel_hi2 = getattr(inst, 'opsel_hi2', 1) if getattr(inst, 'opsel_hi2', 1) is not None else 1
  neg, neg_hi = getattr(inst, 'neg', 0) or 0, getattr(inst, 'neg_hi', 0) or 0

  if is_pk_mov_b32:
    src_offs = [ctx.inst_field(type(inst).src0), ctx.inst_field(type(inst).src1)]
    def read_src_pair(src_lo: UOp, src_off: UOp) -> tuple[UOp, UOp]:
      is_vgpr = src_off >= _c(256)
      vgpr_lo = ctx.rvgpr_dyn(src_off - _c(256), lane) if lane is not None else _c(0)
      vgpr_hi = ctx.rvgpr_dyn(src_off - _c(256) + _c(1), lane) if lane is not None else _c(0)
      # Scalar 64-bit packed sources use adjacent registers, including inline constants.
      # Literal source (255) is a single 32-bit immediate and is broadcast to both halves.
      scalar_hi_valid = src_off < _c(255)
      scalar_hi = ctx.rsgpr_dyn(src_off + _c(1), scalar_hi_valid)
      lo = is_vgpr.where(vgpr_lo, src_lo)
      hi = is_vgpr.where(vgpr_hi, scalar_hi_valid.where(scalar_hi, src_lo))
      return lo, hi
    s0_lo, s0_hi = read_src_pair(src0, src_offs[0])
    s1_lo, s1_hi = read_src_pair(src1, src_offs[1])
    # Pre-resolve OPSEL (constant per instruction) to avoid dynamic bit-slice expressions in pcode.
    sel0 = s0_hi if (opsel & 1) else s0_lo
    sel1 = s1_hi if (opsel & 2) else s1_lo
    srcs = {'S0': _u64(sel0, sel0), 'S1': _u64(sel1, sel1), 'OPSEL': UOp.const(dtypes.uint32, 0)}
  elif is_pk_f32:
    # CDNA packed F32: read 32-bit sources, build 64-bit packed values using opsel.
    # For VGPRs: opsel selects between v[reg] (0) and v[reg+1] (1) for each half.
    # For scalar sources (off < 255): low/high halves come from adjacent 32-bit sources.
    # Literal source (255) is a single 32-bit immediate and is broadcast to both halves.
    src_offs = [ctx.inst_field(type(inst).src0), ctx.inst_field(type(inst).src1), ctx.inst_field(type(inst).src2)]
    def build_pk_f32(src_lo: UOp, src_off: UOp, opsel_lo: int, opsel_hi_bit: int, neg_lo: int, neg_hi_bit: int) -> UOp:
      is_vgpr = src_off >= _c(256)
      vgpr_lo = ctx.rvgpr_dyn(src_off - _c(256), lane) if lane is not None else _c(0)
      vgpr_hi = ctx.rvgpr_dyn(src_off - _c(256) + _c(1), lane) if lane is not None else _c(0)
      scalar_hi_valid = src_off < _c(255)
      sgpr_hi = ctx.rsgpr_dyn(src_off + _c(1), scalar_hi_valid)
      # Low packed lane selection (opsel): 0->src_lo, 1->src_hi.
      scalar_lo_sel = src_lo if not opsel_lo else scalar_hi_valid.where(sgpr_hi, src_lo)
      lo = is_vgpr.where(vgpr_hi if opsel_lo else vgpr_lo, scalar_lo_sel)
      # High packed lane selection (opsel_hi/opsel_hi2): same selection rule on high lane.
      scalar_hi_sel = src_lo if not opsel_hi_bit else scalar_hi_valid.where(sgpr_hi, src_lo)
      hi = is_vgpr.where(vgpr_hi if opsel_hi_bit else vgpr_lo, scalar_hi_sel)
      if neg_lo: lo = lo ^ UOp.const(dtypes.uint32, 0x80000000)
      if neg_hi_bit: hi = hi ^ UOp.const(dtypes.uint32, 0x80000000)
      return _u64(lo, hi)
    srcs = {'S0': build_pk_f32(src0, src_offs[0], opsel & 1, opsel_hi & 1, neg & 1, neg_hi & 1),
            'S1': build_pk_f32(src1, src_offs[1], opsel & 2, opsel_hi & 2, neg & 2, neg_hi & 2),
            'S2': build_pk_f32(src2, src_offs[2], opsel & 4, 1 if opsel_hi2 else 0, neg & 4, neg_hi & 4)}
  elif 'FMA_MIX' in op_name:
    combined_opsel_hi = (opsel_hi & 0x3) | ((opsel_hi2 & 0x1) << 2)
    # For FMA_MIX: neg_hi is ABS (not neg!), neg is actual negation
    def apply_abs(v, bit, opsel_hi_bit, opsel_bit):
      if not (neg_hi & bit): return v
      # Apply abs based on whether source is f32 or f16
      if not (combined_opsel_hi & opsel_hi_bit): return v & UOp.const(dtypes.uint32, 0x7FFFFFFF)  # f32 abs
      if opsel & opsel_bit: return v & UOp.const(dtypes.uint32, 0x7FFF0000)  # f16 hi abs (preserve lo)
      return v & UOp.const(dtypes.uint32, 0xFFFF7FFF)  # f16 lo abs (preserve hi)
    def apply_neg_mix(v, bit, opsel_hi_bit, opsel_bit):
      if not (neg & bit): return v
      if not (combined_opsel_hi & opsel_hi_bit): return v ^ UOp.const(dtypes.uint32, 0x80000000)  # f32 neg
      if opsel & opsel_bit: return v ^ UOp.const(dtypes.uint32, 0x80000000)  # f16 hi neg
      return v ^ UOp.const(dtypes.uint32, 0x00008000)  # f16 lo neg
    s0_mod = apply_neg_mix(apply_abs(src0, 1, 1, 1), 1, 1, 1)
    s1_mod = apply_neg_mix(apply_abs(src1, 2, 2, 2), 2, 2, 2)
    s2_mod = apply_neg_mix(apply_abs(src2, 4, 4, 4), 4, 4, 4)
    srcs = {'S@0': s0_mod, 'S@1': s1_mod, 'S@2': s2_mod,
            'OPSEL_HI': UOp.const(dtypes.uint32, combined_opsel_hi), 'OPSEL': UOp.const(dtypes.uint32, opsel)}
  else:
    def get_half_bits(val: UOp, use_hi: bool, apply_neg: bool = False) -> UOp:
      bits = ((val >> UOp.const(dtypes.uint32, 16)) if use_hi else val) & UOp.const(dtypes.uint32, 0xFFFF)
      if apply_neg: bits = bits.cast(dtypes.uint16).bitcast(dtypes.half).neg().bitcast(dtypes.uint16).cast(dtypes.uint32)
      return bits
    def build_remapped_src(src: UOp, opsel_lo_bit: int, opsel_hi_bit: int, neg_lo_bit: int, neg_hi_bit: int) -> UOp:
      lo = get_half_bits(src, bool(opsel_lo_bit), bool(neg_lo_bit))
      hi = get_half_bits(src, bool(opsel_hi_bit), bool(neg_hi_bit))
      return lo | (hi << UOp.const(dtypes.uint32, 16))
    # DOT IU instructions use NEG bits for signed/unsigned selection, not fp16 negation
    is_dot_iu = 'DOT' in op_name and 'IU' in op_name
    n0, n1, n2, nh0, nh1, nh2 = (0, 0, 0, 0, 0, 0) if is_dot_iu else (neg & 1, neg & 2, neg & 4, neg_hi & 1, neg_hi & 2, neg_hi & 4)
    srcs = {'S0': build_remapped_src(src0, opsel & 1, opsel_hi & 1, n0, nh0),
            'S1': build_remapped_src(src1, opsel & 2, opsel_hi & 2, n1, nh1),
            'S2': build_remapped_src(src2, opsel & 4, 1 if opsel_hi2 else 0, n2, nh2)}
    if is_dot_iu: srcs['NEG'] = UOp.const(dtypes.uint32, neg)
  # Some CDNA VOP3P pcodes reference these control fields directly.
  srcs.setdefault('OPSEL', UOp.const(dtypes.uint32, opsel))
  srcs.setdefault('OPSEL_HI', UOp.const(dtypes.uint32, (opsel_hi & 0x3) | ((opsel_hi2 & 0x1) << 2)))
  srcs.setdefault('NEG', UOp.const(dtypes.uint32, neg))
  srcs.setdefault('NEG_HI', UOp.const(dtypes.uint32, neg_hi))
  return ctx.compile_vop_pcode(inst.op, srcs, lane, vdst_reg, exec_mask)

def _compile_vopd(inst: ir3.VOPD | ir4.VOPD, ctx: _Ctx) -> UOp:
  exec_mask = ctx.rexec()
  # Read operands dynamically - use type(inst) to get correct field descriptors
  inst_type = type(inst)
  vdstx_reg = ctx.inst_field(inst_type.vdstx)
  # vdsty has complex encoding: actual = (raw << 1) | ((vdstx & 1) ^ 1)
  vdsty_raw = ctx.inst_field(inst_type.vdsty)
  vdsty_reg = (vdsty_raw << _c(1)) | ((vdstx_reg & _c(1)) ^ _c(1))
  srcx0_off = ctx.inst_field(inst_type.srcx0)
  srcy0_off = ctx.inst_field(inst_type.srcy0)
  vsrcx1_reg = ctx.inst_field(inst_type.vsrcx1)
  vsrcy1_reg = ctx.inst_field(inst_type.vsrcy1)
  literal = ctx.inst_field(inst_type.literal) if hasattr(inst_type, 'literal') else None

  lane = ctx.range()
  srcy0, srcy1 = ctx.rsrc_dyn(srcy0_off, lane, literal=literal), ctx.rvgpr_dyn(vsrcy1_reg, lane)
  all_stores = []
  for op, src0_off, vsrc1_reg, vdst_reg, label in [(inst.opx, srcx0_off, vsrcx1_reg, vdstx_reg, 'X'),
                                                    (inst.opy, srcy0_off, vsrcy1_reg, vdsty_reg, 'Y')]:
    vop = VOPD_TO_VOP2.get(op)
    assert vop is not None, f"no VOP mapping for VOPD {label}: {op}"
    if label == 'Y': srcs = {'S0': srcy0, 'S1': srcy1, 'D0': ctx.rvgpr_dyn(vdst_reg, lane)}
    else: srcs = {'S0': ctx.rsrc_dyn(src0_off, lane, literal=literal), 'S1': ctx.rvgpr_dyn(vsrc1_reg, lane), 'D0': ctx.rvgpr_dyn(vdst_reg, lane)}
    # VOP2_FMAAK/FMAMK_(DTYPE)_E32
    if vop in (ir3.VOP2Op.V_FMAAK_F32_E32, ir3.VOP2Op.V_FMAMK_F32_E32, ir3.VOP2Op.V_FMAAK_F32_E32, ir3.VOP2Op.V_FMAMK_F32_E32):
      assert literal is not None
      srcs['SIMM32'] = literal
    if op in (ir3.VOPDOp.V_DUAL_CNDMASK_B32, ir4.VOPDOp.V_DUAL_CNDMASK_B32): srcs['VCC'] = ctx.rvcc()
    pcode = get_pcode(vop)
    srcs.update({'VCC': ctx.rvcc(), 'EXEC': exec_mask, 'SCC': ctx.rsgpr_dyn(_c(SCC.offset)), 'laneId': lane})
    for dest, val in parse_pcode(pcode, srcs)[1]:
      if dest.startswith('D0'): all_stores.append(ctx.wvgpr_dyn(vdst_reg, lane, _val_to_u32(val), exec_mask, after=srcy1))
  return UOp.sink(UOp.group(*all_stores).end(lane), *ctx.inc_pc())

def _compile_mem_op(inst: ir3.DS|ir3.FLAT|ir3.GLOBAL|ir3.SCRATCH|ir4.DS|ir4.VFLAT|ir4.VGLOBAL|ir4.VSCRATCH
                    |irc.DS|irc.FLAT|irc.GLOBAL|irc.SCRATCH, ctx: _Ctx) -> UOp:
  """Unified memory operation compiler for DS, FLAT, GLOBAL, SCRATCH."""
  exec_mask, op_name = ctx.rexec(), _op_name(inst)
  pcode = get_pcode(inst.op)
  # CDNA pcode uses CalcGlobalAddr/CalcDsAddr to compute address from raw components, but make_addr already handles this.
  # Strip the addr computation line and use pre-computed ADDR directly (rename 'addr' -> 'ADDR' in remaining pcode).
  if isinstance(inst, (irc.GLOBAL, irc.FLAT, irc.SCRATCH, irc.DS)) and 'Calc' in pcode and 'Addr' in pcode:
    pcode = re.sub(r'addr\s*=\s*Calc\w+Addr\([^)]*\)\s*;?\n?', '', pcode).replace('MEM[addr', 'MEM[ADDR')

  is_lds = isinstance(inst, (ir3.DS, ir4.DS, irc.DS))
  is_scratch = isinstance(inst, (ir3.SCRATCH, ir4.VSCRATCH, irc.SCRATCH))
  mem = ctx.lds if is_lds else ctx.scratch if is_scratch else ctx.vmem
  addr_shift = UOp.const(dtypes.uint32 if is_lds else dtypes.uint64, 2)

  # Extract register info - all dynamic for deduplication
  if is_lds:
    addr_reg = ctx.inst_field(type(inst).addr)  # type: ignore[union-attr]
    vdata_reg = ctx.inst_field(type(inst).data0)  # type: ignore[union-attr]
    vdst_reg = ctx.inst_field(type(inst).vdst)
    offset0 = ctx.inst_field(type(inst).offset0)  # type: ignore[union-attr]
    offset1 = ctx.inst_field(type(inst).offset1)  # type: ignore[union-attr]
    offset = (offset1 << _c(8)) | offset0  # DS offset is 16-bit: (offset1 << 8) | offset0
    saddr_reg = None
  elif isinstance(inst, (ir4.VGLOBAL, ir4.VSCRATCH, ir4.VFLAT)):  # RDNA4: vaddr, vsrc, ioffset
    addr_reg = ctx.inst_field(type(inst).vaddr)
    vdata_reg = ctx.inst_field(type(inst).vsrc)
    vdst_reg = ctx.inst_field(type(inst).vdst)
    offset = ctx.inst_field_signed(type(inst).ioffset)
    offset0, offset1 = _c(0), _c(0)
    saddr_reg = ctx.inst_field(type(inst).saddr) if hasattr(type(inst), 'saddr') else None
  else:  # RDNA3: addr, data, offset
    addr_reg = ctx.inst_field(type(inst).addr)  # type: ignore[union-attr]
    vdata_reg = ctx.inst_field(type(inst).data)  # type: ignore[union-attr]
    vdst_reg = ctx.inst_field(type(inst).vdst)
    offset = ctx.inst_field_signed(type(inst).offset)  # type: ignore[union-attr]
    offset0, offset1 = _c(0), _c(0)
    saddr_reg = ctx.inst_field(type(inst).saddr) if hasattr(type(inst), 'saddr') else None  # type: ignore[union-attr]

  # Data width from canonical_op_bits (32/64/96/128), default to 32 for untyped ops
  data_bits_mem = inst.canonical_op_bits.get('data', 32)
  is_atomic, glc = 'ATOMIC' in op_name, getattr(inst, 'glc', 0)
  has_data1 = is_lds and hasattr(inst, 'data1') and inst.data1 is not None
  data1_reg = ctx.inst_field(type(inst).data1) if is_lds else _c(0)  # type: ignore[union-attr]

  # DS_PERMUTE/DS_BPERMUTE: cross-lane VGPR access via pcode
  if is_lds and 'PERMUTE' in op_name:
    pcode = get_pcode(inst.op)
    srcs = {'ADDR': addr_reg, 'DATA0': vdata_reg, 'VDST': vdst_reg, 'OFFSET': offset,
            'EXEC': exec_mask.cast(dtypes.uint64), '_vgpr': ctx.vgpr}
    _, assigns = parse_pcode(pcode, srcs)
    stores = [ctx.vgpr.index(val[0].cast(dtypes.int)).store(val[1].cast(dtypes.uint32)) for dest, val in assigns if dest.startswith('VGPR[')]
    return UOp.sink(*stores, *ctx.inc_pc())

  def make_addr(lane: UOp) -> UOp:
    if is_lds: return ctx.rvgpr_dyn(addr_reg, lane)
    offset64 = offset.cast(dtypes.uint64)
    # Dynamic saddr check: saddr < 124 means valid SGPR, otherwise use VGPR pair for address
    use_saddr = (saddr_reg < _c(124)) if saddr_reg is not None else UOp.const(dtypes.bool, False)
    if is_scratch:
      scratch_stride = ctx.rsgpr_dyn(_c(SCRATCH_STRIDE_IDX)).cast(dtypes.uint64)
      base = ctx.rsgpr_dyn(_c(SCRATCH_BASE_IDX)).cast(dtypes.uint64) + lane.cast(dtypes.uint64) * scratch_stride
      # SVE (Scratch VGPR Enable): when SVE=1, VADDR is used as offset; when SVE=0, VADDR is ignored
      sve = getattr(inst, 'sve', 0)
      vaddr = ctx.rvgpr_dyn(addr_reg, lane).cast(dtypes.uint64)
      addr_offset = vaddr if sve == 1 else UOp.const(dtypes.uint64, 0)
      # Add saddr value only if use_saddr is true (saddr < 124)
      saddr_contrib = use_saddr.where(ctx.rsgpr_dyn(saddr_reg).cast(dtypes.uint64), UOp.const(dtypes.uint64, 0)) \
        if saddr_reg is not None else UOp.const(dtypes.uint64, 0)
      return base + addr_offset + saddr_contrib + offset64
    # FLAT/GLOBAL: when saddr is valid, addr is a 32-bit offset; otherwise addr is a 64-bit VGPR pair.
    saddr_base = _u64(ctx.rsgpr_dyn(saddr_reg), ctx.rsgpr_dyn(saddr_reg + _c(1))) if saddr_reg is not None else UOp.const(dtypes.uint64, 0)
    vaddr_base = _u64(ctx.rvgpr_dyn(addr_reg, lane), ctx.rvgpr_dyn(addr_reg + _c(1), lane))
    base_addr = use_saddr.where(saddr_base + ctx.rvgpr_dyn(addr_reg, lane).cast(dtypes.uint64), vaddr_base)
    return base_addr + offset64

  def wmem(addr: UOp, val: UOp, active: UOp) -> UOp:
    safe_addr = active.where(addr, UOp.const(addr.dtype, 0))
    idx = mem.index((safe_addr >> addr_shift).cast(dtypes.int))
    return idx.store(active.where(val, idx.load()))

  access_bytes = max(data_bits_mem // 8, 1)
  def _addr_in_valid_range(addr: UOp) -> UOp:
    # Range-gating GPU memory accesses is useful for OOB debugging, but can over-prune
    # legitimate accesses when virtual mappings are fragmented or stale.
    if is_lds or is_scratch or not ENFORCE_MEM_RANGES or not _VALID_MEM_RANGES: return UOp.const(dtypes.bool, True)
    last = addr + UOp.const(dtypes.uint64, access_bytes - 1)
    valid = UOp.const(dtypes.bool, False)
    for st, sz in _VALID_MEM_RANGES:
      start = UOp.const(dtypes.uint64, st)
      end = UOp.const(dtypes.uint64, st + sz)
      valid = valid | ((addr >= start) & (last < end))
    return valid

  def make_srcs(lane: UOp) -> dict:
    addr = make_addr(lane)
    if is_lds:
      if data_bits_mem == 128:
        data = {'DATA': ctx.rvgpr_dyn(vdata_reg, lane), 'DATA1': ctx.rvgpr_dyn(vdata_reg + _c(1), lane),
                'DATA2': ctx.rvgpr_dyn(vdata_reg + _c(2), lane), 'DATA3': ctx.rvgpr_dyn(vdata_reg + _c(3), lane)}
      elif data_bits_mem == 96:
        data = {'DATA': ctx.rvgpr_dyn(vdata_reg, lane), 'DATA1': ctx.rvgpr_dyn(vdata_reg + _c(1), lane),
                'DATA2': ctx.rvgpr_dyn(vdata_reg + _c(2), lane)}
      elif data_bits_mem == 32:
        data = {'DATA': ctx.rvgpr_dyn(vdata_reg, lane), 'DATA2': ctx.rvgpr_dyn(data1_reg, lane) if has_data1 else UOp.const(dtypes.uint32, 0)}
      else:
        data = {'DATA': _u64(ctx.rvgpr_dyn(vdata_reg, lane), ctx.rvgpr_dyn(vdata_reg + _c(1), lane)),
                'DATA2': _u64(ctx.rvgpr_dyn(data1_reg, lane), ctx.rvgpr_dyn(data1_reg + _c(1), lane)) if has_data1 else UOp.const(dtypes.uint64, 0)}
      # RDNA3 uses ADDR/OFFSET, RDNA4 uses vgpr_a/offset (lowercase) + CalcDsAddr function
      return {'ADDR': addr, 'ADDR_BASE': addr, 'OFFSET': offset, 'OFFSET0': offset0, 'OFFSET1': offset1, '_lds': mem, 'laneId': lane,
              'vgpr_a': ctx.rvgpr_dyn(addr_reg, lane), 'offset': offset, **data}
    active = _lane_active(exec_mask, lane) & _addr_in_valid_range(addr)
    # saddr < 124 means valid SGPR pair, otherwise use 0 (NULL means no saddr contribution)
    use_saddr = (saddr_reg < _c(124)) if saddr_reg is not None else UOp.const(dtypes.bool, False)
    saddr_raw = _u64(ctx.rsgpr_dyn(saddr_reg), ctx.rsgpr_dyn(saddr_reg + _c(1))) if saddr_reg is not None else UOp.const(dtypes.uint64, 0)
    saddr_base = use_saddr.where(saddr_raw, UOp.const(dtypes.uint64, 0))
    # Sign-extend offset to 64-bit for the final address calculation
    ioffset64 = offset.cast(dtypes.int64).cast(dtypes.uint64)
    # v_addr for CalcGlobalAddr: when saddr is valid use 32-bit vaddr offset; otherwise use full 64-bit address.
    vaddr_full = _u64(ctx.rvgpr_dyn(addr_reg, lane), ctx.rvgpr_dyn(addr_reg + _c(1), lane))
    vaddr_lo = ctx.rvgpr_dyn(addr_reg, lane).cast(dtypes.uint64)
    vaddr_base = use_saddr.where(vaddr_lo + ioffset64, vaddr_full + ioffset64)
    addr_safe = active.where(addr, UOp.const(dtypes.uint64, 0))
    if is_atomic:
      atomic_data = _u64(ctx.rvgpr_dyn(vdata_reg, lane), ctx.rvgpr_dyn(vdata_reg + _c(1), lane)) \
        if data_bits_mem == 64 else ctx.rvgpr_dyn(vdata_reg, lane)
      return {'ADDR': addr_safe, 'DATA': atomic_data, '_vmem': mem, '_active': active,
              'laneId': lane, 'v_addr': vaddr_base, 's_saddr': saddr_base}
    vdata = ctx.rvgpr_dyn(vdata_reg, lane).cast(dtypes.uint64) if 'STORE' in op_name \
      else ctx.rvgpr_dyn(vdst_reg, lane) if 'D16' in op_name else UOp.const(dtypes.uint32, 0)
    if 'STORE' in op_name and data_bits_mem >= 64:
      vdata = vdata | (ctx.rvgpr_dyn(vdata_reg + _c(1), lane).cast(dtypes.uint64) << UOp.const(dtypes.uint64, 32))
    srcs = {'ADDR': addr_safe, 'laneId': lane, 'VDATA': vdata, 'v_addr': vaddr_base, 'v_addr_off': addr_safe, '_vmem': mem,
            's_saddr': saddr_base, 'SADDR': saddr_base,'s_saddr_off': UOp.const(dtypes.uint64, 0), '_active': active, 'OFFSET': offset}
    for i in range(data_bits_mem // 32):
      srcs[f'VDATA{i}'] = ctx.rvgpr_dyn(vdata_reg + _c(i), lane) if 'STORE' in op_name else UOp.const(dtypes.uint32, 0)
    return srcs

  def make_stores(dest: str, val: UOp, lane: UOp, active: UOp, writes_return_data: bool) -> list[UOp]:
    # Parse bit width from dest format: MEM[...].b32 or RETURN_DATA[63:32].b64
    parts = dest.rsplit('.', 1)
    data_bits = int(parts[1][1:]) if len(parts) == 2 else 32
    if dest.startswith('MEM['):
      if is_lds or is_atomic: return _write_val(data_bits, val[1], wmem, val[0], active, is_mem=True)
      if is_scratch: return _mem_store_bytes(mem, val[0], val[1], active, data_bits)
      return _mem_store(mem, val[0], val[1], active, 64, data_bits)
    if dest.startswith('RETURN_DATA') and writes_return_data:
      if (m := re.match(r'RETURN_DATA\[(\d+)\s*:\s*(\d+)\]', dest)):
        bit_width, dword_idx = int(m.group(1)) - int(m.group(2)) + 1, int(m.group(2)) // 32
        return _write_val(bit_width, val, lambda r, v, l, e: ctx.wvgpr_dyn(r, l, v, e), vdst_reg + _c(dword_idx), lane, exec_mask)
      return _write_val(data_bits, val, lambda r, v, l, e: ctx.wvgpr_dyn(r, l, v, e), vdst_reg, lane, exec_mask)
    return []

  # DS-specific: check for 2ADDR pattern needing separate ranges
  if is_lds:
    dummy_lane = ctx.range()
    _, assigns = parse_pcode(pcode, make_srcs(dummy_lane))
    mem_assigns = [d for d, _ in assigns if d.startswith('MEM[')]
    mem_addrs = set(m.group(1) if (m := re.match(r'MEM\[([^\]]+)\]', d)) else d for d in mem_assigns)
    use_separate_ranges = (len(mem_addrs) > 1 or '2ADDR' in op_name) and 'STOREXCHG' not in op_name
    if use_separate_ranges:
      ended: list[UOp] = []
      for i, (dest, _) in enumerate(assigns):
        lane = ctx.range()
        active = _lane_active(exec_mask, lane)
        _, lane_assigns = parse_pcode(pcode, make_srcs(lane))
        ended.extend(s.end(lane) for s in make_stores(dest, lane_assigns[i][1], lane, active, True))
      return UOp.sink(*ended, *ctx.inc_pc())

  # Standard path: single lane range
  writes_return_data = '_RTN' in op_name or (is_lds and (op_name.startswith('DS_LOAD') or op_name.startswith('DS_READ'))) or bool(is_atomic and glc)
  lane = ctx.range()
  active = _lane_active(exec_mask, lane)
  pcode_vars, assigns = parse_pcode(pcode, make_srcs(lane))
  stores = [s for dest, val in assigns for s in make_stores(dest, val, lane, active, writes_return_data)]

  # FLAT/GLOBAL/SCRATCH: collect VDATA slices for loads
  if not is_lds and not is_atomic:
    for dword_idx, val in sorted(_collect_data_slices(assigns, 'VDATA', pcode_vars, op_name).items()):
      stores.append(ctx.wvgpr_dyn(vdst_reg + _c(dword_idx), lane, val, exec_mask))

  return UOp.sink(UOp.group(*stores).end(lane), *ctx.inc_pc())

# Dispatch table: instruction type -> handler function
_INST_HANDLERS: dict[type, Callable[..., UOp]] = {
  ir3.SOPP: _compile_sopp, ir3.SMEM: _compile_smem, ir3.SOP1: _compile_sop, ir3.SOP2: _compile_sop, ir3.SOPC: _compile_sop, ir3.SOPK: _compile_sop,
  ir3.VOP1: _compile_vop12, ir3.VOP1_SDST: _compile_vop12, ir3.VOP2: _compile_vop12, ir3.VOPC: _compile_vopc, ir3.VOP3: _compile_vop3,
  ir3.VOP3_SDST: _compile_vop3, ir3.VOP3SD: _compile_vop3sd, ir3.VOP3P: _compile_vop3p, ir3.VOPD: _compile_vopd,
  ir3.DS: _compile_mem_op, ir3.FLAT: _compile_mem_op, ir3.GLOBAL: _compile_mem_op, ir3.SCRATCH: _compile_mem_op,
  # RDNA4 instruction classes
  ir4.SOPP: _compile_sopp, ir4.SMEM: _compile_smem, ir4.SOP1: _compile_sop, ir4.SOP2: _compile_sop, ir4.SOPC: _compile_sop, ir4.SOPK: _compile_sop,
  ir4.VOP1: _compile_vop12, ir4.VOP1_SDST: _compile_vop12, ir4.VOP2: _compile_vop12, ir4.VOPC: _compile_vopc, ir4.VOP3: _compile_vop3,
  ir4.VOP3_SDST: _compile_vop3, ir4.VOP3SD: _compile_vop3sd, ir4.VOP3P: _compile_vop3p, ir4.VOPD: _compile_vopd,
  ir4.DS: _compile_mem_op, ir4.VFLAT: _compile_mem_op, ir4.VGLOBAL: _compile_mem_op, ir4.VSCRATCH: _compile_mem_op,
  # CDNA instruction classes
  irc.SOPP: _compile_sopp, irc.SMEM: _compile_smem, irc.SOP1: _compile_sop, irc.SOP2: _compile_sop, irc.SOPC: _compile_sop, irc.SOPK: _compile_sop,
  irc.VOP1: _compile_vop12, irc.VOP2: _compile_vop12, irc.VOPC: _compile_vopc, irc.VOP3: _compile_vop3,
  irc.VOP3_SDST: _compile_vop3, irc.VOP3SD: _compile_vop3sd, irc.VOP3P: _compile_vop3p,
  irc.DS: _compile_mem_op, irc.FLAT: _compile_mem_op, irc.GLOBAL: _compile_mem_op, irc.SCRATCH: _compile_mem_op,
}

# ═══════════════════════════════════════════════════════════════════════════════
# PROGRAM DECODE AND COMPILATION
# ═══════════════════════════════════════════════════════════════════════════════

_canonical_runner_cache: list[tuple[int, int, int, object]] = []  # [(base, mask, size, runner), ...]
_VALID_MEM_RANGES: tuple[tuple[int, int], ...] = ()
_CUR_LIB_BASE: int = 0
_SCRATCH_BASE_PTR: int = 0
_SCRATCH_TOTAL_BYTES: int = 0

@functools.cache
def _get_runner(inst_bytes: bytes, arch: str = "rdna3"):
  """Build and compile instruction to CompiledRunner. Cached by instruction bytes, with canonical dedup."""
  inst = decode_inst(inst_bytes, arch)
  inst_size = inst.size()
  inst_int = int.from_bytes(inst_bytes[:inst_size], 'little')
  use_canonical_dedup = arch != "cdna"

  # Check if instruction matches any cached canonical pattern
  if use_canonical_dedup:
    for base, mask, size, runner in _canonical_runner_cache:
      if inst_size == size and (inst_int & mask) == base: return runner

  # Look up handler by type, falling back to base classes for _LIT variants
  handler = _INST_HANDLERS.get(type(inst))
  if handler is None:
    for cls in type(inst).__mro__:
      if cls in _INST_HANDLERS:
        handler = _INST_HANDLERS[cls]
        break
  if handler is None: raise RuntimeError(f"[emu] unimplemented instruction type: {type(inst).__name__} {_op_name(inst)}")

  ctx = _Ctx(inst_size)
  sink = handler(inst, ctx)
  if use_canonical_dedup:
    base, mask, size = ctx.canonical_mask(inst_bytes)
    canonical_name = f"{_op_name(inst).lower()}_{base.to_bytes(size, 'little').hex()}"
  else:
    base, mask, size = inst_int, (1 << (inst_size * 8)) - 1, inst_size
    canonical_name = f"{_op_name(inst).lower()}_{inst_bytes[:inst_size].hex()}"
  sink = sink.replace(arg=KernelInfo(name=canonical_name)).rtag(1)

  # NOTE: renderer output is not reproducible because of _MXCSRContext. PROFILE=0 prevents emulator instruction runners from polluting profiling.
  check_oob = int(getenv("MOCKGPU_CDNA_CHECK_OOB", 1 if arch == "cdna" else 0))
  with Context(NOOPT=1, CHECK_OOB=check_oob, TUPLE_ORDER=0, EMULATED_DTYPES="", CAPTURE_PROCESS_REPLAY=0, PROFILE=0):
    runner = get_runner('CPU', sink)
  if use_canonical_dedup: _canonical_runner_cache.append((base, mask, size, runner))
  return runner

_BARRIER_OPS = {ir3.SOPPOp.S_BARRIER, irc.SOPPOp.S_BARRIER}
if hasattr(ir4.SOPPOp, 'S_BARRIER_WAIT'): _BARRIER_OPS.add(ir4.SOPPOp.S_BARRIER_WAIT)
_BRANCH_OPS: set[int] = {op.value for op in (ir3.SOPPOp.S_BRANCH, ir3.SOPPOp.S_CBRANCH_SCC0, ir3.SOPPOp.S_CBRANCH_SCC1,
  ir3.SOPPOp.S_CBRANCH_VCCZ, ir3.SOPPOp.S_CBRANCH_VCCNZ, ir3.SOPPOp.S_CBRANCH_EXECZ, ir3.SOPPOp.S_CBRANCH_EXECNZ)}

def _decode_at(pc: int, arch: str):
  """Decode and compile instruction at absolute address pc. Returns (runner, decoded_inst)."""
  inst_bytes = bytes((ctypes.c_char * 16).from_address(pc).raw)
  inst = decode_inst(inst_bytes, arch)
  try: return _get_runner(bytes(inst_bytes[:inst.size() + 4]), arch), inst
  except Exception as e:
    try: inst_str = repr(inst)
    except Exception: inst_str = f"<{type(inst).__name__}>"
    raise RuntimeError(f"[emu] Failed to compile {inst_str}: {type(e).__name__}: {e}") from e

# ═══════════════════════════════════════════════════════════════════════════════
# WAVE STATE
# ═══════════════════════════════════════════════════════════════════════════════

# Inline float constants (as bit patterns) for GPU instructions
F32_INLINE = {240: 0x3f000000, 241: 0xbf000000, 242: 0x3f800000, 243: 0xbf800000,  # 0.5, -0.5, 1.0, -1.0
              244: 0x40000000, 245: 0xc0000000, 246: 0x40800000, 247: 0xc0800000, 248: 0x3e22f983}  # 2.0, -2.0, 4.0, -4.0, 1/(2*pi)

class WaveState:
  __slots__ = ('vgpr_buf', 'sgpr_buf', 'accvgpr_buf', '_vgpr_mv', '_sgpr_mv', '_accvgpr_mv', 'n_lanes')

  def __init__(self, n_lanes: int = WAVE_SIZE):
    self.n_lanes = n_lanes
    self.vgpr_buf = Buffer('CPU', VGPR_SIZE, dtypes.uint32).ensure_allocated()
    self.sgpr_buf = Buffer('CPU', SGPR_COUNT, dtypes.uint32).ensure_allocated()
    self.accvgpr_buf = Buffer('CPU', VGPR_SIZE, dtypes.uint32).ensure_allocated()
    self._vgpr_mv = self.vgpr_buf.as_memoryview(force_zero_copy=True).cast('I')
    self._sgpr_mv = self.sgpr_buf.as_memoryview(force_zero_copy=True).cast('I')
    self._accvgpr_mv = self.accvgpr_buf.as_memoryview(force_zero_copy=True).cast('I')
    # Zero memory using ctypes memset (much faster than Python loops)
    ctypes.memset(self.vgpr_buf._buf.va_addr, 0, VGPR_SIZE * 4)
    ctypes.memset(self.sgpr_buf._buf.va_addr, 0, SGPR_COUNT * 4)
    ctypes.memset(self.accvgpr_buf._buf.va_addr, 0, VGPR_SIZE * 4)
    # Pre-populate inline constants at indices 128-255
    for i in range(65): self._write_sgpr(128 + i, i)  # 128-192: integers 0-64
    for i in range(16): self._write_sgpr(193 + i, (-(i + 1)) & MASK32)  # 193-208: -1 to -16
    for off, val in F32_INLINE.items(): self._write_sgpr(off, val)  # 240-248: float constants
    exec_mask = (1 << n_lanes) - 1 if n_lanes < 64 else (1 << 64) - 1
    self._write_sgpr(EXEC_LO.offset, exec_mask & MASK32)
    if WAVE_SIZE == 64: self._write_sgpr(EXEC_LO.offset + 1, (exec_mask >> 32) & MASK32)
    self._write_sgpr(PC_LO_IDX, 0)
    self._write_sgpr(PC_HI_IDX, 0)

  def _write_sgpr(self, idx: int, val: int): self._sgpr_mv[idx] = val & MASK32
  def _read_sgpr(self, idx: int) -> int: return self._sgpr_mv[idx]
  def _write_vgpr(self, reg: int, lane: int, val: int): self._vgpr_mv[reg * WAVE_SIZE + lane] = val & MASK32
  def _read_vgpr(self, reg: int, lane: int) -> int: return self._vgpr_mv[reg * WAVE_SIZE + lane]
  def _write_accvgpr(self, reg: int, lane: int, val: int): self._accvgpr_mv[reg * WAVE_SIZE + lane] = val & MASK32
  def _read_accvgpr(self, reg: int, lane: int) -> int: return self._accvgpr_mv[reg * WAVE_SIZE + lane]

  @property
  def pc(self) -> int: return self._read_sgpr(PC_LO_IDX) | (self._read_sgpr(PC_HI_IDX) << 32)
  @pc.setter
  def pc(self, val: int):
    self._write_sgpr(PC_LO_IDX, val & MASK32)
    self._write_sgpr(PC_HI_IDX, (val >> 32) & MASK32)

_CDNA_MFMA_RE = re.compile(r'^V_MFMA_F32_16X16X(\d+)_(F16|BF16)$')

def _read_src32_python(st: WaveState, src_off: int, lane: int) -> int:
  if src_off >= 256: return st._read_vgpr(src_off - 256, lane)
  return st._read_sgpr(src_off)

def _exec_cdna_accvgpr_python(st: WaveState, inst) -> bool:
  if getenv("MOCKGPU_DISABLE_CDNA_ACCVGPR_FALLBACK", 0): return False
  opn = _op_name(inst)
  if opn not in ('V_ACCVGPR_READ', 'V_ACCVGPR_WRITE'): return False
  if any(not hasattr(getattr(inst, x), 'offset') for x in ('src0', 'vdst')): return False
  src0_off, vdst_off = inst.src0.offset, inst.vdst.offset
  src_off, dst_off = src0_off, vdst_off
  if getenv("MOCKGPU_TRACE_ACC", 0):
    print(f"[emu-acc] {opn} src0={src0_off} vdst={vdst_off} inst={inst!r}", flush=True)
  if dst_off < 256: return False
  dst_reg = dst_off - 256
  exec_mask = st._read_sgpr(EXEC_LO.offset) | (st._read_sgpr(EXEC_LO.offset + 1) << 32)
  if getenv("MOCKGPU_ACC_IGNORE_EXEC", 0): exec_mask = (1 << WAVE_SIZE) - 1
  for lane in range(WAVE_SIZE):
    if ((exec_mask >> lane) & 1) == 0: continue
    acc_lane = lane & 31 if getenv("MOCKGPU_ACC_LANE32", 0) else lane
    if opn == 'V_ACCVGPR_WRITE':
      st._write_accvgpr(dst_reg, acc_lane, _read_src32_python(st, src_off, lane))
    else:
      if src_off < 256: continue
      st._write_vgpr(dst_reg, lane, st._read_accvgpr(src_off - 256, acc_lane))
  st.pc = st.pc + inst.size()
  return True

def _exec_cdna_mfma_python(st: WaveState, inst) -> bool:
  m = _CDNA_MFMA_RE.match(_op_name(inst))
  if m is None or WAVE_SIZE != 64: return False
  if any(not hasattr(getattr(inst, x), 'offset') for x in ('src0', 'src1', 'src2', 'vdst')): return False
  src0_off, src1_off, src2_off, vdst_off = inst.src0.offset, inst.src1.offset, inst.src2.offset, inst.vdst.offset
  if min(src0_off, src1_off, src2_off, vdst_off) < 256: return False

  op_info = CDNA_OPERANDS.get(inst.op, {})
  src0_bits, src1_bits, src2_bits = op_info.get('src0', (None, 0, None))[1], op_info.get('src1', (None, 0, None))[1], op_info.get('src2', (None, 0, None))[1]
  if src0_bits % 16 != 0 or src1_bits % 16 != 0 or src2_bits % 32 != 0: return False
  num_a, num_b, num_c = src0_bits // 16, src1_bits // 16, src2_bits // 32
  k_dim, is_bf16 = int(m.group(1)), m.group(2) == 'BF16'
  if num_a != num_b or num_c != 4 or k_dim != num_a * 4: return False

  import numpy as np
  lanes, src_words = 64, src0_bits // 32
  src0_base, src1_base, src2_base, vdst_base = src0_off - 256, src1_off - 256, src2_off - 256, vdst_off - 256

  def load_ab_frag(base_reg: int):
    words = np.empty((lanes, src_words), dtype=np.uint32)
    for lane in range(lanes):
      for wi in range(src_words): words[lane, wi] = st._read_vgpr(base_reg + wi, lane)
    elems = np.empty((lanes, src_words * 2), dtype=np.uint16)
    elems[:, 0::2], elems[:, 1::2] = (words & 0xFFFF).astype(np.uint16), ((words >> 16) & 0xFFFF).astype(np.uint16)
    if is_bf16: return (elems.astype(np.uint32) << 16).view(np.float32)
    return elems.view(np.float16).astype(np.float32)

  a_frag, b_frag = load_ab_frag(src0_base), load_ab_frag(src1_base)  # [lane, elem]
  a_mat, b_mat = np.empty((16, k_dim), dtype=np.float32), np.empty((16, k_dim), dtype=np.float32)
  for k in range(k_dim):
    grp, idx = (k // num_a) * 16, k % num_a
    a_mat[:, k], b_mat[:, k] = a_frag[grp:grp+16, idx], b_frag[grp:grp+16, idx]

  c_words = np.empty((lanes, num_c), dtype=np.uint32)
  for lane in range(lanes):
    for e in range(num_c): c_words[lane, e] = st._read_accvgpr(src2_base + e, lane)
  c_frag, c_mat = c_words.view(np.float32), np.zeros((16, 16), dtype=np.float32)
  for lane in range(lanes):
    col, row_base = lane & 15, (lane >> 4) * 4
    for e in range(num_c): c_mat[row_base + e, col] = c_frag[lane, e]

  d_mat = a_mat @ b_mat.T + c_mat
  exec_mask = st._read_sgpr(EXEC_LO.offset) | (st._read_sgpr(EXEC_LO.offset + 1) << 32)
  for lane in range(lanes):
    if ((exec_mask >> lane) & 1) == 0: continue
    col, row_base = lane & 15, (lane >> 4) * 4
    for e in range(num_c):
      st._write_accvgpr(vdst_base + e, lane, int(np.float32(d_mat[row_base + e, col]).view(np.uint32)))

  st.pc = st.pc + inst.size()
  return True

def _addr_in_valid_ranges(addr: int, nbytes: int) -> bool:
  for st_addr, sz in _VALID_MEM_RANGES:
    if st_addr <= addr and addr + nbytes <= st_addr + sz: return True
  return not _VALID_MEM_RANGES

def _exec_cdna_scratch_load_store_python(st: WaveState, inst) -> bool:
  if not isinstance(inst, irc.SCRATCH): return False
  if not hasattr(inst, 'op'): return False
  opn = inst.op.name
  if not (opn.startswith('SCRATCH_LOAD_') or opn.startswith('SCRATCH_STORE_')): return False
  is_store = opn.startswith('SCRATCH_STORE_')
  req = ('addr', 'data', 'saddr') if is_store else ('addr', 'vdst', 'saddr')
  if any(not hasattr(getattr(inst, x), 'offset') for x in req): return False

  nbytes, words, signed = 0, 1, False
  if opn in ('SCRATCH_LOAD_UBYTE', 'SCRATCH_LOAD_SBYTE', 'SCRATCH_LOAD_UBYTE_D16', 'SCRATCH_LOAD_UBYTE_D16_HI',
             'SCRATCH_LOAD_SBYTE_D16', 'SCRATCH_LOAD_SBYTE_D16_HI', 'SCRATCH_STORE_BYTE', 'SCRATCH_STORE_BYTE_D16_HI'):
    nbytes, signed = 1, ('_SBYTE' in opn)
  elif opn in ('SCRATCH_LOAD_USHORT', 'SCRATCH_LOAD_SSHORT', 'SCRATCH_LOAD_SHORT_D16', 'SCRATCH_LOAD_SHORT_D16_HI',
               'SCRATCH_STORE_SHORT', 'SCRATCH_STORE_SHORT_D16_HI'):
    nbytes, signed = 2, ('SCRATCH_LOAD_SSHORT' in opn)
  elif opn in ('SCRATCH_LOAD_DWORD', 'SCRATCH_STORE_DWORD'):
    nbytes = 4
  elif opn in ('SCRATCH_LOAD_DWORDX2', 'SCRATCH_STORE_DWORDX2'):
    nbytes, words = 8, 2
  elif opn in ('SCRATCH_LOAD_DWORDX3', 'SCRATCH_STORE_DWORDX3'):
    nbytes, words = 12, 3
  elif opn in ('SCRATCH_LOAD_DWORDX4', 'SCRATCH_STORE_DWORDX4'):
    nbytes, words = 16, 4
  else:
    return False

  addr_off = inst.addr.offset
  data_off = inst.data.offset if is_store else inst.vdst.offset
  saddr_off = inst.saddr.offset
  if min(addr_off, data_off) < 256: return False
  addr_reg, data_reg = addr_off - 256, data_off - 256
  acc_mode = int(getattr(inst, 'acc', 0) or 0)
  use_saddr = saddr_off < 124
  saddr_base = (st._read_sgpr(saddr_off) | (st._read_sgpr(saddr_off + 1) << 32)) if use_saddr else 0
  sve = int(getattr(inst, 'sve', 0) or 0)
  off_field = type(inst).offset
  off_bits = off_field.hi - off_field.lo + 1
  off_raw = int(inst.offset) & ((1 << off_bits) - 1)
  ioff = off_raw - (1 << off_bits) if (off_raw & (1 << (off_bits - 1))) else off_raw
  d16_hi = opn.endswith('D16_HI')
  exec_mask = st._read_sgpr(EXEC_LO.offset) | (st._read_sgpr(EXEC_LO.offset + 1) << 32)
  trace = int(getenv("MOCKGPU_TRACE_SCRATCH", 0))
  trace_lane = int(getenv("MOCKGPU_TRACE_LANE", 0))
  lane_private_bytes = st._read_sgpr(SCRATCH_STRIDE_IDX)
  wave_base = st._read_sgpr(SCRATCH_BASE_IDX)
  lane_dword_stride = WAVE_SIZE * 4

  def _read_data(reg: int, lane_id: int) -> int:
    return st._read_accvgpr(reg, lane_id) if acc_mode else st._read_vgpr(reg, lane_id)
  def _write_data(reg: int, lane_id: int, val: int) -> None:
    if acc_mode: st._write_accvgpr(reg, lane_id, val)
    else: st._write_vgpr(reg, lane_id, val)

  def _scratch_phys_off(lane_id: int, private_byte_off: int) -> int:
    # CDNA scratch/private memory uses lane-interleaved dword layout:
    #   wave_base + (dword_index * wave_size * 4) + lane*4 + byte_in_dword
    dword_idx, byte_in_dword = private_byte_off >> 2, private_byte_off & 0x3
    return int(wave_base + dword_idx * lane_dword_stride + lane_id * 4 + byte_in_dword)

  for lane in range(WAVE_SIZE):
    if ((exec_mask >> lane) & 1) == 0: continue
    vaddr = st._read_vgpr(addr_reg, lane) if sve == 1 else 0
    private_off = int(vaddr + saddr_base + ioff)
    if private_off < 0 or private_off + nbytes > lane_private_bytes:
      if not is_store:
        for wi in range(words): _write_data(data_reg + wi, lane, 0)
      continue

    if is_store:
      if trace and lane == trace_lane:
        v0 = _read_data(data_reg, lane)
        print(f"[emu-scratch] lane={lane} pc=0x{st.pc:x} off=0x{st.pc - _CUR_LIB_BASE:x} store {opn} poff={private_off} nbytes={nbytes} "
              f"v0=0x{v0:x} acc={acc_mode} inst={inst!r}", flush=True)
      data_bytes = bytearray(nbytes)
      if nbytes == 1:
        data = _read_data(data_reg, lane)
        data_bytes[0] = ((data >> 8) if d16_hi else data) & 0xFF
      elif nbytes == 2:
        data = _read_data(data_reg, lane)
        val16 = ((data >> 16) if d16_hi else data) & 0xFFFF
        data_bytes[0], data_bytes[1] = val16 & 0xFF, (val16 >> 8) & 0xFF
      else:
        for wi in range(words):
          wval = _read_data(data_reg + wi, lane)
          bi = wi * 4
          data_bytes[bi + 0], data_bytes[bi + 1] = wval & 0xFF, (wval >> 8) & 0xFF
          data_bytes[bi + 2], data_bytes[bi + 3] = (wval >> 16) & 0xFF, (wval >> 24) & 0xFF
      for bi, b in enumerate(data_bytes):
        phys = _scratch_phys_off(lane, private_off + bi)
        if phys < 0 or phys >= _SCRATCH_TOTAL_BYTES: continue
        ctypes.c_uint8.from_address(_SCRATCH_BASE_PTR + phys).value = b
    else:
      raw_bytes = bytearray(nbytes)
      in_range = True
      for bi in range(nbytes):
        phys = _scratch_phys_off(lane, private_off + bi)
        if phys < 0 or phys >= _SCRATCH_TOTAL_BYTES:
          in_range = False
          break
        raw_bytes[bi] = ctypes.c_uint8.from_address(_SCRATCH_BASE_PTR + phys).value
      if not in_range:
        for wi in range(words): _write_data(data_reg + wi, lane, 0)
        continue
      if trace and lane == trace_lane:
        raw0 = (raw_bytes[0] | (raw_bytes[1] << 8) | (raw_bytes[2] << 16) | (raw_bytes[3] << 24)) if nbytes >= 4 else raw_bytes[0]
        print(f"[emu-scratch] lane={lane} pc=0x{st.pc:x} off=0x{st.pc - _CUR_LIB_BASE:x} load {opn} poff={private_off} nbytes={nbytes} "
              f"mem0=0x{raw0:x} acc={acc_mode} inst={inst!r}", flush=True)
      if words > 1:
        for wi in range(words):
          bi = wi * 4
          w = raw_bytes[bi] | (raw_bytes[bi + 1] << 8) | (raw_bytes[bi + 2] << 16) | (raw_bytes[bi + 3] << 24)
          _write_data(data_reg + wi, lane, w)
      elif nbytes == 1:
        raw = raw_bytes[0]
        if '_D16' in opn:
          val16 = (ctypes.c_int8(raw).value & 0xFFFF) if signed else (raw & 0xFFFF)
          _write_data(data_reg, lane, ((val16 << 16) if d16_hi else val16) & MASK32)
        else:
          _write_data(data_reg, lane, (ctypes.c_int8(raw).value if signed else raw) & MASK32)
      elif nbytes == 2:
        raw = raw_bytes[0] | (raw_bytes[1] << 8)
        if '_D16' in opn:
          _write_data(data_reg, lane, ((raw << 16) if d16_hi else raw) & MASK32)
        else:
          _write_data(data_reg, lane, (ctypes.c_int16(raw).value if signed else raw) & MASK32)
      else:
        raw = raw_bytes[0] | (raw_bytes[1] << 8) | (raw_bytes[2] << 16) | (raw_bytes[3] << 24)
        _write_data(data_reg, lane, raw)

  st.pc = st.pc + inst.size()
  return True

def _exec_cdna_global_load_python(st: WaveState, inst) -> bool:
  if not isinstance(inst, irc.GLOBAL): return False
  if not hasattr(inst, 'op') or not inst.op.name.startswith('GLOBAL_LOAD_'): return False
  opn = inst.op.name
  if opn.startswith('GLOBAL_LOAD_LDS_'): return False
  if any(not hasattr(getattr(inst, x), 'offset') for x in ('addr', 'vdst', 'saddr')): return False

  # Decode width/sign from opcode.
  load_nbytes, load_words, signed = 0, 1, False
  if opn in ('GLOBAL_LOAD_UBYTE', 'GLOBAL_LOAD_SBYTE', 'GLOBAL_LOAD_UBYTE_D16', 'GLOBAL_LOAD_UBYTE_D16_HI',
             'GLOBAL_LOAD_SBYTE_D16', 'GLOBAL_LOAD_SBYTE_D16_HI'):
    load_nbytes, signed = 1, opn.startswith('GLOBAL_LOAD_SBYTE')
  elif opn in ('GLOBAL_LOAD_USHORT', 'GLOBAL_LOAD_SSHORT', 'GLOBAL_LOAD_SHORT_D16', 'GLOBAL_LOAD_SHORT_D16_HI'):
    load_nbytes, signed = 2, opn in ('GLOBAL_LOAD_SSHORT',)
  elif opn == 'GLOBAL_LOAD_DWORD':
    load_nbytes = 4
  elif opn == 'GLOBAL_LOAD_DWORDX2':
    load_nbytes, load_words = 8, 2
  elif opn == 'GLOBAL_LOAD_DWORDX3':
    load_nbytes, load_words = 12, 3
  elif opn == 'GLOBAL_LOAD_DWORDX4':
    load_nbytes, load_words = 16, 4
  else:
    return False

  addr_off, vdst_off, saddr_off = inst.addr.offset, inst.vdst.offset, inst.saddr.offset
  if min(addr_off, vdst_off) < 256: return False
  addr_reg, vdst_reg = addr_off - 256, vdst_off - 256
  acc_mode = int(getattr(inst, 'acc', 0) or 0)

  def _write_dst(reg: int, lane_id: int, val: int) -> None:
    if acc_mode: st._write_accvgpr(reg, lane_id, val)
    else: st._write_vgpr(reg, lane_id, val)
  def _read_dst(reg: int, lane_id: int) -> int:
    return st._read_accvgpr(reg, lane_id) if acc_mode else st._read_vgpr(reg, lane_id)

  use_saddr = saddr_off < 124
  saddr_base = (st._read_sgpr(saddr_off) | (st._read_sgpr(saddr_off + 1) << 32)) if use_saddr else 0

  off_field = type(inst).offset
  off_bits = off_field.hi - off_field.lo + 1
  off_sign = 1 << (off_bits - 1)
  off_mask = (1 << off_bits) - 1
  off_raw = int(inst.offset) & off_mask
  ioff = off_raw - (1 << off_bits) if (off_raw & off_sign) else off_raw

  is_d16, load_hi = ('_D16' in opn), opn.endswith('_D16_HI')
  exec_mask = st._read_sgpr(EXEC_LO.offset) | (st._read_sgpr(EXEC_LO.offset + 1) << 32)
  trace = bool(getenv("MOCKGPU_TRACE_LOAD", 0))
  loads, invalids = 0, 0
  first = None

  for lane in range(WAVE_SIZE):
    if ((exec_mask >> lane) & 1) == 0: continue
    if use_saddr:
      vaddr = ctypes.c_int32(st._read_vgpr(addr_reg, lane)).value
      addr = (saddr_base + vaddr + ioff) & 0xFFFFFFFFFFFFFFFF
    else:
      vaddr = st._read_vgpr(addr_reg, lane) | (st._read_vgpr(addr_reg + 1, lane) << 32)
      addr = (vaddr + ioff) & 0xFFFFFFFFFFFFFFFF

    if not _addr_in_valid_ranges(addr, load_nbytes):
      invalids += 1
      for wi in range(load_words): _write_dst(vdst_reg + wi, lane, 0)
      continue

    if load_words > 1:
      for wi in range(load_words):
        _write_dst(vdst_reg + wi, lane, ctypes.c_uint32.from_address(addr + wi * 4).value)
      if first is None: first = (lane, addr, _read_dst(vdst_reg, lane))
      loads += 1
      continue

    if load_nbytes == 1:
      raw = ctypes.c_uint8.from_address(addr).value
      if is_d16:
        val16 = (ctypes.c_int8(raw).value & 0xFFFF) if signed else (raw & 0xFFFF)
        _write_dst(vdst_reg, lane, ((val16 << 16) if load_hi else val16) & MASK32)
      else:
        val = ctypes.c_int8(raw).value if signed else raw
        _write_dst(vdst_reg, lane, val & MASK32)
      if first is None: first = (lane, addr, _read_dst(vdst_reg, lane))
      loads += 1
    elif load_nbytes == 2:
      raw = ctypes.c_uint16.from_address(addr).value
      if is_d16:
        val16 = raw & 0xFFFF
        _write_dst(vdst_reg, lane, ((val16 << 16) if load_hi else val16) & MASK32)
      else:
        val = ctypes.c_int16(raw).value if signed else raw
        _write_dst(vdst_reg, lane, val & MASK32)
      if first is None: first = (lane, addr, _read_dst(vdst_reg, lane))
      loads += 1
    else:
      _write_dst(vdst_reg, lane, ctypes.c_uint32.from_address(addr).value)
      if first is None: first = (lane, addr, _read_dst(vdst_reg, lane))
      loads += 1

  if trace:
    print(f"[emu-gload] {opn} loads={loads} invalid={invalids} saddr={saddr_off} addr={addr_off} vdst={vdst_off} acc={acc_mode} off={inst.offset}", flush=True)
    if first is not None: print(f"[emu-gload] first lane={first[0]} addr=0x{first[1]:x} val=0x{int(first[2]):x}", flush=True)

  st.pc = st.pc + inst.size()
  return True

def _exec_cdna_global_store_python(st: WaveState, inst) -> bool:
  if not isinstance(inst, irc.GLOBAL): return False
  if not hasattr(inst, 'op') or not inst.op.name.startswith('GLOBAL_STORE_'): return False
  if any(not hasattr(getattr(inst, x), 'offset') for x in ('addr', 'data', 'saddr')): return False

  opn = inst.op.name
  store_nbytes = 0
  store_words = 1
  if opn in ('GLOBAL_STORE_BYTE', 'GLOBAL_STORE_BYTE_D16_HI'): store_nbytes = 1
  elif opn in ('GLOBAL_STORE_SHORT', 'GLOBAL_STORE_SHORT_D16_HI'): store_nbytes = 2
  elif opn == 'GLOBAL_STORE_DWORD': store_nbytes = 4
  elif opn == 'GLOBAL_STORE_DWORDX2': store_nbytes, store_words = 8, 2
  elif opn == 'GLOBAL_STORE_DWORDX3': store_nbytes, store_words = 12, 3
  elif opn == 'GLOBAL_STORE_DWORDX4': store_nbytes, store_words = 16, 4
  else: return False

  addr_off, data_off, saddr_off = inst.addr.offset, inst.data.offset, inst.saddr.offset
  if min(addr_off, data_off) < 256: return False
  addr_reg, data_reg = addr_off - 256, data_off - 256
  acc_mode = int(getattr(inst, 'acc', 0) or 0)

  def _read_data(reg: int, lane_id: int) -> int:
    return st._read_accvgpr(reg, lane_id) if acc_mode else st._read_vgpr(reg, lane_id)

  # GLOBAL address semantics: when saddr is valid, VADDR_LO is treated as 32-bit offset added to saddr base.
  use_saddr = saddr_off < 124
  saddr_base = (st._read_sgpr(saddr_off) | (st._read_sgpr(saddr_off + 1) << 32)) if use_saddr else 0
  # Sign-extend ioffset using the instruction field width.
  off_field = type(inst).offset
  off_bits = off_field.hi - off_field.lo + 1
  off_sign = 1 << (off_bits - 1)
  off_mask = (1 << off_bits) - 1
  off_raw = int(inst.offset) & off_mask
  ioff = off_raw - (1 << off_bits) if (off_raw & off_sign) else off_raw
  store_hi = opn.endswith('D16_HI')
  exec_mask = st._read_sgpr(EXEC_LO.offset) | (st._read_sgpr(EXEC_LO.offset + 1) << 32)
  trace = bool(getenv("MOCKGPU_TRACE_STORE", 0))

  writes = 0
  first = None
  for lane in range(WAVE_SIZE):
    if ((exec_mask >> lane) & 1) == 0: continue
    if use_saddr:
      vaddr = ctypes.c_int32(st._read_vgpr(addr_reg, lane)).value
      addr = (saddr_base + vaddr + ioff) & 0xFFFFFFFFFFFFFFFF
    else:
      vaddr = st._read_vgpr(addr_reg, lane) | (st._read_vgpr(addr_reg + 1, lane) << 32)
      addr = (vaddr + ioff) & 0xFFFFFFFFFFFFFFFF
    if not _addr_in_valid_ranges(addr, store_nbytes): continue

    if store_nbytes == 1:
      data = _read_data(data_reg, lane)
      val8 = ((data >> 8) if store_hi else data) & 0xFF
      ctypes.c_uint8.from_address(addr).value = val8
      if first is None: first = (lane, addr, val8)
    elif store_nbytes == 2:
      data = _read_data(data_reg, lane)
      val16 = ((data >> 16) if store_hi else data) & 0xFFFF
      ctypes.c_uint16.from_address(addr).value = val16
      if first is None: first = (lane, addr, val16)
    else:
      for wi in range(store_words):
        ctypes.c_uint32.from_address(addr + wi * 4).value = _read_data(data_reg + wi, lane)
      if first is None: first = (lane, addr, _read_data(data_reg, lane))
    writes += 1

  if trace:
    vdst_dbg = getattr(getattr(inst, 'vdst', None), 'offset', None)
    print(f"[emu-gstore] pc=0x{st.pc:x} off=0x{(st.pc - _CUR_LIB_BASE):x} {opn} writes={writes} exec=0x{exec_mask:x} saddr={saddr_off} addr={addr_off} data={data_off} acc={acc_mode} "
          f"vdst={vdst_dbg} off={inst.offset} inst={inst!r}", flush=True)
    if first is not None: print(f"[emu-gstore] first lane={first[0]} addr=0x{first[1]:x} val=0x{int(first[2]):x}", flush=True)
    elif writes == 0 and use_saddr:
      raw0 = st._read_vgpr(addr_reg, 0)
      raw1 = st._read_vgpr(addr_reg + 1, 0)
      raw_vdst = st._read_vgpr(vdst_dbg - 256, 0) if (vdst_dbg is not None and vdst_dbg >= 256) else None
      signed0 = ctypes.c_int32(raw0).value
      addr_u = (saddr_base + raw0 + ioff) & 0xFFFFFFFFFFFFFFFF
      addr_s = (saddr_base + signed0 + ioff) & 0xFFFFFFFFFFFFFFFF
      addr_vdst = (saddr_base + (raw_vdst or 0) + ioff) & 0xFFFFFFFFFFFFFFFF if raw_vdst is not None else 0
      addr_pair = (saddr_base + (raw0 | (raw1 << 32)) + ioff) & 0xFFFFFFFFFFFFFFFF
      print(f"[emu-gstore] lane0 raw_lo=0x{raw0:x} raw_hi=0x{raw1:x} raw_vdst={None if raw_vdst is None else hex(raw_vdst)} signed={signed0} "
            f"sbase=0x{saddr_base:x} addr_u=0x{addr_u:x} addr_s=0x{addr_s:x} addr_vdst=0x{addr_vdst:x} addr_pair=0x{addr_pair:x} "
            f"sve={getattr(inst, 'sve', None)} sc0={getattr(inst, 'sc0', None)} sc1={getattr(inst, 'sc1', None)} nt={getattr(inst, 'nt', None)}",
            flush=True)
      lane_dump = []
      for li in range(min(12, WAVE_SIZE)):
        if ((exec_mask >> li) & 1) == 0: continue
        r = st._read_vgpr(addr_reg, li)
        rv = st._read_vgpr(vdst_dbg - 256, li) if (vdst_dbg is not None and vdst_dbg >= 256) else 0
        lane_dump.append((li, r, rv))
      print(f"[emu-gstore] lanes addr_reg/vdst={[(li, hex(r), hex(rv)) for li, r, rv in lane_dump]}", flush=True)

  st.pc = st.pc + inst.size()
  return True

def _exec_cdna_flat_load_python(st: WaveState, inst) -> bool:
  if not isinstance(inst, irc.FLAT): return False
  if not hasattr(inst, 'op') or not inst.op.name.startswith('FLAT_LOAD_'): return False
  opn = inst.op.name
  if any(not hasattr(getattr(inst, x), 'offset') for x in ('addr', 'vdst', 'saddr')): return False

  load_nbytes, load_words, signed = 0, 1, False
  if opn in ('FLAT_LOAD_UBYTE', 'FLAT_LOAD_SBYTE', 'FLAT_LOAD_UBYTE_D16', 'FLAT_LOAD_UBYTE_D16_HI',
             'FLAT_LOAD_SBYTE_D16', 'FLAT_LOAD_SBYTE_D16_HI'):
    load_nbytes, signed = 1, opn.startswith('FLAT_LOAD_SBYTE')
  elif opn in ('FLAT_LOAD_USHORT', 'FLAT_LOAD_SSHORT', 'FLAT_LOAD_SHORT_D16', 'FLAT_LOAD_SHORT_D16_HI'):
    load_nbytes, signed = 2, opn in ('FLAT_LOAD_SSHORT',)
  elif opn == 'FLAT_LOAD_DWORD':
    load_nbytes = 4
  elif opn == 'FLAT_LOAD_DWORDX2':
    load_nbytes, load_words = 8, 2
  elif opn == 'FLAT_LOAD_DWORDX3':
    load_nbytes, load_words = 12, 3
  elif opn == 'FLAT_LOAD_DWORDX4':
    load_nbytes, load_words = 16, 4
  else:
    return False

  addr_off, vdst_off, saddr_off = inst.addr.offset, inst.vdst.offset, inst.saddr.offset
  if min(addr_off, vdst_off) < 256: return False
  addr_reg, vdst_reg = addr_off - 256, vdst_off - 256
  acc_mode = int(getattr(inst, 'acc', 0) or 0)

  def _write_dst(reg: int, lane_id: int, val: int) -> None:
    if acc_mode: st._write_accvgpr(reg, lane_id, val)
    else: st._write_vgpr(reg, lane_id, val)
  use_saddr = saddr_off < 124
  saddr_base = (st._read_sgpr(saddr_off) | (st._read_sgpr(saddr_off + 1) << 32)) if use_saddr else 0
  off_field = type(inst).offset
  off_bits = off_field.hi - off_field.lo + 1
  off_sign = 1 << (off_bits - 1)
  off_mask = (1 << off_bits) - 1
  off_raw = int(inst.offset) & off_mask
  ioff = off_raw - (1 << off_bits) if (off_raw & off_sign) else off_raw
  is_d16, load_hi = ('_D16' in opn), opn.endswith('_D16_HI')
  exec_mask = st._read_sgpr(EXEC_LO.offset) | (st._read_sgpr(EXEC_LO.offset + 1) << 32)

  for lane in range(WAVE_SIZE):
    if ((exec_mask >> lane) & 1) == 0: continue
    if use_saddr:
      vaddr_lo_signed = ctypes.c_int32(st._read_vgpr(addr_reg, lane)).value
      addr = (saddr_base + vaddr_lo_signed + ioff) & 0xFFFFFFFFFFFFFFFF
    else:
      vaddr = st._read_vgpr(addr_reg, lane) | (st._read_vgpr(addr_reg + 1, lane) << 32)
      addr = (vaddr + ioff) & 0xFFFFFFFFFFFFFFFF

    if not _addr_in_valid_ranges(addr, load_nbytes):
      for wi in range(load_words): _write_dst(vdst_reg + wi, lane, 0)
      continue

    if load_words > 1:
      for wi in range(load_words): _write_dst(vdst_reg + wi, lane, ctypes.c_uint32.from_address(addr + wi * 4).value)
      continue

    if load_nbytes == 1:
      raw = ctypes.c_uint8.from_address(addr).value
      if is_d16:
        val16 = (ctypes.c_int8(raw).value & 0xFFFF) if signed else (raw & 0xFFFF)
        _write_dst(vdst_reg, lane, ((val16 << 16) if load_hi else val16) & MASK32)
      else:
        _write_dst(vdst_reg, lane, (ctypes.c_int8(raw).value if signed else raw) & MASK32)
    elif load_nbytes == 2:
      raw = ctypes.c_uint16.from_address(addr).value
      if is_d16:
        val16 = raw & 0xFFFF
        _write_dst(vdst_reg, lane, ((val16 << 16) if load_hi else val16) & MASK32)
      else:
        _write_dst(vdst_reg, lane, (ctypes.c_int16(raw).value if signed else raw) & MASK32)
    else:
      _write_dst(vdst_reg, lane, ctypes.c_uint32.from_address(addr).value)

  st.pc = st.pc + inst.size()
  return True

def _exec_cdna_flat_store_python(st: WaveState, inst) -> bool:
  if not isinstance(inst, irc.FLAT): return False
  if not hasattr(inst, 'op') or not inst.op.name.startswith('FLAT_STORE_'): return False
  if any(not hasattr(getattr(inst, x), 'offset') for x in ('addr', 'data', 'saddr')): return False
  opn = inst.op.name
  store_nbytes, store_words = 0, 1
  if opn in ('FLAT_STORE_BYTE', 'FLAT_STORE_BYTE_D16_HI'): store_nbytes = 1
  elif opn in ('FLAT_STORE_SHORT', 'FLAT_STORE_SHORT_D16_HI'): store_nbytes = 2
  elif opn == 'FLAT_STORE_DWORD': store_nbytes = 4
  elif opn == 'FLAT_STORE_DWORDX2': store_nbytes, store_words = 8, 2
  elif opn == 'FLAT_STORE_DWORDX3': store_nbytes, store_words = 12, 3
  elif opn == 'FLAT_STORE_DWORDX4': store_nbytes, store_words = 16, 4
  else: return False

  addr_off, data_off, saddr_off = inst.addr.offset, inst.data.offset, inst.saddr.offset
  if min(addr_off, data_off) < 256: return False
  addr_reg, data_reg = addr_off - 256, data_off - 256
  acc_mode = int(getattr(inst, 'acc', 0) or 0)

  def _read_data(reg: int, lane_id: int) -> int:
    return st._read_accvgpr(reg, lane_id) if acc_mode else st._read_vgpr(reg, lane_id)
  use_saddr = saddr_off < 124
  saddr_base = (st._read_sgpr(saddr_off) | (st._read_sgpr(saddr_off + 1) << 32)) if use_saddr else 0
  off_field = type(inst).offset
  off_bits = off_field.hi - off_field.lo + 1
  off_sign = 1 << (off_bits - 1)
  off_mask = (1 << off_bits) - 1
  off_raw = int(inst.offset) & off_mask
  ioff = off_raw - (1 << off_bits) if (off_raw & off_sign) else off_raw
  store_hi = opn.endswith('D16_HI')
  exec_mask = st._read_sgpr(EXEC_LO.offset) | (st._read_sgpr(EXEC_LO.offset + 1) << 32)

  for lane in range(WAVE_SIZE):
    if ((exec_mask >> lane) & 1) == 0: continue
    if use_saddr:
      vaddr_lo_signed = ctypes.c_int32(st._read_vgpr(addr_reg, lane)).value
      addr = (saddr_base + vaddr_lo_signed + ioff) & 0xFFFFFFFFFFFFFFFF
    else:
      vaddr = st._read_vgpr(addr_reg, lane) | (st._read_vgpr(addr_reg + 1, lane) << 32)
      addr = (vaddr + ioff) & 0xFFFFFFFFFFFFFFFF
    if not _addr_in_valid_ranges(addr, store_nbytes): continue

    if store_nbytes == 1:
      data = _read_data(data_reg, lane)
      ctypes.c_uint8.from_address(addr).value = ((data >> 8) if store_hi else data) & 0xFF
    elif store_nbytes == 2:
      data = _read_data(data_reg, lane)
      ctypes.c_uint16.from_address(addr).value = ((data >> 16) if store_hi else data) & 0xFFFF
    else:
      for wi in range(store_words):
        ctypes.c_uint32.from_address(addr + wi * 4).value = _read_data(data_reg + wi, lane)

  st.pc = st.pc + inst.size()
  return True

def _exec_cdna_pk_f32_python(st: WaveState, inst) -> bool:
  # Prefer canonical pcode execution for packed FP32 ops; the Python fallback is
  # kept as an opt-in debugging path.
  if getenv("MOCKGPU_DISABLE_PK_FALLBACK", 0) or not getenv("MOCKGPU_ENABLE_PK_FALLBACK", 0): return False
  opn = _op_name(inst)
  base_op = opn[:-4] if opn.endswith('_E64') else opn
  if base_op not in ('V_PK_ADD_F32', 'V_PK_MUL_F32', 'V_PK_FMA_F32', 'V_PK_MOV_B32'): return False
  if getenv("MOCKGPU_TRACE_PK", 0):
    tc = getattr(_exec_cdna_pk_f32_python, "_trace_cnt", 0)
    off = st.pc - _CUR_LIB_BASE
    if tc < 64 or 0x5800 <= off <= 0x5d00:
      s0 = getattr(getattr(inst, 'src0', None), 'offset', None)
      s1 = getattr(getattr(inst, 'src1', None), 'offset', None)
      s2 = getattr(getattr(inst, 'src2', None), 'offset', None)
      vd = getattr(getattr(inst, 'vdst', None), 'offset', None)
      print(f"[emu-pk] off=0x{off:x} {base_op} s0={s0} s1={s1} s2={s2} vd={vd} opsel={getattr(inst, 'opsel', None)} opsel_hi={getattr(inst, 'opsel_hi', None)} "
            f"opsel_hi2={getattr(inst, 'opsel_hi2', None)} neg={getattr(inst, 'neg', None)} neg_hi={getattr(inst, 'neg_hi', None)} inst={inst!r}",
            flush=True)
    setattr(_exec_cdna_pk_f32_python, "_trace_cnt", tc + 1)
  need_src2 = (base_op == 'V_PK_FMA_F32')
  req_fields = ('src0', 'src1', 'vdst', 'src2') if need_src2 else ('src0', 'src1', 'vdst')
  if any(not hasattr(getattr(inst, x), 'offset') for x in req_fields): return False
  src0_off, src1_off, vdst_off = inst.src0.offset, inst.src1.offset, inst.vdst.offset
  src2_off = inst.src2.offset if need_src2 else 0
  if vdst_off < 256: return False
  vdst_reg = vdst_off - 256
  exec_mask = st._read_sgpr(EXEC_LO.offset) | (st._read_sgpr(EXEC_LO.offset + 1) << 32)
  opsel = int(getattr(inst, 'opsel', 0) or 0)
  opsel_hi = int(getattr(inst, 'opsel_hi', 0) or 0)
  opsel_hi2 = int(getattr(inst, 'opsel_hi2', 0) or 0)
  neg = int(getattr(inst, 'neg', 0) or 0)
  neg_hi = int(getattr(inst, 'neg_hi', 0) or 0)

  def read_src32(src_off: int, lane: int, sel_hi: int) -> int:
    if src_off >= 256: return st._read_vgpr(src_off - 256 + (1 if sel_hi else 0), lane)
    if src_off < 255:
      return st._read_sgpr(src_off + (1 if sel_hi else 0))
    return st._read_sgpr(src_off)

  for lane in range(WAVE_SIZE):
    if ((exec_mask >> lane) & 1) == 0: continue
    if base_op == 'V_PK_MOV_B32':
      st._write_vgpr(vdst_reg, lane, read_src32(src0_off, lane, opsel & 1))
      st._write_vgpr(vdst_reg + 1, lane, read_src32(src1_off, lane, (opsel >> 1) & 1))
      continue
    s0_lo = read_src32(src0_off, lane, opsel & 1)
    s1_lo = read_src32(src1_off, lane, (opsel >> 1) & 1)
    s0_hi = read_src32(src0_off, lane, opsel_hi & 1)
    s1_hi = read_src32(src1_off, lane, (opsel_hi >> 1) & 1)
    s2_lo = read_src32(src2_off, lane, (opsel >> 2) & 1) if need_src2 else 0
    s2_hi = read_src32(src2_off, lane, 1 if opsel_hi2 else 0) if need_src2 else 0
    if neg & 1: s0_lo ^= 0x80000000
    if neg & 2: s1_lo ^= 0x80000000
    if need_src2 and (neg & 4): s2_lo ^= 0x80000000
    if neg_hi & 1: s0_hi ^= 0x80000000
    if neg_hi & 2: s1_hi ^= 0x80000000
    if need_src2 and (neg_hi & 4): s2_hi ^= 0x80000000

    f0_lo, f1_lo = struct.unpack('<f', struct.pack('<I', s0_lo))[0], struct.unpack('<f', struct.pack('<I', s1_lo))[0]
    f0_hi, f1_hi = struct.unpack('<f', struct.pack('<I', s0_hi))[0], struct.unpack('<f', struct.pack('<I', s1_hi))[0]
    if base_op == 'V_PK_ADD_F32':
      if getenv("MOCKGPU_PK_ADD_INT", 0):
        lo = struct.unpack('<f', struct.pack('<I', (s0_lo + s1_lo) & 0xFFFFFFFF))[0]
        hi = struct.unpack('<f', struct.pack('<I', (s0_hi + s1_hi) & 0xFFFFFFFF))[0]
      else:
        lo, hi = f0_lo + f1_lo, f0_hi + f1_hi
    elif base_op == 'V_PK_MUL_F32':
      lo, hi = f0_lo * f1_lo, f0_hi * f1_hi
    else:
      import numpy as np
      f2_lo = struct.unpack('<f', struct.pack('<I', s2_lo))[0]
      f2_hi = struct.unpack('<f', struct.pack('<I', s2_hi))[0]
      lo = float(np.float32(np.float32(f0_lo) * np.float32(f1_lo) + np.float32(f2_lo)))
      hi = float(np.float32(np.float32(f0_hi) * np.float32(f1_hi) + np.float32(f2_hi)))
    st._write_vgpr(vdst_reg, lane, struct.unpack('<I', struct.pack('<f', lo))[0])
    st._write_vgpr(vdst_reg + 1, lane, struct.unpack('<I', struct.pack('<f', hi))[0])

  st.pc = st.pc + inst.size()
  return True

# ═══════════════════════════════════════════════════════════════════════════════
# EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

def _init_wave(lib: int, wave_start: int, total_threads: int, lx: int, ly: int, lz: int, args_ptr: int, rsrc2: int,
               scratch_size: int, wave_scratch_base: int, arch: str, gidx: int, gidy: int, gidz: int, user_data: list[int]|None) -> WaveState:
  """Initialize a single wavefront and return WaveState."""
  n_lanes = min(WAVE_SIZE, total_threads - wave_start)
  st = WaveState(n_lanes)
  st.pc = lib
  if user_data:
    for i, val in enumerate(user_data): st._write_sgpr(i, val)
  else:
    st._write_sgpr(0, args_ptr & MASK32)
    st._write_sgpr(1, (args_ptr >> 32) & MASK32)
  sgpr_idx = (rsrc2 & hsa.AMD_COMPUTE_PGM_RSRC_TWO_USER_SGPR_COUNT) >> hsa.AMD_COMPUTE_PGM_RSRC_TWO_USER_SGPR_COUNT_SHIFT
  for enabled, gid in [(hsa.AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_X, gidx),
                       (hsa.AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Y, gidy),
                       (hsa.AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Z, gidz)]:
    if rsrc2 & enabled:
      st._write_sgpr(sgpr_idx, gid)
      sgpr_idx += 1
  if arch == "rdna4":
    st._write_sgpr(ttmp[7].offset, (gidy & 0xFFFF) | ((gidz & 0xFFFF) << 16))
    st._write_sgpr(ttmp[9].offset, gidx)
  for lane in range(n_lanes):
    tid = wave_start + lane
    st._write_vgpr(0, lane, ((tid // (lx * ly)) << 20) | (((tid // lx) % ly) << 10) | (tid % lx))
  st._write_sgpr(SCRATCH_STRIDE_IDX, scratch_size)
  st._write_sgpr(SCRATCH_BASE_IDX, wave_scratch_base)
  return st

def run_asm(lib: int, lib_sz: int, gx: int, gy: int, gz: int, lx: int, ly: int, lz: int, args_ptr: int, rsrc2: int = 0x19c,
            scratch_size: int = 0, arch: str = "rdna3", user_data: list[int]|None = None,
            valid_mem_ranges: set[tuple[int, int]]|None = None) -> int:
  """Execute AMD assembly program. scratch_size is private_segment_fixed_size from kernel descriptor (per-lane)."""
  from tinygrad.renderer.amd.dsl import Inst
  global _VALID_MEM_RANGES, _CUR_LIB_BASE, _SCRATCH_BASE_PTR, _SCRATCH_TOTAL_BYTES
  _CUR_LIB_BASE = lib
  new_ranges = tuple(sorted(valid_mem_ranges)) if valid_mem_ranges is not None else tuple()
  if new_ranges != _VALID_MEM_RANGES:
    _VALID_MEM_RANGES = new_ranges
    _canonical_runner_cache.clear()
    _get_runner.cache_clear()
  program: dict[int, tuple[Callable, list[int], bool, Inst]] = {}  # pc -> (fxn, globals, is_barrier, inst)
  lds_size = ((rsrc2 & hsa.AMD_COMPUTE_PGM_RSRC_TWO_GRANULATED_LDS_SIZE) >> hsa.AMD_COMPUTE_PGM_RSRC_TWO_GRANULATED_LDS_SIZE_SHIFT) * 512
  total_threads = lx * ly * lz

  # Use Buffer objects with external_ptr=0 for vmem
  vmem_buf = Buffer('CPU', 1 << 40, dtypes.uint32, options=BufferSpec(external_ptr=0)).ensure_allocated()
  lds_buf = Buffer('CPU', max(lds_size // 4, 1), dtypes.uint32).ensure_allocated()
  num_waves = (total_threads + WAVE_SIZE - 1) // WAVE_SIZE
  scratch_wave_bytes = scratch_size * WAVE_SIZE
  scratch_total_bytes = scratch_wave_bytes * num_waves
  scratch_buf = Buffer('CPU', scratch_total_bytes, dtypes.uint8).ensure_allocated() if scratch_size else None
  if scratch_buf is not None and scratch_total_bytes > 0:
    ctypes.memset(scratch_buf._buf.va_addr, 0, scratch_total_bytes)
  _SCRATCH_BASE_PTR = scratch_buf._buf.va_addr if scratch_buf is not None else 0
  _SCRATCH_TOTAL_BYTES = scratch_total_bytes if scratch_buf is not None else 0

  # Initialize SQTT encoder — emits packets inline as instructions execute (only when profiling)
  if PROFILE:
    sqtt_emit, sqtt_finish, sqtt_finalize = _init_sqtt_encoder()
  trace_reg = int(getenv("MOCKGPU_TRACE_REG", -1))
  trace_lane = int(getenv("MOCKGPU_TRACE_LANE", 0))
  trace_off_lo = int(getenv("MOCKGPU_TRACE_OFF_LO", 0))
  trace_off_hi = int(getenv("MOCKGPU_TRACE_OFF_HI", 0x7FFFFFFF))

  def _trace_reg_change(st: WaveState, pc0: int, inst, before: int):
    if trace_reg < 0: return
    if trace_lane < 0 or trace_lane >= WAVE_SIZE: return
    after = st._read_vgpr(trace_reg, trace_lane)
    off0 = pc0 - lib
    if after != before and trace_off_lo <= off0 <= trace_off_hi:
      opn = getattr(getattr(inst, 'op', None), 'name', type(inst).__name__)
      print(f"[emu-reg] lane={trace_lane} off=0x{off0:x} reg=v[{trace_reg}] 0x{before:x}->0x{after:x} op={opn} inst={inst!r}", flush=True)

  def _ensure_compiled(pc: int) -> tuple[Callable, list[int], bool, Inst]:
    if pc not in program:
      prev_len = len(_canonical_runner_cache)
      runner, inst = _decode_at(pc, arch)
      is_barrier = isinstance(inst, (ir3.SOPP, ir4.SOPP, irc.SOPP)) and inst.op in _BARRIER_OPS
      program[pc] = (runner._prg.fxn, runner.p.globals, is_barrier, inst)
      if DEBUG >= 3:
        msg = f"[emu] PC={pc - lib}: {inst!r}"
        print(colored(msg, 'green') if len(_canonical_runner_cache) > prev_len else msg)
    return program[pc]

  disable_cdna_mem_fallback = bool(getenv("MOCKGPU_DISABLE_CDNA_MEM_FALLBACK", 0))

  def _trace_div_state(st: WaveState, inst) -> None:
    if not getenv("MOCKGPU_TRACE_DIV", 0): return
    opn = getattr(getattr(inst, 'op', None), 'name', '')
    if 'DIV_SCALE_F32' not in opn and 'DIV_FMAS_F32' not in opn and 'DIV_FIXUP_F32' not in opn: return
    vcc = st._read_sgpr(VCC_LO.offset) | (st._read_sgpr(VCC_LO.offset + 1) << 32)
    s2 = st._read_sgpr(2) | (st._read_sgpr(3) << 32)
    print(f"[emu-div] {opn} vcc=0x{vcc:x} s2=0x{s2:x}", flush=True)

  # Set DAZ+FTZ during emulator execution, restore afterward to avoid breaking hypothesis tests
  # Only trace the first workgroup (like real HW traces one CU/SIMD), subsequent workgroups run but don't add to trace
  tracing = bool(PROFILE)
  with _MXCSRContext():
    for gidz in range(gz):
      for gidy in range(gy):
        for gidx in range(gx):
          # Initialize all wavefronts for this workgroup
          waves: list[tuple[WaveState, list]] = []
          for wave_start in range(0, total_threads, WAVE_SIZE):
            wave_id = wave_start // WAVE_SIZE
            st = _init_wave(lib, wave_start, total_threads, lx, ly, lz, args_ptr, rsrc2, scratch_size,
                            wave_id * scratch_wave_bytes, arch, gidx, gidy, gidz, user_data)
            c_bufs = [ctypes.c_uint64(st.sgpr_buf._buf.va_addr), ctypes.c_uint64(st.vgpr_buf._buf.va_addr),
                      ctypes.c_uint64(vmem_buf._buf.va_addr), ctypes.c_uint64(lds_buf._buf.va_addr),
                      ctypes.c_uint64(scratch_buf._buf.va_addr if scratch_buf else 0)]
            waves.append((st, c_bufs))

          # Execute wavefronts with barrier synchronization
          # Each wave runs until it hits s_barrier or s_endpgm. When all waves have stopped, release barrier waves.
          done = [False] * len(waves)
          for total_inst in range(10_000_000):
            if all(done): break
            for wi, (st, c_bufs) in enumerate(waves):
              if done[wi]: continue
              # Run this wave until barrier or endpgm
              for _ in range(1_000_000):
                pc = st.pc
                reg_before = st._read_vgpr(trace_reg, trace_lane) if (trace_reg >= 0 and 0 <= trace_lane < WAVE_SIZE) else 0
                if pc == ENDPGM_PC:
                  done[wi] = True
                  if tracing: sqtt_finish(wi)
                  break
                inst = decode_inst(bytes((ctypes.c_char * 16).from_address(pc).raw), arch)
                if getenv("MOCKGPU_TRACE_INST", 0): print(f"[emu-pre] pc={pc-lib:06x} op={inst.op.name if hasattr(inst, 'op') else type(inst).__name__}", flush=True)
                if hasattr(inst, 'op') and 'GLOBAL_STORE' in inst.op.name and getenv("MOCKGPU_TRACE_INST", 0):
                  print(f"[emu-inst] {inst!r}", flush=True)
                if _exec_cdna_accvgpr_python(st, inst):
                  _trace_reg_change(st, pc, inst, reg_before)
                  _trace_div_state(st, inst)
                  if tracing:
                    inst_op = inst.op.value if hasattr(inst, 'op') else 0
                    sqtt_emit(wi, inst, (st.pc != ENDPGM_PC and st.pc != pc + inst.size()) if inst_op in _BRANCH_OPS else None)
                  continue
                if not disable_cdna_mem_fallback and _exec_cdna_scratch_load_store_python(st, inst):
                  _trace_reg_change(st, pc, inst, reg_before)
                  _trace_div_state(st, inst)
                  if tracing:
                    inst_op = inst.op.value if hasattr(inst, 'op') else 0
                    sqtt_emit(wi, inst, (st.pc != ENDPGM_PC and st.pc != pc + inst.size()) if inst_op in _BRANCH_OPS else None)
                  continue
                if not disable_cdna_mem_fallback and _exec_cdna_flat_load_python(st, inst):
                  _trace_reg_change(st, pc, inst, reg_before)
                  _trace_div_state(st, inst)
                  if tracing:
                    inst_op = inst.op.value if hasattr(inst, 'op') else 0
                    sqtt_emit(wi, inst, (st.pc != ENDPGM_PC and st.pc != pc + inst.size()) if inst_op in _BRANCH_OPS else None)
                  continue
                if not disable_cdna_mem_fallback and _exec_cdna_flat_store_python(st, inst):
                  _trace_reg_change(st, pc, inst, reg_before)
                  _trace_div_state(st, inst)
                  if tracing:
                    inst_op = inst.op.value if hasattr(inst, 'op') else 0
                    sqtt_emit(wi, inst, (st.pc != ENDPGM_PC and st.pc != pc + inst.size()) if inst_op in _BRANCH_OPS else None)
                  continue
                if not disable_cdna_mem_fallback and _exec_cdna_global_load_python(st, inst):
                  _trace_reg_change(st, pc, inst, reg_before)
                  _trace_div_state(st, inst)
                  if tracing:
                    inst_op = inst.op.value if hasattr(inst, 'op') else 0
                    sqtt_emit(wi, inst, (st.pc != ENDPGM_PC and st.pc != pc + inst.size()) if inst_op in _BRANCH_OPS else None)
                  continue
                if not disable_cdna_mem_fallback and _exec_cdna_global_store_python(st, inst):
                  _trace_reg_change(st, pc, inst, reg_before)
                  _trace_div_state(st, inst)
                  if tracing:
                    inst_op = inst.op.value if hasattr(inst, 'op') else 0
                    sqtt_emit(wi, inst, (st.pc != ENDPGM_PC and st.pc != pc + inst.size()) if inst_op in _BRANCH_OPS else None)
                  continue
                if _exec_cdna_pk_f32_python(st, inst):
                  _trace_reg_change(st, pc, inst, reg_before)
                  _trace_div_state(st, inst)
                  if tracing:
                    inst_op = inst.op.value if hasattr(inst, 'op') else 0
                    sqtt_emit(wi, inst, (st.pc != ENDPGM_PC and st.pc != pc + inst.size()) if inst_op in _BRANCH_OPS else None)
                  continue
                if _exec_cdna_mfma_python(st, inst):
                  _trace_reg_change(st, pc, inst, reg_before)
                  _trace_div_state(st, inst)
                  if tracing:
                    inst_op = inst.op.value if hasattr(inst, 'op') else 0
                    sqtt_emit(wi, inst, (st.pc != ENDPGM_PC and st.pc != pc + inst.size()) if inst_op in _BRANCH_OPS else None)
                  continue
                fxn, globals_list, is_barrier, inst = _ensure_compiled(pc)
                if getenv("MOCKGPU_TRACE_INST", 0): print(f"[emu-exec] pc={pc-lib:06x}", flush=True)
                fxn(*[c_bufs[g] for g in globals_list])
                _trace_reg_change(st, pc, inst, reg_before)
                _trace_div_state(st, inst)
                if tracing:
                  inst_op = inst.op.value if hasattr(inst, 'op') else 0
                  sqtt_emit(wi, inst, (st.pc != ENDPGM_PC and st.pc != pc + inst.size()) if inst_op in _BRANCH_OPS else None)
                if is_barrier: break  # s_barrier hit: PC already advanced past it, pause this wave
              else: raise RuntimeError("exceeded 1M instructions in single wave, likely infinite loop")
            # All waves have either hit barrier or endpgm — release barrier waves for next round
          else: raise RuntimeError("exceeded 10M total scheduling rounds")
          tracing = False  # only trace the first workgroup

          # Reset LDS for next workgroup
          if lds_size > 0: ctypes.memset(lds_buf._buf.va_addr, 0, max(lds_size, 4))

  if PROFILE: sqtt_traces.append(sqtt_finalize())
  return 0
