"""CDNA VOP3 instruction coverage.

Exercises generated CDNA pcode end-to-end in the emulator and compares against
gfx950 hardware when USE_HW=1.
"""
import ctypes, unittest
import tinygrad.runtime.autogen.amd.cdna.ins as cdna
from tinygrad.renderer.amd.dsl import NULL
from test.amd.hw.helpers import USE_HW, assemble
from test.mockgpu.amd.emu import run_asm

LANES = 1

def _insts(instructions: list, out_reg: int = 2, out_addr: int | None = None) -> list:
  load_out_addr = [
    cdna.s_mov_b32(cdna.s[92], out_addr & 0xffffffff),
    cdna.s_mov_b32(cdna.s[93], out_addr >> 32),
  ] if out_addr is not None else [
    cdna.s_load_dwordx2(cdna.s[92:93], cdna.s[80:81], 0, soffset=NULL),
    cdna.s_waitcnt(0),
  ]
  return [
    cdna.s_mov_b32(cdna.s[80], cdna.s[0]),
    cdna.s_mov_b32(cdna.s[81], cdna.s[1]),
    cdna.v_mov_b32_e32(cdna.v[255], cdna.v[0]),
    *instructions,
    *load_out_addr,
    cdna.v_lshlrev_b32_e32(cdna.v[240], 2, cdna.v[255]),
    cdna.v_mov_b32_e32(cdna.v[241], 0),  # the store address is v[240:241]
    cdna.global_store_dword(addr=cdna.v[240], data=cdna.v[out_reg], saddr=cdna.s[92:93], offset=0),
    cdna.s_endpgm(),
  ]

def _code(instructions: list, out_reg: int = 2, out_addr: int | None = None) -> bytes:
  return assemble(_insts(instructions, out_reg, out_addr))

def _run_emu(instructions: list, out_reg: int = 2) -> int:
  out_buf = (ctypes.c_uint32 * LANES)(*([0] * LANES))
  args = (ctypes.c_uint64 * 1)(ctypes.addressof(out_buf))
  code = _code(instructions, out_reg)
  kernel_buf = (ctypes.c_char * len(code)).from_buffer_copy(code)
  result = run_asm(ctypes.addressof(kernel_buf), len(code), 1, 1, 1, LANES, 1, 1, ctypes.addressof(args),
                   0x19c | (128 << 15), 0x10000, arch="cdna")
  assert result == 0, f"run_asm failed with {result}"
  return out_buf[0]

def _run_hw(instructions: list, out_reg: int = 2) -> int:
  from tinygrad.device import Device, Buffer
  from tinygrad.dtype import dtypes
  from tinygrad.uop.ops import UOp, Ops, KernelInfo
  from tinygrad.engine.realize import run_linear

  dev = Device["AMD"]
  if dev.arch != "gfx950": raise unittest.SkipTest("requires gfx950 hardware")
  out_gpu = Buffer(dev.device, LANES * 4).ensure_allocated()
  insts = _insts(instructions, out_reg, out_gpu._buf)
  sink = UOp.sink(UOp.param(0, dtypes.uint32, (LANES,)), UOp.special(LANES, "lidx0"), arg=KernelInfo("test"))
  prg = UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=tuple(UOp(Ops.INS, arg=(x, dtypes.void)) for x in insts))))
  run_linear(UOp(Ops.LINEAR, src=(prg.call(UOp.from_buffer(out_gpu, dtypes.uint32)),)), wait=True)
  return out_gpu.as_memoryview().cast("I")[0]

def run_cdna(instructions: list, out_reg: int = 2) -> int:
  emu = _run_emu(instructions, out_reg)
  if not USE_HW: return emu
  hw = _run_hw(instructions, out_reg)
  if emu != hw: raise AssertionError(f"Emulator vs Hardware mismatch: emu=0x{emu:08x} hw=0x{hw:08x}")
  return hw

class TestCDNAVOP3(unittest.TestCase):
  def test_cvt_pk_fp8_f32_preserves_upper_half(self):
    """V_CVT_PK_FP8_F32 with OPSEL[3]=0 writes only D[15:0]."""
    out = run_cdna([
      cdna.s_mov_b32(cdna.s[0], 0xdeadbeef),
      cdna.v_mov_b32_e32(cdna.v[2], cdna.s[0]),
      cdna.v_mov_b32_e32(cdna.v[0], 1.0),
      cdna.v_mov_b32_e32(cdna.v[1], 2.0),
      cdna.v_cvt_pk_fp8_f32(cdna.v[2], cdna.v[0], cdna.v[1]),
    ])
    self.assertEqual(out, 0xdead4038)

  def test_cvt_pk_bf8_f32_overflow_and_inf(self):
    """V_CVT_PK_BF8_F32 converts finite overflow and infinities to E5M2 infinities."""
    for name, bits, expected in [
      ("finite_overflow", 0x47700000, 0x7c),
      ("pos_inf", 0x7f800000, 0x7c),
      ("neg_inf", 0xff800000, 0xfc),
    ]:
      with self.subTest(name=name):
        out = run_cdna([
          cdna.s_mov_b32(cdna.s[0], 0xdeadbeef),
          cdna.v_mov_b32_e32(cdna.v[2], cdna.s[0]),
          cdna.s_mov_b32(cdna.s[0], bits),
          cdna.v_mov_b32_e32(cdna.v[0], cdna.s[0]),
          cdna.v_mov_b32_e32(cdna.v[1], 1.0),
          cdna.v_cvt_pk_bf8_f32(cdna.v[2], cdna.v[0], cdna.v[1]),
        ])
        self.assertEqual(out, 0xdead3c00 | expected)
