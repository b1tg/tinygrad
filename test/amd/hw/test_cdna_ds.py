import unittest
import tinygrad.runtime.autogen.amd.cdna.ins as cdna
from test.amd.hw.test_cdna_vop3 import run_cdna_lanes

class TestCDNADSSwizzle(unittest.TestCase):
  def test_modes(self):
    for offset in (0x041f, 0x00a0, 0x801b, 0xc020, 0xe000):
      with self.subTest(offset=hex(offset)):
        out = run_cdna_lanes([cdna.ds_swizzle_b32(vdst=cdna.v[2], addr=cdna.v[255], offset0=offset & 255, offset1=offset >> 8), cdna.s_waitcnt(0)])
        self.assertEqual(sorted(out), [5]*32 + [37]*32 if offset == 0x00a0 else list(range(64)))

if __name__ == "__main__":
  unittest.main()
