import tempfile, pathlib
from tinygrad.helpers import system
from tinygrad.device import Compiler, CompileError

class MCCCompiler(Compiler):
  def __init__(self, arch:str, cache_key:str="musa"):
    self.arch = arch
    super().__init__(f"compile_{cache_key}_{arch}")
  def compile(self, src:str) -> bytes:
    with tempfile.TemporaryDirectory() as td:
      sf, of = pathlib.Path(td)/"k.mu", pathlib.Path(td)/"k.fatbin"
      sf.write_text(src)
      try: system(f"mcc -x musa -mtgpu --offload-arch={self.arch} -O2 --cuda-device-only -o {of} {sf}")
      except Exception as e: raise CompileError(str(e)) from e
      return of.read_bytes()
