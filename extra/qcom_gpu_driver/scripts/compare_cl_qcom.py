#!/usr/bin/env python3
"""Compare CL vs QCOM backend results for LLM _attention block.
Identifies whether the issue is in dispatch (register setup) or compilation.

Usage: source extra/cl_android.sh
  OPENCL_PATH=/system/vendor/lib64/libOpenCL.so python3 extra/qcom_gpu_driver/scripts/compare_cl_qcom.py
"""
import os
os.environ.setdefault("LIBC_PATH", "/system/lib64/libc.so")
os.environ.setdefault("LLVM_QCOM_PATH", "/vendor/lib64/libllvm-qcom.so")
os.environ.setdefault("SPEC", "0")

from tinygrad import Tensor, dtypes, Device
from tinygrad.llm.model import Transformer
import numpy as np, sys

gguf_path = sys.argv[1] if len(sys.argv) > 1 else "qwen3.5:0.8b"
gguf = Tensor.from_url(gguf_path) if "/" in gguf_path or "." in gguf_path else None

# NOOPT=1 reference on QCOM
os.environ["DEV"] = "QCOM"
os.environ["NOOPT"] = "1"
if gguf is None:
  from tinygrad.llm.cli import fetch_model
  gguf = Tensor.from_url(fetch_model(gguf_path))
model, kv = Transformer.from_gguf(gguf)
tok = Tensor([[151643]], dtype=dtypes.int)
blk = model.blk[0]
x_in = model.token_embd(tok).float()
blk._init_state(x_in)
ref = blk._attention(blk.attn_norm(x_in), 0).numpy().flatten()
print(f"NOOPT=1 ref: {ref[:4]}  sum={ref.sum():.4f}")

# QCOM NOOPT=0
os.environ["NOOPT"] = "0"
blk._init_state(x_in)
qcom_out = blk._attention(blk.attn_norm(x_in), 0).numpy().flatten()
print(f"QCOM NOOPT=0: {qcom_out[:4]}  diff={np.abs(ref - qcom_out).max():.6f}")

# CL NOOPT=0
os.environ["DEV"] = "CL"
os.environ["OPENCL_PATH"] = os.environ.get("OPENCL_PATH", "/system/vendor/lib64/libOpenCL.so")
Device._opened_devices.clear()
os.environ["NOOPT"] = "0"
model_cl, _ = Transformer.from_gguf(gguf)
tok_cl = Tensor([[151643]], dtype=dtypes.int)
blk_cl = model_cl.blk[0]
x_in_cl = model_cl.token_embd(tok_cl).float()
blk_cl._init_state(x_in_cl)
cl_out = blk_cl._attention(blk_cl.attn_norm(x_in_cl), 0).numpy().flatten()
print(f"CL   NOOPT=0: {cl_out[:4]}  diff={np.abs(ref - cl_out).max():.6f}")

print(f"\nCL vs ref:  {np.abs(ref - cl_out).max():.6f}")
print(f"QCOM vs ref: {np.abs(ref - qcom_out).max():.6f}")
if np.abs(ref - cl_out).max() < 0.01:
  print("=> CL correct, QCOM wrong => dispatch issue")
else:
  print("=> Both wrong => kernel codegen issue")
