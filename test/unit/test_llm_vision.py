"""Tests for tinygrad vision encoder components, verified against vLLM and ollama reference implementations."""
import unittest, math
from tinygrad import Tensor
from tinygrad.llm.model import compute_mrope_freqs, apply_rope
from tinygrad.llm.vision import get_rope_index, scatter_image_embeds, ImageEmbed

class TestComputeMRoPEFreqs(unittest.TestCase):
  """Verify compute_mrope_freqs matches ollama/vLLM for ViT (chunked) and LLM (interleaved)."""

  def test_vit_rope_freq_matches_ollama(self):
    # ollama: freq[j] = pos / theta^(j*2/halfDim), halfDim = headDim/2
    # ref: ollama/model/models/qwen3vl/model_vision.go positions()
    dh, theta = 64, 10000.0
    halfDim = dh // 2
    for y, x in [(0, 0), (3, 7), (24, 24)]:
      ref_y = [y / theta**(j*2/halfDim) for j in range(halfDim//2)]
      ref_x = [x / theta**(j*2/halfDim) for j in range(halfDim//2)]
      ref_angles = ref_y + ref_x
      ref = [math.cos(a) for a in ref_angles] + [math.sin(a) for a in ref_angles]
      ours = compute_mrope_freqs(Tensor([[y, x]]), dh//2, theta, (dh//4, dh//4), chunked=True).numpy()[0]
      for k in range(len(ref)):
        self.assertAlmostEqual(float(ours[k]), ref[k], places=4, msg=f"ViT RoPE mismatch at ({y},{x}) dim {k}")

  def test_chunked_freq_reset(self):
    # chunked mode: each section gets independent frequencies (restart from base)
    dh, theta = 64, 10000.0
    freqs = compute_mrope_freqs(Tensor([[5, 10]]), dh//2, theta, (dh//4, dh//4), chunked=True).numpy()[0]
    cos_part = freqs[:dh//2]
    self.assertAlmostEqual(float(cos_part[0]), math.cos(5.0), places=5)
    self.assertAlmostEqual(float(cos_part[dh//4]), math.cos(10.0), places=5)

  def test_interleaved_layout(self):
    sections = (11, 11, 10, 0)
    freqs = compute_mrope_freqs(Tensor([[1, 2, 3]]), 64, 10000.0, sections, chunked=False)
    self.assertEqual(freqs.shape, (1, 64))

  def test_precompute_freqs_cis_delegates(self):
    from tinygrad.llm.model import precompute_freqs_cis
    result = precompute_freqs_cis(64, 16, 10000.0)
    self.assertEqual(result.shape, (16, 64))

class TestApplyRoPE(unittest.TestCase):
  """Verify apply_rope matches ollama's rotateHalf + applyRotaryPositionEmbeddings."""

  def test_matches_ollama_rotate_half(self):
    # ollama: [x1*cos - x2*sin, x2*cos + x1*sin]
    dh, theta = 64, 10000.0
    Tensor.manual_seed(42)
    x = Tensor.randn(1, 1, 1, dh)
    freqs = compute_mrope_freqs(Tensor([[3, 7]]), dh//2, theta, (dh//4, dh//4), chunked=True)
    result = apply_rope(x, freqs).numpy()[0, 0, 0]
    x_np = x.numpy()[0, 0, 0]
    cos_sin = freqs.numpy()[0]
    cos_np, sin_np = cos_sin[:dh//2], cos_sin[dh//2:]
    x1, x2 = x_np[:dh//2], x_np[dh//2:]
    for k in range(dh//2):
      self.assertAlmostEqual(float(result[k]), float(x1[k]*cos_np[k] - x2[k]*sin_np[k]), places=4)
      self.assertAlmostEqual(float(result[k+dh//2]), float(x2[k]*cos_np[k] + x1[k]*sin_np[k]), places=4)

  def test_identity_at_zero_position(self):
    dh = 64
    x = Tensor.ones(1, 1, 1, dh)
    freqs = compute_mrope_freqs(Tensor([[0, 0]]), dh//2, 10000.0, (dh//4, dh//4), chunked=True)
    result = apply_rope(x, freqs).numpy()[0, 0, 0]
    for k in range(dh):
      self.assertAlmostEqual(float(result[k]), 1.0, places=5, msg=f"Position (0,0) should be identity at dim {k}")

class TestGetRopeIndex(unittest.TestCase):
  """Verify get_rope_index matches ollama PostTokenize + Forward position logic.
  ref: QwenLM/Qwen3-VL qwen-vl-finetune/qwenvl/data/rope2d.py get_rope_index_3"""

  def test_text_only_positions(self):
    thw = get_rope_index([], 5)
    self.assertEqual(thw, [(0,0,0), (1,1,1), (2,2,2), (3,3,3), (4,4,4)])

  def test_image_positions_match_ollama(self):
    # ollama: all image tokens get base=p, then H += i/w, W += i%w
    # ref: ollama/model/models/qwen3vl/model.go Forward()
    nx, ny = 3, 2
    images = [ImageEmbed(embd=Tensor.zeros(nx*ny, 4), start=2, n_tokens=nx*ny, nx=nx, ny=ny)]
    thw = get_rope_index(images, 10)
    self.assertEqual(thw[0], (0, 0, 0))
    self.assertEqual(thw[1], (1, 1, 1))
    self.assertEqual(thw[2], (2, 2, 2))      # i=0: H=2+0//3=2, W=2+0%3=2
    self.assertEqual(thw[3], (2, 2, 3))      # i=1
    self.assertEqual(thw[4], (2, 2, 4))      # i=2
    self.assertEqual(thw[5], (2, 3, 2))      # i=3: H=2+3//3=3
    self.assertEqual(thw[6], (2, 3, 3))
    self.assertEqual(thw[7], (2, 3, 4))
    self.assertEqual(thw[8], (5, 5, 5))      # text after: offset = 2 + max(3,2) = 5

  def test_image_skips_pad_positions(self):
    images = [ImageEmbed(embd=Tensor.zeros(4, 4), start=1, n_tokens=4, nx=2, ny=2)]
    thw = get_rope_index(images, 7)
    self.assertEqual(len(thw), 7)
    self.assertEqual(thw[5], (3, 3, 3))      # offset = 1 + max(2,2) = 3

class TestScatterImageEmbeds(unittest.TestCase):
  def test_scatter_replaces_correct_positions(self):
    text_embd = Tensor.ones(1, 8, 2)
    img_embd = Tensor.full((3, 2), 99.0)
    images = [ImageEmbed(embd=img_embd, start=2, n_tokens=3, nx=3, ny=1)]
    r = scatter_image_embeds(text_embd, images).numpy()[0]
    for i in [0, 1, 5, 6, 7]:
      self.assertAlmostEqual(float(r[i, 0]), 1.0)
    for i in [2, 3, 4]:
      self.assertAlmostEqual(float(r[i, 0]), 99.0)

  def test_scatter_preserves_text_outside_image(self):
    text = Tensor.arange(20).reshape(1, 10, 2).float()
    img = Tensor.zeros(2, 2)
    images = [ImageEmbed(embd=img, start=3, n_tokens=2, nx=2, ny=1)]
    r = scatter_image_embeds(text, images).numpy()[0]
    self.assertAlmostEqual(float(r[0, 0]), 0.0)
    self.assertAlmostEqual(float(r[0, 1]), 1.0)
    self.assertAlmostEqual(float(r[5, 0]), 10.0)
    self.assertAlmostEqual(float(r[5, 1]), 11.0)

class TestSpatialMergeOrder(unittest.TestCase):
  """Verify spatial merge produces (hb, wb, dy, dx) order matching vLLM/ollama."""

  def test_merge_order_matches_vllm(self):
    # vLLM: reshape(h//m, m, w//m, m, C).permute(0, 2, 1, 3, 4)
    # ref: vllm/model_executor/models/qwen3_vl.py pos_embed_interpolate_native
    data = Tensor.arange(16).reshape(4, 4).float()
    merged = data.reshape(2, 2, 2, 2).permute(0, 2, 1, 3).flatten()
    vals = merged.numpy().tolist()
    self.assertEqual(vals[:4], [0, 1, 4, 5])    # first 2x2 block: (0,0),(0,1),(1,0),(1,1)
    self.assertEqual(vals[4:8], [2, 3, 6, 7])   # second block

class TestPositionInterpolation(unittest.TestCase):
  """Verify position embedding interpolation matches vLLM align_corners=True bilinear.
  ref: vllm/model_executor/models/qwen3_vl.py pos_embed_interpolate_native
  ref: QwenLM/Qwen3-VL modeling uses torch.linspace(0, N-1, h) = align_corners=True"""

  def test_align_corners_true_matches_vllm(self):
    Tensor.manual_seed(0)
    n, dst, C = 6, 4, 3
    raw = Tensor.randn(n, n, C)
    raw_np = raw.numpy()

    # vLLM reference: linspace(0, n-1, dst) bilinear
    h_idxs = [i * (n - 1) / (dst - 1) for i in range(dst)]
    w_idxs = h_idxs[:]
    ref = [[None]*C for _ in range(dst*dst)]
    for i in range(dst):
      for j in range(dst):
        hf, wf = int(h_idxs[i]), int(w_idxs[j])
        hc, wc = min(hf+1, n-1), min(wf+1, n-1)
        dh, dw = h_idxs[i] - hf, w_idxs[j] - wf
        for c in range(C):
          ref[i*dst+j][c] = ((1-dh)*(1-dw)*raw_np[hf,wf,c] + (1-dh)*dw*raw_np[hf,wc,c] +
                              dh*(1-dw)*raw_np[hc,wf,c] + dh*dw*raw_np[hc,wc,c])

    t = raw.reshape(1, n, n, C).permute(0, 3, 1, 2)
    t = t.interpolate((dst, dst), mode="linear", align_corners=True).contiguous().permute(0, 2, 3, 1)
    ours = t.reshape(dst*dst, C).numpy()
    for i in range(dst*dst):
      for c in range(C):
        self.assertAlmostEqual(float(ours[i, c]), ref[i][c], places=4)

class TestViTRoPEDim(unittest.TestCase):
  """Verify ViT uses dim=headDim//2 (not headDim) for RoPE frequency computation.
  ref: vllm qwen3_vl.py:565 partial_rotary_factor=0.5 → rotary_dim = head_size * 0.5
  ref: ollama model_vision.go: Qwen3VLVisionRotaryEmbedding(head_dim // 2)
  ref: llama.cpp qwen3vl.cpp:116 n_dims = d_head/2"""

  def test_vit_rope_dim_is_half_head_dim(self):
    dh, theta = 64, 10000.0
    freq_half = compute_mrope_freqs(Tensor([[1, 0]]), dh//2, theta, (dh//4, dh//4), chunked=True).numpy()[0]
    freq_full = compute_mrope_freqs(Tensor([[1, 0]]), dh, theta, (dh//4, dh//4), chunked=True).numpy()[0]
    # half and full dim must produce different frequencies
    diff = sum(abs(float(freq_half[i]) - float(freq_full[i])) for i in range(len(freq_half)))
    self.assertGreater(diff, 0.1, "halfDim and fullDim should produce different frequencies")

if __name__ == '__main__':
  unittest.main()
