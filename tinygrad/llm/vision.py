from __future__ import annotations
import pathlib
from tinygrad import Tensor, nn
from tinygrad.llm.gguf import gguf_load
from tinygrad.llm.model import apply_rope, compute_mrope_freqs

def get_rope_index(tokens: list[int], images: list[tuple], max_context: int, token_embd) -> tuple[list, Tensor]:
  img_ranges = {img[1]: img for img in images}
  pos, cp, i = [], 0, 0
  while i < max_context:
    if i < len(tokens) and i in img_ranges:
      _, _, count, nx, ny = img_ranges[i]
      pos += [[cp, cp + j // nx, cp + j % nx] for j in range(count)]
      cp += max(nx, ny)
      i += count
    else:
      pos.append([cp, cp, cp])
      cp += 1
      i += 1
  embd = token_embd(Tensor(tokens, dtype="int32").reshape(1, -1)).float()
  for img_embd, start, count, _, _ in sorted(images, key=lambda x: -x[1]):
    embd = embd[:, :start].cat(img_embd.reshape(1, count, -1).float(), embd[:, start+count:], dim=1)
  return pos, embd

def preprocess_image(img_path: str, patch_size: int = 16, merge_size: int = 2, mean: list|None = None, std: list|None = None) -> Tensor:
  from PIL import Image
  img = Image.open(img_path).convert('RGB')
  w, h = img.size
  grid = patch_size * merge_size
  new_w, new_h = max(grid, round(w / grid) * grid), max(grid, round(h / grid) * grid)
  while (new_h // patch_size) * (new_w // patch_size) // 4 > 4096:
    new_w, new_h = max(grid, new_w - grid), max(grid, new_h - grid)
  img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
  mean, std = mean or [0.5, 0.5, 0.5], std or [0.5, 0.5, 0.5]
  t = Tensor(list(img.getdata())).reshape(new_h, new_w, 3).float() / 255.0
  return ((t - Tensor(mean)) / Tensor(std)).permute(2, 0, 1).reshape(1, 3, new_h, new_w)

class VisionBlock:
  def __init__(self, n_embd: int, n_head: int, ffn_dim: int, eps: float):
    self.ln1 = {"weight": Tensor.ones(n_embd), "bias": Tensor.zeros(n_embd)}
    self.ln2 = {"weight": Tensor.ones(n_embd), "bias": Tensor.zeros(n_embd)}
    self.attn_qkv = nn.Linear(n_embd, n_embd * 3)
    self.attn_out = nn.Linear(n_embd, n_embd)
    self.ffn_up, self.ffn_down = nn.Linear(n_embd, ffn_dim), nn.Linear(ffn_dim, n_embd)
    self.n_head, self.eps = n_head, eps

  def _ln(self, x: Tensor, ln: dict) -> Tensor:
    mean = x.mean(-1, keepdim=True)
    return (x - mean) / ((x - mean).square().mean(-1, keepdim=True) + self.eps).sqrt() * ln["weight"] + ln["bias"]

  def __call__(self, x: Tensor, freqs_cis: Tensor) -> Tensor:
    B, T, D = x.shape
    dh, rot = D // self.n_head, D // self.n_head // 2
    h = self._ln(x, self.ln1)
    qkv = self.attn_qkv(h).reshape(B, T, 3, self.n_head, dh)
    q, k, v = qkv[:, :, 0].transpose(1, 2), qkv[:, :, 1].transpose(1, 2), qkv[:, :, 2].transpose(1, 2)
    q = apply_rope(q[..., :rot], freqs_cis).cat(q[..., rot:], dim=-1)
    k = apply_rope(k[..., :rot], freqs_cis).cat(k[..., rot:], dim=-1)
    x = x + self.attn_out(q.scaled_dot_product_attention(k, v).transpose(1, 2).reshape(B, T, D))
    return x + self.ffn_down(self.ffn_up(self._ln(x, self.ln2)).gelu())

class _VisionCore:
  def __init__(self, n_embd, n_head, n_layer, ffn_dim, patch_size, eps, max_pos_embd):
    self.blk = [VisionBlock(n_embd, n_head, ffn_dim, eps) for _ in range(n_layer)]
    self.patch_embd = {"weight": Tensor.zeros(n_embd, 3, patch_size, patch_size), "bias": Tensor.zeros(n_embd)}
    self.patch_embd_1 = {"weight": Tensor.zeros(n_embd, 3, patch_size, patch_size)}
    self.post_ln = {"weight": Tensor.ones(n_embd), "bias": Tensor.zeros(n_embd)}
    self.position_embd = {"weight": Tensor.zeros(max_pos_embd, n_embd)}

class VisionEncoder:
  def __init__(self, n_embd, n_head, n_layer, ffn_dim, patch_size, projection_dim, eps, max_pos_embd, image_size,
               image_mean=None, image_std=None):
    self.v = _VisionCore(n_embd, n_head, n_layer, ffn_dim, patch_size, eps, max_pos_embd)
    self.mm_0, self.mm_2 = nn.Linear(n_embd * 4, n_embd * 4), nn.Linear(n_embd * 4, projection_dim)
    self.n_embd, self.n_head, self.patch_size, self.eps, self.image_size = n_embd, n_head, patch_size, eps, image_size
    self._image_mean, self._image_std = image_mean or [0.5, 0.5, 0.5], image_std or [0.5, 0.5, 0.5]

  def __call__(self, image: Tensor) -> tuple[Tensor, int, int]:
    B, _, H, W = image.shape
    ps, ne = self.patch_size, self.n_embd
    ph, pw = H // ps, W // ps
    dh = ne // self.n_head

    x = image.conv2d(self.v.patch_embd["weight"], bias=self.v.patch_embd["bias"], stride=ps) + image.conv2d(self.v.patch_embd_1["weight"], stride=ps)
    x = x.reshape(B, ne, ph // 2, 2, pw // 2, 2).permute(0, 2, 4, 3, 5, 1).reshape(B, ph * pw, ne)

    # learned position embeddings (crop if image smaller than training size)
    max_ph = max_pw = self.image_size // ps
    pos_w = self.v.position_embd["weight"].reshape(1, max_ph, max_pw, ne)
    if ph != max_ph or pw != max_pw: pos_w = pos_w[:, :ph, :pw, :]
    pos_w = pos_w.reshape(1, ph, pw, ne)
    pos_w = pos_w.reshape(1, ph // 2, 2, pw // 2, 2, ne).permute(0, 1, 3, 2, 4, 5).reshape(1, ph * pw, ne)
    x = x + pos_w

    pos = [[y+dy, xp+dx] for y in range(0, ph, 2) for xp in range(0, pw, 2) for dy in range(2) for dx in range(2)]
    freqs_cis = compute_mrope_freqs(Tensor(pos), dh // 2, 10000.0, (dh // 8, dh // 8), interleaved=False)

    for block in self.v.blk: x = block(x, freqs_cis)

    ln = self.v.post_ln
    mean = x.mean(-1, keepdim=True)
    x = (x - mean) / ((x - mean).square().mean(-1, keepdim=True) + self.eps).sqrt() * ln["weight"] + ln["bias"]

    x = x.reshape(B, ph * pw // 4, ne * 4)
    x = self.mm_2(self.mm_0(x).gelu())
    return x, pw // 2, ph // 2

  def encode_image(self, img_path: str) -> tuple[Tensor, int, int]:
    embd, nx, ny = self(preprocess_image(img_path, self.patch_size, mean=self._image_mean, std=self._image_std))
    return embd.realize().squeeze(0), nx, ny

  @staticmethod
  def from_gguf(path) -> VisionEncoder:
    kv, sd = gguf_load(path if isinstance(path, pathlib.Path) else pathlib.Path(path))
    ne, nh, nl = kv['clip.vision.embedding_length'], kv['clip.vision.attention.head_count'], kv['clip.vision.block_count']
    ps, proj_dim = kv['clip.vision.patch_size'], kv['clip.vision.projection_dim']
    eps = kv.get('clip.vision.attention.layer_norm_epsilon', 1e-6)
    image_size = kv.get('clip.vision.image_size', 768)
    ffn_dim = sd['v.blk.0.ffn_up.weight'].shape[0]
    max_pos_embd = sd['v.position_embd.weight'].shape[0]
    sd = {k.replace('patch_embd.weight.1', 'patch_embd_1.weight').replace('mm.0.', 'mm_0.').replace('mm.2.', 'mm_2.'): v for k, v in sd.items()}
    enc = VisionEncoder(ne, nh, nl, ffn_dim, ps, proj_dim, eps, max_pos_embd, image_size,
                        kv.get('clip.vision.image_mean'), kv.get('clip.vision.image_std'))
    nn.state.load_state_dict(enc, sd, verbose=False, consume=True, realize=False)
    return enc
