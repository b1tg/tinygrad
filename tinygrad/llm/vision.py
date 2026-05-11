from __future__ import annotations
import pathlib
from tinygrad import Tensor, nn
from tinygrad.llm.gguf import gguf_load
from tinygrad.llm.model import apply_rope, compute_mrope_freqs

def get_rope_index(images: list[tuple], max_context: int) -> list:
  pos, cp, img_at = [], 0, {img[1]: img for img in images}
  for i in range(max_context):
    if i in img_at:
      _, _, count, nx, ny = img_at[i]
      pos += [[cp, cp + j // nx, cp + j % nx] for j in range(count)]
      cp += max(nx, ny)
    else:
      pos.append([cp, cp, cp])
      cp += 1
  return pos

def scatter_image_embeds(embd: Tensor, images: list[tuple]) -> Tensor:
  seq_len = embd.shape[1]
  idx = Tensor.arange(seq_len).reshape(1, -1, 1)
  mask = Tensor.zeros(1, seq_len, 1)
  img_embd_full = Tensor.zeros_like(embd)
  for ie, start, count, _, _ in images:
    mask = mask + ((idx >= start) * (idx < start + count)).float()
    img_embd_full = img_embd_full + ie.reshape(1, count, -1).float().pad(((0, 0), (start, seq_len - start - count), (0, 0)))
  return mask.where(img_embd_full, embd)

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
    self.ln1, self.ln2 = nn.LayerNorm(n_embd, eps), nn.LayerNorm(n_embd, eps)
    self.attn_qkv = nn.Linear(n_embd, n_embd * 3)
    self.attn_out = nn.Linear(n_embd, n_embd)
    self.ffn_up, self.ffn_down = nn.Linear(n_embd, ffn_dim), nn.Linear(ffn_dim, n_embd)
    self.n_head = n_head

  def __call__(self, x: Tensor, freqs_cis: Tensor) -> Tensor:
    B, T, D = x.shape
    dh, rot = D // self.n_head, D // self.n_head // 2
    h = self.ln1(x)
    qkv = self.attn_qkv(h).reshape(B, T, 3, self.n_head, dh)
    q, k, v = qkv[:, :, 0].transpose(1, 2), qkv[:, :, 1].transpose(1, 2), qkv[:, :, 2].transpose(1, 2)
    q = apply_rope(q[..., :rot], freqs_cis).cat(q[..., rot:], dim=-1)
    k = apply_rope(k[..., :rot], freqs_cis).cat(k[..., rot:], dim=-1)
    x = x + self.attn_out(q.scaled_dot_product_attention(k, v).transpose(1, 2).reshape(B, T, D))
    return x + self.ffn_down(self.ffn_up(self.ln2(x)).gelu())

class _VisionCore:
  def __init__(self, n_embd, n_head, n_layer, ffn_dim, patch_size, eps, max_pos_embd):
    self.blk = [VisionBlock(n_embd, n_head, ffn_dim, eps) for _ in range(n_layer)]
    self.patch_embd = {"weight": Tensor.zeros(n_embd, 3, patch_size, patch_size), "bias": Tensor.zeros(n_embd)}
    self.patch_embd_1 = {"weight": Tensor.zeros(n_embd, 3, patch_size, patch_size)}
    self.post_ln = nn.LayerNorm(n_embd, eps)
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

    pos = Tensor([[y+dy, xp+dx] for y in range(0, ph, 2) for xp in range(0, pw, 2) for dy in range(2) for dx in range(2)])
    freqs_cis = compute_mrope_freqs(pos, dh // 2, 10000.0, (dh // 8, dh // 8), chunked=True)

    for block in self.v.blk: x = block(x, freqs_cis)
    x = self.v.post_ln(x)

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
