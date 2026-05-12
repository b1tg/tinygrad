import pathlib
from typing import NamedTuple
from tinygrad import Tensor, nn
from tinygrad.llm.gguf import gguf_load
from tinygrad.llm.model import apply_rope, compute_mrope_freqs

class ImageEmbed(NamedTuple):
  embd: Tensor    # (n_tokens, dim) vision encoder output
  start: int      # token position in the sequence
  n_tokens: int   # nx * ny merged patches
  nx: int         # grid width after merge
  ny: int         # grid height after merge

def get_rope_index(images: list[ImageEmbed], max_context: int) -> list[tuple[int, int, int]]:
  # time/height/width positions for M-RoPE
  img_by_start = {img.start: img for img in images}
  thw, offset, i = [], 0, 0
  while i < max_context:
    img = img_by_start.get(i)
    count, nx, ny = (img.n_tokens, img.nx, img.ny) if img else (1, 1, 1)
    thw += [(offset, offset + j // nx, offset + j % nx) for j in range(count)]
    offset += max(nx, ny)
    i += count
  return thw

def scatter_image_embeds(embd: Tensor, images: list[ImageEmbed]) -> Tensor:
  seq_len = embd.shape[1]
  idx = Tensor.arange(seq_len).reshape(1, -1, 1)
  mask = Tensor.zeros(1, seq_len, 1)
  img_embd_full = Tensor.zeros_like(embd)
  for img in images:
    mask = mask + ((idx >= img.start) * (idx < img.start + img.n_tokens)).float()
    img_embd_full = img_embd_full + img.embd.reshape(1, img.n_tokens, -1).float().pad(((0,0), (img.start, seq_len-img.start-img.n_tokens), (0,0)))
  return mask.where(img_embd_full, embd)

class VisionBlock:
  def __init__(self, n_embd: int, n_head: int, ffn_dim: int, eps: float):
    self.ln1, self.ln2 = nn.LayerNorm(n_embd, eps), nn.LayerNorm(n_embd, eps)
    self.attn_qkv = nn.Linear(n_embd, n_embd * 3)
    self.attn_out = nn.Linear(n_embd, n_embd)
    self.ffn_up, self.ffn_down = nn.Linear(n_embd, ffn_dim), nn.Linear(ffn_dim, n_embd)
    self.n_head = n_head

  def __call__(self, x: Tensor, freqs_cis: Tensor) -> Tensor:
    B, T, dh = x.shape[0], x.shape[1], x.shape[-1] // self.n_head
    h = self.ln1(x)
    qkv = self.attn_qkv(h).reshape(B, T, 3, self.n_head, dh)
    q, k, v = qkv[:, :, 0].transpose(1, 2), qkv[:, :, 1].transpose(1, 2), qkv[:, :, 2].transpose(1, 2)
    q = apply_rope(q, freqs_cis)
    k = apply_rope(k, freqs_cis)
    x = x + self.attn_out(q.scaled_dot_product_attention(k, v).transpose(1, 2).reshape(B, T, -1))
    return x + self.ffn_down(self.ffn_up(self.ln2(x)).gelu())

class _VisionCore:
  def __init__(self, n_embd, n_head, n_layer, ffn_dim, patch_size, eps, max_pos_embd):
    self.blk = [VisionBlock(n_embd, n_head, ffn_dim, eps) for _ in range(n_layer)]
    self.patch_embd = nn.Conv2d(3, n_embd, patch_size, stride=patch_size)
    self.patch_embd_1 = nn.Conv2d(3, n_embd, patch_size, stride=patch_size, bias=False)
    self.post_ln = nn.LayerNorm(n_embd, eps)
    self.position_embd = {"weight": Tensor.zeros(max_pos_embd, n_embd)}

class VisionEncoder:
  def __init__(self, n_embd:int, n_head:int, n_layer:int, ffn_dim:int, patch_size:int, projection_dim:int, eps:float,
               image_size:int, merge_size:int, image_mean:list[float], image_std:list[float]):
    self.v = _VisionCore(n_embd, n_head, n_layer, ffn_dim, patch_size, eps, (image_size // patch_size) ** 2)
    ms2 = merge_size * merge_size
    self.mm = {0: nn.Linear(n_embd * ms2, n_embd * ms2), 2: nn.Linear(n_embd * ms2, projection_dim)}
    self.n_embd, self.n_head, self.patch_size, self.image_size, self.merge_size = n_embd, n_head, patch_size, image_size, merge_size
    self._image_mean, self._image_std = image_mean, image_std

  def __call__(self, image: Tensor):
    ms, ms2 = self.merge_size, self.merge_size * self.merge_size
    ph, pw = image.shape[2] // self.patch_size, image.shape[3] // self.patch_size
    x = self.v.patch_embd(image) + self.v.patch_embd_1(image)
    x = x.reshape(-1, self.n_embd, ph//ms, ms, pw//ms, ms).permute(0, 2, 4, 3, 5, 1).reshape(-1, ph*pw, self.n_embd)
    n_per_side = self.image_size // self.patch_size
    if ph != n_per_side or pw != n_per_side:
      pos_w = self.v.position_embd["weight"].reshape(1, n_per_side, n_per_side, self.n_embd).permute(0, 3, 1, 2)
      pos_w = pos_w.interpolate((ph, pw), mode="linear", align_corners=True).contiguous().permute(0, 2, 3, 1)
    else:
      pos_w = self.v.position_embd["weight"].reshape(1, n_per_side, n_per_side, self.n_embd)
    x = x + pos_w.reshape(1, ph//ms, ms, pw//ms, ms, self.n_embd).permute(0, 1, 3, 2, 4, 5).reshape(1, ph*pw, self.n_embd)
    dh = self.n_embd // self.n_head
    pos = [[y+dy, xp+dx] for y in range(0, ph, ms) for xp in range(0, pw, ms) for dy in range(ms) for dx in range(ms)]
    freqs = compute_mrope_freqs(Tensor(pos), dh//2, 10000.0, (dh//4, dh//4), chunked=True)
    for block in self.v.blk: x = block(x, freqs)
    x = self.v.post_ln(x)
    return self.mm[2](self.mm[0](x.reshape(-1, ph*pw//ms2, self.n_embd*ms2)).gelu()), pw//ms, ph//ms

  def encode_image(self, img_path: str) -> tuple[Tensor, int, int]:
    from PIL import Image
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    grid = self.patch_size * self.merge_size
    new_w, new_h = max(grid, round(w / grid) * grid), max(grid, round(h / grid) * grid)
    while (new_h // self.patch_size) * (new_w // self.patch_size) > 4096:
      new_w, new_h = max(grid, new_w - grid), max(grid, new_h - grid)
    img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    t = Tensor(list(img.getdata())).reshape(new_h, new_w, 3).float() / 255.0
    image = ((t - Tensor(self._image_mean)) / Tensor(self._image_std)).permute(2, 0, 1).reshape(1, 3, new_h, new_w)
    embd, nx, ny = self(image)
    return embd.realize().squeeze(0), nx, ny

  @staticmethod
  def from_gguf(path):
    kv, sd = gguf_load(path if isinstance(path, pathlib.Path) else pathlib.Path(path))
    assert kv['general.architecture'] == 'clip'
    ne, nh, nl = kv['clip.vision.embedding_length'], kv['clip.vision.attention.head_count'], kv['clip.vision.block_count']
    ps, proj_dim = kv['clip.vision.patch_size'], kv['clip.vision.projection_dim']
    eps = kv['clip.vision.attention.layer_norm_epsilon']
    image_size = kv['clip.vision.image_size']
    ffn_dim = kv['clip.vision.feed_forward_length']
    merge_size = kv['clip.vision.spatial_merge_size']
    sd = {k.replace('patch_embd.weight.1', 'patch_embd_1.weight'): v for k, v in sd.items()}
    enc = VisionEncoder(ne, nh, nl, ffn_dim, ps, proj_dim, eps, image_size, merge_size, kv['clip.vision.image_mean'], kv['clip.vision.image_std'])
    nn.state.load_state_dict(enc, sd, verbose=False, consume=True, realize=False)
    return enc
