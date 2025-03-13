"""k-diffusion transformer diffusion models, version 2.
Codes adopted from https://github.com/crowsonkb/k-diffusion
"""

import math
import torch
import torch._dynamo
from torch import nn

from . import flags
from configs.config import get_juicefs_path
from configs.node_fields import PUILD_EVA_CLIP_MAPPINGS
from configs.node_fields import get_field_pre_values
if flags.get_use_compile():
    torch._dynamo.config.suppress_errors = True

def rotate_half(x):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    *shape, d, r = x.shape
    return x.view(*shape, d * r)

def apply_rotary_emb(freqs, t, start_index=0, scale=1.0):
    freqs = freqs.to(t)
    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim
    assert rot_dim <= t.shape[-1], f"feature dimension {t.shape[-1]} is not sufficient for rotation {rot_dim}"
    t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    return torch.cat((t_left, t, t_right), dim=-1)

def centers(start, stop, num, dtype=None, device=None):
    edges = torch.linspace(start, stop, num + 1, dtype=dtype, device=device)
    return (edges[:-1] + edges[1:]) / 2

def make_grid(h_pos, w_pos):
    grid = torch.stack(torch.meshgrid(h_pos, w_pos, indexing='ij'), dim=-1)
    h, w, d = grid.shape
    return grid.view(h * w, d)

def bounding_box(h, w, pixel_aspect_ratio=1.0):
    w_adj = w
    h_adj = h * pixel_aspect_ratio
    ar_adj = w_adj / h_adj
    y_min, y_max, x_min, x_max = -1.0, 1.0, -1.0, 1.0
    if ar_adj > 1:
        y_min, y_max = -1 / ar_adj, 1 / ar_adj
    elif ar_adj < 1:
        x_min, x_max = -ar_adj, ar_adj
    return y_min, y_max, x_min, x_max

def make_axial_pos(h, w, pixel_aspect_ratio=1.0, align_corners=False, dtype=None, device=None):
    y_min, y_max, x_min, x_max = bounding_box(h, w, pixel_aspect_ratio)
    if align_corners:
        h_pos = torch.linspace(y_min, y_max, h, dtype=dtype, device=device)
        w_pos = torch.linspace(x_min, x_max, w, dtype=dtype, device=device)
    else:
        h_pos = centers(y_min, y_max, h, dtype=dtype, device=device)
        w_pos = centers(x_min, x_max, w, dtype=dtype, device=device)
    return make_grid(h_pos, w_pos)

def load_cached_freqs(cache_key, shape):  # @liblib adapter: 从共享缓存加载频率参数
    shared_drive_path = get_juicefs_path()  # 获取共享存储路径
    cache_path = f"{shared_drive_path}/cache/{cache_key}.pt"
    # 如果缓存存在，直接加载
    if torch.exists(cache_path):
        return torch.load(cache_path)
    
    # 如果缓存不存在，创建并保存
    freqs = torch.linspace(1.0, 10.0 / 2, shape[-1]) * math.pi
    torch.save(freqs, cache_path)  # 保存到共享存储
    return freqs.expand(shape)

class AxialRoPE(nn.Module):
    def __init__(self, dim, n_heads, start_index=0):
        super().__init__()
        self.n_heads = n_heads
        self.start_index = start_index
        self.freqs_h = nn.Parameter(load_cached_freqs("freqs_h", (n_heads, dim // 4)))
        self.freqs_w = nn.Parameter(load_cached_freqs("freqs_w", (n_heads, dim // 4)))

    def extra_repr(self):
        dim = (self.freqs_h.shape[-1] + self.freqs_w.shape[-1]) * 2
        return f"dim={dim}, n_heads={self.n_heads}, start_index={self.start_index}"

    def get_freqs(self, pos):
        if pos.shape[-1] != 2:
            raise ValueError("input shape must be (..., 2)")
        freqs_h = pos[..., None, None, 0] * self.freqs_h.exp()
        freqs_w = pos[..., None, None, 1] * self.freqs_w.exp()
        freqs = torch.cat((freqs_h, freqs_w), dim=-1).repeat_interleave(2, dim=-1)
        return freqs.transpose(-2, -3)

    def forward(self, x, pos):
        freqs = self.get_freqs(pos)
        return apply_rotary_emb(freqs, x, self.start_index)
