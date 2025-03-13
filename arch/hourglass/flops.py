"""k-diffusion transformer diffusion models, version 2.
Codes adopted from https://github.com/crowsonkb/k-diffusion
"""

from contextlib import contextmanager
import math
import threading
import torch
from liblib import get_shared_cache_path  # @liblib adapter
import os

# @liblib adapter: FLOP 计数存储路径
FLOP_CACHE_PATH = get_shared_cache_path("k_diffusion_flop_count.pt")

state = threading.local()
state.flop_counter = None

@contextmanager
def flop_counter(enable=True):
    try:
        old_flop_counter = state.flop_counter
        state.flop_counter = FlopCounter() if enable else None
        yield state.flop_counter
        if state.flop_counter:
            state.flop_counter.save()  # @liblib adapter: 计算完成后保存 FLOP 结果
    finally:
        state.flop_counter = old_flop_counter


class FlopCounter:
    def __init__(self):
        self.ops = []
        self.load()  # @liblib adapter: 尝试加载已有 FLOP 记录

    def op(self, op, *args, **kwargs):
        self.ops.append((op, args, kwargs))

    @property
    def flops(self):
        flops = 0
        for op, args, kwargs in self.ops:
            flops += op(*args, **kwargs)
        return flops

    # @liblib adapter: 读取 FLOP 计数
    def load(self):
        if os.path.exists(FLOP_CACHE_PATH):
            self.ops = torch.load(FLOP_CACHE_PATH)
    
    # @liblib adapter: 保存 FLOP 计数
    def save(self):
        torch.save(self.ops, FLOP_CACHE_PATH)


def op(op, *args, **kwargs):
    if getattr(state, "flop_counter", None):
        state.flop_counter.op(op, *args, **kwargs)


def op_linear(x, weight):
    return math.prod(x) * weight[0]


def op_attention(q, k, v):
    *b, s_q, d_q = q
    *b, s_k, d_k = k
    *b, s_v, d_v = v
    return math.prod(b) * s_q * s_k * (d_q + d_v)


def op_natten(q, k, v, kernel_size):
    *q_rest, d_q = q
    *_, d_v = v
    return math.prod(q_rest) * (d_q + d_v) * kernel_size**2
