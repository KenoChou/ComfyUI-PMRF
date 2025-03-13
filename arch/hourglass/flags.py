"""k-diffusion transformer diffusion models, version 2.
Codes adopted from https://github.com/crowsonkb/k-diffusion
"""

from contextlib import contextmanager
from functools import update_wrapper
import os
import threading
import torch
from liblib import get_shared_cache_path  # @liblib adapter

# @liblib adapter: 使用共享存储路径管理配置
CONFIG_PATH = get_shared_cache_path("k_diffusion_config.pt")

# @liblib adapter: 读取共享存储中的配置
def load_config():
    if os.path.exists(CONFIG_PATH):
        return torch.load(CONFIG_PATH)
    return {"use_compile": True, "use_flash_attention_2": True}

config = load_config()

def get_use_compile():
    return config.get("use_compile", True)

def get_use_flash_attention_2():
    return config.get("use_flash_attention_2", True)

state = threading.local()
state.checkpointing = False

@contextmanager
def checkpointing(enable=True):
    try:
        old_checkpointing, state.checkpointing = state.checkpointing, enable
        yield
    finally:
        state.checkpointing = old_checkpointing

def get_checkpointing():
    return getattr(state, "checkpointing", False)

class compile_wrap:
    def __init__(self, function, *args, **kwargs):
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self._compiled_function = None
        update_wrapper(self, function)

    @property
    def compiled_function(self):
        if self._compiled_function is not None:
            return self._compiled_function
        if get_use_compile():
            try:
                self._compiled_function = torch.compile(self.function, *self.args, **self.kwargs)
            except RuntimeError:
                print("[WARN] torch.compile failed, using uncompiled function")  # @liblib adapter
                self._compiled_function = self.function
        else:
            self._compiled_function = self.function
        return self._compiled_function

    def __call__(self, *args, **kwargs):
        return self.compiled_function(*args, **kwargs)
