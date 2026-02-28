from __future__ import annotations

import torch
from torch.nn import Module
from torch import nn, Tensor

# helpers

def exists(v):
    return v is not None

# classes

class BufferDict(Module):
    def __init__(
        self,
        key_value_dict: dict[str, Tensor]
    ):

        super().__init__()
        self.key_list = list(key_value_dict.keys())
        
        self.key_map = {k: k.replace('.', '_') for k in self.key_list}
        self.reverse_key_map = {v: k for k, v in self.key_map.items()}

        for k, v in key_value_dict.items():
            self.register_buffer(self.key_map[k], v)

    def __getitem__(self, key: str) -> Tensor:
        if key in self.key_map:
            return getattr(self, self.key_map[key])

        if key in self.reverse_key_map:
            return getattr(self, key)

        raise KeyError(key)

    def __setitem__(self, key: str, value: Tensor):
        self[key].copy_(value)

    def __len__(self):
        return len(self.key_list)

    def __contains__(self, key: str):
        return key in self.key_map or key in self.reverse_key_map

    def keys(self):
        return iter(self.key_list)

    def values(self):
        for k in self.key_list:
            yield self[k]

    def items(self):
        for k in self.key_list:
            yield k, self[k]
