import torch
from torch import nn
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from einops import einsum, rearrange, repeat, pack, unpack

from adam_atan2_pytorch.adam_atan2_with_wasserstein_reg import Adam

import gymnasium as gym

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# classes

class StreamingDeepRL(Module):
    raise NotImplementedError
