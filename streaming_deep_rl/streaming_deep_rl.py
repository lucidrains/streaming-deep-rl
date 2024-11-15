import torch
from torch import nn, tensor
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from einops import einsum, rearrange, repeat, reduce, pack, unpack

from adam_atan2_pytorch.adam_atan2_with_wasserstein_reg import Adam

import gymnasium as gym

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# online normalization from Welford in 1962

class NormalizeObservation(Module):
    """
    Algorithm 6 in https://arxiv.org/abs/2410.14606
    """

    def __init__(
        self,
        dim = 1,
        eps = 1e-5
    ):
        super().__init__()
        self.dim = dim
        self.eps = eps

        self.register_buffer('step', tensor(1))
        self.register_buffer('running_mean', torch.zeros(dim))
        self.register_buffer('running_estimate_p', torch.ones(dim))

    @property
    def variance(self):
        p = self.running_estimate_p

        if self.step.item() == 1:
            return torch.ones_like(p)

        return (p / (self.step - 1))

    def forward(
        self,
        obs
    ):
        normalized = (obs - self.running_mean) / self.variance.clamp(min = self.eps).sqrt()

        if not self.training:
            return normalized

        step = self.step.item()
        mean = self.running_mean
        estimate_p = self.running_estimate_p

        if self.dim == 1:
            obs_mean = obs.mean()
        else:
            obs_mean = reduce(obs, '... d -> d', 'mean')

        delta = obs_mean - mean

        mean = mean + delta / step
        estimate_p = estimate_p + (obs_mean - mean) * delta

        self.running_mean.copy_(mean)
        self.running_estimate_p.copy_(estimate_p)

        self.step.add_(1)

        return normalized

class ScaleReward(Module):
    """
    Algorithm 5
    """

    def __init__(
        self,
        eps = 1e-5,
        discount_factor = 0.999
    ):
        super().__init__()
        self.eps = eps
        self.discount_factor = discount_factor

        self.register_buffer('step', tensor(0))
        self.register_buffer('running_reward', tensor(0.))
        self.register_buffer('running_estimate_p', tensor(0.))

    @property
    def variance(self):
        p = self.running_estimate_p

        if self.step.item() == 1:
            return torch.ones_like(p)

        return (p / (self.step - 1))

    def forward(
        self,
        reward,
        is_terminal = False
    ):

        normed_reward = reward / self.variance.clamp(min = self.eps).sqrt()

        if not self.training:
            return normed_reward

        self.step.add_(1)
        step = self.step.item()

        running_reward = self.running_reward.item()
        estimate_p = self.running_estimate_p.item()

        next_reward = running_reward * self.discount_factor * (1. - float(is_terminal)) + reward

        mu_hat = running_reward - running_reward / step
        next_estimate_p = estimate_p + running_reward * mu_hat

        self.running_reward.copy_(next_reward)
        self.running_estimate_p.copy_(next_estimate_p)

        return normed_reward

# classes

class StreamingDeepRL(Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError

# sanity check

if __name__ == '__main__':

    x = torch.randn((50000,)) * 10 + 2

    norm_obs = NormalizeObservation()

    for el in x:
        norm_obs(el)

    print(f'true mean: {x.mean()} | true std: {x.std()}')

    print(f'online mean: {norm_obs.running_mean.item()} | online std: {norm_obs.variance.sqrt().item()}')

    norm_reward = ScaleReward()

    for el in x:
        norm_reward(el, is_terminal = True)

    print(f'scaled reward std: {norm_reward.variance.sqrt().item()}')
