import torch
from torch import nn, tensor, atan2, sqrt
from torch.nn import Module, ModuleList
from torch.optim.optimizer import Optimizer

import torch.nn.functional as F

from einops import einsum, rearrange, repeat, reduce, pack, unpack

from adam_atan2_pytorch.adam_atan2_with_wasserstein_reg import Adam

import gymnasium as gym

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# initialization

def sparse_init_(
    l: nn.Linear,
    sparsity = 0.9
):
    """
    Algorithm 1
    """
    weight, bias = l.weight, l.bias
    _, fan_in = weight.shape

    value = fan_in ** -0.5
    nn.init.uniform_(weight, -value, value)

    assert 0. <= sparsity <= 1.

    n = int(fan_in * sparsity)
    sparse_indices = torch.randperm(fan_in)[:n]
    nn.init.zeros_(weight[:, sparse_indices])

    if exists(bias):
        nn.init.zeros_(bias)

# online normalization from Welford in 1962

class NormalizeObservation(Module):
    """
    Algorithm 6 in https://arxiv.org/abs/2410.14606
    """

    def __init__(
        self,
        dim = 1,
        eps = 1e-5,
        time_dilate_factor = 1.
    ):
        super().__init__()
        self.dim = dim
        self.eps = eps

        self.time_dilate_factor = time_dilate_factor

        self.register_buffer('step', tensor(1))
        self.register_buffer('running_mean', torch.zeros(dim))
        self.register_buffer('running_estimate_p', torch.ones(dim))

    def reset_step(self):
        self.step.zero_()

    @property
    def time(self):
        return self.step / self.time_dilate_factor

    @property
    def variance(self):
        p = self.running_estimate_p

        if self.step.item() == 1:
            return torch.ones_like(p)

        return (p / (self.time - 1. / self.time_dilate_factor))

    def forward(
        self,
        obs
    ):
        normalized = (obs - self.running_mean) / self.variance.clamp(min = self.eps).sqrt()

        if not self.training:
            return normalized

        time = self.time.item()
        mean = self.running_mean
        estimate_p = self.running_estimate_p

        if self.dim == 1:
            obs_mean = obs.mean()
        else:
            obs_mean = reduce(obs, '... d -> d', 'mean')

        delta = obs_mean - mean

        mean = mean + delta / time
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
        discount_factor = 0.999,
        time_dilate_factor = 1.
    ):
        super().__init__()
        self.eps = eps
        self.discount_factor = discount_factor
        self.time_dilate_factor = time_dilate_factor

        self.register_buffer('step', tensor(0))
        self.register_buffer('running_reward', tensor(0.))
        self.register_buffer('running_estimate_p', tensor(0.))

    def reset_step(self):
        self.step.zero_()

    @property
    def time(self):
        return self.step / self.time_dilate_factor

    @property
    def variance(self):
        p = self.running_estimate_p

        if self.step.item() == 1:
            return torch.ones_like(p)

        return (p / (self.time - 1. / self.time_dilate_factor))

    def forward(
        self,
        reward,
        is_terminal = False
    ):

        normed_reward = reward / self.variance.clamp(min = self.eps).sqrt()

        if not self.training:
            return normed_reward

        self.step.add_(1)
        time = self.time.item()

        running_reward = self.running_reward.item()
        estimate_p = self.running_estimate_p.item()

        next_reward = running_reward * self.discount_factor * (1. - float(is_terminal)) + reward

        mu_hat = running_reward - running_reward / time
        next_estimate_p = estimate_p + running_reward * mu_hat

        self.running_reward.copy_(next_reward)
        self.running_estimate_p.copy_(next_estimate_p)

        return normed_reward

# "bro" mlp
# in this paper, they claim layernorm should be before activation, without any gamma + bias
# will go with BroMLP for now as i have seen it work well, and it is heavily normalized already

class BroMLP(Module):

    def __init__(
        self,
        dim,
        dim_out,
        dim_hidden = None,
        depth = 3,
        dropout = 0.,
        expansion_factor = 2,
        final_norm = False,
    ):
        super().__init__()

        dim_hidden = default(dim_hidden, dim * 2)

        layers = []

        self.proj_in = nn.Sequential(
            nn.Linear(dim, dim_hidden),
            nn.ReLU(),
            nn.LayerNorm(dim_hidden, bias = False),
        )

        dim_inner = dim_hidden * expansion_factor

        for _ in range(depth):

            layer = nn.Sequential(
                nn.Linear(dim_hidden, dim_inner),
                nn.Dropout(dropout),
                nn.ReLU(),
                nn.LayerNorm(dim_inner, bias = False),
                nn.Linear(dim_inner, dim_hidden),
                nn.LayerNorm(dim_hidden, bias = False),
            )

            nn.init.constant_(layer[-1].weight, 1e-5)
            layers.append(layer)

        # final layer out

        self.layers = ModuleList(layers)

        self.final_norm = nn.LayerNorm(dim_hidden) if final_norm else nn.Identity()

        self.proj_out = nn.Linear(dim_hidden, dim_out)

        self.apply(self.init_)

    def init_(self, module):
        if isinstance(module, nn.Linear):
            sparse_init_(module)

    def forward(self, x):

        x = self.proj_in(x)

        for layer in self.layers:
            x = layer(x) + x

        x = self.final_norm(x)
        return self.proj_out(x)

# adam atan2 w/ regenerative reg for improved continual learning
# todo - add the eligibility trace logic

class AdamAtan2(Optimizer):
    def __init__(
        self,
        params,
        lr = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay = 0.,
        regen_reg_rate = 0.,
        decoupled_wd = False,
        a = 1.27,
        b = 1.
    ):
        assert lr > 0.
        assert all([0. <= beta <= 1. for beta in betas])
        assert weight_decay >= 0.
        assert regen_reg_rate >= 0.
        assert not (weight_decay > 0. and regen_reg_rate > 0.)

        self._init_lr = lr
        self.decoupled_wd = decoupled_wd

        defaults = dict(
            lr = lr,
            betas = betas,
            a = a,
            b = b,
            weight_decay = weight_decay,
            regen_reg_rate = regen_reg_rate,
        )

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(
        self,
        closure: Callable | None = None
    ):

        loss = None
        if exists(closure):
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in filter(lambda p: exists(p.grad), group['params']):

                grad, lr, wd, regen_rate, beta1, beta2, a, b, state, init_lr = p.grad, group['lr'], group['weight_decay'], group['regen_reg_rate'], *group['betas'], group['a'], group['b'], self.state[p], self._init_lr

                # maybe decoupled weight decay

                if self.decoupled_wd:
                    wd /= init_lr

                # weight decay

                if wd > 0.:
                    p.mul_(1. - lr * wd)

                # regenerative regularization from Kumar et al. https://arxiv.org/abs/2308.11958

                if regen_rate > 0. and 'param_init' in state:
                    param_init = state['param_init']
                    p.lerp_(param_init, lr / init_lr * regen_rate)

                # init state if needed

                if len(state) == 0:
                    state['steps'] = 0
                    state['exp_avg'] = torch.zeros_like(grad)
                    state['exp_avg_sq'] = torch.zeros_like(grad)

                    if regen_rate > 0.:
                        state['param_init'] = p.clone()

                # get some of the states

                exp_avg, exp_avg_sq, steps = state['exp_avg'], state['exp_avg_sq'], state['steps']

                steps += 1

                # bias corrections

                bias_correct1 = 1. - beta1 ** steps
                bias_correct2 = 1. - beta2 ** steps

                # decay running averages

                exp_avg.lerp_(grad, 1. - beta1)
                exp_avg_sq.lerp_(grad * grad, 1. - beta2)

                # the following line is the proposed change to the update rule
                # using atan2 instead of a division with epsilon in denominator
                # a * atan2(exp_avg / bias_correct1, b * sqrt(exp_avg_sq / bias_correct2))

                den = exp_avg_sq.mul(b * b / bias_correct2).sqrt_()
                update = exp_avg.mul(1. / bias_correct1).atan2_(den)

                # update parameters

                p.add_(update, alpha = -lr * a)

                # increment steps

                state['steps'] = steps

        return loss

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

    sparse_init_(nn.Linear(512, 512))

    mlp = BroMLP(5, 5)