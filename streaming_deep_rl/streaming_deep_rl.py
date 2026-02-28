from __future__ import annotations
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn, tensor, is_tensor, atan2, sqrt
from torch.nn import Module, ModuleList, Linear, Sequential, ParameterDict

from torch.optim.optimizer import Optimizer

from einops import einsum, rearrange, repeat, reduce, pack, unpack

from hl_gauss_pytorch import HLGaussLayer

from discrete_continuous_embed_readout import Readout

from torch_einops_utils import tree_map_tensor

from ema_pytorch import EMA

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def cast_tensor(t):
    return tensor(t) if not is_tensor(t) else t

def to_device(tree, device):
    return tree_map_tensor(lambda t: t.to(device), tree)

# initialization

def sparse_init_(
    l: Linear,
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

class ObservationNormalizer(Module):
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
        obs,
        update = None
    ):
        normalized = (obs - self.running_mean) / self.variance.clamp(min = self.eps).sqrt()

        update = default(update, self.training)

        if not update:
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

class ScaleRewardNormalizer(Module):
    """
    Algorithm 5
    """

    def __init__(
        self,
        eps = 1e-5,
        discount_factor = 0.999,
        time_dilate_factor = 1.,
        mean_center = False
    ):
        super().__init__()
        self.eps = eps
        self.discount_factor = discount_factor
        self.time_dilate_factor = time_dilate_factor
        self.mean_center = mean_center

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
        is_terminal = False,
        update = None
    ):

        normed_reward = reward / self.variance.clamp(min = self.eps).sqrt()

        update = default(update, self.training)

        if not update:
            return normed_reward

        self.step.add_(1)
        time = self.time.item()

        running_reward = self.running_reward.item()
        estimate_p = self.running_estimate_p.item()

        next_reward = running_reward * self.discount_factor * (1. - float(is_terminal)) + reward

        mu_hat = next_reward

        # rewards are not mean centered

        if self.mean_center:
            mu_hat = mu_hat - next_reward / time

        next_estimate_p = estimate_p + next_reward * mu_hat

        self.running_reward.copy_(next_reward)
        self.running_estimate_p.copy_(next_estimate_p)

        return normed_reward

# classes

# streaming AC variant

class StreamingACLambda(Module):
    def __init__(
        self,
        *,
        actor: Module,
        critic: Module,
        dim_state,
        dim_actor,
        num_discrete_actions = 0,
        num_continuous_actions = 0,
        discount_factor = 0.999,
        eligibility_trace_decay = 0.8,
        actor_kappa = 3.,
        critic_kappa = 2.,
        actor_lr = 1e-4,
        critic_lr = 1e-4,
        critic_ema_beta = 0.95,
        entropy_weight = 0.01,
        init_sparsity = 0.9
    ):
        super().__init__()

        # state and reward normalization

        self.state_norm = ObservationNormalizer(dim_state)

        self.reward_norm = ScaleRewardNormalizer()
        self.register_buffer('discounted_return', tensor(0.))

        # actor

        self.actor = actor

        assert num_discrete_actions > 0 or num_continuous_actions > 0

        self.readout = Readout(
            dim_actor,
            num_discrete = num_discrete_actions,
            num_continuous = num_continuous_actions
        )

        self.actor_with_readout = Sequential(actor, self.readout)

        # critic

        self.critic = critic

        self.critic_ema = EMA(critic, beta = critic_ema_beta)

        # td related

        self.discount_factor = discount_factor

        # entropy related

        self.entropy_weight = entropy_weight

        # eligibility traces

        self.actor_trace = {name: torch.zeros_like(param) for name, param in self.actor_with_readout.named_parameters()}

        self.critic_trace = {name: torch.zeros_like(param) for name, param in self.critic.named_parameters()}

        self.eligibility_trace_decay = eligibility_trace_decay # lambda in paper

        # adaptive step related (obsgd)

        self.actor_kappa = actor_kappa
        self.critic_kappa = critic_kappa

        # learning rates

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        # sparse init

        self.init_sparsity = init_sparsity

        self.apply(self.init_)

    def init_(self, module):
        if not isinstance(module, Linear):
            return

        sparse_init_(module, sparsity = self.init_sparsity)

    def reset_trace_(self):
        for trace in self.actor_trace.values():
            trace.zero_()

        for trace in self.critic_trace.values():
            trace.zero_()

    @torch.no_grad()
    def update(
        self,
        state,
        action,
        next_state,
        reward,
        is_terminal = False
    ):
        reward = cast_tensor(reward)
        is_terminal = cast_tensor(is_terminal)

        state = self.state_norm(state, update = True)
        next_state = self.state_norm(next_state, update = False)

        # normalize the rewards

        normed_reward = self.reward_norm(reward, is_terminal = is_terminal, update = True)

        # get value prediction and grad

        with torch.enable_grad():
            value_pred = self.critic(state).mean()

            # actor grad

            action_logits = self.actor_with_readout(state)
            log_prob = self.readout.log_prob(action_logits, action).mean()

            actor_grads = torch.autograd.grad(log_prob, self.actor_with_readout.parameters())
            actor_grad = {name: grad for (name, _), grad in zip(self.actor_with_readout.named_parameters(), actor_grads)}

            # critic grad

            critic_grads = torch.autograd.grad(value_pred, self.critic.parameters())
            value_grad = {name: grad for (name, _), grad in zip(self.critic.named_parameters(), critic_grads)}

        next_value_pred = self.critic_ema(next_state).mean()

        assert isinstance(reward, (int, float)) or reward.numel() == 1

        td_error = normed_reward + next_value_pred * self.discount_factor * (~is_terminal).float() - value_pred

        # update actor eligibility trace

        decay = self.eligibility_trace_decay * self.discount_factor

        for name, trace in self.actor_trace.items():
            trace.mul_(decay).add_(actor_grad[name])

        # update critic eligibility trace

        for name, trace in self.critic_trace.items():
            trace.mul_(decay).add_(value_grad[name])

        # overstepping-bounds gradient descent related

        actor_trace_l1norm = sum([trace.norm(p = 1) for trace in self.actor_trace.values()])
        critic_trace_l1norm = sum([trace.norm(p = 1) for trace in self.critic_trace.values()])

        td_error_factor = td_error.abs().clamp(min = 1.) # if td is small, should not influence

        scale_actor = (self.actor_kappa * td_error_factor * actor_trace_l1norm).reciprocal().clamp(max = 1.)
        scale_critic = (self.critic_kappa * td_error_factor * critic_trace_l1norm).reciprocal().clamp(max = 1.)

        # update actor params

        for name, param in self.actor_with_readout.named_parameters():
            actor_trace = self.actor_trace[name]
            update = td_error  * actor_trace * scale_actor * self.actor_lr
            param.data.add_(update)

        # update critic params

        for name, param in self.critic.named_parameters():
            critic_trace = self.critic_trace[name]
            update = td_error * critic_trace * scale_critic * self.critic_lr
            param.data.add_(update)

        self.critic_ema.update()

    @torch.no_grad()
    def forward_value(self, state):
        return self.critic(state)

    @torch.no_grad()
    def forward_action(self, state):
        state = self.state_norm(state, update = False)
        return self.actor_with_readout(state)

    def sample_action(self, action_dist):
        return self.readout.sample(action_dist)

    @torch.no_grad()
    def forward(self, state):
        return self.forward_action(state)

# streaming Q variant

class StreamingQLambda(Module):
    def __init__(
        self,
        network: Module
    ):
        super().__init__()
        self.network = network

        # sparse init

        self.apply(self.init_)

    def init_(self, module):
        if not isinstance(module, Linear):
            return

        sparse_init_(module)

    def update(
        self,
        state,
        action,
        next_state,
        rewards
    ):
        raise NotImplementedError

    def forward(
        self,
        state
    ):
        raise NotImplementedError
