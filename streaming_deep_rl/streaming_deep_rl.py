from __future__ import annotations
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn, tensor, atan2, sqrt
from torch.nn import Module, ModuleList, Linear, Sequential, ParameterDict

from torch.optim.optimizer import Optimizer

from torch.func import functional_call, vmap, grad, grad_and_value

from einops import einsum, rearrange, repeat, reduce, pack, unpack

from hl_gauss_pytorch import HLGaussLayer

from discrete_continuous_embed_readout import Readout

from torch_einops_utils import tree_map_tensor

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

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

# classes

# streaming AC variant

class StreamingACLambda(Module):
    def __init__(
        self,
        *,
        actor: Module,
        critic: Module,
        dim_actor,
        dim_critic,
        num_discrete_actions = 0,
        num_continuous_actions = 0,
        discount_factor = 0.999,
        eligibility_trace_decay = 0.8,
        actor_kappa = 3.,
        critic_kappa = 2.,
        actor_lr = 1e-4,
        critic_lr = 1e-4,
        value_min = -5.,
        value_max = 5.,
        num_critic_bins = 32,
        entropy_weight = 0.01
    ):
        super().__init__()

        # actor

        self.actor = actor

        assert num_discrete_actions > 0 or num_continuous_actions > 0

        self.readout = Readout(
            dim_actor,
            num_discrete = num_discrete_actions,
            num_continuous = num_continuous_actions
        )

        actor_with_readout = Sequential(actor, self.readout)
        self.actor_params = dict(actor_with_readout.named_parameters())

        # critic

        self.critic = critic
        hl_gauss_layer = HLGaussLayer(dim_critic, hl_gauss_loss = dict(
            min_value = value_min,
            max_value = value_max,
            num_bins = num_critic_bins
        ))

        critic_with_hl_gauss = Sequential(critic, hl_gauss_layer)
        self.critic_params = dict(critic_with_hl_gauss.named_parameters())

        # state -> actions

        def actor_forward(params, inputs):
            return functional_call(actor_with_readout, params, inputs)

        self.actor_forward = actor_forward

        def actor_forward_log_prob(params, inputs):
            states, actions = inputs
            action_logits = functional_call(actor_with_readout, params, states)
            log_prob = self.readout.log_prob(action_logits, actions)
            return log_prob.mean()

        self.actor_grad = grad(actor_forward_log_prob)

        # state -> value

        def critic_forward(params, inputs):
            return functional_call(critic_with_hl_gauss, params, inputs)

        self.critic_forward = critic_forward

        self.critic_grad_and_value = grad_and_value(critic_forward)

        # td related

        self.discount_factor = discount_factor

        # entropy related

        self.entropy_weight = entropy_weight

        # eligibility traces

        self.actor_trace = {name: torch.zeros_like(param) for name, param in self.actor_params.items()}

        self.critic_trace = {name: torch.zeros_like(param) for name, param in self.critic_params.items()}

        self.eligibility_trace_decay = eligibility_trace_decay # lambda in paper

        # adaptive step related (obsgd)

        self.actor_kappa = actor_kappa
        self.critic_kappa = critic_kappa

        # learning rates

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        # sparse init

        self.apply(self.init_)

    def init_(self, module):
        if not isinstance(module, Linear):
            return

        sparse_init_(module)

    @torch.no_grad()
    def update(
        self,
        state,
        action,
        next_state,
        reward,
        is_terminal = tensor(False)
    ):
        value_grad, value_pred = self.critic_grad_and_value(self.critic_params, state)

        actor_grad = self.actor_grad(self.actor_params, (state, action))

        next_value_pred = self.forward_value(next_state)

        assert isinstance(reward, (int, float)) or reward.numel() == 1

        td_error = reward + next_value_pred * self.discount_factor * (~is_terminal).float() - value_pred

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

        scale_actor = 1. / (self.actor_kappa * td_error_factor * actor_trace_l1norm).clamp(min = 1.)
        scale_critic = 1. / (self.critic_kappa * td_error_factor * critic_trace_l1norm).clamp(min = 1.)

        # update actor params

        for name, param in self.actor_params.items():

            actor_trace = self.actor_trace[name]
            update = td_error * self.actor_lr * actor_trace * scale_actor
            param.add_(update)

        # update critic params

        for name, param in self.critic_params.items():
            critic_trace = self.critic_trace[name]
            update = td_error * self.critic_lr * critic_trace * scale_critic
            param.add_(update)

    @torch.no_grad()
    def forward_value(self, state):
        return self.critic_forward(self.critic_params, state)

    @torch.no_grad()
    def forward_action(self, state):
        return self.actor_forward(self.actor_params, state)

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
