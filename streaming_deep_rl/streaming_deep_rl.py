from __future__ import annotations
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn, tensor, is_tensor, atan2, sqrt
from torch.nn import Module, Linear, Sequential

from einops import einsum, rearrange, repeat, reduce, pack, unpack

from discrete_continuous_embed_readout import Readout

from torch_einops_utils import tree_map_tensor

from ema_pytorch import EMA
from hl_gauss_pytorch import HLGaussLayer

from streaming_deep_rl.buffer_dict import BufferDict

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

    n = int(weight.numel() * sparsity)
    flat_weight = weight.view(-1)
    sparse_indices = torch.randperm(flat_weight.numel())[:n]
    nn.init.zeros_(flat_weight[sparse_indices])

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
        self.register_buffer('running_estimate_p', torch.zeros(dim))

    def reset_step(self):
        self.step.fill_(1)

    @property
    def time(self):
        return self.step / self.time_dilate_factor

    @property
    def variance(self):
        p = self.running_estimate_p

        if self.step.item() <= 1:
            return torch.ones_like(p)

        return (p / (self.time - 1. / self.time_dilate_factor)).clamp(min = self.eps)

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

        self.register_buffer('step', tensor(1))
        self.register_buffer('running_reward', tensor(0.))
        self.register_buffer('running_estimate_p', tensor(0.))

    def reset_step(self):
        self.step.fill_(1)

    @property
    def time(self):
        return self.step / self.time_dilate_factor

    @property
    def variance(self):
        p = self.running_estimate_p

        if self.step.item() <= 1:
            return torch.ones_like(p)

        return (p / (self.time - 1. / self.time_dilate_factor)).clamp(min = self.eps)

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
        adaptive = False,
        rms_beta = 0.99,
        eps = 1e-5,
        actor_use_ema = False,
        actor_ema_beta = 0.7,
        critic_ema_beta = 0.7,
        entropy_weight = 0.01,
        init_sparsity = 0.9,
        dim_critic = 128,
        val_min = -3.,
        val_max = 3.,
        num_bins = 64,
        sigma = 1.
    ):
        super().__init__()

        # hl-gauss

        self.hl_gauss_layer = HLGaussLayer(
            dim = dim_critic,
            hl_gauss_loss = dict(
                min_value = val_min,
                max_value = val_max,
                num_bins = num_bins,
                sigma = sigma
            )
        )

        # state and reward normalization

        self.state_norm = ObservationNormalizer(dim_state)

        self.reward_norm = ScaleRewardNormalizer()

        # actor

        self.actor = actor

        assert num_discrete_actions > 0 or num_continuous_actions > 0

        self.readout = Readout(
            dim_actor,
            num_discrete = num_discrete_actions,
            num_continuous = num_continuous_actions
        )

        self.actor_with_readout = Sequential(actor, self.readout)

        self.actor_use_ema = actor_use_ema
        self.actor_with_readout_ema = EMA(self.actor_with_readout, beta = actor_ema_beta) if actor_use_ema else None

        # critic

        self.critic = critic

        self.critic_full = Sequential(critic, self.hl_gauss_layer)

        self.critic_ema = EMA(self.critic_full, beta = critic_ema_beta)

        # td related

        self.discount_factor = discount_factor

        # entropy related

        self.entropy_weight = entropy_weight

        # eligibility traces

        self.actor_trace = BufferDict({name: torch.zeros_like(param) for name, param in self.actor_with_readout.named_parameters()})
        self.critic_trace = BufferDict({name: torch.zeros_like(param) for name, param in self.critic_full.named_parameters()})

        self.adaptive = adaptive
        self.rms_beta = rms_beta
        self.eps = eps

        if adaptive:
            self.actor_v = BufferDict({name: torch.zeros_like(param) for name, param in self.actor_with_readout.named_parameters()})
            self.critic_v = BufferDict({name: torch.zeros_like(param) for name, param in self.critic_full.named_parameters()})

        self.eligibility_trace_decay = eligibility_trace_decay # lambda in paper

        # adaptive step related (obgd)

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

        if self.adaptive:
            for v in self.actor_v.values():
                v.zero_()

            for v in self.critic_v.values():
                v.zero_()

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

        # all gradient related

        # all gradient related

        with torch.enable_grad():
            # critic pred and value loss (reformatted as categorical HL-Gauss)

            embed = self.critic(state)
            value_pred = self.hl_gauss_layer(embed)

            next_value_pred = self.critic_ema(next_state)
            td_target = (normed_reward + next_value_pred * self.discount_factor * (~is_terminal).float()).detach()
            
            value_loss = self.hl_gauss_layer(embed, td_target)

            critic_params = list(self.critic_full.parameters())
            critic_loss_grads = torch.autograd.grad(value_loss, critic_params)

            td_error = td_target - value_pred
            td_error_sign = td_error.detach().sign()

            # The ObGD optimizer requires a trace of `nabla_w v(s)`. 
            # We construct a pseudo-gradient such that when ObGD multiplies by delta,
            # it recovers exact `-grad(value_loss)` for the current step.
            # note: Gemini Pro 3.1 thought of this pseudo-gradient logic

            safe_delta = torch.where(td_error.detach() >= 0, 1.0, -1.0) * td_error.detach().abs().clamp(min=1e-6)

            value_grad = {
                name: -grad / safe_delta
                for (name, _), grad in zip(self.critic_full.named_parameters(), critic_loss_grads)
            }

            # actor grad with entropy regularization (Appendix E)

            action_logits = self.actor_with_readout(state)
            log_prob = self.readout.log_prob(action_logits, action).mean()
            entropy = self.readout.entropy(action_logits).mean()

            total_actor_loss = log_prob + self.entropy_weight * td_error_sign * entropy

            actor_params = list(self.actor_with_readout.parameters())
            actor_grads = torch.autograd.grad(total_actor_loss, actor_params)
            actor_grad = {name: grad for (name, _), grad in zip(self.actor_with_readout.named_parameters(), actor_grads)}

        td_error = td_error.detach().squeeze()
        td_error_sign = td_error_sign.detach().squeeze()
        value_pred = value_pred.detach()
        next_value_pred = next_value_pred.detach()

        # update actor eligibility trace

        decay = self.eligibility_trace_decay * self.discount_factor

        for name, trace in self.actor_trace.items():
            trace.mul_(decay).add_(actor_grad[name])

        # update critic eligibility trace

        for name, trace in self.critic_trace.items():
            trace.mul_(decay).add_(value_grad[name])

        # overstepping-bounds gradient descent related

        td_error_factor = td_error.abs().clamp(min = 1.)

        actor_grad_norm = torch.stack([g.norm(p=1) for g in actor_grad.values()]).mean()
        critic_grad_norm = torch.stack([g.norm(p=1) for g in value_grad.values()]).mean()

        def update_params(params, traces, vs, kappa, lr):
            if self.adaptive:
                for name, v in vs.items():
                    v.mul_(self.rms_beta).add_((td_error * traces[name]) ** 2, alpha = 1 - self.rms_beta)

                adapted_traces = {name: trace / (vs[name] + self.eps).sqrt() for name, trace in traces.items()}
            else:
                adapted_traces = traces

            trace_norms = []
            scales = []

            for name, param in params.named_parameters():
                trace = adapted_traces[name]
                
                # per-parameter effective step size and scaling
                param_trace_norm = trace.norm(p=1)
                
                param_scale = (kappa * td_error_factor * param_trace_norm).reciprocal().clamp(max=1.)
                param_update = td_error * trace * param_scale * lr
                
                param.data.add_(param_update)
                
                trace_norms.append(param_trace_norm)
                scales.append(param_scale)

            mean_trace_norm = torch.stack(trace_norms).mean()
            mean_scale = torch.stack(scales).mean()
            return mean_trace_norm, mean_scale


        actor_trace_norm, actor_scale = update_params(
            self.actor_with_readout,
            self.actor_trace,
            self.actor_v if self.adaptive else None,
            self.actor_kappa,
            self.actor_lr
        )

        critic_trace_norm, critic_scale = update_params(
            self.critic_full,
            self.critic_trace,
            self.critic_v if self.adaptive else None,
            self.critic_kappa,
            self.critic_lr
        )

        if self.actor_use_ema:
            self.actor_with_readout_ema.update()

        self.critic_ema.update()

        return dict(
            td_error = td_error.item(),
            value_pred = value_pred.item(),
            actor_grad_norm = actor_grad_norm.item(),
            critic_grad_norm = critic_grad_norm.item(),
            actor_trace_norm = actor_trace_norm.item(),
            critic_trace_norm = critic_trace_norm.item(),
            actor_scale = actor_scale.item(),
            critic_scale = critic_scale.item()
        )

    @torch.no_grad()
    def forward_value(self, state):
        return self.critic_full(state)

    @torch.no_grad()
    def forward_action(self, state, use_ema = True):
        state = self.state_norm(state, update = False)

        actor = self.actor_with_readout_ema if use_ema and self.actor_use_ema else self.actor_with_readout

        return actor(state)

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
