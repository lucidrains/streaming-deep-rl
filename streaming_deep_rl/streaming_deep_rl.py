from __future__ import annotations
import math
from collections import deque
from typing import Callable, NamedTuple

import torch
import torch.nn.functional as F
from torch import nn, tensor, is_tensor
from torch.nn import Module, Linear, Sequential

from einops import reduce

from discrete_continuous_embed_readout import Readout

from torch_einops_utils import tree_map_tensor

from ema_pytorch import EMA

from hl_gauss_pytorch import HLGaussLayer

from streaming_deep_rl.buffer_dict import BufferDict

# helpers

def exists(v):
    return v is not None

def divisible_by(num, den):
    return (num % den) == 0

def default(v, d):
    return v if exists(v) else d

def cast_tensor(t):
    return tensor(t) if not is_tensor(t) else t

def to_device(tree, device):
    return tree_map_tensor(lambda t: t.to(device), tree)

# regenerative regularization

class RegenerativeRegularization(Module):
    def __init__(
        self,
        networks: list[Module],
        rate: float,
        every: int
    ):
        super().__init__()
        self.networks = networks
        self.rate = rate
        self.every = every
        self.register_buffer('step', tensor(0))

        self.init_params = nn.ModuleList([
            BufferDict({name: param.detach().clone() for name, param in network.named_parameters()})
            for network in networks
        ])

    def forward(self):
        self.step.add_(1)

        if self.rate <= 0. or not divisible_by(self.step.item(), self.every):
            return

        for network, init_params in zip(self.networks, self.init_params):
            for name, param in network.named_parameters():
                param.data.lerp_(init_params[name], self.rate)

# return types

class UpdateMetrics(NamedTuple):
    td_error: float
    value_pred: float
    actor_grad_norm: float
    critic_grad_norm: float
    actor_trace_norm: float
    critic_trace_norm: float
    actor_scale: float
    critic_scale: float

# initialization

@torch.no_grad()
def sparse_init_(
    l: Linear,
    sparsity = 0.9
):
    """
    Algorithm 1
    """
    weight, bias = l.weight, l.bias
    device = weight.device

    fan_out, fan_in = weight.shape

    value = fan_in ** -0.5
    nn.init.uniform_(weight, -value, value)

    assert 0. <= sparsity <= 1.

    num_zeros = int(math.ceil(sparsity * fan_in))

    random_scores = torch.randn(fan_out, fan_in, device = device)
    zero_indices = random_scores.argsort(dim = -1)[:, :num_zeros]

    weight.scatter_(1, zero_indices, 0.)

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
        regen_reg_rate = 0.,
        regen_reg_every = 1,
        delay_steps = 1
    ):
        super().__init__()
        assert delay_steps > 0, 'delay steps must be greater than 0'

        self.delay_steps = delay_steps
        self.delay_buffer = deque()

        # quick validate

        with torch.no_grad():
            mock_state = torch.randn(dim_state)
            actor_embed = actor(mock_state)
            critic_embed = critic(mock_state)
            assert actor_embed.shape[-1] == dim_actor
            assert critic_embed.shape[-1] == dim_critic

        # hl-gauss

        self.hl_gauss_layer = HLGaussLayer(
            dim = dim_critic,
            hl_gauss_loss = dict(
                min_value = val_min,
                max_value = val_max,
                num_bins = num_bins,
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

        # regenerative regularization

        self.regen_reg = RegenerativeRegularization(
            networks = [self.actor_with_readout, self.critic_full],
            rate = regen_reg_rate,
            every = regen_reg_every
        )

        # set actor and critic lambda if eligibility trace is used

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

        normed_reward = self.reward_norm(reward, is_terminal = is_terminal, update = True)

        self.delay_buffer.append((state, action, normed_reward, next_state, is_terminal))

        # regenerative regularization

        self.regen_reg()

        # process all pending transitions in the buffer

        should_learn = len(self.delay_buffer) == self.delay_steps or is_terminal.item()

        if not should_learn:
            return UpdateMetrics(0., 0., 0., 0., 0., 0., 0., 0.)

        # process all pending transitions in the buffer

        metrics = None

        while len(self.delay_buffer) > 0:
            metrics = self._learn_step(list(self.delay_buffer))
            self.delay_buffer.popleft()

            if not is_terminal.item():
                break

        return metrics

    def _learn_step(self, buffer_slice):
        """single learning step for the oldest transition in buffer_slice, using n-step return"""

        oldest_state, oldest_action, _, _, _ = buffer_slice[0]
        _, _, _, last_next_state, last_is_term = buffer_slice[-1]

        # n-step discounted return

        n_step_return = 0.
        discount = 1.0

        for _, _, reward, _, is_term in buffer_slice:
            n_step_return = n_step_return + discount * reward
            discount *= self.discount_factor
            if is_term.item():
                break

        # all gradient related

        with torch.enable_grad():

            # critic

            embed = self.critic(oldest_state)
            value_pred = self.hl_gauss_layer(embed)

            if not last_is_term.item():
                next_value_pred = self.critic_ema(last_next_state)
                td_target = (n_step_return + discount * next_value_pred).detach()
            else:
                next_value_pred = torch.zeros_like(value_pred)
                td_target = n_step_return.detach()

            value_loss = self.hl_gauss_layer(embed, td_target)

            critic_params = list(self.critic_full.parameters())
            critic_loss_grads = torch.autograd.grad(value_loss, critic_params)

            td_error = td_target - value_pred
            td_error_sign = td_error.detach().sign()

            # ObGD pseudo-gradient

            safe_delta = torch.where(td_error.detach() >= 0, 1.0, -1.0) * td_error.detach().abs().clamp(min = 1e-6)

            value_grad = {
                name: -grad / safe_delta
                for (name, _), grad in zip(self.critic_full.named_parameters(), critic_loss_grads)
            }

            # actor with entropy regularization

            action_logits = self.actor_with_readout(oldest_state)
            log_prob = self.readout.log_prob(action_logits, oldest_action).mean()
            entropy = self.readout.entropy(action_logits).mean()

            total_actor_loss = log_prob + self.entropy_weight * td_error_sign * entropy

            actor_params = list(self.actor_with_readout.parameters())
            actor_grads = torch.autograd.grad(total_actor_loss, actor_params)
            actor_grad = {name: grad for (name, _), grad in zip(self.actor_with_readout.named_parameters(), actor_grads)}

        td_error = td_error.detach().squeeze()
        td_error_sign = td_error_sign.detach().squeeze()
        value_pred = value_pred.detach()

        # update eligibility traces

        decay = self.eligibility_trace_decay * self.discount_factor

        for name, trace in self.actor_trace.items():
            trace.mul_(decay).add_(actor_grad[name])

        for name, trace in self.critic_trace.items():
            trace.mul_(decay).add_(value_grad[name])

        # overstepping-bounds gradient descent

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

            trace_sum = 0.0
            for name, trace in adapted_traces.items():
                trace_sum += trace.abs().sum()

            global_scale = (kappa * td_error_factor * trace_sum).reciprocal().clamp(max = 1.)

            for name, param in params.named_parameters():
                update = td_error * adapted_traces[name] * global_scale * lr
                param.data.add_(update)

            return trace_sum, global_scale

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

        return UpdateMetrics(
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

    def sample_action(self, action_dist):
        return self.readout.sample(action_dist)

    @torch.no_grad()
    def forward_action(
        self,
        state,
        use_ema = True,
        sample = False
    ):
        state = self.state_norm(state, update = False)

        actor = self.actor_with_readout_ema if use_ema and self.actor_use_ema else self.actor_with_readout

        action_dist = actor(state)

        if not sample:
            return action_dist

        action = self.sample_action(action_dist)
        return action, action_dist

    @torch.no_grad()
    def forward(
        self,
        state,
        sample = False
    ):
        return self.forward_action(state, sample = sample)

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
