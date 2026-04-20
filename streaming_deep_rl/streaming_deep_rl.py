from __future__ import annotations
import math
from collections import deque
from typing import Callable, NamedTuple

import torch
import torch.nn.functional as F
from torch import nn, tensor, is_tensor, cat, stack
from torch.nn import Module, Linear, Sequential

from einops import reduce

from discrete_continuous_embed_readout import Readout, Embed

from torch_einops_utils import tree_map_tensor, pack_with_inverse

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

def l2norm(t):
    return F.normalize(t, p = 2, dim = -1)

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

# orthogonal projection for SPR in streaming setting
# https://arxiv.org/abs/2602.09396

def orthogonal_project(x, y):
    x, inverse = pack_with_inverse(x, '*')
    y, _ = pack_with_inverse(y, '*')

    dtype = x.dtype
    x, y = x.double(), y.double()
    unit = l2norm(y)

    parallel = (x * unit).sum(dim = -1, keepdim = True) * unit
    orthogonal = x - parallel

    return inverse(orthogonal).to(dtype)

class SelfPredictRepr(Module):
    def __init__(
        self,
        dim_embed,
        num_discrete_actions = 0,
        num_continuous_actions = 0,
        dim_predict = None
    ):
        super().__init__()
        assert num_discrete_actions > 0 or num_continuous_actions > 0

        dim_predict = default(dim_predict, dim_embed)

        self.to_action_embed = Embed(dim_embed, num_discrete = num_discrete_actions, num_continuous = num_continuous_actions)

        self.to_projection = nn.Sequential(
            nn.RMSNorm(dim_embed),
            nn.Linear(dim_embed, dim_predict, bias = False)
        )

        self.to_prediction = nn.Sequential(
            nn.Linear(dim_predict + dim_embed, dim_predict, bias = False)
        )

    def forward(
        self,
        state_embed,
        actions,
        next_state_embed
    ):
        projected = self.to_projection(state_embed)

        with torch.no_grad():
            next_projected = self.to_projection(next_state_embed.detach())

        action_embed = self.to_action_embed(actions)

        predicted = self.to_prediction(cat((projected, action_embed), dim = -1))

        # cosine sim loss

        return F.mse_loss(l2norm(predicted), l2norm(next_projected))

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
        actor_lr = 1.0,
        critic_lr = 1.0,
        adaptive = True,
        adam_beta1 = 0.9,
        adam_beta2 = 0.999,
        eps = 1e-5,
        actor_use_ema = False,
        actor_ema_beta = 0.7,
        critic_ema_beta = 0.7,
        self_predict_repr = False,
        entropy_weight = 0.01,
        init_sparsity = 0.9,
        dim_critic = 128,
        val_min = -3.,
        val_max = 3.,
        num_bins = 64,
        l2_weight_decay = 0.,
        l1_weight_decay = 0.,
        cautious_wd = False,
        wd_towards_init = False,
        delay_steps = 1,
        enable_pilar = False,
        pilar_mixing_param = 0.5,
        use_critic_ema = True,
        use_minto = False,
        use_delightful_pg = False,
        delightful_eta = 1.0
    ):
        super().__init__()
        assert delay_steps > 0, 'delay steps must be greater than 0'

        # delay buffer and pilar

        self.delay_steps = delay_steps
        self.delay_buffer = deque()

        self.enable_pilar = enable_pilar
        self.pilar_mixing_param = pilar_mixing_param

        # minto (use online network if you can)

        assert not (use_minto and not use_critic_ema), 'minto can only be used with critic ema'
        self.use_minto = use_minto

        # delightful pg

        self.use_delightful_pg = use_delightful_pg
        self.delightful_eta = delightful_eta

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

        self.reward_norm = ScaleRewardNormalizer(discount_factor = discount_factor)

        # self-predictive representation
        # Schwarzer et al. https://arxiv.org/abs/2007.05929

        self.self_predict_repr = self_predict_repr

        if self_predict_repr:

            actor_use_ema |= True

            self.actor_self_predict_repr = SelfPredictRepr(
                dim_actor,
                num_discrete_actions = num_discrete_actions,
                num_continuous_actions = num_continuous_actions,
            )

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

        self.use_critic_ema = use_critic_ema
        self.critic_ema = EMA(self.critic_full, beta = critic_ema_beta) if self.use_critic_ema else None

        # td related

        self.discount_factor = discount_factor

        # entropy related

        self.entropy_weight = entropy_weight

        # eligibility traces

        self.actor_trace = BufferDict({name: torch.zeros_like(param) for name, param in self.actor_with_readout.named_parameters()})
        self.critic_trace = BufferDict({name: torch.zeros_like(param) for name, param in self.critic_full.named_parameters()})

        self.adaptive = adaptive
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.eps = eps

        if adaptive:
            self.register_buffer('step', tensor(0))
            self.actor_m = BufferDict({name: torch.zeros_like(param) for name, param in self.actor_with_readout.named_parameters()})
            self.critic_m = BufferDict({name: torch.zeros_like(param) for name, param in self.critic_full.named_parameters()})

            self.actor_v = BufferDict({name: torch.zeros_like(param) for name, param in self.actor_with_readout.named_parameters()})
            self.critic_v = BufferDict({name: torch.zeros_like(param) for name, param in self.critic_full.named_parameters()})

        self.eligibility_trace_decay = eligibility_trace_decay # lambda in paper

        # adaptive step related (obgd)

        self.actor_kappa = actor_kappa
        self.critic_kappa = critic_kappa

        # learning rates

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        self.has_l2_weight_decay = l2_weight_decay > 0.
        self.has_l1_weight_decay = l1_weight_decay > 0.

        assert not (self.has_l2_weight_decay and self.has_l1_weight_decay), 'l2 and l1 weight decay are mutually exclusive'

        self.has_any_wd = self.has_l2_weight_decay or self.has_l1_weight_decay

        self.l2_weight_decay = l2_weight_decay
        self.l1_weight_decay = l1_weight_decay
        self.cautious_wd = cautious_wd
        self.wd_towards_init = wd_towards_init

        # sparse init

        self.init_sparsity = init_sparsity

        self.apply(self.init_)

        self.init_params = None
        if self.has_any_wd and wd_towards_init:
            self.init_params = nn.ModuleDict({
                'actor': BufferDict({name: param.detach().clone() for name, param in self.actor_with_readout.named_parameters()}),
                'critic': BufferDict({name: param.detach().clone() for name, param in self.critic_full.named_parameters()})
            })

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
            self.step.zero_()

            for m in self.actor_m.values():
                m.zero_()

            for m in self.critic_m.values():
                m.zero_()

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
        is_terminal = False,
        delay_steps = None,
        drain = False
    ):
        reward = cast_tensor(reward)
        is_terminal = cast_tensor(is_terminal)

        state = self.state_norm(state, update = True)
        next_state = self.state_norm(next_state, update = False)

        normed_reward = self.reward_norm(reward, is_terminal = is_terminal, update = True)

        self.delay_buffer.append((state, action, normed_reward, next_state, is_terminal))

        # process all pending transitions in the buffer

        delay_steps = default(delay_steps, self.delay_steps)

        should_learn = len(self.delay_buffer) >= delay_steps or is_terminal.item() or drain

        if not should_learn:
            return UpdateMetrics(0., 0., 0., 0., 0., 0., 0., 0.)

        # process all pending transitions in the buffer

        metrics = None

        while len(self.delay_buffer) > 0:
            metrics = self._learn_step(list(self.delay_buffer))
            self.delay_buffer.popleft()

            if not is_terminal.item() and not drain:
                break

        return metrics

    def _learn_step(self, buffer_slice):
        """single learning step for the oldest transition in buffer_slice, using n-step return"""

        oldest_state, oldest_action, oldest_reward, oldest_next_state, oldest_is_term = buffer_slice[0]
        last_next_state = buffer_slice[-1][3]

        # n-step discounted return

        n_step_return = 0.
        discount = 1.0

        for _, _, reward, _, is_term in buffer_slice:
            n_step_return = n_step_return + discount * reward

            if is_term.item():
                discount = 0.
                break

            discount *= self.discount_factor

        # all gradient related

        with torch.enable_grad():

            # critic

            embed = self.critic(oldest_state)
            value_pred = self.hl_gauss_layer(embed)

            # getting the bootstrapped value

            def get_next_value_pred(state):
                next_value = self.critic_ema(state) if self.use_critic_ema else self.critic_full(state)

                if not self.use_minto:
                    return next_value

                # Hendawy et al. https://openreview.net/forum?id=rFLuaG9Yq6

                online_next_value = self.critic_full(state)
                return torch.min(next_value, online_next_value)

            # n-step td target

            if discount > 0.:
                td_target = (n_step_return + discount * get_next_value_pred(last_next_state)).detach()
            else:
                td_target = n_step_return.detach()

            # pilar - average of 1-step and n-step td targets
            # https://arxiv.org/abs/2402.03903

            if self.enable_pilar and len(buffer_slice) > 1:
                if oldest_is_term.item():
                    td_target_1 = oldest_reward.detach()
                else:
                    td_target_1 = (oldest_reward + self.discount_factor * get_next_value_pred(oldest_next_state)).detach()

                td_target = torch.lerp(td_target_1, td_target, self.pilar_mixing_param)

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

            actor_loss = log_prob

            actor_params = list(self.actor_with_readout.parameters())

            # policy gradient goes into the eligibility trace

            actor_loss_grads = torch.autograd.grad(actor_loss, actor_params, retain_graph = True)
            actor_grad = {name: grad for (name, _), grad in zip(self.actor_with_readout.named_parameters(), actor_loss_grads)}

            # get the SPR loss

            actor_spr_grad = None
            if self.self_predict_repr:
                actor_embed = self.actor(oldest_state)

                if self.actor_use_ema:
                    actor_next_embed = self.actor_with_readout_ema.ema_model[0](last_next_state)
                else:
                    actor_next_embed = self.actor(last_next_state)

                spr_loss = self.actor_self_predict_repr(actor_embed, oldest_action, actor_next_embed)

                spr_params = list(self.actor_self_predict_repr.parameters())
                spr_grads_tup = torch.autograd.grad(spr_loss, actor_params + spr_params, retain_graph = True, allow_unused = True)
                spr_grads_tup = tuple(default(g, torch.zeros_like(p)) for g, p in zip(spr_grads_tup, actor_params + spr_params))

                actor_spr_grad = {name: grad for (name, _), grad in zip(self.actor_with_readout.named_parameters(), spr_grads_tup[:len(actor_params)])}
                spr_only_grad = spr_grads_tup[len(actor_params):]

                # Update SPR networks via SGD
                for param, grad in zip(spr_params, spr_only_grad):
                    param.data.add_(grad, alpha = -self.actor_lr)

            # entropy gradient decoupled from td_error

            entropy_grads_tup = torch.autograd.grad(entropy, actor_params, retain_graph = True)
            entropy_grad = {name: grad for (name, _), grad in zip(self.actor_with_readout.named_parameters(), entropy_grads_tup)}

        td_error = td_error.detach().squeeze()
        td_error_sign = td_error_sign.detach().squeeze()
        value_pred = value_pred.detach()

        # delightful pg

        gate = 1.
        if self.use_delightful_pg:
            surprisal = -log_prob.detach()
            delight = td_error * surprisal
            gate = (delight / self.delightful_eta).sigmoid()

        # update eligibility traces

        decay = self.eligibility_trace_decay * self.discount_factor

        for name, trace in self.actor_trace.items():
            trace.mul_(decay).add_(actor_grad[name] * gate)

        for name, trace in self.critic_trace.items():
            trace.mul_(decay).add_(value_grad[name])

        # overstepping-bounds gradient descent

        td_error_factor = td_error.abs().clamp(min = 1.)

        actor_grad_norm = torch.stack([g.norm(p=1) for g in actor_grad.values()]).mean()
        critic_grad_norm = torch.stack([g.norm(p=1) for g in value_grad.values()]).mean()

        if self.adaptive:
            self.step.add_(1)
            step = self.step.item()
            bias_correction1 = 1. - self.adam_beta1 ** step
            bias_correction2 = 1. - self.adam_beta2 ** step

        def update_params(params, init_params, traces, ms, vs, kappa, lr, l2_wd, has_l2_wd, l1_wd, has_l1_wd, has_any_wd, cautious_wd, wd_towards_init, aux_grads = None, spr_grads = None):
            if self.adaptive:
                # update first and second moments of semi-gradients

                for name, trace in traces.items():
                    m, v = ms[name], vs[name]
                    grad = td_error * trace

                    m.lerp_(grad, 1. - self.adam_beta1)
                    v.mul_(self.adam_beta2).add_(grad ** 2, alpha = 1. - self.adam_beta2)

                # adam update direction + obgd bound (Algorithm 11)

                adapted_grads = {}
                trace_sum = 0.0

                for name, trace in traces.items():
                    v_hat = vs[name] / bias_correction2
                    denom = v_hat.sqrt() + self.eps

                    m_hat = ms[name] / bias_correction1
                    adapted_grads[name] = m_hat / denom

                    trace_sum += (trace / denom).abs().sum()
            else:
                adapted_grads = {name: td_error * trace for name, trace in traces.items()}

                trace_sum = 0.0
                for trace in traces.values():
                    trace_sum += trace.abs().sum()

            global_scale = (kappa * td_error_factor * trace_sum).reciprocal().clamp(max = 1.)

            for name, param in params.named_parameters():
                update = adapted_grads[name] * global_scale * lr

                if exists(aux_grads):
                    update = update + aux_grads[name] * self.entropy_weight * global_scale * lr

                if exists(spr_grads) and name in spr_grads:
                    spr_update = -spr_grads[name] * lr
                    spr_update = orthogonal_project(spr_update, update)
                    update = update + spr_update

                # add gradient update first

                param.data.add_(update)

                # weight decay (applied after main update for proper L1 sparsity)

                if has_any_wd:
                    target = init_params[name] if wd_towards_init else torch.zeros_like(param)

                    wd_mask = 1.
                    if cautious_wd:
                        wd_mask = (update * (target - param) > 0).float()

                    # l2

                    if has_l2_wd:
                        param.data.lerp_(target, lr * l2_wd * wd_mask * global_scale)

                    # l1 with soft thresholding

                    if has_l1_wd:
                        l1_amount = lr * l1_wd * wd_mask * global_scale
                        diff = param.data - target
                        shrunk_diff = diff.sign() * (diff.abs() - l1_amount).relu()
                        param.data.copy_(target + shrunk_diff)

            return trace_sum, global_scale

        actor_trace_norm, actor_scale = update_params(
            self.actor_with_readout,
            self.init_params['actor'] if exists(self.init_params) else None,
            self.actor_trace,
            self.actor_m if self.adaptive else None,
            self.actor_v if self.adaptive else None,
            self.actor_kappa,
            self.actor_lr,
            self.l2_weight_decay,
            self.has_l2_weight_decay,
            self.l1_weight_decay,
            self.has_l1_weight_decay,
            self.has_any_wd,
            self.cautious_wd,
            self.wd_towards_init,
            entropy_grad,
            actor_spr_grad
        )

        critic_trace_norm, critic_scale = update_params(
            self.critic_full,
            self.init_params['critic'] if exists(self.init_params) else None,
            self.critic_trace,
            self.critic_m if self.adaptive else None,
            self.critic_v if self.adaptive else None,
            self.critic_kappa,
            self.critic_lr,
            self.l2_weight_decay,
            self.has_l2_weight_decay,
            self.l1_weight_decay,
            self.has_l1_weight_decay,
            self.has_any_wd,
            self.cautious_wd,
            self.wd_towards_init,
            None,
            None
        )

        if self.actor_use_ema:
            self.actor_with_readout_ema.update()

        if self.use_critic_ema:
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
