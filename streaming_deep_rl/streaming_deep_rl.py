from __future__ import annotations
import math
from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn, tensor, is_tensor, cat, stack
from torch.nn import Module, Linear, Sequential

import einx
from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from discrete_continuous_embed_readout import Readout, Embed

from torch_einops_utils import tree_map_tensor, pack_with_inverse, masked_mean

from ema_pytorch import EMA

from hl_gauss_pytorch import HLGaussLayer

from x_mlps_pytorch import MLP

from streaming_deep_rl.buffer_dict import BufferDict

# helpers

def exists(v):
    return v is not None

def identity(t):
    return t

def first(t):
    return t[0]

def divisible_by(num, den):
    return (num % den) == 0

def default(v, d):
    return v if exists(v) else d

def no_grad_detach(fn):
    def inner(t):
        with torch.no_grad():
            return fn(t).detach()
    return inner

def cast_tensor(t, dtype = None, device = None):
    t = tensor(t, dtype = dtype) if not is_tensor(t) else t
    return t if not exists(device) else t.to(device)

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

def orthog_project(x, y):
    x, inverse_pack = pack_with_inverse(x, '*')
    y, _ = pack_with_inverse(y, '*')

    dtype = x.dtype

    if x.device.type != 'mps':
        x, y = x.double(), y.double()

    unit = l2norm(y)

    parallel = (x * unit).sum(dim = -1, keepdim = True) * unit
    orthog = x - parallel

    orthog = inverse_pack(orthog, '*')

    return orthog.to(dtype)

# Simplicial Embeddings
# Lavoie et al - https://arxiv.org/abs/2204.00616

class SEM(Module):
    def __init__(
        self,
        dim,
        temperature = 0.1,
        dim_simplex = 8,
        pre_layernorm = False
    ):
        super().__init__()
        assert divisible_by(dim, dim_simplex), f'{dim} must be divisible by {dim_simplex}'

        self.dim = dim
        self.dim_simplex = dim_simplex
        self.temperature = temperature

        self.norm = nn.LayerNorm(dim, bias = False) if pre_layernorm else nn.Identity()

    def forward(
        self,
        t
    ):
        t = self.norm(t)
        t = rearrange(t, '... (l v) -> ... l v', v = self.dim_simplex)
        t = (t / self.temperature).softmax(dim = -1)
        return rearrange(t, '... l v -> ... (l v)')


class SelfPredictRepr(Module):
    def __init__(
        self,
        dim_embed,
        num_discrete_actions = 0,
        num_continuous_actions = 0,
        dim_predict = None,
        target_embed_from_ema = True, # whether the target embed is from an EMA, or from the model itself (traditional vs sigreg from lejepa)
        sigreg_weight = 0.,
        dim_hidden_expand_factor = 4,
        use_sem = False,
        sem_dim_simplex = 8,
        sem_temperature = 0.1
    ):
        super().__init__()
        assert num_discrete_actions > 0 or num_continuous_actions > 0

        dim_predict = default(dim_predict, dim_embed)
        dim_mlp_hidden = int(dim_embed * dim_hidden_expand_factor)

        self.to_action_embed = Embed(dim_embed, num_discrete = num_discrete_actions, num_continuous = num_continuous_actions)

        self.target_embed_from_ema = target_embed_from_ema

        self.to_projection = nn.Sequential(
            nn.RMSNorm(dim_embed),
            MLP(dim_embed, dim_mlp_hidden, dim_predict)
        )

        self.to_prediction = MLP(dim_predict + dim_embed, dim_mlp_hidden, dim_predict)

        self.apply_sigreg = sigreg_weight > 0.
        self.sigreg_weight = sigreg_weight

        self.use_sem = use_sem
        self.sem = SEM(dim_predict, temperature = sem_temperature, dim_simplex = sem_dim_simplex) if use_sem else nn.Identity()

    def forward(
        self,
        state_embed,
        actions,
        next_state_embed
    ):
        projected = self.to_projection(state_embed)

        # maybe no

        decorator = no_grad_detach if self.target_embed_from_ema else identity

        next_projected = decorator(self.to_projection)(next_state_embed)

        if self.use_sem:
            projected = self.sem(projected)
            next_projected = self.sem(next_projected)

        action_embed = self.to_action_embed(actions)

        predicted = self.to_prediction(cat((projected, action_embed), dim = -1))

        # cosine sim loss

        loss = F.mse_loss(l2norm(predicted), l2norm(next_projected))

        if not self.apply_sigreg:
            return loss

        return loss + self.sigreg_weight * sigreg_loss(projected)

# sigreg from lejepa

def sigreg_loss(
    x,
    num_slices = 1024,
    domain = (-5, 5),
    num_knots = 17,
    mask = None
):
    # Randall Balestriero - https://arxiv.org/abs/2511.08544

    dim, device = x.shape[-1], x.device

    # slice sampling

    rand_projs = torch.randn((num_slices, dim), device = device)
    rand_projs = l2norm(rand_projs)

    # integration points

    t = torch.linspace(*domain, num_knots, device = device)

    # theoretical CF for N(0, 1) and Gauss. window

    exp_f = (-0.5 * t.square()).exp()

    # empirical CF

    x_t = einx.dot('... d, m d -> (...) m', x, rand_projs)

    x_t = einx.multiply('n m, t -> n m t', x_t, t)
    ecf = (1j * x_t).exp()

    mask_1d = rearrange(mask, '... -> (...)') if exists(mask) else None
    ecf = masked_mean(ecf, mask_1d, dim = 0)

    # weighted L2 distance

    err = ecf.sub(exp_f).abs().square().mul(exp_f)

    return torch.trapz(err, t, dim = -1).mean()

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
        actor_self_predict_repr = False,
        critic_self_predict_repr = False,
        spr_target_embed_from_ema = True,
        spr_sigreg_weight = 0.,
        spr_dim_hidden_expand_factor = 4,
        spr_use_sem = False,
        spr_sem_dim_simplex = 8,
        spr_sem_temperature = 0.1,
        spr_lr = 5e-2,
        spr_lr_param_update = 1.,
        spr_orth_beta = 0.9,
        entropy_weight = 0.01,
        init_sparsity = 0.9,
        dim_critic = 128,
        l2_weight_decay = 0.,
        l1_weight_decay = 0.,
        cautious_wd = False,
        wd_towards_init = False,
        use_critic_ema = True,
        use_minto = False,
        use_delightful_pg = False,
        delightful_eta = 1.0,
        use_hl_gauss = False,
        hl_gauss_val_min = -3.,
        hl_gauss_val_max = 3.,
        hl_gauss_num_bins = 64
    ):
        super().__init__()

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

        # state and reward normalization

        self.state_norm = ObservationNormalizer(dim_state)

        self.reward_norm = ScaleRewardNormalizer(discount_factor = discount_factor)

        # self-predictive representation
        # Schwarzer et al. https://arxiv.org/abs/2007.05929

        self.actor_self_predict_repr = actor_self_predict_repr
        self.critic_self_predict_repr = critic_self_predict_repr

        self.spr_lr = spr_lr
        self.spr_lr_param_update = spr_lr_param_update

        self.spr_orth_beta = spr_orth_beta

        # actor

        dim_readout_input = dim_actor

        self.actor = actor

        assert num_discrete_actions > 0 or num_continuous_actions > 0

        self.readout = Readout(
            dim_readout_input,
            num_discrete = num_discrete_actions,
            num_continuous = num_continuous_actions
        )

        self.actor_with_readout = Sequential(actor, self.readout)

        self.actor_use_ema = actor_use_ema
        actor_has_ema = actor_use_ema or (actor_self_predict_repr and spr_target_embed_from_ema)
        self.actor_with_readout_ema = EMA(self.actor_with_readout, beta = actor_ema_beta) if actor_has_ema else None

        # critic readout

        self.use_hl_gauss = use_hl_gauss

        if use_hl_gauss:
            self.hl_gauss_layer = HLGaussLayer(
                dim = dim_critic,
                hl_gauss_loss = dict(
                    min_value = hl_gauss_val_min,
                    max_value = hl_gauss_val_max,
                    num_bins = hl_gauss_num_bins,
                )
            )
            self.critic_readout = self.hl_gauss_layer
        else:
            self.critic_readout = Sequential(
                Linear(dim_critic, 1),
                Rearrange('... 1 -> ...')
            )

        # critic

        self.critic = critic

        self.critic_full = Sequential(critic, self.critic_readout)

        self.use_critic_ema = use_critic_ema
        critic_has_ema = use_critic_ema or (critic_self_predict_repr and spr_target_embed_from_ema)
        self.critic_ema = EMA(self.critic_full, beta = critic_ema_beta) if critic_has_ema else None

        # self-predictive representations (Nilaksh et al. 2026)

        spr_kwargs = dict(
            num_discrete_actions = num_discrete_actions,
            num_continuous_actions = num_continuous_actions,
            target_embed_from_ema = spr_target_embed_from_ema,
            sigreg_weight = spr_sigreg_weight,
            dim_hidden_expand_factor = spr_dim_hidden_expand_factor,
            use_sem = spr_use_sem,
            sem_dim_simplex = spr_sem_dim_simplex,
            sem_temperature = spr_sem_temperature
        )

        if actor_self_predict_repr:
            self.actor_spr = SelfPredictRepr(dim_readout_input, **spr_kwargs)

        if critic_self_predict_repr:
            self.critic_spr = SelfPredictRepr(dim_critic, **spr_kwargs)

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
            self.actor_v = BufferDict({name: torch.zeros_like(param) for name, param in self.actor_with_readout.named_parameters()})
            self.critic_v = BufferDict({name: torch.zeros_like(param) for name, param in self.critic_full.named_parameters()})

        self.eligibility_trace_decay = eligibility_trace_decay # lambda in paper

        # spr gradient momentum for orth¹ decorrelation (Nilaksh et al. 2026)

        if actor_self_predict_repr:
            self.actor_spr_momentum = BufferDict({name: torch.zeros_like(param) for name, param in self.actor_with_readout.named_parameters()})
            self.actor_spr_aux_momentum = BufferDict({str(i): torch.zeros_like(param) for i, param in enumerate(self.actor_spr.parameters())})

        if critic_self_predict_repr:
            self.critic_spr_momentum = BufferDict({name: torch.zeros_like(param) for name, param in self.critic_full.named_parameters()})
            self.critic_spr_aux_momentum = BufferDict({str(i): torch.zeros_like(param) for i, param in enumerate(self.critic_spr.parameters())})

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

    def save(self, path, overwrite = True):
        path = Path(path)
        assert overwrite or not path.exists(), f"File {path} already exists"
        torch.save(self.state_dict(), str(path))

    def load(self, path):
        path = Path(path)
        assert path.exists(), f"File {path} does not exist"
        self.load_state_dict(torch.load(str(path)))

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

            for v in self.actor_v.values():
                v.zero_()

            for v in self.critic_v.values():
                v.zero_()

        if self.actor_self_predict_repr:
            for m in self.actor_spr_momentum.values():
                m.zero_()
            for m in self.actor_spr_aux_momentum.values():
                m.zero_()

        if self.critic_self_predict_repr:
            for m in self.critic_spr_momentum.values():
                m.zero_()
            for m in self.critic_spr_aux_momentum.values():
                m.zero_()

    def update(
        self,
        state,
        action,
        next_state,
        reward,
        is_terminal = False
    ):
        device = state.device
        action = cast_tensor(action, device = device)
        reward = cast_tensor(reward, dtype = torch.float32, device = device)
        is_terminal = cast_tensor(is_terminal, dtype = torch.bool, device = device)

        state = self.state_norm(state, update = True)
        next_state = self.state_norm(next_state, update = False)

        normed_reward = self.reward_norm(reward, is_terminal = is_terminal, update = True)

        metrics = self._learn_step((state, action, normed_reward, next_state, is_terminal))

        return metrics

    def _learn_step(self, transition):
        """single learning step for a transition, using 1-step return"""

        state, action, reward, next_state, is_term = transition

        # all gradient related

        with torch.enable_grad():

            # critic

            critic_embed = self.critic(state)
            value_pred = self.critic_readout(critic_embed)

            # getting the bootstrapped value

            def get_next_value_pred(state):
                next_value = self.critic_ema(state) if self.use_critic_ema else self.critic_full(state)

                if not self.use_minto:
                    return next_value

                online_next_value = self.critic_full(state)
                return torch.min(next_value, online_next_value)

            # 1-step td target

            if not is_term.item():
                td_target = (reward + self.discount_factor * get_next_value_pred(next_state)).detach()
            else:
                td_target = reward.detach()

            # critic mse

            td_error = td_target - value_pred
            td_error_sign = td_error.detach().sign()

            critic_params = list(self.critic_full.parameters())

            if self.use_hl_gauss:
                value_loss = self.hl_gauss_layer(critic_embed, td_target)
                critic_trace_grads = torch.autograd.grad(value_loss, critic_params, retain_graph=True, allow_unused=True)

                safe_delta = torch.where(td_error.detach() >= 0, 1.0, -1.0) * td_error.detach().abs().clamp(min = 1e-6)
                critic_trace_grads = [(-grad / safe_delta) if exists(grad) else None for grad in critic_trace_grads]
            else:
                critic_trace_grads = torch.autograd.grad(value_pred, critic_params, retain_graph=True, allow_unused=True)

            value_grad = {
                name: grad
                for (name, _), grad in zip(self.critic_full.named_parameters(), critic_trace_grads)
            }

            # actor with entropy regularization

            action_logits = self.actor_with_readout(state)
            log_prob = self.readout.log_prob(action_logits, action).mean()
            entropy = self.readout.entropy(action_logits).mean()

            # Combine entropy into the actor trace, modulated by sign(td_error)

            actor_loss = log_prob + self.entropy_weight * td_error_sign * entropy

            actor_params = list(self.actor_with_readout.parameters())

            # policy gradient goes into the eligibility trace

            actor_loss_grads = torch.autograd.grad(actor_loss, actor_params, retain_graph = True)
            actor_grad = {name: grad for (name, _), grad in zip(self.actor_with_readout.named_parameters(), actor_loss_grads)}

            # SPR gradient helper

            def get_spr_grad(
                embed,
                network,
                network_with_readout,
                network_ema,
                spr_module,
                spr_momentum,
                spr_aux_momentum,
                network_params
            ):
                # target representation

                if spr_module.target_embed_from_ema:
                    assert exists(network_ema), 'ema must be available if target_embed_from_ema is True'
                    next_embed = first(network_ema.ema_model)(next_state)
                else:
                    next_embed = network(next_state)

                # loss

                spr_loss = spr_module(embed, action, next_embed)

                # gradients

                spr_params = list(spr_module.parameters())
                spr_grads_tup = torch.autograd.grad(spr_loss, network_params + spr_params, retain_graph = True, allow_unused = True)
                spr_grads_tup = tuple(default(g, torch.zeros_like(p)) for g, p in zip(spr_grads_tup, network_params + spr_params))

                spr_grad = {name: grad for (name, _), grad in zip(network_with_readout.named_parameters(), spr_grads_tup[:len(network_params)])}
                spr_only_grad = list(spr_grads_tup[len(network_params):])

                # orth¹: project SPR encoder grads orthogonal to their own EMA momentum
                # this decorrelates the highly correlated streaming data

                names = list(spr_grad.keys())
                all_grads = [spr_grad[n] for n in names] + spr_only_grad
                all_moms = [spr_momentum[n] for n in names] + list(spr_aux_momentum.values())

                all_grads_proj = [orthog_project(g, m) for g, m in zip(all_grads, all_moms)]

                # momentum update

                for grad, mom in zip(all_grads, all_moms):
                    mom.mul_(self.spr_orth_beta).add_(grad, alpha = 1. - self.spr_orth_beta)

                for name, proj_grad in zip(names, all_grads_proj[:len(names)]):
                    spr_grad[name] = proj_grad

                spr_only_grad = all_grads_proj[len(names):]

                # update SPR auxiliary networks via SGD with dedicated spr_lr

                for param, grad in zip(spr_params, spr_only_grad):
                    param.data.add_(grad, alpha = -self.spr_lr)

                return spr_grad

            # get the SPR loss

            actor_spr_grad = None
            if self.actor_self_predict_repr:
                actor_embed = self.actor(state)
                actor_spr_grad = get_spr_grad(
                    actor_embed,
                    self.actor,
                    self.actor_with_readout,
                    self.actor_with_readout_ema,
                    self.actor_spr,
                    self.actor_spr_momentum,
                    self.actor_spr_aux_momentum,
                    actor_params
                )

            # critic SPR loss

            critic_spr_grad = None
            if self.critic_self_predict_repr:
                critic_spr_grad = get_spr_grad(
                    critic_embed,
                    self.critic,
                    self.critic_full,
                    self.critic_ema,
                    self.critic_spr,
                    self.critic_spr_momentum,
                    self.critic_spr_aux_momentum,
                    critic_params
                )

        td_error = td_error.detach().squeeze()
        td_error_sign = td_error_sign.detach().squeeze()
        value_pred = value_pred.detach()

        # delightful pg

        gate = 1.
        if self.use_delightful_pg:
            surprisal = -log_prob.detach()
            delight = td_error.detach() * surprisal
            gate = (delight / self.delightful_eta).sigmoid()

        # update eligibility traces

        decay = self.eligibility_trace_decay * self.discount_factor

        for name, trace in self.actor_trace.items():
            trace.mul_(decay).add_(actor_grad[name] * gate)

        for name, trace in self.critic_trace.items():
            trace.mul_(decay).add_(value_grad[name])

        # overstepping-bounds gradient descent

        td_error_factor = td_error.abs().clamp(min = 1.)

        actor_grad_norm = torch.stack([g.norm(p = 1) for g in actor_grad.values()]).mean()
        critic_grad_norm = torch.stack([g.norm(p = 1) for g in value_grad.values()]).mean()

        if self.adaptive:
            self.step.add_(1)
            step = self.step.item()
            bias_correction2 = 1. - self.adam_beta2 ** step

        def update_params(params, init_params, traces, vs, kappa, lr, l2_wd, has_l2_wd, l1_wd, has_l1_wd, has_any_wd, cautious_wd, wd_towards_init, spr_grads = None):
            if self.adaptive:
                # update second moment of semi-gradients

                for name, trace in traces.items():
                    v = vs[name]
                    grad = td_error * trace
                    v.mul_(self.adam_beta2).add_(grad ** 2, alpha = 1. - self.adam_beta2)

                # adam update direction + obgd bound (Algorithm 11)

                adapted_grads = {}
                trace_sum = 0.0

                for name, trace in traces.items():
                    v_hat = vs[name] / bias_correction2
                    denom = v_hat.sqrt() + self.eps

                    adapted_grads[name] = td_error * trace / denom

                    trace_sum += (trace / denom).abs().sum()
            else:
                adapted_grads = {name: td_error * trace for name, trace in traces.items()}

                trace_sum = 0.0
                for trace in traces.values():
                    trace_sum += trace.abs().sum()

            global_scale = (kappa * td_error_factor * trace_sum).reciprocal().clamp(max = 1.)

            for name, param in params.named_parameters():
                update = adapted_grads[name] * lr

                if exists(spr_grads) and name in spr_grads:
                    spr_update = -spr_grads[name] * self.spr_lr_param_update
                    update = update + orthog_project(spr_update, update)

                update = update * global_scale

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
            spr_grads = actor_spr_grad
        )

        critic_trace_norm, critic_scale = update_params(
            self.critic_full,
            self.init_params['critic'] if exists(self.init_params) else None,
            self.critic_trace,
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
            spr_grads = critic_spr_grad
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
