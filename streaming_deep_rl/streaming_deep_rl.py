from __future__ import annotations
import copy
import math
from collections import deque
from pathlib import Path
from typing import NamedTuple, Literal, Callable, Sequence

import torch
import torch.nn.functional as F
from torch import nn, tensor, is_tensor, cat, stack, Tensor
from torch.nn import Module, Linear, Sequential
from torch.nn.utils import parameters_to_vector

import einx
from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from discrete_continuous_embed_readout import Readout, Embed

from torch_einops_utils import tree_map_tensor, masked_mean, orthogonal_component, dict_add

from ema_pytorch import EMA

from hl_gauss_pytorch import HLGaussLayer

from x_mlps_pytorch import MLP
from x_mlps_pytorch.norms import LayerNorm as XLayerNorm, RMSNorm as XRMSNorm

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

class SSLLosses(NamedTuple):
    total: Tensor
    recon: Tensor

class UpdateMetrics(NamedTuple):
    td_error: float
    value_pred: float
    actor_grad_norm: float
    critic_grad_norm: float
    actor_trace_norm: float
    critic_trace_norm: float
    actor_scale: float
    critic_scale: float
    chunked_td_lambda: float

class RebornMetrics(NamedTuple):
    actor_kl: float
    critic_loss: float
    dim_actor: int
    dim_critic: int
    core_sparsity: float
    right_sparsity: float

# initialization

@torch.no_grad()
def sparse_init_tensor_(
    weight: Tensor,
    sparsity = 0.9,
    scale = 1.
):
    if weight.numel() == 0:
        return weight

    device = weight.device
    fan_out, fan_in = weight.shape

    value = (fan_in ** -0.5) * scale
    nn.init.uniform_(weight, -value, value)

    assert 0. <= sparsity <= 1.

    num_zeros = int(math.ceil(sparsity * fan_in))
    if sparsity < 1.0:
        num_zeros = min(fan_in - 1, num_zeros)

    if num_zeros > 0:
        random_scores = torch.randn(fan_out, fan_in, device = device)
        zero_indices = random_scores.argsort(dim = -1)[:, :num_zeros]
        weight.scatter_(1, zero_indices, 0.)

    return weight

@torch.no_grad()
def sparse_init_(
    l: Linear,
    sparsity = 0.9
):
    """
    Algorithm 1
    """
    sparse_init_tensor_(l.weight, sparsity = sparsity)

    if exists(l.bias):
        nn.init.zeros_(l.bias)

def round_up_multiple(num: int, multiple: int) -> int:
    if multiple <= 1:
        return num
    return int(math.ceil(num / multiple) * multiple)

def _tree_select_rows(tree_a, tree_b, mask_b: Tensor):
    if is_tensor(tree_a):
        view_mask = mask_b.reshape(-1, *([1] * (tree_a.ndim - 1)))
        return torch.where(view_mask, tree_b, tree_a)

    if hasattr(tree_a, '_make'):
        return type(tree_a)._make(_tree_select_rows(a, b, mask_b) for a, b in zip(tree_a, tree_b))

    if isinstance(tree_a, (tuple, list)):
        return type(tree_a)(_tree_select_rows(a, b, mask_b) for a, b in zip(tree_a, tree_b))

    return tree_a

def _kl_mean(kl) -> Tensor:
    if is_tensor(kl):
        return kl.mean()
    return sum(t.mean() for t in kl if exists(t))

def _probe_entropy(readout: Readout, dist) -> Tensor:
    ent = readout.entropy(dist)
    if is_tensor(ent):
        return ent.reshape(ent.shape[0], -1).mean(dim = -1)
    return sum(t.reshape(t.shape[0], -1).mean(dim = -1) for t in ent if exists(t))

def _resolve_parent_choice(
    filter_fn: str | Callable,
    probes: Tensor,
    dists: tuple,
    vals: tuple[Tensor, Tensor],
    parent_fitnesses: Sequence[float] | Tensor | None = None,
    readout: Readout | None = None
) -> Tensor:
    val_a, val_b = vals
    num_probes, device = val_a.shape[0], val_a.device
    fits = torch.as_tensor(parent_fitnesses, device = device, dtype = torch.float32) if exists(parent_fitnesses) else None

    if callable(filter_fn):
        out = filter_fn(probes, dists, vals, fits)
        if is_tensor(out) and out.shape == (2, num_probes):
            return out.argmax(dim = 0).to(device)
        return torch.as_tensor(out, device = device, dtype = torch.long).reshape(num_probes)

    if filter_fn == 'critic':
        return (val_b > val_a).long()

    if filter_fn == 'entropy':
        assert exists(readout), 'parent_filter="entropy" requires readout'
        return (_probe_entropy(readout, dists[1]) < _probe_entropy(readout, dists[0])).long()

    if filter_fn == 'fitness':
        assert exists(fits), 'parent_filter="fitness" requires parent_fitnesses'
        return torch.full((num_probes,), int(fits.argmax().item()), device = device, dtype = torch.long)

    if filter_fn == 'random':
        return torch.randint(0, 2, (num_probes,), device = device)

    raise ValueError(f'unknown parent_filter {filter_fn!r} - choose from "critic", "entropy", "fitness", "random", or a callable')

class SplitNorm(Module):
    """
    Normalizes the consolidated core `[0 : split_dim]` and fresh plastic reservoir
    `[split_dim : dim]` independently while preserving identical parameter/buffer names
    to `x_mlps_pytorch` and `torch.nn` LayerNorm / RMSNorm modules.
    """

    def __init__(
        self,
        dim: int,
        split_dim: int | None = None,
        *,
        use_rms: bool = False,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        use_gamma_offset: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.split_dim = split_dim
        self.use_rms = use_rms
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.use_gamma_offset = use_gamma_offset

        if use_gamma_offset:
            if elementwise_affine:
                self.gamma = nn.Parameter(torch.zeros(dim))
            else:
                self.register_buffer('gamma', torch.ones(dim))
            self.offset = float(elementwise_affine)
        else:
            if elementwise_affine:
                self.weight = nn.Parameter(torch.ones(dim))
                self.bias = None if use_rms else nn.Parameter(torch.zeros(dim))
            else:
                self.register_parameter('weight', None)
                self.register_parameter('bias', None)

    def _norm(self, x: Tensor, w: Tensor | None, b: Tensor | None) -> Tensor:
        shape = x.shape[-1:]
        if self.use_rms:
            return F.rms_norm(x, shape, w, eps = self.eps)
        return F.layer_norm(x, shape, w, b, eps = self.eps)

    def forward(self, x: Tensor) -> Tensor:
        if self.use_gamma_offset:
            w = self.gamma + self.offset
            b = None
        else:
            w = self.weight
            b = self.bias

        s = self.split_dim
        min_part = 1 if self.use_rms else 2

        if not exists(s) or s < min_part or (self.dim - s) < min_part:
            return self._norm(x, w, b)

        w_l = w[:s] if exists(w) else None
        w_r = w[s:] if exists(w) else None
        b_l = b[:s] if exists(b) else None
        b_r = b[s:] if exists(b) else None

        return cat((
            self._norm(x[..., :s], w_l, b_l),
            self._norm(x[..., s:], w_r, b_r)
        ), dim = -1)

@torch.no_grad()
def reborn_backbone_(
    backbone: Module,
    *,
    dim_state: int,
    old_dim: int,
    new_dim: int,
    keep_ratio: float = 0.5,
    sparsity: float = 0.9,
    core_sparsity: float = 0.9,
    cross_scale: float = 0.05,
    align_multiple: int = 2,
    split_norms: bool = True
):
    """
    Reinitializes a backbone as a distilled core `[0 : keep_dim]` plus a fresh sparse
    plastic reservoir `[keep_dim : new_dim]`, resizing hidden widths when `new_dim != old_dim`.
    Returns `(linear_records, norm_records)` used to share one gradient mask per tensor.
    """
    first_param = next(backbone.parameters(), None)
    device = first_param.device if exists(first_param) else torch.device('cpu')
    dtype = first_param.dtype if exists(first_param) else torch.float32

    # find linears that directly consume the raw state (input linears)

    input_linears = set()
    mock_state = torch.zeros(dim_state, device = device, dtype = dtype)

    def _track_input(mod, args):
        if len(args) > 0 and is_tensor(args[0]) and args[0].data_ptr() == mock_state.data_ptr():
            input_linears.add(mod)

    hooks = [m.register_forward_pre_hook(_track_input) for m in backbone.modules() if isinstance(m, Linear)]

    try:
        backbone(mock_state)
    finally:
        for h in hooks:
            h.remove()

    # ResidualNormedMLP with Identity proj_in when growing width

    if hasattr(backbone, 'proj_in') and isinstance(backbone.proj_in, nn.Identity) and new_dim != dim_state:
        backbone.proj_in = Linear(dim_state, new_dim, device = device, dtype = dtype)
        input_linears.add(backbone.proj_in)

    def scale_dim(d):
        if new_dim == old_dim:
            return d
        if d == old_dim:
            return new_dim
        return round_up_multiple(max(align_multiple, int(round(d * new_dim / old_dim))), align_multiple)

    def keep_of(d):
        if d <= 1:
            return d
        return max(1, min(d - 1, int(round(d * keep_ratio))))

    linear_records = []
    norm_records = []

    def _process(parent):
        for name, child in list(parent.named_children()):
            if isinstance(child, Linear):
                is_input = child in input_linears

                in_f = child.in_features if is_input else scale_dim(child.in_features)
                out_f = scale_dim(child.out_features)
                has_bias = exists(child.bias)

                if (in_f, out_f) != (child.in_features, child.out_features):
                    child = Linear(in_f, out_f, bias = has_bias, device = device, dtype = dtype)
                    setattr(parent, name, child)

                in_keep = in_f if is_input else keep_of(in_f)
                out_keep = keep_of(out_f)

                # core | cross | right reservoir (or uniform full-width init when not split_norms)

                if not split_norms:
                    sparse_init_tensor_(child.weight, sparsity = sparsity)
                elif is_input:
                    sparse_init_tensor_(child.weight[:out_keep], sparsity = core_sparsity)
                    if out_f > out_keep:
                        sparse_init_tensor_(child.weight[out_keep:], sparsity = sparsity)
                else:
                    sparse_init_tensor_(child.weight[:out_keep, :in_keep], sparsity = core_sparsity)
                    if in_f > in_keep:
                        sparse_init_tensor_(child.weight[:out_keep, in_keep:], sparsity = sparsity, scale = cross_scale)
                    if out_f > out_keep:
                        sparse_init_tensor_(child.weight[out_keep:], sparsity = sparsity)

                if has_bias:
                    child.bias.zero_()

                linear_records.append((child, is_input, in_keep, out_keep))

            elif isinstance(child, (SplitNorm, XLayerNorm, XRMSNorm, nn.LayerNorm, nn.RMSNorm)):
                if isinstance(child, SplitNorm):
                    elem_affine = child.elementwise_affine
                    cur_d = child.dim
                    use_rms = child.use_rms
                    use_gamma_offset = child.use_gamma_offset
                    eps = child.eps
                else:
                    elem_affine = isinstance(getattr(child, 'gamma', None), nn.Parameter) or getattr(child, 'elementwise_affine', False)
                    cur_d = child.gamma.shape[0] if hasattr(child, 'gamma') else child.normalized_shape[0]
                    use_rms = isinstance(child, (XRMSNorm, nn.RMSNorm))
                    use_gamma_offset = isinstance(child, (XLayerNorm, XRMSNorm))
                    eps = child.eps

                target_d = scale_dim(cur_d)
                split_d = keep_of(target_d) if split_norms else None

                child = SplitNorm(
                    target_d,
                    split_dim = split_d,
                    use_rms = use_rms,
                    eps = eps,
                    elementwise_affine = elem_affine,
                    use_gamma_offset = use_gamma_offset
                ).to(device = device, dtype = dtype)

                setattr(parent, name, child)

                if elem_affine:
                    p = child.gamma if use_gamma_offset else child.weight
                    norm_records.append((p, keep_of(target_d)))
                    if exists(getattr(child, 'bias', None)):
                        norm_records.append((child.bias, keep_of(target_d)))
            else:
                _process(child)

    _process(backbone)
    return linear_records, norm_records

# orthogonal projection for SPR in streaming setting
# https://arxiv.org/abs/2602.09396

def orthog_project_tensors(
    grads: list[Tensor],
    moms: list[Tensor]
):
    if len(grads) == 0:
        return []

    flat_grad = parameters_to_vector(grads)
    flat_mom = parameters_to_vector(moms)

    dtype = flat_grad.dtype

    if flat_grad.device.type != 'mps':
        flat_grad, flat_mom = flat_grad.double(), flat_mom.double()

    flat_orth = orthogonal_component(flat_grad, flat_mom).to(dtype)

    sizes = [grad.numel() for grad in grads]
    return [part.view_as(grad) for part, grad in zip(flat_orth.split(sizes), grads)]

# Simplicial Embeddings
# Lavoie et al - https://arxiv.org/abs/2204.00616

class SEM(Module):
    def __init__(
        self,
        dim,
        dim_in = None,
        project_in = None,
        temperature = 0.1,
        dim_simplex = 8,
        pre_layernorm = False
    ):
        super().__init__()
        assert divisible_by(dim, dim_simplex), f'{dim} must be divisible by {dim_simplex}'

        dim_in = default(dim_in, dim)
        project_in = default(project_in, dim_in != dim)

        self.embedder = Linear(dim_in, dim, bias = False) if project_in else nn.Identity()

        self.dim = dim
        self.dim_simplex = dim_simplex
        self.temperature = temperature

        self.norm = nn.LayerNorm(dim, bias = False) if pre_layernorm else nn.Identity()

    def forward(
        self,
        t
    ):
        t = self.embedder(t)
        t = self.norm(t)
        t = rearrange(t, '... (l v) -> ... l v', v = self.dim_simplex)
        t = (t / self.temperature).softmax(dim = -1)
        return rearrange(t, '... l v -> ... (l v)')

# Max Schwarzer et al. https://arxiv.org/abs/2007.05929

class SelfPredictRepr(Module):
    def __init__(
        self,
        dim_embed,
        num_discrete_actions = 0,
        num_continuous_actions = 0,
        dim_predict = None,
        target_embed_from_ema = True, # whether the target embed is from an EMA, or from the model itself (traditional vs sigreg from lejepa)
        sigreg_weight = 0.,
        sigreg_num_subspaces = 1,
        dim_hidden_expand_factor = 4,
        use_sem = True,
        sem_dim_simplex = 8,
        sem_temperature = 0.1
    ):
        super().__init__()
        assert num_discrete_actions > 0 or num_continuous_actions > 0

        dim_predict = default(dim_predict, dim_embed)
        dim_mlp_hidden = int(dim_embed * dim_hidden_expand_factor)

        # when SEM is active, projects dim_embed -> dim_predict before the projector

        dim_projector_in = dim_predict if use_sem else dim_embed

        self.to_action_embed = Embed(dim_projector_in, num_discrete = num_discrete_actions, num_continuous = num_continuous_actions)

        self.target_embed_from_ema = target_embed_from_ema

        self.to_projection = nn.Sequential(
            nn.RMSNorm(dim_projector_in),
            MLP(dim_projector_in, dim_mlp_hidden, dim_predict)
        )

        self.to_prediction = MLP(dim_predict + dim_projector_in, dim_mlp_hidden, dim_predict)

        self.apply_sigreg = sigreg_weight > 0.
        self.sigreg_weight = sigreg_weight

        # sub-jepa subspace gaussian regularization
        # Kai Zhao et al https://arxiv.org/abs/2605.09241v1

        self.num_subspaces = sigreg_num_subspaces

        if self.num_subspaces > 1:
            dim_subspace, remainder = divmod(dim_predict, self.num_subspaces)
            assert remainder == 0, f'dimension {dim_predict} must be divisible by number of subspaces {self.num_subspaces}'

            projs = stack([nn.init.orthogonal_(torch.empty(dim_subspace, dim_predict)) for _ in range(self.num_subspaces)])
            self.register_buffer('subspace_projs', projs)

        self.use_sem = use_sem
        self.sem = SEM(dim_predict, dim_in = dim_embed, project_in = True, temperature = sem_temperature, dim_simplex = sem_dim_simplex) if use_sem else nn.Identity()

    def forward(
        self,
        state_embed,
        actions,
        next_state_embed,
        ema_to_projection = None
    ):
        if self.use_sem:
            state_embed = self.sem(state_embed)
            next_state_embed = self.sem(next_state_embed)

        projected = self.to_projection(state_embed)

        # maybe no

        decorator = no_grad_detach if self.target_embed_from_ema else identity

        target_to_projection = self.to_projection
        if self.target_embed_from_ema and exists(ema_to_projection):
            target_to_projection = ema_to_projection

        next_projected = decorator(target_to_projection)(next_state_embed)

        action_embed = self.to_action_embed(actions)

        predicted = self.to_prediction(cat((projected, action_embed), dim = -1))

        # cosine sim loss

        recon_loss = F.mse_loss(l2norm(predicted), l2norm(next_projected))

        total_loss = recon_loss

        if self.apply_sigreg:
            sigreg_input = projected

            if self.num_subspaces > 1:
                sigreg_input = einx.dot('... dim, k dim_subspace dim -> k ... dim_subspace', sigreg_input, self.subspace_projs)
            else:
                sigreg_input = rearrange(sigreg_input, '... d -> 1 ... d')

            total_loss = total_loss + self.sigreg_weight * sigreg(sigreg_input)

        return SSLLosses(total = total_loss, recon = recon_loss)

# Dominik Schmidt https://arxiv.org/abs/2312.10812

class LAPO(Module):
    def __init__(
        self,
        dim_embed,
        dim_latent_action,
        dim_project = None,
        num_discrete_actions = 0,
        num_continuous_actions = 0,
        target_embed_from_ema = True,
        sem_dim_simplex = 4,
        sem_temperature = 0.1,
        expansion_factor = 4,
        pred_action_loss_weight = 0.,
    ):
        super().__init__()
        assert num_discrete_actions > 0 or num_continuous_actions > 0

        self.target_embed_from_ema = target_embed_from_ema

        # dimensions

        dim_project = default(dim_project, dim_embed)
        dim_mlp_hidden = int(dim_embed * expansion_factor)

        self.to_projection = nn.Sequential(
            nn.RMSNorm(dim_embed),
            MLP(dim_embed, dim_mlp_hidden, dim_project, activation = nn.SiLU())
        )

        # state + next state projected embeddings -> latent action - IDM

        self.to_latent_action_embed = nn.Linear(dim_project * 2, dim_latent_action, bias = False)

        # will use simplicial embed for action bottleneck

        assert divisible_by(dim_latent_action, sem_dim_simplex)
        self.sem = SEM(dim_latent_action, temperature = sem_temperature, dim_simplex = sem_dim_simplex)

        # for action prediction loss

        self.has_pred_action_loss = pred_action_loss_weight > 0.
        self.pred_action_loss_weight = pred_action_loss_weight
        self.action_readout = Readout(dim_latent_action, num_discrete = num_discrete_actions, num_continuous = num_continuous_actions) if self.has_pred_action_loss else None

        # predicting the next state embed

        self.to_pred_next_state_embed = MLP(dim_project + dim_latent_action, dim_mlp_hidden, dim_project, activation = nn.SiLU())

        self.register_buffer('zero', tensor(0.), persistent = False)

    def forward(
        self,
        state_embed,
        actions,
        next_state_embed,
        ema_to_projection = None
    ):
        state_projected = self.to_projection(state_embed)

        next_state_transform_decorator = no_grad_detach if self.target_embed_from_ema else identity
        next_state_projector = default(ema_to_projection, self.to_projection)
        next_state_projected = next_state_transform_decorator(next_state_projector)(next_state_embed)

        # inverse dynamics model

        latent_action_embed = self.to_latent_action_embed(cat((state_projected, next_state_projected), dim = -1))

        simplicial_embedded_latent_action = self.sem(latent_action_embed)

        # maybe action loss

        pred_action_loss = self.zero

        if self.has_pred_action_loss:
            pred_action_loss = self.action_readout(simplicial_embedded_latent_action, targets = actions, return_loss = True)
            pred_action_loss = pred_action_loss * self.pred_action_loss_weight

        # now do the forward dynamics model

        pred_next_state_projected = self.to_pred_next_state_embed(cat((state_projected, simplicial_embedded_latent_action), dim = -1))

        # cosine sim loss

        recon_loss = F.mse_loss(l2norm(pred_next_state_projected), l2norm(next_state_projected).detach())

        # lapo actually does a decoder back to state in pixel space, although probably not crucial given SPR

        return SSLLosses(total = recon_loss + pred_action_loss, recon = recon_loss)

# sigreg from lejepa

def sigreg(
    x,
    num_slices = 1024,
    domain = (-5, 5),
    num_knots = 17,
    mask = None
):
    # Randall Balestriero - https://arxiv.org/abs/2511.08544
    # Sub-JEPA - https://arxiv.org/abs/2605.09241v1

    dim, device = x.shape[-1], x.device

    # slice sampling

    rand_projs = torch.randn((num_slices, dim), device = device)
    rand_projs = l2norm(rand_projs)

    # integration points

    t = torch.linspace(*domain, num_knots, device = device)

    # theoretical CF for N(0, 1) and Gauss. window

    exp_f = (-0.5 * t.square()).exp()

    # empirical CF

    x_t = einx.dot('k ... d, m d -> k (...) m', x, rand_projs)

    x_t = einx.multiply('k n m, t -> k n m t', x_t, t)
    ecf = (1j * x_t).exp()

    mask_1d = rearrange(mask, 'k ... -> k (...)') if exists(mask) else None
    ecf = masked_mean(ecf, mask_1d, dim = 1)

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
        mean_center = True
    ):
        super().__init__()
        self.eps = eps
        self.discount_factor = discount_factor
        self.time_dilate_factor = time_dilate_factor
        self.mean_center = mean_center

        self.register_buffer('step', tensor(1))
        self.register_buffer('running_reward', tensor(0.))
        self.register_buffer('running_mean', tensor(0.))
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
        running_mean = self.running_mean.item()
        estimate_p = self.running_estimate_p.item()

        next_reward = running_reward * self.discount_factor * (1. - float(is_terminal)) + reward

        # rewards are not mean centered

        if self.mean_center:
            next_mean = running_mean + (next_reward - running_mean) / time
            next_estimate_p = estimate_p + (next_reward - running_mean) * (next_reward - next_mean)
            self.running_mean.copy_(next_mean)
        else:
            next_estimate_p = estimate_p + next_reward * next_reward

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
        actor_lr = 10.0,
        critic_lr = 10.0,
        adaptive = True,
        adam_beta1 = 0.9,
        adam_beta2 = 0.999,
        eps = 1e-5,
        actor_use_ema = False,
        actor_ema_beta = 0.7,
        critic_ema_beta = 0.7,
        ssl_type: Literal['spr', 'lapo'] | None = None,
        spr_kwargs: dict = dict(),
        lapo_kwargs: dict = dict(),
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
        hl_gauss_num_bins = 64,
        use_chunked_td = False,
        reborn_every = 0,
        reborn_buffer_size = 2048,
        reborn_keep_ratio = 0.5,
        reborn_sparsity: float | None = None,
        reborn_core_sparsity: float | None = None,
        reborn_epochs = 16,
        reborn_batch_size = 128,
        reborn_lr = 3e-3,
        reborn_cross_scale = 0.05,
        reborn_grow = False,
        reborn_grow_factor = 1.5,
        reborn_max_dim: int | None = None,
        reborn_scale_lr_with_grow = False,
        reborn_retain = 1.0
    ):
        super().__init__()

        self.dim_state = dim_state
        self.dim_actor = dim_actor
        self.dim_critic = dim_critic
        self.num_discrete_actions = num_discrete_actions
        self.num_continuous_actions = num_continuous_actions
        self.actor_ema_beta = actor_ema_beta
        self.critic_ema_beta = critic_ema_beta
        self.spr_kwargs = spr_kwargs
        self.lapo_kwargs = lapo_kwargs
        self.hl_gauss_kwargs = dict(
            min_value = hl_gauss_val_min,
            max_value = hl_gauss_val_max,
            num_bins = hl_gauss_num_bins,
        )

        # minto (use online network if you can)

        assert not (use_minto and not use_critic_ema), 'minto can only be used with critic ema'
        self.use_minto = use_minto

        # chunked td

        assert not (use_chunked_td and ssl_type is None), 'chunked td requires a predictive model (ssl_type)'
        self.use_chunked_td = use_chunked_td

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

        assert ssl_type in (None, 'spr', 'lapo'), 'ssl_type must be either "spr", "lapo", or None'
        self.ssl_type = ssl_type

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

        # squeezing more from stream - Nilaksh et al. https://arxiv.org/abs/2602.09396
        # application of self predictive representation

        ssl_action_kwargs = dict(num_discrete_actions = num_discrete_actions, num_continuous_actions = num_continuous_actions)

        ssl_modules = dict(
            spr = lambda dim: SelfPredictRepr(dim, **ssl_action_kwargs, **spr_kwargs),
            lapo = lambda dim: LAPO(dim, dim_latent_action = dim, **ssl_action_kwargs, **lapo_kwargs)
        )

        self.actor_spr = ssl_modules[ssl_type](dim_readout_input) if exists(ssl_type) else None
        self.critic_spr = ssl_modules[ssl_type](dim_critic) if exists(ssl_type) else None

        self.actor_use_ema = actor_use_ema
        actor_has_ema = actor_use_ema or (exists(self.actor_spr) and self.actor_spr.target_embed_from_ema)
        self.actor_with_readout_ema = EMA(self.actor_with_readout, beta = actor_ema_beta) if actor_has_ema else None
        self.actor_spr_ema = EMA(self.actor_spr.to_projection, beta = actor_ema_beta) if exists(self.actor_spr) and self.actor_spr.target_embed_from_ema else None

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
        critic_has_ema = use_critic_ema or (exists(self.critic_spr) and self.critic_spr.target_embed_from_ema)
        self.critic_ema = EMA(self.critic_full, beta = critic_ema_beta) if critic_has_ema else None
        self.critic_spr_ema = EMA(self.critic_spr.to_projection, beta = critic_ema_beta) if exists(self.critic_spr) and self.critic_spr.target_embed_from_ema else None

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

        if exists(self.ssl_type):
            self.actor_spr_momentum = BufferDict({name: torch.zeros_like(param) for name, param in self.actor_with_readout.named_parameters()})
            self.actor_spr_aux_momentum = BufferDict({str(i): torch.zeros_like(param) for i, param in enumerate(self.actor_spr.parameters())})
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

        # continual reborn: half-distill + sparse reinit, optional width growth

        assert 0. < reborn_keep_ratio < 1., 'reborn_keep_ratio must be in (0, 1)'
        assert reborn_grow_factor >= 1., 'reborn_grow_factor must be >= 1.0'
        assert 0. <= reborn_retain <= 1., 'reborn_retain must be in [0, 1]'

        self.reborn_every = reborn_every
        self.reborn_buffer_size = reborn_buffer_size
        self.reborn_keep_ratio = reborn_keep_ratio
        self.reborn_sparsity = default(reborn_sparsity, init_sparsity)
        self.reborn_core_sparsity = reborn_core_sparsity
        self.reborn_epochs = reborn_epochs
        self.reborn_batch_size = reborn_batch_size
        self.reborn_lr = reborn_lr
        self.reborn_cross_scale = reborn_cross_scale
        self.reborn_grow = reborn_grow
        self.reborn_grow_factor = reborn_grow_factor
        self.reborn_max_dim = reborn_max_dim
        self.reborn_scale_lr_with_grow = reborn_scale_lr_with_grow
        self.reborn_retain = reborn_retain

        self.reborn_states = deque(maxlen = reborn_buffer_size) if reborn_buffer_size > 0 else None

        self.register_buffer('update_step', tensor(0, dtype = torch.long), persistent = False)


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

        if exists(self.ssl_type):
            for m in self.actor_spr_momentum.values():
                m.zero_()
            for m in self.actor_spr_aux_momentum.values():
                m.zero_()
            for m in self.critic_spr_momentum.values():
                m.zero_()
            for m in self.critic_spr_aux_momentum.values():
                m.zero_()

    @torch.no_grad()
    def _record_reborn_state_(self, normed_state: Tensor):
        if not exists(self.reborn_states):
            return

        for row in normed_state.detach().reshape(-1, self.dim_state):
            self.reborn_states.append(row.clone())

    def _align_multiple(self) -> int:
        mult = 2
        if self.ssl_type == 'spr':
            use_sem = self.spr_kwargs.get('use_sem', True)
            if use_sem:
                mult = math.lcm(mult, int(self.spr_kwargs.get('sem_dim_simplex', 8)))
            num_sub = int(self.spr_kwargs.get('sigreg_num_subspaces', 1))
            if num_sub > 1:
                mult = math.lcm(mult, num_sub)
        elif self.ssl_type == 'lapo':
            mult = math.lcm(mult, int(self.lapo_kwargs.get('sem_dim_simplex', 4)))
        return mult

    def _compute_target_dim(
        self,
        cur_dim: int,
        grow: bool,
        grow_factor: float,
        max_dim: int | None,
        align_multiple: int
    ) -> int:
        if not grow or grow_factor <= 1.:
            return cur_dim

        target = round_up_multiple(max(cur_dim + align_multiple, int(round(cur_dim * grow_factor))), align_multiple)
        if exists(max_dim):
            target = min(target, (max_dim // align_multiple) * align_multiple)
        return max(cur_dim, target)

    def _reset_after_reborn_(self, device: torch.device):
        ssl_action_kwargs = dict(
            num_discrete_actions = self.num_discrete_actions,
            num_continuous_actions = self.num_continuous_actions
        )

        if exists(self.ssl_type):
            ssl_modules = dict(
                spr = lambda dim: SelfPredictRepr(dim, **ssl_action_kwargs, **self.spr_kwargs),
                lapo = lambda dim: LAPO(dim, dim_latent_action = dim, **ssl_action_kwargs, **self.lapo_kwargs)
            )
            self.actor_spr = ssl_modules[self.ssl_type](self.dim_actor).to(device)
            self.critic_spr = ssl_modules[self.ssl_type](self.dim_critic).to(device)
            self.actor_spr.apply(self.init_)
            self.critic_spr.apply(self.init_)

        actor_has_ema = self.actor_use_ema or (exists(self.actor_spr) and self.actor_spr.target_embed_from_ema)
        self.actor_with_readout_ema = EMA(self.actor_with_readout, beta = self.actor_ema_beta).to(device) if actor_has_ema else None
        self.actor_spr_ema = EMA(self.actor_spr.to_projection, beta = self.actor_ema_beta).to(device) if exists(self.actor_spr) and self.actor_spr.target_embed_from_ema else None

        critic_has_ema = self.use_critic_ema or (exists(self.critic_spr) and self.critic_spr.target_embed_from_ema)
        self.critic_ema = EMA(self.critic_full, beta = self.critic_ema_beta).to(device) if critic_has_ema else None
        self.critic_spr_ema = EMA(self.critic_spr.to_projection, beta = self.critic_ema_beta).to(device) if exists(self.critic_spr) and self.critic_spr.target_embed_from_ema else None

        self.actor_trace = BufferDict({name: torch.zeros_like(param) for name, param in self.actor_with_readout.named_parameters()})
        self.critic_trace = BufferDict({name: torch.zeros_like(param) for name, param in self.critic_full.named_parameters()})

        if self.adaptive:
            self.step.zero_()
            self.actor_v = BufferDict({name: torch.zeros_like(param) for name, param in self.actor_with_readout.named_parameters()})
            self.critic_v = BufferDict({name: torch.zeros_like(param) for name, param in self.critic_full.named_parameters()})

        if exists(self.ssl_type):
            self.actor_spr_momentum = BufferDict({name: torch.zeros_like(param) for name, param in self.actor_with_readout.named_parameters()})
            self.actor_spr_aux_momentum = BufferDict({str(i): torch.zeros_like(param) for i, param in enumerate(self.actor_spr.parameters())})
            self.critic_spr_momentum = BufferDict({name: torch.zeros_like(param) for name, param in self.critic_full.named_parameters()})
            self.critic_spr_aux_momentum = BufferDict({str(i): torch.zeros_like(param) for i, param in enumerate(self.critic_spr.parameters())})

        if self.has_any_wd and self.wd_towards_init:
            self.init_params = nn.ModuleDict({
                'actor': BufferDict({name: param.detach().clone() for name, param in self.actor_with_readout.named_parameters()}),
                'critic': BufferDict({name: param.detach().clone() for name, param in self.critic_full.named_parameters()})
            })

    @torch.no_grad()
    def clone(
        self,
        reset_traces: bool = True,
        device: torch.device | str | None = None
    ) -> StreamingACLambda:
        saved_states = self.reborn_states
        self.reborn_states = None
        try:
            cloned = copy.deepcopy(self)
        finally:
            self.reborn_states = saved_states

        if exists(device):
            cloned = cloned.to(device)

        target_device = next(cloned.actor_with_readout.parameters()).device
        cloned.reborn_states = deque(maxlen = cloned.reborn_buffer_size) if cloned.reborn_buffer_size > 0 else None

        if reset_traces:
            cloned._reset_after_reborn_(target_device)

        return cloned

    @torch.no_grad()
    def _teacher_targets(
        self,
        states: Tensor,
        *,
        raw_probes: Tensor | None,
        other_states: Tensor | None,
        parent_pair: tuple[StreamingACLambda, StreamingACLambda] | None,
        parent_filter: str | Callable,
        parent_fitnesses: Sequence[float] | Tensor | None,
        retain: float
    ) -> tuple[object, Tensor]:
        if not exists(parent_pair):
            teacher_actor = self.actor_with_readout_ema if self.actor_use_ema and exists(self.actor_with_readout_ema) else self.actor_with_readout
            teacher_critic = self.critic_ema if self.use_critic_ema and exists(self.critic_ema) else self.critic_full
            return (
                tree_map_tensor(lambda t: t.detach().clone() * retain, teacher_actor(states)),
                teacher_critic(states).detach().clone() * retain
            )

        pa, pb = parent_pair
        states_a = pa.state_norm(raw_probes, update = False) if exists(raw_probes) else states
        states_b = pb.state_norm(raw_probes, update = False) if exists(raw_probes) else (
            other_states.detach().reshape(-1, self.dim_state).to(device = states.device, dtype = states.dtype) if exists(other_states) else states
        )

        actor_a = pa.actor_with_readout_ema if pa.actor_use_ema and exists(pa.actor_with_readout_ema) else pa.actor_with_readout
        actor_b = pb.actor_with_readout_ema if pb.actor_use_ema and exists(pb.actor_with_readout_ema) else pb.actor_with_readout
        critic_a = pa.critic_ema if pa.use_critic_ema and exists(pa.critic_ema) else pa.critic_full
        critic_b = pb.critic_ema if pb.use_critic_ema and exists(pb.critic_ema) else pb.critic_full

        dist_a = tree_map_tensor(lambda t: t.detach().clone(), actor_a(states_a))
        dist_b = tree_map_tensor(lambda t: t.detach().clone(), actor_b(states_b))

        self_std = self.reward_norm.variance.sqrt().clamp(min = 1e-5)
        scale_a = float((pa.reward_norm.variance.sqrt() / self_std).item()) if pa.reward_norm is not self.reward_norm else 1.
        scale_b = float((pb.reward_norm.variance.sqrt() / self_std).item()) if pb.reward_norm is not self.reward_norm else 1.

        val_a = critic_a(states_a).detach().clone().reshape(-1) * scale_a
        val_b = critic_b(states_b).detach().clone().reshape(-1) * scale_b

        choice = _resolve_parent_choice(parent_filter, default(raw_probes, states), (dist_a, dist_b), (val_a, val_b), parent_fitnesses, readout = pa.readout)
        mask_b = choice.bool()

        chosen_dist = _tree_select_rows(dist_a, dist_b, mask_b)
        return (
            tree_map_tensor(lambda t: t * retain, chosen_dist),
            torch.where(mask_b, val_b, val_a) * retain
        )

    def reborn_(
        self,
        states: Tensor | None = None,
        *,
        raw_states: Tensor | None = None,
        other_parent: StreamingACLambda | None = None,
        parents: tuple[StreamingACLambda, StreamingACLambda] | None = None,
        other_states: Tensor | None = None,
        parent_filter: str | Callable = 'critic',
        parent_fitnesses: Sequence[float] | Tensor | None = None,
        grow: bool | None = None,
        grow_factor: float | None = None,
        max_dim: int | None = None,
        keep_ratio: float | None = None,
        sparsity: float | None = None,
        core_sparsity: float | None = None,
        epochs: int | None = None,
        batch_size: int | None = None,
        lr: float | None = None,
        cross_scale: float | None = None,
        retain: float | None = None
    ) -> RebornMetrics | None:
        """
        Consolidates policy/value into `[0 : keep_dim]` (`SplitNorm`) via distillation while
        reinitializing `[keep_dim : new_dim]` as a fresh sparse plastic reservoir.
        When `other_parent` or `parents` is given, performs two-parent phenotypic crossover,
        selecting per probe state between the two parents via their value critics or `parent_filter`.
        """
        first_param = next(self.actor_with_readout.parameters())
        device, dtype = first_param.device, first_param.dtype

        if exists(parents):
            assert len(parents) == 2, 'parents must be a pair of two StreamingACLambda individuals'
            parent_pair = (parents[0], parents[1])
        elif exists(other_parent):
            parent_pair = (self, other_parent)
        else:
            parent_pair = None

        grow = default(grow, self.reborn_grow)
        grow_factor = default(grow_factor, self.reborn_grow_factor)
        max_dim = default(max_dim, self.reborn_max_dim)
        keep_ratio = default(keep_ratio, self.reborn_keep_ratio)
        sparsity = default(sparsity, self.reborn_sparsity)
        core_sparsity = default(core_sparsity, default(self.reborn_core_sparsity, sparsity))
        epochs = default(epochs, self.reborn_epochs)
        batch_size = default(batch_size, self.reborn_batch_size)
        lr = default(lr, self.reborn_lr)
        cross_scale = default(cross_scale, self.reborn_cross_scale)
        retain = default(retain, self.reborn_retain)
        do_distill = retain > 0.

        raw_probes = None
        teacher_dist, teacher_vals = None, None

        if do_distill:
            if exists(raw_states):
                raw_probes = raw_states.detach().reshape(-1, self.dim_state).to(device = device, dtype = dtype)
                if raw_probes.shape[0] == 0:
                    return None
                states = self.state_norm(raw_probes, update = False)
            elif not exists(states):
                if exists(parent_pair):
                    pa, pb = parent_pair
                    buf_a = list(pa.reborn_states) if exists(pa.reborn_states) else []
                    buf_b = [] if (pa.reborn_states is pb.reborn_states or not exists(pb.reborn_states)) else list(pb.reborn_states)
                    combined = buf_a + buf_b
                    if len(combined) == 0:
                        return None
                    if self.reborn_buffer_size > 0 and len(combined) > self.reborn_buffer_size:
                        step_idx = torch.linspace(0, len(combined) - 1, steps = self.reborn_buffer_size).round().long()
                        combined = [combined[int(i)] for i in step_idx]
                    states = torch.stack(combined)
                else:
                    if not exists(self.reborn_states) or len(self.reborn_states) == 0:
                        return None
                    states = torch.stack(list(self.reborn_states))

            states = states.detach().reshape(-1, self.dim_state).to(device = device, dtype = dtype)
            if states.shape[0] == 0:
                return None

            teacher_dist, teacher_vals = self._teacher_targets(
                states,
                raw_probes = raw_probes,
                other_states = other_states,
                parent_pair = parent_pair,
                parent_filter = parent_filter,
                parent_fitnesses = parent_fitnesses,
                retain = retain
            )

        # resize and reinitialize backbones & readouts

        align_mult = self._align_multiple()
        old_dim_actor, old_dim_critic = self.dim_actor, self.dim_critic
        base_dim_actor = max(old_dim_actor, *(p.dim_actor for p in parent_pair)) if exists(parent_pair) else old_dim_actor
        base_dim_critic = max(old_dim_critic, *(p.dim_critic for p in parent_pair)) if exists(parent_pair) else old_dim_critic

        new_dim_actor = self._compute_target_dim(base_dim_actor, grow, grow_factor, max_dim, align_mult)
        new_dim_critic = self._compute_target_dim(base_dim_critic, grow, grow_factor, max_dim, align_mult)
        keep_dim_actor = max(1, min(new_dim_actor - 1, int(round(new_dim_actor * keep_ratio)))) if new_dim_actor > 1 else 1
        keep_dim_critic = max(1, min(new_dim_critic - 1, int(round(new_dim_critic * keep_ratio)))) if new_dim_critic > 1 else 1

        actor_resized = new_dim_actor != old_dim_actor
        critic_resized = new_dim_critic != old_dim_critic

        actor_lins, actor_norms = reborn_backbone_(
            self.actor,
            dim_state = self.dim_state,
            old_dim = old_dim_actor,
            new_dim = new_dim_actor,
            keep_ratio = keep_ratio,
            sparsity = sparsity,
            core_sparsity = core_sparsity,
            cross_scale = cross_scale,
            align_multiple = align_mult,
            split_norms = do_distill
        )
        critic_lins, critic_norms = reborn_backbone_(
            self.critic,
            dim_state = self.dim_state,
            old_dim = old_dim_critic,
            new_dim = new_dim_critic,
            keep_ratio = keep_ratio,
            sparsity = sparsity,
            core_sparsity = core_sparsity,
            cross_scale = cross_scale,
            align_multiple = align_mult,
            split_norms = do_distill
        )

        if self.reborn_scale_lr_with_grow:
            if actor_resized:
                self.actor_lr = self.actor_lr * (new_dim_actor / old_dim_actor)
            if critic_resized:
                self.critic_lr = self.critic_lr * (new_dim_critic / old_dim_critic)

        self.dim_actor = new_dim_actor
        self.dim_critic = new_dim_critic

        if actor_resized:
            self.readout = Readout(
                new_dim_actor,
                num_discrete = self.num_discrete_actions,
                num_continuous = self.num_continuous_actions
            ).to(device = device, dtype = dtype)
            self.actor_with_readout = Sequential(self.actor, self.readout)

        if self.use_hl_gauss:
            if critic_resized:
                self.hl_gauss_layer = HLGaussLayer(dim = new_dim_critic, hl_gauss_loss = self.hl_gauss_kwargs).to(device = device, dtype = dtype)
                self.critic_readout = self.hl_gauss_layer
                self.critic_full = Sequential(self.critic, self.critic_readout)
            critic_out_linear = self.hl_gauss_layer.to_pred
        else:
            if critic_resized:
                self.critic_readout[0] = Linear(new_dim_critic, 1, device = device, dtype = dtype)
                self.critic_full = Sequential(self.critic, self.critic_readout)
            critic_out_linear = self.critic_readout[0]

        with torch.no_grad():
            if exists(self.readout.embeddings):
                if do_distill:
                    nn.init.normal_(self.readout.embeddings.weight[:, :keep_dim_actor], std = 1e-2)
                    if new_dim_actor > keep_dim_actor:
                        nn.init.normal_(self.readout.embeddings.weight[:, keep_dim_actor:], std = 1e-2 * cross_scale)
                else:
                    nn.init.normal_(self.readout.embeddings.weight, std = 1e-2)

            if do_distill:
                sparse_init_tensor_(critic_out_linear.weight[:, :keep_dim_critic], sparsity = core_sparsity)
                if new_dim_critic > keep_dim_critic:
                    sparse_init_tensor_(critic_out_linear.weight[:, keep_dim_critic:], sparsity = sparsity, scale = cross_scale)
            else:
                sparse_init_tensor_(critic_out_linear.weight, sparsity = sparsity)

            if exists(critic_out_linear.bias):
                critic_out_linear.bias.zero_()

        # distill core while masking gradients on the fresh plastic half

        if do_distill and epochs > 0:
            weight_masks = []
            with torch.no_grad():
                for lin, is_input, in_keep, out_keep in actor_lins + critic_lins:
                    mask = torch.zeros_like(lin.weight, dtype = torch.bool)
                    if is_input:
                        mask[:out_keep] = lin.weight[:out_keep] != 0
                    else:
                        mask[:out_keep, :in_keep] = lin.weight[:out_keep, :in_keep] != 0
                    weight_masks.append((lin, mask))

            with torch.enable_grad():
                distill_params = list(self.actor_with_readout.parameters()) + list(self.critic_full.parameters())
                optimizer = torch.optim.Adam(distill_params, lr = lr)

                num_samples = states.shape[0]
                eff_batch = min(batch_size, max(16, num_samples // 8)) if num_samples >= 32 else num_samples

                for _ in range(epochs):
                    perm = torch.randperm(num_samples, device = device)
                    for start in range(0, num_samples, eff_batch):
                        idx = perm[start:start + eff_batch]
                        sb = states[idx]

                        pred_dist = self.actor_with_readout(sb)
                        teacher_batch_dist = tree_map_tensor(lambda t: t[idx], teacher_dist)
                        loss_pi = _kl_mean(self.readout.kl_div(teacher_batch_dist, pred_dist))

                        if self.use_hl_gauss:
                            t_val = teacher_vals[idx].clamp(min = self.hl_gauss_kwargs['min_value'], max = self.hl_gauss_kwargs['max_value'])
                            loss_v = self.hl_gauss_layer(self.critic(sb), t_val)
                        else:
                            loss_v = F.mse_loss(self.critic_full(sb), teacher_vals[idx])

                        optimizer.zero_grad()
                        (loss_pi + loss_v).backward()

                        with torch.no_grad():
                            for lin, mask in weight_masks:
                                if exists(lin.weight.grad):
                                    lin.weight.grad.mul_(mask)

                            for lin, _, _, out_keep in actor_lins + critic_lins:
                                if exists(lin.bias) and exists(lin.bias.grad):
                                    lin.bias.grad[out_keep:].zero_()

                            for norm_param, n_keep in actor_norms + critic_norms:
                                if exists(norm_param.grad) and norm_param.shape[0] > n_keep:
                                    norm_param.grad[n_keep:].zero_()

                            if exists(self.readout.embeddings) and exists(self.readout.embeddings.weight.grad) and new_dim_actor > keep_dim_actor:
                                self.readout.embeddings.weight.grad[:, keep_dim_actor:].zero_()

                            if exists(critic_out_linear.weight.grad) and new_dim_critic > keep_dim_critic:
                                critic_out_linear.weight.grad[:, keep_dim_critic:].zero_()

                        optimizer.step()

            with torch.no_grad():
                if exists(self.readout.embeddings) and new_dim_actor > keep_dim_actor:
                    core_std = self.readout.embeddings.weight[:, :keep_dim_actor].std().clamp(min = 1e-2).item()
                    nn.init.normal_(self.readout.embeddings.weight[:, keep_dim_actor:], std = core_std * cross_scale)

        self._reset_after_reborn_(device)

        with torch.no_grad():
            if do_distill:
                post_dist = self.actor_with_readout(states)
                actor_kl = float(_kl_mean(self.readout.kl_div(teacher_dist, post_dist)).item())
                critic_loss = float(F.mse_loss(self.critic_full(states), teacher_vals).item())

                core_zero_fracs, right_zero_fracs = [], []
                for lin, is_input, in_keep, out_keep in actor_lins + critic_lins:
                    w_core = lin.weight[:out_keep, :] if is_input else lin.weight[:out_keep, :in_keep]
                    core_zero_fracs.append(float((w_core == 0).float().mean().item()))
                    if lin.out_features > out_keep:
                        right_zero_fracs.append(float((lin.weight[out_keep:] == 0).float().mean().item()))

                core_sp = float(sum(core_zero_fracs) / len(core_zero_fracs)) if len(core_zero_fracs) > 0 else 0.
                right_sp = float(sum(right_zero_fracs) / len(right_zero_fracs)) if len(right_zero_fracs) > 0 else 0.
            else:
                actor_kl, critic_loss = 0., 0.
                zero_fracs = [float((lin.weight == 0).float().mean().item()) for lin, _, _, _ in actor_lins + critic_lins]
                core_sp = right_sp = float(sum(zero_fracs) / len(zero_fracs)) if len(zero_fracs) > 0 else 0.

        return RebornMetrics(
            actor_kl = actor_kl,
            critic_loss = critic_loss,
            dim_actor = new_dim_actor,
            dim_critic = new_dim_critic,
            core_sparsity = core_sp,
            right_sparsity = right_sp
        )

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

        if self.reborn_buffer_size > 0:
            self._record_reborn_state_(state)

        normed_reward = self.reward_norm(reward, is_terminal = is_terminal, update = True)

        metrics = self._learn_step((state, action, normed_reward, next_state, is_terminal))

        self.update_step.add_(1)

        if (
            self.reborn_every > 0 and
            divisible_by(int(self.update_step.item()), self.reborn_every) and
            exists(self.reborn_states) and
            len(self.reborn_states) >= min(64, self.reborn_buffer_size)
        ):
            self.reborn_()

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

            @torch.no_grad()
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

                safe_delta = torch.where(td_error.detach() >= 0, 1.0, -1.0) * td_error.detach().abs().clamp(min = 1.0)
                critic_trace_grads = [(-grad / safe_delta) if exists(grad) else torch.zeros_like(p) for grad, p in zip(critic_trace_grads, critic_params)]
            else:
                critic_trace_grads = torch.autograd.grad(value_pred, critic_params, retain_graph=True, allow_unused=True)
                critic_trace_grads = [default(grad, torch.zeros_like(p)) for grad, p in zip(critic_trace_grads, critic_params)]

            value_grad = {
                name: grad
                for (name, _), grad in zip(self.critic_full.named_parameters(), critic_trace_grads)
            }

            # actor with entropy regularization

            actor_embed = self.actor(state)
            action_logits = self.readout(actor_embed)
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
                network_params,
                ema_to_projection = None
            ):
                # target representation

                if spr_module.target_embed_from_ema:
                    assert exists(network_ema), 'ema must be available if target_embed_from_ema is True'
                    next_embed = first(network_ema.ema_model)(next_state)
                else:
                    next_embed = network(next_state)

                # loss

                ssl_losses = spr_module(embed, action, next_embed, ema_to_projection = ema_to_projection)

                # gradients (use total loss for backprop)

                spr_params = list(spr_module.parameters())
                spr_grads_tup = torch.autograd.grad(ssl_losses.total, network_params + spr_params, retain_graph = True, allow_unused = True)
                spr_grads_tup = tuple(default(g, torch.zeros_like(p)) for g, p in zip(spr_grads_tup, network_params + spr_params))

                spr_grad = {name: grad for (name, _), grad in zip(network_with_readout.named_parameters(), spr_grads_tup[:len(network_params)])}
                spr_only_grad = list(spr_grads_tup[len(network_params):])

                # orth¹: project SPR encoder grads orthogonal to their own EMA momentum
                # this decorrelates the highly correlated streaming data

                enc_names = [n for n in spr_grad.keys() if n.startswith('0.')]
                enc_grads = [spr_grad[n] for n in enc_names]
                enc_moms = [spr_momentum[n] for n in enc_names]

                enc_grads_proj = orthog_project_tensors(enc_grads, enc_moms)

                aux_grads = spr_only_grad
                aux_moms = list(spr_aux_momentum.values())
                aux_grads_proj = orthog_project_tensors(aux_grads, aux_moms)

                # momentum update

                for grad, mom in zip([*enc_grads, *aux_grads], [*enc_moms, *aux_moms]):
                    mom.mul_(self.spr_orth_beta).add_(grad, alpha = 1. - self.spr_orth_beta)

                for name, proj_grad in zip(enc_names, enc_grads_proj):
                    spr_grad[name] = proj_grad

                spr_only_grad = aux_grads_proj

                # update SPR auxiliary networks via SGD with dedicated spr_lr

                for param, grad in zip(spr_params, spr_only_grad):
                    param.data.add_(grad, alpha = -self.spr_lr)

                # return recon-only loss for chunked-td (not contaminated by sigreg / action loss)

                return spr_grad, ssl_losses.recon.detach()

            # get the SPR loss

            actor_spr_grad = None
            actor_spr_loss = None
            if exists(self.ssl_type):
                actor_spr_grad, actor_spr_loss = get_spr_grad(
                    actor_embed,
                    self.actor,
                    self.actor_with_readout,
                    self.actor_with_readout_ema,
                    self.actor_spr,
                    self.actor_spr_momentum,
                    self.actor_spr_aux_momentum,
                    actor_params,
                    ema_to_projection = self.actor_spr_ema
                )

            # critic SPR loss

            critic_spr_grad = None
            critic_spr_loss = None
            if exists(self.ssl_type):
                critic_spr_grad, critic_spr_loss = get_spr_grad(
                    critic_embed,
                    self.critic,
                    self.critic_full,
                    self.critic_ema,
                    self.critic_spr,
                    self.critic_spr_momentum,
                    self.critic_spr_aux_momentum,
                    critic_params,
                    ema_to_projection = self.critic_spr_ema
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

        chunked_td_lambda = 1.

        if self.use_chunked_td and exists(critic_spr_loss):
            chunked_td_lambda = max(0., 1. - critic_spr_loss.item() / 2.)
            decay = decay * chunked_td_lambda

        for name, trace in self.actor_trace.items():
            trace.mul_(decay).add_(actor_grad[name] * gate)

        for name, trace in self.critic_trace.items():
            trace.mul_(decay).add_(value_grad[name])

        # overstepping-bounds gradient descent

        td_error_factor = td_error.abs().clamp(min = 1.)

        actor_grad_norm = stack([g.norm(p = 1) for g in actor_grad.values()]).mean()
        critic_grad_norm = stack([g.norm(p = 1) for g in value_grad.values()]).mean()

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

            rl_updates = {name: adapted_grads[name] * lr for name, _ in params.named_parameters()}

            if exists(spr_grads):
                enc_names = [n for n in spr_grads.keys() if n.startswith('0.')]
                enc_spr_updates = [-spr_grads[n] * self.spr_lr_param_update for n in enc_names]
                enc_rl_updates = [rl_updates[n] for n in enc_names]

                enc_spr_updates_proj = orthog_project_tensors(enc_spr_updates, enc_rl_updates)

                spr_updates_proj = dict(zip(enc_names, enc_spr_updates_proj))
            else:
                spr_updates_proj = {}

            updates = dict_add(rl_updates, spr_updates_proj)

            for name, param in params.named_parameters():
                update = updates[name] * global_scale

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

        if exists(self.ssl_type) and exists(self.actor_spr_ema):
            self.actor_spr_ema.update()

        if exists(self.ssl_type) and exists(self.critic_spr_ema):
            self.critic_spr_ema.update()

        return UpdateMetrics(
            td_error = td_error.item(),
            value_pred = value_pred.item(),
            actor_grad_norm = actor_grad_norm.item(),
            critic_grad_norm = critic_grad_norm.item(),
            actor_trace_norm = actor_trace_norm.item(),
            critic_trace_norm = critic_trace_norm.item(),
            actor_scale = actor_scale.item(),
            critic_scale = critic_scale.item(),
            chunked_td_lambda = chunked_td_lambda
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

def __getattr__(name: str):
    if name in (
        'AgentPool',
        'ArchiveEntry',
        'crossover_distill',
        'reproduce_lerp',
        'reproduce_shrink_perturb',
        'policy_kl_distance',
        'value_l1_distance'
    ):
        from streaming_deep_rl import agent_pool
        return getattr(agent_pool, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
