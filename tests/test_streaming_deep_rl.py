import pytest
param = pytest.mark.parametrize

import torch
import torch.nn.functional as F

from x_mlps_pytorch.normed_mlp import MLP
from x_mlps_pytorch.residual_normed_mlp import ResidualNormedMLP

from streaming_deep_rl import (
    StreamingACLambda,
    SplitNorm,
    AgentPool,
    crossover_distill,
    reproduce_shrink_perturb,
    sparse_init_
)
from streaming_deep_rl.streaming_deep_rl import orthog_project_tensors

# helper

def make_agent(
    dim_state = 4,
    dim = 32,
    discrete = 2,
    continuous = 0,
    use_residual = False,
    **kwargs
):
    if use_residual:
        actor = ResidualNormedMLP(dim, dim_in = dim_state, depth = 2, residual_every = 1)
        critic = ResidualNormedMLP(dim, dim_in = dim_state, depth = 2, residual_every = 1)
    else:
        actor = MLP(dim_state, dim, dim, norm_elementwise_affine = False, activate_last = True)
        critic = MLP(dim_state, dim, dim, norm_elementwise_affine = False, activate_last = True)

    return StreamingACLambda(
        actor = actor,
        critic = critic,
        dim_state = dim_state,
        dim_actor = dim,
        dim_critic = dim,
        num_discrete_actions = discrete,
        num_continuous_actions = continuous,
        **kwargs
    )

# core streaming ac(lambda)

@param('ssl_type', (None, 'spr', 'lapo'))
@param('spr_target_embed_from_ema', (False, True))
@param('spr_use_sem', (False, True))
@param('spr_sigreg_num_subspaces', (1, 16))
def test_streaming(
    ssl_type,
    spr_target_embed_from_ema,
    spr_use_sem,
    spr_sigreg_num_subspaces
):
    actor = MLP(5, 128, norm_elementwise_affine = False, activate_last = False)
    critic = MLP(5, 128, norm_elementwise_affine = False, activate_last = False)

    ssl_kwargs = dict(target_embed_from_ema = spr_target_embed_from_ema)

    if ssl_type == 'spr':
        ssl_kwargs.update(
            use_sem = spr_use_sem,
            sigreg_weight = 1.0 if not spr_target_embed_from_ema else 0.0,
            sigreg_num_subspaces = spr_sigreg_num_subspaces
        )
    elif ssl_type == 'lapo':
        ssl_kwargs.update(pred_action_loss_weight = 1.0)

    agent = StreamingACLambda(
        actor = actor,
        critic = critic,
        num_continuous_actions = 1,
        dim_state = 5,
        dim_actor = 128,
        dim_critic = 128,
        ssl_type = ssl_type,
        spr_kwargs = ssl_kwargs if ssl_type == 'spr' else dict(),
        lapo_kwargs = ssl_kwargs if ssl_type == 'lapo' else dict()
    )

    agent.reset_trace_()

    state = torch.randn(5)
    action_dist_params = agent(state)
    value = agent.forward_value(state)

    assert action_dist_params.shape == (1, 2)
    assert value.shape == ()

    actions = agent.sample_action(action_dist_params)
    agent.update(state, actions, state, 1.)

    agent.save('./test_streaming_ckpt.pt')
    agent.load('./test_streaming_ckpt.pt')

def test_sparse_init_small_fan_in():
    for fan_in in (1, 2, 4, 8):
        layer = torch.nn.Linear(fan_in, 64)
        sparse_init_(layer, sparsity = 0.9)
        assert (layer.weight != 0).any(dim = -1).all()

def test_orthog_project_tensors():
    grads = [torch.tensor([1., 0.]), torch.tensor([0., 1.])]
    moms = [torch.tensor([1., 0.]), torch.tensor([0., 0.])]

    projected = orthog_project_tensors(grads, moms)

    assert torch.allclose(projected[0], torch.zeros(2))
    assert torch.allclose(projected[1], torch.tensor([0., 1.]))

# continual plasticity: reborn & crossover

@param('grow', (False, True))
@param('discrete', (True, False))
@param('use_residual', (False, True))
@param('retain', (0., 1.))
def test_reborn(grow, discrete, use_residual, retain):
    torch.manual_seed(0)

    agent = make_agent(
        dim_state = 6,
        dim = 64,
        discrete = 4 if discrete else 0,
        continuous = 0 if discrete else 2,
        use_residual = use_residual,
        reborn_buffer_size = 128,
        reborn_epochs = 8,
        reborn_grow = grow,
        reborn_grow_factor = 1.5,
        reborn_max_dim = 96
    )

    state = torch.randn(6)
    for _ in range(96):
        action, _ = agent(state, sample = True)
        next_state = torch.randn(6)
        agent.update(state, action, next_state, 1., is_terminal = False)
        state = next_state

    with torch.no_grad():
        metrics = agent.reborn_(retain = retain)

    expected_dim = 96 if grow else 64
    assert metrics is not None
    assert agent.dim_actor == expected_dim
    assert metrics.right_sparsity > 0.75
    assert any(isinstance(m, SplitNorm) for m in agent.actor.modules())

    if retain == 1.:
        assert metrics.actor_kl < 0.05
        assert metrics.critic_loss < 0.05

    action, _ = agent(state, sample = True)
    agent.update(state, action, torch.randn(6), 0.5, is_terminal = False)
    keep_dim = expected_dim // 2
    assert sum(t[keep_dim:].abs().sum().item() for t in agent.actor_trace.values() if t.ndim == 2 and t.shape[0] == expected_dim) > 0.

@param('filter_fn', ('critic', 'entropy', 'fitness', 'random'))
def test_crossover_distill(filter_fn):
    torch.manual_seed(0)

    pa = make_agent(use_critic_ema = False, reborn_epochs = 40, reborn_lr = 1e-2)
    pb = make_agent(use_critic_ema = False, reborn_epochs = 40, reborn_lr = 1e-2)

    probes = torch.randn(128, 4)
    probes[:64, 0] = -torch.rand(64) - 0.25
    probes[64:, 0] = torch.rand(64) + 0.25

    opt_a = torch.optim.Adam([*pa.actor_with_readout.parameters(), *pa.critic_full.parameters()], lr = 2e-2)
    opt_b = torch.optim.Adam([*pb.actor_with_readout.parameters(), *pb.critic_full.parameters()], lr = 2e-2)

    for _ in range(80):
        opt_a.zero_grad()
        (F.cross_entropy(pa.actor_with_readout(probes), torch.zeros(128, dtype = torch.long)) +
         F.mse_loss(pa.critic_full(probes), torch.where(probes[:, 0] < 0, 1., -1.))).backward()
        opt_a.step()

        opt_b.zero_grad()
        (F.cross_entropy(pb.actor_with_readout(probes), torch.ones(128, dtype = torch.long)) +
         F.mse_loss(pb.critic_full(probes), torch.where(probes[:, 0] > 0, 1., -1.))).backward()
        opt_b.step()

    child, metrics = crossover_distill(
        pa, pb, probes,
        filter_fn = filter_fn,
        parent_fitnesses = (1., 2.),
        retain = 1.
    )
    assert metrics is not None and child is not pa and child is not pb

    if filter_fn == 'critic':
        preds = child.actor_with_readout(child.state_norm(probes, update = False)).argmax(dim = -1)
        assert (preds == (probes[:, 0] > 0).long()).float().mean().item() > 0.90

    # hybrid discrete + continuous readout
    ha, hb = make_agent(discrete = 2, continuous = 2), make_agent(discrete = 2, continuous = 2)
    hc, hm = crossover_distill(ha, hb, probes[:32], filter_fn = filter_fn, parent_fitnesses = (1., 2.), epochs = 2)
    assert hm is not None and hc is not ha

# single-stream agent pool

def test_agent_pool_e2e(tmp_path):
    # 1. pool_size = 1 is an exact zero-overhead pass-through

    torch.manual_seed(42)
    ref = make_agent()
    torch.manual_seed(42)
    single = AgentPool(make_agent(), pool_size = 1)

    states = [torch.randn(4) for _ in range(30)]
    torch.manual_seed(99)
    for i, s in enumerate(states):
        a, _ = ref(s, sample = True)
        ref.update(s, a, states[(i + 1) % 30], 1., is_terminal = (i == 14))
        if i == 14:
            ref.reset_trace_()

    torch.manual_seed(99)
    for i, s in enumerate(states):
        a, _ = single(s, sample = True)
        single.update(s, a, states[(i + 1) % 30], 1., is_terminal = (i == 14))
        if i == 14:
            single.reset_trace_()

    for (_, p_ref), (_, p_pool) in zip(ref.state_dict().items(), single.agents[0].state_dict().items()):
        assert torch.equal(p_ref, p_pool)

    # 2. pool_size = 4 non-stationary survival: bud -> crossover -> archive -> recall

    torch.manual_seed(7)
    pool = AgentPool(
        lambda: make_agent(discount_factor = 0.96, use_critic_ema = False, reborn_buffer_size = 256, reborn_epochs = 8),
        pool_size = 4,
        archive_size = 4,
        evolve_every = 150,
        min_mature_steps = 50
    )

    def run_phase(steps, good_action):
        s = torch.tensor([1., -0.5, 0.25, -0.25]) + 0.05 * torch.randn(4)
        pool.reset_trace_()
        ep_len, bad_streak, returns = 0, 0, []
        for _ in range(steps):
            a, _ = pool(s, sample = True)
            ep_len += 1
            bad_streak = 0 if int(a.item()) == good_action else bad_streak + 1
            term = bad_streak >= 2 or ep_len >= 50
            trunc = ep_len >= 50 and bad_streak < 2
            ns = torch.tensor([1., -0.5, 0.25, -0.25]) + 0.05 * torch.randn(4)
            pool.update(s, a, ns, 1., is_terminal = (term and not trunc))
            if term:
                returns.append(float(ep_len))
                ep_len, bad_streak = 0, 0
                pool.reset_trace_()
            s = ns
        return returns

    assert max(run_phase(700, good_action = 1)[-5:]) > 35.
    assert max(run_phase(750, good_action = 0)[-5:]) > 35.
    assert max(run_phase(500, good_action = 1)[-4:]) > 35.

    events = {ev['event'] for ev in pool.pool_events}
    assert {'bud_pool', 'two_parent_crossover', 'reintroduce_archive'}.issubset(events)

    test_s = torch.randn(4)
    pool.eval()
    assert torch.allclose(pool(test_s), pool.best_agent(test_s))
    pool.train()

    ckpt = tmp_path / 'pool.pt'
    pool.save(ckpt)
    pool.load(ckpt)

# researcher hooks: string presets & custom callables at a single glance

@param('actor_selector, reproduce_fn, crossover_filter, novelty_fn', (
    ('softmax', 'crossover', 'entropy', 'value'),
    ('round_robin', 'lerp', 'fitness', 'kl'),
    ('uniform', 'shrink_perturb', 'random', 'value'),
    ('ucb', 'reborn', 'critic', 'kl')
))
def test_agent_pool_preset_hooks(actor_selector, reproduce_fn, crossover_filter, novelty_fn):
    torch.manual_seed(11)

    pool = AgentPool(
        lambda: make_agent(dim = 16, discrete = 3, reborn_epochs = 2),
        pool_size = 3,
        archive_size = 3,
        evolve_every = 40,
        min_evolve_episodes = 2,
        min_mature_steps = 20,
        actor_selector = actor_selector,
        reproduce_fn = reproduce_fn,
        crossover_filter = crossover_filter,
        novelty_fn = novelty_fn
    )

    s = torch.randn(4)
    for step in range(120):
        a, _ = pool(s, sample = True)
        ns = torch.randn(4)
        pool.update(s, a, ns, 1., is_terminal = ((step + 1) % 10 == 0))
        s = ns

    assert pool.active_count == 3
    assert len(pool.status()) == 3

def test_agent_pool_custom_hooks_e2e():
    torch.manual_seed(3)

    # 1. actor election: (means, pulls) -> scores or slot index
    noisy_bandit = lambda means, pulls: means + torch.randn_like(means) / (pulls + 1.).sqrt()

    # 2. crossover state filter: (probes, dists, vals, fitnesses) -> 0/1 parent choice per state
    margin_filter = lambda probes, dists, vals, fits: (vals[1] > vals[0] + 0.05).long()

    # 3. reproduction operator: (parents, probes, fitnesses, retain) -> child agent
    def custom_reproduce(parents, probes, fitnesses, retain):
        if len(parents) >= 2 and probes is not None and probes.shape[0] >= 32:
            child, _ = crossover_distill(parents[0], parents[1], probes, filter_fn = margin_filter, retain = retain, epochs = 4)
            return child
        return reproduce_shrink_perturb(parents[0], shrink = retain, perturb = 1. - retain)

    # 4. archive novelty distance: (agent_a, agent_b, probes) -> float
    def prob_l1_novelty(a, b, probes):
        pa = a.actor_with_readout(a.state_norm(probes, update = False)).softmax(dim = -1)
        pb = b.actor_with_readout(b.state_norm(probes, update = False)).softmax(dim = -1)
        return (pa - pb).abs().mean().item()

    # 5. fitness objective: (ep_return, ep_steps, agent) -> float
    speed_fitness = lambda ep_return, ep_steps, agent: ep_return + 0.1 * ep_steps

    pool = AgentPool(
        make_agent(reborn_epochs = 4),
        pool_size = 3,
        archive_size = 4,
        archive_kl_threshold = 0.02,
        evolve_every = 80,
        min_evolve_episodes = 3,
        min_mature_steps = 30,
        actor_selector = noisy_bandit,
        reproduce_fn = custom_reproduce,
        novelty_fn = prob_l1_novelty,
        fitness_fn = speed_fitness
    )

    state = torch.tensor([1., -0.5, 0.25, -0.25])
    for step in range(320):
        action, _ = pool(state, sample = True)
        reward = 1. if int(action.item()) == 1 else -0.2
        done = (step + 1) % 20 == 0
        next_state = torch.tensor([1., -0.5, 0.25, -0.25]) + 0.05 * torch.randn(4)
        pool.update(state, action, next_state, reward, is_terminal = done)
        state = next_state

    assert any('custom_reproduce' in row['origin'] for row in pool.status())

    pool.evolve_()
    pool.eval()
    assert pool(state).shape == (2,)
