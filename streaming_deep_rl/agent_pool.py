from __future__ import annotations

import inspect
from collections import deque
from pathlib import Path
from typing import NamedTuple, Callable, Sequence

import torch
from torch import nn, is_tensor, Tensor
from torch.nn import Module, Linear

from torch_einops_utils import tree_map_tensor

from streaming_deep_rl.streaming_deep_rl import (
    StreamingACLambda,
    RebornMetrics,
    sparse_init_tensor_,
    exists,
    default,
    cast_tensor,
    _kl_mean
)

# types

class ArchiveEntry(NamedTuple):
    agent: StreamingACLambda
    fitness: float
    raw_states: Tensor | None
    tested_at: int

# reproduction & novelty operators

@torch.no_grad()
def crossover_distill(
    parent_a: StreamingACLambda,
    parent_b: StreamingACLambda,
    raw_states: Tensor,
    *,
    child: StreamingACLambda | None = None,
    filter_fn: str | Callable = 'critic',
    parent_fitnesses: Sequence[float] | Tensor | None = None,
    retain: float = 0.85,
    keep_ratio: float = 0.5,
    core_sparsity: float | None = None,
    **reborn_kwargs
) -> tuple[StreamingACLambda, RebornMetrics | None]:
    if exists(child):
        assert child is not parent_a and child is not parent_b, 'crossover child must not alias either parent'
    else:
        child = parent_a.clone(reset_traces = False)

    core_sparsity = default(core_sparsity, default(child.reborn_core_sparsity, min(0.70, child.reborn_sparsity)))

    metrics = child.reborn_(
        raw_states = raw_states,
        parents = (parent_a, parent_b),
        parent_filter = filter_fn,
        parent_fitnesses = parent_fitnesses,
        grow = False,
        keep_ratio = keep_ratio,
        core_sparsity = core_sparsity,
        retain = retain,
        **reborn_kwargs
    )

    return child, metrics

@torch.no_grad()
def reproduce_lerp(
    parent_a: StreamingACLambda,
    parent_b: StreamingACLambda | None = None,
    *,
    alpha = 0.5
) -> StreamingACLambda:
    child = parent_a.clone(reset_traces = False)

    if exists(parent_b):
        for pa, pb in zip(child.actor_with_readout.parameters(), parent_b.actor_with_readout.parameters()):
            if pa.shape == pb.shape:
                pa.lerp_(pb, alpha)
        for pa, pb in zip(child.critic_full.parameters(), parent_b.critic_full.parameters()):
            if pa.shape == pb.shape:
                pa.lerp_(pb, alpha)

    child._reset_after_reborn_(next(child.actor_with_readout.parameters()).device)
    return child

@torch.no_grad()
def reproduce_shrink_perturb(
    parent: StreamingACLambda,
    *,
    shrink = 0.8,
    perturb = 0.2
) -> StreamingACLambda:
    child = parent.clone(reset_traces = False)

    for m in (*child.actor.modules(), *child.critic.modules()):
        if isinstance(m, Linear):
            noise = sparse_init_tensor_(torch.empty_like(m.weight), sparsity = child.init_sparsity)
            m.weight.mul_(shrink).add_(noise, alpha = perturb)

    child._reset_after_reborn_(next(child.actor_with_readout.parameters()).device)
    return child

@torch.no_grad()
def policy_kl_distance(agent_a: StreamingACLambda, agent_b: StreamingACLambda, raw_states: Tensor) -> float:
    dev_a = raw_states.device
    dev_b = next(agent_b.actor_with_readout.parameters()).device
    dist_a = tree_map_tensor(lambda t: t.detach(), agent_a.actor_with_readout(agent_a.state_norm(raw_states, update = False)))
    dist_b = tree_map_tensor(lambda t: t.detach().to(dev_a), agent_b.actor_with_readout(agent_b.state_norm(raw_states.to(dev_b), update = False)))
    return float(_kl_mean(agent_a.readout.kl_div(dist_a, dist_b)).item())

@torch.no_grad()
def value_l1_distance(agent_a: StreamingACLambda, agent_b: StreamingACLambda, raw_states: Tensor) -> float:
    dev_a = raw_states.device
    dev_b = next(agent_b.critic_full.parameters()).device
    val_a = agent_a.critic_full(agent_a.state_norm(raw_states, update = False)).reshape(-1)
    val_b = agent_b.critic_full(agent_b.state_norm(raw_states.to(dev_b), update = False)).reshape(-1).to(dev_a)
    return float((val_a - val_b).abs().mean().item())

# single-stream evolutionary agent pool

class AgentPool(Module):
    _SCALAR_KEYS = (
        'active_count', 'active_index', 'dwell_eps_left',
        'total_steps', 'total_episodes', 'last_evolve_step', 'last_evolve_episode', '_spawn_count'
    )

    def __init__(
        self,
        agent: StreamingACLambda | Sequence[StreamingACLambda] | Callable[[], StreamingACLambda],
        *,
        pool_size = 4,
        archive_size = 8,
        archive_kl_threshold = 0.20,
        evolve_every = 1200,
        min_evolve_episodes = None,
        min_mature_steps = None,
        probe_buffer_size = 1024,
        ucb_decay = 0.90,
        ucb_c = 0.90,
        retain_temps = (0.85, 0.35, 0.10),
        actor_selector: str | Callable = 'ucb',
        reproduce_fn: str | Callable = 'crossover',
        crossover_filter: str | Callable = 'critic',
        novelty_fn: str | Callable = 'kl',
        fitness_fn: Callable | None = None
    ):
        super().__init__()
        assert pool_size >= 1
        assert archive_size >= 0
        assert 0. < ucb_decay < 1.
        assert len(retain_temps) >= 1

        self.pool_size = pool_size
        self.archive_size = archive_size
        self.archive_kl_threshold = archive_kl_threshold
        self.ucb_decay = ucb_decay
        self.ucb_c = ucb_c
        self.retain_temps = retain_temps

        self.actor_selector = actor_selector
        self.reproduce_fn = reproduce_fn
        self.crossover_filter = crossover_filter
        self.novelty_fn = novelty_fn
        self.fitness_fn = fitness_fn

        if isinstance(agent, StreamingACLambda):
            members = [agent] + [agent.clone() for _ in range(pool_size - 1)]
        elif callable(agent):
            members = [agent() for _ in range(pool_size)]
        else:
            members = list(agent)
            assert len(members) == pool_size

        self.agents = nn.ModuleList(members)
        self.active_count = 1
        self.active_index = 0
        self.slot_returns = [deque(maxlen = 100) for _ in range(pool_size)]
        self.slot_origin = ['founder'] + ['unused'] * (pool_size - 1)
        self.archive: list[ArchiveEntry] = []
        self.pool_events: list[dict] = []

        if pool_size == 1:
            self.evolve_every = 0
            self.min_evolve_episodes = 0
            self.min_mature_steps = 0
            return

        primary = self.agents[0]
        first_p = next(primary.actor_with_readout.parameters())

        self.evolve_every = evolve_every
        self.min_evolve_episodes = default(min_evolve_episodes, min(24, max(4, evolve_every // 150)))
        self.min_mature_steps = default(min_mature_steps, max(60, evolve_every // 2))

        self.probe_capacity = max(64, probe_buffer_size, primary.reborn_buffer_size)
        self.register_buffer('probe_buf', torch.zeros(self.probe_capacity, primary.dim_state, device = first_p.device, dtype = first_p.dtype), persistent = False)
        self.probe_ptr = 0
        self.probe_count = 0

        for m in self.agents:
            m.reborn_every = 0
            m.reborn_buffer_size = 0
            m.reborn_states = None

        self.register_buffer('fitness_sum', torch.zeros(pool_size))
        self.register_buffer('pull_count', torch.zeros(pool_size))
        self.register_buffer('slot_steps', torch.zeros(pool_size, dtype = torch.long))
        self.register_buffer('slot_episodes', torch.zeros(pool_size, dtype = torch.long))
        self.register_buffer('slot_trial_episodes', torch.full((pool_size,), 3, dtype = torch.long))

        self.dwell_eps_left = 0
        self.total_steps = 0
        self.total_episodes = 0
        self.last_evolve_step = 0
        self.last_evolve_episode = 0
        self._spawn_count = 0
        self._ep_steps = 0
        self._ep_return = 0.0
        self.recent_returns = deque(maxlen = 32)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name.startswith('_') or 'active_index' not in self.__dict__:
                raise
            modules = self.__dict__.get('_modules')
            if exists(modules) and 'agents' in modules and len(modules['agents']) > 0:
                idx = self.__dict__['active_index'] if self.__dict__.get('training', True) else self.best_index
                return getattr(modules['agents'][idx], name)
            raise

    @property
    def active_agent(self) -> StreamingACLambda:
        return self.agents[self.active_index]

    @property
    def best_index(self) -> int:
        if self.pool_size == 1 or self.active_count <= 1:
            return 0
        mature = [i for i in range(self.active_count) if len(self.slot_returns[i]) >= 10]
        if len(mature) > 0:
            return max(mature, key = lambda i: sum(self.slot_returns[i]) / len(self.slot_returns[i]))
        return int(self._slot_means().argmax().item())

    @property
    def best_agent(self) -> StreamingACLambda:
        return self.agents[self.best_index]

    @property
    def current_agent(self) -> StreamingACLambda:
        return self.active_agent if self.training else self.best_agent

    def best_rolling_reward(self, min_episodes = 20) -> float | None:
        if self.pool_size == 1:
            return None
        eligible = [sum(r) / len(r) for r in self.slot_returns[:self.active_count] if len(r) >= min_episodes]
        return float(max(eligible)) if len(eligible) > 0 else None

    def status(self) -> list[dict]:
        means = self._slot_means().tolist() if self.pool_size > 1 else [0.0]
        return [
            dict(
                slot = i,
                origin = self.slot_origin[i],
                episodes = int(self.slot_episodes[i]) if self.pool_size > 1 else len(self.slot_returns[0]),
                steps = int(self.slot_steps[i]) if self.pool_size > 1 else 0,
                fitness = round(means[i], 2),
                rolling_reward = round(sum(self.slot_returns[i]) / len(self.slot_returns[i]), 2) if len(self.slot_returns[i]) > 0 else None
            )
            for i in range(self.active_count)
        ]

    def _device(self) -> torch.device:
        return next(self.agents[0].actor_with_readout.parameters()).device

    def _log(self, event: str, **kw):
        self.pool_events.append(dict(step = self.total_steps, event = event, archive_len = len(self.archive), **kw))

    def _slot_means(self) -> Tensor:
        n = self.active_count
        pulls = self.pull_count[:n]
        fallback = float(sum(self.recent_returns) / len(self.recent_returns)) if len(self.recent_returns) > 0 else 0.0
        return torch.where(pulls > 1e-6, self.fitness_sum[:n] / pulls.clamp(min = 1e-6), fallback)

    def _mean_fit(self, idx: int) -> float:
        return float(self._slot_means()[idx].item())

    @torch.no_grad()
    def _record_probe_(self, state: Tensor):
        rows = cast_tensor(state, dtype = self.probe_buf.dtype, device = self.probe_buf.device).detach().reshape(-1, self.probe_buf.shape[-1])
        for row in rows:
            self.probe_buf[self.probe_ptr].copy_(row)
            self.probe_ptr = (self.probe_ptr + 1) % self.probe_capacity
            self.probe_count = min(self.probe_capacity, self.probe_count + 1)

    def _sample_probes(self, max_n = 512) -> Tensor | None:
        if self.probe_count == 0:
            return None
        n = min(max_n, self.probe_count)
        idx = (self.probe_ptr - n + torch.arange(n, device = self.probe_buf.device)) % self.probe_capacity
        return self.probe_buf[idx]

    def _novelty(self, agent_a: StreamingACLambda, agent_b: StreamingACLambda, probes: Tensor) -> float:
        if callable(self.novelty_fn):
            return float(self.novelty_fn(agent_a, agent_b, probes))
        if self.novelty_fn == 'kl':
            return policy_kl_distance(agent_a, agent_b, probes)
        if self.novelty_fn == 'value':
            return value_l1_distance(agent_a, agent_b, probes)
        raise ValueError(f'unknown novelty_fn {self.novelty_fn!r} - choose from "kl", "value", or a callable')

    def _reset_slot_(self, slot_idx: int, *, trial_episodes: int, origin: str):
        self.fitness_sum[slot_idx] = 0.0
        self.pull_count[slot_idx] = 0.0
        self.slot_steps[slot_idx] = 0
        self.slot_episodes[slot_idx] = 0
        self.slot_trial_episodes[slot_idx] = trial_episodes
        self.slot_returns[slot_idx].clear()
        self.slot_origin[slot_idx] = origin

    @torch.no_grad()
    def _reproduce(
        self,
        parents: Sequence[StreamingACLambda],
        fitnesses: Sequence[float],
        retain: float
    ) -> tuple[StreamingACLambda, bool]:
        probes = self._sample_probes(512)
        has_probes = exists(probes) and probes.shape[0] >= 32
        two_parent = len(parents) >= 2

        if callable(self.reproduce_fn):
            n_params = len(inspect.signature(self.reproduce_fn).parameters)
            args = (parents, probes, fitnesses, retain) if n_params >= 4 else (parents, probes, fitnesses)
            return self.reproduce_fn(*args), two_parent

        pa = parents[0]
        pb = parents[1] if two_parent else None
        keep_ratio = 0.35 if retain < 0.2 else 0.50
        core_sp = default(pa.reborn_core_sparsity, min(0.70, pa.reborn_sparsity))

        if self.reproduce_fn == 'crossover':
            if two_parent and has_probes:
                child, _ = crossover_distill(
                    pa, pb, probes,
                    filter_fn = self.crossover_filter,
                    parent_fitnesses = fitnesses,
                    retain = retain,
                    keep_ratio = keep_ratio,
                    core_sparsity = core_sp
                )
                return child, True

            child = pa.clone(reset_traces = True)
            if has_probes:
                child.reborn_(raw_states = probes, grow = False, keep_ratio = keep_ratio, core_sparsity = core_sp, retain = retain)
            return child, False

        if self.reproduce_fn == 'reborn':
            child = pa.clone(reset_traces = True)
            if has_probes:
                child.reborn_(raw_states = probes, grow = False, keep_ratio = keep_ratio, core_sparsity = core_sp, retain = retain)
            return child, False

        if self.reproduce_fn == 'lerp':
            return reproduce_lerp(pa, pb), two_parent

        if self.reproduce_fn == 'shrink_perturb':
            return reproduce_shrink_perturb(pa, shrink = retain, perturb = 1. - retain), False

        raise ValueError(f'unknown reproduce_fn {self.reproduce_fn!r} - choose from "crossover", "reborn", "lerp", "shrink_perturb", or a callable')

    @torch.no_grad()
    def _spawn_into_slot_(
        self,
        slot_idx: int,
        parents: Sequence[StreamingACLambda],
        fitnesses: Sequence[float],
        *,
        retain = 0.85,
        event_tag = 'offspring'
    ):
        child, used_two_parent = self._reproduce(parents, fitnesses, retain)
        self.agents[slot_idx] = child.to(self._device())
        tag = self.reproduce_fn if isinstance(self.reproduce_fn, str) else getattr(self.reproduce_fn, '__name__', 'custom')
        self._reset_slot_(slot_idx, trial_episodes = 3, origin = f'{tag}({retain:.2f})')
        self._log(event_tag, slot = slot_idx, retain = retain, two_parent = used_two_parent)

    @torch.no_grad()
    def _snapshot_elite(self, elite: StreamingACLambda, fitness: float, tested_at: int) -> ArchiveEntry:
        probes = self._sample_probes(512)
        snap_agent = elite.clone(reset_traces = False, device = 'cpu')
        snap_agent.reset_trace_()
        snap_states = probes.detach().cpu().clone() if exists(probes) and probes.shape[0] >= 32 else None
        return ArchiveEntry(agent = snap_agent, fitness = float(fitness), raw_states = snap_states, tested_at = tested_at)

    @torch.no_grad()
    def _maybe_archive_elite_(self, elite_idx: int):
        if self.archive_size <= 0 or self.probe_count < 64:
            return
        if int(self.slot_steps[elite_idx]) < self.min_mature_steps or float(self.pull_count[elite_idx]) < 1.5:
            return

        elite = self.agents[elite_idx]
        elite_fit = self._mean_fit(elite_idx)

        if len(self.recent_returns) >= 8:
            sorted_rets = sorted(self.recent_returns)
            if elite_fit < sorted_rets[int(0.40 * (len(sorted_rets) - 1))]:
                return

        probes = self._sample_probes(256)
        if not exists(probes):
            return

        for idx_a, entry in enumerate(self.archive):
            if self._novelty(elite, entry.agent, probes) < self.archive_kl_threshold:
                if elite_fit >= entry.fitness * 0.95:
                    self.archive[idx_a] = self._snapshot_elite(elite, max(elite_fit, entry.fitness), entry.tested_at)
                return

        if len(self.archive) >= self.archive_size:
            self.archive.pop(0)

        self.archive.append(self._snapshot_elite(elite, elite_fit, self.total_steps))
        self._log('archive_save', slot = elite_idx, fitness = elite_fit)

    @torch.no_grad()
    def _reintroduce_archive_(self, archive_idx: int, slot_idx: int, consolidate = True):
        entry = self.archive[archive_idx]
        device = self._device()
        restored = entry.agent.clone(reset_traces = True, device = device)

        if consolidate and exists(entry.raw_states) and entry.raw_states.shape[0] >= 32:
            core_sp = default(restored.reborn_core_sparsity, min(0.70, restored.reborn_sparsity))
            restored.reborn_(raw_states = entry.raw_states.to(device), grow = False, core_sparsity = core_sp, retain = 0.95)

        self.agents[slot_idx] = restored
        self.archive[archive_idx] = entry._replace(tested_at = self.total_steps)
        self._reset_slot_(slot_idx, trial_episodes = 1, origin = 'archive')
        self._log('reintroduce_archive', slot = slot_idx, archive_idx = archive_idx)

    def select_actor(self) -> int:
        if self.dwell_eps_left > 0 and self.active_index < self.active_count:
            self.dwell_eps_left -= 1
            return self.active_index

        n = self.active_count
        if n <= 1:
            return 0

        for i in range(n):
            if int(self.slot_episodes[i]) == 0:
                self.dwell_eps_left = max(0, int(self.slot_trial_episodes[i]) - 1)
                return i

        means = self._slot_means()
        pulls = self.pull_count[:n]
        std = means.std(correction = 0)
        norm_means = (means - means.mean()) / (std + 1e-5) if std > 1e-4 else torch.zeros_like(means)

        if callable(self.actor_selector):
            out = self.actor_selector(means, pulls)
            chosen = int(out.argmax().item()) if is_tensor(out) and out.ndim > 0 else int(out)
        elif self.actor_selector == 'ucb':
            scores = norm_means + self.ucb_c * ((pulls.sum() + 2.).log() / (pulls + 0.25)).sqrt()
            chosen = int(scores.argmax().item())
        elif self.actor_selector == 'softmax':
            chosen = int(torch.multinomial(norm_means.softmax(dim = -1), 1).item())
        elif self.actor_selector == 'round_robin':
            chosen = (self.active_index + 1) % n
        elif self.actor_selector == 'uniform':
            chosen = int(torch.randint(0, n, ()).item())
        else:
            raise ValueError(f'unknown actor_selector {self.actor_selector!r} - choose from "ucb", "softmax", "round_robin", "uniform", or a callable')

        self.dwell_eps_left = 1 if int(self.slot_steps[chosen]) < self.min_mature_steps else 0
        return chosen

    def evolve_(self):
        if self.pool_size == 1:
            return

        # 1. initial budding once founder matures

        if self.active_count < self.pool_size:
            if int(self.slot_steps[0]) >= self.min_mature_steps and int(self.slot_episodes[0]) >= max(2, self.min_evolve_episodes // 2):
                self.last_evolve_step = self.total_steps
                self.last_evolve_episode = self.total_episodes
                fit0 = self._mean_fit(0)
                for offset, slot_idx in enumerate(range(self.active_count, self.pool_size)):
                    ret_val = self.retain_temps[offset % len(self.retain_temps)]
                    self._spawn_into_slot_(slot_idx, (self.agents[0],), (fit0,), retain = ret_val, event_tag = 'bud_pool')
                self.active_count = self.pool_size
            return

        # 2. rank active individuals; archive distinct elite; protect top elite & long-horizon champion

        means = self._slot_means()
        ranked = means.argsort(descending = True).tolist()
        elite_idx = ranked[0]
        self._maybe_archive_elite_(elite_idx)

        protected = {elite_idx, self.best_index}
        if len(protected) < 2 and self.active_count >= 2:
            protected.add(ranked[1])

        cull_candidates = [
            i for i in reversed(ranked)
            if i not in protected and int(self.slot_episodes[i]) >= int(self.slot_trial_episodes[i])
        ]
        if len(cull_candidates) == 0:
            return

        self.last_evolve_step = self.total_steps
        self.last_evolve_episode = self.total_episodes

        # 3. slot A: reproduce from top-2 viable elites (or single elite right after shift)

        ret_choice = self.retain_temps[self._spawn_count % len(self.retain_temps)]
        self._spawn_count += 1

        fit_a, runner_idx = float(means[elite_idx]), ranked[1]
        fit_b = float(means[runner_idx])
        floor = min(self.recent_returns) if len(self.recent_returns) > 0 else min(fit_a, fit_b)
        viable_second = int(self.slot_episodes[runner_idx]) >= 2 and (fit_b - floor) >= 0.50 * max(0.0, fit_a - floor)

        parents = (self.agents[elite_idx], self.agents[runner_idx]) if viable_second else (self.agents[elite_idx],)
        fits = (fit_a, fit_b) if viable_second else (fit_a,)
        self._spawn_into_slot_(
            cull_candidates[0],
            parents,
            fits,
            retain = ret_choice,
            event_tag = 'two_parent_crossover' if viable_second else 'single_parent_reborn'
        )

        # 4. slot B: recall behaviorally distinct archived specialist

        if len(cull_candidates) >= 2 and len(self.archive) > 0 and self.probe_count >= 64:
            probes = self._sample_probes(256)
            elite = self.agents[elite_idx]
            if exists(probes):
                distinct = [(i, e) for i, e in enumerate(self.archive) if self._novelty(elite, e.agent, probes) >= self.archive_kl_threshold]
                if len(distinct) > 0:
                    cycle = max(1, len(distinct)) * max(1, self.evolve_every)
                    fresh = [(i, e) for i, e in distinct if (self.total_steps - e.tested_at) >= cycle]
                    picked = max(fresh, key = lambda x: x[1].fitness)[0] if len(fresh) > 0 else min(distinct, key = lambda x: x[1].tested_at)[0]
                    self._reintroduce_archive_(picked, cull_candidates[1], consolidate = True)

    _evolve_ = evolve_

    def _finish_episode_(self):
        k = self.active_index
        ep_ret, ep_steps = self._ep_return, self._ep_steps
        self._ep_return = 0.0
        self._ep_steps = 0

        self.agents[k].reward_norm.running_reward.zero_()
        self.agents[k].reset_trace_()

        score = float(self.fitness_fn(ep_ret, ep_steps, self.agents[k])) if callable(self.fitness_fn) else ep_ret

        self.total_episodes += 1
        self.slot_episodes[k] += 1
        self.slot_returns[k].append(ep_ret)
        self.recent_returns.append(score)

        self.fitness_sum[:self.active_count].mul_(self.ucb_decay)
        self.pull_count[:self.active_count].mul_(self.ucb_decay)
        self.fitness_sum[k] += score
        self.pull_count[k] += 1.0

        if (
            self.evolve_every > 0 and
            (self.total_steps - self.last_evolve_step) >= self.evolve_every and
            (self.total_episodes - self.last_evolve_episode) >= self.min_evolve_episodes
        ):
            self.evolve_()

        self.active_index = self.select_actor()

    def reset_trace_(self):
        if self.pool_size == 1:
            self.agents[0].reset_trace_()
            return

        if self._ep_steps > 0:
            self._finish_episode_()

        self.active_agent.reward_norm.running_reward.zero_()
        self.active_agent.reset_trace_()

    def update(self, state, action, next_state, reward, is_terminal = False):
        if self.pool_size == 1:
            return self.agents[0].update(state, action, next_state, reward, is_terminal = is_terminal)

        self._record_probe_(state)
        metrics = self.active_agent.update(state, action, next_state, reward, is_terminal = is_terminal)
        is_term = bool(is_terminal.item()) if is_tensor(is_terminal) else bool(is_terminal)

        self.total_steps += 1
        self._ep_steps += 1
        self.slot_steps[self.active_index] += 1
        self._ep_return += float(reward.item()) if is_tensor(reward) else float(reward)

        if is_term:
            self._finish_episode_()
            self.active_agent.reward_norm.running_reward.zero_()
            self.active_agent.reset_trace_()

        return metrics

    def reborn_(self, states: Tensor | None = None, **kwargs):
        if self.pool_size == 1:
            return self.agents[0].reborn_(states, **kwargs)

        probes = self._sample_probes(512) if not exists(states) else None
        ranked = self._slot_means().argsort(descending = True).tolist()

        if self.active_count < 2:
            return self.agents[0].reborn_(states, raw_states = probes, **kwargs)

        p0, p1 = ranked[0], ranked[1]
        pa, pb = self.agents[p0].clone(reset_traces = False), self.agents[p1].clone(reset_traces = False)
        fits = (self._mean_fit(p0), self._mean_fit(p1))

        metrics = self.agents[p0].reborn_(
            states,
            raw_states = probes,
            parents = (pa, pb),
            parent_filter = self.crossover_filter,
            parent_fitnesses = fits,
            **kwargs
        )
        for idx in ranked[1:]:
            self.agents[idx].reborn_(states, raw_states = probes, **kwargs)

        return metrics

    @torch.no_grad()
    def forward_value(self, state):
        return self.current_agent.forward_value(state)

    def sample_action(self, action_dist):
        return self.current_agent.sample_action(action_dist)

    @torch.no_grad()
    def forward_action(self, state, use_ema = True, sample = False):
        return self.current_agent.forward_action(state, use_ema = use_ema, sample = sample)

    @torch.no_grad()
    def forward(self, state, sample = False):
        return self.forward_action(state, sample = sample)

    def save(self, path, overwrite = True):
        if self.pool_size == 1:
            return self.agents[0].save(path, overwrite = overwrite)

        path = Path(path)
        assert overwrite or not path.exists(), f'File {path} already exists'
        torch.save(dict(
            state_dict = self.state_dict(),
            scalars = {k: getattr(self, k) for k in self._SCALAR_KEYS},
            slot_returns = [list(d) for d in self.slot_returns],
            slot_origin = self.slot_origin
        ), str(path))

    def load(self, path):
        if self.pool_size == 1:
            return self.agents[0].load(path)

        path = Path(path)
        assert path.exists(), f'File {path} does not exist'
        payload = torch.load(str(path), weights_only = False)
        self.load_state_dict(payload['state_dict'], strict = False)
        for k, v in payload.get('scalars', {}).items():
            setattr(self, k, v)
        if 'slot_returns' in payload:
            self.slot_returns = [deque(r, maxlen = 100) for r in payload['slot_returns']]
        if 'slot_origin' in payload:
            self.slot_origin = list(payload['slot_origin'])
