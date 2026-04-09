# /// script
# dependencies = [
#   "accelerate",
#   "torch",
#   "einops",
#   "gymnasium[box2d,other]",
#   "fire",
#   "tqdm",
#   "x-mlps-pytorch>=0.0.8",
#   "einx",
#   "rich",
#   "wandb"
# ]
# ///

from __future__ import annotations

import os
import torch
import numpy as np
import fire
from collections import deque
from shutil import rmtree

from torch import tensor, nn

import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.console import Console, Group
from rich import box
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    MofNCompleteColumn
)

from accelerate import Accelerator

from streaming_deep_rl.streaming_deep_rl import StreamingACLambda
from x_mlps_pytorch.residual_normed_mlp import ResidualNormedMLP

# constants

VIDEO_FOLDER = './lunar-video-streaming'

# helpers

def exists(v):
    return v is not None

def divisible_by(num, den):
    return (num % den) == 0

# UI implementation

class Dashboard:
    def __init__(self, num_episodes, hyperparams=None):
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            expand = True
        )
        self.pbar_task = self.progress.add_task("Episodes", total = num_episodes)

        self.hyperparams = hyperparams or {}

        self.episode_info = {
            "avg_cum_reward_100": 0.0,
            "avg_steps_100": 0.0,
            "last_eps_reward": 0.0,
            "last_eps_steps": 0,
            "td_error": "0.0000",
            "value_pred": "0.0000",
            "actor_grad": "0.0000",
            "actor_norm": "0.0000",
            "actor_scale": "0.0000",
            "critic_grad": "0.0000",
            "critic_norm": "0.0000",
            "critic_scale": "0.0000"
        }

    def update_episode_info(self, **kwargs):
        self.episode_info.update(kwargs)

    def update_diagnostics(self, **kwargs):
        self.episode_info.update(kwargs)

    def advance_progress(self):
        self.progress.update(self.pbar_task, advance = 1)

    def _make_table(self, data, columns, styles):
        table = Table(box = box.ROUNDED, expand = True)
        for col, style in zip(columns, styles):
            table.add_column(col, style = style, width = 30)
        for k, v in data.items():
            table.add_row(k, str(v))
        return table

    def create_renderable(self):
        progress_panel = Panel(self.progress, title = "Progress", border_style = "green")
        metrics_panel = Panel(self._make_table(self.episode_info, ("Metric", "Value"), ("cyan", "magenta")), title = "[b]Streaming RL Training[/b]", border_style = "blue")
        config_panel = Panel(self._make_table(self.hyperparams, ("Hyperparameter", "Value"), ("yellow", "white")), title = "[b]Configuration[/b]", border_style = "yellow")

        return Group(progress_panel, metrics_panel, config_panel)

# main

def main(
    num_episodes = 20_000,
    max_timesteps = 1000,
    actor_lr = 1.0,
    critic_lr = 1.0,
    entropy_weight = 0.01,
    discount_factor = 0.99,
    eligibility_trace_decay = 0.8,
    use_wandb = False,
    render = True,
    render_every_eps = 250,
    cpu = True,
    adaptive = True,
    val_min = -2.5,
    val_max = 2.5,
    num_bins = 127,
    delay_steps = 7,
    init_sparsity = 0.9,
    dim_actor = 128,
    dim_critic = 128,
    enable_pilar = False,
    pilar_mixing_param = 0.5,
    l2_weight_decay = 0.,
    l1_weight_decay = 0.,
    cautious_wd = False,
    wd_towards_init = False,
    use_critic_ema = True,
    use_minto = False,
    use_delightful_pg = False,
    delightful_eta = 1.0
):
    if render:
        rmtree(VIDEO_FOLDER, ignore_errors = True)

    # accelerator

    accelerator = Accelerator(
        log_with = 'wandb' if use_wandb else None,
        cpu = cpu
    )

    if use_wandb:
        accelerator.init_trackers(
            project_name = 'streaming-deep-rl',
            config = locals()
        )

    device = accelerator.device

    # environment

    env_config = dict(
        id = 'LunarLander-v3',
        continuous = False,
        render_mode = 'rgb_array' if render else None
    )

    env = gym.make(**env_config)

    # env dimensions

    dim_state = int(env.observation_space.shape[0])
    num_discrete_actions = int(env.action_space.n)

    if render:
        console = Console()
        console.print(f"\n[bold green]Video recording enabled.[/bold green]")
        console.print(f"[bold blue]Videos will be stored in:[/bold blue] [cyan]{VIDEO_FOLDER}[/cyan]\n")

        record_video_config = dict(
            env = env,
            video_folder = VIDEO_FOLDER,
            name_prefix = 'lunar-video',
            episode_trigger = lambda eps: divisible_by(eps, render_every_eps),
            disable_logger = True
        )

        env = RecordVideo(**record_video_config)

    # agent

    actor = ResidualNormedMLP(
        dim_actor,
        dim_in = dim_state,
        depth = 4,
        residual_every = 1
    ).to(device)

    critic = ResidualNormedMLP(
        dim_critic,
        dim_in = dim_state,
        depth = 4,
        residual_every = 1
    ).to(device)

    agent = StreamingACLambda(
        actor = actor,
        critic = critic,
        dim_state = dim_state,
        dim_actor = dim_actor,
        num_discrete_actions = num_discrete_actions,
        actor_lr = actor_lr,
        critic_lr = critic_lr,
        entropy_weight = entropy_weight,
        adaptive = adaptive,
        discount_factor = discount_factor,
        eligibility_trace_decay = eligibility_trace_decay,
        dim_critic = dim_critic,
        val_min = val_min,
        val_max = val_max,
        num_bins = num_bins,
        init_sparsity = init_sparsity,
        delay_steps = delay_steps,
        l2_weight_decay = l2_weight_decay,
        l1_weight_decay = l1_weight_decay,
        cautious_wd = cautious_wd,
        wd_towards_init = wd_towards_init,
        use_critic_ema = use_critic_ema,
        use_minto = use_minto,
        use_delightful_pg = use_delightful_pg,
        delightful_eta = delightful_eta
    )

    # metrics

    rolling_reward = deque(maxlen = 100)
    rolling_steps = deque(maxlen = 100)

    dashboard = Dashboard(num_episodes, hyperparams = dict(
        actor_lr = actor_lr,
        critic_lr = critic_lr,
        entropy_weight = entropy_weight,
        discount_factor = discount_factor,
        eligibility_trace_decay = eligibility_trace_decay,
        adaptive = adaptive,
        delay_steps = delay_steps,
        init_sparsity = init_sparsity,
        enable_pilar = enable_pilar,
        pilar_mixing_param = pilar_mixing_param,
        l2_weight_decay = l2_weight_decay,
        l1_weight_decay = l1_weight_decay,
        cautious_wd = cautious_wd,
        wd_towards_init = wd_towards_init,
        use_critic_ema = use_critic_ema,
        use_minto = use_minto,
        use_delightful_pg = use_delightful_pg,
        delightful_eta = delightful_eta
    ))

    # training loop

    with Live(dashboard.create_renderable(), refresh_per_second = 4) as live:
        for episode in range(num_episodes):
            state_np, _ = env.reset()
            state = torch.from_numpy(state_np).float().to(device)

            eps_reward = 0.0
            eps_steps = 0

            # reset eligibility traces

            agent.reset_trace_()

            for timestep in range(max_timesteps):

                # action

                action_logits = agent.forward_action(state)
                action = agent.readout.sample(action_logits)

                # step

                next_state_np, reward, terminated, truncated, _ = env.step(int(action.item()))

                reward_t = tensor(reward, dtype = torch.float32, device = device)
                is_terminal = tensor(terminated, dtype = torch.bool, device = device)
                next_state = torch.from_numpy(next_state_np).float().to(device)

                # update

                metrics = agent.update(
                    state = state,
                    action = action,
                    next_state = next_state,
                    reward = reward_t,
                    is_terminal = is_terminal,
                    drain = terminated or truncated
                )

                dashboard.update_diagnostics(
                    td_error = f"{metrics.td_error:.4f}",
                    value_pred = f"{metrics.value_pred:.4f}",
                    actor_grad = f"{metrics.actor_grad_norm:.4e}",
                    actor_norm = f"{metrics.actor_trace_norm:.4f}",
                    actor_scale = f"{metrics.actor_scale:.4e}",
                    critic_grad = f"{metrics.critic_grad_norm:.4e}",
                    critic_norm = f"{metrics.critic_trace_norm:.4f}",
                    critic_scale = f"{metrics.critic_scale:.4e}",
                    last_eps_reward = f"{eps_reward:.2f}",

                    last_eps_steps = eps_steps
                )

                live.update(dashboard.create_renderable())

                state = next_state
                eps_reward += reward
                eps_steps += 1

                if terminated or truncated:
                    break

            rolling_reward.append(eps_reward)
            rolling_steps.append(eps_steps)

            dashboard.advance_progress()

            avg_reward = np.mean(rolling_reward)
            avg_steps = np.mean(rolling_steps)

            dashboard.update_episode_info(
                avg_cum_reward_100 = f"{avg_reward:.2f}",
                avg_steps_100 = f"{avg_steps:.1f}"
            )

            live.update(dashboard.create_renderable())

if __name__ == '__main__':
    fire.Fire(main)
