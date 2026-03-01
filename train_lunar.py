# /// script
# dependencies = [
#   "accelerate",
#   "torch",
#   "einops",
#   "gymnasium[box2d,other]",
#   "fire",
#   "tqdm",
#   "x-mlps-pytorch>=0.0.8",
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
from x_mlps_pytorch.normed_mlp import MLP
from x_mlps_pytorch.nff import nFeedforwards

# constants

VIDEO_FOLDER = './lunar-video-streaming'

# helpers

def exists(v):
    return v is not None

def divisible_by(num, den):
    return (num % den) == 0

# UI implementation

class Dashboard:
    def __init__(self, num_episodes):
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            expand = True
        )
        self.pbar_task = self.progress.add_task("Episodes", total = num_episodes)
        
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

    def create_renderable(self):
        table = Table(box = box.ROUNDED, expand = True)
        table.add_column("Metric", style = "cyan", width = 30)
        table.add_column("Value", style = "magenta", width = 20)
        
        for k, v in self.episode_info.items():
            table.add_row(k, str(v))
            
        group = Group(
            Panel(self.progress, title = "Progress", border_style = "green"),
            Panel(table, title = "[b]Streaming RL Training[/b]", border_style = "blue")
        )
        return group

# main

def main(
    num_episodes = 10_000,
    max_timesteps = 1000,
    actor_lr = 3e-4,
    critic_lr = 3e-4,
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
    sigma = 0.5,
    use_nff = False,
    init_sparsity = 0.9,
    dim_actor = 128,
    dim_critic = 128
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

    if use_nff:
        actor = nFeedforwards(
            dim = dim_actor,
            depth = 2,
            dim_in = dim_state,
        ).to(device)

        critic = nFeedforwards(
            dim = dim_critic,
            depth = 2,
            dim_in = dim_state,
            dim_out = dim_critic
        ).to(device)
    else:
        actor = MLP(
            dim_state, dim_actor, dim_actor, dim_actor,
            norm_elementwise_affine = False,
            activation = nn.SiLU(),
            activate_last = True
        ).to(device)

        critic = MLP(
            dim_state, dim_critic, dim_critic, dim_critic,
            norm_elementwise_affine = False,
            activation = nn.SiLU(),
            activate_last = True
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
        sigma = sigma,
        init_sparsity = init_sparsity
    )

    # metrics

    rolling_reward = deque(maxlen = 100)
    rolling_steps = deque(maxlen = 100)

    # dashboard

    dashboard = Dashboard(num_episodes)

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
                    is_terminal = is_terminal
                )

                if use_nff:
                    actor.norm_weights_()
                    critic.norm_weights_()

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
