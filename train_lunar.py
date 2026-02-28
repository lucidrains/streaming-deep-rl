# /// script
# dependencies = [
#   "torch",
#   "einops",
#   "gymnasium[box2d]",
#   "fire",
#   "tqdm",
#   "wandb",
#   "discrete-continuous-embed-readout",
#   "torch-einops-utils",
#   "x-mlps-pytorch",
#   "accelerate",
#   "rich",
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
            "avg_reward_100": 0.0,
            "avg_steps_100": 0.0,
            "last_eps_reward": 0.0,
            "last_eps_steps": 0
        }

    def update_episode_info(self, **kwargs):
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
    num_episodes = 100_000,
    max_timesteps = 1000,
    actor_lr = 1e-4,
    critic_lr = 1e-4,
    entropy_weight = 0.01,
    discount_factor = 0.99,
    eligibility_trace_decay = 0.8,
    use_wandb = False,
    render = False,
    render_every_eps = 100,
    clear_videos = False,
    cpu = True
):
    if clear_videos:
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

    actor = MLP(
        8, 128, 128, 128,
        norm_elementwise_affine = False,
        activation = nn.LeakyReLU(),
        activate_last = True
    ).to(device)

    critic = MLP(
        8, 128, 128, 1,
        norm_elementwise_affine = False,
        activation = nn.LeakyReLU(),
        activate_last = False
    ).to(device)

    agent = StreamingACLambda(
        actor = actor,
        critic = critic,
        dim_state = 8,
        dim_actor = 128,
        num_discrete_actions = 4,
        actor_lr = actor_lr,
        critic_lr = critic_lr,
        entropy_weight = entropy_weight,
        discount_factor = discount_factor,
        eligibility_trace_decay = eligibility_trace_decay
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

            for trace in agent.actor_trace.values():
                trace.zero_()

            for trace in agent.critic_trace.values():
                trace.zero_()

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

                agent.update(
                    state = state, 
                    action = action, 
                    next_state = next_state, 
                    reward = reward_t, 
                    is_terminal = is_terminal
                )
                
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
                avg_reward_100 = f"{avg_reward:.2f}",
                avg_steps_100 = f"{avg_steps:.1f}",
                last_eps_reward = f"{eps_reward:.2f}",
                last_eps_steps = eps_steps
            )
            
            live.update(dashboard.create_renderable())
            
if __name__ == '__main__':
    fire.Fire(main)
