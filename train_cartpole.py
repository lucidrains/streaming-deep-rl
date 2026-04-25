# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "streaming-deep-rl",
#     "x-mlps-pytorch>=0.3.3",
#     "gymnasium",
#     "numpy",
#     "torch",
#     "fire",
# ]
# ///

import fire
import numpy as np
from collections import deque

import torch
import gymnasium as gym

from x_mlps_pytorch.residual_normed_mlp import ResidualNormedMLP
from streaming_deep_rl.streaming_deep_rl import StreamingACLambda

# helpers

def exists(v):
    return v is not None

def divisible_by(num, den):
    return (num % den) == 0

# convergence test

def sanity_test(
    env_name = 'CartPole-v1',
    dim = 64,
    depth = 3,
    adaptive = False,
    spr = False,
    spr_target_embed_from_ema = False,
    spr_sigreg_weight = 0.1,
    spr_use_sem = False,
    spr_sem_dim_simplex = 8,
    spr_sem_temperature = 0.1,
    use_minto = False,
    use_delightful_pg = False,
    use_hl_gauss = False,
    converge_threshold = 100.,
    max_episodes = 500,
    print_every = 10
):
    env = gym.make(env_name)
    dim_state = int(env.observation_space.shape[0])
    num_actions = int(env.action_space.n)

    actor = ResidualNormedMLP(dim_in = dim_state, dim_out = dim, dim = dim, depth = depth, residual_every = 1)
    critic = ResidualNormedMLP(dim_in = dim_state, dim_out = dim, dim = dim, depth = depth, residual_every = 1)

    agent = StreamingACLambda(
        actor = actor,
        critic = critic,
        dim_state = dim_state,
        dim_actor = dim,
        dim_critic = dim,
        num_discrete_actions = num_actions,
        eligibility_trace_decay = 0.8,
        actor_lr = 1.0,
        critic_lr = 1.0,
        adaptive = adaptive,
        actor_self_predict_repr = spr,
        critic_self_predict_repr = spr,
        spr_target_embed_from_ema = spr_target_embed_from_ema,
        spr_sigreg_weight = spr_sigreg_weight,
        spr_use_sem = spr_use_sem,
        spr_sem_dim_simplex = spr_sem_dim_simplex,
        spr_sem_temperature = spr_sem_temperature,
        use_minto = use_minto,
        use_delightful_pg = use_delightful_pg,
        use_hl_gauss = use_hl_gauss
    )

    rolling_reward = deque(maxlen = 50)

    for eps in range(max_episodes):
        state, _ = env.reset()
        state = torch.from_numpy(state).float()
        eps_reward = 0.

        for t in range(500):
            action_logits = agent.forward_action(state)
            action = agent.readout.sample(action_logits)

            next_state, reward, terminated, truncated, _ = env.step(int(action.item()))
            next_state = torch.from_numpy(next_state).float()

            agent.update(state, action, next_state, reward, terminated)

            state = next_state
            eps_reward += reward

            if terminated or truncated:
                break

        agent.reset_trace_()
        rolling_reward.append(eps_reward)

        if divisible_by(eps + 1, print_every):
            avg = np.mean(rolling_reward)
            print(f'episode {eps + 1}: avg reward {avg:.1f}')

            if avg >= converge_threshold:
                print(f'converged at episode {eps + 1}')
                return True

    print(f'did not converge. final avg: {np.mean(rolling_reward):.1f}')

if __name__ == '__main__':
    fire.Fire(sanity_test)
