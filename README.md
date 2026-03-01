<img src="./streaming.png" width="400px"></img>

## Streaming Deep RL

Explorations into the proposed [Streaming Deep Reinforcement Learning](https://arxiv.org/abs/2410.14606), from University of Alberta.

Once completed, if it checks out, will reach to integrate the Stream Q(λ) with [Q-Transformer](https://github.com/lucidrains/q-transformer).

A recent testimony to Streaming AC(λ) variant can be found [here](https://blog.9600baud.net/streaming-deep-rl-honeypot.html). Will be incorporated into the repository as well with a few improvements.

[Paper reading](https://www.youtube.com/live/5NFAzluHkcY) by Youtube AI/ML educator [@hu-po](https://www.youtube.com/@hu-po).

The official repository can be found <a href="https://github.com/mohmdelsayed/streaming-drl">here</a>.

## Install

```bash
$ pip install streaming-deep-rl
```

## Usage

```python
import torch

from streaming_deep_rl import StreamingACLambda
from x_mlps_pytorch.normed_mlp import MLP

# actor and critic

actor = MLP(
    8, 128, 128, 128,
    norm_elementwise_affine = False,
    activate_last = True
)

critic = MLP(
    8, 128, 128,
    norm_elementwise_affine = False
)

# agent

agent = StreamingACLambda(
    actor = actor,
    critic = critic,
    dim_state = 8,
    dim_actor = 128,
    num_discrete_actions = 4
)

# get action from state and pass to environment or world model

state = torch.randn(8)
action, action_dist = agent(state, sample = True)

# environment or world model gives back

next_state = torch.randn(8)
reward = torch.tensor(1.)
done = torch.tensor(False)

# update at each timestep, "streaming"

agent.update(
    state = state,
    action = action,
    next_state = next_state,
    reward = reward,
    is_terminal = done
)
```

## Citations

```bibtex
@inproceedings{Elsayed2024StreamingDR,
    title   = {Streaming Deep Reinforcement Learning Finally Works},
    author  = {Mohamed Elsayed and Gautham Vasan and A. Rupam Mahmood},
    year    = {2024},
    url     = {https://api.semanticscholar.org/CorpusID:273482696}
}
```

```bibtex
@article{Nauman2024BiggerRO,
    title   = {Bigger, Regularized, Optimistic: scaling for compute and sample-efficient continuous control},
    author  = {Michal Nauman and Mateusz Ostaszewski and Krzysztof Jankowski and Piotr Milo's and Marek Cygan},
    journal = {ArXiv},
    year    = {2024},
    volume  = {abs/2405.16158},
    url     = {https://api.semanticscholar.org/CorpusID:270063045}
}
```
