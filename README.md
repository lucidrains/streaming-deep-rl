<img src="./streaming.png" width="400px"></img>

## Streaming Deep RL

Explorations into the proposed [Streaming Deep Reinforcement Learning](https://arxiv.org/abs/2410.14606), from University of Alberta.

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
    num_discrete_actions = 4,
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

## Continual Plasticity (`reborn_`, `crossover_distill` & `AgentPool`)

Consolidate the current policy/value into `[0 : keep_dim]` with independent subspace normalization (`SplitNorm`), while reinitializing `[keep_dim : dim]` as a sparse plastic reservoir. Supports standalone two-parent phenotypic distillation crossover (`crossover_distill`, decided per probe state by the parents' value critics) and `AgentPool` for single-stream natural selection, crossover, and specialist archive recall (with zero-overhead fallback when `pool_size = 1`, and `pool.eval()` routing to `pool.best_agent`).

```python
from streaming_deep_rl import StreamingACLambda, AgentPool, crossover_distill

agent = StreamingACLambda(
    actor = actor,
    critic = critic,
    dim_state = 8,
    dim_actor = 64,
    dim_critic = 64,
    num_discrete_actions = 4,
    reborn_sparsity = 0.9,      # fresh-half sparsity
    reborn_grow = True,         # optional width growth
    reborn_grow_factor = 1.5,
    reborn_max_dim = 256
)

# single-stream evolutionary pool (falls back to exact 1-agent behavior when pool_size = 1)
pool = AgentPool(
    agent,
    pool_size = 4,
    archive_size = 8,
    actor_selector = 'ucb',      # 'ucb' | 'softmax' | 'round_robin' | 'uniform' | Callable
    reproduce_fn = 'crossover',  # 'crossover' | 'reborn' | 'lerp' | 'shrink_perturb' | Callable
    crossover_filter = 'critic', # 'critic' | 'entropy' | 'fitness' | 'random' | Callable
    novelty_fn = 'kl',           # 'kl' | 'value' | Callable
    fitness_fn = None            # optional (ep_return, ep_steps, agent) -> float
)
```

## Cartpole

```bash
$ uv run train_cartpole.py --spr
```

## Lunar Lander

```bash
$ uv run train_lunar.py
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

```bibtex
@misc{chen2026cautiousweightdecay,
    title   = {Cautious Weight Decay},
    author  = {Lizhang Chen and Jonathan Li and Kaizhao Liang and Baiyu Su and Cong Xie and Nuo Wang Pierse and Chen Liang and Ni Lao and Qiang Liu},
    year    = {2026},
    eprint  = {2510.12402},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/2510.12402},
}
```

```bibtex
@article{dohare2024loss,
    title   = {Loss of Plasticity in Deep Continual Learning},
    author  = {Dohare, Shibhansh and Hernandez-Garcia, J. Fernando and Lan, Qingfeng and Rahman, Parash and Mahmood, A. Rupam and Sutton, Richard S.},
    journal = {Nature},
    volume  = {632},
    number  = {8026},
    pages   = {768--774},
    year    = {2024},
    url     = {https://arxiv.org/abs/2306.13812},
}
```

```bibtex
@misc{kumar2024maintainingplasticitycontinuallearning,
    title   = {Maintaining Plasticity in Continual Learning via Regenerative Regularization},
    author  = {Saurabh Kumar and Henrik Marklund and Benjamin Van Roy},
    year    = {2024},
    eprint  = {2308.11958},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/2308.11958},
}
```

```bibtex
@inproceedings{furlanello2018born,
    title   = {Born Again Neural Networks},
    author  = {Furlanello, Tommaso and Lipton, Zachary and Tschannen, Michael and Itti, Laurent and Anandkumar, Anima},
    booktitle = {Proceedings of the 35th International Conference on Machine Learning},
    pages   = {1607--1616},
    year    = {2018},
    volume  = {80},
    series  = {Proceedings of Machine Learning Research},
    publisher = {PMLR},
    url     = {https://proceedings.mlr.press/v80/furlanello18a.html},
}
```

```bibtex
@inproceedings{agarwal2022reincarnating,
    title   = {Reincarnating Reinforcement Learning: Reusing Prior Computation to Accelerate Progress},
    author  = {Agarwal, Rishabh and Schwarzer, Max and Castro, Pablo Samuel and Courville, Aaron C. and Bellemare, Marc},
    booktitle = {Advances in Neural Information Processing Systems},
    year    = {2022},
    volume  = {35},
    url     = {https://arxiv.org/abs/2206.01626},
}
```

```bibtex
@misc{osband2026delightfulpolicygradient,
    title   = {Delightful Policy Gradient},
    author  = {Ian Osband},
    year    = {2026},
    eprint  = {2603.14608},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/2603.14608},
}
```

```bibtex
@inproceedings{hendawy2026use,
    title   = {Use the Online Network If You Can: Towards Fast and Stable Reinforcement Learning},
    author  = {Ahmed Hendawy and Henrik Metternich and Th{\'e}o Vincent and Mahdi Kallel and Jan Peters and Carlo D'Eramo},
    booktitle = {The Fourteenth International Conference on Learning Representations},
    year    = {2026},
    url     = {https://openreview.net/forum?id=rFLuaG9Yq6}
}
```

```bibtex
@misc{schwarzer2021dataefficientreinforcementlearningselfpredictive,
    title   = {Data-Efficient Reinforcement Learning with Self-Predictive Representations},
    author  = {Max Schwarzer and Ankesh Anand and Rishab Goel and R Devon Hjelm and Aaron Courville and Philip Bachman},
    year    = {2021},
    eprint  = {2007.05929},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/2007.05929},
}
```

```bibtex
@misc{nilaksh2026squeezingstreamlearning,
    title   = {Squeezing More from the Stream : Learning Representation Online for Streaming Reinforcement Learning},
    author  = {Nilaksh and Antoine Clavaud and Mathieu Reymond and François Rivest and Sarath Chandar},
    year    = {2026},
    eprint  = {2602.09396},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/2602.09396},
}
```

```bibtex
@misc{maes2026leworldmodelstableendtoendjointembedding,
    title   = {LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
    author  = {Lucas Maes and Quentin Le Lidec and Damien Scieur and Yann LeCun and Randall Balestriero},
    year    = {2026},
    eprint  = {2603.19312},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/2603.19312},
}
```

```bibtex
@misc{lavoie2022simplicialembeddingsselfsupervisedlearning,
    title   = {Simplicial Embeddings in Self-Supervised Learning and Downstream Classification},
    author  = {Samuel Lavoie and Christos Tsirigotis and Max Schwarzer and Ankit Vani and Michael Noukhovitch and Kenji Kawaguchi and Aaron Courville},
    year    = {2022},
    eprint  = {2204.00616},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/2204.00616},
}
```

```bibtex
@inproceedings{bodnar2020proximal,
    title     = {Proximal Distilled Evolutionary Reinforcement Learning},
    author    = {Cristian Bodnar and Ben Day and Pietro Li{\`o}},
    booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
    volume    = {34},
    number    = {04},
    pages     = {3283--3290},
    year      = {2020},
    url       = {https://arxiv.org/abs/1906.09807}
}
```

```bibtex
@misc{schmidt2024learningactactions,
    title   = {Learning to Act without Actions},
    author  = {Dominik Schmidt and Minqi Jiang},
    year    = {2024},
    eprint  = {2312.10812},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/2312.10812},
}
```
