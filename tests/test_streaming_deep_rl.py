from functools import partial

import pytest
param = pytest.mark.parametrize

import torch
from torch.nn import LayerNorm

def test_streaming():

    from streaming_deep_rl.streaming_deep_rl import (
        StreamingACLambda
    )

    from x_mlps_pytorch.normed_mlp import MLP

    actor = MLP(5, 128, norm_elementwise_affine = False, activate_last = True)

    critic = MLP(5, 128, norm_elementwise_affine = False, activate_last = True)

    streaming_actor_critic = StreamingACLambda(
        actor = actor,
        critic = critic,
        dim_actor = 128,
        dim_critic = 128
    )

    state = torch.randn(5,)
    actions = streaming_actor_critic(state)

    value = streaming_actor_critic.forward_value(state)

    assert actions.shape == (1,)
    assert value.shape == ()

    streaming_actor_critic.update(state, actions, state, 1.)
