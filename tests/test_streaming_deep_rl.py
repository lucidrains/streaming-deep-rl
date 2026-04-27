import pytest
param = pytest.mark.parametrize

import torch

@param('ssl_type', (None, 'spr', 'lapo'))
@param('spr_target_embed_from_ema', (False, True))
@param('spr_use_sem', (False, True))
def test_streaming(
    ssl_type,
    spr_target_embed_from_ema,
    spr_use_sem
):

    from streaming_deep_rl.streaming_deep_rl import (
        StreamingACLambda
    )

    from x_mlps_pytorch.normed_mlp import MLP

    actor = MLP(5, 128, norm_elementwise_affine = False, activate_last = False)

    critic = MLP(5, 128, norm_elementwise_affine = False, activate_last = False)

    ssl_kwargs = dict(
        target_embed_from_ema = spr_target_embed_from_ema,
    )

    if ssl_type == 'spr':
        ssl_kwargs.update(
            use_sem = spr_use_sem,
            sigreg_weight = 1.0 if not spr_target_embed_from_ema else 0.0
        )
    elif ssl_type == 'lapo':
        ssl_kwargs.update(
            pred_action_loss_weight = 1.0
        )

    streaming_actor_critic = StreamingACLambda(
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

    streaming_actor_critic.reset_trace_()

    state = torch.randn(5)
    action_dist_params = streaming_actor_critic(state)

    value = streaming_actor_critic.forward_value(state)

    assert action_dist_params.shape == (1, 2)
    assert value.shape == ()

    actions = streaming_actor_critic.sample_action(action_dist_params)
    streaming_actor_critic.update(state, actions, state, 1.)

    # test save and load

    streaming_actor_critic.save('./test_streaming_ckpt.pt')
    streaming_actor_critic.load('./test_streaming_ckpt.pt')
