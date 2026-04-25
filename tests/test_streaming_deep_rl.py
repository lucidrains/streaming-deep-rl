import pytest
param = pytest.mark.parametrize

import torch

@param('actor_self_predict_repr', (False, True))
@param('critic_self_predict_repr', (False, True))
@param('spr_target_embed_from_ema', (False, True))
@param('spr_use_sem', (False, True))
def test_streaming(
    actor_self_predict_repr,
    critic_self_predict_repr,
    spr_target_embed_from_ema,
    spr_use_sem
):

    from streaming_deep_rl.streaming_deep_rl import (
        StreamingACLambda
    )

    from x_mlps_pytorch.normed_mlp import MLP

    actor = MLP(5, 128, norm_elementwise_affine = False, activate_last = False)

    critic = MLP(5, 128, norm_elementwise_affine = False, activate_last = False)

    streaming_actor_critic = StreamingACLambda(
        actor = actor,
        critic = critic,
        num_continuous_actions = 1,
        dim_state = 5,
        dim_actor = 128,
        dim_critic = 128,
        actor_self_predict_repr = actor_self_predict_repr,
        critic_self_predict_repr = critic_self_predict_repr,
        spr_target_embed_from_ema = spr_target_embed_from_ema,
        spr_sigreg_weight = 1.0 if not spr_target_embed_from_ema else 0.0,
        spr_use_sem = spr_use_sem
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
