import torch

from wan_va.rl.algorithms import expectile_loss, iql_losses, mc_q_loss
from wan_va.rl.critics import MaskedTokenPool, TwinQCritic, ValueCritic


def test_masked_pool_uses_all_valid_chunk_tokens():
    tokens = torch.tensor(
        [[[[1.0], [3.0]], [[5.0], [100.0]]]]
    )
    action_mask = torch.tensor(
        [[[[[True], [True]], [[True], [False]]]]]
    )
    pooled = MaskedTokenPool()(tokens, action_mask)
    assert torch.allclose(pooled, torch.tensor([[3.0]]))


def test_twin_q_and_value_return_one_scalar_per_chunk():
    action_tokens = torch.randn(2, 3, 4, 8, dtype=torch.bfloat16)
    action_mask = torch.ones(2, 5, 3, 4, 1, dtype=torch.bool)
    video_tokens = torch.randn(2, 3, 6, 8, dtype=torch.bfloat16)
    latent_mask = torch.ones(2, 3, dtype=torch.bool)

    critic = TwinQCritic(8, hidden_dim=16, num_layers=1)
    value = ValueCritic(8, hidden_dim=16, num_layers=1)
    q1, q2 = critic(action_tokens, action_mask)

    assert q1.shape == (2,)
    assert q2.shape == (2,)
    assert value(video_tokens, latent_mask).shape == (2,)


def test_mc_and_iql_losses_have_expected_targets():
    q1 = torch.tensor([0.2, 0.4], requires_grad=True)
    q2 = torch.tensor([0.1, 0.5], requires_grad=True)
    returns = torch.tensor([1.0, 0.0])
    assert mc_q_loss(q1, q2, returns).item() > 0

    value = torch.tensor([0.0, 0.3], requires_grad=True)
    next_value = torch.tensor([0.8, 4.0])
    reward = torch.tensor([0.0, 1.0])
    discount = torch.tensor([0.5, 0.0])
    losses = iql_losses(
        q1,
        q2,
        value,
        next_value,
        reward,
        discount,
        expectile=0.7,
        value_mask=torch.tensor([False, True]),
    )

    assert torch.allclose(losses.target, torch.tensor([0.4, 1.0]))
    assert torch.allclose(losses.value, torch.tensor(0.007), atol=1e-6)
    assert losses.total.item() > 0
    assert expectile_loss(value, torch.minimum(q1, q2), 0.7).item() > 0

