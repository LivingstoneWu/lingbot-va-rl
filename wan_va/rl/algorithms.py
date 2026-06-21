from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class IQLLosses:
    total: torch.Tensor
    q: torch.Tensor
    value: torch.Tensor
    target: torch.Tensor


def expectile_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    expectile: float,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if not 0.0 < expectile < 1.0:
        raise ValueError("expectile must be in (0, 1)")
    difference = target - prediction
    weight = torch.where(
        difference >= 0,
        torch.as_tensor(expectile, device=difference.device),
        torch.as_tensor(1.0 - expectile, device=difference.device),
    )
    loss = weight * difference.square()
    if mask is None:
        return loss.mean()
    valid = mask.to(device=loss.device, dtype=loss.dtype)
    return (loss * valid).sum() / valid.sum().clamp_min(1.0)


def mc_q_loss(
    q1: torch.Tensor,
    q2: torch.Tensor,
    returns: torch.Tensor,
) -> torch.Tensor:
    target = returns.detach().to(q1.dtype)
    return F.mse_loss(q1, target) + F.mse_loss(q2, target)


def iql_losses(
    q1: torch.Tensor,
    q2: torch.Tensor,
    value: torch.Tensor,
    next_target_value: torch.Tensor,
    reward: torch.Tensor,
    discount: torch.Tensor,
    expectile: float,
    value_weight: float = 1.0,
    value_mask: torch.Tensor | None = None,
) -> IQLLosses:
    q_min = torch.minimum(q1, q2).detach()
    value_loss = expectile_loss(
        value, q_min, expectile, mask=value_mask
    )
    target = (
        reward.to(next_target_value.dtype)
        + discount.to(next_target_value.dtype)
        * next_target_value.detach()
    )
    q_loss = (
        F.mse_loss(q1, target.to(q1.dtype))
        + F.mse_loss(q2, target.to(q2.dtype))
    )
    total = q_loss + value_weight * value_loss
    return IQLLosses(
        total=total,
        q=q_loss,
        value=value_loss,
        target=target,
    )

