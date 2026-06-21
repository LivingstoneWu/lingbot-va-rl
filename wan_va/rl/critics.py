from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn


class MaskedTokenPool(nn.Module):
    """Pool all valid tokens in one inference-sized latent/action chunk."""

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if tokens.ndim != 4:
            raise ValueError(
                f"tokens must have shape [B,F,T,D], got {tuple(tokens.shape)}"
            )
        token_mask = self._normalize_mask(mask, tokens)
        weights = token_mask.to(tokens.dtype).unsqueeze(-1)
        denominator = weights.sum(dim=(1, 2)).clamp_min(1.0)
        return (tokens * weights).sum(dim=(1, 2)) / denominator

    @staticmethod
    def _normalize_mask(
        mask: torch.Tensor,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        batch, frames, token_count, _ = tokens.shape
        if mask.ndim == 5:
            # Action mask [B,C,F,N,1] -> token validity [B,F,N].
            mask = mask.squeeze(-1).any(dim=1)
        elif mask.ndim == 2:
            # Latent frame mask [B,F] -> all spatial context tokens in frame.
            mask = mask.unsqueeze(-1).expand(-1, -1, token_count)
        elif mask.ndim != 3:
            raise ValueError(
                "mask must have shape [B,C,F,N,1], [B,F,T], or [B,F]"
            )
        expected = (batch, frames, token_count)
        if tuple(mask.shape) != expected:
            raise ValueError(
                f"mask shape {tuple(mask.shape)} does not match {expected}"
            )
        return mask.bool()


class ScalarMLP(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        layers: list[nn.Module] = [nn.LayerNorm(feature_dim)]
        input_dim = feature_dim
        for _ in range(num_layers):
            layers.extend(
                [nn.Linear(input_dim, hidden_dim), nn.GELU()]
            )
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


class TwinQCritic(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.pool = MaskedTokenPool()
        self.q1 = ScalarMLP(feature_dim, hidden_dim, num_layers)
        self.q2 = ScalarMLP(feature_dim, hidden_dim, num_layers)

    def forward(
        self,
        action_tokens: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = self.pool(action_tokens, action_mask).float()
        return self.q1(pooled), self.q2(pooled)


class ValueCritic(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.pool = MaskedTokenPool()
        self.value = ScalarMLP(feature_dim, hidden_dim, num_layers)

    def forward(
        self,
        video_tokens: torch.Tensor,
        latent_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.value(
            self.pool(video_tokens, latent_mask).float()
        )


class CriticBundle(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 2,
        with_value: bool = True,
    ) -> None:
        super().__init__()
        self.q = TwinQCritic(feature_dim, hidden_dim, num_layers)
        self.value = (
            ValueCritic(feature_dim, hidden_dim, num_layers)
            if with_value
            else None
        )
        self.target_value = (
            deepcopy(self.value).requires_grad_(False)
            if self.value is not None
            else None
        )

    @torch.no_grad()
    def update_target(self, rate: float) -> None:
        if self.value is None or self.target_value is None:
            return
        if not 0.0 <= rate <= 1.0:
            raise ValueError("EMA rate must be in [0, 1]")
        for target, source in zip(
            self.target_value.parameters(), self.value.parameters()
        ):
            target.lerp_(source, rate)


CRITIC_REGISTRY = {
    "twin_mlp_v1": CriticBundle,
}


def build_critic_bundle(
    critic_type: str,
    **kwargs,
) -> CriticBundle:
    try:
        constructor = CRITIC_REGISTRY[critic_type]
    except KeyError as error:
        raise ValueError(
            f"Unknown critic_type {critic_type!r}; "
            f"available: {sorted(CRITIC_REGISTRY)}"
        ) from error
    return constructor(**kwargs)

