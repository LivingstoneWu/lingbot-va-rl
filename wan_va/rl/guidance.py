from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .checkpoint import CHECKPOINT_SCHEMA_VERSION
from .config import CriticTrainingConfig
from .critics import CriticBundle, build_critic_bundle


def denoising_time_scaling(
    time: torch.Tensor,
    beta: float = 2.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Denoising-step-aware Q guidance scale for time descending 1 -> 0."""
    time = time.clamp(0.0, 1.0)
    r_square = time.square() / (
        time.square() + (1.0 - time).square()
    ).clamp_min(eps)
    scaling = time / ((1.0 - time) * r_square + eps)
    return torch.minimum(scaling, torch.as_tensor(beta, device=time.device))


class QGuidanceAdapter(nn.Module):
    """Server-facing Q-guidance adapter contract."""

    def objective(
        self,
        action_tokens: torch.Tensor,
        action_mask: torch.Tensor,
        mode: str = "min",
    ) -> torch.Tensor:
        raise NotImplementedError


class TwinMLPQGuidanceAdapter(QGuidanceAdapter):
    """Inference adapter for current twin scalar-Q critic checkpoints."""

    def __init__(self, bundle: CriticBundle) -> None:
        super().__init__()
        self.bundle = bundle

    def q_values(
        self,
        action_tokens: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.bundle.q(action_tokens, action_mask)

    def objective(
        self,
        action_tokens: torch.Tensor,
        action_mask: torch.Tensor,
        mode: str = "min",
    ) -> torch.Tensor:
        q1, q2 = self.q_values(action_tokens, action_mask)
        if mode == "min":
            return torch.minimum(q1, q2)
        if mode == "mean":
            return 0.5 * (q1 + q2)
        if mode == "q1":
            return q1
        if mode == "q2":
            return q2
        raise ValueError(
            f"Unknown Q guidance objective {mode!r}; "
            "expected one of: min, mean, q1, q2"
        )


Q_GUIDANCE_REGISTRY: dict[str, type[QGuidanceAdapter]] = {
    "twin_mlp_v1": TwinMLPQGuidanceAdapter,
}


@dataclass(frozen=True)
class QGuidanceArtifact:
    config: CriticTrainingConfig
    manifest: dict[str, Any]
    adapter: QGuidanceAdapter

    @property
    def feature_layer(self) -> int:
        return self.config.feature_layer

    @property
    def feature_layers(self) -> tuple[int, ...]:
        return self.config.feature_layers

    @property
    def feature_aggregation(self) -> str:
        return self.config.feature_aggregation

    @property
    def feature_normalization(self) -> str:
        return self.config.feature_normalization


def load_q_guidance_artifact(
    checkpoint_dir: str | Path,
    device: torch.device,
) -> QGuidanceArtifact:
    """Load a critic checkpoint for inference-time Q guidance."""
    checkpoint = Path(checkpoint_dir)
    config = CriticTrainingConfig.from_json(checkpoint / "config.json")
    with (checkpoint / "manifest.json").open() as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported checkpoint schema: {manifest.get('schema_version')}"
        )
    _validate_manifest_matches_config(manifest, config)
    try:
        adapter_type = Q_GUIDANCE_REGISTRY[config.critic_type]
    except KeyError as error:
        raise ValueError(
            f"Unsupported Q guidance critic_type {config.critic_type!r}; "
            f"available: {sorted(Q_GUIDANCE_REGISTRY)}"
        ) from error

    bundle = build_critic_bundle(
        critic_type=config.critic_type,
        feature_dim=config.feature_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        with_value=config.algorithm == "iql",
    ).to(device=device, dtype=torch.float32)
    state = torch.load(
        checkpoint / "training_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    bundle.load_state_dict(state["critic"])
    bundle.eval().requires_grad_(False)
    adapter = adapter_type(bundle).eval()
    return QGuidanceArtifact(
        config=config,
        manifest=manifest,
        adapter=adapter,
    )


def _validate_manifest_matches_config(
    manifest: dict[str, Any],
    config: CriticTrainingConfig,
) -> None:
    expected = {
        "algorithm": config.algorithm,
        "critic_type": config.critic_type,
        "feature_dim": config.feature_dim,
        "infer_latent_chunk_size": config.infer_latent_chunk_size,
        "feature_layers": list(config.feature_layers),
        "feature_aggregation": config.feature_aggregation,
        "feature_normalization": config.feature_normalization,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(
                f"Incompatible Q checkpoint {key}: "
                f"expected {value!r}, found {manifest.get(key)!r}"
            )


def build_clean_feature_input(
    latents: torch.Tensor,
    actions: torch.Tensor,
    text_emb: torch.Tensor,
    patch_size: tuple[int, int, int],
    chunk_size: int,
    window_size: int,
) -> dict[str, Any]:
    return {
        "latent_dict": _clean_stream(
            latents, text_emb, patch_size, action_mode=False
        ),
        "action_dict": _clean_stream(
            actions, text_emb, patch_size, action_mode=True
        ),
        "chunk_size": chunk_size,
        "window_size": window_size,
    }


def _clean_stream(
    tensor: torch.Tensor,
    text_emb: torch.Tensor,
    patch_size: tuple[int, int, int],
    action_mode: bool,
) -> dict[str, torch.Tensor]:
    batch, _, frames, height, width = tensor.shape
    patch_f, patch_h, patch_w = (1, 1, 1) if action_mode else patch_size
    grid_id = _get_mesh_id(
        frames // patch_f,
        height // patch_h,
        width // patch_w,
        t=1 if action_mode else 0,
        action=action_mode,
    ).to(tensor.device)
    timesteps = torch.zeros(batch, frames, device=tensor.device)
    return {
        "timesteps": timesteps,
        "cond_timesteps": timesteps.clone(),
        "noisy_latents": tensor,
        "latent": tensor,
        "text_emb": text_emb,
        "grid_id": grid_id.unsqueeze(0).repeat(batch, 1, 1),
    }


def _get_mesh_id(
    frames: int,
    height: int,
    width: int,
    t: int,
    f_w: int = 1,
    f_shift: int = 0,
    action: bool = False,
) -> torch.Tensor:
    frame_idx = torch.arange(f_shift, frames + f_shift) * f_w
    height_idx = torch.arange(height)
    width_idx = torch.arange(width)
    ff, hh, ww = torch.meshgrid(
        frame_idx, height_idx, width_idx, indexing="ij"
    )
    if action:
        frame_offset = (
            torch.ones([height]).cumsum(0) / (height + 1)
        ).view(1, -1, 1)
        ff = ff + frame_offset
        hh = torch.ones_like(hh) * -1
        ww = torch.ones_like(ww) * -1

    grid_id = torch.cat(
        [ff.unsqueeze(0), hh.unsqueeze(0), ww.unsqueeze(0)], dim=0
    ).flatten(1)
    return torch.cat(
        [grid_id, torch.full_like(grid_id[:1], t)], dim=0
    )


def build_action_guidance_mask(
    actions: torch.Tensor,
    valid_action_channels: torch.Tensor,
    clamp_first_frame: bool,
) -> torch.Tensor:
    mask = valid_action_channels.to(device=actions.device).view(
        1, -1, 1, 1, 1
    )
    mask = mask.expand(
        actions.shape[0], -1, actions.shape[2], actions.shape[3], 1
    )
    if clamp_first_frame:
        mask = mask.clone()
        mask[:, :, 0:1] = False
    return mask.bool()
