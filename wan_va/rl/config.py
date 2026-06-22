from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CriticTrainingConfig:
    base_config_name: str
    output_dir: str
    infer_latent_chunk_size: int
    algorithm: str = "mc"
    critic_type: str = "twin_mlp_v1"
    transformer_path: str | None = None
    feature_dim: int = 3072
    feature_layers: tuple[int, ...] = (-1,)
    feature_aggregation: str = "single"
    hidden_dim: int = 512
    num_layers: int = 2
    gamma: float = 0.99
    expectile: float = 0.7
    value_loss_weight: float = 1.0
    target_ema_rate: float = 0.005
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 1
    num_workers: int = 4
    num_steps: int = 10000
    log_interval: int = 10
    save_interval: int = 1000
    window_size: int = 64
    seed: int = 42
    resume_from: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.feature_layers, (list, tuple)):
            raise TypeError("feature_layers must be a list or tuple")
        if any(
            not isinstance(layer, int) or isinstance(layer, bool)
            for layer in self.feature_layers
        ):
            raise TypeError("feature_layers must contain only integers")
        object.__setattr__(self, "feature_layers", tuple(self.feature_layers))
        if len(self.feature_layers) != 1:
            raise ValueError("Phase 1 requires exactly one feature layer")
        if self.feature_layers[0] < -1:
            raise ValueError("feature layer must be -1 or non-negative")
        if self.feature_aggregation != "single":
            raise ValueError("feature_aggregation must be 'single'")

        if self.algorithm not in {"mc", "iql"}:
            raise ValueError("algorithm must be 'mc' or 'iql'")
        if self.infer_latent_chunk_size <= 0:
            raise ValueError("infer_latent_chunk_size must be positive")
        if self.log_interval <= 0:
            raise ValueError("log_interval must be positive")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if not 0.0 < self.expectile < 1.0:
            raise ValueError("expectile must be in (0, 1)")

    @property
    def feature_layer(self) -> int:
        return self.feature_layers[0]

    @property
    def feature_normalization(self) -> str:
        if self.feature_layer == -1:
            return "final_adaptive_norm_v1"
        return "raw_block_output_v1"

    @classmethod
    def from_json(cls, path: str | Path) -> "CriticTrainingConfig":
        with Path(path).open() as handle:
            values: dict[str, Any] = json.load(handle)
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(values).difference(known))
        if unknown:
            raise ValueError(f"Unknown critic config keys: {unknown}")
        if "feature_layers" in values:
            values["feature_layers"] = tuple(values["feature_layers"])
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

