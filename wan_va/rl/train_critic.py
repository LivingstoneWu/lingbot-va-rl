from __future__ import annotations

import argparse
import json
import os
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from wan_va.configs import VA_CONFIGS
from wan_va.dataset import MultiLatentLeRobotDataset
from wan_va.distributed.fsdp import shard_model
from wan_va.distributed.util import _configure_model
from wan_va.modules.utils import load_transformer
from wan_va.utils import get_mesh_id, init_logger

from .algorithms import iql_losses, mc_q_loss
from .checkpoint import load_critic_checkpoint, save_critic_checkpoint
from .config import CriticTrainingConfig
from .critics import CriticBundle, build_critic_bundle
from .transitions import ChunkTransitionDataset


def _init_runtime() -> tuple[int, int, int, torch.device]:
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    if not torch.cuda.is_available():
        raise RuntimeError("Critic training requires CUDA")
    torch.cuda.set_device(local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )
    return rank, local_rank, world_size, torch.device("cuda", local_rank)


def _transformer_path(base_config: Any, critic_config: CriticTrainingConfig) -> str:
    if critic_config.transformer_path:
        return critic_config.transformer_path
    resume_from = getattr(base_config, "resume_from", None)
    if resume_from:
        return str(Path(resume_from) / "transformer")
    return str(
        Path(base_config.wan22_pretrained_model_name_or_path) / "transformer"
    )


def _clean_stream(
    tensor: torch.Tensor,
    text_emb: torch.Tensor,
    patch_size: tuple[int, int, int],
    action_mode: bool,
) -> dict[str, torch.Tensor]:
    batch, _, frames, height, width = tensor.shape
    patch_f, patch_h, patch_w = (1, 1, 1) if action_mode else patch_size
    grid_id = get_mesh_id(
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


def _prepare_clean_input(
    batch: dict[str, torch.Tensor],
    prefix: str,
    patch_size: tuple[int, int, int],
    chunk_size: int,
    window_size: int,
    device: torch.device,
    model_dtype: torch.dtype,
) -> dict[str, Any]:
    def get(name: str) -> torch.Tensor:
        tensor = batch[f"{prefix}{name}"].to(
            device, non_blocking=True
        )
        if tensor.is_floating_point():
            tensor = tensor.to(dtype=model_dtype)
        return tensor

    latents = get("latents")
    actions = get("actions")
    text_emb = get("text_emb")
    latent_dict = _clean_stream(
        latents, text_emb, patch_size, action_mode=False
    )
    action_dict = _clean_stream(
        actions, text_emb, patch_size, action_mode=True
    )
    latent_dict["latents_mask"] = get("latents_mask")
    action_dict["actions_mask"] = get("actions_mask")
    return {
        "latent_dict": latent_dict,
        "action_dict": action_dict,
        "chunk_size": chunk_size,
        "window_size": window_size,
    }


class CriticTrainer:
    def __init__(
        self,
        config: CriticTrainingConfig,
        rank: int,
        local_rank: int,
        world_size: int,
        device: torch.device,
    ) -> None:
        self.config = config
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = device
        self.step = 0

        if config.base_config_name not in VA_CONFIGS:
            raise KeyError(
                f"Unknown base config: {config.base_config_name}"
            )
        self.base_config = deepcopy(VA_CONFIGS[config.base_config_name])
        self.base_config.cfg_prob = 0.0
        self.base_config.jepa_loss_enabled = False
        self.base_config.batch_size = config.batch_size

        inference_chunk_size = int(self.base_config.frame_chunk_size)
        if config.infer_latent_chunk_size != inference_chunk_size:
            raise ValueError(
                "infer_latent_chunk_size must match base-config "
                f"frame_chunk_size ({inference_chunk_size})"
            )
        action_per_frame = int(self.base_config.action_per_frame)
        base_dataset = MultiLatentLeRobotDataset(self.base_config)
        self.dataset = ChunkTransitionDataset(
            base_dataset=base_dataset,
            infer_latent_chunk_size=config.infer_latent_chunk_size,
            action_per_frame=action_per_frame,
            gamma=config.gamma,
            reward_source=config.reward_source,
            include_sparse_success_reward=(
                config.include_sparse_success_reward
            ),
            jepa_reward_weight=config.jepa_reward_weight,
            success_reward_weight=config.success_reward_weight,
            require_outcomes=True,
            include_previous=config.algorithm == "iql",
        )
        self.sampler = (
            DistributedSampler(
                self.dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=config.seed,
            )
            if world_size > 1
            else None
        )
        self.loader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=self.sampler is None,
            sampler=self.sampler,
            num_workers=config.num_workers,
            pin_memory=True,
            persistent_workers=config.num_workers > 0,
        )

        transformer = load_transformer(
            _transformer_path(self.base_config, config),
            torch_dtype=torch.float32,
            torch_device="cpu",
        )
        self.transformer = _configure_model(
            model=transformer,
            shard_fn=shard_model,
            param_dtype=self.base_config.param_dtype,
            device=device,
            eval_mode=True,
        )
        self.transformer.eval().requires_grad_(False)
        transformer_layers = len(self.transformer.blocks)
        if config.feature_layer >= transformer_layers:
            raise ValueError(
                f"feature layer {config.feature_layer} is out of range for "
                f"a transformer with {transformer_layers} blocks"
            )

        self.bundle = build_critic_bundle(
            critic_type=config.critic_type,
            feature_dim=config.feature_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            with_value=config.algorithm == "iql",
        ).to(device=device, dtype=torch.float32)
        self.optimizer = torch.optim.AdamW(
            [
                parameter
                for parameter in self.bundle.parameters()
                if parameter.requires_grad
            ],
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        if config.resume_from:
            self.step = load_critic_checkpoint(
                config.resume_from,
                self.bundle,
                self.optimizer,
                expected_manifest={
                    "algorithm": config.algorithm,
                    "critic_type": config.critic_type,
                    "feature_dim": config.feature_dim,
                    "infer_latent_chunk_size": (
                        config.infer_latent_chunk_size
                    ),
                    "feature_layers": list(config.feature_layers),
                    "feature_aggregation": config.feature_aggregation,
                    "feature_normalization": config.feature_normalization,
                },
            )

        self.output_dir = Path(config.output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        if rank == 0:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            (self.checkpoint_dir / "config.json").write_text(
                json.dumps(config.to_dict(), indent=2) + "\n"
            )
        self.log_path = self.checkpoint_dir / "loss.jsonl"

    @torch.no_grad()
    def _extract_features(
        self,
        batch: dict[str, torch.Tensor],
        prefix: str = "",
    ) -> dict[str, torch.Tensor]:
        input_dict = _prepare_clean_input(
            batch=batch,
            prefix=prefix,
            patch_size=tuple(self.base_config.patch_size),
            chunk_size=self.config.infer_latent_chunk_size,
            window_size=self.config.window_size,
            device=self.device,
            model_dtype=self.base_config.param_dtype,
        )
        outputs = self.transformer(
            input_dict,
            train_mode=True,
            return_features=True,
            critic_feature_layer=self.config.feature_layer,
        )
        features = {
            key: value.detach()
            for key, value in outputs[3].items()
        }
        actual_dim = features["action_tokens"].shape[-1]
        if actual_dim != self.config.feature_dim:
            raise ValueError(
                f"Configured feature_dim={self.config.feature_dim}, "
                f"transformer returned {actual_dim}"
            )
        return features

    def _all_reduce_gradients(self) -> None:
        if self.world_size == 1:
            return
        for parameter in self.bundle.parameters():
            if parameter.grad is not None:
                dist.all_reduce(parameter.grad, op=dist.ReduceOp.AVG)

    def _train_batch(
        self,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        features = self._extract_features(batch)
        action_mask = batch["actions_mask"].to(
            self.device, non_blocking=True
        )
        latent_mask = batch["latents_mask"].to(
            self.device, non_blocking=True
        )
        q1, q2 = self.bundle.q(features["action_tokens"], action_mask)

        if self.config.algorithm == "mc":
            returns = batch["mc_return"].to(self.device)
            total_loss = mc_q_loss(q1, q2, returns)
            metrics = {
                "loss": total_loss,
                "q_loss": total_loss,
                "value_loss": torch.zeros_like(total_loss),
                "target_mean": returns.float().mean(),
            }
        else:
            assert self.bundle.value is not None
            assert self.bundle.target_value is not None
            previous_features = self._extract_features(
                batch, prefix="previous_"
            )
            previous_latent_mask = batch["previous_latents_mask"].to(
                self.device, non_blocking=True
            )
            value = self.bundle.value(
                previous_features["video_tokens"], previous_latent_mask
            )
            with torch.no_grad():
                next_value = self.bundle.target_value(
                    features["video_tokens"], latent_mask
                )
            losses = iql_losses(
                q1=q1,
                q2=q2,
                value=value,
                next_target_value=next_value,
                reward=batch["reward"].to(self.device),
                discount=batch["discount"].to(self.device),
                expectile=self.config.expectile,
                value_weight=self.config.value_loss_weight,
                value_mask=batch["state_valid"].to(self.device),
            )
            total_loss = losses.total
            metrics = {
                "loss": losses.total,
                "q_loss": losses.q,
                "value_loss": losses.value,
                "target_mean": losses.target.float().mean(),
            }

        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        self._all_reduce_gradients()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [
                parameter
                for parameter in self.bundle.parameters()
                if parameter.requires_grad
            ],
            max_norm=10.0,
        )
        self.optimizer.step()
        if self.config.algorithm == "iql":
            self.bundle.update_target(self.config.target_ema_rate)

        metrics.update(
            {
                "q1_mean": q1.detach().float().mean(),
                "q2_mean": q2.detach().float().mean(),
                "q_disagreement": (q1 - q2).detach().abs().float().mean(),
                "grad_norm": torch.as_tensor(grad_norm).float(),
            }
        )
        return {
            key: float(value.detach().cpu())
            for key, value in metrics.items()
        }

    @torch.no_grad()
    def _parameter_norm(self) -> float:
        squared_norm = torch.zeros(
            (), device=self.device, dtype=torch.float32
        )
        for parameter in self.bundle.parameters():
            if parameter.requires_grad:
                squared_norm += parameter.detach().float().square().sum()
        return float(squared_norm.sqrt().cpu())

    def _save(self) -> None:
        if self.rank != 0:
            return
        save_critic_checkpoint(
            directory=self.checkpoint_dir / f"checkpoint_{self.step:08d}",
            bundle=self.bundle,
            optimizer=self.optimizer,
            config=self.config,
            step=self.step,
            manifest={
                "base_config_name": self.config.base_config_name,
                "base_transformer_path": _transformer_path(
                    self.base_config, self.config
                ),
                "action_per_frame": int(
                    self.base_config.action_per_frame
                ),
                "feature_source": (
                    "final_normalized_action_tokens_v1"
                    if self.config.feature_layer == -1
                    else "raw_block_action_tokens_v1"
                ),
                "video_feature_source": (
                    "final_normalized_video_tokens_v2"
                    if self.config.feature_layer == -1
                    else "raw_block_video_tokens_v1"
                ),
                "value_state_alignment": (
                    "previous_video_for_v_current_video_for_target_v"
                ),
                "reward_source": self.config.reward_source,
                "include_sparse_success_reward": (
                    self.config.include_sparse_success_reward
                ),
                "jepa_reward_weight": self.config.jepa_reward_weight,
                "success_reward_weight": self.config.success_reward_weight,
            },
        )

    def train(self) -> None:
        self.bundle.train()
        progress = tqdm(
            total=self.config.num_steps,
            initial=self.step,
            disable=self.rank != 0,
            desc="critic",
        )
        epoch = 0
        while self.step < self.config.num_steps:
            if self.sampler is not None:
                self.sampler.set_epoch(epoch)
            for batch in self.loader:
                metrics = self._train_batch(batch)
                self.step += 1
                if self.rank == 0:
                    progress.update(1)
                    progress.set_postfix(
                        loss=f"{metrics['loss']:.4f}",
                        q=f"{metrics['q1_mean']:.3f}",
                    )
                    if self.step % self.config.log_interval == 0:
                        entry = {
                            "step": self.step,
                            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "total_loss": metrics["loss"],
                            "q_loss": metrics["q_loss"],
                            "value_loss": metrics["value_loss"],
                            "grad_norm": metrics["grad_norm"],
                            "param_norm": self._parameter_norm(),
                        }
                        with self.log_path.open("a") as handle:
                            handle.write(json.dumps(entry) + "\n")
                if self.step % self.config.save_interval == 0:
                    self._save()
                if self.step >= self.config.num_steps:
                    break
            epoch += 1
        self._save()
        progress.close()


def run(config_path: str) -> None:
    config = CriticTrainingConfig.from_json(config_path)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    rank, local_rank, world_size, device = _init_runtime()
    trainer = CriticTrainer(
        config, rank, local_rank, world_size, device
    )
    trainer.train()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train frozen-backbone MC/IQL critics"
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    init_logger()
    main()
