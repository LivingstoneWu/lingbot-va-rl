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
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from preprocessing.extract_jepa_features import build_jepa_encoder
from wan_va.configs import VA_CONFIGS
from wan_va.dataset import MultiLatentLeRobotDataset
from wan_va.distributed.fsdp import shard_model
from wan_va.distributed.util import _configure_model
from wan_va.modules.utils import load_transformer, load_vae
from wan_va.utils import FlowMatchScheduler, data_seq_to_patch, get_mesh_id, init_logger

from .algorithms import iql_losses
from .checkpoint import load_critic_checkpoint, save_critic_checkpoint
from .config import CriticTrainingConfig
from .critics import build_critic_bundle
from .train_critic import _init_runtime, _transformer_path
from .transitions import ChunkTransitionDataset


def _zero_timestep_stream(
    tensor: torch.Tensor,
    text_emb: torch.Tensor,
    patch_size: tuple[int, int, int],
    action_mode: bool,
    timestep: float,
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
    timesteps = torch.full(
        (batch, frames),
        float(timestep) * 1000.0,
        device=tensor.device,
    )
    return {
        "timesteps": timesteps,
        "cond_timesteps": torch.zeros_like(timesteps),
        "noisy_latents": tensor,
        "latent": tensor,
        "text_emb": text_emb,
        "grid_id": grid_id.unsqueeze(0).repeat(batch, 1, 1),
    }


def _masked_mean_distance(
    predicted: torch.Tensor,
    actual: torch.Tensor,
    mask: torch.Tensor,
    available: torch.Tensor | None,
) -> torch.Tensor:
    predicted = F.normalize(predicted.float(), dim=-1)
    actual = F.normalize(actual.float(), dim=-1)
    distance = 1.0 - (predicted * actual).sum(dim=-1)
    per_frame = distance.mean(dim=(2, 3))
    valid = mask.to(device=per_frame.device, dtype=per_frame.dtype)
    if available is not None:
        valid = valid * available.to(valid.device, valid.dtype).view(-1, 1)
    return (per_frame * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)


class Phase3CriticTrainer:
    """Train critics on generated-video-conditioned action features."""

    def __init__(
        self,
        config: CriticTrainingConfig,
        rank: int,
        local_rank: int,
        world_size: int,
        device: torch.device,
    ) -> None:
        if config.training_distribution != "predicted_video_conditioned_action":
            raise ValueError(
                "train_critic_phase3.py requires "
                "training_distribution='predicted_video_conditioned_action'"
            )
        self.config = config
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = device
        self.step = 0

        self.base_config = deepcopy(VA_CONFIGS[config.base_config_name])
        self.base_config.cfg_prob = 0.0
        self.base_config.jepa_loss_enabled = True
        self.base_config.batch_size = config.batch_size
        if config.infer_latent_chunk_size != int(self.base_config.frame_chunk_size):
            raise ValueError(
                "infer_latent_chunk_size must match base-config frame_chunk_size"
            )

        base_dataset = MultiLatentLeRobotDataset(self.base_config)
        missing_jepa = base_dataset.get_missing_jepa_files()
        if missing_jepa:
            raise FileNotFoundError(
                "Phase 3 requires cached actual JEPA targets; missing first "
                f"files: {missing_jepa[:5]}"
            )
        self.dataset = ChunkTransitionDataset(
            base_dataset=base_dataset,
            infer_latent_chunk_size=config.infer_latent_chunk_size,
            action_per_frame=int(self.base_config.action_per_frame),
            gamma=config.gamma,
            reward_source=config.reward_source,
            include_sparse_success_reward=config.include_sparse_success_reward,
            jepa_reward_weight=config.jepa_reward_weight,
            success_reward_weight=config.success_reward_weight,
            require_outcomes=True,
            include_previous=True,
            include_next=True,
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
        if config.feature_layer >= len(self.transformer.blocks):
            raise ValueError("feature_layer is out of range for transformer")

        self.video_scheduler = FlowMatchScheduler(
            shift=self.base_config.snr_shift,
            sigma_min=0.0,
            extra_one_step=True,
        )
        self.action_scheduler = FlowMatchScheduler(
            shift=self.base_config.action_snr_shift,
            sigma_min=0.0,
            extra_one_step=True,
        )

        self.vae = load_vae(
            Path(self.base_config.wan22_pretrained_model_name_or_path) / "vae",
            torch_dtype=self.base_config.param_dtype,
            torch_device=device,
        )
        self.vae.eval().requires_grad_(False)
        self.jepa_encoder = build_jepa_encoder(
            config.phase3_jepa_checkpoint,
            device,
        )
        self.jepa_encoder.eval().requires_grad_(False)

        self.bundle = build_critic_bundle(
            critic_type=config.critic_type,
            feature_dim=config.feature_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            with_value=True,
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
                    "infer_latent_chunk_size": config.infer_latent_chunk_size,
                    "feature_layers": list(config.feature_layers),
                    "feature_aggregation": config.feature_aggregation,
                    "feature_normalization": config.feature_normalization,
                    "training_distribution": config.training_distribution,
                    "reward_source": config.reward_source,
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

    def _to_device(self, batch: dict[str, torch.Tensor], key: str) -> torch.Tensor:
        tensor = batch[key].to(self.device, non_blocking=True)
        if tensor.is_floating_point():
            tensor = tensor.to(dtype=self.base_config.param_dtype)
        return tensor

    def _packed_streams(
        self,
        history_latents: torch.Tensor,
        current_latents: torch.Tensor,
        history_actions: torch.Tensor,
        current_actions: torch.Tensor,
        text_emb: torch.Tensor,
        latent_timestep: float,
        action_timestep: float,
    ) -> dict[str, Any]:
        latents = torch.cat([history_latents, current_latents], dim=2)
        actions = torch.cat([history_actions, current_actions], dim=2)
        return {
            "latent_dict": _zero_timestep_stream(
                latents,
                text_emb,
                tuple(self.base_config.patch_size),
                action_mode=False,
                timestep=latent_timestep,
            ),
            "action_dict": _zero_timestep_stream(
                actions,
                text_emb,
                tuple(self.base_config.patch_size),
                action_mode=True,
                timestep=action_timestep,
            ),
            "chunk_size": self.config.infer_latent_chunk_size,
            "window_size": self.config.window_size,
        }

    @torch.no_grad()
    def _predict_video(
        self,
        history_latents: torch.Tensor,
        current_gt_latents: torch.Tensor,
        text_emb: torch.Tensor,
    ) -> torch.Tensor:
        current = torch.randn_like(current_gt_latents)
        video_steps = (
            self.config.phase3_video_num_inference_steps
            or int(self.base_config.num_inference_steps)
        )
        self.video_scheduler.set_timesteps(video_steps)
        timesteps = F.pad(
            self.video_scheduler.timesteps,
            (0, 1),
            mode="constant",
            value=0,
        )
        if self.config.phase3_video_exec_step != -1:
            timesteps = timesteps[: self.config.phase3_video_exec_step]

        dummy_history_actions = torch.zeros(
            history_latents.shape[0],
            int(self.base_config.action_dim),
            history_latents.shape[2],
            int(self.base_config.action_per_frame),
            1,
            device=self.device,
            dtype=history_latents.dtype,
        )
        dummy_current_actions = torch.zeros_like(dummy_history_actions)
        for index, timestep in enumerate(timesteps):
            last_step = index == len(timesteps) - 1
            inputs = self._packed_streams(
                history_latents,
                current,
                dummy_history_actions,
                dummy_current_actions,
                text_emb,
                latent_timestep=float(timestep.item()) / 1000.0,
                action_timestep=0.0,
            )
            latent_pred, _, _ = self.transformer(inputs, train_mode=True)
            if not last_step or self.config.phase3_video_exec_step != -1:
                latent_pred = data_seq_to_patch(
                    tuple(self.base_config.patch_size),
                    latent_pred,
                    history_latents.shape[2] + current.shape[2],
                    current.shape[3],
                    current.shape[4],
                    batch_size=current.shape[0],
                )
                velocity = latent_pred[:, :, -current.shape[2] :]
                current = self.video_scheduler.step(
                    velocity,
                    timestep,
                    current,
                    return_dict=False,
                )
        return current

    @torch.no_grad()
    def _predict_actions(
        self,
        history_latents: torch.Tensor,
        predicted_latents: torch.Tensor,
        history_actions: torch.Tensor,
        current_gt_actions: torch.Tensor,
        current_action_mask: torch.Tensor,
        text_emb: torch.Tensor,
    ) -> torch.Tensor:
        current = torch.randn_like(current_gt_actions)
        current = current * current_action_mask.to(current.dtype)
        action_steps = (
            self.config.phase3_action_num_inference_steps
            or int(self.base_config.action_num_inference_steps)
        )
        self.action_scheduler.set_timesteps(action_steps)
        timesteps = F.pad(
            self.action_scheduler.timesteps,
            (0, 1),
            mode="constant",
            value=0,
        )
        for index, timestep in enumerate(timesteps):
            last_step = index == len(timesteps) - 1
            inputs = self._packed_streams(
                history_latents,
                predicted_latents,
                history_actions,
                current,
                text_emb,
                latent_timestep=0.0,
                action_timestep=float(timestep.item()) / 1000.0,
            )
            _, action_pred, _ = self.transformer(inputs, train_mode=True)
            if not last_step:
                action_pred = rearrange(
                    action_pred,
                    "b (f n) c -> b c f n 1",
                    f=history_actions.shape[2] + current.shape[2],
                )
                velocity = action_pred[:, :, -current.shape[2] :]
                current = self.action_scheduler.step(
                    velocity,
                    timestep,
                    current,
                    return_dict=False,
                )
                current = current * current_action_mask.to(current.dtype)
        return current

    @torch.no_grad()
    def _extract_clean_features(
        self,
        history_latents: torch.Tensor,
        predicted_latents: torch.Tensor,
        history_actions: torch.Tensor,
        predicted_actions: torch.Tensor,
        text_emb: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        inputs = self._packed_streams(
            history_latents,
            predicted_latents,
            history_actions,
            predicted_actions,
            text_emb,
            latent_timestep=self.config.phase3_v_feature_timestep,
            action_timestep=self.config.phase3_q_feature_timestep,
        )
        outputs = self.transformer(
            inputs,
            train_mode=True,
            return_features=True,
            critic_feature_layer=self.config.feature_layer,
        )
        features = outputs[3]
        if features["action_tokens"].shape[-1] != self.config.feature_dim:
            raise ValueError("Transformer feature_dim does not match config")
        chunk = self.config.infer_latent_chunk_size
        return {
            "previous_video_tokens": features["video_tokens"][:, :chunk].detach(),
            "action_tokens": features["action_tokens"][:, -chunk:].detach(),
        }

    @torch.no_grad()
    def _extract_real_current_video_features(
        self,
        history_latents: torch.Tensor,
        current_latents: torch.Tensor,
        history_actions: torch.Tensor,
        current_actions: torch.Tensor,
        text_emb: torch.Tensor,
    ) -> torch.Tensor:
        inputs = self._packed_streams(
            history_latents,
            current_latents,
            history_actions,
            current_actions,
            text_emb,
            latent_timestep=self.config.phase3_v_feature_timestep,
            action_timestep=0.0,
        )
        outputs = self.transformer(
            inputs,
            train_mode=True,
            return_features=True,
            critic_feature_layer=self.config.feature_layer,
        )
        features = outputs[3]
        if features["video_tokens"].shape[-1] != self.config.feature_dim:
            raise ValueError("Transformer feature_dim does not match config")
        chunk = self.config.infer_latent_chunk_size
        return features["video_tokens"][:, -chunk:].detach()

    def _split_robotwin_latents(
        self,
        latents: torch.Tensor,
    ) -> list[tuple[str, torch.Tensor]]:
        if self.base_config.env_type == "robotwin_tshape":
            high_h = int(self.base_config.height) // 16
            high_w = int(self.base_config.width) // 16
            wrist_h = high_h // 2
            wrist_w = high_w // 2
            wrist = latents[:, :, :, :wrist_h, :]
            high = latents[:, :, :, wrist_h : wrist_h + high_h, :high_w]
            left = wrist[:, :, :, :, :wrist_w]
            right = wrist[:, :, :, :, wrist_w : wrist_w + wrist_w]
            return [
                ("observation.images.cam_high", high),
                ("observation.images.cam_left_wrist", left),
                ("observation.images.cam_right_wrist", right),
            ]
        cameras = list(self.base_config.obs_cam_keys)
        width = latents.shape[-1] // len(cameras)
        return [
            (camera, latents[:, :, :, :, i * width : (i + 1) * width])
            for i, camera in enumerate(cameras)
        ]

    def _decode_latent_camera(self, latents: torch.Tensor) -> torch.Tensor:
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = (
            1.0
            / torch.tensor(self.vae.config.latents_std)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        decoded = self.vae.decode(
            latents / latents_std + latents_mean,
            return_dict=False,
        )[0]
        return (decoded.float() * 0.5 + 0.5).clamp(0.0, 1.0)

    @staticmethod
    def _align_jepa_to_latents(
        features: torch.Tensor,
        latent_frames: int,
    ) -> torch.Tensor:
        aligned = torch.empty(
            features.shape[0],
            latent_frames,
            features.shape[2],
            features.shape[3],
            features.shape[4],
            dtype=torch.float32,
            device=features.device,
        )
        feats = features.float()
        aligned[:, 0] = feats[:, 0]
        if latent_frames == 1:
            return aligned
        odd = feats[:, 1::2]
        even = feats[:, 2::2]
        pairs = min(odd.shape[1], even.shape[1], latent_frames - 1)
        if pairs > 0:
            aligned[:, 1 : 1 + pairs] = 0.5 * (
                odd[:, :pairs] + even[:, :pairs]
            )
        if pairs < latent_frames - 1:
            last = aligned[:, pairs] if pairs > 0 else aligned[:, 0]
            aligned[:, 1 + pairs :] = last[:, None].expand(
                -1, latent_frames - 1 - pairs, -1, -1, -1
            )
        return aligned

    @torch.no_grad()
    def _predicted_jepa_target(self, latents: torch.Tensor) -> torch.Tensor:
        aligned: list[torch.Tensor] = []
        for _, camera_latents in self._split_robotwin_latents(latents):
            video = self._decode_latent_camera(camera_latents)
            video = torch.cat([video[:, :, 0:1], video], dim=2)
            mean = torch.tensor(
                [0.485, 0.456, 0.406],
                device=video.device,
                dtype=video.dtype,
            ).view(1, 3, 1, 1, 1)
            std = torch.tensor(
                [0.229, 0.224, 0.225],
                device=video.device,
                dtype=video.dtype,
            ).view(1, 3, 1, 1, 1)
            feats = self.jepa_encoder((video - mean) / std)
            batch, _, frames, height, width = video.shape
            h_patches = height // 16
            w_patches = width // 16
            t_jepa = frames // 2
            feats = feats.view(
                batch,
                t_jepa,
                h_patches,
                w_patches,
                feats.shape[-1],
            )
            aligned.append(self._align_jepa_to_latents(feats, latents.shape[2]))

        if self.base_config.env_type == "robotwin_tshape":
            wrist = torch.cat(aligned[1:], dim=3)
            combined = torch.cat([wrist, aligned[0]], dim=2)
        else:
            combined = torch.cat(aligned, dim=3)
        batch, frames, height, width, dim = combined.shape
        pooled = F.avg_pool2d(
            combined.permute(0, 1, 4, 2, 3).flatten(0, 1),
            kernel_size=2,
            stride=2,
        )
        return pooled.view(batch, frames, dim, height // 2, width // 2).permute(
            0, 1, 3, 4, 2
        )

    @torch.no_grad()
    def _phase3_sample(
        self,
        batch: dict[str, torch.Tensor],
        current_prefix: str,
        history_prefix: str,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        history_latents = self._to_device(batch, f"{history_prefix}latents")
        current_latents = self._to_device(batch, f"{current_prefix}latents")
        history_actions = self._to_device(batch, f"{history_prefix}actions")
        current_actions = self._to_device(batch, f"{current_prefix}actions")
        current_action_mask = self._to_device(
            batch, f"{current_prefix}actions_mask"
        )
        text_emb = self._to_device(batch, f"{current_prefix}text_emb")

        predicted_latents = self._predict_video(
            history_latents,
            current_latents,
            text_emb,
        )
        predicted_actions = self._predict_actions(
            history_latents,
            predicted_latents,
            history_actions,
            current_actions,
            current_action_mask,
            text_emb,
        )
        features = self._extract_clean_features(
            history_latents,
            predicted_latents,
            history_actions,
            predicted_actions,
            text_emb,
        )
        return predicted_latents, predicted_actions, features

    def _train_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        predicted_latents, _, features = self._phase3_sample(
            batch,
            current_prefix="",
            history_prefix="previous_",
        )

        predicted_jepa = self._predicted_jepa_target(predicted_latents)
        actual_jepa = self._to_device(batch, "jepa_target")
        latent_mask = batch["latents_mask"].to(self.device, non_blocking=True)
        jepa_available = batch.get("jepa_available")
        if jepa_available is not None:
            jepa_available = jepa_available.to(self.device, non_blocking=True)
        distance = _masked_mean_distance(
            predicted_jepa,
            actual_jepa,
            latent_mask,
            jepa_available,
        )
        reward = -self.config.jepa_reward_weight * distance
        if self.config.include_sparse_success_reward:
            reward = reward + batch["reward"].to(self.device)

        assert self.bundle.value is not None
        assert self.bundle.target_value is not None
        action_mask = batch["actions_mask"].to(self.device, non_blocking=True)
        q1, q2 = self.bundle.q(features["action_tokens"], action_mask)
        previous_latent_mask = batch["previous_latents_mask"].to(
            self.device,
            non_blocking=True,
        )
        value = self.bundle.value(
            features["previous_video_tokens"],
            previous_latent_mask,
        )
        with torch.no_grad():
            current_real_video_tokens = self._extract_real_current_video_features(
                self._to_device(batch, "previous_latents"),
                self._to_device(batch, "latents"),
                self._to_device(batch, "previous_actions"),
                self._to_device(batch, "actions"),
                self._to_device(batch, "text_emb"),
            )
            next_value = self.bundle.target_value(
                current_real_video_tokens,
                latent_mask,
            )
        losses = iql_losses(
            q1=q1,
            q2=q2,
            value=value,
            next_target_value=next_value,
            reward=reward,
            discount=batch["discount"].to(self.device),
            expectile=self.config.expectile,
            value_weight=self.config.value_loss_weight,
            value_mask=batch["state_valid"].to(self.device),
        )

        self.optimizer.zero_grad(set_to_none=True)
        losses.total.backward()
        if self.world_size > 1:
            for parameter in self.bundle.parameters():
                if parameter.grad is not None:
                    dist.all_reduce(parameter.grad, op=dist.ReduceOp.AVG)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [
                parameter
                for parameter in self.bundle.parameters()
                if parameter.requires_grad
            ],
            max_norm=10.0,
        )
        self.optimizer.step()
        self.bundle.update_target(self.config.target_ema_rate)
        return {
            "total_loss": float(losses.total.detach().cpu()),
            "q_loss": float(losses.q.detach().cpu()),
            "value_loss": float(losses.value.detach().cpu()),
            "reward_mean": float(reward.detach().float().mean().cpu()),
            "jepa_distance": float(distance.detach().float().mean().cpu()),
            "q1_mean": float(q1.detach().float().mean().cpu()),
            "q2_mean": float(q2.detach().float().mean().cpu()),
            "grad_norm": float(torch.as_tensor(grad_norm).float().cpu()),
        }

    @torch.no_grad()
    def _parameter_norm(self) -> float:
        squared_norm = torch.zeros((), device=self.device, dtype=torch.float32)
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
                    self.base_config,
                    self.config,
                ),
                "training_distribution": self.config.training_distribution,
                "reward_source": self.config.reward_source,
                "phase3_video_exec_step": self.config.phase3_video_exec_step,
                "phase3_q_feature_timestep": self.config.phase3_q_feature_timestep,
                "phase3_v_feature_timestep": self.config.phase3_v_feature_timestep,
                "phase3_jepa_checkpoint": self.config.phase3_jepa_checkpoint,
                "value_state_alignment": (
                    "previous_real_video_for_v_current_real_video_for_target_v"
                ),
            },
        )

    def train(self) -> None:
        self.bundle.train()
        progress = tqdm(
            total=self.config.num_steps,
            initial=self.step,
            disable=self.rank != 0,
            desc="phase3-critic",
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
                        loss=f"{metrics['total_loss']:.4f}",
                        reward=f"{metrics['reward_mean']:.3f}",
                    )
                    if self.step % self.config.log_interval == 0:
                        entry = {
                            "step": self.step,
                            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            **metrics,
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
    trainer = Phase3CriticTrainer(config, rank, local_rank, world_size, device)
    trainer.train()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Phase 3 generated-video-conditioned IQL critics"
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    init_logger()
    main()
