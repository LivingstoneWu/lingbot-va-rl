# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from __future__ import annotations

import argparse
import os
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch.nn.attention.flex_attention import flex_attention as raw_flex_attention
from tqdm import tqdm

from wan_va.rl.guidance import (
    build_action_guidance_mask,
    build_clean_feature_input,
    denoising_time_scaling,
    load_q_guidance_artifact,
)
from wan_va.wan_va_server import (
    VA_CONFIGS,
    VA_Server,
    data_seq_to_patch,
    init_distributed,
    init_logger,
    logger,
    run_async_server_mode,
    save_async,
)


class QGuidedVA_Server(VA_Server):
    """VA server variant that applies Q-gradient guidance to action sampling."""

    def __init__(
        self,
        job_config,
        q_checkpoint: str,
        q_guidance_scale: float,
        q_guidance_beta: float = 2.0,
        q_guidance_start_step: int = 0,
        q_guidance_end_step: int = -1,
        q_guidance_interval: int = 1,
        q_objective: str = "min",
        q_grad_clip: float = 0.0,
        q_grad_normalize: bool = False,
        robotwin_eval: bool = False,
    ) -> None:
        super().__init__(job_config, robotwin_eval=robotwin_eval)
        self.q_guidance_scale = float(q_guidance_scale)
        self.q_guidance_beta = float(q_guidance_beta)
        self.q_guidance_start_step = int(q_guidance_start_step)
        self.q_guidance_end_step = int(q_guidance_end_step)
        if q_guidance_interval <= 0:
            raise ValueError("q_guidance_interval must be positive")
        self.q_guidance_interval = int(q_guidance_interval)
        self.q_objective = q_objective
        self.q_grad_clip = float(q_grad_clip)
        self.q_grad_normalize = bool(q_grad_normalize)
        self.q_artifact = load_q_guidance_artifact(q_checkpoint, self.device)
        self._validate_q_artifact()

    def _validate_q_artifact(self) -> None:
        config = self.q_artifact.config
        manifest = self.q_artifact.manifest
        if config.infer_latent_chunk_size != self.job_config.frame_chunk_size:
            raise ValueError(
                "Q checkpoint infer_latent_chunk_size must match inference "
                f"frame_chunk_size: {config.infer_latent_chunk_size} != "
                f"{self.job_config.frame_chunk_size}"
            )
        action_per_frame = manifest.get("action_per_frame")
        if (
            action_per_frame is not None
            and int(action_per_frame) != int(self.job_config.action_per_frame)
        ):
            raise ValueError(
                "Q checkpoint action_per_frame must match inference "
                f"action_per_frame: {action_per_frame} != "
                f"{self.job_config.action_per_frame}"
            )
        if config.feature_layer >= len(self.transformer.blocks):
            raise ValueError(
                f"Q checkpoint feature layer {config.feature_layer} is out "
                f"of range for {len(self.transformer.blocks)} transformer blocks"
            )

    def _should_apply_q_guidance(self, step_index: int) -> bool:
        if self.q_guidance_scale == 0.0:
            return False
        if step_index < self.q_guidance_start_step:
            return False
        if self.q_guidance_end_step >= 0 and step_index > self.q_guidance_end_step:
            return False
        if (step_index - self.q_guidance_start_step) % self.q_guidance_interval:
            return False
        return True

    def _scheduler_sigma(
        self,
        timestep: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        timestep_cpu = timestep.detach().cpu()
        timestep_id = torch.argmin(
            (self.action_scheduler.timesteps - timestep_cpu).abs()
        )
        sigma = self.action_scheduler.sigmas[timestep_id]
        return sigma.to(device=self.device, dtype=dtype)

    def _flex_attention_classes(self) -> set[type]:
        classes = set()
        for block in self.transformer.blocks:
            attn_op = getattr(block.attn1, "attn_op", None)
            attn_cls = type(attn_op)
            if hasattr(attn_cls, "flex_attn"):
                classes.add(attn_cls)
        return classes

    @contextmanager
    def _raw_flex_attention_context(self):
        saved = {
            attn_cls: attn_cls.__dict__.get("flex_attn")
            for attn_cls in self._flex_attention_classes()
        }
        for attn_cls in saved:
            attn_cls.flex_attn = staticmethod(raw_flex_attention)
        try:
            yield
        finally:
            for attn_cls, flex_attn in saved.items():
                if flex_attn is None:
                    delattr(attn_cls, "flex_attn")
                else:
                    attn_cls.flex_attn = flex_attn

    def _clear_flex_attention_masks(self) -> None:
        for block in self.transformer.blocks:
            attn_op = getattr(block.attn1, "attn_op", None)
            attn_cls = type(attn_op)
            if hasattr(attn_cls, "attention_mask"):
                attn_cls.attention_mask = None
            if hasattr(attn_cls, "cross_attention_mask"):
                attn_cls.cross_attention_mask = None

    @contextmanager
    def _q_feature_extraction_context(self):
        missing = object()
        saved_caches = []
        for block in self.transformer.blocks:
            caches = getattr(block.attn1, "attn_caches", None)
            if caches is None:
                continue
            saved = caches.get(self.cache_name, missing)
            saved_caches.append((caches, saved))
            caches[self.cache_name] = None
        try:
            yield
        finally:
            for caches, saved in saved_caches:
                if saved is missing:
                    caches.pop(self.cache_name, None)
                else:
                    caches[self.cache_name] = saved
            # The training-style feature pass creates square block masks for
            # the current chunk. The regular server forward attends over its
            # live KV cache, so those masks are invalid after this context.
            self._clear_flex_attention_masks()

    def _q_guided_velocity(
        self,
        actions: torch.Tensor,
        latents: torch.Tensor,
        action_velocity: torch.Tensor,
        timestep: torch.Tensor,
        action_cond: torch.Tensor | None,
        step_index: int,
    ) -> torch.Tensor:
        if not self._should_apply_q_guidance(step_index):
            return action_velocity

        sigma = self._scheduler_sigma(timestep, dtype=torch.float32)
        clean_action = (
            actions.float() - sigma.view(1, 1, 1, 1, 1) * action_velocity.float()
        ).detach()
        if action_cond is not None:
            clean_action = clean_action.clone()
            clean_action[:, :, 0:1] = action_cond.float()
        clean_action = clean_action.to(dtype=self.dtype).requires_grad_(True)

        action_mask = build_action_guidance_mask(
            clean_action,
            valid_action_channels=self.action_mask,
            clamp_first_frame=action_cond is not None,
        )
        feature_input = build_clean_feature_input(
            latents=latents.detach().to(dtype=self.dtype),
            actions=clean_action,
            text_emb=self.prompt_embeds.to(self.dtype).clone(),
            patch_size=tuple(self.job_config.patch_size),
            chunk_size=self.q_artifact.config.infer_latent_chunk_size,
            window_size=self.q_artifact.config.window_size,
        )
        with (
            torch.enable_grad(),
            self._q_feature_extraction_context(),
            self._raw_flex_attention_context(),
        ):
            outputs = self.transformer(
                feature_input,
                train_mode=True,
                return_features=True,
                critic_feature_layer=self.q_artifact.feature_layer,
            )
            q_objective = self.q_artifact.adapter.objective(
                outputs[3]["action_tokens"],
                action_mask,
                mode=self.q_objective,
            ).sum()
            grad = torch.autograd.grad(
                q_objective,
                clean_action,
                retain_graph=False,
                create_graph=False,
                only_inputs=True,
            )[0]

        grad = grad.float() * action_mask.to(grad.dtype)
        if self.q_grad_normalize:
            rms = (
                grad.square().sum()
                / action_mask.to(grad.dtype).sum().clamp_min(1.0)
            ).sqrt()
            grad = grad / rms.clamp_min(1e-6)
        if self.q_grad_clip > 0:
            grad = grad.clamp(-self.q_grad_clip, self.q_grad_clip)

        time = sigma.clamp(0.0, 1.0)
        step_scale = denoising_time_scaling(
            time, beta=self.q_guidance_beta
        ).view(1, 1, 1, 1, 1)
        guidance = (
            self.q_guidance_scale
            * step_scale
            * grad
            / sigma.clamp_min(1e-6).view(1, 1, 1, 1, 1)
        )
        guided_velocity = action_velocity.float() - guidance
        return guided_velocity.to(dtype=action_velocity.dtype)

    def _infer(self, obs, frame_st_id=0, initial_state=None):
        frame_chunk_size = self.job_config.frame_chunk_size
        if frame_st_id == 0:
            init_latent = self._encode_obs(obs)
            self.init_latent = init_latent

        latents = torch.randn(
            1,
            48,
            frame_chunk_size,
            self.latent_height,
            self.latent_width,
            device=self.device,
            dtype=self.dtype,
        )
        actions = torch.randn(
            1,
            self.job_config.action_dim,
            frame_chunk_size,
            self.action_per_frame,
            1,
            device=self.device,
            dtype=self.dtype,
        )

        self.scheduler.set_timesteps(self.job_config.num_inference_steps)
        self.action_scheduler.set_timesteps(
            self.job_config.action_num_inference_steps
        )
        timesteps = F.pad(
            self.scheduler.timesteps, (0, 1), mode="constant", value=0
        )
        video_step = self.job_config.video_exec_step
        if video_step != -1:
            timesteps = timesteps[:video_step]
        action_timesteps = F.pad(
            self.action_scheduler.timesteps,
            (0, 1),
            mode="constant",
            value=0,
        )

        with torch.no_grad():
            for i, t in enumerate(tqdm(timesteps)):
                last_step = i == len(timesteps) - 1
                latent_cond = (
                    init_latent[:, :, 0:1].to(self.dtype)
                    if frame_st_id == 0
                    else None
                )
                input_dict = self._prepare_latent_input(
                    latents,
                    None,
                    t,
                    t,
                    latent_cond,
                    None,
                    frame_st_id=frame_st_id,
                )
                self._clear_flex_attention_masks()
                video_noise_pred = self.transformer(
                    self._repeat_input_for_cfg(input_dict["latent_res_lst"]),
                    update_cache=1 if last_step else 0,
                    cache_name=self.cache_name,
                    action_mode=False,
                )
                if not last_step or video_step != -1:
                    video_noise_pred = data_seq_to_patch(
                        self.job_config.patch_size,
                        video_noise_pred,
                        frame_chunk_size,
                        self.latent_height,
                        self.latent_width,
                        batch_size=2 if self.use_cfg else 1,
                    )
                    if self.job_config.guidance_scale > 1:
                        video_noise_pred = (
                            video_noise_pred[1:]
                            + self.job_config.guidance_scale
                            * (video_noise_pred[:1] - video_noise_pred[1:])
                        )
                    else:
                        video_noise_pred = video_noise_pred[:1]
                    latents = self.scheduler.step(
                        video_noise_pred, t, latents, return_dict=False
                    )
                latents[:, :, 0:1] = (
                    latent_cond if frame_st_id == 0 else latents[:, :, 0:1]
                )

        if frame_st_id == 0:
            if initial_state is None:
                raise ValueError(
                    "_infer requires initial_state when frame_st_id == 0. "
                    "Pass the current robot state in used-channel space."
                )
            state = np.asarray(initial_state, dtype=np.float32)
            if state.ndim == 1:
                state = state[None, :]
            state_rep = np.tile(state[:1], (self.action_per_frame, 1))
            inv_ids = np.array(self.job_config.inverse_used_action_channel_ids)
            state_padded = np.pad(state_rep, ((0, 0), (0, 1)))
            state_aligned = state_padded[:, inv_ids]
            denom = np.maximum(self.q99 - self.q01, 1e-2)
            state_norm = np.where(
                self.action_valid[None, :],
                (state_aligned - self.q01[None, :])
                / (denom[None, :] + 1e-6)
                * 2.0
                - 1.0,
                0.0,
            )
            action_cond = torch.from_numpy(
                state_norm.T[None, :, None, :, None].astype(np.float32)
            ).to(self.device, self.dtype)
        else:
            action_cond = None

        for i, t in enumerate(tqdm(action_timesteps)):
            last_step = i == len(action_timesteps) - 1
            input_dict = self._prepare_latent_input(
                None,
                actions,
                t,
                t,
                None,
                action_cond,
                frame_st_id=frame_st_id,
            )
            with torch.no_grad():
                self._clear_flex_attention_masks()
                action_noise_pred = self.transformer(
                    self._repeat_input_for_cfg(input_dict["action_res_lst"]),
                    update_cache=1 if last_step else 0,
                    cache_name=self.cache_name,
                    action_mode=True,
                )

            if not last_step:
                action_noise_pred = rearrange(
                    action_noise_pred,
                    "b (f n) c -> b c f n 1",
                    f=frame_chunk_size,
                )
                if self.job_config.action_guidance_scale > 1:
                    action_noise_pred = (
                        action_noise_pred[1:]
                        + self.job_config.action_guidance_scale
                        * (action_noise_pred[:1] - action_noise_pred[1:])
                    )
                else:
                    action_noise_pred = action_noise_pred[:1]
                action_noise_pred = self._q_guided_velocity(
                    actions,
                    latents,
                    action_noise_pred,
                    t,
                    action_cond,
                    i,
                )
                with torch.no_grad():
                    actions = self.action_scheduler.step(
                        action_noise_pred,
                        t,
                        actions,
                        return_dict=False,
                    )
            actions[:, :, 0:1] = (
                action_cond if frame_st_id == 0 else actions[:, :, 0:1]
            )

        actions[:, ~self.action_mask] *= 0
        self.predicted_actions = actions.clone()
        save_async(
            latents, os.path.join(self.exp_save_root, f"latents_{frame_st_id}.pt")
        )
        save_async(
            actions, os.path.join(self.exp_save_root, f"actions_{frame_st_id}.pt")
        )
        with self._job_chunks_lock:
            self._job_latent_chunks.append(latents.cpu())
            self._job_action_chunks.append(actions.cpu())

        actions = self.postprocess_action(actions)
        torch.cuda.empty_cache()
        return actions, latents

    def infer(self, obs):
        reset = obs.get("reset", False)
        prompt = obs.get("prompt", None)
        compute_kv_cache = obs.get("compute_kv_cache", False)

        if reset:
            logger.info("******************* Reset server ******************")
            with torch.no_grad():
                self._reset(prompt=prompt)
            return dict()
        if compute_kv_cache:
            logger.info("################# Compute KV Cache #################")
            with torch.no_grad():
                self._compute_kv_cache(obs)
            return dict()

        logger.info("################# Infer One Q-Guided Chunk #################")
        action, _ = self._infer(
            obs,
            frame_st_id=self.frame_st_id,
            initial_state=obs.get("state", None),
        )
        return dict(action=action)


def run(args):
    config = VA_CONFIGS[args.config_name]
    config.wan22_finetuned_model_name_or_path = args.eval_model_path

    port = config.port if args.port is None else args.port
    if args.save_root is not None:
        config.save_root = args.save_root
    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        init_distributed(world_size, local_rank, rank)
    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size
    model = QGuidedVA_Server(
        config,
        q_checkpoint=args.q_checkpoint,
        q_guidance_scale=args.q_guidance_scale,
        q_guidance_beta=args.q_guidance_beta,
        q_guidance_start_step=args.q_guidance_start_step,
        q_guidance_end_step=args.q_guidance_end_step,
        q_guidance_interval=args.q_guidance_interval,
        q_objective=args.q_objective,
        q_grad_clip=args.q_grad_clip,
        q_grad_normalize=args.q_grad_normalize,
        robotwin_eval=args.robotwin,
    )
    if config.infer_mode == "i2va":
        logger.info(
            "******************************USE I2AV mode******************************"
        )
        model.generate()
    elif config.infer_mode == "server":
        logger.info(
            "***********************USE Q-Guided Server mode***********************"
        )
        metadata = {
            "action_per_frame": config.action_per_frame,
            "frame_chunk_size": config.frame_chunk_size,
            "q_guidance": "q_gradient_v1",
        }
        run_async_server_mode(model, local_rank, config.host, port, metadata=metadata)
    else:
        raise ValueError(f"Unknown infer mode: {config.infer_mode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", type=str, default="robotwin")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--save_root", type=str, default=None)
    parser.add_argument("--eval_model_path", type=str, default=None)
    parser.add_argument("--robotwin", action="store_true")
    parser.add_argument("--q_checkpoint", type=str, required=True)
    parser.add_argument("--q_guidance_scale", type=float, default=0.0)
    parser.add_argument("--q_guidance_beta", type=float, default=2.0)
    parser.add_argument("--q_guidance_start_step", type=int, default=0)
    parser.add_argument("--q_guidance_end_step", type=int, default=-1)
    parser.add_argument("--q_guidance_interval", type=int, default=1)
    parser.add_argument(
        "--q_objective",
        type=str,
        default="min",
        choices=("min", "mean", "q1", "q2"),
    )
    parser.add_argument("--q_grad_clip", type=float, default=0.0)
    parser.add_argument("--q_grad_normalize", action="store_true")
    args = parser.parse_args()
    run(args)
    logger.info("Finish all process!!!!!!!!!!!!")


if __name__ == "__main__":
    init_logger()
    main()
