# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import argparse
import os
import sys
import time
from functools import partial
from PIL import Image
from diffusers.video_processor import VideoProcessor
from diffusers.utils import export_to_video

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.pipelines.wan.pipeline_wan import prompt_clean
from einops import rearrange
from tqdm import tqdm


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs import VA_CONFIGS
from distributed.fsdp import shard_model
from distributed.util import _configure_model, init_distributed
from modules.utils import (
    WanVAEStreamingWrapper,
    load_text_encoder,
    load_tokenizer,
    load_transformer,
    load_vae,
)
from utils import (
    FlowMatchScheduler,
    data_seq_to_patch,
    get_mesh_id,
    init_logger,
    logger,
    run_async_server_mode,
    save_async,
)


class VA_Server:

    def __init__(self, job_config, robotwin_eval=False):
        self.cache_name = 'pos'
        self.job_config = job_config
        self.save_root = job_config.save_root
        self.dtype = job_config.param_dtype
        self.device = torch.device(f"cuda:{job_config.local_rank}")
        self.enable_offload = getattr(job_config, 'enable_offload', True)  # offload vae & text_encoder to save vram

        # Per-job accumulators: collect latent and action tensors from every
        # _infer() call, then cat + save them when the next _reset() fires.
        import threading as _threading
        self._job_latent_chunks: list = []
        self._job_action_chunks: list = []
        self._job_chunks_lock   = _threading.Lock()

        import atexit as _atexit
        _atexit.register(self._flush_job_chunks)

        self.scheduler = FlowMatchScheduler(shift=self.job_config.snr_shift,
                                            sigma_min=0.0,
                                            extra_one_step=True)
        self.action_scheduler = FlowMatchScheduler(
            shift=self.job_config.action_snr_shift,
            sigma_min=0.0,
            extra_one_step=True)
        self.scheduler.set_timesteps(1000, training=True)
        self.action_scheduler.set_timesteps(1000, training=True)

        self.vae = load_vae(
            os.path.join(job_config.wan22_pretrained_model_name_or_path,
                         'vae'),
            torch_dtype=self.dtype,
            torch_device='cpu' if self.enable_offload else self.device,
        )
        self.streaming_vae = WanVAEStreamingWrapper(self.vae)

        self.tokenizer = load_tokenizer(
            os.path.join(job_config.wan22_pretrained_model_name_or_path,
                         'tokenizer'), )

        self.text_encoder = load_text_encoder(
            os.path.join(job_config.wan22_pretrained_model_name_or_path,
                         'text_encoder'),
            torch_dtype=self.dtype,
            torch_device='cpu' if self.enable_offload else self.device,
        )

        self.transformer = load_transformer(
            #os.path.join(job_config.wan22_pretrained_model_name_or_path,
            os.path.join(job_config.wan22_finetuned_model_name_or_path,
                         'transformer'),
            torch_dtype=self.dtype,
            torch_device=self.device,
        )
        shard_fn = shard_model
        self.transformer = _configure_model(model=self.transformer,
                                            shard_fn=shard_fn,
                                            param_dtype=self.dtype,
                                            device=self.device,
                                            eval_mode=True,
                                            )

        self.env_type = job_config.env_type
        self.streaming_vae_half = None
        if self.env_type == 'robotwin_tshape':
            vae_half = load_vae(
                os.path.join(job_config.wan22_pretrained_model_name_or_path,
                             'vae'),
                torch_dtype=self.dtype,
                torch_device='cpu' if self.enable_offload else self.device,
            )
            self.streaming_vae_half = WanVAEStreamingWrapper(vae_half)
        # robotwin eval need to return action (C, F, H)
        self.robotwin_eval = robotwin_eval

    def _get_t5_prompt_embeds(
        self,
        prompt=None,
        num_videos_per_prompt=1,
        max_sequence_length=512,
        device=None,
        dtype=None,
    ):
        device = device or self.device
        dtype = dtype or self.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        text_encoder_device = next(self.text_encoder.parameters()).device
        prompt_embeds = self.text_encoder(text_input_ids.to(text_encoder_device),
                                          mask.to(text_encoder_device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack([
            torch.cat(
                [u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))])
            for u in prompt_embeds
        ],
                                    dim=0)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt,
                                           seq_len, -1)

        return prompt_embeds.to(device)

    def encode_prompt(
        self,
        prompt,
        negative_prompt=None,
        do_classifier_free_guidance=True,
        num_videos_per_prompt=1,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        max_sequence_length=226,
        device=None,
        dtype=None,
    ):
        r"""
        TODO
        """
        device = device or self.device
        dtype = dtype or self.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(
                negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(
                    negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}.")
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`.")

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
        return prompt_embeds, negative_prompt_embeds

    def normalize_latents(
        self,
        latents: torch.Tensor,
        latents_mean: torch.Tensor,
        latents_std: torch.Tensor,
    ) -> torch.Tensor:
        latents_mean = latents_mean.view(1, -1, 1, 1,
                                         1).to(device=latents.device)
        latents_std = latents_std.view(1, -1, 1, 1,
                                       1).to(device=latents.device)
        latents = ((latents.float() - latents_mean) * latents_std).to(latents)
        return latents

    # def preprocess_action(self, action):
    #     action_model_input = torch.from_numpy(action)
    #     CA, FA, HA = action_model_input.shape  # C, F, H
    #     action_model_input_paded = F.pad(action_model_input,
    #                                      [0, 0, 0, 0, 0, 1],
    #                                      mode='constant',
    #                                      value=0)

    #     action_model_input = action_model_input_paded[
    #         self.job_config.inverse_used_action_channel_ids]

    #     if self.action_norm_method == 'quantiles':
    #         action_model_input = (action_model_input - self.actions_q01) / (
    #             self.actions_q99 - self.actions_q01 + 1e-6) * 2. - 1.
    #     else:
    #         raise NotImplementedError
    #     return action_model_input.unsqueeze(0).unsqueeze(-1)  # B, C, F, H, W
    
    
    # ---- old preprocess_action (kept for reference, no longer used) ----------
    # def preprocess_action(self, action):
    #     """
    #     输入:
    #         action:
    #             1) 原始时序动作/状态: (T, D)，例如 (1, 7), (4, 7), (8, 7)
    #             2) 也兼容单帧: (D,) -> 自动转成 (1, D)
    #
    #     输出:
    #         action_model_input: (B, C, F, H, W)
    #             其中:
    #             - B = 1
    #             - C = len(inverse_used_action_channel_ids)
    #             - F = latent_frame_num
    #             - H = 4
    #             - W = 1
    #     """
    #     if isinstance(action, torch.Tensor):
    #         action = action.detach().cpu().numpy()
    #     else:
    #         action = np.asarray(action)
    #     if action.ndim == 1:
    #         action = action[None, :]
    #     elif action.ndim != 2:
    #         raise ValueError(f"preprocess_action expects action shape (T, D) or (D,), got {action.shape}")
    #     T, D = action.shape
    #     action_mask = np.ones_like(action, dtype=bool)
    #     pad_len = self.job_config.action_per_frame
    #     action = np.pad(action, pad_width=((pad_len, 0), (0, 0)), mode='constant', constant_values=0)
    #     action_mask = np.pad(action_mask, pad_width=((pad_len, 0), (0, 0)), mode='constant', constant_values=False)
    #     total_len = action.shape[0]
    #     required_action_num = ((total_len + 3) // 4) * 4
    #     if total_len < required_action_num:
    #         extra = required_action_num - total_len
    #         action = np.pad(action, pad_width=((0, extra), (0, 0)), mode='constant', constant_values=0)
    #         action_mask = np.pad(action_mask, pad_width=((0, extra), (0, 0)), mode='constant', constant_values=False)
    #     else:
    #         action = action[:required_action_num]
    #         action_mask = action_mask[:required_action_num]
    #     latent_frame_num = required_action_num // 4
    #     print("action dim before padding: ", action.shape)
    #     print("inverse_ids max: ", max(self.job_config.inverse_used_action_channel_ids))
    #     action_paded = np.pad(action, ((0, 0), (0, 1)), mode='constant', constant_values=0)
    #     action_mask_padded = np.pad(action_mask, ((0, 0), (0, 1)), mode='constant', constant_values=False)
    #     inverse_ids = np.array(self.job_config.inverse_used_action_channel_ids)
    #     action_aligned      = action_paded[:,       inverse_ids]
    #     action_mask_aligned = action_mask_padded[:, inverse_ids]
    #     if self.action_norm_method == 'quantiles':
    #         action_aligned[:, self.action_valid] = np.where(
    #             action_mask_aligned[:, self.action_valid],
    #             (action_aligned[:, self.action_valid] - self.q01[self.action_valid]) /
    #             (self.q99[self.action_valid] - self.q01[self.action_valid] + 1e-6) * 2.0 - 1.0,
    #             0.0
    #         )
    #         action_aligned[:, ~self.action_valid] = 0.0
    #     else:
    #         raise NotImplementedError
    #     action_aligned = rearrange(action_aligned, "(f n) c -> c f n 1", f=latent_frame_num)
    #     action_mask_aligned = rearrange(action_mask_aligned, "(f n) c -> c f n 1", f=latent_frame_num)
    #     action_aligned = action_aligned * action_mask_aligned
    #     action_model_input = torch.from_numpy(action_aligned).float().unsqueeze(0)
    #     return action_model_input
    # ---- end old preprocess_action -------------------------------------------

    def preprocess_action(self, action):
        """Encode observed robot states → model action conditioning tensor.

        Converts a sequence of real robot states into the format expected by
        the transformer's action conditioning input.  No zero-padding is
        applied; the caller is responsible for providing exactly
        F * action_per_frame rows (where F = frame_chunk_size).

        Args:
            action: array-like of shape (T, D) or (D,).
                T must equal frame_chunk_size * action_per_frame.
                D is the robot state dimension in *used-channel space*
                (same layout as what the client sends).

        Returns:
            torch.Tensor of shape (1, C_model, F, action_per_frame, 1),
            ready to be passed as action_model_input to _prepare_latent_input.
        """
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        else:
            action = np.asarray(action, dtype=np.float32)

        if action.ndim == 1:
            action = action[None, :]
        elif action.ndim != 2:
            raise ValueError(
                f"preprocess_action expects shape (T, D) or (D,), got {action.shape}"
            )

        T, D = action.shape
        if T % self.action_per_frame != 0:
            raise ValueError(
                f"T={T} must be divisible by action_per_frame={self.action_per_frame}. "
                f"Pass frame_chunk_size * action_per_frame rows."
            )
        F = T // self.action_per_frame

        # Pad D by 1 so that inverse_ids (which may equal len(used_ids) for
        # padding channels) always has a valid index to land on.
        inv_ids = np.array(self.job_config.inverse_used_action_channel_ids)
        action_padded = np.pad(action.astype(np.float32), ((0, 0), (0, 1)))  # (T, D+1)
        action_aligned = action_padded[:, inv_ids]                            # (T, C_model)

        # Per-channel quantile normalisation.
        if self.action_norm_method != 'quantiles':
            raise NotImplementedError(
                f"Unsupported action_norm_method: {self.action_norm_method!r}"
            )
        denom = np.maximum(self.q99 - self.q01, 1e-2)
        action_aligned[:, self.action_valid] = (
            (action_aligned[:, self.action_valid] - self.q01[self.action_valid])
            / (denom[self.action_valid] + 1e-6) * 2.0 - 1.0
        )
        action_aligned[:, ~self.action_valid] = 0.0

        # Reshape (T=F*N, C) → (C, F, N, 1) then add batch dim → (1, C, F, N, 1).
        action_aligned = rearrange(
            action_aligned, "(f n) c -> c f n 1",
            f=F, n=self.action_per_frame,
        )
        return torch.from_numpy(action_aligned.astype(np.float32)).float().unsqueeze(0)

    # def postprocess_action(self, action):
    #     action = action.cpu()  # B, C, F, H, W

    #     action = action[0, ..., 0]  #C, F, H
    #     if self.action_norm_method == 'quantiles':
    #         action = (action + 1) / 2 * (self.actions_q99 - self.actions_q01 +
    #                                      1e-6) + self.actions_q01
    #     else:
    #         raise NotImplementedError
    #     action = action.squeeze(0).detach().cpu().numpy()
    #     return action[self.job_config.used_action_channel_ids]
        
    
    # def postprocess_action(self, action):

    #     action = action.detach().cpu()   # (B, C, F, H, W)
    #     action = action[0, ..., 0]       # (C, F, H)

    #     if self.action_norm_method != 'quantiles':
    #         raise NotImplementedError

    #     # self.actions_q01 / self.actions_q99: (7, 1, 1)
    #     q01 = self.actions_q01.detach().cpu().numpy().reshape(self.actions_q01.shape[0], -1)[:, 0][None, :]
    #     q99 = self.actions_q99.detach().cpu().numpy().reshape(self.actions_q99.shape[0], -1)[:, 0][None, :]

    #     # pad 到 (1, 8)
    #     q01_paded = np.pad(q01, ((0, 0), (0, 1)), mode='constant', constant_values=0)
    #     q99_paded = np.pad(q99, ((0, 0), (0, 1)), mode='constant', constant_values=0)

    #     inverse_ids = np.array(self.job_config.inverse_used_action_channel_ids)

    #     # 对齐到模型通道空间
    #     q01_aligned = q01_paded[:, inverse_ids]   # (1, C_model)
    #     q99_aligned = q99_paded[:, inverse_ids]   # (1, C_model)

    #     valid = inverse_ids < q01.shape[1]        # 只有真实原始通道有效

    #     q01_aligned = torch.from_numpy(q01_aligned[0]).float().unsqueeze(-1).unsqueeze(-1)  # (C_model,1,1)
    #     q99_aligned = torch.from_numpy(q99_aligned[0]).float().unsqueeze(-1).unsqueeze(-1)  # (C_model,1,1)
    #     valid_t = torch.from_numpy(valid).bool().unsqueeze(-1).unsqueeze(-1)                 # (C_model,1,1)

    #     action = torch.where(
    #         valid_t,
    #         (action + 1.0) / 2.0 * (q99_aligned - q01_aligned + 1e-6) + q01_aligned,
    #         torch.zeros_like(action)
    #     )

    #     action = action.numpy()   # (C_model, F, H)
        
    #     action = action[self.job_config.used_action_channel_ids]

    #     return action
    
    def postprocess_action(self, action):
        action = action.detach().cpu()   # (B, C, F, H, W)
        action = action[0, ..., 0]       # (C, F, H)

        if self.action_norm_method != 'quantiles':
            raise NotImplementedError

        # Reshape to (C_model, 1, 1) for broadcasting against (C_model, F, H)
        q01 = torch.from_numpy(self.q01).float().view(-1, 1, 1)
        q99 = torch.from_numpy(self.q99).float().view(-1, 1, 1)
        valid = torch.from_numpy(self.action_valid).bool().view(-1, 1, 1)

        # 反归一化
        action = torch.where(
            valid,
            (action + 1.0) / 2.0 * (torch.maximum(q99 - q01, torch.tensor(1e-2)) + 1e-6) + q01,
            torch.zeros_like(action)
        )

        action = action.numpy()   # (C_model, F, H)

        # 取出真实用到的原始动作通道
        action = action[self.job_config.used_action_channel_ids]   # (C_used, F, H)
        # 通常这里是 (7, F, H)

        # COMMENT: 现在暂时用env_type 判断返回动作维度，Robotwin需要三维
        # Output shape depends on the downstream client.  The RoboTwin
        # evaluation client (eval_polict_client_*.py) indexes the returned
        # tensor as action[:, i, j] and inspects action.shape[0/1/2], i.e. it
        # expects 3-D (C_used, F, H).  All other clients (ManipArena adapter
        # wan_va_policy*.py, real-robot deployment) expect a flat time series
        # (F*H, C_used).  Branching on env_type is a pragmatic short-term fix:
        # currently each checkpoint is paired with exactly one client pipeline,
        # so env_type reliably identifies the expected shape.  A cleaner long-
        # term design would be a per-request flag in obs (e.g. obs['action_shape'])
        # or a handshake field, decoupling wire format from training metadata.
        if self.robotwin_eval:
            # Keep the model-native 3-D layout for the RoboTwin eval client.
            return action  # (C_used, F, H)

        # 变成时间序列格式: (C, F, H) -> (F, H, C) -> (F*H, C)
        action = np.transpose(action, (1, 2, 0))   # (F, H, C_used)
        action = action.reshape(-1, action.shape[-1])   # (N, C_used)

        return action
    
    def _repeat_input_for_cfg(self, input_dict):
        if self.use_cfg:
            input_dict['noisy_latents'] = input_dict['noisy_latents'].repeat(2, 1, 1, 1, 1)
            input_dict['text_emb'] = torch.cat([self.prompt_embeds.to(self.dtype).clone(), self.negative_prompt_embeds.to(self.dtype).clone()], dim=0)
            input_dict['grid_id'] = input_dict['grid_id'][None].repeat(2, 1, 1)
            input_dict['timesteps'] = input_dict['timesteps'][None].repeat(2, 1)
        else:
            input_dict['grid_id'] = input_dict['grid_id'][None]
            input_dict['timesteps'] = input_dict['timesteps'][None]
        return input_dict

    def _prepare_latent_input(self,
                              latent_model_input,
                              action_model_input,
                              latent_t=0,
                              action_t=0,
                              latent_cond=None,
                              action_cond=None,
                              frame_st_id=0,
                              patch_size=(1, 2, 2)):
        logger.info(f"FRAME START ID: {frame_st_id}")
        input_dict = dict()
        if latent_model_input is not None:
            input_dict['latent_res_lst'] = {
                'noisy_latents':
                latent_model_input,
                'timesteps':
                torch.ones([latent_model_input.shape[2]],
                           dtype=torch.float32,
                           device=self.device) * latent_t,
                'grid_id':
                get_mesh_id(latent_model_input.shape[-3] // patch_size[0],
                            latent_model_input.shape[-2] // patch_size[1],
                            latent_model_input.shape[-1] // patch_size[2], 0,
                            1, frame_st_id).to(self.device),
                'text_emb':
                self.prompt_embeds.to(self.dtype).clone(),
            }
            if latent_cond is not None:
                input_dict['latent_res_lst'][
                    'noisy_latents'][:, :, 0:1] = latent_cond[:, :, 0:1]
                input_dict['latent_res_lst']['timesteps'][0:1] *= 0

        if action_model_input is not None:
            input_dict['action_res_lst'] = {
                'noisy_latents':
                action_model_input,
                'timesteps':
                torch.ones([action_model_input.shape[2]],
                           dtype=torch.float32,
                           device=self.device) * action_t,
                'grid_id':
                get_mesh_id(action_model_input.shape[-3],
                            action_model_input.shape[-2],
                            action_model_input.shape[-1],
                            1,
                            1,
                            frame_st_id,
                            action=True).to(self.device),
                'text_emb':
                self.prompt_embeds.to(self.dtype).clone(),
            }

            if action_cond is not None:
                input_dict['action_res_lst'][
                    'noisy_latents'][:, :, 0:1] = action_cond[:, :, 0:1]
                input_dict['action_res_lst']['timesteps'][0:1] *= 0
            input_dict['action_res_lst']['noisy_latents'][:, ~self.
                                                          action_mask] *= 0
        return input_dict

    def _encode_obs(self, obs):
        images = obs['obs']
        if not isinstance(images, list):
            images = [images]
        if len(images) < 1:
            return None
        videos = []
        for k_i, k in enumerate(self.job_config.obs_cam_keys):
            if self.env_type == 'robotwin_tshape':
                if k_i == 0:  # camera high
                    height_i, width_i = self.height, self.width
                else:
                    height_i, width_i = self.height // 2, self.width // 2
            else:
                height_i, width_i = self.height, self.width

            history_video_k = torch.from_numpy(
                np.stack([each[k]
                          for each in images])).float().permute(3, 0, 1, 2)
            history_video_k = F.interpolate(history_video_k,
                                            size=(height_i, width_i),
                                            mode='bilinear',
                                            align_corners=False).unsqueeze(0)
            videos.append(history_video_k)

        if self.env_type == 'robotwin_tshape':
            videos_high = videos[0] / 255.0 * 2.0 - 1.0
            videos_left_and_right = torch.cat(videos[1:],
                                              dim=0) / 255.0 * 2.0 - 1.0
            vae_device = next(self.streaming_vae.vae.parameters()).device
            enc_out_high = self.streaming_vae.encode_chunk(
                videos_high.to(vae_device).to(self.dtype))
            enc_out_left_and_right = self.streaming_vae_half.encode_chunk(
                videos_left_and_right.to(vae_device).to(self.dtype))
            enc_out = torch.cat([
                torch.cat(enc_out_left_and_right.split(1, dim=0), dim=-1),
                enc_out_high
            ],
                                dim=-2)
        else:
            videos = torch.cat(videos, dim=0) / 255.0 * 2.0 - 1.0
            vae_device = next(self.streaming_vae.vae.parameters()).device
            videos_chunk = videos.to(vae_device).to(self.dtype)
            enc_out = self.streaming_vae.encode_chunk(videos_chunk)

        mu, logvar = torch.chunk(enc_out, 2, dim=1)
        latents_mean = torch.tensor(self.vae.config.latents_mean).to(mu.device)
        latents_std = torch.tensor(self.vae.config.latents_std).to(mu.device)
        mu_norm = self.normalize_latents(mu, latents_mean, 1.0 / latents_std)
        video_latent = torch.cat(mu_norm.split(1, dim=0), dim=-1)
        return video_latent.to(self.device)

    def _flush_job_chunks(self):
        """Concatenate and save latents+actions from the completed job.

        Called at the top of _reset() (before state is cleared) and also
        registered with atexit so any final partial job is saved on shutdown.
        Safe to call multiple times — snapshots and clears the lists atomically.
        """
        import threading as _t
        with self._job_chunks_lock:
            latent_chunks = list(self._job_latent_chunks)
            action_chunks = list(self._job_action_chunks)
            self._job_latent_chunks = []
            self._job_action_chunks = []

        if not latent_chunks:
            return

        save_dir = getattr(self, 'exp_save_root', None)
        if save_dir is None:
            return

        def _worker(latents, actions, d):
            try:
                cat_latents = torch.cat(latents, dim=2)  # (1, C, T_total, H, W)
                cat_actions = torch.cat(actions, dim=2)  # (1, C, T_total, N, 1)
                save_async(cat_latents.cpu(), os.path.join(d, 'latents_all.pt'))
                save_async(cat_actions.cpu(), os.path.join(d, 'actions_all.pt'))
                print(f"[flush_job] saved latents {tuple(cat_latents.shape)}  "
                      f"actions {tuple(cat_actions.shape)}  → {d}")
            except Exception as e:
                print(f"[flush_job] ERROR: {e}")

        _t.Thread(target=_worker, args=(latent_chunks, action_chunks, save_dir),
                  daemon=True).start()

    def _reset(self, prompt=None):
        # Save accumulated latents/actions from the job that just ended
        self._flush_job_chunks()

        logger.info('Reset.')
        self.use_cfg = (self.job_config.guidance_scale > 1) or (self.job_config.action_guidance_scale > 1)
        #### Reset all parameters
        self.frame_st_id = 0
        self.init_latent = None
        #### clean vae and transformer cache
        self.transformer.clear_cache(self.cache_name)
        self.streaming_vae.clear_cache()

        self.action_per_frame = self.job_config.action_per_frame
        self.height, self.width = self.job_config.height, self.job_config.width

        if self.env_type == 'robotwin_tshape':
            self.latent_height, self.latent_width = (
                (self.height // 16) * 3) // 2, self.width // 16
            self.streaming_vae_half.clear_cache()
        else:
            self.latent_height, self.latent_width = self.height // 16, self.width // 16 * len(
                self.job_config.obs_cam_keys)

        patch_size = self.job_config.patch_size
        latent_token_per_chunk = (self.job_config.frame_chunk_size *
                                  self.latent_height * self.latent_width) // (
                                      patch_size[0] * patch_size[1] *
                                      patch_size[2])
        action_token_per_chunk = self.job_config.frame_chunk_size * self.action_per_frame
        self.transformer.create_empty_cache(self.cache_name,
                                            self.job_config.attn_window,
                                            latent_token_per_chunk,
                                            action_token_per_chunk,
                                            dtype=self.dtype,
                                            device=self.device,
                                            batch_size = 2 if self.use_cfg else 1
                                            )

        self.predicted_actions = None

        self.action_mask = torch.zeros([self.job_config.action_dim]).bool()
        self.action_mask[self.job_config.used_action_channel_ids] = True

        used_ids    = np.array(self.job_config.used_action_channel_ids)
        inverse_ids = np.array(self.job_config.inverse_used_action_channel_ids)
        q01 = np.array(self.job_config.norm_stat['q01'], dtype=np.float32)
        q99 = np.array(self.job_config.norm_stat['q99'], dtype=np.float32)
        q01_used = np.pad(q01[used_ids], (0, 1))   # (n_used+1,)
        q99_used = np.pad(q99[used_ids], (0, 1))   # (n_used+1,)
        self.q01 = q01_used[inverse_ids]            # (C_model,)
        self.q99 = q99_used[inverse_ids]            # (C_model,)
        self.action_valid = inverse_ids < len(used_ids)   # (C_model,) bool

        self.action_norm_method = self.job_config.action_norm_method

        ##### get prompt
        if prompt is None:
            self.prompt_embeds = self.negative_prompt_embeds = None
        else:
            self.prompt_embeds, self.negative_prompt_embeds = self.encode_prompt(
                prompt=prompt,
                negative_prompt=None,
                do_classifier_free_guidance=self.job_config.guidance_scale > 1,
                num_videos_per_prompt=1,
                prompt_embeds=None,
                negative_prompt_embeds=None,
                max_sequence_length=512,
                device=self.device,
                dtype=self.dtype,
            )

        self.exp_name = f"{time.strftime('%Y%m%d_%H%M%S')}"
        self.exp_save_root = os.path.join(self.save_root, 'real', self.exp_name)
        os.makedirs(self.exp_save_root, exist_ok=True)
        torch.cuda.empty_cache()

    def _infer(self, obs, frame_st_id=0, initial_state=None):
        """
        Args:
            initial_state: Current robot state in raw used-channel space,
                shape (D_used,) or (T, D_used). Required when frame_st_id == 0.
                Repeated action_per_frame times, normalised inline using
                self.q01/q99/action_valid, and used as action_cond (inpainting)
                for the first latent frame.  Shape produced: (1, C_model, 1, N, 1).
        """
        frame_chunk_size = self.job_config.frame_chunk_size
        if frame_st_id == 0:
            init_latent = self._encode_obs(obs)
            self.init_latent = init_latent

        latents = torch.randn(1,
                              48,
                              frame_chunk_size,
                              self.latent_height,
                              self.latent_width,
                              device=self.device,
                              dtype=self.dtype)
        actions = torch.randn(1,
                              self.job_config.action_dim,
                              frame_chunk_size,
                              self.action_per_frame,
                              1,
                              device=self.device,
                              dtype=self.dtype)

        video_inference_step = self.job_config.num_inference_steps
        action_inference_step = self.job_config.action_num_inference_steps
        video_step = self.job_config.video_exec_step

        self.scheduler.set_timesteps(video_inference_step)
        self.action_scheduler.set_timesteps(action_inference_step)
        timesteps = self.scheduler.timesteps
        action_timesteps = self.action_scheduler.timesteps

        timesteps = F.pad(timesteps, (0, 1), mode='constant', value=0)

        if video_step != -1:
            timesteps = timesteps[:video_step]

        action_timesteps = F.pad(
            action_timesteps,
            (0,
             1),  # pad 1 element at the end (right side) of the last dimension
            mode='constant',
            value=0)

        with (
                torch.no_grad(),
        ):
            # 1. Video Generation Loop
            for i, t in enumerate(tqdm(timesteps)):
                last_step = i == len(timesteps) - 1
                latent_cond = init_latent[:, :, 0:1].to(
                    self.dtype) if frame_st_id == 0 else None
                input_dict = self._prepare_latent_input(
                    latents,
                    None,
                    t,
                    t,
                    latent_cond,
                    None,
                    frame_st_id=frame_st_id)

                video_noise_pred = self.transformer(
                    self._repeat_input_for_cfg(input_dict['latent_res_lst']),
                    update_cache=1 if last_step else 0,
                    cache_name=self.cache_name,
                    action_mode=False)

                if not last_step or video_step != -1:
                    video_noise_pred = data_seq_to_patch(
                        self.job_config.patch_size, video_noise_pred,
                        frame_chunk_size, self.latent_height,
                        self.latent_width, batch_size=2 if self.use_cfg else 1)
                    if self.job_config.guidance_scale > 1:
                        video_noise_pred = video_noise_pred[1:] + self.job_config.guidance_scale * (video_noise_pred[:1] - video_noise_pred[1:])
                    else:
                        video_noise_pred = video_noise_pred[:1]
                    latents = self.scheduler.step(video_noise_pred,
                                                  t,
                                                  latents,
                                                  return_dict=False)

                latents[:, :, 0:1] = latent_cond if frame_st_id == 0 else latents[:, :, 0:1]

            # Build action_cond once before the loop (only used at frame_st_id==0).
            if frame_st_id == 0:
                if initial_state is None:
                    raise ValueError(
                        "_infer requires initial_state when frame_st_id == 0. "
                        "Pass the current robot state in used-channel space."
                    )
                state = np.asarray(initial_state, dtype=np.float32)
                if state.ndim == 1:
                    state = state[None, :]               # (1, D_used)
                # Repeat the current state action_per_frame times → (N, D_used)
                state_rep = np.tile(state[:1], (self.action_per_frame, 1))
                # Pad to D_used+1 so inverse_ids (which may equal len(used_ids) for
                # padding channels) has a valid index to land on.
                inv_ids = np.array(self.job_config.inverse_used_action_channel_ids)
                state_padded  = np.pad(state_rep, ((0, 0), (0, 1)))   # (N, D_used+1)
                state_aligned = state_padded[:, inv_ids]               # (N, C_model)
                denom = np.maximum(self.q99 - self.q01, 1e-2)
                state_norm = np.where(
                    self.action_valid[None, :],
                    (state_aligned - self.q01[None, :]) / (denom[None, :] + 1e-6) * 2.0 - 1.0,
                    0.0,
                )  # (N, C_model)
                # Reshape to (1, C_model, 1, N, 1) — the expected action_cond shape
                action_cond = torch.from_numpy(
                    state_norm.T[None, :, None, :, None].astype(np.float32)
                ).to(self.device, self.dtype)             # (1, C_model, 1, N, 1)
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
                    frame_st_id=frame_st_id)
                action_noise_pred = self.transformer(
                    self._repeat_input_for_cfg(input_dict['action_res_lst']),
                    update_cache=1 if last_step else 0,
                    cache_name=self.cache_name,
                    action_mode=True)

                if not last_step:
                    action_noise_pred = rearrange(action_noise_pred,
                                                  'b (f n) c -> b c f n 1',
                                                  f=frame_chunk_size)
                    if self.job_config.action_guidance_scale > 1:
                        action_noise_pred = action_noise_pred[1:] + self.job_config.action_guidance_scale * (action_noise_pred[:1] - action_noise_pred[1:])
                    else:
                        action_noise_pred = action_noise_pred[:1]
                    actions = self.action_scheduler.step(action_noise_pred,
                                                         t,
                                                         actions,
                                                         return_dict=False)

                actions[:, :, 0:1] = action_cond if frame_st_id == 0 else actions[:, :, 0:1]

        actions[:, ~self.action_mask] *= 0

        # Buffer predicted actions for use as KV-cache action conditioning on the next call.
        self.predicted_actions = actions.clone()

        save_async(latents, os.path.join(self.exp_save_root, f'latents_{frame_st_id}.pt'))
        save_async(actions, os.path.join(self.exp_save_root, f'actions_{frame_st_id}.pt'))

        # Accumulate for whole-job concat save (flushed on next _reset / atexit)
        with self._job_chunks_lock:
            self._job_latent_chunks.append(latents.cpu())
            self._job_action_chunks.append(actions.cpu())

        actions = self.postprocess_action(actions)
        torch.cuda.empty_cache()
        return actions, latents

    def _compute_kv_cache(self, obs):
        ### optional async save obs for debug
        self.transformer.clear_pred_cache(self.cache_name)
        save_async(obs['obs'], os.path.join(self.exp_save_root, f'obs_data_{self.frame_st_id}.pt'))
        latent_model_input = self._encode_obs(obs)

        # Determine action conditioning:
        #   - If obs['state'] is provided (non-None), use preprocess_action for backward
        #     compatibility with clients that send real robot state.
        #   - Otherwise use the buffered predicted actions from the previous _infer call.
        #     This is the normal inference path: the robot sends no state, and we rely on
        #     the model's own predictions as conditioning (as agreed, prediction error is
        #     assumed small enough to be negligible).
        state = obs.get('state', None)
        if state is not None:
            action_model_input = self.preprocess_action(state).to(latent_model_input)
        else:
            if self.predicted_actions is None:
                raise RuntimeError(
                    "_compute_kv_cache called before any _infer; "
                    "predicted_actions is None. Send obs['state'] on the first call "
                    "or ensure _infer runs before _compute_kv_cache."
                )
            action_model_input = self.predicted_actions.to(latent_model_input)
        logger.info(
            f"get KV cache obs: {latent_model_input.shape} {action_model_input.shape}"
        )
        input_dict = self._prepare_latent_input(latent_model_input,
                                                action_model_input,
                                                frame_st_id=self.frame_st_id)

        with (
                torch.no_grad(),
        ):
            self.transformer(self._repeat_input_for_cfg(input_dict['latent_res_lst']),
                             update_cache=2,
                             cache_name=self.cache_name,
                             action_mode=False)

            self.transformer(self._repeat_input_for_cfg(input_dict['action_res_lst']),
                             update_cache=2,
                             cache_name=self.cache_name,
                             action_mode=True)
        torch.cuda.empty_cache()
        self.frame_st_id += latent_model_input.shape[2]

    @torch.no_grad()
    def infer(self, obs):
        reset = obs.get('reset', False)
        prompt = obs.get('prompt', None)
        compute_kv_cache = obs.get('compute_kv_cache', False)

        if reset:
            logger.info(f"******************* Reset server ******************")
            self._reset(prompt=prompt)
            return dict()
        elif compute_kv_cache:
            logger.info(
                f"################# Compute KV Cache #################")
            self._compute_kv_cache(obs)
            return dict()
        else:
            logger.info(f"################# Infer One Chunk #################")
            action, _ = self._infer(obs, frame_st_id=self.frame_st_id,
                                    initial_state=obs.get('state', None))
            return dict(action=action)
    
    def decode_one_video(self, latents, output_type):
        latents = latents.to(self.vae.dtype)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std + latents_mean
        video = self.vae.decode(latents, return_dict=False)[0]
        video = self.video_processor.postprocess_video(video, output_type=output_type)
        return video
    
    def load_init_obs(self):
        imf_dict = {v: np.array(Image.open(os.path.join(self.job_config.input_img_path, f"{v}.png")).convert("RGB")) for v in self.job_config.obs_cam_keys}
        init_obs = {}
        init_obs['obs'] = [imf_dict]
        return init_obs
    
    @torch.no_grad()
    def generate(self):
        self.video_processor = VideoProcessor(vae_scale_factor=1)
        self._reset(self.job_config.prompt)
        init_obs = self.load_init_obs()
        pred_latent_lst = []
        pred_action_lst = []
        # i2va demo mode: no real robot state available, use zeros for first chunk
        zero_state = np.zeros(len(self.job_config.used_action_channel_ids), dtype=np.float32)
        for chunk_id in range(self.job_config.num_chunks_to_infer):
            frame_st = chunk_id * self.job_config.frame_chunk_size
            actions, latents = self._infer(
                init_obs, frame_st_id=frame_st,
                initial_state=zero_state if frame_st == 0 else None)
            actions = torch.from_numpy(actions)
            pred_latent_lst.append(latents)
            pred_action_lst.append(actions)
        pred_latent = torch.cat(pred_latent_lst, dim=2)
        pred_action = torch.cat(pred_action_lst, dim=1).flatten(1)
        self.transformer.clear_cache(self.cache_name)
        self.streaming_vae.clear_cache()
        if self.streaming_vae_half:
            self.streaming_vae_half.clear_cache()
        del self.transformer
        del self.streaming_vae_half
        del self.text_encoder
        torch.cuda.empty_cache()
        
        # Move VAE to GPU for decoding
        if self.enable_offload:
            self.vae = self.vae.to(self.device).to(self.dtype)
        
        decoded_video = self.decode_one_video(pred_latent, 'np')[0]
        export_to_video(decoded_video, os.path.join(self.save_root, "demo.mp4"), fps=10)

def run(args):    
    
    config = VA_CONFIGS[args.config_name]
    ### added by lhc
    config.wan22_finetuned_model_name_or_path = args.eval_model_path

    port = config.port if args.port is None else args.port
    if args.save_root is not None:
        config.save_root = args.save_root
    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        init_distributed(world_size, local_rank, rank)
    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size
    model = VA_Server(config, robotwin_eval=args.robotwin)
    if config.infer_mode == 'i2va':
        logger.info(f"******************************USE I2AV mode******************************")
        model.generate()
    elif config.infer_mode == 'server':
        logger.info(f"******************************USE Server mode******************************")
        metadata = {
            'action_per_frame': config.action_per_frame,
            'frame_chunk_size': config.frame_chunk_size,
        }
        run_async_server_mode(model, local_rank, config.host, port, metadata=metadata)
    else:
        raise ValueError(f"Unknown infer mode: {config.infer_mode}")

def main():
    """
    TODO
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-name",
        type=str,
        required=False,
        default='robotwin',
        help="config name.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help='(start) port'
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default=None,
        help='save root'
    )
    parser.add_argument(
        "--eval_model_path",
        type=str,
        default=None,
        help='eval model path'
    )
    parser.add_argument(
        "--robotwin",
        action='store_true',
    )
    args = parser.parse_args()
    run(args)
    logger.info("Finish all process!!!!!!!!!!!!")


if __name__ == "__main__":
    init_logger()
    main()
