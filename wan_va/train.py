# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import argparse
import os
import sys
import time
from pathlib import Path
import wandb

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from safetensors.torch import save_file, load_file
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs import VA_CONFIGS
from distributed.fsdp import shard_model, apply_ac
from distributed.util import (
    _configure_model, 
    init_distributed, 
    dist_mean, 
    dist_max
)
from einops import rearrange
from modules.utils import (
    load_transformer,
)
from modules.model import JepaProjectionHead
from utils import (
    init_logger, 
    logger, 
    get_mesh_id, 
    sample_timestep_id,
    data_seq_to_patch,
    warmup_constant_lambda,
    FlowMatchScheduler
)

from dataset import MultiLatentLeRobotDataset
import gc
import re


def _config_to_dict(cfg) -> dict:
    """Recursively convert an EasyDict config to a plain dict safe for json.dump.

    Values that are not natively JSON-serialisable (e.g. torch.dtype, Path,
    numpy arrays) are converted to their str() representation so the file
    stays human-readable without losing any information.
    """
    import torch
    _NATIVE = (bool, int, float, str, type(None))

    def _convert(v):
        if isinstance(v, dict):
            return {str(k): _convert(vv) for k, vv in v.items()}
        if isinstance(v, (list, tuple)):
            converted = [_convert(x) for x in v]
            return converted
        if isinstance(v, _NATIVE):
            return v
        # torch.dtype, Path, numpy scalars, …
        return str(v)

    return _convert(dict(cfg))

from contextlib import nullcontext
# import setproctitle
# setproctitle.setproctitle('lingbot_lhc')


class Trainer:
    def __init__(self, config):
        if config.enable_wandb and config.rank == 0:
            wandb.login(host=os.environ['WANDB_BASE_URL'], key=os.environ['WANDB_API_KEY'])
            self.wandb = wandb
            self.wandb.init(
                entity=os.environ["WANDB_TEAM_NAME"],
                project=os.getenv("WANDB_PROJECT", "va_robotwin"),
                # dir=log_dir,
                config=config,
                mode="online",
                name='test_lln'
                # name=os.path.basename(os.path.normpath(job_config.job.dump_folder))
            )
            logger.info("WandB logging enabled")
        self.step = 0
        self.config = config
        self.device = torch.device(f"cuda:{config.local_rank}")
        self.dtype = config.param_dtype
        self.patch_size = config.patch_size

        # Load models
        logger.info("Loading models...")

        # Load and shard transformer with FSDP
        logger.info("Loading transformer...")

        if hasattr(config, 'resume_from') and config.resume_from:
            transformer_path = os.path.join(config.resume_from, 'transformer')
            if config.rank == 0:
                logger.info(f"Resuming from checkpoint: {transformer_path}")
        else:
            transformer_path = os.path.join(config.wan22_pretrained_model_name_or_path, 'transformer')

        self.transformer = load_transformer(
            transformer_path,
            torch_dtype=torch.float32,
            torch_device='cpu',
        )

        logger.info("Setting up activation checkpointing ...")
        apply_ac(self.transformer)

        logger.info("Setting up FSDP...")
        shard_fn = shard_model
        self.transformer = _configure_model(
            model=self.transformer,
            shard_fn=shard_fn,
            param_dtype=self.dtype,
            device=self.device,
            eval_mode=False,
        )
        self.transformer.train()
        self.transformer.requires_grad_(True)

        # ── JEPA projection head (separate from FSDP transformer) ────────────
        if getattr(config, 'jepa_loss_enabled', False):
            self.jepa_head = JepaProjectionHead(
                inner_dim=3072,   # 24 heads × 128 dim
                jepa_dim=1664,    # JEPA ViT-Gigantic embed_dim
                head_type=config.jepa_head_type,
            ).to(device=self.device, dtype=self.dtype)
            self.jepa_head.train()
            self.jepa_head.requires_grad_(True)
            if hasattr(config, 'resume_from') and config.resume_from:
                jepa_head_resume = Path(config.resume_from) / "jepa_head.pt"
                if jepa_head_resume.exists():
                    self.jepa_head.load_state_dict(
                        torch.load(jepa_head_resume, map_location=self.device, weights_only=True)
                    )
                    if config.rank == 0:
                        logger.info(f"Loaded jepa_head from {jepa_head_resume}")
                elif config.rank == 0:
                    logger.warning(f"[JEPA] resume_from set but no jepa_head.pt found at {jepa_head_resume}")
        else:
            self.jepa_head = None

        # Optimizer
        trainable_params = [p for p in self.transformer.parameters() if p.requires_grad]
        if self.jepa_head is not None:
            trainable_params += list(self.jepa_head.parameters())
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=1e-8,
            weight_decay=config.weight_decay,
            fused=True,
            foreach=False,
        )

        warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1e-8,   # avoid exactly 0
            end_factor=1.0,
            total_iters=config.warmup_steps,
        )
        cosine_annealing_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_steps - config.warmup_steps,
            eta_min=config.min_lr,
        )
        self.lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[warmup_lr_scheduler, cosine_annealing_lr_scheduler],    
            milestones=[config.warmup_steps]
        )


        # Setup dataloaders
        logger.info("Setting up datasets...")
        train_dataset = MultiLatentLeRobotDataset(config=config)
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=config.world_size,
            rank=config.rank,
            shuffle=True,
            seed=42
        ) if config.world_size > 1 else None
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=(train_sampler is None), 
            num_workers=config.load_worker,
            sampler=train_sampler,
        )

        self.train_scheduler_latent = FlowMatchScheduler(shift=self.config.snr_shift, sigma_min=0.0, extra_one_step=True)
        self.train_scheduler_latent.set_timesteps(1000, training=True)
        self.train_scheduler_action = FlowMatchScheduler(shift=self.config.action_snr_shift, sigma_min=0.0, extra_one_step=True)
        self.train_scheduler_action.set_timesteps(1000, training=True)

        self.save_dir = Path(config.save_root) / "checkpoints"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.save_dir / 'log.jsonl'

        if config.rank == 0:
            config_save_path = self.save_dir / 'config.json'
            with open(config_save_path, 'w') as _f:
                json.dump(_config_to_dict(config), _f, indent=2)
            logger.info(f"Config saved to {config_save_path}")

            if getattr(config, 'jepa_loss_enabled', False):
                missing = train_dataset.get_missing_jepa_files()
                if missing:
                    missing_log = self.save_dir / 'missing_jepa.log'
                    missing_log.write_text('\n'.join(missing) + '\n')
                    logger.warning(
                        f"[JEPA] {len(missing)} missing JEPA file(s) — "
                        f"see {missing_log}"
                    )
                else:
                    logger.info("[JEPA] All JEPA feature files present.")

        self.gradient_accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)
        self.grad_log_freq   = getattr(config, 'grad_log_freq', 0)
        self.grad_stats_file = self.save_dir / 'grad_stats.jsonl'
        self.train_loader_iter = None
        # if hasattr(config, 'resume_from') and config.resume_from:
        #     self._load_training_state(config.resume_from)
    
    def _get_next_batch(self):
        """Get next batch from iterator, reset if epoch is finished."""
        if self.train_loader_iter is None:
            self.train_loader_iter = iter(self.train_loader)
        
        try:
            batch = next(self.train_loader_iter)
        except StopIteration:
            # Reset sampler and iterator when epoch finishes
            if hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(self.train_loader.sampler.epoch + 1)
            self.train_loader_iter = iter(self.train_loader)
            batch = next(self.train_loader_iter)
        
        return batch

    @torch.no_grad()
    def _add_noise(self, latent, train_scheduler, action_mask=False, action_mode=False, noisy_cond_prob=0.):
        B, C, F, H, W = latent.shape

        timestep_ids = sample_timestep_id(batch_size=F, num_train_timesteps=train_scheduler.num_train_timesteps)
        noise = torch.zeros_like(latent).normal_()
        timesteps = train_scheduler.timesteps[timestep_ids].to(device=self.device)
        noisy_latents =train_scheduler.add_noise(latent, noise, timesteps, t_dim=2)
        targets =train_scheduler.training_target(latent, noise, timesteps)

        patch_f, patch_h, patch_w = self.patch_size
        if action_mode:
            patch_f = patch_h = patch_w = 1
        
        latent_grid_id = get_mesh_id(
            latent.shape[-3] // patch_f,  # F
            latent.shape[-2] // patch_h,  # H
            latent.shape[-1] // patch_w,  # W
            t=1 if action_mode else 0,  # 1 for action mode (0 for latent), not used
            f_w=1,
            f_shift=0,
            action=action_mode
        ).to(self.device)  # shape: [4, seq_len]
        latent_grid_id = latent_grid_id[None].repeat(B, 1, 1)

        if torch.rand(1).item() < noisy_cond_prob:
            cond_timestep_ids = sample_timestep_id(
                    batch_size=F,
                    min_timestep_bd=0.5, 
                    max_timestep_bd=1.0, 
                    num_train_timesteps=train_scheduler.num_train_timesteps,
                )
            noise = torch.zeros_like(latent).normal_()
            cond_timesteps = train_scheduler.timesteps[cond_timestep_ids].to(device=self.device)
            latent = train_scheduler.add_noise(latent, noise, cond_timesteps, t_dim=2)
        else:
            cond_timesteps = torch.zeros_like(timesteps)

        if action_mask is not None:
            noisy_latents *= action_mask.float()
            targets *= action_mask.float()
            latent *= action_mask.float()

        return dict(
            timesteps=timesteps[None].repeat(B, 1),
            noisy_latents=noisy_latents,
            targets=targets,
            latent=latent,
            cond_timesteps=cond_timesteps[None].repeat(B, 1),
            grid_id=latent_grid_id,
        )

    @torch.no_grad()
    def _prepare_input_dict(self, batch_dict):
        """Prepare input dict following infer code pattern from wan_va_server.py."""
        # Generate grid_id following infer code (no batch dimension yet)
        # For action mode: get_mesh_id(shape[-3], shape[-2], shape[-1], t=1, f_w=1, f_shift, action=True)
        latent_dict = self._add_noise(
            latent=batch_dict['latents'], 
            train_scheduler=self.train_scheduler_latent, 
            action_mask=None, 
            action_mode=False,
            noisy_cond_prob=0.5)
        
        action_dict = self._add_noise(
            latent=batch_dict['actions'], 
            train_scheduler=self.train_scheduler_action, 
            action_mask=batch_dict['actions_mask'], 
            action_mode=True,
            noisy_cond_prob=0.0)

        latent_dict['text_emb']     = batch_dict['text_emb']
        latent_dict['latents_mask'] = batch_dict['latents_mask']   # (B, F_max) bool
        action_dict['text_emb']     = batch_dict['text_emb']
        action_dict['actions_mask'] = batch_dict['actions_mask']

        input_dict = {
            'latent_dict': latent_dict,
            'action_dict': action_dict,
            'chunk_size': torch.randint(1, 5, (1,)).item(),
            'window_size': torch.randint(4, 65, (1,)).item(),
        }
        return input_dict

    def convert_input_format(self, input_dict):
        """Convert input dict to match transformer input format if needed."""
        for key, value in input_dict.items():
            input_dict[key] = value.to(self.device).to(self.dtype)
        return input_dict

    def compute_loss(self,
        input_dict,
        pred
    ):
        latent_pred, action_pred = pred
        action_pred = rearrange(action_pred, 'b (f n) c -> b c f n 1', f=input_dict['action_dict']['targets'].shape[-3])
        latent_pred = data_seq_to_patch(
                        self.patch_size, latent_pred,
                        input_dict['latent_dict']['targets'].shape[-3], input_dict['latent_dict']['targets'].shape[-2],
                        input_dict['latent_dict']['targets'].shape[-1], batch_size=latent_pred.shape[0])
        Bn, Fn = input_dict['latent_dict']['timesteps'].shape
        latent_loss_weight = self.train_scheduler_latent.training_weight(input_dict['latent_dict']['timesteps'].flatten()).reshape(Bn, Fn)
        action_loss_weight = self.train_scheduler_action.training_weight(input_dict['action_dict']['timesteps'].flatten()).reshape(Bn, Fn)

        # Frame-wise video loss calculation
        latent_loss = F.mse_loss(latent_pred.float(), input_dict['latent_dict']['targets'].float().detach(), reduction='none')
        latent_loss = latent_loss * latent_loss_weight[:, None, :, None, None]
        # Permute to (B, F, H, W, C) and flatten to (B*F, H*W*C)
        latent_loss = latent_loss.permute(0, 2, 3, 4, 1)  # (B, C, F, H, W) -> (B, F, H, W, C)
        latent_loss = latent_loss.flatten(0, 1).flatten(1)  # (B, F, H, W, C) -> (B*F, H*W*C)
        # Sum per frame and compute element count per frame
        latent_loss_per_frame = latent_loss.sum(dim=1)           # (B*F_max,)
        latent_elems_per_frame = torch.ones_like(latent_loss).sum(dim=1)  # (B*F_max,)

        # latents_mask: (B, F_max) → (B*F_max,) float; zeros out padded frames.
        # flatten() preserves the same (B, F_max) → (B*F_max,) row-major order as
        # latent_loss.permute(...).flatten(0,1), so frame indices align correctly.
        lm = input_dict['latent_dict']['latents_mask'].flatten().float()  # (B*F_max,)
        latent_loss = (
            (latent_loss_per_frame / (latent_elems_per_frame + 1e-6)) * lm
        ).sum() / (lm.sum() + 1e-6)

        # Frame-wise action loss calculation
        action_loss = F.mse_loss(action_pred.float(), input_dict['action_dict']['targets'].float().detach(), reduction='none')
        # COMMENT: DEBUG
        with torch.no_grad():
            raw = F.mse_loss(action_pred.float(), 
                         input_dict['action_dict']['targets'].float(),
                         reduction='none')
            msk = input_dict['action_dict']['actions_mask'].float()
            # Per-element loss for ONLY the 7 active channels
            active_raw = (raw * msk).sum() / (msk.sum() + 1e-6)
            wt = action_loss_weight[:, None, :, None, None]
            active_weighted = (raw * wt * msk).sum() / (msk.sum() + 1e-6)
            logger.info(f"[RAW MSE] active-only raw={active_raw:.4f}  weighted={active_weighted:.4f}  ratio={active_weighted/active_raw:.4f}")

        action_loss = action_loss * action_loss_weight[:, None, :, None, None]
        action_loss = action_loss * input_dict['action_dict']['actions_mask'].float()
        # Permute to (B, F, H, W, C) and flatten to (B*F, H*W*C)
        action_loss = action_loss.permute(0, 2, 3, 4, 1)  # (B, C, F, H, W) -> (B, F, H, W, C)
        action_mask = input_dict['action_dict']['actions_mask'].float().permute(0, 2, 3, 4, 1)  # (B, C, F, H, W) -> (B, F, H, W, C)
        action_loss = action_loss.flatten(0, 1).flatten(1)  # (B, F, H, W, C) -> (B*F, H*W*C)
        action_mask = action_mask.flatten(0, 1).flatten(1)  # (B, F, H, W, C) -> (B*F, H*W*C)
        # Sum per frame, normalise by real action slots, then exclude padded frames
        # via lm (same frame ordering as latent loss, reused from above).
        action_loss_per_frame = action_loss.sum(dim=1)  # (B*F_max,)
        action_mask_per_frame = action_mask.sum(dim=1)  # (B*F_max,)
        action_loss = (
            (action_loss_per_frame / (action_mask_per_frame + 1e-6)) * lm
        ).sum() / (lm.sum() + 1e-6)

        return latent_loss / self.gradient_accumulation_steps, action_loss / self.gradient_accumulation_steps

    def compute_jepa_loss(
        self,
        jepa_hidden: torch.Tensor,
        batch: dict,
        input_dict: dict,
    ) -> torch.Tensor:
        """
        Cosine similarity loss between projected DiT hidden states and JEPA targets.

        jepa_hidden: [1, B*F*H_tokens*W_tokens, inner_dim] — noisy-latent stream
                     at config.jepa_loss_layer, captured inside forward_train.

        jepa_target in the batch is already 2×2-pooled in the dataset loader so
        that its spatial dims match the DiT token grid (VAE stride 16px per cell,
        DiT patch_size (1,2,2) → 32px per token; JEPA patch 16px → pool by 2).
        Returns the unscaled mean loss (caller divides by gradient_accumulation_steps).
        """
        jepa_target    = batch['jepa_target']          # [B, F, H_tok, W_tok, D]  bfloat16
        jepa_available = batch.get('jepa_available')   # [B] float (0 or 1 after convert)
        latent_dict    = input_dict['latent_dict']

        B     = latent_dict['noisy_latents'].shape[0]
        F_lat = latent_dict['noisy_latents'].shape[2]
        H_tok = latent_dict['noisy_latents'].shape[3] // self.patch_size[1]
        W_tok = latent_dict['noisy_latents'].shape[4] // self.patch_size[2]

        assert jepa_target.shape[2] == H_tok and jepa_target.shape[3] == W_tok, (
            f"JEPA target spatial dims {jepa_target.shape[2:][:2]} do not match "
            f"DiT token grid ({H_tok}, {W_tok}). "
            f"Check that JEPA features were extracted at native camera resolution "
            f"and that _load_jepa_target applied 2×2 avg-pool."
        )

        # Reshape hidden states to match spatial layout of jepa_target
        hidden = jepa_hidden.squeeze(0).reshape(B, F_lat, H_tok, W_tok, -1)

        # Project: [B, F, H', W', inner_dim] → [B, F, H', W', jepa_dim]
        jepa_pred = self.jepa_head(hidden.to(self.dtype))

        # L2-normalise both sides; dot product = cosine similarity
        jepa_pred   = F.normalize(jepa_pred.float(),              dim=-1)
        jepa_target = F.normalize(jepa_target.float(),            dim=-1)

        cos_sim       = (jepa_pred * jepa_target).sum(dim=-1)  # [B, F, H', W']
        loss_per_frame = 1.0 - cos_sim.mean(dim=(-2, -1))     # [B, F]

        # Validity mask: real latent frames, within timestep gate, available samples
        lm     = latent_dict['latents_mask'].float()           # [B, F]
        # timesteps are in [0, num_train_timesteps] (= sigmas * 1000); normalise
        # so jepa_loss_t_max stays in the intuitive [0, 1] range in the config.
        # t_max = 1.0 → all timesteps contribute; 0.8 → top-80%-noise only.
        t_norm = latent_dict['timesteps'].float() / self.train_scheduler_latent.num_train_timesteps
        t_gate = (t_norm < self.config.jepa_loss_t_max).float()

        if jepa_available is not None:
            avail = jepa_available.float().view(B, 1)
        else:
            avail = torch.ones(B, 1, device=self.device)

        mask = lm * t_gate * avail                             # [B, F]
        denom = mask.sum() + 1e-6
        return (loss_per_frame * mask).sum() / denom

    def _train_step(self, batch, batch_idx):
        """Train a single batch, returns losses for logging."""
        batch = self.convert_input_format(batch)
        input_dict = self._prepare_input_dict(batch)

        should_sync = (batch_idx + 1) % self.gradient_accumulation_steps == 0

        if not should_sync:
            self.transformer.set_requires_gradient_sync(False)
        else:
            self.transformer.set_requires_gradient_sync(True)

        jepa_layer = (
            self.config.jepa_loss_layer
            if getattr(self.config, 'jepa_loss_enabled', False)
            else -1
        )

        # ── PROFILER START (temporary – remove after debugging) ───────────────
        # _do_profile = (
        #     self.config.rank == 0
        #     and not getattr(self, '_profiler_done', False)
        # )
        # if _do_profile:
        #     _prof = torch.profiler.profile(
        #         activities=[
        #             torch.profiler.ProfilerActivity.CPU,
        #             torch.profiler.ProfilerActivity.CUDA,
        #         ],
        #         with_stack=False,
        #         record_shapes=False,
        #     )
        #     _prof.__enter__()
        # ── PROFILER START END ────────────────────────────────────────────────

        latent_pred, action_pred, jepa_hidden = self.transformer(
            input_dict, train_mode=True, jepa_capture_layer=jepa_layer
        )
        latent_loss, action_loss = self.compute_loss(
            input_dict, (latent_pred, action_pred)
        )

        jepa_loss_raw = torch.tensor(0.0, device=self.device)
        if self.jepa_head is not None and jepa_hidden is not None:
            jepa_loss_raw = self.compute_jepa_loss(jepa_hidden, batch, input_dict)

        loss = (
            latent_loss
            + action_loss
            + self.config.jepa_loss_weight
            * jepa_loss_raw
            / self.gradient_accumulation_steps
        )
        loss.backward()

        # Manually sync jepa_head gradients across ranks (head is not FSDP-wrapped).
        if should_sync and self.jepa_head is not None and self.config.world_size > 1:
            for p in self.jepa_head.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

        # ── PROFILER END (temporary – remove after debugging) ─────────────────
        # if _do_profile:
        #     torch.cuda.synchronize()
        #     _prof.__exit__(None, None, None)
        #     logger.info(
        #         "\n[PROFILER] Top 20 ops by CUDA time:\n" +
        #         _prof.key_averages().table(
        #             sort_by="cuda_time_total", row_limit=20
        #         )
        #     )
        #     self._profiler_done = True   # only profile once
        # ── PROFILER END END ──────────────────────────────────────────────────

        losses = {
            'latent_loss': latent_loss.detach(),
            'action_loss': action_loss.detach(),
            'jepa_loss': (jepa_loss_raw / self.gradient_accumulation_steps).detach(),
        }

        # Only update weights after accumulating gradients
        if should_sync:
            # DTensor (FSDP2 transformer) and plain Tensor (jepa_head) cannot
            # be mixed in a single clip_grad_norm_ call (_foreach_mul_ rejects
            # heterogeneous tensor types).  Clip each group separately; the
            # transformer call is identical to the pre-jepa version.
            total_norm = torch.nn.utils.clip_grad_norm_(
                list(self.transformer.parameters()), 2.0
            )
            if self.jepa_head is not None:
                torch.nn.utils.clip_grad_norm_(
                    list(self.jepa_head.parameters()), 2.0
                )
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

            losses['total_norm'] = total_norm
            losses['should_log'] = True
        else:
            losses['should_log'] = False

        return losses

    # ── Gradient alignment helpers ──────────────────────────────────────────

    @staticmethod
    def _local_tensor(t: torch.Tensor) -> torch.Tensor:
        """Return the plain local shard of a DTensor, or the tensor itself."""
        if hasattr(t, 'to_local'):
            return t.to_local()
        return t

    def _snapshot_block_grads(self) -> dict:
        """
        Copy this rank's gradient shards to CPU (bfloat16) keyed by parameter name.
        Every tensor is flattened to 1-D.  Params with grad=None are stored as a
        sentinel (None) so that _write_grad_stats can canonicalise sizes correctly.

        Returns {param_name (str): cpu bfloat16 1-D tensor, or None}.
        """
        snap: dict = {}
        for name, param in self.transformer.named_parameters():
            if param.grad is not None:
                snap[name] = (self._local_tensor(param.grad)
                              .detach().flatten()
                              .to(device='cpu', dtype=torch.bfloat16))
            else:
                snap[name] = None   # resolved to zeros in _write_grad_stats
        return snap

    def _write_grad_stats(
        self,
        block_grads_per_term: dict,
        mean_t_latent: float,
        mean_t_action: float,
    ) -> None:
        """
        Compute per-block pairwise cosine similarities and gradient magnitudes
        via a single all-reduce, then write one JSON record to grad_stats.jsonl.

        block_grads_per_term: {term: {param_name: cpu bfloat16 1-D tensor}}

        Statistics are accumulated on CPU as scalar running sums (one param at a
        time), so no large GPU or CPU buffers are allocated.  A single small
        scalar tensor is all-reduced to recover global statistics across ranks.
        Only rank 0 writes the file.
        """
        terms = list(block_grads_per_term.keys())
        if len(terms) < 2:
            return

        pairs = [(ta, tb) for i, ta in enumerate(terms) for tb in terms[i + 1:]]

        # Discover all block ids from parameter names (same set for every term).
        param_names = list(next(iter(block_grads_per_term.values())).keys())
        bid_for: dict = {}
        for name in param_names:
            m = re.search(r'blocks\.(\d+)\.', name)
            bid_for[name] = int(m.group(1)) if m else -1
        all_bids = sorted(set(bid_for.values()))

        if not all_bids:
            return

        # ── Canonicalise per-param sizes and accumulate stats on CPU ─────────
        # Rule: the canonical numel for a param is the numel of the FIRST term
        # whose snapshot is not None (i.e. the first term that actually produced
        # a gradient for this param).  None-grad terms get zero tensors of that
        # canonical size.  If two non-None snapshots have DIFFERENT numels (an
        # FSDP2 sharding-state inconsistency across backward passes), the param
        # is skipped entirely for that step — better to omit than to corrupt.
        sq_accum:  dict = {term: {bid: 0.0 for bid in all_bids}
                           for term in terms}
        dot_accum: dict = {(ta, tb): {bid: 0.0 for bid in all_bids}
                           for ta, tb in pairs}
        n_skipped = 0

        for name in param_names:
            bid = bid_for[name]

            # Determine canonical numel from the first non-None snapshot.
            canon_n = None
            for term in terms:
                t = block_grads_per_term[term].get(name)
                if t is not None:
                    canon_n = t.numel()
                    break
            if canon_n is None:
                # All terms have grad=None for this param; nothing to contribute.
                continue

            # Resolve snapshots: zeros for None-grad terms; check non-None sizes.
            gs: dict = {}
            skip = False
            for term in terms:
                t = block_grads_per_term[term].get(name)
                if t is None:
                    gs[term] = torch.zeros(canon_n, dtype=torch.float32)
                elif t.numel() != canon_n:
                    # Two non-None snapshots disagree on size: FSDP2 state mismatch.
                    # Skip this param to avoid producing incorrect statistics.
                    n_skipped += 1
                    skip = True
                    break
                else:
                    gs[term] = t.float()
            if skip:
                continue

            for term in terms:
                sq_accum[term][bid] += gs[term].pow(2).sum().item()
            for ta, tb in pairs:
                dot_accum[(ta, tb)][bid] += (gs[ta] * gs[tb]).sum().item()

        if n_skipped:
            logger.warning(
                f"_write_grad_stats step {self.step}: skipped {n_skipped} params "
                "due to FSDP2 grad/data numel mismatch across terms. "
                "Statistics for affected blocks may be underestimated."
            )

        # ── Build flat scalar tensor → single all-reduce ──────────────────────
        scalars: list = []
        idx: dict = {}

        for bid in all_bids:
            for term in terms:
                idx[(bid, f'sq_{term}')] = len(scalars)
                scalars.append(torch.tensor(
                    sq_accum[term][bid], dtype=torch.float32, device=self.device
                ))
            for ta, tb in pairs:
                idx[(bid, f'dot_{ta}_{tb}')] = len(scalars)
                scalars.append(torch.tensor(
                    dot_accum[(ta, tb)][bid], dtype=torch.float32, device=self.device
                ))

        stats = torch.stack(scalars)
        if dist.is_initialized():
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)

        if self.config.rank != 0:
            return  # only rank 0 writes

        block_stats: dict = {}
        for bid in all_bids:
            entry: dict = {}
            for term in terms:
                sq = stats[idx[(bid, f'sq_{term}')]].item()
                entry[f'mag_{term}'] = round(sq ** 0.5, 6)
            for (ta, tb) in pairs:
                dot  = stats[idx[(bid, f'dot_{ta}_{tb}')]].item()
                sq_a = stats[idx[(bid, f'sq_{ta}')]].item()
                sq_b = stats[idx[(bid, f'sq_{tb}')]].item()
                cos  = dot / ((sq_a ** 0.5) * (sq_b ** 0.5) + 1e-8)
                entry[f'cos_{ta}_{tb}'] = round(float(cos), 6)
            block_stats[str(bid)] = entry

        record = {
            'step':     self.step,
            't_latent': round(mean_t_latent, 2),
            't_action': round(mean_t_action, 2),
            'blocks':   block_stats,
        }
        with open(self.grad_stats_file, 'a') as f:
            f.write(json.dumps(record) + '\n')
        logger.info(
            f"Grad stats logged at step {self.step} "
            f"({len(all_bids)} blocks, terms={terms})"
        )

    def _logging_window_step(self, raw_batches: list) -> dict:
        """
        Full logging-window optimizer step:
          1. Prepare all K micro-batches (device transfer + noise sampling).
          2. For each active loss term, run an isolated K-micro-batch backward
             pass (with proper FSDP gradient sync on the last micro-batch) and
             snapshot the per-block gradient shards.
          3. All-reduce the shard statistics and write grad_stats.jsonl.
          4. Run the regular combined backward over the same K micro-batches
             (this is the update that actually changes the weights).
          5. Clip, optimizer.step(), zero_grad().

        The gradients captured in step 2 are the exact per-term contributions
        to the combined gradient computed in step 4 (same batches, same noise,
        same scaling), so the cosine similarities reflect what truly drove the
        weight update at this optimizer step.

        Returns a losses dict with keys:
            micro_latent_losses, micro_action_losses, micro_jepa_losses,
            total_norm, should_log=True
        """
        K = len(raw_batches)

        # ── 1. Prepare all micro-batches ──────────────────────────────────
        prepared = []   # list of (input_dict, converted_batch)
        for raw in raw_batches:
            batch     = self.convert_input_format(raw)
            input_dict = self._prepare_input_dict(batch)
            prepared.append((input_dict, batch))

        # Representative mean timesteps (averaged over micro-batches)
        mean_t_latent = float(
            sum(inp['latent_dict']['timesteps'].float().mean().item()
                for inp, _ in prepared) / K
        )
        mean_t_action = float(
            sum(inp['action_dict']['timesteps'].float().mean().item()
                for inp, _ in prepared) / K
        )

        jepa_enabled  = getattr(self.config, 'jepa_loss_enabled', False)
        jepa_layer    = self.config.jepa_loss_layer if jepa_enabled else -1
        active_terms  = ['latent', 'action']
        if self.jepa_head is not None:
            active_terms.append('jepa')

        # ── 2. Per-loss gradient passes ───────────────────────────────────
        block_grads_per_term: dict = {}

        for term in active_terms:
            self.optimizer.zero_grad(set_to_none=True)

            for i, (input_dict, batch) in enumerate(prepared):
                enable_sync   = (i == K - 1)
                capture_layer = jepa_layer if term == 'jepa' else -1
                self.transformer.set_requires_gradient_sync(enable_sync)

                latent_pred, action_pred, jepa_hidden = self.transformer(
                    input_dict, train_mode=True,
                    jepa_capture_layer=capture_layer,
                )

                if term == 'latent':
                    loss, _ = self.compute_loss(input_dict, (latent_pred, action_pred))
                elif term == 'action':
                    _, loss  = self.compute_loss(input_dict, (latent_pred, action_pred))
                else:  # jepa
                    if jepa_hidden is not None:
                        jepa_raw = self.compute_jepa_loss(
                            jepa_hidden, batch, input_dict
                        )
                        loss = (
                            self.config.jepa_loss_weight
                            * jepa_raw
                            / self.gradient_accumulation_steps
                        )
                    else:
                        # jepa_hidden unexpectedly None; skip this micro-batch
                        continue

                loss.backward()

                # Sync jepa_head grads manually (not FSDP-wrapped) on last micro-step
                if (enable_sync and term == 'jepa'
                        and self.jepa_head is not None
                        and self.config.world_size > 1):
                    for p in self.jepa_head.parameters():
                        if p.grad is not None:
                            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

            block_grads_per_term[term] = self._snapshot_block_grads()

        # ── 3. Write grad stats ───────────────────────────────────────────
        self._write_grad_stats(block_grads_per_term, mean_t_latent, mean_t_action)

        # ── 4. Regular combined backward (the actual update) ──────────────
        # Free per-term grad snapshots (CPU) and flush any unreferenced GPU
        # memory before the most memory-intensive pass.
        del block_grads_per_term
        gc.collect()
        torch.cuda.empty_cache()
        self.optimizer.zero_grad(set_to_none=True)
        micro_latent_losses: list = []
        micro_action_losses: list = []
        micro_jepa_losses:   list = []

        for i, (input_dict, batch) in enumerate(prepared):
            enable_sync = (i == K - 1)
            self.transformer.set_requires_gradient_sync(enable_sync)

            latent_pred, action_pred, jepa_hidden = self.transformer(
                input_dict, train_mode=True, jepa_capture_layer=jepa_layer
            )
            latent_loss, action_loss = self.compute_loss(
                input_dict, (latent_pred, action_pred)
            )

            jepa_loss_raw = torch.tensor(0.0, device=self.device)
            if self.jepa_head is not None and jepa_hidden is not None:
                jepa_loss_raw = self.compute_jepa_loss(
                    jepa_hidden, batch, input_dict
                )

            loss = (
                latent_loss
                + action_loss
                + self.config.jepa_loss_weight
                * jepa_loss_raw
                / self.gradient_accumulation_steps
            )
            loss.backward()

            if (enable_sync and self.jepa_head is not None
                    and self.config.world_size > 1):
                for p in self.jepa_head.parameters():
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

            micro_latent_losses.append(latent_loss.detach())
            micro_action_losses.append(action_loss.detach())
            micro_jepa_losses.append(
                (jepa_loss_raw / self.gradient_accumulation_steps).detach()
            )

        # ── 5. Optimizer step ─────────────────────────────────────────────
        # Clip transformer (DTensor) and jepa_head (plain Tensor) separately
        # to avoid the _foreach_mul_ mixed-type error.
        total_norm = torch.nn.utils.clip_grad_norm_(
            list(self.transformer.parameters()), 2.0
        )
        if self.jepa_head is not None:
            torch.nn.utils.clip_grad_norm_(
                list(self.jepa_head.parameters()), 2.0
            )
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

        return {
            'micro_latent_losses': micro_latent_losses,
            'micro_action_losses': micro_action_losses,
            'micro_jepa_losses':   micro_jepa_losses,
            'total_norm':          total_norm,
            'should_log':          True,
        }

    def save_checkpoint(self,):
        """Save model checkpoint in the same format as pretrained model."""
        try:
            state_dict = get_model_state_dict(
                self.transformer,
                options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            )
            state_dict_bf16 = {k: v.to(torch.bfloat16) for k, v in state_dict.items()}
            # optim_state = get_optimizer_state_dict(
            #         self.transformer, self.optimizer,
            #         options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            #     )

            # Only rank 0 saves the checkpoint
            if self.config.rank == 0:
                checkpoint_dir = self.save_dir / f"checkpoint_step_{self.step}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)

                # Save transformer in the same format as pretrained model
                transformer_dir = checkpoint_dir / "transformer"
                transformer_dir.mkdir(parents=True, exist_ok=True)

                logger.info(f"Saving transformer to {transformer_dir}")

                # Manually save in diffusers format (outside FSDP context to avoid deadlock)
                # Save model weights
                model_file = transformer_dir / "diffusion_pytorch_model.safetensors"
                save_file(state_dict_bf16, model_file)

                # Save config (copy from original transformer config and update _name_or_path)
                config_file = transformer_dir / "config.json"
                config_dict = dict(self.transformer.config)
                config_dict.pop('_name_or_path', None)
                with open(config_file, 'w') as f:
                    json.dump(config_dict, f, indent=2)

                # # Save optimizer state and training metadata in PyTorch format
                # training_state_path = checkpoint_dir / "training_state.pt"
                # logger.info(f"Saving training state to {training_state_path}")
                # torch.save({
                #     'step': self.step,
                #     'optimizer_state_dict': optim_state,
                #     'config': vars(self.config),
                # }, training_state_path)

                if self.jepa_head is not None:
                    jepa_head_path = checkpoint_dir / "jepa_head.pt"
                    torch.save(self.jepa_head.state_dict(), jepa_head_path)
                    logger.info(f"Saved jepa_head to {jepa_head_path}")

                logger.info(f"Checkpoint saved successfully at step {self.step}")

            # Synchronize all processes after saving
            if dist.is_initialized():
                dist.barrier()

        except Exception as e:
            if self.config.rank == 0:
                logger.error(f"Failed to save checkpoint: {e}")
                import traceback
                logger.error(traceback.format_exc())
            # Ensure all processes stay synchronized even on error
            if dist.is_initialized():
                dist.barrier()

    def _load_training_state(self, checkpoint_path):
        """Load training state (optimizer + step) after FSDP and optimizer creation."""
        checkpoint_dir = Path(checkpoint_path)
        training_state_path = checkpoint_dir / "training_state.pt"

        if not training_state_path.exists():
            if self.config.rank == 0:
                logger.warning(f"Training state not found: {training_state_path}, starting from step 0")
            return

        if self.config.rank == 0:
            logger.info(f"Loading training state from {training_state_path}")

        # All ranks load the training state directly
        training_state = torch.load(training_state_path, map_location='cpu', weights_only=False)

        # All ranks load optimizer state (required for FSDP)
        set_optimizer_state_dict(
            self.transformer, self.optimizer,
            optim_state_dict=training_state['optimizer_state_dict'],
            options=StateDictOptions(full_state_dict=True, strict=False)
        )
        self.step = training_state.get('step', 0)

        if self.config.rank == 0:
            logger.info(f"Training state loaded, resuming from step {self.step}")

        # Synchronize all ranks
        if dist.is_initialized():
            dist.barrier()

    def train(self):
        """Main training loop - train by steps instead of epochs."""
        logger.info(f"Starting training for {self.config.num_steps} steps...")
        self.transformer.train()

        progress_bar = tqdm(
            total=self.config.num_steps,
            desc="Training",
            disable=(self.config.rank != 0),
            leave=True,
            dynamic_ncols=True,
            initial=self.step
        )

        self.optimizer.zero_grad()
        accumulated_latent_losses = []
        accumulated_action_losses = []
        accumulated_jepa_losses   = []
        step_in_accumulation = 0

        while self.step < self.config.num_steps:
            # ── Detect start of a gradient-logging accumulation window ──────
            # Check at step_in_accumulation==0 (window boundary) only.
            # self.step==0 satisfies the modulo for any grad_log_freq, so the
            # very first optimizer step is always a logging step when enabled.
            is_logging_window = (
                step_in_accumulation == 0
                and self.grad_log_freq > 0
                and self.step % self.grad_log_freq == 0
            )

            if is_logging_window:
                # Collect all K micro-batches up front so we can replay them
                # for the per-loss isolated passes AND the combined update pass.
                raw_batches = [
                    self._get_next_batch()
                    for _ in range(self.gradient_accumulation_steps)
                ]
                losses = self._logging_window_step(raw_batches)
                # _logging_window_step already ran the optimizer step; pop the
                # per-micro-batch loss lists before entering the logging block.
                accumulated_latent_losses.extend(losses.pop('micro_latent_losses'))
                accumulated_action_losses.extend(losses.pop('micro_action_losses'))
                accumulated_jepa_losses.extend(losses.pop('micro_jepa_losses'))
            else:
                # Normal single micro-batch step
                batch  = self._get_next_batch()
                losses = self._train_step(batch, step_in_accumulation)
                accumulated_latent_losses.append(losses['latent_loss'])
                accumulated_action_losses.append(losses['action_loss'])
                accumulated_jepa_losses.append(losses['jepa_loss'])
                step_in_accumulation += 1

            # Log and checkpoint when optimizer steps
            if losses['should_log']:
                lr = self.lr_scheduler.get_last_lr()[0]

                # Average accumulated losses
                latent_loss_show = dist_mean(torch.stack(accumulated_latent_losses).sum()).detach().cpu().item()
                action_loss_show = dist_mean(torch.stack(accumulated_action_losses).sum()).detach().cpu().item()
                jepa_loss_show   = dist_mean(torch.stack(accumulated_jepa_losses).sum()).detach().cpu().item()
                max_latent_loss_show = dist_max(torch.stack(accumulated_latent_losses).sum()).detach().cpu().item()
                max_action_loss_show = dist_max(torch.stack(accumulated_action_losses).sum()).detach().cpu().item()

                # Clear accumulated losses
                accumulated_latent_losses = []
                accumulated_action_losses = []
                accumulated_jepa_losses   = []
                step_in_accumulation = 0

                torch.cuda.synchronize()
                if self.step % self.config.gc_interval == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

                if self.config.rank == 0:
                    total_norm = losses['total_norm']
                    # Progress bar total is in optimizer steps (num_steps), so
                    # update by 1 after each optimizer step.
                    progress_bar.update(1)
                    postfix = {
                        'latent_loss': f'{latent_loss_show:.4f}',
                        'action_loss': f'{action_loss_show:.4f}',
                        'step': self.step,
                        'grad_norm': f'{total_norm.item():.2f}',
                        'lr': f'{lr:.2e}',
                    }
                    if getattr(self.config, 'jepa_loss_enabled', False):
                        postfix['jepa_loss'] = f'{jepa_loss_show:.4f}'
                    progress_bar.set_postfix(postfix)
                    logger.info(
                        f"step={self.step} "
                        f"latent_loss={latent_loss_show:.4f} "
                        f"action_loss={action_loss_show:.4f} "
                        + (f"jepa_loss={jepa_loss_show:.4f} "
                           if getattr(self.config, 'jepa_loss_enabled', False) else "")
                        + f"total_loss={latent_loss_show + action_loss_show:.4f} "
                        f"grad_norm={total_norm.item():.4f} "
                        f"lr={lr:.2e}"
                    )
                    if self.step % self.config.log_freq == 0:
                        log_entry = {
                            'step': self.step,
                            'time': time.strftime('%Y-%m-%dT%H:%M:%S'),
                            'latent_loss': round(latent_loss_show, 6),
                            'max_latent_loss': round(max_latent_loss_show, 6),
                            'action_loss': round(action_loss_show, 6),
                            'max_action_loss': round(max_action_loss_show, 6),
                            'total_loss': round(latent_loss_show + action_loss_show, 6),
                            'grad_norm': round(total_norm.item(), 6),
                            'lr': lr,
                        }
                        if getattr(self.config, 'jepa_loss_enabled', False):
                            log_entry['jepa_loss'] = round(jepa_loss_show, 6)
                        with open(self.log_file, 'a') as f:
                            f.write(json.dumps(log_entry) + '\n')
                    if self.config.enable_wandb:
                        wandb_log = {
                            'loss_metrics/global_avg_video_loss': latent_loss_show,
                            'loss_metrics/global_avg_action_loss': action_loss_show,
                            'loss_metrics/global_max_video_loss': max_latent_loss_show,
                            'loss_metrics/global_max_action_loss': max_action_loss_show,
                            'grad_norm': total_norm.item(),
                            'lr': lr,
                        }
                        if getattr(self.config, 'jepa_loss_enabled', False):
                            wandb_log['loss_metrics/jepa_loss'] = jepa_loss_show
                        self.wandb.log(wandb_log, step=self.step)
                
                self.step += 1
                
                if self.step % self.config.save_interval == 0:
                    if self.config.rank == 0:
                        logger.info(f"Starting save model at step {self.step}")
                    self.save_checkpoint()

        progress_bar.close()
        logger.info("Training completed!")


def run(args):
    """Main entry point."""
    config = VA_CONFIGS[args.config_name]

    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    init_distributed(world_size, local_rank, rank)

    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size

    if args.save_root is not None:
        config.save_root = args.save_root

    if rank == 0:
        logger.info(f"Using config: {args.config_name}")
        logger.info(f"World size: {world_size}, Local rank: {local_rank}")

    trainer = Trainer(config)
    trainer.train()


def main():
    """Parse arguments and run training."""
    parser = argparse.ArgumentParser(description="Train WAN model for robotics")
    parser.add_argument(
        "--config-name",
        type=str,
        default='robotwin_train',
        help="Config name",
    )
    parser.add_argument(
        "--save-root",
        type=str,
        default=None,
        help="Root directory for saving checkpoints",
    )

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    init_logger()
    main()
