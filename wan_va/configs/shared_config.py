# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import torch
from easydict import EasyDict

va_shared_cfg = EasyDict()

va_shared_cfg.host = '0.0.0.0'
va_shared_cfg.port = 29536

va_shared_cfg.param_dtype = torch.bfloat16
va_shared_cfg.save_root = './checkpoints'

va_shared_cfg.patch_size = (1, 2, 2)

va_shared_cfg.enable_offload = False
va_shared_cfg.log_freq = 10  # log metrics to log.jsonl every N optimizer steps

# MYEDIT: added experiment configs

# Maximum number of latent frames per chunk after VAE encoding.
# All dataset samples are padded to this length along the F dimension so that
# the default PyTorch collate can stack them into batches of size > 1.
# Set to (chunk_size - 1) // 4 + 1 where chunk_size is the extraction chunk_size.
va_shared_cfg.max_latent_frames = 126

# ── JEPA auxiliary cosine loss ────────────────────────────────────────────────
va_shared_cfg.jepa_loss_enabled  = False

# DiT layer (0-indexed, out of 30) whose hidden states are projected to JEPA
# feature space.  Middle-to-deep layers carry rich semantic content without
# being over-specialised for denoising.
va_shared_cfg.jepa_loss_layer    = 20

# Projection head architecture mapped from DiT inner_dim → JEPA embed_dim.
#   "linear" : single nn.Linear(3072, 1664)
#   "mlp"    : Linear(3072, 3072) → GELU → Linear(3072, 1664)
va_shared_cfg.jepa_head_type     = "linear"

# λ in:  loss = loss_diffusion + λ * loss_jepa
va_shared_cfg.jepa_loss_weight   = 0.1

# Apply JEPA loss only when the normalised diffusion timestep σ < this value.
# 1.0 = apply at all timesteps; lower values focus on less-noisy steps where
# hidden states carry stronger semantic content.
va_shared_cfg.jepa_loss_t_max    = 1.0

# ── Gradient alignment logging ────────────────────────────────────────────────
# Every grad_log_freq optimizer steps, run per-loss isolated backward passes on
# the same accumulation window that drives the update, compute per-transformer-
# block cosine similarities and gradient magnitudes, and append one record to
# checkpoints/grad_stats.jsonl.  Set to 0 to disable (default).
# Step 0 (before the first weight update) is always logged when enabled.
va_shared_cfg.grad_log_freq = 100