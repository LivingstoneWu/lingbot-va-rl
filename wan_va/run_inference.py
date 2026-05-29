#!/usr/bin/env python3
"""
test_inference.py
=================
Sanity-check the WAN-VA action prediction pipeline against dataset samples,
without running the full WebSocket server or needing a physical robot.

For each sampled item the script:
  1. Loads precomputed latents + GT actions from the latent dataset.
  2. Injects the dataset latent directly into VA_Server._infer() (bypasses
     the VAE encoder and any camera pipeline issues).
  3. Saves per-sample outputs to <output_dir>/sample_NNN_idxM/:
       pred_actions_raw.pt     – raw action tensor from _infer (1, C, F, N, 1)
       gt_actions_raw.pt       – ground truth action tensor   (C, F, N, 1)
       pred_actions_denorm.npy – postprocessed actions        (T, C_used)
       pred_latents.pt         – denoised video latents       (1, C, F, H, W)
       stats.json              – L1 / RMSE error in normalised action space
  4. Writes a summary.json to <output_dir>/.

NOTE: Pass a *training* config name (e.g. rc_single_set_the_plates_config_train)
because those configs carry dataset_path, empty_emb_path, and norm_stat.

Usage:
  python wan_va/test_inference.py \\
      --config rc_single_set_the_plates_config_train \\
      --eval_model_path /path/to/finetuned \\
      --num_samples 5 \\
      --output_dir ./test_inference_out
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers.utils import export_to_video
from diffusers.video_processor import VideoProcessor

# ── ensure project root is on the path ───────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan_va.configs import VA_CONFIGS
from wan_va.wan_va_server import VA_Server
from wan_va.dataset.lerobot_latent_dataset import MultiLatentLeRobotDataset


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_server(server: VA_Server, text_emb: torch.Tensor):
    """
    Reset the server caches (with CFG disabled so we only need one embedding),
    then override prompt_embeds with the precomputed dataset embedding.

    CFG is disabled for the entire test session: guidance_scale and
    action_guidance_scale are set to 1 and NOT restored.  This is intentional —
    _infer checks job_config.guidance_scale directly (not self.use_cfg) when
    applying the CFG combination step, so restoring a value > 1 would cause
    video_noise_pred[1:] to be an empty tensor (batch=1, no CFG repeat) and
    corrupt latents to shape (0,...) from the first denoising step onward.
    """
    server.job_config.guidance_scale        = 1.0
    server.job_config.action_guidance_scale = 1.0

    server._reset(prompt='test')  # initialises KV-cache buffers, action_mask, etc.

    # Replace the dummy text embedding with the dataset's precomputed one.
    # Shape expected by the transformer: (1, seq_len, hidden_dim).
    emb = text_emb.to(server.device).to(server.dtype)
    if emb.dim() == 2:          # (seq, dim) → (1, seq, dim)
        emb = emb.unsqueeze(0)
    server.prompt_embeds          = emb
    server.negative_prompt_embeds = emb   # unused when use_cfg=False


@torch.no_grad()
def decode_pred_latents_to_video_np(server: VA_Server, latents: torch.Tensor) -> np.ndarray:
    """
    Decode normalized WAN latents [B,C,F,H,W] to numpy video frames.
    """
    if latents.ndim != 5:
        raise ValueError(f"Expected latents [B,C,F,H,W], got {tuple(latents.shape)}")
    if latents.shape[0] != 1:
        raise ValueError(f"Expected batch size B=1, got {latents.shape[0]}")

    vae = server.vae
    vae_device = next(vae.parameters()).device
    vae_dtype = next(vae.parameters()).dtype

    latents = latents.to(vae_device).to(vae_dtype)
    latents_mean = (
        torch.tensor(vae.config.latents_mean)
        .view(1, vae.config.z_dim, 1, 1, 1)
        .to(vae_device, vae_dtype)
    )
    latents_std = (
        1.0
        / torch.tensor(vae.config.latents_std)
        .view(1, vae.config.z_dim, 1, 1, 1)
        .to(vae_device, vae_dtype)
    )
    latents = latents / latents_std + latents_mean

    video = vae.decode(latents, return_dict=False)[0]
    video_np = VideoProcessor(vae_scale_factor=1).postprocess_video(video, output_type="np")[0]
    return video_np


def run_sample(server: VA_Server, sample: dict, sample_dir: Path,
               kv_cache_window: int = 1, num_chunks: int = 3,
               video_fps: int = 10) -> dict | None:
    """
    Run multi-chunk inference following the actual deployment flow:

      reset()
      chunk 0 : _infer(frame_st_id=0)          ← no KV cache, zero action_cond
      chunk k>0: _compute_kv_cache(kv_window)   ← updates server.frame_st_id
                 _infer(frame_st_id=server.frame_st_id)

    _encode_obs is monkey-patched so the pre-computed dataset latents are used
    instead of running the VAE encoder.

    Args:
        kv_cache_window: number of latent frames fed to _compute_kv_cache each
                         time (mirrors the chunk_server KV-cache window size).
        num_chunks:      how many predict-then-update-KV iterations to run.
    """
    latent     = sample['latents']       # (C, F_total, H, W)
    gt_actions = sample['actions']       # (C_model, F_total, N, 1)  – normalised
    gt_mask    = sample['actions_mask']  # (C_model, F_total, N, 1)  – bool
    text_emb   = sample['text_emb']      # (seq, dim)

    frame_chunk_size = server.job_config.frame_chunk_size
    F_total = latent.shape[1]

    # Need at least num_chunks * frame_chunk_size latent frames from f_start.
    max_start = F_total - num_chunks * frame_chunk_size
    if max_start < 0:
        print(f"  Skipping: only {F_total} latent frames, need {num_chunks * frame_chunk_size}")
        return None
    f_start = random.randint(0, max_start)

    prepare_server(server, text_emb)

    original_encode_obs = server._encode_obs
    fake_obs = {'obs': [{}]}   # content irrelevant; _encode_obs is patched

    all_pred_denorm = []
    all_gt_denorm   = []
    all_pred_raw    = []
    all_gt_mask     = []
    all_latents     = []

    try:
        for chunk_idx in range(num_chunks):
            # f_obs: latent frame index of the current observation.
            # GT actions at gt_actions[:, f_obs:f_obs+frame_chunk_size] are the
            # actions taken *during* those latent frames, which is exactly what
            # the model predicts given the observation at f_obs.
            # The VAE "1, 4, 4" pattern (anchor + groups-of-4) is already baked
            # into the dataset's action_per_frame alignment, so latent-frame
            # indexing here is correct without any extra offset.
            f_obs = f_start + chunk_idx * frame_chunk_size

            if chunk_idx > 0:
                # ── KV cache: kv_cache_window latent frames ending at f_obs ──
                f_kv = max(0, f_obs - kv_cache_window)
                kv_frames = latent[:, f_kv:f_obs]   # (C, kv_cache_window, H, W)
                server._encode_obs = lambda _, _f=kv_frames: (
                    _f.unsqueeze(0).to(server.device).to(server.dtype)
                )
                # Override predicted_actions with GT so KV cache conditioning
                # uses ground-truth actions instead of accumulated drift.
                # Shape: (1, C_model, kv_cache_window, N, 1)
                server.predicted_actions = (
                    gt_actions[:, f_kv:f_obs]
                    .unsqueeze(0).to(server.device).to(server.dtype)
                )
                server._compute_kv_cache(fake_obs)
                # server.frame_st_id is now updated by _compute_kv_cache

            # ── infer: single obs frame at f_obs ─────────────────────────────
            obs_frame = latent[:, f_obs:f_obs + 1]   # (C, 1, H, W)
            server._encode_obs = lambda _, _f=obs_frame: (
                _f.unsqueeze(0).to(server.device).to(server.dtype)
            )

            # For the first chunk (frame_st_id==0), pass the GT state at f_obs
            # as initial_state for action_cond inpainting.
            # gt_actions[:, f_obs:f_obs+1, 0:1, :] = first action step of frame f_obs
            # → (C_model, 1, 1, 1) → postprocess_action → (1, C_used) denormalised
            if server.frame_st_id == 0:
                gt_state_norm = gt_actions[:, f_obs:f_obs + 1, 0:1, :].unsqueeze(0)
                gt_state_denorm = server.postprocess_action(gt_state_norm)  # (1, C_used)
                initial_state = gt_state_denorm[0]   # (C_used,)
            else:
                initial_state = None

            pred_denorm, pred_latents = server._infer(
                fake_obs,
                frame_st_id=server.frame_st_id,
                initial_state=initial_state
            )

            # ── GT: actions during frames f_obs..f_obs+frame_chunk_size-1 ────
            gt_chunk   = gt_actions[:, f_obs:f_obs + frame_chunk_size]   # (C_model, frame_chunk_size, N, 1)
            mask_chunk = gt_mask[:,    f_obs:f_obs + frame_chunk_size]
            gt_denorm  = server.postprocess_action(gt_chunk.unsqueeze(0)) # (T_chunk, C_used)

            all_pred_denorm.append(pred_denorm)
            all_gt_denorm.append(gt_denorm)
            all_pred_raw.append(server.predicted_actions.cpu().float())   # (1, C_model, frame_chunk_size, N, 1)
            all_gt_mask.append(mask_chunk)
            all_latents.append(pred_latents)

    finally:
        server._encode_obs = original_encode_obs

    # ── save outputs ─────────────────────────────────────────────────────────
    sample_dir.mkdir(parents=True, exist_ok=True)

    pred_all = np.concatenate(all_pred_denorm, axis=0)   # (T_total, C_used)
    gt_all   = np.concatenate(all_gt_denorm,   axis=0)   # (T_total, C_used)
    np.save(sample_dir / 'pred_actions_denorm.npy', pred_all)
    np.save(sample_dir / 'gt_actions_denorm.npy',   gt_all)

    # Save per-chunk latents stacked along the frame dimension
    pred_latents_cat = torch.cat(all_latents, dim=2)
    torch.save(pred_latents_cat, sample_dir / 'pred_latents.pt')
    try:
        pred_video_np = decode_pred_latents_to_video_np(server, pred_latents_cat)

        # GT latents for the same window: (C, F_used, H, W) → (1, C, F_used, H, W)
        F_used = num_chunks * frame_chunk_size
        gt_latents_cat = (
            latent[:, f_start:f_start + F_used]
            .unsqueeze(0)
            .to(server.device, server.dtype)
        )
        gt_video_np = decode_pred_latents_to_video_np(server, gt_latents_cat)

        # Stack GT (top) and pred (bottom) along the height axis for each frame.
        # Both lists are length F_used, each element is (H, W, C) float32 in [0, 1].
        combined = [
            np.concatenate([g, p], axis=0)
            for g, p in zip(gt_video_np, pred_video_np)
        ]
        export_to_video(combined, str(sample_dir / 'pred_video.mp4'), fps=video_fps)
    except Exception as e:
        print(f"  WARNING: failed to decode/save pred_video.mp4: {e}")

    # ── compute error in normalised action space (per chunk, then aggregate) ──
    chunk_stats = []
    for chunk_idx, (pred_raw, gt_chunk, mask_chunk) in enumerate(
            zip(all_pred_raw, [gt_actions[:, f_start + i * frame_chunk_size:
                                           f_start + (i + 1) * frame_chunk_size]
                                for i in range(num_chunks)], all_gt_mask)):
        pred = pred_raw.squeeze(0)          # (C_model, frame_chunk_size, N, 1)
        gt   = gt_chunk.float().cpu()
        mask = mask_chunk.bool().cpu()
        if not mask.any():
            chunk_stats.append({'l1': float('nan'), 'rmse': float('nan')})
            continue
        diff = (pred[mask] - gt[mask]).abs()
        chunk_stats.append({
            'l1':   diff.mean().item(),
            'rmse': ((pred[mask] - gt[mask]) ** 2).mean().sqrt().item(),
        })

    valid_chunk_stats = [s for s in chunk_stats if not np.isnan(s['l1'])]
    mean_l1   = float(np.mean([s['l1']   for s in valid_chunk_stats])) if valid_chunk_stats else float('nan')
    mean_rmse = float(np.mean([s['rmse'] for s in valid_chunk_stats])) if valid_chunk_stats else float('nan')

    stats = {
        'f_start':       f_start,
        'num_chunks':    num_chunks,
        'kv_cache_window': kv_cache_window,
        'mean_l1':       mean_l1,
        'mean_rmse':     mean_rmse,
        'per_chunk':     chunk_stats,
        'latent_shape':  list(latent.shape),
    }
    with open(sample_dir / 'stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"  f_start={f_start}  chunks={num_chunks}  mean_L1={mean_l1:.4f}  mean_RMSE={mean_rmse:.4f}")
    per_chunk_l1 = [f"{s['l1']:.3f}" if not np.isnan(s['l1']) else 'nan' for s in chunk_stats]
    print(f"  per_chunk_l1: {per_chunk_l1}")
    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', required=True,
                        help='Key in VA_CONFIGS — use a *_train config that has '
                             'dataset_path and norm_stat (e.g. rc_single_set_the_plates_config_train)')
    parser.add_argument('--eval_model_path', default=None,
                        help='Override wan22_finetuned_model_name_or_path in the config')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of dataset samples to evaluate')
    parser.add_argument('--num_chunks', type=int, default=3,
                        help='Inference chunks per sample (reset → infer → kv_cache → infer …)')
    parser.add_argument('--kv_cache_window', type=int, default=1,
                        help='Latent frames fed to _compute_kv_cache each update')
    parser.add_argument('--output_dir', default='./test_inference_out',
                        help='Directory to write per-sample results and summary')
    parser.add_argument('--video_fps', type=int, default=10,
                        help='FPS used when saving decoded pred_video.mp4')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── config ────────────────────────────────────────────────────────────────
    config = VA_CONFIGS[args.config]
    if args.eval_model_path:
        config.wan22_finetuned_model_name_or_path = args.eval_model_path
    config.local_rank = 0
    config.rank       = 0
    config.world_size = 1
    # Route async debug saves inside the output dir so they don't pollute cwd.
    config.save_root  = os.path.join(args.output_dir, 'server_debug')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── dataset ───────────────────────────────────────────────────────────────
    print("Loading dataset …")
    dataset = MultiLatentLeRobotDataset(config)
    print(f"Dataset size: {len(dataset)} chunks")

    indices = random.sample(range(len(dataset)), min(args.num_samples, len(dataset)))

    # ── model ─────────────────────────────────────────────────────────────────
    print("Loading model …")
    server = VA_Server(config)

    # ── inference loop ────────────────────────────────────────────────────────
    all_stats = []
    for i, idx in enumerate(indices):
        print(f"\n[{i+1}/{len(indices)}] dataset_idx={idx}")
        sample     = dataset[idx]
        sample_dir = output_dir / f"sample_{i:03d}_idx{idx}"
        stats = run_sample(server, sample, sample_dir,
                           kv_cache_window=args.kv_cache_window,
                           num_chunks=args.num_chunks,
                           video_fps=args.video_fps)
        if stats is None:
            continue
        all_stats.append({'dataset_idx': idx, **stats})

    # ── summary ───────────────────────────────────────────────────────────────
    summary = {
        'config':            args.config,
        'num_samples':       len(all_stats),
        'num_chunks':        args.num_chunks,
        'kv_cache_window':   args.kv_cache_window,
        'mean_l1':           float(np.mean([s['mean_l1']   for s in all_stats])),
        'mean_rmse':         float(np.mean([s['mean_rmse'] for s in all_stats])),
        'samples':           all_stats,
    }
    summary_path = output_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"mean L1   = {summary['mean_l1']:.4f}  (over {summary['num_chunks']} chunks/sample)")
    print(f"mean RMSE = {summary['mean_rmse']:.4f}")
    print(f"Results   → {output_dir}")


if __name__ == '__main__':
    main()
