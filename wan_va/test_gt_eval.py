#!/usr/bin/env python3
"""
test_gt_eval.py
===============
Evaluate WAN-VA inference quality on a single dataset item using
ground-truth latents and actions as KV-cache conditioning.

Unlike test_inference.py (which uses kv_cache_window=1 and random chunk
sampling), this script:

  • Uses kv_window = frame_chunk_size (4) so that frame_st_id advances
    exactly as in real deployment, preserving transformer positional encoding.
  • Processes the whole dataset item sequentially from chunk 0 to the end.
  • Provides ground-truth actions as predicted_actions before each KV call,
    eliminating drift from the model's own predictions.
  • Produces per-dimension GT-vs-prediction plots matching the offline rollout
    plots the user already has.

Usage
-----
# Config registered in VA_CONFIGS:
python wan_va/test_gt_eval.py \\
    --config my_train_config \\
    --eval-model-path /path/to/checkpoint \\
    --dataset-idx 0 \\
    --output-dir ./gt_eval_out

# Unregistered config (dotted import path to the EasyDict object):
python wan_va/test_gt_eval.py \
    --config wan_va.configs.ma_configs.ma_preliminary_config \
    --eval-model-path /liujinxin/code/lhc/wy/wms/lingbot-va/checkpoints/ma_preliminary/bs16lr2.5e-5*1e-6_resume1000/checkpoints/checkpoint_step_11000 \
    --dataset-path /liujinxin/code/lhc/wy/wms/lingbot-va/datasets/maniparena/multi_datasets/preliminary/place_ring_on_rod \
    --dataset-idx 0 \
    --output-dir ./repeat_eval_out/test1_gt
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

# ── repo root on path ─────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan_va.wan_va_server import VA_Server
from wan_va.dataset.lerobot_latent_dataset import MultiLatentLeRobotDataset


# ═══════════════════════════════════════════════════════════════════════════════
# Config loader
# ═══════════════════════════════════════════════════════════════════════════════

def load_config(config_spec: str):
    """Load config by VA_CONFIGS key, or by dotted module path.

    Examples
    --------
    ``my_train_config``                        → VA_CONFIGS['my_train_config']
    ``wan_va.configs.ma_configs.ma_sim_config`` → imports that attribute
    """
    try:
        from wan_va.configs import VA_CONFIGS
        if config_spec in VA_CONFIGS:
            return VA_CONFIGS[config_spec]
    except Exception:
        pass

    # Dotted import: split on the last '.' to get module + attribute.
    try:
        mod_path, attr = config_spec.rsplit('.', 1)
        mod = importlib.import_module(mod_path)
        cfg = getattr(mod, attr)
        print(f"[config] loaded '{attr}' from '{mod_path}'")
        return cfg
    except (ValueError, ModuleNotFoundError, AttributeError) as exc:
        raise RuntimeError(
            f"Cannot load config '{config_spec}': {exc}\n"
            "Pass either a VA_CONFIGS key or a dotted import path, e.g.\n"
            "  wan_va.configs.ma_configs.ma_sim_config"
        ) from exc


# ═══════════════════════════════════════════════════════════════════════════════
# Server setup helper
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_server(server: VA_Server, text_emb: torch.Tensor):
    """Reset server, disable CFG, inject dataset text embedding.

    CFG must be disabled before _reset so that create_empty_cache allocates
    batch_size=1 (not 2).  Restoring guidance_scale > 1 after the fact
    would make video_noise_pred[1:] an empty tensor and corrupt denoising.
    """
    server.job_config.guidance_scale        = 1.0
    server.job_config.action_guidance_scale = 1.0
    server._reset(prompt='placeholder')

    emb = text_emb.to(server.device).to(server.dtype)
    if emb.dim() == 2:
        emb = emb.unsqueeze(0)          # (seq, dim) → (1, seq, dim)
    server.prompt_embeds          = emb
    server.negative_prompt_embeds = emb   # unused (use_cfg=False)


# ═══════════════════════════════════════════════════════════════════════════════
# Core inference
# ═══════════════════════════════════════════════════════════════════════════════

def _cuda_sync():
    """Flush the CUDA stream so wall-clock timers measure real GPU completion."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_episode_gt_eval(
    server: VA_Server,
    sample: dict,
    num_chunks: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Run sequential inference over one dataset item with GT KV conditioning.

    Frame-st-id progression mirrors real deployment:

      chunk 0  : frame_st_id = 0
                 _infer(frame_st_id=0, initial_state=gt_state_0)
                 (frame_st_id stays 0 — _infer does NOT advance it)

      chunk k>0: _compute_kv_cache(gt_latents[prev_chunk])
                    → frame_st_id += frame_chunk_size
                 _infer(frame_st_id = k * frame_chunk_size)

    ``server.predicted_actions`` is overwritten with GT before every KV call so
    the model receives perfect action context rather than accumulated drift.

    Parameters
    ----------
    server      : initialised VA_Server (model already loaded)
    sample      : dict from LatentLeRobotDataset.__getitem__
    num_chunks  : cap on number of chunks; None = run the whole item

    Returns
    -------
    pred_all : (T, C_used) float32 numpy — denormalised predicted actions
    gt_all   : (T, C_used) float32 numpy — denormalised GT actions
    timing   : list of per-chunk dicts with keys
               'chunk', 'kv_s' (None for chunk 0), 'infer_s', 'total_s'
    """
    F  = server.job_config.frame_chunk_size   # 4
    N  = server.job_config.action_per_frame              # 8

    latents      = sample['latents']          # (C_lat, F_max, H, W)
    gt_actions   = sample['actions']          # (C_model, F_max, N, 1)  normalised
    latents_mask = sample['latents_mask']     # (F_max,) bool
    text_emb     = sample['text_emb']

    # ── Trim to real (non-padded) frames ──────────────────────────────────────
    F_real   = int(latents_mask.sum().item())
    n_chunks = F_real // F
    if num_chunks is not None:
        n_chunks = min(n_chunks, num_chunks)
    if n_chunks == 0:
        raise ValueError(
            f"Only {F_real} real latent frames but frame_chunk_size={F}. "
            "Not enough data for a single chunk."
        )

    latents    = latents[:, :F_real]          # (C_lat, F_real, H, W)
    gt_actions = gt_actions[:, :F_real]       # (C_model, F_real, N, 1)

    # ── Reset server and inject text embedding ────────────────────────────────
    prepare_server(server, text_emb)

    original_encode_obs = server._encode_obs
    fake_obs = {'obs': [{}]}   # content unused; _encode_obs is monkey-patched

    all_pred:   list[np.ndarray] = []
    all_gt:     list[np.ndarray] = []
    timing:     list[dict]       = []

    try:
        for k in range(n_chunks):
            f0 = k * F        # first latent frame of this chunk
            f1 = f0 + F       # one past the last
            t_chunk_start = time.perf_counter()
            kv_s = None

            if k == 0:
                # ── Chunk 0: obs-conditioned infer, no KV cache ───────────────
                #
                # Patch _encode_obs to return the first latent frame so
                # _infer can store it as self.init_latent.  Only the first
                # time-step slice (index 0) is used as latent_cond.
                obs_lat = latents[:, f0:f0 + 1]   # (C, 1, H, W)
                server._encode_obs = lambda _, _f=obs_lat: (
                    _f.unsqueeze(0).to(server.device).to(server.dtype)
                )

                # initial_state: first GT action step, denormalised, shape (C_used,)
                # gt_actions[:, f0:f0+1, 0:1, :] → (C_model, 1, 1, 1)
                # .unsqueeze(0) → (1, C_model, 1, 1, 1)
                # postprocess_action → (1, C_used)
                gt_st_norm     = gt_actions[:, f0:f0 + 1, 0:1, :].unsqueeze(0)
                initial_state  = server.postprocess_action(gt_st_norm)[0].copy()

                t_infer_start = time.perf_counter()
                pred, _ = server._infer(
                    fake_obs, frame_st_id=0, initial_state=initial_state
                )
                _cuda_sync()
                infer_s = time.perf_counter() - t_infer_start
                # frame_st_id stays 0 — _infer does NOT advance it.

            else:
                # ── Chunk k>0: KV cache with previous chunk's GT ──────────────
                #
                # KV covers the PREVIOUS chunk's frame_chunk_size latent frames.
                # Using GT actions as predicted_actions gives the model perfect
                # context and isolates pure inference quality.
                f_kv0 = f0 - F
                f_kv1 = f0

                kv_lat = latents[:, f_kv0:f_kv1]   # (C, F, H, W)
                server._encode_obs = lambda _, _f=kv_lat: (
                    _f.unsqueeze(0).to(server.device).to(server.dtype)
                )
                server.predicted_actions = (
                    gt_actions[:, f_kv0:f_kv1]
                    .unsqueeze(0)
                    .to(server.device)
                    .to(server.dtype)
                )
                t_kv_start = time.perf_counter()
                server._compute_kv_cache(fake_obs)
                _cuda_sync()
                kv_s = time.perf_counter() - t_kv_start
                # frame_st_id is now k * F (advanced inside _compute_kv_cache)

                # _infer at frame_st_id > 0 does NOT call _encode_obs, so no
                # further patch is needed here.
                t_infer_start = time.perf_counter()
                pred, _ = server._infer(
                    fake_obs, frame_st_id=server.frame_st_id
                    # initial_state=None is correct for frame_st_id > 0
                )
                _cuda_sync()
                infer_s = time.perf_counter() - t_infer_start

            total_s = time.perf_counter() - t_chunk_start

            # ── Ground truth for this chunk ───────────────────────────────────
            # postprocess_action expects (1, C_model, F, N, 1) → (F*N, C_used)
            gt_chunk = gt_actions[:, f0:f1]
            gt       = server.postprocess_action(gt_chunk.unsqueeze(0)).copy()

            all_pred.append(pred)   # (F*N, C_used)
            all_gt.append(gt)       # (F*N, C_used)
            timing.append({'chunk': k, 'kv_s': kv_s,
                           'infer_s': infer_s, 'total_s': total_s})

            kv_str = f"kv={kv_s:.2f}s  " if kv_s is not None else "kv=  —     "
            print(
                f"  chunk {k:3d}  frame_st_id={server.frame_st_id:4d}  "
                f"{kv_str}infer={infer_s:.2f}s  total={total_s:.2f}s"
            )

    finally:
        server._encode_obs = original_encode_obs

    pred_all = np.concatenate(all_pred, axis=0)   # (T, C_used)
    gt_all   = np.concatenate(all_gt,   axis=0)   # (T, C_used)
    return pred_all, gt_all, timing


# ═══════════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════════

_DIM_LABELS = [
    "L_joint_0", "L_joint_1", "L_joint_2", "L_joint_3", "L_joint_4", "L_joint_5",
    "R_joint_0", "R_joint_1", "R_joint_2", "R_joint_3", "R_joint_4", "R_joint_5",
    "L_gripper",
    "R_gripper",
]


def plot_trajectories(
    pred: np.ndarray,
    gt:   np.ndarray,
    save_path: Path,
    frame_chunk_size: int = 4,
    action_per_frame: int = 8,
    title: str = "GT vs Predicted Actions",
) -> None:
    """Plot each output dimension as GT (blue) vs prediction (orange).

    A vertical dashed line is drawn at every KV-cache update boundary
    (every frame_chunk_size * action_per_frame steps) to show where the model
    receives fresh context.

    Parameters
    ----------
    pred / gt   : (T, C_used) arrays in postprocess_action order
                  [L_joint×6, R_joint×6, L_gripper, R_gripper]
    save_path   : file path for the saved figure (PNG)
    """
    T, C = gt.shape
    chunk_size = frame_chunk_size * action_per_frame   # steps per KV update

    fig, axes = plt.subplots(C, 1, figsize=(16, C * 2.2), sharex=True)
    fig.suptitle(title, fontsize=13, y=1.002)

    t = np.arange(T)
    for dim in range(C):
        ax = axes[dim]
        ax.plot(t, gt  [:, dim], color='steelblue',  linewidth=1.0, label='Ground Truth')
        ax.plot(t, pred[:, dim], color='darkorange',  linewidth=0.9, label='Prediction', alpha=0.85)

        # Mark KV cache update boundaries
        for boundary in range(chunk_size, T, chunk_size):
            ax.axvline(boundary, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)

        label = _DIM_LABELS[dim] if dim < len(_DIM_LABELS) else f"Dim {dim}"
        ax.set_ylabel(label, fontsize=8, rotation=0, labelpad=60, va='center')
        ax.tick_params(axis='y', labelsize=7)
        ax.tick_params(axis='x', labelsize=7)
        if dim == 0:
            ax.legend(fontsize=8, loc='upper right')

    axes[-1].set_xlabel("Time Step", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"[plot] saved → {save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--config', required=True,
        help=(
            'VA_CONFIGS key OR dotted import path to the config EasyDict, e.g.\n'
            '  my_train_config\n'
            '  wan_va.configs.ma_configs.ma_sim_config'
        ),
    )
    parser.add_argument(
        '--eval-model-path', default=None,
        help='Override wan22_pretrained_model_name_or_path / finetuned path in config',
    )
    parser.add_argument(
        '--dataset-path', default=None,
        help='Override the dataset_path in config (path to the LeRobot latent dataset root)',
    )
    parser.add_argument(
        '--dataset-idx', type=int, default=0,
        help='Index into MultiLatentLeRobotDataset to evaluate (default: 0)',
    )
    parser.add_argument(
        '--num-chunks', type=int, default=None,
        help='Max inference chunks to run (default: whole item)',
    )
    parser.add_argument(
        '--output-dir', default='./gt_eval_out',
        help='Directory to write arrays, stats and plots',
    )
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Config ────────────────────────────────────────────────────────────────
    config = load_config(args.config)
    if args.eval_model_path:
        for attr in ('wan22_finetuned_model_name_or_path'):
            if hasattr(config, attr):
                setattr(config, attr, args.eval_model_path)
    config.local_rank  = 0
    config.rank        = 0
    config.world_size  = 1
    config.save_root   = str(output_dir / 'server_debug')
    # cfg_prob=0: never replace text_emb with empty_emb during eval
    config.cfg_prob    = 0.0

    # ── Dataset ───────────────────────────────────────────────────────────────
    if args.dataset_path:
        config.dataset_path = args.dataset_path
        print(f"[dataset] path overridden → {args.dataset_path}")
    print("Loading dataset …")
    dataset = MultiLatentLeRobotDataset(config)
    n_total = len(dataset)
    print(f"Dataset size: {n_total} items")
    if args.dataset_idx >= n_total:
        raise IndexError(
            f"--dataset-idx {args.dataset_idx} out of range (dataset has {n_total} items)"
        )
    sample = dataset[args.dataset_idx]
    latents_mask = sample['latents_mask']
    F_real = int(latents_mask.sum().item())
    F  = config.frame_chunk_size
    N  = config.action_per_frame
    print(f"Item {args.dataset_idx}: {F_real} real latent frames → "
          f"{F_real // F} complete chunks × {F}×{N} = {F_real // F * F * N} steps")

    # ── Model ─────────────────────────────────────────────────────────────────
    print("Loading model …")
    server = VA_Server(config)

    # ── Inference ─────────────────────────────────────────────────────────────
    print("\nRunning GT-conditioned inference …")
    pred_all, gt_all, timing = run_episode_gt_eval(
        server, sample, num_chunks=args.num_chunks
    )

    # ── Save arrays ───────────────────────────────────────────────────────────
    np.save(output_dir / 'pred_actions.npy', pred_all)
    np.save(output_dir / 'gt_actions.npy',   gt_all)
    print(f"\nArrays saved to {output_dir}/  (shapes: pred={pred_all.shape}, gt={gt_all.shape})")

    # ── Per-chunk L1 stats ────────────────────────────────────────────────────
    chunk_steps = F * N
    n_chunks = pred_all.shape[0] // chunk_steps
    chunk_l1s = []
    for k in range(n_chunks):
        p = pred_all[k * chunk_steps:(k + 1) * chunk_steps]
        g = gt_all  [k * chunk_steps:(k + 1) * chunk_steps]
        chunk_l1s.append(float(np.abs(p - g).mean()))

    per_dim_l1 = np.abs(pred_all - gt_all).mean(axis=0)
    stats = {
        'dataset_idx':  args.dataset_idx,
        'n_chunks':     n_chunks,
        'total_steps':  int(pred_all.shape[0]),
        'mean_L1':      float(np.abs(pred_all - gt_all).mean()),
        'per_chunk_L1': chunk_l1s,
        'per_dim_L1':   {
            (_DIM_LABELS[d] if d < len(_DIM_LABELS) else f'dim_{d}'): float(per_dim_l1[d])
            for d in range(len(per_dim_l1))
        },
    }
    with open(output_dir / 'stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\nmean L1 = {stats['mean_L1']:.4f}")
    print("per-dim L1:")
    for name, val in stats['per_dim_L1'].items():
        print(f"  {name:12s}: {val:.4f}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    title = (
        f"GT KV eval — item {args.dataset_idx}  "
        f"({n_chunks} chunks, mean L1={stats['mean_L1']:.3f})"
    )
    plot_trajectories(
        pred_all, gt_all,
        save_path=output_dir / 'trajectories.png',
        frame_chunk_size=F,
        action_per_frame=N,
        title=title,
    )

    # ── Timing summary ────────────────────────────────────────────────────────
    infer_times = [r['infer_s'] for r in timing]
    kv_times    = [r['kv_s']    for r in timing if r['kv_s'] is not None]
    total_times = [r['total_s'] for r in timing]

    def _fmt(vals):
        return f"{np.mean(vals):.2f}s  (min {np.min(vals):.2f}s, max {np.max(vals):.2f}s)"

    print("\n" + "━" * 54)
    print("  Timing summary")
    print("━" * 54)
    print(f"  chunks run        : {len(timing)}")
    if kv_times:
        print(f"  kv_cache / chunk  : {_fmt(kv_times)}  [chunks 1+]")
    print(f"  infer    / chunk  : {_fmt(infer_times)}")
    print(f"  total    / chunk  : {_fmt(total_times)}")
    print(f"  wall time total   : {sum(total_times):.1f}s")
    print("━" * 54)

    print(f"\nDone. Results in {output_dir}/")


if __name__ == '__main__':
    main()
