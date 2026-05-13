#!/usr/bin/env python3
"""
test_repeat_eval.py
===================
Evaluate WAN-VA inference quality under *deployment-style* KV conditioning:
for each KV update, one representative video frame per latent slot is read
from the raw MP4 and repeated ×4 through the streaming VAE — exactly what
the deployed chunk server (build_kv_cache_obs) does.

Comparison with test_gt_eval.py
--------------------------------
  test_gt_eval.py   : injects pre-encoded GT latents directly (upper bound)
  test_repeat_eval.py: encodes "last frame ×4" from raw video (deployment)

Both scripts keep GT actions as ``predicted_actions`` for every KV call so
that action-conditioning drift is not a confound; only the visual encoding
differs.

No monkey-patching of _encode_obs is required — obs dicts with real images
are built here and passed directly to _compute_kv_cache / _infer.

Frame mapping
-------------
The latent file stores a ``frame_ids`` tensor of the original (downsampled)
video frame indices that fed the streaming VAE.  For latent frame k:

    representative video frame = frame_ids[4 * k]   (last of the 4 inputs)

This aligns with the deployed robot sending its most-recent image at
inference time (end of the executed chunk window).

Usage
-----
python wan_va/test_repeat_eval.py \
    --config wan_va.configs.ma_configs.ma_preliminary_config \
    --eval-model-path /liujinxin/code/lhc/wy/wms/lingbot-va/checkpoints/ma_preliminary/bs16lr2.5e-5*1e-6_resume1000/checkpoints/checkpoint_step_11000 \
    --dataset-path /liujinxin/code/lhc/wy/wms/lingbot-va/datasets/maniparena/multi_datasets/preliminary/place_ring_on_rod \
    --dataset-idx 0 \
    --output-dir ./repeat_eval_out/test1
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

import cv2
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
# Config loader  (identical to test_gt_eval.py)
# ═══════════════════════════════════════════════════════════════════════════════

def load_config(config_spec: str):
    try:
        from wan_va.configs import VA_CONFIGS
        if config_spec in VA_CONFIGS:
            return VA_CONFIGS[config_spec]
    except Exception:
        pass
    try:
        mod_path, attr = config_spec.rsplit('.', 1)
        mod = importlib.import_module(mod_path)
        cfg = getattr(mod, attr)
        print(f"[config] loaded '{attr}' from '{mod_path}'")
        return cfg
    except (ValueError, ModuleNotFoundError, AttributeError) as exc:
        raise RuntimeError(
            f"Cannot load config '{config_spec}': {exc}\n"
            "Pass either a VA_CONFIGS key or a dotted import path."
        ) from exc


# ═══════════════════════════════════════════════════════════════════════════════
# Server setup helper  (identical to test_gt_eval.py)
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_server(server: VA_Server, text_emb: torch.Tensor):
    """Reset server, disable CFG, inject dataset text embedding."""
    server.job_config.guidance_scale        = 1.0
    server.job_config.action_guidance_scale = 1.0
    server._reset(prompt='placeholder')

    emb = text_emb.to(server.device).to(server.dtype)
    if emb.dim() == 2:
        emb = emb.unsqueeze(0)
    server.prompt_embeds          = emb
    server.negative_prompt_embeds = emb


# ═══════════════════════════════════════════════════════════════════════════════
# Video frame loading
# ═══════════════════════════════════════════════════════════════════════════════

def get_item_video_meta(dataset: MultiLatentLeRobotDataset, idx: int) -> dict:
    """Return video-loading metadata for dataset item idx.

    Extracted directly from dataset internals so that __getitem__ does not
    need to be modified (and collation in training is unaffected).

    Returns
    -------
    dict with keys:
        repo_root      : Path  — dataset root (videos/ lives here)
        episode_index  : int
        episode_chunk  : int
        frame_ids      : 1-D LongTensor of episode-relative video frame indices
        cam_keys       : list[str] — obs_cam_keys in model order
    """
    dset_id  = dataset.item_id_to_dataset_id[idx]
    cur_dset = dataset._datasets[dset_id]
    local_idx = idx - dataset.acc_dset_num[dset_id]
    cur_meta  = cur_dset.new_metas[local_idx]

    episode_index = cur_meta['episode_index']
    episode_chunk = cur_dset.meta.get_episode_chunk(episode_index)

    # frame_ids are stored inside the latent .pth file
    latent_data = cur_dset._get_range_latent_data(
        cur_meta['start_frame'], cur_meta['end_frame'], episode_index
    )
    cam0 = cur_dset.used_video_keys[0]
    frame_ids = latent_data[f"{cam0}.frame_ids"]   # (1 + (F_real-1)*4,) LongTensor

    return {
        'repo_root':     cur_dset.root,            # Path (HF_LEROBOT_HOME / repo_id)
        'episode_index': episode_index,
        'episode_chunk': episode_chunk,
        'frame_ids':     frame_ids,
        'cam_keys':      cur_dset.used_video_keys,
    }


def load_video_frame(
    repo_root: Path,
    episode_chunk: int,
    episode_index: int,
    cam_key: str,
    frame_idx: int,
    img_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Read one frame (episode-relative index) from the MP4 as (H,W,3) RGB uint8.

    Parameters
    ----------
    img_size : (W, H) tuple passed to cv2.resize, or None to keep native size.
               Should be set to (config.width, config.height) = (256, 256) so
               that _encode_obs receives already-downscaled frames and does not
               allocate a large intermediate tensor at the original resolution.
    """
    video_path = (
        repo_root
        / 'videos'
        / f"chunk-{episode_chunk:03d}"
        / cam_key
        / f"episode_{episode_index:06d}.mp4"
    )
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, bgr = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(
            f"cv2 could not read frame {frame_idx} from {video_path}.\n"
            "Check that the video exists and the frame index is in range."
        )
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if img_size is not None:
        rgb = cv2.resize(rgb, img_size, interpolation=cv2.INTER_LINEAR)
    return rgb.astype(np.uint8)


def build_obs(
    repo_root: Path,
    episode_chunk: int,
    episode_index: int,
    cam_keys: list[str],
    frame_ids,
    latent_frame_indices: list[int],
    repeat: int,
    img_size: tuple[int, int] | None = None,
) -> dict:
    """Build an obs dict suitable for VA_Server._encode_obs.

    For each latent frame index f in ``latent_frame_indices``, loads the
    representative video frame (frame_ids[4*f]) for every camera, then
    appends it ``repeat`` times.

    The resulting obs['obs'] list has len(latent_frame_indices) * repeat
    entries — matching the streaming VAE's 1,4,4,… temporal compression when
    repeat=4 and len(latent_frame_indices)=frame_chunk_size.

    Parameters
    ----------
    latent_frame_indices : local indices within the dataset item (0-based)
    repeat               : 4 for KV-cache obs (deployment), 1 for init frame
    img_size             : (W, H) for cv2.resize before stacking; pass
                           (config.width, config.height) to avoid allocating
                           full-resolution tensors inside _encode_obs.
    """
    obs_list = []
    for f_local in latent_frame_indices:
        vid_frame_idx = int(frame_ids[4 * f_local])
        frame_dict = {
            cam_key: load_video_frame(
                repo_root, episode_chunk, episode_index, cam_key,
                vid_frame_idx, img_size=img_size,
            )
            for cam_key in cam_keys
        }
        for _ in range(repeat):
            # shallow copy of dict — values are numpy arrays (immutable buffer)
            obs_list.append(dict(frame_dict))

    return {'obs': obs_list}


# ═══════════════════════════════════════════════════════════════════════════════
# Core inference
# ═══════════════════════════════════════════════════════════════════════════════

def _cuda_sync():
    """Flush the CUDA stream so wall-clock timers measure real GPU completion."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def run_episode_repeat_eval(
    server: VA_Server,
    sample: dict,
    video_meta: dict,
    num_chunks: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Sequential inference with deployment-style (repeated-frame) KV encoding.

    Differences from run_episode_gt_eval in test_gt_eval.py
    --------------------------------------------------------
    KV cache (chunk k > 0):
      GT eval     → inject pre-encoded GT latent directly (monkey-patch)
      This eval   → load last video frame per latent slot, repeat ×4,
                    encode through streaming VAE (_encode_obs runs naturally)

    Chunk 0 init:
      GT eval     → inject first GT latent frame (monkey-patch)
      This eval   → load first video frame, encode through VAE naturally

    Both: GT actions as predicted_actions (isolates visual quality effect).

    Parameters
    ----------
    server      : initialised VA_Server
    sample      : dict from MultiLatentLeRobotDataset.__getitem__
    video_meta  : dict from get_item_video_meta() with video path info
    num_chunks  : cap on chunks to run; None = whole item

    Returns
    -------
    pred_all : (T, C_used) float32 — denormalised predicted actions
    gt_all   : (T, C_used) float32 — denormalised GT actions
    timing   : list of per-chunk dicts with keys
               'chunk', 'video_load_s' (None for chunk 0), 'kv_s' (None for
               chunk 0), 'infer_s', 'total_s'
    """
    F = server.job_config.frame_chunk_size   # 4
    N = server.job_config.action_per_frame              # 8

    gt_actions   = sample['actions']         # (C_model, F_max, N, 1) normalised
    latents_mask = sample['latents_mask']    # (F_max,) bool
    text_emb     = sample['text_emb']

    # Video metadata
    repo_root     = video_meta['repo_root']
    episode_chunk = video_meta['episode_chunk']
    episode_index = video_meta['episode_index']
    frame_ids     = video_meta['frame_ids']   # episode-relative video frame indices
    cam_keys      = video_meta['cam_keys']

    # Pre-resize frames to the model's input resolution before stacking so
    # that _encode_obs never allocates a full-resolution intermediate tensor.
    # cv2.resize takes (W, H), config stores height/width separately.
    img_size = (server.job_config.width, server.job_config.height)  # (256, 256)

    F_real   = int(latents_mask.sum().item())
    n_chunks = F_real // F
    if num_chunks is not None:
        n_chunks = min(n_chunks, num_chunks)
    if n_chunks == 0:
        raise ValueError(
            f"Only {F_real} real latent frames but frame_chunk_size={F}."
        )

    gt_actions = gt_actions[:, :F_real]

    prepare_server(server, text_emb)

    all_pred: list[np.ndarray] = []
    all_gt:   list[np.ndarray] = []
    timing:   list[dict]       = []

    for k in range(n_chunks):
        f0 = k * F
        f1 = f0 + F
        t_chunk_start = time.perf_counter()
        video_load_s = None
        kv_s         = None

        if k == 0:
            # ── Chunk 0: encode single representative frame for init_latent ───
            #
            # In deployment, the first inference uses the robot's current image
            # (one frame).  We use the video frame corresponding to latent f=0.
            # _infer calls _encode_obs naturally — no monkey-patching.
            t_vid = time.perf_counter()
            obs_init = build_obs(
                repo_root, episode_chunk, episode_index, cam_keys,
                frame_ids,
                latent_frame_indices=[0],   # first latent frame only
                repeat=1,                   # single frame for init_latent
                img_size=img_size,
            )
            video_load_s = time.perf_counter() - t_vid

            # initial_state from GT (same as test_gt_eval.py)
            gt_st_norm    = gt_actions[:, f0:f0 + 1, 0:1, :].unsqueeze(0)
            initial_state = server.postprocess_action(gt_st_norm)[0].copy()

            t_infer_start = time.perf_counter()
            pred, _ = server._infer(
                obs_init, frame_st_id=0, initial_state=initial_state
            )
            _cuda_sync()
            infer_s = time.perf_counter() - t_infer_start
            # frame_st_id stays 0 — _infer does NOT advance it.

        else:
            # ── Chunk k>0: deployment-style KV — last frame of each slot ×4 ──
            #
            # The chunk server collects the last F frames (one per executed
            # chunk) and repeats each ×4 to fill the streaming VAE's temporal
            # window.  We replicate this exactly: for each of the F latent
            # frame slots in the previous chunk, load frame_ids[4*f_local]
            # (the last video frame of that slot) and repeat it 4 times.
            f_kv0 = f0 - F
            f_kv1 = f0

            t_vid = time.perf_counter()
            obs_kv = build_obs(
                repo_root, episode_chunk, episode_index, cam_keys,
                frame_ids,
                latent_frame_indices=list(range(f_kv0, f_kv1)),  # F latent frames
                repeat=4,    # each repeated ×4 → F*4 total obs frames
                img_size=img_size,
            )
            video_load_s = time.perf_counter() - t_vid

            # GT actions as KV action conditioning (isolates visual effect)
            server.predicted_actions = (
                gt_actions[:, f_kv0:f_kv1]
                .unsqueeze(0)
                .to(server.device)
                .to(server.dtype)
            )
            t_kv_start = time.perf_counter()
            server._compute_kv_cache(obs_kv)
            _cuda_sync()
            kv_s = time.perf_counter() - t_kv_start
            # frame_st_id is now k * F (advanced inside _compute_kv_cache)

            # For frame_st_id > 0, _infer does NOT call _encode_obs, so the
            # obs content is irrelevant.
            t_infer_start = time.perf_counter()
            pred, _ = server._infer(
                {'obs': [{}]}, frame_st_id=server.frame_st_id
            )
            _cuda_sync()
            infer_s = time.perf_counter() - t_infer_start

        total_s = time.perf_counter() - t_chunk_start

        # ── Ground truth for this chunk ───────────────────────────────────────
        gt_chunk = gt_actions[:, f0:f1]
        gt       = server.postprocess_action(gt_chunk.unsqueeze(0)).copy()

        all_pred.append(pred)
        all_gt.append(gt)
        timing.append({'chunk': k, 'video_load_s': video_load_s,
                       'kv_s': kv_s, 'infer_s': infer_s, 'total_s': total_s})

        vid_str = f"vid={video_load_s:.3f}s  " if video_load_s is not None else ""
        kv_str  = f"kv={kv_s:.2f}s  "          if kv_s         is not None else "kv=  —     "
        print(
            f"  chunk {k:3d}  frame_st_id={server.frame_st_id:4d}  "
            f"{vid_str}{kv_str}infer={infer_s:.2f}s  total={total_s:.2f}s"
        )

    pred_all = np.concatenate(all_pred, axis=0)
    gt_all   = np.concatenate(all_gt,   axis=0)
    return pred_all, gt_all, timing


# ═══════════════════════════════════════════════════════════════════════════════
# Plotting  (identical to test_gt_eval.py)
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
    title: str = "Repeat-frame KV vs GT Actions",
) -> None:
    T, C = gt.shape
    chunk_size = frame_chunk_size * action_per_frame

    fig, axes = plt.subplots(C, 1, figsize=(16, C * 2.2), sharex=True)
    fig.suptitle(title, fontsize=13, y=1.002)

    t = np.arange(T)
    for dim in range(C):
        ax = axes[dim]
        ax.plot(t, gt  [:, dim], color='steelblue',  linewidth=1.0, label='Ground Truth')
        ax.plot(t, pred[:, dim], color='darkorange',  linewidth=0.9, label='Prediction', alpha=0.85)

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
    parser.add_argument('--config', required=True,
        help='VA_CONFIGS key OR dotted import path, e.g. '
             'wan_va.configs.ma_configs.ma_sim_config')
    parser.add_argument('--eval-model-path', default=None,
        help='Override wan22_pretrained_model_name_or_path / finetuned path')
    parser.add_argument('--dataset-path', default=None,
        help='Override dataset_path in config')
    parser.add_argument('--dataset-idx', type=int, default=0,
        help='Index into MultiLatentLeRobotDataset (default: 0)')
    parser.add_argument('--num-chunks', type=int, default=None,
        help='Max chunks to run (default: whole item)')
    parser.add_argument('--output-dir', default='./repeat_eval_out',
        help='Directory for arrays, stats and plots')
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
        for attr in ('wan22_pretrained_model_name_or_path',
                     'wan22_finetuned_model_name_or_path'):
            if hasattr(config, attr):
                setattr(config, attr, args.eval_model_path)
    config.local_rank  = 0
    config.rank        = 0
    config.world_size  = 1
    config.save_root   = str(output_dir / 'server_debug')
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
            f"--dataset-idx {args.dataset_idx} out of range ({n_total} items)"
        )

    sample     = dataset[args.dataset_idx]
    video_meta = get_item_video_meta(dataset, args.dataset_idx)

    F_real = int(sample['latents_mask'].sum().item())
    F  = config.frame_chunk_size
    N  = config.action_per_frame
    print(f"Item {args.dataset_idx}: {F_real} real latent frames → "
          f"{F_real // F} complete chunks × {F}×{N} = {F_real // F * F * N} steps")
    print(f"Video root : {video_meta['repo_root']}")
    print(f"Episode    : {video_meta['episode_index']}  "
          f"chunk {video_meta['episode_chunk']}")

    # ── Model ─────────────────────────────────────────────────────────────────
    print("Loading model …")
    server = VA_Server(config)

    # ── Inference ─────────────────────────────────────────────────────────────
    print("\nRunning repeat-frame KV inference …")
    pred_all, gt_all, timing = run_episode_repeat_eval(
        server, sample, video_meta, num_chunks=args.num_chunks
    )

    # ── Save arrays ───────────────────────────────────────────────────────────
    np.save(output_dir / 'pred_actions.npy', pred_all)
    np.save(output_dir / 'gt_actions.npy',   gt_all)
    print(f"\nArrays saved (shapes: pred={pred_all.shape}, gt={gt_all.shape})")

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
        f"Repeat-frame KV eval — item {args.dataset_idx}  "
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
    infer_times = [r['infer_s']       for r in timing]
    kv_times    = [r['kv_s']          for r in timing if r['kv_s']          is not None]
    vid_times   = [r['video_load_s']  for r in timing if r['video_load_s']  is not None]
    total_times = [r['total_s']       for r in timing]

    def _fmt(vals):
        return f"{np.mean(vals):.2f}s  (min {np.min(vals):.2f}s, max {np.max(vals):.2f}s)"

    print("\n" + "━" * 54)
    print("  Timing summary")
    print("━" * 54)
    print(f"  chunks run        : {len(timing)}")
    if vid_times:
        print(f"  video_load/ chunk : {_fmt(vid_times)}")
    if kv_times:
        print(f"  kv_cache / chunk  : {_fmt(kv_times)}  [chunks 1+]")
    print(f"  infer    / chunk  : {_fmt(infer_times)}")
    print(f"  total    / chunk  : {_fmt(total_times)}")
    print(f"  wall time total   : {sum(total_times):.1f}s")
    print("━" * 54)

    print(f"\nDone. Results in {output_dir}/")


if __name__ == '__main__':
    main()
