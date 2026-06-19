"""
Scan all datasets under a root, report per-dataset and global maximum
latent frame count after optional temporal downsampling + VAE encoding.

For each chunk in action_config:
    chunk_frames       = end_frame - start_frame
    downsampled_frames = ceil(chunk_frames / downsample_rate)
    latent_frames      = (downsampled_frames - 1) // 4 + 1

One .pth file per dataset is sampled and loaded to validate that the
actual stored latent frame count matches the formula at downsample_rate=1
(since existing files were encoded without downsampling).

Usage
-----
# Check with no downsampling (current state):
python preprocessing/check_max_latent_frames.py \\
    --dataset-root /path/to/datasets \\
    --cam-key observation.images.cam_high

# Check what max_latent_frames would be with 2× temporal downsampling:
python preprocessing/check_max_latent_frames.py \\
    --dataset-root /path/to/datasets \\
    --cam-key observation.images.cam_high \\
    --downsample-rate 2
"""
import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import torch
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Formula helpers
# ---------------------------------------------------------------------------

def chunk_latent_frames(chunk_len: int, downsample_rate: int) -> int:
    """Number of VAE latent frames for a chunk of `chunk_len` raw frames."""
    downsampled = math.ceil(chunk_len / downsample_rate)
    return max(1, (downsampled - 1) // 4 + 1)


# ---------------------------------------------------------------------------
# .pth validation
# ---------------------------------------------------------------------------

def find_pth_sample(dataset_dir: Path, cam_key: str | None, episodes_file: str):
    """
    Return (pth_path, episode_index, start_frame, end_frame, chunk_len) for the
    first chunk in episodes.jsonl that has an existing .pth file on disk.
    Returns None if nothing can be found.
    """
    ep_path = dataset_dir / episodes_file
    if not ep_path.exists():
        return None

    with open(ep_path) as f:
        lines = [l.strip() for l in f if l.strip()]

    random.shuffle(lines)   # sample randomly rather than always taking ep 0

    for line in lines:
        ep = json.loads(line)
        episode_index = ep.get("episode_index")
        chunk_size    = ep.get("chunk_size", 1000)
        episode_chunk = episode_index // chunk_size

        for acfg in ep.get("action_config", []):
            start_frame = acfg["start_frame"]
            end_frame   = acfg["end_frame"]
            fname       = f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"

            # Determine which cam keys to try
            latents_dir = dataset_dir / "latents" / f"chunk-{episode_chunk:03d}"
            if cam_key:
                keys_to_try = [cam_key]
            else:
                # Auto-discover: pick the first subdirectory under the chunk dir
                if latents_dir.exists():
                    keys_to_try = [p.name for p in sorted(latents_dir.iterdir())
                                   if p.is_dir()]
                else:
                    keys_to_try = []

            for key in keys_to_try:
                pth = latents_dir / key / fname
                if pth.exists():
                    return pth, episode_index, start_frame, end_frame, end_frame - start_frame

    return None


def validate_pth(pth_path: Path):
    """
    Load a .pth dict saved by extract_latent_vae.py and return
    (ok, actual_latent_frames, video_num_frames, note).

    The dict contains:
        'latent_num_frames' : int  — latent frames after VAE (rate=1)
        'video_num_frames'  : int  — raw video frames fed to the VAE
    """
    try:
        data = torch.load(pth_path, map_location="cpu", weights_only=False)
    except Exception as e:
        return False, None, None, f"load error: {e}"

    if not isinstance(data, dict):
        return False, None, None, f"unexpected type {type(data).__name__}"

    actual_lf  = data.get("latent_num_frames")
    actual_vf  = data.get("video_num_frames")

    if actual_lf is None or actual_vf is None:
        return False, actual_lf, actual_vf, (
            f"missing keys (have: {list(data.keys())})"
        )

    # Cross-check: latent_num_frames should equal (video_num_frames - 1) // 4 + 1
    expected_lf = max(1, (actual_vf - 1) // 4 + 1)
    if actual_lf == expected_lf:
        return True, actual_lf, actual_vf, (
            f"video_frames={actual_vf} → latent_frames={actual_lf} ✓"
        )
    else:
        return False, actual_lf, actual_vf, (
            f"video_frames={actual_vf} → expected latent={expected_lf}, "
            f"got {actual_lf} ✗"
        )


# ---------------------------------------------------------------------------
# Per-dataset analysis
# ---------------------------------------------------------------------------

def analyse_dataset(dataset_dir: Path, cam_key: str | None,
                    episodes_file: str, downsample_rate: int):
    """
    Returns a dict with keys:
        max_latent, max_chunk_len, chunk_count, hist,
        validation_ok, validation_note, validation_shape,
        max_latent_at_rate1   (for the recommendation line)
    """
    ep_path = dataset_dir / episodes_file
    if not ep_path.exists():
        return None

    max_latent   = 0
    max_chunk_len = 0
    chunk_count  = 0
    max_latent_rate1 = 0
    hist         = {}   # latent_frames -> count (at the requested rate)

    with open(ep_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ep = json.loads(line)
            for acfg in ep.get("action_config", []):
                chunk_len = acfg["end_frame"] - acfg["start_frame"]
                lf        = chunk_latent_frames(chunk_len, downsample_rate)
                lf_rate1  = chunk_latent_frames(chunk_len, 1)
                hist[lf]  = hist.get(lf, 0) + 1
                if lf > max_latent:
                    max_latent    = lf
                    max_chunk_len = chunk_len
                if lf_rate1 > max_latent_rate1:
                    max_latent_rate1 = lf_rate1
                chunk_count += 1

    if chunk_count == 0:
        return None

    # Validate one .pth file at rate=1 (files on disk were encoded without downsampling)
    sample = find_pth_sample(dataset_dir, cam_key, episodes_file)
    if sample is not None:
        pth_path, ep_idx, sf, ef, cl = sample
        ok, actual_lf, actual_vf, note = validate_pth(pth_path)
        val_ok   = ok
        val_note = f"ep{ep_idx:06d}[{sf}:{ef}] {note}"
    else:
        val_ok   = None
        val_note = "no .pth file found for validation"

    return {
        "max_latent":       max_latent,
        "max_chunk_len":    max_chunk_len,
        "max_latent_rate1": max_latent_rate1,
        "chunk_count":      chunk_count,
        "hist":             hist,
        "validation_ok":    val_ok,
        "validation_note":  val_note,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def percentile_from_hist(hist: dict, p: float) -> int:
    """Compute a percentile value from a {value: count} histogram."""
    total   = sum(hist.values())
    target  = total * p / 100.0
    running = 0
    for v in sorted(hist):
        running += hist[v]
        if running >= target:
            return v
    return max(hist)


def fmt_val(val_ok):
    if val_ok is True:
        return "✓"
    elif val_ok is False:
        return "✗"
    else:
        return "?"


def main():
    parser = argparse.ArgumentParser(
        description="Report max latent frame counts across datasets."
    )
    parser.add_argument("--dataset-root", required=True,
                        help="Root directory containing dataset folders.")
    parser.add_argument("--cam-key", default=None,
                        help="Camera key for .pth validation "
                             "(auto-discovered if omitted).")
    parser.add_argument("--downsample-rate", type=int, default=1,
                        help="Temporal downsampling factor to apply before VAE "
                             "(default 1 = no downsampling).")
    parser.add_argument("--episodes-file", default="meta/episodes.jsonl",
                        help="Relative path to episodes file "
                             "(default: meta/episodes.jsonl).")
    parser.add_argument("--show-hist", action="store_true",
                        help="Print per-dataset latent-frame histograms.")
    args = parser.parse_args()

    root = Path(args.dataset_root)
    # pathlib.rglob does not follow symlinks; use os.walk with followlinks=True.
    target = args.episodes_file   # e.g. "meta/episodes.jsonl"
    ep_paths = sorted(
        (Path(dirpath) / target
         for dirpath, dirnames, filenames in os.walk(root, followlinks=True)
         if (Path(dirpath) / target).exists()),
        key=lambda p: str(p),
    )

    if not ep_paths:
        print(f"No datasets found under {root} "
              f"(looked for '{args.episodes_file}').")
        sys.exit(1)

    datasets = [p.parent.parent for p in ep_paths]
    print(f"Found {len(datasets)} dataset(s) under {root}")
    print(f"Downsample rate : {args.downsample_rate}")
    print(f"Camera key      : {args.cam_key or '(auto)'}")
    print()

    col_w = max(len(str(d.name)) for d in datasets)
    col_w = max(col_w, 12)
    rate_suffix = f"r={args.downsample_rate}" if args.downsample_rate > 1 else "r=1"
    header = (f"{'Dataset':<{col_w}}  {'chunks':>7}  "
              f"{'max(r=1)':>8}  {f'max({rate_suffix})':>10}  "
              f"{'p50':>4}  {'p90':>4}  {'p99':>4}  "
              f"{'val':>4}  validation detail")

    global_max_lf      = 0
    global_max_lf_r1   = 0
    global_chunks      = 0
    global_hist: dict  = {}
    all_results        = []
    output_lines       = []   # collect rows; print table after bar finishes

    with tqdm(datasets, desc="Scanning", unit="dataset", dynamic_ncols=True) as pbar:
        for ds in pbar:
            pbar.set_postfix_str(ds.name[:40], refresh=True)
            result = analyse_dataset(ds, args.cam_key, args.episodes_file,
                                     args.downsample_rate)
            if result is None:
                output_lines.append(f"{'  [SKIP]':<{col_w}}  {ds}")
                continue

            all_results.append((ds, result))
            global_max_lf    = max(global_max_lf,    result["max_latent"])
            global_max_lf_r1 = max(global_max_lf_r1, result["max_latent_rate1"])
            global_chunks   += result["chunk_count"]
            for v, c in result["hist"].items():
                global_hist[v] = global_hist.get(v, 0) + c

            p50 = percentile_from_hist(result["hist"], 50)
            p90 = percentile_from_hist(result["hist"], 90)
            p99 = percentile_from_hist(result["hist"], 99)

            output_lines.append(
                f"{ds.name:<{col_w}}  {result['chunk_count']:>7}  "
                f"{result['max_latent_rate1']:>8}  {result['max_latent']:>10}  "
                f"{p50:>4}  {p90:>4}  {p99:>4}  "
                f"{fmt_val(result['validation_ok']):>4}  "
                f"{result['validation_note']}"
            )

            if args.show_hist:
                for v in sorted(result["hist"]):
                    bar = "█" * min(40, result["hist"][v])
                    output_lines.append(
                        f"    lf={v:3d}  {result['hist'][v]:6d}  {bar}"
                    )
                output_lines.append("")

    # Print the full table once the progress bar is done
    print()
    print(header)
    print("-" * len(header))
    for line in output_lines:
        print(line)

    # Global summary
    print()
    print("=" * len(header))
    gp50 = percentile_from_hist(global_hist, 50) if global_hist else "-"
    gp90 = percentile_from_hist(global_hist, 90) if global_hist else "-"
    gp99 = percentile_from_hist(global_hist, 99) if global_hist else "-"
    print(f"{'GLOBAL':<{col_w}}  {global_chunks:>7}  "
          f"{global_max_lf_r1:>8}  {global_max_lf:>10}  "
          f"{gp50:>4}  {gp90:>4}  {gp99:>4}")
    print()

    if args.downsample_rate == 1:
        print(f"→  max_latent_frames (no downsampling) = {global_max_lf}")
        print(f"   Set va_shared_cfg.max_latent_frames = {global_max_lf}")
    else:
        print(f"→  max_latent_frames (rate=1, current files) = {global_max_lf_r1}")
        print(f"   max_latent_frames (rate={args.downsample_rate}, after downsampling) = {global_max_lf}")
        print(f"   Set va_shared_cfg.max_latent_frames = {global_max_lf}  "
              f"(if you re-extract latents at {args.downsample_rate}× downsampling)")


if __name__ == "__main__":
    main()