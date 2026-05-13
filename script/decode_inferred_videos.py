"""
decode_inferred_videos.py
=========================
Decode saved inference latents back to video and save as MP4.

Given a `real/` directory produced by the WAN-VA server, the script:
  1. Discovers all per-inference sub-directories (timestamped folders).
  2. For each one, loads the per-chunk latent files (latents_1.pt, latents_5.pt …
     or a single latents_all.pt if the unified save was used), concatenates them
     along the time dimension, decodes through the WAN VAE, and writes an MP4.
  3. Skips any folder that already contains a decoded video.
  4. Supports multi-GPU / multi-process decoding via --workers (one process per
     inference folder, round-robin across GPUs).

Directory layout expected
-------------------------
real/
  20260420_180325/
    latents_1.pt   latents_5.pt   latents_9.pt   ...  (per-chunk, frame_st_id-named)
    latents_all.pt                                     (optional, preferred if present)
    actions_*.pt   obs_data_*.pt                       (ignored by this script)
    decoded.mp4                                        (written here; skipped if exists)

Usage examples
--------------
# Single GPU, sequential
python script/decode_inferred_videos.py \
    --real-dir /path/to/checkpoints/run/real \
    --vae-path /path/to/pretrained/model/vae

# 4 parallel workers spread across 2 GPUs (gpu 0 and 1)
python script/decode_inferred_videos.py \
    --real-dir /path/to/checkpoints/run/real \
    --vae-path /path/to/pretrained/model/vae \
    --workers 4 --gpus 0 1

# Decode only into a specific output file name; use bf16
python script/decode_inferred_videos.py \
    --real-dir /path/to/checkpoints/run/real \
    --vae-path /path/to/pretrained/model/vae \
    --output-name video.mp4 --dtype bf16 --fps 15
"""

import argparse
import os
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the wan_va package is importable when run from any working directory
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_WAN_VA_DIR = _SCRIPT_DIR.parent   # wan_va/
sys.path.insert(0, str(_WAN_VA_DIR))


# ============================================================
# Per-worker decode function (importable for multiprocessing)
# ============================================================
def _decode_folder(args_tuple):
    """Decode a single inference folder.  Designed to be called in a worker
    process so it imports torch/diffusers locally to avoid CUDA context issues
    when forking.

    Parameters
    ----------
    args_tuple : (folder: Path, vae_path: str, device: str, dtype_str: str,
                  output_name: str, fps: int)
    """
    folder, vae_path, device, dtype_str, output_name, fps = args_tuple

    import torch
    from diffusers import AutoencoderKLWan
    from diffusers.video_processor import VideoProcessor
    from diffusers.utils import export_to_video

    out_path = folder / output_name
    if out_path.exists():
        print(f"[skip]  {folder.name}  (already decoded)")
        return str(out_path)

    # ── 1. Load latents ──────────────────────────────────────────────────────
    all_pt = folder / "latents_all.pt"
    if all_pt.exists():
        # Preferred: single concatenated file written by _flush_job_chunks
        latents = torch.load(all_pt, map_location="cpu", weights_only=True)
        print(f"[load]  {folder.name}  latents_all.pt  {tuple(latents.shape)}")
    else:
        # Fall back to per-chunk files named latents_{frame_st_id}.pt
        chunk_files = sorted(
            folder.glob("latents_*.pt"),
            key=lambda p: int(re.search(r"latents_(\d+)\.pt", p.name).group(1))
        )
        if not chunk_files:
            print(f"[skip]  {folder.name}  no latent files found")
            return None

        chunks = []
        for cf in chunk_files:
            t = torch.load(cf, map_location="cpu", weights_only=True)
            chunks.append(t)
            print(f"  loaded {cf.name}  {tuple(t.shape)}")

        # Latent shape per chunk: (1, C, T_chunk, H, W)
        # Concatenate along dim=2 (time)
        latents = torch.cat(chunks, dim=2)
        print(f"[concat] {folder.name}  combined {tuple(latents.shape)}")

    # ── 2. Load VAE ──────────────────────────────────────────────────────────
    dtype = {"fp32": torch.float32,
             "fp16": torch.float16,
             "bf16": torch.bfloat16}[dtype_str]

    vae = AutoencoderKLWan.from_pretrained(vae_path, torch_dtype=dtype)
    vae = vae.to(device).eval()

    # ── 3. Unnormalise latents ───────────────────────────────────────────────
    # Inverse of the normalisation applied during encoding:
    #   norm = (mu - latents_mean) * (1 / latents_std)
    # → unnorm = norm * latents_std + latents_mean
    latents = latents.to(device=device, dtype=dtype)
    latents_mean = (
        torch.tensor(vae.config.latents_mean)
        .view(1, vae.config.z_dim, 1, 1, 1)
        .to(device=device, dtype=dtype)
    )
    latents_std = (
        (1.0 / torch.tensor(vae.config.latents_std))
        .view(1, vae.config.z_dim, 1, 1, 1)
        .to(device=device, dtype=dtype)
    )
    latents_unnorm = latents / latents_std + latents_mean

    # ── 4. Decode ────────────────────────────────────────────────────────────
    with torch.no_grad():
        video_tensor = vae.decode(latents_unnorm, return_dict=False)[0]

    video_processor = VideoProcessor(vae_scale_factor=1)
    frames_np = video_processor.postprocess_video(video_tensor, output_type="np")[0]
    # frames_np: list of (H, W, 3) float32 in [0, 1]

    # ── 5. Save MP4 ──────────────────────────────────────────────────────────
    export_to_video(frames_np, str(out_path), fps=fps)
    print(f"[saved] {out_path}  ({len(frames_np)} frames @ {fps} fps)")

    # Clean up GPU memory before the next folder
    del vae, latents, latents_unnorm, video_tensor
    if "cuda" in device:
        torch.cuda.empty_cache()

    return str(out_path)


# ============================================================
# Discovery helpers
# ============================================================
def _find_inference_folders(real_dir: Path) -> list:
    """Return all sub-directories of real_dir that contain at least one
    latents_*.pt file, sorted by name (chronological for timestamp names)."""
    folders = []
    for entry in sorted(real_dir.iterdir()):
        if not entry.is_dir():
            continue
        has_latents = (entry / "latents_all.pt").exists() or \
                      bool(list(entry.glob("latents_*.pt")))
        if has_latents:
            folders.append(entry)
    return folders


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Decode WAN-VA inference latents → MP4 videos"
    )
    parser.add_argument(
        "--real-dir", required=True, type=Path,
        help="Path to the real/ directory containing timestamped inference folders"
    )
    parser.add_argument(
        "--vae-path", type=str, default="/liujinxin/weights/lingbot-va-base/vae",
        help="Path to the pretrained VAE directory (e.g. .../pretrained_model/vae)"
    )
    parser.add_argument(
        "--output-name", default="decoded.mp4", type=str,
        help="Filename for the decoded video inside each inference folder "
             "(default: decoded.mp4)"
    )
    parser.add_argument(
        "--fps", default=10, type=int,
        help="Frames per second for the output video (default: 10)"
    )
    parser.add_argument(
        "--dtype", default="bf16", choices=["fp32", "fp16", "bf16"],
        help="VAE dtype (default: bf16)"
    )
    parser.add_argument(
        "--gpus", nargs="+", type=int, default=[0],
        metavar="GPU_ID",
        help="GPU IDs to use (default: 0). Multiple IDs enable multi-GPU decoding."
    )
    parser.add_argument(
        "--workers", default=1, type=int,
        help="Number of parallel worker processes (default: 1 = sequential). "
             "Workers are round-robin assigned to the specified --gpus."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-decode even if the output video already exists"
    )
    args = parser.parse_args()

    real_dir: Path = args.real_dir.resolve()
    if not real_dir.is_dir():
        sys.exit(f"ERROR: --real-dir {real_dir} does not exist or is not a directory")
    if not Path(args.vae_path).is_dir():
        sys.exit(f"ERROR: --vae-path {args.vae_path} does not exist or is not a directory")

    folders = _find_inference_folders(real_dir)
    if not folders:
        sys.exit(f"No inference folders with latents found under {real_dir}")

    # If --force, delete existing decoded videos so _decode_folder re-runs them
    if args.force:
        for f in folders:
            out = f / args.output_name
            if out.exists():
                out.unlink()
                print(f"[force] removed {out}")

    print(f"Found {len(folders)} inference folder(s) under {real_dir}")
    print(f"GPUs: {args.gpus}   workers: {args.workers}   dtype: {args.dtype}")
    print()

    # Build task list: assign each folder to a device round-robin
    devices = [f"cuda:{g}" for g in args.gpus] if args.gpus else ["cpu"]
    tasks = [
        (
            folder,
            args.vae_path,
            devices[i % len(devices)],
            args.dtype,
            args.output_name,
            args.fps,
        )
        for i, folder in enumerate(folders)
    ]

    if args.workers <= 1:
        # Sequential — useful for single-GPU to avoid OOM from concurrent loads
        results = [_decode_folder(t) for t in tasks]
    else:
        # Parallel workers
        # Use "spawn" start method so each worker gets a clean CUDA context
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.workers) as pool:
            results = pool.map(_decode_folder, tasks)

    succeeded = [r for r in results if r is not None]
    skipped   = results.count(None)  # None means no latents or already decoded
    print(f"\nDone.  {len(succeeded)} decoded,  {skipped} skipped.")


if __name__ == "__main__":
    main()
