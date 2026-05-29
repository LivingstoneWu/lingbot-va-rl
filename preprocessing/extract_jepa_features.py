"""
extract_jepa_features.py — Extract V-JEPA 2.1 ViT-Gigantic features for robot video datasets.
==============================================================================================

For each episode mp4, samples frames at `--target-fps`, encodes the entire episode
in one JEPA forward pass (no windowing — RoPE makes this resolution-agnostic), and
saves per-episode features as:

    {dataset_root}/jepa/chunk-{NNN}/episode_{NNNNNN}.pt

Each .pt file is a dict:
    {
        'observation.images.left_wrist': tensor[T_jepa, H', W', 1664],  # bfloat16
        'observation.images.front':      tensor[T_jepa, H', W', 1664],  # bfloat16
        'frame_ids': int64 array[N_sampled_frames],
    }

Temporal alignment with VAE latents
------------------------------------
The Wan VAE uses a (1, 4, 4, 4, …) temporal compression pattern: the first video
frame maps to latent 0 alone, then every subsequent group of 4 frames maps to one
latent (groups: [1-4], [5-8], …).  JEPA uses tubelet_size=2, so token t covers
frames [2t, 2t+1].

To align the two grids we prepend a duplicate of frame 0 before the JEPA forward
pass (turning 4k+1 frames into 4k+2).  The resulting token layout is:

    JEPA token 0         → frames [f0, f0]   → VAE latent 0
    JEPA tokens 1,2      → frames [f1,f2],[f3,f4]  → VAE latent 1
    JEPA tokens 2j-1, 2j → [f_{4j-3},f_{4j-2}],[f_{4j-1},f_{4j}] → VAE latent j

In the dataset loader, VAE latent j is predicted from the average of JEPA tokens
2j-1 and 2j (or token 0 alone for j=0).

NOTE: this fixed arithmetic mapping is only exact when a VAE chunk starts at
episode frame 0 (i.e. one chunk per episode, as is the case for the current
datasets).  Once multiple chunks with randomised starting offsets are introduced,
this script will need to read the VAE latent frame_ids and match JEPA tokens to
each chunk's exact frame range to preserve alignment across arbitrary offsets.

During training, frame_ids (saved without the prepended duplicate) is available
for this future alignment step.

Usage (multi-GPU via torchrun):
    torchrun --nproc_per_node=8 preprocessing/extract_jepa_features.py \\
        --dataset-root  /data/robochallenge/scan_QR_code_trim_stage2_test \\
        --checkpoint    /checkpoints/vjepa2_1_vitG_384.pt \\
        --camera-keys   observation.images.left_wrist observation.images.front \\
        --target-fps    12.5 \\
        --cam-resolutions observation.images.cam_high:256x320 \\
                          observation.images.cam_left_wrist:128x160 \\
                          observation.images.cam_right_wrist:128x160 \\
        --batch-size    4 \\
        --skip-existing

Per-camera resolution (`--cam-resolutions`):
    JEPA patches are 16×16 px, matching one VAE latent cell (also stride 16).
    The DiT then patchifies with patch_size=(1,2,2), so each DiT token covers
    32×32 px (2×2 latent cells).  To obtain JEPA targets that align with the
    DiT token grid after a 2×2 avg-pool in the dataset loader, extract each
    camera at its native resolution so that the token grid matches the VAE
    latent spatial dimensions:
        cam H_tokens = video_height / 16   (= VAE latent_height)
        cam W_tokens = video_width  / 16   (= VAE latent_width)
    The _DEFAULT_CAM_RESOLUTIONS dict below encodes known camera resolutions;
    --cam-resolutions overrides it per camera; --height/--width is the final
    fallback for unlisted cameras.

Single-GPU:
    python preprocessing/extract_jepa_features.py --dataset-root ... --checkpoint ...
"""

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

# ── repo path wiring ──────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent
VJEPA_ROOT = REPO_ROOT / 'vjepa2'
for _p in (str(VJEPA_ROOT), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Default per-camera resolutions ───────────────────────────────────────────
# Each camera should be extracted at its native video resolution so that the
# JEPA token grid (patch 16px) matches the VAE latent spatial dimensions.
# Add new camera keys here as new datasets are introduced.
# These are overridden by --cam-resolutions on the CLI; cameras not found in
# either fall back to --height / --width.
_DEFAULT_CAM_RESOLUTIONS: dict[str, tuple[int, int]] = {
    # RoboTwin T-shape rig: front camera 256×320, wrist cameras 128×160
    'observation.images.cam_high':        (256, 320),
    'observation.images.cam_left_wrist':  (128, 160),
    'observation.images.cam_right_wrist': (128, 160),
}

# ImageNet normalisation (JEPA uses this, distinct from the VAE's [-1, 1])
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def _clean_state_dict(sd: dict) -> dict:
    """Strip 'module.' (DDP/FSDP) and 'backbone.' (wrapper) prefixes."""
    out = {}
    for k, v in sd.items():
        k = k.replace('module.', '')
        k = k.replace('backbone.', '')
        out[k] = v
    return out


def build_jepa_encoder(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """
    Instantiate JEPA 2.1 ViT-Gigantic and load the target_encoder (EMA) weights.

    Key facts baked in here:
    - Uses app.vjepa_2_1.models.vision_transformer (the 2.1-specific ViT)
    - vit_gigantic_xformers: embed_dim=1664, depth=48, num_heads=26
    - use_rope=True  →  RoPE positions computed on-the-fly; 256×256 works fine
      even though checkpoint was trained at 384×384
    - img_temporal_dim_size=1  →  single-frame inputs are treated as images;
      any T>1 input is treated as video and uses PatchEmbed3D
    - target_encoder key  →  EMA-smoothed teacher, better downstream features
      than the online encoder
    """
    from app.vjepa_2_1.models.vision_transformer import vit_gigantic_xformers

    encoder = vit_gigantic_xformers(
        patch_size=16,
        img_size=(256, 256),
        num_frames=64,          # only needed for is_video flag; RoPE ignores it
        tubelet_size=2,
        use_sdpa=True,
        use_silu=False,
        wide_silu=True,
        uniform_power=False,
        use_rope=True,
        img_temporal_dim_size=1,
        interpolate_rope=True,
    )

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    sd   = _clean_state_dict(ckpt['target_encoder'])
    missing, unexpected = encoder.load_state_dict(sd, strict=False)
    if missing:
        print(f'[JEPA] Warning – {len(missing)} missing keys '
              f'(first 3: {missing[:3]})')
    if unexpected:
        print(f'[JEPA] Warning – {len(unexpected)} unexpected keys '
              f'(first 3: {unexpected[:3]})')

    # Run in float32: RoPE explicitly casts Q/K to float32 for the rotation,
    # so bfloat16 V causes a dtype mismatch in SDPA.  float32 throughout is
    # the safe choice; output features are cast to bfloat16 only on save.
    encoder = encoder.to(device=device, dtype=torch.float32)
    encoder.eval()
    return encoder


# ─────────────────────────────────────────────────────────────────────────────
# Video I/O
# ─────────────────────────────────────────────────────────────────────────────

def read_frames(video_path: str,
                target_fps: float,
                height: int,
                width: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample frames from `video_path` at `target_fps`, resize to (height, width).

    Replicates the *episode-level* sampling from extract_latent_vae.py exactly,
    including the (n-1) % 4 == 0 adjustment, so that the resulting frame_ids are
    a strict superset of every chunk's frame_ids stored in the VAE latent files.
    The np.searchsorted alignment in the dataset loader then works correctly.

    Returns
    -------
    frames    : uint8  [N, H, W, 3]  RGB
    frame_ids : int64  [N]           original-video frame indices
    """
    cap          = cv2.VideoCapture(video_path)
    orig_fps     = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0 or orig_fps <= 0:
        cap.release()
        return np.empty((0, height, width, 3), dtype=np.uint8), np.empty(0, dtype=np.int64)

    # ── replicate extract_latent_vae.py episode-level sampling exactly ────────
    step = max(1, int(orig_fps / target_fps)) if target_fps < orig_fps else 1
    target_count = int(total_frames / step)

    # Enforce (n-1) % 4 == 0 so JEPA frame_ids are a superset of every VAE
    # chunk's frame_ids (VAE chunks are contiguous slices of this sequence).
    adjusted_count = target_count
    if (adjusted_count - 1) % 4 != 0:
        adjusted_count = ((adjusted_count - 1) // 4) * 4 + 1
    if adjusted_count > target_count:
        adjusted_count -= 4
    adjusted_count = max(adjusted_count, 1)

    indices   = np.linspace(0, total_frames - 1, adjusted_count, dtype=int)
    # frame_ids are the actual original-video indices (may contain duplicates
    # from linspace rounding, which is fine — they are kept for alignment).
    frame_ids_all = indices.tolist()

    # For the physical decode pass we only seek each unique frame index once.
    unique_wanted = sorted(set(frame_ids_all))
    wanted_set    = set(unique_wanted)

    decoded: dict[int, np.ndarray] = {}
    cur = 0
    wp  = 0

    while wp < len(unique_wanted):
        ret, frame = cap.read()
        if not ret:
            break
        if cur in wanted_set:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)
            decoded[cur] = frame
            wp += 1
        cur += 1

    # Reconstruct ordered list respecting original linspace order (duplicates
    # map to the same decoded frame — consistent with the VAE script behaviour).
    frames    = [decoded[fid] for fid in frame_ids_all if fid in decoded]
    frame_ids = [fid          for fid in frame_ids_all if fid in decoded]

    cap.release()

    if not frames:
        return np.empty((0, height, width, 3), dtype=np.uint8), np.empty(0, dtype=np.int64)

    return np.stack(frames, axis=0), np.array(frame_ids, dtype=np.int64)   # [N,H,W,3], [N]


def frames_to_tensor(frames: np.ndarray) -> torch.Tensor:
    """
    [N, H, W, 3] uint8  →  [1, 3, N, H, W] bfloat16, ImageNet-normalised.
    The leading batch-1 dimension lets us cat multiple episodes easily.
    """
    t    = torch.from_numpy(frames).float().div_(255.0)      # [N, H, W, 3]
    t    = t.permute(3, 0, 1, 2)                             # [3, N, H, W]
    mean = _IMAGENET_MEAN.view(3, 1, 1, 1)
    std  = _IMAGENET_STD.view(3, 1, 1, 1)
    t    = (t - mean) / std                                   # [3, N, H, W]
    return t.unsqueeze(0)                                    # [1, 3, N, H, W] float32


# ─────────────────────────────────────────────────────────────────────────────
# JEPA encoding
# ─────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def encode_batch(
    encoder:  torch.nn.Module,
    tensors:  list[torch.Tensor],   # each [1, 3, N, H, W] float32
    device:   torch.device,
) -> list[torch.Tensor]:
    """
    Run one JEPA forward pass on a list of same-length, same-resolution episode tensors.

    H_patches and W_patches are derived from the input tensor shape so that
    different cameras (potentially at different resolutions) are handled correctly
    without any external bookkeeping.

    Returns a list of cpu bfloat16 tensors, each [T_jepa, H_patches, W_patches, D].
    """
    batch     = torch.cat(tensors, dim=0).to(device)  # [B, 3, N, H, W]
    N         = batch.shape[2]
    H_patches = batch.shape[3] // 16
    W_patches = batch.shape[4] // 16
    T_jepa    = N // 2

    # [B, T_jepa*H_patches*W_patches, D]
    feats = encoder(batch)
    D     = feats.shape[-1]

    # Reshape to spatial grid and cast to bfloat16 for compact storage
    feats = feats.view(feats.shape[0], T_jepa, H_patches, W_patches, D)  # [B, T, H', W', D]
    feats = feats.to(torch.bfloat16)

    return [feats[i].cpu() for i in range(feats.shape[0])]


# ─────────────────────────────────────────────────────────────────────────────
# Batch accumulator
# ─────────────────────────────────────────────────────────────────────────────

class EpisodeBatchAccumulator:
    """
    Accumulates episode read results grouped by frame count.
    When a bucket reaches `batch_size`, encodes and saves the whole group.

    Each pending item is:
        (chunk_str, ep_id, cam_data)
    where cam_data = {cam_key: (video_tensor [1,3,N,H,W], frame_ids int64[N])}
    """

    def __init__(self, encoder, device, jepa_dir, camera_keys, batch_size):
        self.encoder      = encoder
        self.device       = device
        self.jepa_dir     = jepa_dir
        self.camera_keys  = camera_keys
        self.batch_size   = batch_size
        # frame_count → list of (chunk_str, ep_id, cam_data)
        self._buckets: dict[int, list] = defaultdict(list)

    def push(self, chunk_str: str, ep_id: int,
             cam_data: dict[str, tuple[torch.Tensor, np.ndarray]]) -> None:
        """Add an episode. Flush the bucket automatically when full."""
        # All cameras should have the same frame count; use first camera's count
        n_frames = next(iter(cam_data.values()))[0].shape[2]
        self._buckets[n_frames].append((chunk_str, ep_id, cam_data))
        if len(self._buckets[n_frames]) >= self.batch_size:
            self._flush(n_frames)

    def flush_all(self) -> None:
        """Flush every remaining partial bucket."""
        for n_frames in list(self._buckets.keys()):
            self._flush(n_frames)

    # ── internal ──────────────────────────────────────────────────────────────

    def _flush(self, n_frames: int) -> None:
        items = self._buckets.pop(n_frames)

        # Encode camera-by-camera (each camera is an independent video stream)
        # features_by_cam[cam_key] = list of [T_jepa, H', W', D] cpu bfloat16
        features_by_cam: dict[str, list[torch.Tensor]] = {}

        for cam_key in self.camera_keys:
            tensors = [item[2][cam_key][0] for item in items
                       if cam_key in item[2]]
            if not tensors:
                features_by_cam[cam_key] = []
                continue
            features_by_cam[cam_key] = encode_batch(
                self.encoder, tensors, self.device,
            )

        # Save one file per episode.
        # cam_feat_cursors tracks how many episodes we've already assigned
        # for each camera — must live OUTSIDE the items loop so it increments
        # monotonically across the whole batch (not reset per episode).
        cam_feat_cursors: dict[str, int] = defaultdict(int)

        for chunk_str, ep_id, cam_data in items:
            out_dir = self.jepa_dir / f'chunk-{chunk_str}'
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f'episode_{ep_id:06d}.pt'

            save_dict: dict = {}

            # frame_ids — shared timeline across all cameras for this episode
            for cam_key in self.camera_keys:
                if cam_key in cam_data:
                    save_dict['frame_ids'] = cam_data[cam_key][1]
                    break

            # Per-camera features: pick the i-th result from encode_batch,
            # where i = number of previous items that had this camera key.
            for cam_key in self.camera_keys:
                if cam_key not in cam_data:
                    continue
                feat_list = features_by_cam.get(cam_key, [])
                idx       = cam_feat_cursors[cam_key]
                if idx < len(feat_list):
                    save_dict[cam_key] = feat_list[idx]
                cam_feat_cursors[cam_key] += 1

            torch.save(save_dict, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset discovery
# ─────────────────────────────────────────────────────────────────────────────

def find_dataset_roots(root: Path, max_depth: int = 4) -> list[Path]:
    """
    Return the list of dataset roots under `root`.

    A dataset root is any directory that directly contains a `videos/`
    subdirectory.  If `root` itself is a dataset root it is returned as a
    single-element list.  Otherwise the tree is searched up to `max_depth`
    levels deep and all matching directories are returned sorted.
    """
    if (root / 'videos').is_dir():
        return [root]

    found: list[Path] = []

    def _scan(path: Path, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            for child in sorted(path.iterdir()):
                if not child.is_dir():
                    continue
                if (child / 'videos').is_dir():
                    found.append(child)
                else:
                    _scan(child, depth + 1)
        except PermissionError:
            pass

    _scan(root, 1)
    return found


# ─────────────────────────────────────────────────────────────────────────────
# Per-dataset processing
# ─────────────────────────────────────────────────────────────────────────────

def process_dataset(
    dataset_root:    Path,
    encoder:         torch.nn.Module,
    device:          torch.device,
    rank:            int,
    local_rank:      int,
    world_size:      int,
    cam_resolutions: dict[str, tuple[int, int]],
    default_height:  int,
    default_width:   int,
    target_fps:      float,
    batch_size:      int,
    chunk_ids:       list[str] | None,
    camera_keys:     list[str] | None,
    skip_existing:   bool,
    ds_index:        int,
    ds_total:        int,
) -> tuple[int, int, int]:
    """
    Extract JEPA features for one dataset root.  Returns (processed, skipped, errors).
    """
    videos_dir = dataset_root / 'videos'
    jepa_dir   = dataset_root / 'jepa'
    jepa_dir.mkdir(parents=True, exist_ok=True)

    # ── chunk dirs ────────────────────────────────────────────────────────────
    if chunk_ids:
        chunk_dirs = [videos_dir / f'chunk-{cid}'
                      for cid in chunk_ids
                      if (videos_dir / f'chunk-{cid}').exists()]
    else:
        chunk_dirs = sorted(
            d for d in videos_dir.iterdir()
            if d.is_dir() and d.name.startswith('chunk-')
        )

    if not chunk_dirs:
        if rank == 0:
            print(f'  [skip] no chunk dirs found under {videos_dir}')
        return 0, 0, 0

    # ── camera key detection ──────────────────────────────────────────────────
    if camera_keys:
        cam_keys = camera_keys
    else:
        cam_key_set: set[str] = set()
        for chunk_dir in chunk_dirs:
            for sub in chunk_dir.iterdir():
                if sub.is_dir() and any(sub.glob('episode_*.mp4')):
                    cam_key_set.add(sub.name)
        cam_keys = sorted(cam_key_set)
        if not cam_keys:
            if rank == 0:
                print(f'  [skip] no camera dirs with episode_*.mp4 found under {videos_dir}')
            return 0, 0, 0

    if rank == 0:
        print(f'  cameras : {cam_keys}')

    # ── build job list and shard across ranks ─────────────────────────────────
    all_jobs: list[tuple[str, int]] = []
    for chunk_dir in chunk_dirs:
        chunk_str   = chunk_dir.name[len('chunk-'):]
        episode_ids: set[int] = set()
        for cam_key in cam_keys:
            cam_dir = chunk_dir / cam_key
            if not cam_dir.is_dir():
                continue
            for mp4 in cam_dir.glob('episode_*.mp4'):
                episode_ids.add(int(mp4.stem[len('episode_'):]))
        for ep_id in sorted(episode_ids):
            all_jobs.append((chunk_str, ep_id))

    my_jobs = all_jobs[rank::world_size]

    if rank == 0:
        print(f'  episodes total={len(all_jobs)} | this rank={len(my_jobs)}')

    # ── episode loop ──────────────────────────────────────────────────────────
    accumulator = EpisodeBatchAccumulator(
        encoder     = encoder,
        device      = device,
        jepa_dir    = jepa_dir,
        camera_keys = cam_keys,
        batch_size  = batch_size,
    )

    pbar = tqdm(
        my_jobs,
        desc         = f'Rank {rank} [{ds_index}/{ds_total}]',
        position     = local_rank,
        leave        = True,
        dynamic_ncols = True,
    )

    skipped = errors = 0

    for chunk_str, ep_id in pbar:
        out_path = jepa_dir / f'chunk-{chunk_str}' / f'episode_{ep_id:06d}.pt'

        if skip_existing and out_path.exists():
            skipped += 1
            pbar.set_postfix(skip=skipped, err=errors)
            continue

        cam_data: dict[str, tuple[torch.Tensor, np.ndarray]] = {}
        episode_ok = True

        for cam_key in cam_keys:
            mp4_path = (videos_dir / f'chunk-{chunk_str}'
                        / cam_key / f'episode_{ep_id:06d}.mp4')
            if not mp4_path.exists():
                tqdm.write(f'[Rank {rank}] Missing: {mp4_path}')
                episode_ok = False
                break

            h, w = cam_resolutions.get(
                cam_key, _DEFAULT_CAM_RESOLUTIONS.get(cam_key, (default_height, default_width))
            )
            frames, fids = read_frames(str(mp4_path), target_fps, h, w)

            if len(frames) < 2:
                tqdm.write(f'[Rank {rank}] Too few frames ({len(frames)}): {mp4_path}')
                episode_ok = False
                break

            # Prepend a duplicate of frame 0 so JEPA token 0 = [f0, f0], then
            # tokens 2j-1 and 2j exactly cover the 4-frame window of VAE latent j.
            # 4k+1 frames + 1 prepend = 4k+2, always even — no drop needed.
            frames_jepa = np.concatenate([frames[0:1], frames], axis=0)

            cam_data[cam_key] = (frames_to_tensor(frames_jepa), fids)

        if not episode_ok:
            errors += 1
            pbar.set_postfix(skip=skipped, err=errors)
            continue

        accumulator.push(chunk_str, ep_id, cam_data)
        pbar.set_postfix(chunk=chunk_str, ep=f'{ep_id:06d}',
                         skip=skipped, err=errors)

    accumulator.flush_all()
    pbar.close()

    processed = len(my_jobs) - skipped - errors
    return processed, skipped, errors


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Extract V-JEPA 2.1 ViT-Gigantic features from robot videos. '
                    '--dataset-root may point to a single dataset (contains videos/) '
                    'or a parent directory; in the latter case all datasets found '
                    'within are processed with the model loaded only once.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--dataset-root',  required=True,
                        help='Single dataset root (contains videos/) OR a parent '
                             'directory whose subtree contains multiple such roots.')
    parser.add_argument('--checkpoint',    default="/liujinxin/weights/vjepa2.1/vjepa2_1_vitG_384.pt",
                        help='Path to vjepa2.1 .pt checkpoint file')
    parser.add_argument('--camera-keys',   nargs='+', default=None,
                        help='Camera sub-directory names under videos/chunk-NNN/. '
                             'Auto-detected per dataset when omitted.')
    parser.add_argument('--target-fps',    type=float, default=12.5,
                        help='Frame sampling rate — MUST match VAE latent extraction')
    parser.add_argument('--height',        type=int,   default=256,
                        help='Default extraction height for cameras not listed in '
                             '--cam-resolutions or _DEFAULT_CAM_RESOLUTIONS.')
    parser.add_argument('--width',         type=int,   default=256,
                        help='Default extraction width (same fallback as --height).')
    parser.add_argument('--cam-resolutions', nargs='*', default=None,
                        metavar='CAM_KEY:HxW',
                        help='Per-camera resolution overrides in CAM_KEY:HxW format, '
                             'e.g. observation.images.cam_high:256x320 '
                             'observation.images.cam_left_wrist:128x160. '
                             'Overrides _DEFAULT_CAM_RESOLUTIONS for listed cameras. '
                             'Cameras absent from both use --height/--width.')
    parser.add_argument('--batch-size',    type=int,   default=1,
                        help='Episodes per JEPA forward pass (same frame-count only)')
    parser.add_argument('--chunk-ids',     nargs='+',  default=None,
                        help='Restrict to specific chunk IDs, e.g. 000 001 '
                             '(applied to every dataset)')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip episodes whose jepa/chunk-NNN/episode_NNNNNN.pt '
                             'already exists — safe for resuming interrupted runs')
    parser.add_argument('--local-rank',    type=int,   default=0,
                        help='Overridden automatically by torchrun')
    args = parser.parse_args()

    # ── parse per-camera resolutions ─────────────────────────────────────────
    cli_cam_resolutions: dict[str, tuple[int, int]] = {}
    if args.cam_resolutions:
        for spec in args.cam_resolutions:
            try:
                cam_key, res = spec.rsplit(':', 1)
                h_str, w_str = res.lower().split('x')
                cli_cam_resolutions[cam_key] = (int(h_str), int(w_str))
            except (ValueError, AttributeError):
                raise ValueError(
                    f'Invalid --cam-resolutions entry {spec!r}. '
                    f'Expected format: cam_key:HxW  '
                    f'(e.g. observation.images.cam_high:256x320)'
                )

    # ── distributed setup ─────────────────────────────────────────────────────
    rank       = int(os.environ.get('RANK',       args.local_rank))
    local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    device     = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)

    # ── discover datasets ─────────────────────────────────────────────────────
    root     = Path(args.dataset_root)
    datasets = find_dataset_roots(root)

    if not datasets:
        raise RuntimeError(
            f'No dataset roots (directories containing videos/) found under {root}.'
        )

    if rank == 0:
        print(f'[extract_jepa_features] world_size={world_size}')
        print(f'  root     : {root}')
        print(f'  datasets : {len(datasets)} found')
        for ds in datasets:
            print(f'    {ds}')
        print(f'  fps={args.target_fps}  default_res={args.height}×{args.width}  '
              f'batch_size={args.batch_size}  skip_existing={args.skip_existing}')
        _eff = {**_DEFAULT_CAM_RESOLUTIONS, **cli_cam_resolutions}
        if _eff:
            print('  per-camera resolutions (defaults + CLI overrides):')
            for _k, (_h, _w) in sorted(_eff.items()):
                _src = 'cli' if _k in cli_cam_resolutions else 'default'
                print(f'    [{_src}] {_k}: {_h}×{_w}')

    # ── load model once ───────────────────────────────────────────────────────
    if rank == 0:
        print(f'\nLoading JEPA encoder from {args.checkpoint} ...')
    encoder = build_jepa_encoder(args.checkpoint, device)
    if rank == 0:
        n_params = sum(p.numel() for p in encoder.parameters()) / 1e9
        print(f'  Loaded. Parameters: {n_params:.2f}B\n')

    # ── iterate over datasets ─────────────────────────────────────────────────
    total_processed = total_skipped = total_errors = 0

    for ds_idx, dataset_root in enumerate(datasets, start=1):
        if rank == 0:
            print(f'── Dataset [{ds_idx}/{len(datasets)}]: {dataset_root}')

        processed, skipped, errors = process_dataset(
            dataset_root    = dataset_root,
            encoder         = encoder,
            device          = device,
            rank            = rank,
            local_rank      = local_rank,
            world_size      = world_size,
            cam_resolutions = cli_cam_resolutions,
            default_height  = args.height,
            default_width   = args.width,
            target_fps      = args.target_fps,
            batch_size      = args.batch_size,
            chunk_ids       = args.chunk_ids,
            camera_keys     = args.camera_keys,
            skip_existing   = args.skip_existing,
            ds_index        = ds_idx,
            ds_total        = len(datasets),
        )

        total_processed += processed
        total_skipped   += skipped
        total_errors    += errors

        if rank == 0:
            print(f'  → processed={processed} skipped={skipped} errors={errors}\n')

    if rank == 0:
        print('════════════════════════════════════════')
        print(f'All {len(datasets)} dataset(s) complete.')
        print(f'  Processed : {total_processed}')
        print(f'  Skipped   : {total_skipped}')
        print(f'  Errors    : {total_errors}')
        print('════════════════════════════════════════')


if __name__ == '__main__':
    main()
