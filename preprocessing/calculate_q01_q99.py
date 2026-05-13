import argparse
import glob
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan_va.configs import VA_CONFIGS

# DATASET_DIR = "/liujinxin/code/lhc/wy/wms/lingbot-va/datasets/robochallenge/"
DATASET_DIR = "/liujinxin/code/lhc/wy/wms/lingbot-va/datasets/"

def parse_channel_ids(value: str) -> List[int]:
    # Accept formats like: "0,1,2" or "0 1 2"
    tokens = value.replace(",", " ").split()
    return [int(v) for v in tokens]


def build_inverse_ids(action_dim: int, used_action_channel_ids: List[int]) -> List[int]:
    inverse_ids = [len(used_action_channel_ids)] * action_dim
    for i, j in enumerate(used_action_channel_ids):
        if j < 0 or j >= action_dim:
            raise ValueError(f"Channel id {j} is out of range for action_dim={action_dim}")
        inverse_ids[j] = i
    return inverse_ids


def resolve_inverse_ids_from_config(config_name: str):
    if config_name not in VA_CONFIGS:
        raise KeyError(f"Unknown config_name='{config_name}'. Available: {list(VA_CONFIGS.keys())}")
    cfg = VA_CONFIGS[config_name]
    if not hasattr(cfg, "action_dim"):
        raise ValueError(f"Config '{config_name}' has no action_dim")
    if not hasattr(cfg, "inverse_used_action_channel_ids"):
        raise ValueError(f"Config '{config_name}' has no inverse_used_action_channel_ids")

    action_dim = int(cfg.action_dim)
    inverse_ids = list(cfg.inverse_used_action_channel_ids)
    if len(inverse_ids) != action_dim:
        raise ValueError(
            f"Config '{config_name}' mismatch: len(inverse_used_action_channel_ids)={len(inverse_ids)} "
            f"!= action_dim={action_dim}"
        )
    slicing_ids = list(cfg.action_slicing_ids) if hasattr(cfg, "action_slicing_ids") else None
    pad_to_dim  = int(cfg.action_pad_to_dim)   if hasattr(cfg, "action_pad_to_dim")   else None
    return action_dim, inverse_ids, slicing_ids, pad_to_dim


def to_2d_float_array(values: np.ndarray) -> np.ndarray:
    # values is typically an object array of per-row vectors
    if len(values) == 0:
        return np.empty((0, 0), dtype=np.float32)
    first = values[0]
    if isinstance(first, np.ndarray):
        data = np.stack(values)
    elif isinstance(first, (list, tuple)):
        data = np.asarray(values)
    else:
        raise TypeError(f"Unsupported action element type: {type(first)}")
    return data.reshape(len(values), -1).astype(np.float32)


def align_actions_like_training(
    actions: np.ndarray,
    inverse_ids: List[int],
) -> np.ndarray:
    """
    Mimic channel remap logic used in training dataset:
    1) pad one trailing zero channel
    2) index with inverse_used_action_channel_ids (length = action_dim)
    """
    actions_padded = np.pad(actions, ((0, 0), (0, 1)), mode="constant", constant_values=0)
    if actions_padded.shape[1] <= max(inverse_ids):
        raise ValueError(
            f"actions_padded has width {actions_padded.shape[1]}, but max inverse id is {max(inverse_ids)}. "
            "This usually means inverse_used_action_channel_ids does not match the source action layout."
        )
    return actions_padded[:, inverse_ids]


def compute_q01_q99_parquet(
    dataset_root: str,
    key: str = "action",
    sample_ratio: float = 1.0,
    seed: int = 42,
    inverse_ids: Optional[List[int]] = None,
    slicing_ids: Optional[List[int]] = None,
    pad_to_dim: Optional[int] = None,
):
    parquet_pattern = os.path.join(dataset_root, "**", "data", "**", "episode_*.parquet")
    print("finding files under ", parquet_pattern)
    files = sorted(glob.glob(parquet_pattern, recursive=True))
    if not files:
        raise FileNotFoundError(f"No parquet files found under: {parquet_pattern}")

    rng = np.random.default_rng(seed)
    all_data = []

    for fpath in tqdm(files, desc="Reading parquet"):
        df = pd.read_parquet(fpath, columns=[key])
        if key not in df.columns:
            continue

        data = to_2d_float_array(df[key].values)
        if data.size == 0:
            continue

        if sample_ratio < 1.0:
            n = max(1, int(len(data) * sample_ratio))
            idx = rng.choice(len(data), n, replace=False)
            data = data[idx]

        # Pad to common width before slicing so datasets that lack certain dims
        # (e.g. mobile chassis dims) are zero-filled in those slots, matching the
        # training-time behaviour in _action_post_process.
        if pad_to_dim is not None and data.shape[1] < pad_to_dim:
            pad_width = pad_to_dim - data.shape[1]
            data = np.pad(data, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)

        if slicing_ids is not None:
            if data.shape[1] < max(slicing_ids) + 1:
                print(f"WARNING: skipping {fpath} — action dim {data.shape[1]} < required {max(slicing_ids) + 1}")
                continue
            data = data[:, slicing_ids]   # (T, D_hf) → (T, n_used)

        if inverse_ids is not None:
            data = align_actions_like_training(
                data,
                inverse_ids=inverse_ids,
            )

        all_data.append(data)

    if not all_data:
        raise RuntimeError(f"No valid '{key}' data found in parquet files under: {dataset_root}")

    all_data = np.concatenate(all_data, axis=0)
    q01 = np.quantile(all_data, 0.01, axis=0)
    q99 = np.quantile(all_data, 0.99, axis=0)
    return q01, q99, all_data.shape


def format_float_list(values: np.ndarray, precision: int = 8) -> str:
    return "[" + ", ".join(f"{float(v):.{precision}g}" for v in values.tolist()) + "]"


def main():
    parser = argparse.ArgumentParser(
        description="Compute q01/q99 from LeRobot parquet action field and print paste-ready norm_stat."
    )
    parser.add_argument(
        "--repo-name",
        type=str,
        required=True,
        help="repo name under the huggingface dir",
    )
    parser.add_argument(
        "--key",
        type=str,
        default="action",
        help="Column key in parquet to compute quantiles from.",
    )
    parser.add_argument(
        "--sample-ratio",
        type=float,
        default=1.0,
        help="Optional subsampling ratio in (0, 1].",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when sample_ratio < 1.",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default=None,
        help="Config name in wan_va.configs. If set, action remap uses config.action_dim + config.inverse_used_action_channel_ids.",
    )
    parser.add_argument(
        "--used-action-channel-ids",
        type=str,
        default=None,
        help="Manual mode only: comma/space separated standard action channel ids used by source action layout.",
    )
    parser.add_argument(
        "--action-dim",
        type=int,
        default=None,
        help="Manual mode only: target action_dim for building inverse ids from --used-action-channel-ids.",
    )
    parser.add_argument(
        "--action-slicing-ids",
        type=str,
        default=None,
        help="Manual mode only: comma/space separated column indices to select from the raw HF action "
             "before inverse mapping (mirrors config.action_slicing_ids). Read from config automatically "
             "when --config-name is set.",
    )
    parser.add_argument(
        "--action-pad-to-dim",
        type=int,
        default=None,
        help="Manual mode only: pad HF action to this width before slicing so that datasets with fewer "
             "dims (e.g. non-mobile tasks) share the same slicing config. Read from config automatically "
             "when --config-name is set.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=8,
        help="Number formatting precision for output.",
    )
    args = parser.parse_args()

    if not (0 < args.sample_ratio <= 1.0):
        raise ValueError(f"sample_ratio must be in (0, 1], got {args.sample_ratio}")

    inverse_ids = None
    slicing_ids = None
    pad_to_dim  = None
    effective_action_dim = None

    if args.config_name is not None:
        effective_action_dim, inverse_ids, slicing_ids, pad_to_dim = resolve_inverse_ids_from_config(args.config_name)
    elif args.used_action_channel_ids is not None:
        used_ids = parse_channel_ids(args.used_action_channel_ids)
        if args.action_dim is None:
            raise ValueError("--action-dim is required when --used-action-channel-ids is provided")
        effective_action_dim = int(args.action_dim)
        inverse_ids = build_inverse_ids(effective_action_dim, used_ids)
        if args.action_slicing_ids is not None:
            slicing_ids = parse_channel_ids(args.action_slicing_ids)
        if args.action_pad_to_dim is not None:
            pad_to_dim = args.action_pad_to_dim

    q01, q99, shape = compute_q01_q99_parquet(
        dataset_root=DATASET_DIR+args.repo_name,
        key=args.key,
        sample_ratio=args.sample_ratio,
        seed=args.seed,
        inverse_ids=inverse_ids,
        slicing_ids=slicing_ids,
        pad_to_dim=pad_to_dim,
    )

    print(f"Collected samples shape: {shape}")
    if args.config_name is not None:
        print(f"Mode: aligned by config '{args.config_name}' (action_dim={effective_action_dim})")
    elif inverse_ids is not None:
        print(f"Mode: aligned by manual inverse mapping (action_dim={effective_action_dim})")
    else:
        print("Mode: raw parquet action (no channel remap)")
    print(f"q01 dim: {len(q01)}")
    print(f"q99 dim: {len(q99)}")
    print("\n# Paste this block into your train config:")
    print("norm_stat = {")
    print(f'    "q01": {format_float_list(q01, precision=args.precision)},')
    print(f'    "q99": {format_float_list(q99, precision=args.precision)},')
    print("}")


if __name__ == "__main__":
    main()
