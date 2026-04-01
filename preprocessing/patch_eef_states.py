import os

os.environ['HF_LEROBOT_HOME'] = '/liujinxin/code/lhc/wy/wms/lingbot-va/datasets/robochallenge'

import json
import glob
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import tyro

HF_LEROBOT_HOME = Path(os.environ['HF_LEROBOT_HOME'])
DATA_DIR = "/liujinxin/dataset/robochallenge/"

# The correct key after the fix, and how many dims it covers
EEF_KEY = 'ee_positions'
EEF_DIM = 7  # dims 0–6 of observation.state and action


def read_jsonl(path: str) -> List[dict]:
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def get_valid_source_episodes(raw_dataset_path: str) -> List[Tuple[str, int]]:
    """
    Returns (states_path, n_frames) for every source episode that has enough
    data to have been included in the converted dataset (i.e. states_len >= 2).
    Sorted in the same order the conversion script processes them.
    """
    episode_dirs = sorted(glob.glob(os.path.join(raw_dataset_path, 'data', 'episode_*')))
    valid = []
    for ep_dir in episode_dirs:
        states_path = os.path.join(ep_dir, 'states', 'states.jsonl')
        if not os.path.exists(states_path):
            continue
        data = read_jsonl(states_path)
        if len(data) >= 2:
            valid.append((states_path, len(data) - 1))  # n_frames = states_len - 1
    return valid


def extract_eef(states_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Re-reads a states.jsonl and returns corrected eef arrays using the fixed key.

    Returns:
        eef_states:  (N, EEF_DIM) — observation eef at each timestep
        eef_actions: (N, EEF_DIM) — action eef at each timestep (next state's eef)
    """
    data = read_jsonl(states_path)
    n = len(data) - 1
    eef_states = np.zeros((n, EEF_DIM), dtype=np.float32)
    eef_actions = np.zeros((n, EEF_DIM), dtype=np.float32)
    for i in range(n):
        eef_states[i] = np.array(data[i].get(EEF_KEY, np.zeros(EEF_DIM)), dtype=np.float32)
        eef_actions[i] = np.array(data[i + 1].get(EEF_KEY, np.zeros(EEF_DIM)), dtype=np.float32)
    return eef_states, eef_actions


def patch_parquet(
    parquet_path: str,
    episode_eef_map: Dict[int, Tuple[np.ndarray, np.ndarray]],
) -> Tuple[int, List[int]]:
    """
    Patches observation.state[:, 0:EEF_DIM] and action[:, 0:EEF_DIM] in one
    parquet file for the given episodes.

    Returns:
        (rows_patched, skipped_episode_indices)
    """
    table = pq.read_table(parquet_path)
    df = table.to_pandas()

    rows_patched = 0
    skipped = []

    for ep_idx, (eef_states, eef_actions) in episode_eef_map.items():
        mask = df['episode_index'] == ep_idx
        n_rows = mask.sum()

        if n_rows == 0:
            continue

        if n_rows != len(eef_states):
            print(f"  ⚠ episode {ep_idx}: converted={n_rows} frames, "
                  f"source={len(eef_states)} frames — frame count mismatch, skipping")
            skipped.append(ep_idx)
            continue

        # Stack the full (N, 14) arrays, patch dims 0:EEF_DIM, write back as rows of lists
        obs = np.stack(df.loc[mask, 'observation.state'].values)   # (N, 14)
        act = np.stack(df.loc[mask, 'action'].values)              # (N, 14)

        obs[:, :EEF_DIM] = eef_states
        act[:, :EEF_DIM] = eef_actions

        df.loc[mask, 'observation.state'] = [row for row in obs]
        df.loc[mask, 'action'] = [row for row in act]

        rows_patched += n_rows

    # Write back preserving the original pyarrow schema
    patched_table = pa.Table.from_pandas(df, schema=table.schema, preserve_index=False)
    pq.write_table(patched_table, parquet_path)

    return rows_patched, skipped


def main(
    repo_name: str,
    raw_dataset_name: str,
    data_dir: str = DATA_DIR,
    dry_run: bool = False,
):
    """
    Patches eef (end-effector) values in a converted LeRobot dataset.

    Only observation.state[:, 0:7] and action[:, 0:7] are updated — joints,
    gripper, and all video files are left completely untouched.

    The episode-to-source mapping is reconstructed from sorted episode directory
    order (same order the conversion script used, since shuffle is disabled).
    Frame counts are verified before any file is written.

    Args:
        repo_name:         LeRobot dataset name under HF_LEROBOT_HOME
        raw_dataset_name:  Original raw dataset folder name under data_dir
        data_dir:          Root directory containing raw datasets
        dry_run:           Verify mapping and frame counts without writing anything
    """
    dataset_path = HF_LEROBOT_HOME / repo_name
    raw_dataset_path = os.path.join(data_dir, raw_dataset_name)

    print(f"Converted dataset : {dataset_path}")
    print(f"Raw dataset       : {raw_dataset_path}")
    print(f"EEF key           : '{EEF_KEY}' (dims 0–{EEF_DIM - 1})")
    print(f"Dry run           : {dry_run}")
    print()

    # ── 1. Source episodes ────────────────────────────────────────────────────
    source_episodes = get_valid_source_episodes(raw_dataset_path)
    print(f"Valid source episodes : {len(source_episodes)}")

    # ── 2. Converted dataset parquet files ────────────────────────────────────
    data_dir_path = dataset_path / 'data'
    parquet_files = sorted(data_dir_path.glob('train-*.parquet'))
    if not parquet_files:
        parquet_files = sorted(data_dir_path.glob('*.parquet'))
    if not parquet_files:
        print("ERROR: no parquet files found under", data_dir_path)
        return
    print(f"Parquet files        : {len(parquet_files)}")

    # ── 3. Episode frame counts from the converted dataset ────────────────────
    meta_frames = pd.concat([
        pd.read_parquet(f, columns=['episode_index', 'frame_index'])
        for f in parquet_files
    ])
    episode_frame_counts: Dict[int, int] = (
        meta_frames.groupby('episode_index').size().to_dict()
    )
    sorted_ep_indices = sorted(episode_frame_counts.keys())
    n_converted = len(sorted_ep_indices)
    print(f"Converted episodes   : {n_converted}")
    print()

    # ── 4. Verify counts match ────────────────────────────────────────────────
    if len(source_episodes) != n_converted:
        print(
            f"ERROR: {len(source_episodes)} valid source episodes but "
            f"{n_converted} converted episodes — counts don't match.\n"
            "Cannot safely reconstruct the mapping; aborting."
        )
        return

    # ── 5. Build episode_index → (eef_states, eef_actions) map ───────────────
    print("Verifying frame counts and building episode map...")
    episode_eef_map: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    abort = False

    for ep_idx, (source_path, source_n_frames) in zip(sorted_ep_indices, source_episodes):
        converted_n_frames = episode_frame_counts[ep_idx]
        ep_dir_name = Path(source_path).parts[-3]  # e.g. episode_0042

        if source_n_frames != converted_n_frames:
            print(
                f"  ✗ episode_index {ep_idx} ↔ {ep_dir_name}: "
                f"source={source_n_frames} frames, converted={converted_n_frames} frames — MISMATCH"
            )
            abort = True
        else:
            print(f"  ✓ episode_index {ep_idx} ↔ {ep_dir_name}: {source_n_frames} frames")
            eef_states, eef_actions = extract_eef(source_path)
            episode_eef_map[ep_idx] = (eef_states, eef_actions)

    if abort:
        print("\nFrame count mismatches detected — aborting to avoid writing incorrect data.")
        return

    if dry_run:
        print(f"\nDry run complete — {len(episode_eef_map)} episodes verified, nothing written.")
        return

    # ── 6. Patch parquet files ────────────────────────────────────────────────
    print(f"\nPatching {len(parquet_files)} parquet file(s)...")
    total_rows = 0
    all_skipped = []

    for pf in parquet_files:
        ep_indices_in_file = set(
            pd.read_parquet(pf, columns=['episode_index'])['episode_index'].unique()
        )
        relevant = {k: v for k, v in episode_eef_map.items() if k in ep_indices_in_file}
        if not relevant:
            continue

        print(f"  {pf.name} — {len(relevant)} episode(s)...", end=' ', flush=True)
        rows, skipped = patch_parquet(str(pf), relevant)
        print(f"{rows} rows patched" + (f", {len(skipped)} skipped" if skipped else ""))
        total_rows += rows
        all_skipped.extend(skipped)

    print(f"\nDone. {total_rows} rows patched across {len(parquet_files)} file(s).")
    if all_skipped:
        print(f"Skipped episode indices (frame count mismatch): {all_skipped}")


if __name__ == '__main__':
    tyro.cli(main)
