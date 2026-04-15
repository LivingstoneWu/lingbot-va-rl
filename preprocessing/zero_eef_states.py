"""
zero_eef_action_dims.py

Zero out the first N dimensions of the `action` column in every parquet file
of a converted LeRobot dataset.

Motivation: EEF poses (dims 0–13 for bimanual, 0–6 for single-arm) are not
available at inference time, so we remove them from the training data so the
model never learns to depend on them for KV-cache conditioning.

Usage:
    python zero_eef_action_dims.py --repo-name rc_aloha_pencil_case --n-dims 14
    python zero_eef_action_dims.py --repo-name rc_set_the_plates   --n-dims 7  --dry-run
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

DATASET_ROOT = Path("/liujinxin/code/lhc/wy/wms/lingbot-va/datasets/robochallenge")


def zero_eef_dims(repo_name: str, n_dims: int, dry_run: bool) -> None:
    dataset_path = DATASET_ROOT / repo_name
    data_dir = dataset_path / "data"

    parquet_files = sorted(data_dir.rglob("*.parquet"))
    if not parquet_files:
        print(f"No parquet files found under {data_dir}")
        return

    print(f"Dataset : {dataset_path}")
    print(f"Zeroing : action[:, 0:{n_dims}]")
    print(f"Files   : {len(parquet_files)}")
    print(f"Dry run : {dry_run}")
    print()

    total_rows_patched = 0

    for pq_path in tqdm(parquet_files, desc="Patching parquet files"):
        table = pq.read_table(pq_path)
        df = table.to_pandas()

        if "action" not in df.columns:
            print(f"  WARNING: 'action' column missing in {pq_path.name}, skipping")
            continue

        # Stack the action column into a 2-D numpy array (rows, action_dim)
        action_full = np.stack(df["action"].values)  # (N, action_dim)
        action_dim = action_full.shape[1]

        if n_dims > action_dim:
            print(
                f"  WARNING: n_dims={n_dims} > action_dim={action_dim} "
                f"in {pq_path.name}, clamping to {action_dim}"
            )
            actual_dims = action_dim
        else:
            actual_dims = n_dims

        # Zero out the target dimensions
        action_full[:, :actual_dims] = 0.0

        if dry_run:
            total_rows_patched += len(df)
            continue

        # Assign back as a Python list of 1-D arrays (required to avoid
        # pandas broadcasting errors on object-dtype columns)
        df["action"] = list(action_full)

        patched_table = pa.Table.from_pandas(df, schema=table.schema, preserve_index=False)
        pq.write_table(patched_table, pq_path)
        total_rows_patched += len(df)

    print()
    if dry_run:
        print(f"[DRY RUN] Would have patched {total_rows_patched} rows across {len(parquet_files)} files.")
    else:
        print(f"Done. Patched {total_rows_patched} rows across {len(parquet_files)} files.")


def main():
    parser = argparse.ArgumentParser(description="Zero out EEF action dimensions in a LeRobot dataset.")
    parser.add_argument(
        "--repo-name",
        required=True,
        help="Dataset folder name under DATASET_ROOT (e.g. rc_aloha_pencil_case)",
    )
    parser.add_argument(
        "--n-dims",
        type=int,
        default=14,
        help="Number of leading action dimensions to zero out (default: 14 for bimanual EEF)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without writing any files",
    )
    args = parser.parse_args()

    zero_eef_dims(
        repo_name=args.repo_name,
        n_dims=args.n_dims,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
