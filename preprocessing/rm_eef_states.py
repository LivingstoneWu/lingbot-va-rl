"""
rm_eef_state.py

Remove the first N dimensions of the `action` column in every parquet file
of a converted LeRobot dataset.

This keeps CLI arguments aligned with zero_eef_action_dims.py:
    --repo-name
    --n-dims
    --dry-run
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

DATASET_ROOT = Path("/liujinxin/code/lhc/wy/wms/lingbot-va/datasets/robochallenge")


def _updated_schema_with_new_action_dim(schema: pa.Schema, new_action_dim: int) -> pa.Schema:
    fields = list(schema)
    action_idx = schema.get_field_index("action")
    if action_idx < 0:
        return schema

    old_field = schema.field(action_idx)
    if pa.types.is_fixed_size_list(old_field.type):
        value_type = old_field.type.value_type
    elif pa.types.is_list(old_field.type):
        value_type = old_field.type.value_type
    else:
        value_type = pa.float32()

    fields[action_idx] = pa.field(
        old_field.name,
        pa.list_(value_type, new_action_dim),
        nullable=old_field.nullable,
        metadata=old_field.metadata,
    )

    new_schema = pa.schema(fields, metadata=schema.metadata)

    # Keep HuggingFace metadata in sync, if present.
    md = dict(new_schema.metadata or {})
    hf_raw = md.get(b"huggingface")
    if hf_raw is not None:
        try:
            hf = json.loads(hf_raw.decode())
            info = hf.get("info", {})
            features = info.get("features", {})
            action_feat = features.get("action")
            if isinstance(action_feat, dict) and "length" in action_feat:
                action_feat["length"] = int(new_action_dim)
                md[b"huggingface"] = json.dumps(hf).encode()
                new_schema = new_schema.with_metadata(md)
        except Exception:
            pass

    return new_schema


def remove_eef_dims(repo_name: str, n_dims: int, dry_run: bool) -> None:
    dataset_path = DATASET_ROOT / repo_name
    data_dir = dataset_path / "data"

    parquet_files = sorted(data_dir.rglob("*.parquet"))
    if not parquet_files:
        print(f"No parquet files found under {data_dir}")
        return

    print(f"Dataset : {dataset_path}")
    print(f"Removing: action[:, 0:{n_dims}]")
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

        # Remove the target dimensions.
        action_full = action_full[:, actual_dims:]

        if dry_run:
            total_rows_patched += len(df)
            continue

        # Assign back as a Python list of 1-D arrays (required to avoid
        # pandas broadcasting errors on object-dtype columns)
        df["action"] = list(action_full)

        new_schema = _updated_schema_with_new_action_dim(table.schema, action_full.shape[1])
        patched_table = pa.Table.from_pandas(df, schema=new_schema, preserve_index=False)
        pq.write_table(patched_table, pq_path)
        total_rows_patched += len(df)

    print()
    if dry_run:
        print(f"[DRY RUN] Would have patched {total_rows_patched} rows across {len(parquet_files)} files.")
    else:
        print(f"Done. Patched {total_rows_patched} rows across {len(parquet_files)} files.")


def main():
    parser = argparse.ArgumentParser(description="Remove leading EEF action dimensions in a LeRobot dataset.")
    parser.add_argument(
        "--repo-name",
        required=True,
        help="Dataset folder name under DATASET_ROOT (e.g. rc_aloha_pencil_case)",
    )
    parser.add_argument(
        "--n-dims",
        type=int,
        default=14,
        help="Number of leading action dimensions to remove (default: 14 for bimanual EEF)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without writing any files",
    )
    args = parser.parse_args()

    remove_eef_dims(
        repo_name=args.repo_name,
        n_dims=args.n_dims,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
