import argparse
from pathlib import Path

import numpy as np
import torch


def convert_pt_to_npy(root_dir: Path, overwrite: bool = False) -> None:
    pt_files = sorted(root_dir.rglob("*.pt"))
    if not pt_files:
        print(f"No .pt files found under: {root_dir}")
        return

    converted = 0
    skipped = 0
    failed = 0

    for pt_path in pt_files:
        npy_path = pt_path.with_suffix(".npy")
        if npy_path.exists() and not overwrite:
            skipped += 1
            continue

        try:
            obj = torch.load(pt_path, map_location="cpu", weights_only=False)
            if not torch.is_tensor(obj):
                print(f"[SKIP non-tensor] {pt_path}")
                skipped += 1
                continue

            arr = obj.to(torch.float32).detach().cpu().numpy()
            np.save(npy_path, arr)
            converted += 1
        except Exception as exc:
            print(f"[FAIL] {pt_path} -> {exc}")
            failed += 1

    print(f"Done. converted={converted}, skipped={skipped}, failed={failed}, total={len(pt_files)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recursively convert tensor .pt files to .npy files in-place (same directory)."
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root directory to scan recursively for .pt files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .npy files.",
    )
    args = parser.parse_args()

    root_dir = Path(args.root).resolve()
    if not root_dir.exists():
        raise FileNotFoundError(f"Root directory not found: {root_dir}")
    if not root_dir.is_dir():
        raise NotADirectoryError(f"Root path is not a directory: {root_dir}")

    convert_pt_to_npy(root_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
