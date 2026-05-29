#!/usr/bin/env python3
"""
Randomly sample RobotWin datasets from clean/aug roots and symlink both variants.

Example:
  python sample_robotwin_datasets.py \
    --clean-root /path/to/robotwin_clean \
    --aug-root /path/to/robotwin_aug \
    --output-root /path/to/output \
    --num-datasets 10 \
    --seed 42
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Set, Tuple


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample dataset names and symlink clean/aug pairs into output directory."
    )
    parser.add_argument("--clean-root", type=Path, required=True, help="Directory containing clean datasets.")
    parser.add_argument("--aug-root", type=Path, required=True, help="Directory containing augmented datasets.")
    parser.add_argument("--output-root", type=Path, required=True, help="Directory to place symlinks.")
    parser.add_argument("--num-datasets", type=int, required=True, help="How many dataset names to sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace existing links/files in output if names conflict.",
    )
    return parser.parse_args()


def _list_dataset_names(root: Path) -> Set[str]:
    if not root.exists():
        raise FileNotFoundError(f"Root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Root is not a directory: {root}")
    return {p.name for p in root.iterdir() if p.is_dir()}


def _prefix_before_first_hyphen(name: str) -> str:
    return name.split("-", 1)[0]


def _build_prefix_map(names: Set[str], side: str) -> Dict[str, str]:
    """
    Build prefix -> dataset_name map.
    Raises if one side has duplicate names sharing the same prefix.
    """
    out: Dict[str, str] = {}
    dup: Dict[str, List[str]] = {}
    for n in sorted(names):
        p = _prefix_before_first_hyphen(n)
        if p in out:
            dup.setdefault(p, [out[p]]).append(n)
        else:
            out[p] = n
    if dup:
        lines = [f"{k}: {v}" for k, v in sorted(dup.items())]
        raise ValueError(
            f"Found duplicate prefixes in {side} root; cannot do 1:1 matching by prefix.\n"
            + "\n".join(lines)
        )
    return out


def _safe_symlink(src: Path, dst: Path, replace: bool) -> None:
    if dst.exists() or dst.is_symlink():
        if not replace:
            raise FileExistsError(f"Destination already exists: {dst}")
        if dst.is_dir() and not dst.is_symlink():
            raise IsADirectoryError(
                f"Destination is a real directory, refusing to remove: {dst}"
            )
        dst.unlink()
    dst.symlink_to(src.resolve())


def main() -> int:
    args = _parse_args()

    clean_root = args.clean_root.resolve()
    aug_root = args.aug_root.resolve()
    output_root = args.output_root.resolve()

    clean_names = _list_dataset_names(clean_root)
    aug_names = _list_dataset_names(aug_root)
    clean_by_prefix = _build_prefix_map(clean_names, side="clean")
    aug_by_prefix = _build_prefix_map(aug_names, side="aug")
    common_prefixes: List[str] = sorted(set(clean_by_prefix.keys()) & set(aug_by_prefix.keys()))

    if not common_prefixes:
        raise RuntimeError("No matching dataset prefixes found between clean-root and aug-root.")

    if args.num_datasets <= 0:
        raise ValueError("--num-datasets must be > 0")
    if args.num_datasets > len(common_prefixes):
        raise ValueError(
            f"Requested {args.num_datasets} datasets, but only {len(common_prefixes)} matching prefixes found."
        )

    random.seed(args.seed)
    selected_prefixes = sorted(random.sample(common_prefixes, args.num_datasets))
    selected_pairs: List[Tuple[str, str, str]] = [
        (p, clean_by_prefix[p], aug_by_prefix[p]) for p in selected_prefixes
    ]

    output_root.mkdir(parents=True, exist_ok=True)

    for prefix, clean_name, aug_name in selected_pairs:
        clean_src = clean_root / clean_name
        aug_src = aug_root / aug_name

        clean_dst = output_root / f"{prefix}_clean"
        aug_dst = output_root / f"{prefix}_aug"

        _safe_symlink(clean_src, clean_dst, replace=args.replace)
        _safe_symlink(aug_src, aug_dst, replace=args.replace)

    print(f"Matching prefixes found: {len(common_prefixes)}")
    print(f"Sampled prefixes       : {len(selected_pairs)}")
    print(f"Output directory      : {output_root}")
    print("Sampled mappings:")
    for prefix, clean_name, aug_name in selected_pairs:
        print(f"  - {prefix}")
        print(f"      clean: {clean_name}")
        print(f"      aug  : {aug_name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
