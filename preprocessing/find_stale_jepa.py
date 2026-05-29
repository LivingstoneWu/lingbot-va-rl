"""
find_stale_jepa.py — Scan a dataset tree for JEPA files extracted at the old
                     256×256 resolution (16×16 tokens per camera).

Any camera tensor whose spatial dims are (16, 16) is considered stale and
needs to be re-extracted at the correct native camera resolution.

Usage:
    python preprocessing/find_stale_jepa.py \\
        --root /path/to/datasets \\
        [--workers 8]   # parallel workers (default: CPU count)
        [--delete]      # also delete the stale files so re-extraction picks them up

Output:
    Prints stale file paths to stdout (one per line) and a summary to stderr.
    With --delete, removes each stale file after printing its path.
"""

import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm


# ── worker ────────────────────────────────────────────────────────────────────

def _check_file(pt_path: Path) -> tuple[Path, bool, str | None]:
    """
    Worker function: check one .pt file.
    Returns (path, is_stale, error_message_or_None).
    Runs in a child process — no shared state, no tqdm.
    """
    try:
        data = torch.load(pt_path, map_location='cpu', weights_only=False)
    except Exception as exc:
        return pt_path, False, str(exc)

    for key, val in data.items():
        if key == 'frame_ids':
            continue
        if not isinstance(val, torch.Tensor) or val.ndim != 4:
            continue
        # shape: [T_jepa, H_tokens, W_tokens, D]
        if val.shape[1] == 16 and val.shape[2] == 16:
            return pt_path, True, None

    return pt_path, False, None


# ── scanner ───────────────────────────────────────────────────────────────────

def scan(root: Path, delete: bool, workers: int) -> tuple[int, int, int]:
    """
    Recursively find all episode_*.pt files under jepa/ dirs and check them
    in parallel.  Returns (stale, errors, total).
    """
    print(f'Collecting file list under {root} ...', file=sys.stderr)
    jepa_files = sorted(root.rglob('jepa/chunk-*/episode_*.pt'))
    total = len(jepa_files)
    print(f'Found {total} JEPA file(s).  Scanning with {workers} worker(s) ...',
          file=sys.stderr)

    stale_paths: list[Path] = []
    errors = 0

    with mp.Pool(processes=workers) as pool:
        results = pool.imap_unordered(_check_file, jepa_files, chunksize=32)
        with tqdm(total=total, desc='Scanning', unit='file',
                  dynamic_ncols=True) as pbar:
            for pt_path, file_is_stale, err in results:
                pbar.update(1)
                if err is not None:
                    tqdm.write(f'[WARN] {pt_path}: {err}')
                    errors += 1
                elif file_is_stale:
                    stale_paths.append(pt_path)
                    tqdm.write(str(pt_path))   # stdout: one path per line

    # Print to stdout (for piping / wc -l) and optionally delete
    for pt_path in stale_paths:
        if delete:
            try:
                pt_path.unlink()
            except OSError as exc:
                print(f'[WARN] Could not delete {pt_path}: {exc}', file=sys.stderr)

    return len(stale_paths), errors, total


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Find (and optionally delete) stale 16×16 JEPA feature files.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--root', required=True,
                        help='Root directory to search (searched recursively).')
    parser.add_argument('--workers', type=int,
                        default=min(os.cpu_count() or 4, 16),
                        help='Number of parallel worker processes.')
    parser.add_argument('--delete', action='store_true',
                        help='Delete each stale file after printing its path. '
                             'Re-run the extraction script afterwards (without '
                             '--skip-existing) to regenerate at correct resolution.')
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f'ERROR: {root} does not exist.', file=sys.stderr)
        sys.exit(1)

    stale, errors, total = scan(root, args.delete, args.workers)

    action = 'deleted' if args.delete else 'found'
    print('════════════════════════════════════════', file=sys.stderr)
    print(f'Total scanned : {total}',          file=sys.stderr)
    print(f'Stale ({action}): {stale}',        file=sys.stderr)
    if errors:
        print(f'Errors        : {errors}',     file=sys.stderr)
    print('════════════════════════════════════════', file=sys.stderr)

    if stale and not args.delete:
        print('Re-run with --delete to remove them, then re-extract JEPA features.',
              file=sys.stderr)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
