#!/usr/bin/env python3
# Dependencies: matplotlib, tqdm
#
# Summary:
# - Reads multiple JSONL training logs and plots all loss curves in one figure.
# - Uses `iteration` for x and `demo_sample_action_l1_loss` for y by default.
#
# Usage:
#   python plot_two_loss_curves.py \
#     --logs /path/to/run_a/loss.jsonl /path/to/run_b/loss.jsonl /path/to/run_c/loss.jsonl \
#     --output /path/to/compare.png

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Tuple

import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


X_KEY_DEFAULT = "step"
Y_KEY_DEFAULT = "action_loss"
SMOOTH_WINDOW_DEFAULT = 5
SMOOTH_MODE_DEFAULT = "centered"
LOGS_DEFAULT = [
    "/liujinxin/code/lhc/wy/wms/lingbot-va/checkpoints/rc_stack_color_blocks_perfectaligned/bs4lr2.5e-5/checkpoints/log.jsonl",
    "/liujinxin/code/lhc/wy/wms/lingbot-va/checkpoints/rc_stack_color_blocks_perfectaligned/bs16lr2.5e-5/checkpoints/log.jsonl",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot multiple loss-step curves from JSONL logs in one figure.")
    parser.add_argument(
        "--logs",
        nargs="+",
        default=LOGS_DEFAULT,
        help="Paths to JSONL log files. If omitted, uses LOGS_DEFAULT in script.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional legend labels. Must have same length as --logs if provided.",
    )
    parser.add_argument("--output", required=True, help="Output image path.")
    parser.add_argument("--x-key", default=X_KEY_DEFAULT, help="Key for x axis.")
    parser.add_argument("--y-key", default=Y_KEY_DEFAULT, help="Key for y axis.")
    parser.add_argument("--include-summary", action="store_true", help="Also include summary lines if y-key exists there.")
    parser.add_argument("--smooth-window", type=int, default=SMOOTH_WINDOW_DEFAULT, help="Optional moving-average window.")
    parser.add_argument(
        "--smooth-mode",
        default=SMOOTH_MODE_DEFAULT,
        choices=["centered", "causal"],
        help="Smoothing mode: centered (symmetric window) or causal (running average).",
    )
    parser.add_argument("--skip-first", type=int, default=0, help="Skip first N points from each run before plotting.")
    parser.add_argument("--title", default="", help="Optional plot title.")
    return parser.parse_args()


def _moving_average(values: List[float], window: int, mode: str) -> List[float]:
    if window <= 1 or window > len(values):
        return values
    if mode == "causal":
        out: List[float] = []
        running = 0.0
        for i, v in enumerate(values):
            running += v
            if i >= window:
                running -= values[i - window]
            out.append(running / min(i + 1, window))
        return out

    half = window // 2
    out = []
    n = len(values)
    for i in range(n):
        left = max(0, i - half)
        right = min(n, i + half + 1)
        segment = values[left:right]
        out.append(sum(segment) / float(len(segment)))
    return out


def _load_points(path: str, x_key: str, y_key: str, include_summary: bool) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []
    progress = tqdm(desc=f"Reading {os.path.basename(path)}", unit="line") if tqdm else None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if progress is not None:
                progress.update(1)
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if x_key not in obj:
                continue
            if y_key in obj:
                pass
            elif include_summary:
                matching = [k for k in obj.keys() if k.endswith("/" + y_key.split("/", 1)[-1])]
                if len(matching) == 1:
                    y_val = obj[matching[0]]
                    x_val = obj[x_key]
                    try:
                        points.append((float(x_val), float(y_val)))
                    except (TypeError, ValueError):
                        pass
                continue
            else:
                continue
            x_val = obj[x_key]
            y_val = obj[y_key]
            try:
                points.append((float(x_val), float(y_val)))
            except (TypeError, ValueError):
                continue
    if progress is not None:
        progress.close()
    points.sort(key=lambda p: p[0])
    return points


def _plot_one(points: List[Tuple[float, float]], label: str, smooth_window: int, smooth_mode: str, color: str) -> None:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ys_smooth = _moving_average(ys, smooth_window, smooth_mode) if smooth_window > 1 else ys

    plt.plot(xs, ys, color=color, linewidth=1.0, alpha=0.25, label=f"{label}-raw")
    if smooth_window > 1:
        plt.plot(xs, ys_smooth, color=color, linewidth=2.0, alpha=0.95, label=f"{label}-ma{smooth_window}-{smooth_mode}")


def main() -> int:
    args = _parse_args()
    if not args.logs:
        print("No logs provided.", file=sys.stderr)
        return 1

    for p in args.logs:
        if not os.path.isfile(p):
            print(f"log not found: {p}", file=sys.stderr)
            return 1

    labels: List[str]
    if args.labels is None:
        labels = [os.path.basename(os.path.dirname(p)) or os.path.basename(p) for p in args.logs]
    else:
        if len(args.labels) != len(args.logs):
            print(
                f"--labels length ({len(args.labels)}) must match --logs length ({len(args.logs)}).",
                file=sys.stderr,
            )
            return 1
        labels = list(args.labels)

    all_points: List[List[Tuple[float, float]]] = []
    for p in args.logs:
        pts = _load_points(p, args.x_key, args.y_key, args.include_summary)
        if args.skip_first > 0:
            pts = pts[args.skip_first:]
        all_points.append(pts)

    for idx, pts in enumerate(all_points):
        if not pts:
            print(f"No matching points found in log: {args.logs[idx]}", file=sys.stderr)
            return 1

    smooth_window = max(0, int(args.smooth_window))
    global_max_x = max(max(x for x, _ in pts) for pts in all_points)

    plt.figure(figsize=(10, 6))
    cmap = plt.get_cmap("tab10")
    for i, (pts, label) in enumerate(zip(all_points, labels)):
        _plot_one(pts, label, smooth_window, args.smooth_mode, color=cmap(i % 10))
    plt.xlabel(args.x_key)
    plt.ylabel(args.y_key)
    plt.title(args.title or f"{args.y_key} comparison vs {args.x_key} (max step={global_max_x:g})")
    plt.xlim(left=0, right=global_max_x)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=160)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
