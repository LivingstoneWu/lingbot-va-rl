#!/usr/bin/env python3
# Dependencies: matplotlib, tqdm
#
# Summary:
# - Reads a JSONL training log and plots loss vs step.
# - Uses `iteration` for x and `train/loss` for y by default.
# - Ignores summary lines like `train@10/...` unless requested.
#
# Usage:
#   python plot_loss_curve.py --log /path/to/log.jsonl --output /path/to/loss.png

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

LOG_DEFAULT = "/liujinxin/code/lhc/wy/wms/cosmos-policy/checkpoints/aloha_rc/put_pen_into_pencil_case/cosmos_policy_from_ALOHA_checkpoint_put_pen_into_pencil_case_bs200/checkpoints/loss.jsonl"
OUTPUT_DEFAULT = "/liujinxin/code/lhc/wy/wms/cosmos-policy/checkpoints/aloha_rc/put_pen_into_pencil_case/cosmos_policy_from_ALOHA_checkpoint_put_pen_into_pencil_case_bs200/checkpoints/l1_loss.png"
X_KEY_DEFAULT = "step"
Y_KEY_DEFAULT = "action_loss"
SMOOTH_WINDOW_DEFAULT = 5
SMOOTH_MODE_DEFAULT = "centered"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot loss-step curve from a JSONL log.")
    parser.add_argument("--log", default=LOG_DEFAULT, help="Path to JSONL log file.")
    parser.add_argument("--output", default=OUTPUT_DEFAULT, help="Output image path.")
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
    parser.add_argument("--skip-first", type=int, default=0)
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

    # centered window smoothing with boundary clipping
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
    progress = tqdm(desc="Reading", unit="line") if tqdm else None
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


def main() -> int:
    args = _parse_args()
    if not os.path.isfile(args.log):
        print(f"log not found: {args.log}", file=sys.stderr)
        return 1

    points = _load_points(args.log, args.x_key, args.y_key, args.include_summary)
    if not points:
        print("No matching points found in log.", file=sys.stderr)
        return 1
    points = points[max(args.skip_first, 0):]

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    smooth_window = max(0, int(args.smooth_window))
    ys_smooth = _moving_average(ys, smooth_window, args.smooth_mode) if smooth_window > 1 else ys

    plt.figure(figsize=(10, 6))
    plt.plot(xs, ys, "r", linewidth=1.0, alpha=0.5, label="raw")
    if smooth_window > 1:
        plt.plot(xs, ys_smooth, linewidth=2.0, label=f"ma{smooth_window}-{args.smooth_mode}")
        plt.legend()
    plt.xlabel(args.x_key)
    plt.ylabel(args.y_key)
    plt.title(args.title or f"{args.y_key} vs {args.x_key}")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(args.output, dpi=160)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
