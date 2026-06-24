from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr


def find_jepa_roots(path: Path) -> list[Path]:
    if list((path / "jepa").glob("chunk-*/episode_*.pt")):
        return [path]
    roots = sorted(
        chunk_dir.parent.parent
        for chunk_dir in path.rglob("chunk-*")
        if chunk_dir.parent.name == "jepa"
        and list(chunk_dir.glob("episode_*.pt"))
    )
    deduped: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if root not in seen:
            seen.add(root)
            deduped.append(root)
    return deduped


def load_dense_episode_features(
    path: Path,
    camera_keys: list[str] | None = None,
) -> tuple[torch.Tensor, dict[str, tuple[int, ...]], dict[str, Any]]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    keys = camera_keys or sorted(k for k in data if k != "frame_ids")
    if not keys:
        raise ValueError(f"No camera feature tensors found in {path}")

    shapes: dict[str, tuple[int, ...]] = {}
    tensors: list[torch.Tensor] = []
    time_lengths: dict[str, int] = {}
    feature_dim = None
    for key in keys:
        if key not in data:
            raise KeyError(f"{path} is missing camera key {key!r}")
        tensor = data[key]
        if not torch.is_tensor(tensor):
            raise TypeError(f"{path}:{key} is not a tensor")
        if tensor.ndim < 2:
            raise ValueError(
                f"{path}:{key} must have at least [T,D], got {tuple(tensor.shape)}"
            )
        if feature_dim is None:
            feature_dim = int(tensor.shape[-1])
        elif tensor.shape[-1] != feature_dim:
            raise ValueError(
                f"{path}:{key} shape {tuple(tensor.shape)} is incompatible "
                f"with D={feature_dim}"
            )
        shapes[key] = tuple(tensor.shape)
        time_lengths[key] = int(tensor.shape[0])
        tensors.append(tensor)

    aligned_time_len = min(time_lengths.values())
    if aligned_time_len < 2:
        raise ValueError(f"{path} has fewer than 2 aligned time steps: {time_lengths}")
    alignment = {
        "time_lengths": time_lengths,
        "aligned_time_len": aligned_time_len,
        "time_aligned": len(set(time_lengths.values())) > 1,
    }
    flattened = [
        tensor[:aligned_time_len].reshape(aligned_time_len, -1, tensor.shape[-1])
        for tensor in tensors
    ]

    return torch.cat(flattened, dim=1).float(), shapes, alignment


def terminal_distances(features: torch.Tensor) -> np.ndarray:
    if features.ndim != 3:
        raise ValueError(f"features must be [T,N,D], got {tuple(features.shape)}")
    if features.shape[0] < 2:
        raise ValueError("Need at least two time steps to compute progress")
    normalized = F.normalize(features, dim=-1)
    goal = normalized[-1:]
    cosine = (normalized * goal).sum(dim=-1)
    return (1.0 - cosine).mean(dim=-1).cpu().numpy()


def trajectory_stats(dist: np.ndarray, delta: np.ndarray) -> dict[str, Any]:
    time = np.linspace(0.0, 1.0, len(dist))
    corr = spearmanr(time, dist).correlation
    return {
        "trajectory_length": int(len(dist)),
        "num_delta_steps": int(len(delta)),
        "mean_delta": float(np.mean(delta)),
        "median_delta": float(np.median(delta)),
        "fraction_positive": float(np.mean(delta > 0.0)),
        "fraction_negative": float(np.mean(delta < 0.0)),
        "fraction_abs_lt_1e-4": float(np.mean(np.abs(delta) < 1e-4)),
        "terminal_distance": float(dist[-1]),
        "initial_distance": float(dist[0]),
        "spearman_time_distance": None if np.isnan(corr) else float(corr),
    }


def interpolate_series(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    if len(values) == 1:
        return np.full_like(grid, float(values[0]), dtype=np.float64)
    x = np.linspace(0.0, 1.0, len(values))
    return np.interp(grid, x, values)


def plot_progress(
    trajectories: list[dict[str, Any]],
    aggregate_delta: np.ndarray,
    output_path: Path,
    task_name: str,
    seed: int,
    sample_count: int,
    grid_size: int,
) -> None:
    rng = random.Random(seed)
    sample = rng.sample(
        trajectories, k=min(sample_count, len(trajectories))
    )
    grid = np.linspace(0.0, 1.0, grid_size)
    interp_delta = np.stack(
        [interpolate_series(item["delta"], grid) for item in trajectories],
        axis=0,
    )
    median_delta = np.median(interp_delta, axis=0)
    interp_distance = np.stack(
        [interpolate_series(item["distance"], grid) for item in trajectories],
        axis=0,
    )
    median_distance = np.median(interp_distance, axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(20, 5), constrained_layout=True)
    fig.suptitle(task_name, fontsize=14)

    ax = axes[0]
    for item in sample:
        distance = item["distance"]
        x = np.linspace(0.0, 1.0, len(distance))
        ax.plot(x, distance, linewidth=0.8, alpha=0.25)
    ax.plot(grid, median_distance, color="black", linewidth=2.0, label="median")
    ax.axhline(0.0, color="red", linestyle="--", linewidth=1.0)
    ax.set_title("JEPA distance to oracle terminal goal")
    ax.set_xlabel("Normalized trajectory time")
    ax.set_ylabel("mean token cosine distance")
    ax.legend()

    ax = axes[1]
    for item in sample:
        delta = item["delta"]
        x = np.linspace(0.0, 1.0, len(delta))
        ax.plot(x, delta, linewidth=0.8, alpha=0.25)
    ax.plot(grid, median_delta, color="black", linewidth=2.0, label="median")
    ax.axhline(0.0, color="red", linestyle="--", linewidth=1.0)
    ax.set_title("Per-step JEPA progress to oracle terminal goal")
    ax.set_xlabel("Normalized trajectory time")
    ax.set_ylabel("delta = dist[t] - dist[t+1]")
    ax.legend()

    ax = axes[2]
    ax.hist(aggregate_delta, bins=80, color="#4C78A8", alpha=0.85)
    ax.axvline(0.0, color="red", linestyle="--", linewidth=1.2)
    frac_pos = np.mean(aggregate_delta > 0.0)
    frac_neg = np.mean(aggregate_delta < 0.0)
    ax.set_title(
        f"All deltas: pos={frac_pos:.3f}, neg={frac_neg:.3f}, "
        f"median={np.median(aggregate_delta):.3g}"
    )
    ax.set_xlabel("Per-step progress delta")
    ax.set_ylabel("Count")

    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose whether oracle terminal-state JEPA distance behaves like "
            "a dense progress signal."
        )
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--camera-keys", nargs="+", default=None)
    parser.add_argument("--grid-size", type=int, default=100)
    parser.add_argument("--sample-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--print-shapes", type=int, default=5)
    args = parser.parse_args()

    roots = find_jepa_roots(Path(args.dataset_root))
    if not roots:
        raise RuntimeError(f"No jepa/chunk-* directories found under {args.dataset_root}")
    if len(roots) > 1 and args.output_dir is None:
        raise ValueError(
            "Multiple dataset roots found; pass --output-dir explicitly"
        )
    root = roots[0]
    output_dir = Path(args.output_dir) if args.output_dir else root
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted((root / "jepa").glob("chunk-*/episode_*.pt"))
    if not files:
        raise RuntimeError(f"No JEPA episode files found under {root / 'jepa'}")

    trajectories: list[dict[str, Any]] = []
    all_delta: list[np.ndarray] = []
    terminal_distances_seen = []
    time_aligned_episodes: list[dict[str, Any]] = []

    for index, path in enumerate(files):
        features, shapes, alignment = load_dense_episode_features(path, args.camera_keys)
        if alignment["time_aligned"]:
            time_aligned_episodes.append(
                {
                    "path": str(path),
                    "time_lengths": alignment["time_lengths"],
                    "aligned_time_len": alignment["aligned_time_len"],
                }
            )
        if index < args.print_shapes:
            print(f"{path}: dense={tuple(features.shape)} cameras={shapes}")
            if alignment["time_aligned"]:
                print(
                    "  aligned camera time lengths "
                    f"{alignment['time_lengths']} -> {alignment['aligned_time_len']}"
                )
        dist = terminal_distances(features)
        delta = dist[:-1] - dist[1:]
        terminal_distances_seen.append(float(dist[-1]))
        if index < args.print_shapes:
            print(f"  terminal_distance={dist[-1]:.8g}")
        if not np.isclose(dist[-1], 0.0, atol=1e-5):
            raise ValueError(
                f"Final-state distance for {path} should be ~0, got {dist[-1]}"
            )
        stats = trajectory_stats(dist, delta)
        trajectories.append(
            {
                "path": str(path),
                "episode": path.stem,
                "distance": dist,
                "delta": delta,
                "stats": stats,
            }
        )
        all_delta.append(delta)

    aggregate_delta = np.concatenate(all_delta)
    aggregate_stats = {
        "num_trajectories": len(trajectories),
        "num_delta_steps": int(len(aggregate_delta)),
        "fraction_positive": float(np.mean(aggregate_delta > 0.0)),
        "fraction_negative": float(np.mean(aggregate_delta < 0.0)),
        "fraction_abs_lt_1e-4": float(np.mean(np.abs(aggregate_delta) < 1e-4)),
        "mean_delta": float(np.mean(aggregate_delta)),
        "median_delta": float(np.median(aggregate_delta)),
        "min_delta": float(np.min(aggregate_delta)),
        "max_delta": float(np.max(aggregate_delta)),
        "max_abs_terminal_distance": float(np.max(np.abs(terminal_distances_seen))),
        "num_time_aligned_episodes": len(time_aligned_episodes),
    }

    figure_path = output_dir / "jepa_oracle_goal_progress.png"
    stats_path = output_dir / "jepa_oracle_goal_progress_stats.json"
    plot_progress(
        trajectories,
        aggregate_delta,
        figure_path,
        task_name=root.name,
        seed=args.seed,
        sample_count=args.sample_count,
        grid_size=args.grid_size,
    )

    stats_json = {
        "dataset_root": str(root),
        "figure": str(figure_path),
        "aggregate": aggregate_stats,
        "time_aligned_episodes": time_aligned_episodes,
        "trajectories": [
            {
                "path": item["path"],
                "episode": item["episode"],
                **item["stats"],
            }
            for item in trajectories
        ],
    }
    stats_path.write_text(json.dumps(stats_json, indent=2) + "\n")

    print(json.dumps(aggregate_stats, indent=2))
    print(f"Saved figure: {figure_path}")
    print(f"Saved stats : {stats_path}")


if __name__ == "__main__":
    main()
