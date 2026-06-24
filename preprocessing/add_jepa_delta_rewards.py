"""
Add JEPA delta-distance dense rewards to LeRobot episodes.jsonl.

This script consumes features produced by preprocessing/extract_jepa_features.py
and writes a reward_config into each action_config entry:

    {
      "reward_config": {
        "reward_source": "jepa_delta_distance",
        "distance_metric": "cosine",
        "goal_episode_index": 12,
        "goal_selection": "self_success" | "closest_success_final",
        "latent_rewards": [...],
        "distance_to_goal": [...]
      }
    }

The critic loader later applies reward weights from its training config.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


REWARD_SOURCE = "jepa_delta_distance"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def find_jepa_file(dataset_root: Path, episode_index: int) -> Path:
    matches = sorted(
        (dataset_root / "jepa").glob(
            f"chunk-*/episode_{episode_index:06d}.pt"
        )
    )
    if not matches:
        raise FileNotFoundError(
            f"Missing JEPA features for episode {episode_index}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"Multiple JEPA feature files for episode {episode_index}: "
            f"{matches}"
        )
    return matches[0]


def find_latent_file(
    dataset_root: Path,
    episode_index: int,
    start_frame: int,
    end_frame: int,
) -> Path:
    matches = sorted(
        (dataset_root / "latents").glob(
            f"chunk-*/*/episode_{episode_index:06d}_"
            f"{start_frame}_{end_frame}.pth"
        )
    )
    if not matches:
        raise FileNotFoundError(
            "Missing latent file for "
            f"episode {episode_index} {start_frame}:{end_frame}"
        )
    return matches[0]


def segment_latent_frame_count(
    dataset_root: Path,
    episode_index: int,
    action_config: dict[str, Any],
) -> int:
    latent_file = find_latent_file(
        dataset_root,
        episode_index,
        int(action_config["start_frame"]),
        int(action_config["end_frame"]),
    )
    latent_data = torch.load(latent_file, map_location="cpu", weights_only=False)
    return int(latent_data["latent_num_frames"])


def align_camera_to_latents(
    features: torch.Tensor,
    latent_count: int,
) -> torch.Tensor:
    """Map JEPA tokens to VAE latent indices using the extraction convention."""
    if latent_count <= 0:
        raise ValueError("latent_count must be positive")
    if features.ndim != 4:
        raise ValueError(
            f"JEPA camera features must be [T,H,W,D], got {features.shape}"
        )
    aligned = torch.empty(
        latent_count,
        features.shape[1],
        features.shape[2],
        features.shape[3],
        dtype=torch.float32,
    )
    feats = features.float()
    aligned[0] = feats[0]
    if latent_count == 1:
        return aligned
    odd = feats[1::2]
    even = feats[2::2]
    pairs = min(len(odd), len(even), latent_count - 1)
    if pairs > 0:
        aligned[1 : 1 + pairs] = 0.5 * (odd[:pairs] + even[:pairs])
    if pairs < latent_count - 1:
        last = aligned[pairs] if pairs > 0 else aligned[0]
        aligned[1 + pairs :] = last.unsqueeze(0).expand(
            latent_count - 1 - pairs, -1, -1, -1
        )
    return aligned


def load_episode_features(
    dataset_root: Path,
    episode_index: int,
    latent_count: int,
    camera_keys: list[str] | None,
) -> dict[str, torch.Tensor]:
    data = torch.load(
        find_jepa_file(dataset_root, episode_index),
        map_location="cpu",
        weights_only=False,
    )
    keys = camera_keys or sorted(key for key in data if key != "frame_ids")
    if not keys:
        raise ValueError(f"Episode {episode_index} has no JEPA camera features")
    missing = [key for key in keys if key not in data]
    if missing:
        raise KeyError(
            f"Episode {episode_index} is missing JEPA cameras: {missing}"
        )
    return {
        key: align_camera_to_latents(data[key], latent_count)
        for key in keys
    }


def dense_cosine_distance(
    features: dict[str, torch.Tensor],
    goal: dict[str, torch.Tensor],
) -> torch.Tensor:
    distances: list[torch.Tensor] = []
    for key, value in features.items():
        if key not in goal:
            raise KeyError(f"Goal features are missing camera {key!r}")
        goal_value = goal[key]
        if value.shape[1:] != goal_value.shape:
            raise ValueError(
                f"JEPA shape mismatch for camera {key}: "
                f"{tuple(value.shape[1:])} != {tuple(goal_value.shape)}"
            )
        value_norm = F.normalize(value.float(), dim=-1)
        goal_norm = F.normalize(goal_value.float(), dim=-1)
        patch_distance = 1.0 - (value_norm * goal_norm).sum(dim=-1)
        distances.append(patch_distance.mean(dim=(1, 2)))
    return torch.stack(distances, dim=0).mean(dim=0)


def episode_latent_count(
    dataset_root: Path,
    episode: dict[str, Any],
) -> int:
    return sum(
        segment_latent_frame_count(
            dataset_root, int(episode["episode_index"]), action_config
        )
        for action_config in episode.get("action_config", [])
    )


def build_episode_feature_cache(
    dataset_root: Path,
    episodes: list[dict[str, Any]],
    camera_keys: list[str] | None,
) -> tuple[dict[int, dict[str, torch.Tensor]], dict[int, int]]:
    feature_cache: dict[int, dict[str, torch.Tensor]] = {}
    latent_counts: dict[int, int] = {}
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        latent_count = episode_latent_count(dataset_root, episode)
        if latent_count <= 0:
            continue
        latent_counts[episode_index] = latent_count
        feature_cache[episode_index] = load_episode_features(
            dataset_root, episode_index, latent_count, camera_keys
        )
    return feature_cache, latent_counts


def final_feature(
    features: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {key: value[-1] for key, value in features.items()}


def select_goals(
    episodes: list[dict[str, Any]],
    feature_cache: dict[int, dict[str, torch.Tensor]],
    allow_self_goal_for_failed: bool,
) -> dict[int, tuple[int, str, dict[str, torch.Tensor]]]:
    success_goals: dict[int, dict[str, torch.Tensor]] = {
        int(episode["episode_index"]): final_feature(
            feature_cache[int(episode["episode_index"])]
        )
        for episode in episodes
        if episode.get("success", False)
        and int(episode["episode_index"]) in feature_cache
    }
    if not success_goals:
        if not allow_self_goal_for_failed:
            raise ValueError("No successful trajectories with JEPA features found")
        return {
            episode_index: (
                episode_index,
                "self_failed_fallback",
                final_feature(features),
            )
            for episode_index, features in feature_cache.items()
        }

    goals: dict[int, tuple[int, str, dict[str, torch.Tensor]]] = {}
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        if episode_index not in feature_cache:
            continue
        if episode.get("success", False):
            goals[episode_index] = (
                episode_index,
                "self_success",
                success_goals[episode_index],
            )
            continue

        current_final = final_feature(feature_cache[episode_index])
        best_episode = None
        best_distance = None
        for goal_episode, goal_features in success_goals.items():
            distance = dense_cosine_distance(
                {key: value.unsqueeze(0) for key, value in current_final.items()},
                goal_features,
            )[0].item()
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_episode = goal_episode
        if best_episode is None:
            if not allow_self_goal_for_failed:
                raise ValueError(
                    f"Could not select successful goal for episode {episode_index}"
                )
            goals[episode_index] = (
                episode_index,
                "self_failed_fallback",
                current_final,
            )
        else:
            goals[episode_index] = (
                best_episode,
                "closest_success_final",
                success_goals[best_episode],
            )
    return goals


def annotate_episode(
    dataset_root: Path,
    episode: dict[str, Any],
    features: dict[str, torch.Tensor],
    goal_episode_index: int,
    goal_selection: str,
    goal_features: dict[str, torch.Tensor],
    skip_existing: bool,
) -> int:
    episode_index = int(episode["episode_index"])
    distances = dense_cosine_distance(features, goal_features)
    latent_rewards = torch.zeros_like(distances)
    if len(distances) > 1:
        latent_rewards[:-1] = distances[:-1] - distances[1:]
    latent_offset = 0
    updated = 0
    action_configs = sorted(
        episode.get("action_config", []),
        key=lambda item: (int(item["start_frame"]), int(item["end_frame"])),
    )
    for action_config in action_configs:
        if skip_existing and isinstance(action_config.get("reward_config"), dict):
            latent_offset += segment_latent_frame_count(
                dataset_root, episode_index, action_config
            )
            continue
        latent_count = segment_latent_frame_count(
            dataset_root, episode_index, action_config
        )
        start = latent_offset
        end = latent_offset + latent_count
        segment_rewards = latent_rewards[start:end]
        segment_distances = distances[start:end]

        action_config["reward_config"] = {
            "reward_source": REWARD_SOURCE,
            "distance_metric": "cosine",
            "goal_episode_index": int(goal_episode_index),
            "goal_selection": goal_selection,
            "latent_rewards": [
                float(value.item()) for value in segment_rewards
            ],
            "distance_to_goal": [
                float(value.item()) for value in segment_distances
            ],
        }
        latent_offset += latent_count
        updated += 1
    return updated


def annotate_dataset(
    dataset_root: Path,
    camera_keys: list[str] | None,
    skip_existing: bool,
    allow_self_goal_for_failed: bool,
) -> int:
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    episodes = load_jsonl(episodes_path)
    feature_cache, _ = build_episode_feature_cache(
        dataset_root, episodes, camera_keys
    )
    goals = select_goals(
        episodes, feature_cache, allow_self_goal_for_failed
    )
    updated = 0
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        if episode_index not in feature_cache:
            continue
        goal_episode, goal_selection, goal_features = goals[episode_index]
        updated += annotate_episode(
            dataset_root,
            episode,
            feature_cache[episode_index],
            goal_episode,
            goal_selection,
            goal_features,
            skip_existing,
        )
    backup_path = episodes_path.with_suffix(".jsonl.before_jepa_rewards")
    if not backup_path.exists():
        shutil.copy2(episodes_path, backup_path)
    write_jsonl(episodes_path, episodes)
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate episodes.jsonl with JEPA delta-distance rewards."
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--camera-keys", nargs="*", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--allow-self-goal-for-failed",
        action="store_true",
        help="Use a failed episode's own final feature only if no success goal exists.",
    )
    args = parser.parse_args()

    updated = annotate_dataset(
        dataset_root=Path(args.dataset_root),
        camera_keys=args.camera_keys,
        skip_existing=args.skip_existing,
        allow_self_goal_for_failed=args.allow_self_goal_for_failed,
    )
    print(f"Updated reward_config on {updated} action_config entries")


if __name__ == "__main__":
    main()
