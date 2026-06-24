import json

import pytest
import torch

from preprocessing.add_jepa_delta_rewards import annotate_dataset


def test_annotate_dataset_writes_per_latent_rewards(tmp_path):
    dataset = tmp_path / "dataset"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "latents" / "chunk-000" / "observation.images.cam_high").mkdir(
        parents=True
    )
    (dataset / "jepa" / "chunk-000").mkdir(parents=True)

    episode = {
        "episode_index": 0,
        "success": True,
        "action_config": [{"start_frame": 0, "end_frame": 5}],
    }
    (dataset / "meta" / "episodes.jsonl").write_text(
        json.dumps(episode) + "\n"
    )
    torch.save(
        {"latent_num_frames": 3},
        dataset
        / "latents"
        / "chunk-000"
        / "observation.images.cam_high"
        / "episode_000000_0_5.pth",
    )

    state0 = torch.tensor([1.0, 0.0])
    state1 = torch.tensor([0.0, 1.0])
    state2 = torch.tensor([0.0, 1.0])
    tokens = torch.stack(
        [state0, state1, state1, state2, state2], dim=0
    ).view(5, 1, 1, 2)
    torch.save(
        {
            "frame_ids": torch.arange(5),
            "observation.images.cam_high": tokens,
        },
        dataset / "jepa" / "chunk-000" / "episode_000000.pt",
    )

    updated = annotate_dataset(
        dataset_root=dataset,
        camera_keys=["observation.images.cam_high"],
        skip_existing=False,
        allow_self_goal_for_failed=False,
    )

    assert updated == 1
    annotated = json.loads(
        (dataset / "meta" / "episodes.jsonl").read_text()
    )
    reward_config = annotated["action_config"][0]["reward_config"]
    assert reward_config["reward_source"] == "jepa_delta_distance"
    assert reward_config["goal_selection"] == "self_success"
    assert reward_config["latent_rewards"] == pytest.approx([1.0, 0.0, 0.0])
    assert reward_config["distance_to_goal"] == pytest.approx([1.0, 0.0, 0.0])
