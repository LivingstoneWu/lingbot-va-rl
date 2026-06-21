import math

import pytest
import torch
from torch.utils.data import Dataset

from wan_va.rl.transitions import ChunkTransitionDataset


class FakeLatentDataset(Dataset):
    def __init__(self, success=True, include_outcome=True):
        self.metadata = {
            "dataset_idx": 0,
            "dataset_id": 0,
            "episode_index": 7,
            "start_frame": 0,
            "end_frame": 20,
            "latent_frame_count": 5,
            "truncated": False,
        }
        if include_outcome:
            self.metadata["success"] = success
        self.sample = {
            "latents": torch.arange(5.0).view(1, 5, 1, 1),
            "actions": torch.ones(2, 5, 3, 1),
            "actions_mask": torch.ones(2, 5, 3, 1, dtype=torch.bool),
            "latents_mask": torch.ones(5, dtype=torch.bool),
            "text_emb": torch.ones(4, 6),
        }

    def __len__(self):
        return 1

    def get_rl_segment_metadata(self, idx):
        assert idx == 0
        return self.metadata

    def __getitem__(self, idx):
        assert idx == 0
        return self.sample


def test_chunks_rewards_successors_and_mc_returns():
    dataset = ChunkTransitionDataset(
        FakeLatentDataset(),
        infer_latent_chunk_size=2,
        action_per_frame=3,
        gamma=0.5,
    )

    assert len(dataset) == 3
    assert dataset.records[0].previous_record_idx is None
    assert dataset.records[1].previous_record_idx == 0
    assert dataset.records[2].previous_record_idx == 1
    assert dataset.records[0].next_record_idx == 1
    assert dataset.records[1].next_record_idx == 2
    assert dataset.records[2].next_record_idx is None
    assert dataset.records[2].done
    assert dataset.records[2].reward == 1.0
    assert math.isclose(dataset.records[1].mc_return, 0.5**6)
    assert math.isclose(dataset.records[0].mc_return, 0.5**12)

    first = dataset[0]
    assert not first["state_valid"]
    assert not first["previous_latents_mask"].any()

    terminal = dataset[2]
    assert terminal["latents"].shape == (1, 2, 1, 1)
    assert terminal["actions"].shape == (2, 2, 3, 1)
    assert terminal["latents_mask"].tolist() == [True, False]
    assert not terminal["actions_mask"][:, 1].any()
    assert terminal["state_valid"]
    assert terminal["previous_latents"].flatten().tolist() == [2.0, 3.0]
    assert terminal["next_transition_idx"].item() == -1


def test_failed_terminal_chunk_has_zero_return():
    dataset = ChunkTransitionDataset(
        FakeLatentDataset(success=False),
        infer_latent_chunk_size=2,
        action_per_frame=3,
        gamma=0.99,
    )
    assert all(record.mc_return == 0.0 for record in dataset.records)
    assert dataset.records[-1].done
    assert dataset.records[-1].reward == 0.0


def test_nonfinal_partial_chunk_is_rejected():
    class MisalignedSegments(FakeLatentDataset):
        def __len__(self):
            return 2

        def get_rl_segment_metadata(self, idx):
            metadata = dict(self.metadata)
            metadata["dataset_idx"] = idx
            metadata["start_frame"] = idx * 10
            metadata["end_frame"] = (idx + 1) * 10
            metadata["latent_frame_count"] = 3 if idx == 0 else 2
            return metadata

    with pytest.raises(ValueError, match="Non-final storage segment"):
        ChunkTransitionDataset(
            MisalignedSegments(),
            infer_latent_chunk_size=2,
            action_per_frame=3,
        )



def test_missing_success_label_is_rejected():
    with pytest.raises(ValueError, match="no success label"):
        ChunkTransitionDataset(
            FakeLatentDataset(include_outcome=False),
            infer_latent_chunk_size=2,
            action_per_frame=3,
        )

