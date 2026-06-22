import json

import pytest

from wan_va.rl.checkpoint import (
    load_critic_checkpoint,
    save_critic_checkpoint,
)
from wan_va.rl.config import CriticTrainingConfig
from wan_va.rl.critics import CriticBundle


def test_checkpoint_rejects_incompatible_chunk_size(tmp_path):
    config = CriticTrainingConfig(
        base_config_name="demo_train",
        output_dir=str(tmp_path),
        infer_latent_chunk_size=4,
    )
    bundle = CriticBundle(
        feature_dim=8,
        hidden_dim=16,
        num_layers=1,
        with_value=False,
    )

    class OptimizerStub:
        @staticmethod
        def state_dict():
            return {}

    checkpoint = save_critic_checkpoint(
        tmp_path / "checkpoint",
        bundle,
        OptimizerStub(),
        config,
        step=3,
        manifest={},
    )

    with pytest.raises(ValueError, match="infer_latent_chunk_size"):
        load_critic_checkpoint(
            checkpoint,
            bundle,
            expected_manifest={"infer_latent_chunk_size": 2},
        )


def test_checkpoint_persists_and_validates_feature_spec(tmp_path):
    config = CriticTrainingConfig(
        base_config_name="demo_train",
        output_dir=str(tmp_path),
        infer_latent_chunk_size=4,
        feature_layers=(20,),
    )
    bundle = CriticBundle(
        feature_dim=8,
        hidden_dim=16,
        num_layers=1,
        with_value=False,
    )

    class OptimizerStub:
        @staticmethod
        def state_dict():
            return {}

    checkpoint = save_critic_checkpoint(
        tmp_path / "feature_checkpoint",
        bundle,
        OptimizerStub(),
        config,
        step=3,
        manifest={},
    )

    with (checkpoint / "manifest.json").open() as handle:
        manifest = json.load(handle)
    assert manifest["feature_layers"] == [20]
    assert manifest["feature_aggregation"] == "single"
    assert manifest["feature_normalization"] == "raw_block_output_v1"

    with pytest.raises(ValueError, match="feature_layers"):
        load_critic_checkpoint(
            checkpoint,
            bundle,
            expected_manifest={"feature_layers": [-1]},
        )
