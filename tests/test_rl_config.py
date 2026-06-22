import json

import pytest

from wan_va.rl.config import CriticTrainingConfig


def test_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / "critic.json"
    path.write_text(
        json.dumps(
            {
                "base_config_name": "demo_train",
                "output_dir": "out",
                "infer_latent_chunk_size": 4,
                "typo_key": True,
            }
        )
    )
    with pytest.raises(ValueError, match="Unknown critic config keys"):
        CriticTrainingConfig.from_json(path)


def test_config_validates_algorithm():
    with pytest.raises(ValueError, match="algorithm"):
        CriticTrainingConfig(
            base_config_name="demo_train",
            output_dir="out",
            infer_latent_chunk_size=4,
            algorithm="unknown",
        )


def test_config_validates_log_interval():
    with pytest.raises(ValueError, match="log_interval"):
        CriticTrainingConfig(
            base_config_name="demo_train",
            output_dir="out",
            infer_latent_chunk_size=4,
            log_interval=0,
        )

def test_config_normalizes_single_feature_layer_from_json(tmp_path):
    path = tmp_path / "critic.json"
    path.write_text(
        json.dumps(
            {
                "base_config_name": "demo_train",
                "output_dir": "out",
                "infer_latent_chunk_size": 4,
                "feature_layers": [20],
                "feature_aggregation": "single",
            }
        )
    )

    config = CriticTrainingConfig.from_json(path)

    assert config.feature_layers == (20,)
    assert config.feature_layer == 20
    assert config.feature_normalization == "raw_block_output_v1"


def test_config_defaults_to_final_normalized_features():
    config = CriticTrainingConfig(
        base_config_name="demo_train",
        output_dir="out",
        infer_latent_chunk_size=4,
    )

    assert config.feature_layers == (-1,)
    assert config.feature_normalization == "final_adaptive_norm_v1"


def test_config_rejects_feature_layer_mixing_in_phase1():
    with pytest.raises(ValueError, match="exactly one"):
        CriticTrainingConfig(
            base_config_name="demo_train",
            output_dir="out",
            infer_latent_chunk_size=4,
            feature_layers=(10, 20),
        )


def test_config_rejects_invalid_feature_layer_and_aggregation():
    with pytest.raises(ValueError, match="-1 or non-negative"):
        CriticTrainingConfig(
            base_config_name="demo_train",
            output_dir="out",
            infer_latent_chunk_size=4,
            feature_layers=(-2,),
        )
    with pytest.raises(ValueError, match="feature_aggregation"):
        CriticTrainingConfig(
            base_config_name="demo_train",
            output_dir="out",
            infer_latent_chunk_size=4,
            feature_aggregation="mean",
        )
