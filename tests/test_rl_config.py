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


def test_config_validates_reward_source():
    config = CriticTrainingConfig(
        base_config_name="demo_train",
        output_dir="out",
        infer_latent_chunk_size=4,
        reward_source="jepa_delta_distance",
        include_sparse_success_reward=True,
        jepa_reward_weight=0.5,
        success_reward_weight=2.0,
    )

    assert config.reward_source == "jepa_delta_distance"
    assert config.include_sparse_success_reward
    assert config.jepa_reward_weight == 0.5
    assert config.success_reward_weight == 2.0

    with pytest.raises(ValueError, match="reward_source"):
        CriticTrainingConfig(
            base_config_name="demo_train",
            output_dir="out",
            infer_latent_chunk_size=4,
            reward_source="unknown",
        )


def test_phase3_config_requires_predicted_distribution():
    config = CriticTrainingConfig(
        base_config_name="demo_train",
        output_dir="out",
        infer_latent_chunk_size=2,
        algorithm="iql",
        reward_source="negative_predicted_actual_jepa_distance",
        training_distribution="predicted_video_conditioned_action",
        phase3_q_feature_timestep=0.0,
        phase3_v_feature_timestep=0.5,
    )

    assert config.training_distribution == "predicted_video_conditioned_action"
    assert config.reward_source == "negative_predicted_actual_jepa_distance"
    assert config.phase3_v_feature_timestep == 0.5

    with pytest.raises(ValueError, match="training_distribution"):
        CriticTrainingConfig(
            base_config_name="demo_train",
            output_dir="out",
            infer_latent_chunk_size=2,
            algorithm="iql",
            reward_source="negative_predicted_actual_jepa_distance",
        )

    with pytest.raises(ValueError, match="currently requires algorithm='iql'"):
        CriticTrainingConfig(
            base_config_name="demo_train",
            output_dir="out",
            infer_latent_chunk_size=2,
            algorithm="mc",
            reward_source="negative_predicted_actual_jepa_distance",
            training_distribution="predicted_video_conditioned_action",
        )
