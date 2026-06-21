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

