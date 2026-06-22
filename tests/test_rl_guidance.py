import json

import pytest
import torch

from wan_va.rl.checkpoint import save_critic_checkpoint
from wan_va.rl.config import CriticTrainingConfig
from wan_va.rl.critics import CriticBundle
from wan_va.rl.guidance import (
    build_action_guidance_mask,
    denoising_time_scaling,
    load_q_guidance_artifact,
)


def test_denoising_time_scaling_matches_expected_shape():
    time = torch.tensor([1.0, 0.5, 0.0])
    scale = denoising_time_scaling(time, beta=2.0)

    assert scale.shape == time.shape
    assert scale[0].item() == pytest.approx(2.0)
    assert scale[1].item() == pytest.approx(2.0)
    assert scale[2].item() == pytest.approx(0.0)


def test_action_guidance_mask_excludes_invalid_and_clamped_tokens():
    actions = torch.ones(1, 3, 2, 4, 1)
    channel_mask = torch.tensor([True, False, True])

    mask = build_action_guidance_mask(
        actions,
        valid_action_channels=channel_mask,
        clamp_first_frame=True,
    )

    assert mask.shape == actions.shape
    assert not mask[:, :, 0].any()
    assert not mask[:, 1].any()
    assert mask[:, [0, 2], 1].all()


def test_load_q_guidance_artifact_builds_registered_adapter(tmp_path):
    config = CriticTrainingConfig(
        base_config_name="demo_train",
        output_dir=str(tmp_path),
        infer_latent_chunk_size=4,
        feature_dim=8,
        hidden_dim=16,
        num_layers=1,
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
        manifest={"action_per_frame": 4},
    )

    artifact = load_q_guidance_artifact(checkpoint, torch.device("cpu"))
    tokens = torch.randn(2, 4, 4, 8)
    mask = torch.ones(2, 3, 4, 4, 1, dtype=torch.bool)
    q = artifact.adapter.objective(tokens, mask)

    assert artifact.config.critic_type == "twin_mlp_v1"
    assert artifact.manifest["action_per_frame"] == 4
    assert q.shape == (2,)


def test_load_q_guidance_artifact_accepts_schema2_final_feature(tmp_path):
    config = CriticTrainingConfig(
        base_config_name="demo_train",
        output_dir=str(tmp_path),
        infer_latent_chunk_size=4,
        feature_dim=8,
    )
    bundle = CriticBundle(feature_dim=8, with_value=False)

    class OptimizerStub:
        @staticmethod
        def state_dict():
            return {}

    checkpoint = save_critic_checkpoint(
        tmp_path / "legacy_checkpoint",
        bundle,
        OptimizerStub(),
        config,
        step=3,
        manifest={},
    )
    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] = 2
    manifest.pop("feature_layers")
    manifest.pop("feature_aggregation")
    manifest.pop("feature_normalization")
    manifest_path.write_text(json.dumps(manifest))

    artifact = load_q_guidance_artifact(checkpoint, torch.device("cpu"))

    assert artifact.manifest["schema_version"] == 3
    assert artifact.manifest["feature_layers"] == [-1]
    assert artifact.manifest["feature_aggregation"] == "single"
    assert artifact.manifest["feature_normalization"] == (
        "final_adaptive_norm_v1"
    )


def test_load_q_guidance_artifact_rejects_manifest_mismatch(tmp_path):
    config = CriticTrainingConfig(
        base_config_name="demo_train",
        output_dir=str(tmp_path),
        infer_latent_chunk_size=4,
        feature_dim=8,
    )
    bundle = CriticBundle(feature_dim=8, with_value=False)

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
    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["feature_dim"] = 16
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="feature_dim"):
        load_q_guidance_artifact(checkpoint, torch.device("cpu"))
