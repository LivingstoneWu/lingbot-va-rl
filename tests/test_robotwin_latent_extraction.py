from types import SimpleNamespace

import pytest

from preprocessing.extract_latent_vae_robotwin import (
    robotwin_camera_size,
    validate_robotwin_config,
)


CAMERA_KEYS = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]
CLI_CAMERA_KEYS = [
    "cam_high",
    "cam_left_wrist",
    "cam_right_wrist",
]


def make_config(**overrides):
    values = {
        "env_type": "robotwin_tshape",
        "obs_cam_keys": CAMERA_KEYS,
        "height": 256,
        "width": 320,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def make_args(**overrides):
    values = {
        "config_name": "robotwin",
        "camera_keys": CLI_CAMERA_KEYS,
        "height": 256,
        "width": 320,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_robotwin_camera_sizes_form_t_layout():
    config = make_config()

    assert robotwin_camera_size("cam_high", 256, 320, config) == (256, 320)
    assert robotwin_camera_size(
        "cam_left_wrist", 256, 320, config
    ) == (128, 160)
    assert robotwin_camera_size(
        "cam_right_wrist", 256, 320, config
    ) == (128, 160)


def test_robotwin_config_accepts_expected_layout():
    validate_robotwin_config(make_config(), make_args())


def test_robotwin_config_rejects_wrong_camera_order():
    args = make_args(camera_keys=list(reversed(CLI_CAMERA_KEYS)))

    with pytest.raises(ValueError, match="must preserve"):
        validate_robotwin_config(make_config(), args)


def test_robotwin_config_rejects_uniform_square_extraction():
    args = make_args(width=256)

    with pytest.raises(ValueError, match="must match"):
        validate_robotwin_config(make_config(), args)


def test_robotwin_config_rejects_non_tshape_config():
    config = make_config(env_type="none")

    with pytest.raises(ValueError, match="robotwin_tshape"):
        validate_robotwin_config(config, make_args())
