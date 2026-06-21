"""Extract RobotWin latents using LingBot-VA's T-shaped camera layout."""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _camera_keys(config) -> list[str]:
    return [
        key.removeprefix("observation.images.")
        for key in config.obs_cam_keys
    ]


def validate_robotwin_config(config, args) -> None:
    if config.env_type != "robotwin_tshape":
        raise ValueError(
            "RobotWin extraction requires env_type='robotwin_tshape'; "
            f"config {args.config_name!r} uses {config.env_type!r}"
        )

    expected_keys = _camera_keys(config)
    if len(expected_keys) != 3:
        raise ValueError(
            "RobotWin T-shape requires one high camera followed by two "
            f"wrist cameras; configured keys: {expected_keys}"
        )
    if list(args.camera_keys) != expected_keys:
        raise ValueError(
            "RobotWin camera keys must preserve high/left-wrist/right-wrist "
            f"order: expected {expected_keys}, got {list(args.camera_keys)}"
        )

    requested_size = (args.height, args.width)
    configured_size = (int(config.height), int(config.width))
    if requested_size != configured_size:
        raise ValueError(
            "RobotWin high-camera size must match the selected config: "
            f"expected {configured_size}, got {requested_size}"
        )
    wrist_size = (args.height // 2, args.width // 2)
    if args.height % 2 or args.width % 2:
        raise ValueError("RobotWin high-camera dimensions must be even")
    if wrist_size[0] % 32 or wrist_size[1] % 32:
        raise ValueError(
            "RobotWin wrist dimensions must be multiples of 32 after "
            f"halving; got {wrist_size}"
        )


def robotwin_camera_size(
    camera_key: str,
    target_height: int,
    target_width: int,
    config,
) -> tuple[int, int]:
    keys = _camera_keys(config)
    if camera_key == keys[0]:
        return target_height, target_width
    if camera_key in keys[1:]:
        return target_height // 2, target_width // 2
    raise ValueError(f"Unknown RobotWin camera key: {camera_key!r}")


def main() -> None:
    from preprocessing.extract_latent_vae import main as extract_main

    extract_main(
        camera_size_resolver=robotwin_camera_size,
        config_validator=validate_robotwin_config,
        default_config_name="robotwin",
    )


if __name__ == "__main__":
    main()
