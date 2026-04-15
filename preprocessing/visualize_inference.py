import argparse
import os
import sys
from pathlib import Path

import torch
from diffusers.utils import export_to_video
from diffusers.video_processor import VideoProcessor

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan_va.configs import VA_CONFIGS
from wan_va.modules.utils import load_vae


def resolve_vae_path(config):
    # Prefer finetuned path when available, fallback to base pretrained path.
    model_root = getattr(config, "wan22_finetuned_model_name_or_path", None)
    if not model_root:
        model_root = config.wan22_pretrained_model_name_or_path
    vae_path = os.path.join(model_root, "vae")
    if not os.path.isdir(vae_path):
        raise FileNotFoundError(f"VAE path not found: {vae_path}")
    return vae_path


@torch.no_grad()
def decode_latents_to_video_np(vae, latents):
    if latents.ndim != 5:
        raise ValueError(f"Expected latent tensor shape [B,C,F,H,W], got {tuple(latents.shape)}")
    if latents.shape[0] != 1:
        raise ValueError(f"This script expects B=1 for visualization, got B={latents.shape[0]}")

    latents = latents.to(next(vae.parameters()).device).to(next(vae.parameters()).dtype)

    latents_mean = (
        torch.tensor(vae.config.latents_mean)
        .view(1, vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents_std = (
        1.0
        / torch.tensor(vae.config.latents_std)
        .view(1, vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents = latents / latents_std + latents_mean

    video = vae.decode(latents, return_dict=False)[0]
    video = VideoProcessor(vae_scale_factor=1).postprocess_video(video, output_type="np")[0]
    return video


def main():
    parser = argparse.ArgumentParser(description="Decode one saved inference latent chunk (.pt) into mp4.")
    parser.add_argument("--path", type=str, required=True, help="Path to saved latents_*.pt")
    parser.add_argument("--config", type=str, required=True, help="Config name from wan_va.configs.VA_CONFIGS")
    parser.add_argument("--output", type=str, default=None, help="Output mp4 path (default: alongside pt file)")
    parser.add_argument("--fps", type=int, default=10, help="Output video FPS")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for VAE decode, e.g. cuda:0 or cpu")
    args = parser.parse_args()

    if args.config not in VA_CONFIGS:
        raise KeyError(f"Unknown config_name='{args.config}'. Available: {list(VA_CONFIGS.keys())}")
    config = VA_CONFIGS[args.config]

    pt_path = Path(args.path)
    if not pt_path.exists():
        raise FileNotFoundError(f"Latent file not found: {pt_path}")

    output_path = Path(args.output) if args.output else pt_path.with_suffix(".mp4")

    latents = torch.load(pt_path, map_location="cpu", weights_only=False)
    if not torch.is_tensor(latents):
        raise TypeError(f"Expected tensor in {pt_path}, got {type(latents)}")

    print(f"Loaded: {pt_path}")
    print(f"Latent shape [B,C,F,H,W]: {tuple(latents.shape)}")
    print(f"F/H/W from tensor: F={latents.shape[2]}, H={latents.shape[3]}, W={latents.shape[4]}")
    print(f"Config frame_chunk_size: {getattr(config, 'frame_chunk_size', 'N/A')}")

    vae_path = resolve_vae_path(config)
    vae = load_vae(vae_path, torch_dtype=torch.float32, torch_device=args.device)
    video_np = decode_latents_to_video_np(vae, latents)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(video_np, str(output_path), fps=args.fps)
    print(f"Saved decoded video: {output_path}")


if __name__ == "__main__":
    main()
