import argparse
from pathlib import Path

import numpy as np
import torch

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan_va.configs import VA_CONFIGS


def decode_inferred_action(action_tensor: torch.Tensor, config) -> np.ndarray:
    """
    Decode saved inferred action tensor using the same logic as wan_va_server.py::postprocess_action.

    Input:
      action_tensor: [B, C_model, F, H, W]
    Output:
      decoded action sequence: [F*H, C_used]
    """
    if action_tensor.ndim != 5:
        raise ValueError(f"Expected action tensor [B,C,F,H,W], got {tuple(action_tensor.shape)}")
    if action_tensor.shape[0] != 1:
        raise ValueError(f"This script expects B=1 for visualization, got B={action_tensor.shape[0]}")

    action = action_tensor.detach().cpu()[0, ..., 0]  # (C_model, F, H)

    action_norm_method = getattr(config, "action_norm_method", "quantiles")
    if action_norm_method != "quantiles":
        raise NotImplementedError(f"Unsupported action_norm_method: {action_norm_method}")

    q01 = np.array(config.norm_stat["q01"], dtype=np.float32)[None, :]  # (1, D)
    q99 = np.array(config.norm_stat["q99"], dtype=np.float32)[None, :]  # (1, D)
    inverse_ids = np.array(config.inverse_used_action_channel_ids, dtype=np.int64)
    used_ids = np.array(config.used_action_channel_ids, dtype=np.int64)

    q01_padded = np.pad(q01, ((0, 0), (0, 1)), mode="constant", constant_values=0)
    q99_padded = np.pad(q99, ((0, 0), (0, 1)), mode="constant", constant_values=0)

    q01_aligned = q01_padded[:, inverse_ids]
    q99_aligned = q99_padded[:, inverse_ids]
    valid = inverse_ids < q01.shape[1]

    q01_aligned = torch.from_numpy(q01_aligned[0]).float().unsqueeze(-1).unsqueeze(-1)  # (C_model,1,1)
    q99_aligned = torch.from_numpy(q99_aligned[0]).float().unsqueeze(-1).unsqueeze(-1)  # (C_model,1,1)
    valid_t = torch.from_numpy(valid).bool().unsqueeze(-1).unsqueeze(-1)  # (C_model,1,1)
    eps_t = torch.tensor(1e-2, dtype=q01_aligned.dtype)

    # De-normalize only valid channels.
    action = torch.where(
        valid_t,
        (action + 1.0) / 2.0 * (torch.maximum(q99_aligned - q01_aligned, eps_t) + 1e-6) + q01_aligned,
        torch.zeros_like(action),
    )

    action = action.numpy()  # (C_model, F, H)
    action = action[used_ids]  # (C_used, F, H)
    action = np.transpose(action, (1, 2, 0))  # (F, H, C_used)
    action = action.reshape(-1, action.shape[-1])  # (F*H, C_used)
    return action


def save_array(arr: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".csv":
        np.savetxt(output_path, arr, delimiter=",")
    else:
        # default to .npy for exact preservation
        np.save(output_path, arr)


def main():
    parser = argparse.ArgumentParser(description="Decode one saved inferred action chunk (.pt).")
    parser.add_argument("--path", type=str, required=True, help="Path to saved actions_*.pt")
    parser.add_argument("--config", type=str, required=True, help="Config name from wan_va.configs.VA_CONFIGS")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for decoded actions (.npy or .csv). Default: <pt_stem>_decoded.npy",
    )
    args = parser.parse_args()

    if args.config not in VA_CONFIGS:
        raise KeyError(f"Unknown config='{args.config}'. Available: {list(VA_CONFIGS.keys())}")
    config = VA_CONFIGS[args.config]

    pt_path = Path(args.path)
    if not pt_path.exists():
        raise FileNotFoundError(f"Action file not found: {pt_path}")

    out_path = Path(args.output) if args.output else pt_path.with_name(f"{pt_path.stem}_decoded.npy")

    action_tensor = torch.load(pt_path, map_location="cpu", weights_only=False)
    if not torch.is_tensor(action_tensor):
        raise TypeError(f"Expected tensor in {pt_path}, got {type(action_tensor)}")

    print(f"Loaded: {pt_path}")
    print(f"Action tensor shape [B,C,F,H,W]: {tuple(action_tensor.shape)}")

    decoded = decode_inferred_action(action_tensor, config)
    print(f"Decoded action shape [T,C_used]: {decoded.shape}")
    print(f"Decoded stats: min={decoded.min():.6g}, max={decoded.max():.6g}, mean={decoded.mean():.6g}, std={decoded.std():.6g}")

    save_array(decoded, out_path)
    print(f"Saved decoded actions: {out_path}")


if __name__ == "__main__":
    main()
