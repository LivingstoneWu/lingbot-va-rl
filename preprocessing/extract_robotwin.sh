#!/usr/bin/env bash

# Extract RobotWin latents with the high/wrist T-shaped camera layout.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  echo "Usage: bash $0 [num_gpus] [dataset_root]"
  echo
  echo "Examples:"
  echo "  bash $0 8 /path/to/robotwin_dataset"
  echo "  CUDA_VISIBLE_DEVICES=2,3 bash $0 2 /path/to/robotwin_dataset"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

DEFAULT_DATASET_ROOT="/luhongchao/shared/dataset/robotwin_converted/lingbot_rollout/place_can_basket/place_can_basket_lingbot_rollouts"
REQUESTED_GPUS="${1:-${NPROC_PER_NODE:-4}}"
DATASET_ROOT="${2:-${DATASET_ROOT:-${DEFAULT_DATASET_ROOT}}}"

if [[ ! "${REQUESTED_GPUS}" =~ ^[0-9]+$ || "${REQUESTED_GPUS}" -lt 1 ]]; then
  echo "num_gpus must be a positive integer, got: ${REQUESTED_GPUS}" >&2
  usage >&2
  exit 1
fi

if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "dataset_root does not exist or is not a directory: ${DATASET_ROOT}" >&2
  exit 1
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES="$(python - "${REQUESTED_GPUS}" <<'PY'
import sys
n = int(sys.argv[1])
print(",".join(str(i) for i in range(n)))
PY
)"
fi
export CUDA_VISIBLE_DEVICES

VISIBLE_GPU_COUNT="$(python - <<'PY'
import os
visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
devices = [item for item in visible.split(",") if item.strip()]
print(len(devices) if devices else 1)
PY
)"
if [[ "${REQUESTED_GPUS}" -gt "${VISIBLE_GPU_COUNT}" ]]; then
  echo "num_gpus=${REQUESTED_GPUS} exceeds CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (${VISIBLE_GPU_COUNT} visible devices)" >&2
  exit 1
fi

NPROC_PER_NODE="${REQUESTED_GPUS}"
BATCH_SIZE="${BATCH_SIZE:-3}"

# for debug

export PYTHONFAULTHANDLER=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
export TORCH_SHOW_CPP_STACKTRACES=1
export CUDA_VISIBLE_DEVICES

if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  torchrun --nproc_per_node "${NPROC_PER_NODE}" "${SCRIPT_DIR}/extract_latent_vae_robotwin.py" \
    --config-name "robotwin_qgf_v1_cfg_place_can_basket_generated" \
    --dataset-root "${DATASET_ROOT}" \
    --camera-keys "cam_high" "cam_left_wrist" "cam_right_wrist" \
    --target-fps "12.5" \
    --height "256" \
    --width "320" \
    --chunk-size "501" \
    --batch-size "${BATCH_SIZE}"
else
  python "${SCRIPT_DIR}/extract_latent_vae_robotwin.py" \
    --config-name "robotwin_qgf_v1_cfg_place_can_basket_generated" \
    --dataset-root "${DATASET_ROOT}" \
    --camera-keys "cam_high" "cam_left_wrist" "cam_right_wrist" \
    --target-fps "12.5" \
    --height "256" \
    --width "320" \
    --chunk-size "501" \
    --batch-size "${BATCH_SIZE}"
fi
