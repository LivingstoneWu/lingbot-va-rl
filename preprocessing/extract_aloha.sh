#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES=4,5,6,7

NPROC_PER_NODE=4
BATCH_SIZE=80

if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  torchrun --nproc_per_node "${NPROC_PER_NODE}" "${SCRIPT_DIR}/extract_latent_vae.py" \
    --config-name "demo" \
    --dataset-root "/liujinxin/code/lhc/wy/wms/lingbot-va/datasets/robochallenge/make_vegetarian_sandwich_trim" \
    --camera-keys "top" "left_wrist" "right_wrist" \
    --target-fps "10" \
    --height "256" \
    --width "256" \
    --chunk-size "241" \
    --batch-size "${BATCH_SIZE}"
else
  python "${SCRIPT_DIR}/extract_latent_vae.py" \
    --config-name "demo" \
    --dataset-root "/liujinxin/code/lhc/wy/wms/lingbot-va/datasets/robochallenge/make_vegetarian_sandwich_trim" \
    --camera-keys "top" "left_wrist" "right_wrist" \
    --target-fps "10" \
    --height "256" \
    --width "256" \
    --chunk-size "241" \
    --batch-size "${BATCH_SIZE}"
fi
