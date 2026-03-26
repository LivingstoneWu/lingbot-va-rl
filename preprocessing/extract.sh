#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

NPROC_PER_NODE=8
BATCH_SIZE=160

if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  torchrun --nproc_per_node "${NPROC_PER_NODE}" "${SCRIPT_DIR}/extract_latent_vae.py" \
    --config-name "demo" \
    --dataset-root "/liujinxin/code/lhc/wy/wms/lingbot-va/datasets/robochallenge/put_pen_into_pencil_case" \
    --camera-keys "top" "left_wrist" "right_wrist" \
    --target-fps "15" \
    --height "256" \
    --width "256" \
    --chunk-size "500" \
    --batch-size "${BATCH_SIZE}"
else
  python "${SCRIPT_DIR}/extract_latent_vae.py" \
    --config-name "demo" \
    --dataset-root "/liujinxin/code/lhc/wy/wms/lingbot-va/datasets/robochallenge/put_pen_into_pencil_case" \
    --camera-keys "top" "left_wrist" "right_wrist" \
    --target-fps "15" \
    --height "256" \
    --width "256" \
    --chunk-size "500" \
    --batch-size "${BATCH_SIZE}"
fi
