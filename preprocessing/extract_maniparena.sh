#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

NPROC_PER_NODE=4
BATCH_SIZE=160

# 👇 pass dataset root as argument, or set default
DATASET_ROOT="${1:-/liujinxin/code/lhc/wy/wms/lingbot-va/datasets/maniparena/multi_datasets/final}"

echo "Processing datasets under: $DATASET_ROOT"

# Iterate through subdirectories (depth = 1)
for subdir in "${DATASET_ROOT}"/*; do
  if [[ -d "$subdir" ]]; then
    echo "======================================"
    echo "Processing: $subdir"
    echo "======================================"

    if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
      torchrun --nproc_per_node "${NPROC_PER_NODE}" "${SCRIPT_DIR}/extract_latent_vae.py" \
        --config-name "demo" \
        --dataset-root "$subdir" \
        --camera-keys "faceImg" "leftImg" "rightImg" \
        --target-fps "10" \
        --height "256" \
        --width "256" \
        --chunk-size "501" \
        --batch-size "${BATCH_SIZE}"
    else
      python "${SCRIPT_DIR}/extract_latent_vae.py" \
        --config-name "demo" \
        --dataset-root "$subdir" \
        --camera-keys "faceImg" "leftImg" "rightImg" \
        --target-fps "10" \
        --height "256" \
        --width "256" \
        --chunk-size "501" \
        --batch-size "${BATCH_SIZE}"
    fi
  fi
done