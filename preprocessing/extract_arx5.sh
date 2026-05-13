#!/usr/bin/env bash

# Used to extract latents for ur5

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

NPROC_PER_NODE=4
BATCH_SIZE=3

# for debug

export PYTHONFAULTHANDLER=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
export TORCH_SHOW_CPP_STACKTRACES=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  torchrun --nproc_per_node "${NPROC_PER_NODE}" "${SCRIPT_DIR}/extract_latent_vae.py" \
    --config-name "demo" \
    --dataset-root "/liujinxin/code/lhc/wy/wms/lingbot-va/datasets/robochallenge/wipe_the_table" \
    --camera-keys "top" "wrist" "scene" \
    --target-fps "10" \
    --height "256" \
    --width "256" \
    --chunk-size "501" \
    --batch-size "${BATCH_SIZE}"
else
  python "${SCRIPT_DIR}/extract_latent_vae.py" \
    --config-name "demo" \
    --dataset-root "/liujinxin/code/lhc/wy/wms/lingbot-va/datasets/robochallenge/wipe_the_table" \
    --camera-keys "top" "wrist" "scene" \
    --target-fps "10" \
    --height "256" \
    --width "256" \
    --chunk-size "501" \
    --batch-size "${BATCH_SIZE}"
fi
