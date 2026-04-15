#!/usr/bin/env bash

# Used to extract latents for ur5

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

NPROC_PER_NODE=1
BATCH_SIZE=3

# for debug

export PYTHONFAULTHANDLER=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
export TORCH_SHOW_CPP_STACKTRACES=1


if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  torchrun --nproc_per_node "${NPROC_PER_NODE}" "${SCRIPT_DIR}/extract_latent_vae.py" \
    --config-name "demo" \
    --dataset-root "/liujinxin/code/lhc/wy/wms/lingbot-va/datasets/ur5e/stack_color_blocks_action_corrected" \
    --camera-keys "top" "wrist" \
    --target-fps "10" \
    --height "256" \
    --width "256" \
    --chunk-size "501" \
    --batch-size "${BATCH_SIZE}"
else
  python "${SCRIPT_DIR}/extract_latent_vae.py" \
    --config-name "demo" \
    --dataset-root "/liujinxin/code/lhc/wy/wms/lingbot-va/datasets/ur5e/stack_color_blocks_action_corrected" \
    --camera-keys "top" "wrist" \
    --target-fps "10" \
    --height "256" \
    --width "256" \
    --chunk-size "501" \
    --batch-size "${BATCH_SIZE}"
fi
