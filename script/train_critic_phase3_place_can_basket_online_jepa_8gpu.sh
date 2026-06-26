#!/usr/bin/env bash

set -euo pipefail
set -x

umask 007

REPO_ROOT="${REPO_ROOT:-/luhongchao/wy/lingbot-va-rl}"
CONDA_ROOT="${CONDA_ROOT:-/luhongchao/anaconda3}"
CONDA_ENV="${CONDA_ENV:-lingbot}"
CONFIG="${CONFIG:-wan_va/rl_configs/qgf_phase3_robotwin_place_can_basket_200rollout_50success_online_jepa.json}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29529}"
LOG_RANK="${LOG_RANK:-0}"
LOG_DIR="${LOG_DIR:-training_logs/critic_phase3_place_can_basket_online_jepa}"

cd "${REPO_ROOT}"
source "${CONDA_ROOT}/bin/activate" "${CONDA_ENV}"

export TOKENIZERS_PARALLELISM=false
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"

mkdir -p "${LOG_DIR}"

python -m torch.distributed.run \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  --local-ranks-filter="${LOG_RANK}" \
  --log_dir="${LOG_DIR}" \
  --tee 3 \
  -m wan_va.rl.train_critic_phase3 \
  --config "${CONFIG}"
