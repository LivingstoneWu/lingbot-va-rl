#!/usr/bin/bash

set -x

umask 007

cd /liujinxin/code/lhc/wy/wms/lingbot-va
source /liujinxin/conda3/bin/activate wy-lingbotva
 
NGPU=${NGPU:-"8"}
MASTER_PORT=${MASTER_PORT:-"29502"}
PORT=${PORT:-"1107"}
LOG_RANK=${LOG_RANK:-"0"}
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}
# CONFIG_NAME=${CONFIG_NAME:-"robotwin_train"}
CONFIG_NAME=${CONFIG_NAME:-"rc_ur5_set_the_plates"}

overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

# export WANDB_API_KEY="your key"
# export WANDB_BASE_URL="your url"
# export WANDB_TEAM_NAME="your team name"
# export WANDB_PROJECT="your project"

## node setting
num_gpu=${NGPU}
master_port=${MASTER_PORT}
log_rank=${LOG_RANK}
torchft_lighthouse=${TORCHFT_LIGHTHOUSE}
config_name=${CONFIG_NAME}

# export CUDA_VISIBLE_DEVICES=4,5,6,7

## cmd setting
export TOKENIZERS_PARALLELISM=false
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" TORCHFT_LIGHTHOUSE=${torchft_lighthouse} \
 python -m torch.distributed.run \
     --nproc_per_node=${num_gpu} \
     --local-ranks-filter=${log_rank} \
     --master_port ${master_port} \
     --tee 3 \
     -m wan_va.train --config-name ${config_name} $overrides
#python    -m wan_va.train --config-name ${config_name} $overrides
 
