#!/usr/bin/env bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/extract_latent_vae.py" \
  --config-name "demo" \
  --dataset-root "/liujinxin/code/lhc/lingbot-va/datasets/robochallenge/turn_on_faucet_trim" \
  --camera-keys "top" \
  --target-fps "15" \
  --height "256" \
  --width "256" \
  --chunk-size "500"
