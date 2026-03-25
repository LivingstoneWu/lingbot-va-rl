#!/usr/bin/env bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/extract_latent_vae.py" \
  --config-name "demo" \
  --dataset-root "/liujinxin/code/lhc/wy/wms/lingbot-va/datasets/robochallenge/put_pen_into_pencil_case" \
  --camera-keys "top" "left_wrist" "right_wrist" \
  --target-fps "15" \
  --height "256" \
  --width "256" \
  --chunk-size "500"
