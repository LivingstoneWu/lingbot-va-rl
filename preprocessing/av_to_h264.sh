#!/usr/bin/env bash
set -euo pipefail

ROOT="/luhongchao/shared/dataset/robotwin_converted/lingbot_rollout/place_can_basket/place_can_basket_lingbot_rollouts"
NPROC=40

find "$ROOT" -type d -name videos -print0 |
while IFS= read -r -d '' video_dir; do
  find "$video_dir" -type f -name '*.mp4' -print0
done |
xargs -0 -n 1 -P "$NPROC" bash -c '
  f="$1"
  tmp="${f%.mp4}.tmp.mp4"

  echo "Processing: $f"

  if ffmpeg -y -i "$f" -c:v libx264 -preset fast -crf 18 "$tmp"; then
    mv "$tmp" "$f"
  else
    echo "Failed: $f" >&2
    rm -f "$tmp"
    exit 1
  fi
' _