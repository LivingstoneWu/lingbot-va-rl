find /liujinxin/code/lhc/wy/wms/lingbot-va/datasets/ur5e/ur5e_stack_color_blocks_action_corrected/videos \
  -type f -name '*.mp4' -print0 |
xargs -0 -n 1 -P 80 -I {} bash -c '
  f="$1"
  tmp="${f%.mp4}.tmp.mp4"
  ffmpeg -y -i "$f" -c:v libx264 -preset fast -crf 18 "$tmp" &&
  mv "$tmp" "$f"
' _ {}