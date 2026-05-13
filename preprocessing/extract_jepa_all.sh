#!/usr/bin/env bash
# run_extract_jepa_all.sh
# ─────────────────────────────────────────────────────────────────────────────
# Iteratively run extract_jepa_features.py for every dataset found under
# DATASETS_ROOT.  A "dataset" is any directory that contains a videos/
# subdirectory.  Datasets are processed one at a time; all GPUs are used for
# each run.
#
# Usage:
#   bash preprocessing/run_extract_jepa_all.sh \
#       --datasets-root /data/lingbot-va/datasets \
#       --checkpoint    /checkpoints/vjepa2_1_vitG_384.pt \
#       --nproc         8 \
#       [--target-fps 10] [--height 256] [--width 256] \
#       [--batch-size 4] [--skip-existing] [--chunk-ids 000 001]
#
# All flags after --datasets-root / --checkpoint / --nproc are forwarded
# verbatim to extract_jepa_features.py.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTRACT_SCRIPT="$SCRIPT_DIR/extract_jepa_features.py"
LOG_DIR="$SCRIPT_DIR/../logs/extract_jepa"
mkdir -p "$LOG_DIR"

# ── parse our own flags, collect the rest as passthrough ─────────────────────
DATASETS_ROOT=""
CHECKPOINT="/liujinxin/weights/vjepa2.1/vjepa2_1_vitG_384.pt"
NPROC=1
PASSTHROUGH=()   # forwarded verbatim to extract_jepa_features.py

while [[ $# -gt 0 ]]; do
    case "$1" in
        --datasets-root)  DATASETS_ROOT="$2"; shift 2 ;;
        --checkpoint)     CHECKPOINT="$2";    shift 2 ;;
        --nproc)          NPROC="$2";          shift 2 ;;
        *)                PASSTHROUGH+=("$1"); shift   ;;
    esac
done

if [[ -z "$DATASETS_ROOT" || -z "$CHECKPOINT" ]]; then
    echo "Usage: $0 --datasets-root <path> --checkpoint <path> [--nproc N] [extract args...]"
    exit 1
fi

# ── find all dataset roots (dirs containing a videos/ subdirectory) ───────────
# Use find -maxdepth to avoid descending into videos/ itself.
mapfile -t DATASETS < <(
    find "$DATASETS_ROOT" -maxdepth 4 -type d -name 'videos' \
        | sed 's|/videos$||' \
        | sort
)

if [[ ${#DATASETS[@]} -eq 0 ]]; then
    echo "No datasets found under $DATASETS_ROOT (looked for dirs containing videos/)."
    exit 1
fi

TOTAL=${#DATASETS[@]}
echo "Found $TOTAL dataset(s) under $DATASETS_ROOT"
for ds in "${DATASETS[@]}"; do echo "  $ds"; done
echo ""

# ── run extract for each dataset ──────────────────────────────────────────────
PASS=0
FAIL=0
FAIL_LIST=()

for i in "${!DATASETS[@]}"; do
    DS="${DATASETS[$i]}"
    DS_NAME="$(basename "$DS")"
    IDX=$((i + 1))
    LOG_FILE="$LOG_DIR/${DS_NAME}.log"
    TIMESTAMP="$(date '+%Y-%m-%d %H:%M:%S')"

    echo "[$TIMESTAMP] [$IDX/$TOTAL] Starting: $DS_NAME"
    echo "  log → $LOG_FILE"

    CMD=(
        torchrun
            --nproc_per_node="$NPROC"
            --standalone          # single-node, no rendezvous server needed
        "$EXTRACT_SCRIPT"
            --dataset-root  "$DS"
            --checkpoint    "$CHECKPOINT"
            "${PASSTHROUGH[@]}"
    )

    # Run and tee to both terminal and log file; capture exit code without
    # triggering set -e so we can continue to the next dataset on failure.
    if "${CMD[@]}" 2>&1 | tee "$LOG_FILE"; then
        PASS=$((PASS + 1))
        echo "  ✓ Done: $DS_NAME"
    else
        FAIL=$((FAIL + 1))
        FAIL_LIST+=("$DS_NAME")
        echo "  ✗ FAILED: $DS_NAME  (see $LOG_FILE)"
    fi

    echo ""
done

# ── summary ───────────────────────────────────────────────────────────────────
echo "════════════════════════════════════════"
echo "All datasets processed."
echo "  Passed : $PASS / $TOTAL"
echo "  Failed : $FAIL / $TOTAL"
if [[ $FAIL -gt 0 ]]; then
    echo "  Failed datasets:"
    for name in "${FAIL_LIST[@]}"; do echo "    - $name"; done
fi
echo "════════════════════════════════════════"
