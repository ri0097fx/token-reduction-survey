#!/usr/bin/env bash
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
until grep -q "FINETUNE4 DONE" "$REPO/repro/logs/finetune/progress.txt"; do sleep 60; done
VTR_WORK="${VTR_WORK:?}" "$VTR_WORK/venv_base/bin/python" "$REPO/repro/run_backbones.py" "$1" \
  --models deit3_s_384 --methods none,topk,tome,mctf --batch 64 --workers 8
echo "RESOLUTION DONE $(date '+%F %T')" >> "$REPO/repro/logs/imagenet_backbones/progress.txt"
