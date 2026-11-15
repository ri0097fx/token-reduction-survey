#!/usr/bin/env bash
set -uo pipefail
DATA=${1:?validation folder}
DEVICE=${2:-cuda}
WORK=${VTR_WORK:?set VTR_WORK (see repro/setup_local.sh)}
export VTR_WORK
REPO=$(cd "$(dirname "$0")/.." && pwd)
R="$REPO/repro/official/run_official.py"
LOG="$REPO/repro/logs/imagenet_curve_tb"
OUT="$REPO/repro/results_imagenet_curve_tb_official.csv"
mkdir -p "$LOG"
P04="$WORK/venv_timm04/bin/python"
BATCH=${BATCH:-64}; WORKERS=${WORKERS:-8}

off() {
  local tag="$1_$2_$3"
  echo "$(date '+%F %T') start $tag" | tee -a "$LOG/progress.txt"
  if "$P04" "$R" --model "$1" --method "$2" --setting "$3" --data "$DATA" --device "$DEVICE" \
       --batch "$BATCH" --workers "$WORKERS" --out "$OUT" > "$LOG/$tag.log" 2>&1; then
    echo "$(date '+%F %T') done  $tag" | tee -a "$LOG/progress.txt"
  else
    echo "$(date '+%F %T') FAILED $tag (see $LOG/$tag.log)" | tee -a "$LOG/progress.txt"
  fi
}

for m in deit_tiny_patch16_224 deit_base_patch16_224; do
  for r in 4 8 13 19; do off $m tome $r; done
  for k in 0.9 0.8 0.6; do off $m evit $k; done
done
echo "$(date '+%F %T') ALL DONE" | tee -a "$LOG/progress.txt"
