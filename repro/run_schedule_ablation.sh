#!/usr/bin/env bash
set -uo pipefail
DATA=${1:?validation folder}
DEVICE=${2:-cuda}
WORK=${VTR_WORK:?set VTR_WORK (see repro/setup_local.sh)}
export VTR_WORK
REPO=$(cd "$(dirname "$0")/.." && pwd)
R="$REPO/repro/official/run_official.py"
LOG="$REPO/repro/logs/imagenet_schedule"
OUT="$REPO/repro/results_imagenet_schedule.csv"
mkdir -p "$LOG"
P04="$WORK/venv_timm04/bin/python"
BATCH=${BATCH:-64}; WORKERS=${WORKERS:-8}

off() {
  local tag="$1_$2_${3//[@,]/_}"
  echo "$(date '+%F %T') start $tag" | tee -a "$LOG/progress.txt"
  if "$P04" "$R" --model "$1" --method "$2" --setting "$3" --data "$DATA" --device "$DEVICE" \
       --batch "$BATCH" --workers "$WORKERS" --out "$OUT" > "$LOG/$tag.log" 2>&1; then
    echo "$(date '+%F %T') done  $tag" | tee -a "$LOG/progress.txt"
  else
    echo "$(date '+%F %T') FAILED $tag (see $LOG/$tag.log)" | tee -a "$LOG/progress.txt"
  fi
}

for m in deit_small_patch16_224 deit_tiny_patch16_224 deit_base_patch16_224; do
  for r in 20 35 50 65 80 98; do off $m tome "$r@4,7,10"; done
  for k in 0.97 0.95 0.93 0.91 0.89 0.87 0.85; do off $m evit "$k@1-12"; done
  for k in 0.95 0.92 0.89 0.86 0.83 0.80 0.77; do off $m evit "$k@4-12"; done
done
echo "$(date '+%F %T') ALL DONE" | tee -a "$LOG/progress.txt"
