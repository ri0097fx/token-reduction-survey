#!/usr/bin/env bash
set -uo pipefail
DATA=${1:?validation folder}
DEVICE=${2:-cuda}
WORK=${VTR_WORK:?set VTR_WORK}
export VTR_WORK
REPO=$(cd "$(dirname "$0")/.." && pwd)
R="$REPO/repro/official/run_official.py"
LOG="$REPO/repro/logs/imagenet_tricks"
OUT="$REPO/repro/results_imagenet_tricks.csv"
PRED="$REPO/repro/preds"
mkdir -p "$LOG" "$PRED"
P04="$WORK/venv_timm04/bin/python"
run() {
  local tag="$4"
  echo "$(date '+%F %T') start $tag" | tee -a "$LOG/progress.txt"
  if "$P04" "$R" --model "$1" --method "$2" --setting "$3" --data "$DATA" --device "$DEVICE" --batch "$5" --workers 8 \
       --out "$OUT" --save-preds "$PRED/$tag.npy" > "$LOG/$tag.log" 2>&1; then
    echo "$(date '+%F %T') done  $tag" | tee -a "$LOG/progress.txt"
  else
    echo "$(date '+%F %T') FAILED $tag (see $LOG/$tag.log)" | tee -a "$LOG/progress.txt"
  fi
}
for spec in "mae_vit_base_patch16:11:16:64" "mae_vit_large_patch16:6:8:32" "mae_vit_huge_patch14:5:8:16"; do
  IFS=: read -r M R1 R2 B <<< "$spec"
  run $M none "" "none_$M" $B
  for r in $R1 $R2; do
    run $M tome "$r,prop" "prop_${M}_r$r" $B
    run $M tome "$r" "noprop_${M}_r$r" $B
  done
done
echo "$(date '+%F %T') MAE DONE" | tee -a "$LOG/progress.txt"
