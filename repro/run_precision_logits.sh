#!/usr/bin/env bash
set -uo pipefail
DATA=${1:?validation folder}
DEVICE=${2:-cuda}
WORK=${VTR_WORK:?set VTR_WORK (see repro/setup_local.sh)}
export VTR_WORK
REPO=$(cd "$(dirname "$0")/.." && pwd)
R="$REPO/repro/official/run_official.py"
LOG="$REPO/repro/logs/precision"
OUT="$REPO/repro/results_imagenet_precision.csv"
LGT="$WORK/logits"
mkdir -p "$LOG" "$LGT"
P04="$WORK/venv_timm04/bin/python"
BATCH=${BATCH:-64}; WORKERS=${WORKERS:-8}

run() {
  local model="$1" method="$2" setting="$3" tag="$4"; shift 4
  echo "$(date '+%F %T') start $tag" | tee -a "$LOG/progress.txt"
  if "$P04" "$R" --model "$model" --method "$method" --setting "$setting" --data "$DATA" --device "$DEVICE" \
       --batch "$BATCH" --workers "$WORKERS" --out "$OUT" --save-logits "$LGT/$tag.npy" "$@" > "$LOG/$tag.log" 2>&1; then
    echo "$(date '+%F %T') done  $tag" | tee -a "$LOG/progress.txt"
  else
    echo "$(date '+%F %T') FAILED $tag (see $LOG/$tag.log)" | tee -a "$LOG/progress.txt"
  fi
}

S=deit_small_patch16_224
for d in fp32 fp16 bf16; do
  run $S none "" "${d}_none" --dtype $d
  run $S tome 13 "${d}_tome_r13" --dtype $d
  run $S evit 0.7 "${d}_evit_0.7" --dtype $d
  run $S topk 0.7 "${d}_topk_0.7" --dtype $d
  run $S diffrate 2.9 "${d}_diffrate_2.9" --dtype $d
done
run $S tome "13,metric32" "bf16m32_tome_r13" --dtype bf16
run $S tome "13,metric32" "fp16m32_tome_r13" --dtype fp16
echo "$(date '+%F %T') ALL DONE" | tee -a "$LOG/progress.txt"
