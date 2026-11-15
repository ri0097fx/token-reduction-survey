#!/usr/bin/env bash
set -uo pipefail
DATA=${1:?validation folder}
DEVICE=${2:-cuda}
WORK=${VTR_WORK:?set VTR_WORK (see repro/setup_local.sh)}
export VTR_WORK
REPO=$(cd "$(dirname "$0")/.." && pwd)
R="$REPO/repro/official/run_official.py"
LOG="$REPO/repro/logs/imagenet_tricks"
OUT="$REPO/repro/results_imagenet_tricks.csv"
PRED="$REPO/repro/preds"
mkdir -p "$LOG" "$PRED"
P04="$WORK/venv_timm04/bin/python"
BATCH=${BATCH:-64}; WORKERS=${WORKERS:-8}

run() {
  local py="$1" model="$2" method="$3" setting="$4" tag="$5"; shift 5
  echo "$(date '+%F %T') start $tag" | tee -a "$LOG/progress.txt"
  if "$py" "$R" --model "$model" --method "$method" --setting "$setting" --data "$DATA" --device "$DEVICE" \
       --batch "$BATCH" --workers "$WORKERS" --out "$OUT" --save-preds "$PRED/$tag.npy" "$@" > "$LOG/$tag.log" 2>&1; then
    echo "$(date '+%F %T') done  $tag" | tee -a "$LOG/progress.txt"
  else
    echo "$(date '+%F %T') FAILED $tag (see $LOG/$tag.log)" | tee -a "$LOG/progress.txt"
  fi
}

S=deit_small_patch16_224; T=deit_tiny_patch16_224; B=deit_base_patch16_224

for m in $S $T $B; do
  for r in 11 16; do
    run $P04 $m tome "$r" "prop_${m}_r$r"
    run $P04 $m tome "$r,noprop" "noprop_${m}_r$r"
  done
done

for i in 1 2 3; do run $P04 $S tome 13 "repeat${i}_tome_r13"; done
for i in 1 2; do run $P04 $S evit 0.7 "repeat${i}_evit_0.7"; done
for i in 1 2; do run $P04 $S diffrate 2.9 "repeat${i}_diffrate_2.9"; done
for i in 1 2; do run $P04 $S tome 13 "det${i}_tome_r13" --deterministic; done

for d in fp32 fp16 bf16; do
  run $P04 $S none "" "${d}_none" --dtype $d
  run $P04 $S tome 13 "${d}_tome_r13" --dtype $d
  run $P04 $S evit 0.7 "${d}_evit_0.7" --dtype $d
  run $P04 $S diffrate 2.9 "${d}_diffrate_2.9" --dtype $d
done
run $P04 $S tome "13,metric32" "bf16m32_tome_r13" --dtype bf16
run $P04 $S tome "13,metric32" "fp16m32_tome_r13" --dtype fp16
echo "$(date '+%F %T') ALL DONE" | tee -a "$LOG/progress.txt"
