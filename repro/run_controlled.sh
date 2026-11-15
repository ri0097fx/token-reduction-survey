#!/usr/bin/env bash
set -uo pipefail
NAME=${1:?dataset name}
DATA=${2:?validation folder}
DEVICE=${3:-cpu}
MODELS=${4:-"deit_small_patch16_224 deit_tiny_patch16_224 deit_base_patch16_224"}
WORK=${VTR_WORK:?set VTR_WORK (see repro/setup_local.sh)}
export VTR_WORK
REPO=$(cd "$(dirname "$0")/.." && pwd)
R="$REPO/repro/official/run_official.py"
O="$REPO/repro/results_${NAME}_official.csv"
LOG="$REPO/repro/logs/$NAME"
mkdir -p "$LOG"
P04="$WORK/venv_timm04/bin/python"; P067="$WORK/venv_timm067/bin/python"
P092="$WORK/venv_timm092/bin/python"; PB="$WORK/venv_base/bin/python"
[ -x "$WORK/venv_atc/bin/python" ] && PA="$WORK/venv_atc/bin/python" || PA="$PB"
BATCH=${BATCH:-64}; WORKERS=${WORKERS:-8}
LIMIT_ARG=""; [ -n "${LIMIT:-}" ] && LIMIT_ARG="--limit $LIMIT"

run() {
  local tag="$2_$3_${4:-default}"
  echo "$(date '+%F %T') start $tag" | tee -a "$LOG/progress.txt"
  if "$1" "$R" --model "$2" --method "$3" --setting "${4:-}" --data "$DATA" --device "$DEVICE" \
       --batch "$BATCH" --workers "$WORKERS" $LIMIT_ARG --out "$O" > "$LOG/$tag.log" 2>&1; then
    echo "$(date '+%F %T') done  $tag $(grep '^{' "$LOG/$tag.log" | tail -1)" | tee -a "$LOG/progress.txt"
  else
    echo "$(date '+%F %T') FAILED $tag (see $LOG/$tag.log)" | tee -a "$LOG/progress.txt"
  fi
}

for M in $MODELS; do
  case $M in deit_small*) D1=2.9; D2=2.3;; deit_tiny*) D1=0.8; D2=0.6;; deit_base*) D1=11.5; D2=8.7;; esac
  run "$P04" "$M" none ""
  run "$P04" "$M" diffrate "$D1"; run "$P04" "$M" diffrate "$D2"
  run "$P04" "$M" tome 11;        run "$P04" "$M" tome 16
  run "$P04" "$M" mctf 13;        run "$P04" "$M" mctf 19
  run "$P04" "$M" pitome 0.925;   run "$P04" "$M" pitome 0.875
  run "$P04" "$M" ppt 50
  run "$PA"  "$M" atc 0.7;        run "$PA"  "$M" atc 0.5
  run "$P04" "$M" evit 0.7;       run "$P04" "$M" evit 0.5
  run "$P04" "$M" topk 0.7;       run "$P04" "$M" topk 0.5
  run "$P092" "$M" gtp 11;        run "$P092" "$M" gtp 16
  run "$PB"  "$M" reprshift 0.2;  run "$PB"  "$M" reprshift 0.31
  run "$P067" "$M" papr 0.6;      run "$P067" "$M" papr 0.45
  [ "$M" = deit_small_patch16_224 ] && run "$P04" "$M" ats ""
  tag="${M}_reimplementations"
  echo "$(date '+%F %T') start $tag" | tee -a "$LOG/progress.txt"
  SETS="--setting none: --setting tofu:r=11 --setting tofu:r=16 --setting token_pooling:r=14 --setting token_pooling:r=19 \
--setting long_bat:keep_rate=0.7 --setting long_bat:keep_rate=0.5 --setting zero_tprune:retention=1/0.8/0.8/0.8/1 \
--setting zero_tprune:retention=1/0.63/0.63/0.63/1 --setting etps:keep_rate=0.7 --setting etps:keep_rate=0.5"
  ok=1
  (cd "$REPO/repro" && "$PB" evaluate.py --model "$M.fb_in1k" --data "$DATA" --device "$DEVICE" --batch "$BATCH" \
     --workers "$WORKERS" $LIMIT_ARG --out "results_${NAME}.csv" $SETS > "$LOG/$tag.log" 2>&1) || ok=0
  if [ $ok = 0 ] && [ "$DEVICE" != cpu ]; then
    echo "$(date '+%F %T') retry $tag on the CPU" | tee -a "$LOG/progress.txt"
    (cd "$REPO/repro" && "$PB" evaluate.py --model "$M.fb_in1k" --data "$DATA" --device cpu --batch "$BATCH" \
       --workers "$WORKERS" $LIMIT_ARG --out "results_${NAME}.csv" $SETS > "$LOG/${tag}_cpu.log" 2>&1) && ok=1
  fi
  [ $ok = 1 ] && echo "$(date '+%F %T') done  $tag" | tee -a "$LOG/progress.txt" \
              || echo "$(date '+%F %T') FAILED $tag" | tee -a "$LOG/progress.txt"
done
echo "$(date '+%F %T') ALL DONE" | tee -a "$LOG/progress.txt"
[ -z "${NOBUILD:-}" ] && { "$PB" "$REPO/tools/build_controlled.py" "$NAME" || true; }
