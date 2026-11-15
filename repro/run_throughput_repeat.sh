#!/usr/bin/env bash
set -uo pipefail
DATA=${1:?validation folder}
WORK=${VTR_WORK:?set VTR_WORK}; export VTR_WORK
REPO=$(cd "$(dirname "$0")/.." && pwd)
R="$REPO/repro/official/run_official.py"; LOG="$REPO/repro/logs/resources"; mkdir -p "$LOG"
P04="$WORK/venv_timm04/bin/python"; P067="$WORK/venv_timm067/bin/python"
P092="$WORK/venv_timm092/bin/python"; PB="$WORK/venv_base/bin/python"
[ -x "$WORK/venv_atc/bin/python" ] && PA="$WORK/venv_atc/bin/python" || PA="$PB"
OUT="$REPO/repro/resources_gpu_throughput_repeat.csv"
run() { for rep in 1 2 3; do
  "$1" "$R" --model deit_small_patch16_224 --method "$2" --setting "${3:-}" --data "$DATA" --device cuda --throughput 256 \
    --out "$OUT" > "$LOG/repeat_${2}_${3:-default}_$rep.log" 2>&1 || echo "FAILED $2 $3 $rep"
  done; echo "$(date '+%T') $2 $3 $(grep -o "'img_per_s': [0-9.]*" "$LOG/repeat_${2}_${3:-default}_3.log")"; }
run "$P04" none ""
run "$P04" diffrate 2.9;  run "$P04" diffrate 2.3
run "$P04" tome 11;       run "$P04" tome 16
run "$P04" mctf 13;       run "$P04" mctf 19
run "$P04" pitome 0.925;  run "$P04" pitome 0.875
run "$P04" ppt 50
run "$PA"  atc 0.7;       run "$PA"  atc 0.5
run "$P04" topk 0.7;      run "$P04" topk 0.5
run "$P04" evit 0.7;      run "$P04" evit 0.5
run "$P092" gtp 11;       run "$P092" gtp 16
run "$PB"  reprshift 0.2; run "$PB"  reprshift 0.31
run "$P067" papr 0.6;     run "$P067" papr 0.45
run "$P04" ats ""
echo "REPEAT DONE $(date '+%F %T')"
