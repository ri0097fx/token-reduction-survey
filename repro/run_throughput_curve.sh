#!/usr/bin/env bash
set -uo pipefail
DATA=${1:?validation folder}
WORK=${VTR_WORK:?set VTR_WORK}; export VTR_WORK
REPO=$(cd "$(dirname "$0")/.." && pwd)
R="$REPO/repro/official/run_official.py"; LOG="$REPO/repro/logs/resources"; mkdir -p "$LOG"
P04="$WORK/venv_timm04/bin/python"
OUT="$REPO/repro/resources_gpu_throughput_curve.csv"
run() { for rep in 1 2 3; do
  "$P04" "$R" --model deit_small_patch16_224 --method "$1" --setting "${2:-}" --data "$DATA" --device cuda --throughput 256 \
    --out "$OUT" > "$LOG/curve_${1}_${2:-default}_$rep.log" 2>&1 || echo "FAILED $1 $2 $rep"
  done; echo "$(date '+%T') $1 $2 $(grep -o "'img_per_s': [0-9.]*" "$LOG/curve_${1}_${2:-default}_3.log")"; }
run none ""
for s in 4 8 11 13 16 19; do run tome $s; done
for s in 4 8 11 13 16 19; do run mctf $s; done
for s in 0.975 0.95 0.925 0.9 0.875 0.85; do run pitome $s; done
for s in 3.1 2.9 2.7 2.5 2.3; do run diffrate $s; done
for s in 0.9 0.8 0.7 0.6 0.5; do run evit $s; done
for s in 0.9 0.8 0.7 0.6 0.5; do run topk $s; done
run none ""
echo "CURVE DONE $(date '+%F %T')"
