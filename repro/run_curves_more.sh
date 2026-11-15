#!/usr/bin/env bash
set -uo pipefail
DATA=${1:?validation folder}
WORK=${VTR_WORK:?set VTR_WORK}; export VTR_WORK
REPO=$(cd "$(dirname "$0")/.." && pwd)
R="$REPO/repro/official/run_official.py"; LOG="$REPO/repro/logs/imagenet_curve"; mkdir -p "$LOG" "$REPO/repro/logs/resources"
P04="$WORK/venv_timm04/bin/python"; P067="$WORK/venv_timm067/bin/python"
P092="$WORK/venv_timm092/bin/python"; PB="$WORK/venv_base/bin/python"
[ -x "$WORK/venv_atc/bin/python" ] && PA="$WORK/venv_atc/bin/python" || PA="$PB"
both() {
  local tag="deit_small_patch16_224_$2_$3"
  "$1" "$R" --model deit_small_patch16_224 --method "$2" --setting "$3" --data "$DATA" --device cuda --batch 64 --workers 8 \
    --out "$REPO/repro/results_imagenet_curve_official.csv" > "$LOG/more_$tag.log" 2>&1 || { echo "FAILED acc $2 $3"; return; }
  for rep in 1 2 3; do
    "$1" "$R" --model deit_small_patch16_224 --method "$2" --setting "$3" --data "$DATA" --device cuda --throughput 256 \
      --out "$REPO/repro/resources_gpu_throughput_curve.csv" > "$REPO/repro/logs/resources/curve_${2}_${3}_$rep.log" 2>&1 || echo "FAILED thr $2 $3 $rep"
  done
  echo "$(date '+%T') $2 $3 $(grep -o "'top1': [0-9.]*, 'gmacs': [0-9.]*" "$LOG/more_$tag.log" | tail -1) $(grep -o "'img_per_s': [0-9.]*" "$REPO/repro/logs/resources/curve_${2}_${3}_3.log")"
}
for s in 0.06 0.12 0.2 0.26 0.31 0.36; do both "$PB" reprshift $s; done
for s in 0.85 0.72 0.6 0.52 0.45 0.38; do both "$P067" papr $s; done
for s in 4 8 11 13 16 19; do both "$P092" gtp $s; done
for s in 15 30 40 50 60 70; do both "$P04" ppt $s; done
for s in 0.9 0.8 0.7 0.6 0.5 0.4; do both "$PA" atc $s; done
for s in 197 160 130 100 80; do both "$P04" ats $s; done
echo "MORE DONE $(date '+%F %T')"
