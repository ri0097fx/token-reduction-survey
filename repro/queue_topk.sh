#!/usr/bin/env bash
set -uo pipefail
DATA=${1:?validation folder}
WORK=${VTR_WORK:?}; export VTR_WORK
REPO=$(cd "$(dirname "$0")/.." && pwd)
R="$REPO/repro/official/run_official.py"
P04="$WORK/venv_timm04/bin/python"
LOG="$REPO/repro/logs/topk"; mkdir -p "$LOG" "$WORK/logits"
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG/progress.txt"; }
one() {
  local tag="$1"; shift
  say "start $tag"
  if "$P04" "$R" "$@" > "$LOG/$tag.log" 2>&1; then say "done  $tag $(grep '^{' "$LOG/$tag.log" | tail -1)"
  else say "FAILED $tag"; fi
}
for M in deit_small_patch16_224 deit_tiny_patch16_224 deit_base_patch16_224; do
  for k in 0.7 0.5; do
    one "levels_${M}_$k" --model $M --method topk --setting $k --data "$DATA" --device cuda --batch 64 --workers 8 \
        --out "$REPO/repro/results_imagenet_official.csv"
  done
done
for k in 0.9 0.8 0.7 0.6 0.5; do
  one "curve_$k" --model deit_small_patch16_224 --method topk --setting $k --data "$DATA" --device cuda --batch 64 \
      --workers 8 --out "$REPO/repro/results_imagenet_curve_official.csv"
done
for d in fp32 fp16 bf16; do
  one "precision_${d}" --model deit_small_patch16_224 --method topk --setting 0.7 --data "$DATA" --device cuda \
      --batch 64 --workers 8 --out "$REPO/repro/results_imagenet_precision.csv" --save-logits "$WORK/logits/${d}_topk_0.7.npy" --dtype $d
done
say "start grid"
"$P04" "$REPO/repro/run_schedule_grid.py" "$DATA" --methods topk --batch 64 --workers 8 >> "$LOG/grid.log" 2>&1; say "grid rc=$?"
for k in 0.7 0.5; do
  for mode in gpu_throughput gpu_latency cpu_latency; do
    case $mode in
      gpu_throughput) extra="--device cuda --throughput 256";;
      gpu_latency)    extra="--device cuda --latency 50";;
      cpu_latency)    extra="--device cpu --latency 50 --threads 16";;
    esac
    one "${mode}_$k" --model deit_small_patch16_224 --method topk --setting $k --data "$DATA" $extra \
        --out "$REPO/repro/resources_$mode.csv"
  done
done
say "TOPK DONE"
