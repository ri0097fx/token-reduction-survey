#!/usr/bin/env bash
set -uo pipefail
DATA=${1:?validation folder}
DEVICE=${2:-cuda}
WORK=${VTR_WORK:?set VTR_WORK (see repro/setup_local.sh)}
export VTR_WORK
REPO=$(cd "$(dirname "$0")/.." && pwd)
R="$REPO/repro/official/run_official.py"
LOG="$REPO/repro/logs/imagenet_curve"
mkdir -p "$LOG"
P04="$WORK/venv_timm04/bin/python"; PB="$WORK/venv_base/bin/python"
BATCH=${BATCH:-64}; WORKERS=${WORKERS:-8}

off() {
  local tag="deit_small_patch16_224_$2_$3"
  echo "$(date '+%F %T') start $tag" | tee -a "$LOG/progress.txt"
  if "$1" "$R" --model deit_small_patch16_224 --method "$2" --setting "$3" --data "$DATA" --device "$DEVICE" \
       --batch "$BATCH" --workers "$WORKERS" --out "$REPO/repro/results_imagenet_curve_official.csv" > "$LOG/$tag.log" 2>&1; then
    echo "$(date '+%F %T') done  $tag" | tee -a "$LOG/progress.txt"
  else
    echo "$(date '+%F %T') FAILED $tag (see $LOG/$tag.log)" | tee -a "$LOG/progress.txt"
  fi
}

own() {
  local model=$1 out=$2 tag=$3; shift 3
  echo "$(date '+%F %T') start $tag" | tee -a "$LOG/progress.txt"
  if (cd "$REPO/repro" && "$PB" evaluate.py --model "$model" --data "$DATA" --device "$DEVICE" --batch "$BATCH" \
        --workers "$WORKERS" --out "$out" "$@" > "$LOG/$tag.log" 2>&1); then
    echo "$(date '+%F %T') done  $tag" | tee -a "$LOG/progress.txt"
  else
    echo "$(date '+%F %T') FAILED $tag (see $LOG/$tag.log)" | tee -a "$LOG/progress.txt"
  fi
}

for s in 4 8 11 13 16 19; do off "$P04" tome $s; done
for s in 4 8 11 13 16 19; do off "$P04" mctf $s; done
for s in 0.975 0.95 0.925 0.9 0.875 0.85; do off "$P04" pitome $s; done
for s in 3.1 2.9 2.7 2.5 2.3; do off "$P04" diffrate $s; done
for s in 0.9 0.8 0.7 0.6 0.5; do off "$P04" evit $s; done
for s in 0.9 0.8 0.7 0.6 0.5; do off "$P04" topk $s; done

own deit_small_patch16_224.fb_in1k results_imagenet_curve.csv deit_small_curves \
  --setting none: \
  --setting tofu:r=4 --setting tofu:r=8 --setting tofu:r=11 --setting tofu:r=13 --setting tofu:r=16 --setting tofu:r=19 \
  --setting etps:keep_rate=0.9 --setting etps:keep_rate=0.8 --setting etps:keep_rate=0.7 \
  --setting etps:keep_rate=0.6 --setting etps:keep_rate=0.5 \
  --setting zero_tprune:retention=1/1/1/1/1 --setting zero_tprune:retention=1/0.9/0.9/0.9/1 \
  --setting zero_tprune:retention=1/0.8/0.8/0.8/1 --setting zero_tprune:retention=1/0.7/0.7/0.7/1 \
  --setting zero_tprune:retention=1/0.63/0.63/0.63/1 --setting zero_tprune:retention=1/0.55/0.55/0.55/1

L1=196/195/193/188/169/140/121/110/73/38/7/0
L3=196/194/187/163/118/85/58/47/20/12/2/0
for m in deit_tiny_patch16_224 deit_small_patch16_224 deit_base_patch16_224; do
  own "$m.fb_in1k" results_imagenet.csv "${m}_reimplementations" \
    --setting none: --setting tofu:r=11 --setting tofu:r=16 \
    --setting etps:keep_rate=0.7 --setting etps:keep_rate=0.5 \
    --setting zero_tprune:retention=1/0.8/0.8/0.8/1 --setting zero_tprune:retention=1/0.63/0.63/0.63/1 \
    --setting long_bat:keep_rate=0.7 --setting long_bat:keep_rate=0.5 \
    --setting token_pooling:ks=$L1,prop=1 --setting token_pooling:ks=$L3,prop=1
done
echo "$(date '+%F %T') ALL DONE" | tee -a "$LOG/progress.txt"
