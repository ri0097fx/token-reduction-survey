#!/bin/bash
cd "$(dirname "$0")/../.."
: "${VTR_WORK:?}"
W=$VTR_WORK; D=${VTR_DATASETS:?}/imagenet-1k/imagenet
P04=$W/venv_timm04/bin/python
L=repro/logs/finetune; mkdir -p $L
until grep -q "FINETUNE2 DONE" $L/progress.txt; do sleep 120; done
run() { echo "$(date '+%F %T') start $*" >> $L/progress.txt
  "$P04" repro/finetune/run_finetune_learned.py --data $D "$@" > "$L/$(echo "$*" | tr -d ' -' | tr -d '=').log" 2>&1
  rc=$?; echo "$(date '+%F %T') exit $rc $*" >> $L/progress.txt; }
run --method dtps --rate 0.7
run --method dtps --rate 0.5
run --method dtps --rate 0.7 --freeze
run --method dtps --rate 0.5 --freeze
run --method avit
run --method avit --lr 1.5e-4
echo "$(date '+%F %T') FINETUNE3 DONE" >> $L/progress.txt
