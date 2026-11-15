#!/bin/bash
cd "$(dirname "$0")/../.."
: "${VTR_WORK:?}"
W=$VTR_WORK; D=${VTR_DATASETS:?}/imagenet-1k/imagenet
P04=$W/venv_timm04/bin/python; PB=$W/venv_base/bin/python
L=repro/logs/finetune; mkdir -p $L
run() { echo "$(date '+%F %T') start $2 $3" >> $L/progress.txt
  "$1" repro/finetune/run_finetune.py --method $2 --setting "$3" --data $D > "$L/$2_$3.log" 2>&1
  rc=$?; echo "$(date '+%F %T') exit $rc $2 $3" >> $L/progress.txt; }
run $P04 tome 16
run $P04 evit 0.5
run $P04 diffrate 2.3
run $P04 mctf 19
run $PB reprshift 0.31
run $P04 none ""
echo "$(date '+%F %T') FINETUNE DONE" >> $L/progress.txt
