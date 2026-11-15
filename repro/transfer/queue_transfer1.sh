#!/bin/bash
cd "$(dirname "$0")/../.."
: "${VTR_WORK:?}"
W=$VTR_WORK; P04=$W/venv_timm04/bin/python; PB=$W/venv_base/bin/python
LR=${1:-5e-5}
L=repro/logs/transfer; mkdir -p $L
for d in cub200 cars pets flowers indoor67 cifar10 cifar100; do
  echo "$(date '+%F %T') start finetune $d lr=$LR" >> $L/progress.txt
  "$P04" repro/transfer/finetune_transfer.py --dataset $d --lr $LR --workers 8 > "$L/finetune_$d.log" 2>&1
  echo "$(date '+%F %T') exit $? finetune $d" >> $L/progress.txt
  echo "$(date '+%F %T') start eval $d" >> $L/progress.txt
  "$PB" repro/transfer/eval_transfer.py --datasets $d --workers 8 > "$L/eval_$d.log" 2>&1
  echo "$(date '+%F %T') exit $? eval $d" >> $L/progress.txt
done
echo "$(date '+%F %T') TRANSFER1 DONE" >> $L/progress.txt
