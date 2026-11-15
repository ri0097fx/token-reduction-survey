#!/bin/bash
cd "$(dirname "$0")/../.."
: "${VTR_WORK:?}"
W=$VTR_WORK; D=${VTR_DATASETS:?}/imagenet-1k/imagenet
P04=$W/venv_timm04/bin/python; PB=$W/venv_base/bin/python
L=repro/logs/finetune; mkdir -p $L
run() { echo "$(date '+%F %T') start $2 $3" >> $L/progress.txt
  "$1" repro/finetune/run_finetune.py --method $2 --setting "$3" --data $D > "$L/$2_$3.log" 2>&1
  rc=$?; echo "$(date '+%F %T') exit $rc $2 $3" >> $L/progress.txt; }
dyvit() { echo "$(date '+%F %T') start dynamicvit $*" >> $L/progress.txt
  "$P04" repro/finetune/run_finetune_dyvit.py --data $D "$@" > "$L/dynamicvit_$(echo "$*" | tr -d ' -').log" 2>&1
  rc=$?; echo "$(date '+%F %T') exit $rc dynamicvit $*" >> $L/progress.txt; }
learned() { echo "$(date '+%F %T') start $*" >> $L/progress.txt
  "$P04" repro/finetune/run_finetune_learned.py --data $D "$@" > "$L/$(echo "$*" | tr -d ' -' | tr -d '=').log" 2>&1
  rc=$?; echo "$(date '+%F %T') exit $rc $*" >> $L/progress.txt; }
run $P04 none ""
run $P04 topk 0.5
run $P04 evit 0.5
run $P04 tome 16
run $P04 diffrate 2.3
run $P04 mctf 19
run $PB reprshift 0.31
run $P04 topk 0.7
run $P04 evit 0.7
run $P04 tome 11
run $P04 diffrate 2.9
run $P04 mctf 13
run $PB reprshift 0.2
dyvit --rate 0.7
dyvit --rate 0.5
dyvit --rate 0.7 --distill
dyvit --rate 0.5 --distill
dyvit --rate 0.7 --freeze
dyvit --rate 0.5 --freeze
dyvit --rate 0.7 --freeze --distill
dyvit --rate 0.5 --freeze --distill
learned --method dtps --rate 0.7
learned --method dtps --rate 0.5
learned --method dtps --rate 0.7 --freeze
learned --method dtps --rate 0.5 --freeze
learned --method avit
learned --method avit --lr 1.5e-4
echo "$(date '+%F %T') FINETUNE4 DONE" >> $L/progress.txt
