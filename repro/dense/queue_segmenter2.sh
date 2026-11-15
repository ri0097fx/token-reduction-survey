#!/bin/bash
W=${VTR_WORK:?}
cd "$(dirname "$0")/../.."
while pgrep -f queue_segmenter.sh > /dev/null; do sleep 30; done
for m in s b; do
  $W/venv_timm04/bin/python -I repro/dense/run_segmenter_ade.py --model $m --runs expedite:784:4,expedite:576:4,expedite:400:4,expedite:256:4 2>&1 | grep -v -i warning
done
echo "SEGMENTER2 DONE $(date '+%F %T')"
