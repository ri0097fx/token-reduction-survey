#!/bin/bash
W=${VTR_WORK:?}
cd "$(dirname "$0")/../.."
RUNS="algm:0.95,algm:0.92,algm:0.88,algm:0.85,algm:0.8,algm:0.75,tome:40,tome:60,tome:80,tome:100,tome:130,tomeu:40,tomeu:60,tomeu:80,tomeu:100,tomeu:130,topk:0.9,topk:0.8,topk:0.7,topk:0.6,topk:0.5"
for m in s b; do
  $W/venv_timm04/bin/python -I repro/dense/run_segmenter_ade.py --model $m --runs $RUNS 2>&1 | grep -v -i warning
done
echo "SEGMENTER DONE $(date '+%F %T')"
