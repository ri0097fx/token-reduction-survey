#!/bin/bash
W=${VTR_WORK:?}
cd "$(dirname "$0")/../.."
while pgrep -f 'queue_segmenter.sh|queue_segmenter2.sh' > /dev/null; do sleep 30; done
$W/venv_timm04/bin/python -I repro/dense/run_segmenter_ade.py --model l --runs none,algm:0.98,algm:0.95,algm:0.92,algm:0.88,algm:0.85,expedite:1296:9,expedite:1024:9,expedite:784:9,expedite:576:9,expedite:400:9,tome:40,tome:80,tome:120,tome:160,tomeu:40,tomeu:80,tomeu:120,tomeu:160,topk:0.9,topk:0.8,topk:0.7,topk:0.6,topk:0.5 2>&1 | grep -v -i warning
echo "SEGMENTER3 DONE $(date '+%F %T')"
