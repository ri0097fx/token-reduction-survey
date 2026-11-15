#!/bin/bash
W=${VTR_WORK:?}
export HF_HOME=$W/hf_cache VTR_WORK=$W
cd "$(dirname "$0")/../.."
$W/venv_hf/bin/python -I repro/dense/run_yolos_coco.py --runs none,topk:0.9,topk:0.8,topk:0.7,topk:0.6,topk:0.5,tome:50,tome:100,tome:150,tome:200,tome:250,tome:300 2>&1 | grep -v -i 'warn\|slow'
echo "YOLOS DONE $(date '+%F %T')"
