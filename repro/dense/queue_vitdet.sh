#!/bin/bash
cd "$(dirname "$0")/../.."
export DETECTRON2_DATASETS=${VTR_DATASETS:?}
~/anaconda3/envs/d2env/bin/python repro/dense/run_vitdet_coco.py --runs none,attn:0.25,attn:0.5,attn:0.75,attn_mlp:0.25,attn_mlp:0.5,attn_mlp:0.75 2>&1 | grep -E "box_ap|Error|error"
echo "VITDET DONE $(date '+%F %T')"
