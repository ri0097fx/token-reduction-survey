# Code of the comparison

Released code of each method runs unchanged through `official/run_official.py`. `official/repos.txt` lists the repositories and commits. Methods without released code are implemented in `tokred/methods.py`.

## Setup

- `VTR_WORK`: folder with the cloned repositories (`official/`), the weights (`weights/`), and the Python environments (`setup_local.sh`).
- `VTR_DATASETS`: folder with the datasets (ImageNet-1k, COCO, ADE20K, and the seven small datasets).

## Changes needed to run released code

| Method | Change |
|---|---|
| DiffRate, ToMe, PPT, MCTF, ATC, PaPr | none |
| Top-K | the EViT code with `fuse_token=False` |
| EViT | the model function is called directly, because its registered names clash with timm |
| ATS | a stub of the fvcore registry, and the package is imported without its `__init__.py` |
| PiToMe | only `patch/deit.py` is imported |
| Representation Shift | a stub for the `flash_attn` import, which the released code does not use |
| GTP-ViT | see `official/run_official.py` |
| ALGM | none |
| Expedite | its clustering and reconstruction operators are called from our forward |

All methods load the same timm weights and use the same evaluation transform (bicubic resize to 248, center crop 224).

## Experiments

| Script | Result files |
|---|---|
| `run_controlled.sh`, `queue_topk.sh` | `results_imagenet_official.csv`, `results_imagenet.csv` |
| `run_curves.sh`, `run_curves_more.sh`, `run_curves_tb.sh` | `results_imagenet_curve_official.csv` |
| `run_schedule_grid.py`, `run_schedule_methods.py` | `results_imagenet_schedule_grid.csv`, `results_imagenet_schedule_methods.csv` |
| `finetune/` | `results_finetune.csv` |
| `run_backbones.py`, `run_backbones_resolution.sh` | `results_imagenet_backbones.csv` |
| `transfer/` | `results_transfer_*.csv` |
| `dense/` | `results_ade20k_segmenter.csv`, `results_coco_yolos.csv`, `results_coco_vitdet.csv` |
| `run_resources.sh`, `run_throughput_repeat.sh`, `run_throughput_curve.sh` | `resources_*.csv` |
| `bench_compile.py`, `bench_sync.py`, `bench_sdpa.py` | `bench_*.csv` |
| `run_tricks_accuracy.sh`, `run_precision_logits.sh`, `precision_kl.py`, `run_mae_propattn.sh` | `results_imagenet_tricks.csv`, `results_imagenet_precision*.csv`, `preds/` |

The scripts in `tools/` (`build_*.py`) generate the tables and the figures from these files.
