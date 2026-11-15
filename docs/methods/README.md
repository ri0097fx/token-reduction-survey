# Method table: schema and extraction rules

One filled table per group (`g1_*_filled.csv` etc.). The filled tables are merged into `docs/methods_table.csv`, which feeds the appendix table of the survey.

## Columns of the filled table

| column | allowed values / format |
|---|---|
| key | BibTeX key from paper/references.bib (copy from the group CSV) |
| name | short method name as the paper uses it (e.g. ToMe, DynamicViT). Empty if the paper has no name |
| year | publication year |
| target | one or more of: `classification`, `dense` (segmentation / detection / other per-token outputs), `video`, `vlm` (image--text models or multimodal LLMs), `generation` (diffusion and other generators), `nlp`, `other`. Separate several with `;` |
| operation | what happens to the reduced tokens. One or more of: `prune` (dropped), `fuse` (dropped tokens are averaged into one extra token), `merge` (similar tokens are combined pairwise or by matching), `cluster` (tokens are grouped by clustering / pooling into fewer tokens), `halt` (a token stops being updated but stays in the sequence), `resample` (new tokens are generated from the old ones, e.g. TokenLearner), `resolution` (the number of tokens is changed through patch size or input resolution), `hybrid` (the paper combines prune and merge or chooses between them), `reactivate` (reduced tokens are restored later for a dense head), `other` |
| criterion | how tokens are scored or paired. One or more of: `cls-attention` (attention from the class token or similar attention statistics), `learned-predictor` (a trained module predicts keep / drop), `similarity` (feature similarity between tokens), `norm` (token norm or magnitude), `gradient` (gradient, Hessian or Taylor-based importance), `diversity` (keeps a diverse set), `query-aware` (text or task query decides), `frequency`, `random`, `none` (no scoring, e.g. fixed pattern), `other` |
| schedule | how many tokens are reduced and where. One of: `fixed` (a fixed count or keep ratio per block, same for every image), `learned-rate` (per-block rate learned or searched offline, same for every image), `threshold` (a threshold makes the count vary per image), `halting` (per-token stopping), `budget` (chosen from a FLOPs / latency budget by an explicit procedure), `input-adaptive` (per-image decision by a confidence or difficulty estimate, e.g. early exit over resolutions), `other` |
| training | one of: `training-free` (works on a frozen pre-trained model without any training; may need calibration data), `light` (trains only a small added module or a few parameters, backbone frozen), `fine-tune` (fine-tunes the backbone), `scratch` (must train from scratch), `mixed` (paper reports both training-free and fine-tuned variants; say which in notes) |
| backbones | backbones evaluated, separated by `;` (e.g. `DeiT-S; DeiT-B; LV-ViT`) |
| main_result | one headline number copied from the paper in the form `<backbone>: <metric> <value> at <cost>` (e.g. `DeiT-S: top-1 79.8 at 2.9 GFLOPs (baseline 79.8 at 4.6)`). Prefer DeiT-S on ImageNet-1k if present. Empty if not applicable |
| deits_top1 | DeiT-S ImageNet-1k top-1 (%) of the main setting, if reported |
| deits_gflops | GFLOPs of that setting |
| deits_throughput | throughput (img/s) of that setting, if reported, with the GPU in notes |
| latency_reported | `yes` if the paper reports wall-clock latency or throughput, `no` if only FLOPs |
| batch1 | `yes` if latency at batch size 1 is reported |
| code | URL of the official code or `none` |
| notes | free text, short |
| evidence | one or two short quotes from the paper (abstract or method section) that support `operation`, `criterion`, `schedule` and `training`. Put the section name in parentheses. Quotes only, no paraphrase |
| source | URL of the text you read (arXiv abs / html / pdf, or the publisher page) |

## Rules

- Read the paper. The abstract alone is not enough for `training`, `schedule` and the DeiT-S numbers. Use the arXiv API for the abstract and download the arXiv PDF (`https://arxiv.org/pdf/<id>`) and run `pdftotext -layout` on it, or use `https://arxiv.org/html/<id>` when it exists. Put downloads in your own directory under the scratchpad, never in the repository.
- If a value cannot be found in the paper, write `?` and say in notes what is missing. Never guess.
- Keep each cell short. Quotes in `evidence` are verbatim.
- Save the filled table as `docs/methods/<group>_filled.csv` with exactly the columns above in that order, UTF-8, quoted fields.
