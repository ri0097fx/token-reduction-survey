# Token Reduction in Vision Transformers: A Survey and Empirical Comparison

Data files of the survey by Ryuto Ishibashi and Lin Meng (Ritsumeikan University).
Every table and figure with numbers in the paper is generated from the files below.

## Method table

| File | Content | Used in |
|---|---|---|
| [docs/methods_table.csv](docs/methods_table.csv) | The 173 token reduction methods of the literature search on the five axes of the taxonomy (operation, criterion, schedule, training, target), with the backbones, the main reported result, the reported DeiT-S numbers, flags for reported time and batch size 1, the code URL, and a quote from the paper as evidence for each value | Table 21 (Appendix A), Tables 3 to 5, Figures 4 and 8 |
| [docs/methods/](docs/methods) | The same records in the groups in which they were collected (`*_filled.csv`), the column definitions, and the notes on the bibliography of each group | source of `methods_table.csv` |
| [docs/excluded_methods.txt](docs/excluded_methods.txt) | Papers of the search that are not counted as methods (analysis papers and surveys) | Section 1 |
| [docs/citation_counts.csv](docs/citation_counts.csv) | Citation counts used to select the methods that the text discusses | Section 1 |
| [docs/bib_notes.md](docs/bib_notes.md) | Source of every bibliography entry (publisher record, Crossref, or proceedings page) | References |
| [docs/image_credits.json](docs/image_credits.json) | Authors, Flickr pages, and licenses of the COCO photographs in the figures | Table 23 (Appendix C) |

## Results of the comparison on ImageNet-1k

| File | Content | Used in |
|---|---|---|
| [repro/results_imagenet_official.csv](repro/results_imagenet_official.csv) | Released code of 10 methods and Top-K on DeiT-T, DeiT-S, and DeiT-B at about 65% and 50% of the GMACs, without training | Tables 6 and 22, Figure 14 |
| [repro/results_imagenet.csv](repro/results_imagenet.csv) | Our implementations of the methods without released code, same protocol | Tables 6 and 22 |
| [repro/results_imagenet_curve_official.csv](repro/results_imagenet_curve_official.csv), [repro/results_imagenet_curve.csv](repro/results_imagenet_curve.csv) | Accuracy of DeiT-S over several levels of GMACs, released code and our implementations | Figure 9 |
| [repro/results_imagenet_curve_tb_official.csv](repro/results_imagenet_curve_tb_official.csv) | The same curves on DeiT-T and DeiT-B | read by `tools/build_controlled.py` |
| [repro/results_imagenet_schedule.csv](repro/results_imagenet_schedule.csv) | First schedule ablation (exchanged schedules of ToMe and EViT) | read by `tools/build_controlled.py` |
| [repro/results_imagenet_schedule_grid.csv](repro/results_imagenet_schedule_grid.csv) | Top-K and ToMe under fixed schedules (unit, first reduction block, set of reduction blocks) | Figure 10, Table 7 |
| [repro/results_imagenet_schedule_methods.csv](repro/results_imagenet_schedule_methods.csv) | Seven methods under four of these schedules | Table 8 |
| [repro/results_finetune.csv](repro/results_finetune.csv) | 5 epochs of fine-tuning under one recipe | Table 9 |
| [repro/results_imagenet_backbones.csv](repro/results_imagenet_backbones.csv) | 11 methods on 18 pre-trained ViTs, and DeiT III at two resolutions | Tables 11 and 12 |

## Other datasets and dense prediction

| File | Content | Used in |
|---|---|---|
| [repro/results_transfer_finetune.csv](repro/results_transfer_finetune.csv) | Fine-tuning of the unreduced DeiT-S on seven small datasets | Table 13 |
| [repro/results_transfer_free.csv](repro/results_transfer_free.csv) | 11 methods inside the fine-tuned models, without further training | Table 13 |
| [repro/results_transfer_cub_foreground.csv](repro/results_transfer_cub_foreground.csv) | Foreground and background patches behind the tokens on CUB-200-2011 | Section 9.7 |
| [repro/results_ade20k_segmenter.csv](repro/results_ade20k_segmenter.csv) | Segmenter ViT-S, ViT-B, and ViT-L on ADE20K (mIoU) | Figure 11 |
| [repro/results_coco_yolos.csv](repro/results_coco_yolos.csv) | YOLOS-S on COCO val2017 (box AP) | Figure 11 |
| [repro/results_coco_vitdet.csv](repro/results_coco_vitdet.csv) | ViTDet ViT-B Mask R-CNN on COCO val2017 (box and mask AP) | Figure 11 |

## Time, memory, and implementation choices

| File | Content | Used in |
|---|---|---|
| [repro/resources_gpu_throughput.csv](repro/resources_gpu_throughput.csv), [repro/resources_gpu_throughput_repeat.csv](repro/resources_gpu_throughput_repeat.csv) | Images per second and peak memory on the GPU at batch size 256 | Table 14, Figure 14 |
| [repro/resources_gpu_throughput_curve.csv](repro/resources_gpu_throughput_curve.csv) | Throughput over the settings of the accuracy curve | Figure 14 |
| [repro/resources_gpu_latency.csv](repro/resources_gpu_latency.csv), [repro/resources_cpu_latency.csv](repro/resources_cpu_latency.csv) | Latency at batch size 1 on the GPU and on the CPU | Table 14 |
| [repro/bench_compile.csv](repro/bench_compile.csv), [repro/bench_compile_diffrate.csv](repro/bench_compile_diffrate.csv) | Eager mode, `torch.compile`, and CUDA graph | Table 15 |
| [repro/bench_sync.csv](repro/bench_sync.csv) | Synchronization inside the reduction step | Table 15 |
| [repro/bench_sdpa.csv](repro/bench_sdpa.csv) | Reduction on fused attention (SDPA) | Table 15 |
| [repro/results_imagenet_precision.csv](repro/results_imagenet_precision.csv), [repro/results_imagenet_precision_kl.csv](repro/results_imagenet_precision_kl.csv) | fp32, fp16, and bf16 | Table 16 |
| [repro/results_imagenet_tricks.csv](repro/results_imagenet_tricks.csv), [repro/results_imagenet_v1.csv](repro/results_imagenet_v1.csv), [repro/preds/](repro/preds) | Proportional attention on DeiT and MAE, the fused token and last-block entries of the implementation table, and the predicted class of every validation image for the precision and repeat runs | Tables 16 to 18 |

## Notes

- Accuracy is the top-1 on the 50,000 images of the ImageNet-1k validation set unless a file states another dataset. GMACs count the multiply-accumulate operations of the model for one image.
- [repro/README.md](repro/README.md) lists the scripts that produce each file.
- The datasets are public and are not part of this repository: ImageNet-1k, COCO 2017, ADE20K, CUB-200-2011, Stanford Cars, Oxford-IIIT Pets, Oxford Flowers-102, MIT Indoor-67, CIFAR-10, and CIFAR-100.

## License

The code and the data files are released under the MIT License ([LICENSE](LICENSE)). The fonts in `tools/fonts` are under the SIL Open Font License 1.1.
