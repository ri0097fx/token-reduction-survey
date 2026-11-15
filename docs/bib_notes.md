# references.bib の確認記録

各エントリの書誌情報をどこで確認したかを記録する。出版社の公式記録(doi.org の BibTeX / Crossref)、公式 proceedings ページ(NeurIPS、PMLR、ACL Anthology、CVF open access、OpenReview)、または arXiv API で確認したものだけを references.bib に入れる。

照会には `tools/verify_refs.py`(arXiv API、Crossref、doi.org、OpenReview v2 検索)と `tools/fetch_neurips_bib.py`(NeurIPS proceedings の公式 BibTeX)を使った。候補は `docs/candidates.csv` と `docs/candidates_beyond.csv`。

## 確認元の種類

- `doi`: doi.org から取得した出版社の BibTeX(Crossref 経由)。IEEE、Springer、AAAI、ACL、NeurIPS 2023 以降はこれで確認。
- `neurips`: proceedings.neurips.cc の公式 BibTeX(巻、ページ)。
- `pmlr`: proceedings.mlr.press の公式 BibTeX(ICML)。
- `acl`: aclanthology.org の公式 BibTeX。
- `openreview`: OpenReview の検索 API で会場(ICLR / TMLR)と著者を確認。ICLR と TMLR には DOI がないので url に forum を入れる。
- `cvf`: openaccess.thecvf.com の公式 BibTeX(CVPR / ICCV / WACV)。
- `arxiv`: arXiv API。査読付きの掲載先が確認できなかったものは journal = {arXiv} として引用する。

## 2026-10-17 に追加したエントリ

| キー | 確認元 | メモ |
|---|---|---|
| vaswani2017attention | neurips | NeurIPS 2017 のページには BibTeX リンクがなく、巻 30 のみ記載。ページ番号は書かない |
| wang2021pvt | doi | ICCV 2021, pp. 548--558 |
| darcet2024registers | openreview | ICLR 2024 oral, forum 2dnO3LLiJ1 |
| beyer2023flexivit | doi | CVPR 2023 |
| dehghani2023navit | neurips, doi | NeurIPS 36, DOI あり |
| liu2023patchdropout | doi | WACV 2023 |
| goyal2020powerbert | pmlr | PMLR 119 |
| dai2020funnel | neurips | NeurIPS 33 |
| ye2021trbert | acl, doi | NAACL-HLT 2021 |
| kim2021lengthadaptive | acl, doi | ACL-IJCNLP 2021 |
| rao2023dss | doi | TPAMI 45(9)。DynamicViT の論文誌版 |
| pan2021iared2 | neurips | NeurIPS 34。arXiv API では題名の上付き 2 の表記ゆれで一致せず |
| chen2021svite | neurips | NeurIPS 34 |
| song2021dge | neurips | NeurIPS 34 |
| wang2021dvt | neurips | NeurIPS 34 |
| ryoo2021tokenlearner | neurips | NeurIPS 34。NeurIPS 版の題名は "Adaptive Space-Time Tokenization for Videos"(arXiv 版と異なる) |
| xu2022evovit | doi | AAAI 36(3) |
| meng2022adavit | doi | CVPR 2022 |
| tang2022psvit | doi | CVPR 2022 |
| fayyaz2022ats | doi | ECCV 2022 (LNCS) |
| zong2022sit | doi | ECCV 2022 (LNCS) |
| li2022sait | arxiv | 掲載先は見つからず |
| zhou2023longimageshort | doi | ECML PKDD 2023。著者は Zhou and Zhu(arXiv の第一著者とは順序が違う) |
| chen2023cfvit | doi | AAAI 37(6) |
| wei2023tps | doi | CVPR 2023 |
| long2023beyond | doi | CVPR 2023 |
| chang2023stvit | doi | CVPR 2023 |
| chen2023sparsevit | doi | CVPR 2023 |
| lin2023supervit | doi | IJCV 131(12) |
| xu2024gtpvit | doi | WACV 2024 |
| wang2024zerotprune | doi | CVPR 2024 |
| mahmud2024papr | doi | ECCV 2024 (LNCS) |
| bergner2025cropr | doi, openreview | CVPR 2025 |
| haurum2023whichtokens | doi | ICCV Workshops 2023 |
| haurum2024atc | doi, openreview | ECCV 2024 (LNCS) |
| renggli2022patchmerger | arxiv | 掲載先は見つからず |
| zeng2022tcformer | doi | CVPR 2022 |
| grainger2023pacavit | doi | CVPR 2023 |
| bolya2023tomesd | doi | CVPR Workshops 2023 |
| ronen2023mixedres | doi | CVPR Workshops 2023 |
| havtorn2023msvit | doi | ICCV **Workshops** 2023(本会議ではない) |
| lee2024dtem | neurips, doi | NeurIPS 37 |
| jie2024tocom | doi | ECCV 2024 (LNCS) |
| shi2024crossget | pmlr | PMLR 235 |
| peng2025votemix | doi | ICME 2025。出版社記録の著者は 5 名(arXiv 版は 6 名)。第一著者は Peng |
| zeng2025tokentransforming | arxiv | 掲載先は見つからず |
| lee2025freqaware | doi | NeurIPS 38 (2025)。proceedings.neurips.cc の 2025 索引は確認時点で Creative AI track しか載っていないので doi.org で確認 |
| choi2026rapid | arxiv | |
| bercy2026g2tm | arxiv | arXiv のコメントでは CCIS (Springer) に掲載予定。確認できたら更新 |
| nguyen2025tokencompression | doi | APSIPA ASC 2025 |
| papa2024efficientvit | doi | TPAMI 46(12) |
| chen2024compressionsurvey | doi | Journal of Information, Technology and Policy。巻号の記録なし |
| saha2025edgesurvey | doi | Neurocomputing 643, 130417 |
| kong2025beyond | arxiv | |
| shao2026tokencompression | openreview | TMLR、公開日 2026-02-13、Survey Certification。OpenReview の公式 BibTeX は year 2026 |
| jin2025efficientmllm | doi | Visual Intelligence 3(1), 記事番号 27 |
| roh2022sparsedetr | openreview | ICLR 2022 poster, forum RRGVCN8kjim(v1 検索 API で確認) |
| zheng2023focusdetr | doi | ICCV 2023 |
| lu2023cts | doi | CVPR 2023 |
| tang2023dtop | doi | ICCV 2023 |
| liu2024revisiting | doi | WACV 2024 |
| norouzi2024algm | doi | CVPR 2024 |
| dou2023tore | doi | ICCV 2023 |
| wang2022stts | doi | ECCV 2022 (LNCS) |
| ding2023sta | doi | ICCV 2023 |
| ren2023testa | acl, doi | Findings of EMNLP 2023。正式題名は "... for Long-form Video-Language Understanding" |
| choi2024vidtldr | doi | CVPR 2024 |
| pollard2025videomerge | doi | CVPR Workshops 2025 |
| shang2025prumerge | doi | ICCV 2025(arXiv 初出は 2024) |
| chen2024fastv | doi | ECCV 2024 (LNCS) |
| zhang2025sparsevlm | pmlr | PMLR 267 |
| yang2025visionzip | doi | CVPR 2025 |
| xing2025vico | cvf | PyramidDrop (arXiv:2410.17247) の CVPR 2025 版。題名が "Conical Visual Concentration for Efficient Large Vision-Language Models" に変わっている。著者と要旨で同一論文と確認 |
| alvar2025divprune | doi | CVPR 2025 |
| zhang2025cdpruner | doi | NeurIPS 38 (2025) |
| lin2025vtw | doi | AAAI 39(5) |
| arif2025hired | doi | AAAI 39(2) |
| li2025tokenpacker | doi | IJCV 133(10)(arXiv 初出は 2024) |
| endo2025feather | doi | ICCV 2025(arXiv 初出は 2024) |
| wang2025effivlmbench | acl, doi | ACL 2025 |
| ye2025fitprune | doi | AAAI 39(21) |
| wang2024atedm | doi | CVPR 2024 |
| wu2025importancemerge | doi | ICCV 2025。正式題名は "Importance-Based Token Merging for Efficient Image and Video Generation" |
| zhao2025dydit | openreview | ICLR 2025 poster, forum taHwqSrbrb |
| zhan2024mambatoken | neurips, doi | NeurIPS 37 |
| ma2025mambatokenfree | arxiv | 第一著者は Ma |

## 2026-10-18 に追加したエントリ(第 2 ラウンド)

剪定・強化学習の 16 本は `docs/methods/g6_bib_notes.md`、統合系の 12 本は `docs/methods/g7_bib_notes.md`、dense の 15 本は `docs/methods/g8_bib_notes.md`、生成モデルと分析の 14 本は `docs/methods/g9_bib_notes.md`、その他の 19 本は `docs/methods/g10_bib_notes.md`、残りの 22 本は `docs/methods/g11_bib_notes.md`、第 3 ラウンド前半の 19 本は `docs/methods/g12_bib_notes.md`、後半の 19 本は `docs/methods/g13_bib_notes.md` に確認元を記録した。

## 既存エントリ

先行する論文の執筆時(2026-10-05 と 2026-10-08)に確認済み。

## 2026-10-18 に追加したエントリ(照合で見つかった引用漏れ)

| キー | 確認元 | メモ |
|---|---|---|
| dao2022flashattention | neurips | NeurIPS 35, pp. 16344--16359, doi 10.52202/068431-1189(proceedings.neurips.cc の公式 BibTeX) |

## 2026-11-14: models and data of the dense comparison

| key | source | note |
|---|---|---|
| strudel2021segmenter | doi.org BibTeX (10.1109/ICCV48922.2021.00717) | ICCV 2021, pp. 7242--7252 |
| zhou2017ade20k | doi.org BibTeX (10.1109/CVPR.2017.544) | CVPR 2017, pp. 5122--5130 |
| fang2021yolos | proceedings.neurips.cc citation metadata | NeurIPS 34, pp. 26183--26197 |
| li2022vitdet | doi.org BibTeX (10.1007/978-3-031-20077-9_17) | ECCV 2022 (LNCS), pp. 280--296 |
