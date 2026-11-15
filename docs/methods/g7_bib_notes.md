# g7(merging 第 2 弾)の書誌確認記録

2026-10-18 に確認。docs/bib_notes.md と同じく、出版社の公式記録(doi.org の BibTeX / Crossref)、CVF open access、PMLR、OpenReview だけを使った。エントリは docs/methods/g7_refs.bib にある(paper/refs.bib にはまだ入れていない)。

| キー | 確認元 | メモ |
|---|---|---|
| pan2021hvt | doi | ICCV 2021, pp. 367--376。HVT は DeiT のブロック列を段に分け、各段の最初のブロックの後で 1D max pooling(kernel 3, stride 2)をかけて系列長を約半分にする構造型の手法で、学習はスクラッチから |
| koner2024lookupvit | doi | ECCV 2024 (LNCS), pp. 322--337。Crossref に LNCS の巻番号がないので volume は書かない。LookupViT は入力トークンを双線形補間で固定数 M(例 5×5)の compressed token に縮小し、重い処理はその M 個だけで行う構造型の手法(元の N 個の lookup token は軽い cross-attention と小さい MLP だけで更新される) |
| kim2025fpet | doi | CVPR 2025, pp. 30189--30198 |
| zhang2025tfatm | doi | ACM MM 2025(33rd), pp. 7172--7180。本文は ACM DL が自動取得を拒否し、OA 版も arXiv 版も見つからない。手法表は要旨(OpenAlex 経由で取得した出版社の要旨)だけで埋めた |
| choudhury2026apt | openreview | ICLR 2026 poster, forum SzoowJtd14。OpenReview の正式記録の題名は "Faster Vision Transformers with Adaptive Patches" で、CSV と arXiv(2510.18091v2)の題名 "Accelerating Vision Transformers with Adaptive Patch Sizes" と違う。著者 6 名は一致。bib は OpenReview の題名にした(arXiv 題名の別記録 vqeZag7jHY は OpenReview Archive への直接アップロードで、会場の記録ではない)。APT は画像のエントロピーが低い領域に大きなパッチ(32×32、64×64)を割り当て、入力段階でトークン数を減らす tokenization 型の手法 |
| xie2026sadtm | cvf, openreview | CVPR 2026, pp. 32184--32193(CVF の公式 BibTeX)。OpenReview の CVPR 2026 記録(forum OkrPIllBT7)でも会場と著者 7 名を確認。Crossref に DOI はまだない |
| ichikawa2026hac | doi, cvf | WACV 2026, pp. 3908--3917。CVF 版の題名は "Head-wise"、DOI 版は "Head-Wise"。bib は DOI 版に合わせた |
| zhou2026calibrated | pmlr, openreview | ICML 2026, PMLR 306, pp. 166294--166311。OpenReview(forum 9A6aXciRTI, "ICML 2026 regular")でも確認 |
| feng2023tokenmerger | doi | TIP 32, pp. 4156--4169。号の記載なし。本文は IEEE の有料記事で OA 版も arXiv 版も見つからない。手法表は要旨(OpenAlex 経由で取得した出版社の要旨)だけで埋めた |
| wang2025ltm | doi | TPAMI 47(11), pp. 9597--9608。本文は arXiv 2407.15219v2 で読んだ |
| chen2026hyptm | doi | TCSVT 36(10), pp. 14786--14799。本文は IEEE の有料記事で、Crossref と OpenAlex にも要旨がなく、arXiv 版も見つからない。手法表は題名から言えること以外 `?` |
| mao2025pruneandmerge | doi | TMM 27, pp. 4670--4683。号の記載なし。Crossref の著者表記は "Hua, Xiansheng" と "Shen, Hengtao"。本文は arXiv 2503.23455v1 で読んだが、arXiv 版の題名は "Efficient Token Compression for Vision Transformer with Spatial Information Preserved" で、著者は 5 名(Heng Tao Shen がいない)。要旨と DeiT-S の数値(1.64 倍、0.2% 低下)は TMM 版の要旨と一致 |
