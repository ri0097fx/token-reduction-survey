# g12(ギャップ候補 第 3 弾 a)の書誌確認記録

2026-10-18 に確認。docs/bib_notes.md と同じく、出版社の公式記録(doi.org の BibTeX / Crossref)、CVF open access の一覧、OpenReview の検索 API、arXiv API だけを使った。照会は doi.org(`Accept: application/x-bibtex`)、Crossref API(works)、openaccess.thecvf.com の一覧ページ、api.openreview.net / api2.openreview.net の notes/search で行い、要求にメールアドレスなどの連絡先は付けていない。エントリは docs/methods/g12_refs.bib にある(paper/refs.bib にはまだ入れていない)。

本文の入手先は手法表(docs/methods/g12_gap_r3a_filled.csv)の source 列にある。19 本のうち 17 本は arXiv の PDF、DTMFormer は AAAI の公式 PDF(ojs.aaai.org)で本文を読んだ。STTABT だけは本文を読めていない(OpenReview の PDF がブラウザ認証 [challenge] で取得できず、arXiv 版もない)。STTABT は OpenReview の要旨と公式リポジトリの README だけで埋め、残りは `?` とした。

| キー | 確認元 | メモ |
|---|---|---|
| cordonnier2021dps | doi | CVPR 2021, pp. 2351--2360。選んだパッチを処理する特徴ネットワークは ResNet で、Transformer は集約部(3 層)にだけ使う |
| yue2021progsampling | doi | ICCV 2021, pp. 377--386 |
| huang2022svit | arxiv | 査読付きの版は見つからなかった。CVF open access の CVPR 2023 一覧(2353 件、`?day=all`)に題名も "Super Token" を含む題名もなく、Huang_Vision_Transformer_With_Super_Token_Sampling_CVPR_2023_paper.html は 404。CVPR 2024 と ICCV 2023 の一覧にもない。Crossref の題名検索(IEEE プレフィックス 10.1109 に絞った検索を含む)と著者名検索でも該当なし。arXiv API の記録は v2(2024-01-25 更新)で、journal_ref と DOI はない。著者 5 名は arXiv API で確認。公式 README も arXiv だけを挙げている。CVPR 2023 版があるという指摘は確認できなかったので journal = {arXiv} のままにした |
| lee2023sttabt | openreview | ICLR 2023 poster, forum VV0hSE8AxCw(v1 検索 API の venue / venueid と _bibtex で確認)。v2 では dblp の記録 PYrXtu2WcB も同じ forum を指す。本文 PDF は取得できず |
| bergner2023ips | openreview | ICLR 2023 poster, forum QCrw0u9LQ7(v1 検索 API)。arXiv 2210.13007v2 は "Published as a conference paper at ICLR 2023" |
| kong2023trilevel | doi | AAAI 37(7), pp. 8360--8368。doi.org の BibTeX は @article(Proceedings of the AAAI Conference on Artificial Intelligence)だが、refs.bib の AAAI の書き方に合わせて @inproceedings にした。本文は arXiv 2211.10801v1 で読んだ |
| li2023flip | doi | CVPR 2023, pp. 23390--23400。出版社の題名は "Pre-Training"(arXiv 版は "Pre-training") |
| yang2023aclip | doi | ICCV 2023, pp. 2759--2769。著者 11 名 |
| zhou2023monoatt | doi | CVPR 2023, pp. 17493--17503 |
| xu2023lgvit | doi | ACM MM 2023(Proceedings of the 31st ACM International Conference on Multimedia), pp. 9103--9114。arXiv 版の本文にも ACM の DOI が印字されている |
| wang2024dtmformer | doi | AAAI 38(6), pp. 5814--5822。arXiv 版はない。本文は AAAI の公式 PDF(article/download/28394/28770)で読んだ |
| gao2024sparseformer | openreview | ICLR 2024 poster, forum 2pvECsmld3(v2 検索 API の venue / venueid と _bibtex で確認)。本文は arXiv 2304.03768v1 で読んだ(arXiv は v1 のみ) |
| gao2024bootsparseformer | doi | CVPR 2024, pp. 17710--17721 |
| jain2024mone | doi | NeurIPS 37 (2024), pp. 58480--58497(doi.org の BibTeX と Crossref)。本文は arXiv 2407.19985v2("Preprint. Under review." の版)で読んだ |
| zhang2024toc3d | doi | ECCV 2024 (LNCS), pp. 56--72。Crossref に LNCS の巻番号がないので volume は書かない |
| olszewski2025tore | doi | WACV 2025, pp. 8606--8616。著者名の Wójcik、Zieliński は doi.org の BibTeX どおり |
| wu2025patchranking | doi | WACV 2025, pp. 5842--5851 |
| yoa2025imagepiece | doi | AAAI 39(9), pp. 9544--9552。出版社記録の第 3 著者は "Cho, Hye-Seung"(arXiv では Hyeseung Cho)。出版社の表記に合わせた |
| rios2025cla | doi | ICASSP 2025, pp. 1--5。手法名は CLCA(CLA head と CLC を合わせたもの) |
