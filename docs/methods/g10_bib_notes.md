# g10(その他 第 2 弾)の書誌確認記録

2026-10-18 に確認。docs/bib_notes.md と同じく、出版社の公式記録(doi.org の BibTeX / Crossref)と CVF open access だけを使った。照会は doi.org(`Accept: application/x-bibtex`)、Crossref API、openaccess.thecvf.com の一覧ページで行い、要求にメールアドレスなどの連絡先は付けていない。エントリは docs/methods/g10_refs.bib にある(paper/refs.bib にはまだ入れていない)。

| キー | 確認元 | メモ |
|---|---|---|
| rangrej2022stam | doi | CVPR 2022, pp. 2508--2517。STAM は画像全体を見ずに 32×32 の glimpse を 1 枚ずつ選び、選んだ glimpse のパッチだけを DeiT に入れる。次の位置は one-step actor-critic で学習した actor が決める |
| pardyl2024adaglimpse | doi | ECCV 2024 (LNCS), pp. 112--129。Crossref に LNCS の巻番号がないので volume は書かない。glimpse の位置と大きさを Soft Actor-Critic で選ぶ能動的視覚探索の手法で、トークン削減は「見た glimpse のパッチだけを処理する」形 |
| chowdhury2025bandit | doi, cvf | WACV 2025, pp. 9597--9606。Crossref の第 3 著者の表記は family "K J"、given "Prabuchandran"(CVF では "Prabuchandran K J")。bib では {K J}, Prabuchandran とした。UCB bandit で各クエリが参照するキー(上位 p=5 個)を選ぶ方式で、トークン自体は系列から消えない。トークン削減というより疎な注意に近い |
| erak2026adaptivetokenmerging | doi | IEEE OJ-COMS 7, pp. 4112--4128 (2026)。号の記載なし。本文は arXiv 2509.09955v1(投稿中の版)で読んだ。層ごとの類似度しきい値を多目的ベイズ最適化(GP + EHVI)で探す方式で、RL は使っていない。主目的は semantic communication の計算量と通信量の削減 |
| huo2026mame | doi, cvf | CVPR 2026 Findings, pp. 2863--2872。Crossref の会場名は "2026 IEEE/CVF Conference on Computer Vision and Pattern Recognition- FINDINGS Track (CVPRF)"、CVF の公式 BibTeX の booktitle は "Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings"。bib は CVF の名前に合わせた。doi.org の BibTeX と Crossref には題名はあるが著者がないので、著者は CVF の公式 BibTeX から取った。arXiv 2604.13432 は MaRe を加えた拡張版(題名も違う)で、本文は CVF の版で読んだ |
| jajal2026adaperceiver | doi, cvf | CVPR 2026 Findings, pp. 2618--2628。会場名と著者の扱いは huo2026mame と同じ(著者は CVF の公式 BibTeX)。CVF の著者表記は "Thiruvathukal, George K"。本文は CVF の版で読んだ |
| schlesinger2026spot | doi, cvf | CVPR 2026 Findings, pp. 2335--2345。会場名と著者の扱いは huo2026mame と同じ。本文は CVF の版、付録(学習設定とスループット)は arXiv 2511.10488v1 で読んだ |
| fiza2026dynavit | doi, cvf | CVPR 2026 Findings, pp. 2844--2851。会場名と著者の扱いは huo2026mame と同じ。第一著者は CVF の公式 BibTeX で "Rubab, Syeda Fiza"(姓は Rubab)。キーは CSV のまま fiza2026dynavit にした。arXiv 版は見つからない。本文と補足資料は CVF で読んだ |
| jiang2024lemevit | doi | IJCAI 2024, pp. 929--937。Crossref の会場名は "Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence"。LeMeViT は 16 個の学習可能な meta token と画像トークンの間で dual cross-attention をとる構造型の手法で、画像トークンの数は減らさない |
| jajal2025vittm | doi, cvf | WACV 2025, pp. 7898--7907。DOI 版と CVF 版の著者表記は "Nick John Eliopoulos"、"Thiravathukal"(AdaPerceiver の CVF 版では "Thiruvathukal")。bib は公式記録の表記のままにした。少数の process token(大きいパッチ)だけを encoder に通し、多数の memory token とは linear attention で読み書きする構造型の手法 |
| wang2024ranmerformer | doi | Neurocomputing 573, 記事番号 127216 (2024)。CC BY の open access だが ScienceDirect が自動取得を拒否し、リポジトリ版も arXiv 版も見つからない。手法表は要旨(OpenAlex 経由で取得した出版社の要旨)だけで埋めた |
| marchetti2025tram | doi | Expert Systems with Applications 279, 記事番号 127449 (2025)。本文は著者所属機関のリポジトリ(iris.univpm.it、出版社版 PDF)で読んだ |
| xu2023idlevit | doi | AI 2023: Advances in Artificial Intelligence(AJCAI 2023、LNCS), pp. 28--41。Crossref に LNCS の巻番号がないので volume は書かない。booktitle は Springer の書名に、arXiv コメント "Accepted to AJCAI2023" の略称を添えた。本文は arXiv 2310.05654v2 で読んだ |
| niu2025atmformer | doi | Remote Sensing 17(4), 記事番号 660 (2025)。本文は MDPI の PDF(mdpi-res.com)で読んだ |
| haurum2022sinkhorn | doi | Automation in Construction 144, 記事番号 104614 (2022)。CC BY の open access だが ScienceDirect と Aalborg 大学のリポジトリ(Cloudflare)が自動取得を拒否した。手法表は要旨だけで埋めた。下水管の欠陥分類のための CNN + 軽量 ViT のハイブリッドで、Sinkhorn クラスタリングでトークン数を減らす部分が対象 |
| parikh2024fpga | doi | FCCM 2024, pp. 78--89。本文は arXiv 2403.14047v2 で読んだ(arXiv コメント "FCCM 2024")。重みのブロック枝刈りとトークン枝刈りを組み合わせた FPGA 向けの協調設計 |
| sun2025drvit | doi | Journal of Parallel and Distributed Computing 199, 記事番号 105042 (2025)。本文は有料で、Crossref、OpenAlex、Semantic Scholar にも要旨がなく、arXiv 版も見つからない。手法表は題名から言えること以外 `?` |
| lee2024vittogo | doi | DATE 2024, pp. 1--6。Crossref と doi.org の題名は "ViT- ToGo"(空白あり)、DATE の proceedings の PDF では "ViT-ToGo :"。bib は {ViT-ToGo} にした。本文は DATE 2024 の公式 proceedings archive(past.date-conference.com)の PDF で読んだ |
| yoo2024adaptiv | doi | MICRO 2024(57th), pp. 64--77。本文は IEEE の有料記事で、OA 版も arXiv 版も見つからない。手法表は要旨(OpenAlex 経由で取得した出版社の要旨)だけで埋めた |

CSV に書かれた会場はすべて公式記録で確認できたので、書かなかったエントリはない。CVPR 2026 Findings の 4 本は IEEE の DOI(CVPRF71909)があり、Crossref では会場が "FINDINGS Track (CVPRF)" と記録されている。主会議(CVPR)と区別するため、bib の booktitle には "(CVPR) Findings" を残した。
