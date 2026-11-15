# g6 の書誌情報の確認記録

`docs/methods/g6_pruning_rl_r2.csv` の 16 本について、2026-10-18 に確認した。エントリは `docs/methods/g6_refs.bib`。確認元の種類は `docs/bib_notes.md` の冒頭と同じ(`doi`、`neurips`、`openreview`、`cvf`)。照会は doi.org(`Accept: application/x-bibtex`)、Crossref API、`tools/fetch_neurips_bib.py`、OpenReview v2 検索 API、proceedings.iclr.cc、openaccess.thecvf.com で行い、要求にメールアドレスなどの連絡先は付けていない。

| キー | 確認元 | メモ |
|---|---|---|
| liu2023asvit | doi | IJCAI 2023, pp. 1222--1230。arXiv コメントの "Accepted by IJCAI2023" とも一致。トークン数をしきい値で画像ごとに変える枝刈りで、RL は使わない |
| liu2024metr | openreview | ICLR 2024 poster, forum gJeYtRuguR。DOI なし。proceedings.iclr.cc にも掲載(pp. 7397--7410)があるが、既存の ICLR エントリに合わせてページは書かない。OpenReview の著者表記 "Xilin CHEN" は Chen, Xilin とした |
| zhang2024star | openreview | ICLR 2024 poster, forum COO51g41Q4。proceedings.iclr.cc では pp. 34095--34113。公式記録の著者表記は "Freris, Nikolaos"(論文 PDF では Nikolaos M. Freris) |
| hu2024lfvit | doi | AAAI 38(3), pp. 2274--2284 |
| luo2024ltrp | doi | CVPR 2024, pp. 22831--22840(IEEE Xplore 版) |
| wang2022vtclfc | neurips, doi | NeurIPS 35, pp. 13974--13988。proceedings.neurips.cc の BibTeX と doi.org で一致 |
| zhao2024dyt | neurips, doi | NeurIPS 37, pp. 114765--114796。トークン削減の位置づけ: 学習したディスパッチャが選ばなかったトークンはブロック(MLP)を飛ばしてアダプタだけを通るが、系列からは消えない。主目的はパラメータ効率のよい適応で、トークン削減そのものではない |
| zhang2023mgvit | neurips, doi | NeurIPS 36, pp. 69328--69347 |
| zhang2026tofe | doi | IEEE TNNLS 37(10), pp. 5545--5558 (2026)。出版社記録の著者は 4 名(Hu, Chenyu が加わる)で、arXiv 2507.16260 の 3 名と違う |
| yu2022miaformer | doi | AAAI 36(8), pp. 8962--8970。出版社版の題名は "Multi-Grained Input-Adaptation"(G が大文字) |
| luo2025trpts | doi | ICCV 2025, pp. 4360--4369。トークン削減の位置づけ: [CLS] 注意の上位トークンを残し、残りを重み付き平均で 1 トークンにまとめる。主目的はパラメータ効率のよい微調整で、トークン削減は補助的な部分 |
| lu2025rl4evit | doi | ICME 2025, pp. 1--6。DOI の接尾辞 11210023 |
| yao2026vpruner | doi | AAAI 40(40), pp. 34396--34404 (2026)。arXiv 版は見つからない。本文は AAAI の公開 PDF で読んだ |
| li2026prance | doi | IEEE TPAMI 48(1), pp. 283--298 (2026)。早期公開は 2025 年(DOI 2025.3605239)。本文は arXiv 2407.05010 で読んだ |
| liu2026lmft | cvf | CVPR 2026, pp. 31262--31271(CVF open access の公式 BibTeX)。確認時点で Crossref に DOI が見つからないので doi は書かず、url に CVF のページを入れた。IEEE Xplore 版が出たらページと DOI を更新する。トークン削減の位置づけ: 動きの小さい入力トークンを ViT に入れる前に捨て、そのしきい値を RL(REINFORCE)で学習する。主目的は動画の教師なしドメイン適応で、トークン削減は適応と効率化の両方の手段 |
| cauteruccio2025agentvit | doi | Applied Intelligence 55(7), 記事番号 607 (2025)。記事番号は Crossref で確認。Springer の open access |

CSV に書かれた会場はすべて公式記録で確認できたので、書かなかったエントリはない。
