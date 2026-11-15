# g13(ギャップ第 3 弾 b)の書誌確認記録

2026-10-18 に確認。docs/bib_notes.md と同じく、出版社の公式記録(doi.org の BibTeX / Crossref)、CVF open access の公式 BibTeX、ICLR proceedings(proceedings.iclr.cc)と OpenReview の検索 API、PMLR の公式 BibTeX、AAAI の OJS 記録だけを使った。照会に連絡先(メールアドレスなど)は付けていない。エントリは docs/methods/g13_refs.bib にある(paper/refs.bib にはまだ入れていない)。OpenReview の notes API と PDF は challenge(403)で取得できなかったので、OpenReview は検索 API(notes/search)の結果で会場と著者を確認した。

| キー | 確認元 | メモ |
|---|---|---|
| li2025simprune | openreview | ICLR 2025 poster, forum GTcEe5fayC。著者 11 名は OpenReview と ICLR proceedings(proceedings.iclr.cc)の記録で一致。本文は ICLR proceedings の PDF で読んだ。arXiv 版は見つからない |
| aich2025proscale | openreview | ICLR 2025 poster, forum dmzM5UdAq6。ICLR proceedings にも掲載。本文は ICLR proceedings の PDF で読んだ(arXiv 2404.14657v3 のコメントも "Accepted to ICLR 2025") |
| lei2025peftr | doi, cvf | CVPR 2025, pp. 14954--14964。本文は CVF の版で読んだ。arXiv 版は見つからない |
| wang2025tca | doi | ICCV 2025, pp. 144--154。DOI 版の題名は "Training-Free Test-Time Adaptation"(大文字)。bib は DOI 版に合わせた。本文は arXiv 2410.14729v3 で読んだ |
| choi2025reprshift | doi | ICCV 2025, pp. 20456--20466。DOI 版の題名は "Flashattention" と小文字化されているので、bib は arXiv と論文本文の表記 {FlashAttention} にした。本文は arXiv 2508.00367v1 で読んだ |
| huang2025tosa | doi | IROS 2025, pp. 9654--9660。本文は arXiv 2506.20066v1(コメント "Accepted by IROS 2025")で読んだ |
| fuller2025lookwhere | doi, openreview | NeurIPS 2025(巻 38), pp. 96710--96748, DOI 10.52202/085713-2910。OpenReview では forum a1fWrvr72O、venue "NeurIPS 2025 poster"。doi.org の著者表記は "Green, James"(OpenReview は "James R Green")で、bib は doi.org に合わせた。本文は arXiv 2505.18051v3(NeurIPS 2025 の版)で読んだ |
| wang2026cait | doi | IEEE TPAMI 48(2), pp. 1373--1389 (2026)。公式の題名は "Compression Toward"(CSV と arXiv は "Towards")。bib は DOI 版に合わせた。本文は arXiv 2309.15755v2(コメント "TPAMI 2025 Camera-ready Version")で読んだ |
| he2026visionmor | doi | AAAI 2026, 巻 40, 号 6, pp. 4699--4707。AAAI OJS の記録(著者 7 名)とも一致。本文は AAAI OJS の PDF で読んだ |
| rave2026mpm | doi, cvf | CVPR 2026 Findings, pp. 2998--3008。doi.org の BibTeX と Crossref には題名も著者もないので、題名と著者は CVF の公式 BibTeX から取った(g10 の huo2026mame と同じ扱い)。booktitle は CVF の "(CVPR) Findings" に合わせた。本文は CVF の版で読んだ |
| li2026edgerecvit | cvf | CVPR 2026, pp. 12987--12996(CVF の公式 BibTeX)。Crossref に DOI がまだないので doi は書かず、url に CVF のページを入れた。arXiv 版は見つからない。本文は CVF の版で読んだ |
| ji2026sepatch3d | cvf | CVPR 2026, pp. 18776--18785。DOI なし(扱いは li2026edgerecvit と同じ)。本文は CVF の版、学習設定は CVF の補足資料で読んだ |
| liu2026bigain | cvf | CVPR 2026, pp. 31931--31940。DOI なし。本文は CVF の版で読んだ |
| wang2026httm | cvf | CVPR 2026, pp. 26379--26388。DOI なし。CVF の著者表記 "De La Parra, Cecilia" は bib で {De La Parra} とした。本文は CVF の版で読んだ |
| chen2026come | cvf | CVPR 2026, pp. 14590--14599。DOI なし。CVF の題名は "Confidence Guided"(arXiv 2511.14751v2 は "Confidence-Guided")。bib は CVF に合わせた。本文は CVF の版で読んだ |
| shu2026litevggt | cvf | CVPR 2026, pp. 36422--36432。DOI なし。著者 11 名は CVF の公式 BibTeX のまま。本文は CVF の版で読んだ |
| li2026sremerge | pmlr, openreview | ICML 2026, PMLR 306, pp. 71576--71593(PMLR の公式 BibTeX)。OpenReview では forum IJ2C56ra1a、venue "ICML 2026 regular"。本文は PMLR の PDF で読んだ |
| nguyen2026tokencompbreaks | doi | ECCV 2026 (LNCS), pp. 90--105。Crossref の書名は "Computer Vision – ECCV 2026"(Lecture Notes in Computer Science)で、会場はこれで確認した。Crossref に LNCS の巻番号がないので volume は書かない。ECVA の論文一覧(ecva.net)には確認時点で ECCV 2026 がまだない。公式の題名は "Under High Compression"(CSV は "under")。本文は arXiv 2607.02237v1 で読んだ |
| han2022dynamicnn | doi | IEEE TPAMI 44(11), pp. 7436--7456 (2022)。サーベイ論文。手法表は none / n/a で埋めた |

CSV に書かれた会場はすべて公式記録で確認できたので、書かなかったエントリはない。CVPR 2026 主会議の 6 本(li2026edgerecvit, ji2026sepatch3d, liu2026bigain, wang2026httm, chen2026come, shu2026litevggt)は Crossref に DOI がまだなく、ページは CVF の公式 BibTeX のもの。IEEE Xplore 版が出たらページと DOI を確認する。
