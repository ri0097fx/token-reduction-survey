# g11(残りの候補 第 2 弾)の書誌確認記録

2026-10-18 に確認。docs/bib_notes.md と同じく、出版社の公式記録(doi.org の BibTeX / Crossref)と出版社の公式ページだけを使った。照会は doi.org(`Accept: application/x-bibtex`)、Crossref API(works と、Springer の章では会議情報の assertion と書籍レコード)、IOS Press と SPIE の公式ページで行い、要求にメールアドレスなどの連絡先は付けていない。エントリは docs/methods/g11_refs.bib にある(paper/refs.bib にはまだ入れていない)。

本文または要旨の入手先は手法表(docs/methods/g11_rest_r2_filled.csv)の source 列にある。22 本のうち本文を読めたのは 9 本(arXiv 4 本、IJCAI 公式 PDF 2 本、DATE の公式 archive、SCITEPRESS、HAL)。残り 13 本は有料で OA 版も arXiv 版も見つからず、要旨だけで埋めた(OpenAlex 経由の出版社の要旨 12 本、TREWA は著者所属機関のリポジトリの要旨)。

| キー | 確認元 | メモ |
|---|---|---|
| chen2026nastrack | doi | IJCAI 2026, pp. 1007--1015。本文は ijcai.org の公式 PDF で読んだ。追跡のための NAS で、層ごとの keep ratio を探索空間に入れる |
| yu2024rctp | doi | ISCAS 2024, pp. 1--5(IEEE の仮のページ番号)。第 4 著者は Crossref で "Wu, An-Yeu Andy"。要旨のみ |
| wang2026tinydrop | doi | ICASSP 2026, pp. 10247--10251。本文は arXiv 2509.03379v1 で読んだ |
| lee2023tavt | doi | DATE 2023, pp. 1--6。本文は DATE 2023 の公式 proceedings archive(past.date-conference.com)の PDF(6 ページ)で読んだ。ハードウェア設計はなく、細粒度分類のトークン枝刈り |
| zou2025pf2avit | doi | IEEE TCAD 44(11), pp. 4127--4140 (2025)。題名の "Quantization–Encoding" は公式記録どおり en dash。bib の題名は {PF$^{2}$A-ViT} とした。要旨のみ |
| yang2025twdp | doi | ASP-DAC 2025(30th), pp. 177--182。DOI は ACM(10.1145)。要旨のみ。zou2025pf2avit と同じグループ |
| li2025stem | doi | ICCAD 2025, pp. 1--9(IEEE の仮のページ番号)。要旨のみ。目的は敵対的パッチへの防御で、効率化は副次的 |
| chen2025trfpa | doi | ISLPED 2025, pp. 1--7(IEEE の仮のページ番号)。手法名 TRFPA は要旨から。要旨のみ |
| cho2025dppvit | doi | AICAS 2025(7th), pp. 1--5(IEEE の仮のページ番号)。要旨のみ |
| peng2024jmp | doi | Journal of Electronic Imaging 33(4), 記事番号 043045 (2024)。Crossref の page は "1-13"(記事内のページ数)なので、bib の pages には記事番号を入れた。記事番号は doi.org から転送される SPIE の公式 URL(.../volume-33/issue-04/043045/...)で確認。要旨のみ |
| jiang2024tsmvt | doi | IEEE Internet of Things Journal 11(19), pp. 32129--32142 (2024)。要旨のみ |
| haroun2025lotm | doi | ICONIP 2024 (CCIS), pp. 286--300, 出版は 2025 年。会議名は Crossref の Springer assertion("ICONIP", 第 31 回, Auckland, 2024)で確認。CCIS の巻番号は Crossref にないので volume は書かない。第 3 著者は Crossref では family "Chehida"、given "Karim Ben" だが、論文本文と DHTM の Crossref 記録では "Ben Chehida" が姓なので {Ben Chehida}, Karim とした。本文は HAL の著者版(hal-04781040, submitted version)で読んだ |
| haroun2025dhtm | doi | VISAPP 2025(VISIGRAPP 2025 の Volume 3), pp. 677--684。Crossref の会場名は "Proceedings of the 20th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications"、event は "20th International Conference on Computer Vision Theory and Applications"。本文は SCITEPRESS の open access PDF(CC BY-NC-ND)で読んだ |
| shin2025atom | doi | IEEE Transactions on Computers 74(5), pp. 1620--1633 (2025)。要旨のみ |
| bao2025ftcformer | doi | ECAI 2025, Frontiers in Artificial Intelligence and Applications 413, pp. 114--121。doi.org の BibTeX と Crossref には巻とページがなく、IOS Press の公式ページ(ebooks.iospress.nl, "Volume 413: ECAI 2025", "Pages 114 - 121")で確認。本文は arXiv 2507.10283v1 で読んだ |
| buratti2026trewa | doi | ICPR 2026 (LNCS), pp. 583--597。書名は "Pattern Recognition"。会議名は Crossref の Springer assertion("ICPR", 第 28 回, Lyon, 2026)で確認。LNCS の巻番号は Crossref にないので volume は書かない。手法名 TREWA は要旨から。要旨は著者所属機関のリポジトリ(iris.univpm.it, hdl 11566/354992)の記録から取った。PDF はログインが必要で読めなかった |
| wang2024camixersr | doi | CVPR 2024, pp. 25837--25846。本文は arXiv 2402.19289v2 で読んだ。超解像の手法で ViT ではない |
| zhou2023tokensparsemedseg | doi | IPMI 2023 (LNCS), pp. 743--754。会議名は Crossref の Springer assertion で確認。LNCS の巻番号は Crossref にないので volume は書かない。本文は arXiv 2303.06522v1 で読んだ |
| liu2025estj | doi | ICME 2025, pp. 1--6(IEEE の仮のページ番号)。要旨のみ |
| fang2025fr2vit | doi | ICASSP 2025, pp. 1--5(IEEE の仮のページ番号)。題名の上付き 2 は Crossref では <sup>2</sup>。bib は {FR$^{2}$ViT} とした。要旨のみ |
| liang2025tasktp | doi | IROS 2025, pp. 7826--7833。要旨のみ |
| smith2024todo | doi | IJCAI 2024 Demonstrations Track, pp. 8801--8804。Demo track であることは IJCAI の公式 PDF のヘッダ("Demonstrations Track")で確認し、booktitle に残した。本文は ijcai.org の公式 PDF で読んだ |

CSV に書かれた会場はすべて公式記録で確認できたので、書かなかったエントリはない。IEEE の会議論文で pages が 1--N のもの(ISCAS、DATE、ICCAD、ISLPED、AICAS、ICME、ICASSP 2025)は、Crossref の値をそのまま入れた(g10 の lee2024vittogo と同じ扱い)。
