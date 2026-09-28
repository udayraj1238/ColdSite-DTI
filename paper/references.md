# References — verified

Every entry below was checked on 2026-09-12 against the registry that issued its
identifier (Crossref for DOIs, the arXiv API for preprints, Europe PMC for abstracts), not
written from memory. No citation style is fixed yet; format for the venue's author
guidelines when it is chosen. **Add nothing to the paper that is not on this list, and
verify anything new the same way.**

## Models audited

| key | reference | identifier | verified |
|---|---|---|---|
| DeepDTA 2018 | Öztürk H, Özgür A, Ozkirimli E. DeepDTA: deep drug–target binding affinity prediction. *Bioinformatics* 34:i821–i829 (2018) | doi:10.1093/bioinformatics/bty593 | Crossref |
| MolTrans 2021 | Huang K, et al. MolTrans: Molecular Interaction Transformer for drug–target interaction prediction. *Bioinformatics* 37:830–836 (2021) | doi:10.1093/bioinformatics/btaa880 | Crossref |
| HyperAttentionDTI 2022 | Zhao Q, et al. HyperAttentionDTI: improving drug–protein interaction prediction by sequence-based deep learning with attention mechanism. *Bioinformatics* 38:655–662 (2022) | doi:10.1093/bioinformatics/btab715 | Crossref |

Added 2026-09-19 (the paper cited these without a verified entry):

| key | reference | identifier | verified |
|---|---|---|---|
| DrugBAN 2023 | Bai P, et al. Interpretable bilinear attention network with domain adaptation improves drug–target prediction. *Nature Machine Intelligence* 5:126–136 (2023) | doi:10.1038/s42256-022-00605-1 | Crossref, 2026-09-19 |

## Data

| key | reference | identifier | verified |
|---|---|---|---|
| DAVIS 2011 | Davis MI, et al. Comprehensive analysis of kinase inhibitor selectivity. *Nature Biotechnology* 29:1046–1051 (2011) | doi:10.1038/nbt.1990 | Crossref |
| KIBA 2014 | Tang J, et al. Making sense of large-scale kinase inhibitor bioactivity data sets: a comparative and integrative analysis. *J. Chem. Inf. Model.* 54:735–743 (2014) | doi:10.1021/ci400709d | Crossref |
| BindingDB 2016 | Gilson MK, et al. BindingDB in 2015: a public database for medicinal chemistry, computational chemistry and systems pharmacology. *Nucleic Acids Res.* 44:D1045–D1053 (2016) | doi:10.1093/nar/gkv1072 | Crossref |
| UniProt 2025 | The UniProt Consortium. UniProt: the Universal Protein Knowledgebase in 2025. *Nucleic Acids Res.* 53:D609–D617 (2025) | doi:10.1093/nar/gkae1010 | Crossref |
| UniProt 2023 | The UniProt Consortium. UniProt: the Universal Protein Knowledgebase in 2023. *Nucleic Acids Res.* 51:D523–D531 (2023) | doi:10.1093/nar/gkac1052 | Crossref |

Cite the UniProt release that matches when the binding sites were fetched (2026-08;
the 2025 paper is the current one).

## Attention as explanation, and faithfulness metrics

| key | reference | identifier | verified |
|---|---|---|---|
| Jain & Wallace 2019 | Jain S, Wallace BC. Attention is not Explanation. *Proc. NAACL-HLT 2019*, 3543–3556 | doi:10.18653/v1/N19-1357 | Crossref |
| Serrano & Smith 2019 | Serrano S, Smith NA. Is Attention Interpretable? *Proc. ACL 2019*, 2931–2951 | doi:10.18653/v1/P19-1282 | Crossref |
| Wiegreffe & Pinter 2019 | Wiegreffe S, Pinter Y. Attention is not not Explanation. *Proc. EMNLP-IJCNLP 2019*, 11–20 | doi:10.18653/v1/D19-1002 | Crossref |
| ERASER 2020 | DeYoung J, et al. ERASER: A Benchmark to Evaluate Rationalized NLP Models. *Proc. ACL 2020*, 4443–4458 — comprehensiveness and sufficiency | doi:10.18653/v1/2020.acl-main.408 | Crossref |
| ROAR 2019 | Hooker S, Erhan D, Kindermans P-J, Kim B. A Benchmark for Interpretability Methods in Deep Neural Networks. *NeurIPS 2019* — removal takes inputs off the training distribution | arXiv:1806.10758 | arXiv API |
| AOPC 2017 | Samek W, et al. Evaluating the Visualization of What a Deep Neural Network Has Learned. *IEEE Trans. Neural Netw. Learn. Syst.* 28:2660–2673 (2017) | doi:10.1109/TNNLS.2016.2599820 | Crossref |

## Ground truth, explanation method, protein embeddings (added 2026-09-19)

| key | reference | identifier | verified |
|---|---|---|---|
| KLIFS 2014 | van Linden OPJ, et al. KLIFS: a knowledge-based structural database to navigate kinase–ligand interaction space. *J. Med. Chem.* 57:249–277 (2014) | doi:10.1021/jm400378w | Crossref, 2026-09-19 |
| KLIFS 2021 | Kanev GK, et al. KLIFS: an overhaul after the first 5 years of supporting kinase research. *Nucleic Acids Res.* 49:D562–D569 (2021) | doi:10.1093/nar/gkaa895 | Crossref, 2026-09-19 |
| IG 2017 | Sundararajan M, Taly A, Yan Q. Axiomatic Attribution for Deep Networks. *Proc. ICML 2017* (PMLR 70) | arXiv:1703.01365 | arXiv API, 2026-09-19 |
| ProtTrans 2022 | Elnaggar A, et al. ProtTrans: Toward Understanding the Language of Life Through Self-Supervised Learning. *IEEE TPAMI* 44:7112–7127 (2022) | doi:10.1109/TPAMI.2021.3095381 | Crossref, 2026-09-19 |

Cite KLIFS 2014 for the 85-residue pocket definition and KLIFS 2021 for the database
release used.

## Statistics

| key | reference | identifier | verified |
|---|---|---|---|
| Holm 1979 | Holm S. A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics* 6:65–70 (1979) | JSTOR 4615733 (no DOI) | several catalogues |

## DTI interpretability and cold-start (Related Work §1)

| key | reference | identifier | verified |
|---|---|---|---|
| DMFF-DTA 2025 | He H, et al. Dual modality feature fused neural network integrating binding site information for drug target affinity prediction. *npj Digital Medicine* 8:67 (2025) | doi:10.1038/s41746-025-01464-x | Crossref; abstract via Europe PMC |
| EviDTI 2025 | Zhao Y, et al. Evidential deep learning-based drug-target interaction prediction. *Nature Communications* 16:6915 (2025) | doi:10.1038/s41467-025-62235-6 | Crossref; abstract via Europe PMC |
| GPS-DTI 2025 | Xiong A, et al. An interpretable geometric graph neural network for enhancing the generalizability of drug–target interaction prediction. *BMC Biology* 23:350 (2025) — the paper's model is named GPS-DTI in its abstract | doi:10.1186/s12915-025-02456-9 (PMID 41299450) | Crossref; Europe PMC |
| ColdDTI 2025 | Zhang Z, Wang Y, Sun Y, Ye M, Yao Q. Attending on Multilevel Structure of Proteins enables Accurate Prediction of Cold-Start Drug-Target Interactions. arXiv preprint (2025) | arXiv:2510.04126 | arXiv API — **preprint** |
| CS-DTA 2026 | Jiang Z, et al. CS-DTA: a language model-driven framework for robust drug-target affinity prediction under strict cold-start… *Frontiers in Chemistry* 14:1834317 (2026) | doi:10.3389/fchem.2026.1834317 | Crossref; abstract via Europe PMC |
| KANPM-DTA 2026 | Rakib MDYK, et al. KANPM-DTA: improving drug–target affinity prediction with Kolmogorov–Arnold networks and pretrained models. *Briefings in Bioinformatics* 27:bbag112 (2026) | doi:10.1093/bib/bbag112 | Crossref; abstract via Europe PMC |

First authors and initials are from each DOI's Crossref record; take the complete author
lists from the same records when formatting for the venue.

## Explanations under distribution shift (Related Work §2)

| key | reference | identifier | verified |
|---|---|---|---|
| Gupta 2025 | Gupta M, Prasad C V, Ramakrishnan G. Uncertainty-Aware Subset Selection for Robust Visual Explainability under Distribution Shifts. arXiv preprint (2025) — insertion/deletion scores drop "up to 40%" OOD — in the full text, not the abstract; give the page when citing | arXiv:2512.08445 | arXiv full text — **preprint** |
| Zhang 2026 | Zhang D, Betala S, Agarwal C. Quantifying Explanation Quality in Graph Neural Networks using Out-of-Distribution Generalization. arXiv preprint (2026) — the Explanation-Generalization Score | arXiv:2602.07708 | arXiv API — **preprint** |
| Sun 2025 | Sun S. CIRR: Causal-Invariant Retrieval-Augmented Recommendation with Faithful Explanations under Distribution Shift. arXiv preprint (2025) | arXiv:2512.18683 | arXiv API — **preprint** |

**Removed:** the claim that sparse-autoencoder faithfulness "has been formalized as a
geometric faithfulness gap" (`docs/00_MASTER_PLAN_V2.md` §1). No source makes it; the
nearest, Bal (2026, arXiv:2607.12166), is about in-distribution faithfulness.
