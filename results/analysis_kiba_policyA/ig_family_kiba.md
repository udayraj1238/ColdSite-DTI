# KIBA integrated-gradients family (UniProt annotated residues)

Holm–Bonferroni over **4 cells**, each cell's p the median over its seeds.

| cell | precision@10 | chance | × chance | p (median) | threshold | survives |
|---|---|---|---|---|---|---|
| `hyperattentiondti_ig|kiba|random` | 0.042 ± 0.020 (3 seeds) | 0.023 | 1.82× | 0.000999 | 0.0125 | **yes** |
| `hyperattentiondti_ig|kiba|cold_drug` | 0.043 ± 0.034 (3 seeds) | 0.023 | 1.90× | 0.001998 | 0.01667 | **yes** |
| `moltrans_ig|kiba|cold_drug` | 0.031 ± 0.007 (3 seeds) | 0.023 | 1.38× | 0.004995 | 0.025 | **yes** |
| `moltrans_ig|kiba|random` | 0.031 ± 0.017 (3 seeds) | 0.023 | 1.34× | 0.5265 | 0.05 | no |

**3 of 4 cells survive correction.**

Cells with no ladder file (not corrected, and not counted in the family):

* `hyperattentiondti_ig|kiba|cold_target`
* `hyperattentiondti_ig|kiba|cold_pair`
* `moltrans_ig|kiba|cold_target`
* `moltrans_ig|kiba|cold_pair`
