# Do a cell's three seeds agree with each other?

Each cell trained three times; precision@10 against UniProt's annotated
residues, and the per-seed permutation p **uncorrected** -- what a
single-seed report would have quoted.

| model | dataset | level | seed 1 | seed 2 | seed 3 | chance | spread | seeds above alpha |
|---|---|---|---|---|---|---|---|---|
| ColdSite-DTI | DAVIS | cold-drug | 0.027 | 0.027 | 0.011 | 0.020 | 0.015 | `**.` |
| ColdSite-DTI | DAVIS | cold-pair | 0.018 | 0.013 | 0.008 | 0.019 | 0.010 | `...` |
| ColdSite-DTI | DAVIS | cold-target | 0.015 | 0.019 | 0.018 | 0.019 | 0.004 | `...` |
| ColdSite-DTI | DAVIS | warm | 0.023 | 0.009 | 0.013 | 0.020 | 0.013 | `...` |
| HyperAttentionDTI | DAVIS | cold-drug | 0.025 | 0.077 | 0.019 | 0.020 | 0.057 | `**.` |
| HyperAttentionDTI | DAVIS | cold-pair | 0.013 | 0.022 | 0.032 | 0.019 | 0.019 | `..*` |
| HyperAttentionDTI | DAVIS | cold-target | 0.018 | 0.031 | 0.025 | 0.019 | 0.013 | `.*.` |
| HyperAttentionDTI | DAVIS | warm | 0.035 | 0.039 | 0.028 | 0.020 | 0.011 | `***` |
| HyperAttentionDTI | KIBA | cold-drug | 0.019 | 0.020 | 0.025 | 0.023 | 0.006 | `...` |
| HyperAttentionDTI | KIBA | warm | 0.017 | 0.022 | 0.053 | 0.023 | 0.036 | `..*` |
| MolTrans | DAVIS | cold-drug | 0.022 | 0.032 | 0.024 | 0.020 | 0.010 | `.*.` |
| MolTrans | DAVIS | cold-pair | 0.004 | 0.032 | 0.024 | 0.019 | 0.028 | `.*.` |
| MolTrans | DAVIS | cold-target | 0.012 | 0.038 | 0.029 | 0.019 | 0.026 | `.**` |
| MolTrans | DAVIS | warm | 0.024 | 0.021 | 0.019 | 0.020 | 0.005 | `*..` |
| MolTrans | KIBA | cold-drug | 0.028 | 0.063 | 0.017 | 0.023 | 0.046 | `.*.` |
| MolTrans | KIBA | warm | 0.020 | 0.053 | 0.022 | 0.023 | 0.032 | `.*.` |

`*` = that seed alone clears alpha; `.` = it does not.

**11 of 16 cells have seeds that disagree** about their own verdict. In **15 of 16** the spread across seeds is larger than the cell's distance from chance. **1** cell has all three seeds above alpha; **4** have none.
