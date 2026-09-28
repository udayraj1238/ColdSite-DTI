# Faithfulness — kiba_seed3

| Level | comp. | random control | **delta** | suff. | suff. random | AOPC | n | load-bearing? |
|---|---|---|---|---|---|---|---|---|
| Warm | 0.4734 | 0.1551 | **0.3183** | 1.7547 | 1.7765 | 0.5170 | 200 | yes |
| Cold-Drug | 0.7211 | 0.1404 | **0.5807** | 1.1689 | 1.1795 | 0.6820 | 200 | yes |

`delta` = comprehensiveness minus its random-masking control, and it is the only column that is a result. Masking anything moves the prediction, so the raw comprehensiveness means nothing on its own.

`load-bearing? no` means masking the attended residues perturbed the prediction no more than masking arbitrary ones — the explanation is decoration at that level. That is a finding, not a failed run.
