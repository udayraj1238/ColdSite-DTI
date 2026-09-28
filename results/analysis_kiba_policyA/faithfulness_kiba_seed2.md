# Faithfulness — kiba_seed2

| Level | comp. | random control | **delta** | suff. | suff. random | AOPC | n | load-bearing? |
|---|---|---|---|---|---|---|---|---|
| Warm | 0.7422 | 0.1752 | **0.5670** | 1.6729 | 1.6508 | 0.7093 | 200 | yes |
| Cold-Drug | 0.3255 | 0.1053 | **0.2202** | 1.0923 | 1.0607 | 0.3339 | 200 | yes |

`delta` = comprehensiveness minus its random-masking control, and it is the only column that is a result. Masking anything moves the prediction, so the raw comprehensiveness means nothing on its own.

`load-bearing? no` means masking the attended residues perturbed the prediction no more than masking arbitrary ones — the explanation is decoration at that level. That is a finding, not a failed run.
