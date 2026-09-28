# Faithfulness — drugban_davis_seed1

| Level | comp. | random control | **delta** | suff. | suff. random | AOPC | n | load-bearing? |
|---|---|---|---|---|---|---|---|---|
| Warm | 0.1974 | 0.2067 | **-0.0094** | 0.6751 | 0.6734 | 0.0963 | 200 | **no** |
| Cold-Drug | 0.3566 | 0.3209 | **0.0357** | 1.1022 | 1.1403 | 0.1527 | 200 | yes |
| Cold-Target | 0.1235 | 0.1228 | **0.0007** | 1.7352 | 1.7279 | 0.0659 | 200 | yes |
| Cold-Pair | 0.1734 | 0.1765 | **-0.0031** | 1.7119 | 1.6991 | 0.0917 | 200 | **no** |

`delta` = comprehensiveness minus its random-masking control, and it is the only column that is a result. Masking anything moves the prediction, so the raw comprehensiveness means nothing on its own.

`load-bearing? no` means masking the attended residues perturbed the prediction no more than masking arbitrary ones — the explanation is decoration at that level. That is a finding, not a failed run.
