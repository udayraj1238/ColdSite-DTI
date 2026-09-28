# Faithfulness — drugban_davis_seed3

| Level | comp. | random control | **delta** | suff. | suff. random | AOPC | n | load-bearing? |
|---|---|---|---|---|---|---|---|---|
| Warm | 0.0936 | 0.0872 | **0.0064** | 0.8974 | 0.8933 | 0.1084 | 200 | yes |
| Cold-Drug | 0.1210 | 0.1081 | **0.0128** | 1.0044 | 0.9910 | 0.1294 | 200 | yes |
| Cold-Target | 0.0574 | 0.0608 | **-0.0034** | 0.8398 | 0.8331 | 0.0673 | 200 | **no** |
| Cold-Pair | 0.1225 | 0.1197 | **0.0027** | 0.7299 | 0.7468 | 0.1312 | 200 | yes |

`delta` = comprehensiveness minus its random-masking control, and it is the only column that is a result. Masking anything moves the prediction, so the raw comprehensiveness means nothing on its own.

`load-bearing? no` means masking the attended residues perturbed the prediction no more than masking arbitrary ones — the explanation is decoration at that level. That is a finding, not a failed run.
