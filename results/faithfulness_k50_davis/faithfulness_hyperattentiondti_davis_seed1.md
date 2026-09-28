# Faithfulness — hyperattentiondti_davis_seed1

| Level | comp. | random control | **delta** | suff. | suff. random | AOPC | n | load-bearing? |
|---|---|---|---|---|---|---|---|---|
| Warm | 1.2881 | 1.0897 | **0.1984** | 3.5664 | 3.5529 | 0.7112 | 200 | yes |
| Cold-Drug | 0.7120 | 0.6058 | **0.1062** | 3.7927 | 3.7979 | 0.3965 | 200 | yes |
| Cold-Target | 0.3482 | 0.4198 | **-0.0716** | 1.2564 | 1.2747 | 0.2083 | 200 | **no** |
| Cold-Pair | 0.3052 | 0.3130 | **-0.0078** | 1.1731 | 1.1826 | 0.1817 | 200 | **no** |

`delta` = comprehensiveness minus its random-masking control, and it is the only column that is a result. Masking anything moves the prediction, so the raw comprehensiveness means nothing on its own.

`load-bearing? no` means masking the attended residues perturbed the prediction no more than masking arbitrary ones — the explanation is decoration at that level. That is a finding, not a failed run.
