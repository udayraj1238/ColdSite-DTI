# Faithfulness — hyperattentiondti_davis_seed3

| Level | comp. | random control | **delta** | suff. | suff. random | AOPC | n | load-bearing? |
|---|---|---|---|---|---|---|---|---|
| Warm | 0.6372 | 0.5149 | **0.1223** | 2.0051 | 2.0171 | 0.3268 | 200 | yes |
| Cold-Drug | 0.4460 | 0.5229 | **-0.0769** | 2.7384 | 2.7220 | 0.2449 | 200 | **no** |
| Cold-Target | 0.3417 | 0.3295 | **0.0122** | 1.0073 | 1.0024 | 0.1778 | 200 | yes |
| Cold-Pair | 0.2802 | 0.2855 | **-0.0052** | 0.8390 | 0.8333 | 0.1568 | 200 | **no** |

`delta` = comprehensiveness minus its random-masking control, and it is the only column that is a result. Masking anything moves the prediction, so the raw comprehensiveness means nothing on its own.

`load-bearing? no` means masking the attended residues perturbed the prediction no more than masking arbitrary ones — the explanation is decoration at that level. That is a finding, not a failed run.
