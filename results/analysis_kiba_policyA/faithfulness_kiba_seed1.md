# Faithfulness — kiba_seed1

| Level | comp. | random control | **delta** | suff. | suff. random | AOPC | n | load-bearing? |
|---|---|---|---|---|---|---|---|---|
| Warm | 0.4138 | 0.1348 | **0.2790** | 1.6564 | 1.8231 | 0.4935 | 200 | yes |
| Cold-Drug | 0.3439 | 0.1473 | **0.1966** | 1.2606 | 1.2412 | 0.3681 | 200 | yes |

`delta` = comprehensiveness minus its random-masking control, and it is the only column that is a result. Masking anything moves the prediction, so the raw comprehensiveness means nothing on its own.

`load-bearing? no` means masking the attended residues perturbed the prediction no more than masking arbitrary ones — the explanation is decoration at that level. That is a finding, not a failed run.
