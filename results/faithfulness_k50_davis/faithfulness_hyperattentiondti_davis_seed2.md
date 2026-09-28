# Faithfulness — hyperattentiondti_davis_seed2

| Level | comp. | random control | **delta** | suff. | suff. random | AOPC | n | load-bearing? |
|---|---|---|---|---|---|---|---|---|
| Warm | 0.9008 | 0.7891 | **0.1117** | 2.9170 | 2.8927 | 0.4690 | 200 | yes |
| Cold-Drug | 0.6724 | 0.6699 | **0.0024** | 3.8375 | 3.8208 | 0.3887 | 200 | yes |
| Cold-Target | 0.4985 | 0.3995 | **0.0990** | 1.2713 | 1.2804 | 0.2927 | 200 | yes |
| Cold-Pair | 0.2950 | 0.2740 | **0.0211** | 0.8486 | 0.8517 | 0.1757 | 200 | yes |

`delta` = comprehensiveness minus its random-masking control, and it is the only column that is a result. Masking anything moves the prediction, so the raw comprehensiveness means nothing on its own.

`load-bearing? no` means masking the attended residues perturbed the prediction no more than masking arbitrary ones — the explanation is decoration at that level. That is a finding, not a failed run.
