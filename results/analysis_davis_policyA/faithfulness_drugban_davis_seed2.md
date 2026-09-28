# Faithfulness — drugban_davis_seed2

| Level | comp. | random control | **delta** | suff. | suff. random | AOPC | n | load-bearing? |
|---|---|---|---|---|---|---|---|---|
| Warm | 0.1053 | 0.1072 | **-0.0018** | 1.6084 | 1.5907 | 0.1252 | 200 | **no** |
| Cold-Drug | 0.0619 | 0.0655 | **-0.0036** | 0.7760 | 0.7544 | 0.0884 | 200 | **no** |
| Cold-Target | 0.0649 | 0.0646 | **0.0002** | 0.9902 | 0.9948 | 0.0693 | 200 | yes |
| Cold-Pair | 0.1242 | 0.1280 | **-0.0038** | 0.6659 | 0.6814 | 0.1362 | 200 | **no** |

`delta` = comprehensiveness minus its random-masking control, and it is the only column that is a result. Masking anything moves the prediction, so the raw comprehensiveness means nothing on its own.

`load-bearing? no` means masking the attended residues perturbed the prediction no more than masking arbitrary ones — the explanation is decoration at that level. That is a finding, not a failed run.
