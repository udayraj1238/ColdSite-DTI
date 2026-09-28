# Faithfulness — drugban_davis_seed3

| Level | comp. | random control | **delta** | suff. | suff. random | AOPC | n | load-bearing? |
|---|---|---|---|---|---|---|---|---|
| Warm | 0.2360 | 0.2283 | **0.0077** | 0.8456 | 0.8369 | 0.1106 | 200 | yes |
| Cold-Drug | 0.2730 | 0.2561 | **0.0169** | 0.7931 | 0.8283 | 0.1291 | 200 | yes |
| Cold-Target | 0.1393 | 0.1268 | **0.0125** | 0.7011 | 0.7029 | 0.0680 | 200 | yes |
| Cold-Pair | 0.2233 | 0.2438 | **-0.0205** | 0.6254 | 0.6683 | 0.1311 | 200 | **no** |

`delta` = comprehensiveness minus its random-masking control, and it is the only column that is a result. Masking anything moves the prediction, so the raw comprehensiveness means nothing on its own.

`load-bearing? no` means masking the attended residues perturbed the prediction no more than masking arbitrary ones — the explanation is decoration at that level. That is a finding, not a failed run.
