# Faithfulness — drugban_davis_seed1

| Level | comp. | random control | **delta** | suff. | suff. random | AOPC | n | load-bearing? |
|---|---|---|---|---|---|---|---|---|
| Warm | 0.0768 | 0.0757 | **0.0011** | 0.7483 | 0.7521 | 0.0968 | 200 | yes |
| Cold-Drug | 0.1081 | 0.1154 | **-0.0073** | 1.1302 | 1.1247 | 0.1526 | 200 | **no** |
| Cold-Target | 0.0616 | 0.0611 | **0.0005** | 1.9061 | 1.9093 | 0.0659 | 200 | yes |
| Cold-Pair | 0.0839 | 0.0876 | **-0.0037** | 2.2679 | 2.2507 | 0.0916 | 200 | **no** |

`delta` = comprehensiveness minus its random-masking control, and it is the only column that is a result. Masking anything moves the prediction, so the raw comprehensiveness means nothing on its own.

`load-bearing? no` means masking the attended residues perturbed the prediction no more than masking arbitrary ones — the explanation is decoration at that level. That is a finding, not a failed run.
