# Faithfulness — drugban_davis_seed2

| Level | comp. | random control | **delta** | suff. | suff. random | AOPC | n | load-bearing? |
|---|---|---|---|---|---|---|---|---|
| Warm | 0.2810 | 0.2706 | **0.0104** | 1.5437 | 1.5711 | 0.1299 | 200 | yes |
| Cold-Drug | 0.2146 | 0.1949 | **0.0196** | 0.7510 | 0.7518 | 0.0884 | 200 | yes |
| Cold-Target | 0.1377 | 0.1397 | **-0.0020** | 0.9686 | 0.9642 | 0.0690 | 200 | **no** |
| Cold-Pair | 0.2611 | 0.2591 | **0.0020** | 0.6367 | 0.6498 | 0.1362 | 200 | yes |

`delta` = comprehensiveness minus its random-masking control, and it is the only column that is a result. Masking anything moves the prediction, so the raw comprehensiveness means nothing on its own.

`load-bearing? no` means masking the attended residues perturbed the prediction no more than masking arbitrary ones — the explanation is decoration at that level. That is a finding, not a failed run.
