# Explanation-fidelity ladder — drugban_receptive_davis_seed2

| Level | precision@10 | normalised | ceiling | chance | p | n |
|---|---|---|---|---|---|---|
| Warm | 0.119 | 0.119 | 0.998 | 0.143 | 1.000 | 350 |
| Cold-Drug | 0.130 | 0.130 | 0.998 | 0.143 | 0.983 | 350 |
| Cold-Target | 0.173* | 0.173 | 1.000 | 0.140 | 0.008 | 67 |
| Cold-Pair | 0.180* | 0.180 | 1.000 | 0.136 | 0.001 | 71 |

`*` = significantly above chance (p < 0.05).
`n` = proteins, one test pair each.
