# Positional control — drugban, davis

Does precision@k survive when each protein is given **another protein's** attention map? A borrowed map keeps the model's positional habit and loses everything specific to the protein. See `src/evaluation/positional_control.py`. k = 10; 1000 reassignments per cell; non-kinase sites exclude cotransport ions (the primary setting).

| seed | level | arm | p@10 | borrowed, absolute (p) | borrowed, relative (p) | same-residue shuffle (p) | in-span shuffle (p) | top-10 in site span | span / chain | first 10 residues | last 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | random | kinase | 0.019 | 0.022 (0.879) | 0.021 (0.739) | 0.020 (0.664) | 0.020 (0.706) | 0.23 | 0.23 | 0.004 | 0.001 |
| 1 | random | non_kinase | 0.008 | 0.013 (0.891) | 0.013 (0.849) | 0.012 (0.883) | 0.012 (0.824) | 0.39 | 0.39 | 0.000 | 0.005 |
| 1 | cold_drug | kinase | 0.022 | 0.021 (0.365) | 0.021 (0.359) | 0.021 (0.309) | 0.021 (0.410) | 0.24 | 0.23 | 0.004 | 0.001 |
| 1 | cold_drug | non_kinase | 0.014 | 0.013 (0.509) | 0.013 (0.456) | 0.013 (0.522) | 0.012 (0.451) | 0.39 | 0.39 | 0.000 | 0.005 |
| 1 | cold_target | kinase | 0.018 | 0.022 (0.828) | 0.020 (0.630) | 0.019 (0.636) | 0.019 (0.654) | 0.21 | 0.22 | 0.000 | 0.000 |
| 1 | cold_target | non_kinase | 0.010 | 0.015 (0.893) | 0.012 (0.685) | 0.013 (0.827) | 0.013 (0.761) | 0.42 | 0.39 | 0.000 | 0.005 |
| 1 | cold_pair | kinase | 0.036 | 0.020 (0.007*) | 0.020 (0.015*) | 0.021 (0.005*) | 0.018 (0.002*) | 0.21 | 0.21 | 0.000 | 0.000 |
| 1 | cold_pair | non_kinase | 0.014 | 0.012 (0.438) | 0.012 (0.416) | 0.011 (0.260) | 0.012 (0.448) | 0.39 | 0.39 | 0.000 | 0.005 |
| 2 | random | kinase | 0.015 | 0.021 (0.999) | 0.019 (0.950) | 0.019 (0.983) | 0.019 (0.986) | 0.22 | 0.23 | 0.004 | 0.001 |
| 2 | random | non_kinase | 0.012 | 0.013 (0.600) | 0.013 (0.559) | 0.014 (0.658) | 0.011 (0.427) | 0.36 | 0.39 | 0.000 | 0.005 |
| 2 | cold_drug | kinase | 0.021 | 0.021 (0.613) | 0.021 (0.517) | 0.020 (0.316) | 0.020 (0.347) | 0.22 | 0.23 | 0.004 | 0.001 |
| 2 | cold_drug | non_kinase | 0.007 | 0.014 (0.962) | 0.013 (0.948) | 0.012 (0.951) | 0.013 (0.950) | 0.42 | 0.39 | 0.000 | 0.005 |
| 2 | cold_target | kinase | 0.028 | 0.021 (0.123) | 0.019 (0.103) | 0.019 (0.078) | 0.022 (0.200) | 0.25 | 0.22 | 0.000 | 0.000 |
| 2 | cold_target | non_kinase | 0.007 | 0.013 (0.943) | 0.014 (0.945) | 0.012 (0.942) | 0.012 (0.921) | 0.37 | 0.39 | 0.000 | 0.005 |
| 2 | cold_pair | kinase | 0.022 | 0.019 (0.316) | 0.018 (0.288) | 0.018 (0.207) | 0.019 (0.263) | 0.21 | 0.21 | 0.000 | 0.000 |
| 2 | cold_pair | non_kinase | 0.012 | 0.013 (0.561) | 0.013 (0.584) | 0.013 (0.636) | 0.013 (0.571) | 0.39 | 0.39 | 0.000 | 0.005 |
| 3 | random | kinase | 0.019 | 0.021 (0.729) | 0.020 (0.566) | 0.020 (0.704) | 0.020 (0.678) | 0.23 | 0.23 | 0.004 | 0.001 |
| 3 | random | non_kinase | 0.010 | 0.012 (0.684) | 0.014 (0.747) | 0.013 (0.701) | 0.013 (0.723) | 0.40 | 0.39 | 0.000 | 0.005 |
| 3 | cold_drug | kinase | 0.021 | 0.023 (0.796) | 0.021 (0.568) | 0.019 (0.296) | 0.021 (0.624) | 0.24 | 0.23 | 0.004 | 0.001 |
| 3 | cold_drug | non_kinase | 0.017 | 0.014 (0.291) | 0.013 (0.249) | 0.013 (0.201) | 0.012 (0.156) | 0.39 | 0.39 | 0.000 | 0.005 |
| 3 | cold_target | kinase | 0.022 | 0.022 (0.483) | 0.020 (0.313) | 0.018 (0.224) | 0.019 (0.260) | 0.21 | 0.22 | 0.000 | 0.000 |
| 3 | cold_target | non_kinase | 0.012 | 0.012 (0.584) | 0.013 (0.639) | 0.012 (0.543) | 0.012 (0.565) | 0.38 | 0.39 | 0.000 | 0.005 |
| 3 | cold_pair | kinase | 0.024 | 0.019 (0.186) | 0.020 (0.285) | 0.021 (0.297) | 0.020 (0.267) | 0.22 | 0.21 | 0.000 | 0.000 |
| 3 | cold_pair | non_kinase | 0.012 | 0.013 (0.646) | 0.012 (0.604) | 0.015 (0.787) | 0.013 (0.645) | 0.41 | 0.39 | 0.000 | 0.005 |

`*` = the real maps beat the null (p < 0.05, before correction). *Same-residue shuffle* = attention permuted only among residues of the same amino acid (keeps residue-type preference, removes location). *In-span shuffle* = attention permuted within the stretch from the first to the last site, and separately outside it (keeps how much attention reaches that stretch -- for the KLIFS pocket, the kinase domain -- removes where in it). *Near an end* = within the first or last 10% of the chain.
