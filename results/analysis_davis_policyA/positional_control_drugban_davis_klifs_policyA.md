# Positional control — drugban, davis

Does precision@k survive when each protein is given **another protein's** attention map? A borrowed map keeps the model's positional habit and loses everything specific to the protein. See `src/evaluation/positional_control.py`. k = 10; 1000 reassignments per cell; non-kinase sites exclude cotransport ions (the primary setting).

| seed | level | arm | p@10 | borrowed, absolute (p) | borrowed, relative (p) | same-residue shuffle (p) | in-span shuffle (p) | top-10 in site span | span / chain | first 10 residues | last 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | random | kinase | 0.145 | 0.152 (0.909) | 0.147 (0.605) | 0.143 (0.418) | 0.144 (0.483) | 0.25 | 0.25 | 0.011 | 0.010 |
| 1 | random | non_kinase | 0.008 | 0.013 (0.891) | 0.013 (0.849) | 0.012 (0.883) | 0.012 (0.824) | 0.39 | 0.39 | 0.000 | 0.005 |
| 1 | cold_drug | kinase | 0.152 | 0.151 (0.479) | 0.149 (0.362) | 0.142 (0.053) | 0.150 (0.355) | 0.26 | 0.25 | 0.011 | 0.011 |
| 1 | cold_drug | non_kinase | 0.014 | 0.013 (0.509) | 0.013 (0.456) | 0.013 (0.522) | 0.012 (0.451) | 0.39 | 0.39 | 0.000 | 0.005 |
| 1 | cold_target | kinase | 0.145 | 0.145 (0.523) | 0.141 (0.407) | 0.137 (0.256) | 0.141 (0.325) | 0.25 | 0.25 | 0.000 | 0.000 |
| 1 | cold_target | non_kinase | 0.010 | 0.015 (0.893) | 0.012 (0.685) | 0.013 (0.827) | 0.013 (0.761) | 0.42 | 0.39 | 0.000 | 0.005 |
| 1 | cold_pair | kinase | 0.138 | 0.127 (0.195) | 0.141 (0.569) | 0.137 (0.486) | 0.134 (0.340) | 0.23 | 0.24 | 0.000 | 0.021 |
| 1 | cold_pair | non_kinase | 0.014 | 0.012 (0.438) | 0.012 (0.416) | 0.011 (0.260) | 0.012 (0.448) | 0.39 | 0.39 | 0.000 | 0.005 |
| 2 | random | kinase | 0.134 | 0.145 (0.978) | 0.141 (0.820) | 0.144 (0.962) | 0.140 (0.929) | 0.25 | 0.25 | 0.011 | 0.010 |
| 2 | random | non_kinase | 0.012 | 0.013 (0.600) | 0.013 (0.559) | 0.014 (0.658) | 0.011 (0.427) | 0.36 | 0.39 | 0.000 | 0.005 |
| 2 | cold_drug | kinase | 0.133 | 0.153 (1.000) | 0.146 (0.960) | 0.140 (0.897) | 0.142 (0.988) | 0.25 | 0.25 | 0.011 | 0.011 |
| 2 | cold_drug | non_kinase | 0.007 | 0.014 (0.962) | 0.013 (0.948) | 0.012 (0.951) | 0.013 (0.950) | 0.42 | 0.39 | 0.000 | 0.005 |
| 2 | cold_target | kinase | 0.145 | 0.142 (0.439) | 0.139 (0.398) | 0.142 (0.427) | 0.158 (0.929) | 0.28 | 0.25 | 0.000 | 0.000 |
| 2 | cold_target | non_kinase | 0.007 | 0.013 (0.943) | 0.014 (0.945) | 0.012 (0.942) | 0.012 (0.921) | 0.37 | 0.39 | 0.000 | 0.005 |
| 2 | cold_pair | kinase | 0.134 | 0.141 (0.730) | 0.143 (0.718) | 0.136 (0.595) | 0.139 (0.730) | 0.25 | 0.24 | 0.000 | 0.021 |
| 2 | cold_pair | non_kinase | 0.012 | 0.013 (0.561) | 0.013 (0.584) | 0.013 (0.636) | 0.013 (0.571) | 0.39 | 0.39 | 0.000 | 0.005 |
| 3 | random | kinase | 0.138 | 0.153 (0.992) | 0.148 (0.909) | 0.144 (0.863) | 0.147 (0.985) | 0.26 | 0.25 | 0.011 | 0.010 |
| 3 | random | non_kinase | 0.010 | 0.012 (0.684) | 0.014 (0.747) | 0.013 (0.701) | 0.013 (0.723) | 0.40 | 0.39 | 0.000 | 0.005 |
| 3 | cold_drug | kinase | 0.152 | 0.150 (0.372) | 0.145 (0.188) | 0.142 (0.040*) | 0.150 (0.334) | 0.26 | 0.25 | 0.011 | 0.011 |
| 3 | cold_drug | non_kinase | 0.017 | 0.014 (0.291) | 0.013 (0.249) | 0.013 (0.201) | 0.012 (0.156) | 0.39 | 0.39 | 0.000 | 0.005 |
| 3 | cold_target | kinase | 0.145 | 0.143 (0.475) | 0.138 (0.354) | 0.141 (0.398) | 0.145 (0.563) | 0.26 | 0.25 | 0.000 | 0.000 |
| 3 | cold_target | non_kinase | 0.012 | 0.012 (0.584) | 0.013 (0.639) | 0.012 (0.543) | 0.012 (0.565) | 0.38 | 0.39 | 0.000 | 0.005 |
| 3 | cold_pair | kinase | 0.144 | 0.133 (0.224) | 0.142 (0.479) | 0.141 (0.425) | 0.143 (0.515) | 0.25 | 0.24 | 0.000 | 0.021 |
| 3 | cold_pair | non_kinase | 0.012 | 0.013 (0.646) | 0.012 (0.604) | 0.015 (0.787) | 0.013 (0.645) | 0.41 | 0.39 | 0.000 | 0.005 |

`*` = the real maps beat the null (p < 0.05, before correction). *Same-residue shuffle* = attention permuted only among residues of the same amino acid (keeps residue-type preference, removes location). *In-span shuffle* = attention permuted within the stretch from the first to the last site, and separately outside it (keeps how much attention reaches that stretch -- for the KLIFS pocket, the kinase domain -- removes where in it). *Near an end* = within the first or last 10% of the chain.
