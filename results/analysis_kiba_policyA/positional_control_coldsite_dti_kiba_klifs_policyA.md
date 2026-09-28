# Positional control — coldsite_dti, kiba

Does precision@k survive when each protein is given **another protein's** attention map? A borrowed map keeps the model's positional habit and loses everything specific to the protein. See `src/evaluation/positional_control.py`. k = 10; 1000 reassignments per cell; non-kinase sites exclude cotransport ions (the primary setting).

| seed | level | arm | p@10 | borrowed, absolute (p) | borrowed, relative (p) | same-residue shuffle (p) | in-span shuffle (p) | top-10 in site span | span / chain | first 10 residues | last 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | random | kinase | 0.173 | 0.153 (0.008*) | 0.153 (0.015*) | 0.153 (0.005*) | 0.164 (0.034*) | 0.30 | 0.27 | 0.005 | 0.008 |
| 1 | random | non_kinase | 0.013 | 0.014 (0.561) | 0.013 (0.536) | 0.016 (0.742) | 0.013 (0.542) | 0.40 | 0.38 | 0.000 | 0.005 |
| 1 | cold_drug | kinase | 0.195 | 0.155 (0.001*) | 0.154 (0.001*) | 0.165 (0.001*) | 0.172 (0.001*) | 0.31 | 0.27 | 0.005 | 0.008 |
| 1 | cold_drug | non_kinase | 0.010 | 0.013 (0.794) | 0.013 (0.773) | 0.013 (0.784) | 0.012 (0.712) | 0.38 | 0.38 | 0.000 | 0.005 |
| 2 | random | kinase | 0.177 | 0.153 (0.002*) | 0.155 (0.018*) | 0.156 (0.003*) | 0.168 (0.077) | 0.30 | 0.27 | 0.005 | 0.008 |
| 2 | random | non_kinase | 0.025 | 0.014 (0.009*) | 0.013 (0.004*) | 0.017 (0.059) | 0.013 (0.004*) | 0.40 | 0.38 | 0.000 | 0.005 |
| 2 | cold_drug | kinase | 0.140 | 0.153 (0.961) | 0.151 (0.867) | 0.124 (0.007*) | 0.135 (0.155) | 0.25 | 0.27 | 0.005 | 0.008 |
| 2 | cold_drug | non_kinase | 0.018 | 0.012 (0.101) | 0.011 (0.059) | 0.014 (0.226) | 0.012 (0.109) | 0.35 | 0.38 | 0.000 | 0.005 |
| 3 | random | kinase | 0.147 | 0.156 (0.872) | 0.153 (0.746) | 0.139 (0.131) | 0.150 (0.710) | 0.27 | 0.27 | 0.005 | 0.008 |
| 3 | random | non_kinase | 0.012 | 0.013 (0.630) | 0.014 (0.654) | 0.016 (0.813) | 0.013 (0.649) | 0.37 | 0.38 | 0.000 | 0.005 |
| 3 | cold_drug | kinase | 0.121 | 0.152 (1.000) | 0.148 (1.000) | 0.156 (1.000) | 0.122 (0.584) | 0.22 | 0.27 | 0.005 | 0.008 |
| 3 | cold_drug | non_kinase | 0.020 | 0.014 (0.117) | 0.013 (0.087) | 0.016 (0.247) | 0.014 (0.134) | 0.38 | 0.38 | 0.000 | 0.005 |

`*` = the real maps beat the null (p < 0.05, before correction). *Same-residue shuffle* = attention permuted only among residues of the same amino acid (keeps residue-type preference, removes location). *In-span shuffle* = attention permuted within the stretch from the first to the last site, and separately outside it (keeps how much attention reaches that stretch -- for the KLIFS pocket, the kinase domain -- removes where in it). *Near an end* = within the first or last 10% of the chain.
