# Audit grid

| Model | Warm | Cold-Drug | Cold-Target | Cold-Pair | Drop |
|---|---|---|---|---|---|
| coldsite_dti | 0.015 ± 0.007 | 0.022 ± 0.009 | 0.017 ± 0.002 | 0.013 ± 0.005 | +0.002 |
| drugban | 0.018 ± 0.003 | 0.021 ± 0.001 | 0.023 ± 0.005 | 0.027 ± 0.008 | -0.010 |
| hyperattentiondti | 0.034 ± 0.006 | 0.040 ± 0.031 | 0.024 ± 0.008 | 0.022 ± 0.010 | +0.012 |
| moltrans | 0.021 ± 0.003 | 0.027 ± 0.005 | 0.028 ± 0.016 | 0.020 ± 0.014 | +0.001 |
| uniform_control | 0.020 ± 0.001 | 0.020 ± 0.001 | 0.018 ± 0.005 | 0.017 ± 0.001 | +0.003 |

`±` is the standard deviation over seeds. `!` marks a cell with fewer than 3 seeds — not a usable estimate.

## Significance (Holm-Bonferroni over the whole grid)

1 of 20 cells survive correction at alpha = 0.05.

- `hyperattentiondti|davis|random` p=9.999e-05 (threshold 0.0025) -> yes
- `coldsite_dti|davis|cold_drug` p=0.006599 (threshold 0.002632) -> no
- `moltrans|davis|cold_target` p=0.0141 (threshold 0.002778) -> no
- `moltrans|davis|cold_drug` p=0.0318 (threshold 0.002941) -> no
- `hyperattentiondti|davis|cold_drug` p=0.0373 (threshold 0.003125) -> no
- `hyperattentiondti|davis|cold_target` p=0.1219 (threshold 0.003333) -> no
- `moltrans|davis|cold_pair` p=0.2036 (threshold 0.003571) -> no
- `drugban|davis|cold_pair` p=0.2036 (threshold 0.003846) -> no
- `drugban|davis|cold_target` p=0.2928 (threshold 0.004167) -> no
- `hyperattentiondti|davis|cold_pair` p=0.2935 (threshold 0.004545) -> no
- `uniform_control|davis|cold_drug` p=0.4462 (threshold 0.005) -> no
- `drugban|davis|cold_drug` p=0.4477 (threshold 0.005556) -> no
- `moltrans|davis|random` p=0.4782 (threshold 0.00625) -> no
- `uniform_control|davis|random` p=0.6326 (threshold 0.007143) -> no
- `coldsite_dti|davis|cold_target` p=0.6581 (threshold 0.008333) -> no
- `uniform_control|davis|cold_pair` p=0.696 (threshold 0.01) -> no
- `drugban|davis|random` p=0.7109 (threshold 0.0125) -> no
- `uniform_control|davis|cold_target` p=0.7656 (threshold 0.01667) -> no
- `coldsite_dti|davis|cold_pair` p=0.9296 (threshold 0.025) -> no
- `coldsite_dti|davis|random` p=0.9997 (threshold 0.05) -> no

## Confound control (kinase vs non-kinase)

**No stratified comparison was possible.** Fewer than 20 non-kinase targets were available in every cell. The unstratified ladder must NOT be presented as if the kinase confound were absent -- state it as an explicit limitation in the Discussion, or enlarge the antiviral subset (Track A, Priority 1).