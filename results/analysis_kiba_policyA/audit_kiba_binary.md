# Audit grid

| Model | Warm | Cold-Drug | Cold-Target | Cold-Pair | Drop |
|---|---|---|---|---|---|
| coldsite_dti | 0.019 ± 0.007 | 0.020 ± 0.005 | n/a | n/a | -0.001 |
| hyperattentiondti | 0.032 ± 0.020 | 0.025 ± 0.002 | n/a | n/a | +0.008 |
| moltrans | 0.032 ± 0.018 | 0.036 ± 0.024 | n/a | n/a | -0.003 |
| uniform_control | 0.023 ± 0.004 | 0.025 ± 0.001 | n/a | n/a | -0.002 |

`±` is the standard deviation over seeds. `!` marks a cell with fewer than 3 seeds — not a usable estimate.

## Significance (Holm-Bonferroni over the whole grid)

0 of 8 cells survive correction at alpha = 0.05.

- `moltrans|kiba|cold_drug` p=0.06786 (threshold 0.00625) -> no
- `uniform_control|kiba|cold_drug` p=0.2315 (threshold 0.007143) -> no
- `hyperattentiondti|kiba|cold_drug` p=0.2754 (threshold 0.008333) -> no
- `uniform_control|kiba|random` p=0.3014 (threshold 0.01) -> no
- `hyperattentiondti|kiba|random` p=0.477 (threshold 0.0125) -> no
- `moltrans|kiba|random` p=0.5369 (threshold 0.01667) -> no
- `coldsite_dti|kiba|cold_drug` p=0.8104 (threshold 0.025) -> no
- `coldsite_dti|kiba|random` p=0.9361 (threshold 0.05) -> no

## Confound control (kinase vs non-kinase)

**No stratified comparison was possible.** Fewer than 20 non-kinase targets were available in every cell. The unstratified ladder must NOT be presented as if the kinase confound were absent -- state it as an explicit limitation in the Discussion, or enlarge the antiviral subset (Track A, Priority 1).

## Missing cells (18)

- coldsite_dti/kiba/cold_target/seed1 — no checkpoint at /Users/mahimagarwal/ColdSite-results/kiba_binary/coldsite_dti_kiba_cold_target_binary_seed1.pt
- coldsite_dti/kiba/cold_target/seed2 — no checkpoint at /Users/mahimagarwal/ColdSite-results/kiba_binary/coldsite_dti_kiba_cold_target_binary_seed2.pt
- coldsite_dti/kiba/cold_target/seed3 — no checkpoint at /Users/mahimagarwal/ColdSite-results/kiba_binary/coldsite_dti_kiba_cold_target_binary_seed3.pt
- coldsite_dti/kiba/cold_pair/seed1 — no checkpoint at /Users/mahimagarwal/ColdSite-results/kiba_binary/coldsite_dti_kiba_cold_pair_binary_seed1.pt
- coldsite_dti/kiba/cold_pair/seed2 — no checkpoint at /Users/mahimagarwal/ColdSite-results/kiba_binary/coldsite_dti_kiba_cold_pair_binary_seed2.pt
- coldsite_dti/kiba/cold_pair/seed3 — no checkpoint at /Users/mahimagarwal/ColdSite-results/kiba_binary/coldsite_dti_kiba_cold_pair_binary_seed3.pt
- hyperattentiondti/kiba/cold_target/seed1 — no checkpoint at /Users/mahimagarwal/ColdSite-results/kiba_binary/coldsite_dti_kiba_cold_target_binary_seed1_hyperattentiondti.pt
- hyperattentiondti/kiba/cold_target/seed2 — no checkpoint at /Users/mahimagarwal/ColdSite-results/kiba_binary/coldsite_dti_kiba_cold_target_binary_seed2_hyperattentiondti.pt
- hyperattentiondti/kiba/cold_target/seed3 — no checkpoint at /Users/mahimagarwal/ColdSite-results/kiba_binary/coldsite_dti_kiba_cold_target_binary_seed3_hyperattentiondti.pt
- hyperattentiondti/kiba/cold_pair/seed1 — no checkpoint at /Users/mahimagarwal/ColdSite-results/kiba_binary/coldsite_dti_kiba_cold_pair_binary_seed1_hyperattentiondti.pt
- hyperattentiondti/kiba/cold_pair/seed2 — no checkpoint at /Users/mahimagarwal/ColdSite-results/kiba_binary/coldsite_dti_kiba_cold_pair_binary_seed2_hyperattentiondti.pt
- hyperattentiondti/kiba/cold_pair/seed3 — no checkpoint at /Users/mahimagarwal/ColdSite-results/kiba_binary/coldsite_dti_kiba_cold_pair_binary_seed3_hyperattentiondti.pt
- moltrans/kiba/cold_target/seed1 — no checkpoint at /Users/mahimagarwal/ColdSite-results/kiba_binary/coldsite_dti_kiba_cold_target_binary_seed1_moltrans.pt
- moltrans/kiba/cold_target/seed2 — no checkpoint at /Users/mahimagarwal/ColdSite-results/kiba_binary/coldsite_dti_kiba_cold_target_binary_seed2_moltrans.pt
- moltrans/kiba/cold_target/seed3 — no checkpoint at /Users/mahimagarwal/ColdSite-results/kiba_binary/coldsite_dti_kiba_cold_target_binary_seed3_moltrans.pt
- moltrans/kiba/cold_pair/seed1 — no checkpoint at /Users/mahimagarwal/ColdSite-results/kiba_binary/coldsite_dti_kiba_cold_pair_binary_seed1_moltrans.pt
- moltrans/kiba/cold_pair/seed2 — no checkpoint at /Users/mahimagarwal/ColdSite-results/kiba_binary/coldsite_dti_kiba_cold_pair_binary_seed2_moltrans.pt
- moltrans/kiba/cold_pair/seed3 — no checkpoint at /Users/mahimagarwal/ColdSite-results/kiba_binary/coldsite_dti_kiba_cold_pair_binary_seed3_moltrans.pt