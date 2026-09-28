# Kinase confound control — coldsite_dti, kiba, seed 3

The non-kinase arm is a **transfer** condition, not a stratification: 60 BindingDB proteins that no model trained on this dataset has seen, with their own UniProt binding-site annotation. It is therefore harder than this dataset's own cold_target level, where the protein is unseen but still a kinase.

| Level | kinase p@10 | non-kinase p@10 | gap | kinase n | non-kinase n |
|---|---|---|---|---|---|
| random | 0.026 | 0.012 | +0.014 | 211 | 60 |
| cold_drug | 0.015 | 0.020 | -0.005 | 212 | 60 |
| cold_target | n/a | n/a | n/a | — | — |
| cold_pair | n/a | n/a | n/a | — | — |

`*` = significant before correction. Correct across the whole grid with `run_audit`, not per cell.

Control arm: **60 distinct non-kinase targets**, `control_is_usable: True`.

## Levels that could not be scored

- `cold_target` — no checkpoint at /Users/mahimagarwal/ColdSite-results/kiba_binary/coldsite_dti_kiba_cold_target_binary_seed3.pt
- `cold_pair` — no checkpoint at /Users/mahimagarwal/ColdSite-results/kiba_binary/coldsite_dti_kiba_cold_pair_binary_seed3.pt