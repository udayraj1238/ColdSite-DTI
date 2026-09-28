# Kinase confound control — drugban, davis, seed 3

The non-kinase arm is a **transfer** condition, not a stratification: 60 BindingDB proteins that no model trained on this dataset has seen, with their own UniProt binding-site annotation. It is therefore harder than this dataset's own cold_target level, where the protein is unseen but still a kinase.

| Level | kinase p@10 | non-kinase p@10 | gap | kinase n | non-kinase n |
|---|---|---|---|---|---|
| random | 0.019 | 0.010 | +0.009 | 349 | 59 |
| cold_drug | 0.021 | 0.019 | +0.002 | 349 | 59 |
| cold_target | 0.022 | 0.014 | +0.008 | 68 | 59 |
| cold_pair | 0.024 | 0.014 | +0.010 | 72 | 59 |

`*` = significant before correction. Correct across the whole grid with `run_audit`, not per cell.

Control arm: **60 distinct non-kinase targets**, `control_is_usable: True`.