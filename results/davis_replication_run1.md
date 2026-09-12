# DAVIS grid — replication run (run 1)

The 12 DAVIS regression cells were trained twice on Kaggle (T4 x2), with the same
notebook and the same training code. **The committed results in this directory are
run 2**, the only run whose checkpoints were kept. This file records run 1, whose
checkpoints were not saved, so that the agreement between the two can be checked.

Run 1's numbers were copied from the notebook's printed summary. There are no
checkpoints or per-epoch histories for it, so it cannot be used for the
interpretability steps — only as evidence of how well the accuracy results replicate.

## Same code, different runs

Between the two runs the only changes to the training code were to its progress
output (commits 9349174, 9cca03f, a3754bc). The model, data pipeline, splits and
optimisation were identical. The difference between the runs is therefore run-to-run
nondeterminism, not a code change.

## Run 1 — test metrics

| split | seed | CI | MSE |
|---|---|---|---|
| random | 1 | 0.8007 | 0.4935 |
| random | 2 | 0.8798 | 0.2609 |
| random | 3 | 0.8820 | 0.2578 |
| cold_drug | 1 | 0.5989 | 0.7232 |
| cold_drug | 2 | 0.6556 | 0.6313 |
| cold_drug | 3 | 0.6439 | 0.6699 |
| cold_target | 1 | 0.7980 | 0.4512 |
| cold_target | 2 | 0.8049 | 0.4504 |
| cold_target | 3 | 0.7923 | 0.4293 |
| cold_pair | 1 | 0.5515 | 0.6083 |
| cold_pair | 2 | 0.5911 | 0.6463 |
| cold_pair | 3 | 0.6617 | 0.5737 |

## Agreement with run 2 (committed)

**Split means replicate.**

| split | run 1 CI | run 2 CI | diff |
|---|---|---|---|
| random | 0.8542 | 0.8460 | −0.0081 |
| cold_target | 0.7984 | 0.8038 | +0.0054 |
| cold_drug | 0.6328 | 0.6566 | +0.0238 |
| cold_pair | 0.6014 | 0.6018 | +0.0004 |

**The drug/target asymmetry replicates.** Relative to `random`, an unseen target costs
−0.056 (run 1) and −0.042 (run 2); an unseen drug costs −0.221 and −0.189. The drug-side
drop is **4.0x** the target-side drop in run 1 and **4.5x** in run 2.

**Individual seeds do not replicate.** With the same seed, CI moved by a mean of 0.020 and
up to **0.058** (cold_drug seed 1: 0.5989 -> 0.6573). The seed does not fully determine the
result — cuDNN's LSTM kernels are nondeterministic — so a single seed's number is not a
reproducible quantity.

## What this means for reporting

- Report split-level means with their spread across seeds, never a single seed.
- Do not claim the per-seed numbers are exactly reproducible; they are not, even on
  identical code and hardware.
- The between-run shift in a split mean (up to 0.024, on cold_drug) is a lower bound on
  how small a difference between conditions can be before it is indistinguishable from
  rerunning the same experiment.
