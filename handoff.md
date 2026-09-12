# Handoff — continuing the GPU work

Repo: https://github.com/Mahim56207/ColdSite-DTI_New (fork of udayraj1238/ColdSite-DTI).
Local clone: `/Users/mahimagarwal/Documents/ColdSite-DTI`. `origin` = the fork,
`upstream` = udayraj1238. **Work on `main`.**

`gh` is authenticated as Mahim56207 with `repo` scope, so commits and pushes work
from a *local* session — confirm before any push.

**Run local, not cloud.** A cloud container has no browser (so no Colab, which
needs your own Google session), no GPU, and no access to the macOS keyring
holding the `gh` token. Cloud is fine for read-only investigation; anything that
touches Colab or pushes must run locally.

## Branch note — read this first

`project-completion` is **stale**: it forked at `c9c7b8e` and `main` is 24
commits ahead of that point. Everything below describes `main`. Do not launch
from `project-completion`; it lacks the finished control arm, the re-fetched
ground truth, the gene map, and ~180 tests.

## State

577 tests pass. `python -m src.model.run_grid --preflight` reports
`ready_to_launch: True`.

STATUS.md items 0–5 are **done** — this is the part most likely to be
misremembered, because an older section of STATUS.md still lists them as open:

| | |
|---|---|
| Splits | built for real, all 8 dirs, leakage clean |
| Antiviral subset | 10,548 pairs, **3 targets** (HIV-1 protease, HIV-1 RT, influenza NA). SARS-CoV-2 documented unavailable — it is 3, not the 5 originally planned |
| Ground truth | re-fetched with `type` on every feature; DAVIS 442/406 usable, KIBA 229/212; contamination gone |
| KIBA gene map | 229/229 |
| Unmapped DAVIS targets | resolved or documented; 19 genuinely unresolvable |
| **Non-kinase control panel** | **60 targets, 21,145 pairs, `control_is_usable: True`** |

The control arm **exists**. Any statement that it does not is stale.

## Data is gitignored — rebuild it wherever you compute

`data/splits` (417 MB) and `src/data/baselines/deepdta/data` are both gitignored,
so they do not travel with the repo. On Colab, rebuild first:

```bash
mkdir -p src/data/baselines/deepdta/data/{davis,kiba}
for ds in davis kiba; do for f in ligands_can.txt proteins.txt Y; do
  curl -sL https://raw.githubusercontent.com/hkmztrk/DeepDTA/master/data/$ds/$f \
    -o src/data/baselines/deepdta/data/$ds/$f; done; done
python -m src.data.build_splits
python -m src.model.run_grid --preflight   # expect ready_to_launch: True
```

Takes ~20 s. Then persist splits + source files to Drive so later sessions skip
it. Verified end to end from a clean clone. Expect exactly:

```
davis   30056 measured pairs,   68 drugs, 442 targets, Y range [5.000, 10.796]
kiba   118254 measured pairs, 2111 drugs, 229 targets, Y range [0.000, 17.200]
```

DAVIS Y is converted in-code to pKd via `-log10(Y/1e9)`, so use the raw-Kd files.

## THERE ARE TWO GRIDS — do not conflate them

STATUS.md describes both and calls each "every real number" in different
sections. They are different runs:

| | 24-run grid | 36-run grid |
|---|---|---|
| what | **ColdSite-DTI, our own model** | **3 published baselines, the audit** |
| shape | 2 datasets × 4 splits × 3 seeds | 3 models × 4 splits × 3 seeds |
| task | regression, DAVIS + KIBA | **binary, DAVIS only** |
| command | `python -m src.model.run_grid` | `./run_davis_grid.sh` |
| status | STATUS.md item 6, "the critical path" | STATUS.md item 7, "the long pole" |

Confirm which one is wanted before spending GPU hours. The 24-run grid is the
one `run_grid --preflight` checks.

## Two numbers that keep drifting

- The 24-run grid is 2 × 4 × 3 **training seeds** = 24, **not 72**. Seeds vary
  weight init and batch order on one fixed split per cell — so seed error bars
  measure initialisation variance, not split-selection variance. That belongs in
  the paper.
- Cold-pair **trains on ~71%** of the pairs the other levels get (davis 72%,
  kiba 70%) but **uses ~54%** of all measured pairs once discarded rows are
  counted (davis 55%, kiba 54%). Quoting 54% as the training ratio overstates
  the confound by about 2×. `pct_of_largest_split` reports the second number.

## Colab disconnects are safe now

`run_grid` used to decide a cell was finished by testing only whether a
checkpoint file existed. But `train()` writes one on the first improving epoch —
epoch 0 in practice — while the results JSON is written only after the test pass.
A cell killed mid-training was therefore skipped on the next run as though it had
completed, entering the results table with no accuracy while the grid reported
success.

Fixed in `d9a03c1`: a cell is skipped only if `verify_cell` calls it valid,
otherwise it prints why it looks interrupted and retrains it. After a drop, just
re-run the same command. There is **no within-cell resume** — no optimizer or
scheduler state is saved, so an interrupted cell restarts at epoch 0 by design.

## Known rough edges

- `notebooks/colab_davis_grid.ipynb` is for the **36-run** grid, and it clones
  `udayraj1238/ColdSite-DTI` and pushes results back *there*, not to this fork.
  Repoint it before use. There is no Colab notebook for the 24-run grid yet.
- STATUS.md contradicts itself: the "Remaining work, by owner" table near the end
  and the §2.2 zero-count control table are **stale**, and list items 1–5 as open
  when the current tables mark them done. Believe the tables that say done. This
  cleanup is worth doing.

## Next

1. Decide which grid is being launched (see above).
2. Rebuild data on Colab, confirm preflight.
3. Launch. Re-run the same command after any disconnect.
