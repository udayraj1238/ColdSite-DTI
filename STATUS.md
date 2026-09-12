# Project status

> **Framing changed.** The project is now an audit of published DTI
> interpretability claims, not a single-model paper. Read
> `docs/00_MASTER_PLAN_V2.md`, then your `docs/PART2_GUIDE_<roll>.md`.

Last updated: 2026-09-12. Regenerate the numbers with `python -m pytest tests/ -q`
and `python -m src.data.ground_truth`.

This file tracks Part 1 against the checklists in each guide. **Part 1 is not
complete.** The remaining items are listed at the bottom with the exact command
to run. Track A's step-by-step is `docs/RUNBOOK_124AD0008.md`.

> **2026-08-09 — the splits exist. Track B is unblocked.**
> `python -m src.data.build_splits` has been run for real against the canonical
> DeepDTA-direct loader. All 8 split directories are built (2 datasets × 4
> levels, three-way, leakage checks clean). `data/splits/` is gitignored, so
> regenerate locally: put the DeepDTA files at
> `src/data/baselines/deepdta/data/{davis,kiba}/` then run the command.
> Counts are in `results/split_summary.md`.
>
> Note for 124AD0015: **DAVIS cold-pair validation is 264 rows.** Inherent to
> requiring both drug and target unseen, not a bug — but early stopping on it
> will be noisy, so decide how to handle that before the grid runs.

---

## 2026-09-12 — where things stand

**The scope is unchanged from 2026-08-18: the 36-run binary grid on DAVIS is the
paper's spine.** Two older "what is left" tables further down (§ *What is left, and
why* and § *Remaining work, by owner*) predate that decision and are marked
superseded; read this section instead.

### Trained

| model | task | DAVIS | KIBA | where |
|---|---|---|---|---|
| DeepDTA | regression | 12/12 | 12/12 | `results.md` |
| ColdSite-DTI | regression | 12/12, plus a full replication | deferred | `results/davis_*_regression_*` |
| DeepDTA | **binary** | **12/12** | — | Kaggle, in progress |
| ColdSite-DTI | **binary** | 0/12 | — | Kaggle, in progress |
| HyperAttentionDTI | **binary** | 0/12 | — | Kaggle, in progress |

The regression cells do not enter the audit table -- HyperAttentionDTI and MolTrans
are binary-only, so the table compares AUROC. The ColdSite-DTI regression grid shows
it on par with DeepDTA (CI 0.846 / 0.657 / 0.804 / 0.602 against 0.877 / 0.640 /
0.818 / 0.603 across random / cold-drug / cold-target / cold-pair).

**DeepDTA, binary, DAVIS** -- test AUROC, mean over 3 seeds, from the sanity gate of
the first Kaggle commit:

| random | cold_target | cold_pair | cold_drug |
|---|---|---|---|
| 0.929 | 0.908 | 0.728 | 0.692 |

The drug/target asymmetry documented for regression holds for the binary task, and
more strongly: an unseen target costs 0.022 AUROC, an unseen drug 0.238.

### Running

`notebooks/kaggle_davis_binary_grid36.ipynb`, Kaggle T4 x2. Each commit starts in an
empty container, so finished cells are carried forward through a restore dataset; the
notebook stops itself at 11 hours so a commit always saves its output. Expect 2-3
commits.

### Fixed since 2026-08-18

- **ColdSite-DTI could not train on the binary task** (`61ac412`). Its loader passed
  raw pKd values to the loss, and the run died at its first AUROC. `--dummy` hid it --
  its random labels are already 0/1. A third of the binary grid would have failed.
  The three trainers now share one `BINARY_THRESHOLD` (DAVIS pKd >= 7.0), so they are
  scored against identical labels. The 2026-08-18 note below that "nothing above the
  line is waiting on code" was wrong on this point: that path had never been run on
  real data.
- **Resuming a grid was unsafe** (`d9a03c1`, `f9d36f0`). An interrupted cell was
  banked as finished, and every restart retrained a finished cell.
- **The grid status file showed half the grid** on two GPUs (`bdad54c`).

### What the replication showed

The DAVIS regression grid was trained twice on identical code and hardware. Split
means agree within 0.024 CI and the drug/target asymmetry holds in both (4.0x and
4.5x). **Individual seeds do not reproduce**: the same seed moved by up to 0.058 CI,
because cuDNN's LSTM kernels are nondeterministic. Report split-level means with their
spread, never a single seed. Details: `results/davis_replication_run1.md`.

### Dry run of the interpretability analysis

`run_faithfulness` and `run_ladder` had only ever run on synthetic fixtures. They
were run on the real DAVIS **regression** ColdSite-DTI checkpoints, seed 1, to find
out before the binary grid finished whether they work. **They do**, end to end; the
accuracy hand-off matches the trainer's results exactly. Faithfulness used 40 pairs
per level; the ladder used every test pair.

**Faithfulness -- the attended residues are load-bearing at every level.** Masking
them moves the prediction 2-4x more than masking random residues:

| level | comprehensiveness | random control | delta |
|---|---|---|---|
| warm | 0.093 | 0.022 | 0.071 |
| cold-target | 0.123 | 0.054 | 0.069 |
| cold-drug | 0.094 | 0.041 | 0.053 |
| cold-pair | 0.078 | 0.030 | 0.048 |

**Ladder -- but they are barely binding sites.** precision@10 against the ceiling
(~0.99) and chance:

| level | precision@10 | chance | p | accuracy (CI) |
|---|---|---|---|---|
| warm | 0.036 | 0.020 | 0.001 | 0.780 |
| cold-drug | 0.023 | 0.020 | 0.001 | 0.657 |
| cold-target | **0.018** | 0.019 | **0.987** | **0.797** |
| cold-pair | 0.035 | 0.019 | 0.001 | 0.581 |

Of the ten residues ColdSite-DTI attends to most, on average fewer than half of one is
an annotated binding site. The attention is **faithful but not plausible**: it drives
the model's predictions without pointing at the biology. And the two axes come apart --
cold-target has the *highest* accuracy and the *lowest* fidelity, indistinguishable
from chance. That dissociation is the audit's thesis, visible in our own model.

Preliminary: one seed, regression rather than the binary checkpoints the audit uses.
The significance is easy at n > 1,000 per level; the effect size is what to report.

**Two things it surfaced, both since fixed:**

1. **Ground truth numbered along the wrong sequence** (`cb92832`). 54 targets had a DAVIS
   sequence differing from UniProt's -- fragments, isoforms, constructs -- so UniProt
   residue numbers pointed at the wrong residues. `src/data/align_ground_truth.py` now
   re-numbers every site along the DAVIS sequence. It also found **four targets carrying
   another protein's sites** (PKAC-alpha, PAK1, MLCK, CDK11; corrected via overrides) and
   DAVIS sequences that omit the kinase domain entirely (ROCK2, MLK1). Details:
   `data/GROUND_TRUTH_README.md`.
2. **The headline figure hid the scale** (`865cfe7`). It now draws chance and states
   the ceiling.

**The ladder re-run on the corrected ground truth** (same checkpoints, seed 1):

| level | precision@10 before | after | chance |
|---|---|---|---|
| warm | 0.036 | **0.040** | 0.020 |
| cold-drug | 0.023 | **0.028** | 0.020 |
| cold-target | 0.018 | **0.018** (p = 0.98) | 0.019 |
| cold-pair | 0.035 | **0.033** | 0.019 |

The misaligned sites were diluting the signal: where the model has one, it rose. The
conclusion is unchanged -- at best ~2x chance against a ceiling of 0.99, and cold-target
indistinguishable from chance.

### What is left

1. Finish the binary grid: ColdSite-DTI and HyperAttentionDTI, 12 cells each.
2. Faithfulness, then the ladder, per seed; then the audit grid; then the control,
   with and without `--exclude-cotransport-ions`. All run in the notebook's last
   section once 36/36 exist.
3. Volume-matched control for cold-pair -- `notebooks/colab_volume_control.ipynb`,
   ColdSite-DTI on `random` cut to 15,190 rows, three seeds, on Colab.
4. Writing decisions below: the ladder's framing, and the cotransport-ion treatment.
5. MolTrans: trainer not yet written; measured at ~3-12 GPU-hours for DAVIS on two T4s,
   far cheaper than KIBA. Decide its protocol (published 13 epochs vs early stopping).

**Deferred, to be written up as limitations:** KIBA for ColdSite-DTI and the binary
grid; MolTrans.

---

## 2026-08-18 — the week's scope, and what the hardware allows

**Scope for this sprint: DAVIS only, binary task, three models.** That is
3 models × 4 splits × 3 seeds = 36 runs, and it is the whole scientific spine
of the paper — the ladder, faithfulness, the Holm-corrected audit table, the
stratified control, the headline figure.

**Deferred, deliberately, and to be written up as limitations:**

- **KIBA.** 118,254 pairs against DAVIS's 30,056. It multiplies the grid's
  12–30 GPU-hours to 50–120, which does not fit the window.
- **MolTrans.** No trainer exists for it, it is 62.8M parameters against
  DeepDTA's 1.9M, and the `config['batch_size']` reshape in its vendored
  `forward` (`baselines/MolTrans/models.py:96`) will scramble training the
  same way it scrambled the suspect AUPRC row. The adapter is written and
  passes the contract; the audit cell is what is missing.

### Measured memory, so nobody discovers a batch size by OOM

Peak activation, forward + backward, 1000-residue protein:

| model | params | batch | peak |
|---|---|---|---|
| DeepDTA (our port) | 1.9M | 256 | 0.7 GB |
| ColdSite-DTI | 0.6M | 16 / 32 / **64 (default)** | 2.3 / 4.4 / **8.7 GB** |
| HyperAttentionDTI | 2.3M | **8 (default)** / 32 (vendored) | **1.7** / ~5 GB |
| MolTrans | 62.8M | 16 (vendored) / 32 | 5.6 / 10.9 GB |

Two consequences. **ColdSite-DTI's own default batch of 64 needs 8.7 GB** and
will not start on a 4 GB card — `src/model/run_grid.py` now takes
`--batch-size` so the grid can be sized to the card it runs on. And vendored
MolTrans divides by `torch.cuda.device_count()`, so it raises
`ZeroDivisionError` on any CPU-only machine, CI runners included.

The grid runs on Linux (Kaggle / Colab / HPC), so the Windows `.bat` runners
cannot launch it. `run_davis_grid.sh` is the portable equivalent, resumable
per cell, batch sizes overridable by environment variable.

### Fixed in this pass

- Two tests still asserted the baseline adapters were unimplemented stubs, so
  the suite was **red on `main`** from the commit that implemented them. They
  now assert the real contract: every adapter implemented, DeepDTA implemented
  but deliberately not auditable. **540 passing, 0 failing.**
- `python -m src.evaluation.target_family --panel` was documented in two places
  and implemented in none — the flag was ignored and the command re-printed the
  DAVIS/KIBA report saying `control_is_usable: False`. It now reads the panel's
  own family assignments: **60 non-kinase targets, 0 unknown,
  `control_is_usable: True`**. The control arm was never broken, only unwired.
- `data/davis_uniprot_to_gene.json` is committed. It is built offline from the
  provenance file (`python -m src.data.build_gene_map --dataset davis`, no
  network, 442/442 resolved), and it takes DAVIS from 241 kinase / 201 unknown
  to **429 kinase / 13 unknown / 0 non-kinase** — which is the confound stated
  exactly: DAVIS cannot supply its own control, so the non-kinase panel is the
  only one there is.
- `results/*.md` and `results/*.csv` are no longer gitignored, so the numbers
  the paper quotes are versioned. Checkpoints, per-run JSON and figures stay
  ignored, and anything named `DUMMY_PLACEHOLDER` is re-ignored after the
  exception so a synthetic run still cannot be committed.

### Built since — everything that does not need a GPU

1. **`run_audit`'s real collector** — `src/evaluation/collect.py`. Real mode
   used to raise `SystemExit`; the grid now runs against trained checkpoints.
   Each model is tokenised with its own vendored tables, never a shared one.
   Verified end to end on a 30-target fixture across 4 levels × 3 seeds.
2. **The panel evaluation path** — `src/evaluation/run_control.py`. Reported as
   a **transfer** condition, not a stratification: DAVIS has 0 non-kinase
   targets to stratify into, so the arm is 60 out-of-distribution BindingDB
   proteins, which is harder than the dataset's own cold-target level.
   `--dry-run` checks the gate without a checkpoint.
3. **One checkpoint per model per cell** — `MODEL_SUFFIX` in
   `checkpoint_naming.py`. Existing DeepDTA paths are unchanged, pinned by a
   test. `discover_checkpoints` could previously never find a baseline at all.
4. **The cotransport-ion decision is a flag**, and its cost is measured: 29 of
   396 panel positions across SLC6A2/3/4 and DRD4, no target lost, DAVIS
   untouched. Default excludes nothing; zinc deliberately survives.
5. **Related Work rewritten** for the audit reframe — honest CS-DTA paragraph,
   OOD-explainability section, attention-faithfulness dispute.

Found and fixed on the way: **`run_audit` could not see a single baseline.**
Registration is an import side-effect of `@register`, nothing imported
`baseline_adapters`, and `--models deepdta` failed with "unknown model" plus
advice to write an adapter that already existed and already passed
`validate_adapter`.

### What is left, and all of it needs the GPU

| # | Item | Command | Blocking |
|---|---|---|---|
| 1 | Rebuild splits on the machine holding the DeepDTA files | `python -m src.data.build_splits` | everything |
| 2 | **36 training runs** — DAVIS, binary, 3 models × 4 splits × 3 seeds | `./run_davis_grid.sh` | every real number |
| 3 | Ladder + faithfulness per seed | `run_faithfulness` then `run_ladder` | the headline figure |
| 4 | The audit grid | `run_audit --models ... --ground-truth ...` | Results |
| 5 | The control, both ways | `run_control` ± `--exclude-cotransport-ions` | the confound section |

Every one of those is a command that exists and is tested. Nothing above the
line is waiting on code.

### Still an open decision, not a code gap

**The ladder is not monotonic on DAVIS** (`results.md`): cold-drug 0.640 is far
harder than cold-target 0.818, because holding out 20% of 68 drugs leaves 49 to
train on. Nothing in the code assumes monotonicity — the axis is categorical
throughout — so this is a writing decision, and **option 3 is the cheap,
defensible one**: keep the four levels as categories, drop the "increasing
severity" framing, and report the non-monotonicity as a finding. "Cold-start
severity is not a single ordered axis; it depends on which entity space is
sparse" is exactly the kind of claim an audit paper is well placed to make.
Settle it before the headline figure is drawn.

---

## Track A — 124AD0008 (Data, splits, baselines)

| | Item | Status |
|---|---|---|
| A1 | DAVIS + KIBA load with stats | ✅ `src/data/load_data.py` — 30,056/68/442 and 118,254/2,111/229, matching the published benchmark |
| A2 | Four splits, train/valid/test, both datasets | ✅ `src/data/build_splits.py` |
| A3 | Leakage check in code | ✅ + 18 tests, incl. injected-leakage cases |
| A4 | Split files saved | ✅ **run for real 2026-08-09**, all 8 dirs, leakage clean |
| A5 | Split summary table | ✅ auto-written to `results/split_summary.md` |
| A6 | Antiviral subset | ✅ **10,548 pairs, 3 targets** (HIV-1 protease, HIV-1 RT, influenza NA). SARS-CoV-2 unavailable — see below |
| A7 | Binding-site ground truth | ✅ **re-fetched** — DAVIS 442 targets / 406 usable, KIBA 229 / 212, `type` on every feature, contamination gone |
| A8 | Ground-truth README | ✅ `data/GROUND_TRUTH_README.md` |
| A9 | Three baseline adapters, passing `validate_adapter` | ✅ **all three PASS** — `python -m src.evaluation.check_adapters` |
| A10 | Results table, 3×4×2×3 seeds | ⏳ **8 of 24 cells** — DeepDTA regression complete. Runners ready for DeepDTA-binary and HyperAttentionDTI; MolTrans needs 124AD0015 |
| A11 | KIBA accession → gene map | ✅ `data/kiba_uniprot_to_gene.json`, 229/229 with a gene symbol |
| A12 | Unmapped DAVIS targets | ✅ resolved or documented — 19 unresolvable (real UniProt annotation gaps), 3 fixed by hand |
| A13 | **Non-kinase control panel** | ✅ **60 distinct human targets, 21,145 pairs, `control_is_usable: True`** |

### A7 — what "clean" means here

`dropped_feature_type` reads **0**, and that is the correct post-fix state, not a
failure. The rewritten fetcher never *collects* UniProt's `Site` catch-all, so
there is nothing left downstream to drop. The number that shows the fix landed
is **`dropped_description`: 125 → 0** — the description heuristic that the code
itself called "a stopgap with known false negatives" now has nothing to guess
at. Every feature carries a `type`; the file holds 1,369 `Binding site` and 427
`Active site` annotations and nothing else.

### A7 — three silent wrong-protein bugs, found and fixed

None of these crashed. Each produced a real, reviewed UniProt entry for the
wrong protein, and the target then contributed a wrong or empty site list to
every average:

| Target | Was | Should be | How it was caught |
|---|---|---|---|
| `IKK-epsilon` | Q96MC9, "Putative uncharacterized protein IKBKE-AS1" (antisense transcript) | Q14164, the kinase | gene search took the first of `size=1` |
| `PRKCH` | C0HM02, "PRKCH upstream open reading frame 2", 52 aa | P24723, the kinase, 683 aa | uORF is filed under the *same* gene symbol — only sequence length separates them |
| `MST1` | P26927, macrophage-stimulating 1 (hepatocyte growth factor-like) | Q13043 = **STK4** | gene-symbol collision; normal length, exact symbol match, so only the "not described as a kinase in a kinase panel" sweep found it |

The fetcher now requests 10 candidates and picks on exact gene-symbol match then
longest sequence. All three checks live in
`python -m src.data.resolve_unmapped --dataset davis --audit`, which currently
flags one target (`CASK`, correct — UniProt just names it oddly).

Also fixed: `organism_id` → `taxonomy_id`, without which every non-human target
(`PFCDPK1`, `PKNB`, `PFPK5`) was unresolvable, because their reviewed entries
sit under *strain* taxa rather than the species id.

### A10 — the ladder is not monotonic on DAVIS

DeepDTA is done, and its accuracy ladder says something the plan did not expect:

    davis   random 0.877  ->  cold_drug 0.640  ->  cold_target 0.818  ->  cold_pair 0.603
    kiba    random 0.854  ->  cold_drug 0.736  ->  cold_target 0.727  ->  cold_pair 0.637

On DAVIS, **cold-drug is far harder than cold-target** (0.640 vs 0.818, against
a seed spread of 0.02). DAVIS has 68 drugs and 442 targets, so cold-drug trains
on 49 drugs while cold-target trains on 310. KIBA, with 2,111 drugs, behaves.

The v2 plan treats warm → cold-drug → cold-target → cold-pair as increasing
severity and the headline figure plots fidelity along that axis. On DAVIS that
ordering is false, so a fidelity curve drawn along it would show a shape that is
neither noise nor a finding — just a mislabelled x-axis. Three options and the
reasoning are in `results.md`; the choice is 124AD0067's as Results lead, and it
should be settled before the figure is drawn.

### A10 — the four models do not share an accuracy metric

HyperAttentionDTI (`Linear(512,2)` + CrossEntropyLoss) and MolTrans (`BCELoss`)
are **binary-only**. DeepDTA and ColdSite-DTI are regression. So the audit grid
as specified yields CI for two models and AUROC for the other two, which cannot
be compared — and "do the interpretable models pay an accuracy cost", the
reason DeepDTA is in the grid at all, needs a single axis.

Binary is the only task all four share, and the thresholds are already verified
(DAVIS pKd ≥ 7.0 → 8.3% positive, KIBA ≥ 12.1 → 21.0%). Changing the two binary
models to regression would mean auditing a head their authors never published.

**Action for 124AD0015: the 24-run grid should be `--task binary`, or run
twice.** The Part 2 guide currently says `--task regression`. Full reasoning in
`results.md`. The explanation axis — precision@k, faithfulness — is unaffected;
attention is attention whatever the loss.

### A13 — why the control arm was rebuilt

The v2 plan's control arm is five antiviral proteins. `confound_report` gates
the control at **≥20 distinct non-kinase targets**, so five could never clear
it — that was true before any extraction problem. Two further findings:

- **SARS-CoV-2 cannot be extracted from BindingDB 2026-07-31.** All 18,149
  SARS-CoV-2 rows are filed under "Replicase polyprotein 1ab" carrying the full
  **7,096-residue** polyprotein. Mpro is residues 3264–3569; exactly 3 rows say
  so. Nothing separates an Mpro measurement from an RdRp one by target name, and
  7,096 residues sits almost entirely outside the 1,000-residue window anyway.
  Mpro and RdRp are now `OPTIONAL_TARGETS` with the reason recorded in code.
- That leaves three antiviral proteins, which are now a **named case study**
  rather than the control arm.

`src/data/build_nonkinase_panel.py` builds the real arm: 60 distinct human
non-kinase targets with UniProt binding-site annotation, inside the model's
window. **The filter that matters is not "is this a kinase" but "does this bind
a nucleotide"** — HSP90, DNA gyrase B, helicases, myosins and NADPH-dependent
oxidoreductases are non-kinases with nucleotide pockets, and admitting them
would put the confound inside the arm built to exclude it. Checked against
UniProt's annotated ligand, not the protein name.

**Open question for 124AD0067:** three panel targets have most of their
annotated sites on cotransport ions — `SLC6A3` (14/20), `SLC6A4` (10/16),
`DRD4` (2/4). No drug binds a sodium-coordination residue, so those positions
inflate precision@k the same way the `Site` catch-all did. Deliberately **not**
filtered: zinc cuts the other way, since carbonic anhydrase and HDAC inhibitors
chelate the catalytic zinc directly, so zinc *is* the drug site there. The
ligand is on every feature in `data/nonkinase_ground_truth_sites.json` so it can
be excluded or reported separately — that call is Track C's.

## Track B — 124AD0015 (Core model)

| | Item | Status |
|---|---|---|
| B1 | Drug encoder | ✅ tested, padding-invariant |
| B2 | Protein encoder, attention exposed | ✅ tested, zero weight on padding |
| B3 | Fusion + prediction head | ✅ end-to-end on dummy data |
| B4 | Training loop with checkpointing | ✅ verified `--dummy` |
| B5 | Trained on real DAVIS/KIBA, all four splits | ⏳ **no longer blocked — A4 is done.** The 24-run grid is now the critical path for the whole project |
| B6 | SMILES vocab from train split only | ✅ tested |

## Track C — 124AD0067 (Evaluation, case study, literature)

| | Item | Status |
|---|---|---|
| C1 | `precision@k` module | ✅ rewritten with input guards |
| C2 | Multiple k values | ✅ + achievable-ceiling reporting |
| C3 | Tested on several cases incl. edge cases | ✅ **271 tests**, was 0 |
| C4 | Permutation significance test | ✅ + split-level test |
| C5 | Headline figure mock-up | ✅ `src/evaluation/plots.py` |
| C6 | Antiviral reference sheet | ✅ `antiviral_targets_reference.md` |
| C7 | Five-paper differentiation | ✅ `literature_differentiation.md` |
| C8 | Cross-check the 5 targets against A's data | ✅ done — it found A6 |

---

## What changed in this pass

### Correctness fixes

**The Track A → Track C coordinate mismatch.** UniProt reports 1-indexed
inclusive ranges; `precision_at_k` expects 0-indexed positions. Nothing
converted between them, and the error is silent — a model with *perfect*
attention scored 0.67 instead of 1.00. `src/data/ground_truth.py` now owns
that conversion, and `test_off_by_one_regression` pins it.

**`precision_at_k` returned wrong numbers instead of raising.** `k=500` on a
300-residue protein returned 0.012: `argsort(x)[-500:]` yields 300 indices,
divided by 500 anyway. `k=0` raised `ZeroDivisionError`. Both now raise with a
message naming the problem.

**Tie-breaking bias.** `np.argsort` is stable, so `[-k:]` on tied attention
returned the highest indices — biasing selection toward the C-terminus. On a
flat attention map, sites near the protein's end scored 1.0 on a model that had
learned nothing. Ties are now broken at random against a seeded generator.

**Unsafe shuffle in the split builder.** `rng.shuffle` on a pandas
`StringArray` warns that it may leave duplicates. A duplicated ID puts the same
drug in train and test and voids the paper's central claim. Now shuffles a
`dtype=object` array.

**`explain()` length mismatch.** It counted non-pad tokens while the encoder
measured to the last non-pad position. These disagree on any sequence with an
interior pad, returning a shorter array than the residues it covers and
misaligning every ground-truth index past that point.

**Ground-truth contamination.** The fetcher collected UniProt's `Site`
catch-all, which carries protease cleavage points and chromosomal breakpoints —
positions no drug binds to, scored as correct answers. ~136 DAVIS and ~68 KIBA
annotations. Removed from the fetcher; filtered by description heuristic in the
adapter until the files are re-fetched.

**`coverage_report` hid its own losses.** It summed drop counters over the
*filtered* mapping, so a protein whose sites were all discarded took its counts
out of the total. Truncation losses read as 22 when the real figure was 283.

**p-value floor.** The permutation test used a bare mean, which can return
exactly 0.0 — infinite confidence from 1000 draws. Now `(1+hits)/(1+trials)`.

**Statistical framing.** Added a split-level permutation test. Per-protein
p-values across 400 proteins are 400 hypothesis tests; ~20 land under 0.05 by
luck. The paper needs one test on the mean, per split.

### Added

- `src/data/ground_truth.py` — the A→C adapter, with explicit truncation policy
- `tests/` — 160 tests across 7 files, including an end-to-end integration test
  that runs model → `explain()` → adapter → precision@k → p-value
- `src/data/extract_antiviral.py` — replaces the single-target pipeline; refuses
  to write unless all five targets are present; converts affinities to p-scale
  and keeps the measurement type instead of pooling IC50/Ki/Kd/EC50
- `achievable_ceiling` / `normalised_precision_at_k` — six sites at k=20 caps at
  0.30, so raw precision is not comparable across proteins
- `summarise_splits` — surfaces the cold-pair volume confound, which would
  otherwise be read as pure difficulty. **Two different numbers, do not mix
  them up:** cold-pair *trains* on roughly **71%** of the pairs the other levels
  get, and *uses* roughly **54%** of all measured pairs once the rows it
  discards (one cold entity, not two) are counted. The `pct_of_largest_split`
  column reports the second. Quoting 54% as the training ratio overstates the
  confound by about a factor of two.
- `src/evaluation/run_ladder.py` — the full ladder: loads checkpoints, collects
  explanations, computes fidelity + significance at every level, and writes the
  JSON, the markdown table and the headline figure. Runs today with `--dummy`,
  runs unchanged on real checkpoints in October. Guide C Step 3 asks for exactly
  this: build the plumbing before the numbers arrive, not under time pressure.
- `.github/workflows/tests.yml` — CI runs the suite, verifies the ground truth
  still converts, and smoke-tests the ladder on every push
- `conftest.py` + `pytest.ini` — the suite runs identically from any directory

### Consolidated / removed

- Three overlapping fetchers (`binding_sites.py`, `fetch_binding_sites.py`,
  `fetch_kiba_binding_sites.py`) that filtered features differently → one
- `clean_antiviral.py` → `src/data/extract_antiviral.py`

### A sanity floor worth keeping

`test_an_untrained_model_scores_around_chance_not_above_it` runs an untrained
model through the whole pipeline and asserts the result is **not** significant.
If it ever fires, the metric is measuring an artefact — padding, index bias,
tie ordering — and every real number produced afterwards is suspect.

---

## What is left, and why

> **Superseded -- see § 2026-09-12 at the top.** Items 0-5 are accurate. Item 6, the
> 24-run regression grid called "the critical path" here, was replaced as the critical
> path by the 36-run binary grid on 2026-08-18. Its DAVIS half has since been trained;
> its KIBA half is deferred.

None of these are code problems. They need data or compute — `rest.uniprot.org`
and `bindingdb.org` are unreachable from the environment the code was written
in, so every network step has to run on a team member's own machine.

| # | Item | Owner | Command | Est. |
|---|---|---|---|---|
| 0 | ~~Build the splits~~ | A | `python -m src.data.build_splits` | ✅ **done 2026-08-09** |
| 1 | ~~Antiviral subset~~ | A | `python -m src.data.extract_antiviral --from-cache` | ✅ **done** (3 targets; SARS-CoV-2 documented unavailable) |
| 2 | ~~Re-fetch ground truth with feature types~~ | A | `python -m src.data.fetch_binding_sites --dataset davis` / `kiba` | ✅ **done** |
| 3 | ~~KIBA gene map~~ | A | `python -m src.data.build_gene_map --dataset kiba` | ✅ **done**, 229/229 |
| 4 | ~~Unmapped DAVIS targets~~ | A | `python -m src.data.resolve_unmapped --dataset davis --apply` | ✅ **done**, 19 documented unresolvable |
| 5 | ~~Non-kinase control panel~~ | A | `python -m src.data.build_nonkinase_panel --scan/--select/--build` | ✅ **done**, 60 targets |
| 6 | **Train ColdSite-DTI: 2 datasets × 4 splits × 3 seeds** | **B** | `python -m src.model.run_grid --preflight`, then `python -m src.model.run_grid` | **24 runs, HPC — the critical path** |
| 7 | Fill the remaining 23 baseline cells | A | per `src/data/baselines/README.md` | the long pole |
| 8 | Decide how to treat cotransport-ion sites in the panel | C | see A13 above | a judgement call, not code |

**Everything Track A was blocking is now unblocked.** Item 6 is the critical
path for the entire project: no real precision@k number exists until checkpoints
do. Item 7 is Track A's remaining work and is the long pole overall.

Item 3 used to be a two-hour UniProt ID-mapping job. It is now one command,
because `fetch_binding_sites.py` records the gene symbol and protein name for
every target into `*_provenance.json` — the names were always inside the entries
it downloads, so the map falls out of item 2 with no second pass over the API,
and the symbols are guaranteed to come from the same UniProt snapshot as the
sites they stratify.

Verified download link for item 1 (BindingDB release 2026-07-31, 565 MB zipped,
~3–4 GB unzipped) — `curl -L` does not work in PowerShell, use this:

```powershell
$url = "https://www.bindingdb.org/rwd/bind/chemsearch/marvin/SDFdownload.jsp?download_file=/rwd/bind/downloads/BindingDB_All_202608_tsv.zip"
Invoke-WebRequest -Uri $url -OutFile "data\raw\BindingDB_All_tsv.zip"
Expand-Archive -Path "data\raw\BindingDB_All_tsv.zip" -DestinationPath "data\raw\" -Force
```

After 1 and 2, delete `test_committed_antiviral_file_is_still_incomplete` in
`tests/test_antiviral.py` — it is a deliberate guard on the current broken
artefact and should fail once the artefact is fixed.

## Running the checks

```bash
pip install -r requirements.txt
python -m pytest tests/ -q          # 540 tests, ~13s
python -m src.data.ground_truth     # ground-truth coverage report
```

## Producing the headline figure

Once items 1, 2 and 4 above are done, the ladder is one command per seed. Run
`run_faithfulness` first — it produces the `--accuracy-json` file the ladder
needs (next section).

```bash
python -m src.evaluation.run_ladder \
    --dataset davis --seed 1 \
    --ground-truth data/davis_ground_truth_sites.json \
    --checkpoint-dir results \
    --accuracy-json results/accuracy_davis_seed1.json
```

It writes `results/ladder_davis_seed1.json`, `results/ladder_davis_seed1.md`
(the table for the paper) and `results/headline_davis_seed1.png`.

`--seed` selects which training run to read, and it appears in the output names
too: three ladder runs writing one `ladder_davis.json` would overwrite each
other exactly as unseeded checkpoints did. Checkpoint and result filenames are
built by `src/model/checkpoint_naming.py` at both ends — the writer
(`src.model.train`) and the reader (`run_ladder`) import the same function, so
the two cannot drift apart:

    results/coldsite_dti_{dataset}_{split}_{task}_seed{N}.pt
    results/{dataset}_{split}_{task}_seed{N}_results.json

Two behaviours of `run_ladder` are deliberate. It **refuses to draw the figure**
without accuracy values, because fidelity plotted alone is half the paper's
claim. And `--dummy` output is tagged `DUMMY_PLACEHOLDER` in every filename so a
synthetic run cannot later be mistaken for a result.

## Faithfulness and the accuracy hand-off

```bash
python -m src.evaluation.run_faithfulness --dummy          # no data needed

python -m src.evaluation.run_faithfulness \
    --dataset davis --seed 1 --checkpoint-dir results
```

It writes `results/faithfulness_davis_seed1.json`,
`results/faithfulness_davis_seed1.md` and
`results/faithfulness_davis_seed1.png` — comprehensiveness against its
random-masking control per level, with the **delta** as the only column that is
a result — plus `results/accuracy_davis_seed1.json`, the flat
`{level: accuracy}` file `run_ladder --accuracy-json` reads. Accuracy is read
out of the trainer's own `*_results.json` rather than recomputed, so the
figure's accuracy axis cannot disagree with the runs it reports.


---

# Audit reframe — what was added

New modules, all tested, all runnable today:

| module | purpose |
|---|---|
| `src/evaluation/faithfulness.py` | comprehensiveness, sufficiency, AOPC — **with random-masking controls**. Promoted from optional stretch goal to core. |
| `src/evaluation/model_registry.py` | the adapter contract every audited model must satisfy, plus `validate_adapter` and a uniform-attention control |
| `src/evaluation/aggregate.py` | seed aggregation (mean ± std, flags <3 seeds), Holm-Bonferroni across the grid, the audit table |
| `src/evaluation/target_family.py` | kinase / non-kinase stratification — the confound control |
| `src/evaluation/run_audit.py` | the audit grid: models × splits × datasets × seeds, with Holm correction applied once over the whole family |
| `src/evaluation/baseline_adapters.py` | registered stubs for DeepDTA / HyperAttentionDTI / MolTrans that raise with instructions until filled in |
| `src/evaluation/plots.py` | rewritten: multi-model curves with seed error bars, control floor, stratified panels, faithfulness-vs-control bars |

## The confound, measured

`python -m src.evaluation.target_family` on the committed data:

| | DAVIS | KIBA |
|---|---|---|
| targets | 409 | 224 |
| kinase | 230 | 0 |
| **non-kinase** | **0** | **0** |
| control usable | **no** | **no** |

KIBA is zero because it uses UniProt accessions, not gene symbols, and cannot be
classified without a mapping. **The control arm does not currently exist.** This
is the highest-priority gap in the project.

> **Stale.** The control arm now exists: the non-kinase panel (A13) supplies 60
> targets, `control_is_usable: True`, and KIBA is mapped 229/229.

## Remaining work, by owner

> **Stale -- kept for history.** Written before the splits existed. Every item below
> is either done or superseded; see § 2026-09-12 at the top.

| # | Item | Owner | Blocking |
|---|---|---|---|
| 1 | Antiviral rebuild — 5 targets, ≥20 non-kinase | A | the entire control arm |
| 2 | Ground-truth re-fetch with feature types | A | metric validity |
| 3 | KIBA accession → gene mapping | A | control on KIBA |
| 4 | 33 unmapped DAVIS targets | A | coverage |
| 5 | **Split files for both datasets** | A | items 6, 7 below — the whole grid |
| 6 | 24 training runs (2 × 4 × 3 seeds) | B | every real number |
| 7 | Baseline adapters (3 models) | A + B | the audit framing |
| 8 | Differentiation doc rewrite | C | Related Work |
| 9 | Audit grid + Holm correction | C | Results |

Items 1–3 unblock 6, which unblocks 9. **Item 5 is the hard blocker for Track
B**: `data/splits/` is empty and the DeepDTA source files under
`src/data/baselines/deepdta/data/` that `build_splits.py` reads are not present.

---

# Track B (124AD0015) Part 2 — state

| Item | Status |
|---|---|
| Seed-aware checkpoint naming, shared writer/reader | ✅ `src/model/checkpoint_naming.py` |
| Faithfulness runner + random-masking control + delta | ✅ `src/evaluation/run_faithfulness.py` |
| Accuracy hand-off to Track C | ✅ `accuracy_{dataset}_seed{N}.json` |
| Truncation decision (`exclude`, max_len 1000) | ✅ decided, evidenced, written up |
| Cold-pair volume decision (report, don't subsample) | ✅ decided; `--train-subsample` ready for the control |
| 24-cell grid runner with preflight + one-cell validation | ✅ `src/model/run_grid.py` |
| Attention extraction for HyperAttentionDTI / MolTrans | ✅ `src/evaluation/attention_projection.py` |
| `validate_adapter` passing for ColdSite-DTI | ✅ padded and unpadded |
| Methods / Model Architecture draft | ✅ `paper/methods_track_b.md` |
| **24 training runs** | ⛔ blocked on item 5 |
| **Faithfulness on real checkpoints** | ⛔ blocked on the grid |
| **Volume-matched sensitivity run** | ⛔ blocked on the grid |

Everything above the line runs today and is covered by the test suite. The
three blocked items are blocked on data, not on code:

```bash
python -m src.model.run_grid --preflight   # says exactly what is missing
```

## Grid semantics, settled

**2 datasets × 4 splits × 3 TRAINING seeds = 24 runs**, not 72. The three seeds
vary weight initialisation and batch order on one fixed split per cell. The
repository is consistent on this: the Part 2 guide's loop varies only `--seed`
against a seed-independent `--split-dir`; `build_all_splits()` takes no seed
argument and writes one split per cell; `run_audit.build_grid` has a single seed
axis; and both the guide and this file state the count as 24.

One consequence belongs in the paper: seed error bars measure **initialisation
variance, not split-selection variance**.
