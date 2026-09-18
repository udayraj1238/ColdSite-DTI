# Methods — data, ground truth, the audited models, and evaluation

Draft for the paper's Methods section, companion to `methods_track_b.md` (ColdSite-DTI's
architecture, its training, faithfulness on our model, and attention extraction for the
baselines). This file covers what that one does not: the data and splits, the binding-site
ground truth, the family control, how every audited model is trained, the plausibility and
significance machinery, faithfulness for the published models, and the controls that
validate the measurement itself.

Every number here was produced from the code and data in this repository on 2026-09-12
and can be regenerated with the command given beside it. Nothing depends on the audit's
results. Citations are by model name and year until the venue's style is fixed.

---

## 1. Data

***Figure 0*** (`results/figures/fig0_design.pdf`) is this section and the next eight in
one picture: what was trained, how each explanation is read, what it is scored against,
and where the correction is applied.

### 1.1 Datasets

DAVIS and KIBA are loaded directly from the files published with DeepDTA (2018),
following that repository's own loading procedure (`src/data/load_data.py`). DAVIS
dissociation constants are converted to pKd = −log10(Kd / 10⁹); KIBA scores are used as
published.

| | pairs | drugs | targets | label range |
|---|---|---|---|---|
| DAVIS | 30,056 | 68 | 442 | pKd 5.000–10.796 |
| KIBA | 118,254 | 2,111 | 229 | KIBA score 0.0–17.2 |

### 1.2 The binary task

The audit compares models on one accuracy metric, so every model does the same task.
HyperAttentionDTI (2022) and MolTrans (2021) are classifiers in their published form, so
the task is binary: a pair binds if **pKd ≥ 7.0 (DAVIS)** or **KIBA score ≥ 12.1 (KIBA)**,
DeepDTA's published thresholds. One constant (`BINARY_THRESHOLD`, `src/model/dataset.py`)
supplies the threshold to all four trainers and to training, validation and test alike,
so every model is trained, early-stopped and scored against identical labels.

8.3% of DAVIS pairs and 21.0% of KIBA pairs are positive. A model that always predicts
"does not bind" is 92% accurate on DAVIS, so **accuracy is not reported as a result;
AUROC is the accuracy axis**. The positive rate also varies across DAVIS test sets
(7.7% random, 6.0% cold-drug, 7.5% cold-target, 5.7% cold-pair), which moves AUPRC's
chance level from one level to the next; AUPRC is reported alongside AUROC but not
compared across levels.

---

## 2. Splits

### 2.1 Four levels of difficulty

Each dataset is split four ways (`src/data/build_splits.py`, seed 42, 70 / 10 / 20 of the
held-out entity for train / validation / test):

| level | held out of training |
|---|---|
| random (warm) | nothing: pairs are sampled at random |
| cold-drug | every pair of 20% (test) and 10% (validation) of the **drugs** |
| cold-target | every pair of 20% and 10% of the **targets** |
| cold-pair | test and validation pairs need an unseen drug **and** an unseen target; pairs with only one of the two unseen are discarded |

Validation and test entities never overlap each other or training; this is asserted
when the splits are built (`check_no_leakage`). The split files were rebuilt and found
identical on three machines, and every training notebook refuses to start unless its
row counts match these:

| dataset | level | train | valid | test | test targets | test drugs |
|---|---|---|---|---|---|---|
| DAVIS | random | 21,039 | 3,006 | 6,011 | 442 | 68 |
| | cold-drug | 21,658 | 2,652 | 5,746 | 442 | 13 |
| | cold-target | 21,080 | 2,992 | 5,984 | 88 | 68 |
| | cold-pair | 15,190 | 264 | 1,144 | 88 | 13 |
| KIBA | random | 82,778 | 11,825 | 23,651 | 228 | 2,057 |
| | cold-drug | 83,807 | 12,073 | 22,374 | 229 | 422 |
| | cold-target | 85,452 | 10,701 | 22,101 | 45 | 2,093 |
| | cold-pair | 58,041 | 1,334 | 4,375 | 45 | 380 |

There is **one fixed split per level**. The three seeds reported for every cell are
training seeds (weight initialisation and batch order), so the spread across seeds is
initialisation variance, not split-selection variance, and is labelled as such.

### 2.2 The levels are categories, not a ladder of severity

On DAVIS the levels do not order by difficulty the way their names suggest: for every
model trained so far, cold-target is the most accurate cold level, close to random, and
cold-drug falls far below it, near cold-pair (DeepDTA binary AUROC: random 0.929,
cold-target 0.884 and cold-pair 0.749 on targets unseen by sequence — 0.908 and 0.728 on
all test rows, §2.4 — cold-drug 0.692). Holding out 13 of 68 drugs removes
far more of what a model learns from than holding out 88 of 442 kinases that share a
binding pocket with the rest. The levels are therefore
treated as four categories of distribution shift throughout, and no analysis assumes a
monotone order.

### 2.3 Cold-pair's training volume, and the control for it

Because cold-pair discards every pair with exactly one unseen entity, it trains on less:
15,190 DAVIS rows against 21,039 for random (72.2%), and it uses 55.2% of all measured
pairs (KIBA: 58,041 against 82,778, 70.1%; 53.9% used). Its DAVIS validation set is 264
rows. Part of any cold-pair accuracy drop could therefore be *less data* rather than *a
harder task*.

The other levels are not subsampled to match; each level is trained on everything
legitimately available to it. Instead a **volume-matched control** bounds the effect:
ColdSite-DTI is retrained on DAVIS random with its training set cut to 15,190 rows, three
seeds, each seed drawing its own subsample; validation and test are not cut, so the
control is scored on exactly the random test set. Full random minus the control is the
cost of fewer rows; the control minus cold-pair is the genuine cold-pair difficulty.
(`notebooks/colab_volume_control.ipynb`; outputs are renamed `_trainsub15190` and kept
apart from the grid, since `src/model/train.py` would otherwise name them exactly like the full
random cells.)

A second comparability gap cannot be fixed by subsampling: plausibility is averaged per
protein, and cold-target and cold-pair test sets hold 88 DAVIS / 45 KIBA proteins against
442 / 229 for the others. Wider seed-to-seed variance in those cells is expected on that
basis alone.

### 2.4 Targets are sequences, not names (DAVIS)

The splits hold targets out by name. In DeepDTA's DAVIS sequence file, which this study
and most DTI benchmarks use, a name is not a sequence: all 54 variant targets with a
wild-type entry (ABL1(T315I), EGFR(T790M), BRAF(V600E) and the rest) carry exactly the
wild-type sequence, so the 442 targets are 379 distinct sequences, and ten targets'
sequences contain few or none of the 85 KLIFS ATP-pocket residues (RET and its three
mutants are RET's extracellular residues 1–430) (`src/data/sequence_audit.py`,
`results/sequence_audit_davis.md`). Two consequences follow. At cold-target, 12 of the 88
held-out targets (816 of 5,984 test rows, 13.6%) and at cold-pair 11 of 88 (143 of 1,144,
12.5%) are identical in sequence to a training target, so they are not unseen; cold-pair's
validation set is 13.6% such rows. And at every level, averaging per target name enters one
sequence up to seventeen times.

Nothing is retrained; the evaluation stops counting what it should not
(`src/evaluation/exclusions.py`). **Accuracy** at cold-target and cold-pair is reported on
the test rows whose target is unseen by sequence, beside the full-test value, each cell
re-scored by its own trainer's test pass, which reproduces every recorded test AUROC to
four decimals (`src/evaluation/clean_accuracy.py`). **Explanation metrics** drop the
seen-by-sequence targets at the cold levels and the ten pocketless targets at every level,
and count one protein per distinct sequence. What cannot be undone without retraining is the
leak into cold-pair's validation set, which influenced checkpoint selection; it is stated as
a limitation. KIBA has none of these problems: its 229 targets are 229 distinct sequences,
no cold test target is seen by sequence, and every sequence contains its pocket
(`results/sequence_audit_kiba.md`).

---

## 3. Binding-site ground truth

### 3.1 Source and feature types

Binding sites are UniProt sequence features of type **Binding site**, **Active site** and
**Nucleotide binding** (the last because DAVIS and KIBA are kinase panels and UniProt
annotates the ATP pocket under it). UniProt's catch-all **Site** type is excluded: it
carries protease cleavage points and chromosomal breakpoints, which no drug binds and
which would count as correct answers. UniProt's 1-indexed inclusive ranges are converted
to 0-indexed residue positions by one module (`src/data/ground_truth.py`) that every
consumer goes through; reading the raw coordinates directly costs about a third of the
score of a perfect explanation, silently.

### 3.2 Resolving targets to UniProt entries

KIBA names its targets by UniProt accession, which are used as given. DAVIS names them by
gene symbol, resolved through UniProt gene search. 63 DAVIS identifiers are mutant or
phosphorylation variants (e.g. ABL1(T315I)); these take the annotation of their wild-type
entry, so a mutant is scored against wild-type sites. Five DAVIS targets are resolved by
manual override because name search chose the wrong protein: MST1, and four found by the
sequence alignment below (PKAC-alpha, PAK1, MLCK and CDK11, whose DAVIS sequences are
identical to PRKACA, PAK1, MYLK3 and CDK19 and align to nothing in the entries search had
returned). Overrides are recorded per target in `data/davis_target_overrides.json`.

### 3.3 Numbering sites along the sequence the model reads

UniProt numbers residues along its canonical sequence; the models read the dataset's own.
Where the two differ, a UniProt residue number points at the wrong residue of what the
model saw, or past its end. Each dataset sequence is therefore aligned to its UniProt
sequence (`src/data/align_ground_truth.py`) and every annotated residue is carried across
individually:

- inside an identical aligned stretch of at least 10 residues, it maps across;
- a single-residue substitution between two such stretches maps too, so point mutants
  keep sites at the mutated residue (the gatekeeper among them);
- anything else, outside a fragment or inside a divergent region, is dropped;
- a target whose UniProt sequence has changed length since its sites were fetched keeps
  no sites.

On DAVIS, 54 targets with sites hold a sequence different from UniProt's (kinase-domain
fragments, longer isoforms, constructs). Some hold a fragment that omits the annotated
kinase domain entirely (DAVIS's ROCK2 is UniProt residues 686–1388, its MLK1 732–1104);
five targets lose every site this way and leave the evaluation, rather than being scored
against residues they do not contain. On KIBA, 212 of 221 targets with sites are
identical to UniProt; seven differ at one or two residues, none of them sites; two are
longer isoforms (PIM1, 404 residues against UniProt's 313; SGK2, 427 against 367), whose
24 site residues move. No KIBA site is dropped. The UniProt-numbered files are kept
beside the aligned ones, with a per-target report of what happened to every site.

### 3.4 The 1,000-residue window

Proteins are truncated to their first 1,000 residues, and sites beyond the window are
excluded (`truncation="exclude"`); a protein left with no site is removed from evaluation
rather than scored zero (`python -m src.data.ground_truth`):

| | DAVIS | KIBA |
|---|---|---|
| targets in file | 442 | 229 |
| usable within the window | 402 | 212 |
| annotated positions used | 5,035 | 2,726 |
| positions past the window | 280, on 24 targets | 164, on 14 targets |
| targets removed by the window | 16 | 9 |

The removed targets are systematically the longest (median final annotated residue 1,320
against 312 for retained DAVIS targets; 1,212 against 272 on KIBA), mostly large
multidomain receptor kinases. Results should not be extrapolated to proteins much longer
than the window. The reasoning for excluding rather than retaining those sites is in
`methods_track_b.md` §4.1.

### 3.5 Sites on cotransport ions

Three non-kinase panel targets (§4) carry most of their annotated sites on cotransport
ions: SLC6A3 (14 of 20 positions), SLC6A4 (10 of 16) and DRD4 (2 of 4), all sodium or
chloride. No drug binds a sodium-coordination residue. Excluding every metal would be
wrong in the other direction: carbonic-anhydrase and HDAC inhibitors chelate the catalytic
zinc, which is then the correct answer. **The primary result excludes the cotransport
ions** (Na⁺, K⁺, Cl⁻; `run_control --exclude-cotransport-ions`, outputs suffixed
`_noions`), because the ground truth is meant to mark where drugs bind; zinc and every
other ligand stay in. Including them is reported as a sensitivity analysis. A residue
annotated for an ion and for another ligand keeps its site through the other annotation.
The choice affects only the non-kinase panel: DAVIS and KIBA carry no cotransport-ion
sites, and the panel carries 34 ion features, on 4 targets.

---

### 3.6 A second ground truth: the KLIFS ATP pocket

UniProt's annotation is protein-level and sparse (about a dozen residues per kinase) and
uneven across kinases. As a second, structure-derived ground truth we use the KLIFS pocket
(Kanev et al., *Nucleic Acids Res.* 2021): the same 85 residues lining the ATP cleft of
every kinase, defined by structural alignment, where the ATP-competitive inhibitors that
make up most of DAVIS bind. KLIFS gives each kinase's UniProt accession and its 85-residue
pocket sequence; the pocket is placed on the UniProt canonical sequence in order, with
KLIFS's 19 structural regions (van Linden et al., *J. Med. Chem.* 2014) kept contiguous
where the kinase has no insertion, and carried onto each dataset's sequences by the same
alignment as the UniProt sites (§3.3). Placement agrees with KLIFS's own residue numbers
for all 41 kinases checked against a crystal structure carrying KLIFS's reference pocket
(36 identical, 5 at a constant numbering offset) (`src/data/klifs_pocket.py`). DAVIS: 432 of
442 targets are covered (10 are not in KLIFS); for kinases with two kinase domains the
assayed domain is taken from the target name (JH1/JH2, KinDom.1/2). Precision@k against
the pocket has a chance level near 0.13 (85 residues of a typical chain) rather than 0.02,
and the two ground truths answer different questions: whether attention reaches the
pocket, and whether it lands on the residues UniProt annotates.

## 4. The kinase confound, and the control for it

DAVIS and KIBA are kinase panels. A cold-target kinase shares the ATP-pocket architecture
of hundreds of kinases seen in training, and UniProt annotates that pocket in nearly all
of them, so a model could score well on cold-target plausibility by recognising "an ATP
pocket" rather than anything specific to the protein. Stratifying within the datasets
cannot test this: classified by gene family (`src/evaluation/target_family.py`), DAVIS is
429 kinase / 0 non-kinase / 13 unknown and KIBA 227 / 0 / 2.

The control is therefore a **transfer** condition. A panel of **60 non-kinase proteins**
from BindingDB, each with UniProt binding-site annotation, is scored with the same
checkpoints and pipeline as the model's own test split (`src/evaluation/run_control.py`).
Sequences come from UniProt, not from BindingDB's target-chain column, because the site
coordinates are UniProt's and BindingDB chains are often tagged or truncated constructs.
The panel's protein, family and drugs are all unseen, so this is strictly harder than
cold-target, and it is described as transfer rather than as stratification. BindingDB
writes ChemAxon extended SMILES, whose annotations (e.g. `|r|`, relative
stereochemistry) follow a space, on 4,694 of the panel's 21,145 pairs; every SMILES is
read up to its first whitespace. 16 pairs whose molecule contains a wildcard atom or a
dative bond, which no audited model's alphabet can represent, are excluded for all
models alike; all 60 proteins remain. DAVIS and KIBA contain neither. The panel is
sized to clear the ≥20-target minimum the analysis sets for any family comparison; the
antiviral targets (HIV-1 protease and reverse transcriptase, influenza neuraminidase)
remain inside it as ordinary members. They were once planned as a case study of their
own; that was cut, because the 2026-07-31 BindingDB release collapsed all 18,149
SARS-CoV-2 rows under a single 7,096-residue polyprotein and what remained was three
proteins — too few to carry a claim, and already covered by the panel.

---

## 5. The audited models and how each is trained

| model | role | explanation |
|---|---|---|
| DeepDTA (2018) | accuracy anchor | none; never audited |
| ColdSite-DTI (this work) | subject | single-query cross-attention (`methods_track_b.md` §1) |
| HyperAttentionDTI (2022) | subject | attention over convolution positions, projected to residues |
| MolTrans (2021) | subject | attention over ESPF subword tokens, projected to residues |

DeepDTA has no attention and is not given one: a saliency map computed for it would put
a different method's output in a table read as DeepDTA's. It is present so a reader can
see whether the interpretable models pay an accuracy cost.

Each published model is trained with its authors' recipe and tokeniser, from the
vendored repository, with only the data loading replaced; an audit that retrained a
subject under a different optimiser would measure a model its authors never released.

**A published model can also be audited without retraining it, when the claim is about
what its explanation is a function of.** Results §7e reports one such case. The procedure
is stated here because it is a method, not an anecdote: obtain the released code at a
recorded commit and licence, locate the tensor the paper's figure plots, and trace which
inputs reach it in the forward pass. If a drug tensor never reaches a per-residue map,
then for a fixed protein that map is identical for every ligand, for any weights — a fact
about the computation graph that no amount of retraining can change and no measurement is
needed to establish. What such a reading cannot support is any statement about how well
that model's map agrees with a ground truth, or about its accuracy; we report neither for
a model we did not train. The record for the one case in this paper, with line references,
licence, access date and a re-check recipe, is `results/evidti_code_audit.md`.

- **DeepDTA**: a PyTorch port of the published architecture (the original is TF1-era
  Keras), Adam, learning rate 10⁻³, batch 256, `BCEWithLogitsLoss`, gradient clipping at 5.
- **ColdSite-DTI**: `methods_track_b.md` §2, with `BCEWithLogitsLoss`; batch 64 on GPUs
  with ≥14 GB, 16 otherwise.
- **HyperAttentionDTI**: AdamW, learning rate 5×10⁻⁵, weight decay 10⁻⁴ on weights and 0
  on biases, CyclicLR from the base rate to 10× stepped per optimiser update,
  cross-entropy with no class weighting (the published per-dataset weights do not match
  our splits' prevalence), effective batch 32. On smaller GPUs the batch is 8 with 4
  gradient-accumulation steps, which leaves the effective batch and the gradient
  unchanged; CyclicLR still steps per optimiser update.
- **MolTrans**: Adam, learning rate 10⁻⁴, batch 16 (`BIN_config_DBPE`). The published loss
  is a sigmoid followed by binary cross-entropy; `BCEWithLogitsLoss` is the same function,
  numerically stabler. Two defects in the published model are handled without editing it.
  Its forward pass reshapes by the configured batch size rather than the tensor's, so a
  final partial batch is scored silently wrong; the batch size is set per batch
  (`_fit_batch_size`). And its interaction map calls dropout without the training flag,
  so dropout stays on at inference and test predictions carry that noise, as in the
  published model.

All four share one checkpoint-selection rule (`src/model/early_stopping.py`): up to 100
epochs; the checkpoint is the lowest validation loss **among epochs ≥ 10**; early
stopping fires after a patience of 15 epochs without improvement, and never before
epoch 10. DeepDTA's patience is 10, on both datasets: it is the accuracy anchor, its
checkpoints never enter the explanation axis, and its DAVIS cells were trained with it,
so aligning it would split DeepDTA across two settings rather than unify the grid. The floor exists because DAVIS cold-pair validation is 264 rows:
without it, validation loss bottomed out by epoch 2 on the sparse levels while random
trained to epoch 16, and the explanation axis would have compared an undertrained cold
checkpoint against a trained warm one. MolTrans's published script trains a fixed number
of epochs (50 by default; its configuration lists 13) and keeps the best validation
AUROC; it is trained under the shared rule instead, so that every subject's checkpoint is
chosen the same way.

Seeds 1, 2 and 3 are run for every cell. cuDNN's LSTM kernels are not deterministic: an
identical re-run of the DAVIS regression grid moved single seeds by up to 0.058 CI while
split means agreed within 0.024. Only split means with their spread across seeds are
reported, never a single seed.

---

## 6. Plausibility

**precision@k** is the fraction of a protein's *k* most-attended residues that are
annotated sites (`src/evaluation/precision_at_k.py`). Ties in attention are broken at
random rather than by a stable sort: a stable sort returns tied blocks lowest-index
first, which on the exact zeros and saturated softmaxes attention maps produce biases
selection towards one end of the protein and turns a positional artefact into apparent
quality. *k* ∈ {5, 10, 20}; *k* = 10 is the headline.

A protein with fewer than *k* annotated sites cannot score 1.0, so each value is reported
with its **achievable ceiling**, min(|sites|, *k*)/*k*, and as **normalised precision**
(precision / ceiling). Proteins with no usable site, or shorter than *k*, are skipped and
counted, never scored zero.

Every model's explanation reaches this metric as one non-negative weight per residue the
model actually saw (`methods_track_b.md` §1.6, §5); `validate_adapter` and
`check_adapters` verify the length against trained checkpoints, because an explanation of
the wrong length does not fail, it misaligns every site.

**One window for every model.** Explanations are scored over the first 1,000 residues,
the window the ground truth is cut to (§3.4). ColdSite-DTI and HyperAttentionDTI never
read past it. MolTrans reads 545 subword tokens, which reach past residue 1,000 on 115 of
DAVIS's 442 proteins and 44 of KIBA's 229; its explanation is cut to the window
(`collect.py`), because attention past it would compete for the top *k* where no site
can exist. MolTrans never covers fewer residues than the window on either dataset, so the
cut removes attention without leaving any window residue unseen. Faithfulness uses the
same window (§8), so both axes describe the same top *k*.

**Unit of averaging.** Both the ladder and the audit table average over proteins, one
test pair each: the first pair of each protein in the test file (`pairs_per_target = 1`,
`src/evaluation/collect.py`, `run_ladder.collect_explanations`). precision@k is a
per-protein quantity, and averaging every pair would enter a protein once per drug it
was measured against, weighting proteins by how many drugs they were tested with and
reporting correlated pairs as independent observations. *n* is therefore the number of
proteins with usable sites: 402 on DAVIS random and cold-drug, 79 on cold-target and
cold-pair (KIBA 211, 212, 42, 41).

## 7. Significance and aggregation

**Split-level permutation test** (`src/evaluation/significance_test.py`). The null draws
*k* uniformly random positions for every protein in the split and takes the mean,
repeated 1,000 times (ladder) or 500 (audit grid). Drawing per protein keeps the split's
own mix of lengths and site counts: a short protein with many sites has a much higher
chance level than a long one with two, and a pooled null would wash that out. The p-value
uses the add-one estimator, (1 + #null ≥ observed) / (1 + trials). One test per split
mean, never per protein; hundreds of per-protein tests would yield significant proteins
by chance alone.

**Across seeds**, each cell reports the mean and sample standard deviation (ddof = 1)
over three seeds; a cell with fewer than three is flagged and not quoted as an estimate.
A cell's p-value is the median over its seeds, not the smallest.

**Multiple comparisons.** Holm–Bonferroni is applied once, over the whole audit family
(every model × dataset × level), after every cell's raw p-value has been collected.
Correcting within each model and pooling would define the family after seeing the
results. Each dataset's arm is its own family, sized by the cells that arm measures: **16
on DAVIS** (three audited models and the uniform control × four levels) and **6 on KIBA**
(two audited models and the uniform control × two levels, §11). The control is scored only
at levels the arm trains, so an untrained level cannot enlarge a family and make every
threshold stricter than the design specifies. DAVIS's sixteen attention cells and the
twelve integrated-gradient cells of Results §7c are separate families, and no claim
compares a corrected p from one with a corrected p from another.

**Effect size.** With a thousand or more evaluations per level almost any difference is
significant, so precision@k is always reported beside its chance level and ceiling, and
the result is the distance from chance (and normalised precision), not the p-value.

**Negative control.** A uniform explainer (`uniform_control`, flat weight over every
residue) runs through the same grid. A model indistinguishable from it has no
explanatory content at that level, whatever its accuracy.

## 8. Faithfulness for the published models

Faithfulness is measured as in `methods_track_b.md` §3 (comprehensiveness against a
random-masking control; the reported quantity is the difference), on the first 200 test
pairs of each level, *k* = 10, 5 random-masking trials per pair. Two changes are needed
to apply it to the published models, and both keep the intervention identical across
models (`src/evaluation/residue_space.py`):

- **Residues are masked in residue space.** ColdSite-DTI masks a residue by writing its
  unknown token, one token per residue. In HyperAttentionDTI's alphabet that token value
  is alanine, and MolTrans's input is subword tokens that each cover several residues.
  For both, a masked residue is instead replaced by `X` (unknown amino acid) in the
  sequence, which is then re-tokenised by the model's own tokeniser. Masking is
  confined to the 1,000-residue window (§6); residues past it are passed through
  unchanged. Every model therefore receives the
  same intervention: the residue becomes an unknown amino acid, as seen through that
  model's input encoding.
- **The prediction compared is the model's own decision quantity.** HyperAttentionDTI
  outputs two logits; faithfulness uses their difference (the log-odds), the quantity its
  AUROC is computed on, because the positive logit alone can move without the prediction
  moving. MolTrans's inference-time dropout is held fixed by running every forward pass
  under the same random seed, so the difference between two predictions reflects the
  input alone.

## 9. Validating the measurement: positive control

Near-chance plausibility is only a finding if the pipeline recognises a good explanation
when it sees one. `src/evaluation/positive_control.py` scores explanations of known
quality with the audit's own functions, on the real test proteins and ground truth. For a
dose *d*, every residue gets a random weight and each annotated site is lifted above all
non-sites with probability *d*: *d* = 1 is an oracle, *d* = 0 noise. Faithfulness is
validated the same way on a planted model whose prediction depends only on the residues
at a protein's annotated sites.

On both datasets and all four levels: no annotated site lies outside the sequence the
model sees; the oracle scores its ceiling exactly and is significant; noise scores chance;
and the oracle explanation is load-bearing on the planted model while noise is not. The
permutation test reliably detects an explanation that ranks as few as 2% of true sites
first at every level, including cold-target's 79 DAVIS and 42 KIBA proteins
(`results/positive_control_{davis,kiba}.md`). The same run fails on DAVIS's sites before
their re-numbering (146 sites outside the sequence), so it detects the class of error §3.3
corrects. An audited model's precision@k can be read against this curve as the fraction
of sites its attention effectively ranks first.

---

## 10. Decisions taken (2026-09-12)

These were open in the first draft; each is now settled and applied in the code.

1. **MolTrans's window.** Its explanation is cut to the same 1,000 residues as every
   other model and the ground truth (§6), for plausibility and faithfulness alike.
2. **Unit of averaging.** The ladder scores one pair per protein, the same pairs as the
   audit table (§6). Ladders computed before this, including the dry-run numbers in
   STATUS.md, averaged over every pair.
3. **Early-stopping patience.** DeepDTA keeps patience 10 on both datasets, stated rather
   than aligned (§5). Each model's patience is pinned explicitly in the grid notebook.
4. **Cotransport ions.** Excluded in the primary result; all ligands as sensitivity (§3.5).
5. **Truncation figures.** Recomputed on the re-numbered ground truth
   (`methods_track_b.md` §4.1): 3.8% (DAVIS) and 4.1% (KIBA) deflation under the
   retaining policy, level-dependent from 2.3% to 7.1%; ceilings within 0.8%.

## 11. The KIBA replication arm

KIBA is a replication, not a second exploration, and its scope was fixed before any KIBA
number existed (2026-09-14, on compute grounds; Limitations states that honestly). It
reuses every decision above unchanged — splits, binary threshold, ground-truth
re-numbering, the 1,000-residue window, one pair per protein, 200 faithfulness pairs, three
seeds, the same nulls and the same positive control — and nothing in it was tuned on KIBA.

**Scope.** Two levels, **random and cold-drug**, for the two audited published models
(HyperAttentionDTI, MolTrans) and the DeepDTA accuracy anchor: 18 cells. Cold-drug is the
level KIBA can support and DAVIS cannot (422 held-out drugs against 13); cold-target and
cold-pair are weaker on KIBA than on DAVIS (45 held-out targets against 88) and are not
trained. ColdSite-DTI is not included, so our own model is audited on one dataset and the
published ones on two. The explanation-side analyses of Results §7–§7c (readout variants,
per-pair drug contacts) are DAVIS-only. Integrated gradients run on both: KIBA uses the
identical implementation and settings (32 steps, the padding-embedding baseline), over the
two audited models at both trained levels, three seeds each.

**Numerical precision is per model.** DeepDTA and HyperAttentionDTI train under float16
autocast with loss scaling; MolTrans trains in full precision, because its vendored
hand-written LayerNorm divides by `sqrt(var + 1e-12)` and 1e-12 underflows in float16,
producing NaN weights from the first epoch. MolTrans's KIBA cells are therefore full
precision like its DAVIS cells; the other two carry the precision caveat Limitations
states.

**Sequence policy.** None of DAVIS's three sequence problems (§2.4) occurs in KIBA: no
target is seen by sequence in training at either level, and no target lacks the kinase
pocket. The policy therefore excludes no KIBA target, and `clean_accuracy` is not needed.

**Cells longer than one compute session** continue from their last finished epoch,
restoring model, optimiser, scheduler, loss scaler and every random-number generator, so a
cell interrupted by a session limit is not restarted and not partially scored.

---

## 12. Reproducibility: what a reader needs to re-run this

Every number in Results is written by a command in this repository, into a file the
Results section names. Nothing is typed by hand, and nothing is averaged in a spreadsheet.

**The commands.** Training is one entry point per model
(`src/model/train.py`, `train_deepdta.py`, `train_hyperattentiondti.py`,
`train_moltrans.py`, `train_drugban.py`), each taking `--split-dir --dataset --split
--seed` and writing a checkpoint plus a `_results.json` with its test metrics, selected
epoch and the arguments it ran under. Analysis is one command over a finished grid:

    python -m src.evaluation.run_all --dataset {davis|kiba} --checkpoint-dir <grid>

which runs faithfulness, both ladders, the audit with its Holm correction, the non-kinase
control in both ion settings and the positive control, and writes
`analysis_summary_<dataset>.md`. The explanation variants (integrated gradients,
alternative readouts) are `run_ladder --model <name>_ig`; their family correction is
`src/evaluation/ladder_family.py`; the figures are `src/evaluation/paper_figures.py`.

**Determinism and what is not deterministic.** Splits are built by
`src/data/build_splits.py` from the published DAVIS and KIBA files and are byte-identical
on rebuild (checked 2026-09-18) and across three machines. Ground-truth re-numbering is
`src/data/align_ground_truth.py`, run once per dataset, and its output is in the
repository. Training on a GPU is *not* bit-reproducible — cuDNN kernel selection is
nondeterministic — which is why every cell is trained three times and no claim rests on a
single seed. Interrupted cells resume from their last finished epoch
(`src/model/resume.py`), and resumption is bit-identical on a CPU, which the tests assert.

**Where the numbers came from, physically.** Training ran on Kaggle's two-T4 sessions
under the notebooks in `notebooks/`; each notebook records its plan, self-stops an hour
inside the session limit, and writes the same file layout as a local run. Analysis ran
locally on CPU except where a section says otherwise. Result folders are committed as
their markdown tables (`results/analysis_*`), with `RUN_NOTES.txt` in each recording any
way that run departed from a plain `run_all` and why.

**Checks.** 924 tests run against the analysis and training code, including planted-case
tests for every instrument the audit trusts: the positive control detects a 2% dose, the
projection from convolution positions to residues is checked on synthetic maps, the
matched masking control is checked draw-for-draw against the unmatched one, and the
adapter contract is checked on every registered model before it is used for real.

**Vendored models.** The three published subjects are cloned unmodified into `baselines/`
with their licences (MolTrans BSD-3, DrugBAN MIT) and a `PROVENANCE.md` recording the
commit and date. An audit that edited its subject would be measuring something else; every
adaptation lives outside those directories, in an adapter that exposes `predict` and
`explain` and nothing more.
