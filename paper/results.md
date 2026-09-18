# Results (draft)

Drafted 2026-09-13 from the cells finished so far. Every number below is read from a
trained cell's `_results.json` (DAVIS binary grid, Kaggle account 1, commit v1 of
`kaggle_davis_binary_grid36.ipynb`) or from a file named beside it. The DAVIS grid is
complete: 48 of 48 cells (4 models x 4 levels x 3 seeds), verified cell by cell against
the AUROC each recorded. KIBA, the replication, is complete (18 of 18 cells; §8, added
2026-09-16), and its integrated-gradient arm with it (§8.4, added 2026-09-18).
No *[PENDING]* marks remain: the antiviral case study was cut on 2026-09-18, because
after the 2026-07-31 BindingDB release put all 18,149 SARS-CoV-2 rows under one
7,096-residue polyprotein it was three proteins, and §6's 60-protein panel is the
non-kinase arm it was meant to be. All
values are test-set means ± sample standard deviation over three training seeds; a
difference smaller than the spread is not reported as one.

---

## 1. Accuracy across the four split levels (DAVIS, binary)

**Table R1.** Test AUROC, mean ± sd over seeds 1–3. AUPRC is given with each test set's
positive rate, which is its chance level; AUPRC is not comparable across levels because
that rate differs.

| model | random | cold-target | cold-drug | cold-pair |
|---|---|---|---|---|
| DeepDTA (anchor) | 0.929 ± 0.002 | 0.907 ± 0.003 | 0.692 ± 0.044 | 0.728 ± 0.035 |
| ColdSite-DTI (ours) | 0.924 ± 0.001 | 0.857 ± 0.011 | 0.721 ± 0.008 | 0.624 ± 0.099 |
| HyperAttentionDTI | 0.937 ± 0.005 | 0.915 ± 0.001 | **0.760 ± 0.042** | 0.694 ± 0.038 |
| MolTrans | 0.923 ± 0.002 | 0.874 ± 0.005 | 0.685 ± 0.020 | 0.569 ± 0.021 |

| | random | cold-target | cold-drug | cold-pair |
|---|---|---|---|---|
| test pairs | 6,011 | 5,984 | 5,746 | 1,144 |
| test drugs / targets | 68 / 442 | 68 / 88 | 13 / 442 | 13 / 88 |
| positive rate (AUPRC chance) | 0.077 | 0.075 | 0.060 | 0.057 |
| DeepDTA AUPRC | 0.631 ± 0.013 | 0.615 ± 0.004 | 0.199 ± 0.045 | 0.206 ± 0.012 |
| ColdSite-DTI AUPRC | 0.612 ± 0.010 | 0.464 ± 0.047 | 0.201 ± 0.035 | 0.132 ± 0.028 |
| HyperAttentionDTI AUPRC | 0.669 ± 0.021 | 0.644 ± 0.014 | 0.284 ± 0.032 | 0.187 ± 0.028 |
| MolTrans AUPRC | 0.616 ± 0.004 | 0.532 ± 0.010 | 0.133 ± 0.021 | 0.098 ± 0.024 |

**Table R1b.** The cold levels on targets **unseen by sequence** (§1b; option A): the same
checkpoints re-scored by their own trainers' test passes, which reproduce all 24 recorded
AUROCs (`results/clean_accuracy_davis.md`; to four decimals for the three deterministic
models, within the range of five passes for MolTrans, which keeps dropout on at inference
as published). Cold-target keeps 5,168 of 5,984 test rows, cold-pair 1,001 of 1,144.

| model | cold-target, all rows | cold-target, unseen | cold-pair, all rows | cold-pair, unseen |
|---|---|---|---|---|
| DeepDTA | 0.907 ± 0.003 | **0.884 ± 0.003** | 0.728 ± 0.035 | **0.749 ± 0.044** |
| ColdSite-DTI | 0.857 ± 0.011 | **0.835 ± 0.015** | 0.624 ± 0.099 | **0.607 ± 0.128** |
| HyperAttentionDTI | 0.915 ± 0.001 | **0.893 ± 0.001** | 0.694 ± 0.038 | **0.713 ± 0.050** |
| MolTrans | 0.874 ± 0.006 | **0.833 ± 0.008** | 0.566 ± 0.022 | **0.530 ± 0.024** |

Leakage inflated cold-target for **every one of the four models**, by 0.021–0.023 for
DeepDTA, ColdSite-DTI and HyperAttentionDTI and by **0.041 for MolTrans** — 11 of the 12
seeds move down (ColdSite-DTI seed 2, −0.002, is the exception). At cold-pair there is no
consistent effect: the 11 leaked targets were the harder ones for DeepDTA (+0.021) and
HyperAttentionDTI (+0.019) and the easier ones for MolTrans (−0.037) and ColdSite-DTI
(−0.017, its seed 2 falling to 0.510 — chance — on unseen proteins). The
unseen-by-sequence values are the ones the paper reports for the cold levels; the all-rows
values are kept for comparison with work that uses the same DAVIS files.

That the effect is largest for MolTrans, the model with by far the most parameters here,
is what memorising a training sequence would predict, but four models is not a sample:
we report it as an observation, not a trend.

**The levels are not a ladder of increasing difficulty.** On DAVIS an unseen target costs
little (DeepDTA 0.929 → 0.884 on targets unseen by sequence) and an unseen drug costs a
great deal (→ 0.692); cold-pair is no harder than cold-drug for DeepDTA (0.749 vs 0.692,
within the cold-drug spread).
The drug axis dominates: DAVIS has 68 drugs, so cold-drug trains on 49 and tests on 13,
while cold-target still trains on 310 targets and its 88 held-out targets are kinases
like them. Levels are therefore reported as categories, not as a severity scale.

**ColdSite-DTI is not the most accurate model, and does not need to be.** It matches
DeepDTA on random (0.924 vs 0.929) and cold-drug (0.721 ± 0.008 vs 0.692 ± 0.044, inside
DeepDTA's spread), and falls below it on cold-target (0.835 vs 0.884 unseen by sequence)
and cold-pair (0.607 vs 0.749). The audit asks whether each model's
explanation survives, not which model predicts best; accuracy is reported so that every
explanation result can be read against how well the same checkpoint predicts.

**Cold-pair is the least stable cell.** ColdSite-DTI's three cold-pair seeds score 0.738,
0.557 and 0.577, and all three checkpoints come from epoch 11, one epoch after the
selection floor: validation loss on cold-pair's 264 validation pairs was lowest almost
immediately. Random ran long by comparison (best epochs 40, 40, 32). Every cold-pair
result below is therefore quoted with its spread, never from one seed.

**The two published models do not degrade alike.** HyperAttentionDTI is the most accurate
model at every level (random 0.937, cold-target 0.893 unseen — the best cold-target figure
in the table), holds up at cold-pair (0.713 unseen), and loses the least on unseen drugs:
**0.760 ± 0.042 at cold-drug**, against 0.692 ± 0.044 for DeepDTA, 0.721 ± 0.008 for
ColdSite-DTI and 0.685 ± 0.020 for MolTrans. On the axis that costs every other model the
most, it loses the least — and KIBA, whose cold-drug level holds out 422 drugs rather than
13, reproduces that ordering but not the size of the drop (0.086–0.107 there, against
0.177–0.238 here; §8). DAVIS's cold-drug severity is therefore read as a property of its
13-drug split, not of unseen chemistry in general. MolTrans is close to the
others on random (0.923) but loses more at every cold level, and at cold-pair it reaches
**0.530 ± 0.024 on unseen proteins — chance**, with AUPRC 0.098 against a 0.057 positive
rate. Whatever its attention means at cold-pair, it is attached to a model that cannot
predict there; the audit reports that beside its explanation scores, because an
explanation of a prediction no better than chance is not an explanation of anything.

*MolTrans's three seeds are the retrained ones: the first grid's seeds 2 and 3 trained as seed 1 (the vendored `baselines/MolTrans/models.py`
reseeds torch on import; fixed 2026-09-13), and only the corrected cells are used here.*

## 1b. DAVIS's sequences: leakage and pseudo-variants

Found 2026-09-13 building the KLIFS ground truth; `results/sequence_audit_davis.md`
(`src/data/sequence_audit.py`). In DeepDTA's DAVIS `proteins.txt`, which this project
and most DTI papers use: (i) all 54 variant targets with a wild-type entry carry exactly
the wild-type sequence — ABL1(T315I), EGFR(T790M), BRAF(V600E) and the rest — so 442
targets are 379 distinct sequences; (ii) at cold-target 12 of 88 test targets (816 of
5,984 rows, 13.6%) and at cold-pair 11 of 88 (143 of 1,144, 12.5%) are unseen by name
but identical in sequence to a training target, and cold-pair validation is 13.6% such
rows; (iii) ten targets' sequences hold few or none of the 85 KLIFS pocket residues
(RET and its three mutants are RET's extracellular residues 1–430). KIBA has none of
the three. **Decided 2026-09-13: option A** (Methods §2.4) — cold-level accuracy on targets unseen
by sequence (Table R1b), explanation metrics without the seen-by-sequence and pocketless
targets and counting one protein per distinct sequence; nothing retrained. The leak into
cold-pair's validation set, which influenced checkpoint selection, is a limitation.

**What the leak is worth, measured by retraining without it.** Re-scoring changes which
rows are *measured*; it cannot remove what the model *learned*. So the cold splits were
rebuilt with the leak removed — no sequence shared between training, validation and test,
the test file untouched — beside a control that keeps the leak at the same row count and
the same number of positives (`src/data/seqclean_splits.py`). DeepDTA trained on both,
three seeds each; all three arms are scored on the one test set they share
(`results/leakage_retrain_davis.md`, `src/evaluation/leakage_retrain.py`).

**Table R1c.** DeepDTA test AUROC, mean ± sd over seeds 1–3.

| level | arm | training rows | leak | test AUROC |
|---|---|---|---|---|
| cold-target | original | 21,080 | in | 0.907 ± 0.003 |
| | volume-matched control | 17,748 | in | 0.888 ± 0.011 |
| | sequence-clean | 17,748 | out | **0.869 ± 0.013** |
| cold-pair | original | 15,190 | in | 0.728 ± 0.035 |
| | volume-matched control | 12,936 | in | 0.666 ± 0.027 |
| | sequence-clean | 12,936 | out | **0.686 ± 0.018** |

At cold-target the published 0.907 is **0.019 smaller training set + 0.019 leakage**: the
control minus the clean arm is −0.005, −0.015 and −0.037 in seeds 1, 2 and 3 — the same
direction in every seed, mean **0.019**. That is the figure re-scoring estimated
independently (0.023 for DeepDTA, Table R1b) by a method with nothing in common with this
one, which is the reason we report both. At cold-pair removing the leak costs nothing (the
clean arm is 0.021 *higher*, in all three seeds, inside the seed spread): cold-pair's
difficulty is its unseen drugs, as §2 finds, not its leaked proteins.

Retraining every model this way was not affordable; DeepDTA, the anchor, is the one
retrained, and the agreement between the two methods on it is what licenses using the
re-scored values for the other three.

## 2. The cold-pair drop is mostly task, not volume

Cold-pair trains on 15,190 rows against random's 21,039. Retraining ColdSite-DTI on
random with its training set cut to 15,190 rows (three seeds, each its own subsample;
full random validation and test sets) gives AUROC **0.887 ± 0.018**
(`results/volume_control_davis.md`):

| ColdSite-DTI cell | training rows | AUROC |
|---|---|---|
| random, full | 21,039 | 0.924 ± 0.001 |
| random, volume-matched | 15,190 | 0.887 ± 0.018 |
| cold-pair | 15,190 | 0.624 ± 0.099 |

Fewer rows cost 0.037; the remaining 0.263 is cold-pair itself — about 12% and 88% of
the 0.300 drop. The second term includes everything that differs between the two test
sets (cold-pair's own 1,144 pairs of unseen drugs and unseen targets), which is what
cold-pair difficulty means here. The control was run for ColdSite-DTI only.

## 3. The plausibility metric can see a real signal

Before any model is scored, the metric is scored against explanations of known quality
(`results/positive_control_davis.md`, recomputed 2026-09-14 under the sequence policy of
§1b): synthetic attention that places a fraction *d* of the true binding sites first.
Precision@10 rises with *d* at every level, the oracle (*d* = 1) reaches the ceiling
(0.989–0.991), and a dose of **2% of the sites is detected as significant at all four
levels** — on the policy's own protein sets, 349 proteins at random and cold-drug and only
68 and 72 at cold-target and cold-pair. A model whose precision@10 sits at chance is
therefore a real null, not a test too weak to see anything. On a planted model whose prediction depends only on the annotated
sites, the faithfulness measure separates the oracle (comprehensiveness delta ≈ +10.2)
from no signal (≈ 0) at every level.

## 4. ColdSite-DTI's attention is load-bearing, finds the pocket region, misses the annotated residues

Computed 2026-09-13 on the 12 binary checkpoints under the sequence policy of §1b (option A:
one protein per distinct sequence, seen-by-sequence targets dropped at the cold levels,
pocketless targets dropped everywhere). Outputs: `results/analysis_davis_policyA/` (not
committed). Faithfulness uses up to 200 test pairs per level; precision@10 scores one test
pair per protein — 349 proteins at random and cold-drug, 68 at cold-target and 72 at
cold-pair (UniProt; KLIFS: 350 / 350 / 67 / 71) — against a 1,000-trial permutation null.

**Table R2.** ColdSite-DTI, DAVIS binary, seeds 1 / 2 / 3.

| level | faithfulness delta | mean ± sd | precision@10 (UniProt) | mean ± sd | chance | ceiling |
|---|---|---|---|---|---|---|
| random | 0.801 / 1.993 / 0.770 | 1.19 ± 0.70 | 0.023 / 0.009 / 0.013 | 0.015 ± 0.007 | 0.020 | 0.99 |
| cold-drug | 1.801 / 0.966 / 0.771 | 1.18 ± 0.55 | 0.027 / 0.027 / 0.011 | 0.022 ± 0.009 | 0.020 | 0.99 |
| cold-target | 0.163 / 0.891 / 0.496 | 0.52 ± 0.36 | 0.015 / 0.019 / 0.018 | 0.017 ± 0.002 | 0.019 | 0.99 |
| cold-pair | 0.619 / 0.944 / 0.432 | 0.67 ± 0.26 | 0.018 / 0.013 / 0.008 | 0.013 ± 0.005 | 0.019 | 0.99 |

Faithfulness delta = comprehensiveness (absolute change in the predicted logit when the 10
most-attended residues are masked) minus the same for 10 random residues; only the delta
is a result.

**Faithful at every level.** Masking the residues ColdSite-DTI attends to moves its
prediction more than masking random residues in all 12 cells (every delta positive,
flagged load-bearing in each seed's report). The attention is not decoration: the model
uses the residues it points at, at every split level.

**Against UniProt's annotated residues: at chance.** Mean precision@10 is 0.013–0.022
against a chance of 0.019–0.020 and a ceiling of 0.99; no level exceeds chance on average.
The only cells significant before correction are cold-drug seeds 1 and 2 (0.027, p = 0.008
and 0.010), which seed 3 does not reproduce (0.011, p = 1.0).

Read against the dose curve of §3, which was recomputed on these very protein sets, every
one of the 12 cells is worth an **equivalent dose of 0.006 or less** — the fraction of true
sites a synthetic explanation would have to rank first to match it — and 7 of the 12 sit at
or below chance (`results/positive_control_davis.md`, "Audited models, read against the
curve"). The same test detects a dose of 0.02 at every level, so this is a null with
resolution to spare, not an underpowered test: whatever ColdSite-DTI's attention carries,
it is worth under 1% of the annotated residues being ranked first.

**Against the KLIFS ATP pocket: above chance, mostly by finding the domain.** The same
checkpoints scored against KLIFS's 85-residue ATP pocket (Methods §3.6):

**Table R3.** ColdSite-DTI precision@10 against the KLIFS pocket, seeds 1 / 2 / 3
(`results/analysis_davis_policyA/positional_control_coldsite_dti_davis_klifs_policyA.md`).

| level | precision@10 | mean ± sd | chance | beats in-span shuffle (p) | top-10 inside the pocket's span | span / chain |
|---|---|---|---|---|---|---|
| random | 0.191 / 0.226 / 0.239 | 0.22 ± 0.02 | 0.14 | 0.205 / 0.001 / 0.001 | 0.33–0.36 | 0.25 |
| cold-drug | 0.300 / 0.259 / 0.338 | 0.30 ± 0.04 | 0.14 | 0.001 / 0.001 / 0.001 | 0.39–0.46 | 0.25 |
| cold-target | 0.228 / 0.204 / 0.296 | 0.24 ± 0.05 | 0.14 | 0.049 / 0.196 / 0.001 | 0.34–0.43 | 0.25 |
| cold-pair | 0.310 / 0.259 / 0.245 | 0.27 ± 0.03 | 0.14 | 0.005 / 0.009 / 0.211 | 0.40–0.48 | 0.24 |

Every cell beats, at p = 0.001, both a map borrowed from another protein (position alone)
and the protein's own attention shuffled among residues of the same amino acid
(residue-type preference alone). So the attention does find the pocket region of each
kinase. How: 33–48% of its top ten residues fall inside the stretch the pocket spans (the
kinase domain core), which is 24–25% of the chain; shuffled within that stretch it keeps
most of its score, and beats the within-stretch shuffle in 9 of 12 cells (p ≤ 0.049,
before correction) by a modest margin (e.g. cold-drug seed 1: 0.300 vs 0.250). Most of the
pocket signal is knowing the domain; a smaller part is knowing the pocket inside it.

Taken together: the attention is causally used (every level), is **coarsely plausible** —
it concentrates on the kinase domain and its ATP pocket at about twice chance, beyond
position and amino-acid preference — and is **not finely plausible**: it does not land on
the residues UniProt annotates. "Is attention plausible?" has a different answer at each
ground-truth resolution, so both are reported. The pattern is the same with and without
the sequence policy (pre-policy values: `results/positional_control_coldsite_dti_davis_klifs.md`).
*[Formal statement waits for the audit table (§5), Holm over the whole family.]*

## 5. The audit table: the published model's residue-level claim holds only on the random split

*See **Figure 1** (`results/figures/fig1_plausibility.pdf`), top-left panel: every DAVIS
cell against UniProt with its three seeds, and **Figure 2** for the same cells one seed at
a time.*

Computed 2026-09-14 on two T4 GPUs (`notebooks/kaggle_analysis_davis.ipynb`), **one Holm
correction over all sixteen cells** — three audited models and the uniform control, four
levels each. Correcting per model would have inflated every claim in the table.

**Table R4.** precision@10 against UniProt's annotated residues, mean ± sd over seeds
1–3 (`results/analysis_davis_policyA/audit_davis_binary.md`). `uniform_control` is an
attention map of equal weight everywhere — the metric's own floor.

| model | random | cold-drug | cold-target | cold-pair |
|---|---|---|---|---|
| ColdSite-DTI (ours) | 0.015 ± 0.007 | 0.022 ± 0.009 | 0.017 ± 0.002 | 0.013 ± 0.005 |
| HyperAttentionDTI (published) | **0.034 ± 0.006** | 0.040 ± 0.031 | 0.024 ± 0.008 | 0.022 ± 0.010 |
| MolTrans (published) | 0.021 ± 0.003 | 0.027 ± 0.005 | 0.028 ± 0.016 | 0.020 ± 0.014 |
| uniform control | 0.020 ± 0.001 | 0.020 ± 0.001 | 0.018 ± 0.005 | 0.017 ± 0.001 |

**One cell of sixteen survives Holm–Bonferroni: HyperAttentionDTI on the random split**
(p = 0.0020 against a threshold of 0.0031; *it does not replicate on KIBA — §8.1, p = 0.48,
one seed of three above chance*). The next four in the ordering all fail:
ColdSite-DTI cold-drug (p = 0.0060 vs 0.0033), MolTrans cold-target (p = 0.012 vs 0.0036)
and cold-drug (p = 0.020 vs 0.0038), and HyperAttentionDTI cold-drug (p = 0.050 vs
0.0042, on a seed spread of ±0.031). HyperAttentionDTI's cold-target (p = 0.11) and
cold-pair (p = 0.30) are not close, and ColdSite-DTI survives nowhere.

**MolTrans is at the metric's floor everywhere.** Its four cells (0.020–0.028) sit within
one standard deviation of the uniform control's (0.017–0.020) — an attention map of equal
weight everywhere scores the same as its trained attention. Its best cell, cold-target
0.028 ± 0.016, is also where its accuracy is 0.833 unseen (Table R1b); at cold-pair,
where it predicts at chance (0.530), its attention scores 0.020 against a 0.017 floor.
Both of the published models we audit therefore fail the residue-level claim under
shift, and one of them fails it everywhere.

**The warm signal is real, not an artefact.** At the random split all three of
HyperAttentionDTI's seeds beat every null in `positional_control`: a map borrowed from
another protein (0.021, p = 0.001), attention permuted among residues of the same amino
acid (0.021, p = 0.001) and permuted within the site-spanning stretch (0.026, p ≤ 0.003 in
two seeds of three). So on the random split this model's attention carries
protein-specific, residue-level information about where the annotated residues are —
1.67× chance — which is exactly the claim the interpretability literature makes, and it is
supported.

**It does not survive distribution shift.** By cold-target the same model is at 1.26×
chance and no longer beats a borrowed map in two of three seeds; at cold-pair it is at
1.18× and beats nothing. The cold cells are also where the seeds disagree most
(cold-drug 0.019–0.077 across seeds), so single-seed evidence there would be worthless in
either direction.

**Faithfulness tells the same story in a different currency.** HyperAttentionDTI's
attention is load-bearing at every level, but the margin over random masking collapses as
the split hardens: **0.184 ± 0.056** (random), 0.113 ± 0.052 (cold-drug),
0.063 ± 0.046 (cold-target), 0.056 ± 0.008 (cold-pair). The attention is still used
under shift; it is simply used for something that no longer coincides with the annotated
site.

**Against the KLIFS pocket, the coarse signal persists where the fine one does not**
(`results/analysis_davis_policyA_klifs/`): 0.242 ± 0.038 at random (1.70× its 0.143
chance, p = 0.001 in every seed) and 1.33–1.37× at all three cold levels. Read beside
§4, the two audited models differ in kind rather than in degree: ColdSite-DTI is
coarsely plausible everywhere and finely plausible nowhere, while HyperAttentionDTI is
both at random and only coarsely so once the split is cold.

*Device check: every cell of this table was also computed on a CPU during the same night.
The two agree exactly in 11 of 12 cells and by 0.0003 in the twelfth, so the GPU runs
carry no device-specific drift.*

*The kinase-family confound could not be stratified away: fewer than 20 non-kinase targets
are available in every cell (`audit_davis_binary.md`), so the unstratified table must be
read with §6's finding in mind.*

**How large is the surviving cell, in absolute terms?** Read against the dose curve of §3 —
recomputed on these protein sets — HyperAttentionDTI's random cell is worth an **equivalent
dose of 0.007–0.017**: a synthetic explanation would have to rank about 1% of the true
annotated sites first to score what this model scores
(`results/analysis_davis_policyA/positive_control_davis_hat_compare.md`). Its cold cells are
worth 0.014 or less, and five of its nine cold seed-cells are at or below chance. So the one
cell that survives correction is real, reproducible against every null, and **small**: 1.7×
chance is a statistically solid effect that still corresponds to recovering a low single-digit
percentage of the annotated site. The audit reports it as such rather than as a vindication.

*The same reading was not computed for MolTrans, whose cells sit at the uniform control's
floor; the dose that matches a floor score is not a meaningful quantity.*

## 5b. Faithfulness, and an intervention that was not the same size in both arms

*See **Figure 4** (`results/figures/fig4_faithfulness.pdf`): the residue-space panels and
MolTrans's token-space panel, which is why the two are never plotted on one axis.*

A masking test subtracts a random-masking control from the explanation's
comprehensiveness, which is only meaningful if both arms change the input by the same
amount. For a model that reads residues they do: masking k residues changes k input
positions either way. **MolTrans does not read residues.** It reads ESPF sub-word tokens,
so replacing a residue with `X` re-segments the protein, and how much of the token
sequence changes depends on where the masked residues sit. Measured over 20 DAVIS
proteins (`src/evaluation/mask_comparability.py`): masking MolTrans's ten most-attended
residues changes **48%** of its tokens, and masking ten random residues changes **95%** —
for RIPK5, 0.8% against 99%, a factor of 124.

Under that test MolTrans's faithfulness delta was negative in 11 of 12 cells, reproducing
to 0.005 across a laptop and a T4. Read naively it says its attention points at residues
that matter *less* than arbitrary ones. It says no such thing: subtracting a larger
intervention from a smaller one returns a negative number whatever the attention does.

Measured in the space the model reads — the explanation's arm removes the tokens carrying
its top-10 attention, the control removes **the same number** of tokens at random, so both
arms change an identical amount of input by construction
(`src/evaluation/token_faithfulness.py`) — the sign reverses:

**Table R4b.** MolTrans comprehensiveness delta, 200 pairs per level (75 and 76 at the
cold levels under the sequence policy), mean over seeds 1–3.

| level | residue space (arms unequal) | token space (arms matched) |
|---|---|---|
| random | −0.299 | **+0.458** |
| cold-drug | −0.407 | **+0.362** |
| cold-target | −0.176 | **+0.347** |
| cold-pair | −0.087 | **+0.276** |

**MolTrans's attention is load-bearing at every level** — positive in all 12 cells. The
seed spread is wide (0.125 to 0.766 at random), so the sign is the result and the
magnitude is not. Its *sufficiency* deltas are mostly negative: keeping only the attended
tokens preserves the prediction less well than keeping the same number of random ones,
which is what a thinly spread attention looks like — removing its top tokens matters,
but they do not carry the prediction alone.

Two consequences beyond this model. Masking-based faithfulness **does not transfer across
tokenisations**, so any audit of a sub-word protein model that compares k-residue masks is
measuring its own intervention; and because the unit differs (tokens here, residues for
the other two models), these deltas are comparable within a model across levels and seeds
— which is how the audit uses them — and not numerically across models. Methods states
both.

## 6. Kinase-family control, and why the confound cannot be tested inside DAVIS

Every model here trained on a kinase panel, so a plausibility score could reflect
knowledge of one protein family rather than of binding sites. The natural test is to
stratify each cell by family and compare. **That test is impossible on these benchmarks.**
DAVIS's 6,011 test rows contain 3,307 rows on targets our classifier recognises as
kinases and **zero** on a non-kinase (`src/evaluation/target_family.py`, measured
2026-09-14); the remaining 2,704 are kinases its gene-symbol heuristic does not name.
KIBA is the same kind of object — 229 kinases. No panel size fixes this: the gate counts
non-kinase targets *in the cell being scored*, and there are none to count. That the two
standard DTI benchmarks cannot answer the family-confound question is a fact about the
benchmarks, and belongs beside §1b's leakage finding rather than in a limitations list.

What can be done is to score the same trained attention on proteins from **outside** the
training family: 60 BindingDB proteins with UniProt-annotated sites, none of them kinases,
none seen by any model here (`src/data/build_nonkinase_panel.py`; sequences and site
numbering both from UniProt, never BindingDB's construct chains). Chance differs between
the arms because the proteins differ — 0.020 for DAVIS's kinases, 0.012 for the panel —
so each arm is read against its own.

**Table R5.** precision@10, mean ± sd over seeds 1–3, cotransport ions excluded (the
primary setting; all ligands as sensitivity). `chance` in brackets.

| model | level | kinase arm (n = 349) | non-kinase panel (n = 60) |
|---|---|---|---|
| ColdSite-DTI | random | 0.015 ± 0.007 (0.020) | 0.027 ± 0.021 (0.012) |
| | cold-drug | 0.022 ± 0.009 (0.020) | 0.029 ± 0.014 (0.012) |
| | cold-target | 0.017 ± 0.002 (0.019) | 0.014 ± 0.008 (0.012) |
| | cold-pair | 0.013 ± 0.005 (0.019) | **0.044 ± 0.019** (0.012) |
| HyperAttentionDTI | random | **0.034 ± 0.006** (0.020) | 0.015 ± 0.002 (0.012) |
| | cold-drug | 0.040 ± 0.031 (0.020) | 0.013 ± 0.004 (0.012) |
| | cold-target | 0.024 ± 0.008 (0.019) | 0.017 ± 0.004 (0.012) |
| | cold-pair | 0.022 ± 0.010 (0.019) | 0.007 ± 0.003 (0.012) |
| MolTrans | random | 0.021 ± 0.003 (0.020) | 0.012 ± 0.007 (0.012) |
| | cold-drug | 0.027 ± 0.005 (0.020) | 0.012 ± 0.002 (0.012) |
| | cold-target | 0.028 ± 0.016 (0.019) | 0.014 ± 0.005 (0.012) |
| | cold-pair | 0.020 ± 0.014 (0.019) | 0.008 ± 0.006 (0.012) |

**The two published models score no better than chance off the training family.**
HyperAttentionDTI and MolTrans sit within one standard deviation of the panel's 0.012 in
all eight cells, so whatever HyperAttentionDTI's warm-split signal is (§5), it does not
travel to proteins outside the family it trained on. That is the answer the confound
question wanted, obtained without a stratification the data cannot support.

**ColdSite-DTI is the exception, and the exception is an artefact.** It is above the
panel's chance in three cells, most clearly at cold-pair (0.044 against 0.012) — where
its kinase arm is at 0.013. A higher score on unseen proteins from another family than on
the family it trained on is not knowledge, and the nulls of §4 identify what it is.

It is **not positional**: maps borrowed from another protein score 0.012–0.014 on the
panel, no better than chance. It is **amino-acid preference**. ColdSite-DTI's top-ten
attention is enriched in histidine (3× in seed 1, 11–15× in seeds 2 and 3); the panel's
annotated sites are histidine-rich (8.8×, many of them metal sites); kinase ATP sites are
not (they are enriched in glycine, aspartate and lysine). Permuting the attention among
residues of the *same amino acid* — which keeps the preference and destroys any knowledge
of position — recovers most of the panel score (seed 2 random: 0.045 of 0.050, p = 0.28),
and three of twelve cells keep a remainder significant before correction
(p = 0.004–0.048). The ordering across seeds follows the preference rather than the
accuracy: seed 2, with the strongest histidine enrichment, has the highest panel scores.

So the panel's apparent signal is a liking for one amino acid meeting sites that happen to
be rich in it — the same preference that makes the model *miss* the glycine-rich kinase
ATP site it was trained on. Read together with §5, the family confound does not rescue any
model's plausibility: the two published models are at chance off their training family,
and ours is above chance there for a reason that has nothing to do with binding.

*Sensitivity: with cotransport ions included (`_noions` dropped) the panel's chance level
rises and the same pattern holds; both settings are in
`results/analysis_davis_policyA/control_*.json`. The panel's 60 proteins are what limit
this comparison, and no protein-level interval was computed for it (§7d): the ± given here
are seed spreads, and the closest measured analogue — a 68-protein UniProt cell — is
±0.005–0.009 wide. Only the ColdSite-DTI cold-pair cell stands clear of the panel's chance
level by more than its own spread; the claim that the two published models are at chance
rests on eight cells agreeing rather than on any one of them.*

## 7. Does attention know *which* drug binds? A per-pair ground truth

Both ground truths so far are per protein, and the model is asked about a **pair**. A
reviewer is entitled to ask whether attention marks the site of *this* binding event, and
neither UniProt's annotations nor the KLIFS pocket can answer that.

KLIFS publishes, for every co-crystal structure, an **interaction fingerprint**: 85 pocket
positions × 7 interaction types, recording which pocket residues touch the bound ligand.
Where a DAVIS drug is that ligand, the residues it actually touches are *measured* rather
than annotated. `src/data/klifs_ligand_contacts.py` matches DAVIS drugs to KLIFS ligands
by InChIKey (RDKit, from the SMILES the models trained on) and places the contacts through
the same steps as the pocket ground truth — same KLIFS entry by UniProt accession, same
placement, same remapping — so both ground truths share one coordinate frame. A position
counts when it is contacted in at least half of the pair's structures; imatinib has 18
ABL1 structures and they do not agree residue for residue (the agreed set is 93% of the
union). **217 pairs, 101 proteins, 28 drugs**, median 19 contacted residues per pair. Two
checks: every contact lies inside its protein, and every contact lies inside the KLIFS
pocket of the same protein. ABL1–imatinib (2hyy) contacts 24 of 85 positions, and the
contacted residues include the VAIK lysine and the DFG motif.

**What DAVIS can support.** A pair is scorable only where that exact drug has been
crystallised with that exact kinase:

| level | test rows | scorable pairs | proteins | drugs |
|---|---|---|---|---|
| random | 6,011 | 38 | 29 | 17 |
| cold-drug | 5,746 | 60 | 57 | 6 |
| cold-target | 5,984 | 39 | 20 | 19 |
| cold-pair | 1,144 | **12** | 12 | 5 |

**The control that makes it interpretable.** A drug's contacts sit inside one pocket, so a
model that merely finds the pocket scores well against *any* drug's contacts there. The
comparison therefore holds the protein and the number of sites fixed and changes only
*which drug the contacts belong to*: each pair is scored against another drug's contacts on
the same protein (`swap_drugs`, a rotation by sorted drug id). Only proteins with at least
two crystallised drugs can be swapped, so both arms are restricted to exactly those pairs
— identical keys, identical n — and the difference between the columns is the drug and
nothing else.

**Table R6.** precision@10 against crystallographic contacts, mean over seeds 1–3 with
95% intervals from resampling proteins (§7d). Chance is higher than against UniProt's
annotations because a drug touches ~19 residues rather than ~12.

| model | level | n | chance | the pair's own drug | another drug, same pocket |
|---|---|---|---|---|---|
| ColdSite-DTI | random | 20 | 0.025 | 0.032 [0.010–0.062] | 0.025 [0.005–0.053] |
| | cold-drug | 36 | 0.023 | 0.030 [0.019–0.043] | 0.032 [0.020–0.046] |
| | cold-target | 4 | 0.025 | 0.046 [0.008–0.096] | 0.038 [0.007–0.082] |
| HyperAttentionDTI | random | 29 | 0.025 | 0.036 [0.028–0.045] | 0.033 [0.025–0.041] |
| | cold-drug | 39 | 0.023 | 0.025 [0.015–0.036] | 0.015 [0.005–0.027] |
| | cold-target | 14 | 0.025 | 0.060 [0.040–0.079] | 0.060 [0.043–0.076] |
| MolTrans | random | 29 | 0.025 | 0.011 [0.005–0.021] | 0.009 [0.002–0.021] |
| | cold-drug | 39 | 0.023 | 0.036 [0.025–0.048] | **0.056** [0.042–0.072] |
| | cold-target | 14 | 0.025 | 0.048 [0.036–0.060] | 0.045 [0.033–0.057] |

**Attention does not know which drug binds.** In every row the two intervals overlap
almost entirely; the largest gain from using the correct drug is +0.011, and in two rows
the *wrong* drug scores higher — MolTrans's cold-drug cell by 0.021, outside the paired
arm's interval. Whatever agreement exists with crystallographic contacts is agreement with
the pocket those contacts lie in, not with the binding event the model was asked about.

Cold-pair is omitted from the table: three scorable pairs after the sequence policy, where
MolTrans scores 0.000 in both arms, ColdSite-DTI 0.011, and HyperAttentionDTI's interval is
0.067 wide on those three proteins (§7d) — wider than any difference the table above
reports. Twelve pairs before the policy is the honest ceiling DAVIS offers at that level,
and it is not enough to say anything.

## 7b. Is the verdict the model's, or the readout's?

Between the tensor inside a network and the one weight per residue that precision@k scores,
somebody chooses: which axis to reduce, which layer to read, how to spread a convolution
position or a sub-word token over the residues it covers. Every number above rests on
choices made once. `src/evaluation/readout_variants.py` scores the same checkpoints through
readouts another author could reasonably have picked, each changing exactly one documented
choice, with the published readouts re-scored in the same run so nothing is compared across
devices or code states.

**Table R7.** precision@10, mean ± sd over seeds 1–3. Chance is 0.020 against UniProt's
annotated residues and 0.143 against the KLIFS pocket.

| model | readout | UniProt: random / cold-target | KLIFS: random / cold-target |
|---|---|---|---|
| ColdSite-DTI | cross-attention *(published)* | 0.015 / 0.017 | 0.219 / 0.243 |
| | protein self-attention | 0.010 / 0.022 | 0.238 / 0.268 |
| HyperAttentionDTI | channel mean, centre *(published)* | 0.034 / 0.025 | 0.242 / 0.186 |
| | **channel max** | 0.038 / 0.034 | **0.335 / 0.367** |
| | **receptive-field spread** | 0.027 / 0.012 | **0.157 / 0.081** |
| MolTrans | head mean, last layer *(published)* | 0.021 / 0.026 | 0.157 / 0.176 |
| | head max | 0.021 / 0.029 | 0.155 / 0.177 |
| | first layer | 0.023 / 0.023 | 0.173 / 0.189 |

Three findings, in order of how much they should worry a reader of the literature.

**The residues a readout points at are largely a property of the readout.** Across 25
proteins, the top-ten residues of an alternative readout overlap the published readout's
by **2–12%** — HyperAttentionDTI's channel-max and receptive-field readouts share 2% and
4% of their top ten with the published one — with MolTrans's head-max the single exception
at 90%. Two defensible readouts of one checkpoint therefore highlight almost disjoint sets
of residues. A published attention figure is, to that extent, a picture of a reduction
choice.

**A readout choice can move a verdict across chance.** HyperAttentionDTI against the KLIFS
pocket reads 0.367 at cold-target under channel-max (2.6× chance) and **0.081** under the
receptive-field projection (*below* the 0.143 chance level), against 0.186 as published.
The claim "this model's attention finds the ATP pocket under distribution shift" is true,
false, or unsupported depending on a choice no paper reports.

**But the audit's own verdicts survive.** Every readout of every model stays at chance
against UniProt's annotated residues (0.010–0.057 against 0.020, all within the seed
spread bar HyperAttentionDTI's channel-max cold-drug cell at 0.057 ± 0.018), and no
readout lifts MolTrans above the pocket's chance level (0.120–0.189 against 0.143). The
residue-level null of §5 and the floor of §6 are therefore not artefacts of how we read
attention; what the readout choice changes is the *size* of the coarse, pocket-level
signal, not the existence of the fine-grained one.

**A readout that cannot see the drug does as well as one that can.** ColdSite-DTI's
protein-tower self-attention — computed by its forward pass and discarded, and
independent of the drug by construction — scores **0.238** against the KLIFS pocket at
random where its drug-conditioned cross-attention scores 0.219, and 0.268 against 0.243 at
cold-target. Its reported explanation owes nothing to the pair. Read with §7, where using
the correct drug's contacts buys at most +0.011 over another drug's, two independent
measurements say the same thing: the drug is not doing work in these explanations.

## 7c. Attention versus the gradient: is it the explanation or the model?

*See **Figure 3** (`results/figures/fig3_attention_vs_ig.pdf`): all twelve cells, both
ground truths, attention beside the gradient on identical weights.*

Every measurement so far scores **attention**. When attention misses the site, two
opposite things could be true — the attention is a poor report of a model that does
represent the site, or the model never learned it — and no attention measurement
separates them. Integrated gradients do: the attribution comes from the trained weights
and the gradient of the model's own prediction, with no interpretability head
(`src/evaluation/integrated_gradients.py`; path from the padding embedding, the same
"no residue here" the masking uses, 32 steps, explaining each model's own `predict`). The
variants read the *same checkpoints*, so this is two explanations of one model.

**Table R8.** precision@10 against UniProt's annotated residues, all three audited models,
mean ± sd over seeds 1–3. `attention` is the audit table of §5. Holm is applied over the
twelve cells of this family, with each cell's p taken as the median of its three seeds —
the same rule `run_audit` uses for the attention family.

| model | level | attention | integrated gradients | IG / attn | × chance | survives Holm |
|---|---|---|---|---|---|---|
| ColdSite-DTI | random | 0.015 | 0.021 ± 0.009 | 1.4× | 1.03× | no (p = 0.73) |
| | cold-drug | 0.022 | **0.055 ± 0.022** | 2.5× | **2.69×** | **yes** (p = 0.0010) |
| | cold-target | 0.017 | **0.044 ± 0.027** | 2.6× | **2.25×** | **yes** (p = 0.0030) |
| | cold-pair | 0.013 | 0.014 ± 0.001 | 1.1× | 0.76× | no (p = 0.87) |
| HyperAttentionDTI | random | 0.034 | **0.055 ± 0.034** | 1.6× | **2.70×** | **yes** (p = 0.0010) |
| | cold-drug | 0.040 | **0.079 ± 0.005** | 2.0× | **3.88×** | **yes** (p = 0.0010) |
| | cold-target | 0.025 | **0.079 ± 0.009** | 3.2× | **4.07×** | **yes** (p = 0.0010) |
| | cold-pair | 0.022 | **0.060 ± 0.028** | 2.7× | **3.20×** | **yes** (p = 0.0010) |
| MolTrans | random | 0.021 | 0.018 ± 0.001 | 0.9× | 0.90× | no (p = 0.84) |
| | cold-drug | 0.027 | 0.027 ± 0.003 | 1.0× | 1.32× | yes (p = 0.0050) |
| | cold-target | 0.028 | 0.029 ± 0.001 | 1.1× | 1.52× | no (p = 0.050) |
| | cold-pair | 0.020 | 0.021 ± 0.003 | 1.0× | 1.11× | no (p = 0.38) |

**Table R8b.** The same against the 85-residue KLIFS ATP pocket (chance 0.136–0.143).
Eleven of these twelve cells survive Holm; only MolTrans's cold-pair does not.

| model | random | cold-drug | cold-target | cold-pair |
|---|---|---|---|---|
| ColdSite-DTI attention | 0.219 | 0.299 | 0.243 | 0.271 |
| ColdSite-DTI **IG** | **0.280 ± 0.014** | **0.451 ± 0.159** | **0.325 ± 0.040** | **0.314 ± 0.044** |
| HyperAttentionDTI attention | 0.242 | 0.193 | 0.186 | 0.185 |
| HyperAttentionDTI **IG** | **0.427 ± 0.070** | **0.384 ± 0.078** | **0.538 ± 0.068** | **0.326 ± 0.055** |
| MolTrans attention | 0.157 | 0.154 | 0.176 | 0.136 |
| MolTrans **IG** | 0.168 ± 0.045 | 0.165 ± 0.025 | 0.162 ± 0.016 | 0.125 ± 0.024 |

**Seven of twelve cells survive Holm for the gradient, against one of sixteen for the
attention.** The comparison is as controlled as it can be made: the same checkpoints, the
same ground truth, the same protein sets, the same permutation test, the same *k* — only
the explanation differs. Where the attention of the best-generalising model is at chance
under shift, its gradient is at 3.2–4.1× chance and survives correction at every level.

**And the gap appears exactly where the attention carries something.** For the two models
whose attention is at least coarsely plausible, the gradient recovers far more: ColdSite-DTI
2.5–2.6× its attention at the two cold levels where it clears correction, HyperAttentionDTI
1.6–3.2× at all four. For **MolTrans the gradient matches its attention to within noise**
(0.9–1.1× on annotated residues, 0.9–1.1× on the pocket) and both sit at the floor. That is
the control this section needed. It says the two failures are different in kind:

* HyperAttentionDTI and ColdSite-DTI **do** represent the binding site, and their attention
  under-reports it — a reporting failure.
* MolTrans's attention is not under-reporting anything. Its gradient, which has no
  interpretability head to blame, is at the floor too. Its failure is the **model**.

No attention measurement could have drawn that distinction, which is the argument for
including a second explanation method in an audit of this kind at all.

Two honest qualifications. **ColdSite-DTI's gradient is noisy where it matters**: its
cold-target cell is 0.074 / 0.021 / 0.037 across seeds (± 0.027) and its cold-drug pocket
cell is 0.352 / 0.634 / 0.366 (± 0.159), so the effect is established by the permutation
test rather than by a precise estimate, and the direction is what we report. **MolTrans's
three surviving KLIFS cells are significant but tiny** — 1.16–1.18× chance on 350 proteins
— and are read as the floor, not as a signal; §5's rule of reporting effect size beside p
is why.

*Reproducibility: this table was computed twice, on two Kaggle accounts with independent T4
allocations, one on commit `139b103` and one on `3ca50aa`. All 24 cells (2 models × 4 levels
× 3 seeds × 2 ground truths) agree to **0.0e+00** — bit-identical — which is what the
attribution's determinism check predicted and is worth recording because the first attempt
at this analysis crashed on a non-deterministic cuDNN path.*

*The correction family: these twelve cells are Holm-corrected among themselves, not pooled
with the sixteen attention cells of §5. Integrated gradients were added **after** the
attention results were seen, so this is a secondary analysis and is labelled one; the
attention audit remains the pre-specified primary family. Methods states both, and no claim
in this section rests on comparing a corrected p from one family with a corrected p from the
other.*

## 7d. How precisely does a cell of this size measure anything?

Every mean above carries a seed spread, which says how much the *training* varied. It does
not say how precisely a cell measures its protein population — a cell of 68 proteins and a
cell of 349 can report the same ± and mean very different things. `src/evaluation/bootstrap_ci.py`
resamples the **proteins** a cell scored, 10,000 times, each protein entering with all of
its seeds (so seed variation stays inside the interval rather than being averaged away
first). The mean is the number the ladder already reports.

**Table R9.** 95% percentile intervals, precision@10 (`results/ci_davis.md`). Chance is
0.019–0.020 against UniProt's annotated residues and 0.136–0.143 against the KLIFS pocket.

| ground truth | model | random (n = 349/350) | cold-drug | cold-target (n = 68/67) | cold-pair (n = 72/71) |
|---|---|---|---|---|---|
| UniProt | ColdSite-DTI | 0.015 [0.013–0.018] | 0.022 [0.019–0.024] | 0.017 [0.012–0.023] | 0.013 [0.007–0.020] |
| | HyperAttentionDTI | **0.034 [0.030–0.038]** | 0.040 [0.037–0.044] | 0.025 [0.018–0.031] | 0.022 [0.016–0.029] |
| | MolTrans | 0.021 [0.017–0.026] | 0.026 [0.021–0.031] | 0.026 [0.018–0.035] | 0.020 [0.013–0.027] |
| KLIFS | ColdSite-DTI | 0.219 [0.208–0.229] | 0.299 [0.286–0.312] | 0.243 [0.218–0.268] | 0.271 [0.241–0.303] |
| | HyperAttentionDTI | 0.242 [0.232–0.253] | 0.193 [0.183–0.203] | 0.186 [0.163–0.209] | 0.185 [0.165–0.206] |
| | MolTrans | 0.157 [0.143–0.170] | 0.154 [0.142–0.166] | 0.176 [0.150–0.201] | 0.136 [0.114–0.158] |

Three things this settles that the seed spreads could not.

**HyperAttentionDTI's random cell excludes chance; ColdSite-DTI's excludes it downwards.**
The surviving cell of §5 reads 0.034 [0.030–0.038] against a chance of 0.020 — the whole
interval above it. ColdSite-DTI at random reads 0.015 [0.013–0.018]: the entire interval
lies *below* chance, which is a stronger statement than "at chance" and is consistent with
§6's finding that its attention prefers an amino acid the kinase ATP site is poor in.

**The cold cells are imprecise, but not as imprecise as their seed spreads suggest.**
Against UniProt, a 68-protein cell measures precision@10 to ±0.005–0.009 — narrower than
the ±0.008–0.016 seed spreads of Table R4, because averaging three seeds per protein
removes noise the spread reports. Against KLIFS the same cells are ±0.020–0.031, five
times the random level's, so the cold-level pocket numbers are the loosest in the paper.

**The per-pair drug arms are too small to carry their point estimates.** The intervals for
§7's three arms (`results/ci_drug_arms_davis.md`) run to 0.087 wide at ColdSite-DTI's
cold-target paired cell (4 proteins) and 0.075 at its swapped cell, and every cold-pair
arm rests on 3 proteins. This is why §7 omits cold-pair and reads the paired-versus-swapped
comparison off overlapping intervals rather than off the difference of two means.

No interval is computed for the 60-protein non-kinase panel of §6: its proteins are drawn
from a different population, so the bootstrap would need its own run. The closest measured
analogue is a 68-protein UniProt cell at ±0.005–0.009, which is why §6's claim rests on
eight cells agreeing rather than on any one of them.

## 7e. An explanation that cannot depend on the drug

§7 asked whether attention knows *which* drug binds, and answered it by measurement: the
correct drug's own contacts buy at most +0.011 precision@10 over another drug's contacts
in the same pocket, and in two of nine cells the wrong drug scores higher. §7b found the
same thing from the other side: ColdSite-DTI's protein-tower self-attention — computed by
its forward pass, discarded, and **independent of the drug by construction** — agrees with
the KLIFS pocket *better* than its drug-conditioned cross-attention (0.238 against 0.219
at random, 0.268 against 0.243 at cold-target).

Both are our own models measured. This section is about a published one, and the finding
is not statistical but structural.

**EviDTI** (Zhao et al., *Nature Communications* 16:6915, 2025) predicts drug–target
interaction with evidential uncertainty and reports an interpretability analysis: its
Figure 6 shows "attention scores of all the residues in the four randomly selected
drug–target complexes", and the text concludes that "residues with high attention values
coincide with the binding site, underscoring ... the attention mechanism's efficacy". It
is a current model — its protein tower is a protein language model (ProtTrans) — and the
claim is the one this audit exists to test.

Its residue attention is computed, in each of its three model files, as

```python
attention = self.attention_convolution(t_1D)    # t_1D: the protein's ProtTrans embedding
att_AA    = torch.mean(attention, dim=1)        # the per-residue map the figure plots
```

and the drug branches are first used afterwards, at `cat_v = torch.cat((t_o, d_o,
atom_h), 1)`. There is no cross-attention between drug and protein anywhere in the model.
**No drug tensor reaches `att_AA`.** The consequence needs no experiment and holds for any
weights: for a fixed protein, every drug produces the identical residue map. Two of the
four complexes in that figure would carry the same highlighted residues if they shared a
target. The full record, with line references and a one-minute recipe for re-checking it,
is `results/evidti_code_audit.md`.

This does not say the model is wrong, and it says nothing about its uncertainty
quantification, which is its contribution. It says the evidence offered for the
interpretability claim cannot support it: a map that cannot vary with the ligand can
agree with a binding site — an ATP pocket is a property of the kinase, not of the drug —
while carrying no information about *this* pair. We did not retrain EviDTI (its two drug
encoders need TensorFlow and PaddlePaddle, and the 3D encoder's pretrained weights are
not in its repository), so we report no precision@k for it and make no claim about how
well its map agrees with annotated residues. The claim here is about what the explanation
is a function of, which is visible in the source and independent of training.

**Three observations follow, and the third is the one for the field.**

*The measured and the structural cases agree.* Our subjects' explanations are weakly
drug-dependent where they are drug-dependent at all (§7); EviDTI's cannot be. The same
pattern appears whether one measures it or reads it off the architecture.

*A figure of drug–target complexes is the wrong evidence for a protein-only map.* Nothing
in Figure 6 is incorrect. What makes it misleading is the pairing: showing per-complex
pictures implies the map is per-complex. The honest version of that figure is one map per
protein, captioned as protein saliency, and it would support a much weaker claim.

*This is checkable before it is published, by anyone, in a minute.* Whether an
explanation can depend on the drug is a property of the computation graph, not an
empirical question — and a referee, an author or a reader can settle it with `grep`. We
propose it as a routine check for interpretability claims in this field: **state which
inputs the explanation is a function of.** DrugBAN passes it: its bilinear map is indexed by
drug atom and protein position, so the drug is in the explanation by construction, and
the adapter's tests assert that its map moves when the drug changes
(`tests/test_drugban_adapter.py`; its trained cells are not in this draft yet).
ColdSite-DTI passes it formally and fails it in practice, which is why §7b's comparison
is in the paper. EviDTI does not pass it.

## 7f. What would a single-seed paper have concluded?

Every cell here is trained three times and the audit takes the median p over the seeds
(§7). That is a choice, and it is worth showing what it costs — or rather, what reporting
one seed would have bought. The table is generated by
`src/evaluation/seed_agreement.py` from the same ladder files as §5 and §8, with each
seed's own permutation p, **uncorrected**, because an uncorrected per-seed p is exactly
what a single-seed report quotes.

**Table R12.** Sixteen attention cells with three seeds each — three models at four DAVIS
levels, two models at KIBA's two. `*` marks a seed that on its own clears α = 0.05;
`spread` is the largest minus the smallest precision@10 in the cell
(`results/seed_agreement.md`).

| model | dataset | level | seed 1 | seed 2 | seed 3 | chance | spread | seeds above α |
|---|---|---|---|---|---|---|---|---|
| ColdSite-DTI | DAVIS | cold-drug | 0.027 | 0.027 | 0.011 | 0.020 | 0.015 | `**.` |
| ColdSite-DTI | DAVIS | cold-pair | 0.018 | 0.013 | 0.008 | 0.019 | 0.010 | `...` |
| ColdSite-DTI | DAVIS | cold-target | 0.015 | 0.019 | 0.018 | 0.019 | 0.004 | `...` |
| ColdSite-DTI | DAVIS | warm | 0.023 | 0.009 | 0.013 | 0.020 | 0.013 | `...` |
| HyperAttentionDTI | DAVIS | cold-drug | 0.025 | 0.077 | 0.019 | 0.020 | 0.057 | `**.` |
| HyperAttentionDTI | DAVIS | cold-pair | 0.013 | 0.022 | 0.032 | 0.019 | 0.019 | `..*` |
| HyperAttentionDTI | DAVIS | cold-target | 0.018 | 0.031 | 0.025 | 0.019 | 0.013 | `.*.` |
| HyperAttentionDTI | DAVIS | warm | 0.035 | 0.039 | 0.028 | 0.020 | 0.011 | `***` |
| HyperAttentionDTI | KIBA | cold-drug | 0.019 | 0.020 | 0.025 | 0.023 | 0.006 | `...` |
| HyperAttentionDTI | KIBA | warm | 0.017 | 0.022 | 0.053 | 0.023 | 0.036 | `..*` |
| MolTrans | DAVIS | cold-drug | 0.022 | 0.032 | 0.024 | 0.020 | 0.010 | `.*.` |
| MolTrans | DAVIS | cold-pair | 0.004 | 0.032 | 0.024 | 0.019 | 0.028 | `.*.` |
| MolTrans | DAVIS | cold-target | 0.012 | 0.038 | 0.029 | 0.019 | 0.026 | `.**` |
| MolTrans | DAVIS | warm | 0.024 | 0.021 | 0.019 | 0.020 | 0.005 | `*..` |
| MolTrans | KIBA | cold-drug | 0.028 | 0.063 | 0.017 | 0.023 | 0.046 | `.*.` |
| MolTrans | KIBA | warm | 0.020 | 0.053 | 0.022 | 0.023 | 0.032 | `.*.` |

**In 11 of the 16 cells the three seeds disagree about their own verdict**, and in **15 of
16 the spread across seeds is larger than the cell's distance from chance**. A paper
reporting one training run would therefore have had an above-chance, uncorrected result
available in eleven of these sixteen cells — including cells this audit reports as null,
and including MolTrans, whose attention is otherwise indistinguishable from a uniform map.
Which seed was drawn decides the claim.

**One cell behaves differently, and it is the one that survives.** HyperAttentionDTI at
DAVIS random is the only cell in the table where all three seeds clear α individually
(`***`), and it is also the only cell of sixteen that survives Holm correction over the
whole family (§5). The two criteria were computed independently — one is agreement among
replicate runs, the other family-wise error control over a grid — and they select the same
cell. That is the reassurance the audit's method needs: the correction is not discarding
real effects, because the effects it discards are the ones that do not reproduce across
retraining, and the effect it keeps is the one that does.

Four cells have no seed above α at all: three of ColdSite-DTI's four DAVIS cells (warm,
cold-target, cold-pair) and HyperAttentionDTI's cold-drug on KIBA. Those are the audit's
unambiguous nulls — the cells where no draw of the dice would have produced a claim.

**What this does not say.** It is not a claim that these models are unusually unstable —
we have no comparison against a field norm, because single-seed reporting is the field
norm and there is nothing published to compare against. Nor does it apply to the
accuracy axis: on KIBA the same three seeds give test AUROC standard deviations of
0.001–0.009 (§8), against explanation spreads of 0.006–0.046 on precision@10 whose own
chance level is 0.023 — the accuracy is reproducible at a scale the explanation is not. The
instability is specific to the explanation, which is the quantity the literature reports
from one run and a figure.

## 8. KIBA: the replication

*See **Figure 1** (right-hand panels) for KIBA beside DAVIS, and **Figure 2** for the
per-seed strip that this section's central claim rests on.*

KIBA was trained and analysed after every DAVIS result above was fixed, as a replication
rather than a second exploration: the same trainers, the same analysis code, the same
decisions, and no parameter chosen by looking at KIBA. Its scope was set in advance
(Methods §11): **random** and **cold-drug** only, for the two published models and the
DeepDTA anchor, three seeds each — 18 cells. Cold-drug is the level KIBA can test and
DAVIS cannot (422 held-out drugs against DAVIS's 13); cold-target and cold-pair would have
added a weaker copy of levels DAVIS covers with twice KIBA's held-out targets. ColdSite-DTI
was not retrained, so our own model is audited on one dataset and the published ones on two.
HyperAttentionDTI and DeepDTA trained under mixed precision (validated on DAVIS,
`results/amp_validation_davis.md`); MolTrans in full precision, because its hand-written
LayerNorm produces NaN in float16 on KIBA. Every number below is read from
`results/analysis_kiba_policyA/` (UniProt) or `results/analysis_kiba_policyA_klifs/` (KLIFS);
`RUN_NOTES.txt` there records how the run was executed. KIBA needs no sequence policy: none
of DAVIS's three sequence problems occurs in it (§1b).

**Table R10.** KIBA test AUROC and AUPRC, mean ± sd over seeds 1–3, beside DAVIS (Table R1).

| model | random | cold-drug | AUPRC random | AUPRC cold-drug | DAVIS random | DAVIS cold-drug |
|---|---|---|---|---|---|---|
| DeepDTA (anchor) | 0.918 ± 0.001 | 0.832 ± 0.001 | 0.793 ± 0.003 | 0.619 ± 0.008 | 0.929 | 0.692 |
| HyperAttentionDTI | **0.933 ± 0.001** | **0.844 ± 0.004** | 0.829 ± 0.001 | 0.648 ± 0.006 | 0.937 | 0.760 |
| MolTrans | 0.919 ± 0.004 | 0.812 ± 0.009 | 0.795 ± 0.011 | 0.594 ± 0.011 | 0.923 | 0.685 |

| | random | cold-drug |
|---|---|---|
| test pairs | 23,651 | 22,374 |
| test drugs / targets | 2,057 / 228 | 422 / 229 |
| positive rate (AUPRC chance) | 0.209 | 0.219 |

**Accuracy replicates at random and corrects DAVIS at cold-drug.** Every model's random AUROC
is within 0.011 of its DAVIS value, and the ordering holds (HyperAttentionDTI first). At
cold-drug the ordering holds too — HyperAttentionDTI is again the model that loses least on
unseen drugs — but nothing collapses: the drop from random is 0.086–0.107 on KIBA against
0.177–0.238 on DAVIS. DAVIS's cold-drug level tests 13 drugs, so its severity says as much
about which 13 were drawn as about unseen chemistry; §1's "collapse" is a property of
DAVIS's cold-drug split, not of the task, and is stated as such. The seed spreads are also
an order of magnitude tighter (±0.001–0.009 against ±0.020–0.044), which is what a
4× larger test set with 32× as many held-out drugs should give.

### 8.1 Residue-level plausibility: the one DAVIS survivor does not replicate

**Table R11.** Precision@10 against UniProt's annotated binding residues, one test pair per
protein, mean ± sd over seeds (audit grid, `audit_kiba_binary.md`); chance 0.023, ceiling
0.995, n = 211 (random) and 212 (cold-drug) proteins. Holm over the pre-specified KIBA
family of 6 cells (two audited models and the uniform control × two levels), on the median
p over seeds.

| model | random | p (median) | cold-drug | p (median) |
|---|---|---|---|---|
| HyperAttentionDTI | 0.032 ± 0.020 | 0.477 | 0.025 ± 0.002 | 0.275 |
| MolTrans | 0.032 ± 0.018 | 0.537 | 0.036 ± 0.024 | 0.068 |
| uniform control | 0.023 ± 0.004 | 0.301 | 0.025 ± 0.001 | 0.232 |

**None of the six cells survives correction**; the smallest p, MolTrans cold-drug's 0.068,
is eight times its threshold of 0.0083. In particular **HyperAttentionDTI at random — the
only one of DAVIS's sixteen cells to survive Holm (§5) — does not replicate** (p = 0.48).
Its three seeds read 0.017, 0.022 and 0.053 (ladder, p = 0.98, 0.65 and 0.001): one seed
carries an effect larger than DAVIS's (2.3× chance against DAVIS's 1.67×, beating every
positional and residue-identity null in
`positional_control_hyperattentiondti_kiba_policyA.md`), and two sit at chance. On DAVIS all three of the same cell's seeds were above chance and beat the borrowed-map
and same-residue nulls (§5). Across the two
datasets, therefore, no residue-level attention claim of either published model survives
correction, and the one that survived on one dataset is seed-dependent on the other.

*The audit and the ladder differ by 0.001–0.002 per seed for HyperAttentionDTI (0.018 /
0.023 / 0.055 against 0.017 / 0.023 / 0.053): the audit averages 500 random tie-breaks
among equally weighted residues and the ladder breaks ties once. DAVIS shows the same;
no verdict depends on it.*

**MolTrans leaves the floor in one seed of three.** On DAVIS its attention scored the same
as the uniform control at every level (§5). On KIBA seed 2 reaches 0.053 (random) and 0.063
(cold-drug), beats every null at both levels, and does so on kinases only (non-kinase
panel 0.013 and 0.015, `control_moltrans_kiba_seed2_noions.md`); seeds 1 and 3 are at
chance (0.017–0.028). The mean of 0.032–0.036 is therefore not a small effect present in
every model but a large one present in one — which a single-seed audit would have
reported as MolTrans's attention either working or not, depending on the seed.

Read against the metric's own dose curve on these protein sets
(`positive_control_kiba.md`: a 2% dose is detected at both levels), HyperAttentionDTI's
cells are worth an equivalent dose of 0.024 (random, one seed) and 0.004 (cold-drug, one
seed), with the other two seeds at or below chance in each; MolTrans's are 0.024 (random,
one seed) and 0.021 ± 0.020 (cold-drug, two seeds). As on DAVIS, where a signal exists it
corresponds to ranking a low single-digit percentage of the annotated sites first.

### 8.2 Pocket-level plausibility replicates, and holds on unseen drugs

**Table R12.** Precision@10 against the KLIFS 85-residue ATP pocket (ladder), chance 0.151,
ceiling 0.996, n = 210 / 211; per-seed values with their p in brackets.

| model | level | mean ± sd | seed 1 | seed 2 | seed 3 | DAVIS (Table R9) |
|---|---|---|---|---|---|---|
| HyperAttentionDTI | random | **0.207 ± 0.011** | 0.199 (0.001) | 0.204 (0.001) | 0.220 (0.001) | 0.242 |
| | cold-drug | **0.189 ± 0.024** | 0.164 (0.073) | 0.191 (0.001) | 0.211 (0.001) | 0.193 |
| MolTrans | random | 0.161 ± 0.049 | 0.122 (1.000) | 0.216 (0.001) | 0.146 (0.747) | 0.157 |
| | cold-drug | 0.188 ± 0.040 | 0.171 (0.006) | 0.233 (0.001) | 0.158 (0.200) | 0.154 |

**HyperAttentionDTI finds the pocket region on both datasets.** At random all three seeds
are at 1.32–1.46× chance and beat all four nulls — a map borrowed from another protein
(absolute and relative position), attention permuted among residues of the same amino acid,
and attention permuted within the stretch that spans the pocket
(`positional_control_hyperattentiondti_kiba_klifs_policyA.md`). At cold-drug seeds 2 and 3
beat all four and seed 1 beats two (borrowed-absolute and same-residue, not the
within-span shuffle): on unseen drugs the attention still points into the pocket, and in
two seeds of three at residues within it rather than merely at the stretch it occupies.
This is §5's reading — coarsely plausible under shift, finely plausible nowhere — now on
a cold-drug level with 422 drugs rather than 13.

**MolTrans's pocket signal is the same seed-2 effect as its residue signal.** Seed 2 beats
every null at both levels (0.216 and 0.233); seed 1 does at cold-drug against three of four
nulls; seed 1 at random (0.122) and seed 3 at both levels beat none. Its DAVIS values
(0.154–0.157) were at chance; KIBA does not change the verdict that its attention carries no
reproducible pocket information, and adds that it can carry some in a particular training
run.

### 8.3 Faithfulness replicates for both models

**Table R13.** Comprehensiveness delta over a size-matched random control, 200 pairs per
level, mean ± sd over seeds; HyperAttentionDTI in residue space, MolTrans in token space
(§5b; the unit differs, so the two rows are comparable within a model, not across).

| model | random | cold-drug | cells > 0 | DAVIS random | DAVIS cold-drug |
|---|---|---|---|---|---|
| HyperAttentionDTI (residues) | 0.132 ± 0.078 | 0.095 ± 0.066 | 6 / 6 | 0.184 ± 0.056 | 0.113 ± 0.052 |
| MolTrans (tokens) | 0.394 ± 0.263 | 0.291 ± 0.075 | 6 / 6 | 0.458 | 0.362 |

**Both models' attention is load-bearing in every KIBA cell**, at magnitudes within the DAVIS
seed spreads, and for both the margin is smaller on unseen drugs than at random — the
direction DAVIS showed. Faithfulness is the half of the interpretability claim that
replicates cleanly: the attention is used; §8.1 is what it is not used *for*.

*A sensitivity result that must be reported with it: masking MolTrans's top-attended
**residues** (its token attention projected onto residues) against random residues chosen
to change the same fraction of its tokens gives deltas near zero — −0.045 ± 0.112 at random
and −0.005 ± 0.032 at cold-drug, positive in 2 of 6 cells
(`faithfulness_moltrans_kiba_seed*.md`). The token arm removes what the model attended to;
the residue arm removes residues the projection assigns that attention to, which the
tokeniser then re-segments. That the two disagree for MolTrans is §5b's point again —
masking faithfulness does not transfer across tokenisations — and the token-space test,
fixed on DAVIS before KIBA was run, is the one the audit uses. This matched residue-space
control did not exist when DAVIS's MolTrans cells were measured, so DAVIS has no
counterpart to compare it with.*

### 8.4 The gradient recovers the residues the attention misses — on KIBA too

§7c's result was DAVIS-only until now: read the same trained weights with integrated
gradients instead of attention and the residue-level signal appears. KIBA was run with the
same code and the same settings DAVIS used — 32 steps, the path from the padding
embedding — over both audited models, both levels and three seeds — twelve ladders, the correction over the four cells the family
contains (`results/analysis_kiba_policyA/ig_family_kiba.md`,
`src/evaluation/ladder_family.py`).

**Table R11.** KIBA, precision@10 at k = 10, mean ± sd over seeds 1–3, one test pair per
protein. `attention` is §8.1–8.2's audit readout; `IG` is the gradient of the same
checkpoint. p is the median over seeds.

| ground truth | model | level | attention | integrated gradients | chance | IG / attn |
|---|---|---|---|---|---|---|
| UniProt residues | HyperAttentionDTI | random | 0.030 ± 0.019 (p = 0.65) | **0.042 ± 0.020** (p = 0.001) | 0.023 | 1.36× |
| | | cold-drug | 0.021 ± 0.003 (p = 0.80) | **0.043 ± 0.034** (p = 0.002) | 0.023 | **2.02×** |
| | MolTrans | random | 0.032 ± 0.018 (p = 0.65) | 0.031 ± 0.017 (p = 0.53) | 0.023 | 0.97× |
| | | cold-drug | 0.036 ± 0.024 (p = 0.068) | 0.031 ± 0.007 (p = 0.005) | 0.023 | 0.88× |
| KLIFS pocket | HyperAttentionDTI | random | 0.207 ± 0.011 | **0.272 ± 0.026** | 0.151 | 1.31× |
| | | cold-drug | 0.189 ± 0.024 | **0.257 ± 0.069** | 0.151 | 1.36× |
| | MolTrans | random | 0.161 ± 0.049 | 0.190 ± 0.050 | 0.151 | 1.18× |
| | | cold-drug | 0.188 ± 0.040 | 0.174 ± 0.011 | 0.151 | 0.93× |

**Three of the four IG cells survive Holm, where none of the six attention cells did**
(thresholds 0.0125 to 0.05): HyperAttentionDTI at random (p = 0.0010) and cold-drug
(p = 0.0020), and MolTrans at cold-drug (p = 0.0050). Only MolTrans at random fails
(p = 0.53).

**The cell that carries the claim is cold-drug against annotated residues.** There
HyperAttentionDTI's attention is at chance — 0.021 against 0.023, the audit's clearest
null — while the gradient of those same weights is at 0.043, **1.9× chance**, over 422
held-out drugs. The information about which residues matter is in the model; the attention
map does not report it. That is §7c's conclusion, reproduced on the replication dataset at
the level DAVIS could not test.

**MolTrans behaves as the control it was on DAVIS.** Its gradient tracks its attention
(0.88–1.18×) rather than beating it, which is what should happen for a model whose
explanation is at the floor either way: the gradient is not a better readout in general,
it is a better readout of a model that has something to report. Its cold-drug cell does
survive where its attention did not, on equal precision (0.031 against 0.036) but a third
of the seed spread (± 0.007 against ± 0.024) — a difference in stability, not in signal,
and too small to carry a claim.

### 8.5 Controls

**Non-kinase transfer panel** (60 unseen BindingDB proteins, primary analysis excluding
cotransport ions; `control_*_kiba_seed*_noions.md`): HyperAttentionDTI 0.014 ± 0.001
(random) and 0.016 ± 0.004 (cold-drug), MolTrans 0.013 ± 0.008 and 0.016 ± 0.001 — at or
below the kinase cells and at the DAVIS panel's level. Whatever signal the kinase cells
carry does not transfer to non-kinases, for either model, in any seed. KIBA is 229
kinases, so, as on DAVIS, the confound cannot be stratified inside the dataset
(`audit_kiba_binary.md`); the panel is the test.

**Positive control** (`positive_control_kiba.md`): every hard check passes on KIBA's
splits and re-numbered ground truth, and a 2% dose is detected at every level, so the null
results of §8.1 are not a failure of the metric to see a signal of the size DAVIS found.

### 8.6 What KIBA changes

| DAVIS finding | on KIBA |
|---|---|
| HyperAttentionDTI's residue-level claim holds at random (1 of 16 cells survives Holm) | **does not replicate** (0 of 6; one seed of three) |
| The attention finds the pocket region, including under shift | **replicates** for HyperAttentionDTI, at a 422-drug cold level |
| The attention is load-bearing at every level | **replicates** for both models (12 / 12 cells) |
| MolTrans's attention sits at the uniform floor | holds in 2 seeds of 3; seed 2 carries a kinase-only signal |
| Cold-drug collapses accuracy | **does not replicate**: a property of DAVIS's 13-drug split |
| The gradient beats the attention on the same weights (7 of 12 cells) | **replicates** (3 of 4 cells; 1.9× chance where the attention is at chance) |

The claim the two datasets support together is narrower and firmer than either alone:
the published models' attention is **used** and points **into the binding pocket**, but
does not mark **binding residues** — and the one exception on one dataset depends on the
training seed on the other. §7c's finding that integrated gradients recover the residues the
attention misses **replicates** (§8.4): on KIBA's 422-drug cold level the same weights
score 1.9× chance through the gradient and at chance through the attention.

---

## Figures

Built by `python -m src.evaluation.paper_figures` from the files each section cites; full
captions and sources in `results/figures/CAPTIONS.md`. Rebuild after any re-analysis.

| # | file | shows | cited in |
|---|---|---|---|
| 0 | `fig0_design` | the audit's design: datasets, splits, models, the three readings of the explanation, both measurement axes, and the correction | Methods §1, Introduction ¶5 |
| 1 | `fig1_plausibility` | precision@10 for every model × level × dataset against UniProt residues and the KLIFS pocket, seeds as dots, per-level chance | §4, §5, §8.1, §8.2 |
| 2 | `fig2_seeds` | every UniProt cell as three seed dots against chance — the seed-dependence finding | §5, §8.1, Discussion §5b |
| 3 | `fig3_attention_vs_ig` | attention against integrated gradients on identical checkpoints, both datasets, both ground truths | §7c, §8.4 |
| 4 | `fig4_faithfulness` | comprehensiveness delta over a size-matched control; MolTrans's token-space panel kept separate | §5b, §8.3 |

**Figure 1 is the one to keep** if the venue limits the count: it carries the headline on
its own. Figure 2 can fold into its caption, and Figures 3 and 4 into supplementary.
