# Discussion — Limitations (draft)

Draft for the Limitations part of the Discussion. Rewritten 2026-09-14 once the DAVIS audit
was complete: the earlier version was written before the results and four of its items had
gone stale — it declared gradient attributions out of scope, expected the readout choice not
to matter, described the ground truth as protein-level only, and spoke of KIBA as trained.
Each is corrected below. Extended 2026-09-16 once the KIBA replication landed (Results §8):
the KIBA items are no longer conditional, and two new ones state what the replication
exposed — seed dependence that three seeds cannot quantify, and MolTrans's
masking-space-dependent faithfulness. Numbers come from `paper/results.md` and
`paper/methods_data_and_evaluation.md`, where each is sourced. No *[PENDING]* marks remain.

---

**Kinase-only training data, and a confound that cannot be stratified.** DAVIS and KIBA are
kinase panels, and the natural test — compare kinase and non-kinase targets inside a cell —
is not merely underpowered here but impossible: DAVIS's 6,011 test rows contain 3,307 rows
our classifier names as kinases and **zero** non-kinase rows, and KIBA is 229 kinases
(Results §6). No panel size fixes it, because the gate counts non-kinase targets in the cell
being scored. The substitute is transfer to 60 BindingDB proteins no model has seen, which
is strictly harder than cold-target: protein, family and drugs all change at once. A
difference between the arms therefore bounds the family effect rather than isolating it,
and rests on 60 proteins whose affinities come from different assays.

**What that does and does not limit.** It limits the *generality* of the verdict, not its
*validity*. The claims this paper audits were made on DAVIS and KIBA — they are the
benchmarks on which DTI interpretability is reported — so testing them there is testing
them where they live; a null found on some other family would leave the published claims
untouched. Two things follow, and the paper needs both stated. Nothing here licenses
"attention fails for drug-target interaction in general": it licenses "attention fails for
these models, on the benchmarks their claims are made on, at four levels of shift, on two
datasets". And the evidence that the failure is not merely a kinase artefact is the
transfer panel rather than a stratification: every audited model is at chance on 60 unseen
non-kinase proteins (Results §6), and the few above-chance non-kinase cells trace to an amino-acid
preference — histidine, 3-15x enriched in the top ten — meeting histidine-rich metal
sites, which is a property of the attention rather than of the family. A reader who wants
the kinase-free version of this audit needs a kinase-free benchmark carrying residue-level
ground truth, and building one is a paper of its own.

**Three ground truths, none of them per-pair at a usable scale.** UniProt annotations
(binding, active and nucleotide-binding sites) and the 85-residue KLIFS ATP pocket are both
defined per protein, so every drug measured against a protein is scored against the same
residues; a compound binding outside the annotated pocket — an allosteric inhibitor — is
scored as if it bound the ATP site. Annotation is also incomplete, so precision@k is a lower
bound. DAVIS's mutant and phosphorylated variants (63 identifiers) are scored against their
wild-type entry's sites. The third ground truth added here *is* per pair — KLIFS
interaction fingerprints, the residues a drug is measured to contact in its own co-crystal
(Results §7) — and its limitation is arithmetic: a pair is scorable only where that exact
drug was crystallised with that exact kinase, which is 38, 60, 39 and 12 of DAVIS's test
pairs at the four levels, falling to 3 at cold-pair under the sequence policy. Its
paired-versus-swapped comparison is read off overlapping intervals rather than off a
difference of means, and cold-pair is not reported at all. KLIFS covers kinases only, so the
non-kinase panel is scored against UniProt alone.

**A verdict can depend on how the attention is read out.** The earlier draft of this section
expected alternative projections to change the numbers "though not the comparison between
levels". That expectation was wrong and the paper now reports it as a finding rather than a
caveat (Results §7b): across 25 proteins, an alternative readout's top-ten residues overlap
the published readout's by **2–12%**, and HyperAttentionDTI's KLIFS agreement at cold-target
reads 0.367 under a channel-max reduction and 0.081 — below the 0.143 chance level — under a
receptive-field projection, against 0.186 as published. What survives the choice is the
residue-level null (every readout of every model stays at chance against annotated residues,
all inside the seed spread bar HyperAttentionDTI's channel-max cold-drug cell) and MolTrans's
pocket floor. What does not survive it is any statement about the *size* of the coarse
signal. Five alternative readouts were tried beside the three published ones; the space of
defensible readouts is larger, and a reader should treat every pocket-level magnitude in
this paper as one reading among several.

**What faithfulness can say, and the intervention-size problem.** Comprehensiveness replaces
the top-*k* attended residues with an unknown amino acid and measures the change in
prediction. The masked input is off the training distribution, which moves predictions for
reasons unrelated to the explanation; the random-masking control removes that on average but
not per pair. For a sub-word model the two arms are not even the same size of intervention:
masking MolTrans's ten most-attended residues changes 48% of its tokens where ten random
residues change 95%, and its residue-space delta was negative in 11 of 12 cells for that
reason alone (Results §5b). MolTrans is therefore measured in token space, where both arms
remove the same number of tokens — which fixes the comparison *within* the model across
levels and seeds, and makes its deltas **not numerically comparable** with the two
residue-level models', because the unit differs. Faithfulness uses the first 200 test pairs
per level at *k* = 10 (75 and 76 at the cold levels under the sequence policy) and
establishes whether the attended residues are load-bearing, not that they are the model's
full reason.

**One alternative explanation method, with its own choices.** Integrated gradients are used
to separate "the attention is a poor report" from "the model never learned the site"
(Results §7c) — a question no attention measurement can answer. IG is itself parameterised:
the path starts at the padding embedding (the same "no residue here" the masking uses), 32
steps by the midpoint rule, attributions taken as magnitude over the protein embedding, and
for ColdSite-DTI's recurrent tower the attribution runs in train mode with every stochastic
component switched off (dropout modules to eval, the dropout *attributes* of
`MultiheadAttention` and `RNNBase` zeroed), verified deterministic and agreeing with the
eval-mode path to 1.5e-8. A different baseline or step count would give different
magnitudes. One alternative method is enough to show attention under-reports; it is not a
survey of attribution methods, and a different attribution might place the gap differently.
Two further caveats belong to the result itself: ColdSite-DTI's gradient is **noisy** where
it matters (cold-target 0.074 / 0.021 / 0.037 across seeds; cold-drug against the pocket
0.352 / 0.634 / 0.366), so the effect rests on the permutation test rather than on a precise
estimate; and **integrated gradients were added after the attention results were seen**, so
they are a secondary analysis, Holm-corrected within their own family and never pooled
with the attention cells. A reader should treat "7 of 12 for the gradient against 1
of 16 for the attention" on DAVIS, and "3 of 4 against 0 of 6" on KIBA, as separately
corrected families, which is how Results §7c and §8.4 state it. Being decided in advance is
what KIBA's arm adds: its four cells were run after DAVIS's result was known, but with the
protocol, the step count and the family fixed by that earlier run rather than chosen to suit
the outcome.

**One split per level, three training seeds, and two kinds of interval.** Each level has a
single fixed split, and the three seeds vary initialisation and batch order only, so reported
spreads exclude split-selection variance. Training is not bit-reproducible on GPU (cuDNN's
LSTM kernels are nondeterministic; an identical re-run moved single seeds by up to 0.058
CI). Seed spreads and bootstrap intervals over proteins (Results §7d) answer different
questions and neither substitutes for the other; where a cell is small, the interval is the
honest one.

**Small held-out sets on the cold levels.** DAVIS has 68 drugs, so cold-drug holds out 13 —
close to anecdote for a claim about unseen chemistry. After the sequence policy, cold-target
and cold-pair test sets contain 68 and 72 proteins with usable sites (KIBA: 42 and 41)
against 349 on random. Measured, that costs less precision than the seed spreads suggest
(±0.005–0.009 against UniProt at n = 68) and more against KLIFS (±0.020–0.031, five times the
random level's), so the cold-level *pocket* magnitudes are the loosest numbers in the paper.
The positive control shows the permutation test still detects an explanation that ranks 2% of
true sites first at these sizes, so a result at chance there is a null rather than a lack of
power; it does not make the estimates as precise as the random level's.

**DAVIS's sequence file, and what only retraining could remove.** DeepDTA's DAVIS protein
file, used here as in most DTI benchmarks, gives all 54 variants with a wild-type entry
exactly the wild-type sequence, so 442 targets are 379 distinct sequences and no sequence
model can tell a mutant from its wild type; ten targets' sequences hold few or none of the
ATP pocket's residues (RET and its three mutants are residues 1–430). 13.6% of cold-target
and 12.5% of cold-pair test rows are proteins seen in training under another name. We score
the cold levels on targets unseen by sequence, count one protein per sequence, and drop the
pocketless targets (Methods §2.4), with all-rows accuracy beside it. Retraining on
sequence-clean splits puts the leak at **0.019 of cold-target's 0.038 total drop** — but
that was done for DeepDTA only, the anchor, and the licence for applying the re-scored
values to the other three models is that the two independent methods agree on DeepDTA
(0.019 retrained, 0.023 re-scored). Two effects remain: 13.6% of cold-pair's validation rows
are seen by sequence, which influenced which epoch was kept, and every model learned from
duplicated sequences. Results on this file elsewhere in the literature carry the same
leakage. KIBA has none of these properties.

**Single-model controls.** The volume-matched control (Results §2) was run for ColdSite-DTI
only, so the 12%/88% division of the cold-pair drop between fewer rows and genuine
difficulty is established for one model and assumed for the others. The leakage retraining
was run for DeepDTA only, as above. Retraining every model for either control was not
affordable at the compute available.

**The 1,000-residue window.** Sequences are truncated to 1,000 residues and sites beyond it
excluded; 16 DAVIS and 9 KIBA targets lose every site and leave the evaluation. They are
systematically the longest proteins (median final annotated residue 1,320 against 312 for
retained DAVIS targets), mostly large multidomain receptor kinases, so nothing here should
be extended to proteins much longer than the window. MolTrans reads beyond it; its
explanation is cut to the same window for comparability, which scores the model on less than
it saw.

**A binary task for every model.** Two of the audited models are classifiers in their
published form, so all four are compared on a binary task at DeepDTA's published thresholds
(DAVIS pKd ≥ 7.0, KIBA score ≥ 12.1). ColdSite-DTI and DeepDTA were designed for regression;
their binary results are not their regression results, and a different threshold would change
the class balance and every AUPRC.

**Published models retrained under a shared protocol.** Each published model uses its
authors' optimiser, learning rate, batch size and tokeniser, but checkpoints are selected by
one rule for all (lowest validation loss after a 10-epoch floor, patience 15; DeepDTA 10).
MolTrans's published script trains a fixed number of epochs and keeps the best validation
AUROC; HyperAttentionDTI's published class weights are not used. Each subject is therefore
the published architecture and recipe under our protocol, not the published checkpoint.
MolTrans's published code keeps dropout active at inference and its accuracy is reported
with that noise, as published (faithfulness holds the RNG fixed per forward pass). DeepDTA
runs as a PyTorch port of the original Keras implementation. One trainer bug of our own is
worth recording: the vendored MolTrans module reseeds torch on import, so an earlier run's
three seeds were one seed three times; the corrected cells are the ones used here.

**What the non-kinase control can separate.** The panel's binding sites differ from kinase
ATP sites in composition as well as family: histidine-rich (8.8× background), many of them
metal sites, where kinase sites are glycine-, aspartate- and lysine-rich. ColdSite-DTI's
attention prefers histidine (3–15× enriched), and most of its above-chance panel precision is
recovered by shuffling attention among residues of the same amino acid. A kinase–non-kinase
gap therefore mixes family with amino-acid composition; the same-residue null is reported
beside it for every model so the two can be told apart. No bootstrap interval was computed
for the panel arm, whose proteins are a different population (Results §7d).

**Scope of the audit.** Three attention-based models are audited (HyperAttentionDTI, MolTrans
and our own ColdSite-DTI), with DeepDTA as an accuracy anchor. Three published
attention-based DTI models is a small sample of a large literature, and the models chosen are
those whose code we could run faithfully; a claim about attention-based DTI interpretability
in general rests on the argument that these are representative, not on the sample size.

**Compute-driven choices on KIBA, and a precision that is not uniform.** KIBA trained
2026-09-14 to 09-15, 18 of 18 cells (Results §8). It uses mixed precision (float16 autocast with loss scaling) for DeepDTA and
HyperAttentionDTI, which ran 2.3× and 2.0× faster on the T4s available, and **full
precision for MolTrans**. That asymmetry was forced, not chosen: under autocast on KIBA,
MolTrans produced NaN losses from batch ~4,040 of its first epoch and could not be trained
at all. The cause is in the vendored implementation — it defines its own LayerNorm as
`(x − µ)/sqrt(var + 1e-12)`, and 1e-12 is below float16's smallest subnormal (~6e-8), so a
zero-variance row yields 0/0; PyTorch's autocast keeps `nn.LayerNorm` in float32 but cannot
recognise a hand-written one, and a NaN arising in the forward pass is in the weights
thereafter, which a gradient scaler does not address. We report this because it is a
reproducibility hazard for anyone applying mixed precision to published DTI baselines, and
because it means **precision is a per-model property of the KIBA arm** and must be read
that way. It also has one convenient consequence: MolTrans's KIBA cells are full precision
like its DAVIS cells, so its cross-dataset comparison carries no precision caveat.

The mixed-precision evidence itself is thinner than the use made of it. It was validated on
**one model and one cell**: DAVIS HyperAttentionDTI cold-pair, three seeds each, where
mixed precision moved test AUROC by −0.013 and precision@10 by +0.001, both inside the
full-precision seed spread (0.038 and 0.012; `results/amp_validation_davis.md`). Three
seeds rule out only a gross effect, one mixed-precision seed (0.603) sat below every
full-precision seed, and the two arms also differed in PyTorch version (2.10 against 2.11)
and in the device the ladder ran on. Extending that validation to every model was the
assumption that MolTrans falsified. DeepDTA's KIBA cells under autocast train normally
(test AUROC 0.918 ± 0.001 at random, 0.832 ± 0.001 at cold-drug), but they inherit the same untested assumption, and a reader
should treat DAVIS-versus-KIBA accuracy differences for DeepDTA and HyperAttentionDTI as
carrying a precision caveat that MolTrans's do not. Cells longer than one 11-hour compute
session continue from their last finished epoch, restoring model, optimiser, scheduler,
loss scaler and RNG state. The arm's scope was **decided on compute
grounds, 2026-09-14**: random and cold-drug only, for HyperAttentionDTI, MolTrans and the
DeepDTA anchor — 18 cells, ~101 GPU-hours over four accounts. The honest statement is that
the replication's breadth was set by available GPU hours, not by the question, and the
paragraph below says exactly what that leaves uncovered.

**An asymmetric replication.** KIBA is the replication (Results §8) and repairs DAVIS's weakest axis (422 held-out drugs at
cold-drug against 13), but it cannot repair the family confound — it is also kinases — and
its cold-target level holds out only 45 targets (42 with usable sites), fewer than DAVIS's
68. The KIBA arm is also narrower than the DAVIS one in three ways, all decided by
available GPU hours rather than by the question, and all of which we state rather than
leave a reader to infer: it trains **random and cold_drug only** (KIBA's cold-target is
weaker than DAVIS's, and cold_pair would repeat DAVIS's checkpoint-selection instability on
a 1,334-row validation set); it covers the two **published** models and the accuracy anchor
but **not ColdSite-DTI**, so our own model is audited on one dataset where the models whose
claims this paper is about are audited on two; and two explanation-side analyses — readout
variants and per-pair drug contacts — remain DAVIS-only. Anything the KIBA arm does not
cover is a DAVIS result, and the Results section says so cell by cell. Integrated gradients
are no longer in that list: Results §8.4 replicates them on KIBA for both audited models, which
leaves the readout-dependence result (§7b) as the largest unreplicated claim — and it is a
claim about the instrument, so a reader should ask whether it holds for KIBA's proteins
before relying on its magnitude.

**The EviDTI result is a reading of source code, not a measurement.** Results §7e states
that a 2025 published model's residue attention cannot depend on the drug. That claim
rests on its released code (CC-BY-4.0, read 2026-09-18, recorded with line references in
`results/evidti_code_audit.md`), not on retraining it: its two drug encoders need
TensorFlow and PaddlePaddle, and the 3D encoder's pretrained weights are not in the
repository, so it is not one of this audit's trained subjects. Two consequences belong in
the paper rather than in a reader's inference. We report **no** precision@k, no
faithfulness and no accuracy for EviDTI, and nothing here says its predictions are poor or
its uncertainty quantification unsound — that is its actual contribution and we did not
test it. And the claim is only as current as the code we read: if the authors release a
version whose attention takes the drug as an input, Results §7e describes the version we read and
should be re-checked against theirs, which takes a minute.

**Three seeds detect seed dependence; they cannot measure it.** The replication's central
result is that a residue-level verdict moves across chance between training seeds of the
same cell: HyperAttentionDTI's KIBA random cell reads 0.017, 0.022 and 0.053 against a
chance of 0.023, and MolTrans's cells are at the floor in two seeds and well above it in
the third. Three seeds are enough to establish that this variance exists and to stop a
single-seed claim; they are far too few to estimate its distribution, to say how often a
seed would clear a Holm threshold, or to distinguish a bimodal outcome from a wide
unimodal one. The per-seed values are reported for every cell so that a reader can see the
spread rather than infer it from a ±, but the paper cannot say what fraction of training
runs would support the published claim. Ten or more seeds on the two random-split cells is
the obvious follow-up and was beyond the compute available here.

**MolTrans's faithfulness depends on which space the masking happens in, and the two
spaces were not measured on both datasets.** The audit's MolTrans faithfulness is measured
in token space, fixed on DAVIS before KIBA ran, where the explanation's arm and the control
remove the same number of tokens by construction. On KIBA the residue-space variant with a
token-matched control — masking the residues its token attention projects onto, against
random residues chosen to disturb the same fraction of tokens — gives deltas near zero
(−0.045 ± 0.112 at random, −0.005 ± 0.032 at cold-drug) where the token-space test gives
+0.394 and +0.291. Both are reported (Results §8.3). The disagreement is informative
rather than contradictory — it is the non-transferability of masking across tokenisations
again — but it means a single number cannot be quoted for "MolTrans's faithfulness", and
the matched residue-space control did not exist when DAVIS's MolTrans cells were measured,
so DAVIS has no counterpart for that column.
