# Discussion (draft)

Drafted 2026-09-13, rewritten 2026-09-14 once the DAVIS audit was complete (three models,
four levels, three seeds, Holm over all sixteen cells). Every number here is in
`paper/results.md` with its source file; nothing states a result for a model whose numbers
are not there. Extended 2026-09-16 with §5b, the KIBA replication (Results §8), and 2026-09-18 with its
integrated-gradient arm (Results §8.4); no *[PENDING]* marks remain. Limitations are in `paper/limitations.md`.

---

## 1. What the audit found

Of sixteen cells — three attention-based models and a uniform-attention control, at four
levels of distribution shift — **one supports the residue-level claim after correction**:
HyperAttentionDTI on the random split, at 1.7× chance (precision@10 0.034 against 0.020,
p = 0.0020 against a Holm threshold of 0.0031). It beats the nulls that could explain it
away: a map borrowed from another protein and attention permuted among residues of the same
amino acid in every seed (both 0.021, p = 0.001), and attention permuted within the
site-spanning stretch in two seeds of three. On the split that published work reports, for
the model that generalises best, the interpretability claim holds — and that is the only
place it holds. **On KIBA it does not hold even there** (§5b): the same model at the same
level is above chance in one seed of three, and none of KIBA's six cells survives
correction. Across both datasets, then, the residue-level claim survives nowhere that a
replication confirms.

The same model is at 1.26× chance on unseen targets and 1.18× on unseen pairs, neither
distinguishable from chance. MolTrans sits at the metric's floor at every level: its four
cells (0.020–0.028) are within one standard deviation of what an attention map of *equal
weight everywhere* scores (0.017–0.020). ColdSite-DTI, ours, is at chance at all four
levels (0.013–0.022 against 0.019–0.020) and survives no cell. So the audit's answer to its
own question is: **the residue-level
interpretability claim survives only on the easiest split, for one of the three models,
and nowhere under the distribution shift that deployment implies.**

Three further findings turned out to matter more than that verdict, and all three are
about measurement rather than about these models. Which residues an attention map
highlights depends mostly on an undocumented reduction choice (§3b). A masking-based
faithfulness test can invert its own conclusion when the model reads sub-words (§3b). And
the gradient of the same checkpoint finds the pocket about twice as well as the attention
does (§4) — so where attention fails, it is often the *report* that fails rather than the
model.

## 2. Attention that is used but does not point at the binding residues

All three models show the same dissociation, which answers the question this section was
left open on: it is not a property of our model.

For **ColdSite-DTI** the three measurements come apart cleanly. Its attention is
**faithful** — at every level, masking the ten residues it attends to moves the prediction
more than masking ten random ones (12/12 cells). It is **coarsely plausible**: against the
85-residue KLIFS ATP pocket it scores about twice chance, beyond what position or
amino-acid preference explain, mostly because it concentrates on the kinase domain. And it
is **not finely plausible**: against the dozen residues UniProt annotates it sits at chance
at all four levels (0.013–0.022 against 0.019–0.020, 7 of its 12 cells at or below chance),
and at random the whole bootstrap interval over proteins — 0.015 [0.013–0.018] — lies
*below* the 0.020 chance level. The positive control says this is a real null rather than an
underpowered test: a 2% dose of true sites is detectable at every level, on these very
protein sets, while its cells are worth an equivalent dose of 0.006 or less.

**HyperAttentionDTI** is the same shape with one extra step: finely plausible at random
(§1), coarse-only once the split is cold. **MolTrans** has the shape without the content —
faithful in token space at every level (§5b of Results), and at the floor against both
ground truths. So the pattern across three models is: *attention is used, and it marks a
region rather than a site.*

Two things ColdSite-DTI demonstrably uses: the kinase domain as a region, and histidine —
its top-ten attention is 3–15× enriched in histidine, a preference that produces
above-chance "hits" on histidine-rich non-kinase metal sites (§6 of Results) and *misses*
the glycine-rich kinase ATP site it trained on. A reader inspecting its attention maps
would see highlights near the pocket and could take them as residue-level explanations.
They are not.

Two independent measurements then show the drug plays no part in these explanations. Given
a drug's own crystallographic contacts versus another drug's contacts in the same pocket,
the correct drug buys at most +0.011 precision@10, and in two of nine cells the *wrong*
drug scores higher. And ColdSite-DTI's protein-tower self-attention — computed by its
forward pass, discarded, and independent of the drug by construction — scores *higher*
against the pocket than its drug-conditioned cross-attention (0.238 against 0.219 at
random; 0.268 against 0.243 at cold-target). An explanation that does not change with the
drug is not explaining a drug–target interaction.

## 3. Plausibility depends on the resolution of the ground truth

"Does attention mark the binding site?" has no single answer: the same checkpoints are at
chance at residue resolution and above chance at pocket resolution. Claims in the
literature are usually made at residue resolution — a highlighted residue shown beside a
crystal-structure contact — and should be tested there. A pocket-level agreement is a
weaker statement that a protein-sequence model can satisfy by learning where the kinase
domain is.

This audit used three resolutions, and they disagree in an informative order: ~85 pocket
residues (above chance), ~19 residues contacted by the specific drug (at chance once the
pocket is controlled for), and the ~12 residues UniProt annotates (at chance). Plausibility
should be reported at every resolution a paper's claim spans, each beside its own chance
level and ceiling, because the chance level itself moves with the resolution — 0.143 for
the pocket, 0.025 for drug contacts, 0.020 for annotations.

## 3b. What the measurement depends on

Three results here are about the instrument, and the first two would have produced a
wrong published claim of our own if we had not checked. The third is about somebody
else's, and it needed no measurement at all.

**The residues a readout points at are mostly a property of the readout.** Between the
tensor inside a network and one weight per residue, somebody chooses which axis to reduce,
which layer to read, and how to spread a convolution position or sub-word token over
residues. Scoring the same checkpoints through readouts another author could reasonably
have chosen, the top-ten residues overlap the published readout's by **2–12%** (one
exception at 90%). One choice moves HyperAttentionDTI's pocket agreement at cold-target
from 0.081 — *below* the 0.143 chance level — to 0.367, against 0.186 as published. A
published attention figure is, to that extent, a picture of a reduction choice. The
audit's own verdicts do survive this: every readout of every model stays at chance against
annotated residues (0.010–0.057 against 0.020, all inside the seed spread bar one cell),
and none lifts MolTrans above the pocket's chance level (0.120–0.189 against 0.143). What
the choice changes is the size of the coarse signal, not the existence of the fine one.

**An explanation can be incapable of the claim made for it, and the source says so.**
A 2025 *Nature Communications* model (EviDTI) presents per-residue attention for four
drug–target complexes and concludes that high-attention residues coincide with the binding
site. Its attention is computed from the protein's language-model embedding alone; no drug
tensor reaches it, and there is no cross-attention in the model. For a fixed protein,
every drug therefore yields the identical map, for any weights (Results §7e). The claim is
not refuted by a better measurement — it is refuted by the computation graph, which anyone
can read before the figure is drawn. That suggests a cheap, general check for this
literature: **state which inputs the explanation is a function of.** Our own audit's
per-pair result (§7 of Results: the correct drug buys at most +0.011 precision@10) is the
measured version of the same problem, and the two agree.

**A masking-based faithfulness test can invert its own conclusion.** Faithfulness
subtracts a random-masking control from the explanation's comprehensiveness, which assumes
both arms change the input by the same amount. For a model reading sub-word tokens they do
not: masking MolTrans's ten most-attended residues changes 48% of its tokens, and ten
random residues change 95% — for one protein, 0.8% against 99%. Under that test its
faithfulness delta was negative in 11 of 12 cells, which reads as "its attention points at
residues that matter less than arbitrary ones". Measured in the space the model actually
reads, with both arms removing the same number of tokens, the sign reverses in all twelve.
The lesson generalises beyond this model: **an intervention-based explanation metric is
only interpretable when the intervention is the same size in both arms**, and for
tokenised inputs that is not automatic.

## 4. Does explanation quality degrade with distribution shift?

Yes, but as a step rather than a slope, and DAVIS's levels are categories rather than a
severity ladder — an unseen drug costs far more accuracy than an unseen target, and
cold-pair is no harder than cold-drug for the accuracy anchor. The question is therefore
per level.

The fine-grained signal is what degrades. HyperAttentionDTI's residue-level agreement goes
1.7× chance → 1.26× → 1.18× from random to cold-target to cold-pair, crossing from
"survives Holm over sixteen cells" to "not distinguishable from chance". Faithfulness
degrades the same way without vanishing: its margin over random masking falls from
0.184 ± 0.056 at random to 0.056 ± 0.008 at cold-pair. The attention is still load-bearing
under shift; it is simply load-bearing for something that no longer coincides with the
annotated site.

The coarse signal is more robust — pocket-level agreement stays at 1.3–1.4× chance across
the cold levels for HyperAttentionDTI and ~2× for ColdSite-DTI — which is consistent with
the region, not the site, being what these models learned.

**And for two of the three models the degradation is a reporting failure, not an ignorance
failure.** Integrated gradients on the same checkpoints — same ground truth, same protein
sets, same test, only the explanation changed — survive Holm in **seven of twelve DAVIS cells,
against one of sixteen for the attention**, and in **three of four KIBA cells against none
of six** — the one comparison in this paper that replicates in the direction that rescues
the models rather than indicting them. HyperAttentionDTI's gradient is at 2.7–4.1×
chance at *all four* levels, including the cold ones where its attention is at 1.2–1.3× and
fails correction; ColdSite-DTI's is at 2.3–2.7× at cold-drug and cold-target, where its
attention is at chance. The information is in the weights; the attention head does not
report it.

**MolTrans is the control that makes this a finding rather than an artefact.** Its gradient
matches its attention to within noise (0.9–1.1× on annotated residues and on the pocket on
DAVIS; 0.9–1.2× on KIBA), and both sit at the floor. So the two failures are different in kind: for two models the
attention under-reports a site the model does represent, and for the third there is nothing
to report. An audit that measured only attention could not have told those apart, and would
have filed all three under the same verdict.

The practical form of this is the most useful thing in the paper: **an attention map is a
lossy summary of what a model uses, and it is lossiest exactly where interpretability is
supposed to earn its keep** — under distribution shift, where the gradient of the same
weights recovers three to four times chance and the attention recovers nothing. A
practitioner reading an attention figure is seeing less than the model knows; a paper
validating a model by its attention map is measuring its interpretability head rather than
its knowledge.

Two accuracy caveats belong beside all of this. Cold-target accuracy was inflated for
every model by sequence leakage — 0.021–0.023 for three models and 0.041 for MolTrans —
so part of the "cold" difficulty in published DAVIS work is not difficulty at all. And
MolTrans at cold-pair predicts at chance on unseen proteins (AUROC 0.530 ± 0.024), so its
explanation scores there describe an explanation of nothing.

## 5. What an attention-explanation claim should be tested against

The controls this audit needed. Each one changed or protected a conclusion, and the last
three did not exist in our plan until a number forced them.

- **A positive control for the metric** (explanations of known quality): without it a
  result at chance cannot be told from an underpowered test. A 2% dose of true sites is
  detectable at every level here, so the nulls are real nulls.
- **A masking control for faithfulness**: masking anything moves a prediction; only the
  excess over random masking is evidence.
- **Position and residue-type nulls for plausibility**: a uniform null is fooled by an
  attention map with a positional or amino-acid habit. The non-kinase "signal" survived
  the positional null and not the residue-type null.
- **An intervention matched in size, not in units** (§3b): for tokenised inputs, k
  residues is not a fixed-size intervention, and comparing unequal interventions inverted
  a conclusion.
- **A control that changes only the thing being claimed**: for a per-drug claim, the same
  protein and the same number of sites with a *different drug's* contacts. Without it,
  pocket-finding reads as drug-specific binding-site recovery.
- **More than one defensible readout** (§3b): a verdict that depends on the reduction
  choice is a property of the reduction.
- **A second explanation method** (§4): it separates "the attention is a poor report" from
  "the model does not know", which no attention measurement can do.
- **One protein per distinct sequence**: averaging per name or per pair counts one protein
  many times; DAVIS's 442 names are 379 sequences.
- **Seeds, spreads, and intervals over proteins**: single seeds on the cold levels move by
  more than most reported differences (ColdSite-DTI cold-pair: 0.56–0.74 AUROC across
  seeds), and a cell resting on three scorable pairs carries an interval up to 0.067 wide —
  wider than every difference that table reports. A seed spread and an interval over
  proteins answer different questions, and a cell of 68 needs both.

## 5b. Does the verdict replicate? KIBA

*See **Figure 2** (`results/figures/fig2_seeds.pdf`): each cell as three seed dots against
its chance level, which is the evidence for the seed-dependence argument below.*

The audit's one positive residue-level result was a single cell of sixteen, so the
replication was aimed at it. KIBA repeats the two published models at random and cold-drug,
three seeds, with every DAVIS decision unchanged and nothing tuned on KIBA (Results §8).

**The surviving cell does not replicate.** None of KIBA's six cells survives Holm, and
HyperAttentionDTI at random — DAVIS's one survivor — reads p = 0.48 across seeds, with one
seed of three above chance (0.053, 2.3× chance and beating every null) and two at chance.
The honest reading is not "the effect is absent on KIBA" but something more uncomfortable
for the literature: **an effect of this size is not stable across training seeds**, and a
paper reporting one seed would have called it either a confirmation or a refutation
depending on which seed it drew. MolTrans shows the same instability from the other
direction: at the uniform floor in every DAVIS cell, it produces a kinase-specific signal
in KIBA seed 2 at both levels (0.053 and 0.063, beating every null) and nothing in seeds 1
and 3. Three seeds are enough to see that the variance is there; they are not enough to
estimate it, which is why we report per-seed values throughout rather than means alone —
and why Results §7f counts the disagreement across every cell instead of leaving it as two
anecdotes.

**What replicates is the coarse signal and the faithfulness.** HyperAttentionDTI's
attention points into the KLIFS pocket in all three KIBA seeds at random (1.32–1.46×
chance, every null beaten) and in two of three on unseen drugs, where KIBA holds out 422
drugs against DAVIS's 13 — so the "coarsely plausible under shift" finding survives on the
axis DAVIS could not support. Both models' attention is load-bearing in all twelve KIBA
cells, at magnitudes inside the DAVIS seed spreads. The dissociation of §2 is therefore a
property of these models rather than of one benchmark: the attention is used, it is in the
right neighbourhood, and it does not mark the residues.

**And one DAVIS finding turns out to be a property of DAVIS.** Cold-drug costs 0.18–0.24
AUROC on DAVIS and 0.09–0.11 on KIBA. With 13 held-out drugs, DAVIS's cold-drug level
measures which 13 were drawn as much as it measures unseen chemistry. The model ordering
holds on both (HyperAttentionDTI loses least), but the severity does not, and any paper
quoting DAVIS cold-drug as evidence about unseen compounds — ours included, before this
replication — was quoting a 13-drug sample.

## 6. Benchmark hygiene

**DAVIS leaks, and the leak is worth about half of its cold-target difficulty.** Every
mutant in DeepDTA's DAVIS file carries the wild-type sequence, so 442 targets are 379
distinct sequences and cold splits by target *name* test 12–14% of rows on proteins seen
in training. Re-scoring on sequence-unseen targets says leakage inflated cold-target by
0.021–0.041 depending on the model; retraining on sequence-clean splits — with a control
that keeps the leak at matched row count and class balance — puts the number at **0.019 of
the 0.038 total drop**, the other half being the smaller training set. Two methods with
nothing in common agree. Ten targets' sequences also lack the kinase domain the drugs
bind, and no mutant-specific claim can be tested on this file at all.

**The family confound cannot be tested on these benchmarks.** DAVIS's test set contains
3,307 kinase rows and zero non-kinase; KIBA is 229 kinases. A stratified
kinase-versus-non-kinase comparison inside a cell is therefore impossible at any panel
size, and an external panel — 60 non-kinase proteins no model here has seen — is the
substitute, with the accompanying loss of control over drugs and protein length.

**The drug axis is the thinnest part of the design.** DAVIS holds out 13 drugs at
cold-drug; KIBA holds out 422. A cold-drug result on 13 compounds is close to anecdote,
and this is where KIBA does not merely replicate but repairs.

We therefore recommend: split and deduplicate by sequence and report the sequence audit
beside the split; use sequences that contain the mutation for any mutant claim; state the
family composition of the benchmark when a family confound is possible; and prefer
benchmarks whose held-out drug set is large enough to support a cold-drug claim.

## 7. Conclusion

We audited three attention-based DTI models under four levels of distribution shift, with
one multiplicity correction across the whole family, three ground truths at different
resolutions, a positive control for the metric, and nulls for position, amino-acid
preference and drug identity.

One of sixteen cells supports the residue-level interpretability claim: the
best-generalising model, on the random split, at 1.7× chance. On a second dataset that cell
does not replicate — it is above chance in one training seed of three — so the claim
survives nowhere that a replication confirms. Under distribution shift no model's attention
marks annotated residues better than chance, none distinguishes the
drug's own crystallographic contacts from another drug's in the same pocket, and one
model's attention is indistinguishable from a uniform map everywhere on DAVIS and in two of
three KIBA seeds. What survives at every level, on both datasets, is coarser: attention
that is load-bearing, and that concentrates on the right region — including on a cold-drug
level with 422 held-out drugs.

The measurement lessons may outlast the verdict. The residues an attention map highlights
depend mostly on an unreported reduction choice; a masking-based faithfulness test can
invert its own sign when the intervention is not size-matched; and the gradient of the same
checkpoint recovers the pocket about twice as well as the attention, so an attention map
understates what the model uses — most of all under the shift where interpretability is
supposed to earn its keep. Claims of the form "our attention identifies binding sites"
should be tested at the resolution they are made at, against nulls that can explain them
away, under the shift the model will meet, and with more than one way of reading the
attention out.

A fourth lesson came from the replication itself, and it is the one we would most like the
field to take up: **an interpretability verdict of this size is seed-dependent**. Counted
over every cell rather than anecdotally (Results §7f, Table R12), the three seeds disagree
about their own verdict in **11 of 16 cells**, and in **15 of 16 the spread across seeds is
larger than the cell's distance from chance**. A paper reporting one training run would
therefore have had an above-chance result available in eleven of these sixteen cells —
including cells this audit reports as null, and including the model whose attention is
otherwise indistinguishable from a uniform map. A single-seed attention figure, which is
what published work almost always shows, cannot establish or refute the claim it
illustrates. Report every seed, or report none.

There is a reassurance inside that number, and it belongs to the method rather than to the
models. Exactly one cell of the sixteen has all three seeds above α on their own, and it is
the same cell — HyperAttentionDTI at DAVIS random — that survives Holm correction over the
whole family. Agreement among replicate runs and family-wise error control were computed
independently and select the same cell, so the correction is not discarding real effects:
what it discards is what does not reproduce when the model is retrained.
