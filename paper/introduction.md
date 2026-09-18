# Introduction (draft)

Paragraph-by-paragraph, with the results filled in from `paper/results.md` (2026-09-14).
Every number here is one this paper reports; the KIBA replication was folded into ¶5-¶7 on
2026-09-16 and no *[PENDING]* marks remain. Citations are by model name and year until the
venue's style is chosen; `paper/references.md` holds the verified entries.

---

**¶1 — Why explanations matter in DTI prediction.** Deep models now predict drug–target
binding from sequence alone, and the most cited of them present an explanation beside the
prediction: an attention map over the protein that is said to mark where the drug binds.
In drug discovery that map is not decoration. A chemist deciding which residues to mutate,
or which series to pursue against a new target, reads it as mechanistic evidence, and the
published figures invite exactly that reading. *[Cite: HyperAttentionDTI (2022), MolTrans
(2021); recent examples that present attention or interpretability alongside cold-start
results — DMFF-DTA (2025), EviDTI (2025), GPS-DTI (2025), CS-DTA (2026).]*

**¶2 — How those claims are validated.** The evidence for such claims is usually an
inspected attention map, a case study, or a hit rate on a random split, where every test
drug and protein has close relatives in training. The setting that motivates the models
is the opposite: a new target, a new chemical series, or both. Recent models report
accuracy under these cold-start conditions, often beside an interpretability analysis;
whether explanation quality itself holds up as the shift grows has not, to our knowledge,
been measured. *[Cite cold-start DTI evaluation work; ColdDTI (2025), GPS-DTI (2025).]*

**¶3 — What is known outside DTI.** That explanations degrade under distribution shift is
established elsewhere — for attribution methods in vision, for explainers of graph neural
networks, and for recommender explanations — and whether attention is an explanation at all
has been disputed since Jain & Wallace (2019), Serrano & Smith (2019) and Wiegreffe &
Pinter (2019). Two properties must be kept apart: *plausibility*, whether the explanation
looks right to an expert, and *faithfulness*, whether the model actually depends on what it
highlights. The dangerous case is an explanation that is plausible but not faithful, because
no expert can detect it by eye. *[Cite Gupta et al. (2025), Zhang et al. (2026) and Sun
(2025) from Related Work §2 — all arXiv preprints, check for published versions — and
DeYoung et al. (2020) for comprehensiveness and sufficiency.]*

**¶4 — The gap.** DTI's interpretability claims have not been tested against either
concern. The closest prior work, CS-DTA (2026), reports interpretability for its own model
under cold-start and includes a non-kinase validation; it does not measure faithfulness,
compare published models, test against a floor, or correct for multiple comparisons.
*[Keep this paragraph exactly as unflattering to our novelty as Related Work §1.4.]*

**¶5 — This work: an audit.** We audit the interpretability claims of published
attention-based DTI models — HyperAttentionDTI and MolTrans, with our own ColdSite-DTI held
to the same standard and DeepDTA as an accuracy anchor — across four levels of distribution
shift on DAVIS — random, unseen drug, unseen target and both — and replicate the audit of
the two published models on KIBA at the random and unseen-drug levels, the latter holding
out 422 drugs where DAVIS holds out 13. Each model is retrained on identical splits with its
authors' recipe, three seeds per cell, and the KIBA arm reuses every DAVIS decision
unchanged. We measure plausibility as precision@k against three ground
truths at different resolutions — UniProt's annotated residues, the 85-residue KLIFS ATP
pocket, and the residues a drug is measured to contact in its own co-crystal structure —
each read against its own chance level and achievable ceiling. We measure faithfulness as
the change in prediction when the attended residues are masked, against a random-masking
control, in the input space each model actually reads. A uniform attention map provides the
floor; a positive control establishes that the pipeline recognises a genuinely good
explanation, and at what resolution; nulls for position, amino-acid preference and drug
identity test the explanations that would otherwise be read as binding-site recovery;
significance is corrected once across each arm's whole family (sixteen cells on DAVIS, six
on KIBA); a panel of 60
non-kinase proteins stands in for a family stratification the benchmarks cannot support;
alternative attention readouts test whether a verdict belongs to the model or to the
reduction; and integrated gradients on the same checkpoints separate a poor explanation from
a model that never learned the site. **Figure 0** shows the design in one picture.

**¶6 — What we find.** One of sixteen cells supports the residue-level interpretability
claim after correction: HyperAttentionDTI on the random split, at 1.7× chance
(precision@10 0.034 [0.030–0.038] against 0.020), where it also beats a borrowed attention
map and an amino-acid-preserving permutation. **That cell does not replicate on KIBA**: none
of KIBA's six cells survives correction, and the same model at the same level is above
chance in one training seed of three — so across two datasets no residue-level attention
claim survives, and the one that did is seed-dependent. Under distribution shift no model's
attention marks the annotated residues better than chance, MolTrans is indistinguishable
from a uniform map at every DAVIS level (and in two KIBA seeds of three), and no model's
attention distinguishes a drug's own crystallographic contacts from another drug's in the
same pocket. What survives everywhere, and replicates, is coarser: attention that is
load-bearing — masking the attended residues moves the prediction more than masking random
ones, at every level of every model on both datasets — and that concentrates on the kinase
domain at 1.3–2.1× chance for the two models that clear the pocket floor at all, with
HyperAttentionDTI's attention still pointing into the ATP pocket on KIBA's 422 unseen
drugs. Three findings concern the measurement rather than the models, and we expect them
to matter beyond DTI: which residues
an attention map highlights is mostly a property of an unreported reduction choice
(alternative readouts share 2–12% of their top-ten residues with the published one, and one
choice moves a pocket-level verdict from below chance to 2.6× chance); a masking-based
faithfulness test inverts its own sign for a sub-word model, because k residues is not a
fixed-size intervention; and integrated gradients on the same checkpoints survive correction
in **seven of twelve cells where the attention survives in one of sixteen**, and on KIBA in
**three of four where the attention survives in none of six**, reaching
2.3–4.1× chance at levels where the attention is at chance — so for two of the three models
the attention under-reports a binding site the model does represent, and under-reports it
worst under the shift where interpretability is supposed to earn its keep. The third model
is the control for that claim: its gradient matches its attention and both sit at the
metric's floor, so its failure is the model rather than the report — a distinction no
attention measurement can draw. Along the way the
benchmark itself required correction: DAVIS's protein file gives every mutant its wild-type
sequence, and retraining without that leak accounts for 0.019 of cold-target's 0.038
apparent difficulty.

**¶7 — Contributions.**
1. **An audit, not a model.** Three attention-based DTI models (two published, one ours)
   and a no-attention accuracy anchor, one measurement suite, four levels of distribution
   shift, three seeds, and family-wise error control applied once across the whole family
   rather than per model — with the published models' audit repeated on a second dataset,
   where the only cell that survived correction fails to replicate.
2. **A protocol other people can run, with its instruments validated rather than
   asserted.** Plausibility against a uniform-attention floor, three ground-truth
   resolutions, and a **positive control** that reports at what dose a real signal would
   have been visible — so a null is distinguishable from a weak test, which is the
   objection every negative interpretability result attracts. Faithfulness against random
   masking, size-matched in the space each model reads. Nulls for position, amino-acid
   preference and drug identity. Every instrument is released with the code that produced
   the numbers, and each was checked on planted cases before being trusted on real ones.
3. **Three findings about interpretability measurement itself, applicable outside DTI.**
   Attention-based plausibility is largely a property of an unreported readout choice
   (2–12% top-ten overlap between defensible readouts; one choice moves a pocket-level
   verdict from below chance to 2.6× chance). Masking-based faithfulness does not transfer
   across tokenisations, and inverts its own sign for a sub-word model. And the gradient of
   the same weights recovers the site the attention misses (7 of 12 DAVIS cells against 1
   of 16; 3 of 4 on KIBA against 0 of 6), so a weak attention map often indicts the report
   rather than the model — with a third model as the control in which gradient and
   attention agree at the floor.
4. **A check the field can apply before publishing, and a published claim that fails it.**
   Whether an explanation *can* depend on the drug is a property of the computation graph,
   not an empirical question. A 2025 *Nature Communications* model presents per-residue
   attention for four drug–target complexes; that map is computed from the protein
   embedding alone, so for a fixed protein every drug yields the same figure (Results §7e). We
   propose the check as routine: **state which inputs the explanation is a function of.**
5. **A data-quality audit of DAVIS as the field uses it** — sequence-identical mutants,
   pocketless sequences, the leak quantified by retraining (0.019 of cold-target's 0.038
   apparent difficulty) — and a kinase-family confound shown to be untestable inside either
   standard benchmark, with a 60-protein transfer panel built because of it.
6. **Our own model audited on the same terms**, with its results reported whether or not
   they flatter it: ColdSite-DTI's attention is load-bearing everywhere, finely plausible
   nowhere, and its one above-chance result off the training family traced to an
   amino-acid preference rather than to knowledge of binding. It is not one of the
   published models whose claims this paper is about; it is the model we can open, which
   is why the readout and drug-dependence comparisons that need a model's internals use it.
