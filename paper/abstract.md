# Abstract (draft)

Drafted 2026-09-16, after the KIBA replication (Results §8). Every number is in
`paper/results.md` with its source file. Two versions of the same content, because the
venue is not chosen: **A** is unstructured with a Key Points box (*Briefings in
Bioinformatics* style), **B** is structured (*Bioinformatics* style). Both are ~250 words;
check each journal's current author guidelines for the limit and the required headings
before submission.

---

## A. Unstructured (Briefings in Bioinformatics)

Attention-based drug–target interaction (DTI) models present attention over the protein
as evidence of where the drug binds; we audit that claim.
Four attention-based models (three published, including DrugBAN, 2023; one ours) and a
no-attention anchor were retrained on identical splits at four levels of shift — random,
unseen drug, unseen target, both — three seeds per cell, on DAVIS (60 cells) and at two
levels on KIBA (24 cells). Plausibility was scored against UniProt residues, the KLIFS
ATP pocket and crystallographic contacts, each against chance, a uniform floor, a positive
control and positional, residue-type and drug-identity nulls; faithfulness by size-matched masking in the space each model reads. One of
twenty DAVIS cells supported the residue-level claim after Holm correction (HyperAttentionDTI,
random split, 1.7× chance) and it did not replicate on KIBA, where
none of eight survived. In twelve of twenty-two three-seed cells the seeds disagree about
their own verdict; the surviving cell is the only one where all three agree. The 2023 model,
the only one whose map changes with the drug, is at chance against the pocket at every
level. What replicated was coarser: the older models' top-ten attention
is load-bearing everywhere, and HyperAttentionDTI's localises the ATP pocket at 1.3–1.5×
chance on 422 unseen drugs. Three measurement findings generalise: plausibility depends on
an unreported readout choice (2–12% top-ten overlap), masking faithfulness does not
transfer across tokenisations, and integrated gradients on the same weights survive
correction where attention does not (7/12 DAVIS cells against 1/20; 3/4 on KIBA against 0/8).

*(248 words)*

### Key Points

* Published DTI interpretability claims are tested, not assumed: four attention-based
  models (including DrugBAN, 2023), four levels of distribution shift, two benchmarks,
  three seeds, and one family-wise correction per arm.
* Residue-level binding-site recovery survives in 1 of 20 DAVIS cells and in none of 8 on
  KIBA. Across all 22 cells with three seeds, the seeds disagree about their own verdict
  in 12, and in 21 the spread across seeds exceeds the cell's distance from chance — so a
  single-seed attention figure, the field's norm, cannot support the claim it illustrates.
  The one cell whose seeds all agree is the one cell that survives correction.
* The older models' attention is faithful (its top ten residues are load-bearing in every
  cell of both datasets, though not its top fifty under shift)
  and, for HyperAttentionDTI, coarsely plausible (ATP pocket at 1.3–1.5× chance, including
  on 422 unseen drugs) while missing the annotated residues. The 2023 model's map is the
  only one that changes with the drug, and it is at chance against residues and pocket
  alike — faithfulness, plausibility and drug-dependence must be reported separately.
* Which residues an attention map highlights is mostly a property of an undocumented
  readout choice, and one such choice moves a pocket-level verdict from 2.6× chance to
  below chance.
* Integrated gradients on the same checkpoints recover the site where attention does not
  (7 of 12 DAVIS cells against 1 of 20, 3 of 4 on KIBA against 0 of 8; 1.9× chance on
  unseen drugs where the attention is at chance), so a weak attention map often indicts the report rather than the model.

---

## B. Structured (Bioinformatics)

**Motivation:** Attention-based DTI models present attention over the protein as evidence
of where the drug binds, usually validated on a random split, by inspection, from one
training run. Whether such claims hold under the distribution shift that motivates
the models — an unseen drug, an unseen target, or both — and whether they replicate, has
not been measured.

**Results:** Four attention-based models (three published, including DrugBAN, 2023; one
ours) and an accuracy anchor were retrained on identical splits at four shift
levels, three seeds per cell, on DAVIS (60 cells) and at two levels on KIBA (24 cells).
Plausibility was scored against UniProt residues, the KLIFS ATP pocket and
crystallographic contacts, each against chance, a uniform floor, a positive control and
positional, residue-type and drug-identity nulls; faithfulness by size-matched masking in
each model's input space. One of twenty DAVIS
cells supported the residue-level claim after Holm correction (1.7× chance, random split)
and did not replicate on KIBA (0 of 8). The older models' top-ten attention was load-bearing in
every cell, and HyperAttentionDTI's localised the ATP pocket at 1.3–1.5× chance with 422
held-out drugs; DrugBAN's, the only map that varied with the drug, was at chance against
residues and pocket at every level. Attention plausibility proved largely a property of
an unreported readout choice (2–12% top-ten overlap), masking faithfulness did not
transfer across tokenisations, and integrated gradients on the same weights survived
correction where attention did not (7 of 12 DAVIS cells against 1 of 20; 3 of 4 on KIBA
against 0 of 8).

**Availability:** Code, splits, ground truth and all per-cell outputs: *[repository URL]*.

*(250 words, excluding headings and Availability)*

---

## Notes for finalising

1. **Numbers to re-check when the draft is frozen** (all section numbers are Results'):
   1.7× chance (§5),
   1.3–1.5× pocket on KIBA (§8.2: 0.199–0.220 against 0.151), 2–12% readout overlap (§7b),
   7 of 12 IG cells (§7c), cell counts 60 DAVIS / 24 KIBA (§1, §8, §9), 1 of 20 and 0 of 8
   Holm families (§5, §8.1), 12 of 22 seed disagreement (§7f), DrugBAN's k = 50 faithfulness
   sensitivity (§9) before 'load-bearing' is used of the older models only.
2. **Both versions state the non-replication explicitly.** Introduction's closing note
   requires it, and it is the result a reader most needs from the abstract.
3. **ColdSite-DTI is named only as "one ours"** — the audit is about the published claims,
   and our model's role (DAVIS and KIBA since 2026-09-19) belongs in Methods and Limitations, not in
   250 words.
4. **Not in the abstract, deliberately:** the DAVIS sequence-leakage audit (a benchmark
   finding, strong in the paper but a distraction here), the non-kinase transfer panel, and
   the equivalent-dose reading. Add the leakage line only if a reviewer asks for it or the
   venue wants the benchmark contribution foregrounded.
5. **Title suggestion, for the same framing:** "Attention in drug–target interaction models
   is faithful, coarse, and seed-dependent: an audit across distribution shift and two
   benchmarks." With DrugBAN in, "faithful" holds only for the three older models; if the
   k = 50 sensitivity leaves DrugBAN not load-bearing, prefer "…is seed-dependent and rarely
   points at binding residues: an audit of four models across distribution shift".
