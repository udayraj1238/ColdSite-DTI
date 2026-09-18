# Abstract (draft)

Drafted 2026-09-16, after the KIBA replication (Results §8). Every number is in
`paper/results.md` with its source file. Two versions of the same content, because the
venue is not chosen: **A** is unstructured with a Key Points box (*Briefings in
Bioinformatics* style), **B** is structured (*Bioinformatics* style). Both are ~250 words;
check each journal's current author guidelines for the limit and the required headings
before submission.

---

## A. Unstructured (Briefings in Bioinformatics)

Attention-based drug–target interaction (DTI) models routinely present attention over the
protein as evidence of where the drug binds; we audit that claim rather than make it.
Three attention-based models (two published, one ours) and a no-attention anchor were
retrained on identical splits at four levels of shift — random, unseen drug, unseen target,
both — three seeds per cell, on DAVIS (48 cells) and, for the published models at two
levels, on KIBA (18 cells). Plausibility was scored against UniProt's annotated
residues, the KLIFS ATP pocket and per-pair crystallographic contacts — each against its own
chance level, a uniform floor, a positive control and nulls for position,
amino-acid preference and drug identity — and faithfulness by size-matched masking in the
space each model reads. One of sixteen DAVIS cells
supported the residue-level claim after Holm correction — the best-generalising model at
random, 1.7× chance — and it did not replicate on KIBA, where none survived.
Across the sixteen three-seed cells the seeds disagree about their own verdict in eleven,
and the surviving cell is the only one where all three agree. What replicated was coarser: attention that is
load-bearing everywhere and localises the ATP pocket at 1.3–1.5× chance, on 422 unseen drugs. Three measurement findings generalise beyond DTI: attention plausibility is largely a property of an
unreported readout choice (2–12% top-ten overlap), masking faithfulness does not transfer
across tokenisations, and integrated gradients on the same weights survive correction where
attention does not (7/12 DAVIS cells against 1/16; 3/4 on KIBA against 0/6).

*(250 words)*

### Key Points

* Published DTI interpretability claims are tested, not assumed: three attention-based
  models, four levels of distribution shift, two benchmarks, three seeds, and one
  family-wise correction per arm.
* Residue-level binding-site recovery survives in 1 of 16 DAVIS cells and in none of 6 on
  KIBA. Across all 16 cells with three seeds, the seeds disagree about their own verdict
  in 11, and in 15 the spread across seeds exceeds the cell's distance from chance — so a
  single-seed attention figure, the field's norm, cannot support the claim it illustrates.
  The one cell whose seeds all agree is the one cell that survives correction.
* Attention is faithful (load-bearing in every cell of both datasets) and coarsely
  plausible (ATP pocket at 1.3–1.5× chance, including on 422 unseen drugs) while missing
  the annotated residues — faithfulness and plausibility must be reported separately.
* Which residues an attention map highlights is mostly a property of an undocumented
  readout choice, and one such choice moves a pocket-level verdict from 2.6× chance to
  below chance.
* Integrated gradients on the same checkpoints recover the site where attention does not
  (7 of 12 DAVIS cells, 3 of 4 on KIBA; 1.9× chance on unseen drugs where the attention is
  at chance), so a weak attention map often indicts the report rather than the model.

---

## B. Structured (Bioinformatics)

**Motivation:** Attention-based DTI models present attention over the protein as evidence
of where the drug binds, usually validated on a random split, by inspection, and from a
single training run. Whether such claims hold under the distribution shift that motivates
the models — an unseen drug, an unseen target, or both — and whether they replicate, has
not been measured.

**Results:** Three attention-based models (two published, one ours) plus a no-attention
accuracy anchor were retrained on identical splits at four shift levels, three seeds per
cell, on DAVIS (48 cells) and, for the published models at two levels, on KIBA (18 cells).
Plausibility was scored against UniProt residues, the KLIFS ATP pocket and per-pair
crystallographic contacts, each against chance, a uniform-attention floor, a validated
positive control and nulls for position, amino-acid preference and drug identity;
faithfulness by size-matched masking in each model's own input space. One of sixteen DAVIS
cells supported the residue-level claim after Holm correction (1.7× chance, random split)
and it failed to replicate on KIBA, where the same cell was above chance in one seed of
three. Attention was load-bearing in every cell and localised the ATP pocket at 1.3–1.5×
chance, including with 422 held-out drugs. Attention plausibility proved largely a property
of an unreported readout choice (2–12% top-ten overlap), masking faithfulness did not
transfer across tokenisations, and integrated gradients on the same weights survived
correction where attention did not (7 of 12 DAVIS cells against 1 of 16; 3 of 4 on KIBA
against 0 of 6).

**Availability:** Code, splits, ground truth and all per-cell outputs: *[repository URL]*.

*(227 words, excluding headings and Availability)*

---

## Notes for finalising

1. **Numbers to re-check when the draft is frozen** (all section numbers are Results'):
   1.7× chance (§5),
   1.3–1.5× pocket on KIBA (§8.2: 0.199–0.220 against 0.151), 2–12% readout overlap (§7b),
   7 of 12 IG cells (§7c), cell counts 48 DAVIS / 18 KIBA (§1, §8).
2. **Both versions state the non-replication explicitly.** Introduction's closing note
   requires it, and it is the result a reader most needs from the abstract.
3. **ColdSite-DTI is named only as "one ours"** — the audit is about the published claims,
   and our model's role (audited on DAVIS only) belongs in Methods and Limitations, not in
   250 words.
4. **Not in the abstract, deliberately:** the DAVIS sequence-leakage audit (a benchmark
   finding, strong in the paper but a distraction here), the non-kinase transfer panel, and
   the equivalent-dose reading. Add the leakage line only if a reviewer asks for it or the
   venue wants the benchmark contribution foregrounded.
5. **Title suggestion, for the same framing:** "Attention in drug–target interaction models
   is faithful, coarse, and seed-dependent: an audit across distribution shift and two
   benchmarks."
