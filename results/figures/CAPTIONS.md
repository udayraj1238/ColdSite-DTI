# Figures — captions and what each one is allowed to claim

Built by `python -m src.evaluation.paper_figures --out-dir results/figures` from the
committed analysis files; no number is typed by hand. PDF for the manuscript, PNG for
drafts. Rebuild after any re-analysis.

---

**Figure 0 — `fig0_design`. The audit in one picture.**
Datasets and their split levels, the models retrained on identical splits with three seeds,
the three ways each explanation is read (attention as published, integrated gradients on the
same weights, a uniform map as the floor), the two measurement axes with their ground truths
and nulls, and where the multiplicity correction is applied. Hand-laid rather than
data-driven: it describes the protocol, so its counts must be updated with Methods.
*Cited in:* Methods §1, Introduction ¶5.

**Figure 1 — `fig1_plausibility`. Does attention mark the binding site?**
precision@10 for every model, level and dataset against two ground truths — four models on
DAVIS, three on KIBA, since DrugBAN is DAVIS-only and ColdSite-DTI joined KIBA on
2026-09-19. Bars are means
over three training seeds, open circles are the seeds themselves, and the dashed segment
over each group is that level's chance level (it depends on the level's proteins, their
lengths and site counts, so it is not one line across a panel). Top row: UniProt's
annotated residues, where chance is 0.019–0.020 on DAVIS and 0.023 on KIBA — no model's
bar stands clear of it on either dataset. Bottom row: the 85-residue KLIFS ATP pocket,
where chance is 0.136–0.143 on DAVIS and 0.151 on KIBA — the 2021–2022 models sit above it
on DAVIS, DrugBAN sits on it at every level, and on KIBA only HyperAttentionDTI clears it
in every seed.
Read together, the figure is the paper's claim: attention finds the pocket region and not
the annotated residues. n is the number of proteins per cell, one test pair each.
*Sources:* `results/analysis_{davis,kiba}_policyA{,_klifs}/ladder_*.json`.

**Figure 2 — `fig2_seeds`. One cell, three training seeds.**
All twenty-two three-seed UniProt cells — four models on DAVIS, three on KIBA — as three
dots each with the cell's chance level as a dashed tick. It is Table R10 as a picture. The figure exists because a ± hides what matters:
HyperAttentionDTI's KIBA warm cell reads 0.017, 0.022 and 0.053 against a chance of 0.023,
and MolTrans's cells straddle chance the same way. A single-seed attention figure — the
norm in this literature — could report either side of the verdict.
*Sources:* the same UniProt ladders as Figure 1.

**Figure 3 — `fig3_attention_vs_ig`. The same checkpoints, read two ways.**
Attention (red) beside integrated gradients (blue) on identical trained weights, for every
cell that has both. Top row DAVIS (the three models with an IG arm, four levels), bottom row KIBA (two models
at the two trained levels, so it is deliberately shorter). DrugBAN is absent by design: its
drug side is a graph and it has no IG adapter, which Results §9 and Limitations state. Left column
UniProt's annotated residues, right column the KLIFS pocket; dots are seeds, the dashed
segment is that group's chance level. A secondary analysis: the gradient was added after
the attention results were seen, and each dataset's family is Holm-corrected within itself
(7 of 12 DAVIS cells, 3 of 4 KIBA cells).
*Cited in:* Results §7c, §8.4.

**Figure 4 — `fig4_faithfulness`. Is the attention load-bearing?**
Comprehensiveness minus a size-matched random-masking control; above zero means masking
what the explanation points at moves the prediction more than masking arbitrary input of
the same size. Left and middle: residues masked, DAVIS (ColdSite-DTI, HyperAttentionDTI, DrugBAN) and
KIBA (ColdSite-DTI, HyperAttentionDTI). DrugBAN's bars are flat at this scale — its deltas
are within ±0.013 of zero while ColdSite-DTI's reach 1.2 — so the panel says so in text
rather than leaving a legend entry a reader cannot find. Right: MolTrans, where the
intervention is a **token**, the unit its model reads. The panels are separate because the
units differ — a token delta and a residue delta are not comparable numbers (Results §5b) —
and every cell of both datasets is positive for the three 2021–2022 models. All of this is
at k = 10, the pre-specified dose; at k = 50 HyperAttentionDTI's margin survives only at
random (Results §5).
*Sources:* `faithfulness_*.json` and `token_faithfulness_moltrans_*.json` in the two
policyA folders.

---

## Notes

* **What is deliberately not drawn:** the non-kinase transfer panel (60 proteins from
  another population — it belongs in a table, not beside these bars), the positional and
  residue nulls (four arms per cell would swamp the figure; they are quoted in the text
  where a cell's claim depends on them), the readout-variant table of §7b — including
  MolTrans's interaction map, which is a row there rather than a bar here — and the
  drug-dependence check of §7e.
* **Figure 1 is the one a reader will judge the paper by.** If only one figure survives
  review, keep it and fold Figure 2's message into its caption.
* **Colour:** ColdSite-DTI blue, HyperAttentionDTI red, MolTrans green, DrugBAN purple,
  throughout all four figures. Distinguishable in greyscale by position, not by colour alone — check
  before submission if the journal prints mono.
