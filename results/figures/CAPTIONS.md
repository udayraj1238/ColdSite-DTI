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
precision@10 for every model, level and dataset against two ground truths. Bars are means
over three training seeds, open circles are the seeds themselves, and the dashed segment
over each group is that level's chance level (it depends on the level's proteins, their
lengths and site counts, so it is not one line across a panel). Top row: UniProt's
annotated residues, where chance is 0.019–0.020 on DAVIS and 0.023 on KIBA — no model's
bar stands clear of it on either dataset. Bottom row: the 85-residue KLIFS ATP pocket,
where chance is 0.136–0.143 on DAVIS and 0.151 on KIBA — the audited models sit above it.
Read together, the figure is the paper's claim: attention finds the pocket region and not
the annotated residues. n is the number of proteins per cell, one test pair each.
*Sources:* `results/analysis_{davis,kiba}_policyA{,_klifs}/ladder_*.json`.

**Figure 2 — `fig2_seeds`. One cell, three training seeds.**
Every UniProt cell of the two published models on both datasets, as three dots with the
cell's chance level as a dashed tick. The figure exists because a ± hides what matters:
HyperAttentionDTI's KIBA warm cell reads 0.017, 0.022 and 0.053 against a chance of 0.023,
and MolTrans's cells straddle chance the same way. A single-seed attention figure — the
norm in this literature — could report either side of the verdict.
*Sources:* the same UniProt ladders as Figure 1.

**Figure 3 — `fig3_attention_vs_ig`. The same checkpoints, read two ways.**
Attention (red) beside integrated gradients (blue) on identical trained weights, for every
cell that has both. Top row DAVIS (three models, four levels), bottom row KIBA (the two
audited models at the two trained levels, so it is deliberately shorter). Left column
UniProt's annotated residues, right column the KLIFS pocket; dots are seeds, the dashed
segment is that group's chance level. A secondary analysis: the gradient was added after
the attention results were seen, and each dataset's family is Holm-corrected within itself
(7 of 12 DAVIS cells, 3 of 4 KIBA cells).
*Cited in:* Results §7c, §8.4.

**Figure 4 — `fig4_faithfulness`. Is the attention load-bearing?**
Comprehensiveness minus a size-matched random-masking control; above zero means masking
what the explanation points at moves the prediction more than masking arbitrary input of
the same size. Left and middle: residues masked, DAVIS and KIBA. Right: MolTrans, where the
intervention is a **token**, the unit its model reads. The panels are separate because the
units differ — a token delta and a residue delta are not comparable numbers (Results §5b) —
and every cell of both datasets is positive.
*Sources:* `faithfulness_*.json` and `token_faithfulness_moltrans_*.json` in the two
policyA folders.

---

## Notes

* **What is deliberately not drawn:** the non-kinase transfer panel (60 proteins from
  another population — it belongs in a table, not beside these bars), the positional and
  residue nulls (four arms per cell would swamp the figure; they are quoted in the text
  where a cell's claim depends on them), and the readout-variant table of §7b.
* **Figure 1 is the one a reader will judge the paper by.** If only one figure survives
  review, keep it and fold Figure 2's message into its caption.
* **Colour:** ColdSite-DTI blue, HyperAttentionDTI red, MolTrans green, throughout all
  four figures. Distinguishable in greyscale by position, not by colour alone — check
  before submission if the journal prints mono.
