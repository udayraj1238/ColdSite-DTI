# ColdSite-DTI — Project Report

**Mahim Agarwal** · Supervisor: **Dr. Chandra Mohan Dasari**
Report date: 20 September 2026 · Work period: 31 July – 20 September 2026

*Companion document: `draft_for_supervisor.docx` — the condensed paper draft with the
figures. This report covers the project as a whole: what was built, what was run, what went
wrong, how we know the numbers are right, and what is left.*

---

## 1. The project in one page

**The question.** Papers on drug–target interaction (DTI) prediction routinely show a plot
of the model's attention over a protein and conclude that the model has found the binding
site. That claim is usually made on a random split, from one training run, by eye, and
without a chance level. **Does it survive a realistic evaluation?**

**What we did.** We retrained four attention-based DTI models — three published (MolTrans
2021, HyperAttentionDTI 2022, DrugBAN 2023) and our own ColdSite-DTI — plus a no-attention
accuracy anchor (DeepDTA), on identical data splits at four levels of difficulty, three
times each, on two benchmarks. We then scored their explanations against experimentally
annotated binding sites, with a chance level, a ceiling, a uniform-attention floor, a
positive control and several nulls, and applied one statistical correction across the whole
family of tests.

**What we found.** Of twenty DAVIS cells, **one** supports the residue-level claim after
correction, and it does not replicate on KIBA. In twelve of twenty-two cells the three
training seeds disagree about their own verdict. What does survive is coarser and worth
reporting: the attention is genuinely used by the model, and it concentrates on the binding
*pocket region* rather than on the annotated residues. Three findings about the measurement
itself generalise beyond DTI (§5).

**Status.** All training and analysis are complete: 84 trained cells, both audits, the
replication, the controls, the figures. The paper's sections are drafted. What remains is
editorial — condensing to a venue's length, the citation style, and a DOI for the code
release. Nothing in the scientific plan is outstanding.

---

## 2. Why an audit, and why this is not a model paper

The easy version of this project would have been "our model gets a better AUROC". We
deliberately did not do that, for three reasons:

1. **The gap is real and unfilled.** The NLP community established in 2019 that attention
   weights need not be explanations. DTI papers continued to present attention as binding-site
   evidence, and nobody had tested those claims under the distribution shift the models are
   built for.
2. **An audit can be wrong in a way a model paper cannot.** If we report a null, the first
   objection is "your test is too weak". That forced the protocol to carry its own evidence:
   a positive control that says at what strength a real signal would have been detected, a
   uniform-attention floor, and nulls that a spurious signal would pass.
3. **Our own model is a subject, not a hero.** ColdSite-DTI is audited on exactly the same
   terms, and the report includes results that are unfavourable to it (§5, §6).

---

## 3. What was built

| Component | What it does | Size |
|---|---|---|
| Data pipeline | DAVIS and KIBA loading, four split levels per dataset, binary thresholds, a non-kinase transfer panel (60 BindingDB proteins), an antiviral subset | `src/data/` |
| Ground truth | UniProt binding-site residues re-numbered onto each dataset's own sequences; the 85-residue KLIFS ATP pocket; per-pair crystallographic contacts from KLIFS interaction fingerprints | 3 independent ground truths |
| Trainers | One per model, each using its authors' own recipe and tokeniser from the vendored repository, with only data loading replaced; epoch-level resume so an 11-hour cloud session can continue | 5 models |
| Model adapters | A uniform interface (`predict`, `explain`) over five very different architectures, so every metric runs unchanged on each | `src/evaluation/` |
| Analysis | Plausibility ladders, faithfulness, the audit grid with Holm correction, positional and residue-identity nulls, the non-kinase control, the positive control, readout variants, integrated gradients, seed agreement, drug-dependence, clean accuracy | 33 modules |
| Orchestration | `run_all` — one command that runs the whole analysis for a dataset and writes a summary; resumable | 1 entry point |
| Cloud notebooks | Kaggle notebooks for each training grid, with self-stop before the platform's time limit, restore-from-previous-commit, and pre-flight checks that refuse to start if the splits do not match the record | 20 notebooks |
| Tests | Unit and integration tests, including tests on planted cases where the right answer is known | **938 tests** |

**Totals:** 67 Python modules, ~17,300 lines, 200 commits over eight weeks.

---

## 4. What was run

**Training — 84 cells, each a full training run with its own checkpoint and test pass:**

| dataset | models | levels | seeds | cells |
|---|---|---|---|---|
| DAVIS | DeepDTA, ColdSite-DTI, HyperAttentionDTI, MolTrans | random, cold-drug, cold-target, cold-pair | 3 | 48 |
| DAVIS | DrugBAN (added Sept 18) | all four | 3 | 12 |
| KIBA | DeepDTA, HyperAttentionDTI, MolTrans | random, cold-drug | 3 | 18 |
| KIBA | ColdSite-DTI (added Sept 19) | random, cold-drug | 3 | 6 |

All training ran on free cloud GPUs (Kaggle T4s), because the work does not fit on a laptop:
the KIBA arm alone took about **101 GPU-hours** across four accounts. Checkpoints total 4.6 GB
and are backed up outside the machine.

**Analysis** runs locally on CPU: plausibility ladders against two ground truths, faithfulness,
both audits, the nulls and controls, integrated gradients, readout variants, and the figures.

---

## 5. What we found

Full numbers, tables and figures are in the companion draft. In brief:

1. **One cell of twenty survives correction** (HyperAttentionDTI, random split, 1.7× chance),
   and it beats every null. It **does not replicate on KIBA**, where none of eight survives.
2. **Attention is used but coarse.** Masking the attended residues moves the prediction more
   than masking random ones in every cell of both datasets (for the three 2021–2022 models),
   and attention concentrates on the ATP pocket at 1.3–2.1× chance — while missing the
   annotated residues inside it.
3. **A single training run cannot support the claim.** In 12 of 22 cells the three seeds
   disagree about their own verdict; in 21 of 22 the seed spread exceeds the cell's distance
   from chance.
4. **The verdict can belong to the readout.** Alternative, equally defensible ways of reducing
   the same attention tensor share only 2–12% of their top-ten residues, and one choice moves a
   pocket-level verdict from 2.6× chance to below chance.
5. **The gradient finds what the attention misses.** Integrated gradients on the *same*
   checkpoints survive correction in 7 of 12 DAVIS cells against 1 of 20 for attention. A weak
   attention map often indicts the report, not the model.
6. **The newest model is the clearest failure.** DrugBAN (2023) is at chance against both
   ground truths in all 12 cells and its attention is not load-bearing — while being the only
   model whose map genuinely changes with the drug.
7. **A check the field can adopt before publishing:** state which inputs the explanation is a
   function of. A 2025 *Nature Communications* model's residue attention cannot depend on the
   drug at all — visible in its source code, no experiment needed.
8. **A benchmark finding:** in the DAVIS protein file this literature uses, all 54 mutant
   targets carry the wild-type sequence, so "unseen" targets are partly seen. Leakage inflated
   cold-target accuracy for every model.

---

## 6. Problems found during the work, and how each was handled

This section is the honest core of the project: most of the effort went into catching
mistakes that would have produced confident wrong numbers.

| # | Problem | Consequence if missed | Resolution |
|---|---|---|---|
| 1 | **DAVIS mutants carry wild-type sequences** (found while building the KLIFS ground truth) | "Unseen" targets are seen; cold-target accuracy inflated for every model | Re-scored all cold cells on genuinely unseen targets; explanation metrics exclude seen-by-sequence targets; reported as a benchmark finding |
| 2 | **MolTrans seed bug** — its vendored code reseeds PyTorch on import, so three "seeds" trained identically | Three copies of one run reported as a seed spread | Fixed; seeds 2 and 3 retrained; seed 1 verified bit-for-bit identical, so it stands |
| 3 | **Masking arms not the same size** for a sub-word model | Its faithfulness delta was negative in 11 of 12 cells — would have read as "anti-faithful attention" | Measured the asymmetry (48% vs 95% of tokens changed), built a size-matched control, re-ran in token space; sign reverses in all 12 |
| 4 | **We were scoring the wrong MolTrans map** | The audit's MolTrans verdict would not have been about the map its paper actually shows | Read their paper's full text, implemented their interaction map as a second readout, scored both; at chance either way, so the verdict holds |
| 5 | **Permutation resolution capped the p-values** at 1/501 | The one surviving result looked marginal (0.0020 vs a 0.0025 threshold) | Re-ran both audits at 10,000 permutations: p = 0.0001, a 25-fold margin, no verdict changed |
| 6 | **Two vendored repositories both define `models.py`** | The audit crashed when it loaded both models in one process | Isolated module loading, with a test that loads both in either order |
| 7 | **`run_all` skips finished outputs** — running it on a partial grid would silently freeze partial results | A level trained later would never be scored, with no error | Rule and guard: probe partial grids into a scratch folder only |
| 10 | **A control crashed on one molecule of 16,033** — DrugBAN's loader caps a drug at 290 atoms | Every DrugBAN control run aborted; the easy response would have been to drop the panel | The collector skips such a row, reports it, and refuses to continue if more than 1% of rows are lost |
| 8 | **Faithfulness depends on the dose** (k = 10 vs k = 50) | "Load-bearing at every level" was broader than the evidence | Ran the sensitivity; the claim is now stated at its pre-specified k, with the k = 50 result reported |
| 9 | **A plausible but wrong interpretation** — we had attributed leakage sensitivity to model size | An unsupported causal story in the paper | DrugBAN contradicted it; the claim was removed rather than defended |

---

## 7. How we know the numbers are right

- **938 automated tests**, including tests on *planted* cases — synthetic data where the
  correct answer is known — for every metric that produces a headline number.
- **A positive control for the metric itself**: synthetic explanations of known quality,
  scored by the audit's own code on the real splits. It detects a signal as weak as 2% of
  true sites, so our nulls are nulls and not weak tests. It also *fails* on the pre-correction
  ground truth, which proves it can catch that class of bug.
- **Exact reproduction across machines.** The DAVIS audit was first computed on cloud GPUs and
  later recomputed on the laptop with a fourth model added: all sixteen original p-values
  reproduced to the last digit. The KIBA audit reproduced the same way.
- **Every number traces to a committed file.** The paper's sections cite the file each number
  comes from, and the figure captions name their sources; no number is typed by hand into a
  figure.
- **Protocol decisions are recorded, not implicit** — including the ones we later revisited
  (§6, item 4).

---

## 8. Where the project stands

**Done:** data pipeline, three ground truths, five trainers, five adapters, the full analysis
suite, 84 trained cells, both audits with correction, the KIBA replication, all controls and
nulls, integrated gradients, readout variants, the seed analysis, four figures, and drafts of
every paper section (~27,600 words).

**Completed since this report was drafted:** DrugBAN's last two controls. Its own data
loader refuses one ligand in the non-kinase panel (322 atoms against its 290-atom cap), and
that single molecule out of 16,033 was aborting the run; the collector now drops such a row,
names it, and fails the cell if more than 1% of rows would be lost. DrugBAN shows no transfer
off kinases, and its one above-chance seed-cell beats all four positional nulls while the
other two seeds of that cell beat none — the seed-dependence finding reaching even the model
that is at chance everywhere.

**Remaining, editorial:**
1. **Condense** the ~27,600 words of section drafts into a main text (a journal allows roughly
   5,000–8,000) plus a supplement. This is the largest remaining task and depends on the venue.
2. **Citation style**, which follows the venue.
3. **A DOI for the code release** (Zenodo) and the repository URL in the paper.

**Optional, not required for any claim:** integrated gradients for DrugBAN, DrugBAN on KIBA,
readout variants on KIBA. Each is currently stated as a limitation, which is how papers
handle scope.

---

## 9. Questions for you

1. **Venue.** Undecided. Candidates: *Briefings in Bioinformatics* (publishes benchmarking and
   critical-assessment work, flexible length — our best fit), *Bioinformatics* (needs heavy
   condensing), ISMB (most competitive, tightest limit). The choice fixes the length target
   and the citation style.
2. **Framing.** An audit of published claims (current), or a benchmark-and-protocol
   contribution with the audit as the demonstration?
3. **Scope.** Is the current evidence sufficient, or would you want DrugBAN on KIBA as well
   (about 20–30 GPU-hours) before submission?
4. **Our own model's role.** ColdSite-DTI is currently a subject audited on the same terms,
   named only in Methods and Limitations. Should it stay that way?

---

## Appendix — where everything lives

| What | Where |
|---|---|
| Paper sections (full) | `paper/introduction.md`, `methods_data_and_evaluation.md`, `methods_track_b.md`, `results.md`, `discussion.md`, `limitations.md`, `abstract.md`, `references.md` |
| Condensed draft + figures | `paper/DRAFT_FOR_SUPERVISOR.md` → `.docx` / `.html` |
| Figures | `results/figures/` (PDF and PNG) with `CAPTIONS.md` |
| Analysis outputs | `results/analysis_davis_policyA*/`, `results/analysis_kiba_policyA*/` |
| Trained checkpoints | outside the repository (4.6 GB), backed up to Drive |
| Code | `src/data/`, `src/model/`, `src/evaluation/`; tests in `tests/` |
| Training notebooks | `notebooks/` |
| Verified citations | `paper/references.md` — every entry checked against Crossref or arXiv |
| Project status and rules | `CLAUDE.md` |

**Reproducing the analysis:** with the checkpoints in place, one command per dataset —
`python -m src.evaluation.run_all --dataset davis --checkpoint-dir <checkpoints>` — runs
faithfulness, the ladders, the audit, the controls and the summary, and is resumable.
