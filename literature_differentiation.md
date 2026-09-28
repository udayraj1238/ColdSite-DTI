# Related Work and Differentiation

Backbone for the paper's Related Work section, under the **audit** framing of
`docs/00_MASTER_PLAN_V2.md`. The question is no longer "how is ColdSite-DTI
different from these models" — ColdSite-DTI is one subject among several. The
question is:

> **Do published interpretability claims in DTI prediction survive realistic
> evaluation?**

Three bodies of work bear on that: the DTI papers that make interpretability
claims (§1), the evidence from outside DTI that explanations degrade under
distribution shift (§2), and the unresolved dispute about whether attention is
an explanation at all (§3). §4 states what is left for us to claim once all
three are taken seriously.

---

## 1. Interpretability claims in DTI prediction

### 1.1 DMFF-DTA (He et al., 2025, *npj Digital Medicine*)

Builds a binding-site-focused protein graph from AlphaFold2 contact maps, reports
generalization to completely unseen drugs and targets (over 8% better than prior
methods), and includes an interpretability analysis said to validate the model's
biological relevance. *(Corrected 2026-09-12: an earlier version of this paragraph said
its pipeline ran only on random splits; the abstract shows it does not. Whether its
interpretability analysis was run on the unseen-drug/target setting, and whether it
tests against binding sites statistically, must be checked in the full text.)*
**What we add:** explanation quality measured as a quantity at each of four difficulty
levels, for several published models, against a floor and with faithfulness — rather
than an interpretability analysis reported alongside a generalization result.

### 1.2 EviDTI (Zhao et al., 2025, *Nature Communications*)

Computes a binding-site hit ratio for high-attention residues *and* evaluates robustness
on novel, unseen drug–target pairs — but, as far as the abstract and tables show, as two
independent experiments. Interpretive fidelity is never expressed as a
function of generalization difficulty. **What we add:** the connection. The
two axes are measured on the same splits, the same proteins and the same
checkpoints, so a change in one can be read against the other.

*Read in full, with the released code, 2026-09-18.* Its Figure 6 shows per-residue
attention for four drug–target complexes and concludes that high-attention residues
coincide with the binding site. That attention is computed from the protein's ProtTrans
embedding alone — no drug tensor reaches it, and the model has no cross-attention — so for
a fixed protein it is the same map whatever the drug binds (Results §7e;
`results/evidti_code_audit.md`). This is stated as a structural fact about the released
model, not as a failure of its uncertainty quantification, which is its contribution and
which this audit does not test.

### 1.3 ColdDTI (Zhang et al., 2025, arXiv preprint)

Strong predictive accuracy under cold-start using a hierarchical attention mechanism over
multi-level protein structure, with no assessment of explanation quality in its abstract
*(a preprint at the time of writing — re-check for a published version)*. It
establishes that cold-start accuracy is achievable; it does not ask whether
the model reaches those predictions for defensible reasons. **What we add:**
that question, on the same difficulty spectrum.

### 1.4 CS-DTA (Jiang et al., 2026, *Frontiers in Chemistry*) — the closest prior work

**This paragraph is deliberately unflattering, and must stay that way.** A
reviewer who has read CS-DTA will notice any softening, and it costs more
credibility than the differentiation gains.

CS-DTA reports state-of-the-art performance across warm and strict cold-start
scenarios, runs interpretability analyses highlighting localized protein
regions with plausible binding relevance, **and includes non-kinase
validation** — the control arm we are only now constructing. On the surface,
it has already done what v1 of this project proposed to do.

The honest differentiation is narrow and specific:

| | CS-DTA | this work |
|---|---|---|
| models examined | one (its own) | several published models, ours included |
| interpretability reported | as a property demonstrated | as a quantity measured across difficulty |
| faithfulness | not measured | comprehensiveness / sufficiency against random-masking controls |
| explanation floor | none | uniform-attention control |
| multiple comparisons | not applicable | Holm–Bonferroni across the whole grid |
| family control | non-kinase validation included | non-kinase **transfer** panel, 60 targets |

So: they report interpretability under cold-start for one model; we measure it
as a function of split difficulty across several published models, with
faithfulness, a control floor, and family-wise error control. That is a real
difference, and it is a difference of *method*, not of claim. It is not a
claim to have discovered that cold-start hurts interpretability.

### 1.5 GPS-DTI (Xiong et al., 2025, *BMC Biology*)

Targets generalization to unseen drugs and targets with ESM-2 protein representations and
a drug–protein cross-attention module, and presents visualised cross-attention maps as
"interpretable insights into key molecular interactions". *(Corrected 2026-09-12: it
does make an interpretability claim — by inspection of attention maps.)* **What we add:**
the audit treats "high accuracy" and "attention points at the right residues" as separate
claims requiring separate, quantitative evidence at each difficulty level — the pairing
GPS-DTI supports by visualisation rather than measurement *(confirm it reports no
quantitative binding-site test)*.

### 1.6 KANPM-DTA (Rakib et al., 2026, *Briefings in Bioinformatics*)

Named in the master plan as venue evidence rather than as a differentiation target: a
2026 *Briefings in Bioinformatics* DTA model that itself lists interpretability and
generalization to unseen pairs among its aims. Relevant mainly for calibrating where this
work publishes.

---

## 2. Explanations degrade under distribution shift — established outside DTI

This is the section that keeps a reviewer from dismissing the paper as naive.
"Explanations get worse out of distribution" is **not a new finding**, and
claiming it as one would be the fastest possible rejection. It is established
in several separate literatures:

- **Vision attribution.** Subset-selection attribution methods that perform well
  in-distribution degrade severely out of distribution, with insertion and deletion
  scores dropping by up to 40% across curated in-/out-of-distribution pairs (Gupta,
  Prasad C & Ramakrishnan, 2025, arXiv:2512.08445 — the figure is in the full text, not
  the abstract, so cite it with its page).
- **Graph neural networks.** The Explanation-Generalization Score uses out-of-distribution
  generalization as the test of whether a GNN explanation captures causal structure
  (Zhang, Betala & Agarwal, 2026, arXiv:2602.07708) — explanation quality judged by what
  survives distribution shift.
- **Recommender systems.** CIRR targets faithful explanations under distribution shift,
  motivated by recommenders' degradation out of distribution (Sun, 2025,
  arXiv:2512.18683).
- ~~**Mechanistic interpretability.** Sparse-autoencoder faithfulness formalized as a
  geometric "faithfulness gap".~~ **Removed 2026-09-12: no source could be found that
  makes this claim.** The nearest, an audit finding that sparse-autoencoder features can be
  geometrically recovered yet causally inert (Bal, 2026, arXiv:2607.12166), concerns
  faithfulness in-distribution, not degradation under shift. Reinstate only with the
  original source.

**Our position against this literature.** We do not claim the phenomenon is
new. We claim that DTI's published interpretability results have not been
checked against it — that a field making biological claims from attention maps
supports them by inspection, on random splits, or beside a separate cold-start
accuracy result (§1), but not, to our knowledge, as a quantity measured across levels of
distribution shift, while deploying the models in precisely the cold-start regime where
the phenomenon is known to bite. *(Revised 2026-09-12: "almost exclusively on random
splits" overstated it — DMFF-DTA, EviDTI, GPS-DTI and CS-DTA all pair interpretability
with cold-start results.)* State this directly in the Intro. A reviewer who believes we
are unaware of this work will reject; one who sees we have positioned against
it deliberately will not.

> **Citation status (2026-09-12).** The three remaining claims are verified against
> their sources and listed in `paper/references.md`. All three are **arXiv preprints**:
> check each for a peer-reviewed version before submission, and prefer it. The 40%
> figure is in the full text of arXiv:2512.08445 (not its abstract). One claim was
> removed for lack of a source — see above.

---

## 3. Is attention an explanation at all?

The dispute is live and unresolved, and it is the single best justification
for this paper's design:

- **Jain & Wallace (2019), *Attention is not Explanation*** — attention
  weights correlate poorly with gradient-based importance, and adversarial
  attention distributions can leave predictions unchanged.
- **Serrano & Smith (2019), *Is Attention Interpretable?*** — erasing
  high-attention components often fails to change the decision, so high
  attention does not establish high influence.
- **Wiegreffe & Pinter (2019), *Attention is not not Explanation*** — pushes
  back: the claim depends on what "explanation" is taken to mean, and under
  reasonable definitions attention can be faithful.

**What that literature could not settle, and what this paper adds.** A reviewer will ask
whether "attention is not explanation" was established in 2019 and this is its application
to a new domain. The honest answer names what was missing there. Those results argue from
*internal* evidence — adversarial attention distributions, erasure, correlation with
gradients — because in sentiment or NLI there is no external fact about which token is the
true evidence. A protein has one: annotated binding residues, a structurally defined
pocket, and the contacts a specific ligand makes in a crystal. That external ground truth
is what turns "attention may not be explanation" into a measurable quantity with a chance
level, a ceiling, a positive control and a p-value, and it is what makes the two axes
separable rather than conflated: this audit finds explanations that are **faithful and not
plausible**, a cell the 2019 methodology cannot name because it has no notion of
plausibility to place against faithfulness.

Three further things have no precedent in that work, and each cost this paper a control
rather than an argument. (i) The verdict is measured **as a function of distribution
shift**, which is the setting these models are sold for and where no attention study of
either field reports interpretability. (ii) The **readout** between a network's tensor and
a per-residue weight is shown to carry most of the apparent signal (2-12% top-ten overlap
between defensible reductions) — a degree of freedom that does not arise for one attention
weight per token in NLP, and is undocumented in every DTI paper we read. (iii) Masking
faithfulness is shown **not to transfer across tokenisations**, inverting its own sign for
a sub-word model; the 2019 erasure results assume, correctly for their setting and
incorrectly here, that erasing k units is the same intervention in both arms.

So the debt is real and the differentiation is narrow: they asked whether attention is
explanation, without a way to check where the explanation should point. We ask whether it
survives *correction, shift, replication and its own readout* against a ground truth that
says where it should point — and answer for the models a reader of the DTI literature is
actually being asked to trust.

**Why this determines our methodology.** Plausibility and faithfulness are
independent properties:

| | faithful | not faithful |
|---|---|---|
| **plausible** | the good case | **a convincing lie** |
| **not plausible** | honest oddity | noise |

`precision@k` against UniProt binding sites measures the rows — does the
explanation *look* biologically sensible. `src/evaluation/faithfulness.py`
measures the columns — does masking the highlighted residues actually change
the prediction, against a random-masking control.

The top-right cell is the one a domain expert cannot detect by eye: an
explanation that lands on real binding sites while contributing nothing to the
prediction. A 2026 paper reporting only that attention looks biologically
sensible is measuring plausibility alone, and plausibility alone does not
clear a serious bar. Measuring both is the reason this paper is worth writing.

---

## 4. What we can honestly claim

1. **A measurement, not a model.** The contribution is the audit: several
   published DTI models, one metric suite, four difficulty levels, three
   seeds, family-wise error control.
2. **Two axes, not one.** Plausibility and faithfulness, each against its own
   control — the uniform-attention floor and random masking respectively.
3. **A family control that DAVIS and KIBA cannot supply.** Both datasets are
   essentially pure kinase panels (measured: DAVIS 429 kinase / 0 non-kinase;
   KIBA 227 / 0). The control is therefore a **transfer** panel of 60
   non-kinase BindingDB proteins — a harder condition than the datasets' own
   cold-target level, and it must be described as transfer rather than
   stratification.
4. **Our own model audited on the same terms.** If ColdSite-DTI comes off
   worst, that is a credibility asset and it gets reported without softening.

What we must **not** claim: that explanation degradation under distribution
shift is a new phenomenon (§2), or that measuring interpretability under
cold-start is unprecedented in DTI (§1.4).

---

## Before submission

- [x] Fill in authors, venues and DOIs for every §2 claim (2026-09-12; one claim
      removed for lack of a source). Remaining: swap the three arXiv preprints for
      published versions if they exist.
- [x] Confirm the CS-DTA description against the published abstract (2026-09-12: it
      matches — warm and strict cold-start, interpretability on localized protein
      regions, non-kinase validation as "preliminary evidence of partial
      transferability"). Still to check in the full text: that it measures no
      faithfulness and uses no floor or correction.
- [ ] Full-text checks flagged in §1: DMFF-DTA (where its interpretability analysis ran),
      EviDTI (hit ratio and cold-start as separate experiments), GPS-DTI (no quantitative
      binding-site test), ColdDTI (published version?)
- [ ] Re-check whether anything newer than CS-DTA has appeared. Seen in passing on
      2026-09-12, not yet read: PCIM-DTA (2026, *Bioinformatics*, cold-start DTA),
      CMA-DTI (2026, *Frontiers in Bioinformatics*, "interpretable" DTI), ColdstartMHDTI
      (2026, *Frontiers in Chemistry*), CDI-DTI ("cross-domain interpretable", JCIM),
      TrustDTI (2026, *IEEE Access*)
- [ ] Cross-check §4's claims against what the finished grid actually shows;
      any claim the numbers do not support comes out
