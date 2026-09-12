# Binding-site ground truth — format and how to use it

Track A deliverable for 124AD0067 (docs/01_GUIDE_124AD0008.md Step 4).

## Do not read the JSON directly

```python
# WRONG -- costs about a third of the score, silently
sites = json.load(open("data/davis_ground_truth_sites.json"))["AAK1"]
precision_at_k(attention, {s["start"] for s in sites}, k=10)

# RIGHT
from src.data.ground_truth import load_site_sets
sites = load_site_sets("data/davis_ground_truth_sites.json", max_len=1000)
precision_at_k(attention, sites["AAK1"].positions, k=10)
```

UniProt reports **1-indexed, inclusive residue ranges**. `precision_at_k`
expects a **0-indexed set of array positions**. Feeding one to the other does
not raise — it returns a plausible-looking number that is wrong. On a model
with perfect attention it returns 0.67 instead of 1.00
(`tests/test_ground_truth.py::test_off_by_one_regression`).

## File format

`data/{dataset}_ground_truth_sites.json`

```json
{
  "AAK1": [
    {"start": 176, "end": 176, "type": "Active site", "description": "Proton acceptor"},
    {"start": 52,  "end": 60,  "type": "Nucleotide binding", "description": "ATP"}
  ]
}
```

| field | meaning |
|---|---|
| `start`, `end` | 1-indexed, **inclusive** residue numbers. `52–60` is nine residues. For DAVIS they number **the DAVIS sequence the model reads**; for KIBA, UniProt's. See below. |
| `type` | UniProt feature type. Only `Binding site`, `Active site`, `Nucleotide binding` are collected. |
| `description` | free text, informational only |
| `uniprot_start`, `uniprot_end` | DAVIS only, on remapped features: the UniProt residues the feature came from |

A sibling `*_provenance.json` records, per target, which UniProt accession was
used and how it was resolved.

## DAVIS: two site files, and why

UniProt numbers residues along its own canonical sequence. The model reads DAVIS's
sequence, and those are not always UniProt's: for 54 targets with annotated sites
DAVIS holds a kinase-domain fragment, a longer isoform or a construct with extra
residues. A UniProt residue number then points at the wrong residue of what the model
saw -- or past its end. So DAVIS has two files, each with one writer:

| file | numbering | written by |
|---|---|---|
| `davis_ground_truth_sites_uniprot.json` | UniProt | `fetch_binding_sites`, `resolve_unmapped --apply` |
| `davis_ground_truth_sites.json` | **the DAVIS sequence** -- what evaluation reads | `align_ground_truth`, only |

`python -m src.data.align_ground_truth` aligns each DAVIS sequence to its UniProt
sequence and carries every annotated residue across only where the two agree; a
single-residue substitution between long aligned stretches carries too, so point
mutants keep their gatekeeper. Anything outside a fragment is dropped. The per-target
outcome is in `davis_ground_truth_alignment.json`; the UniProt sequences are cached in
`davis_uniprot_sequences.json`, so re-running needs no network.

**After re-fetching or applying overrides, re-run the alignment**, or evaluation keeps
reading the previous sites.

What the alignment found, beyond the numbering:

- **Four targets carried another protein's binding sites.** Name search had resolved
  PKAC-alpha to PKC-alpha, PAK1 to PKN1 (which carries "PAK1" as an old alias), MLCK to
  smooth-muscle MLCK and CDK11 to CDK11B. In each case the DAVIS sequence is identical,
  residue for residue, to a different entry -- PRKACA, PAK1, MYLK3 and CDK19 -- and none
  of it aligned to the entry that had been chosen. They are corrected through
  `davis_target_overrides.json`, like MST1 before them.
- **Some DAVIS sequences do not contain the kinase domain.** DAVIS's ROCK2 is UniProt
  residues 686-1388 and its MLK1 732-1104, while their annotated sites lie in the kinase
  domain earlier in the chain. The model never sees those sites. Five targets lose every
  site this way and drop out of the ladder as unusable, rather than being scored against
  residues they do not contain.

## What the adapter gives you

`load_site_sets()` returns `{target_id: SiteSet}`:

| attribute | meaning |
|---|---|
| `.positions` | 0-indexed set — this is what the metric consumes |
| `.usable` | `False` if nothing survived filtering; such proteins must be **skipped**, not scored 0.0 |
| `.is_variant` / `.resolved_from` | `ABL1(T315I)p` → `True` / `ABL1` |
| `.n_dropped_truncation` etc. | what was discarded and why, for the Methods section |

## Three decisions that must be recorded in the paper

**1. Truncation.** The model reads at most `max_protein_len` residues (default
1000). Annotated sites past that point can never be retrieved.

- `truncation="exclude"` (default) — drop them, and drop the protein if nothing
  is left. Measures explanation quality on residues the model can see.
- `truncation="keep"` — retain them; the protein stays in the average and its
  ceiling reflects the unreachable sites.

Current impact at `max_len=1000`:

| | DAVIS | KIBA |
|---|---|---|
| annotated positions past the cut | 280 | 164 |
| targets losing at least one position | **24** | **14** |
| targets losing everything (dropped) | **16** | **9** |

(Regenerated 2026-09-12 by `python -m src.data.ground_truth`, after the DAVIS
alignment.)

> **Figures corrected 2026-08-09 (124AD0015).** This line previously read
> "283 positions / 15 targets" and "165 positions / 10 targets", which conflated
> *targets affected* with *targets dropped*, and — for KIBA — counted 2 targets
> that are unusable because of description filtering rather than the window.
> `coverage_report()` could not express the distinction: `targets_dropped_entirely`
> counts every unusable target whatever the cause, and it is policy-dependent
> (10 under `exclude`, 2 under `keep`). A patch adding
> `targets_affected_by_truncation` and `targets_dropped_by_truncation` — both
> policy-independent — is with 124AD0008; once applied,
> `python -m src.data.ground_truth` prints this table's numbers directly.

Note the asymmetry: this does not change raw precision@k, because there is no
attention out past the cut. It changes which proteins are averaged over, and
the achievable ceiling.

**2. Mutant targets.** DAVIS names point mutants separately (`ABL1(T315I)p`).
UniProt has no entry for them, so all 72 variant IDs resolve to wild-type
accessions — every ABL1 variant gets an identical site list. Defensible;
must be stated.

**3. The ceiling.** With fewer than *k* annotated sites, perfect attention
still cannot reach 1.0 — six sites at k=20 caps at 0.30. Report
`mean_normalised` alongside `mean_precision_at_k`, or the numbers are not
comparable across proteins.

## Current coverage

Run `python -m src.data.ground_truth` to regenerate these numbers.

| | DAVIS | KIBA |
|---|---|---|
| targets in file | 442 | 229 |
| usable at `max_len=1000` | 402 | 212 |
| distinct wild-type accessions | 353 | 212 |
| total 0-indexed positions | 5,035 | 2,726 |

(2026-09-12, DAVIS after the alignment above.) Both files now carry `type` on every
feature -- `dropped_description` is 0 and nothing depends on the heuristic below any
more. The note is kept for history.

> ⚠️ **(Historical -- done.) These files needed regenerating.** They were produced by a fetcher that
> also collected UniProt's `Site` catch-all type, which carries protease
> cleavage points and chromosomal breakpoints — positions no drug binds to,
> counted as correct answers. The adapter currently filters them by description
> heuristic (125 dropped in DAVIS, 67 in KIBA), which has known false negatives.
> The fix is a re-fetch:
>
> ```bash
> python -m src.data.fetch_binding_sites --dataset davis
> python -m src.data.fetch_binding_sites --dataset kiba
> ```
>
> That writes `type` on every record, after which the adapter filters exactly
> and `dropped_feature_type` becomes non-zero instead of `dropped_description`.
> DAVIS coverage is also 409 of 442 targets — the 33 unresolved ones need a
> manual gene→UniProt lookup.
