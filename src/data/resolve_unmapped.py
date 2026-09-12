"""
Track A (124AD0008) -- close the gap between the targets a dataset has and the
targets the ground truth covers.

DAVIS has 442 unique targets; the fetcher resolves ~409 of them. The rest fail
for mundane reasons -- a retired gene symbol, a symbol UniProt files under a
synonym, a non-human entry, a DAVIS-specific spelling of a fusion or a
domain-only construct. They are not distributed at random, and that is the
problem: unresolved targets are silently absent from every precision@k average,
so a systematic gap (say, every non-human or every fusion construct) becomes a
systematic bias in the headline number that nothing downstream can detect.

This script does three things, in this order:

  report   -- list what failed and why, grouped by failure route
  template -- write a hand-editable overrides file, pre-filled with the failures
  apply    -- fetch the accessions you filled in and merge them into the
              ground-truth JSON, recording that they came in by hand

The overrides file is deliberately manual. Automatic fuzzy resolution of a gene
symbol is exactly the kind of step that quietly attaches the wrong protein's
binding sites to a target, and a wrong site list does not crash -- it produces a
plausible, wrong precision@k, the same failure shape as the coordinate bug in
Section 6 of the handover.

Targets that genuinely have no UniProt entry should be marked
`"unresolvable": "<reason>"` rather than left blank or forced onto a near-match.
That count belongs in the Methods section.

Usage
-----
    python -m src.data.resolve_unmapped --dataset davis --report
    python -m src.data.resolve_unmapped --dataset davis --template
    # ... hand-edit data/davis_target_overrides.json ...
    python -m src.data.resolve_unmapped --dataset davis --apply

Requires network access to rest.uniprot.org for --apply only.
"""
import argparse
import collections
import json
import os
import time


def load_json(path: str, what: str) -> dict:
    if not os.path.exists(path):
        raise SystemExit(
            f"No {what} at {path}.\n"
            f"Run the fetcher first:  python -m src.data.fetch_binding_sites "
            f"--dataset <davis|kiba>"
        )
    with open(path) as handle:
        return json.load(handle)


def unresolved_targets(provenance: dict, sites: dict) -> dict:
    """{target_id: reason} for everything without usable ground truth.

    Two distinct failures are collapsed here on purpose, because both have the
    same downstream effect -- the target contributes nothing to any average:

      not_found / failed  -- UniProt returned no entry at all
      no_sites            -- an entry resolved, but carries no binding-relevant
                             feature. That is a real biological answer for some
                             proteins, not necessarily an error, so it is
                             reported separately in the summary.

    `alias_not_found(SYMBOL)` and `alias_failed(SYMBOL)` are matched too. Those
    mean somebody wrote an alias in target_aliases.py that UniProt does not
    recognise -- the most important failure in this list, because it is a wrong
    guess rather than a missing one, and it carries the symbol that was tried so
    it can be corrected rather than just re-run.
    """
    FAILED_ROUTES = ("not_found", "failed", "unknown",
                     "alias_not_found", "alias_failed")
    out = {}
    for target_id, record in provenance.items():
        route = (record or {}).get("resolution", "unknown")
        if route.startswith(FAILED_ROUTES):
            out[target_id] = route
        elif not sites.get(target_id):
            out[target_id] = "no_sites"
    return out


def report(provenance: dict, sites: dict) -> None:
    unresolved = unresolved_targets(provenance, sites)
    by_reason = collections.Counter(unresolved.values())
    routes = collections.Counter(
        (r or {}).get("resolution", "unknown") for r in provenance.values())

    print(f"targets in provenance      {len(provenance)}")
    print(f"targets with >=1 site      {sum(1 for v in sites.values() if v)}")
    print(f"targets needing attention  {len(unresolved)}")
    print("\nresolution routes (all targets):")
    for route, count in routes.most_common():
        print(f"  {route:24s} {count}")
    print("\nneeding attention, by reason:")
    for reason, count in by_reason.most_common():
        print(f"  {reason:24s} {count}")

    print("\ntargets:")
    for target_id, reason in sorted(unresolved.items()):
        print(f"  {target_id:32s} {reason}")


def audit(provenance: dict, dataset: str) -> None:
    """Flag targets that resolved to *something* but possibly the wrong thing.

    A target that fails to resolve is visible: it sits in the not-found list
    with zero sites. A target that resolves to the wrong protein is not. It
    gets a plausible accession, a plausible gene name, and a site list that
    belongs to something else — and it is scored as correct for the rest of the
    project's life.

    Two checks, both cheap, neither needing the network:

    **Inexact gene match.** The fetcher records how it picked among UniProt's
    candidates. `inexact_of_N` means no candidate's gene symbol actually matched
    what was asked for, and the longest sequence was taken as a guess.

    **Not described as a kinase.** DAVIS and KIBA are kinase panels, so every
    target should resolve to a protein kinase. Anything that does not is either
    a gene-symbol collision or a naming quirk. This check found MST1 resolving
    to macrophage-stimulating 1 -- a hepatocyte growth factor-like protein, not
    the STK4 kinase a kinase panel means by that name -- which neither of the
    other two checks would have caught, because the wrong entry is a normal
    length and matched the queried symbol exactly.

    Expect roughly 50 entries here on DAVIS and nearly all of them to be fine:
    UniProt's recommended name for a receptor tyrosine kinase is the receptor
    ("Epidermal growth factor receptor", "Ephrin type-A receptor 2"), and the
    channel-kinase TRPM6 and the myosin-kinases MYO3A/B are named for their
    other domain. The list is short enough to read, which is the point.

    **Length disagreement.** DAVIS ships its own sequence for every target. If
    the UniProt entry is a small fraction of it, the fetcher resolved to a
    fragment, an isoform stub or an ORF artefact rather than the protein — the
    failure that produced "PRKCH upstream open reading frame 2", 50 residues
    standing in for a 683-residue kinase. The reverse (UniProt much longer) is
    normal and not flagged: DAVIS often ships a kinase-domain construct rather
    than the full-length protein.
    """
    from src.data.load_data import load_deepdta_dataset

    df = load_deepdta_dataset(dataset)
    dataset_lengths = (df.drop_duplicates("Target_ID")
                         .set_index("Target_ID")["Target"]
                         .astype(str).str.len().to_dict())

    import re

    # Wide on purpose. A receptor tyrosine kinase is named for its receptor, and
    # the atypical kinases (alpha-kinases, myosin-III, TRPM6/7) are named for
    # their other domain, so "receptor" and "myosin" count as kinase-consistent
    # in a kinase panel. The aim is a list short enough that somebody reads it.
    kinase_like = re.compile(
        r"kinase|phosphotransferase|receptor|myosin|"
        r"transient receptor potential|phosphatidylinositol|PI3|PI4",
        re.IGNORECASE)

    inexact, short, not_kinase = [], [], []
    seen_accessions = set()
    for target_id, record in sorted(provenance.items()):
        route = (record or {}).get("resolution", "")
        if "inexact" in route:
            inexact.append((target_id, route, record.get("protein_name", "")))

        uniprot_len = record.get("sequence_length") or 0
        dataset_len = dataset_lengths.get(target_id, 0)
        if uniprot_len and dataset_len and uniprot_len < 0.5 * dataset_len:
            short.append((target_id, uniprot_len, dataset_len,
                          record.get("protein_name", "")))

        accession = record.get("uniprot_accession")
        protein = record.get("protein_name", "") or ""
        if protein and accession not in seen_accessions:
            seen_accessions.add(accession)
            if not kinase_like.search(protein):
                not_kinase.append((target_id, accession, protein))

    print(f"=== audit: {dataset} ===\n")
    print(f"resolved by an inexact gene match: {len(inexact)}")
    for target_id, route, name in inexact:
        print(f"  {target_id:28s} {route:24s} {name[:48]}")

    print(f"\nUniProt entry under half the length of the dataset's own "
          f"sequence: {len(short)}")
    for target_id, u, d, name in short:
        print(f"  {target_id:28s} uniprot={u:>6}  {dataset}={d:>6}   {name[:44]}")

    print(f"\nnot described as a kinase, in a kinase panel: {len(not_kinase)}")
    for target_id, accession, protein in not_kinase:
        print(f"  {target_id:24s} {str(accession):9s} {protein[:52]}")
    if not_kinase:
        print("  Most of these are fine -- receptor tyrosine kinases are named "
              "for the receptor.\n  Read the list anyway: a gene-symbol "
              "collision looks exactly like the rest of it.")

    if not inexact and not short and not not_kinase:
        print("\nnothing flagged.")
    else:
        print("\nCheck each flagged target on uniprot.org. Anything wrong goes "
              "in the overrides file (--template) with the correct accession; "
              "a bad entry in src/data/target_aliases.py should be corrected "
              "or removed there.")
    if not any(r.get("sequence_length") for r in provenance.values()):
        print("\nNOTE: this provenance file has no sequence_length field, so "
              "the length check could not run. Re-fetch to record it.")


def write_template(provenance: dict, sites: dict, path: str) -> None:
    """Pre-fill an overrides file, preserving anything already filled in."""
    unresolved = unresolved_targets(provenance, sites)

    existing = {}
    if os.path.exists(path):
        with open(path) as handle:
            existing = json.load(handle)

    template = {}
    for target_id, reason in sorted(unresolved.items()):
        prior = existing.get(target_id, {})
        template[target_id] = {
            "reason_it_failed": reason,
            "uniprot_accession": prior.get("uniprot_accession", ""),
            "unresolvable": prior.get("unresolvable", ""),
            "note": prior.get("note", ""),
        }

    # never silently drop a decision already recorded for a target that has
    # since started resolving
    for target_id, record in existing.items():
        template.setdefault(target_id, record)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as handle:
        json.dump(template, handle, indent=2, sort_keys=True)

    filled = sum(1 for v in template.values() if v.get("uniprot_accession"))
    marked = sum(1 for v in template.values() if v.get("unresolvable"))
    print(f"Saved -> {path}")
    print(f"  {len(template)} targets listed, {filled} with an accession, "
          f"{marked} marked unresolvable")
    print("\nFill in `uniprot_accession` by searching uniprot.org for the target,")
    print("or set `unresolvable` to a short reason. Then run --apply.")


def apply_overrides(dataset: str, sites_path: str, provenance_path: str,
                    overrides_path: str, delay: float = 0.2) -> None:
    from src.data.fetch_binding_sites import extract_features, extract_names, _get_json
    from src.data.fetch_binding_sites import UNIPROT_ENTRY

    sites = load_json(sites_path, "ground truth")
    provenance = load_json(provenance_path, "provenance")
    overrides = load_json(overrides_path, "overrides file")

    applied, skipped, failed = 0, 0, []
    for target_id, record in sorted(overrides.items()):
        accession = (record.get("uniprot_accession") or "").strip()
        if not accession:
            skipped += 1
            continue

        try:
            entry = _get_json(UNIPROT_ENTRY.format(accession))
        except Exception as exc:
            failed.append((target_id, accession, str(exc)))
            continue
        time.sleep(delay)

        features = extract_features(entry)
        gene, protein = extract_names(entry)

        sites[target_id] = features
        provenance[target_id] = {
            "resolved_from": target_id,
            "uniprot_accession": accession,
            # tagged so the Methods section can report exactly how many entries
            # came in by hand rather than by lookup
            "resolution": "manual_override",
            "is_variant": False,
            "n_features": len(features),
            "gene_name": gene,
            "protein_name": protein,
            "note": record.get("note", ""),
        }
        applied += 1
        print(f"  {target_id:32s} -> {accession} ({len(features)} features)")

    with open(sites_path, "w") as handle:
        json.dump(sites, handle, indent=2)
    with open(provenance_path, "w") as handle:
        json.dump(provenance, handle, indent=2)

    unresolvable = sum(1 for v in overrides.values() if v.get("unresolvable"))
    print(f"\napplied      {applied}")
    print(f"skipped      {skipped} (no accession filled in)")
    print(f"unresolvable {unresolvable} (documented, counts toward the Methods number)")
    if failed:
        print(f"failed       {len(failed)}")
        for target_id, accession, err in failed:
            print(f"  {target_id} -> {accession}: {err}")
    print(f"\nUpdated {sites_path}")
    print(f"Updated {provenance_path}")
    print("\nRe-check coverage:  python -m src.data.ground_truth")


def main():
    parser = argparse.ArgumentParser(
        description="Report on and resolve targets with no ground truth")
    parser.add_argument("--dataset", choices=["davis", "kiba"], default="davis")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--audit", action="store_true",
                        help="flag targets that resolved to possibly the wrong "
                             "protein (inexact gene match, or a UniProt entry "
                             "far shorter than the dataset's own sequence)")
    parser.add_argument("--template", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--delay", type=float, default=0.2)
    args = parser.parse_args()

    # Resolution works on the UniProt-numbered sites; for DAVIS the file evaluation
    # reads is derived from them by align_ground_truth, and must not be written here.
    from src.data.align_ground_truth import uniprot_numbered_path
    sites_path = uniprot_numbered_path(args.dataset)
    provenance_path = f"data/{args.dataset}_ground_truth_sites_provenance.json"
    overrides_path = f"data/{args.dataset}_target_overrides.json"

    if not (args.report or args.audit or args.template or args.apply):
        args.report = True

    if args.report:
        report(load_json(provenance_path, "provenance"),
               load_json(sites_path, "ground truth"))
    if args.audit:
        audit(load_json(provenance_path, "provenance"), args.dataset)
    if args.template:
        write_template(load_json(provenance_path, "provenance"),
                       load_json(sites_path, "ground truth"),
                       overrides_path)
    if args.apply:
        apply_overrides(args.dataset, sites_path, provenance_path,
                        overrides_path, delay=args.delay)
        if args.dataset == "davis":
            print("\nNow re-align to DAVIS's sequences:  python -m src.data.align_ground_truth")


if __name__ == "__main__":
    main()
