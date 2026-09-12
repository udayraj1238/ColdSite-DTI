"""
Track A (124AD0008) — collect real binding-site ground truth from UniProt.

This replaces three overlapping scripts (binding_sites.py,
fetch_binding_sites.py, fetch_kiba_binding_sites.py) that had drifted apart and
filtered features differently from each other.

Two things changed versus those versions, and both affect the paper's numbers:

1. The UniProt 'Site' feature type is NO LONGER collected. 'Site' is a
   catch-all that carries protease cleavage points, chromosomal breakpoints and
   interaction residues alongside genuine pockets. Roughly 7% of the previously
   committed annotations were of that kind. Every one of them is a position a
   drug does not bind to, counted as a correct answer -- which inflates
   precision@k and flatters the paper's central claim.

2. Every record now carries its 'type'. Without it, downstream filtering has to
   guess from free-text descriptions, which has known false negatives. Storing
   the type makes src/data/ground_truth.py filter exactly.

3. The provenance file now records the UniProt gene symbol and protein name for
   every target. This is what makes the kinase-confound control possible on
   KIBA: KIBA identifies its targets by accession ('O00141'), and
   src/evaluation/target_family.py classifies by gene symbol, so every KIBA
   target reads UNKNOWN and the control arm cannot run there at all. The names
   are already inside the entry JSON this fetcher downloads, so capturing them
   here means the accession -> gene map falls out of this one pass instead of
   costing a second trip over ~229 more accessions. Build the map afterwards
   with `python -m src.data.build_gene_map --dataset kiba`.

Usage
-----
    python -m src.data.fetch_binding_sites --dataset davis
    python -m src.data.fetch_binding_sites --dataset kiba
    python -m src.data.fetch_binding_sites --ids P00533 P04626 --out data/custom.json

Requires network access to rest.uniprot.org.
"""
import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

from src.data.ground_truth import BINDING_FEATURE_TYPES, normalise_target_id
from src.data.target_aliases import resolve_alias

UNIPROT_ENTRY = "https://rest.uniprot.org/uniprotkb/{}.json"

# size=10, not size=1. A gene-name search is a *substring-ish* match, so asking
# for one result and trusting it is how 'IKBKE' returned "Putative
# uncharacterized protein IKBKE-AS1" (an antisense transcript) and 'PRKCH'
# returned "PRKCH upstream open reading frame 2" -- both reviewed, both filed
# under a matching gene name, neither one the kinase. Neither failed. They
# produced an entry with a plausible accession and zero binding sites, and the
# target quietly dropped out of every precision@k average.
UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search?query={}&size=10"

HUMAN_TAXON = 9606

# `organism_id` matches one exact taxon. `taxonomy_id` matches the subtree, and
# that difference decides whether non-human targets resolve at all: the reviewed
# entries for P. falciparum CDPK1 and M. tuberculosis PknB are filed under
# strain taxa (3D7, H37Rv), not under the species id, so an organism_id query
# for the species finds nothing.
UNIPROT_TAXON_FIELD = "taxonomy_id"

# Protein names that mark a record as an artefact of the locus rather than the
# gene product itself. Used only to break ties, never to reject outright -- a
# real protein occasionally carries "putative" in its name.
def _entry_sequence_length(entry: dict) -> int:
    return int(((entry or {}).get("sequence") or {}).get("length") or 0)


def _entry_gene_names(entry: dict) -> set:
    """Every symbol this entry answers to, lowercased."""
    names = set()
    for gene in (entry or {}).get("genes") or []:
        value = (gene.get("geneName") or {}).get("value")
        if value:
            names.add(value.lower())
        for synonym in gene.get("synonyms") or []:
            if synonym.get("value"):
                names.add(synonym["value"].lower())
    return names


def choose_entry(results: list, symbol: str) -> tuple:
    """Pick the entry that is actually the protein `symbol` names.

    Two signals, in order:

    1. **Exact gene-symbol match.** Drops IKBKE-AS1 when IKBKE was asked for,
       because the antisense transcript's symbol is a different string.

    2. **Longest sequence.** Breaks the case exact matching cannot: an upstream
       open reading frame inside the PRKCH locus is genuinely filed under gene
       PRKCH, so no symbol comparison separates it from the kinase. It is ~50
       residues against the kinase's ~680, and for a drug-binding ground truth
       the large gene product is the one meant every time.

    Returns (entry, how) where `how` records which rule decided, so provenance
    can show it and a human can audit the ones that fell through to the weaker
    rule.
    """
    if not results:
        return None, "none"

    wanted = symbol.strip().lower()
    exact = [e for e in results if wanted in _entry_gene_names(e)]

    if exact:
        best = max(exact, key=_entry_sequence_length)
        how = "exact" if len(exact) == 1 else f"exact_longest_of_{len(exact)}"
        return best, how

    # No symbol matched. Still return something, but say so loudly in the route
    # name -- this is the case most likely to be wrong.
    best = max(results, key=_entry_sequence_length)
    return best, f"inexact_of_{len(results)}"


def _get_json(url: str, timeout: int = 20):
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def resolve_entry(target_id: str) -> tuple:
    """Target ID -> (UniProt entry JSON, accession, how it was resolved).

    DAVIS names point mutants like 'ABL1(T315I)p'. UniProt has no entry for
    those, so the mutation is stripped and the wild-type entry is used. That is
    a real approximation, not a formatting detail: every ABL1 variant ends up
    with an identical site list, and the Methods section has to say so. The
    resolution route is recorded per target so the paper can report how many
    entries came in that way.
    """
    # An alias short-circuits the normaliser entirely. DAVIS shorthand like
    # 'AMPK-alpha1' or 'PKNB(Mtuberculosis)' is not a damaged gene symbol that
    # normalisation can repair -- it is a different naming scheme, and for the
    # organism-bracketed ones the normaliser actively makes things worse by
    # discarding the only clue that the protein is not human.
    alias = resolve_alias(target_id)
    if alias:
        symbol, taxon = alias
        query = urllib.parse.quote(
            f"(gene:{symbol}) AND ({UNIPROT_TAXON_FIELD}:{taxon}) AND (reviewed:true)"
        )
        try:
            results = _get_json(UNIPROT_SEARCH.format(query)).get("results", [])
        except Exception:
            return None, symbol, f"alias_failed({symbol})"
        entry, how = choose_entry(results, symbol)
        if entry is None:
            # Deliberately not falling through to a broader search. A missing
            # alias target is a naming question for a human, not something to
            # resolve by loosening the query until something matches.
            return None, symbol, f"alias_not_found({symbol})"
        return entry, entry.get("primaryAccession", symbol), f"alias({symbol}|{how})"

    base = normalise_target_id(target_id)

    try:
        return _get_json(UNIPROT_ENTRY.format(base)), base, "accession"
    except urllib.error.HTTPError as exc:
        if exc.code not in (400, 404):
            raise
    except Exception:
        return None, base, "failed"

    query = urllib.parse.quote(
        f"(gene:{base}) AND ({UNIPROT_TAXON_FIELD}:{HUMAN_TAXON}) AND (reviewed:true)"
    )
    try:
        results = _get_json(UNIPROT_SEARCH.format(query)).get("results", [])
    except Exception:
        return None, base, "failed"
    entry, how = choose_entry(results, base)
    if entry is None:
        return None, base, "not_found"
    return entry, entry.get("primaryAccession", base), f"gene_search({how})"


def extract_features(entry: dict) -> list:
    """UniProt entry -> the binding-relevant features, with their types kept."""
    features = []
    for feature in entry.get("features", []):
        ftype = feature.get("type")
        if ftype not in BINDING_FEATURE_TYPES:
            continue
        location = feature.get("location", {})
        start = location.get("start", {}).get("value")
        end = location.get("end", {}).get("value", start)
        if start is None:
            continue
        features.append({
            "start": int(start),
            "end": int(end if end is not None else start),
            "type": ftype,
            "description": feature.get("description", "") or "",
        })
    return features


def extract_names(entry: dict) -> tuple:
    """UniProt entry -> (gene symbol, recommended protein name).

    Both are best-effort. An entry with no `genes` block is normal for some
    non-human and unreviewed records, and an empty string is a truthful answer
    there -- better than inventing a symbol, because a wrong gene symbol would
    silently mis-stratify a target into the wrong family, and the family split
    is the paper's control arm.
    """
    if not entry:
        return "", ""

    gene = ""
    genes = entry.get("genes") or []
    if genes:
        gene = (genes[0].get("geneName") or {}).get("value", "") or ""
        if not gene:
            synonyms = genes[0].get("synonyms") or []
            if synonyms:
                gene = synonyms[0].get("value", "") or ""

    protein = ""
    description = entry.get("proteinDescription") or {}
    recommended = description.get("recommendedName") or {}
    protein = (recommended.get("fullName") or {}).get("value", "") or ""
    if not protein:
        submitted = description.get("submissionNames") or []
        if submitted:
            protein = (submitted[0].get("fullName") or {}).get("value", "") or ""

    return gene, protein


def fetch_for_targets(target_ids, delay: float = 0.2, verbose: bool = True) -> dict:
    """Fetch every target, caching by resolved accession.

    DAVIS has 72 variant IDs collapsing onto ~17 base genes, so caching removes
    a few hundred redundant requests and, more importantly, guarantees the
    variants of one gene get byte-identical annotations rather than two
    different snapshots of a database that changed mid-run.
    """
    ground_truth, provenance, cache = {}, {}, {}
    total = len(target_ids)

    for i, target_id in enumerate(target_ids, start=1):
        base = normalise_target_id(target_id)
        if base in cache:
            features, accession, route, gene, protein, length = cache[base]
            route = f"{route}[cached]"
        else:
            entry, accession, route = resolve_entry(target_id)
            features = extract_features(entry) if entry else []
            gene, protein = extract_names(entry)
            length = _entry_sequence_length(entry)
            cache[base] = (features, accession, route, gene, protein, length)
            time.sleep(delay)

        ground_truth[target_id] = features
        provenance[target_id] = {
            "resolved_from": base,
            "uniprot_accession": accession,
            "resolution": route,
            "is_variant": base != str(target_id).strip(),
            "n_features": len(features),
            "gene_name": gene,
            "protein_name": protein,
            # Recorded so the length cross-check in resolve_unmapped --audit can
            # compare it against the sequence DAVIS itself ships for this
            # target. A UniProt entry a fraction of the dataset's own sequence
            # length is the signature of having resolved to a fragment or an
            # ORF artefact rather than the kinase.
            "sequence_length": length,
        }
        if verbose:
            status = "ok" if features else "NO SITES"
            print(f"[{i}/{total}] {target_id} -> {accession} ({route}) {status}")

    return {"sites": ground_truth, "provenance": provenance}


def main():
    parser = argparse.ArgumentParser(description="Fetch UniProt binding sites")
    parser.add_argument("--dataset", choices=["davis", "kiba"],
                        help="pull target IDs from this dataset")
    parser.add_argument("--ids", nargs="+", help="explicit UniProt IDs instead")
    parser.add_argument("--out", help="output JSON path")
    parser.add_argument("--delay", type=float, default=0.2)
    args = parser.parse_args()

    if args.dataset:
        from src.data.load_data import load_deepdta_dataset
        from src.data.align_ground_truth import uniprot_numbered_path
        target_ids = list(load_deepdta_dataset(args.dataset)["Target_ID"].unique())
        # UniProt numbers its own sequence, and DAVIS's sequences are not all
        # UniProt's -- some are fragments or other isoforms. The fetch writes the
        # UniProt-numbered file; align_ground_truth derives the one evaluation reads.
        out_path = args.out or uniprot_numbered_path(args.dataset)
    elif args.ids:
        target_ids = args.ids
        out_path = args.out or "data/custom_ground_truth_sites.json"
    else:
        parser.error("give either --dataset or --ids")

    print(f"Fetching binding sites for {len(target_ids)} targets...")
    result = fetch_for_targets(target_ids, delay=args.delay)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result["sites"], f, indent=2)
    # The provenance name predates the split into two site files; keep it.
    provenance_path = (f"data/{args.dataset}_ground_truth_sites_provenance.json"
                       if args.dataset and not args.out
                       else out_path.replace(".json", "_provenance.json"))
    with open(provenance_path, "w") as f:
        json.dump(result["provenance"], f, indent=2)

    with_sites = sum(1 for v in result["sites"].values() if v)
    variants = sum(1 for p in result["provenance"].values() if p["is_variant"])
    print(f"\n{with_sites}/{len(target_ids)} targets have annotated sites")
    print(f"{variants} were variant IDs resolved to a wild-type entry")
    print(f"Saved -> {out_path}")
    print(f"Saved -> {provenance_path}")
    if args.dataset == "davis" and not args.out:
        print("\nNow align to DAVIS's sequences:  python -m src.data.align_ground_truth")
    print("\nNow verify with:  python -m src.data.ground_truth")


if __name__ == "__main__":
    main()
