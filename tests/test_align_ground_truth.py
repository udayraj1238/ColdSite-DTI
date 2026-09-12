"""Carrying UniProt-numbered binding sites onto the sequences DAVIS actually holds.

For 54 targets the DAVIS sequence is a fragment, a longer isoform or a construct,
and a UniProt residue number points at the wrong residue of what the model saw.
These pin how each case maps, and -- on the real files -- that every aligned site
lands on the amino acid it annotated.
"""
import json
import os
import random

import pytest

from src.data.align_ground_truth import (
    MIN_BLOCK, align_target, map_features, residue_map)

AA = "ACDEFGHIKLMNPQRSTVWY"


def _protein(n, seed=0):
    rng = random.Random(seed)
    return "".join(rng.choice(AA) for _ in range(n))


def _site(start, end=None):
    return {"start": start, "end": end or start, "type": "Binding site", "description": ""}


def test_identical_sequences_keep_their_sites():
    seq = _protein(300)
    aligned, report = align_target([_site(50, 58)], seq, seq, 300)
    assert report["status"] == "identical"
    assert aligned == [_site(50, 58)]


def test_a_fragment_shifts_sites_and_drops_those_outside_it():
    """RET's case: DAVIS holds residues 101-400 of the UniProt protein."""
    uniprot = _protein(1000)
    davis = uniprot[100:400]
    aligned, report = align_target([_site(150), _site(50), _site(398, 405)],
                                   uniprot, davis, 1000)
    starts = [(f["start"], f["end"]) for f in aligned]
    assert (50, 50) in starts                  # UniProt 150 -> DAVIS 50
    assert all(f["uniprot_start"] != 50 for f in aligned)   # 50 lies outside the fragment
    assert (298, 300) in starts                # 398-400 kept, 401-405 fall off the end
    assert report["status"] == "remapped"
    assert report["residues_dropped"] == 1 + 5


def test_an_n_terminal_extension_shifts_every_site():
    """PIM1's case: DAVIS carries a longer isoform, 91 residues longer at the start."""
    uniprot = _protein(313, seed=1)
    davis = _protein(91, seed=2) + uniprot
    aligned, _ = align_target([_site(44), _site(121, 128)], uniprot, davis, 313)
    assert [(f["start"], f["end"]) for f in aligned] == [(135, 135), (212, 219)]


def test_a_point_mutation_at_a_site_is_kept():
    """ABL1(T315I): the mutated residue is the gatekeeper, a binding residue. A strict
    same-amino-acid rule would drop exactly the site that matters most."""
    uniprot = _protein(400, seed=3)
    davis = uniprot[:314] + ("I" if uniprot[314] != "I" else "V") + uniprot[315:]
    aligned, report = align_target([_site(315)], uniprot, davis, 400)
    assert [(f["start"], f["end"]) for f in aligned] == [(315, 315)]
    assert report["substitutions_mapped"] == 1


def test_a_divergent_region_is_not_mapped():
    """An isoform-specific stretch shares positions but not residues: nothing there
    corresponds, so a site inside it is dropped rather than guessed."""
    uniprot = _protein(400, seed=4)
    davis = uniprot[:200] + _protein(30, seed=5) + uniprot[230:]
    aligned, _ = align_target([_site(210), _site(300)], uniprot, davis, 400)
    assert [f["uniprot_start"] for f in aligned] == [300]


def test_a_changed_uniprot_sequence_keeps_nothing():
    """The sites were numbered against a sequence of the recorded length. If UniProt's
    sequence is now a different length, there is nothing sound to align them to."""
    seq = _protein(300)
    aligned, report = align_target([_site(50)], seq, seq, 310)
    assert aligned == [] and report["status"] == "uniprot_sequence_changed"


def test_an_insertion_inside_a_site_splits_it_in_two():
    """DAVIS carries five extra residues in the middle of an annotated range. Those
    five were never annotated, so the aligned range must not cover them."""
    uniprot = _protein(300, seed=6)
    davis = uniprot[:150] + _protein(5, seed=9) + uniprot[150:]
    out, kept, dropped = map_features([_site(140, 170)], residue_map(uniprot, davis)[0])
    assert [(f["start"], f["end"]) for f in out] == [(140, 150), (156, 175)]
    assert (kept, dropped) == (31, 0)


def test_a_deletion_inside_a_site_drops_only_the_missing_residues():
    """DAVIS lacks UniProt 151-155. Those five cannot map; the rest are adjacent in
    DAVIS's own numbering and stay one feature."""
    uniprot = _protein(300, seed=6)
    davis = uniprot[:150] + uniprot[155:]
    out, kept, dropped = map_features([_site(140, 170)], residue_map(uniprot, davis)[0])
    assert [(f["start"], f["end"]) for f in out] == [(140, 165)]
    assert (kept, dropped) == (26, 5)


def test_short_chance_matches_are_not_trusted():
    """Unrelated sequences share short identical runs by chance; none may map."""
    mapping, _ = residue_map(_protein(500, seed=7), _protein(500, seed=8))
    assert mapping == {}, f"{len(mapping)} residues mapped between unrelated proteins"
    assert MIN_BLOCK >= 8


# --------------------------------------------------------------------------
# the real files
# --------------------------------------------------------------------------

REAL = ("data/davis_ground_truth_sites.json", "data/davis_ground_truth_sites_uniprot.json",
        "data/davis_uniprot_sequences.json", "data/davis_ground_truth_sites_provenance.json",
        "src/data/baselines/deepdta/data/davis/proteins.txt")


def test_every_aligned_site_lands_on_the_residue_it_annotated():
    """The guarantee the alignment exists to give. For every site in the aligned file,
    the DAVIS residue it now points at is the UniProt residue it was annotated on --
    or a mapped point substitution, of which there are only a handful."""
    if not all(os.path.exists(p) for p in REAL):
        pytest.skip("aligned ground truth or its inputs not present")
    aligned, _orig, uniprot, prov, davis = (json.load(open(p)) for p in REAL)

    checked = mismatched = 0
    for target, features in aligned.items():
        seq_u = uniprot[prov[target]["uniprot_accession"]]
        for f in features:
            if "uniprot_start" not in f:       # identical sequence, numbering unchanged
                assert davis[target] == seq_u
                continue
            for offset in range(f["end"] - f["start"] + 1):
                checked += 1
                if davis[target][f["start"] - 1 + offset] != seq_u[f["uniprot_start"] - 1 + offset]:
                    mismatched += 1
    assert checked > 0
    assert mismatched <= checked * 0.02, f"{mismatched} of {checked} remapped residues differ"


def test_no_aligned_site_points_past_the_end_of_its_sequence():
    if not all(os.path.exists(p) for p in REAL):
        pytest.skip("aligned ground truth or its inputs not present")
    aligned = json.load(open(REAL[0]))
    davis = json.load(open(REAL[4]))
    for target, features in aligned.items():
        for f in features:
            assert 1 <= f["start"] <= f["end"] <= len(davis[target]), (target, f)
