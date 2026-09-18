"""Would a single-seed paper have reported a different verdict?

Results §7f rests on two counts -- how many cells' seeds disagree with each other, and in
how many the spread across seeds exceeds the cell's distance from chance. Both are easy to
compute in a way that flatters the argument, so these tests pin the arithmetic on cells
whose answer is known by construction, and pin the one real number the section leans on:
the cell where all three seeds agree is the cell that survives correction.
"""
import json
import os

import pytest

from src.evaluation.seed_agreement import cells, report, summarise


def write_ladder(folder, model, dataset, seed, per_level):
    payload = {level: {"by_k": {"10": {"precision_at_k": p, "p_value": pv,
                                       "chance": ch, "n_evaluated": 100}}}
               for level, (p, pv, ch) in per_level.items()}
    path = folder / f"ladder_{model}_{dataset}_seed{seed}.json"
    path.write_text(json.dumps(payload))


def test_a_cell_whose_seeds_disagree_is_counted_once(tmp_path):
    for seed, (p, pv) in enumerate(((0.05, 0.001), (0.02, 0.400), (0.02, 0.600)), start=1):
        write_ladder(tmp_path, "m", "davis", seed, {"random": (p, pv, 0.02)})
    table = cells({("M", "davis"): f"{tmp_path}/ladder_m_davis_seed*.json"})
    stats = summarise(table)
    assert stats == {"cells": 1, "seeds_disagree": 1, "all_three": 0, "none": 0,
                     "spread_exceeds_signal": 1}


def test_unanimous_cells_are_not_counted_as_disagreeing(tmp_path):
    for seed in (1, 2, 3):
        write_ladder(tmp_path, "m", "davis", seed, {"random": (0.05, 0.001, 0.02)})
    stats = summarise(cells({("M", "davis"): f"{tmp_path}/ladder_m_davis_seed*.json"}))
    assert stats["all_three"] == 1 and stats["seeds_disagree"] == 0
    assert stats["spread_exceeds_signal"] == 0        # zero spread, real distance


def test_a_cell_with_fewer_than_three_seeds_is_ignored(tmp_path):
    """Two seeds cannot show disagreement across three, and quoting such a cell beside
    the others would make the counts mean different things per row."""
    for seed in (1, 2):
        write_ladder(tmp_path, "m", "davis", seed, {"random": (0.05, 0.001, 0.02)})
    assert cells({("M", "davis"): f"{tmp_path}/ladder_m_davis_seed*.json"}) == {}


def test_spread_is_compared_against_distance_from_chance_not_against_zero(tmp_path):
    """A wide spread around a far-from-chance mean is not the finding; a spread wider
    than the signal is."""
    for seed, p in enumerate((0.20, 0.21, 0.22), start=1):       # far above chance, tight
        write_ladder(tmp_path, "m", "davis", seed, {"random": (p, 0.001, 0.02)})
    stats = summarise(cells({("M", "davis"): f"{tmp_path}/ladder_m_davis_seed*.json"}))
    assert stats["spread_exceeds_signal"] == 0


def test_the_report_states_both_counts_and_marks_each_seed(tmp_path):
    for seed, (p, pv) in enumerate(((0.05, 0.001), (0.02, 0.400), (0.02, 0.600)), start=1):
        write_ladder(tmp_path, "m", "davis", seed, {"random": (p, pv, 0.02)})
    text = report(cells({("M", "davis"): f"{tmp_path}/ladder_m_davis_seed*.json"}))
    assert "`*..`" in text
    assert "**1 of 1 cells have seeds that disagree**" in text


def test_the_published_counts_reproduce_from_the_committed_ladders():
    """Results §7f: 11 of 16 disagree, 15 of 16 have a spread wider than their signal,
    and exactly one cell -- the one that survives Holm -- has all three seeds above α."""
    from src.evaluation.seed_agreement import default_sources

    davis = "results/analysis_davis_policyA"
    kiba = "results/analysis_kiba_policyA"
    if not (os.path.isdir(davis) and os.path.isdir(kiba)):
        pytest.skip("committed ladder folders not present")
    table = cells(default_sources(davis, kiba))
    stats = summarise(table)
    assert stats["cells"] == 16, stats
    assert stats["seeds_disagree"] == 11, stats
    assert stats["spread_exceeds_signal"] == 15, stats
    assert stats["all_three"] == 1, stats
    unanimous = [key for key, seeds in table.items()
                 if all(p < 0.05 for _precision, p, _chance in seeds)]
    assert unanimous == [("HyperAttentionDTI", "davis", "random")], unanimous
