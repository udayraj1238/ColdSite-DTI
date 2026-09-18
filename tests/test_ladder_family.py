"""Holm over a family of ladder cells, read from the ladder files.

Results §7c's twelve-cell correction was done by hand. This module re-does it from the
files, so these tests pin both the arithmetic and the two rules that make the family a
family: a cell's p is the MEDIAN over its seeds, and a missing cell is reported rather
than quietly shrinking the family (which would loosen every threshold).
"""
import json

import pytest

from src.evaluation.ladder_family import cells, report


def write_ladder(folder, model, dataset, seed, per_level):
    payload = {level: {"by_k": {"10": {"precision_at_k": p, "p_value": pv,
                                       "chance": 0.02, "n_evaluated": 100}}}
               for level, (p, pv) in per_level.items()}
    path = folder / f"ladder_{model}_{dataset}_seed{seed}.json"
    path.write_text(json.dumps(payload))
    return path


def test_a_cells_p_is_the_median_of_its_seeds(tmp_path):
    for seed, pv in ((1, 0.001), (2, 0.400), (3, 0.900)):
        write_ladder(tmp_path, "m_ig", "kiba", seed, {"random": (0.05, pv)})
    p_values, precision, missing = cells(str(tmp_path), "kiba", ["m_ig"])
    assert p_values["m_ig|kiba|random"] == pytest.approx(0.400)      # not 0.001
    assert precision["m_ig|kiba|random"]["n_seeds"] == 3
    # the three levels this ladder does not contain are named, not silently dropped
    assert sorted(missing) == ["m_ig|kiba|cold_drug", "m_ig|kiba|cold_pair",
                               "m_ig|kiba|cold_target"]


def test_the_family_is_every_measured_cell_and_missing_ones_are_named(tmp_path):
    write_ladder(tmp_path, "m_ig", "kiba", 1,
                 {"random": (0.05, 0.001), "cold_drug": (0.04, 0.02)})
    p_values, _precision, missing = cells(str(tmp_path), "kiba", ["m_ig", "other_ig"])
    assert set(p_values) == {"m_ig|kiba|random", "m_ig|kiba|cold_drug"}
    assert [k for k in missing if k.startswith("other_ig")]
    assert any("cold_target" in k for k in missing)


def test_thresholds_are_holm_not_bonferroni(tmp_path):
    """alpha/(m-i), so the smallest p is tested against the strictest threshold and the
    largest against alpha itself -- a flat alpha/m would reject less."""
    for i, pv in enumerate((0.001, 0.02, 0.5, 0.9)):
        write_ladder(tmp_path, f"m{i}_ig", "kiba", 1, {"random": (0.05, pv)})
    p_values, precision, missing = cells(str(tmp_path), "kiba",
                                         [f"m{i}_ig" for i in range(4)])
    text = report(p_values, precision, missing, "t")
    assert "Holm–Bonferroni over **4 cells**" in text
    assert "0.0125 |" in text            # 0.05/4 for the smallest p
    assert "0.05 |" in text              # 0.05/1 for the largest
    assert "1 of 4 cells survive" in text


def test_the_davis_ig_family_reproduces_the_published_correction():
    """Results §7c, computed by hand: 7 of 12 cells survive. Skipped where the IG
    ladders are not on this machine."""
    import os
    folder = os.path.expanduser("~/ColdSite-results/integrated_gradients/ig_davis")
    if not os.path.isdir(folder):
        pytest.skip("DAVIS IG ladders not present")
    p_values, precision, missing = cells(
        folder, "davis", ["coldsite_dti_ig", "hyperattentiondti_ig", "moltrans_ig"])
    assert len(p_values) == 12, missing
    text = report(p_values, precision, missing, "t")
    assert "7 of 12 cells survive correction" in text
    # the four published anchors of Table R8
    assert precision["hyperattentiondti_ig|davis|cold_target"]["mean"] == pytest.approx(0.079, abs=5e-4)
    assert precision["coldsite_dti_ig|davis|cold_drug"]["mean"] == pytest.approx(0.055, abs=5e-4)
    assert precision["moltrans_ig|davis|random"]["mean"] == pytest.approx(0.018, abs=5e-4)
    assert p_values["moltrans_ig|davis|cold_target"] == pytest.approx(0.04995, rel=1e-3)
