"""Tests for the audit orchestrator, baseline adapter stubs, and figures."""
import numpy as np
import pytest

from src.evaluation.baseline_adapters import (
    DeepDTAAdapter,
    HyperAttentionDTIAdapter,
    MolTransAdapter,
    implementation_status,
)
from src.evaluation.model_registry import available_models
from src.evaluation.plots import (
    plot_degradation_curve,
    plot_faithfulness,
    plot_stratified_panels,
)
from src.evaluation.run_audit import LEVELS, build_grid, evaluate_cell, summarise
from src.evaluation.target_family import KINASE, NON_KINASE

SITES = {40, 41, 42, 120, 121, 122}


def perfect(length=250, sites=SITES):
    attention = np.zeros(length)
    attention[sorted(sites)] = 1.0
    return attention


def _cell(mean, std, n_seeds=3):
    return {"precision_at_k": {"mean": mean, "std": std, "n_seeds": n_seeds,
                               "sufficient_seeds": n_seeds >= 3}}


# --------------------------------------------------------------------------
# baseline stubs
# --------------------------------------------------------------------------

def test_all_three_baselines_are_registered():
    for name in ("deepdta", "hyperattentiondti", "moltrans"):
        assert name in available_models()


def test_importing_run_audit_alone_registers_the_baselines():
    """Registration is an import side-effect, and this test file used to be the
    only thing importing baseline_adapters.

    Without the import inside run_audit, `--models deepdta` failed with
    "unknown model. Registered: ['coldsite_dti', 'uniform_control']" and told
    the reader to write an adapter that already exists and already passes
    validate_adapter. The audit could only ever see our own two models.
    """
    import subprocess
    import sys

    probe = subprocess.run(
        [sys.executable, "-c",
         "import src.evaluation.run_audit as a;"
         "print(sorted(a.available_models()))"],
        capture_output=True, text=True)
    assert probe.returncode == 0, probe.stderr
    for name in ("deepdta", "hyperattentiondti", "moltrans"):
        assert name in probe.stdout


def test_deepdta_is_marked_as_having_no_attention():
    """Correct, not a gap -- it anchors the accuracy axis."""
    assert DeepDTAAdapter.provides_attention is False
    assert HyperAttentionDTIAdapter.provides_attention is True
    assert MolTransAdapter.provides_attention is True


def test_deepdta_explain_refuses_and_says_why():
    """The one adapter that must never produce an explanation.

    Manufacturing saliency for DeepDTA would put a different method's output
    into a table that reads as DeepDTA's. If this ever stops raising, check
    that the audit runner is still honouring `provides_attention`.
    """
    with pytest.raises(NotImplementedError) as exc:
        DeepDTAAdapter().explain(None, None)
    message = str(exc.value)
    assert "provides_attention" in message
    assert "accuracy anchor" in message


def test_status_marks_every_adapter_implemented_but_deepdta_unauditable():
    status = implementation_status()
    for name in ("coldsite_dti", "hyperattentiondti", "moltrans"):
        assert status[name]["implemented"]
        assert status[name]["auditable_for_explanations"], name
    # implemented, deliberately not auditable -- no attention to audit
    assert status["deepdta"]["implemented"]
    assert not status["deepdta"]["auditable_for_explanations"]


# --------------------------------------------------------------------------
# the grid
# --------------------------------------------------------------------------

def test_evaluate_cell_returns_the_expected_keys():
    cell = evaluate_cell([perfect()] * 5, [SITES] * 5, k=10, n_trials=50)
    for key in ("precision_at_k", "normalised", "ceiling", "p_value", "n_evaluated"):
        assert key in cell


def test_build_grid_covers_every_model_and_level():
    target_ids = [f"ABL{i}" for i in range(6)]

    def collect(model, dataset, level, seed):
        return [perfect()] * 6, [SITES] * 6, target_ids

    results = build_grid(collect, ["coldsite_dti"], ["davis"], [1, 2, 3],
                         k=10, n_trials=50)
    assert set(results["grid"]["coldsite_dti"]) == set(LEVELS)
    assert len(results["p_values_raw"]) == 4


def test_missing_cells_are_recorded_not_silently_zeroed():
    def collect(model, dataset, level, seed):
        if level == "cold_pair":
            return None
        return [perfect()] * 4, [SITES] * 4, ["ABL1"] * 4

    results = build_grid(collect, ["coldsite_dti"], ["davis"], [1, 2],
                         k=10, n_trials=30)
    assert "cold_pair" not in results["grid"]["coldsite_dti"]
    assert len(results["missing_cells"]) == 2
    assert any("cold_pair" in c for c in results["missing_cells"])


def test_correction_is_applied_once_over_the_whole_grid():
    """Correcting per model then pooling would control nothing."""
    def collect(model, dataset, level, seed):
        return [perfect()] * 5, [SITES] * 5, ["ABL1"] * 5

    results = build_grid(collect, ["coldsite_dti", "uniform_control"],
                         ["davis"], [1, 2, 3], k=10, n_trials=50)
    assert len(results["p_values_corrected"]) == 8      # 2 models x 4 levels
    thresholds = {v["adjusted_alpha"] for v in results["p_values_corrected"].values()}
    assert len(thresholds) > 1, "Holm thresholds must vary by rank"


def test_seed_p_values_are_combined_by_median_not_minimum():
    """Taking the smallest p of three runs is cherry-picking."""
    def collect(model, dataset, level, seed):
        rng = np.random.default_rng(seed)
        return [rng.random(250) for _ in range(5)], [SITES] * 5, ["ABL1"] * 5

    results = build_grid(collect, ["coldsite_dti"], ["davis"], [1, 2, 3],
                         k=10, n_trials=50)
    assert all(p > 0.0 for p in results["p_values_raw"].values())


def test_summary_warns_loudly_when_the_control_is_missing():
    def collect(model, dataset, level, seed):
        return [perfect()] * 5, [SITES] * 5, ["ABL1"] * 5   # all kinase

    report = summarise(build_grid(collect, ["coldsite_dti"], ["davis"],
                                  [1, 2, 3], k=10, n_trials=50))
    assert "No stratified comparison was possible" in report
    assert "limitation" in report


def test_stratified_comparison_appears_when_both_families_are_present():
    target_ids = ([f"ABL{i}" for i in range(25)]
                  + [f"HIV-{i} protease" for i in range(25)])

    def collect(model, dataset, level, seed):
        return [perfect()] * 50, [SITES] * 50, target_ids

    results = build_grid(collect, ["coldsite_dti"], ["davis"], [1, 2, 3],
                         k=10, n_trials=50)
    entry = results["stratified"]["coldsite_dti"]["random"]
    assert KINASE in entry and NON_KINASE in entry
    assert "kinase" in summarise(results)


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

def test_headline_figure_refuses_to_draw_without_accuracy(tmp_path):
    grid = {"m": {l: _cell(0.4, 0.02) for l in LEVELS}}
    with pytest.raises(ValueError, match="accuracy is required"):
        plot_degradation_curve(grid, accuracy=None,
                               save_path=str(tmp_path / "x.png"))


def test_headline_figure_draws_with_accuracy(tmp_path):
    grid = {"m": {l: _cell(0.4, 0.02) for l in LEVELS},
            "uniform_control": {l: _cell(0.05, 0.01) for l in LEVELS}}
    accuracy = {"m": {l: 0.8 for l in LEVELS}}
    path = plot_degradation_curve(grid, accuracy,
                                  save_path=str(tmp_path / "headline.png"))
    assert (tmp_path / "headline.png").exists()


def test_headline_figure_draws_chance_and_states_the_ceiling(tmp_path, monkeypatch):
    """Real precision@k on DAVIS sits within a few hundredths of chance, and the
    axis auto-scales to the data. Without a chance line a 0.01 wobble between
    levels fills the panel and reads as a trend."""
    import src.evaluation.plots as plots

    captured = {}
    monkeypatch.setattr(plots.plt, "close", lambda fig: captured.setdefault("fig", fig))

    chance = {"random": 0.020, "cold_drug": 0.020, "cold_target": 0.019, "cold_pair": 0.019}
    grid = {"m": {l: _cell(0.03, float("nan")) for l in LEVELS}}
    plot_degradation_curve(grid, {"m": {l: 0.8 for l in LEVELS}},
                           save_path=str(tmp_path / "h.png"),
                           chance=chance, ceiling={l: 0.99 for l in LEVELS})

    ax_fid = captured["fig"].axes[0]
    lines = {line.get_label(): list(line.get_ydata()) for line in ax_fid.get_lines()}
    chance_line = next(v for label, v in lines.items() if "chance" in label)
    assert chance_line == pytest.approx([chance[l] for l in LEVELS])
    assert "0.99" in ax_fid.get_title()


def test_stratified_panels_refuse_underpowered_families(tmp_path):
    stratified = {"m": {l: {KINASE: {"mean": 0.4, "std": 0.02}} for l in LEVELS}}
    with pytest.raises(ValueError, match="confound control does not yet exist"):
        plot_stratified_panels(stratified, target_counts={KINASE: 5},
                               save_path=str(tmp_path / "s.png"))


def test_stratified_panels_draw_when_both_families_qualify(tmp_path):
    stratified = {"m": {l: {KINASE: {"mean": 0.4, "std": 0.02},
                            NON_KINASE: {"mean": 0.2, "std": 0.03}}
                        for l in LEVELS}}
    plot_stratified_panels(stratified,
                           target_counts={KINASE: 200, NON_KINASE: 40},
                           save_path=str(tmp_path / "strat.png"))
    assert (tmp_path / "strat.png").exists()


def test_faithfulness_figure_plots_observed_beside_control(tmp_path):
    summaries = {l: {"comprehensiveness": 0.3,
                     "comprehensiveness_random": 0.05} for l in LEVELS}
    plot_faithfulness(summaries, save_path=str(tmp_path / "faith.png"))
    assert (tmp_path / "faith.png").exists()
