"""Tests for the 24-cell training grid runner.

The grid is a week of HPC time. Everything here is about failing before that is
spent, not after: the cell count, the uniqueness of every run tag, and the
artefact checks that decide whether a finished run can be attributed to a cell
at all.
"""
import json

import pytest

from src.model.checkpoint_naming import checkpoint_name, results_path, run_tag
from src.model.train import accuracy_metric_for
from src.model.run_grid import (
    DATASETS,
    EPOCHS,
    SEEDS,
    SPLITS,
    TASK,
    format_preflight,
    grid_cells,
    main,
    preflight,
    status_table,
    train_command,
    verify_cell,
    write_status,
)


# --------------------------------------------------------------------------
# the grid is 24 cells, not 72
# --------------------------------------------------------------------------

def test_the_grid_is_twenty_four_cells():
    """2 datasets x 4 splits x 3 TRAINING seeds. The three seeds are training
    seeds on one fixed split per cell -- 72 would mean regenerating splits per
    seed, which build_all_splits() cannot do and no guide asks for."""
    assert len(grid_cells()) == 24
    assert len(DATASETS) == 2 and len(SPLITS) == 4 and len(SEEDS) == 3


def test_the_grid_covers_exactly_the_required_cells():
    cells = {(c["dataset"], c["split"], c["seed"]) for c in grid_cells()}
    expected = {(d, s, seed) for d in ("davis", "kiba")
                for s in ("random", "cold_drug", "cold_target", "cold_pair")
                for seed in (1, 2, 3)}
    assert cells == expected


def test_the_task_is_regression_at_one_hundred_epochs():
    assert TASK == "regression"
    assert EPOCHS == 100


# --------------------------------------------------------------------------
# preflight
# --------------------------------------------------------------------------

def _make_splits(root, datasets=("davis", "kiba"), splits=SPLITS):
    for dataset in datasets:
        for split in splits:
            directory = root / dataset / split
            directory.mkdir(parents=True, exist_ok=True)
            for name in ("train.csv", "valid.csv", "test.csv"):
                (directory / name).write_text("a,b\n1,2\n")
    return root


def test_no_two_cells_can_overwrite_each_other(tmp_path):
    """The bug the naming module exists to prevent, asserted at grid level."""
    report = preflight(grid_cells(), str(_make_splits(tmp_path / "s")),
                       str(tmp_path / "r"))
    assert report["unique_tags"] == report["n_cells"] == 24
    assert report["problems"] == []


def test_preflight_blocks_when_split_files_are_missing(tmp_path):
    report = preflight(grid_cells(), str(tmp_path / "absent"), str(tmp_path))
    assert not report["splits_ready"]
    assert not report["ready_to_launch"]
    # 8 split directories (2 datasets x 4 splits) x 3 files each. The three
    # seeds share a directory, so the paths dedup to 24, not 72.
    assert len(report["missing_split_files"]) == 8 * 3
    assert any("train.csv" in p for p in report["missing_split_files"])


def test_preflight_names_track_a_as_the_owner_of_the_blocker(tmp_path):
    """A blocked grid must say whose blocker it is, not just that it failed."""
    text = format_preflight(preflight(grid_cells(), str(tmp_path / "absent"),
                                      str(tmp_path)))
    assert "BLOCKED" in text
    assert "124AD0008" in text and "build_splits" in text


def test_preflight_passes_when_every_split_is_present(tmp_path):
    report = preflight(grid_cells(), str(_make_splits(tmp_path / "s")),
                       str(tmp_path / "r"))
    assert report["ready_to_launch"]


def test_existing_checkpoints_warn_rather_than_silently_rerun(tmp_path):
    splits = _make_splits(tmp_path / "s")
    results = tmp_path / "r"
    results.mkdir()
    (results / checkpoint_name("davis", "random", TASK, 1)).touch()
    report = preflight(grid_cells(), str(splits), str(results))
    assert any("already exists" in w for w in report["warnings"])
    assert report["ready_to_launch"]


# --------------------------------------------------------------------------
# the launch command
# --------------------------------------------------------------------------

def test_the_command_carries_the_seed_and_the_matching_split_dir():
    """The most expensive mistake available: a cold-target number produced by a
    model trained on cold-drug."""
    command = train_command({"dataset": "davis", "split": "cold_target", "seed": 3})
    joined = " ".join(command)
    assert "--seed 3" in joined
    assert "data/splits/davis/cold_target" in joined
    assert "--dataset davis" in joined and "--split cold_target" in joined
    assert "--task regression" in joined


def test_every_cell_gets_a_distinct_command():
    commands = {" ".join(train_command(c)) for c in grid_cells()}
    assert len(commands) == 24


# --------------------------------------------------------------------------
# verification of a finished cell
# --------------------------------------------------------------------------

def _finish_cell(results_dir, cell, metrics=None, dataset_field=None):
    """Write the artefacts a completed run leaves behind."""
    import torch

    tag = run_tag(cell["dataset"], cell["split"], TASK, cell["seed"])
    torch.save({"model_state": {"w": torch.zeros(2)}, "epoch": 1},
               results_dir / checkpoint_name(cell["dataset"], cell["split"],
                                             TASK, cell["seed"]))
    payload = {"tag": tag, "dataset": dataset_field or cell["dataset"],
               "split": cell["split"], "task": TASK, "seed": cell["seed"],
               "test_metrics": metrics if metrics is not None
               else {"mse": 0.4, "ci": 0.72, "pearson": 0.6}}
    with open(results_path(str(results_dir), tag), "w") as f:
        json.dump(payload, f)


@pytest.mark.slow
def test_a_complete_cell_verifies(tmp_path):
    cell = {"dataset": "davis", "split": "cold_target", "seed": 2}
    _finish_cell(tmp_path, cell)
    report = verify_cell(cell, str(tmp_path))
    assert report["valid"], report["problems"]
    assert report["accuracy"] == 0.72


def test_a_missing_checkpoint_is_caught(tmp_path):
    report = verify_cell({"dataset": "davis", "split": "random", "seed": 1},
                         str(tmp_path))
    assert not report["valid"]
    assert any("no checkpoint" in p for p in report["problems"])


@pytest.mark.slow
def test_a_results_file_describing_another_run_is_caught(tmp_path):
    """The silent one: a number attributed to the wrong cell."""
    cell = {"dataset": "davis", "split": "cold_pair", "seed": 1}
    _finish_cell(tmp_path, cell, dataset_field="kiba")
    report = verify_cell(cell, str(tmp_path))
    assert not report["valid"]
    assert any("wrong run" in p for p in report["problems"])


@pytest.mark.slow
def test_a_run_with_no_accuracy_metric_is_caught(tmp_path):
    """Without it the Track C hand-off is empty and the headline figure cannot
    be drawn."""
    cell = {"dataset": "davis", "split": "random", "seed": 1}
    _finish_cell(tmp_path, cell, metrics={"mse": 0.3})
    report = verify_cell(cell, str(tmp_path))
    assert not report["valid"]
    assert any(accuracy_metric_for("regression") in p for p in report["problems"])


@pytest.mark.slow
def test_an_unloadable_checkpoint_is_caught(tmp_path):
    cell = {"dataset": "davis", "split": "random", "seed": 1}
    _finish_cell(tmp_path, cell)
    (tmp_path / checkpoint_name("davis", "random", TASK, 1)).write_text("junk")
    report = verify_cell(cell, str(tmp_path))
    assert not report["valid"]
    assert any("will not load" in p for p in report["problems"])


# --------------------------------------------------------------------------
# resuming after an interrupted run
# --------------------------------------------------------------------------

def _run_grid_over_one_cell(tmp_path, monkeypatch, cell):
    """Drive main() over a single cell, recording whether it trained.

    Returns the list of cells run_cell was called on, so a test can tell a
    skipped cell from a retrained one.
    """
    import src.model.run_grid as run_grid

    trained = []

    def fake_run_cell(cell, split_root, results_dir, task, epochs, extra=(),
                      dry_run=False):
        trained.append(cell)
        _finish_cell(tmp_path / "r", cell)
        return {"cell": cell, "status": "ok", "accuracy": 0.5,
                "checkpoint": str(tmp_path / "r" / checkpoint_name(
                    cell["dataset"], cell["split"], TASK, cell["seed"]))}

    monkeypatch.setattr(run_grid, "run_cell", fake_run_cell)
    monkeypatch.setattr("sys.argv", [
        "run_grid",
        "--datasets", cell["dataset"], "--splits", cell["split"],
        "--seeds", str(cell["seed"]),
        "--split-root", str(_make_splits(tmp_path / "s")),
        "--results-dir", str(tmp_path / "r"),
        "--skip-validation-cell",
    ])
    main()
    return trained


@pytest.mark.slow
def test_an_interrupted_cell_is_retrained_not_skipped(tmp_path, monkeypatch):
    """A checkpoint on disk does not mean the cell finished.

    train() saves on the first improving epoch -- usually epoch 0 -- and writes
    the results JSON only after the test pass. A dropped session therefore
    leaves a checkpoint with no results JSON. Skipping on the checkpoint alone
    banked a half-trained model as a finished cell, with no accuracy, and the
    grid reported success.
    """
    import torch

    cell = {"dataset": "davis", "split": "random", "seed": 1}
    results = tmp_path / "r"
    results.mkdir()
    # what an interrupted run leaves behind: a checkpoint, no results JSON
    torch.save({"model_state": {"w": torch.zeros(2)}, "epoch": 0},
               results / checkpoint_name("davis", "random", TASK, 1))
    assert not verify_cell(cell, str(results))["valid"]

    trained = _run_grid_over_one_cell(tmp_path, monkeypatch, cell)

    assert trained == [cell], "an interrupted cell must be retrained, not skipped"


@pytest.mark.slow
def test_the_validation_cell_is_the_first_unfinished_one(tmp_path, monkeypatch):
    """Resuming must not retrain a finished cell just because it sorts first.

    The validation cell used to be cells[0] unconditionally. On a resumed
    session that cell is usually already complete, so every restart spent an
    hour retraining it -- and overwrote a checkpoint and results JSON whose
    accuracy was already reported, with cuDNN free to return a slightly
    different number the second time.
    """
    import src.model.run_grid as run_grid

    results = tmp_path / "r"
    results.mkdir()
    done = {"dataset": "davis", "split": "random", "seed": 1}
    todo = {"dataset": "davis", "split": "random", "seed": 2}
    _finish_cell(results, done)
    assert verify_cell(done, str(results))["valid"]

    trained = []

    def fake_run_cell(cell, split_root, results_dir, task, epochs, extra=(),
                      dry_run=False):
        trained.append(cell)
        _finish_cell(results, cell)
        return {"cell": cell, "status": "ok", "accuracy": 0.5,
                "checkpoint": str(results / checkpoint_name(
                    cell["dataset"], cell["split"], TASK, cell["seed"]))}

    monkeypatch.setattr(run_grid, "run_cell", fake_run_cell)
    monkeypatch.setattr("sys.argv", [
        "run_grid", "--datasets", "davis", "--splits", "random",
        "--seeds", "1,2",
        "--split-root", str(_make_splits(tmp_path / "s")),
        "--results-dir", str(results),
    ])
    main()

    assert done not in trained, (
        "the finished cell was retrained as the validation cell")
    assert trained == [todo], f"expected only the unfinished cell, got {trained}"


@pytest.mark.slow
def test_a_complete_cell_is_still_skipped(tmp_path, monkeypatch):
    """The other half of the same decision: finished work is not redone."""
    cell = {"dataset": "davis", "split": "random", "seed": 1}
    results = tmp_path / "r"
    results.mkdir()
    _finish_cell(results, cell)
    assert verify_cell(cell, str(results))["valid"]

    trained = _run_grid_over_one_cell(tmp_path, monkeypatch, cell)

    assert trained == [], "a complete cell must not be retrained"


# --------------------------------------------------------------------------
# the status table
# --------------------------------------------------------------------------

def test_status_table_has_the_requested_columns():
    results = [{"cell": {"dataset": "davis", "split": "random", "seed": 1},
                "checkpoint": "results/coldsite_dti_davis_random_regression_seed1.pt",
                "accuracy": 0.83, "status": "ok"}]
    table = status_table(results)
    for column in ("dataset", "split", "seed", "checkpoint", "accuracy", "status"):
        assert column in table
    assert "0.8300" in table and "seed1" in table


def test_status_table_surfaces_failures_rather_than_burying_them():
    results = [{"cell": {"dataset": "kiba", "split": "cold_pair", "seed": 3},
                "status": "train-failed", "problems": ["trainer exited 1"]}]
    table = status_table(results)
    assert "## Failures" in table and "trainer exited 1" in table
    assert "0 of 1 cells complete" in table


# --------------------------------------------------------------------------
# the accuracy metric follows --task, it is not a constant
# --------------------------------------------------------------------------

@pytest.mark.slow
def test_a_binary_grid_verifies_against_auroc_not_ci(tmp_path):
    """Regression. The metric was hardcoded to "ci" while --task binary is an
    offered choice, so every cell of a binary grid failed verification with
    "no 'ci' in test_metrics" -- which reads like a training failure rather
    than a metric-name mismatch."""
    import json

    import torch

    cell = {"dataset": "davis", "split": "random", "seed": 1}
    tag = run_tag("davis", "random", "binary", 1)
    torch.save({"model_state": {"w": torch.zeros(2)}, "epoch": 1},
               tmp_path / checkpoint_name("davis", "random", "binary", 1))
    with open(results_path(str(tmp_path), tag), "w") as f:
        json.dump({"tag": tag, "dataset": "davis", "split": "random",
                   "task": "binary", "seed": 1,
                   "test_metrics": {"auroc": 0.83, "auprc": 0.7,
                                    "accuracy": 0.8}}, f)

    report = verify_cell(cell, str(tmp_path), task="binary")
    assert report["valid"], report["problems"]
    assert report["accuracy"] == 0.83


def test_the_metric_mapping_has_one_definition():
    """Two copies of it is how the grid's status table and Track C's hand-off
    end up reporting different quantities, both labelled "accuracy"."""
    from src.evaluation import run_faithfulness
    from src.model import train

    assert run_faithfulness.DEFAULT_ACCURACY_METRIC is train.DEFAULT_ACCURACY_METRIC


def test_an_unknown_task_raises_rather_than_guessing():
    with pytest.raises(ValueError, match="no accuracy metric"):
        accuracy_metric_for("ranking")


# --------------------------------------------------------------------------
# the status file, when two processes share a results directory
# --------------------------------------------------------------------------

def _share(split_types):
    return [{"dataset": "davis", "split": sp, "seed": seed}
            for sp in split_types for seed in SEEDS]


@pytest.mark.slow
def test_the_last_process_to_finish_writes_the_whole_grid(tmp_path):
    """Two GPUs, one results directory, each process training half the split
    types. The status used to be built from the caller's memory, so the last
    process to finish overwrote the other's file with its own half: a 12-cell
    grid reported "6 of 6 cells complete". Built from disk, the last writer
    describes all twelve.
    """
    gpu0, gpu1 = _share(["random", "cold_drug"]), _share(["cold_target", "cold_pair"])
    for cell in gpu0 + gpu1:
        _finish_cell(tmp_path, cell)

    def outcomes(cells):
        return [{"cell": c, "status": "ok", "accuracy": 0.72,
                 "checkpoint": str(tmp_path / checkpoint_name(
                     c["dataset"], c["split"], TASK, c["seed"]))} for c in cells]

    write_status(outcomes(gpu1), str(tmp_path), ["davis"])       # finishes first
    _, table_path, rows = write_status(outcomes(gpu0), str(tmp_path), ["davis"])

    text = open(table_path).read()
    assert "12 of 12 cells complete" in text, text
    assert len(rows) == 12
    # a cell only the *other* process trained must still be in the table
    assert "| davis | cold_pair | 3 |" in text


@pytest.mark.slow
def test_a_resumed_grid_counts_skipped_cells_as_complete(tmp_path):
    """On a resumed run every finished cell comes back as "skipped", and the
    count only included "ok" -- a complete grid read as nearly empty."""
    cells = _share(["random", "cold_drug", "cold_target", "cold_pair"])
    for cell in cells:
        _finish_cell(tmp_path, cell)
    skipped = [{"cell": c, "status": "skipped", "accuracy": 0.72} for c in cells]

    _, table_path, _ = write_status(skipped, str(tmp_path), ["davis"])

    assert "12 of 12 cells complete" in open(table_path).read()


def test_a_cell_nobody_has_reached_is_not_a_failure(tmp_path):
    """The other GPU's unstarted share is "missing", not a failure."""
    _, table_path, rows = write_status([], str(tmp_path), ["davis"])
    text = open(table_path).read()

    assert {r["status"] for r in rows} == {"missing"}
    assert "0 of 12 cells complete" in text
    assert "## Failures" not in text
