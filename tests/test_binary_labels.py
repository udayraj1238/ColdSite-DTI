"""The binary task: where labels come from, and that all three models agree.

The split files hold raw affinities. ColdSite-DTI's loader passed them straight
through, so `--task binary` handed BCEWithLogitsLoss pKd values of 5-11 as if
they were classes, and the run died at its first AUROC ("continuous format is
not supported"). `--dummy` never showed it: its random labels are already 0/1.

The audit table compares the three models' AUROCs, which only means something
if they are scored against the same labels -- hence one threshold, imported by
all three rather than copied into each.
"""
import json
import subprocess
import sys

import pandas as pd
import pytest

from src.model.dataset import BINARY_THRESHOLD, load_split


def _write_split(directory, n=48):
    """A small split whose affinities straddle the DAVIS threshold."""
    directory.mkdir(parents=True, exist_ok=True)
    aa, smi = "ACDEFGHIKLMNPQRSTVWY", "CCONc1ccccc1"
    for part, offset in (("train", 0), ("valid", 1), ("test", 2)):
        rows = range(n)
        pd.DataFrame({
            "compound_iso_smiles": [smi[(i + offset) % len(smi):] + "CC" for i in rows],
            "target_sequence": [(aa * 4)[(i + offset) % 20:][:60] for i in rows],
            # alternates 5.0 / 9.0, so both classes are present in every part
            "affinity": [5.0 if (i + offset) % 2 else 9.0 for i in rows],
        }).to_csv(directory / f"{part}.csv", index=False)
    return directory


def _labels(loader):
    return [float(y) for _d, _p, batch in loader for y in batch]


def test_binary_labels_are_classes_at_the_threshold(tmp_path):
    split = _write_split(tmp_path / "s")
    train, valid, test, _dv, _pv = load_split(
        str(split), batch_size=8, binary_threshold=BINARY_THRESHOLD["davis"])

    for loader in (train, valid, test):
        labels = _labels(loader)
        assert set(labels) == {0.0, 1.0}, sorted(set(labels))

    raw = pd.read_csv(split / "test.csv")["affinity"]
    assert _labels(test) == [float(v >= 7.0) for v in raw]


def test_the_threshold_is_inclusive(tmp_path):
    """pKd exactly 7.0 binds -- DeepDTA's definition is >=, not >."""
    split = tmp_path / "edge"
    split.mkdir()
    for part in ("train", "valid", "test"):
        pd.DataFrame({"compound_iso_smiles": ["CCO"] * 3,
                      "target_sequence": ["ACDEFGHIKL"] * 3,
                      "affinity": [6.99, 7.0, 7.01]}).to_csv(split / f"{part}.csv",
                                                             index=False)
    _t, _v, test, _dv, _pv = load_split(str(split), batch_size=8,
                                        binary_threshold=7.0)
    assert _labels(test) == [0.0, 1.0, 1.0]


def test_regression_keeps_the_raw_affinities(tmp_path):
    """No threshold, no change: the regression grid must be unaffected."""
    split = _write_split(tmp_path / "s")
    _t, _v, test, _dv, _pv = load_split(str(split), batch_size=8)
    assert _labels(test) == list(pd.read_csv(split / "test.csv")["affinity"])


def test_all_three_models_share_one_threshold():
    """Not merely equal values -- the same object, so they cannot drift."""
    import src.model.train_deepdta as deepdta
    import src.model.train_hyperattentiondti as hyperattention
    import src.model.dataset as dataset

    assert deepdta.BINARY_THRESHOLD is dataset.BINARY_THRESHOLD
    assert hyperattention.BINARY_THRESHOLD is dataset.BINARY_THRESHOLD
    assert dataset.BINARY_THRESHOLD == {"davis": 7.0, "kiba": 12.1}


@pytest.mark.slow
def test_coldsite_trains_end_to_end_on_the_binary_task(tmp_path):
    """The crash itself: the real trainer, real CLI, binary task, to a results
    file with an AUROC in it."""
    split = _write_split(tmp_path / "s")
    out = tmp_path / "out"
    run = subprocess.run(
        [sys.executable, "-m", "src.model.train", "--split-dir", str(split),
         "--dataset", "davis", "--split", "random", "--task", "binary",
         "--seed", "1", "--epochs", "1", "--batch-size", "16",
         "--results-dir", str(out)],
        capture_output=True, text=True)
    assert run.returncode == 0, run.stderr[-2000:]

    payload = json.load(open(out / "davis_random_binary_seed1_results.json"))
    auroc = payload["test_metrics"]["auroc"]
    assert 0.0 <= auroc <= 1.0


def test_binary_without_a_known_dataset_is_refused(tmp_path):
    """Guessing a threshold would silently define a different task."""
    split = _write_split(tmp_path / "s")
    run = subprocess.run(
        [sys.executable, "-m", "src.model.train", "--split-dir", str(split),
         "--dataset", "bindingdb", "--split", "random", "--task", "binary",
         "--seed", "1", "--epochs", "1"],
        capture_output=True, text=True)
    assert run.returncode != 0
    assert "--task binary needs --dataset" in run.stderr
