"""A model may refuse an input its own published loader refuses.

DrugBAN caps a drug at 290 atoms (DRUG.MAX_NODES); the 60-protein non-kinase panel holds one
BindingDB ligand of 322, and that single molecule out of 16,033 aborted every DrugBAN control
run on 2026-09-20. Dropping such a row is right. Dropping many would change which proteins a
cell covers without saying so, which is what these tests pin.
"""
import numpy as np
import pytest

from src.evaluation import collect as collect_module
from src.evaluation.collect import MissingCell, collect_cell

GROUND_TRUTH = "data/davis_ground_truth_sites.json"


def _cell(monkeypatch, refuse_every, max_proteins=12, capsys=None):
    """Collect a real cell with `refuse_every`-th row refused by the encoder."""
    real = collect_module._explain_row
    state = {"n": 0}

    def flaky(model_name, adapter, vocabs, smiles, sequence, max_protein_len):
        state["n"] += 1
        if refuse_every and state["n"] % refuse_every == 0:
            raise ValueError("322 atoms exceeds DrugBAN's DRUG.MAX_NODES = 290; "
                             "their dataloader cannot encode this drug")
        return real(model_name, adapter, vocabs, smiles, sequence, max_protein_len)

    monkeypatch.setattr(collect_module, "_explain_row", flaky)
    from src.evaluation.collect import load_site_sets
    return collect_cell("uniform_control", "davis", "random", 1,
                        site_sets=load_site_sets(GROUND_TRUTH, max_len=1000),
                        max_proteins=max_proteins, device="cpu", verbose=True)


def _have_data():
    import os
    return os.path.exists(GROUND_TRUTH) and os.path.isdir("data/splits/davis/random")


def test_one_refused_row_is_skipped_not_fatal(monkeypatch, capsys):
    if not _have_data():
        pytest.skip("DAVIS splits or ground truth not present")
    weights, sites, ids = _cell(monkeypatch, refuse_every=12)
    assert len(weights) == len(sites) == len(ids)
    assert len(weights) >= 10, "the whole cell was lost over one row"
    assert all(np.asarray(w).ndim == 1 for w in weights)
    out = capsys.readouterr().out
    assert "skipped 1 row(s) its own encoder refuses" in out, out
    assert "MAX_NODES" in out, "the reason must be printed, not swallowed"


def test_many_refused_rows_fail_the_cell(monkeypatch):
    """Silently scoring the survivors would report a different population under the same
    name. Above the limit the cell must fail loudly."""
    if not _have_data():
        pytest.skip("DAVIS splits or ground truth not present")
    with pytest.raises(MissingCell) as excinfo:
        _cell(monkeypatch, refuse_every=2)
    message = str(excinfo.value)
    assert "could not encode" in message and "MAX_NODES" in message


def test_a_misalignment_is_never_swallowed(monkeypatch):
    """Only ValueError means 'I cannot represent this input'. A RuntimeError is the
    adapter saying its explanation is misaligned, which must still crash the run."""
    if not _have_data():
        pytest.skip("DAVIS splits or ground truth not present")

    def broken(*_args, **_kwargs):
        raise RuntimeError("explanation has 37 weights for a 400-residue sequence")

    monkeypatch.setattr(collect_module, "_explain_row", broken)
    from src.evaluation.collect import load_site_sets
    with pytest.raises(RuntimeError, match="misaligned|weights for"):
        collect_cell("uniform_control", "davis", "random", 1,
                     site_sets=load_site_sets(GROUND_TRUTH, max_len=1000),
                     max_proteins=5, device="cpu", verbose=False)
