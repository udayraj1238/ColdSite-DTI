"""MolTrans's published explanation is its drug x protein interaction map (its paper:
"visualize the strength of individual sub-structural interaction pair from the interaction
map", Fig. 3), not the protein encoder's self-attention this adapter reads by default.
These tests pin the properties that make the interaction readout worth having: it obeys the
explanation contract, and unlike the default it actually depends on the drug.

Skipped where the vendored MolTrans or subword_nmt is missing.
"""
import numpy as np
import pytest

SEQUENCE = ("MSGPRAGFYRQELNKTVWEVPQRLQGLRPVGSGAYGSVCSAYDARLRQKVAVKKLSRPFQSLIHARRTYRELRLLKHLKHE"
            "NVIGLLDVFTPATSIEDFSEVYLVTTLMGADLNNIVKCQALSDEHVQFLVYQLLRGLKYIHSAGIIHRDLKPSNVAVNEDC"
            "ELRILDFGLARQADEEMTGYVATRWYRAPEIMLNWMHYNQTVDIWSVGCIMAELLQGKALFPGSDYIDQLKRIMEVVGTPS")
DRUGS = ["CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]


def _adapter(**kwargs):
    pytest.importorskip("subword_nmt")
    from src.evaluation.baseline_adapters import MolTransAdapter
    try:
        MolTransAdapter.encode("C", "MKV")
    except Exception as exc:                                   # vendored repo missing
        pytest.skip(f"MolTrans not available: {exc}")
    return MolTransAdapter(checkpoint_path=None, device="cpu", **kwargs)


def _explain(adapter, smiles, sequence):
    d, dm, p, pm, tokens = type(adapter).encode(smiles, sequence)
    return np.asarray(adapter.explain(d, p, protein_tokens=tokens,
                                      drug_mask=dm, protein_mask=pm), dtype=float)


def test_the_interaction_readout_meets_the_explanation_contract():
    weights = _explain(_adapter(explanation="interaction"), DRUGS[0], SEQUENCE)
    assert weights.shape == (len(SEQUENCE),), weights.shape
    assert np.all(weights >= 0), "the contract requires non-negative weights"
    assert np.isfinite(weights).all()
    assert weights.max() > 0, "an all-zero map would rank nothing"


def test_the_interaction_readout_depends_on_the_drug_and_the_default_does_not():
    """The reason this readout exists. The encoder self-attention is computed before the
    drug is involved, so its map is identical for two different drugs; the interaction
    map is a product of both encoders and must move."""
    default = _adapter()
    interaction = _adapter(explanation="interaction")
    top = lambda w: set(np.argsort(-w)[:10])

    a, b = (_explain(default, s, SEQUENCE) for s in DRUGS)
    assert np.array_equal(a, b), "the default readout is protein-only; it must not move"

    a, b = (_explain(interaction, s, SEQUENCE) for s in DRUGS)
    assert not np.array_equal(a, b), "the interaction map did not change with the drug"
    assert len(top(a) & top(b)) < 10, "its top-10 residues are drug-independent too"


def test_max_and_sum_over_drug_substructures_are_different_readouts():
    one = _explain(_adapter(explanation="interaction"), DRUGS[0], SEQUENCE)
    other = _explain(_adapter(explanation="interaction", interaction_reduce="sum"),
                     DRUGS[0], SEQUENCE)
    assert one.shape == other.shape
    assert not np.array_equal(one, other)


def test_both_variants_are_registered_on_moltrans_weights():
    from src.evaluation.model_registry import available_models
    from src.model.checkpoint_naming import base_model_name, model_suffix
    for name in ("moltrans_interaction", "moltrans_interaction_sum"):
        assert name in available_models()
        assert base_model_name(name) == "moltrans"
        assert model_suffix(name) == model_suffix("moltrans")
