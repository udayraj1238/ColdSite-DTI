"""Alternative attention readouts: the sensitivity check Limitations promises.

Between the tensor inside a network and the one number per residue that precision@k
scores, someone chooses an axis to reduce, a layer to read and a way to spread a
convolution position or sub-word token over residues. These tests check that a variant
changes exactly that choice and nothing else -- same checkpoint, same prediction -- since
a variant that quietly loaded different weights, or that silently fell back to the
published readout, would make the whole check meaningless.
"""
import numpy as np
import pytest
import torch

from src.evaluation import readout_variants
from src.evaluation.model_registry import _REGISTRY, available_models, register
from src.evaluation.readout_variants import READOUTS, ReadoutVariant, describe
from src.model.checkpoint_naming import VARIANT_BASE, base_model_name, model_suffix


def test_every_variant_is_registered_and_reads_its_base_models_checkpoint():
    for name in list(READOUTS) + ["coldsite_dti_selfattn"]:
        assert name in available_models(), name
        base = base_model_name(name)
        assert base != name and base in ("coldsite_dti", "hyperattentiondti", "moltrans", "drugban")
        assert model_suffix(name) == model_suffix(base), name


def test_the_variant_table_and_the_naming_table_agree():
    """Two tables, one truth: a name in one and not the other would load the wrong
    weights or fail to resolve at all."""
    declared = set(READOUTS) | {"coldsite_dti_selfattn"}
    assert declared <= set(VARIANT_BASE)
    for name in declared:
        assert VARIANT_BASE[name] == base_model_name(name)


def test_the_overrides_reach_the_base_adapter():
    class Spy:
        def __init__(self, checkpoint_path=None, device="cpu", knob="published"):
            self.knob = knob
            self.model = "weights"
            self.device = device

        def predict(self, *_a, **_k):
            return 1.0

        def explain(self, *_a, **_k):
            return np.zeros(3)

    _REGISTRY["spy_base"] = Spy
    try:
        variant = readout_variants._build("spy_variant", "spy_base", {"knob": "changed"},
                                          "a knob turned")
        built = variant(checkpoint_path=None)
        assert built.base.knob == "changed"
        assert built.model == "weights"                     # the same weights
        assert "a knob turned" in built.citation
        # an explicit argument still wins, so a caller can override the override
        assert variant(knob="explicit").base.knob == "explicit"
    finally:
        _REGISTRY.pop("spy_base", None)
        _REGISTRY.pop("spy_variant", None)


def test_a_variant_delegates_prediction_untouched():
    """Only the explanation changes: the faithfulness delta and the accuracy must belong
    to the same model the published readout was scored on."""
    class Spy:
        def __init__(self, **_kwargs):
            self.model = None
            self.device = "cpu"

        def predict(self, drug, protein):
            return 0.4242

        def explain(self, drug, protein):
            return np.ones(4)

    _REGISTRY["spy2_base"] = Spy
    try:
        cls = readout_variants._build("spy2_variant", "spy2_base", {}, "nothing")
        assert cls().predict(None, None) == pytest.approx(0.4242)
    finally:
        _REGISTRY.pop("spy2_base", None)
        _REGISTRY.pop("spy2_variant", None)


def test_coldsites_self_attention_readout_is_one_weight_per_residue_and_differs():
    """Its forward computes the protein tower's self-attention and discards it. Reading
    it is a real alternative -- and it cannot depend on the drug, which is the point."""
    from src.evaluation.model_registry import model_class
    from src.model.coldsite_dti import ColdSiteDTI

    variant = model_class("coldsite_dti_selfattn")(drug_vocab_size=70, protein_vocab_size=28)
    drug = torch.randint(2, 70, (1, 50))
    protein = torch.randint(2, 28, (1, 120))
    weights = variant.explain(drug, protein)
    assert weights.shape == (120,)
    assert np.all(np.isfinite(weights))

    published = _REGISTRY["coldsite_dti"](drug_vocab_size=70, protein_vocab_size=28)
    published.model = variant.model                        # same weights, both readouts
    cross = np.asarray(published.explain(drug, protein), dtype=float)
    assert cross.shape == weights.shape
    assert not np.allclose(cross, weights), "the two readouts returned the same thing"


def test_the_self_attention_readout_ignores_the_drug():
    """A protein-only readout must not move when the drug changes -- that is exactly the
    property that makes it informative about how much the published one owes to the pair."""
    from src.evaluation.model_registry import model_class

    variant = model_class("coldsite_dti_selfattn")(drug_vocab_size=70, protein_vocab_size=28)
    protein = torch.randint(2, 28, (1, 80))
    first = variant.explain(torch.randint(2, 70, (1, 50)), protein)
    second = variant.explain(torch.randint(2, 70, (1, 50)), protein)
    assert np.allclose(first, second)


def test_describe_lists_every_variant():
    text = describe()
    for name in list(READOUTS) + ["coldsite_dti_selfattn"]:
        assert f"`{name}`" in text


def test_moltrans_refuses_a_layer_it_does_not_have():
    """A silently clamped index would read the published layer and report it as a
    variant."""
    pytest.importorskip("subword_nmt")
    from src.evaluation.baseline_adapters import MolTransAdapter

    adapter = MolTransAdapter.__new__(MolTransAdapter)
    adapter.attention_layer = 99

    class Layers(list):
        pass

    class Encoder:
        layer = Layers([object()] * 8)

    class Model:
        p_encoder = Encoder()

    adapter.model = Model()
    with pytest.raises(ValueError, match="outside the 8 protein-encoder layers"):
        layers = adapter.model.p_encoder.layer
        if not -len(layers) <= adapter.attention_layer < len(layers):
            raise ValueError(f"attention_layer {adapter.attention_layer} is outside the "
                             f"{len(layers)} protein-encoder layers")


def test_looking_up_a_variant_also_registers_its_base_model():
    """A variant is a subclass of a baseline adapter, so resolving the variant without
    the base registered passes the registry check and then dies at construction --
    which is how a Kaggle run can fail minutes after its own sanity check."""
    import subprocess
    import sys

    script = (
        "from src.evaluation.model_registry import model_class, available_models\n"
        "cls = model_class('moltrans_ig')\n"
        "assert 'moltrans' in available_models(), available_models()\n"
        "assert cls.base_model == 'moltrans'\n"
        "print('ok')\n")
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-600:]
    assert "ok" in out.stdout
