"""DrugBAN as an audit subject: the parts that decide whether its numbers mean anything.

DrugBAN (Bai et al., Nature Machine Intelligence 2023) is the current-generation subject,
added 2026-09-18 so the audit's verdict is not only about 2020-2022 models. Its published
explanation is a 2D map over (drug atom, convolution position), so turning it into one
weight per residue takes two reductions and a projection that the published code does not
contain. These tests pin the three things that would silently produce a wrong number:
the reduction, the residue count, and the geometry of the convolution stack.

DGL has no macOS-ARM wheel on PyPI, so anything needing the real model is skipped unless
the vendored repo and DGL are both present.
"""
import numpy as np
import pytest
import torch

from src.evaluation.attention_projection import conv_output_length, project_conv_attention
from src.evaluation.drugban_adapter import (DRUGBAN_MAX_PROTEIN_LEN,
                                            DRUGBAN_PROTEIN_KERNELS, _normalise,
                                            _real_residue_count, _reduce)


def test_the_reduction_is_the_one_named():
    att = np.array([[1.0, 5.0], [3.0, 1.0]])
    assert list(_reduce(att, "mean", axis=0)) == [2.0, 3.0]
    assert list(_reduce(att, "max", axis=0)) == [3.0, 5.0]
    assert list(_reduce(att, "sum", axis=0)) == [4.0, 6.0]
    with pytest.raises(ValueError, match="unknown reduction"):
        _reduce(att, "median", axis=0)


def test_the_map_is_normalised_the_way_their_own_layer_would():
    """BANLayer returns bilinear LOGITS (its softmax argument defaults to False and
    DrugBAN.forward never passes it), so they can be negative. Their softmax=True branch
    normalises over the flattened atom-by-position map, per head -- which is what makes
    the map a joint distribution whose marginal over atoms is a per-residue weight."""
    att = np.array([[[-3.0, 1.0], [0.0, 2.0]],
                    [[5.0, 5.0], [5.0, 5.0]]])          # 2 heads, 2 atoms, 2 positions
    p = _normalise(att)
    assert p.shape == att.shape
    assert (p >= 0).all()
    assert p[0].sum() == pytest.approx(1.0)             # each head sums to one
    assert p[1].sum() == pytest.approx(1.0)
    assert p[1].ravel() == pytest.approx([0.25] * 4)    # equal logits -> uniform
    # order is preserved within a head: the largest logit keeps the largest weight
    assert np.argmax(p[0]) == np.argmax(att[0])


def test_normalisation_is_shift_invariant_so_a_large_logit_cannot_overflow():
    att = np.array([[[1000.0, 1001.0]]])
    p = _normalise(att)
    assert np.isfinite(p).all()
    assert p.sum() == pytest.approx(1.0)


def test_residue_count_is_measured_to_the_last_residue_not_counted():
    """integer_label_protein writes 0 for padding AND for any character outside its
    25-letter table. A sequence with an interior unknown residue therefore has interior
    zeros; counting non-zeros would return a length shorter than the protein spans and
    shift every ground-truth index after that point."""
    encoded = torch.zeros(1, DRUGBAN_MAX_PROTEIN_LEN, dtype=torch.long)
    encoded[0, :10] = torch.tensor([1, 2, 3, 4, 0, 6, 7, 8, 9, 10])   # one unknown at 4
    assert _real_residue_count(encoded) == 10        # not 9


def test_a_full_length_protein_is_cut_to_what_the_model_encodes():
    encoded = torch.ones(1, DRUGBAN_MAX_PROTEIN_LEN + 500, dtype=torch.long)
    assert _real_residue_count(encoded) == DRUGBAN_MAX_PROTEIN_LEN


def test_the_map_is_shorter_than_the_sequence_by_the_stack_geometry():
    """Three valid convolutions with kernels 3, 6, 9 shorten the sequence by 2 + 5 + 8,
    so a weight at position j did not come from residue j. Projecting with the wrong
    kernel sizes would shift the explanation along the protein."""
    assert conv_output_length(1000, kernel_sizes=DRUGBAN_PROTEIN_KERNELS) == 1000 - 15


def test_projection_returns_one_weight_per_residue_seen():
    real_length = 300
    positions = conv_output_length(real_length, kernel_sizes=DRUGBAN_PROTEIN_KERNELS)
    weights = np.zeros(positions)
    weights[100] = 1.0
    projected = project_conv_attention(weights, real_length,
                                       kernel_sizes=DRUGBAN_PROTEIN_KERNELS,
                                       mode="centre",
                                       max_input_len=DRUGBAN_MAX_PROTEIN_LEN)
    assert projected.shape == (real_length,)
    assert projected.min() >= 0
    # the weight lands inside the window that position 100 actually saw
    assert 100 <= int(np.argmax(projected)) <= 100 + 15


def test_a_protein_longer_than_the_encoding_still_projects_to_the_encoded_part():
    real_length = DRUGBAN_MAX_PROTEIN_LEN
    positions = conv_output_length(real_length, kernel_sizes=DRUGBAN_PROTEIN_KERNELS)
    projected = project_conv_attention(np.ones(positions), real_length,
                                       kernel_sizes=DRUGBAN_PROTEIN_KERNELS,
                                       max_input_len=DRUGBAN_MAX_PROTEIN_LEN)
    assert projected.shape == (DRUGBAN_MAX_PROTEIN_LEN,)


# ---------------------------------------------------------------------------
# the real model, when the machine can load it
# ---------------------------------------------------------------------------

def _real_model_available():
    import importlib.util
    import os
    if importlib.util.find_spec("dgl") is None or importlib.util.find_spec("dgllife") is None:
        return False
    return os.path.isdir(os.path.join("baselines", "DrugBAN"))


requires_dgl = pytest.mark.skipif(not _real_model_available(),
                                  reason="DGL or the vendored DrugBAN is absent")


@requires_dgl
def test_registered_and_satisfies_the_adapter_contract():
    from src.evaluation.model_registry import get_model, validate_adapter

    model = get_model("drugban")
    sequence = "MKKFFDSRREQGGSGLGSGSSGGGGSTSGLGSGYIGRVFGIGRQQVTVDEVLAEGGFAIVFLV"
    drug, protein = type(model).encode("CCO", sequence)
    report = validate_adapter(model, drug, protein, expected_length=len(sequence))
    assert report["valid"], report["problems"]
    assert report["n_weights"] == len(sequence)


@requires_dgl
def test_the_explanation_changes_with_the_drug():
    """The property EviDTI's published map lacks by construction (Results 7e): a
    drug-target explanation must depend on the drug. DrugBAN's bilinear map does."""
    from src.evaluation.model_registry import get_model

    model = get_model("drugban")
    sequence = "MKKFFDSRREQGGSGLGSGSSGGGGSTSGLGSGYIGRVFGIGRQQVTVDEVLAEGGFAIVFLV"
    first = model.explain(*type(model).encode("CCO", sequence))
    second = model.explain(*type(model).encode("c1ccccc1C(=O)Nc1ccccc1", sequence))
    assert first.shape == second.shape == (len(sequence),)
    assert not np.allclose(first, second), "attention did not move with the drug"


def test_an_empty_training_epoch_says_why_rather_than_failing_inside_numpy():
    """drop_last on a split smaller than one batch leaves zero batches, and the epoch
    used to die in np.concatenate with 'need at least one array to concatenate' --
    which names neither the split, the batch size, nor the fix. Found 2026-09-18 while
    validating the Kaggle notebook's command line against a 60-row split at batch 64."""
    from src.model.train_drugban import run_epoch

    class EmptyLoader:
        dataset = range(60)
        batch_size = 64
        drop_last = True

        def __len__(self):
            return 0

        def __iter__(self):
            return iter(())

    class Model:
        def train(self, mode=True):
            return self

    with pytest.raises(ValueError, match="no batches"):
        run_epoch(Model(), EmptyLoader(), None, "cpu", label="epoch 1 train")
