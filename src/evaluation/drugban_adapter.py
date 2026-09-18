"""DrugBAN (Bai et al., *Nature Machine Intelligence* 2023) as an audit subject.

Why this model is here
----------------------
The three models audited on DAVIS are from 2020-2022, and the first thing a reviewer asks
of an audit is whether its verdict binds on what people build now. DrugBAN is the
current-generation subject: a bilinear attention network whose title claims
interpretability, which reports cross-domain (cluster-split) generalisation, and whose
code is maintained and MIT-licensed. If its attention marks binding residues under shift,
the audit's verdict is dated; if it does not, the verdict reaches the present.

What its explanation is, and the choice we have to make
-------------------------------------------------------
`BANLayer` returns `att_maps` of shape (batch, heads, drug_atoms, protein_positions) --
a two-dimensional map over atom-residue pairs, which is what the paper visualises. The
audit needs one weight per residue, so the map must be reduced over heads and over drug
atoms, and then projected from convolution positions back to residues. Neither step is
in the published code, because the published artefact is the 2D picture.

`BANLayer.forward` returns those maps as **pre-softmax logits** (its `softmax` argument
defaults to False and `DrugBAN.forward` does not pass it), so they can be negative and are
not weights at all until normalised. Their own layer contains the normalisation it would
apply -- `p = softmax(att_maps.view(-1, h_out, v_num * q_num), 2)`, i.e. a softmax over
the flattened atom-by-position map -- so this adapter applies exactly that, per head. The
result is a joint distribution over (atom, residue-position) pairs, and summing it over
the drug's atoms gives the marginal over protein positions: one non-negative weight per
position, which is what the audit's contract asks for and what a reader takes an
"attention weight" to be.

Both remaining choices are recorded rather than hidden, and both are variables:

* `head_reduce` -- "mean" (default) or "max", over the two BAN heads.
* `atom_reduce` -- "sum" (default: the marginal) or "max" (the single best-matching atom).
* `projection_mode` -- how a convolution position's weight is spread over the residues in
  its receptive field; the same machinery HyperAttentionDTI uses
  (`src/evaluation/attention_projection.project_conv_attention`).

DrugBAN's protein tower is three valid convolutions with kernels 3, 6 and 9, so position
i of the map covers residues i..i+15 and the map is 15 positions shorter than the
sequence. Padding positions carry weight too -- the tower runs over the padded 1,200-long
encoding -- so the projection is given the REAL residue count and the rest is dropped.

Dependencies are imported lazily: DGL has no macOS-ARM wheel on PyPI, and the rest of the
audit must keep running on a machine that cannot install it.
"""
from __future__ import annotations

import os

import numpy as np

from src.evaluation.baseline_adapters import _as_batch, _vendored
from src.evaluation.model_registry import ExplainableDTIModel, register

# Their config: PROTEIN.KERNEL_SIZE = [3, 6, 9], PROTEIN.PADDING = True, BCN.HEADS = 2,
# DRUG.MAX_NODES = 290, and utils.integer_label_protein's max_length = 1200.
DRUGBAN_PROTEIN_KERNELS = (3, 6, 9)
DRUGBAN_MAX_PROTEIN_LEN = 1200
DRUGBAN_MAX_DRUG_NODES = 290


def _config():
    """Their default config object, so the architecture is theirs, not ours."""
    _vendored("DrugBAN", DrugBANAdapter.clone_hint)
    from configs import get_cfg_defaults  # noqa: E402

    cfg = get_cfg_defaults()
    cfg.DECODER.BINARY = 1
    return cfg


@register("drugban")
class DrugBANAdapter(ExplainableDTIModel):
    """predict() -> the binary head's logit; explain() -> one weight per residue."""

    provides_attention = True
    citation = "Bai et al., Nature Machine Intelligence 2023"
    repo_url = "https://github.com/peizhenbai/DrugBAN"
    clone_hint = ("cd baselines && git clone https://github.com/peizhenbai/DrugBAN.git "
                  "DrugBAN")

    def __init__(self, checkpoint_path: str = None, device: str = "cpu",
                 head_reduce: str = "mean", atom_reduce: str = "sum",
                 projection_mode: str = "centre"):
        import torch

        _vendored("DrugBAN", self.clone_hint)
        from models import DrugBAN  # noqa: E402

        self.device = device
        self.checkpoint_path = checkpoint_path
        self.head_reduce = head_reduce
        self.atom_reduce = atom_reduce
        self.projection_mode = projection_mode

        self.model = DrugBAN(**_config())
        if checkpoint_path:
            state = torch.load(checkpoint_path, map_location=device, weights_only=False)
            self.model.load_state_dict(state.get("model_state", state))
        self.model.to(device).eval()

    # ------------------------------------------------------------------
    # encoding: their featuriser, their integer encoding, their padding
    # ------------------------------------------------------------------

    @staticmethod
    def encode(smiles: str, sequence: str):
        """(drug graph, protein tensor) exactly as their DTIDataset builds them.

        The drug graph is padded to 290 nodes with virtual nodes, as in their
        `dataloader.DTIDataset.__getitem__`; without that padding the bilinear map's
        atom axis would vary in length between pairs and the reduction over atoms would
        not mean the same thing twice.
        """
        import dgl
        import torch
        from dgllife.utils import CanonicalAtomFeaturizer, smiles_to_bigraph

        _vendored("DrugBAN", DrugBANAdapter.clone_hint)
        from utils import integer_label_protein  # noqa: E402

        graph = smiles_to_bigraph(smiles=smiles, node_featurizer=CanonicalAtomFeaturizer(),
                                  add_self_loop=True)
        if graph is None:
            raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
        actual = graph.num_nodes()
        virtual = DRUGBAN_MAX_DRUG_NODES - actual
        if virtual < 0:
            raise ValueError(
                f"{actual} atoms exceeds DrugBAN's DRUG.MAX_NODES = "
                f"{DRUGBAN_MAX_DRUG_NODES}; their dataloader cannot encode this drug")
        graph = graph.add_self_loop()
        graph.ndata["h"] = torch.cat(
            (graph.ndata["h"], torch.zeros(actual, 1)), 1)
        virtual_feats = torch.zeros(virtual, graph.ndata["h"].shape[1])
        virtual_feats[:, -1] = 1
        graph.add_nodes(virtual, {"h": virtual_feats})
        graph = graph.add_self_loop()

        protein = torch.from_numpy(
            integer_label_protein(sequence, DRUGBAN_MAX_PROTEIN_LEN)).long()
        return graph, protein

    # ------------------------------------------------------------------
    # the contract
    # ------------------------------------------------------------------

    def _forward(self, drug, protein):
        import dgl
        import torch

        graph = drug if not isinstance(drug, (list, tuple)) else drug[0]
        batched = dgl.batch([graph]).to(self.device)
        v_p = _as_batch(protein).to(self.device)
        with torch.no_grad():
            _v_d, _v_p, score, att = self.model(batched, v_p, mode="eval")
        return score, att

    def predict(self, drug, protein) -> float:
        score, _att = self._forward(drug, protein)
        values = score.reshape(-1)
        if values.numel() != 1:
            raise RuntimeError(
                f"DrugBAN returned {values.numel()} scores for one pair; expected 1 "
                f"(DECODER.BINARY = 1). Its head is a single logit, so this is the "
                f"log-odds already -- no difference of two logits to take.")
        return float(values.item())

    def explain(self, drug, protein) -> np.ndarray:
        """One weight per residue, over the residues the model actually saw."""
        from src.evaluation.attention_projection import project_conv_attention

        _score, att = self._forward(drug, protein)
        att = _normalise(att.detach().cpu().numpy()[0])   # (heads, atoms, positions)
        att = _reduce(att, self.head_reduce, axis=0)      # (atoms, positions)
        weights = _reduce(att, self.atom_reduce, axis=0)  # (positions,)

        real_length = _real_residue_count(protein)
        if real_length == 0:
            raise ValueError("protein tensor is entirely padding")
        return project_conv_attention(weights, real_length,
                                      kernel_sizes=DRUGBAN_PROTEIN_KERNELS,
                                      mode=self.projection_mode,
                                      max_input_len=DRUGBAN_MAX_PROTEIN_LEN)


def _normalise(att: np.ndarray) -> np.ndarray:
    """Their own normalisation: a softmax per head over the flattened atom-by-position
    map (`BANLayer.forward`'s `softmax=True` branch), which `DrugBAN.forward` never
    reaches because it calls the layer with the default. Without it the "attention" is a
    bilinear logit that can be negative, and negative weights are not an explanation --
    the audit's contract rejects them, and precision@k over them would rank a residue the
    model scored -3 above one it scored -5 while calling both "attended"."""
    heads = att.shape[0]
    flat = att.reshape(heads, -1)
    flat = flat - flat.max(axis=1, keepdims=True)        # stable, softmax is shift-invariant
    exp = np.exp(flat)
    return (exp / exp.sum(axis=1, keepdims=True)).reshape(att.shape)


def _reduce(array: np.ndarray, how: str, axis: int) -> np.ndarray:
    if how == "mean":
        return array.mean(axis=axis)
    if how == "max":
        return array.max(axis=axis)
    if how == "sum":
        return array.sum(axis=axis)
    raise ValueError(f"unknown reduction {how!r}; expected 'mean', 'max' or 'sum'")


def _real_residue_count(protein) -> int:
    """Residues the model saw: measured to the LAST non-pad position, not counted.

    `integer_label_protein` writes 0 for padding AND for any character outside its
    25-letter table, logging a warning and calling it padding. A sequence with an
    interior unknown residue therefore has interior zeros, and counting non-zeros would
    return fewer residues than the protein spans -- shifting every ground-truth index
    after that point. The same reasoning as `protein_encoder.real_lengths`, which this
    delegates to.
    """
    import torch

    from src.model.protein_encoder import real_lengths

    flat = torch.as_tensor(protein).reshape(1, -1).long()
    return min(int(real_lengths(flat)[0].item()), DRUGBAN_MAX_PROTEIN_LEN)
