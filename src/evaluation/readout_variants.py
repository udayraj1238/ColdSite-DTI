"""
The same attention, read out a different way — the sensitivity check Limitations promises.

Why
---
"The model's attention" is not a single object. Between the tensor inside the network and
the one number per residue that precision@k scores, somebody chooses: which axis to
average over, which layer to read, how to spread a convolution position or a sub-word
token across the residues it covers. Every one of those choices was made once, early, and
every result in the audit rests on them.

A reviewer is entitled to ask whether the verdict is a property of the model or of our
choices. This module answers it by scoring the same checkpoints through alternative
readouts that another author could reasonably have picked:

    hyperattentiondti_maxchannel   channel max instead of mean: does one channel's
                                   sharp preference matter more than the average?
    hyperattentiondti_receptive    a convolution position's weight spread over the
                                   residues in its receptive field, rather than placed
                                   on the centre residue
    moltrans_maxhead               head max instead of mean over the attention heads
    moltrans_firstlayer            the first protein-encoder layer rather than the last
    moltrans_interaction           the drug x protein interaction map MolTrans's own
                                   paper visualises (Fig. 3), rather than the protein
                                   encoder's self-attention, which cannot depend on
                                   the drug at all
    drugban_maxhead                bilinear-head max instead of mean
    drugban_maxatom                max over drug atoms instead of their sum
    drugban_receptive              receptive-field projection, as for HyperAttentionDTI
    coldsite_dti_selfattn          the protein tower's own self-attention, which its
                                   forward pass computes and discards, instead of the
                                   drug-conditioned cross-attention

Each changes exactly one documented choice, so a difference is attributable. They are
registered as models in their own right, reading the same checkpoints, so every ladder,
null, faithfulness run and audit applies unchanged:

    python -m src.evaluation.run_ladder --model moltrans_maxhead ...

What counts as passing
----------------------
The readouts are not expected to agree to three decimals. The claim they have to support
is the verdict: at chance against UniProt's residues, above chance against the KLIFS
pocket, and the cold levels below the random level. A variant that reverses one of those
is a finding about the fragility of attention explanations and belongs in the paper --
not a bug to be tuned away.
"""
from __future__ import annotations

import numpy as np

from src.evaluation.model_registry import _REGISTRY, ExplainableDTIModel, register

# variant name -> (base model, constructor overrides, one-line description)
READOUTS = {
    "hyperattentiondti_maxchannel": (
        "hyperattentiondti", {"channel_reduce": "max"},
        "channel max instead of mean"),
    "hyperattentiondti_receptive": (
        "hyperattentiondti", {"projection_mode": "receptive_field"},
        "convolution weight spread over its receptive field, not placed on the centre"),
    "moltrans_maxhead": (
        "moltrans", {"head_reduce": "max"},
        "attention-head max instead of mean"),
    "moltrans_firstlayer": (
        "moltrans", {"attention_layer": 0},
        "the first protein-encoder layer instead of the last"),
    "moltrans_interaction": (
        "moltrans", {"explanation": "interaction"},
        "the drug x protein interaction map their paper visualises, max over drug "
        "substructures, instead of the protein encoder's self-attention"),
    "moltrans_interaction_sum": (
        "moltrans", {"explanation": "interaction", "interaction_reduce": "sum"},
        "the same interaction map, summed over drug substructures"),
    "drugban_maxhead": (
        "drugban", {"head_reduce": "max"},
        "bilinear-head max instead of mean"),
    "drugban_maxatom": (
        "drugban", {"atom_reduce": "max"},
        "the strongest drug atom per position instead of the sum over atoms"),
    "drugban_receptive": (
        "drugban", {"projection_mode": "receptive_field"},
        "convolution weight spread over its receptive field, not placed on the centre"),
}


class ReadoutVariant(ExplainableDTIModel):
    """A registered model with one readout choice overridden.

    `predict` and `explain` are the base adapter's own: only the constructor arguments
    differ, which is what keeps the comparison honest -- same weights, same tokenisation,
    same prediction, one documented choice changed.
    """

    base_model: str = ""
    overrides: dict = {}
    description: str = ""

    def __init__(self, *args, **kwargs):
        merged = dict(self.overrides)
        merged.update(kwargs)          # an explicit argument still wins
        self.base = _REGISTRY[self.base_model](*args, **merged)
        self.device = getattr(self.base, "device", "cpu")
        self.model = self.base.model
        self.citation = f"{self.base_model}, {self.description}"

    @classmethod
    def encode(cls, *args, **kwargs):
        return _REGISTRY[cls.base_model].encode(*args, **kwargs)

    def predict(self, *args, **kwargs) -> float:
        return self.base.predict(*args, **kwargs)

    def explain(self, *args, **kwargs) -> np.ndarray:
        return self.base.explain(*args, **kwargs)


def _build(name: str, base: str, overrides: dict, description: str):
    cls = type(name.title().replace("_", ""), (ReadoutVariant,),
               {"base_model": base, "overrides": overrides, "description": description,
                "__doc__": f"{base}, read out with {description}."})
    return register(name)(cls)


for _name, (_base, _overrides, _description) in READOUTS.items():
    _build(_name, _base, _overrides, _description)


@register("coldsite_dti_selfattn")
class ColdSiteSelfAttention(ReadoutVariant):
    """ColdSite-DTI explained by its protein tower's self-attention.

    `ColdSiteDTI.forward` computes the protein encoder's self-attention and throws it
    away, keeping the drug-conditioned cross-attention as the explanation. The discarded
    one is a real alternative: it says which residues the protein representation attends
    to before the drug is consulted. It cannot depend on the drug -- which is the point.
    A protein-only readout that scores as well as the cross-attention would mean the
    reported explanation owes nothing to the pair.
    """

    base_model = "coldsite_dti"
    overrides: dict = {}
    description = "protein self-attention, not drug-conditioned cross-attention"

    def explain(self, drug, protein) -> np.ndarray:
        import torch

        from src.model.protein_encoder import real_lengths

        batch = protein if torch.as_tensor(protein).dim() == 2 else torch.as_tensor(protein)[None]
        batch = torch.as_tensor(batch).to(self.device)
        self.model.eval()
        with torch.no_grad():
            _sequence, self_attention = self.model.protein_encoder(batch)
        if self_attention is None:
            raise RuntimeError("this protein encoder returned no self-attention weights")
        weights = torch.as_tensor(self_attention).float().cpu()
        # (batch, len, len) -> how much attention each residue RECEIVES, averaged over
        # the residues doing the attending; (batch, len) is taken as already reduced.
        if weights.dim() == 3:
            weights = weights.mean(dim=1)
        length = int(real_lengths(batch)[0])
        return np.asarray(weights[0, :length], dtype=float)


def describe() -> str:
    lines = ["# Alternative attention readouts", "",
             "| variant | base model | what changes |", "|---|---|---|"]
    for name in sorted(_REGISTRY):
        cls = _REGISTRY[name]
        if isinstance(cls, type) and issubclass(cls, ReadoutVariant):
            lines.append(f"| `{name}` | {cls.base_model} | {cls.description} |")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
