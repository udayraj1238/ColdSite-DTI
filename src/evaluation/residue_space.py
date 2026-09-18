"""
Residue-space masking for models that do not tokenise one residue per position.

`faithfulness.py` masks by writing a mask token into columns of the protein
tensor, at indices taken from the explanation. That is only correct when
column j of the tensor *is* residue j and the mask token means "unknown
residue". For ColdSite-DTI both hold: its vocabulary is one token per residue
and token 1 is UNK. For the two audited baselines neither does:

    HyperAttentionDTI  one column per residue, but token 1 is ALANINE in its
                       CHARPROTSET. Masking with 1 would be an alanine scan,
                       not a deletion -- a different intervention from the one
                       ColdSite-DTI receives, measured under the same name.
    MolTrans           ESPF subword tokens. Column j is a token covering
                       several residues, so residue indices from the
                       explanation do not address the tensor at all. Index 0
                       is both padding and a real subword, so there is no
                       token id that means "unknown".

The fix is to mask residues, not tokens, and let each model's own tokeniser
decide what a masked residue becomes. `ResidueSpaceModel` hands
`batch_faithfulness` a tensor with exactly one column per residue the model
saw, in an alphabet where 1 means `X` (unknown amino acid). Its `predict()`
turns that tensor back into a sequence string, re-tokenises it the way the
model's authors do, and asks the real adapter. `X` is in both vendored
alphabets: CHARPROTSET maps it to 24, and ESPF has `X` and `XXXX` units.

For ColdSite-DTI this is exactly what the existing path already does -- UNK at
the token level is `X` at the residue level -- so every audited model now
receives the same intervention: the residue is replaced by an unknown amino
acid, and the model sees the result through its own tokeniser.

Because the drug side is never masked, the "drug" tensor handed to
`batch_faithfulness` is just the pair's index into this object. That keeps
`faithfulness.py` untouched: it only ever passes the drug through to
`predict`.

MolTrans is not deterministic at inference
------------------------------------------
`BIN_Interaction_Flat.forward` calls `F.dropout(i_v, p=self.dropout_rate)`
without `training=self.training`, so dropout on the interaction map stays on
in `eval()` and the same input gives a different score on every call. A
faithfulness comparison is a difference between two predictions; with the
noise left in, comprehensiveness and its random control both measure dropout
as well as masking. `predict()` therefore runs every forward pass under the
same fixed RNG seed, in a forked RNG state so nothing outside is disturbed.
The dropout mask has the same shape every time, so every prediction for a
pair uses the identical mask and the difference between two of them reflects
the input alone. The computation is still the published one, dropout
included. Applied to every model; for the deterministic ones it is a no-op.
"""
from __future__ import annotations

import string
from dataclasses import dataclass

import numpy as np
import torch

PAD_CODE = 0
MASK_CODE = 1          # must stay 1: faithfulness.MASK_TOKEN is the default it writes
MASK_RESIDUE = "X"
_LETTERS = [c for c in string.ascii_uppercase if c != MASK_RESIDUE]
_CODE_OF = {MASK_RESIDUE: MASK_CODE, **{c: i + 2 for i, c in enumerate(_LETTERS)}}
_RESIDUE_OF = {code: residue for residue, code in _CODE_OF.items()}

# Any fixed value works; what matters is that it is the same for every call.
PREDICT_SEED = 0

SUPPORTED_MODELS = ("hyperattentiondti", "moltrans", "drugban")
# Models whose input is sub-word tokens rather than residues, so that masking k residues
# is not a fixed-size intervention -- see control_positions below.
SUBWORD_MODELS = ("moltrans",)
RESIDUE_LEVEL_MODELS = ("hyperattentiondti", "drugban")


def encode_residues(sequence: str) -> torch.Tensor:
    """(1, len) residue codes. Raises on a character outside A-Z."""
    try:
        codes = [_CODE_OF[c] for c in sequence.upper()]
    except KeyError as exc:
        raise ValueError(f"residue {exc.args[0]!r} is not a letter A-Z") from None
    return torch.tensor([codes], dtype=torch.long)


def decode_residues(codes: torch.Tensor) -> str:
    flat = codes[0] if codes.dim() == 2 else codes
    return "".join(_RESIDUE_OF[int(c)] for c in flat if int(c) != PAD_CODE)


@dataclass
class _Pair:
    smiles: str
    sequence: str
    n_seen: int          # residues the model actually saw == len(explanation)


class ResidueSpaceModel:
    """A baseline adapter, re-exposed so `batch_faithfulness` can mask residues.

        wrapped = ResidueSpaceModel(adapter, "moltrans")
        drug, protein, attention = wrapped.add_pair(smiles, sequence)
        ...
        batch_faithfulness(wrapped, drugs, proteins, attentions, ...)
    """

    def __init__(self, adapter, model_name: str, device: str = "cpu"):
        # An explanation variant is the same trained model with a different explainer,
        # so it is tokenised and masked exactly like its base.
        from src.model.checkpoint_naming import base_model_name
        model_name = base_model_name(model_name)
        if model_name not in SUPPORTED_MODELS:
            raise ValueError(
                f"{model_name!r} has no residue-space tokenisation. Supported: "
                f"{SUPPORTED_MODELS}. ColdSite-DTI does not need one -- its "
                f"tensor is already one token per residue with UNK = 1.")
        self.adapter = adapter
        self.model_name = model_name
        self.device = str(device)
        self.pairs: list[_Pair] = []

    # -- tokenising the way each model's authors do -------------------------

    def _encode(self, smiles: str, sequence: str):
        if self.model_name in RESIDUE_LEVEL_MODELS:
            # One token per residue for both, so masking a residue to X is a
            # one-token intervention and the uniform random control is already
            # size-matched (X is in DrugBAN's 25-letter table, index 24, and in
            # HyperAttentionDTI's) -- no token matching needed, unlike MolTrans.
            drug, protein = type(self.adapter).encode(smiles, sequence)
            return {"drug": drug, "protein": protein}

        drug, drug_mask, protein, protein_mask, tokens = type(self.adapter).encode(
            smiles, sequence)
        # stream.protein2emb_encoder swallows an unknown subword by returning
        # the single token [0] for the WHOLE protein. Masking can in principle
        # produce a subword the index lacks, and the prediction would then be
        # for an empty protein -- a large, entirely spurious comprehensiveness.
        if int(protein_mask.sum()) == 1 and int(protein[0]) == 0 and len(sequence) > 1:
            raise RuntimeError(
                "MolTrans's ESPF encoder collapsed this sequence to a single "
                "token -- it contains a subword missing from "
                "subword_units_map_uniprot.csv. Refusing to score a prediction "
                "made on an empty protein.")
        return {"drug": drug, "drug_mask": drug_mask, "protein": protein,
                "protein_mask": protein_mask, "tokens": tokens}

    def _forward(self, encoded) -> float:
        devices = []
        if self.device.startswith("cuda") and torch.cuda.is_available():
            devices = [torch.device(self.device).index or 0]
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(PREDICT_SEED)
            if self.model_name in RESIDUE_LEVEL_MODELS:
                return float(self.adapter.predict(encoded["drug"], encoded["protein"]))
            return float(self.adapter.predict(
                encoded["drug"], encoded["protein"],
                drug_mask=encoded["drug_mask"], protein_mask=encoded["protein_mask"]))

    # -- the two things batch_faithfulness needs ----------------------------

    def add_pair(self, smiles: str, sequence: str, max_len: int | None = None):
        """Explain one pair. Returns (drug, protein, attention) for batch_faithfulness.

        The protein tensor is cut to exactly `len(attention)` residues, so that
        `random_control` samples from the same residues the explanation was
        scored over. Residues past that point are re-attached unchanged by
        `predict`, so the model's input differs from the original only where a
        residue was masked.

        `max_len` cuts the explanation to the evaluation window, as `collect`
        does for plausibility: MolTrans can read ~1,400 residues, and without
        the cut its faithfulness top-k could include residues its plausibility
        top-k never competes for, so the two axes would describe different
        explanations.
        """
        sequence = str(sequence).upper()
        encoded = self._encode(smiles, sequence)
        if self.model_name in RESIDUE_LEVEL_MODELS:
            attention = self.adapter.explain(encoded["drug"], encoded["protein"])
        else:
            attention = self.adapter.explain(
                encoded["drug"], encoded["protein"],
                protein_tokens=encoded["tokens"],
                drug_mask=encoded["drug_mask"], protein_mask=encoded["protein_mask"])
        attention = np.asarray(attention, dtype=float)
        if max_len is not None:
            attention = attention[:max_len]

        n_seen = int(attention.size)
        if not 0 < n_seen <= len(sequence):
            raise RuntimeError(
                f"explanation has {n_seen} weights for a {len(sequence)}-residue "
                f"sequence -- the adapter's projection is misaligned")

        index = len(self.pairs)
        self.pairs.append(_Pair(str(smiles), sequence, n_seen))
        drug = torch.tensor([[index]], dtype=torch.long)
        return drug, encode_residues(sequence[:n_seen]), attention

    def control_positions(self, drug, protein, k: int, rng, attended):
        """Random positions whose masking changes about as many TOKENS as `attended`.

        Only for a model whose input is sub-word tokens. Masking a residue there
        re-segments the protein, so k scattered residues are a far larger intervention
        than k attended ones (48% of MolTrans's tokens versus 95%, measured over 20 DAVIS
        proteins) and the faithfulness delta ends up measuring the intervention. Returns
        None for a residue-level model, where k residues is k tokens either way and the
        caller's uniform draw is already the right control.
        """
        if self.model_name not in SUBWORD_MODELS:
            return None
        from src.evaluation.mask_comparability import (protein_token_encoder,
                                                       token_change_fraction,
                                                       token_matched_control)
        pair = self.pairs[int(torch.as_tensor(drug).reshape(-1)[0])]
        encode = protein_token_encoder(type(self.adapter))
        target = token_change_fraction(encode, pair.sequence, attended, pair.smiles)
        positions, _fraction, _tries = token_matched_control(
            encode, pair.sequence, k, target, rng, pair.smiles)
        return positions

    def predict(self, drug, protein) -> float:
        pair = self.pairs[int(torch.as_tensor(drug).reshape(-1)[0])]
        seen = decode_residues(torch.as_tensor(protein))
        if len(seen) != pair.n_seen:
            raise RuntimeError(
                f"got {len(seen)} residues back for a pair that was explained "
                f"over {pair.n_seen} -- the masked tensor lost or gained columns")
        sequence = seen + pair.sequence[pair.n_seen:]
        return self._forward(self._encode(pair.smiles, sequence))
