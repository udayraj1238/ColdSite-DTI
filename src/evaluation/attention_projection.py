"""
Mapping a model's internal attention positions back to REAL residue indices.

Track B (124AD0015) contribution to the baseline adapters that 124AD0008 owns.
This module is the part of that work that is error-prone and testable; the
adapters themselves — checkpoint loading, tokenisation, `predict()` — stay in
`baseline_adapters.py`.

Why this exists
---------------
`model_registry.ExplainableDTIModel.explain` must return exactly one weight per
real residue, 0-indexed. Neither audited attention model produces that natively:

    HyperAttentionDTI   attention lives on CONVOLUTION positions. A stack of
                        [4, 8, 12] kernels turns 1000 residues into 979
                        positions, each one a window over 22 residues.
    MolTrans            attention lives on ESPF SUBWORD tokens. One token spans
                        several residues, and 545 tokens is not 545 residues.

Handing either array straight to `precision_at_k` does not raise. It returns a
number of the wrong length, or the right length indexed by the wrong thing, and
every ground-truth position past the first discrepancy is scored against the
wrong residue. That is the single failure mode `validate_adapter` exists to
catch, and it is why the projection lives in one tested place rather than being
re-derived inside each adapter.

Padding is not a detail here
----------------------------
Both models pad to a fixed width (1000 residues / 545 tokens). A convolution
position whose receptive field reaches into padding is reading zeros, not
protein. Those positions are dropped rather than projected, because a weight
computed over padding is not evidence about any residue.

Open choices, flagged not guessed
---------------------------------
Two projections are genuinely defensible and they give different numbers.
Both are implemented; the default is stated in each function's docstring and
must be recorded in Methods. See `PROJECTION_DECISIONS` at the bottom.
"""
from __future__ import annotations

import numpy as np

# HyperAttentionDTI, baselines/HpyerAttentionDTI/hyperparameter.py
HYPERATTENTION_PROTEIN_KERNELS = (4, 8, 12)
HYPERATTENTION_DRUG_KERNELS = (4, 6, 8)
HYPERATTENTION_MAX_PROTEIN_LEN = 1000

# MolTrans, baselines/MolTrans/stream.py::protein2emb_encoder
MOLTRANS_MAX_PROTEIN_TOKENS = 545


# ---------------------------------------------------------------------------
# convolution geometry
# ---------------------------------------------------------------------------

def conv_stack_geometry(kernel_sizes=HYPERATTENTION_PROTEIN_KERNELS) -> dict:
    """Receptive field of a stack of stride-1, unpadded Conv1d layers.

    Each layer of kernel k shortens the sequence by (k - 1) and widens the
    receptive field by the same amount, so for [4, 8, 12]:

        offset = 3 + 7 + 11 = 21
        width  = 22 residues per output position
        output position j covers input residues [j, j + 21]

    Verified against the vendored model: a (B, 64, 1000) protein embedding
    comes out of `Protein_CNNs` as (B, 160, 979), and 1000 - 21 = 979.
    """
    offset = int(sum(k - 1 for k in kernel_sizes))
    return {
        "kernel_sizes": tuple(kernel_sizes),
        "offset": offset,           # last residue of position 0's window
        "width": offset + 1,        # residues seen by one output position
        "centre_offset": offset // 2,
    }


def conv_output_length(input_length: int, kernel_sizes=HYPERATTENTION_PROTEIN_KERNELS) -> int:
    """Length of the convolution output for an input of `input_length`.

    1000 residues -> 979 positions under [4, 8, 12].
    """
    out = int(input_length) - conv_stack_geometry(kernel_sizes)["offset"]
    return max(out, 0)


def min_length_for_conv_stack(kernel_sizes=HYPERATTENTION_PROTEIN_KERNELS) -> int:
    """Shortest protein with at least one convolution position over real residues."""
    return conv_stack_geometry(kernel_sizes)["width"]


def valid_conv_positions(real_length: int,
                         kernel_sizes=HYPERATTENTION_PROTEIN_KERNELS) -> int:
    """How many convolution positions see only real residues, never padding.

    Position j's window is [j, j + offset], so it is padding-free exactly when
    j + offset <= real_length - 1. A 300-residue protein padded to 1000 has 979
    convolution positions but only 279 that mean anything.
    """
    return max(int(real_length) - conv_stack_geometry(kernel_sizes)["offset"], 0)


def project_conv_attention(conv_weights, real_length: int,
                           kernel_sizes=HYPERATTENTION_PROTEIN_KERNELS,
                           mode: str = "centre",
                           max_input_len: int = HYPERATTENTION_MAX_PROTEIN_LEN) -> np.ndarray:
    """Convolution-position attention -> one weight per real residue.

    conv_weights   1D, one scalar per convolution position (length 979 for a
                   1000-wide padded input). Reduce channels and the drug axis
                   before calling; see `hyperattentiondti_protein_attention`.
    real_length    residues the model actually saw, i.e.
                   min(len(sequence), max_input_len). Padding is excluded here,
                   not by the caller.

    Returns exactly `min(real_length, max_input_len)` weights, 0-indexed,
    non-negative — the `explain()` contract.

    mode='centre' (default)
        Position j's weight lands on its central residue, j + offset//2.
        Residues in the first and last ~10 positions are never a centre and get
        0. Sharp: a pocket that the model localised stays localised.

    mode='receptive_field'
        Position j's weight is spread over all 22 residues it saw, and each
        residue averages the positions covering it. More literally faithful to
        what the convolution computed, but it blurs any weight across a
        22-residue window, which mechanically lowers precision@10.

    'centre' is the default deliberately. This is an audit: a projection that
    smears a sharp signal would manufacture the degradation the paper is
    looking for. The favourable-to-the-model choice keeps a negative result
    attributable to the model rather than to us. Record whichever is used.
    """
    if mode not in ("centre", "receptive_field"):
        raise ValueError(
            f"mode must be 'centre' or 'receptive_field', got {mode!r}")

    conv_weights = np.asarray(conv_weights, dtype=float).ravel()
    geometry = conv_stack_geometry(kernel_sizes)
    offset = geometry["offset"]

    real_length = int(real_length)
    if real_length <= 0:
        raise ValueError(f"real_length must be positive, got {real_length}")
    # A protein longer than the model's window is truncated by the model, so
    # the explanation covers the truncated length -- never the full sequence.
    out_length = min(real_length, int(max_input_len))

    if out_length < geometry["width"]:
        raise ValueError(
            f"protein of {out_length} residues is shorter than the "
            f"{geometry['width']}-residue receptive field of kernels "
            f"{tuple(kernel_sizes)}; no convolution position sees only real "
            f"residues, so no per-residue explanation exists. Skip this "
            f"protein rather than scoring it.")

    n_valid = valid_conv_positions(out_length, kernel_sizes)
    if conv_weights.size < n_valid:
        raise ValueError(
            f"got {conv_weights.size} convolution weights but a "
            f"{out_length}-residue protein needs at least {n_valid}. The "
            f"attention was probably reduced along the wrong axis.")

    # Everything past n_valid reads padding. Dropping is not optional: those
    # weights are computed over zeros and carry no evidence about any residue.
    usable = conv_weights[:n_valid]
    residues = np.zeros(out_length, dtype=float)

    if mode == "centre":
        centres = np.arange(n_valid) + geometry["centre_offset"]
        residues[centres] = usable
        return residues

    counts = np.zeros(out_length, dtype=float)
    for j, weight in enumerate(usable):
        residues[j:j + offset + 1] += weight
        counts[j:j + offset + 1] += 1.0
    covered = counts > 0
    residues[covered] /= counts[covered]
    return residues


# ---------------------------------------------------------------------------
# subword-token geometry (MolTrans / ESPF)
# ---------------------------------------------------------------------------

def espf_token_spans(tokens, sequence: str = None) -> list:
    """ESPF tokens -> the [start, end) residue span each one covers.

    MolTrans builds its protein tokens with `BPE(..., separator='')`, so the
    tokens concatenate back to the original sequence exactly. That is what
    makes an exact residue mapping possible at all, and it is worth asserting:
    a separator, a lowercase pass or an unknown-token substitution anywhere in
    the tokeniser breaks the concatenation and every span after it shifts.

    Pass `sequence` to have that checked. It is cheap and it fails loudly
    instead of silently misaligning the ground truth.
    """
    spans, cursor = [], 0
    for token in tokens:
        spans.append((cursor, cursor + len(token)))
        cursor += len(token)

    if sequence is not None:
        joined = "".join(tokens)
        if joined != sequence[:len(joined)]:
            raise ValueError(
                "ESPF tokens do not concatenate back to the sequence, so the "
                "residue spans would be wrong. Check that the BPE was built "
                "with separator='' and that the sequence was not case-folded "
                "or substituted before tokenising.")
    return spans


def moltrans_covered_residues(tokens, max_tokens: int = MOLTRANS_MAX_PROTEIN_TOKENS) -> int:
    """How many residues the model actually saw.

    MolTrans truncates to 545 **tokens**, not residues, so the number of
    residues behind the window varies per protein and is usually far more than
    545. Treating 545 as a residue count -- in either direction -- misaligns
    everything.
    """
    return int(sum(len(t) for t in list(tokens)[:int(max_tokens)]))


def project_token_attention(token_weights, tokens, max_tokens: int = MOLTRANS_MAX_PROTEIN_TOKENS,
                            divide_by_span: bool = False) -> np.ndarray:
    """Subword-token attention -> one weight per real residue.

    token_weights  1D, one scalar per token position (length 545 for a padded
                   MolTrans input; only the first len(tokens) are real).
    tokens         the ESPF tokens for this protein, in order.

    Returns exactly `moltrans_covered_residues(tokens)` weights.

    Every residue in a token's span receives that token's weight, undivided.
    `divide_by_span=True` splits the weight across the span instead, which
    sounds fairer and is not: precision@k ranks residues, so dividing hands
    every short token a higher per-residue score than a long one purely for
    being short. That would make top-k selection a function of ESPF token
    length rather than of attention. Record the choice in Methods.
    """
    token_weights = np.asarray(token_weights, dtype=float).ravel()
    tokens = list(tokens)

    kept = tokens[:int(max_tokens)]
    if not kept:
        raise ValueError("no tokens: nothing to project onto residues")
    if token_weights.size < len(kept):
        raise ValueError(
            f"got {token_weights.size} token weights for {len(kept)} tokens — "
            f"the attention was reduced along the wrong axis, or the padded "
            f"length was trimmed before projecting")

    spans = espf_token_spans(kept)
    residues = np.zeros(moltrans_covered_residues(kept, max_tokens), dtype=float)
    for i, (start, end) in enumerate(spans):
        weight = token_weights[i]
        residues[start:end] = weight / (end - start) if divide_by_span else weight
    return residues


# ---------------------------------------------------------------------------
# extracting the attention itself, without changing prediction behaviour
# ---------------------------------------------------------------------------

def hyperattentiondti_protein_attention(model, drug, protein,
                                        channel_reduce: str = "mean") -> np.ndarray:
    """HyperAttentionDTI's per-convolution-position protein attention.

    Recomputes the attention branch of `AttentionDTI.forward` up to
    `Protein_atte` and stops. It does not modify the model, does not run the
    prediction head, and forces eval mode first -- the vendored forward applies
    dropout after the attention branch, so extracting in train mode would
    return perturbed weights.

    Shapes, verified against the vendored model at protein length 1000:

        protein_embed    (B, 64, 1000)
        Protein_CNNs     (B, 160, 979)          <- [4, 8, 12], 1000 - 21 = 979
        Atten_matrix     (B, 85, 979, 160)      <- drug pos x protein pos x ch
        Protein_atte     (B, 160, 979)          <- channel gate per position

    Note what the vendored code has already done by this point: it takes
    `torch.mean(Atten_matrix, 1)`, so the DRUG axis is averaged away inside
    forward. The remaining axis is CHANNELS, not drug positions. Reducing it
    with `mean` (default) or `max` gives one scalar per convolution position.

    Returns (B, 979) — still convolution positions. Feed it to
    `project_conv_attention` with the protein's real length to get residues.
    """
    import torch

    if channel_reduce not in ("mean", "max"):
        raise ValueError(f"channel_reduce must be 'mean' or 'max', got {channel_reduce!r}")

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            drug_embed = model.drug_embed(drug).permute(0, 2, 1)
            protein_embed = model.protein_embed(protein).permute(0, 2, 1)
            drug_conv = model.Drug_CNNs(drug_embed)
            protein_conv = model.Protein_CNNs(protein_embed)

            drug_att = model.drug_attention_layer(drug_conv.permute(0, 2, 1))
            protein_att = model.protein_attention_layer(protein_conv.permute(0, 2, 1))

            d_layers = drug_att.unsqueeze(2).repeat(1, 1, protein_conv.shape[-1], 1)
            p_layers = protein_att.unsqueeze(1).repeat(1, drug_conv.shape[-1], 1, 1)
            atten_matrix = model.attention_layer(model.relu(d_layers + p_layers))

            protein_atte = torch.sigmoid(
                torch.mean(atten_matrix, 1).permute(0, 2, 1))   # (B, C, n_pos)
            reduced = (protein_atte.mean(dim=1) if channel_reduce == "mean"
                       else protein_atte.max(dim=1).values)      # (B, n_pos)
        return reduced.cpu().numpy()
    finally:
        if was_training:
            model.train()


def capture_moltrans_attention(self_attention_module) -> tuple:
    """Forward hook that records MolTrans `attention_probs`.

    `SelfAttention.forward` computes `attention_probs` and returns only
    `context_layer`, so the weights are not reachable without either editing
    the vendored file or hooking. Hooking is the safer of the two: prediction
    behaviour is untouched and the baseline stays byte-identical to upstream.

    Returns (store, handle). `store["probs"]` holds the last
    (B, heads, tokens, tokens) tensor seen; call `handle.remove()` when done.

    Two cautions.

    * Extract in eval mode. The vendored code applies dropout to
      `attention_probs` before use; in train mode the captured weights are a
      dropped-out sample, not the model's explanation.
    * This is protein-token SELF-attention (token x token), not drug-protein
      interaction. Reducing it to one weight per token needs a choice of
      reduction, and MolTrans's own published interpretation figure uses the
      drug x protein interaction map instead. See PROJECTION_DECISIONS.
    """
    store: dict = {}

    def hook(_module, inputs, _output):
        import torch

        hidden_states, attention_mask = inputs[0], inputs[1]
        with torch.no_grad():
            query = _module.transpose_for_scores(_module.query(hidden_states))
            key = _module.transpose_for_scores(_module.key(hidden_states))
            scores = torch.matmul(query, key.transpose(-1, -2))
            scores = scores / (_module.attention_head_size ** 0.5)
            scores = scores + attention_mask
            store["probs"] = torch.softmax(scores, dim=-1).detach()

    return store, self_attention_module.register_forward_hook(hook)


# ---------------------------------------------------------------------------

PROJECTION_DECISIONS = {
    "hyperattentiondti.conv_projection": {
        "question": "How does a convolution position become a residue?",
        "options": ["centre (default) — position j -> residue j + 10",
                    "receptive_field — spread over the 22 residues j..j+21"],
        "default_rationale": "centre keeps a localised signal localised; "
                             "receptive_field blurs every weight over 22 "
                             "residues and mechanically lowers precision@10, "
                             "which would manufacture the degradation the "
                             "audit is testing for.",
        "owner": "124AD0008 (adapter owner), to record in Methods",
    },
    "hyperattentiondti.drug_axis": {
        "question": "Can the drug axis be reduced with max instead of mean?",
        "finding": "Not without recomputing. The vendored forward hardcodes "
                   "torch.mean(Atten_matrix, 1), so the drug axis is gone "
                   "before Protein_atte exists. The remaining axis is 160 "
                   "CHANNELS, not drug positions — the baseline_adapters.py "
                   "docstring currently says otherwise. Taking max over drug "
                   "positions means recomputing from Atten_matrix, which is "
                   "(B, 85, 979, 160) — ~13M floats per sample.",
        "owner": "124AD0008, with 124AD0015",
    },
    "moltrans.attention_source": {
        "question": "Self-attention over protein tokens, or the drug x protein "
                    "interaction map?",
        "finding": "SelfAttention gives token x token within the protein; the "
                   "interaction map i_v (drug x protein tokens) is what "
                   "MolTrans's own paper visualises. The audit measures "
                   "published interpretability claims, so the published "
                   "artefact is the defensible subject.",
        "resolved": "2026-09-20: both readouts are scored and both are reported. "
                    "MolTrans's full text confirms the published artefact -- "
                    "'visualize the strength of individual sub-structural "
                    "interaction pair from the interaction map', its Fig. 3 -- so "
                    "the map is registered as `moltrans_interaction` (max over "
                    "drug substructures; `_sum` the alternative) in "
                    "readout_variants.py. The encoder self-attention stays the "
                    "default and is reported beside it. Leaving it as the only "
                    "readout was not defensible once measured: it returns the "
                    "identical map for different drugs on the same protein "
                    "(src/evaluation/drug_dependence.py).",
        "owner": "124AD0008, with 124AD0067 (it affects what claim is audited)",
    },
    "moltrans.span_weighting": {
        "question": "Divide a token's weight across its residue span?",
        "options": ["undivided (default)", "divide_by_span=True"],
        "default_rationale": "dividing makes top-k selection a function of "
                             "ESPF token length rather than attention.",
        "owner": "124AD0008, to record in Methods",
    },
}
