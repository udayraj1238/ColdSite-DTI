"""
The real collector — trained checkpoints in, `(weights, sites, target_ids)` out.

`run_audit.build_grid` takes a `collect_fn` and calls it once per
(model, dataset, level, seed) cell. Until now the only implementation was
`dummy_collect_fn`, so real mode raised `SystemExit` and there was no path from
a checkpoint to the audit table however much training had finished. This module
is that path.

Every model is tokenised with its own vendored tables
-----------------------------------------------------
The one thing this file must not do is standardise the models. Each adapter
already knows how its authors tokenise; the collector's job is to hand each one
a row from the test split in the form that adapter expects, and to keep the
protein's identity attached to the explanation that comes back:

    ColdSite-DTI       the split's own vocabulary, built from train.csv only
    HyperAttentionDTI  CHARISOSMISET / CHARPROTSET, its own label_* functions
    MolTrans           ESPF subword units, plus the token list explain() needs
    uniform_control    any encoding -- the output is flat by construction

DeepDTA is not collectable and that is correct: `provides_attention = False`.
Asking it for an explanation raises rather than returning a saliency map its
authors never published.

One pair per target, by default
-------------------------------
precision@k is a per-protein quantity. A test split holds many pairs per
protein, so collecting every row would put the same protein into the average
dozens of times, once per drug it was measured against. That inflates
`n_evaluated` from "proteins" to "pairs" and makes the permutation test look
far better powered than it is -- 400 correlated samples are not 400
observations.

`pairs_per_target=1` therefore takes one representative pair per protein.
Raise it to compare explanations across drugs for the same protein, which is a
different and also interesting question; just do not report the result as if n
were the number of proteins. **This is a measurement decision that belongs to
Track C** -- it is a parameter here rather than a constant so that the choice
is made explicitly and stated in Methods.

Usage
-----
    from src.evaluation.collect import make_collect_fn
    from src.evaluation.run_audit import build_grid

    collect = make_collect_fn(dataset="davis", task="binary",
                              ground_truth="data/davis_ground_truth_sites.json")
    results = build_grid(collect, ["coldsite_dti", "uniform_control"],
                         ["davis"], [1, 2, 3])
"""
from __future__ import annotations

import os
import re

import numpy as np

from src.data.ground_truth import load_site_sets, site_lookup

# Imported for its registration side-effect: @register runs at import time, so
# without this the registry holds only coldsite_dti and uniform_control and
# every baseline reads as "unknown model" -- with an error telling you to write
# an adapter that already exists and already passes validate_adapter.
from src.evaluation import baseline_adapters  # noqa: F401
from src.evaluation.model_registry import (_REGISTRY, get_model,
                                           load_variant_plugins)
from src.model.checkpoint_naming import (MODEL_SUFFIX, base_model_name,
                                         checkpoint_path)

# Models that need no checkpoint: the control is flat by construction, so
# "untrained" is not a defect and a missing file must not skip the cell.
CHECKPOINT_FREE = {"uniform_control"}

SMILES_COLUMN = "Drug"
SEQUENCE_COLUMN = "Target"
TARGET_ID_COLUMN = "Target_ID"

# A wildcard atom (`*`) or a dative bond (`->`, `<-`) is not a concrete molecule
# any audited model can read: HyperAttentionDTI's alphabet has none of them and
# raises. 16 of the non-kinase panel's 21,145 rows (BindingDB) carry one; no
# DAVIS or KIBA row does. They are dropped for every model alike, so each model
# is scored on the same rows -- dropping them only where one model fails would
# compare models on different panels.
UNREADABLE_SMILES = re.compile(r"[*<>]")


def clean_smiles(smiles: str) -> str:
    """The SMILES itself: everything before the first whitespace.

    BindingDB writes ChemAxon extended SMILES, where annotations such as
    `|r|` (relative stereochemistry) follow a space. They are not atoms. Read
    as part of the molecule they crash HyperAttentionDTI's tokeniser and are
    silently tokenised as chemistry by the other two models. 4,694 of the
    panel's rows carry one; DAVIS and KIBA carry none.
    """
    parts = str(smiles).split()
    return parts[0] if parts else ""


class MissingCell(Exception):
    """This cell cannot be collected yet. Carries the reason for the report."""


def _read_test_rows(split_dir: str, pairs_per_target: int,
                    rows_csv: str | None = None, policy: bool = True,
                    with_drug: bool = False):
    """One row per (target, drug) pair from the test split, capped per target.

    Returns a list of (target_id, smiles, sequence), or of
    (target_id, drug_id, smiles, sequence) when `with_drug` -- which a drug-specific
    ground truth needs, since its sites belong to the pair and not to the protein.
    Rows keep the file's own
    order so a cap of 1 is deterministic rather than whichever pair pandas
    happened to group first.

    `rows_csv` evaluates a model on pairs from somewhere other than its own
    test split -- the non-kinase control panel above all. The vocabulary still
    comes from `split_dir`, because it is a property of the *trained model*:
    rebuilding it from the panel would give the model an embedding table it
    was never trained with.

    `policy` applies `src/evaluation/exclusions.py` (option A, 2026-09-13): test
    targets seen by sequence at the cold levels and targets without the kinase pocket
    are skipped, and the per-protein cap counts distinct sequences, not names. On a
    panel (`rows_csv`) only the per-sequence cap applies -- its proteins are not this
    dataset's targets. `policy=False` reproduces the numbers from before the decision.
    """
    from src.evaluation.exclusions import (dataset_level_from_split_dir,
                                           excluded_target_ids, protein_key)
    import pandas as pd

    path = rows_csv or os.path.join(split_dir, "test.csv")
    if not os.path.exists(path):
        raise MissingCell(f"no test split at {path}")

    frame = pd.read_csv(path)
    for column in (TARGET_ID_COLUMN, SMILES_COLUMN, SEQUENCE_COLUMN):
        if column not in frame.columns:
            raise MissingCell(
                f"{path} has no {column!r} column (found {list(frame.columns)})")

    excluded = (frozenset() if rows_csv
                else excluded_target_ids(*dataset_level_from_split_dir(split_dir), policy))
    seen: dict = {}
    rows = []
    drug_column = frame["Drug_ID"] if "Drug_ID" in frame.columns else frame[SMILES_COLUMN]
    for target_id, drug_id, smiles, sequence in zip(frame[TARGET_ID_COLUMN],
                                                    drug_column,
                                                    frame[SMILES_COLUMN],
                                                    frame[SEQUENCE_COLUMN]):
        if str(target_id) in excluded:
            continue
        # Before the per-target cap, so a protein's first READABLE pair is
        # the one kept. See clean_smiles and UNREADABLE_SMILES.
        smiles = clean_smiles(smiles)
        if not smiles or UNREADABLE_SMILES.search(smiles):
            continue
        key = protein_key(target_id, sequence, policy)
        count = seen.get(key, 0)
        if pairs_per_target and count >= pairs_per_target:
            continue
        seen[key] = count + 1
        row = (str(target_id), str(smiles), str(sequence).upper())
        rows.append((row[0], str(drug_id)) + row[1:] if with_drug else row)
    return rows


def _build_adapter(model_name: str, checkpoint: str | None, split_dir: str,
                   device: str, max_protein_len: int):
    """Instantiate one adapter, loading its checkpoint if it needs one.

    ColdSite-DTI is the only model whose *architecture* depends on the split:
    its drug vocabulary is built from that split's train.csv, so the embedding
    size differs per cell. Rebuilt here the same way the trainer built it --
    train rows only, never valid or test, or the cold splits leak the drugs
    they exist to hold out.

    An explanation variant is built exactly like the model it explains: it IS that model,
    with a different explainer on top (`src/evaluation/integrated_gradients.py`).
    """
    if base_model_name(model_name) == "coldsite_dti":
        import pandas as pd

        from src.model.dataset import SMILES_COLUMNS, find_column
        from src.model.drug_encoder import build_smiles_vocab
        from src.model.protein_encoder import build_protein_vocab

        train_path = os.path.join(split_dir, "train.csv")
        if not os.path.exists(train_path):
            raise MissingCell(f"no train split at {train_path} (needed for the vocab)")
        train_df = pd.read_csv(train_path)
        drug_vocab = build_smiles_vocab(
            train_df[find_column(train_df, SMILES_COLUMNS, "SMILES")]
            .astype(str).tolist())
        protein_vocab = build_protein_vocab()
        # +2 for PAD and UNK, matching run_ladder and the trainer
        return get_model(model_name, checkpoint_path=checkpoint,
                         drug_vocab_size=len(drug_vocab) + 2,
                         protein_vocab_size=len(protein_vocab) + 2,
                         device=device), (drug_vocab, protein_vocab)

    if model_name == "uniform_control":
        from src.model.drug_encoder import build_smiles_vocab
        from src.model.protein_encoder import build_protein_vocab
        # the control needs an encoding only to measure the protein's length
        return get_model(model_name), (build_smiles_vocab(["C"]),
                                       build_protein_vocab())

    return get_model(model_name, checkpoint_path=checkpoint, device=device), None


def _explain_row(model_name: str, adapter, vocabs, smiles: str, sequence: str,
                 max_protein_len: int) -> np.ndarray:
    """One explanation, tokenised the way that model's own authors tokenise.

    An explanation variant is tokenised like the model it explains: `type(adapter).encode`
    below is the base adapter's, reached through the variant's own class.
    """
    import torch

    model_name = base_model_name(model_name)
    if model_name in ("coldsite_dti", "uniform_control"):
        from src.model.drug_encoder import encode_smiles
        from src.model.protein_encoder import encode_protein

        drug_vocab, protein_vocab = vocabs
        drug = torch.tensor(encode_smiles(smiles, drug_vocab, 100), dtype=torch.long)
        protein = torch.tensor(
            encode_protein(sequence, protein_vocab, max_protein_len), dtype=torch.long)
        return np.asarray(adapter.explain(drug, protein), dtype=float)

    if model_name in ("hyperattentiondti", "drugban"):
        # Both encode (smiles, sequence) -> (drug, protein) and return one weight per
        # residue; DrugBAN's drug side is a DGL graph rather than a tensor, which
        # `explain` handles, and nothing here needs to know the difference.
        drug, protein = type(adapter).encode(smiles, sequence)
        return np.asarray(adapter.explain(drug, protein), dtype=float)

    if model_name == "moltrans":
        drug, drug_mask, protein, protein_mask, tokens = type(adapter).encode(
            smiles, sequence)
        return np.asarray(
            adapter.explain(drug, protein, protein_tokens=tokens,
                            drug_mask=drug_mask, protein_mask=protein_mask),
            dtype=float)

    raise MissingCell(
        f"no tokenisation registered for {model_name!r}. Add one in "
        f"src/evaluation/collect.py::_explain_row — guessing an encoding "
        f"produces an array of the wrong length, which misaligns every "
        f"ground-truth index and yields a plausible wrong number.")


def collect_cell(model_name: str, dataset: str, level: str, seed: int, *,
                 site_sets: dict, task: str = "binary",
                 split_root: str = "data/splits",
                 checkpoint_dir: str = "results",
                 max_protein_len: int = 1000,
                 pairs_per_target: int = 1,
                 max_proteins: int | None = None,
                 rows_csv: str | None = None,
                 device: str = "cpu",
                 verbose: bool = True,
                 policy: bool = True):
    """One grid cell. Returns (weights, sites, target_ids), or raises MissingCell.

    A protein with no usable ground truth is skipped rather than scored against
    an empty site set, which would count as a zero and drag every mean down.
    """
    load_variant_plugins(model_name)
    adapter_cls = _REGISTRY.get(model_name)
    if adapter_cls is None:
        raise MissingCell(
            f"unknown model {model_name!r}. Registered: {sorted(_REGISTRY)}")

    if not getattr(adapter_cls, "provides_attention", True):
        raise MissingCell(
            f"{model_name} has provides_attention = False — it anchors the "
            f"accuracy axis and has no explanation to collect")

    split_dir = os.path.join(split_root, dataset, level)

    checkpoint = None
    if model_name not in CHECKPOINT_FREE:
        if model_name not in MODEL_SUFFIX:
            raise MissingCell(
                f"{model_name} has no checkpoint suffix registered; add it to "
                f"MODEL_SUFFIX in src/model/checkpoint_naming.py")
        checkpoint = checkpoint_path(checkpoint_dir, dataset, level, task, seed,
                                     model=model_name)
        if not os.path.exists(checkpoint):
            raise MissingCell(f"no checkpoint at {checkpoint}")

    lookup = site_lookup(site_sets)
    rows = _read_test_rows(split_dir, pairs_per_target, rows_csv, policy=policy,
                           with_drug=lookup.pair_keyed)
    adapter, vocabs = _build_adapter(model_name, checkpoint, split_dir, device,
                                     max_protein_len)

    weights, sites, used_ids = [], [], []
    skipped_no_sites = 0
    for row in rows:
        target_id, drug_id, smiles, sequence = (
            row if lookup.pair_keyed else (row[0], None, row[1], row[2]))
        site_set = lookup(target_id, drug_id)
        if site_set is None or not site_set.usable:
            skipped_no_sites += 1
            continue
        # One evaluation window for every model: the same first
        # `max_protein_len` residues the ground truth is cut to. ColdSite-DTI
        # and HyperAttentionDTI never read past it; MolTrans's 545 tokens reach
        # ~1,400 residues on long proteins (115 of DAVIS's 442), and attention
        # there competes for the top k while no site can exist there to hit.
        weights.append(_explain_row(model_name, adapter, vocabs, smiles,
                                    sequence, max_protein_len)[:max_protein_len])
        sites.append(site_set.positions)
        used_ids.append(f"{drug_id}|{target_id}" if lookup.pair_keyed else target_id)
        if max_proteins and len(weights) >= max_proteins:
            break

    if not weights:
        raise MissingCell(
            f"{len(rows)} test rows, none with usable ground truth "
            f"(skipped {skipped_no_sites}) — check that the ground-truth file "
            + ("has pairs from this split: a drug-specific ground truth only covers "
               "pairs with a co-crystal structure, and a cold split may hold none"
               if lookup.pair_keyed else
               "matches this dataset's Target_ID spelling"))

    if verbose:
        unit = "drug-protein pairs" if lookup.pair_keyed else "proteins"
        print(f"  {model_name}/{dataset}/{level}/seed{seed}: "
              f"{len(weights)} {unit} ({skipped_no_sites} without usable sites)")
    return weights, sites, used_ids


def make_collect_fn(dataset: str, ground_truth: str, task: str = "binary",
                    split_root: str = "data/splits",
                    checkpoint_dir: str = "results",
                    max_protein_len: int = 1000,
                    pairs_per_target: int = 1,
                    max_proteins: int | None = None,
                    rows_csv: str | None = None,
                    device: str = "cpu",
                    skipped: list | None = None):
    """A `collect_fn` for `run_audit.build_grid`, over real checkpoints.

    Returns None for a cell that cannot be collected yet, which is what
    `build_grid` records as missing. The reason is appended to `skipped` so the
    report can say *why* a cell is absent — "no checkpoint" and "no usable
    ground truth" call for very different responses, and a bare count of
    missing cells cannot tell them apart.
    """
    site_sets = load_site_sets(ground_truth, max_len=max_protein_len)

    def collect(model_name, dataset_name, level, seed):
        try:
            return collect_cell(
                model_name, dataset_name, level, seed,
                site_sets=site_sets, task=task, split_root=split_root,
                checkpoint_dir=checkpoint_dir, max_protein_len=max_protein_len,
                pairs_per_target=pairs_per_target, max_proteins=max_proteins,
                rows_csv=rows_csv, device=device)
        except MissingCell as reason:
            if skipped is not None:
                skipped.append(
                    f"{model_name}/{dataset_name}/{level}/seed{seed}: {reason}")
            return None

    return collect
