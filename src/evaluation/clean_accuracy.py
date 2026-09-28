"""
Test accuracy on the targets that are genuinely unseen -- option A of 2026-09-13.

At DAVIS cold-target 12 of 88 test targets (13.6% of rows) and at cold-pair 11 of 88
(12.5%) are identical in sequence to a training target (`src/data/sequence_audit.py`).
The cells were trained and tested on the full test files; this module re-scores the
saved checkpoints and reports test AUROC twice: on every row, and on the rows whose
target is unseen by sequence.

Each model is scored by **its own trainer's test pass** -- the same dataset class,
encoding and `run_epoch` that produced the recorded number -- so the only thing that
changes between the two AUROCs is which rows count. The first number must reproduce
the cell's recorded test AUROC; `reproduces_recorded` says whether it did (CPU against
the GPU that trained it; MolTrans keeps dropout on at inference, as published, so its
re-score is not bit-identical).

Pocketless targets are NOT dropped here: accuracy is a prediction from the sequence the
model was given, whatever that sequence is. Only leakage changes what a test row means.

    python -m src.evaluation.clean_accuracy --checkpoint-dir <results> --models deepdta,coldsite_dti
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from src.model.checkpoint_naming import checkpoint_path, results_path, run_tag
from src.model.dataset import BINARY_THRESHOLD

LEVELS = ("cold_target", "cold_pair")
# Recorded (GPU) vs re-scored (CPU) AUROC. Floating-point order differs between the two;
# anything past this is not arithmetic and means the re-score is not the same test pass.
REPRODUCE_TOLERANCE = 0.005
# MolTrans keeps dropout on at inference, as published, so one test pass is one draw:
# re-scoring DAVIS cold-pair seed 2 six times gave 0.562-0.575 AUROC (sd 0.005) around a
# recorded 0.567 (2026-09-13). Its cells are scored as the mean of this many passes, and
# count as reproduced when the recorded value lies within the passes' range.
DROPOUT_DRAWS = {"moltrans": 5}


def seen_by_sequence(dataset: str, level: str) -> frozenset:
    from src.data.sequence_audit import sequence_leakage
    return frozenset(sequence_leakage(dataset)[level]["test"]["seen_by_sequence"])


def unseen_auroc(dataset: str, model: str, level: str, seed: int,
                 path: str | None = None) -> float | None:
    """The cell's AUROC on targets unseen by sequence, if this module has scored it."""
    path = path or f"results/clean_accuracy_{dataset}.json"
    if level not in LEVELS or not os.path.exists(path):
        return None
    for cell in json.load(open(path)):
        if (cell["model"], cell["dataset"], cell["level"], cell["seed"]) == (model, dataset, level, seed):
            return cell["unseen_by_sequence"]["auroc"]
    return None


def leaks(dataset: str, level: str) -> bool:
    """Does this level's test set hold targets seen by sequence in training?"""
    if level not in LEVELS or not os.path.isdir(os.path.join("data/splits", dataset, level)):
        return False
    return bool(seen_by_sequence(dataset, level))


def _scores(model_name: str, split_dir: str, dataset: str, ckpt: str, recorded: dict,
            seed: int, device: str) -> tuple[np.ndarray, np.ndarray]:
    """(labels, scores) for every test row, in test.csv order, by the trainer's own code."""
    test_csv = os.path.join(split_dir, "test.csv")
    threshold = BINARY_THRESHOLD[dataset]
    state = torch.load(ckpt, map_location=device, weights_only=False)

    if model_name == "deepdta":
        from src.model.deepdta_torch import DeepDTA
        from src.model.train_deepdta import DeepDTADataset, run_epoch
        saved = state.get("args", {})
        model = DeepDTA(drug_kernel=saved.get("drug_kernel", 4),
                        protein_kernel=saved.get("protein_kernel", 8))
        model.load_state_dict(state["model_state"])
        loader = DataLoader(DeepDTADataset(test_csv, "binary", threshold), batch_size=256)
        _loss, labels, scores = run_epoch(model.to(device), loader, nn.BCEWithLogitsLoss(), device)
        return labels, scores

    if model_name == "coldsite_dti":
        from src.model.coldsite_dti import ColdSiteDTI
        from src.model.dataset import load_split
        _tr, _va, loader, drug_vocab, protein_vocab = load_split(
            split_dir, 1000, batch_size=64, binary_threshold=threshold)
        model = ColdSiteDTI(len(drug_vocab) + 2, len(protein_vocab) + 2)
        model.load_state_dict(state["model_state"])
        model.to(device).eval()
        labels, scores = [], []
        with torch.no_grad():
            for drug, protein, y in loader:
                pred, _attn = model(drug.to(device), protein.to(device))
                scores.append(pred.squeeze(-1).cpu().numpy())
                labels.append(y.numpy())
        return np.concatenate(labels), np.concatenate(scores)

    if model_name == "hyperattentiondti":
        from src.model.train_hyperattentiondti import (HyperAttentionDataset, _import_vendored,
                                                       run_epoch)
        AttentionDTI, hyperparameter, *_ = _import_vendored()
        hp = hyperparameter()
        hp.Batch_size = recorded.get("batch_size", 32)
        model = AttentionDTI(hp)
        model.load_state_dict(state["model_state"])
        loader = DataLoader(HyperAttentionDataset(test_csv, threshold), batch_size=hp.Batch_size)
        _loss, labels, scores = run_epoch(model.to(device), loader, nn.CrossEntropyLoss(),
                                          device, label="test")
        return labels, scores

    if model_name == "moltrans":
        from src.model.train_moltrans import MolTransDataset, _import_vendored, run_epoch
        BIN_config_DBPE, BIN_Interaction_Flat, drug_encoder, protein_encoder = _import_vendored()
        config = BIN_config_DBPE()
        config["batch_size"] = recorded.get("batch_size", 16)
        model = BIN_Interaction_Flat(**config)
        model.load_state_dict(state["model_state"])
        torch.manual_seed(seed)                     # its inference dropout, held fixed
        loader = DataLoader(MolTransDataset(test_csv, threshold, (drug_encoder, protein_encoder)),
                            batch_size=config["batch_size"])
        _loss, labels, scores = run_epoch(model.to(device), loader, nn.BCEWithLogitsLoss(),
                                          device, label="test")
        return labels, scores

    if model_name == "drugban":
        # The trainer's own dataset, collate and run_epoch; the model through the adapter,
        # which loads DrugBAN's vendored modules without colliding with MolTrans's.
        from src.evaluation.drugban_adapter import DrugBANAdapter
        from src.model.train_drugban import DrugBANDataset, collate, run_epoch
        model = DrugBANAdapter(checkpoint_path=ckpt, device=device).model
        loader = DataLoader(DrugBANDataset(test_csv, threshold),
                            batch_size=recorded.get("batch_size", 64), collate_fn=collate)
        _loss, labels, scores = run_epoch(model, loader, nn.BCEWithLogitsLoss(), device,
                                          log_every=0, label="test")
        return labels, scores

    raise ValueError(f"no test pass registered for {model_name!r}")


def _metrics(labels, scores) -> dict:
    labels, scores = np.asarray(labels), np.asarray(scores)
    return {"auroc": float(roc_auc_score(labels, scores)),
            "auprc": float(average_precision_score(labels, scores)),
            "rows": int(labels.size), "positive_rate": float(labels.mean())}


def evaluate_cell(model_name: str, dataset: str, level: str, seed: int, checkpoint_dir: str,
                  results_dir: str | None = None, device: str = "cpu") -> dict | None:
    results_dir = results_dir or checkpoint_dir
    ckpt = checkpoint_path(checkpoint_dir, dataset, level, "binary", seed, model=model_name)
    res = results_path(results_dir, run_tag(dataset, level, "binary", seed), model=model_name)
    if not (os.path.exists(ckpt) and os.path.exists(res)):
        return None
    recorded = json.load(open(res))
    split_dir = os.path.join("data/splits", dataset, level)
    test = pd.read_csv(os.path.join(split_dir, "test.csv"))
    leaked = seen_by_sequence(dataset, level)
    keep = ~test.Target_ID.astype(str).isin(leaked).to_numpy()

    draws = []
    for draw in range(DROPOUT_DRAWS.get(model_name, 1)):
        labels, scores = _scores(model_name, split_dir, dataset, ckpt, recorded,
                                 seed * 100 + draw if draw else seed, device)
        if len(test) != len(labels):
            raise RuntimeError(f"{model_name} {level} s{seed}: {len(labels)} scores for "
                               f"{len(test)} test rows -- rows and scores are not aligned")
        draws.append((_metrics(labels, scores), _metrics(labels[keep], scores[keep])))
    every, unseen = combine_draws(draws)
    recorded_auroc = recorded["test_metrics"]["auroc"]
    return {"model": model_name, "dataset": dataset, "level": level, "seed": seed,
            "recorded_auroc": recorded_auroc, "all_rows": every, "unseen_by_sequence": unseen,
            "reproduces_recorded": reproduces(recorded_auroc, [d[0]["auroc"] for d in draws]),
            "dropout_draws": len(draws),
            "leaked_targets": sorted(leaked), "rows_dropped": int((~keep).sum())}


def combine_draws(draws: list) -> tuple:
    """Mean over test passes; with one pass the metrics pass through unchanged."""
    out = []
    for part in (0, 1):
        runs = [d[part] for d in draws]
        merged = dict(runs[0])
        for key in ("auroc", "auprc"):
            values = [r[key] for r in runs]
            merged[key] = float(np.mean(values))
            if len(values) > 1:
                merged[f"{key}_draws"] = [float(v) for v in values]
        out.append(merged)
    return out[0], out[1]


def reproduces(recorded: float, rescored: list) -> bool:
    """One pass: within REPRODUCE_TOLERANCE. Several: the recorded value lies in their range."""
    return (min(rescored) - REPRODUCE_TOLERANCE <= recorded
            <= max(rescored) + REPRODUCE_TOLERANCE)


def merge_cells(existing: list, new: list) -> list:
    """New cells replace old ones for the same (model, dataset, level, seed); others stay."""
    key = lambda c: (c["model"], c["dataset"], c["level"], c["seed"])
    fresh = {key(c) for c in new}
    return [c for c in existing if key(c) not in fresh] + list(new)


def report(cells: list) -> str:
    lines = ["# Test accuracy on targets unseen by sequence — DAVIS\n",
             "Option A (2026-09-13): each cell re-scored by its own trainer's test pass; AUROC "
             "on every test row and on the rows whose target is unseen by sequence "
             "(`src/evaluation/clean_accuracy.py`, `results/sequence_audit_davis.md`).\n",
             "MolTrans keeps dropout on at inference (as published): its re-scores are the mean of "
             f"{DROPOUT_DRAWS['moltrans']} passes and count as reproduced when the recorded value "
             "lies within their range.\n",
             "| model | level | seed | recorded | re-scored, all rows | reproduces? | "
             "unseen by sequence | change | rows kept |",
             "|---|---|---|---|---|---|---|---|---|"]
    for c in cells:
        a, u = c["all_rows"], c["unseen_by_sequence"]
        lines.append(f"| {c['model']} | {c['level']} | {c['seed']} | {c['recorded_auroc']:.4f} | "
                     f"{a['auroc']:.4f} | {'yes' if c['reproduces_recorded'] else '**NO**'} | "
                     f"{u['auroc']:.4f} | {u['auroc'] - a['auroc']:+.4f} | "
                     f"{u['rows']} of {a['rows']} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--dataset", default="davis")
    parser.add_argument("--models", default="deepdta,coldsite_dti,hyperattentiondti,moltrans")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--levels", default=",".join(LEVELS))
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--results-dir")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    cells = []
    for model in args.models.split(","):
        for level in args.levels.split(","):
            for seed in (int(s) for s in args.seeds.split(",")):
                cell = evaluate_cell(model, args.dataset, level, seed, args.checkpoint_dir,
                                     args.results_dir, args.device)
                if cell is None:
                    print(f"[skip] {model} {level} s{seed}: no checkpoint/results")
                    continue
                cells.append(cell)
                print(f"{model:18s} {level:11s} s{seed}  recorded {cell['recorded_auroc']:.4f}  "
                      f"re-scored {cell['all_rows']['auroc']:.4f} "
                      f"({'ok' if cell['reproduces_recorded'] else 'MISMATCH'})  "
                      f"unseen {cell['unseen_by_sequence']['auroc']:.4f}", flush=True)
    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.join(args.out_dir, f"clean_accuracy_{args.dataset}")
    # A run over some models updates their rows and keeps every other model's.
    if os.path.exists(stem + ".json"):
        cells = merge_cells(json.load(open(stem + ".json")), cells)
    order = ["deepdta", "coldsite_dti", "hyperattentiondti", "moltrans"]
    cells.sort(key=lambda c: (order.index(c["model"]) if c["model"] in order else 9,
                              c["level"] != "cold_target", c["seed"]))
    with open(stem + ".json", "w") as f:
        json.dump(cells, f, indent=2)
    with open(stem + ".md", "w") as f:
        f.write(report(cells))
    print(f"Saved -> {stem}.json\nSaved -> {stem}.md")


if __name__ == "__main__":
    main()
