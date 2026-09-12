"""
Train the PyTorch DeepDTA port on our splits.

Why a separate trainer
----------------------
`src/model/train.py` is ColdSite-DTI's, and its `dataset.py` builds a vocabulary
from the training split for ColdSite-DTI's own tokenisation. DeepDTA uses fixed
character tables published with the model (`CHARISOSMISET`, `CHARPROTSET`), so
it needs its own encoding path. Forcing both through one loader would mean
either changing 124AD0015's loader or tokenising DeepDTA in a way its authors
did not — and in an audit, each subject should be reproduced as published.

Everything downstream is shared on purpose. The tag, the checkpoint path and the
results JSON all come from `checkpoint_naming`, and the metrics come from
`train.compute_metrics`, so a DeepDTA cell lands in exactly the same shape as a
ColdSite-DTI cell and `aggregate.py` needs no special case.

Usage
-----
    python -m src.model.train_deepdta --split-dir data/splits/davis/cold_target \\
        --dataset davis --split cold_target --task regression --seed 1 --epochs 100

    # the full grid for this model: 2 datasets x 4 splits x 3 seeds
    for /L %s in (1,1,3) do for %d in (davis kiba) do for %p in (random cold_drug cold_target cold_pair) do ^
        python -m src.model.train_deepdta --split-dir data/splits/%d/%p --dataset %d --split %p --seed %s

STATUS: verified end-to-end on DAVIS cold-target seed 1 (2026-08-09):
test CI 0.811, MSE 0.397, early stop at epoch 29 (best 19).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.model.early_stopping import DEFAULT_MIN_EPOCHS, CheckpointSelector
from src.model.checkpoint_naming import checkpoint_path, results_path, run_tag
from src.model.deepdta_torch import (
    MAX_PROTEIN_LEN,
    MAX_SMILES_LEN,
    DeepDTA,
    encode_protein,
    encode_smiles,
)
from src.model.train import compute_metrics

# DAVIS pKd >= 7.0 and KIBA >= 12.1 are DeepDTA's own published thresholds and
# are already verified against the real data (8.3% and 21.0% positive). Shared
# with the other two models from src.model.dataset -- see the note there.
from src.model.dataset import BINARY_THRESHOLD


class DeepDTADataset(Dataset):
    """Split CSV -> DeepDTA's fixed-charset integer encoding.

    Encoding happens once in __init__, not per __getitem__. DAVIS is 30k rows
    and KIBA 118k; re-encoding a 1000-character sequence on every access makes
    the loader, not the GPU, the bottleneck.
    """

    def __init__(self, csv_path: str, task: str = "regression",
                 threshold: float = None):
        frame = pd.read_csv(csv_path)
        for column in ("Drug", "Target", "Y"):
            if column not in frame.columns:
                raise KeyError(
                    f"{csv_path} has no '{column}' column; found "
                    f"{list(frame.columns)}. Rebuild with "
                    f"`python -m src.data.build_splits`.")

        self.drugs = np.stack([encode_smiles(s, MAX_SMILES_LEN)
                               for s in frame["Drug"]])
        self.proteins = np.stack([encode_protein(s, MAX_PROTEIN_LEN)
                                  for s in frame["Target"]])

        y = frame["Y"].to_numpy(dtype=float)
        if task == "binary":
            if threshold is None:
                raise ValueError("binary task needs a threshold")
            y = (y >= threshold).astype(float)
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return (torch.from_numpy(self.drugs[index]),
                torch.from_numpy(self.proteins[index]),
                torch.tensor(self.y[index], dtype=torch.float32))


def run_epoch(model, loader, loss_fn, device, optimizer=None) -> tuple:
    training = optimizer is not None
    model.train(training)
    total, n, preds, trues = 0.0, 0, [], []

    for drug, protein, y in loader:
        drug, protein, y = drug.to(device), protein.to(device), y.to(device)
        with torch.set_grad_enabled(training):
            out = model(drug, protein)
            loss = loss_fn(out, y)
        if training:
            optimizer.zero_grad()
            loss.backward()
            # Same clipping as train.py. DeepDTA's three stacked convolutions
            # on a 1000-wide input can spike early, and a NaN on epoch 2 of a
            # 24-run grid is expensive to notice late.
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        total += float(loss.item()) * len(y)
        n += len(y)
        preds.append(out.detach().cpu().numpy())
        trues.append(y.detach().cpu().numpy())

    return total / max(n, 1), np.concatenate(trues), np.concatenate(preds)


def main():
    parser = argparse.ArgumentParser(description="Train DeepDTA on our splits")
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--dataset", required=True, choices=["davis", "kiba"])
    parser.add_argument("--split", required=True)
    parser.add_argument("--task", default="regression",
                        choices=["regression", "binary"])
    parser.add_argument("--seed", type=int, required=True,
                        help="TRAINING seed. Mandatory: omitting it is how one "
                             "run silently overwrites another.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument(
        "--min-epochs", type=int, default=DEFAULT_MIN_EPOCHS,
        help="no checkpoint before this epoch; see src/model/early_stopping.py")
    parser.add_argument("--drug-kernel", type=int, default=4)
    parser.add_argument("--protein-kernel", type=int, default=8)
    parser.add_argument("--checkpoint-dir", default="results")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-if-done", action="store_true",
                        help="exit immediately if this cell's results file "
                             "already exists, so a 24-run grid can be resumed "
                             "after an interruption without redoing work")
    args = parser.parse_args()

    tag = run_tag(args.dataset, args.split, args.task, args.seed)
    out_path = results_path(args.results_dir, tag, model="deepdta")
    if args.skip_if_done and os.path.exists(out_path):
        print(f"already done, skipping -> {out_path}")
        return

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    threshold = BINARY_THRESHOLD[args.dataset] if args.task == "binary" else None
    loaders = {}
    for part in ("train", "valid", "test"):
        path = os.path.join(args.split_dir, f"{part}.csv")
        if not os.path.exists(path):
            raise SystemExit(
                f"{path} not found. Build the splits first:\n"
                f"    python -m src.data.build_splits")
        dataset = DeepDTADataset(path, args.task, threshold)
        loaders[part] = DataLoader(dataset, batch_size=args.batch_size,
                                   shuffle=(part == "train"))
        print(f"  {part:5s} {len(dataset):>7,} pairs  <- {path}")

    device = args.device
    model = DeepDTA(drug_kernel=args.drug_kernel,
                    protein_kernel=args.protein_kernel).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss() if args.task == "regression" else nn.BCEWithLogitsLoss()

    # model="deepdta" is what keeps this from colliding with ColdSite-DTI's
    # checkpoint for the same cell -- same dataset, split, task and seed,
    # different model. The suffix lives in checkpoint_naming so the reader
    # resolves the same path without re-deriving it.
    ckpt = checkpoint_path(args.checkpoint_dir, args.dataset, args.split,
                           args.task, args.seed, model="deepdta")
    os.makedirs(os.path.dirname(ckpt) or ".", exist_ok=True)

    history = []
    selector = CheckpointSelector(patience=args.patience,
                                  min_epochs=args.min_epochs,
                                  n_epochs=args.epochs)
    for epoch in range(1, args.epochs + 1):
        train_loss, _, _ = run_epoch(model, loaders["train"], loss_fn, device,
                                     optimizer)
        val_loss, val_true, val_pred = run_epoch(model, loaders["valid"],
                                                 loss_fn, device)
        metrics = compute_metrics(val_true, val_pred, args.task)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, **metrics})
        print(f"  epoch {epoch:>3} train {train_loss:.4f} val {val_loss:.4f} "
              + " ".join(f"{k} {v:.4f}" for k, v in metrics.items()))

        if selector.consider(epoch, val_loss):
            torch.save({"model_state": model.state_dict(), "epoch": epoch,
                        "args": vars(args)}, ckpt)
        elif selector.should_stop(epoch):
            print(f"  early stop at epoch {epoch} "
                  f"(best {selector.best_epoch})")
            break

    model.load_state_dict(torch.load(ckpt, map_location=device,
                                     weights_only=False)["model_state"])
    _loss, test_true, test_pred = run_epoch(model, loaders["test"], loss_fn, device)
    test_metrics = compute_metrics(test_true, test_pred, args.task)

    with open(out_path, "w") as handle:
        json.dump({"tag": tag, "model": "deepdta", "dataset": args.dataset,
                   "split": args.split, "task": args.task, "seed": args.seed,
                   "checkpoint": ckpt, "best_epoch": selector.best_epoch,
                   "selection": selector.summary(),
                   "n_train_rows": len(loaders["train"].dataset),
                   "test_metrics": test_metrics}, handle, indent=2)

    print("\nTest metrics:", json.dumps(test_metrics, indent=2))
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
