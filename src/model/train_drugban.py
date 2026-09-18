"""
Train the vendored DrugBAN (Bai et al., *Nature Machine Intelligence* 2023) on our splits.

Why this model is in the audit
------------------------------
The other three subjects are 2020-2022. DrugBAN is the current-generation one: its title
claims interpretability, it reports cross-domain (cluster-split) generalisation, and its
code is maintained and MIT-licensed. Whether the audit's verdict reaches models people
build now is exactly what its cells answer.

Why a separate trainer
----------------------
Same reasoning as the other three: DrugBAN featurises a drug as a DGL graph
(`dgllife.utils.smiles_to_bigraph` with the canonical atom featuriser, padded to 290
nodes with virtual nodes) and a protein as its own 25-letter integer encoding, so it
cannot use `src/model/dataset.py`'s vocabulary. Everything downstream -- run tag,
checkpoint path, results JSON, metrics -- comes from the shared modules, so a DrugBAN
cell lands in the same shape as any other and nothing downstream needs a special case.

Their recipe, kept
------------------
Adam at 5e-5, batch 64, up to 100 epochs (`baselines/DrugBAN/configs.py`), their model
built from their own config object. What this trainer changes, and why:

* **Early stopping on validation loss**, with our shared `CheckpointSelector` (patience
  15, minimum 10 epochs). Their script trains a fixed 100 epochs and selects on
  validation AUROC. Ours is the protocol every other cell in this audit uses, and a cell
  selected differently from the cells it is compared against is not comparable.
* **`BCEWithLogitsLoss` on the single logit**, rather than their `CrossEntropyLoss` over
  a 2-way head: `DECODER.BINARY = 1` gives one logit, and the adapter reads that same
  logit as the prediction, so faithfulness and AUROC score the same quantity.
* **Domain adaptation off** (`DA.USE = False`, their default for in-domain runs). The
  audit trains every model on the same splits with no adaptation; CDAN would give
  DrugBAN a mechanism no other subject has.

Batching graphs
---------------
`dgl.batch` is not `torch.stack`: a batch of graphs is one disjoint graph, and the
collate function has to build it. `MolecularGCN.forward` then reshapes node features by
`batch_graph.batch_size`, which is why every drug is padded to the same 290 nodes -- with
ragged graphs that reshape would silently mix atoms between molecules.

DGL has no macOS-ARM wheel on PyPI; conda-forge has one (`dgl` 2.3, osx-arm64). The
import is deferred to `main()` so that importing this module -- which the test suite
does -- works on a machine without it.
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

from src.model import precision, resume
from src.model.checkpoint_naming import checkpoint_path, results_path, run_tag
from src.model.dataset import BINARY_THRESHOLD
from src.model.early_stopping import DEFAULT_MIN_EPOCHS, CheckpointSelector
from src.model.train import compute_metrics

RESUME_KEYS = ("amp", "batch_size", "lr", "epochs", "patience", "min_epochs")


class DrugBANDataset(Dataset):
    """Split CSV -> (DGL drug graph, protein tensor, label).

    Encoded once up front, like every other trainer here: KIBA is 118k rows and
    re-running RDKit per access would make the loader the bottleneck.
    """

    def __init__(self, csv_path: str, threshold: float):
        from src.evaluation.drugban_adapter import DrugBANAdapter

        frame = pd.read_csv(csv_path)
        for column in ("Drug", "Target", "Y"):
            if column not in frame.columns:
                raise KeyError(f"{csv_path} has no '{column}' column")

        # One encode per distinct molecule: DAVIS has 68 drugs over 30k rows.
        cache: dict = {}
        self.graphs, self.proteins = [], []
        for smiles, sequence in zip(frame["Drug"], frame["Target"]):
            key = str(smiles)
            if key not in cache:
                cache[key] = DrugBANAdapter.encode(key, "A")[0]
            self.graphs.append(cache[key])
            self.proteins.append(DrugBANAdapter.encode("C", str(sequence))[1])
        self.y = (frame["Y"].to_numpy(dtype=float) >= threshold).astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return (self.graphs[index], self.proteins[index],
                torch.tensor(self.y[index], dtype=torch.float32))


def collate(batch):
    """dgl.batch the graphs, stack the rest."""
    import dgl

    graphs, proteins, labels = zip(*batch)
    return dgl.batch(graphs), torch.stack(proteins), torch.stack(labels)


def run_epoch(model, loader, loss_fn, device, optimizer=None,
              log_every: int = 20, label: str = "", scaler=None, amp: bool = False):
    training = optimizer is not None
    model.train(training)
    total, n, logits, trues = 0.0, 0, [], []
    n_batches = len(loader)
    if training and scaler is None:
        scaler = precision.make_scaler(device, False)

    if n_batches == 0:
        raise ValueError(
            f"{label or 'this loader'} has no batches: {len(loader.dataset)} rows at "
            f"batch size {loader.batch_size} with drop_last="
            f"{bool(getattr(loader, 'drop_last', False))}. A training split smaller than "
            f"one batch drops its only batch and the epoch sees nothing -- lower "
            f"--batch-size for this cell. (DAVIS's smallest split is cold_pair's 15,190 "
            f"rows, so this only bites on a subset.)")

    for batch_index, (bg_d, v_p, y) in enumerate(loader, start=1):
        bg_d, v_p, y = bg_d.to(device), v_p.to(device), y.to(device)
        with torch.set_grad_enabled(training), precision.autocast(device, amp):
            _v_d, _v_p, _f, score = model(bg_d, v_p)
            out = score.squeeze(-1)
            loss = loss_fn(out, y)
        if training:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            precision.step(optimizer, scaler, model.parameters(), clip=5.0)

        batch_loss = float(loss.item())
        # Same guard as train_moltrans: stop at the batch that diverges rather than at
        # the end of the epoch, and say which model and which knob.
        if batch_loss != batch_loss or batch_loss in (float("inf"), float("-inf")):
            raise ValueError(
                f"loss is {batch_loss} at {label} batch {batch_index}/{n_batches}: this "
                "cell has diverged and cannot recover, because a NaN in the forward pass "
                "is already in the weights. Re-run this cell without --amp.")
        total += batch_loss * len(y)
        n += len(y)
        logits.append(out.detach().float().cpu().numpy())
        trues.append(y.detach().cpu().numpy())

        if log_every and (batch_index % log_every == 0 or batch_index == n_batches):
            print(f"\r    {label} batch {batch_index}/{n_batches} "
                  f"loss {total / max(n, 1):.4f}", end="", flush=True)
    if log_every:
        print()

    return total / max(n, 1), np.concatenate(trues), np.concatenate(logits)


def main():
    parser = argparse.ArgumentParser(description="Train DrugBAN on our splits")
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--dataset", required=True, choices=["davis", "kiba"])
    parser.add_argument("--split", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=100,
                        help="their SOLVER.MAX_EPOCH")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="their SOLVER.BATCH_SIZE")
    parser.add_argument("--lr", type=float, default=5e-5, help="their SOLVER.LR")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-epochs", type=int, default=DEFAULT_MIN_EPOCHS)
    parser.add_argument("--checkpoint-dir", default="results")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-if-done", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--log-every", type=int, default=20,
                        help="batches between progress lines; 0 prints none. Kaggle "
                             "renders the whole log to HTML when a commit ends, and a "
                             "20,000-line log took longer to render than the twelve cells "
                             "took to train (2026-09-18). Raise this for long grids.")
    parser.add_argument("--stop-after-epoch", type=int,
                        help="testing only: behave as if killed after this epoch")
    args = parser.parse_args()

    tag = run_tag(args.dataset, args.split, "binary", args.seed)
    out_path = results_path(args.results_dir, tag, model="drugban")
    if args.skip_if_done and os.path.exists(out_path):
        print(f"already done, skipping -> {out_path}")
        return

    from src.evaluation.baseline_adapters import _vendored
    from src.evaluation.drugban_adapter import DrugBANAdapter, _config

    _vendored("DrugBAN", DrugBANAdapter.clone_hint)
    from models import DrugBAN  # noqa: E402

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    threshold = BINARY_THRESHOLD[args.dataset]

    loaders, datasets = {}, {}
    for part in ("train", "valid", "test"):
        path = os.path.join(args.split_dir, f"{part}.csv")
        if not os.path.exists(path):
            raise SystemExit(f"{path} not found. Run build_splits first.")
        datasets[part] = DrugBANDataset(path, threshold)
        # drop_last on TRAIN ONLY: the decoder's BatchNorm needs >1 row in training mode,
        # and dropping a partial batch of validation or test rows would silently shrink
        # an already-small split.
        loaders[part] = DataLoader(datasets[part], batch_size=args.batch_size,
                                   shuffle=(part == "train"),
                                   drop_last=(part == "train"), collate_fn=collate)
        print(f"  {part:5s} {len(datasets[part]):>7,} pairs  "
              f"{float(datasets[part].y.mean()):.1%} positive  <- {path}")

    device = args.device
    print(f"\n  device        {device}")
    print(f"  batch size    {args.batch_size} (vendored: 64)")
    print(f"  precision     {'mixed (float16 autocast)' if args.amp else 'float32'}")

    model = DrugBAN(**_config()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    ckpt = checkpoint_path(args.checkpoint_dir, args.dataset, args.split,
                           "binary", args.seed, model="drugban")
    os.makedirs(os.path.dirname(ckpt) or ".", exist_ok=True)

    selector = CheckpointSelector(patience=args.patience, min_epochs=args.min_epochs,
                                  n_epochs=args.epochs)
    scaler = precision.make_scaler(device, args.amp)
    run = resume.Resumable(ckpt, device, vars(args), RESUME_KEYS, args.stop_after_epoch)
    start = run.begin(model, optimizer, selector, scaler=scaler)
    for epoch in (range(start, args.epochs + 1) if start else ()):
        train_loss, _, _ = run_epoch(model, loaders["train"], loss_fn, device, optimizer,
                                     label=f"epoch {epoch} train", scaler=scaler,
                                     amp=args.amp, log_every=args.log_every)
        val_loss, val_true, val_score = run_epoch(model, loaders["valid"], loss_fn,
                                                  device, label=f"epoch {epoch} valid",
                                                  amp=args.amp, log_every=args.log_every)
        metrics = compute_metrics(val_true, val_score, "binary")
        print(f"  epoch {epoch:>3} train {train_loss:.4f} val {val_loss:.4f} "
              + " ".join(f"{k} {v:.4f}" for k, v in metrics.items()))

        best = ({"model_state": model.state_dict(), "epoch": epoch, "args": vars(args)}
                if selector.consider(epoch, val_loss) else None)
        stopping = best is None and selector.should_stop(epoch)
        run.end_epoch(epoch, finished=stopping or epoch == args.epochs,
                      best_checkpoint=best)
        if stopping:
            print(f"  early stop at epoch {epoch} (best {selector.best_epoch})")
            break
        if run.interrupt_now(epoch):
            return

    model.load_state_dict(torch.load(ckpt, map_location=device,
                                     weights_only=False)["model_state"])
    _loss, test_true, test_score = run_epoch(model, loaders["test"], loss_fn, device,
                                             label="test", amp=args.amp,
                                             log_every=args.log_every)
    test_metrics = compute_metrics(test_true, test_score, "binary")

    with open(out_path, "w") as handle:
        json.dump({"tag": tag, "model": "drugban", "dataset": args.dataset,
                   "split": args.split, "task": "binary", "seed": args.seed,
                   "checkpoint": ckpt, "best_epoch": selector.best_epoch,
                   "selection": selector.summary(),
                   "batch_size": args.batch_size, "lr": args.lr,
                   "amp": args.amp, "resumed_after_epoch": run.resumed_from,
                   "test_positive_rate": float(datasets["test"].y.mean()),
                   "n_train_rows": len(datasets["train"]),
                   "test_metrics": test_metrics}, handle, indent=2)
    run.clear()

    print("\nTest metrics:", json.dumps(test_metrics, indent=2))
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
