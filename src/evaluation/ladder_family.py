"""Holm–Bonferroni over a family of ladder cells, from the ladder files themselves.

`run_audit` corrects the attention family as it builds it. The explanation *variants* --
integrated gradients above all -- are scored by `run_ladder`, one file per model and seed,
and their family was corrected by hand for Results §7c. By hand is not a method: this
module reads the ladder JSONs, takes each cell's p as the median over its seeds (the rule
`run_audit` uses, since the smallest of three is cherry-picking), and applies the same
`holm_bonferroni`.

    python -m src.evaluation.ladder_family \
        --ladder-dir ~/ColdSite-results/ig_kiba --dataset kiba \
        --models hyperattentiondti_ig,moltrans_ig --out results/ig_family_kiba.md

The family is whatever cells the run measured, and the output names them, so a reader can
see what was corrected against what. Cells whose file is missing are reported, not skipped
silently: a family that quietly shrinks makes every threshold looser.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics

from src.evaluation.aggregate import holm_bonferroni

LEVELS = ("random", "cold_drug", "cold_target", "cold_pair")


def cells(ladder_dir: str, dataset: str, models, k: str = "10") -> tuple[dict, dict, list]:
    """(p per cell as the median over seeds, precision per cell, missing cells)."""
    p_values, precision, missing = {}, {}, []
    for model in models:
        for level in LEVELS:
            seeds, ps = [], []
            for path in sorted(glob.glob(os.path.join(
                    ladder_dir, f"ladder_{model}_{dataset}_seed*.json"))):
                payload = json.load(open(path))
                if level not in payload or k not in payload[level].get("by_k", {}):
                    continue
                cell = payload[level]["by_k"][k]
                seeds.append(cell["precision_at_k"])
                ps.append(cell["p_value"])
            key = f"{model}|{dataset}|{level}"
            if not seeds:
                missing.append(key)
                continue
            p_values[key] = float(statistics.median(ps))
            precision[key] = {"mean": float(statistics.mean(seeds)),
                              "sd": float(statistics.stdev(seeds)) if len(seeds) > 1 else 0.0,
                              "values": seeds, "n_seeds": len(seeds),
                              "chance": cell["chance"], "n": cell["n_evaluated"]}
    return p_values, precision, missing


def report(p_values: dict, precision: dict, missing: list, title: str) -> str:
    corrected = holm_bonferroni(p_values)
    lines = [f"# {title}", "",
             f"Holm–Bonferroni over **{len(p_values)} cells**, each cell's p the median "
             f"over its seeds.", "",
             "| cell | precision@10 | chance | × chance | p (median) | threshold | survives |",
             "|---|---|---|---|---|---|---|"]
    for key, entry in sorted(corrected.items(), key=lambda kv: kv[1]["p_value"]):
        stats = precision[key]
        ratio = stats["mean"] / stats["chance"] if stats["chance"] else float("nan")
        lines.append(
            f"| `{key}` | {stats['mean']:.3f} ± {stats['sd']:.3f} "
            f"({stats['n_seeds']} seeds) | {stats['chance']:.3f} | {ratio:.2f}× | "
            f"{entry['p_value']:.4g} | {entry['adjusted_alpha']:.4g} | "
            f"{'**yes**' if entry['significant'] else 'no'} |")
    survived = sum(1 for e in corrected.values() if e["significant"])
    lines += ["", f"**{survived} of {len(corrected)} cells survive correction.**"]
    if missing:
        lines += ["", "Cells with no ladder file (not corrected, and not counted in the "
                  "family):", ""] + [f"* `{key}`" for key in missing]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Holm over a family of ladder cells")
    parser.add_argument("--ladder-dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--models", required=True,
                        help="comma-separated, e.g. hyperattentiondti_ig,moltrans_ig")
    parser.add_argument("--k", default="10")
    parser.add_argument("--title", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    p_values, precision, missing = cells(os.path.expanduser(args.ladder_dir),
                                         args.dataset, models, k=args.k)
    if not p_values:
        raise SystemExit(f"no ladder files for {models} in {args.ladder_dir}")
    text = report(p_values, precision, missing,
                  args.title or f"Explanation family — {args.dataset}")
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as handle:
            handle.write(text)
        print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
