"""Do a cell's three training seeds agree on their own verdict?

Every cell in this audit is trained three times, and the audit reports the median p over
the seeds. That hides a question the field should be asked: what would a paper reporting
ONE seed have concluded? This module answers it from the ladder files -- for each cell,
whether each seed on its own clears alpha, and whether the spread across seeds is larger
than the cell's distance from chance.

The per-seed p-values here are deliberately UNCORRECTED, because that is what a
single-seed report would quote. The audit's own verdicts are elsewhere and are corrected.

    python -m src.evaluation.seed_agreement --out results/seed_agreement.md
"""
from __future__ import annotations

import argparse
import glob
import json
import os

ALPHA = 0.05
LEVEL_LABEL = {"random": "warm", "cold_drug": "cold-drug",
               "cold_target": "cold-target", "cold_pair": "cold-pair"}


def cells(sources: dict, k: str = "10") -> dict:
    """{(model label, dataset, level): [(precision, p, chance), ...]} from ladder files."""
    out: dict = {}
    for (label, dataset), pattern in sources.items():
        for path in sorted(glob.glob(pattern)):
            payload = json.load(open(path))
            for level, entry in payload.items():
                if not isinstance(entry, dict) or "by_k" not in entry:
                    continue
                cell = entry["by_k"][k]
                out.setdefault((label, dataset, level), []).append(
                    (cell["precision_at_k"], cell["p_value"], cell["chance"]))
    return {key: value for key, value in out.items() if len(value) == 3}


def summarise(table: dict, alpha: float = ALPHA) -> dict:
    disagree = agree_all = agree_none = wide = 0
    for seeds in table.values():
        significant = sum(p < alpha for _precision, p, _chance in seeds)
        precisions = [precision for precision, _p, _chance in seeds]
        chance = seeds[0][2]
        disagree += 0 < significant < 3
        agree_all += significant == 3
        agree_none += significant == 0
        wide += (max(precisions) - min(precisions)) > abs(sum(precisions) / 3 - chance)
    return {"cells": len(table), "seeds_disagree": disagree, "all_three": agree_all,
            "none": agree_none, "spread_exceeds_signal": wide}


def report(table: dict, alpha: float = ALPHA) -> str:
    stats = summarise(table, alpha)
    lines = ["# Do a cell's three seeds agree with each other?", "",
             "Each cell trained three times; precision@10 against UniProt's annotated",
             "residues, and the per-seed permutation p **uncorrected** -- what a",
             "single-seed report would have quoted.", "",
             "| model | dataset | level | seed 1 | seed 2 | seed 3 | chance | spread | seeds above alpha |",
             "|---|---|---|---|---|---|---|---|---|"]
    for (label, dataset, level), seeds in sorted(table.items()):
        precisions = [precision for precision, _p, _chance in seeds]
        marks = "".join("*" if p < alpha else "." for _precision, p, _chance in seeds)
        lines.append(
            f"| {label} | {dataset.upper()} | {LEVEL_LABEL.get(level, level)} | "
            + " | ".join(f"{precision:.3f}" for precision in precisions)
            + f" | {seeds[0][2]:.3f} | {max(precisions) - min(precisions):.3f} | `{marks}` |")
    lines += ["", "`*` = that seed alone clears alpha; `.` = it does not.", "",
              f"**{stats['seeds_disagree']} of {stats['cells']} cells have seeds that disagree** "
              f"about their own verdict. In **{stats['spread_exceeds_signal']} of {stats['cells']}** "
              "the spread across seeds is larger than the cell's distance from chance. "
              f"**{stats['all_three']}** cell has all three seeds above alpha; "
              f"**{stats['none']}** have none."]
    return "\n".join(lines) + "\n"


def default_sources(davis_dir: str, kiba_dir: str) -> dict:
    return {
        ("ColdSite-DTI", "davis"): f"{davis_dir}/ladder_davis_seed*.json",
        ("HyperAttentionDTI", "davis"): f"{davis_dir}/ladder_hyperattentiondti_davis_seed*.json",
        ("MolTrans", "davis"): f"{davis_dir}/ladder_moltrans_davis_seed*.json",
        ("HyperAttentionDTI", "kiba"): f"{kiba_dir}/ladder_hyperattentiondti_kiba_seed*.json",
        ("MolTrans", "kiba"): f"{kiba_dir}/ladder_moltrans_kiba_seed*.json",
    }


def main():
    parser = argparse.ArgumentParser(description="Seed agreement per cell")
    parser.add_argument("--davis-dir", default="results/analysis_davis_policyA")
    parser.add_argument("--kiba-dir", default="results/analysis_kiba_policyA")
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--out")
    args = parser.parse_args()

    table = cells(default_sources(args.davis_dir, args.kiba_dir))
    if not table:
        raise SystemExit("no ladder files with three seeds found")
    text = report(table, args.alpha)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        open(args.out, "w").write(text)
        print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
