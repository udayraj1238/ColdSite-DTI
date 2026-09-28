"""
The whole analysis, in one command, once the grids have trained.

    python -m src.evaluation.run_all --dataset davis --checkpoint-dir results
    python -m src.evaluation.run_all --dataset davis --checkpoint-dir results --dry-run
    python -m src.evaluation.run_all --dataset davis --checkpoint-dir results \\
        --volume-control-dir /content/drive/MyDrive/coldsite-volume-control

What it runs, in order
----------------------
    inventory     which (model, level, seed) cells have a checkpoint AND results
    faithfulness  run_faithfulness, per audited model and seed -- it also
                  writes the accuracy file the ladder draws against
    ladder        run_ladder, per model and seed, fed THAT model's accuracy file
    audit         run_audit once, every audited model + the uniform control,
                  all seeds -- Holm-Bonferroni over the whole family
    control       run_control per model and seed, with the cotransport ions
                  excluded (primary) and included (sensitivity)
    positive      positive_control, every ladder read against its dose curve
    summary       one markdown page: accuracy, plausibility with the Holm
                  verdict, faithfulness, the non-kinase control, the positive
                  control and (DAVIS) the volume-matched control, as means over
                  seeds with their spread

Every step is a real command-line call to the runner that owns it, the way
`run_grid` shells out to `train`: the argument handling and file naming the
paper depends on are what get exercised, not a parallel path through the
functions underneath. The command-by-command version of this used to be about
sixty invocations, typed by hand, where one wrong `--accuracy-json` draws
MolTrans's attention against ColdSite-DTI's accuracy and nothing complains.

Resumable
---------
A step whose output already exists is skipped, so a Colab session that drops
halfway can simply be re-run. `--no-skip-existing` recomputes everything.
Cells that have not been trained yet are skipped by each runner and listed by
the inventory; the summary marks any cell with fewer than three seeds.

DeepDTA is the accuracy anchor: it has no attention, so it appears only in
the accuracy table.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time

from src.model.checkpoint_naming import checkpoint_path, results_path, run_tag

LEVELS = ("random", "cold_drug", "cold_target", "cold_pair")
AUDITED = ("coldsite_dti", "hyperattentiondti", "moltrans", "drugban")
ANCHOR = "deepdta"
STEPS = ("inventory", "faithfulness", "ladder", "audit", "control", "positive",
         "summary")
TASK = "binary"
K = 10


# --------------------------------------------------------------------------
# what exists
# --------------------------------------------------------------------------

def inventory(checkpoint_dir: str, results_dir: str, dataset: str,
              models, seeds) -> dict:
    """{(model, level, seed): 'complete' | 'interrupted' | 'missing'}.

    Complete means checkpoint AND results file: a checkpoint alone is a cell
    that was cut off before its test pass, exactly as the grid notebooks count.
    """
    state = {}
    for model in models:
        for level in LEVELS:
            for seed in seeds:
                ckpt = checkpoint_path(checkpoint_dir, dataset, level, TASK, seed,
                                       model=model)
                res = results_path(results_dir, run_tag(dataset, level, TASK, seed),
                                   model=model)
                state[(model, level, seed)] = (
                    "complete" if os.path.exists(ckpt) and os.path.exists(res)
                    else "interrupted" if os.path.exists(ckpt) else "missing")
    return state


def has_cells(state: dict, model: str, seed: int | None = None) -> bool:
    return any(v == "complete" for (m, _l, s), v in state.items()
               if m == model and (seed is None or s == seed))


def output_tag(model: str, dataset: str, seed: int) -> str:
    from src.evaluation.run_faithfulness import output_tag as tag
    return tag(model, dataset, seed)


# --------------------------------------------------------------------------
# the commands
# --------------------------------------------------------------------------

def _py(module: str, *args) -> list:
    return [sys.executable, "-u", "-m", module, *map(str, args)]


def commands(step: str, cfg: dict, state: dict) -> list:
    """[(label, command, expected_output)] for one step, built when it runs.

    Built lazily rather than all up front, because later steps read files the
    earlier ones write -- the ladder's accuracy file, the positive control's
    ladder JSONs.
    """
    d, out = cfg["dataset"], cfg["out_dir"]
    common = ["--split-root", cfg["split_root"],
              "--checkpoint-dir", cfg["checkpoint_dir"]]
    audited = [m for m in cfg["models"] if m in AUDITED and has_cells(state, m)]
    jobs = []

    if step == "faithfulness":
        for m in audited:
            for s in cfg["seeds"]:
                if not has_cells(state, m, s):
                    continue
                tag = output_tag(m, d, s)
                jobs.append((f"faithfulness {m} seed {s}",
                             _py("src.evaluation.run_faithfulness", "--model", m,
                                 "--task", TASK, "--dataset", d, "--seed", s,
                                 *common, "--results-dir", cfg["results_dir"],
                                 "--out-dir", out, "--max-pairs", cfg["max_pairs"],
                                 "--device", cfg["device"]),
                             os.path.join(out, f"faithfulness_{tag}.json")))

    elif step == "ladder":
        for m in audited:
            for s in cfg["seeds"]:
                if not has_cells(state, m, s):
                    continue
                tag = output_tag(m, d, s)
                accuracy = os.path.join(out, f"accuracy_{tag}.json")
                cmd = _py("src.evaluation.run_ladder", "--model", m, "--task", TASK,
                          "--dataset", d, "--seed", s, *common,
                          "--ground-truth", cfg["ground_truth"], "--out-dir", out,
                          "--device", cfg["device"])
                # Only this model's accuracy, never another's: without it the
                # ladder still runs and simply draws no headline figure.
                if os.path.exists(accuracy):
                    cmd += ["--accuracy-json", accuracy]
                jobs.append((f"ladder {m} seed {s}", cmd,
                             os.path.join(out, f"ladder_{tag}.json")))

    elif step == "audit" and audited:
        jobs.append(("audit grid (Holm over the whole family)",
                     _py("src.evaluation.run_audit",
                         "--models", ",".join(audited + ["uniform_control"]),
                         "--datasets", d, "--seeds", ",".join(map(str, cfg["seeds"])),
                         "--task", TASK, "--ground-truth", cfg["ground_truth"],
                         *common, "--out-dir", out, "--device", cfg["device"]),
                     os.path.join(out, f"audit_{d}_{TASK}.json")))

    elif step == "control":
        for m in audited:
            for s in cfg["seeds"]:
                if not has_cells(state, m, s):
                    continue
                for suffix, extra in (("_noions", ["--exclude-cotransport-ions"]),
                                      ("", [])):
                    jobs.append((f"control {m} seed {s} "
                                 f"{'(primary, no ions)' if extra else '(all ligands)'}",
                                 _py("src.evaluation.run_control", "--model", m,
                                     "--dataset", d, "--seed", s, "--task", TASK,
                                     *common, "--out-dir", out,
                                     "--device", cfg["device"], *extra),
                                 os.path.join(out, f"control_{m}_{d}_seed{s}{suffix}.json")))

    elif step == "positive":
        compare = []
        for m in audited:
            for s in cfg["seeds"]:
                ladder = os.path.join(out, f"ladder_{output_tag(m, d, s)}.json")
                if os.path.exists(ladder):
                    compare += ["--compare", f"{m}_seed{s}={ladder}"]
        jobs.append(("positive control",
                     _py("src.evaluation.positive_control", "--dataset", d,
                         "--ground-truth", cfg["ground_truth"],
                         "--split-root", cfg["split_root"], "--out-dir", out,
                         *compare),
                     os.path.join(out, f"positive_control_{d}.json")))
    return jobs


def run_job(label: str, cmd: list, expected: str, skip_existing: bool,
            log_path: str) -> str:
    """Run one command, streaming its output. Returns ok / skipped / failed."""
    if skip_existing and os.path.exists(expected):
        print(f"[skip] {label} -- {os.path.basename(expected)} exists", flush=True)
        return "skipped"
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}", flush=True)
    started = time.time()
    with open(log_path, "a") as log:
        log.write(f"\n### {label}\n$ {' '.join(cmd)}\n")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1,
                                env={**os.environ, "PYTHONUNBUFFERED": "1"})
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
        code = proc.wait()
    took = time.time() - started
    if code != 0 or not os.path.exists(expected):
        why = f"exited {code}" if code != 0 else f"wrote no {os.path.basename(expected)}"
        print(f"!! {label} FAILED ({why}, {took:.0f}s)", flush=True)
        return "failed"
    print(f"   {label} done in {took:.0f}s", flush=True)
    return "ok"


# --------------------------------------------------------------------------
# the summary page
# --------------------------------------------------------------------------

def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _agg(values):
    from src.evaluation.aggregate import aggregate_seeds
    return aggregate_seeds([v for v in values if v is not None])


def _fmt(entry, digits=3):
    if not entry or not entry.get("n_seeds"):
        return "—"
    mean, std, n = entry["mean"], entry["std"], entry["n_seeds"]
    spread = f" ± {std:.{digits}f}" if n > 1 and std == std else ""
    flag = "" if entry.get("sufficient_seeds") else f" ({n} seed{'s' if n != 1 else ''})"
    return f"{mean:.{digits}f}{spread}{flag}"


def _by_k(entry, k=K):
    by_k = (entry or {}).get("by_k", {})
    return by_k.get(str(k), by_k.get(k))


def summary(cfg: dict, state: dict) -> str:
    d, out, seeds = cfg["dataset"], cfg["out_dir"], cfg["seeds"]
    models = [m for m in cfg["models"]]
    audited = [m for m in models if m in AUDITED]
    lines = [f"# Analysis summary — {d.upper()}, binary", "",
             f"Seeds {', '.join(map(str, seeds))}; mean ± sample sd over seeds; a "
             f"count in brackets marks a cell with fewer than three seeds, which is "
             f"not quoted as an estimate. Generated by `python -m "
             f"src.evaluation.run_all`; every number traces to a file in "
             f"`{out}`.", ""]

    # inventory
    lines += ["## Cells trained", "", "| model | " + " | ".join(LEVELS) + " |",
              "|---|" + "---|" * len(LEVELS)]
    for m in models:
        row = []
        for lv in LEVELS:
            done = sum(state.get((m, lv, s)) == "complete" for s in seeds)
            cut = sum(state.get((m, lv, s)) == "interrupted" for s in seeds)
            row.append(f"{done}/{len(seeds)}" + (f" (+{cut} interrupted)" if cut else ""))
        lines.append(f"| {m} | " + " | ".join(row) + " |")
    lines.append("")

    # accuracy
    from src.evaluation.clean_accuracy import leaks, unseen_auroc
    lines += ["## Accuracy — test AUROC", "", "| model | " + " | ".join(LEVELS) + " |",
              "|---|" + "---|" * len(LEVELS)]
    auroc = {}
    for m in models:
        cells = []
        for lv in LEVELS:
            values = []
            for s in seeds:
                res = _load(results_path(cfg["results_dir"], run_tag(d, lv, TASK, s),
                                         model=m))
                if res:
                    # cold levels on targets unseen by sequence where re-scored
                    # (src/evaluation/clean_accuracy.py)
                    clean = (unseen_auroc(d, m, lv, s, cfg["clean_accuracy"])
                             if leaks(d, lv) else None)
                    values.append(clean if clean is not None
                                  else res.get("test_metrics", {}).get("auroc"))
            auroc[(m, lv)] = _agg(values)
            cells.append(_fmt(auroc[(m, lv)]))
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    lines += ["", f"{ANCHOR} has no attention: it anchors this axis and appears "
              "nowhere below.",
              "Where a level's test set holds targets seen by sequence in training, AUROC is "
              "on the unseen targets (`clean_accuracy`); a cell not yet re-scored keeps its "
              "recorded value.", ""]

    # plausibility, with the Holm verdict from the audit
    audit = _load(os.path.join(out, f"audit_{d}_{TASK}.json")) or {}
    holm = audit.get("p_values_corrected", {})
    lines += [f"## Plausibility — precision@{K} (ladder, one pair per protein)", "",
              "| model | level | precision@10 | normalised | chance | ceiling | "
              "n proteins | Holm-significant |", "|---|---|---|---|---|---|---|---|"]
    for m in audited:
        for lv in LEVELS:
            entries = [_by_k((_load(os.path.join(out, f"ladder_{output_tag(m, d, s)}.json"))
                              or {}).get(lv)) for s in seeds]
            entries = [e for e in entries if e]
            if not entries:
                continue
            verdict = holm.get(f"{m}|{d}|{lv}")
            lines.append(
                f"| {m} | {lv} | {_fmt(_agg([e['precision_at_k'] for e in entries]))} | "
                f"{_fmt(_agg([e['normalised'] for e in entries]))} | "
                f"{_fmt(_agg([e['chance'] for e in entries]))} | "
                f"{_fmt(_agg([e['ceiling'] for e in entries]))} | "
                f"{entries[0].get('n_evaluated', '—')} | "
                + ("—" if verdict is None else
                   f"{'yes' if verdict['significant'] else 'no'} "
                   f"(p = {verdict['p_value']:.3g})") + " |")
    lines += ["", "Holm-Bonferroni over every model × level of the audit grid "
              f"(`audit_{d}_{TASK}.md`), on the median p over seeds. Read "
              "precision beside chance and ceiling: at these n the distance from "
              "chance is the result, not p.", ""]

    # faithfulness
    lines += ["## Faithfulness — comprehensiveness delta over random masking", "",
              "| model | " + " | ".join(LEVELS) + " |", "|---|" + "---|" * len(LEVELS)]
    for m in audited:
        cells, any_cell = [], False
        for lv in LEVELS:
            deltas, load_bearing = [], 0
            for s in seeds:
                entry = ((_load(os.path.join(out, f"faithfulness_{output_tag(m, d, s)}.json"))
                          or {}).get("levels", {}).get(lv))
                if entry:
                    deltas.append(entry.get("comprehensiveness_delta"))
                    load_bearing += bool(entry.get("explanation_is_load_bearing"))
            agg = _agg(deltas)
            any_cell |= bool(agg.get("n_seeds"))
            cells.append(_fmt(agg, 4) + (f", {load_bearing}/{agg['n_seeds']} > 0"
                                         if agg.get("n_seeds") else ""))
        if any_cell:
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
    lines += ["", "Positive delta: masking the attended residues moves the prediction "
              "more than masking random ones (the explanation is load-bearing).", ""]

    # non-kinase control
    lines += ["## Kinase confound — non-kinase transfer panel", "",
              "| model | level | kinase p@10 | non-kinase p@10 (primary: no ions) | "
              "non-kinase p@10 (all ligands) |", "|---|---|---|---|---|"]
    for m in audited:
        for lv in LEVELS:
            kin, noions, allig = [], [], []
            for s in seeds:
                primary = (_load(os.path.join(out, f"control_{m}_{d}_seed{s}_noions.json"))
                           or {}).get("levels", {}).get(lv) or {}
                sensitivity = (_load(os.path.join(out, f"control_{m}_{d}_seed{s}.json"))
                               or {}).get("levels", {}).get(lv) or {}
                if primary.get("kinase"):
                    kin.append(primary["kinase"]["precision_at_k"])
                if primary.get("non_kinase"):
                    noions.append(primary["non_kinase"]["precision_at_k"])
                if sensitivity.get("non_kinase"):
                    allig.append(sensitivity["non_kinase"]["precision_at_k"])
            if kin or noions or allig:
                lines.append(f"| {m} | {lv} | {_fmt(_agg(kin))} | {_fmt(_agg(noions))} | "
                             f"{_fmt(_agg(allig))} |")
    lines += ["", "A transfer condition, not a stratification: 60 non-kinase proteins "
              "no DAVIS/KIBA model has seen. Per-seed tables: `control_*.md`.", ""]

    # positive control
    positive = _load(os.path.join(out, f"positive_control_{d}.json"))
    if positive:
        hard = [msg for ok, is_hard, msg in positive.get("verdicts", []) if is_hard and not ok]
        lines += ["## Positive control", "",
                  ("All hard checks pass." if not hard else
                   "**HARD CHECK FAILED — do not report the audit until fixed:**\n\n"
                   + "\n".join(f"- {h}" for h in hard)),
                  "", f"Full curves: `positive_control_{d}.md`.", ""]
        if positive.get("compare"):
            lines += ["| model | " + " | ".join(LEVELS) + " |", "|---|" + "---|" * len(LEVELS)]
            by_model = {}
            for name, per_level in positive["compare"].items():
                model = name.rsplit("_seed", 1)[0]
                for lv, item in per_level.items():
                    by_model.setdefault(model, {}).setdefault(lv, []).append(item)
            for model, per_level in by_model.items():
                cells = []
                for lv in LEVELS:
                    items = per_level.get(lv, [])
                    inside = [i["equivalent_dose"] for i in items
                              if i.get("equivalent_dose") is not None]
                    reasons = sorted({i.get("reason", "outside the curve") for i in items
                                      if i.get("equivalent_dose") is None})
                    cell = _fmt(_agg(inside)) if inside else ""
                    if reasons:
                        # Seeds that could not be placed say why, never "at chance" by default.
                        cell += (f" [{len(items) - len(inside)} seed(s): " if inside else "") \
                            + "; ".join(reasons) + ("]" if inside else "")
                    cells.append(cell or "—")
                lines.append(f"| {model} | " + " | ".join(cells) + " |")
            lines += ["", "Equivalent dose: the fraction of true sites an explanation "
                      "would have to rank first to match the model's precision@10.", ""]

    # volume-matched control (DAVIS)
    vc_dir = cfg.get("volume_control_dir")
    if vc_dir and d == "davis":
        files = sorted(glob.glob(os.path.join(vc_dir, f"davis_random_{TASK}_seed*_trainsub*_results.json")))
        control = _agg([(_load(f) or {}).get("test_metrics", {}).get("auroc") for f in files])
        full, pair = auroc.get(("coldsite_dti", "random")), auroc.get(("coldsite_dti", "cold_pair"))
        lines += ["## Cold-pair volume-matched control (ColdSite-DTI)", "",
                  "| | test AUROC |", "|---|---|",
                  f"| full random | {_fmt(full)} |",
                  f"| random, trained on cold-pair's volume | {_fmt(control)} |",
                  f"| cold-pair | {_fmt(pair)} |", ""]
        if all(x and x.get("n_seeds") for x in (full, control, pair)):
            lines += [f"- Cost of fewer rows alone: {full['mean'] - control['mean']:+.3f}",
                      f"- Genuine cold-pair difficulty: {control['mean'] - pair['mean']:+.3f}",
                      "", "A difference smaller than the seed spread is not a finding.", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run the whole analysis")
    parser.add_argument("--dataset", required=True, choices=["davis", "kiba"])
    parser.add_argument("--models", default=",".join((ANCHOR, *AUDITED)))
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--checkpoint-dir", default="results")
    parser.add_argument("--results-dir",
                        help="where the *_results.json are (default: checkpoint dir)")
    parser.add_argument("--split-root", default="data/splits")
    parser.add_argument("--ground-truth",
                        help="default: data/<dataset>_ground_truth_sites.json")
    parser.add_argument("--out-dir", help="default: results/analysis_<dataset>")
    parser.add_argument("--steps", default=",".join(STEPS))
    parser.add_argument("--max-pairs", type=int, default=200,
                        help="faithfulness pairs per level")
    parser.add_argument("--volume-control-dir",
                        help="folder holding the _trainsub results (DAVIS only)")
    parser.add_argument("--device", default=None,
                        help="default: cuda if available, else cpu")
    parser.add_argument("--no-skip-existing", action="store_true",
                        help="recompute outputs that already exist")
    parser.add_argument("--clean-accuracy", default=None,
                        help="re-scored cold-level accuracy (default "
                             "results/clean_accuracy_<dataset>.json)")
    parser.add_argument("--allow-interrupted", action="store_true",
                        help="analyse even if some checkpoints have no results file "
                             "(unfinished cells); refused by default")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the inventory and the commands, run nothing")
    args = parser.parse_args()

    if args.device is None:
        try:
            import torch
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            args.device = "cpu"

    cfg = {"dataset": args.dataset,
           "models": [m.strip() for m in args.models.split(",") if m.strip()],
           "seeds": [int(s) for s in args.seeds.split(",") if s.strip()],
           "checkpoint_dir": args.checkpoint_dir,
           "results_dir": args.results_dir or args.checkpoint_dir,
           "split_root": args.split_root,
           "ground_truth": args.ground_truth
           or f"data/{args.dataset}_ground_truth_sites.json",
           "out_dir": args.out_dir or os.path.join("results", f"analysis_{args.dataset}"),
           "max_pairs": args.max_pairs, "device": args.device,
           "volume_control_dir": args.volume_control_dir,
           "clean_accuracy": args.clean_accuracy
           or f"results/clean_accuracy_{args.dataset}.json"}
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    unknown = set(steps) - set(STEPS)
    if unknown:
        parser.error(f"unknown step(s) {sorted(unknown)}; known: {STEPS}")
    os.makedirs(cfg["out_dir"], exist_ok=True)
    log_path = os.path.join(cfg["out_dir"], "run_all.log")

    state = inventory(cfg["checkpoint_dir"], cfg["results_dir"], cfg["dataset"],
                      cfg["models"], cfg["seeds"])
    complete = sum(v == "complete" for v in state.values())
    print(f"{cfg['dataset']}: {complete}/{len(state)} cells complete "
          f"(device {cfg['device']}, outputs -> {cfg['out_dir']})")
    for (m, lv, s), v in sorted(state.items()):
        if v != "complete":
            print(f"   {v:11s} {m} {lv} seed {s}")
    # A checkpoint without its results file is a cell that never finished (a commit cut
    # off mid-training). The analysis tools look only for the checkpoint, so they would
    # score a half-trained model -- which happened on 2026-09-14 with HyperAttentionDTI
    # cold-drug seed 1. Refuse, unless told otherwise.
    interrupted = sorted(k for k, v in state.items() if v == "interrupted")
    if interrupted and not args.allow_interrupted and not args.dry_run:
        raise SystemExit(
            f"{len(interrupted)} checkpoint(s) have no results file -- unfinished cells the "
            f"analysis would silently score: {interrupted}. Move them out of "
            f"{cfg['checkpoint_dir']} (or pass --allow-interrupted).")

    outcomes = {}
    for step in steps:
        if step in ("inventory", "summary"):
            continue
        for label, cmd, expected in commands(step, cfg, state):
            if args.dry_run:
                print(f"\n[{step}] {label}\n  {' '.join(cmd[2:])}\n  -> {expected}")
                continue
            outcomes[label] = run_job(label, cmd, expected,
                                      not args.no_skip_existing, log_path)

    if "summary" in steps and not args.dry_run:
        page = os.path.join(cfg["out_dir"], f"analysis_summary_{cfg['dataset']}.md")
        with open(page, "w") as f:
            f.write(summary(cfg, state))
        print(f"\nSaved -> {page}")

    failed = [label for label, status in outcomes.items() if status == "failed"]
    if failed:
        print(f"\n{len(failed)} step(s) failed (see {log_path}):")
        for label in failed:
            print(f"  - {label}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
