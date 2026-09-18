"""The paper's figures, drawn from the committed analysis files.

An audit is read off its figures before its tables, so these are built by code from the
same JSON the tables quote -- never by hand, and never from a number typed twice.

    python -m src.evaluation.paper_figures --out-dir results/figures

Four figures:

  fig1_plausibility   precision@10 against both ground truths, both datasets, every model
                      and level, with each cell's three seeds shown as dots and chance as
                      a line. The paper's headline: nothing clears chance against annotated
                      residues; the pocket signal is real and coarse.
  fig2_seeds          the same UniProt cells as a per-seed strip, which is what makes the
                      replication's seed dependence visible rather than hidden in a +/-.
  fig3_attention_vs_ig  attention against integrated gradients on the same checkpoints
                      (DAVIS only, as measured).
  fig4_faithfulness   comprehensiveness delta over a size-matched control, both datasets;
                      MolTrans in token space, the others in residue space (the units
                      differ, so its panel is drawn separately and labelled).

Each figure is written as PDF (for the manuscript) and PNG (for drafts), and every panel
carries the n it rests on. Missing inputs are skipped with a printed reason: a figure that
silently drops a model would misrepresent the audit.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

LEVELS = ("random", "cold_drug", "cold_target", "cold_pair")
LEVEL_LABEL = {"random": "warm", "cold_drug": "cold-drug",
               "cold_target": "cold-target", "cold_pair": "cold-pair"}
MODELS = ("coldsite_dti", "hyperattentiondti", "moltrans")
MODEL_LABEL = {"coldsite_dti": "ColdSite-DTI (ours)",
               "hyperattentiondti": "HyperAttentionDTI", "moltrans": "MolTrans",
               "uniform_control": "uniform control"}
COLOUR = {"coldsite_dti": "#4C72B0", "hyperattentiondti": "#C44E52",
          "moltrans": "#55A868", "uniform_control": "#999999"}
SEEDS = (1, 2, 3)


def ladder_path(folder: str, model: str, dataset: str, seed: int) -> str:
    stem = (f"ladder_{dataset}_seed{seed}.json" if model == "coldsite_dti"
            else f"ladder_{model}_{dataset}_seed{seed}.json")
    return os.path.join(folder, stem)


def read_ladder_cells(folder: str, model: str, dataset: str, k: str = "10") -> dict:
    """{level: {"values": [per seed], "chance": float, "n": int}} for what exists."""
    out = {}
    for level in LEVELS:
        values, chance, n = [], None, None
        for seed in SEEDS:
            path = ladder_path(folder, model, dataset, seed)
            if not os.path.exists(path):
                continue
            payload = json.load(open(path))
            if level not in payload or k not in payload[level].get("by_k", {}):
                continue
            cell = payload[level]["by_k"][k]
            values.append(cell["precision_at_k"])
            chance, n = cell["chance"], cell["n_evaluated"]
        if values:
            out[level] = {"values": values, "chance": chance, "n": n}
    return out


def read_faithfulness(folder: str, model: str, dataset: str, token: bool = False) -> dict:
    out = {}
    for level in LEVELS:
        values = []
        for seed in SEEDS:
            # ColdSite-DTI's outputs carry no model name (it is the default model), the
            # same asymmetry ladder_path handles.
            if token:
                stem = f"token_faithfulness_{model}_{dataset}_seed{seed}.json"
            elif model == "coldsite_dti":
                stem = f"faithfulness_{dataset}_seed{seed}.json"
            else:
                stem = f"faithfulness_{model}_{dataset}_seed{seed}.json"
            path = os.path.join(folder, stem)
            if not os.path.exists(path):
                continue
            payload = json.load(open(path))
            levels = payload if token else payload.get("levels", {})
            if level in levels and "comprehensiveness_delta" in levels[level]:
                values.append(levels[level]["comprehensiveness_delta"])
        if values:
            out[level] = {"values": values}
    return out


def _panel(ax, cells_by_model, title, ylabel, chance_label="chance"):
    """Grouped bars (mean over seeds) with each seed as a dot and chance as a line."""
    levels = [lv for lv in LEVELS if any(lv in c for c in cells_by_model.values())]
    models = [m for m, c in cells_by_model.items() if c]
    width = 0.8 / max(len(models), 1)

    for i, model in enumerate(models):
        cells = cells_by_model[model]
        xs, means, seeds_x, seeds_y = [], [], [], []
        for j, level in enumerate(levels):
            if level not in cells:
                continue
            values = cells[level]["values"]
            x = j - 0.4 + width * (i + 0.5)
            xs.append(x)
            means.append(float(np.mean(values)))
            seeds_x += [x] * len(values)
            seeds_y += list(values)
        ax.bar(xs, means, width=width * 0.92, color=COLOUR[model],
               label=MODEL_LABEL[model], zorder=2)
        ax.scatter(seeds_x, seeds_y, s=11, color="white", edgecolor="black",
                   linewidth=0.5, zorder=3)

    # Chance is per level (it depends on that level's proteins, their lengths and site
    # counts), so it is drawn as a segment over each group rather than as one line
    # across the panel, which would be the last level's value pretending to be all of them.
    for j, level in enumerate(levels):
        chances = [c[level]["chance"] for c in cells_by_model.values()
                   if level in c and c[level].get("chance") is not None]
        if not chances:
            continue
        ax.plot([j - 0.44, j + 0.44], [np.mean(chances)] * 2, color="black",
                linestyle="--", linewidth=1.1, zorder=4,
                label=chance_label if j == 0 else None)
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([LEVEL_LABEL[lv] for lv in levels],
                       fontsize=8 if len(levels) < 3 else 7,
                       rotation=0 if len(levels) < 3 else 20,
                       ha="center" if len(levels) < 3 else "right")
    ax.set_title(title, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(axis="y", linewidth=0.3, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    everything = [v for c in cells_by_model.values() for lv in c for v in c[lv]["values"]]
    top, bottom = max(everything or [0]), min(everything or [0])
    ax.set_ylim(min(0.0, bottom * 1.18), top * 1.18)


def figure_plausibility(paths: dict, out_dir: str) -> str:
    import matplotlib.pyplot as plt

    panels = [
        ("UniProt annotated residues — DAVIS", paths["davis_uniprot"], "davis"),
        ("UniProt annotated residues — KIBA", paths["kiba_uniprot"], "kiba"),
        ("KLIFS ATP pocket — DAVIS", paths["davis_klifs"], "davis"),
        ("KLIFS ATP pocket — KIBA", paths["kiba_klifs"], "kiba"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
    for ax, (title, folder, dataset) in zip(axes.ravel(), panels):
        cells = {m: read_ladder_cells(folder, m, dataset) for m in MODELS}
        n = {lv: c[lv]["n"] for m in cells for lv, c in [(k, cells[m]) for k in cells[m]]}
        _panel(ax, cells, title, "precision@10")
        counts = sorted({v for v in n.values()})
        if counts:
            ax.annotate("n = " + ", ".join(str(c) for c in counts) + " proteins",
                        xy=(0.015, 0.975), xycoords="axes fraction", fontsize=6.5,
                        va="top", ha="left", color="#444444")
    # Fixed order (models, then the chance line), so the legend cannot reflow into two
    # ragged rows depending on which cells a run happens to have.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    order = [labels.index(MODEL_LABEL[m]) for m in MODELS if MODEL_LABEL[m] in labels]
    order += [i for i, lab in enumerate(labels) if lab == "chance"]
    fig.legend([handles[i] for i in order], [labels[i] for i in order],
               loc="lower center", ncol=len(order), fontsize=8, frameon=False)
    fig.suptitle("Does attention mark the binding site? Dots are training seeds.",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    return _save(fig, out_dir, "fig1_plausibility")


def figure_seeds(paths: dict, out_dir: str) -> str:
    """One row per cell, three dots: the figure that makes seed dependence visible."""
    import matplotlib.pyplot as plt

    rows, labels, chances = [], [], []
    for dataset, folder in (("DAVIS", paths["davis_uniprot"]), ("KIBA", paths["kiba_uniprot"])):
        for model in ("hyperattentiondti", "moltrans"):
            cells = read_ladder_cells(folder, model, dataset.lower())
            for level in LEVELS:
                if level not in cells:
                    continue
                rows.append((model, cells[level]["values"]))
                labels.append(f"{dataset}  {MODEL_LABEL[model]}  {LEVEL_LABEL[level]}")
                chances.append(cells[level]["chance"])

    fig, ax = plt.subplots(figsize=(7.2, 0.30 * len(rows) + 1.4))
    for y, ((model, values), chance) in enumerate(zip(rows, chances)):
        ax.plot([chance, chance], [y - 0.35, y + 0.35], color="black",
                linestyle="--", linewidth=0.9, zorder=2)
        ax.scatter(values, [y] * len(values), s=26, color=COLOUR[model],
                   edgecolor="black", linewidth=0.4, zorder=3)
        ax.plot([min(values), max(values)], [y, y], color=COLOUR[model],
                linewidth=1.2, alpha=0.5, zorder=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("precision@10 against UniProt annotated residues (dashed line = chance)",
                  fontsize=8)
    ax.set_title("One cell, three training seeds: a verdict that moves across chance",
                 fontsize=9)
    ax.tick_params(labelsize=7)
    ax.grid(axis="x", linewidth=0.3, alpha=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return _save(fig, out_dir, "fig2_seeds")


def figure_attention_vs_ig(ig_paths: dict, attn_paths: dict, out_dir: str) -> str:
    """Attention against integrated gradients on the same checkpoints, both datasets.

    Read from the ladder files of both explainers rather than from the summary CSV, so
    every model that has an IG ladder appears (the CSV predates MolTrans's rows) and each
    cell keeps its three seeds. KIBA's row exists from 2026-09-18: two audited models at
    the two trained levels, so it is deliberately shorter than DAVIS's.
    """
    import matplotlib.pyplot as plt

    rows = [("davis", "DAVIS", MODELS),
            ("kiba", "KIBA", [m for m in MODELS if m != "coldsite_dti"])]
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.0))
    for row, (dataset, dataset_label, models) in enumerate(rows):
        for col, truth in enumerate(("uniprot", "klifs")):
            ax = axes[row][col]
            attn_folder = attn_paths[f"{dataset}_{truth}"]
            ig_folder = ig_paths[f"{dataset}_{truth}"]
            labels, x, seen = [], 0, []
            for model in models:
                attn = read_ladder_cells(attn_folder, model, dataset)
                ig = read_ladder_cells(ig_folder, model + "_ig", dataset)
                for level in LEVELS:
                    if level not in attn or level not in ig:
                        continue
                    a, g = attn[level]["values"], ig[level]["values"]
                    ax.bar(x - 0.19, np.mean(a), width=0.36, color="#C44E52",
                           label="attention" if not seen else None, zorder=2)
                    ax.bar(x + 0.19, np.mean(g), width=0.36, color="#4C72B0",
                           label="integrated gradients" if not seen else None, zorder=2)
                    ax.scatter([x - 0.19] * len(a) + [x + 0.19] * len(g), list(a) + list(g),
                               s=9, color="white", edgecolor="black", linewidth=0.4, zorder=3)
                    ax.plot([x - 0.42, x + 0.42], [attn[level]["chance"]] * 2, color="black",
                            linestyle="--", linewidth=1.0, zorder=4,
                            label="chance" if not seen else None)
                    labels.append(f'{MODEL_LABEL[model].split(" (")[0]}\n{LEVEL_LABEL[level]}')
                    seen.append(model)
                    x += 1
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, fontsize=5.2, rotation=90)
            ax.set_title(f"{dataset_label} — "
                         + ("UniProt annotated residues" if truth == "uniprot"
                            else "KLIFS ATP pocket"), fontsize=9)
            ax.set_ylabel("precision@10", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(axis="y", linewidth=0.3, alpha=0.5)
            ax.set_axisbelow(True)
    axes[0][0].legend(fontsize=6.5, frameon=False, loc="upper left")
    fig.suptitle("The same checkpoints, read two ways (secondary analysis)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, out_dir, "fig3_attention_vs_ig")


def figure_faithfulness(paths: dict, out_dir: str) -> str:
    """Three panels, because the unit differs: residues for two models, tokens for MolTrans.

    Plotting MolTrans's token-space delta on the same axis as a residue-space delta would
    invite a comparison the measurement does not support (Results 5b), so it gets its own
    panel and its own label.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.1))
    _panel(axes[0], {m: read_faithfulness(paths["davis_uniprot"], m, "davis")
                     for m in ("coldsite_dti", "hyperattentiondti")},
           "residues masked — DAVIS", "comprehensiveness delta")
    _panel(axes[1], {"hyperattentiondti": read_faithfulness(paths["kiba_uniprot"],
                                                            "hyperattentiondti", "kiba")},
           "residues masked — KIBA", "")
    token = {"davis": read_faithfulness(paths["davis_uniprot"], "moltrans", "davis",
                                        token=True),
             "kiba": read_faithfulness(paths["kiba_uniprot"], "moltrans", "kiba",
                                       token=True)}
    ax, labels, x = axes[2], [], 0
    for dataset, cells in token.items():
        for level in LEVELS:
            if level not in cells:
                continue
            values = cells[level]["values"]
            ax.bar([x], [float(np.mean(values))], width=0.7, color=COLOUR["moltrans"],
                   zorder=2)
            ax.scatter([x] * len(values), values, s=11, color="white", edgecolor="black",
                       linewidth=0.5, zorder=3)
            short = {"random": "warm", "cold_drug": "c-drug",
                     "cold_target": "c-target", "cold_pair": "c-pair"}[level]
            labels.append(f"{dataset.upper()[0]} {short}")
            x += 1
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=6.5, rotation=30, ha="right")
    ax.annotate("D = DAVIS, K = KIBA", xy=(0.02, 0.97), xycoords="axes fraction",
                fontsize=6, va="top", color="#444444")
    ax.set_title("tokens masked — MolTrans", fontsize=9)
    token_values = [v for c in token.values() for lv in c for v in c[lv]["values"]]
    ax.set_ylim(min(0.0, min(token_values) * 1.18), max(token_values) * 1.22)
    ax.tick_params(labelsize=7)
    ax.grid(axis="y", linewidth=0.3, alpha=0.5)
    ax.set_axisbelow(True)

    for a in axes:
        a.axhline(0.0, color="black", linewidth=0.8, zorder=4)
    handles, lab = axes[0].get_legend_handles_labels()
    keep = [(h, l) for h, l in zip(handles, lab) if l != "chance"]
    keep.append((plt.Rectangle((0, 0), 1, 1, color=COLOUR["moltrans"]), "MolTrans"))
    fig.legend([h for h, _ in keep], [l for _, l in keep], loc="lower center",
               ncol=len(keep), fontsize=8, frameon=False)
    fig.suptitle("Is the attention load-bearing? Delta over a size-matched control; "
                 "above zero = yes", fontsize=9.5)
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    return _save(fig, out_dir, "fig4_faithfulness")


def _save(fig, out_dir: str, stem: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    pdf = os.path.join(out_dir, stem + ".pdf")
    fig.savefig(pdf)
    fig.savefig(os.path.join(out_dir, stem + ".png"), dpi=200)
    import matplotlib.pyplot as plt

    plt.close(fig)
    print(f"Saved -> {pdf} (+ .png)")
    return pdf


def main():
    parser = argparse.ArgumentParser(description="Build the paper's figures")
    parser.add_argument("--out-dir", default="results/figures")
    parser.add_argument("--davis-uniprot", default="results/analysis_davis_policyA")
    parser.add_argument("--davis-klifs", default="results/analysis_davis_policyA_klifs")
    parser.add_argument("--kiba-uniprot", default="results/analysis_kiba_policyA")
    parser.add_argument("--kiba-klifs", default="results/analysis_kiba_policyA_klifs")
    parser.add_argument("--ig-uniprot",
                        default=os.path.expanduser(
                            "~/ColdSite-results/integrated_gradients/ig_davis"))
    parser.add_argument("--ig-klifs",
                        default=os.path.expanduser(
                            "~/ColdSite-results/integrated_gradients/ig_davis_klifs"))
    parser.add_argument("--ig-kiba-uniprot",
                        default=os.path.expanduser("~/ColdSite-results/ig_kiba"))
    parser.add_argument("--ig-kiba-klifs",
                        default=os.path.expanduser("~/ColdSite-results/ig_kiba_klifs"))
    parser.add_argument("--only", default="", help="comma-separated figure numbers")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")

    paths = {"davis_uniprot": args.davis_uniprot, "davis_klifs": args.davis_klifs,
             "kiba_uniprot": args.kiba_uniprot, "kiba_klifs": args.kiba_klifs}
    for name, folder in paths.items():
        if not os.path.isdir(folder):
            raise SystemExit(f"missing input folder for {name}: {folder}")

    wanted = {s.strip() for s in args.only.split(",") if s.strip()}

    def run(number, fn, *fn_args):
        if wanted and number not in wanted:
            return
        fn(*fn_args, args.out_dir)

    if not wanted or "0" in wanted:
        figure_design(args.out_dir)
    run("1", figure_plausibility, paths)
    run("2", figure_seeds, paths)
    ig_paths = {"davis_uniprot": args.ig_uniprot, "davis_klifs": args.ig_klifs,
                "kiba_uniprot": args.ig_kiba_uniprot, "kiba_klifs": args.ig_kiba_klifs}
    if all(os.path.isdir(p) for p in ig_paths.values()):
        run("3", figure_attention_vs_ig, ig_paths, paths)
    else:
        print(f"[skip] figure 3: no IG ladders at {ig_paths}")
    run("4", figure_faithfulness, paths)




# ---------------------------------------------------------------------------
# The design schematic: what was trained, explained, measured and corrected
# ---------------------------------------------------------------------------

def figure_design(out_dir: str) -> str:
    """One picture of the audit, so a reader knows what is being compared before any bar.

    Hand-laid rather than data-driven -- it describes the protocol, not a result -- so the
    counts in it are the ones Methods states and must be updated with them.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 11.4)
    ax.axis("off")

    def box(x, y, w, h, title, body, colour="#F2F4F8", edge="#4C4C4C"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.10",
                                    linewidth=0.9, edgecolor=edge, facecolor=colour))
        ax.text(x + w / 2, y + h - 0.26, title, ha="center", va="top", fontsize=8.4,
                fontweight="bold")
        ax.text(x + w / 2, y + h - 0.60, body, ha="center", va="top", fontsize=6.9,
                linespacing=1.45)

    def arrow(x, y0, y1):
        ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>", mutation_scale=9,
                                     linewidth=0.9, color="#4C4C4C"))

    box(0.2, 9.7, 4.6, 1.4, "DAVIS  (primary)",
        "30,056 pairs · 68 drugs · 442 targets\n"
        "4 levels: random, cold-drug,\ncold-target, cold-pair", colour="#E8EEF7")
    box(5.2, 9.7, 4.6, 1.4, "KIBA  (replication)",
        "118,254 pairs · 2,111 drugs · 229 targets\n"
        "2 levels: random, cold-drug\n(422 held-out drugs vs DAVIS's 13)", colour="#EDF5EC")

    arrow(2.5, 9.6, 9.05)
    arrow(7.5, 9.6, 9.05)
    box(0.2, 7.5, 9.6, 1.5, "Retrained on identical splits — 3 seeds per cell",
        "ColdSite-DTI (ours) · HyperAttentionDTI · MolTrans      +  DeepDTA (accuracy anchor, no attention)\n"
        "48 DAVIS cells  ·  18 KIBA cells  ·  every cell scored by AUROC before its explanation is read",
        colour="#FAF4E8")

    arrow(5.0, 7.4, 6.85)
    box(0.2, 5.3, 9.6, 1.5, "The explanation, read three ways",
        "attention (as published)   ·   integrated gradients on the same weights   ·   uniform map = the floor\n"
        "alternative attention readouts test whether a verdict belongs to the model or to the reduction",
        colour="#F7EDF3")

    arrow(2.7, 5.2, 4.60)
    arrow(7.3, 5.2, 4.60)
    box(0.2, 2.35, 4.6, 2.15, "PLAUSIBILITY — does it point there?",
        "precision@10 against\n· UniProt annotated residues\n· KLIFS 85-residue ATP pocket\n"
        "· the drug's own crystal contacts\nnulls: borrowed map, same amino acid,\n"
        "within the site-spanning stretch,\n60 unseen non-kinase proteins", colour="#EDF1F7")
    box(5.2, 2.35, 4.6, 2.15, "FAITHFULNESS — is it used?",
        "mask the top-10 and re-predict,\nagainst a control of the same size\nin the space each model reads\n"
        "(residues; tokens for MolTrans)\n\ndelta > 0  =  load-bearing", colour="#EDF7F1")

    arrow(5.0, 2.30, 1.85)
    box(0.2, 0.25, 9.6, 1.5, "One correction per arm, and a control for the metric itself",
        "Holm–Bonferroni once over the whole family: 16 DAVIS cells, 6 KIBA cells, 12 for the gradient\n"
        "positive control: a planted explanation of known dose is detectable at 2% — so a null is a null, not a weak test\n"
        "bootstrap CIs over proteins · every cell reported with its three seeds",
        colour="#F2F2F2")

    ax.text(5.0, 11.25, "The audit", ha="center", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return _save(fig, out_dir, "fig0_design")

if __name__ == "__main__":
    main()
