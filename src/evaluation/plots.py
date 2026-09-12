"""
Track C (124AD0067) — the paper's headline figure.

Upgraded from the v1 two-line version. The audit needs four things the single
model figure did not:

  * one fidelity line per audited model, not one line total
  * seed error bars -- a curve without them is one run's idiosyncrasy
  * the uniform-attention control drawn as an explicit floor
  * kinase vs non-kinase as separate panels, so the confound is visible rather
    than argued about in prose

Two refusals are built in on purpose:

  plot_degradation_curve refuses to draw without accuracy values, because
  fidelity plotted alone is half the paper's claim -- the contribution is
  fidelity *against* accuracy across the ladder.

  plot_stratified_panels refuses to draw a family panel built on fewer than 20
  targets. A stratified figure a reviewer cannot believe is worse than an
  openly stated limitation.
"""
import matplotlib

matplotlib.use("Agg")  # no display on the HPC
import matplotlib.pyplot as plt
import numpy as np

LEVELS = ("random", "cold_drug", "cold_target", "cold_pair")
LEVEL_LABELS = ["Warm", "Cold-Drug", "Cold-Target", "Cold-Pair"]
MIN_TARGETS_FOR_A_PANEL = 20


def _series(grid_row, metric="precision_at_k"):
    """Extract (means, stds) in ladder order from one model's grid row."""
    means, stds = [], []
    for level in LEVELS:
        entry = grid_row.get(level, {}).get(metric)
        if entry is None:
            means.append(np.nan)
            stds.append(np.nan)
        else:
            means.append(entry.get("mean", np.nan))
            stds.append(entry.get("std", np.nan))
    return np.asarray(means, float), np.asarray(stds, float)


def plot_degradation_curve(grid: dict, accuracy: dict = None,
                           control_model: str = "uniform_control",
                           title: str = "Explanation fidelity vs accuracy across the ladder",
                           save_path: str = "results/headline_figure.png",
                           metric: str = "precision_at_k",
                           chance: dict = None, ceiling: dict = None):
    """The headline: fidelity per model across the ladder, accuracy beside it.

    grid:     {model_name: {level: {metric: {mean, std, ...}}}}
    accuracy: {model_name: {level: value}} -- REQUIRED
    chance:   {level: precision@k of attention placed at random} -- drawn as a
              line. Real precision@k on DAVIS sits within a few hundredths of
              chance, and the axis auto-scales to the data; without the line a
              0.01 wobble between levels fills the panel and reads as a trend.
    ceiling:  {level: best precision@k any attention could reach} -- stated in
              the panel title rather than drawn. It is near 1.0; on the same
              axis it would flatten every model onto the floor.
    """
    if not accuracy:
        raise ValueError(
            "accuracy is required. A fidelity curve on its own is half the "
            "paper's claim -- the contribution is fidelity plotted against "
            "accuracy across the ladder. Pass {model: {level: auroc_or_ci}}."
        )

    fig, (ax_fid, ax_acc) = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    x = np.arange(len(LEVELS))

    # Drawn first, so every model sits on top of it.
    if chance:
        floor = np.asarray([chance.get(l, np.nan) for l in LEVELS], float)
        ax_fid.fill_between(x, 0, floor, color="0.5", alpha=0.10, zorder=0,
                            label="_nolegend_")
        ax_fid.plot(x, floor, color="black", linestyle=":", linewidth=1.8, zorder=1,
                    label="chance (attention placed at random)")

    for model_name in sorted(grid):
        means, stds = _series(grid[model_name], metric)
        is_control = model_name == control_model
        ax_fid.errorbar(
            x, means, yerr=np.nan_to_num(stds, nan=0.0),
            marker="s" if is_control else "o",
            linestyle="--" if is_control else "-",
            linewidth=2.5 if is_control else 2,
            color="0.45" if is_control else None,
            capsize=4, label=f"{model_name} (control)" if is_control else model_name,
        )

    ax_fid.set_ylabel(f"Explanation fidelity ({metric})")
    fid_title = "Do explanations still point at real binding sites?"
    if ceiling:
        top = np.nanmean([ceiling.get(l, np.nan) for l in LEVELS])
        fid_title += f"\n(perfect attention would score ≈ {top:.2f})"
    ax_fid.set_title(fid_title)

    for model_name in sorted(accuracy):
        values = [accuracy[model_name].get(l, np.nan) for l in LEVELS]
        ax_acc.plot(x, values, marker="^", linewidth=2, label=model_name)

    ax_acc.set_ylabel("Prediction accuracy")
    ax_acc.set_title("Is the prediction still correct?")

    for ax in (ax_fid, ax_acc):
        ax.set_xticks(x)
        ax.set_xticklabels(LEVEL_LABELS, rotation=15)
        ax.set_xlabel("Difficulty level")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_ylim(bottom=0)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Saved figure to {save_path}")
    return save_path


def plot_stratified_panels(stratified: dict, target_counts: dict = None,
                           title: str = "Confound control: kinase vs non-kinase",
                           save_path: str = "results/stratified_figure.png"):
    """Kinase and non-kinase ladders side by side.

    stratified: {model: {level: {"kinase": {...}, "non_kinase": {...}}}}
    target_counts: {"kinase": n, "non_kinase": n} -- gates each panel

    If the two panels look different, the unstratified ladder was measuring
    kinase-family similarity, and that is the paper's real finding.
    """
    from src.evaluation.target_family import KINASE, NON_KINASE

    counts = target_counts or {}
    families = []
    for family in (KINASE, NON_KINASE):
        n = counts.get(family)
        if n is not None and n < MIN_TARGETS_FOR_A_PANEL:
            print(f"Skipping the {family} panel: only {n} targets "
                  f"(need {MIN_TARGETS_FOR_A_PANEL}). State this as a "
                  f"limitation rather than plotting it.")
            continue
        if any(family in levels.get(l, {}) for levels in stratified.values()
               for l in LEVELS):
            families.append(family)

    if not families:
        raise ValueError(
            "No family has enough targets to plot. The confound control does "
            "not yet exist -- see Track A Priority 1 (antiviral rebuild). Do "
            "not present the unstratified ladder as if the confound were absent."
        )

    fig, axes = plt.subplots(1, len(families), figsize=(6.5 * len(families), 5),
                             squeeze=False)
    x = np.arange(len(LEVELS))

    for ax, family in zip(axes[0], families):
        for model_name in sorted(stratified):
            means, stds = [], []
            for level in LEVELS:
                entry = stratified[model_name].get(level, {}).get(family)
                means.append(entry.get("mean", np.nan) if entry else np.nan)
                stds.append(entry.get("std", np.nan) if entry else np.nan)
            ax.errorbar(x, means, yerr=np.nan_to_num(stds, nan=0.0),
                        marker="o", linewidth=2, capsize=4, label=model_name)

        n = counts.get(family)
        ax.set_title(f"{family}" + (f"  (n={n})" if n else ""))
        ax.set_xticks(x)
        ax.set_xticklabels(LEVEL_LABELS, rotation=15)
        ax.set_ylabel("Explanation fidelity")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_ylim(bottom=0)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Saved figure to {save_path}")
    return save_path


def plot_faithfulness(summaries: dict,
                      title: str = "Are the explanations load-bearing?",
                      save_path: str = "results/faithfulness_figure.png"):
    """Comprehensiveness against its random-masking control, per level.

    Plots the observed and random bars together rather than the delta alone,
    because a reader has to see that the gap is the claim -- masking anything
    perturbs the prediction, and the raw height means nothing by itself.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(LEVELS))
    width = 0.35

    observed = [summaries.get(l, {}).get("comprehensiveness", np.nan) for l in LEVELS]
    random = [summaries.get(l, {}).get("comprehensiveness_random", np.nan) for l in LEVELS]

    ax.bar(x - width / 2, observed, width, label="Top-k attended residues masked")
    ax.bar(x + width / 2, random, width, label="Random residues masked (control)",
           color="0.6")

    ax.set_xticks(x)
    ax.set_xticklabels(LEVEL_LABELS, rotation=15)
    ax.set_ylabel("Prediction change when masked")
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Saved figure to {save_path}")
    return save_path


if __name__ == "__main__":
    import os

    os.makedirs("results", exist_ok=True)
    levels = LEVELS

    def cell(mean, std):
        return {"precision_at_k": {"mean": mean, "std": std, "n_seeds": 3,
                                   "sufficient_seeds": True}}

    grid = {
        "coldsite_dti": {l: cell(v, 0.03) for l, v in
                         zip(levels, (0.62, 0.51, 0.47, 0.35))},
        "hyperattentiondti": {l: cell(v, 0.04) for l, v in
                              zip(levels, (0.58, 0.49, 0.44, 0.33))},
        "uniform_control": {l: cell(v, 0.01) for l, v in
                            zip(levels, (0.05, 0.05, 0.05, 0.05))},
    }
    accuracy = {
        "coldsite_dti": dict(zip(levels, (0.89, 0.81, 0.78, 0.69))),
        "hyperattentiondti": dict(zip(levels, (0.87, 0.80, 0.76, 0.67))),
    }

    plot_degradation_curve(grid, accuracy,
                           title="DRAFT — placeholder numbers, not results",
                           save_path="results/headline_figure_DRAFT.png")
    print("Placeholder numbers only. Replace with real audit output.")
