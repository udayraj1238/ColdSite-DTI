"""Run tags and checkpoint filenames — the single source of truth.

Track B writes checkpoints (`src/model/train.py`); Track C reads them back by
name (`src/evaluation/run_ladder.py`). Until now the pattern was spelled out
twice, once at each end, as an f-string. Two copies of a filename convention is
one copy too many: the reader silently `[skip]`s a level whose name it cannot
guess, and a skipped level looks exactly like a level that has not been trained
yet.

The bug this module exists to fix
---------------------------------
The old tag was `{dataset}_{split}_{task}`. Seed was absent, so the three runs
the master plan requires per cell all wrote to the same path and each
overwrote the last. Nothing crashes; you end up with one checkpoint, three
identical-looking `*_results.json` rewrites, and an "n_seeds = 3" claim built
on one run.

    coldsite_dti_davis_cold_target_regression.pt      <- seeds 1, 2 and 3
    coldsite_dti_davis_cold_target_regression_seed1.pt   <- one run each
    coldsite_dti_davis_cold_target_regression_seed2.pt
    coldsite_dti_davis_cold_target_regression_seed3.pt

Why this convention is safe to adopt
------------------------------------
A sweep of the repository found exactly one place that ever constructed a
checkpoint name for reading — `run_ladder.py`, which duplicated the trainer's
f-string as a literal. Both ends now import from here, so the "agreement" is
enforced by the import graph rather than by two people remembering the same
string. Nothing else constrains it:

  * `ColdSiteDTIAdapter` and the baseline adapters take an explicit
    `checkpoint_path` and are convention-agnostic.
  * `run_audit.build_grid` takes a `collect_fn`, never a filename.
  * `aggregate.py` and `plots.py` operate on dicts; `plots.py` only names its
    own output PNGs.

So Track C has no incompatible expectation to break. The seed also lands in the
*reader's* output filenames (`ladder_davis_seed1.json`) for the same reason it
lands here: three ladder runs writing one file is the same overwrite, one stage
later.

`seed` is the TRAINING seed
---------------------------
Weight init and batch order. The repository is consistent that the grid is 2
datasets x 4 splits x 3 seeds = 24 runs: the Track B guide's own loop varies
only `--seed` while pointing at a fixed `data/splits/$dataset/$split`;
`build_all_splits()` takes no seed argument at all and writes one split per
cell; `run_audit.build_grid` has a single seed axis; and both the guide and
STATUS.md state the count as 24 rather than 72.

`split_seed=` stays available but unused. Track A's Part 2 definition of done
says "splits built for both datasets, three seeds each", which is the one line
in the repository that could imply a second seed axis. If it turns out to mean
three split variants rather than loose wording, the tag needs both seeds and
`data/splits/` needs a seed level — a parameter change here, not a rewrite.

Round-trip guarantee
--------------------
`parse_run_tag(run_tag(...))` returns the components it was given. That is what
makes it safe for Track C to discover seeds by listing a directory instead of
being told which ones exist. It holds because `dataset` and `task` are
forbidden from containing `_`; `split` may (`cold_target`), so it is recovered
as the remainder.
"""
import glob
import os
import re

CHECKPOINT_PREFIX = "coldsite_dti"
CHECKPOINT_SUFFIX = ".pt"

# Change nothing here without telling both other tracks -- a reader built
# against the old spelling does not fail loudly, it reports "no checkpoint" and
# the level vanishes from the ladder without comment.
NAMING_IS_RATIFIED = True

# One cell -- same dataset, split, task and seed -- now holds one checkpoint per
# audited model, so the model has to be in the name. The suffixes are exactly
# what the baseline trainers were already spelling out by hand:
#
#     ckpt.replace(".pt", "_deepdta.pt")            train_deepdta.py
#     ckpt.replace(".pt", "_hyperattentiondti.pt")  train_hyperattentiondti.py
#
# which is the duplicated-convention bug this module exists to prevent, one
# level down: the writer applied the suffix and the reader had to know to
# apply the same one. ColdSite-DTI keeps the empty suffix, so every path that
# already exists on disk still resolves and no checkpoint is orphaned.
MODEL_SUFFIX = {
    "coldsite_dti": "",
    "deepdta": "_deepdta",
    "hyperattentiondti": "_hyperattentiondti",
    "moltrans": "_moltrans",
    "drugban": "_drugban",
    # Integrated gradients is a second explanation of the SAME trained weights
    # (src/evaluation/integrated_gradients.py), so each variant reads its base model's
    # checkpoint. Its outputs are still named apart, because `output_tag` prefixes any
    # model that is not the default -- ladder_coldsite_dti_ig_davis_seed1.json.
    "coldsite_dti_ig": "",
    "hyperattentiondti_ig": "_hyperattentiondti",
    "moltrans_ig": "_moltrans",
    # Alternative readouts of the same attention (src/evaluation/readout_variants.py):
    # one documented choice changed, same weights, same checkpoint.
    "coldsite_dti_selfattn": "",
    "hyperattentiondti_maxchannel": "_hyperattentiondti",
    "hyperattentiondti_receptive": "_hyperattentiondti",
    "moltrans_maxhead": "_moltrans",
    "moltrans_firstlayer": "_moltrans",
}

# Every explanation variant and the trained model it reads. Explicit rather than a suffix
# rule: a variant whose base is guessed wrongly would silently load another model's
# weights, and `_ig` is not the only suffix any more.
VARIANT_BASE = {
    "coldsite_dti_ig": "coldsite_dti",
    "hyperattentiondti_ig": "hyperattentiondti",
    "moltrans_ig": "moltrans",
    "coldsite_dti_selfattn": "coldsite_dti",
    "hyperattentiondti_maxchannel": "hyperattentiondti",
    "hyperattentiondti_receptive": "hyperattentiondti",
    "moltrans_maxhead": "moltrans",
    "moltrans_firstlayer": "moltrans",
}
IG_SUFFIX = "_ig"


def base_model_name(model: str) -> str:
    """The trained model behind an explanation variant: 'moltrans_maxhead' -> 'moltrans'."""
    return VARIANT_BASE.get(model, model)
DEFAULT_MODEL = "coldsite_dti"


def model_suffix(model: str) -> str:
    """The filename suffix for one audited model. Raises rather than guessing.

    An unknown model returning "" would silently collide with ColdSite-DTI's
    checkpoint for the same cell, which is the overwrite this module exists to
    make impossible.
    """
    if model not in MODEL_SUFFIX:
        raise ValueError(
            f"unknown model {model!r}. Known: {sorted(MODEL_SUFFIX)}.\n"
            f"Add it to MODEL_SUFFIX in src/model/checkpoint_naming.py — a "
            f"model without a registered suffix would overwrite "
            f"{DEFAULT_MODEL}'s checkpoint for the same cell.")
    return MODEL_SUFFIX[model]

_TAG_RE = re.compile(
    r"^(?P<dataset>[^_]+)_"
    r"(?P<split>.+?)_"
    r"(?P<task>[^_]+)"
    r"(?:_split(?P<split_seed>\d+))?"
    r"_seed(?P<seed>\d+)$"
)


def run_tag(dataset: str, split: str, task: str, seed: int,
            split_seed: int | None = None) -> str:
    """The identity of one training run, used for every file it produces.

    `seed` is the training seed and is mandatory: making it optional is how the
    overwrite bug comes back, because the one call site that forgets it silently
    collides with every other seed.

    `split_seed` is for the case where Track A ends up shipping one split per
    seed. Leave it None until that decision is made (see the module docstring).
    """
    for name, value in (("dataset", dataset), ("task", task)):
        if not value:
            raise ValueError(f"{name} must not be empty")
        if "_" in value:
            raise ValueError(
                f"{name}={value!r} must not contain '_': the tag is parsed back "
                f"by splitting on it, and an underscore here makes the run tag "
                f"ambiguous (see parse_run_tag)")
    if not split:
        raise ValueError("split must not be empty")
    if seed is None:
        raise ValueError(
            "seed is required. Runs that omit it overwrite each other and the "
            "loss is invisible -- see src/model/checkpoint_naming.py")

    middle = f"_split{int(split_seed)}" if split_seed is not None else ""
    return f"{dataset}_{split}_{task}{middle}_seed{int(seed)}"


def parse_run_tag(tag: str) -> dict:
    """Inverse of `run_tag`. Raises on a tag that carries no seed.

    A tag without a seed is an artefact from before this module existed. It is
    an error rather than a default, because guessing "probably seed 1" is how a
    single run gets aggregated as three.
    """
    match = _TAG_RE.match(tag)
    if match is None:
        raise ValueError(
            f"{tag!r} is not a run tag. Expected "
            f"'{{dataset}}_{{split}}_{{task}}_seed{{n}}' — a tag with no seed "
            f"predates the seeded convention and cannot be attributed to a run.")
    parts = match.groupdict()
    return {
        "dataset": parts["dataset"],
        "split": parts["split"],
        "task": parts["task"],
        "seed": int(parts["seed"]),
        "split_seed": (int(parts["split_seed"])
                       if parts["split_seed"] is not None else None),
    }


def checkpoint_name(dataset: str, split: str, task: str, seed: int,
                    split_seed: int | None = None,
                    model: str = DEFAULT_MODEL) -> str:
    tag = run_tag(dataset, split, task, seed, split_seed)
    return f"{CHECKPOINT_PREFIX}_{tag}{model_suffix(model)}{CHECKPOINT_SUFFIX}"


def checkpoint_path(checkpoint_dir: str, dataset: str, split: str, task: str,
                    seed: int, split_seed: int | None = None,
                    model: str = DEFAULT_MODEL) -> str:
    return os.path.join(
        checkpoint_dir,
        checkpoint_name(dataset, split, task, seed, split_seed, model))


def history_path(checkpoint_path_: str) -> str:
    """Per-epoch history sits beside the checkpoint it describes."""
    if not checkpoint_path_.endswith(CHECKPOINT_SUFFIX):
        raise ValueError(f"expected a {CHECKPOINT_SUFFIX} path, got {checkpoint_path_!r}")
    return checkpoint_path_[: -len(CHECKPOINT_SUFFIX)] + "_history.json"


def results_path(results_dir: str, tag: str,
                 model: str = DEFAULT_MODEL) -> str:
    """Where the test-set metrics for one run land.

    This is the file Track C consumes as the accuracy axis of the headline
    figure, so its name is part of the hand-off, not an internal detail.
    """
    return os.path.join(results_dir, f"{tag}{model_suffix(model)}_results.json")


def discover_checkpoints(checkpoint_dir: str, dataset: str | None = None,
                         split: str | None = None, task: str | None = None,
                         model: str = DEFAULT_MODEL) -> list[dict]:
    """Every seeded checkpoint on disk for one model, parsed, sorted by seed.

    Track C needs "which seeds exist for this cell" and should not have to be
    told. Unparseable files -- including unseeded ones from before this
    convention -- are left out rather than guessed at; `aggregate_seeds` will
    then flag the cell as under-powered, which is the correct outcome.

    `model` filters by suffix. It matters in both directions: without it a
    DeepDTA checkpoint could be read as ColdSite-DTI's, and baseline
    checkpoints were previously invisible here -- `..._seed1_deepdta` does not
    match a tag regex anchored on `_seed{n}$`, so they were skipped as
    unparseable and every baseline cell looked untrained.
    """
    suffix = model_suffix(model)
    pattern = os.path.join(
        checkpoint_dir, f"{CHECKPOINT_PREFIX}_*{suffix}{CHECKPOINT_SUFFIX}")
    found = []
    for path in sorted(glob.glob(pattern)):
        stem = os.path.basename(path)[len(CHECKPOINT_PREFIX) + 1: -len(CHECKPOINT_SUFFIX)]
        if suffix:
            stem = stem[: -len(suffix)]
        elif any(stem.endswith(other) for other in MODEL_SUFFIX.values() if other):
            # an unsuffixed glob also matches every other model's files
            continue
        try:
            parsed = parse_run_tag(stem)
        except ValueError:
            continue
        if dataset is not None and parsed["dataset"] != dataset:
            continue
        if split is not None and parsed["split"] != split:
            continue
        if task is not None and parsed["task"] != task:
            continue
        parsed["path"] = path
        parsed["tag"] = stem
        parsed["model"] = model
        found.append(parsed)
    return sorted(found, key=lambda entry: entry["seed"])
