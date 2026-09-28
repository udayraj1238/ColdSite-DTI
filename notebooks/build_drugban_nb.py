"""Build notebooks/kaggle_drugban_davis.ipynb.

The runner is the KIBA notebook's, reused rather than rewritten: it is the code that
survived four accounts and a dozen commits. What changes is the plan (one model, four
levels, three seeds, one account) and the install (DGL, which the other subjects do not
need).
"""
import json, os

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runner_patch import RUNNER as runner_src

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}

def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.splitlines(keepends=True)}

cells = []

cells.append(md('''# DrugBAN on DAVIS — the current-generation subject

The audit's three subjects are from 2020-2022, and the first question a reviewer asks is
whether its verdict binds on what people build now. **DrugBAN** (Bai et al., *Nature
Machine Intelligence* 2023) is the answer: its title claims interpretability, it reports
cross-domain generalisation, and its code is maintained and MIT-licensed.

This notebook trains its **12 DAVIS cells** — four levels, three seeds — on two T4s.
Nothing else: KIBA, the ladders and the faithfulness runs happen elsewhere.

**Measured, 2026-09-18: all twelve cells in 5.5 h on two T4s**, inside one commit. The
constant this notebook shipped with projected 0.6 min/epoch and was wrong by roughly
fourfold — DrugBAN pads every drug to 290 atoms and every protein to 1,200 residues and
then builds a bilinear map of 290 × 1,185 per pair per head, so its epoch cost has little
to do with its row count. The queue listing therefore prints no per-cell hours at all;
the STATUS lines report the measured rate.

**Budget the wall clock, not the training.** That run showed 6.3 h in Kaggle's timer
against 5.5 h of training: the difference is Kaggle rendering a 20,000-line log to HTML
afterwards. It is not stuck when it says `[NbConvertApp] Converting notebook`.

| | |
|---|---|
| model | DrugBAN, their architecture from their own `configs.py`, unmodified |
| recipe | Adam 5e-5, batch 64, up to 100 epochs — **theirs** |
| what we change | early stopping on validation loss (patience 15, floor 10) and `BCEWithLogitsLoss` on their single logit, so this cell is selected and scored like every other cell in the audit; domain adaptation off, their own default for in-domain runs |
| cost | **measured, not projected** — read the first STATUS line; 6 cells, 3 per GPU |

**Why it needs an install step the other notebooks do not.** DrugBAN featurises a drug as
a DGL graph. DGL is not on Kaggle's image and its wheels are pinned to a torch/CUDA
build, so section 4 installs it and then *proves* it works by running the real model
forward once. If that check fails, stop: every cell would fail the same way an hour later.

## What to do

1. **Settings** (right panel): Accelerator **GPU T4 x2**, Internet **On**.
2. Run all. Nothing to edit unless you are resuming — see section 6.
3. **Save Version -> Save & Run All (Commit)**.
'''))

cells.append(md('## 1. Settings — normally nothing to change'))
cells.append(code('''# ============================================================================
# SETTINGS
# ============================================================================

RESTORE_FROM = None      # second commit onward: '/kaggle/input' (searches every input)

# ============================================================================
# Everything below is the plan. Read it; do not edit it.
# ============================================================================

DATASET = 'davis'
TASK = 'binary'
BRANCH = 'main'
MODEL = 'drugban'
LEVELS = ['random', 'cold_drug', 'cold_target', 'cold_pair']
SEEDS = [1, 2, 3]

# Their SOLVER block (baselines/DrugBAN/configs.py): batch 64, lr 5e-5, 100 epochs.
# Passed explicitly so this notebook's log records what was trained rather than
# whatever a future edit to their config would silently change.
BATCH_SIZE = 64
LR = 5e-5
EPOCHS = 100

# One progress line every 100 batches, not every 20. Kaggle renders the entire log to
# HTML when a commit ends, and on 2026-09-18 a 20,000-line log took longer to render than
# the twelve cells took to train. Four lines per epoch is enough to see a cell is alive;
# the STATUS block carries the rate.
LOG_EVERY = 100

# Full precision. DrugBAN has not been validated under --amp here, and the one model in
# this audit that was left unvalidated under float16 (MolTrans on KIBA) produced NaN from
# batch ~4,040 of epoch 1. Twelve cells at ~0.5 h do not need the speed-up badly enough
# to risk a silent divergence.
AMP = False

# 12 cells over two GPUs, most expensive first. cold_pair is the cheapest (15,190 train
# rows against 21,658), so it goes last on both queues; the two queues are equal in rows.
CELLS = [(level, seed) for level in LEVELS for seed in SEEDS]
QUEUE0 = [c for i, c in enumerate(CELLS) if i % 2 == 0]
QUEUE1 = [c for i, c in enumerate(CELLS) if i % 2 == 1]
TOTAL_CELLS = len(CELLS)

print(f'{TOTAL_CELLS} cells: {MODEL} x {len(LEVELS)} levels x {len(SEEDS)} seeds')
print('GPU 0:', QUEUE0)
print('GPU 1:', QUEUE1)
print(f'precision: {"mixed" if AMP else "full (fp32)"} | batch {BATCH_SIZE} | lr {LR}')
'''))

cells.append(md('## 2. Check the GPU(s)'))
cells.append(code('''import time
START = time.time()          # the 11-hour self-stop is measured from here

import torch

assert torch.cuda.is_available(), 'No CUDA. Settings -> Accelerator -> GPU.'
N_GPU = torch.cuda.device_count()
for i in range(N_GPU):
    p = torch.cuda.get_device_properties(i)
    print(f'GPU {i}: {p.name}, {p.total_memory/1e9:.1f} GB')
print('torch  :', torch.__version__)
print('cuda   :', torch.version.cuda)

assert N_GPU >= 2, (
    f'Kaggle gave {N_GPU} GPU(s). Settings -> Accelerator -> GPU T4 x2. On one GPU this '
    'notebook still finishes, but in twice the wall time -- re-run this cell after '
    'switching, or accept it knowingly.')
'''))

cells.append(md('''## 3. Clone the repo

DrugBAN itself is vendored in the repo (`baselines/DrugBAN`, MIT, unmodified), so there
is nothing to fetch from GitHub for the model.'''))
cells.append(code('''import os

REPO = 'https://github.com/Mahim56207/ColdSite-DTI_New.git'
WORK = '/kaggle/working'
SRC  = f'{WORK}/ColdSite-DTI_New'

if not os.path.exists(SRC):
    !git clone --branch {BRANCH} {REPO} {SRC}
os.chdir(SRC)
!git checkout {BRANCH}
!git pull origin {BRANCH}
!pip install -q tabulate

for _needed in ('src/model/train_drugban.py', 'src/evaluation/drugban_adapter.py',
                'baselines/DrugBAN/models.py', 'src/model/resume.py'):
    assert os.path.exists(_needed), (
        f'{_needed} is missing from branch {BRANCH!r}. Push the DrugBAN commit to '
        'origin/main before running this notebook, then re-run this cell.')

RESULTS = f'{WORK}/results'
os.makedirs(RESULTS, exist_ok=True)
print()
!git log --oneline -1
print('results ->', RESULTS)
'''))

cells.append(md('''## 4. Install DGL — by pinning torch to a version it builds for

**This is the fragile step, and it fails fast on purpose.** DGL ships a compiled library
per torch version (`libgraphbolt_pytorch_<version>.so`). Kaggle's image is currently torch
**2.10**, for which DGL publishes nothing: the wheel installs and then cannot import
(seen 2026-09-18, three index URLs, same `FileNotFoundError`). DGL's newest build (2.5.0)
targets **torch 2.6 / cu124**, and that is what this section installs — torch first, then
the matching DGL.

Two consequences worth knowing:

* This downgrades torch **inside the session only**, and takes a few minutes. Training
  runs in subprocesses, which pick up the new torch; the kernel keeps the one it imported
  in section 2, which is why the check below runs in a subprocess too.
* A T4 is compute 7.5 and cu124 supports it, so the GPUs stay usable. The check proves
  that rather than assuming it.

If DGL ever publishes for Kaggle's torch, delete the pin and this section gets shorter.'''))
cells.append(code('''# Torch that DGL builds for, then DGL. ~3 minutes, mostly the torch download.
import subprocess, sys

TORCH_PIN = 'torch==2.6.0'
TORCHVISION_PIN = 'torchvision==0.21.0'
DGL_INDEX = 'https://data.dgl.ai/wheels/torch-2.6/cu124/repo.html'

!pip install -q {TORCH_PIN} {TORCHVISION_PIN} --index-url https://download.pytorch.org/whl/cu124
!pip install -q dgl -f {DGL_INDEX}
!pip install -q dgllife rdkit yacs prettytable

# dgl 2.x imports torchdata.datapipes, which torchdata >= 0.10 removed. Only pinned if
# the import actually asks for it -- an unnecessary downgrade is its own risk.
probe = subprocess.run([sys.executable, '-c', 'import dgl'],
                       capture_output=True, text=True,
                       env={**os.environ, 'DGLBACKEND': 'pytorch'})
if 'torchdata.datapipes' in probe.stderr:
    print('-> dgl wants torchdata.datapipes; pinning torchdata==0.7.1')
    !pip install -q "torchdata==0.7.1"
print(probe.stderr.strip().splitlines()[-1] if probe.returncode else 'dgl imports cleanly')
'''))
cells.append(code('''# The proof, in a subprocess so it uses the torch we just installed:
# versions, both GPUs, and one real pair through the real model.
import os, subprocess, sys, textwrap

CHECK = textwrap.dedent(\'\'\'
    import sys
    sys.path.insert(0, 'baselines/DrugBAN')
    import dgl, torch
    from src.evaluation.drugban_adapter import DrugBANAdapter, _config
    from models import DrugBAN

    print('torch', torch.__version__, '| dgl', dgl.__version__)
    assert torch.cuda.is_available(), 'torch sees no GPU after the pin'
    print('GPUs visible:', torch.cuda.device_count(),
          '|', torch.cuda.get_device_name(0))
    # a real kernel launch: a wheel built for the wrong CUDA fails here, not later
    print('cuda matmul ok:',
          float((torch.ones(8, 8, device='cuda') @ torch.ones(8, 8, device='cuda'))[0, 0]))

    seq = ('MKKFFDSRREQGGSGLGSGSSGGGGSTSGLGSGYIGRVFGIGRQQVTVDEVLAEGGFAIVFLVRTSNGMKCALKRMF'
           'VNNEHDLQVCKREIQIMRDLSGHKNIVGYIDSSINNVSSGDVWEVLILM')
    graph, protein = DrugBANAdapter.encode('CC(=O)Oc1ccccc1C(=O)O', seq)
    model = DrugBAN(**_config()).eval()
    with torch.no_grad():
        _d, _p, score, att = model(dgl.batch([graph]), protein.unsqueeze(0), mode='eval')
    print('score', float(score.reshape(-1)[0]))
    print('attention map', tuple(att.shape), '= (batch, heads, drug atoms, positions)')
    assert att.shape[1] == 2 and att.shape[2] == 290, att.shape
    print('OK')
\'\'\')

done = subprocess.run([sys.executable, '-c', CHECK], text=True, capture_output=True,
                      env={**os.environ, 'DGLBACKEND': 'pytorch'})
print(done.stdout)
assert done.returncode == 0 and done.stdout.strip().endswith('OK'), (
    'DGL/DrugBAN did not come up:\\n' + done.stderr[-2000:] +
    '\\n\\nDo not train: every cell would fail the same way. If the error names a missing '
    'libgraphbolt_pytorch_<version>.so, Kaggle moved torch again -- check '
    'https://data.dgl.ai/wheels/ for the newest torch DGL builds for and change '
    "TORCH_PIN and DGL_INDEX above to match.")
print('DGL, the featuriser and DrugBAN all agree, on the GPU. Safe to train.')
'''))

cells.append(md('''## 5. Fetch the DAVIS source files

`build_splits` reads DeepDTA's published DAVIS files, which are not in the repo
(`.gitignore` excludes `src/data/baselines/*`). Both datasets are fetched because
`load_data` reads both; this notebook then builds DAVIS only.

Omitting this step is what made the first run fail: `build_splits` died with
`No such file or directory: ligands_can.txt`, and the next cell reported the confusing
consequence (`data/splits/davis/random/train.csv` missing) rather than the cause.'''))
cells.append(code('''BASE = 'https://raw.githubusercontent.com/hkmztrk/DeepDTA/master/data'
for ds in ('davis', 'kiba'):
    os.makedirs(f'src/data/baselines/deepdta/data/{ds}', exist_ok=True)
    for fname in ('ligands_can.txt', 'proteins.txt', 'Y'):
        target = f'src/data/baselines/deepdta/data/{ds}/{fname}'
        if not os.path.exists(target):
            !curl -sL {BASE}/{ds}/{fname} -o {target}

missing = [f'src/data/baselines/deepdta/data/{ds}/{f}'
           for ds in ('davis', 'kiba')
           for f in ('ligands_can.txt', 'proteins.txt', 'Y')
           if not os.path.exists(f'src/data/baselines/deepdta/data/{ds}/{f}')
           or os.path.getsize(f'src/data/baselines/deepdta/data/{ds}/{f}') == 0]
assert not missing, (
    f'these source files did not download: {missing}. Internet must be ON '
    '(Settings -> Internet), or GitHub raw is unreachable from this session.')
print('DeepDTA source files present')

!python -m src.data.load_data
'''))

cells.append(md('''## 6. Build the splits — and verify they match

The sizes are checked against the ones the other three models were trained on. A split
that does not match is not the same test, and the whole point of adding this model is
that it faces the identical one.'''))
cells.append(code('''import subprocess, sys

import pandas as pd
from src.model.dataset import BINARY_THRESHOLD

# Run it as a subprocess and CHECK: `!python ...` reports a failure only in the output,
# so the first run sailed past a crashed build_splits and failed later on a missing CSV.
built = subprocess.run([sys.executable, '-m', 'src.data.build_splits'],
                       capture_output=True, text=True)
for line in built.stdout.splitlines():
    if 'davis' in line or 'leakage' in line:
        print(line)
assert built.returncode == 0, (
    'build_splits failed:\\n' + (built.stderr or built.stdout)[-1500:] +
    '\\n\\nIf it says a source file is missing, section 5 did not download it.')

# Recorded from data/splits/davis on the Mac, 2026-09-18.
EXPECTED = {
    'random':      (21039, 3006, 6011),
    'cold_drug':   (21658, 2652, 5746),
    'cold_target': (21080, 2992, 5984),
    'cold_pair':   (15190,  264, 1144),
}
assert BINARY_THRESHOLD['davis'] == 7.0, BINARY_THRESHOLD
print(f"\\nbinary threshold: DAVIS pKd >= {BINARY_THRESHOLD['davis']}")

ok = True
for level, expected in EXPECTED.items():
    sizes = tuple(len(pd.read_csv(f'data/splits/{DATASET}/{level}/{part}.csv'))
                  for part in ('train', 'valid', 'test'))
    mark = 'OK' if sizes == expected else f'MISMATCH, expected {expected}'
    ok &= sizes == expected
    print(f'{level:12s} {str(sizes):28s} {mark}')
assert ok, 'splits differ from the recorded ones -- stop and find out why'
print('\\nsplits match the record.')
'''))

cells.append(md('''## 7. Restore from a previous commit

Only needed if a commit was cut short. Set `RESTORE_FROM = '/kaggle/input'` in section 1,
having attached this account's own output as a private dataset. Finished cells are then
skipped and an interrupted one continues from its last finished epoch.'''))
cells.append(code('''import shutil, glob

if RESTORE_FROM:
    assert os.path.isdir(RESTORE_FROM), f'not a directory: {RESTORE_FROM}'
    copied = skipped = 0
    for src in glob.glob(f'{RESTORE_FROM}/**/*', recursive=True):
        name = os.path.basename(src)
        if not name.endswith(('.pt', '_results.json', '_history.json')):
            continue
        dst = os.path.join(RESULTS, name)
        if os.path.exists(dst):
            skipped += 1
            continue
        shutil.copy2(src, dst)
        copied += 1
    print(f'restored {copied} file(s), left {skipped} already present')
else:
    print('RESTORE_FROM is None -- starting from an empty results folder.')
'''))

cells.append(md('''## 8. The runner

Two queues, one per GPU, each cell a separate process so a crash cannot take the other
GPU with it. A STATUS line every 10 minutes carries the measured minutes-per-epoch and
when each queue expects to finish. The whole thing self-stops at 11 hours, an hour inside
Kaggle's limit, so the commit has time to save its output.'''))
cells.append(code(runner_src))

cells.append(md('## 9. Launch'))
cells.append(code('''if hours_left() < 0.5:
    raise SystemExit('less than 30 minutes before the self-stop -- not worth starting')

cut = run_parallel(QUEUES, f'drugban_{DATASET}')
print()
print(f'{cells_done()}/{TOTAL_CELLS} cells complete')
if cut:
    print('CUT SHORT by the 11-hour stop. Download the output, make it a dataset, attach '
          "it, set RESTORE_FROM = '/kaggle/input' in section 1, and run again: finished "
          'cells are skipped and an interrupted cell continues from its last epoch.')
else:
    print('every cell finished')
'''))

cells.append(md('''## 10. What landed

One row per cell. `AUROC` should be believable for DAVIS: ~0.85-0.95 at random, lower at
the cold levels. A value at 0.5 means the cell never learned; above 0.98 means look for
leakage before celebrating.'''))
cells.append(code('''import glob, json

rows = []
for path in sorted(glob.glob(f'{RESULTS}/*_{MODEL}_results.json')):
    r = json.load(open(path))
    rows.append((r['split'], r['seed'], r['test_metrics']['auroc'],
                 r['test_metrics']['auprc'], r['best_epoch'],
                 r.get('resumed_after_epoch') or '-'))

print(f'{"level":12s} {"seed":>4s} {"AUROC":>7s} {"AUPRC":>7s} {"best":>5s} {"resumed":>8s}')
for level, seed, auroc, auprc, best, resumed in sorted(rows):
    print(f'{level:12s} {seed:>4} {auroc:>7.4f} {auprc:>7.4f} {best:>5} {str(resumed):>8s}')
print(f'\\n{len(rows)}/{TOTAL_CELLS} cells')

seen = {}
for level, seed, auroc, *_ in rows:
    if (level, round(auroc, 6)) in seen:
        print(f'!! {level} seed {seed} has the same AUROC as seed {seen[(level, round(auroc, 6))]}'
              ' -- is --seed reaching training? Do not merge these.')
    seen[(level, round(auroc, 6))] = seed
'''))

cells.append(md('''## 11. Take the results with you

`drugban_davis_results.zip` is what the analysis needs. Download it from the Output panel.
Its checkpoints are small (a few MB each), unlike MolTrans's.'''))
cells.append(code('''import subprocess

RES_ZIP = f'{WORK}/drugban_{DATASET}_results.zip'
finished = [p for p in glob.glob(f'{RESULTS}/*')
            if not os.path.basename(p).endswith('_resume.pt')]
resumes = glob.glob(f'{RESULTS}/*_resume.pt')

if os.path.exists(RES_ZIP):
    os.remove(RES_ZIP)
subprocess.run(['zip', '-q', '-j', RES_ZIP, *finished], check=True)
print(f'{os.path.basename(RES_ZIP)}: {len(finished)} file(s), '
      f'{os.path.getsize(RES_ZIP)/1e6:.0f} MB')

if resumes:
    RESUME_ZIP = f'{WORK}/drugban_{DATASET}_resume.zip'
    if os.path.exists(RESUME_ZIP):
        os.remove(RESUME_ZIP)
    subprocess.run(['zip', '-q', '-j', RESUME_ZIP, *resumes], check=True)
    print(f'{os.path.basename(RESUME_ZIP)}: {len(resumes)} unfinished cell(s) -- upload '
          'this one too if you need another commit.')
'''))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python"},
                   "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}
os.makedirs('notebooks', exist_ok=True)
json.dump(nb, open('notebooks/kaggle_drugban_davis.ipynb', 'w'), indent=1)
print('wrote notebooks/kaggle_drugban_davis.ipynb with', len(cells), 'cells')
