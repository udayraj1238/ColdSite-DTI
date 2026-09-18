"""Build notebooks/kaggle_coldsite_kiba.ipynb — ColdSite-DTI's six KIBA cells."""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runner_patch_coldsite import RUNNER as runner_src

def md(t): return {"cell_type":"markdown","metadata":{},"source":t.splitlines(keepends=True)}
def code(t): return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],
                     "source":t.splitlines(keepends=True)}

cells = []
cells.append(md('''# ColdSite-DTI on KIBA — closing the asymmetry

KIBA was trained for the two **published** models and the accuracy anchor; our own model
was cut to fit the compute. The paper therefore says, in Limitations, that our model is
audited on one dataset where the models whose claims the paper is about are audited on
two. A reviewer is entitled to read that as convenient. This notebook removes it: the
**six cells** ColdSite-DTI is missing — random and cold-drug, three seeds — on the
identical splits every other model faced.

| | |
|---|---|
| model | ColdSite-DTI (ours), trained exactly as its DAVIS cells were |
| cells | 2 levels × 3 seeds = 6 |
| precision | **full (fp32)**, see below |
| cost | 3.6 h per cell at the 25-epoch floor, 5.2 h at the 36-epoch median (measured on a T4, `results/speed_test_kiba_t4.md`) |
| plan | 3 cells per GPU, so **11–16 h per GPU: expect two commits** |

**Why full precision.** Mixed precision would buy 15% on this model (0.387 → 0.336
s/batch) and cost the thing that matters more: its DAVIS cells are fp32, and a model
trained in one precision on one dataset and another precision on the other is not the
clean replication the paper claims. MolTrans's KIBA cells were kept fp32 for the same
reason — there because float16 made it diverge, here because consistency is worth more
than 15%.

**Two commits are expected, not a failure.** The runner self-stops at 11 hours so the
commit can save its output. Whatever is unfinished continues from its last finished epoch
on the next commit: download the output, make it a private dataset, attach it, set
`RESTORE_FROM = '/kaggle/input'` in section 1, and run again.

## What to do

1. **Settings** (right panel): Accelerator **GPU T4 x2**, Internet **On**.
2. Run all. Nothing to edit on the first commit.
3. **Save Version → Save & Run All (Commit)**.
'''))

cells.append(md('## 1. Settings — only RESTORE_FROM, and only on a later commit'))
cells.append(code('''# ============================================================================
# SETTINGS
# ============================================================================

RESTORE_FROM = None      # second commit onward: '/kaggle/input' (searches every input)

# ============================================================================
# Everything below is the plan. Read it; do not edit it.
# ============================================================================

DATASET = 'kiba'
TASK = 'binary'
BRANCH = 'main'
MODEL = 'coldsite_dti'
LEVELS = ['random', 'cold_drug']
SEEDS = [1, 2, 3]
EPOCHS = 100
AMP = False              # see the header: its DAVIS cells are fp32 and must stay comparable

# Measured peak memory on a 1,000-residue protein: 8.7 GB at batch 64, 2.3 GB at 16
# (STATUS.md, from the DAVIS grid). A T4 has 15.6 GB, so 64 fits with room.
BATCH_SIZE = 64

CELLS = [(level, seed) for level in LEVELS for seed in SEEDS]
QUEUE0 = [c for i, c in enumerate(CELLS) if i % 2 == 0]
QUEUE1 = [c for i, c in enumerate(CELLS) if i % 2 == 1]
TOTAL_CELLS = len(CELLS)

print(f'{TOTAL_CELLS} cells: ColdSite-DTI x {len(LEVELS)} levels x {len(SEEDS)} seeds')
print('GPU 0:', QUEUE0)
print('GPU 1:', QUEUE1)
print(f'precision: {"mixed" if AMP else "full (fp32)"} | batch {BATCH_SIZE}')
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

if torch.cuda.get_device_properties(0).total_memory / 1e9 < 14:
    BATCH_SIZE = 16
    print(f'small GPU -- batch size lowered to {BATCH_SIZE} (8.7 GB at 64 would not fit)')
'''))

cells.append(md('''## 3. Clone the repo

No vendored model to install: ColdSite-DTI is ours and lives in `src/`.'''))
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

for _needed in ('src/model/run_grid.py', 'src/model/train.py',
                'src/model/resume.py', 'src/model/precision.py'):
    assert os.path.exists(_needed), (
        f'{_needed} is missing from branch {BRANCH!r}. Without resume.py a cell longer '
        'than one 11-hour commit could never finish.')

RESULTS = f'{WORK}/results'
os.makedirs(RESULTS, exist_ok=True)
print()
!git log --oneline -1
print('results ->', RESULTS)
'''))

cells.append(md('''## 4. Fetch the KIBA source files

`build_splits` reads DeepDTA's published files, which `.gitignore` keeps out of the repo.
Both datasets are fetched because `load_data` reads both.'''))
cells.append(code('''BASE = 'https://raw.githubusercontent.com/hkmztrk/DeepDTA/master/data'
for ds in ('davis', 'kiba'):
    os.makedirs(f'src/data/baselines/deepdta/data/{ds}', exist_ok=True)
    for fname in ('ligands_can.txt', 'proteins.txt', 'Y'):
        target = f'src/data/baselines/deepdta/data/{ds}/{fname}'
        if not os.path.exists(target):
            !curl -sL {BASE}/{ds}/{fname} -o {target}

missing = [f'{ds}/{f}' for ds in ('davis', 'kiba')
           for f in ('ligands_can.txt', 'proteins.txt', 'Y')
           if not os.path.exists(f'src/data/baselines/deepdta/data/{ds}/{f}')
           or os.path.getsize(f'src/data/baselines/deepdta/data/{ds}/{f}') == 0]
assert not missing, (f'these source files did not download: {missing}. '
                     'Internet must be ON (Settings -> Internet).')
print('DeepDTA source files present')

!python -m src.data.load_data
'''))

cells.append(md('''## 5. Build the splits — and verify they match

The point of this notebook is that our model faces the identical test the published ones
faced. That is only true if these numbers match, so a mismatch stops the run.'''))
cells.append(code('''import subprocess, sys

import pandas as pd
from src.model.dataset import BINARY_THRESHOLD

built = subprocess.run([sys.executable, '-m', 'src.data.build_splits'],
                       capture_output=True, text=True)
for line in built.stdout.splitlines():
    if 'kiba' in line or 'leakage' in line:
        print(line)
assert built.returncode == 0, (
    'build_splits failed:\\n' + (built.stderr or built.stdout)[-1500:] +
    '\\n\\nIf it names a missing source file, section 4 did not download it.')

# The same figures the four-account KIBA notebook checked, from data/splits/kiba.
EXPECTED = {
    'random':      (82778, 11825, 23651),
    'cold_drug':   (83807, 12073, 22374),
    'cold_target': (85452, 10701, 22101),
    'cold_pair':   (58041,  1334,  4375),
}
assert BINARY_THRESHOLD['kiba'] == 12.1, BINARY_THRESHOLD
print(f"\\nbinary threshold: KIBA score >= {BINARY_THRESHOLD['kiba']}")

ok = True
for level, expected in EXPECTED.items():
    sizes = tuple(len(pd.read_csv(f'data/splits/{DATASET}/{level}/{part}.csv'))
                  for part in ('train', 'valid', 'test'))
    mark = 'OK' if sizes == expected else f'MISMATCH, expected {expected}'
    ok &= sizes == expected
    print(f'{level:12s} {str(sizes):28s} {mark}')
assert ok, ('splits differ from the ones the other models trained on -- stop. Whatever '
            'this notebook produced would not be comparable with Results section 8.')
print('\\nsplits match the record: this model faces the same test.')
'''))

cells.append(md('''## 6. Restore from a previous commit

Two commits are expected. On the second, attach this account's own output as a private
dataset and set `RESTORE_FROM = '/kaggle/input'` in section 1. Finished cells are skipped
and an interrupted one continues from its last finished epoch.'''))
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

cells.append(md('''## 7. The runner

Two queues, one per GPU, each cell its own process. A STATUS line every 10 minutes carries
the measured minutes-per-epoch against the projection and says whether the queue fits
inside this commit. Self-stops at 11 hours.'''))
cells.append(code(runner_src))

cells.append(md('## 8. Launch'))
cells.append(code('''if hours_left() < 0.5:
    raise SystemExit('less than 30 minutes before the self-stop -- not worth starting')

cut = run_parallel(QUEUES, f'coldsite_{DATASET}')
print()
print(f'{cells_done()}/{TOTAL_CELLS} cells complete')
if cut:
    print('CUT SHORT by the 11-hour stop -- expected on the first commit. Download the '
          "output, make it a private dataset, attach it, set RESTORE_FROM = "
          "'/kaggle/input' in section 1, and run again.")
else:
    print('every cell finished')
'''))

cells.append(md('''## 9. What landed

Believable for ColdSite-DTI on KIBA: AUROC around 0.85–0.93 at random and lower on unseen
drugs. Its DAVIS binary cells were 0.884–0.906 at random. 0.5 means it never learned;
above 0.98 means look for leakage before celebrating.'''))
cells.append(code('''import glob, json

rows = []
for path in sorted(glob.glob(f'{RESULTS}/kiba_*_binary_seed*_results.json')):
    if any(tag in os.path.basename(path) for tag in
           ('_deepdta', '_hyperattentiondti', '_moltrans', '_drugban')):
        continue                      # another model's cell, restored alongside ours
    r = json.load(open(path))
    rows.append((r['split'], r['seed'], r['test_metrics']['auroc'],
                 r['test_metrics']['auprc'], r.get('best_epoch', '-'),
                 r.get('resumed_after_epoch') or '-'))

print(f'{"level":12s} {"seed":>4s} {"AUROC":>7s} {"AUPRC":>7s} {"best":>5s} {"resumed":>8s}')
for level, seed, auroc, auprc, best, resumed in sorted(rows):
    print(f'{level:12s} {seed:>4} {auroc:>7.4f} {auprc:>7.4f} {str(best):>5s} {str(resumed):>8s}')
print(f'\\n{len(rows)}/{TOTAL_CELLS} cells')

seen = {}
for level, seed, auroc, *_ in rows:
    key = (level, round(auroc, 6))
    if key in seen:
        print(f'!! {level} seed {seed} has the same AUROC as seed {seen[key]} -- is --seed '
              'reaching training? Do not merge these.')
    seen[key] = seed
'''))

cells.append(md('''## 10. Take the results with you

ColdSite-DTI's checkpoints are small (~2.5 MB), so this zip is megabytes, not gigabytes.'''))
cells.append(code('''import subprocess

RES_ZIP = f'{WORK}/coldsite_{DATASET}_results.zip'
finished = [p for p in glob.glob(f'{RESULTS}/*')
            if not os.path.basename(p).endswith('_resume.pt')]
resumes = glob.glob(f'{RESULTS}/*_resume.pt')

if os.path.exists(RES_ZIP):
    os.remove(RES_ZIP)
subprocess.run(['zip', '-q', '-j', RES_ZIP, *finished], check=True)
print(f'{os.path.basename(RES_ZIP)}: {len(finished)} file(s), '
      f'{os.path.getsize(RES_ZIP)/1e6:.0f} MB')

if resumes:
    RESUME_ZIP = f'{WORK}/coldsite_{DATASET}_resume.zip'
    if os.path.exists(RESUME_ZIP):
        os.remove(RESUME_ZIP)
    subprocess.run(['zip', '-q', '-j', RESUME_ZIP, *resumes], check=True)
    print(f'{os.path.basename(RESUME_ZIP)}: {len(resumes)} unfinished cell(s) -- upload '
          'this one too, or the next commit restarts them from epoch 1.')
'''))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open('notebooks/kaggle_coldsite_kiba.ipynb', 'w'), indent=1)
print('wrote notebooks/kaggle_coldsite_kiba.ipynb with', len(cells), 'cells')
