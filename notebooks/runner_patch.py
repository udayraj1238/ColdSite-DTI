"""Adapt the KIBA runner for DrugBAN: keep the machinery, replace the plan."""
import json

KIBA = json.load(open('notebooks/kaggle_kiba_4accounts.ipynb'))
runner_src = ''.join(KIBA['cells'][14]['source'])

# The KIBA runner leans on `os` and `time` being imported by earlier cells. True there
# and true here, but a cell that states its own imports survives being re-run alone.
runner_src = runner_src.replace('import subprocess, threading, glob, json, re',
                                'import glob, json, os, re, subprocess, threading, time')

head, sep, _tail = runner_src.partition('def amp_flag(model):')
assert sep, 'the KIBA runner no longer ends where this builder expects'

OLD_ASSERT = """assert N_GPU >= len(MY_CELLS), (
    f'this plan has {len(MY_CELLS)} queues but Kaggle gave {N_GPU} GPU(s). '
    'Settings -> Accelerator -> GPU T4 x2, then re-run: on one GPU this account would '
    'take twice as long and overrun its quota.')"""
NEW_ASSERT = """QUEUE_OF = {'0': QUEUE0, '1': QUEUE1}
if N_GPU < 2:                      # one GPU: the same work, one queue, twice the wall time
    QUEUE_OF = {'0': QUEUE0 + QUEUE1}
    print('WARNING: one GPU. Running all cells in a single queue.')"""
assert OLD_ASSERT in head
head = head.replace(OLD_ASSERT, NEW_ASSERT)

OLD_NAMES = """MODEL_NAME = {'src.model.train_deepdta': 'DeepDTA', 'src.model.run_grid': 'ColdSite',
              'src.model.train_hyperattentiondti': 'HAT', 'src.model.train_moltrans': 'MolTrans'}
MODEL_KEY = {'src.model.train_deepdta': 'deepdta', 'src.model.run_grid': 'coldsite_dti',
             'src.model.train_hyperattentiondti': 'hyperattentiondti',
             'src.model.train_moltrans': 'moltrans'}"""
NEW_NAMES = """MODEL_NAME = {'src.model.train_drugban': 'DrugBAN'}
MODEL_KEY = {'src.model.train_drugban': 'drugban'}

# A placeholder, and known to be wrong. The 2026-09-18 run measured DrugBAN far slower
# than this 0.6 min/epoch guess -- it pads every drug to 290 atoms and every protein to
# 1,200 residues, so its epoch cost has little to do with its row count. The STATUS line
# prints the measured rate beside this one with their ratio; believe the measurement, and
# replace this constant once a run has reported it.
MIN_PER_EPOCH = {'drugban': 0.6}
EPOCHS_MIN, EPOCHS_TYPICAL = 25, 36        # the early-stopping floor, and the DAVIS median"""
assert OLD_NAMES in head
head = head.replace(OLD_NAMES, NEW_NAMES)

TAIL = '''def drugban_cmd(level, seed):
    """One cell. Their recipe (batch, lr, epochs) comes from section 1; the early
    stopping and the patience are the audit's, identical to every other subject."""
    return ['python', '-u', '-m', 'src.model.train_drugban',
            '--split-dir', f'data/splits/{DATASET}/{level}', '--dataset', DATASET,
            '--split', level, '--seed', str(seed),
            '--batch-size', str(BATCH_SIZE), '--lr', str(LR),
            '--min-epochs', '10', '--epochs', str(EPOCHS), '--patience', '15',
            '--log-every', str(LOG_EVERY),
            '--checkpoint-dir', RESULTS, '--results-dir', RESULTS,
            '--skip-if-done'] + (['--amp'] if AMP else [])


# Largest split first in each queue: it is the cell most likely to meet the 11-hour stop,
# and the one whose resume file is better written early than late.
TRAIN_ROWS = {'random': 21039, 'cold_drug': 21658, 'cold_target': 21080, 'cold_pair': 15190}

# (high, low) hours per cell, the shape `eta` reads: 36 and 25 epochs at the projected
# rate above. Replaced in practice by the measured rate once the first cell reports.
HOURS = {'drugban': (EPOCHS_TYPICAL * MIN_PER_EPOCH['drugban'] / 60,
                     EPOCHS_MIN * MIN_PER_EPOCH['drugban'] / 60)}

# Triples, because `eta` and `cells_done` both unpack (model, level, seed).
ORDERED = {gpu: [(MODEL, level, seed)
                 for level, seed in sorted(cells, key=lambda c: -TRAIN_ROWS[c[0]])]
           for gpu, cells in sorted(QUEUE_OF.items())}
QUEUES = {gpu: [drugban_cmd(level, seed) for _model, level, seed in cells]
          for gpu, cells in sorted(ORDERED.items())}
MY_CELLS = ORDERED

for gpu, cells in sorted(ORDERED.items()):
    print(f'GPU {gpu}: in this order')
    for _model, level, seed in cells:
        print(f'   drugban  {level:12s} seed {seed}   {TRAIN_ROWS[level]:,} train rows')
print('Per-cell hours are deliberately not printed: the projection this notebook shipped '
      'with was wrong by a wide margin on the 2026-09-18 run. The STATUS lines below '
      'report the MEASURED minutes per epoch and what is left; believe those.')
print(f'{hours_left():.1f} h left before the self-stop')
'''

RUNNER = head + TAIL
