"""Adapt the KIBA runner for ColdSite-DTI's six KIBA cells: keep the machinery, replace
the plan. ColdSite-DTI has no single-cell entry point, so each cell is one `run_grid`
invocation with one split and one seed."""
import json

KIBA = json.load(open('notebooks/kaggle_kiba_4accounts.ipynb'))
runner_src = ''.join(KIBA['cells'][14]['source'])
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
    print('WARNING: one GPU. Six cells in a single queue will need three commits, not two.')"""
assert OLD_ASSERT in head
head = head.replace(OLD_ASSERT, NEW_ASSERT)

OLD_NAMES = """MODEL_NAME = {'src.model.train_deepdta': 'DeepDTA', 'src.model.run_grid': 'ColdSite',
              'src.model.train_hyperattentiondti': 'HAT', 'src.model.train_moltrans': 'MolTrans'}
MODEL_KEY = {'src.model.train_deepdta': 'deepdta', 'src.model.run_grid': 'coldsite_dti',
             'src.model.train_hyperattentiondti': 'hyperattentiondti',
             'src.model.train_moltrans': 'moltrans'}"""
NEW_NAMES = """MODEL_NAME = {'src.model.run_grid': 'ColdSite'}
MODEL_KEY = {'src.model.run_grid': 'coldsite_dti'}

# Measured on a T4, full precision (results/speed_test_kiba_t4.md): 7.5 min/epoch, so
# 3.6 h at the 25-epoch floor and 5.2 h at the 36-epoch DAVIS median. The STATUS line
# prints the measured rate beside this projection with their ratio.
MIN_PER_EPOCH = {'coldsite_dti': 8.7}
EPOCHS_MIN, EPOCHS_TYPICAL = 25, 36"""
assert OLD_NAMES in head
head = head.replace(OLD_NAMES, NEW_NAMES)

TAIL = '''def coldsite_cmd(level, seed):
    """One cell. ColdSite-DTI has no single-cell trainer, so it goes through run_grid with
    one split and one seed -- which is also how its DAVIS cells were trained, so the
    recipe is unchanged. Its patience is fixed at 15 inside run_training and is not a
    flag; --skip-if-done is likewise run_grid's own behaviour, which is why finished cells
    are skipped on a second commit without passing anything."""
    return ['python', '-u', '-m', 'src.model.run_grid',
            '--datasets', DATASET, '--splits', level, '--seeds', str(seed),
            '--task', TASK, '--epochs', str(EPOCHS), '--min-epochs', '10',
            '--batch-size', str(BATCH_SIZE), '--results-dir', RESULTS] + (['--amp'] if AMP else [])


# Both KIBA levels are within 1,000 rows of each other, so the queues are balanced by
# construction; seeds are spread across GPUs so a GPU dying does not cost a whole level.
TRAIN_ROWS = {'random': 82778, 'cold_drug': 83807}
HOURS = {'coldsite_dti': (EPOCHS_TYPICAL * MIN_PER_EPOCH['coldsite_dti'] / 60,
                          EPOCHS_MIN * MIN_PER_EPOCH['coldsite_dti'] / 60)}

ORDERED = {gpu: [(MODEL, level, seed)
                 for level, seed in sorted(cells, key=lambda c: -TRAIN_ROWS[c[0]])]
           for gpu, cells in sorted(QUEUE_OF.items())}
QUEUES = {gpu: [coldsite_cmd(level, seed) for _model, level, seed in cells]
          for gpu, cells in sorted(ORDERED.items())}
MY_CELLS = ORDERED

for gpu, cells in sorted(ORDERED.items()):
    print(f'GPU {gpu}: in this order')
    for _model, level, seed in cells:
        print(f'   coldsite_dti  {level:10s} seed {seed}   {TRAIN_ROWS[level]:,} train rows  '
              f'~{HOURS["coldsite_dti"][1]:.1f}-{HOURS["coldsite_dti"][0]:.1f} h')
print(f'{hours_left():.1f} h left before the self-stop')
'''

RUNNER = head + TAIL
