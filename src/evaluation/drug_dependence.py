"""Does an explanation change when the drug does?  (Results §7e)

For 25 DAVIS proteins (random split, seed 1, the first test row per protein in a fixed
sample) and 4 drugs each, the top-10 residues of the explanation are compared between
every pair of drugs on the same protein. 10/10 in every pair means the map is, in
practice, a function of the protein alone -- whatever the architecture allows.

    python -m src.evaluation.drug_dependence <model> > results/drug_dependence/<model>.txt
"""
import os, sys, itertools, numpy as np, pandas as pd
from src.evaluation.model_registry import model_class
from src.model.checkpoint_naming import checkpoint_path
model = sys.argv[1]
ck = checkpoint_path(os.path.expanduser('~/ColdSite-results/davis_binary'), 'davis', 'random', 'binary', 1, model=model)
test = pd.read_csv('data/splits/davis/random/test.csv')
rng = np.random.default_rng(0)
proteins = test.drop_duplicates('Target_ID').sample(25, random_state=0)
drugs = test.drop_duplicates('Drug')['Drug'].tolist()
m = model_class(model)(checkpoint_path=ck)
top = lambda w: set(np.argsort(-np.asarray(w)[:1000])[:10])
ov = []
for _, r in proteins.iterrows():
    maps = []
    for smi in rng.choice(drugs, 4, replace=False):
        enc = type(m).encode(smi, r.Target)
        if model.startswith('moltrans'):
            d, dm, p, pm, t = enc; maps.append(top(m.explain(d, p, protein_tokens=t, drug_mask=dm, protein_mask=pm)))
        else:
            maps.append(top(m.explain(*enc)))
    ov += [len(a & b) for a, b in itertools.combinations(maps, 2)]
print(f'{model}: top-10 overlap between different drugs on the same protein: mean {np.mean(ov):.2f}/10, '
      f'identical top-10 in {np.mean(np.array(ov)==10):.0%} of {len(ov)} drug pairs (25 proteins x 4 drugs)')
