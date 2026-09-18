# DrugBAN — vendored for the audit

Source: https://github.com/peizhenbai/DrugBAN (MIT, `LICENSE.md` kept beside this file),
cloned 2026-09-18. Paper: Bai et al., "Interpretable bilinear attention network with
domain adaptation improves drug-target prediction", *Nature Machine Intelligence* 5,
126-136 (2023), doi:10.1038/s42256-022-00605-1.

Unmodified. Everything the audit needs is outside this directory:
`src/evaluation/drugban_adapter.py` (predict/explain) and `src/model/train_drugban.py`
(their recipe on our splits). An audit that edited its subject would be measuring
something other than the published model.

`datasets/` (140 MB of their BindingDB/BioSNAP/Human splits) and `image/` are deleted
after cloning and ignored by `.gitignore`'s `baselines/**/datasets/` rule -- added
2026-09-18, because the pre-existing rule said `dataset/` singular and did not match
theirs. This audit trains on its own splits, so their data is never needed.

Requires DGL, which has no macOS-ARM wheel on PyPI. Locally:

    mamba create -n drugban python=3.11 "dgl=2.3" pytorch rdkit scikit-learn pandas -c conda-forge
    ~/miniforge3/envs/drugban/bin/pip install dgllife yacs prettytable "torchdata==0.7.1"

`torchdata` must be 0.7.x: DGL 2.3 imports `torchdata.datapipes`, which later versions
removed. Run with `DGLBACKEND=pytorch`.
