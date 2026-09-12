"""
Track B (124AD0015) — Dataset / DataLoader over Track A's split files.

This is the "fill in the DataLoader / dataset class" TODO left in train.py.
Reads the CSVs in data/splits/. Column names are auto-detected from a few common
spellings so this does not break the first time a file arrives with `Drug`
instead of `compound_iso_smiles`.
"""
import random

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from src.model.drug_encoder import PAD_IDX, build_smiles_vocab, encode_smiles
from src.model.protein_encoder import build_protein_vocab, encode_protein

SMILES_COLUMNS = ["compound_iso_smiles", "smiles", "drug", "drug_smiles", "ligand"]
PROTEIN_COLUMNS = ["target_sequence", "protein", "target", "sequence", "aa_sequence"]
LABEL_COLUMNS = ["affinity", "label", "y", "value", "interaction"]

# The binary task calls a pair binding when its affinity reaches these values:
# DeepDTA's own published thresholds, DAVIS pKd >= 7.0 and KIBA >= 12.1.
#
# The one definition. DeepDTA, HyperAttentionDTI and ColdSite-DTI all import
# it, because the audit table compares their AUROCs, and three models scored
# against three separately maintained thresholds would be three different
# tasks presented as one.
BINARY_THRESHOLD = {"davis": 7.0, "kiba": 12.1}


def find_column(df: pd.DataFrame, candidates: list, role: str) -> str:
    lowered = {c.lower().strip(): c for c in df.columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    raise KeyError(f"No {role} column found. Looked for {candidates}, "
                   f"file has {list(df.columns)}")


class DTIDataset(Dataset):
    """One split of one dataset, e.g. data/splits/davis/cold_target/train.csv"""

    def __init__(self, smiles, proteins, labels, drug_vocab, protein_vocab,
                 max_smiles_len=100, max_protein_len=1000):
        if not (len(smiles) == len(proteins) == len(labels)):
            raise ValueError("smiles, proteins and labels must be the same length")
        self.smiles, self.proteins, self.labels = smiles, proteins, labels
        self.drug_vocab, self.protein_vocab = drug_vocab, protein_vocab
        self.max_smiles_len, self.max_protein_len = max_smiles_len, max_protein_len

    @classmethod
    def from_csv(cls, path, drug_vocab, protein_vocab, **kwargs):
        return cls.from_frame(pd.read_csv(path), drug_vocab, protein_vocab, **kwargs)

    @classmethod
    def from_frame(cls, df, drug_vocab, protein_vocab, label_threshold=None, **kwargs):
        """Same as from_csv, for a frame already in memory.

        load_split needs this: it may subsample the training rows before
        building the vocabulary, so it cannot re-read the file afterwards.

        `label_threshold` turns affinities into binding labels, 1.0 at or above
        it and 0.0 below. The split files hold raw affinities, so without it a
        binary run hands the loss pKd values of 5-11 as though they were
        classes -- and fails at the first AUROC, which refuses continuous
        labels.
        """
        s = find_column(df, SMILES_COLUMNS, "SMILES")
        p = find_column(df, PROTEIN_COLUMNS, "protein sequence")
        y = find_column(df, LABEL_COLUMNS, "label")
        df = df.dropna(subset=[s, p, y])
        labels = df[y].astype(float)
        if label_threshold is not None:
            labels = (labels >= label_threshold).astype(float)
        return cls(df[s].astype(str).tolist(),
                   df[p].astype(str).str.upper().tolist(),
                   labels.tolist(),
                   drug_vocab, protein_vocab, **kwargs)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        drug = torch.tensor(encode_smiles(self.smiles[idx], self.drug_vocab,
                                          self.max_smiles_len), dtype=torch.long)
        protein = torch.tensor(encode_protein(self.proteins[idx], self.protein_vocab,
                                              self.max_protein_len), dtype=torch.long)
        return drug, protein, torch.tensor(self.labels[idx], dtype=torch.float)


def collate_batch(batch):
    """Pad each batch to its own longest drug and longest protein.

    Padding to a global maximum instead would waste most of the compute: DAVIS
    proteins run from roughly 200 to 2500 residues.
    """
    drugs, proteins, labels = zip(*batch)
    return (pad_sequence(drugs, batch_first=True, padding_value=PAD_IDX),
            pad_sequence(proteins, batch_first=True, padding_value=PAD_IDX),
            torch.stack(labels))


def make_loader(dataset, batch_size=64, shuffle=False, workers=0):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate_batch, num_workers=workers)


def load_split(split_dir, max_protein_len=1000, batch_size=64,
               train_subsample=None, subsample_seed=0, binary_threshold=None):
    """Build train/valid/test loaders for one split directory.

    Returns (train_loader, valid_loader, test_loader, drug_vocab, protein_vocab).

    `train_subsample` keeps only that many training rows, chosen at random
    against `subsample_seed`. It exists for one experiment: the cold-pair split
    trains on roughly 71% of the pairs the other three levels get, because it
    discards every row with exactly one cold entity. Without a volume-matched
    control, the accuracy drop at cold-pair reads as pure difficulty when part
    of it is simply less training data. Valid and test are never subsampled --
    the control has to be evaluated on the same test set as the run it is
    being compared against, or it answers a different question.
    """
    train_df = pd.read_csv(f"{split_dir}/train.csv")
    if train_subsample is not None and train_subsample < len(train_df):
        train_df = train_df.sample(n=int(train_subsample),
                                   random_state=subsample_seed)
    # Vocab is built from TRAIN ONLY. Building it from all three splits would leak
    # information about unseen drugs -- precisely what the cold splits measure.
    # Built AFTER subsampling on purpose: a smaller training set genuinely sees
    # fewer SMILES tokens, and that shrinkage is part of the effect being
    # measured, not an artefact to correct for.
    drug_vocab = build_smiles_vocab(
        train_df[find_column(train_df, SMILES_COLUMNS, "SMILES")].astype(str).tolist()
    )
    protein_vocab = build_protein_vocab()

    # binary_threshold: see BINARY_THRESHOLD. Applied to all three parts, so a
    # model is trained, early-stopped and tested against the same labels.
    common = dict(drug_vocab=drug_vocab, protein_vocab=protein_vocab,
                  max_protein_len=max_protein_len, label_threshold=binary_threshold)
    train = DTIDataset.from_frame(train_df, **common)
    valid = DTIDataset.from_csv(f"{split_dir}/valid.csv", **common)
    test = DTIDataset.from_csv(f"{split_dir}/test.csv", **common)

    return (make_loader(train, batch_size, shuffle=True),
            make_loader(valid, batch_size),
            make_loader(test, batch_size),
            drug_vocab, protein_vocab)


def random_dataset(n=256, binary=True, seed=0):
    """Random SMILES-shaped and protein-shaped strings, random labels.

    The labels carry no signal by design. This exists to prove the loop runs, not
    to prove the model learns; expect AUROC near 0.5 and CI near 0.5.
    """
    rng = random.Random(seed)
    smiles_chars = "CNOSPFIcln()[]=#+-123456"
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    smiles = ["".join(rng.choices(smiles_chars, k=rng.randint(10, 90)))
              for _ in range(n)]
    proteins = ["".join(rng.choices(amino_acids, k=rng.randint(60, 400)))
                for _ in range(n)]
    labels = [float(rng.randint(0, 1)) if binary else rng.uniform(5.0, 11.0)
              for _ in range(n)]
    return DTIDataset(smiles, proteins, labels,
                      build_smiles_vocab(smiles), build_protein_vocab())


if __name__ == "__main__":
    ds = random_dataset(32)
    drugs, proteins, labels = next(iter(make_loader(ds, batch_size=8)))
    print("Drug batch:", drugs.shape)        # (8, <= 90)
    print("Protein batch:", proteins.shape)  # (8, <= 400)
    print("Labels:", labels.shape)           # (8,)
