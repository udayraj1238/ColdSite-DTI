"""Where does a training step actually spend its time?

Written because a T4 was taking 2-3 minutes per DAVIS epoch and the guesses
about why (data loading, then attention) were both wrong twice over. This
measures instead of guessing: it times the protein encoder's three branches
separately, then a full training step, so the biggest number decides what is
worth optimising.

    python -m src.model.profile_forward
    python -m src.model.profile_forward --batch-size 32 --protein-len 512

The two numbers to compare are `attention (need_weights=True)` against
`attention (need_weights=False)`. The model discards the protein encoder's
self-attention weights (`coldsite_dti.py` line 54), so any gap between those
two rows is time spent building a tensor nothing reads. If instead the BiLSTM
row dominates both, the attention flag is not the bottleneck and the 1000
sequential timesteps are.
"""
import argparse
import time

import torch
import torch.nn as nn

from src.model.coldsite_dti import ColdSiteDTI
from src.model.drug_encoder import build_smiles_vocab
from src.model.protein_encoder import build_protein_vocab


def _timed(fn, n_iter, device, warmup=3):
    """Median ms per call. Synchronises, because CUDA calls are async and an
    untimed launch queue reports the launch, not the work."""
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(n_iter):
        start = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    times.sort()
    return times[len(times) // 2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--protein-len", type=int, default=1000)
    parser.add_argument("--smiles-len", type=int, default=100)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device      : {device}")
    if device.type == "cuda":
        print(f"gpu         : {torch.cuda.get_device_name(0)}")
    print(f"batch       : {args.batch_size}")
    print(f"protein len : {args.protein_len}")
    print()

    protein_vocab = build_protein_vocab()
    # Timing depends on the embedding's vocab size only through one lookup, so a
    # representative character set is enough -- no split file needed to profile.
    smiles_vocab = build_smiles_vocab(
        ["CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
         "C1CC[NH+](CC1)CCOc1ccc(cc1)C(F)(F)F", "[Na+].[Cl-]", "c1ccc2[nH]ccc2c1"])
    model = ColdSiteDTI(len(smiles_vocab) + 2, len(protein_vocab) + 2).to(device)

    # Full-length sequences with no padding: the worst case, and the one the
    # 1000-residue proteins actually hit.
    protein = torch.randint(2, len(protein_vocab), (args.batch_size, args.protein_len),
                            device=device)
    drug = torch.randint(2, len(smiles_vocab), (args.batch_size, args.smiles_len),
                         device=device)

    enc = model.protein_encoder
    emb = enc.embedding(protein)
    mask = torch.zeros(args.batch_size, args.protein_len, dtype=torch.bool, device=device)

    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
    lengths = torch.full((args.batch_size,), args.protein_len, dtype=torch.long)

    def conv_branch():
        out = torch.relu(enc.conv(emb.permute(0, 2, 1))).permute(0, 2, 1)
        return out

    def lstm_branch():
        packed = pack_padded_sequence(emb, lengths, batch_first=True,
                                      enforce_sorted=False)
        packed_out, _ = enc.bilstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True,
                                     total_length=args.protein_len)
        return out

    merged = enc.pre_norm(enc.merge(torch.cat([conv_branch(), lstm_branch()], dim=-1)))

    def attn_with_weights():
        return enc.attention(merged, merged, merged, key_padding_mask=mask,
                             need_weights=True)

    def attn_without_weights():
        return enc.attention(merged, merged, merged, key_padding_mask=mask,
                             need_weights=False)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    target = torch.randn(args.batch_size, 1, device=device)

    def full_step():
        optimizer.zero_grad()
        pred, _ = model(drug, protein)
        loss = loss_fn(pred, target)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        rows = [
            ("conv branch", _timed(conv_branch, args.iters, device)),
            ("BiLSTM branch", _timed(lstm_branch, args.iters, device)),
            ("attention (need_weights=True)",
             _timed(attn_with_weights, args.iters, device)),
            ("attention (need_weights=False)",
             _timed(attn_without_weights, args.iters, device)),
        ]

    rows.append(("FULL train step (fwd+bwd+opt)",
                 _timed(full_step, args.iters, device)))

    width = max(len(name) for name, _ in rows)
    print(f"{'component'.ljust(width)}   median ms")
    print("-" * (width + 14))
    for name, ms in rows:
        print(f"{name.ljust(width)}   {ms:9.1f}")

    with_w = dict(rows)["attention (need_weights=True)"]
    without_w = dict(rows)["attention (need_weights=False)"]
    step = dict(rows)["FULL train step (fwd+bwd+opt)"]
    saved = with_w - without_w

    print()
    print(f"dropping the discarded self-attention weights saves {saved:.1f} ms "
          f"of a {step:.1f} ms step ({100 * saved / step:.1f}%) on the forward "
          f"pass alone.")
    print()
    print("Per DAVIS epoch (329 batches of 64):")
    print(f"  now              : {step * 329 / 60000:.1f} min")
    print(f"  weights dropped  : {(step - saved) * 329 / 60000:.1f} min  "
          f"(lower bound -- backward saves more)")


if __name__ == "__main__":
    main()
