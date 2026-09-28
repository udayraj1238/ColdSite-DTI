# EviDTI's published residue attention is a function of the protein alone

**Subject.** Zhao et al., "Evidential deep learning-based drug-target interaction
prediction", *Nature Communications* 16:6915 (2025), doi:10.1038/s41467-025-62235-6.
Code: https://github.com/zhaoyanpeng208/EviDTI (CC-BY-4.0; archived at
https://zenodo.org/records/15760471). Read 2026-09-18, repository state of 2025-06-29.

**Their claim.** Figure 6 is captioned "Visualization of attention scores of all the
residues in the four randomly selected drug-target complexes", and the text states that
"residues with high attention values coincide with the binding site, underscoring its
importance in predicting and validating the attention mechanism's efficacy."

**What the code computes.** In `davis_model.py` (identically in `drugbank_model.py` and
`ligheattention.py`), inside `forward`:

    t_1D = ...                                     # data.t_1D_feature: ProtTrans embeddings
    t_o  = self.feature_convolution(t_1D)
    attention = self.attention_convolution(t_1D)   # [batch, embeddings_dim, seq_len]
    attention = attention.masked_fill(mask[:, None, :] == False, -1e9)
    att_AA = torch.mean(attention, dim=1)          # one score per residue

`att_AA` is the per-residue quantity the case study plots: `infernce.py` collects it into
`att_list` and writes `case_att_result.csv`.

**The observation.** `att_AA` is computed from `t_1D` and the padding mask. `t_1D` is the
protein's ProtTrans embedding. No drug tensor appears anywhere in that expression, and
the drug branches (`d_2D_embedding`, the 3D conformer graphs) are first used *after* it,
at the fusion step:

    cat_v = torch.cat((t_o, d_o, atom_h), 1)

There is no cross-attention between the drug and the protein in any of the three model
files. The dependency is therefore syntactic, not empirical: **for a fixed protein, every
drug gives the identical residue map**, for any weights, trained or not.

**What follows, and what does not.**

* The map is a protein-level saliency. It can agree with a binding site -- an ATP pocket
  is a property of the kinase, not of the ligand -- but that agreement cannot be evidence
  that the model has localised *this drug's* interaction, which is what a figure of four
  drug-target complexes invites the reader to conclude. Any two of those four complexes
  that share a protein necessarily carry the same map.
* This bears only on the interpretability claim. EviDTI's contribution is uncertainty
  quantification through evidential deep learning, and nothing here touches it.
* We did not retrain EviDTI, so we report no precision@k for it and make no claim about
  how well its map agrees with annotated residues. Retraining needs PyTorch *and*
  TensorFlow (their 2D drug encoder) *and* PaddlePaddle (their 3D encoder, whose
  pretrained weights are not in the repository) -- the reason it is not one of this
  audit's trained subjects.

**How to re-check this in under a minute.**

    git clone https://github.com/zhaoyanpeng208/EviDTI
    grep -n "att_AA\|attention_convolution\|torch.cat((t_o" EviDTI/davis_model.py

Read the lines around each hit and look for any drug tensor reaching `att_AA`.
