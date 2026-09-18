"""
Model registry — the audit's central abstraction.

Under the original framing, DeepDTA / HyperAttentionDTI / MolTrans were
comparisons: numbers in a table proving ColdSite-DTI was competitive. Under the
audit framing they are SUBJECTS. The paper's claim is about published
interpretability claims in general, so every model with an attention-based
explanation has to run through the identical measurement pipeline.

That only works if they all expose the same two things:

    predict(drug, protein) -> a scalar
    explain(drug, protein) -> one weight per protein residue, 0-indexed

Nothing else about a model matters here. Wrap each one in an adapter satisfying
that contract and the whole of precision@k, the permutation tests and the
faithfulness masking applies unchanged.

Adding a model
--------------
1. Subclass ExplainableDTIModel.
2. Implement predict() and explain().
3. Register it with @register("name").
4. Run tests/test_model_registry.py -- the contract tests catch the mistakes
   that would otherwise show up as a wrong number rather than an error.

The contract is deliberately strict about explain(). A model that returns
attention over tokens rather than residues, or that silently truncates, will
produce a plausible-looking precision@k that is measuring the wrong thing.
"""
from __future__ import annotations

import abc

import numpy as np
import torch

_REGISTRY: dict = {}


def register(name: str):
    """Class decorator adding a model adapter to the registry."""
    def decorator(cls):
        if name in _REGISTRY:
            raise ValueError(f"'{name}' is already registered")
        _REGISTRY[name] = cls
        cls.registry_name = name
        return cls
    return decorator


def load_variant_plugins(name: str = "") -> None:
    """Import the modules that register explanation variants of an existing model.

    An adapter registers itself when its module is imported, and the integrated-gradients
    variants live outside this file (`src/evaluation/integrated_gradients.py`, which
    imports from here -- so it cannot be imported at the top of this one). Every lookup
    path calls this first, so `moltrans_ig` resolves wherever `moltrans` does instead of
    depending on which module happened to be imported already.
    """
    from src.model.checkpoint_naming import VARIANT_BASE
    # The baseline adapters register on import and every other adapter is a subclass of
    # one, so they load for ANY lookup. Skipping them for non-variant names made
    # model_class('moltrans') raise "unknown model 'moltrans'" while the error message
    # -- which calls available_models() and so triggers the import -- listed it.
    from src.evaluation import baseline_adapters    # noqa: F401  (registers on import)
    # DrugBAN lives in its own module because it needs DGL, which has no macOS-ARM wheel
    # on PyPI. Importing the module is still safe there: every DGL import inside it is
    # deferred to the call that needs one, so a machine without DGL can still list the
    # model, read its class-level facts, and run the rest of the suite.
    from src.evaluation import drugban_adapter      # noqa: F401  (same)
    if name and name not in VARIANT_BASE:
        return
    import src.evaluation.integrated_gradients      # noqa: F401  (registers on import)
    import src.evaluation.readout_variants          # noqa: F401  (same)


def available_models() -> list:
    load_variant_plugins()
    return sorted(_REGISTRY)


def model_class(name: str):
    """The adapter class, uninstantiated.

    Needed to read class-level facts -- `provides_attention` above all -- before
    deciding whether to build the thing. Instantiating a baseline adapter loads
    a checkpoint and imports a vendored repo, which is a lot of work to discover
    that the model has no attention to audit.
    """
    load_variant_plugins(name)
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown model '{name}'. Registered: {available_models()}. "
            f"Add an adapter in src/evaluation/model_registry.py."
        )
    return _REGISTRY[name]


def get_model(name: str, **kwargs):
    return model_class(name)(**kwargs)


class ExplainableDTIModel(abc.ABC):
    """Anything the audit can measure.

    `provides_attention = False` marks a model that predicts but cannot
    explain. Those still belong in the audit: they anchor the accuracy axis, so
    a reviewer can see whether the interpretable models pay an accuracy cost.
    """

    registry_name: str = "unnamed"
    provides_attention: bool = True
    citation: str = ""

    @abc.abstractmethod
    def predict(self, drug, protein) -> float:
        """Scalar affinity or logit for one drug-protein pair."""

    @abc.abstractmethod
    def explain(self, drug, protein) -> np.ndarray:
        """1D array, one non-negative weight per protein residue, 0-indexed.

        Length MUST equal the number of real residues the model saw. If the
        model truncates, return the truncated length and record the truncation
        -- do not pad the array back out, because every ground-truth index past
        the cut would then align against a weight that does not exist.
        """

    def explain_batch(self, drugs, proteins) -> list:
        return [self.explain(d, p) for d, p in zip(drugs, proteins)]

    def __repr__(self):
        return f"<{type(self).__name__} name={self.registry_name!r}>"


@register("coldsite_dti")
class ColdSiteDTIAdapter(ExplainableDTIModel):
    """Our own model. Cross-attention weights are the explanation."""

    citation = "This work"

    def __init__(self, checkpoint_path: str = None, drug_vocab_size: int = 70,
                 protein_vocab_size: int = 28, device: str = "cpu"):
        from src.model.coldsite_dti import ColdSiteDTI

        self.device = device
        self.model = ColdSiteDTI(drug_vocab_size, protein_vocab_size)
        if checkpoint_path:
            state = torch.load(checkpoint_path, map_location=device,
                               weights_only=False)
            self.model.load_state_dict(state["model_state"])
        self.model.to(device).eval()

    @torch.no_grad()
    def predict(self, drug, protein) -> float:
        drug, protein = _batchify(drug), _batchify(protein)
        pred, _attn = self.model(drug.to(self.device), protein.to(self.device))
        return float(pred.squeeze().item())

    def explain(self, drug, protein) -> np.ndarray:
        drug, protein = _batchify(drug), _batchify(protein)
        weights = self.model.explain(drug.to(self.device), protein.to(self.device))
        return np.asarray(weights[0], dtype=float)


class UniformBaseline(ExplainableDTIModel):
    """Flat attention over every residue. The null explainer.

    Registered so the audit always has a floor. Any real model whose
    precision@k is statistically indistinguishable from this one has no
    explanatory content, whatever its accuracy -- and reporting that comparison
    is cheap insurance against a tie-breaking or indexing artefact being read
    as a finding.
    """

    citation = "Control, this work"

    def __init__(self, wrapped: ExplainableDTIModel = None):
        self.wrapped = wrapped

    def predict(self, drug, protein) -> float:
        return self.wrapped.predict(drug, protein) if self.wrapped else 0.0

    def explain(self, drug, protein) -> np.ndarray:
        # real_lengths(), not (protein != 0).sum(): counting non-pad tokens and
        # measuring to the last non-pad position disagree on any sequence with
        # an interior pad, and the count is SHORTER than the residues it covers.
        # This is the control the whole audit is compared against -- a floor
        # line one residue short is scored against shifted ground truth, and
        # nothing raises. Same reasoning as ColdSiteDTI.explain().
        from src.model.protein_encoder import real_lengths

        length = int(real_lengths(_batchify(protein))[0].item())
        return np.ones(max(length, 1), dtype=float) / max(length, 1)


_REGISTRY["uniform_control"] = UniformBaseline
UniformBaseline.registry_name = "uniform_control"


def _batchify(tensor: torch.Tensor) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)
    return tensor if tensor.dim() == 2 else tensor.unsqueeze(0)


def validate_adapter(adapter: ExplainableDTIModel, drug, protein,
                     expected_length: int = None) -> dict:
    """Check an adapter honours the contract before it is used for real.

    Run this on every new adapter. A silent contract violation here does not
    crash -- it produces a wrong precision@k, which is much worse.
    """
    problems = []

    try:
        prediction = adapter.predict(drug, protein)
        if not np.isscalar(prediction) and not isinstance(prediction, float):
            problems.append(f"predict() returned {type(prediction)}, want float")
        elif not np.isfinite(prediction):
            problems.append("predict() returned NaN or inf")
    except Exception as exc:
        problems.append(f"predict() raised {type(exc).__name__}: {exc}")
        prediction = None

    weights = None
    try:
        weights = np.asarray(adapter.explain(drug, protein))
        if weights.ndim != 1:
            problems.append(f"explain() returned shape {weights.shape}, want 1D")
        if not np.all(np.isfinite(weights)):
            problems.append("explain() returned NaN or inf")
        if weights.size and weights.min() < 0:
            problems.append("explain() returned negative weights")
        if expected_length is not None and weights.size != expected_length:
            problems.append(
                f"explain() returned {weights.size} weights for a protein of "
                f"{expected_length} residues -- ground-truth indices will "
                f"misalign")
    except Exception as exc:
        problems.append(f"explain() raised {type(exc).__name__}: {exc}")

    return {
        "adapter": adapter.registry_name,
        "valid": not problems,
        "problems": problems,
        "n_weights": int(weights.size) if weights is not None else 0,
    }
