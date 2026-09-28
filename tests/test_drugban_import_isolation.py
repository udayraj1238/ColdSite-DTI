"""DrugBAN and MolTrans both vendor a top-level `models.py`. One process must be able to
load both adapters, in either order -- the DAVIS audit does exactly that. Skipped where
either vendored repo or its dependencies (DGL, subword_nmt) are absent."""
import subprocess
import sys

import pytest

SCRIPT = """
import sys
from src.evaluation.baseline_adapters import MolTransAdapter
from src.evaluation.drugban_adapter import DrugBANAdapter
order = sys.argv[1]
first, second = ((MolTransAdapter, DrugBANAdapter) if order == "moltrans-first"
                 else (DrugBANAdapter, MolTransAdapter))
a = first(checkpoint_path=None, device="cpu")
b = second(checkpoint_path=None, device="cpu")
names = {type(a.model).__name__, type(b.model).__name__}
assert names == {"BIN_Interaction_Flat", "DrugBAN"}, names
assert "models" not in sys.modules or "MolTrans" in sys.modules["models"].__file__
print("ok", order)
"""


def _available():
    try:
        import dgl  # noqa: F401
        import subword_nmt  # noqa: F401
    except Exception:
        return False
    return True


@pytest.mark.parametrize("order", ["moltrans-first", "drugban-first"])
def test_both_vendored_models_load_in_one_process(order):
    if not _available():
        pytest.skip("needs DGL and subword_nmt (the drugban env)")
    out = subprocess.run([sys.executable, "-c", SCRIPT, order], capture_output=True,
                         text=True, env={"DGLBACKEND": "pytorch", **__import__("os").environ})
    assert out.returncode == 0, out.stderr[-2000:]
    assert f"ok {order}" in out.stdout
