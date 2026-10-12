"""Core output schema (reorganization plan Phase 3 step 2).

``fragnnet.model.output_schema`` holds the model-output names and switch defaults that core reads,
so ``SpectrumPL`` no longer imports FraGNNet's model module for them.
"""

from __future__ import annotations

import inspect

from fragnnet.model.fragnnet_model import FraGNNetModel
from fragnnet.model.output_schema import (
    EXPLAINABILITY_OUTPUT_KEYS,
    RETURN_EXPLAINABILITY_TENSORS_DEFAULT,
)


def test_switch_defaults_are_unchanged() -> None:
    """FraGNNet computes its marginals unless told not to; MASSter annotates only on request."""
    assert RETURN_EXPLAINABILITY_TENSORS_DEFAULT == {"fragnnet": True, "masster": False}
    param = inspect.signature(FraGNNetModel.forward).parameters["return_explainability_tensors"]
    assert param.default is RETURN_EXPLAINABILITY_TENSORS_DEFAULT["fragnnet"]


def test_gated_set_excludes_the_always_returned_index_arrays() -> None:
    assert isinstance(EXPLAINABILITY_OUTPUT_KEYS, frozenset)
    assert not {"pred_formula_batch_idxs", "pred_node_batch_idxs"} & EXPLAINABILITY_OUTPUT_KEYS
    assert len(EXPLAINABILITY_OUTPUT_KEYS) == 27
