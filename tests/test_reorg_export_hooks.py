"""Export hooks (reorganization plan Phase 4 step 5b).

The inference exporters in ``fragnnet.utils.script_utils`` call ``fragnnet.hooks.ExportHooks``
methods on the model they hold instead of branching on ``FraGNNetPL``/``MassterPL``. The export
records in ``test_reorg_export.py`` pin the behavior; this file covers what they do not reach:
the nb_iso index variables (no export fixture turns nb_iso on), the per-exporter explainability
switch, and the fallback for objects that are not export-hook models.
"""

from __future__ import annotations

import pytest

from fragnnet.hooks import ExportHooks
from fragnnet.models.fragnnet.fragnnet_pl import FraGNNetPL
from fragnnet.models.masster.masster_pl import MassterPL
from fragnnet.utils import script_utils

PEAKS = ["pred_batch_idxs", "true_batch_idxs"]
ANNOTATION = (
    PEAKS + ["pred_formula_batch_idxs", "pred_node_batch_idxs", "pred_joint_batch_idxs"],
    ["pred_node_node_idxs", "pred_joint_node_idxs"],
    ["pred_formula_formula_idxs", "pred_joint_formula_idxs"],
    [],
)
# The pre-hook get_idx_vars lists for FraGNNet with nb_iso, verbatim.
FRAGNNET_NB_ISO = (
    ANNOTATION[0]
    + ["pred_nb_node_batch_idxs", "pred_nb_joint_batch_idxs", "pred_nb_node_node_batch_idxs"],
    ANNOTATION[1],
    ANNOTATION[2] + ["pred_nb_joint_formula_idxs"],
    ["pred_nb_node_node_idxs", "pred_nb_joint_node_idxs", "pred_nb_node_node_node_idxs"],
)


def _stub(cls):
    """An instance that skips ``__init__``, like the exporter tests' stubs."""
    return cls.__new__(cls)


@pytest.mark.parametrize(
    ("model", "nb_iso", "expected"),
    [
        (None, False, (PEAKS, [], [], [])),
        (object(), False, (PEAKS, [], [], [])),
        ("masster", False, ANNOTATION),
        ("fragnnet", False, ANNOTATION),
        ("fragnnet", True, FRAGNNET_NB_ISO),
    ],
    ids=["none", "plain_object", "masster", "fragnnet", "fragnnet_nb_iso"],
)
def test_index_vars_match_the_pre_hook_lists(model, nb_iso, expected) -> None:
    model = {"masster": _stub(MassterPL), "fragnnet": _stub(FraGNNetPL)}.get(model, model)
    assert script_utils.get_idx_vars(None, model, nb_iso) == tuple(expected)


@pytest.mark.parametrize("model", [None, object(), "masster"])
def test_nb_iso_is_fragnnet_only(model) -> None:
    model = _stub(MassterPL) if model == "masster" else model
    with pytest.raises(AssertionError):
        script_utils.get_idx_vars(None, model, nb_iso=True)


def test_required_index_vars_add_the_nb_iso_names_for_fragnnet_only() -> None:
    subset = set(script_utils.OUTPUT_SUBSET_OPTIONS)
    plain = script_utils.required_index_vars_for_outputs(subset, None, nb_iso=True)
    masster = script_utils.required_index_vars_for_outputs(subset, _stub(MassterPL), True)
    fragnnet = script_utils.required_index_vars_for_outputs(subset, _stub(FraGNNetPL), True)
    assert plain == masster
    nb_batch, nb_formula, nb_node = (
        set(FRAGNNET_NB_ISO[i]) - set(ANNOTATION[i]) for i in (0, 2, 3)
    )
    assert fragnnet[0] == plain[0] | nb_batch
    assert fragnnet[1] == plain[1]
    assert fragnnet[2] == plain[2] | nb_formula
    assert fragnnet[3] == plain[3] | nb_node


@pytest.mark.parametrize(
    ("model", "exporter", "expected"),
    [
        ("plain", "pkl", {}),
        ("plain", "tabular", {}),
        ("fragnnet", "pkl", {"return_explainability_tensors": False}),
        ("fragnnet", "tabular", {"return_explainability_tensors": False}),
        ("masster", "pkl", {}),
        ("masster", "tabular", {"return_explainability_tensors": False}),
    ],
)
def test_explainability_switch_per_model_and_exporter(model, exporter, expected) -> None:
    """FraGNNet gets the switch from both exporters; MASSter only from the tabular ones."""
    hooks = {"plain": ExportHooks(), "fragnnet": _stub(FraGNNetPL), "masster": _stub(MassterPL)}
    assert hooks[model].inference_batch_options(exporter, False) == expected


def test_capability_flags() -> None:
    assert not ExportHooks.exports_annotation_axes and not ExportHooks.supports_rule_provenance
    assert FraGNNetPL.exports_annotation_axes and not FraGNNetPL.supports_rule_provenance
    assert MassterPL.exports_annotation_axes and MassterPL.supports_rule_provenance
