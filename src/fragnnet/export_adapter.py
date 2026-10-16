"""The interface between the inference exporters and the model they export.

``fragnnet.inference.runners``'s exporters call ``ExportAdapter`` methods on the model instead
of branching on its class. FraGNNet and MASSter mix it into their Lightning modules; any other
model gets the defaults. Moved from ``fragnnet.hooks`` (where it was ``ExportHooks``), so
"hooks" now means only the registry-looked-up ``ModelHooks``.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class ExportAdapter:
    """Methods the inference exporters (``fragnnet.inference.runners``) call on the model.

    The defaults describe a model with peak outputs only. FraGNNet and MASSter, whose exports
    carry formula/node annotation, mix this class into their Lightning modules and override what
    they need. The exporters call these on the module they hold and fall back to a default
    instance for any other object. Test stubs skip ``__init__``, so nothing here reads
    ``hparams``.
    """

    # Whether the outputs carry the formula, node and joint annotation axes.
    exports_annotation_axes = False
    # Whether the export can name the SMARTS rules behind each node.
    supports_rule_provenance = False

    def export_index_vars(
        self, nb_iso: bool
    ) -> tuple[list[str], list[str], list[str], list[str]]:
        """Index variables (batch, node, formula, nb-node) the exporter rebatches.

        Args:
            nb_iso: Whether the isomorphic-neighbourhood outputs are exported.
        """
        assert not nb_iso
        batch = ["pred_batch_idxs", "true_batch_idxs"]
        if not self.exports_annotation_axes:
            return batch, [], [], []
        return (
            batch + ["pred_formula_batch_idxs", "pred_node_batch_idxs", "pred_joint_batch_idxs"],
            ["pred_node_node_idxs", "pred_joint_node_idxs"],
            ["pred_formula_formula_idxs", "pred_joint_formula_idxs"],
            [],
        )

    def nb_index_vars(self) -> tuple[set[str], set[str], set[str]]:
        """Batch, formula and nb-node index variables of the nb_iso outputs; none by default."""
        return set(), set(), set()

    def inference_batch_options(self, exporter: str, need_annotation: bool) -> dict[str, Any]:
        """Batch keys to set before ``inference_step``.

        Args:
            exporter: "pkl" or "tabular" (the tabular and HDF5 exporters).
            need_annotation: Whether the export reads the formula or node annotation.
        """
        return {}

    def postprocess_export_peaks(
        self,
        batch_input: dict[str, Any],
        batch_result: dict[str, Any],
        peaks: tuple[Any, Any, Any],
        *,
        need_annotation: bool,
        return_cc_hash: bool,
        return_rule_provenance: bool,
    ) -> tuple[tuple[Any, Any, Any], Any]:
        """Adjust the predicted peaks before export and build any annotation state.

        Args:
            batch_input: The batch, modified in place where the model adds keys.
            batch_result: The ``inference_step`` output, modified in place likewise.
            peaks: Predicted (m/z, intensity, batch index) tensors.
            need_annotation: Whether the export reads the formula or node annotation.
            return_cc_hash: Whether node cc hashes are exported.
            return_rule_provenance: Whether node rule names are exported.

        Returns:
            The peaks to export and an opaque annotation state for the methods below (None).
        """
        return peaks, None

    def export_formula_row_deltas(
        self, batch_input: dict[str, Any], formula_cumsizes: Any, annotation: Any
    ) -> Any:
        """Precursor delta per formula row, for ``pred_formula_adducts``; None leaves NaN."""
        return None

    def export_node_ccs(self, frag_pyg: Any, b_idx: int, node_idxs: Any, annotation: Any) -> Any:
        """One spectrum's node cc bitmasks for ``pred_node_ccs``; NaN when unavailable."""
        return np.nan

    def export_node_rules(self, b_idx: int, node_idxs: Any, annotation: Any) -> Any:
        """One spectrum's node SMARTS rule names for ``pred_node_rules``; NaN when unavailable."""
        return np.nan

    def export_node_rewire_rules(self, b_idx: int, node_idxs: Any, annotation: Any) -> Any:
        """One spectrum's node rewire rule names for ``pred_node_rewire_rules``; NaN when unavailable."""
        return np.nan
