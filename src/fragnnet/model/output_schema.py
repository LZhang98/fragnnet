"""Model-output names and switches that core code reads.

``SpectrumPL`` and the inference exporter consume these, so they live in core even though only
the fragment models produce the outputs (docs/plans/repo_reorg_plan.md, "Annotation output
contract").
"""

from __future__ import annotations

# Default of the per-call ``return_explainability_tensors`` batch switch when a batch omits it.
# FraGNNet computes its explainability marginals unless told not to; MASSter builds its
# per-pathway annotation only when asked.
RETURN_EXPLAINABILITY_TENSORS_DEFAULT = {"fragnnet": True, "masster": False}

# The gated set: outputs of `FraGNNetModel.forward` that only exist when the terminal
# explainability marginals are computed. A forward with `return_explainability_tensors=False`
# returns None for every one of them. This is not the full annotation schema:
# `pred_formula_batch_idxs` and `pred_node_batch_idxs` are always returned (the metrics read
# them).
EXPLAINABILITY_OUTPUT_KEYS = frozenset(
    {
        "pred_formula_logprobs",
        "pred_node_logprobs",
        "pred_node_formula_logprobs",
        "pred_formula_node_logprobs",
        "pred_joint_logprobs",
        "pred_joint_node_idxs",
        "pred_joint_formula_idxs",
        "pred_joint_batch_idxs",
        "pred_h_counts",
        "pred_h_batch_idxs",
        "pred_h_logprobs",
        "pred_formula_formula_idxs",
        "pred_node_node_idxs",
        "pred_joint_h_counts",
        "pred_joint_h_idxs",
        # nb_iso (isomorphic-neighbourhood) marginals ride the same gate
        "pred_nb_node_logprobs",
        "pred_nb_node_formula_logprobs",
        "pred_nb_formula_node_logprobs",
        "pred_nb_node_node_logprobs",
        "pred_nb_node_node_idxs",
        "pred_nb_node_batch_idxs",
        "pred_nb_joint_logprobs",
        "pred_nb_joint_node_idxs",
        "pred_nb_joint_formula_idxs",
        "pred_nb_joint_batch_idxs",
        "pred_nb_node_node_node_idxs",
        "pred_nb_node_node_batch_idxs",
    }
)
