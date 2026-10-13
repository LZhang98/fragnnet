"""The inference export schema: the column specs, the output-subset options a run can select,
and the index variables and score types the exporters need (split out of
``utils/script_utils.py``).
"""

from dataclasses import dataclass
from typing import Any

import torch as th

from fragnnet.export_adapter import ExportAdapter


@dataclass(frozen=True)
class ColumnSpec:
    """One column of the inference export.

    Attributes:
        name: Column name in the DataFrame and dataset name in HDF5.
        family: Which axis the column is ragged along, or "" for one value per spectrum.
            A ragged family's members all carry a per-spectrum block and share one index
            pair; the four families' block lengths are unrelated.
        token: The `output_subset` token that requests this column, or None for a
            structural column (identifiers, and the index arrays that make a family's
            probabilities interpretable) that is always emitted.
        flag: Name of the `process_batch_to_tabular_row` boolean that gates this column
            instead of a token, for the two annotation columns.
    """

    name: str
    family: str = ""
    token: str | None = None
    flag: str | None = None


# The single source of truth for the export: column order, HDF5 family grouping, and the
# `output_subset` vocabulary are all read off this table, so they cannot drift apart.
# Order here is the column order of the emitted DataFrame, ahead of the eval-score columns.
COLUMN_SPECS = (
    ColumnSpec("spec_id"),
    ColumnSpec("mol_id"),
    ColumnSpec("pred_mzs", "peak", token="pred_mzs"),
    ColumnSpec("pred_ints", "peak", token="pred_ints"),
    ColumnSpec("pred_oos_prob", token="pred_oos_prob"),
    ColumnSpec("pred_rt", token="pred_rt"),
    ColumnSpec("pred_ccs", token="pred_ccs"),
    ColumnSpec("frag_formula_peak_idxs", "peak"),
    ColumnSpec("frag_formula_peak_mzs", "peak", token="frag_formula_peak_mzs"),
    ColumnSpec("pred_formula_formula_idxs", "formula"),
    ColumnSpec("pred_formula_probs", "formula", token="pred_formula_probs"),
    ColumnSpec("pred_joint_formula_idxs", "joint", token="pred_joint_formula_idxs"),
    ColumnSpec("pred_formula_node_probs", "joint", token="pred_formula_node_probs"),
    ColumnSpec("pred_joint_node_idxs", "joint", token="pred_joint_node_idxs"),
    ColumnSpec("pred_node_formula_probs", "joint", token="pred_node_formula_probs"),
    ColumnSpec("pred_node_probs", "node", token="pred_node_probs"),
    ColumnSpec("pred_node_node_idxs", "node"),
    ColumnSpec("index_rebased"),
    ColumnSpec("pred_formula_strs", "formula", flag="return_formula_str"),
    ColumnSpec("pred_formula_adducts", "formula", flag="return_annotation"),
    ColumnSpec("pred_node_ccs", "node", flag="return_cc_hash"),
    ColumnSpec("pred_node_rules", "node", flag="return_rule_provenance"),
)


# Valid tokens a user can request in `output_subset` for inference outputs.
OUTPUT_SUBSET_OPTIONS = [c.token for c in COLUMN_SPECS if c.token is not None]


def selected_column_specs(
    return_formula_str: bool,
    return_cc_hash: bool,
    return_rule_provenance: bool = False,
) -> list[ColumnSpec]:
    """The columns this run emits, in order.

    `output_subset` deliberately does NOT narrow this list: a column outside the subset is
    still emitted, holding NaN, and `--drop_none_cols` is what removes it. Both the column
    names and the row values are built from this one function, so a row can never be
    misaligned against its header.

    `return_rule_provenance` adds `pred_node_rules`, MASSter's SMARTS rules behind each
    node. It describes the nodes `pred_node_ccs` identifies, so it requires `return_cc_hash`.

    Raises:
        ValueError: If `return_rule_provenance` is set without `return_cc_hash`.
    """
    if return_rule_provenance and not return_cc_hash:
        raise ValueError("return_rule_provenance annotates cc-hash nodes; set return_cc_hash")
    flags = {
        "return_formula_str": return_formula_str,
        "return_cc_hash": return_cc_hash,
        "return_rule_provenance": return_rule_provenance,
        # `pred_formula_adducts` is what makes either annotation resolvable: it turns a
        # neutral `pred_formula_strs` entry into the observed ion, and tells a `pred_node_ccs`
        # bitmask which adduct that fragment carries on the CRF/CMF row it was reached by.
        "return_annotation": return_formula_str or return_cc_hash,
    }
    return [c for c in COLUMN_SPECS if c.flag is None or flags[c.flag]]


def list_output_subset_options() -> list[str]:
    """List all valid output subset option tokens.

    Returns:
        List of valid output subset tokens that can be used in inference
    """
    return OUTPUT_SUBSET_OPTIONS


def prepare_output_subset(
    output_everything: bool,
    output_subset: set[str] | None,
    model_metric_names: set[str] | None = None,
) -> set[str] | None:
    """
    Prepare output subset for inference results.

    Args:
            output_everything: Whether to output all metrics
            output_subset: Initial output subset
            model_metric_names: Model-specific metric names to include

    Returns:
            Final output subset (None if output_everything is True)
    """
    if output_everything:
        return None

    if output_subset is None:
        output_subset = set()
    else:
        output_subset = set(output_subset)

    if model_metric_names is not None:
        output_subset = output_subset.union(model_metric_names)

    return output_subset


def required_index_vars_for_outputs(
    output_subset: set[str] | None,
    model: th.nn.Module,
    nb_iso: bool = False,
    return_formula_str: bool = False,
    return_cc_hash: bool = False,
) -> tuple[set[str], set[str], set[str], set[str]]:
    """
    Given desired `output_subset` tokens, determine which index variable arrays are
    required to be present in batch/model outputs.

    `return_formula_str` / `return_cc_hash` are the annotation columns. They are
    addressed by the same index arrays as the P(f) / P(n) tokens but carry no
    probability of their own, so they are requested separately - otherwise
    "annotation without probabilities" is unreachable and the annotation columns
    come out as NaN.

    Returns sets of required names for: batch, node, formula, nb_node indices.
    """
    # If no subset specified, assume all indices may be needed
    if output_subset is None:
        output_subset = set(OUTPUT_SUBSET_OPTIONS)

    # Base mapping from output tokens -> required index vars
    # Note: These names correspond to keys produced by model steps and consumed
    # in `process_batch_to_tabular_row`.
    token_to_idx = {
        "pred_mzs": {"batch": {"pred_batch_idxs"}},
        "pred_ints": {"batch": {"pred_batch_idxs"}},
        "pred_oos_prob": {},
        "pred_rt": {},
        "pred_ccs": {},
        "frag_formula_peak_mzs": {"batch": {"pred_batch_idxs"}},
        # P(f)
        "pred_formula_probs": {
            "batch": {"pred_formula_batch_idxs"},
            "formula": {"pred_formula_formula_idxs"},
        },
        # P(n|f) on joint grid, requires joint idxs and joint batch.
        # The P(f) idxs come along because `batch_filter_by_peak_intensity` builds the
        # joint keep-mask from `pred_formula_formula_idxs`; without them it raises
        # NameError as soon as `min_peak_int > 0`.
        "pred_formula_node_probs": {
            "batch": {"pred_joint_batch_idxs", "pred_formula_batch_idxs"},
            "formula": {"pred_joint_formula_idxs", "pred_formula_formula_idxs"},
            "node": {"pred_joint_node_idxs"},
        },
        # P(f|n) on joint grid
        "pred_node_formula_probs": {
            "batch": {"pred_joint_batch_idxs", "pred_formula_batch_idxs"},
            "formula": {"pred_joint_formula_idxs", "pred_formula_formula_idxs"},
            "node": {"pred_joint_node_idxs"},
        },
        # Joint (formula, node) support with no probability attached: which DAG nodes
        # could explain each surviving formula/peak.
        "pred_joint_formula_idxs": {
            "batch": {"pred_joint_batch_idxs", "pred_formula_batch_idxs"},
            "formula": {"pred_joint_formula_idxs", "pred_formula_formula_idxs"},
            "node": {"pred_joint_node_idxs"},
        },
        "pred_joint_node_idxs": {
            "batch": {"pred_joint_batch_idxs", "pred_formula_batch_idxs"},
            "formula": {"pred_joint_formula_idxs", "pred_formula_formula_idxs"},
            "node": {"pred_joint_node_idxs"},
        },
        # P(n)
        "pred_node_probs": {
            "batch": {"pred_node_batch_idxs"},
            "node": {"pred_node_node_idxs"},
        },
    }

    # Aggregate required idx names across tokens
    req_batch, req_node, req_formula, req_nb_node = set(), set(), set(), set()
    for tok in output_subset:
        m = token_to_idx.get(tok, {})
        req_batch |= m.get("batch", set())
        req_node |= m.get("node", set())
        req_formula |= m.get("formula", set())
        req_nb_node |= m.get("nb_node", set())

    # Annotation columns. `pred_formula_strs` is looked up with
    # `pred_formula_formula_idxs`, `pred_node_ccs` with `pred_node_node_idxs`; both are
    # otherwise only requested by their probability token.
    if return_formula_str:
        req_batch |= {"pred_formula_batch_idxs"}
        req_formula |= {"pred_formula_formula_idxs"}
    if return_cc_hash:
        # cc hashes are per DAG node. Pruning them to the nodes behind a surviving peak
        # needs the joint (formula, node) grid, and the joint keep-mask is in turn built
        # from the P(f) indices - so all three come along. The joint index columns are
        # also what links a node back to a peak downstream.
        req_batch |= {
            "pred_node_batch_idxs",
            "pred_joint_batch_idxs",
            "pred_formula_batch_idxs",
        }
        req_node |= {"pred_node_node_idxs", "pred_joint_node_idxs"}
        req_formula |= {"pred_joint_formula_idxs", "pred_formula_formula_idxs"}

    # Model-specific adjustments: the nb_iso outputs' index vars, for models that have them
    if nb_iso:
        nb_batch, nb_formula, nb_node = get_export_adapter(model).nb_index_vars()
        req_batch |= nb_batch
        req_formula |= nb_formula
        req_nb_node |= nb_node

    return req_batch, req_node, req_formula, req_nb_node


def get_required_config_indices(
    output_subset: set[str] | None,
    model: th.nn.Module,
    nb_iso: bool = False,
    return_formula_str: bool = False,
    return_cc_hash: bool = False,
) -> tuple[set[str], set[str], set[str], set[str]]:
    """
    Public API: Given `output_subset` tokens, return required index variable names
    grouped by category (batch, node, formula, nb_node). Useful for configuring
    dataset/model outputs to ensure all needed indices are present.
    """
    return required_index_vars_for_outputs(
        output_subset, model, nb_iso, return_formula_str, return_cc_hash
    )


_DEFAULT_EXPORT_ADAPTER = ExportAdapter()


def get_export_adapter(model: Any) -> ExportAdapter:
    """The model's export adapter; any other object (or None) gets the defaults."""
    return model if isinstance(model, ExportAdapter) else _DEFAULT_EXPORT_ADAPTER


def get_idx_vars(
    output_subset: set[str] | None,
    model: th.nn.Module,
    nb_iso: bool = False,
    return_formula_str: bool = False,
    return_cc_hash: bool = False,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """
    Get final output keys based on subset selection.

    Args:
            output_subset: Subset of output keys to include (None = all)
            default_output_keys: Default set of output keys

    Returns:
            Final set of output keys
    """
    # Determine which index variables to track
    BATCH_IDX_VARS, NODE_IDX_VARS, FORMULA_IDX_VARS, NB_NODE_IDX_VARS = get_export_adapter(
        model
    ).export_index_vars(nb_iso)

    if output_subset is not None:
        req_batch, req_node, req_formula, req_nb_node = required_index_vars_for_outputs(
            output_subset, model, nb_iso, return_formula_str, return_cc_hash
        )
        BATCH_IDX_VARS = [k for k in BATCH_IDX_VARS if k in req_batch]
        NODE_IDX_VARS = [k for k in NODE_IDX_VARS if k in req_node]
        FORMULA_IDX_VARS = [k for k in FORMULA_IDX_VARS if k in req_formula]
        NB_NODE_IDX_VARS = [k for k in NB_NODE_IDX_VARS if k in req_nb_node]

    return BATCH_IDX_VARS, NODE_IDX_VARS, FORMULA_IDX_VARS, NB_NODE_IDX_VARS


def get_score_types(config_d: dict[str, Any]) -> list[str]:
    """Generate a list of evaluation score types based on the configuration dictionary.

    Args:
        config_d: Configuration dictionary containing 'auxiliary_scores' and 'eval_mz_bin_res'

    Returns:
        List of evaluation score types
    """
    eval_score_types = []
    for score_type in config_d["auxiliary_scores"]:
        if "_hun" in score_type:
            # Hungarian algorithm scores don't need binning
            eval_score_types.append(score_type)
            if config_d["eval_hun_sqrt"]:
                eval_score_types.append(score_type + "_sqrt")
        else:
            # Other scores need binning by m/z resolution
            for bin_res in config_d["eval_mz_bin_res"]:
                eval_score_types.append(f"{score_type}_{bin_res}")
                if config_d["eval_bin_sqrt"]:
                    eval_score_types.append(f"{score_type}_sqrt_{bin_res}")
    return eval_score_types


def get_tabular_columns(
    config_d: dict[str, Any],
    return_formula_str: bool = False,
    return_cc_hash: bool = False,
    output_subset: set[str] | None = None,
    return_rule_provenance: bool = False,
) -> list[str]:
    """
    Get column names for tabular inference output.

    Args:
            config_d: Configuration dictionary
            return_formula_str: Whether output includes the formula-string annotation column
            return_cc_hash: Whether output includes the node cc-hash annotation column
            return_rule_provenance: Whether output includes MASSter's per-node SMARTS rules
            output_subset: Optional subset of output tokens. Checked for unknown tokens but
                    NOT used to narrow the column list - see `selected_column_specs`.

    Returns:
            List of column names for DataFrame
    """
    columns_name = [
        c.name
        for c in selected_column_specs(return_formula_str, return_cc_hash, return_rule_provenance)
    ]
    columns_name.extend(get_score_types(config_d))

    if output_subset is not None:
        missing_cols = [c for c in output_subset if c not in columns_name]
        if len(missing_cols) > 0:
            print(
                f"Warning: {len(missing_cols)} columns in output_subset are not in available columns: {missing_cols}"
            )

    return columns_name
