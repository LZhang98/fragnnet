"""Turning one inference batch into export rows: per-spectrum peak, formula, node and joint
arrays, and the optional peak-intensity filter (split out of ``utils/script_utils.py``).
"""

from dataclasses import dataclass

import numpy as np
import torch as th

from fragnnet.inference.export_columns import (
    get_export_adapter,
    rewire_provenance_enabled,
    selected_column_specs,
)
from fragnnet.utils.formula_utils import formula_delta_to_signed_str
from fragnnet.utils.misc_utils import to_device
from fragnnet.utils.spec_utils import batched_mf1000_normalize, get_split_fn


def process_batch_to_tabular_row(
    batch_input: dict,
    model: th.nn.Module,
    device: th.device,
    evaluate: bool,
    eval_score_types: list,
    BATCH_IDX_VARS: list,
    NODE_IDX_VARS: list,
    FORMULA_IDX_VARS: list,
    min_peak_int: float,
    rebase_indices: bool,
    return_formula_str: bool,
    return_cc_hash: bool,
    output_set: set[str] | None = None,
    filter_stats: dict | None = None,
    return_rule_provenance: bool = False,
) -> list:
    """Process a single batch and return list of data rows.

    Converts raw model outputs and batch inputs into tabular format, handling
    index rebasing, formula/node probability extraction, and optional supplementary
    fields (formula strings, node canonical SMILES).

    Args:
        batch_input: Input batch data dictionary
        model: Model instance for inference
        device: Device to run on
        evaluate: Whether to run evaluation step with ground truth
        eval_score_types: List of evaluation score types to compute
        BATCH_IDX_VARS: Names of batch index variables to include in output
        NODE_IDX_VARS: Names of node index variables to include in output
        FORMULA_IDX_VARS: Names of formula index variables to include in output
        min_peak_int: Minimum peak intensity threshold for filtering
        rebase_indices: Whether to rebase indices to 0-based per-molecule
        return_formula_str: Whether to include formula strings
        return_cc_hash: Whether to include node canonical SMILES
        return_rule_provenance: Whether to include MASSter's per-node SMARTS rule names

    Returns:
        List of data rows for this batch, with one row per spectrum-molecule pair

    Raises:
        ValueError: If `return_rule_provenance` is set for a model other than MASSter.
    """
    # Emitted column order; the same call `get_tabular_columns` makes for the header.
    selected_columns = selected_column_specs(
        return_formula_str, return_cc_hash, return_rule_provenance, rewire_provenance_enabled(model.hparams)
    )
    export_adapter = get_export_adapter(model)
    if return_rule_provenance and not export_adapter.supports_rule_provenance:
        raise ValueError("return_rule_provenance is a MASSter SMARTS annotation")
    spec_ids = batch_input["spec_unique_id"]
    mol_ids = batch_input["mol_id"]
    # Extract string/graph data before moving to device (if they are not tensors)
    frag_formula_str = batch_input.get("frag_formula_str")
    frag_pyg = batch_input.get("frag_pyg")
    if "mol_pyg" in batch_input:
        batch_input["mol_pyg"].mol_ids = mol_ids

    batch_input = to_device(batch_input, device)
    _need_expl = bool(
        {"pred_formula_formula_idxs"} & set(FORMULA_IDX_VARS)
        or {"pred_node_node_idxs"} & set(NODE_IDX_VARS)
    )
    batch_input.update(export_adapter.inference_batch_options("tabular", _need_expl))
    with th.inference_mode():
        # `evaluate` only adds ground-truth scoring; the predicted intensities have the
        # same meaning either way.
        batch_result = model.inference_step(batch_input, metrics="test" if evaluate else None)

    # batch_input['batch_size'] is a tensor -> convert to Python int
    batch_size = int(batch_input["batch_size"].detach().cpu().item())
    # split peaks into batchs
    pred_batch_idxs_batched = batch_result["pred_batch_idxs"]
    pred_mzs_batched = batch_result["pred_mzs"]
    pred_ints_raw = batch_result["pred_ints"]
    # The model may rewrite the peaks it exports (MASSter sums its candidate rows per peak) and
    # build its annotation tables (MASSter), which the steps below read.
    (pred_mzs_batched, pred_ints_raw, pred_batch_idxs_batched), annotation = (
        export_adapter.postprocess_export_peaks(
            batch_input,
            batch_result,
            (pred_mzs_batched, pred_ints_raw, pred_batch_idxs_batched),
            need_annotation=_need_expl,
            return_cc_hash=return_cc_hash,
            return_rule_provenance=return_rule_provenance,
        )
    )
    pred_formula_batch_idxs = (
        batch_result["pred_formula_batch_idxs"]
        if "pred_formula_batch_idxs" in BATCH_IDX_VARS
        else None
    )
    pred_node_batch_idxs = (
        batch_result["pred_node_batch_idxs"] if "pred_node_batch_idxs" in BATCH_IDX_VARS else None
    )
    # `inference_step` always returns L1-normalized intensities, so the base-peak rescale
    # is all that is owed.
    pred_ints_batched = batched_mf1000_normalize(pred_ints_raw, pred_batch_idxs_batched)

    # prepare
    # Index arrays can be requested by an annotation column alone (formula strings, cc
    # hashes). When that happens the matching probability is still cheap to derive, but
    # the caller did not ask for it - dropping it here is what makes "annotation without
    # probabilities" actually smaller on disk instead of just differently named.
    def _want(token: str) -> bool:
        return output_set is None or token in output_set

    pred_oos_probs = (
        batch_result["oos_prob"].detach().cpu()
        if "oos_prob" in batch_result and _want("pred_oos_prob")
        else None
    )
    # values for eval scores
    # this will be np.nan if not evaluating
    eval_scores_d = {}
    if evaluate:
        for eval_score_type in eval_score_types:
            eval_scores_d[eval_score_type] = batch_result[eval_score_type].detach().cpu()

    # joint_batch_idxs maps each entry in the joint prediction arrays (like pred_formula_node_logprobs)
    # which are always None for NEIMS since it doesn't output formula/node predictions.
    frag_formula_cumsizes = batch_input.get("frag_formula_cumsizes")
    frag_num_nodes = batch_input.get("frag_num_nodes")
    joint_batch_idxs = (
        batch_result["pred_joint_batch_idxs"].detach().cpu()
        if "pred_joint_batch_idxs" in BATCH_IDX_VARS
        else None
    )

    # =============================
    # extract formula strings and PyG object if needed
    # =============================
    # Extract formula strings and PyG object if needed
    frag_formula_str = batch_input.get("frag_formula_str", None)
    frag_pyg = batch_input.get("frag_pyg", None)

    # ============================s
    # per-spectrum processing
    # ============================
    # Get cumulative counts for rebasing from batch_input
    # These are tensors of shape (batch_size + 1) containing offsets

    def _prob(token: str, logprob_key: str, axis_present) -> th.Tensor | None:
        """exp() a marginal, unless its axis is absent or the caller did not ask for it.

        An axis can be requested by an annotation column alone (`pred_formula_strs`,
        `pred_node_ccs`), which needs the index arrays but no probability. Dropping the
        probability here is what makes "annotation without probabilities" smaller on disk
        rather than merely differently named.
        """
        if axis_present is None or not _want(token):
            return None
        return th.exp(batch_result[logprob_key])

    arrays = batch_filter_by_peak_intensity(
        PredArrays(
            pred_batch_idxs=pred_batch_idxs_batched,
            pred_ints=pred_ints_batched,
            pred_mzs=pred_mzs_batched,
            frag_formula_peak_mzs=batch_input.get("frag_formula_peak_mzs"),
            frag_formula_peak_idxs=batch_input.get("frag_formula_peak_idxs"),
            # The batch-idx arrays come in raw: the gated locals above decide which
            # *payload* arrays are requested, but the filter needs the raw axis to rebase
            # and the post-call rebind restores the same values the tuple unpack did.
            pred_formula_batch_idxs=batch_result.get("pred_formula_batch_idxs"),
            pred_formula_formula_idxs=(
                None
                if pred_formula_batch_idxs is None
                else batch_result.get("pred_formula_formula_idxs")
            ),
            pred_formula_probs=_prob(
                "pred_formula_probs", "pred_formula_logprobs", pred_formula_batch_idxs
            ),
            pred_joint_batch_idxs=batch_result.get("pred_joint_batch_idxs"),
            pred_joint_formula_idxs=(
                None if joint_batch_idxs is None else batch_result.get("pred_joint_formula_idxs")
            ),
            pred_joint_node_idxs=(
                None if joint_batch_idxs is None else batch_result.get("pred_joint_node_idxs")
            ),
            pred_formula_node_probs=_prob(
                "pred_formula_node_probs", "pred_formula_node_logprobs", joint_batch_idxs
            ),
            pred_node_formula_probs=_prob(
                "pred_node_formula_probs", "pred_node_formula_logprobs", joint_batch_idxs
            ),
            pred_node_batch_idxs=batch_result.get("pred_node_batch_idxs"),
            pred_node_probs=_prob("pred_node_probs", "pred_node_logprobs", pred_node_batch_idxs),
            pred_node_node_idxs=(
                None if pred_node_batch_idxs is None else batch_result.get("pred_node_node_idxs")
            ),
        ),
        min_peak_int,
        frag_formula_cumsizes,
        frag_num_nodes,
        filter_stats=filter_stats,
    )
    pred_batch_idx = arrays.pred_batch_idxs
    spec_ints = arrays.pred_ints
    spec_mzs = arrays.pred_mzs
    frag_formula_peak_mzs = arrays.frag_formula_peak_mzs
    frag_formula_peak_idxs = arrays.frag_formula_peak_idxs
    pred_formula_batch_idxs = arrays.pred_formula_batch_idxs
    pred_formula_formula_idxs = arrays.pred_formula_formula_idxs
    pred_formula_probs = arrays.pred_formula_probs
    joint_batch_idxs = arrays.pred_joint_batch_idxs
    pred_joint_formula_idxs = arrays.pred_joint_formula_idxs
    pred_joint_node_idxs = arrays.pred_joint_node_idxs
    pred_formula_node_probs = arrays.pred_formula_node_probs
    pred_node_formula_probs = arrays.pred_node_formula_probs
    pred_node_batch_idxs = arrays.pred_node_batch_idxs
    pred_node_probs = arrays.pred_node_probs
    pred_node_node_idxs = arrays.pred_node_node_idxs
    property_predictions = {
        name: (
            batch_result[name].detach().cpu().numpy()
            if name in batch_result and _want(name)
            else np.full(batch_size, np.nan)
        )
        for name in ("pred_rt", "pred_ccs")
    }
    numpy_pred_oos_probs = (
        pred_oos_probs.detach().cpu().numpy() if pred_oos_probs is not None else None
    )

    # split based on cleaned results
    if joint_batch_idxs is not None:
        joint_split_fn = get_split_fn(joint_batch_idxs, batch_size=batch_size)
        joint_split_values = [
            None if _ is None else joint_split_fn(_)
            for _ in [
                pred_joint_formula_idxs,
                pred_formula_node_probs,
                pred_joint_node_idxs,
                pred_node_formula_probs,
            ]
        ]

    # formula prob batch idxs, maps each entry in pred_formula_logprobs to its batch idx
    if pred_formula_batch_idxs is not None:
        formula_split_fn = get_split_fn(pred_formula_batch_idxs, batch_size=batch_size)
        formula_split_values = [
            None if _ is None else formula_split_fn(_)
            for _ in [pred_formula_formula_idxs, pred_formula_probs]
        ]

    # node prob batch idxs, maps each entry in pred_node_logprobs to its batch idx
    if pred_node_batch_idxs is not None:
        node_split_fn = get_split_fn(pred_node_batch_idxs, batch_size=batch_size)
        node_split_values = [
            None if _ is None else node_split_fn(_) for _ in [pred_node_probs, pred_node_node_idxs]
        ]
    # specs
    batch_split_fn = get_split_fn(pred_batch_idx, batch_size=batch_size)
    pred_mzs_split = batch_split_fn(spec_mzs)
    pred_ints_split = batch_split_fn(spec_ints)
    frag_formula_peak_idxs_split = (
        None if frag_formula_peak_idxs is None else batch_split_fn(frag_formula_peak_idxs)
    )
    frag_formula_peak_mzs_split = (
        None if frag_formula_peak_mzs is None else batch_split_fn(frag_formula_peak_mzs)
    )
    assert len(pred_mzs_split) == len(pred_ints_split)

    # Per-formula-row precursor delta for the annotation columns, indexed exactly like
    # `frag_formula_str`, so a row's neutral formula and its adduct are always read from the
    # same place. FraGNNet derives it from the DAG's formula rows; MASSter builds it with its
    # annotation tables.
    formula_row_deltas = None
    if return_formula_str or return_cc_hash:
        formula_row_deltas = export_adapter.export_formula_row_deltas(
            batch_input, frag_formula_cumsizes, annotation
        )

    batch_rows = None
    for b_idx in range(batch_size):
        # per-spectrum masks and basic fields
        # b_peak_mask = pred_batch_idxs == b_idx
        b_pred_oos_prob = numpy_pred_oos_probs[b_idx] if pred_oos_probs is not None else np.nan
        b_spec_id = spec_ids[b_idx]
        b_mol_id = mol_ids[b_idx]
        b_spec_mzs = pred_mzs_split[b_idx].detach().cpu()
        b_spec_ints = pred_ints_split[b_idx].detach().cpu()
        b_frag_formula_peak_idxs = (
            frag_formula_peak_idxs_split[b_idx]
            if frag_formula_peak_idxs_split is not None
            else None
        )
        b_frag_formula_peak_mzs = (
            frag_formula_peak_mzs_split[b_idx] if frag_formula_peak_mzs_split is not None else None
        )
        if joint_batch_idxs is not None:
            (
                b_pred_joint_formula_idxs,
                b_pred_formula_node_probs,
                b_pred_joint_node_idxs,
                b_pred_node_formula_probs,
            ) = [_[b_idx] if _ is not None else None for _ in joint_split_values]
        else:
            (
                b_pred_joint_formula_idxs,
                b_pred_formula_node_probs,
                b_pred_joint_node_idxs,
                b_pred_node_formula_probs,
            ) = [None for _ in range(4)]
        # Filter formula probability data (P(f)) for the current spectrum/molecule (b_idx)
        # and sort formulas by probability descending
        if pred_formula_batch_idxs is not None:
            b_pred_formula_formula_idxs, b_pred_formula_probs = [
                _[b_idx] if _ is not None else None for _ in formula_split_values
            ]
        else:
            b_pred_formula_formula_idxs, b_pred_formula_probs = [None for _ in range(2)]

        # node probabilities P(n) for the current spectrum/molecule (b_idx)
        if pred_node_batch_idxs is not None:
            b_pred_node_probs, b_pred_node_node_idxs = [
                _[b_idx] if _ is not None else None for _ in node_split_values
            ]
        else:
            b_pred_node_probs, b_pred_node_node_idxs = [None for _ in range(2)]

        # Store original indices before rebasing (for lookups)
        orig_formula_idxs_for_lookup = (
            b_pred_formula_formula_idxs.clone() if b_pred_formula_formula_idxs is not None else None
        )
        orig_node_idxs_for_lookup = (
            b_pred_node_node_idxs.clone() if b_pred_node_node_idxs is not None else None
        )

        if rebase_indices:  # possibly can be put into batch, but Idk what's the popurse here.
            # Rebase formula indices to 0-based contiguous
            if b_pred_formula_formula_idxs is not None and len(b_pred_formula_formula_idxs) > 0:
                unique_f_idxs, inverse = th.unique(
                    b_pred_formula_formula_idxs, sorted=True, return_inverse=True
                )

                # Apply rebasing to formula arrays
                b_pred_formula_formula_idxs = inverse
                if b_pred_joint_formula_idxs is not None:
                    b_pred_joint_formula_idxs = th.searchsorted(
                        unique_f_idxs, b_pred_joint_formula_idxs
                    )
                if b_frag_formula_peak_idxs is not None:
                    b_frag_formula_peak_idxs = th.searchsorted(
                        unique_f_idxs, b_frag_formula_peak_idxs
                    )

            # Rebase node indices to 0-based contiguous
            if b_pred_node_node_idxs is not None and len(b_pred_node_node_idxs) > 0:
                unique_n_idxs, inverse = th.unique(
                    b_pred_node_node_idxs, sorted=True, return_inverse=True
                )
                b_pred_node_node_idxs = inverse
                if b_pred_joint_node_idxs is not None:
                    b_pred_joint_node_idxs = th.searchsorted(unique_n_idxs, b_pred_joint_node_idxs)

        # assemble row, keyed by column name; ordered against COLUMN_SPECS below so the
        # row can never drift out of step with `get_tabular_columns`'s header.
        # An absent array becomes NaN rather than being dropped - see `selected_column_specs`.
        row_d = {
            "spec_id": b_spec_id.item(),
            "mol_id": b_mol_id,
            "pred_mzs": b_spec_mzs,
            "pred_ints": b_spec_ints,
            "pred_oos_prob": b_pred_oos_prob,
            **{name: values[b_idx] for name, values in property_predictions.items()},
            "frag_formula_peak_idxs": b_frag_formula_peak_idxs,
            "frag_formula_peak_mzs": b_frag_formula_peak_mzs,
            "pred_formula_formula_idxs": b_pred_formula_formula_idxs,
            "pred_formula_probs": b_pred_formula_probs,
            "pred_joint_formula_idxs": b_pred_joint_formula_idxs,
            "pred_formula_node_probs": b_pred_formula_node_probs,
            "pred_joint_node_idxs": b_pred_joint_node_idxs,
            "pred_node_formula_probs": b_pred_node_formula_probs,
            "pred_node_probs": b_pred_node_probs,
            "pred_node_node_idxs": b_pred_node_node_idxs,
            # Add rebase flag: True if indices were rebased, False otherwise
            "index_rebased": bool(rebase_indices),
        }

        # Formula-axis annotation. `pred_formula_strs` and `pred_formula_adducts` address the
        # batch-wide formula arrays with the same index, so it is resolved once for both.
        global_f_idxs = None
        if (return_formula_str or return_cc_hash) and frag_formula_cumsizes is not None:
            f_offset = frag_formula_cumsizes[b_idx].item()
            if rebase_indices and orig_formula_idxs_for_lookup is not None:
                # Use original (pre-rebase) indices
                global_f_idxs = orig_formula_idxs_for_lookup + f_offset
            elif not rebase_indices and b_pred_formula_formula_idxs is not None:
                # Use current indices
                global_f_idxs = b_pred_formula_formula_idxs + f_offset
        np_f_idxs = None if global_f_idxs is None else global_f_idxs.cpu().numpy()

        if return_formula_str:
            row_d["pred_formula_strs"] = (
                frag_formula_str[np_f_idxs].tolist()
                if np_f_idxs is not None and frag_formula_str is not None
                else np.nan
            )

        if return_formula_str or return_cc_hash:
            # Signed rendering ("-H", "+Na"), so a peak reads as "C6H4O2 -H" against the
            # neutral formula string, and a cc bitmask as that fragment plus this adduct.
            if np_f_idxs is not None:
                row_d["pred_formula_adducts"] = (
                    [formula_delta_to_signed_str(d) for d in formula_row_deltas[np_f_idxs]]
                    if formula_row_deltas is not None
                    else [np.nan] * len(np_f_idxs)
                )
            else:
                row_d["pred_formula_adducts"] = np.nan

        if return_cc_hash or return_rule_provenance:
            b_node_idxs = orig_node_idxs_for_lookup if rebase_indices else b_pred_node_node_idxs
        if return_cc_hash:
            row_d["pred_node_ccs"] = export_adapter.export_node_ccs(
                frag_pyg, b_idx, b_node_idxs, annotation
            )

        if return_rule_provenance:
            row_d["pred_node_rules"] = export_adapter.export_node_rules(
                b_idx, b_node_idxs, annotation
            )
            if rewire_provenance_enabled(model.hparams):
                row_d["pred_node_rewire_rules"] = export_adapter.export_node_rewire_rules(
                    b_idx, b_node_idxs, annotation
                )

        # Move every exported tensor to the CPU here. Only some of the arrays above are
        # explicitly moved, so without this the saved predictions mix CPU and CUDA tensors
        # and the pickle cannot be loaded on a host without that CUDA device.
        data_row = []
        for c in selected_columns:
            value = row_d[c.name]
            if value is None:
                value = np.nan
            elif th.is_tensor(value):
                value = value.detach().cpu()
            data_row.append(value)
        # Append evaluation scores
        for eval_score_type in eval_score_types:
            if evaluate:
                data_row.append(eval_scores_d[eval_score_type][b_idx].item())
            else:
                data_row.append(np.nan)

        # Append the completed data row for this spectrum/molecule
        if batch_rows is None:
            batch_rows = [[_] for _ in data_row]
        else:
            for d, v in zip(batch_rows, data_row):
                d.append(v)
    return batch_rows


@dataclass
class PredArrays:
    """The batch-level prediction arrays the exporters filter, split and write.

    Grouped by the axis each array is indexed along, because that is what decides how it
    gets filtered and how it is written out:

    - **peak** (P): one entry per predicted spectrum peak.
    - **formula** (F): one entry per predicted formula.
    - **joint** (J): one entry per (formula, node) cell of the joint grid.
    - **node** (N): one entry per DAG node.

    Every field except the peak axis is optional: an array is None when the run did not
    request the output that needs it, and each consumer must tolerate that. The four
    lengths are unrelated, so no family's offsets or masks transfer to another.
    """

    # peak axis (P)
    pred_batch_idxs: th.Tensor
    pred_ints: th.Tensor
    pred_mzs: th.Tensor
    frag_formula_peak_mzs: th.Tensor | None = None
    frag_formula_peak_idxs: th.Tensor | None = None
    # formula axis (F)
    pred_formula_batch_idxs: th.Tensor | None = None
    pred_formula_formula_idxs: th.Tensor | None = None
    pred_formula_probs: th.Tensor | None = None
    # joint axis (J)
    pred_joint_batch_idxs: th.Tensor | None = None
    pred_joint_formula_idxs: th.Tensor | None = None
    pred_joint_node_idxs: th.Tensor | None = None
    pred_formula_node_probs: th.Tensor | None = None
    pred_node_formula_probs: th.Tensor | None = None
    # node axis (N)
    pred_node_batch_idxs: th.Tensor | None = None
    pred_node_probs: th.Tensor | None = None
    pred_node_node_idxs: th.Tensor | None = None


def batch_filter_by_peak_intensity(
    arrays: PredArrays,
    min_peak_int: float,
    frag_formula_cumsizes: th.Tensor,
    frag_num_nodes: th.Tensor,
    filter_stats: dict | None = None,
) -> PredArrays:
    """Drop low-intensity peaks, cascade to formulas/nodes, and rebase indices per molecule.

    Filtering cascades along the DAG: peaks below `min_peak_int` go first, then the
    formulas only those peaks explained, then the joint cells of those formulas, then the
    nodes left with no joint cell. Without `frag_formula_peak_idxs` (models with no DAG
    peak -> formula map, e.g. MASSter) only the peaks are filtered, and any formula,
    joint and node arrays pass through whole. Rebasing runs unconditionally afterwards, including
    when no filtering happened, because the caller's `pred_formula_strs` / `pred_node_ccs`
    lookups index a single molecule's table and so need per-molecule indices.

    Args:
        arrays: Batch-level prediction arrays, with batch-global indices.
        min_peak_int: Intensity threshold on the same 0-1000 scale the exporter emits;
            <= 0 skips filtering but still rebases.
        frag_formula_cumsizes: Per-spectrum formula offsets, shape (batch_size + 1,).
        frag_num_nodes: Per-spectrum node offsets, shape (batch_size + 1,).
        filter_stats: Optional mutable dict accumulating n_peaks_before/n_peaks_after/
            n_inst_filtered/n_inst_total across batches.

    Returns:
        A new `PredArrays` with the surviving entries and per-molecule indices. Fields
        that came in None stay None.
    """
    pred_batch_idxs = arrays.pred_batch_idxs
    pred_ints = arrays.pred_ints
    pred_mzs = arrays.pred_mzs
    frag_formula_peak_mzs = arrays.frag_formula_peak_mzs
    frag_formula_peak_idxs = arrays.frag_formula_peak_idxs
    pred_formula_batch_idxs = arrays.pred_formula_batch_idxs
    pred_formula_formula_idxs = arrays.pred_formula_formula_idxs
    pred_formula_probs = arrays.pred_formula_probs
    pred_joint_batch_idxs = arrays.pred_joint_batch_idxs
    pred_joint_formula_idxs = arrays.pred_joint_formula_idxs
    pred_joint_node_idxs = arrays.pred_joint_node_idxs
    pred_formula_node_probs = arrays.pred_formula_node_probs
    pred_node_formula_probs = arrays.pred_node_formula_probs
    pred_node_batch_idxs = arrays.pred_node_batch_idxs
    pred_node_probs = arrays.pred_node_probs
    pred_node_node_idxs = arrays.pred_node_node_idxs

    # No early return when there is nothing to filter: the index arrays still have to be
    # rebased from batch-global to per-molecule below, or the `pred_formula_strs` /
    # `pred_node_ccs` lookups index a single molecule's table with a batch-wide index
    # (out of bounds for every spectrum after the first).
    valid_pred_batch_idx = pred_batch_idxs
    valid_f_idxs = None
    # Stay None unless the peak filter actually builds them, so a missing P(f) index
    # array can never leave them dangling for the joint/node steps below.
    keep_mask_f = keep_mask_j = None

    # 1. Filter the peaks themselves by intensity. This needs nothing but the peak axis,
    # so it runs whenever a threshold is set. It used to be gated on
    # frag_formula_peak_idxs as well, which made min_peak_int a silent no-op for every
    # model without a DAG peak -> formula map (MASSter exports kept all candidates).
    if min_peak_int > 0:
        n_peaks_before = len(pred_ints)
        valid_peak_mask = pred_ints > min_peak_int
        pred_ints = pred_ints[valid_peak_mask]
        pred_mzs = pred_mzs[valid_peak_mask]
        valid_pred_batch_idx = pred_batch_idxs[valid_peak_mask]
        if frag_formula_peak_mzs is not None:  # Only if frag_formula_peak_mzs is available
            frag_formula_peak_mzs = frag_formula_peak_mzs[valid_peak_mask]
        if frag_formula_peak_idxs is not None:
            frag_formula_peak_idxs = frag_formula_peak_idxs[valid_peak_mask]

        if filter_stats is not None:
            n_peaks_after = int(valid_peak_mask.sum().item())
            batch_size_local = (
                int(pred_batch_idxs.max().item()) + 1 if len(pred_batch_idxs) > 0 else 0
            )
            before_counts = th.bincount(pred_batch_idxs, minlength=batch_size_local)
            after_counts = th.bincount(valid_pred_batch_idx, minlength=batch_size_local)
            filter_stats["n_peaks_before"] = filter_stats.get("n_peaks_before", 0) + n_peaks_before
            filter_stats["n_peaks_after"] = filter_stats.get("n_peaks_after", 0) + n_peaks_after
            filter_stats["n_inst_filtered"] = filter_stats.get("n_inst_filtered", 0) + int(
                (before_counts != after_counts).sum().item()
            )
            filter_stats["n_inst_total"] = filter_stats.get("n_inst_total", 0) + batch_size_local

    # 2. Cascade to the formulas the surviving peaks explain. This needs
    # frag_formula_peak_idxs, which maps each peak to a local formula index; without it
    # the formula, joint and node arrays pass through whole.
    if min_peak_int > 0 and frag_formula_peak_idxs is not None:
        # Offsets come from the unfiltered peak axis, as the filtered one has gaps.
        pred_peak_in_each_batch = th.bincount(pred_batch_idxs)
        pred_peak_in_each_batch_cumsum = th.cumsum(pred_peak_in_each_batch, 0)
        pred_peak_in_each_batch_cumsum_pad0 = th.cat(
            [pred_peak_in_each_batch_cumsum.new_zeros(1), pred_peak_in_each_batch_cumsum], 0
        )
        valid_local_f_idxs = frag_formula_peak_idxs  # here the idx is local not global

        # Map local formula indices (0..N_formulas_in_mol) to global formula indices (if available)
        # b_pred_formula_formula_idxs contains the global indices for the formulas in this molecule
        if pred_formula_formula_idxs is not None:
            # Ensure indices are valid (should be within bounds if data is consistent)
            # We use .long() to ensure they are treated as indices
            # Note: valid_local_f_idxs are indices into the formula array for this molecule
            #
            # The offset turning a per-molecule formula index into a row of the
            # batch-global formula arrays must count FORMULA ROWS per spectrum, not peaks.
            # The model trims one null formula per molecule, so peaks-per-spectrum and
            # formula-rows-per-spectrum drift apart by one for every spectrum; using the
            # peak offset makes each spectrum after the first read another molecule's
            # formulas. Spectrum 0 has offset 0 either way, which is why batch_size=1
            # hid this.
            formula_rows_in_each_batch = th.bincount(
                pred_formula_batch_idxs, minlength=len(pred_peak_in_each_batch)
            )
            formula_row_cumsum_pad0 = th.cat(
                [
                    formula_rows_in_each_batch.new_zeros(1),
                    th.cumsum(formula_rows_in_each_batch, 0),
                ],
                0,
            )
            valid_f_idxs = pred_formula_formula_idxs[
                valid_local_f_idxs.long() + formula_row_cumsum_pad0[valid_pred_batch_idx]
            ]
            valid_f_idxs = th.unique(valid_f_idxs)
            mask = th.zeros(
                pred_formula_formula_idxs.max().item() + 1,
                dtype=th.bool,
                device=valid_f_idxs.device,
            )
            mask[valid_f_idxs] = True
            keep_mask_f = mask[pred_formula_formula_idxs]
            # Guarded explicitly: `mask[None]` would not raise, it would add an axis and
            # hand back a (1, F) tensor that reads as "a mask exists" everywhere below.
            if pred_joint_formula_idxs is not None:
                keep_mask_j = mask[pred_joint_formula_idxs]
        else:
            # Fallback if formula indices are missing (unlikely to be useful but prevents crash)
            valid_f_idxs = th.unique(
                valid_local_f_idxs.long()
                + pred_peak_in_each_batch_cumsum_pad0[valid_pred_batch_idx]
            )
    # 3. Apply filtering to all formula-related tensors.
    # Skipped wholesale when step 2 did not run (min_peak_int <= 0, or no
    # frag_formula_peak_idxs to map peaks to formulas with); the rebase below still runs.

    # Filter formula predictions (P(f))
    if keep_mask_f is not None and pred_formula_formula_idxs is not None:
        # keep_mask_f = th.isin(pred_formula_formula_idxs, valid_f_idxs) # speed up
        pred_formula_batch_idxs = pred_formula_batch_idxs[keep_mask_f]
        pred_formula_formula_idxs = pred_formula_formula_idxs[keep_mask_f]
        if pred_formula_probs is not None:
            pred_formula_probs = pred_formula_probs[keep_mask_f]

    # Filter joint predictions (P(f,n), P(n|f), P(f|n)) down to the surviving formulas.
    # There used to be a second, node-based term on this mask, but it was vacuous:
    # `pred_node_node_idxs` is `arange(num_nodes)`, so every joint entry passed it. (It
    # was also an IndexError waiting to happen, since the mask was sized by the joint's
    # largest node index but written at every node index.) Node pruning now runs the
    # other way round, below.
    if keep_mask_j is not None and pred_joint_formula_idxs is not None:
        pred_joint_batch_idxs = pred_joint_batch_idxs[keep_mask_j]
        pred_joint_formula_idxs = pred_joint_formula_idxs[keep_mask_j]
        if pred_formula_node_probs is not None:
            pred_formula_node_probs = pred_formula_node_probs[keep_mask_j]
        if pred_joint_node_idxs is not None:
            pred_joint_node_idxs = pred_joint_node_idxs[keep_mask_j]
        if pred_node_formula_probs is not None:
            pred_node_formula_probs = pred_node_formula_probs[keep_mask_j]

    # Filter node predictions (P(n)) to the nodes that can still explain a surviving
    # peak, i.e. the ones left in the filtered joint grid. Before this, P(f) was pruned
    # by intensity but P(n) was not, so `pred_node_probs` / `pred_node_node_idxs` - and
    # the `pred_node_ccs` annotation indexed by them - covered every node of the DAG
    # however aggressive `min_peak_int` was.
    #
    # Without the joint there is no node -> formula map available here, so the node
    # arrays pass through whole. `required_index_vars_for_outputs` requests the joint
    # whenever cc-hash annotation is asked for, which keeps that fallback off the
    # inference path.
    if (
        keep_mask_j is not None
        and pred_node_node_idxs is not None
        and pred_joint_node_idxs is not None
        and len(pred_node_node_idxs) > 0
    ):
        keep_mask_n = th.zeros(
            int(pred_node_node_idxs.max().item()) + 1,
            dtype=th.bool,
            device=pred_node_node_idxs.device,
        )
        keep_mask_n[pred_joint_node_idxs] = True
        keep_mask_n = keep_mask_n[pred_node_node_idxs]
        pred_node_batch_idxs = pred_node_batch_idxs[keep_mask_n]
        pred_node_node_idxs = pred_node_node_idxs[keep_mask_n]
        if pred_node_probs is not None:
            pred_node_probs = pred_node_probs[keep_mask_n]
    # rebase
    if pred_joint_batch_idxs is not None:
        if pred_joint_formula_idxs is not None:
            pred_joint_formula_idxs = (
                pred_joint_formula_idxs - frag_formula_cumsizes[pred_joint_batch_idxs]
            )
        if pred_joint_node_idxs is not None:
            pred_joint_node_idxs = pred_joint_node_idxs - frag_num_nodes[pred_joint_batch_idxs]
    if pred_node_batch_idxs is not None and pred_node_node_idxs is not None:
        pred_node_node_idxs = pred_node_node_idxs - frag_num_nodes[pred_node_batch_idxs]
    if pred_formula_batch_idxs is not None and pred_formula_formula_idxs is not None:
        pred_formula_formula_idxs = (
            pred_formula_formula_idxs - frag_formula_cumsizes[pred_formula_batch_idxs]
        )

    return PredArrays(
        pred_batch_idxs=valid_pred_batch_idx,
        pred_ints=pred_ints,
        pred_mzs=pred_mzs,
        frag_formula_peak_mzs=frag_formula_peak_mzs,
        frag_formula_peak_idxs=frag_formula_peak_idxs,
        pred_formula_batch_idxs=pred_formula_batch_idxs,
        pred_formula_formula_idxs=pred_formula_formula_idxs,
        pred_formula_probs=pred_formula_probs,
        pred_joint_batch_idxs=pred_joint_batch_idxs,
        pred_joint_formula_idxs=pred_joint_formula_idxs,
        pred_joint_node_idxs=pred_joint_node_idxs,
        pred_formula_node_probs=pred_formula_node_probs,
        pred_node_formula_probs=pred_node_formula_probs,
        pred_node_batch_idxs=pred_node_batch_idxs,
        pred_node_probs=pred_node_probs,
        pred_node_node_idxs=pred_node_node_idxs,
    )
