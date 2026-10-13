"""The inference exporters: run a model over a dataloader and write predictions as pickles
(``run_inference_pkl``), a table (``run_inference_tabular`` and the streaming
``run_inference_tabular_iter``) or HDF5 (``run_inference_hdf5``).
"""

import logging
from collections.abc import Iterator
from itertools import chain
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch as th
from tqdm import tqdm

from fragnnet.inference.batch_rows import process_batch_to_tabular_row
from fragnnet.inference.export_columns import (
    COLUMN_SPECS,
    get_export_adapter,
    get_idx_vars,
    get_score_types,
    get_tabular_columns,
)
from fragnnet.utils.misc_utils import (
    flatten_lol,
    to_device,
)
from fragnnet.utils.spec_utils import batched_mf1000_normalize

logger = logging.getLogger(__name__)


# ============================================================================
# Inference Configuration and Setup Utilities
# ============================================================================
#
# These functions provide modular utilities for inference evaluation scripts.
# They complement load_model_and_init_config() but serve different use cases:
#
# ============================================================================


# -----------------------------------------------------------------------------
# The export schema
# -----------------------------------------------------------------------------


# Rows per HDF5 chunk. Compression works per chunk, so a chunk large enough to hold long
# runs of similar values is what turns the index columns from 8 bytes per small integer
# into ~0.5. Measured on a real index column: h5py's auto-chunk (1024 rows) gives 9.3 MB,
# this gives 5.1 MB, uncompressed 80 MB.
_HDF5_CHUNK_LEN = 1 << 16

# HDF5 index-pair prefix per ragged family. Readers already expect these exact spellings
# (start_points, node_start_points, formula_start_points, joint_start_points).
RAGGED_FAMILY_PREFIX = {"peak": "", "node": "node_", "formula": "formula_", "joint": "joint_"}

# Probability tokens are exp() of the corresponding raw log-prob model output key.
# The raw FraGNNet output (fragnnet_model out_d) is emitted in log space, while the
# `output_subset` vocabulary above uses probability names. `pred_ints` is handled
# separately because it also needs the intensity untransform. This mirrors the
# derivations in `process_batch_to_tabular_row`.
_PROB_TOKEN_TO_LOGPROB_KEY = {
    "pred_oos_prob": "pred_oos_logprobs",
    "pred_formula_probs": "pred_formula_logprobs",
    "pred_formula_node_probs": "pred_formula_node_logprobs",
    "pred_node_formula_probs": "pred_node_formula_logprobs",
    "pred_node_probs": "pred_node_logprobs",
}


def run_inference_pkl(
    model: th.nn.Module,
    dataloader: th.utils.data.DataLoader,
    device: th.device,
    batch_cutoff: int = -1,
    nb_iso: bool = False,
    output_set: set[str] | None = None,
    evaluate: bool = True,
) -> dict[str, Any]:
    """Run inference and return pickled dictionary of results.

    Executes inference on all (or a subset of) batches, with optional index rebatching
    to consolidate results across batches into a single output dictionary.

    Args:
        model: Model instance for inference
        dataloader: DataLoader providing batches
        device: Device to run inference on
        batch_cutoff: Maximum number of batches to process (-1 for all)
        nb_iso: Whether model uses neighbor isotope predictions
        output_set: Subset of output keys to include (None for all)
        evaluate: Whether to compute evaluation metrics using ground truth

    Returns:
        Dictionary with concatenated inference results from all batches
    """
    # setup accumulators and index vars
    cum_num_datapoints = 0
    cum_num_nodes = 0
    cum_num_formulas = 0
    cum_num_nb_nodes = 0

    export_adapter = get_export_adapter(model)
    BATCH_IDX_VARS, NODE_IDX_VARS, FORMULA_IDX_VARS, NB_NODE_IDX_VARS = get_idx_vars(
        output_set,
        model,
        nb_iso,
    )

    # Setup for evaluation mode (return dict)
    vals = {}
    # Common setup
    print(f"\n>>> Starting inference on {len(dataloader)} batches...")
    if model.training:
        raise RuntimeError(f"> Warning: model {model} is in training mode!")

    # Run inference loops
    total_batches = len(dataloader) if batch_cutoff < 0 else min(len(dataloader), batch_cutoff)
    with th.inference_mode():
        for batch_idx, batch_input in tqdm(
            enumerate(dataloader), total=total_batches, desc="Running inference"
        ):
            # Early exit for batch cutoff
            if batch_idx >= total_batches:
                break

            # Move batch to device
            batch_input = to_device(batch_input, device, non_blocking=False)

            # Enable explainability tensors only when formula/node idx vars are needed
            _need_expl = bool(
                {"pred_formula_formula_idxs"} & set(FORMULA_IDX_VARS)
                or {"pred_node_node_idxs"} & set(NODE_IDX_VARS)
            )
            batch_input.update(export_adapter.inference_batch_options("pkl", _need_expl))

            # Run inference. `evaluate` only adds ground-truth scoring; the predicted
            # intensities have the same meaning either way.
            b_output = model.inference_step(batch_input, metrics="test" if evaluate else None)

            b_output = to_device(b_output, "cpu", non_blocking=False)
            if output_set is not None:
                # Translate user-facing `output_subset` tokens into concrete model
                # output columns. The raw model output uses log-prob keys
                # (e.g. `pred_node_logprobs`), while the token vocabulary uses
                # probability names (e.g. `pred_node_probs`). Probability tokens are
                # exp() of their logprob key; `pred_ints` arrives already untransformed
                # and L1-normalized from `inference_step`, so it only needs rescaling.
                # Index variables are always retained so concatenated columns can be regrouped
                # per-spectrum / per-node downstream.
                all_idx_vars = set(
                    BATCH_IDX_VARS + NODE_IDX_VARS + FORMULA_IDX_VARS + NB_NODE_IDX_VARS
                )
                derived = {k: b_output[k] for k in all_idx_vars if b_output.get(k) is not None}
                for token in output_set:
                    if token == "pred_ints":
                        # `inference_step` always returns L1-normalized intensities, so
                        # the base-peak rescale is all that is owed.
                        derived["pred_ints"] = batched_mf1000_normalize(
                            b_output["pred_ints"], b_output["pred_batch_idxs"]
                        )
                    elif token in _PROB_TOKEN_TO_LOGPROB_KEY:
                        key = _PROB_TOKEN_TO_LOGPROB_KEY[token]
                        if b_output.get(key) is None:
                            raise KeyError(
                                f"output token {token!r} needs model output "
                                f"{key!r}, which is absent/None"
                            )
                        derived[token] = th.exp(b_output[key])
                    elif b_output.get(token) is not None:
                        derived[token] = b_output[token]
                    else:
                        raise KeyError(
                            f"output token {token!r} is not produced by run_inference_pkl; "
                            "use output_format=hdf5 for this token"
                        )
                b_output = derived

            # ============================================================
            # EVALUATION MODE: Collect all outputs with index rebatching
            # ============================================================
            # Calculate batch counts for index rebatching. Any per-batch index that
            # ranges over spectra gives the datapoint count, so fall back through them
            # (e.g. a node-only output_subset has no `pred_batch_idxs` but still needs
            # `pred_node_batch_idxs` rebatched across batches).
            b_num_datapoints = 0
            for _batch_key in (
                "pred_batch_idxs",
                "true_batch_idxs",
                "pred_node_batch_idxs",
                "pred_formula_batch_idxs",
                "pred_joint_batch_idxs",
            ):
                v = b_output.get(_batch_key)
                if v is not None and v.numel() > 0:
                    b_num_datapoints = int(v.max()) + 1
                    break

            # Only a model that emits the annotation axes itself (FraGNNet) has these here.
            if b_output.get("pred_node_node_idxs") is not None:
                b_num_nodes = b_output["pred_node_node_idxs"].max() + 1
            else:
                b_num_nodes = 0

            if b_output.get("pred_formula_formula_idxs") is not None:
                b_num_formulas = b_output["pred_formula_formula_idxs"].max() + 1
            else:
                b_num_formulas = 0

            if nb_iso and b_output.get("pred_nb_node_node_idxs") is not None:
                b_num_nb_nodes = b_output["pred_nb_node_node_idxs"].max() + 1
            else:
                b_num_nb_nodes = 0

            # Rebatch indices
            for k in BATCH_IDX_VARS:
                if k in b_output:
                    b_output[k] = b_output[k] + cum_num_datapoints
            for k in NODE_IDX_VARS:
                if k in b_output:
                    b_output[k] = b_output[k] + cum_num_nodes
            for k in FORMULA_IDX_VARS:
                if k in b_output:
                    b_output[k] = b_output[k] + cum_num_formulas
            for k in NB_NODE_IDX_VARS:
                if k in b_output:
                    b_output[k] = b_output[k] + cum_num_nb_nodes

            # Update cumulative counts
            cum_num_datapoints += b_num_datapoints
            cum_num_nodes += b_num_nodes
            cum_num_formulas += b_num_formulas
            cum_num_nb_nodes += b_num_nb_nodes

            # Collect outputs
            for k in b_output:
                if b_output[k] is None:
                    continue
                if k not in vals:
                    assert batch_idx == 0
                    vals[k] = [b_output[k]]
                else:
                    assert batch_idx > 0
                    vals[k].append(b_output[k])

    # Consolidate collected outputs
    print(f"> num_datapoints = {cum_num_datapoints}")
    print(f"> num_nodes = {cum_num_nodes}")
    print(f"> num_formulas = {cum_num_formulas}")
    print(f"> num_nb_nodes = {cum_num_nb_nodes}")

    for k in vals:
        if isinstance(vals[k][0], list):
            vals[k] = flatten_lol(vals[k])
        elif isinstance(vals[k][0], th.Tensor):
            if vals[k][0].dim() == 0:
                vals[k] = th.stack(vals[k], dim=0)
            else:
                vals[k] = th.cat(vals[k], dim=0)
        elif isinstance(vals[k][0], np.ndarray):
            if vals[k][0].ndim == 0:
                vals[k] = np.stack(vals[k], axis=0)
            else:
                vals[k] = np.concatenate(vals[k], axis=0)
        else:
            raise ValueError(f"unexpected type {type(vals[k][0])}")

    return vals


def run_inference_tabular(
    model: th.nn.Module,
    dataloader: th.utils.data.DataLoader,
    device: th.device,
    batch_cutoff: int = -1,
    evaluate: bool = False,
    output_set: set[str] | None = None,
    nb_iso: bool = False,
    min_peak_int: float = 0.0,
    return_formula_str: bool = False,
    return_cc_hash: bool = False,
    rebase_indices: bool = True,
    return_rule_provenance: bool = False,
) -> pd.DataFrame:
    """
    Run inference and return all results as a single DataFrame.

    This is a convenience wrapper around run_inference_tabular_iter that accumulates
    all batch results into a single DataFrame.

    Args:
            model: Model instance
            dataloader: DataLoader with batches
            device: Device to run inference on
            batch_cutoff: Max batches to process (-1 = all)
            evaluate: Whether to run evaluation step
            output_set: Subset of outputs to include
            nb_iso: Whether model uses neighbor isotope prediction
            min_peak_int: Minimum peak intensity threshold
            return_formula_and_cc: Whether to return formula strings and CCs
            rebase_indices: Whether to rebase indices

    Returns:
            DataFrame with all inference results
    """
    # Get column names from config
    columns_name = get_tabular_columns(
        model.hparams,
        return_formula_str,
        return_cc_hash,
        output_subset=output_set,
        return_rule_provenance=return_rule_provenance,
    )

    # Accumulate all batch results into a column-keyed dict
    preds_l: dict[str, list] = {}
    for batch_rows in run_inference_tabular_iter(
        model=model,
        dataloader=dataloader,
        device=device,
        batch_cutoff=batch_cutoff,
        evaluate=evaluate,
        output_set=output_set,
        nb_iso=nb_iso,
        min_peak_int=min_peak_int,
        return_formula_str=return_formula_str,
        return_cc_hash=return_cc_hash,
        rebase_indices=rebase_indices,
        return_rule_provenance=return_rule_provenance,
    ):
        assert len(batch_rows) == len(
            columns_name
        ), f"batch_rows has {len(batch_rows)} columns, expected {len(columns_name)}"
        for k, v in zip(columns_name, batch_rows):
            preds_l.setdefault(k, []).extend(v)

    pred_df = pd.DataFrame(preds_l)
    return pred_df


def run_inference_hdf5(
    model: th.nn.Module,
    dataloader: th.utils.data.DataLoader,
    device: th.device,
    batch_cutoff: int = -1,
    evaluate: bool = False,
    output_set: set[str] | None = None,
    nb_iso: bool = False,
    min_peak_int: float = 0.0,
    return_formula_str: bool = False,
    return_cc_hash: bool = False,
    rebase_indices: bool = True,
    chunk_size: int = 10000,
    return_rule_provenance: bool = False,
    save_path: str = "",
    drop_none_cols: bool = True,
    compression: str | None = "gzip",
    compression_opts: int | None = 4,
) -> pd.DataFrame:
    """
    Stream inference results to an HDF5 file and return the number of spectra written.

    Unlike run_inference_pkl, results are flushed every `chunk_size` spectra rather than
    held in memory, so this is the writer to use for large eval splits.

    Layout: scalar columns (spec_id, mol_id, scores, ...) hold one value per spectrum and
    are indexed directly by spectrum. Ragged columns are stored flat, concatenated across
    spectra, and sliced with a per-spectrum index pair:

        family    index pair                                 columns
        peak      start_points     / end_points              pred_mzs, pred_ints,
                                                             frag_formula_peak_{idxs,mzs}
        node      node_start_points    / node_end_points     pred_node_probs,
                                                             pred_node_node_idxs,
                                                             pred_node_ccs (2-D: hash pair)
        formula   formula_start_points / formula_end_points  pred_formula_probs,
                                                             pred_formula_formula_idxs,
                                                             pred_formula_strs (vlen str)
        joint     joint_start_points   / joint_end_points    pred_joint_{formula,node}_idxs,
                                                             pred_formula_node_probs,
                                                             pred_node_formula_probs

    So spectrum i's P(n) is pred_node_probs[node_start_points[i]:node_end_points[i]]. The
    four families have independent lengths — a spectrum's peak, node, formula and joint
    counts are unrelated — so each carries its own index pair. A family whose columns were
    not requested writes no datasets at all.

    Args:
            model: Model instance
            dataloader: DataLoader with batches
            device: Device to run inference on
            batch_cutoff: Max batches to process (-1 = all)
            evaluate: Whether to run evaluation step
            output_set: Subset of outputs to include
            nb_iso: Whether model uses neighbor isotope prediction
            min_peak_int: Minimum peak intensity threshold
            return_formula_and_cc: Whether to return formula strings and CCs
            rebase_indices: Whether to rebase indices
            drop_none_cols: Skip scalar columns that hold nothing but NaN placeholders
            compression: HDF5 filter for every dataset, or None to store raw. gzip is a
                    standard HDF5 filter, so the file stays readable outside h5py; lzf is
                    faster but h5py-only.
            compression_opts: Level passed to the filter (gzip 0-9); None for filters that
                    take no level.

    Returns:
            DataFrame with all inference results
    """
    # Get column names from config
    columns_name = get_tabular_columns(
        model.hparams,
        return_formula_str,
        return_cc_hash,
        output_subset=output_set,
        return_rule_provenance=return_rule_provenance,
    )
    # Ragged per-spectrum column families, read off COLUMN_SPECS so they cannot drift from
    # the DataFrame schema. Each spectrum contributes a variable-length block to every
    # column in a family, so each family needs its own running offset plus per-spectrum
    # <prefix>start_points/<prefix>end_points. Writing these at the per-spectrum offset
    # (the way scalar columns are written) makes every chunk flush clobber the blocks the
    # previous flush wrote.
    # Families are independent: a spectrum's peak count, node count, formula count and
    # joint-grid size are all different numbers, so no family's offset can substitute for
    # another's.
    ragged_groups = tuple(
        (prefix, tuple(c.name for c in COLUMN_SPECS if c.family == family))
        for family, prefix in RAGGED_FAMILY_PREFIX.items()
    )
    ragged_columns = {k for _, members in ragged_groups for k in members}
    group_offsets = {prefix: 0 for prefix, _ in ragged_groups}

    def is_blocked(values: list) -> bool:
        """True when a column holds per-spectrum blocks rather than scalar placeholders.

        A column that was not requested comes through as one NaN per spectrum, which must
        keep going down the scalar path.
        """
        return bool(values) and isinstance(values[0], (th.Tensor, np.ndarray, list))

    def is_all_none(save_v: list) -> bool:
        """True when a scalar column carries no value at all, only NaN/None placeholders.

        An unrequested column arrives as one NaN per spectrum (see `is_blocked`), which is
        what `drop_none_cols` removes. Non-float columns are never placeholders.
        """
        if len(save_v) == 0:
            return False
        if save_v.dtype.kind == "f":
            return bool(np.isnan(save_v).all())
        if save_v.dtype == object:
            return all(v is None or (isinstance(v, float) and np.isnan(v)) for v in save_v)
        return False

    def concat_blocks(values: list, column_name: str) -> np.ndarray:
        """Flatten per-spectrum blocks into one array h5py can store."""
        if isinstance(values[0], th.Tensor):
            return th.cat(values, 0).detach().cpu().numpy()
        flat = [item for block in values for item in block]
        if column_name == "pred_formula_adducts":
            # A missing precursor type is NaN in the tabular export. HDF5 needs one
            # string dtype across chunks, including when the first chunk is all missing.
            return np.array(
                ["NaN" if isinstance(v, float) and np.isnan(v) else v for v in flat],
                dtype=h5py.string_dtype(),
            )
        if flat and isinstance(flat[0], str):
            # Formula strings are ragged text, so they need a variable-length string dtype.
            return np.array(flat, dtype=h5py.string_dtype())
        return np.array(flat)

    # Accumulate all batch results
    preds_l = {}
    filter_stats: dict = {}

    # Columns dropped as all-NaN placeholders. The decision is made once, on the first
    # flush, and enforced on every later one: this writer appends to a single file, so a
    # column skipped early and written late would be short by every earlier spectrum.
    placeholder_cols: set[str] = set()

    start_point = 0
    with h5py.File(save_path, "w") as save_file:

        def create_kwargs(dtype: Any, trailing: tuple) -> dict[str, Any]:
            """Chunking/compression for one dataset.

            Compression is what makes this file small: the index columns are small integers
            stored in 8 bytes, so byte-shuffling exposes long runs of zeros and gzip takes
            them ~15-26x. The float probability columns are high-entropy and barely move
            (~1.2x) - they are already float32, and neither a dtype change nor a filter
            helps them. Measured on a real 96k-spectrum eval: an 80 MB index column becomes
            5.1 MB.

            The large chunk needs compression to be safe. HDF5 allocates a whole chunk once
            any of it is touched, so a 65536-row chunk holding a 378-row column costs 528 KB
            raw - but 7 KB once the unwritten tail is compressed.
            """
            if not compression:
                # Let h5py size the chunks; a big one would be pure waste uncompressed.
                return {"chunks": True}
            kwargs = {
                "chunks": (_HDF5_CHUNK_LEN,) + trailing,
                "compression": compression,
            }
            if compression_opts is not None:
                kwargs["compression_opts"] = compression_opts
            if dtype.kind in "biufc":
                # Byte-shuffle only applies to fixed-width numbers; it is a no-op on the
                # variable-length strings in pred_formula_strs.
                kwargs["shuffle"] = True
            return kwargs

        def resize_and_save(k: str, save_v: Any, start_point: int) -> None:
            if k not in save_file:
                # Trailing dims are kept fixed and only axis 0 grows, so 2-D columns such as
                # pred_node_ccs (one hash pair per DAG node) store as (n_nodes, 2).
                save_file.create_dataset(
                    k,
                    shape=(0,) + save_v.shape[1:],
                    dtype=save_v.dtype,
                    maxshape=(None,) + save_v.shape[1:],
                    **create_kwargs(save_v.dtype, save_v.shape[1:]),
                )  # important to avoid bug in large number
            save_file[k].resize(start_point + len(save_v), axis=0)
            save_file[k][start_point : start_point + len(save_v)] = save_v

        save_file.create_dataset(
            "start_points",
            shape=(0,),
            dtype=np.int64,
            maxshape=(None,),
            **create_kwargs(np.dtype(np.int64), ()),
        )
        save_file.create_dataset(
            "end_points",
            shape=(0,),
            dtype=np.int64,
            maxshape=(None,),
            **create_kwargs(np.dtype(np.int64), ()),
        )

        batch_iterator = run_inference_tabular_iter(
            model=model,
            dataloader=dataloader,
            device=device,
            batch_cutoff=batch_cutoff,
            evaluate=evaluate,
            output_set=output_set,
            nb_iso=nb_iso,
            min_peak_int=min_peak_int,
            return_formula_str=return_formula_str,
            return_cc_hash=return_cc_hash,
            rebase_indices=rebase_indices,
            filter_stats=filter_stats if min_peak_int > 0 else None,
            return_rule_provenance=return_rule_provenance,
        )
        # The final sentinel flushes a partial chunk on actual iterator exhaustion,
        # including cutoffs and samplers whose reported length is only an estimate.
        for batch_rows in chain(batch_iterator, (None,)):
            if batch_rows is not None:
                assert len(batch_rows) == len(columns_name)
                for k, v in zip(columns_name, batch_rows):
                    if k not in preds_l:
                        preds_l[k] = []
                    preds_l[k].extend(v)
            if preds_l and (batch_rows is None or len(preds_l["pred_mzs"]) >= chunk_size):
                print("saving ...", start_point)
                # Scalar columns: one value per spectrum, so the per-spectrum offset is
                # right for them. Ragged families are written below at their own offsets.
                for k in preds_l:
                    if k in ragged_columns and is_blocked(preds_l[k]):
                        continue
                    # check whether is tensor
                    if len(preds_l[k]) > 0 and hasattr(preds_l[k][0], "detach"):
                        save_v = th.cat(preds_l[k], 0).detach().cpu().numpy()
                    else:
                        save_v = np.array(preds_l[k])
                    if drop_none_cols:
                        if start_point == 0:
                            if is_all_none(save_v):
                                placeholder_cols.add(k)
                                continue
                        elif k in placeholder_cols:
                            # A metric that is NaN for the first chunk's spectra and real
                            # later would silently lose those later values, so stop instead.
                            if not is_all_none(save_v):
                                raise RuntimeError(
                                    f"Column '{k}' was dropped as an all-NaN placeholder on "
                                    f"the first chunk but holds values at spectrum "
                                    f"{start_point}. Rerun with drop_none_cols=False."
                                )
                            continue
                    resize_and_save(k, save_v, start_point)

                for prefix, members in ragged_groups:
                    present = [k for k in members if k in preds_l and is_blocked(preds_l[k])]
                    if not present:
                        continue  # family not requested for this run
                    # Every column in a family is indexed by the same per-spectrum block,
                    # so their block lengths must agree spectrum by spectrum. Checking only
                    # the concatenated totals would let a per-spectrum misalignment through.
                    block_lengths = [len(block) for block in preds_l[present[0]]]
                    for k in present[1:]:
                        assert [
                            len(b) for b in preds_l[k]
                        ] == block_lengths, (
                            f"{k} block lengths disagree with {present[0]} in family '{prefix}'"
                        )
                    group_start = group_offsets[prefix]
                    offset = group_start
                    starts, ends = [], []
                    for n in block_lengths:
                        starts.append(offset)
                        offset += n
                        ends.append(offset)
                    for k in present:
                        resize_and_save(k, concat_blocks(preds_l[k], k), group_start)
                    # start/end points are per spectrum, so they go at the spectrum offset.
                    resize_and_save(f"{prefix}start_points", np.array(starts), start_point)
                    resize_and_save(f"{prefix}end_points", np.array(ends), start_point)
                    group_offsets[prefix] = offset

                start_point += len(preds_l["pred_mzs"])
                del preds_l
                preds_l = {}
    if filter_stats:
        n_before = filter_stats.get("n_peaks_before", 0)
        n_after = filter_stats.get("n_peaks_after", 0)
        n_inst_filt = filter_stats.get("n_inst_filtered", 0)
        n_inst_total = filter_stats.get("n_inst_total", 0)
        pct_peaks = 100.0 * (n_before - n_after) / n_before if n_before > 0 else 0.0
        pct_inst = 100.0 * n_inst_filt / n_inst_total if n_inst_total > 0 else 0.0
        print(
            f">>> Peak filter stats (min_peak_int={min_peak_int}):\n"
            f"    Peaks before: {n_before:,}  after: {n_after:,}  removed: {n_before - n_after:,} ({pct_peaks:.1f}%)\n"
            f"    Instances filtered: {n_inst_filt:,} / {n_inst_total:,} ({pct_inst:.1f}%)"
        )
    return start_point


def run_inference_tabular_iter(
    model: th.nn.Module,
    dataloader: th.utils.data.DataLoader,
    device: th.device,
    batch_cutoff: int = -1,
    evaluate: bool = False,
    output_set: set[str] | None = None,
    nb_iso: bool = False,
    min_peak_int: float = 0.0,
    return_formula_str: bool = False,
    return_cc_hash: bool = False,
    rebase_indices: bool = True,
    filter_stats: dict | None = None,
    return_rule_provenance: bool = False,
) -> Iterator[pd.DataFrame]:
    """
    Generator that yields batch results for streaming inference processing.

    Yields batch_rows for each batch processed. Caller is responsible for accumulating,
    saving, or processing results as needed.

    batch_rows is COLUMN-major: a list of ``len(columns_name)`` lists, each holding one
    value per spectrum in the batch (NOT a list of per-spectrum rows). Transpose with
    ``zip(*batch_rows)`` to get rows, or zip it against the column names to accumulate
    per column.

    Args:
            model: Model instance
            dataloader: DataLoader with batches
            device: Device to run inference on
            batch_cutoff: Max batches to process (-1 = all)
            evaluate: Whether to run evaluation step
            output_set: Subset of outputs to include
            nb_iso: Whether model uses neighbor isotope prediction
            min_peak_int: Minimum peak intensity threshold
            return_formula_and_cc: Whether to return formula strings and CCs
            rebase_indices: Whether to rebase indices
            output_set: Subset of output keys; probabilities outside it are dropped
            filter_stats: Optional mutable dict to accumulate peak-filtering statistics
                (n_peaks_before, n_peaks_after, n_inst_filtered, n_inst_total).

    Yields:
            One column-major batch_rows per batch

    Example:
            columns_name = [...]  # Define column names once
            for batch_rows in run_inference_tabular_iter(model, dataloader, device):
                    df = pd.DataFrame(list(zip(*batch_rows)), columns=columns_name)
                    df.to_pickle(f"batch_{batch_num}.pkl")
    """
    # A DataLoader over an IterableDataset has no length: batches are formed lazily, so
    # the count is unknown until the stream is exhausted.
    try:
        n_batches = len(dataloader)
    except TypeError:
        n_batches = None

    if n_batches is None:
        total_batches = batch_cutoff if batch_cutoff > 0 else None
    else:
        total_batches = n_batches if batch_cutoff < 0 else min(n_batches, batch_cutoff)
    n_desc = "an unknown number of" if total_batches is None else total_batches
    print(f"\n>>> Starting inference on {n_desc} batches...")
    if model.training:
        raise RuntimeError(f"> Warning: model {model} is in training mode!")

    BATCH_IDX_VARS, NODE_IDX_VARS, FORMULA_IDX_VARS, _ = get_idx_vars(
        output_set,
        model,
        nb_iso,
        return_formula_str=return_formula_str,
        return_cc_hash=return_cc_hash,
    )

    eval_score_types = get_score_types(model.hparams)
    model.eval()
    with th.inference_mode():
        for batch_idx, batch_input in tqdm(
            enumerate(dataloader), total=total_batches, desc="Running inference"
        ):
            # Early exit for batch cutoff
            if batch_cutoff >= 0 and batch_idx >= batch_cutoff:
                print(f"> reached batch cutoff of {batch_cutoff}, stopping inference")
                break
            # Process batch and get data rows
            batch_rows = process_batch_to_tabular_row(
                batch_input=batch_input,
                model=model,
                device=device,
                evaluate=evaluate,
                eval_score_types=eval_score_types,
                BATCH_IDX_VARS=BATCH_IDX_VARS,
                NODE_IDX_VARS=NODE_IDX_VARS,
                FORMULA_IDX_VARS=FORMULA_IDX_VARS,
                min_peak_int=min_peak_int,
                rebase_indices=rebase_indices,
                return_formula_str=return_formula_str,
                return_cc_hash=return_cc_hash,
                output_set=output_set,
                filter_stats=filter_stats,
                return_rule_provenance=return_rule_provenance,
            )
            yield batch_rows


