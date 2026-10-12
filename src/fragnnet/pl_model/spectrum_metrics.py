"""Auxiliary metric setup for ``SpectrumPL`` (moved from ``SpectrumPL._setup_metric_fns``).

``setup_metric_fns`` builds the metric names, their min/max direction and the batch metric
function from the configured ``auxiliary_scores``. ``SpectrumPL._setup_metric_fns`` delegates
here, so subclasses that extend it after ``super()`` (MASSter) are unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch as th

from fragnnet.loss.entropy import sparse_entropy_fn
from fragnnet.output_schema import EXPLAINABILITY_OUTPUT_KEYS
from fragnnet.utils.scatter_utils import safelog, scatter_l1normalize, scatter_reduce
from fragnnet.utils.spec_similarity import (
    batch_cos_hun_helper,
    batch_jss_hun_helper,
    calculate_match_mz_reductions,
    calculate_match_mzs,
    cos_sim_helper,
    jss_helper,
    ndcg_helper,
    opt_cos_sim_helper,
)
from fragnnet.utils.spec_utils import (
    batched_bin_func,
    get_ints_transform_func,
    get_split_fn,
    ragged_to_batch,
    round_aggregate_peaks,
)

if TYPE_CHECKING:
    from fragnnet.pl_model.spectrum_pl import SpectrumPL


def setup_metric_fns(pl_module: SpectrumPL) -> None:
    """
    Setup all auxiliary and main metric functions, including configuration for different evaluation modes.
    """

    # setup metrics
    pl_module.metric_min_max = {"loss": "min"}
    pl_module.metric_bests = ["loss"]
    if "primary_loss" in pl_module.metric_names:
        pl_module.metric_min_max["primary_loss"] = "min"
        pl_module.metric_bests.append("primary_loss")
    for k in pl_module.metric_names:
        if k not in pl_module.metric_min_max:
            pl_module.metric_min_max[k] = None
    pl_module.auxiliary_metric_names = set()

    bin_sqrt_flags = [False]
    if pl_module.hparams.eval_bin_sqrt:
        bin_sqrt_flags.append(True)
    bin_remove_prec_peak_flags = [False]
    if pl_module.hparams.eval_bin_remove_prec_peaks:
        bin_remove_prec_peak_flags.append(True)
    hun_sqrt_flags = [False]
    if pl_module.hparams.eval_hun_sqrt:
        hun_sqrt_flags.append(True)
    hun_remove_prec_peak_flags = [False]
    if pl_module.hparams.eval_hun_remove_prec_peaks:
        hun_remove_prec_peak_flags.append(True)
    nb_flags = [False]
    if pl_module.hparams.nb_iso:
        nb_flags.append(True)
    pl_module.extra_metric_args = []

    eval_mz_bin_reses = pl_module.hparams.eval_mz_bin_res

    if (
        len(pl_module.hparams.auxiliary_scores) > 0
        and not pl_module.hparams.sparse_cosine_similarity
    ):
        raise ValueError("auxiliary_scores require sparse_cosine_similarity")
    compute_match_mzs = False
    compute_full_match_mzs = False
    compute_rounded_match_mzs = False
    compute_bin_specs = False
    compute_entropies = False
    compute_pred_entropies = False
    compute_inst_tol_match_mzs = False
    pl_module.batch_metric_args: set[str] = set()

    for metric_name in pl_module.hparams.auxiliary_scores:
        if metric_name == "cos_sim":
            for remove_prec_peak in bin_remove_prec_peak_flags:
                for sqrt in bin_sqrt_flags:
                    for mz_bin_res in eval_mz_bin_reses:
                        fname_base = "cos_sim"
                        if sqrt:
                            fname_base += "_sqrt"
                        if remove_prec_peak:
                            fname_base += "_np"
                        fname = f"{fname_base}_{mz_bin_res}"
                        pl_module.metric_min_max[fname] = "max"
                        pl_module.metric_bests.append(fname)
                        pl_module.auxiliary_metric_names.add(fname)
                compute_bin_specs = True
        elif metric_name == "jss":
            for remove_prec_peak in bin_remove_prec_peak_flags:
                for sqrt in bin_sqrt_flags:
                    for mz_bin_res in eval_mz_bin_reses:
                        fname_base = "jss"
                        if sqrt:
                            fname_base += "_sqrt"
                        if remove_prec_peak:
                            fname_base += "_np"
                        fname = f"{fname_base}_{mz_bin_res}"
                        pl_module.metric_min_max[fname] = "max"
                        pl_module.metric_bests.append(fname)
                        pl_module.auxiliary_metric_names.add(fname)
        elif metric_name == "recall":
            pl_module.metric_min_max["recall"] = "max"
            pl_module.auxiliary_metric_names.add("recall")
            compute_match_mzs = True
            if pl_module.inst_tol_lookup is not None:
                pl_module.batch_metric_args.add("spec_inst_type")
        elif metric_name == "wrecall":
            pl_module.metric_min_max["wrecall"] = "max"
            pl_module.auxiliary_metric_names.add("wrecall")
            compute_match_mzs = True
            if pl_module.inst_tol_lookup is not None:
                pl_module.batch_metric_args.add("spec_inst_type")
        elif metric_name == "opt_cos_sim":
            for remove_prec_peak in bin_remove_prec_peak_flags:
                for sqrt in bin_sqrt_flags:
                    for mz_bin_res in eval_mz_bin_reses:
                        fname_base = "opt_cos_sim"
                        if sqrt:
                            fname_base += "_sqrt"
                        if remove_prec_peak:
                            fname_base += "_np"
                        fname = f"{fname_base}_{mz_bin_res}"
                        pl_module.metric_min_max[fname] = None
                        pl_module.auxiliary_metric_names.add(fname)
                compute_bin_specs = True
        elif metric_name == "true_spec_e":
            pl_module.metric_min_max["true_spec_e"] = None
            pl_module.auxiliary_metric_names.add("true_spec_e")
            compute_entropies = True
        elif metric_name == "true_spec_ne":
            pl_module.metric_min_max["true_spec_ne"] = None
            pl_module.auxiliary_metric_names.add("true_spec_ne")
            compute_entropies = True
        elif metric_name == "pred_spec_e":
            pl_module.metric_min_max["pred_spec_e"] = None
            pl_module.auxiliary_metric_names.add("pred_spec_e")
            compute_pred_entropies = True
        elif metric_name == "pred_spec_ne":
            pl_module.metric_min_max["pred_spec_ne"] = None
            pl_module.auxiliary_metric_names.add("pred_spec_ne")
            compute_pred_entropies = True
        elif metric_name == "cos_hun":
            for remove_prec_peak in hun_remove_prec_peak_flags:
                for sqrt in hun_sqrt_flags:
                    fname_base = "cos_hun"
                    if sqrt:
                        fname_base += "_sqrt"
                    if remove_prec_peak:
                        fname_base += "_np"
                    fname = f"{fname_base}"
                    pl_module.metric_min_max[fname] = "max"
                    pl_module.metric_bests.append(fname)
                    pl_module.auxiliary_metric_names.add(fname)
            compute_rounded_match_mzs = True
        elif metric_name == "precision":
            pl_module.metric_min_max["precision"] = "max"
            pl_module.metric_bests.append("precision")
            pl_module.auxiliary_metric_names.add("precision")
            compute_match_mzs = True
        elif metric_name == "wprecision":
            pl_module.metric_min_max["wprecision"] = "max"
            pl_module.metric_bests.append("wprecision")
            pl_module.auxiliary_metric_names.add("wprecision")
            compute_match_mzs = True
        # elif metric_name == "dice":
        #   self.metric_min_max["dice"] = "max"
        #   self.metric_bests.append("dice")
        #   compute_match_mzs = True
        elif metric_name == "ndcg":
            for union in [True, False]:
                if union:
                    fname_base = "ndcg_un"
                    tie_break_flags = [True, False]
                else:
                    fname_base = "ndcg_int"
                    tie_break_flags = [True]
                for optimistic in tie_break_flags:
                    if union:
                        suffix = "_opt" if optimistic else "_pess"
                    else:
                        suffix = ""
                    fname = f"{fname_base}{suffix}"
                    pl_module.metric_min_max[fname] = "max"
                    pl_module.metric_bests.append(fname)
                    pl_module.auxiliary_metric_names.add(fname)
            compute_rounded_match_mzs = True
            compute_match_mzs = True
            compute_full_match_mzs = True
        elif metric_name == "jss_hun":
            for remove_prec_peak in hun_remove_prec_peak_flags:
                for sqrt in hun_sqrt_flags:
                    fname_base = "jss_hun"
                    if sqrt:
                        fname_base += "_sqrt"
                    if remove_prec_peak:
                        fname_base += "_np"
                    fname = f"{fname_base}"
                    pl_module.metric_min_max[fname] = "max"
                    pl_module.metric_bests.append(fname)
                    pl_module.auxiliary_metric_names.add(fname)
            compute_rounded_match_mzs = True
        elif metric_name == "true_oos_prob":
            pl_module.metric_min_max["true_oos_prob"] = None
            pl_module.auxiliary_metric_names.add("true_oos_prob")
            compute_match_mzs = True
        elif metric_name == "true_oos_e":
            pl_module.metric_min_max["true_oos_e"] = None
            pl_module.auxiliary_metric_names.add("true_oos_e")
            compute_match_mzs = True
        elif metric_name == "pred_node_count":
            for nb_flag in nb_flags:
                if nb_flag:
                    fname = "pred_nb_node_count"
                    pl_module.extra_metric_args.extend(["pred_nb_node_batch_idxs"])
                else:
                    fname = "pred_node_count"
                    pl_module.extra_metric_args.extend(["pred_node_batch_idxs"])
                pl_module.metric_min_max[fname] = None
                pl_module.auxiliary_metric_names.add(fname)
        elif metric_name == "pred_formula_count":
            fname = "pred_formula_count"
            pl_module.extra_metric_args.extend(["pred_formula_batch_idxs"])
            pl_module.metric_min_max[fname] = None
            pl_module.auxiliary_metric_names.add(fname)
        elif metric_name == "pred_edge_count":
            fname = "pred_edge_count"
            pl_module.extra_metric_args.extend(["pred_edge_batch_idxs"])
            pl_module.metric_min_max[fname] = None
            pl_module.auxiliary_metric_names.add(fname)
        elif metric_name == "cos_hun_inst_tol":
            for remove_prec_peak in hun_remove_prec_peak_flags:
                for sqrt in hun_sqrt_flags:
                    fname_base = "cos_hun_inst_tol"
                    if sqrt:
                        fname_base += "_sqrt"
                    if remove_prec_peak:
                        fname_base += "_np"
                    fname = fname_base
                    pl_module.metric_min_max[fname] = "max"
                    pl_module.metric_bests.append(fname)
                    pl_module.auxiliary_metric_names.add(fname)
            compute_inst_tol_match_mzs = True
            compute_rounded_match_mzs = True
            pl_module.batch_metric_args.add("spec_inst_type")
        elif metric_name == "jss_hun_inst_tol":
            for remove_prec_peak in hun_remove_prec_peak_flags:
                for sqrt in hun_sqrt_flags:
                    fname_base = "jss_hun_inst_tol"
                    if sqrt:
                        fname_base += "_sqrt"
                    if remove_prec_peak:
                        fname_base += "_np"
                    fname = fname_base
                    pl_module.metric_min_max[fname] = "max"
                    pl_module.metric_bests.append(fname)
                    pl_module.auxiliary_metric_names.add(fname)
            compute_inst_tol_match_mzs = True
            compute_rounded_match_mzs = True
            pl_module.batch_metric_args.add("spec_inst_type")
        else:
            raise ValueError(f"metric_name {metric_name} not recognized")

    pl_module.metric_names.update(pl_module.auxiliary_metric_names)
    # Metric inputs that only exist on an explainability forward. Metrics reach
    # `metric_fn` only through `extra_metric_args`, so this intersection is the complete
    # set of metric inputs a `return_explainability_tensors=False` forward would hand
    # back as None (e.g. `pred_nb_node_batch_idxs`, which `pred_node_count` needs under
    # nb_iso). `inference_step` ORs its own demand on when this is non-empty.
    pl_module.expl_metric_args = EXPLAINABILITY_OUTPUT_KEYS.intersection(
        pl_module.extra_metric_args
    )

    def calculate_all_auxiliary_metrics(
        true_mzs: th.Tensor,
        true_ints: th.Tensor,
        true_batch_idxs: th.Tensor,
        pred_mzs: th.Tensor,
        pred_ints: th.Tensor,
        pred_batch_idxs: th.Tensor,
        true_prec_mzs: th.Tensor,
        split: str = "test",
        **kwargs: Any,
    ) -> dict[str, th.Tensor]:
        # assumes both true/pred spectra are already L1-normalized

        batch_size = th.max(true_batch_idxs) + 1
        assert batch_size == th.max(pred_batch_idxs) + 1, (
            batch_size,
            th.max(pred_batch_idxs) + 1,
        )

        # Per-spectrum metrics assign by batch index in test/predict as well.
        metric_d = {
            k: -th.ones([batch_size], dtype=true_ints.dtype, device=true_ints.device)
            for k in pl_module.auxiliary_metric_names
        }
        # define aggregation
        agg = "sum" if pl_module.hparams.sum_ints else "amax"

        # global calculations
        if compute_rounded_match_mzs:
            true_mzs_r, true_ints_r, true_batch_idxs_r = round_aggregate_peaks(
                true_mzs, true_ints, true_batch_idxs, agg=agg
            )
            pred_mzs_r, pred_ints_r, pred_batch_idxs_r = round_aggregate_peaks(
                pred_mzs, pred_ints, pred_batch_idxs, agg=agg
            )

        if compute_bin_specs:
            # Predicted m/z are not range-limited upstream: preproc_spec's (eval, pred)
            # branch only untransforms and normalizes intensities, while true peaks pass
            # through rel_filter_func with this same mz_max. A predicted candidate at or
            # above mz_max -- the precursor of a ~1500 Da molecule, say -- therefore reached
            # batched_bin_func and tripped its range assert. Drop those peaks so both sides
            # are binned over the same [0, mz_max) window. Peaks in range are untouched, so
            # this changes no metric that did not previously crash.
            bin_pred_mzs = pred_mzs
            bin_pred_ints = pred_ints
            bin_pred_batch_idxs = pred_batch_idxs
            if pl_module.hparams.mz_max > 0.0:
                in_bin_range = bin_pred_mzs < pl_module.hparams.mz_max
                bin_pred_mzs = bin_pred_mzs[in_bin_range]
                bin_pred_ints = bin_pred_ints[in_bin_range]
                bin_pred_batch_idxs = bin_pred_batch_idxs[in_bin_range]
            for remove_prec_peak in bin_remove_prec_peak_flags:
                for sqrt in bin_sqrt_flags:
                    if sqrt:
                        ints_transform = get_ints_transform_func("sqrt")
                    else:
                        ints_transform = get_ints_transform_func("none")
                    for mz_bin_res in eval_mz_bin_reses:
                        # bin, transform, normalize
                        true_bin_idxs, true_bin_ints, true_bin_batch_idxs = batched_bin_func(
                            true_mzs,
                            true_ints,
                            true_batch_idxs,
                            mz_max=pl_module.hparams.mz_max,
                            mz_bin_res=mz_bin_res,
                            agg=agg,
                            sparse=True,
                            remove_prec_peaks=remove_prec_peak,
                            prec_mzs=true_prec_mzs,
                        )
                        pred_bin_idxs, pred_bin_ints, pred_bin_batch_idxs = batched_bin_func(
                            bin_pred_mzs,
                            bin_pred_ints,
                            bin_pred_batch_idxs,
                            mz_max=pl_module.hparams.mz_max,
                            mz_bin_res=mz_bin_res,
                            agg=agg,
                            sparse=True,
                            remove_prec_peaks=remove_prec_peak,
                            prec_mzs=true_prec_mzs,
                        )
                        # apply intensity transform (e.g. sqrt); helpers normalize internally
                        true_bin_ints = ints_transform(true_bin_ints)
                        pred_bin_ints = ints_transform(pred_bin_ints)
                        if "cos_sim" in pl_module.hparams.auxiliary_scores:
                            fname_base = "cos_sim"
                            if sqrt:
                                fname_base += "_sqrt"
                            if remove_prec_peak:
                                fname_base += "_np"
                            fname = f"{fname_base}_{mz_bin_res}"
                            cos_sim = cos_sim_helper(
                                true_bin_idxs,
                                true_bin_ints,
                                true_bin_batch_idxs,
                                pred_bin_idxs,
                                pred_bin_ints,
                                pred_bin_batch_idxs,
                            )
                            # print(cos_sim)
                            metric_d[fname] = cos_sim
                        if "opt_cos_sim" in pl_module.hparams.auxiliary_scores:
                            fname_base = "opt_cos_sim"
                            if sqrt:
                                fname_base += "_sqrt"
                            if remove_prec_peak:
                                fname_base += "_np"
                            fname = f"{fname_base}_{mz_bin_res}"
                            opt_cos_sim = opt_cos_sim_helper(
                                true_bin_idxs,
                                true_bin_ints,
                                true_bin_batch_idxs,
                                pred_bin_idxs,
                                pred_bin_ints,
                                pred_bin_batch_idxs,
                            )
                            metric_d[fname] = opt_cos_sim
                        if "jss" in pl_module.hparams.auxiliary_scores:
                            fname_base = "jss"
                            if sqrt:
                                fname_base += "_sqrt"
                            if remove_prec_peak:
                                fname_base += "_np"
                            fname = f"{fname_base}_{mz_bin_res}"
                            jss = jss_helper(
                                true_bin_idxs,
                                true_bin_ints,
                                true_bin_batch_idxs,
                                pred_bin_idxs,
                                pred_bin_ints,
                                pred_bin_batch_idxs,
                                log_min=pl_module.hparams.log_min,
                            )
                            metric_d[fname] = jss

        if compute_entropies:
            true_log_probs = kwargs.get("true_logprobs")
            if true_log_probs is None:
                true_log_probs = safelog(
                    scatter_l1normalize(true_ints, true_batch_idxs), eps=pl_module.hparams.log_min
                )
            true_spec_e, true_spec_ne = sparse_entropy_fn(true_log_probs, true_batch_idxs)
            if "true_spec_e" in pl_module.hparams.auxiliary_scores:
                metric_d["true_spec_e"] = true_spec_e
            if "true_spec_ne" in pl_module.hparams.auxiliary_scores:
                metric_d["true_spec_ne"] = true_spec_ne

        if compute_pred_entropies:
            pred_log_probs = kwargs.get("pred_logprobs")
            if pred_log_probs is None:
                pred_log_probs = safelog(
                    scatter_l1normalize(pred_ints, pred_batch_idxs), eps=pl_module.hparams.log_min
                )
            pred_spec_e, pred_spec_ne = sparse_entropy_fn(pred_log_probs, pred_batch_idxs)
            if "pred_spec_e" in pl_module.hparams.auxiliary_scores:
                metric_d["pred_spec_e"] = pred_spec_e
            if "pred_spec_ne" in pl_module.hparams.auxiliary_scores:
                metric_d["pred_spec_ne"] = pred_spec_ne

        true_ion_num = th.bincount(true_batch_idxs)
        pred_ion_num = th.bincount(pred_batch_idxs)
        if compute_rounded_match_mzs:
            batched_true_ints_r = ragged_to_batch(true_ints_r, true_batch_idxs_r, pad_value=0)
            batched_pred_ints_r = ragged_to_batch(pred_ints_r, pred_batch_idxs_r, pad_value=0)
            sqrt_batched_true_ints_r = get_ints_transform_func("sqrt")(batched_true_ints_r)
            sqrt_batched_pred_ints_r = get_ints_transform_func("sqrt")(batched_pred_ints_r)

        if compute_match_mzs:
            batch_true_ints = ragged_to_batch(true_ints, true_batch_idxs, pad_value=0)
            batch_pred_ints = ragged_to_batch(pred_ints, pred_batch_idxs, pad_value=0)
            batch_true_mzs = ragged_to_batch(true_mzs, true_batch_idxs, pad_value=float("inf"))
            batch_pred_mzs = ragged_to_batch(pred_mzs, pred_batch_idxs, pad_value=float("inf"))
            if compute_full_match_mzs:
                batch_match_mask = calculate_match_mzs(
                    batch_true_mzs,
                    batch_pred_mzs,
                    tolerance=pl_module.tolerance,
                    relative=pl_module.relative,
                    tolerance_min_mz=pl_module.tolerance_min_mz,
                )
                batch_true_match_mask = batch_match_mask.max(-1).values
                batch_pred_match_mask = batch_match_mask.max(-2).values
            else:
                batch_true_match_mask, batch_pred_match_mask = calculate_match_mz_reductions(
                    batch_true_mzs,
                    batch_pred_mzs,
                    tolerance=pl_module.tolerance,
                    relative=pl_module.relative,
                    tolerance_min_mz=pl_module.tolerance_min_mz,
                )
        if compute_rounded_match_mzs:
            batch_true_mzs_r = ragged_to_batch(
                true_mzs_r, true_batch_idxs_r, pad_value=float("inf")
            )
            batch_pred_mzs_r = ragged_to_batch(
                pred_mzs_r, pred_batch_idxs_r, pad_value=float("inf")
            )
            batch_true_prec_mask_r = calculate_match_mzs(
                batch_true_mzs_r,
                true_prec_mzs[..., None],
                tolerance=pl_module.tolerance,
                relative=pl_module.relative,
                tolerance_min_mz=pl_module.tolerance_min_mz,
            ).squeeze(-1)
            batch_pred_prec_mask_r = calculate_match_mzs(
                batch_pred_mzs_r,
                true_prec_mzs[..., None],
                tolerance=pl_module.tolerance,
                relative=pl_module.relative,
                tolerance_min_mz=pl_module.tolerance_min_mz,
            ).squeeze(-1)
            # match
            batch_match_mask_r = calculate_match_mzs(
                batch_true_mzs_r,
                batch_pred_mzs_r,
                tolerance=pl_module.tolerance,
                relative=pl_module.relative,
                tolerance_min_mz=pl_module.tolerance_min_mz,
            )
            batch_true_match_mask_r = batch_match_mask_r.max(
                -1
            ).values  # th.any(b_match_mask_r, dim=1)
            batch_pred_match_mask_r = batch_match_mask_r.max(
                -2
            ).values  # th.any(b_match_mask_r, dim=0)

        # calculate batched cosine and batched jss
        batched_metric = {}
        if "cos_hun" in pl_module.hparams.auxiliary_scores:
            for remove_prec_peak in hun_remove_prec_peak_flags:
                for sqrt in hun_sqrt_flags:
                    if sqrt:
                        inp_true = sqrt_batched_true_ints_r
                        inp_pred = sqrt_batched_pred_ints_r
                        # ints_transform = get_ints_transform_func("sqrt")
                    else:
                        inp_true = batched_true_ints_r
                        inp_pred = batched_pred_ints_r
                        # ints_transform = get_ints_transform_func("none")
                    fname_base = "cos_hun"
                    if sqrt:
                        fname_base += "_sqrt"
                    if remove_prec_peak:
                        fname_base += "_np"
                    fname = f"{fname_base}"
                    # metric_d[fname][b_idx] = b_cos_hun
                    batched_metric[fname] = batch_cos_hun_helper(
                        inp_true,
                        inp_pred,
                        batch_match_mask_r,
                        batch_true_match_mask_r,
                        batch_pred_match_mask_r,
                        remove_prec_peak,
                        batch_true_prec_mask_r,
                        batch_pred_prec_mask_r,
                    )
        if "jss_hun" in pl_module.hparams.auxiliary_scores:
            for remove_prec_peak in hun_remove_prec_peak_flags:
                for sqrt in hun_sqrt_flags:
                    if sqrt:
                        inp_true = sqrt_batched_true_ints_r
                        inp_pred = sqrt_batched_pred_ints_r
                        # ints_transform = get_ints_transform_func("sqrt")
                    else:
                        inp_true = batched_true_ints_r
                        inp_pred = batched_pred_ints_r
                    fname_base = "jss_hun"
                    if sqrt:
                        fname_base += "_sqrt"
                    if remove_prec_peak:
                        fname_base += "_np"
                    fname = f"{fname_base}"

                    batched_metric[fname] = batch_jss_hun_helper(
                        inp_true,
                        inp_pred,
                        batch_match_mask_r,
                        batch_true_match_mask_r,
                        batch_pred_match_mask_r,
                        remove_prec_peak,
                        batch_true_prec_mask_r,
                        batch_pred_prec_mask_r,
                        log_min=pl_module.hparams.log_min,
                    )

        if {"recall", "wrecall"}.intersection(pl_module.hparams.auxiliary_scores):
            recall_match_mask = batch_true_match_mask
            if pl_module.inst_tol_lookup is not None:
                inst_idxs = kwargs["spec_inst_type"].reshape(-1)
                tol_per_sample = pl_module.inst_tol_lookup[inst_idxs].to(dtype=true_mzs.dtype)
                min_mz_per_sample = pl_module.inst_min_mz_lookup[inst_idxs].to(dtype=true_mzs.dtype)
                recall_match_mask, _ = calculate_match_mz_reductions(
                    batch_true_mzs,
                    batch_pred_mzs,
                    tol_per_true=tol_per_sample[:, None],
                    min_mz_per_true=min_mz_per_sample[:, None],
                )
            if "recall" in pl_module.hparams.auxiliary_scores:
                metric_d["recall"] = recall_match_mask.float().sum(-1) / true_ion_num.float()
            if "wrecall" in pl_module.hparams.auxiliary_scores:
                metric_d["wrecall"] = (batch_true_ints * recall_match_mask.float()).sum(-1)
        if "precision" in pl_module.hparams.auxiliary_scores:
            metric_d["precision"] = batch_pred_match_mask.float().sum(-1) / pred_ion_num.float()
        if "wprecision" in pl_module.hparams.auxiliary_scores:
            metric_d["wprecision"] = (batch_pred_ints * batch_pred_match_mask.float()).sum(-1)
        if "cos_hun" in pl_module.hparams.auxiliary_scores:
            for remove_prec_peak in hun_remove_prec_peak_flags:
                for sqrt in hun_sqrt_flags:
                    fname_base = "cos_hun"
                    if sqrt:
                        fname_base += "_sqrt"
                    if remove_prec_peak:
                        fname_base += "_np"
                    fname = f"{fname_base}"
                    metric_d[fname] = batched_metric[fname]
        if "jss_hun" in pl_module.hparams.auxiliary_scores:
            for remove_prec_peak in hun_remove_prec_peak_flags:
                for sqrt in hun_sqrt_flags:
                    fname_base = "jss_hun"
                    if sqrt:
                        fname_base += "_sqrt"
                    if remove_prec_peak:
                        fname_base += "_np"
                    fname = f"{fname_base}"
                    metric_d[fname] = batched_metric[fname]
        if compute_inst_tol_match_mzs and (
            "cos_hun_inst_tol" in pl_module.hparams.auxiliary_scores
            or "jss_hun_inst_tol" in pl_module.hparams.auxiliary_scores
        ):
            spec_inst_type = kwargs.get("spec_inst_type")
            if spec_inst_type is not None and pl_module.inst_tol_lookup is not None:
                inst_idxs = spec_inst_type.reshape(-1)
                dtype = true_ints.dtype
                tol_per_sample = pl_module.inst_tol_lookup[inst_idxs].to(dtype=dtype)
                min_mz_per_sample = pl_module.inst_min_mz_lookup[inst_idxs].to(dtype=dtype)
                # Build per-peak effective absolute tolerance: (B, N_true).
                # tol_rel * clamp(mz, min=min_mz) converts relative ppm to absolute Da.
                safe_mzs = batch_true_mzs_r.clamp(max=pl_module.hparams.mz_max)
                tol_per_peak = tol_per_sample[:, None] * th.maximum(
                    safe_mzs, min_mz_per_sample[:, None]
                )
                batch_match_mask_it = calculate_match_mzs(
                    batch_true_mzs_r,
                    batch_pred_mzs_r,
                    tolerance=0.0,
                    relative=False,
                    tol_per_true=tol_per_peak,
                )
                batch_true_match_mask_it = batch_match_mask_it.max(-1).values
                batch_pred_match_mask_it = batch_match_mask_it.max(-2).values
                if "cos_hun_inst_tol" in pl_module.hparams.auxiliary_scores:
                    for remove_prec_peak in hun_remove_prec_peak_flags:
                        for sqrt in hun_sqrt_flags:
                            inp_true = sqrt_batched_true_ints_r if sqrt else batched_true_ints_r
                            inp_pred = sqrt_batched_pred_ints_r if sqrt else batched_pred_ints_r
                            fname_base = "cos_hun_inst_tol"
                            if sqrt:
                                fname_base += "_sqrt"
                            if remove_prec_peak:
                                fname_base += "_np"
                            metric_d[fname_base] = batch_cos_hun_helper(
                                inp_true,
                                inp_pred,
                                batch_match_mask_it,
                                batch_true_match_mask_it,
                                batch_pred_match_mask_it,
                                remove_prec_peak,
                                batch_true_prec_mask_r,
                                batch_pred_prec_mask_r,
                            )
                if "jss_hun_inst_tol" in pl_module.hparams.auxiliary_scores:
                    for remove_prec_peak in hun_remove_prec_peak_flags:
                        for sqrt in hun_sqrt_flags:
                            inp_true = sqrt_batched_true_ints_r if sqrt else batched_true_ints_r
                            inp_pred = sqrt_batched_pred_ints_r if sqrt else batched_pred_ints_r
                            fname_base = "jss_hun_inst_tol"
                            if sqrt:
                                fname_base += "_sqrt"
                            if remove_prec_peak:
                                fname_base += "_np"
                            metric_d[fname_base] = batch_jss_hun_helper(
                                inp_true,
                                inp_pred,
                                batch_match_mask_it,
                                batch_true_match_mask_it,
                                batch_pred_match_mask_it,
                                remove_prec_peak,
                                batch_true_prec_mask_r,
                                batch_pred_prec_mask_r,
                                log_min=pl_module.hparams.log_min,
                            )
        if (
            "ndcg" in pl_module.hparams.auxiliary_scores
            or "true_oos_prob" in pl_module.hparams.auxiliary_scores
            or "true_oos_e" in pl_module.hparams.auxiliary_scores
        ):
            pred_split_fn = get_split_fn(pred_batch_idxs)
            true_split_fn = get_split_fn(true_batch_idxs)
            split_true_ints = true_split_fn(true_ints)
            split_pred_ints = pred_split_fn(pred_ints)
            for b_idx in range(batch_size):  # the followings are not speed up yet
                b_true_ints = split_true_ints[b_idx]
                b_pred_ints = split_pred_ints[b_idx]

                if compute_match_mzs:
                    b_true_match_mask = batch_true_match_mask[b_idx][: b_true_ints.shape[0]]
                    b_pred_match_mask = batch_pred_match_mask[b_idx][: b_pred_ints.shape[0]]

                if "ndcg" in pl_module.hparams.auxiliary_scores:
                    for union in [True, False]:
                        if union:
                            fname_base = "ndcg_un"
                            tie_break_flags = [True, False]
                        else:
                            fname_base = "ndcg_int"
                            tie_break_flags = [True]
                        for optimistic in tie_break_flags:
                            if union:
                                suffix = "_opt" if optimistic else "_pess"
                            else:
                                suffix = ""
                            b_ndcg = ndcg_helper(
                                b_true_ints,
                                b_pred_ints,
                                batch_match_mask[b_idx][
                                    : b_true_ints.shape[0], : b_pred_ints.shape[0]
                                ],
                                b_true_match_mask,
                                b_pred_match_mask,
                                optimistic,
                                union,
                            )
                            fname = f"{fname_base}{suffix}"
                            metric_d[fname][b_idx] = b_ndcg

                if "true_oos_prob" in pl_module.hparams.auxiliary_scores:
                    b_true_oos_prob = th.sum(b_true_ints[~b_true_match_mask]) / th.sum(b_true_ints)
                    metric_d["true_oos_prob"][b_idx] = b_true_oos_prob

                if "true_oos_e" in pl_module.hparams.auxiliary_scores:
                    b_true_oos_probs = b_true_ints[~b_true_match_mask] / th.sum(
                        b_true_ints[~b_true_match_mask]
                    )
                    b_true_oos_logprobs = safelog(b_true_oos_probs, eps=pl_module.hparams.log_min)
                    b_true_oos_e = -th.sum(b_true_oos_probs * b_true_oos_logprobs)
                    metric_d["true_oos_e"][b_idx] = b_true_oos_e

        if "pred_node_count" in pl_module.hparams.auxiliary_scores:
            for nb_flag in nb_flags:
                if nb_flag:
                    key = "nb_node"
                else:
                    key = "node"
                b_pred_node_count = scatter_reduce(
                    th.ones_like(kwargs[f"pred_{key}_batch_idxs"]),
                    kwargs[f"pred_{key}_batch_idxs"],
                    reduce="sum",
                    dim=0,
                )
                assert th.min(b_pred_node_count) > 0, b_pred_node_count
                metric_d[f"pred_{key}_count"] = b_pred_node_count

        if "pred_formula_count" in pl_module.hparams.auxiliary_scores:
            b_pred_formula_count = scatter_reduce(
                th.ones_like(kwargs["pred_formula_batch_idxs"]),
                kwargs["pred_formula_batch_idxs"],
                reduce="sum",
                dim=0,
            )
            b_pred_formula_count = b_pred_formula_count - 1  # -1 for OOS!
            assert th.min(b_pred_formula_count) > 0, b_pred_formula_count
            metric_d["pred_formula_count"] = b_pred_formula_count

        if "pred_edge_count" in pl_module.hparams.auxiliary_scores:
            b_pred_edge_count = scatter_reduce(
                th.ones_like(kwargs["pred_edge_batch_idxs"]),
                kwargs["pred_edge_batch_idxs"],
                reduce="sum",
                dim=0,
            )
            assert th.min(b_pred_edge_count) > 0, b_pred_edge_count
            metric_d["pred_edge_count"] = b_pred_edge_count

        return metric_d

    pl_module.metric_fn = calculate_all_auxiliary_metrics
