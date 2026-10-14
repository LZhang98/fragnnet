"""Peak statistics of generated DAGs against observed spectra.
"""

import numpy as np
import pandas as pd
import torch as th

from fragnnet.utils.formula_utils import PREC_TYPE_TO_CMF_MASS_DIFF, PREC_TYPE_TO_MASS_DIFF
from fragnnet.utils.misc_utils import PPM


def compute_frag_peak_stats(
    peaks: np.ndarray,
    formula_peak_mzs: np.ndarray,
    formula_peak_probs: np.ndarray,
    idx_by_h_delta: dict,
    prec_mz: float,
    allowed_h_transfer: int,
    tolerance: float = 0.01,
    prec_type: str = "[M+H]+",
    is_ppm: bool = False,
    inst_type: str | None = None,
    inst_type_tol: dict | None = None,
) -> pd.Series:
    """_summary_

    Args:
        peaks (_type_): _description_
        formula_peak_mzs (_type_): _description_
        formula_peak_probs (_type_): _description_
        idx_by_h_delta (_type_): _description_
        allowed_h_transfer (_type_): _description_
        tolerance (float, optional): _description_. Defaults to 0.01.
        is_ppm (bool, optional): _description_. Defaults to False.
        inst_type (str, optional): Instrument type for this spectrum (e.g. "ORB", "QTOF").
            Used to look up a per-instrument tolerance in ``inst_type_tol``. Ignored if
            ``inst_type_tol`` is None or does not contain this key. Defaults to None.
        inst_type_tol (dict, optional): Per-instrument tolerance overrides, same convention
            as the ``inst_type_loss_tol`` training config (e.g.
            ``{"ORB": {"rel": 1.0e-5, "min_mz": 200.0}, "QTOF": {"rel": 2.0e-5, "min_mz": 750.0}}``).
            When ``inst_type`` is present in this dict, the effective m/z tolerance for each
            peak is ``rel * max(peak_mz, min_mz)``, overriding ``tolerance``/``is_ppm`` for
            that spectrum. Defaults to None (falls back to the global ``tolerance``/``is_ppm``
            behavior, unchanged from before this parameter existed).

    Returns:
        _type_: _description_
    """
    inst_tol_cfg = None
    if inst_type_tol is not None and inst_type in inst_type_tol:
        inst_tol_cfg = inst_type_tol[inst_type]
    true_mzs, true_ints = list(zip(*peaks))
    theoretical_mzs = formula_peak_mzs
    theoretical_probs = formula_peak_probs

    allowed_idx = list(idx_by_h_delta[0])
    for h in range(1, allowed_h_transfer):
        allowed_idx += list(idx_by_h_delta[2 * h - 1])
        allowed_idx += list(idx_by_h_delta[2 * h])
    allowed_idx = list(set(allowed_idx))
    # print(allowed_idx)
    indices = th.tensor(allowed_idx)
    theoretical_mzs = th.index_select(theoretical_mzs, 0, indices)
    theoretical_probs = th.index_select(theoretical_probs, 0, indices)

    prec_mask = (theoretical_probs > 0.0).type(th.float32)
    # Compute two sets of theoretical m/z values:
    #   - CMF: charge migrates to H+ before fragmentation (correct for [M+NH4]+, [M+Cl]-, etc.)
    #   - std: adduct mass added directly (correct for [M+H]+, [M-H]- where CMF == std)
    # A true peak is matched if it falls within tolerance of EITHER set.
    cmf_diff = PREC_TYPE_TO_CMF_MASS_DIFF.get(prec_type, PREC_TYPE_TO_MASS_DIFF[prec_type])
    std_diff = PREC_TYPE_TO_MASS_DIFF[prec_type]
    theoretical_mzs_cmf = theoretical_mzs + prec_mask * cmf_diff
    theoretical_mzs_std = theoretical_mzs + prec_mask * std_diff
    # compute overlap
    overlap_true_idxs = []
    overlap_true_ints = []
    overlap_pred_idxs = []
    overlap_pred_peak_counts = []
    overlap_pred_formula_counts = []

    # check true_mzs against union of CMF and standard theoretical peaks
    for true_idx, true_mz in enumerate(true_mzs):
        if inst_tol_cfg is not None:
            eff_tol = inst_tol_cfg["rel"] * max(true_mz, inst_tol_cfg["min_mz"])
            mz_close_cmf = th.abs(theoretical_mzs_cmf - true_mz) < eff_tol
            mz_close_std = th.abs(theoretical_mzs_std - true_mz) < eff_tol
        elif not is_ppm:
            mz_close_cmf = th.abs(theoretical_mzs_cmf - true_mz) < tolerance
            mz_close_std = th.abs(theoretical_mzs_std - true_mz) < tolerance
        else:
            mz_close_cmf = th.abs(theoretical_mzs_cmf - true_mz) < (true_mz * tolerance * PPM)
            mz_close_std = th.abs(theoretical_mzs_std - true_mz) < (true_mz * tolerance * PPM)
        mz_close = mz_close_cmf | mz_close_std
        if th.any(mz_close):
            pred_idx = th.nonzero(mz_close, as_tuple=False)
            num_formula_match = th.sum(th.any(mz_close, dim=1).type(th.int32)).item()
            num_peak_match = pred_idx.shape[0]
            overlap_true_idxs.append(true_idx)
            overlap_true_ints.append(true_ints[true_idx])
            overlap_pred_idxs.append(pred_idx)
            overlap_pred_peak_counts.append(num_peak_match)
            overlap_pred_formula_counts.append(num_formula_match)
    # remove duplicates
    if len(overlap_pred_idxs) > 0:
        overlap_pred_idxs = th.unique(th.cat(overlap_pred_idxs, dim=0), dim=0)
    else:
        overlap_pred_idxs = th.zeros((0, 2))
    recall = len(overlap_true_idxs) / len(true_mzs)
    w_recall = np.sum(overlap_true_ints) / np.sum(true_ints)
    prec = len(overlap_pred_idxs) / th.sum((theoretical_probs > 0.0).type(th.int32)).item()
    if len(overlap_pred_peak_counts) > 0:
        ppt_peak = np.mean(overlap_pred_peak_counts)
        ppt_formula = np.mean(overlap_pred_formula_counts)
    else:
        ppt_peak = np.nan
        ppt_formula = np.nan
    # check prec_mz stuff
    prec_recalls = []
    for comp_mzs in [theoretical_mzs_std, th.tensor(true_mzs)]:
        prec_mz_diffs = th.abs(comp_mzs - prec_mz)
        if inst_tol_cfg is not None:
            eff_tol = inst_tol_cfg["rel"] * max(prec_mz, inst_tol_cfg["min_mz"])
            prec_mz_close = prec_mz_diffs < eff_tol
        elif not is_ppm:
            prec_mz_close = prec_mz_diffs < tolerance
        else:
            prec_mz_close = prec_mz_diffs < (prec_mz * tolerance * PPM)
        if th.any(prec_mz_close):
            prec_recalls.append(1.0)
        else:
            prec_recalls.append(0.0)
    prec_recall, prec_spec_recall = prec_recalls
    return pd.Series([recall, w_recall, prec, ppt_peak, ppt_formula, prec_recall, prec_spec_recall])
