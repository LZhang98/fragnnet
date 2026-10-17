"""Compute the DAG-achievable oracle cos_hun / jss_hun ceiling on NIST20, no training required.

For each spectrum, the "oracle" prediction reproduces the true intensity exactly for every
true peak whose m/z is explainable by some fragment formula in the precomputed DAG (within
the same relative m/z tolerance used in eval), and predicts nothing elsewhere. Scoring this
oracle prediction with the same cos_hun_helper/jss_hun_helper used in training eval gives the
best score the model could possibly achieve given only the DAG's fragment coverage -- i.e. the
ceiling imposed by DAG structure, independent of intensity-prediction quality.

For cos_hun this ceiling has a closed form: sqrt(sum of squared L2-normalized true intensities
over DAG-achievable peaks). For jss_hun the same "reuse true intensities on the achievable
support" construction is used; it is the standard weighted-recall-style ceiling, not a formally
re-optimized JSS-maximizing prediction (the coupling between predicted mass and the mixture
distribution has no simple closed-form optimum), but it is the natural, tractable ceiling and is
consistent with how ppt/w_recall style stats are already computed elsewhere in this codebase.

Three m/z matching tolerances are scored per spectrum, matching the tolerance variants used
elsewhere in eval:
  - cos_hun_opt / jss_hun_opt: fixed 10 ppm (ratio 1e-5), floor 200 Da (default eval tolerance).
  - cos_hun_inst_tol_opt / jss_hun_inst_tol_opt: per-instrument tolerance -- FT 10 ppm/200 Da,
    QTOF 20 ppm/750 Da, other instrument types fall back to the 10 ppm default (matches
    inst_type_loss_tol in config/fragnnet/exp_fraggnn_ma_mi/fraggnn_d3_ma_mi_nist20_amolc_all.yml).
  - cos_hun_0p01da_opt / jss_hun_0p01da_opt: fixed absolute 0.01 Da tolerance.

Every spectrum is scored under four denoise filter tiers, reported stacked so the marginal effect
of each filter on the DAG ceiling is visible (see FILTER_TIERS):
  - 0_raw:           no filter (the spectrum as stored).
  - 1_above_prec:    drop true peaks above the precursor m/z using the per-instrument tolerance.
                     These peaks are outside this DAG's candidate support, but may include isotope,
                     co-isolation, or annotation effects; treat this as a sensitivity tier rather
                     than a definitive noise label. Count reported as n_above_prec_removed.
  - 2_uwpr_rm:       additionally drop every peak matching a known UWPR contaminant m/z (aggressive
                     -- removes even contaminant peaks that coincide with a real DAG fragment).
  - 3_uwpr_keepexpl: drop only peaks labelled ``contaminant_only`` in the supplied feasibility
                     CSV. Peaks labelled ``dag_feasible`` are kept. This tier is still dependent
                     on the DAG used to create that CSV and is a sensitivity analysis, not an
                     intrinsic data-quality label.
Tiers 2-3 require --uwpr_feasibility_csv; without it they equal 1_above_prec. Per-spectrum output
columns are suffixed with the tier, e.g. cos_hun_opt__0_raw, cos_hun_opt__3_uwpr_keepexpl.

Additionally, cos_hun_sqrt_opt / jss_hun_sqrt_opt sqrt-transform intensities before scoring
(at the default 10 ppm tolerance), matching the cos_hun_sqrt / jss_hun_sqrt eval variant that
de-emphasizes the dominant peak.

Usage:
    python scripts/debug/oracle/dag_oracle_scores.py \\
        --frag_dp data/frag/nist20_ma_mi_d3_h4_isoFalse_smpFalse/dags \\
        --frag_dp data/frag/nist20_ma_mi_d3_h4_isoFalse_smpFalse_ringmerge/dags \\
        --split test
"""

from __future__ import annotations

import argparse
import functools
import logging
import math
import os

import numpy as np
import pandas as pd

from oracle_dataset_utils import resolve_dsets_for_spec_fp
import torch as th

from fragnnet.utils import parallel_utils
from fragnnet.utils import dag_format_utils
from fragnnet.utils.formula_utils import PREC_TYPE_TO_CMF_MASS_DIFF, PREC_TYPE_TO_MASS_DIFF
from fragnnet.utils.spec_similarity import calculate_match_mzs, cos_hun_helper, jss_hun_helper

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TOLERANCE = 1.0e-5  # relative ratio (== 10 ppm), matches default loss_tolerance_rel
TOLERANCE_MIN_MZ = 200.0  # matches config/template.yml loss_tolerance_min_mz
LOG_MIN = 1.0e-9  # matches config/template.yml log_min
# Per-instrument tolerance override, matches inst_type_loss_tol in
# config/fragnnet/exp_fraggnn_ma_mi/fraggnn_d3_ma_mi_nist20_amolc_all.yml. Instrument types not
# listed here (IT, QQQ, Other, ...) fall back to the global TOLERANCE/TOLERANCE_MIN_MZ.
INST_TYPE_LOSS_TOL = {
    "ORB": {"rel": 1.0e-5, "min_mz": 200.0},  # 10 ppm, floor 200 Da
    "ORB_F": {"rel": 1.0e-5, "min_mz": 200.0},  # 10 ppm, floor 200 Da
    "QTOF": {"rel": 2.0e-5, "min_mz": 750.0},  # 20 ppm, floor 750 Da
}
ABS_TOLERANCE_DA = 0.01  # fixed absolute Da tolerance, reproduces the cos_sim_0.01 convention
# Loose absolute tolerance, guaranteed wider than every per-instrument tol above (FT ~0.01 Da at
# m/z 1000, QTOF ~0.015 Da at low m/z). Used to probe whether a peak that misses at instrument
# tolerance is a borderline miscalibration (recoverable here) vs. a gross error (still a miss).
ABS_TOLERANCE_DA_LOOSE = 0.05

# Denoise filter tiers applied to the true spectrum before scoring. Every spectrum is scored under
# all four and the DAG ceiling is reported stacked, exposing the marginal effect of each filter.
# Tiers 0-2 are strictly nested; tier 3 relaxes tier 2 (it keeps contaminant peaks the DAG can
# explain), so 3_uwpr_keepexpl >= 2_uwpr_rm in retained peaks.
#   0_raw:            no filter.
#   1_above_prec:     0_raw + drop peaks above the precursor m/z (outside this DAG's support).
#   2_uwpr_rm:        1_above_prec + drop every UWPR contaminant-matched peak (aggressive denoise).
#   3_uwpr_keepexpl:  1_above_prec + drop only rows labelled contaminant_only in the feasibility
#                     CSV. This keeps dag_feasible matches, but the label remains dependent on the
#                     DAG used to create the CSV and must not be treated as intrinsic ground truth.
# Tiers 2-3 are identical to 1_above_prec unless a --uwpr_feasibility_csv is supplied.
FILTER_TIERS = ["0_raw", "1_above_prec", "2_uwpr_rm", "3_uwpr_keepexpl"]

# Per-tier metric base names. Each becomes a "<base>__<tier>" column in the per-spectrum output.
METRIC_BASES = [
    "cos_hun_opt",
    "jss_hun_opt",
    "cos_hun_inst_tol_opt",
    "jss_hun_inst_tol_opt",
    "cos_hun_0p01da_opt",
    "jss_hun_0p01da_opt",
    "cos_hun_0p05da_opt",
    "jss_hun_0p05da_opt",
    "cos_hun_sqrt_opt",
    "jss_hun_sqrt_opt",
    "n_true_peaks",
    "n_achievable_peaks",
    "n_achievable_peaks_inst_tol",
    "n_achievable_peaks_0p01da",
    "n_achievable_peaks_0p05da",
]


def _tolerance_for_inst_type(inst_type: str) -> tuple[float, float]:
    """Return (tolerance, tolerance_min_mz) for the inst_tol variant, with global fallback."""
    if inst_type in INST_TYPE_LOSS_TOL:
        cfg = INST_TYPE_LOSS_TOL[inst_type]
        return cfg["rel"], cfg["min_mz"]
    return TOLERANCE, TOLERANCE_MIN_MZ


def _detect_is_compressed(frag_dp: str, sample_mol_id: int) -> bool:
    """Check on disk whether DAG pickle files under frag_dp are bz2-compressed."""
    if os.path.isfile(frag_dp) and str(frag_dp).endswith((".tar", ".zip")):
        return True
    bz2_sample = os.path.join(frag_dp, f"{int(sample_mol_id)}.pkl.bz2")
    pkl_sample = os.path.join(frag_dp, f"{int(sample_mol_id)}.pkl")
    if os.path.isfile(bz2_sample):
        return True
    if os.path.isfile(pkl_sample):
        return False

    if os.path.isdir(frag_dp):
        with os.scandir(frag_dp) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.endswith(".pkl.bz2"):
                    return True
                if entry.is_file() and entry.name.endswith(".pkl"):
                    return False
    raise FileNotFoundError(
        f"Could not find any .pkl or .pkl.bz2 DAG files under {frag_dp}"
    )


def _allowed_formula_idxs(idx_by_h_delta: list[set], max_h_transfer: int | None) -> list[int]:
    """Collect DAG formula-row indices allowed under a given H-transfer budget.

    Args:
        idx_by_h_delta: List of index sets, one per H-transfer delta bucket (as stored
            in the DAG dict; bucket 0 is zero H-transfer).
        max_h_transfer: Maximum |H-transfer| to allow. None means use every bucket the
            DAG was built with (its full H-transfer budget).

    Returns:
        Sorted list of unique formula-row indices.
    """
    available_h_transfer = (len(idx_by_h_delta) - 1) // 2
    if max_h_transfer is not None and not 0 <= max_h_transfer <= available_h_transfer:
        raise ValueError(
            f"max_h_transfer={max_h_transfer} is outside the DAG-supported range "
            f"0..{available_h_transfer}"
        )

    if max_h_transfer is None:
        allowed = set().union(*idx_by_h_delta) if idx_by_h_delta else set()
    else:
        allowed = set(idx_by_h_delta[0])
        for h in range(1, max_h_transfer + 1):
            allowed |= set(idx_by_h_delta[2 * h - 1])
            allowed |= set(idx_by_h_delta[2 * h])
    return sorted(allowed) if allowed else [0]


def _dag_ion_mzs(
    theo_mzs: th.Tensor, prec_mask: th.Tensor, prec_type: str, include_cmf: bool = True
) -> th.Tensor:
    """Flatten a DAG's neutral fragment masses into candidate ion-form m/z values.

    Mirrors the CMF/std union built in dag.stats.compute_frag_peak_stats: a true peak
    counts as DAG-explainable if it matches either the charge-migration-fragment (CMF)
    or the direct-charge-retention (std) ion-mass convention for this precursor type.

    Args:
        theo_mzs: Neutral fragment masses of shape (num_formulas, max_isotope_peaks).
        prec_mask: 1.0 where the corresponding entry in theo_mzs is a real (non-padding)
            isotope peak, 0.0 otherwise. Same shape as theo_mzs.
        prec_type: Precursor adduct type, e.g. "[M+H]+".
        include_cmf: If False, only the std (direct-charge-retention) ion-mass convention
            is used, halving the candidate m/z set (CMF and std differ for every adduct type
            in PREC_TYPE_TO_CMF_MASS_DIFF, including [M+H]+/[M-H]-).

    Returns:
        Flat 1-D tensor of candidate ion m/z values.
    """
    std_diff = PREC_TYPE_TO_MASS_DIFF[prec_type]
    mzs_std = theo_mzs + prec_mask * std_diff
    if not include_cmf:
        return mzs_std.flatten()
    cmf_diff = PREC_TYPE_TO_CMF_MASS_DIFF.get(prec_type, PREC_TYPE_TO_MASS_DIFF[prec_type])
    mzs_cmf = theo_mzs + prec_mask * cmf_diff
    return th.cat([mzs_cmf.flatten(), mzs_std.flatten()])


def _oracle_scores_for_spectrum(
    true_mzs: th.Tensor,
    true_ints: th.Tensor,
    dag_ion_mzs: th.Tensor,
    remove_prec_peak: bool,
    prec_mz: float,
    tolerance: float = TOLERANCE,
    relative: bool = True,
    tolerance_min_mz: float = TOLERANCE_MIN_MZ,
    sqrt: bool = False,
) -> tuple[float, float, int, int]:
    """Score the DAG-achievable oracle prediction for a single spectrum.

    Args:
        true_mzs: True peak m/z values of shape (N,).
        true_ints: True peak intensities of shape (N,).
        dag_ion_mzs: Candidate ion m/z values implied by the DAG, shape (M,).
        remove_prec_peak: Whether to zero the precursor peak before scoring (matches
            the eval_hun_remove_prec_peaks variant; default eval leaves it in).
        prec_mz: Precursor m/z, used only when remove_prec_peak is True.
        tolerance: m/z match tolerance -- a ratio if relative=True, else absolute Da.
        relative: Whether tolerance is a relative ratio (True) or absolute Da (False).
        tolerance_min_mz: Divisor floor for the relative tolerance. Ignored if relative=False.
        sqrt: If True, sqrt-transform intensities before scoring (matches the cos_hun_sqrt /
            jss_hun_sqrt eval variant, which de-emphasizes the dominant peak). The match mask
            itself is unaffected -- only the intensities fed into the Hungarian/cosine/JS
            calculation change.

    Returns:
        Tuple of (cos_hun_opt, jss_hun_opt in [0, 1], n_achievable_peaks, n_true_peaks).
    """
    match_mask = calculate_match_mzs(
        true_mzs,
        dag_ion_mzs,
        tolerance=tolerance,
        relative=relative,
        tolerance_min_mz=tolerance_min_mz,
    )
    true_match_mask = match_mask.any(dim=1)
    # Oracle predicted spectrum: true intensity on DAG-achievable peaks, zero elsewhere.
    pred_ints = true_ints * true_match_mask.float()
    # pred lives on the same index space as true, so a matched pair is only ever (i, i).
    eye_mask = th.eye(true_ints.shape[0], dtype=th.bool)

    if remove_prec_peak:
        prec_mask = calculate_match_mzs(
            true_mzs,
            th.tensor([prec_mz], dtype=true_mzs.dtype),
            tolerance=tolerance,
            relative=relative,
            tolerance_min_mz=tolerance_min_mz,
        ).any(dim=1)
    else:
        prec_mask = th.zeros_like(true_match_mask)

    score_true_ints, score_pred_ints = (
        (th.sqrt(true_ints), th.sqrt(pred_ints)) if sqrt else (true_ints, pred_ints)
    )
    cos_opt = cos_hun_helper(
        score_true_ints,
        score_pred_ints,
        eye_mask,
        true_match_mask,
        true_match_mask,
        remove_prec_peak,
        prec_mask,
        prec_mask,
    )
    jss_opt = jss_hun_helper(
        score_true_ints,
        score_pred_ints,
        eye_mask,
        true_match_mask,
        true_match_mask,
        remove_prec_peak,
        prec_mask,
        prec_mask,
        LOG_MIN,
    )
    return (
        float(cos_opt),
        float(jss_opt) / math.log(2.0),
        int(true_match_mask.sum()),
        int(true_match_mask.shape[0]),
    )


def _score_all_tolerances(
    true_mzs: th.Tensor,
    true_ints: th.Tensor,
    dag_ion_mzs: th.Tensor,
    remove_prec_peak: bool,
    prec_mz: float,
    inst_tol: float,
    inst_tol_min_mz: float,
) -> dict[str, float]:
    """Score one (already-filtered) spectrum at all four tolerance conventions.

    Args:
        true_mzs: Filtered true peak m/z values of shape (N,).
        true_ints: Filtered true peak intensities of shape (N,).
        dag_ion_mzs: Candidate ion m/z values implied by the DAG, shape (M,).
        remove_prec_peak: Whether to zero the precursor peak before scoring.
        prec_mz: Precursor m/z, used only when remove_prec_peak is True.
        inst_tol: Per-instrument relative tolerance for the inst_tol variant.
        inst_tol_min_mz: Per-instrument tolerance floor for the inst_tol variant.

    Returns:
        Dict keyed by the entries of METRIC_BASES. When N == 0 (the filter emptied the
        spectrum), cos/jss metrics are NaN (excluded from means) and peak counts are 0.
    """
    if true_mzs.numel() == 0:
        return {b: (float("nan") if b.startswith(("cos", "jss")) else 0) for b in METRIC_BASES}

    cos_opt, jss_opt, n_ach, n_true = _oracle_scores_for_spectrum(
        true_mzs,
        true_ints,
        dag_ion_mzs,
        remove_prec_peak,
        prec_mz,
        tolerance=TOLERANCE,
        relative=True,
        tolerance_min_mz=TOLERANCE_MIN_MZ,
    )
    cos_it, jss_it, n_ach_it, _ = _oracle_scores_for_spectrum(
        true_mzs,
        true_ints,
        dag_ion_mzs,
        remove_prec_peak,
        prec_mz,
        tolerance=inst_tol,
        relative=True,
        tolerance_min_mz=inst_tol_min_mz,
    )
    cos_da, jss_da, n_ach_da, _ = _oracle_scores_for_spectrum(
        true_mzs,
        true_ints,
        dag_ion_mzs,
        remove_prec_peak,
        prec_mz,
        tolerance=ABS_TOLERANCE_DA,
        relative=False,
    )
    cos_da5, jss_da5, n_ach_da5, _ = _oracle_scores_for_spectrum(
        true_mzs,
        true_ints,
        dag_ion_mzs,
        remove_prec_peak,
        prec_mz,
        tolerance=ABS_TOLERANCE_DA_LOOSE,
        relative=False,
    )
    cos_sqrt, jss_sqrt, _, _ = _oracle_scores_for_spectrum(
        true_mzs,
        true_ints,
        dag_ion_mzs,
        remove_prec_peak,
        prec_mz,
        tolerance=TOLERANCE,
        relative=True,
        tolerance_min_mz=TOLERANCE_MIN_MZ,
        sqrt=True,
    )
    return {
        "cos_hun_opt": cos_opt,
        "jss_hun_opt": jss_opt,
        "cos_hun_inst_tol_opt": cos_it,
        "jss_hun_inst_tol_opt": jss_it,
        "cos_hun_0p01da_opt": cos_da,
        "jss_hun_0p01da_opt": jss_da,
        "cos_hun_0p05da_opt": cos_da5,
        "jss_hun_0p05da_opt": jss_da5,
        "cos_hun_sqrt_opt": cos_sqrt,
        "jss_hun_sqrt_opt": jss_sqrt,
        "n_true_peaks": n_true,
        "n_achievable_peaks": n_ach,
        "n_achievable_peaks_inst_tol": n_ach_it,
        "n_achievable_peaks_0p01da": n_ach_da,
        "n_achievable_peaks_0p05da": n_ach_da5,
    }


def _process_mol_group(
    mol_id: int,
    group_rows: list[dict],
    frag_dp: str,
    is_compressed: bool,
    remove_prec_peak: bool,
    max_h_transfer: int | None,
    include_cmf: bool = True,
) -> list[dict]:
    """Load one molecule's DAG and score every one of its spectra against it."""
    try:
        dag_d = dag_format_utils.load_frag_d(mol_id, frag_dp, is_compressed=is_compressed)
    except FileNotFoundError:
        dag_d = None  # molecule has no DAG on disk (folder backend raises rather than None)
    if dag_d is None:
        return []

    theo_mzs_full = dag_d["formula_peak_mzs"]
    theo_probs_full = dag_d["formula_peak_probs"]
    allowed_idx = _allowed_formula_idxs(dag_d["idx_by_h_delta"], max_h_transfer)
    idx_t = th.tensor(allowed_idx, dtype=th.long)
    theo_mzs = theo_mzs_full.index_select(0, idx_t)
    prec_mask = (theo_probs_full.index_select(0, idx_t) > 0.0).float()

    results = []
    for row in group_rows:
        peaks = row["peaks"]
        if not peaks:
            continue
        true_mzs_raw, true_ints_raw = zip(*peaks)
        true_mzs = th.tensor(true_mzs_raw, dtype=th.float32)
        true_ints = th.tensor(true_ints_raw, dtype=th.float32)
        inst_tol, inst_tol_min_mz = _tolerance_for_inst_type(row["inst_type"])

        dag_ion_mzs = _dag_ion_mzs(theo_mzs, prec_mask, row["prec_type"], include_cmf=include_cmf)

        # Build the denoise masks over the true peaks (see FILTER_TIERS).
        mzs_np = np.asarray(true_mzs_raw)
        prec_cutoff = row["prec_mz"] + max(row["prec_mz"], inst_tol_min_mz) * inst_tol
        keep_above_prec = mzs_np <= prec_cutoff
        n_above_prec_removed = int((~keep_above_prec).sum())

        # Tier 2 removes every UWPR mass match. Tier 3 removes only peaks classified as
        # contaminant_only by the DAG used to produce the feasibility CSV.
        contam_mask = np.zeros(len(mzs_np), dtype=bool)
        for cmz in row.get("contam_mzs", []):
            contam_mask |= np.abs(mzs_np - cmz) <= 1e-4
        contaminant_only_mask = np.zeros(len(mzs_np), dtype=bool)
        for cmz in row.get("contaminant_only_mzs", []):
            contaminant_only_mask |= np.abs(mzs_np - cmz) <= 1e-4
        dag_expl_mask = (
            calculate_match_mzs(
                true_mzs,
                dag_ion_mzs,
                tolerance=TOLERANCE,
                relative=True,
                tolerance_min_mz=TOLERANCE_MIN_MZ,
            )
            .any(dim=1)
            .numpy()
        )
        n_uwpr_matched = int(contam_mask.sum())
        n_uwpr_explainable = int((contam_mask & dag_expl_mask).sum())

        tier_masks = {
            "0_raw": np.ones(len(mzs_np), dtype=bool),
            "1_above_prec": keep_above_prec,
            "2_uwpr_rm": keep_above_prec & ~contam_mask,
            "3_uwpr_keepexpl": keep_above_prec & ~contaminant_only_mask,
        }

        record = {
            "spec_id": row["spec_id"],
            "mol_id": mol_id,
            "inst_type": row["inst_type"],
            "frag_mode": row["frag_mode"],
            "prec_type": row["prec_type"],
            "n_above_prec_removed": n_above_prec_removed,
            "n_uwpr_matched": n_uwpr_matched,
            "n_uwpr_explainable": n_uwpr_explainable,
            "n_uwpr_contaminant_only": int(contaminant_only_mask.sum()),
            "dag_num_nodes": dag_d["dag_num_nodes"],
        }
        for tier, mask in tier_masks.items():
            keep_t = th.tensor(mask)
            metrics = _score_all_tolerances(
                true_mzs[keep_t],
                true_ints[keep_t],
                dag_ion_mzs,
                remove_prec_peak,
                row["prec_mz"],
                inst_tol,
                inst_tol_min_mz,
            )
            for base, val in metrics.items():
                record[f"{base}__{tier}"] = val
        results.append(record)
    return results


def score_dag(
    spec_df: pd.DataFrame,
    frag_dp: str,
    remove_prec_peak: bool = False,
    max_h_transfer: int | None = None,
    include_cmf: bool = True,
    n_jobs: int = parallel_utils.JOBLIB_N_JOBS,
) -> pd.DataFrame:
    """Compute oracle cos_hun/jss_hun for every spectrum in spec_df against one DAG set.

    Args:
        spec_df: Spectra with columns spec_id, mol_id, peaks, prec_mz, prec_type,
            inst_type, frag_mode.
        frag_dp: Directory (or .tar/.zip archive) of precomputed per-molecule DAG pickles.
        remove_prec_peak: Whether to zero the precursor peak before scoring.
        max_h_transfer: Cap on |H-transfer| considered achievable. None uses the full
            H-transfer budget the DAG was built with.
        include_cmf: Whether to union the charge-migration-fragment ion-mass convention
            with the direct-charge-retention (std) convention. False uses std only.
        n_jobs: Number of parallel worker processes.

    Returns:
        Per-spectrum DataFrame with tier-suffixed score/coverage columns (e.g. cos_hun_opt__0_raw)
        for each filter tier in FILTER_TIERS, plus n_above_prec_removed and n_uwpr_removed.
    """
    groups = spec_df.groupby("mol_id")
    sample_mol_id = int(next(iter(groups.groups.keys())))
    is_compressed = _detect_is_compressed(frag_dp, sample_mol_id)

    mol_inputs = [(int(mol_id), group.to_dict("records")) for mol_id, group in groups]
    worker = functools.partial(
        _process_mol_group,
        frag_dp=frag_dp,
        is_compressed=is_compressed,
        remove_prec_peak=remove_prec_peak,
        max_h_transfer=max_h_transfer,
        include_cmf=include_cmf,
    )
    log.info(
        "Scoring %d molecules (%d spectra) against DAGs in %s",
        len(mol_inputs),
        len(spec_df),
        frag_dp,
    )
    per_mol_results = parallel_utils.par_apply(mol_inputs, worker, need_unpack=True, n_jobs=n_jobs)

    n_missing_dag = sum(1 for r in per_mol_results if len(r) == 0)
    if n_missing_dag > 0:
        log.warning(
            "%d/%d molecules had no DAG found under %s", n_missing_dag, len(mol_inputs), frag_dp
        )

    rows = [r for mol_results in per_mol_results for r in mol_results]
    return pd.DataFrame(rows)


# Score metric bases (cos/jss variants) reported per tier. Subset of METRIC_BASES; excludes the
# peak-count columns, which are summarized separately as coverage.
SCORE_BASES = [
    "cos_hun_opt",
    "jss_hun_opt",
    "cos_hun_inst_tol_opt",
    "jss_hun_inst_tol_opt",
    "cos_hun_0p01da_opt",
    "jss_hun_0p01da_opt",
    "cos_hun_0p05da_opt",
    "jss_hun_0p05da_opt",
    "cos_hun_sqrt_opt",
    "jss_hun_sqrt_opt",
]


def _tier_mean_table(results_df: pd.DataFrame, bases: list[str]) -> pd.DataFrame:
    """Build a (rows = FILTER_TIERS) x (cols = bases) mean table, stacking the filter tiers."""
    data = {b: [results_df[f"{b}__{t}"].mean() for t in FILTER_TIERS] for b in bases}
    return pd.DataFrame(data, index=FILTER_TIERS).round(4)


def summarize(results_df: pd.DataFrame, label: str) -> None:
    """Print the stacked-tier overall summary plus per-inst_type/frag_mode and per-adduct tables."""
    print(f"\n{'=' * 70}\nDAG: {label}\n{'=' * 70}")
    print(f"n spectra scored: {len(results_df)}  (n mols: {results_df['mol_id'].nunique()})")

    print("\nCumulative filter tiers (mean over spectra):")
    print(_tier_mean_table(results_df, SCORE_BASES).to_string())

    cov_rows = []
    for t in FILTER_TIERS:
        num = results_df[f"n_achievable_peaks__{t}"]
        den = results_df[f"n_true_peaks__{t}"]
        cov = (num[den > 0] / den[den > 0]).mean()
        cov_rows.append((t, den.mean(), cov))
    print("\nPer tier: mean n_true_peaks | mean peak coverage (n_achievable/n_true), 10ppm tol:")
    for t, n_true_mean, cov in cov_rows:
        print(f"  {t:<14} n_true={n_true_mean:7.2f}  coverage={cov:.4f}")

    print(
        f"\nMean peaks removed per spectrum: above_prec={results_df['n_above_prec_removed'].mean():.2f}"
        f"  uwpr_matched={results_df['n_uwpr_matched'].mean():.3f}"
        f"  (of which DAG-explainable={results_df['n_uwpr_explainable'].mean():.3f}, "
        f"classified contaminant_only={results_df['n_uwpr_contaminant_only'].mean():.3f})"
    )

    # Per-group breakdowns report the headline cos_hun_opt across the three tiers only, to stay
    # readable (the full metric x tier grid lives in the per-spectrum CSV).
    cos_cols = [f"cos_hun_opt__{t}" for t in FILTER_TIERS]
    print("\nBy inst_type / frag_mode (cos_hun_opt per tier):")
    g = results_df.groupby(["inst_type", "frag_mode"], observed=True)
    breakdown = g[cos_cols].mean().round(4)
    breakdown.columns = FILTER_TIERS
    breakdown["count"] = g.size()
    print(breakdown.to_string())

    print("\nBy adduct (prec_type) (cos_hun_opt per tier):")
    ga = results_df.groupby("prec_type", observed=True)
    adduct_breakdown = ga[cos_cols].mean().round(4)
    adduct_breakdown.columns = FILTER_TIERS
    adduct_breakdown["count"] = ga.size()
    print(adduct_breakdown.to_string())


def _load_uwpr_maps(csv_path: str) -> tuple[dict[int, list[float]], dict[int, list[float]]]:
    """Load all UWPR matches and the conservative ``contaminant_only`` subset."""
    required_cols = ["spec_id", "peak_mz", "feasibility"]
    feasibility_df = pd.read_csv(csv_path, usecols=required_cols)
    if feasibility_df[required_cols].isna().any().any():
        raise ValueError(f"UWPR feasibility CSV contains missing required values: {csv_path}")

    known_labels = {
        "dag_feasible",
        "contaminant_only",
        "no_dag_file",
        "unsupported_prec",
        "missing_metadata",
    }
    unknown_labels = set(feasibility_df["feasibility"].unique()) - known_labels
    if unknown_labels:
        raise ValueError(f"Unknown UWPR feasibility labels: {sorted(unknown_labels)}")

    all_matches = feasibility_df.groupby("spec_id")["peak_mz"].apply(list).to_dict()
    contaminant_only = feasibility_df[
        feasibility_df["feasibility"] == "contaminant_only"
    ]
    conservative_matches = contaminant_only.groupby("spec_id")["peak_mz"].apply(list).to_dict()
    return all_matches, conservative_matches


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--frag_dp",
        action="append",
        required=True,
        help="Directory of precomputed DAG pickles (repeat to compare multiple DAG variants).",
    )
    parser.add_argument(
        "--frag_dp_labels", nargs="+", default=None, help="Labels for each --frag_dp, same order."
    )
    parser.add_argument("--spec_fp", default="data/proc/nist20/spec_df.pkl")
    parser.add_argument(
        "--dsets", "--dset", nargs="+", default=None, help="Filter spec_df to these dataset labels."
    )
    parser.add_argument("--split_dp", default="data/split/nist20_ma_mi_inchikey")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--remove_prec_peak", action="store_true")
    parser.add_argument(
        "--max_h_transfer",
        type=int,
        default=None,
        help="Default: use the DAG's full H-transfer budget.",
    )
    parser.add_argument(
        "--no_cmf",
        action="store_true",
        help="Disable the charge-migration-fragment (CMF) ion-mass convention; match only "
        "the direct-charge-retention (std) convention.",
    )
    parser.add_argument(
        "--limit_mols",
        type=int,
        default=None,
        help="Randomly subsample this many molecules for a quick check.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_jobs", type=int, default=parallel_utils.JOBLIB_N_JOBS)
    parser.add_argument(
        "--out_fp",
        default=None,
        help="Optional path to save per-spectrum results (one CSV per --frag_dp, suffixed by label).",
    )
    parser.add_argument(
        "--uwpr_feasibility_csv",
        default=None,
        help="Optional UWPR feasibility CSV (from scripts/debug/oracle/uwpr_feasibility.py). When set, "
        "tier 2 removes every UWPR mass match (aggressive), while tier 3 removes only rows "
        "labelled 'contaminant_only' (DAG-dependent sensitivity analysis). Without it, both "
        "UWPR tiers are identical to tier 1.",
    )
    args = parser.parse_args()
    args.dsets = resolve_dsets_for_spec_fp(args.spec_fp, args.dsets)

    labels = (
        args.frag_dp_labels
        if args.frag_dp_labels
        else [os.path.basename(os.path.dirname(fp.rstrip("/"))) for fp in args.frag_dp]
    )
    if len(labels) != len(args.frag_dp):
        raise ValueError(
            f"--frag_dp_labels must match --frag_dp in length: {len(labels)} != {len(args.frag_dp)}"
        )

    log.info("Loading %s", args.spec_fp)
    spec_df = pd.read_pickle(args.spec_fp)
    if args.dsets:
        spec_df = spec_df[spec_df["dset"].isin(args.dsets)]
    spec_df = spec_df[spec_df["prec_type"].isin(PREC_TYPE_TO_MASS_DIFF.keys())]
    spec_df = spec_df[spec_df["peaks"].apply(len) > 0]

    if args.split != "all":
        split_fp = os.path.join(args.split_dp, f"{args.split}_ids.csv")
        log.info("Filtering to split=%s via %s", args.split, split_fp)
        split_spec_ids = set(pd.read_csv(split_fp)["spec_id"])
        spec_df = spec_df[spec_df["spec_id"].isin(split_spec_ids)]

    spec_df = spec_df[
        ["spec_id", "mol_id", "peaks", "prec_mz", "prec_type", "inst_type", "frag_mode"]
    ].copy()

    # Keep both the complete mass-match set and the DAG-dependent contaminant_only subset so the
    # aggressive and conservative sensitivity analyses remain explicit.
    if args.uwpr_feasibility_csv is not None:
        log.info("Loading UWPR contaminant hits from %s", args.uwpr_feasibility_csv)
        contam_map, contaminant_only_map = _load_uwpr_maps(args.uwpr_feasibility_csv)
        spec_df["contam_mzs"] = spec_df["spec_id"].map(lambda s: contam_map.get(s, []))
        spec_df["contaminant_only_mzs"] = spec_df["spec_id"].map(
            lambda s: contaminant_only_map.get(s, [])
        )
        n_affected = spec_df["contam_mzs"].apply(bool).sum()
        log.info(
            "UWPR active: %d/%d spectra have >=1 contaminant-matched peak", n_affected, len(spec_df)
        )
    else:
        spec_df["contam_mzs"] = [[] for _ in range(len(spec_df))]
        spec_df["contaminant_only_mzs"] = [[] for _ in range(len(spec_df))]
        log.warning(
            "No --uwpr_feasibility_csv supplied: tiers 2_uwpr_rm and 3_uwpr_keepexpl will be "
            "identical to 1_above_prec."
        )

    if args.limit_mols is not None:
        rng = np.random.default_rng(args.seed)
        mol_ids = spec_df["mol_id"].unique()
        keep_mol_ids = rng.choice(mol_ids, size=min(args.limit_mols, len(mol_ids)), replace=False)
        spec_df = spec_df[spec_df["mol_id"].isin(keep_mol_ids)]

    log.info("Scoring %d spectra across %d molecules", len(spec_df), spec_df["mol_id"].nunique())

    all_results = {}
    for frag_dp, label in zip(args.frag_dp, labels):
        results_df = score_dag(
            spec_df,
            frag_dp,
            remove_prec_peak=args.remove_prec_peak,
            max_h_transfer=args.max_h_transfer,
            include_cmf=not args.no_cmf,
            n_jobs=args.n_jobs,
        )
        all_results[label] = results_df
        summarize(results_df, label)
        if args.out_fp:
            base, ext = os.path.splitext(args.out_fp)
            out_fp = f"{base}_{label}{ext or '.csv'}"
            results_df.to_csv(out_fp, index=False)
            log.info("Saved per-spectrum results to %s", out_fp)

    if len(all_results) > 1:
        print(f"\n{'=' * 70}\nComparison: cos_hun_opt per filter tier (mean)\n{'=' * 70}")
        comparison = pd.DataFrame(
            {
                label: [df[f"cos_hun_opt__{t}"].mean() for t in FILTER_TIERS]
                for label, df in all_results.items()
            },
            index=FILTER_TIERS,
        ).round(4)
        print(comparison.to_string())


if __name__ == "__main__":
    main()
