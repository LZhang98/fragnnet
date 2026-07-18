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
    inst_type_loss_tol in config/exp_fraggnn_ma_mi/fraggnn_d3_ma_mi_nist20_amolc_all.yml).
  - cos_hun_0p01da_opt / jss_hun_0p01da_opt: fixed absolute 0.01 Da tolerance.

Additionally, cos_hun_sqrt_opt / jss_hun_sqrt_opt sqrt-transform intensities before scoring
(at the default 10 ppm tolerance), matching the cos_hun_sqrt / jss_hun_sqrt eval variant that
de-emphasizes the dominant peak.

Usage:
    python scripts/eval/dag_oracle_scores.py \\
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
import torch as th

import fragnnet.utils.data_utils as data_utils
import fragnnet.utils.frag_utils as frag_utils
from fragnnet.utils.formula_utils import PREC_TYPE_TO_CMF_MASS_DIFF, PREC_TYPE_TO_MASS_DIFF
from fragnnet.utils.spec_utils import calculate_match_mzs, cos_hun_helper, jss_hun_helper

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TOLERANCE = 1.0e-5  # relative ratio (== 10 ppm), matches default loss_tolerance_rel
TOLERANCE_MIN_MZ = 200.0  # matches config/template.yml loss_tolerance_min_mz
LOG_MIN = 1.0e-9  # matches config/template.yml log_min
# Per-instrument tolerance override, matches inst_type_loss_tol in
# config/exp_fraggnn_ma_mi/fraggnn_d3_ma_mi_nist20_amolc_all.yml. Instrument types not
# listed here (IT, QQQ, Other, ...) fall back to the global TOLERANCE/TOLERANCE_MIN_MZ.
INST_TYPE_LOSS_TOL = {
    "FT": {"rel": 1.0e-5, "min_mz": 200.0},  # 10 ppm, floor 200 Da
    "QTOF": {"rel": 2.0e-5, "min_mz": 750.0},  # 20 ppm, floor 750 Da
}
ABS_TOLERANCE_DA = 0.01  # fixed absolute Da tolerance, reproduces the cos_sim_0.01 convention


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
    if os.path.isfile(os.path.join(frag_dp, f"{int(sample_mol_id)}.pkl.bz2")):
        return True
    if os.path.isfile(os.path.join(frag_dp, f"{int(sample_mol_id)}.pkl")):
        return False
    raise FileNotFoundError(
        f"Could not find DAG file for mol_id={sample_mol_id} under {frag_dp} "
        "(tried .pkl and .pkl.bz2)"
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

    Mirrors the CMF/std union built in frag_utils.compute_frag_peak_stats: a true peak
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
        true_mzs, dag_ion_mzs, tolerance=tolerance, relative=relative, tolerance_min_mz=tolerance_min_mz
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
        score_true_ints, score_pred_ints, eye_mask, true_match_mask, true_match_mask,
        remove_prec_peak, prec_mask, prec_mask,
    )
    jss_opt = jss_hun_helper(
        score_true_ints, score_pred_ints, eye_mask, true_match_mask, true_match_mask,
        remove_prec_peak, prec_mask, prec_mask, LOG_MIN,
    )
    return (
        float(cos_opt),
        float(jss_opt) / math.log(2.0),
        int(true_match_mask.sum()),
        int(true_match_mask.shape[0]),
    )


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
        dag_d = frag_utils.load_frag_d(mol_id, frag_dp, is_compressed=is_compressed)
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

        # Optionally drop UWPR contaminant_only peaks (matched to a known contaminant m/z and
        # unexplainable by the DAG) so the oracle ceiling is measured on the denoised spectrum.
        contam_mzs = row["contam_mzs"]
        n_uwpr_removed = 0
        if contam_mzs:
            mzs_np = np.asarray(true_mzs_raw)
            keep = np.ones(len(mzs_np), dtype=bool)
            for cmz in contam_mzs:
                keep &= np.abs(mzs_np - cmz) > 1e-4
            n_uwpr_removed = int((~keep).sum())
            if n_uwpr_removed:
                if not keep.any():
                    continue  # entire spectrum is contaminant; oracle undefined, skip
                keep_t = th.tensor(keep)
                true_mzs = true_mzs[keep_t]
                true_ints = true_ints[keep_t]

        dag_ion_mzs = _dag_ion_mzs(theo_mzs, prec_mask, row["prec_type"], include_cmf=include_cmf)

        cos_opt, jss_opt, n_achievable, n_true = _oracle_scores_for_spectrum(
            true_mzs, true_ints, dag_ion_mzs, remove_prec_peak, row["prec_mz"],
            tolerance=TOLERANCE, relative=True, tolerance_min_mz=TOLERANCE_MIN_MZ,
        )
        inst_tol, inst_tol_min_mz = _tolerance_for_inst_type(row["inst_type"])
        cos_opt_it, jss_opt_it, n_achievable_it, _ = _oracle_scores_for_spectrum(
            true_mzs, true_ints, dag_ion_mzs, remove_prec_peak, row["prec_mz"],
            tolerance=inst_tol, relative=True, tolerance_min_mz=inst_tol_min_mz,
        )
        cos_opt_da, jss_opt_da, n_achievable_da, _ = _oracle_scores_for_spectrum(
            true_mzs, true_ints, dag_ion_mzs, remove_prec_peak, row["prec_mz"],
            tolerance=ABS_TOLERANCE_DA, relative=False,
        )
        cos_opt_sqrt, jss_opt_sqrt, _, _ = _oracle_scores_for_spectrum(
            true_mzs, true_ints, dag_ion_mzs, remove_prec_peak, row["prec_mz"],
            tolerance=TOLERANCE, relative=True, tolerance_min_mz=TOLERANCE_MIN_MZ, sqrt=True,
        )

        results.append(
            {
                "spec_id": row["spec_id"],
                "mol_id": mol_id,
                "inst_type": row["inst_type"],
                "frag_mode": row["frag_mode"],
                "prec_type": row["prec_type"],
                "cos_hun_opt": cos_opt,
                "jss_hun_opt": jss_opt,
                "cos_hun_inst_tol_opt": cos_opt_it,
                "jss_hun_inst_tol_opt": jss_opt_it,
                "cos_hun_0p01da_opt": cos_opt_da,
                "jss_hun_0p01da_opt": jss_opt_da,
                "cos_hun_sqrt_opt": cos_opt_sqrt,
                "jss_hun_sqrt_opt": jss_opt_sqrt,
                "n_true_peaks": n_true,
                "n_achievable_peaks": n_achievable,
                "n_achievable_peaks_inst_tol": n_achievable_it,
                "n_achievable_peaks_0p01da": n_achievable_da,
                "n_uwpr_removed": n_uwpr_removed,
                "dag_num_nodes": dag_d["dag_num_nodes"],
            }
        )
    return results


def score_dag(
    spec_df: pd.DataFrame,
    frag_dp: str,
    remove_prec_peak: bool = False,
    max_h_transfer: int | None = None,
    include_cmf: bool = True,
    n_jobs: int = data_utils.JOBLIB_N_JOBS,
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
        Per-spectrum DataFrame with cos_hun_opt, jss_hun_opt, and peak-coverage columns.
    """
    groups = spec_df.groupby("mol_id")
    sample_mol_id = int(next(iter(groups.groups.keys())))
    is_compressed = _detect_is_compressed(frag_dp, sample_mol_id)

    mol_inputs = [
        (int(mol_id), group.to_dict("records")) for mol_id, group in groups
    ]
    worker = functools.partial(
        _process_mol_group,
        frag_dp=frag_dp,
        is_compressed=is_compressed,
        remove_prec_peak=remove_prec_peak,
        max_h_transfer=max_h_transfer,
        include_cmf=include_cmf,
    )
    log.info("Scoring %d molecules (%d spectra) against DAGs in %s", len(mol_inputs), len(spec_df), frag_dp)
    per_mol_results = data_utils.par_apply(mol_inputs, worker, need_unpack=True, n_jobs=n_jobs)

    n_missing_dag = sum(1 for r in per_mol_results if len(r) == 0)
    if n_missing_dag > 0:
        log.warning("%d/%d molecules had no DAG found under %s", n_missing_dag, len(mol_inputs), frag_dp)

    rows = [r for mol_results in per_mol_results for r in mol_results]
    return pd.DataFrame(rows)


SCORE_COLS = [
    "cos_hun_opt",
    "jss_hun_opt",
    "cos_hun_inst_tol_opt",
    "jss_hun_inst_tol_opt",
    "cos_hun_0p01da_opt",
    "jss_hun_0p01da_opt",
    "cos_hun_sqrt_opt",
    "jss_hun_sqrt_opt",
]


def summarize(results_df: pd.DataFrame, label: str) -> None:
    """Print overall, per-inst_type/frag_mode, and per-adduct summaries for one DAG's results."""
    print(f"\n{'=' * 70}\nDAG: {label}\n{'=' * 70}")
    print(f"n spectra scored: {len(results_df)}  (n mols: {results_df['mol_id'].nunique()})")
    overall = results_df[SCORE_COLS].agg(["mean", "median"]).round(4)
    print(overall.to_string())
    print(
        f"mean peak coverage (n_achievable/n_true), 10ppm tol: "
        f"{(results_df['n_achievable_peaks'] / results_df['n_true_peaks']).mean():.4f}"
    )

    print("\nBy inst_type / frag_mode:")
    breakdown = (
        results_df.groupby(["inst_type", "frag_mode"], observed=True)[SCORE_COLS]
        .agg(["mean", "count"])
        .round(4)
    )
    print(breakdown.to_string())

    print("\nBy adduct (prec_type):")
    adduct_breakdown = (
        results_df.groupby("prec_type", observed=True)[SCORE_COLS]
        .agg(["mean", "count"])
        .round(4)
    )
    print(adduct_breakdown.to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--frag_dp", action="append", required=True,
        help="Directory of precomputed DAG pickles (repeat to compare multiple DAG variants).",
    )
    parser.add_argument("--frag_dp_labels", nargs="+", default=None, help="Labels for each --frag_dp, same order.")
    parser.add_argument("--spec_fp", default="data/proc/nist20/spec_df.pkl")
    parser.add_argument("--split_dp", default="data/split/nist20_ma_mi_inchikey")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--remove_prec_peak", action="store_true")
    parser.add_argument("--max_h_transfer", type=int, default=None, help="Default: use the DAG's full H-transfer budget.")
    parser.add_argument(
        "--no_cmf", action="store_true",
        help="Disable the charge-migration-fragment (CMF) ion-mass convention; match only "
        "the direct-charge-retention (std) convention.",
    )
    parser.add_argument("--limit_mols", type=int, default=None, help="Randomly subsample this many molecules for a quick check.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_jobs", type=int, default=data_utils.JOBLIB_N_JOBS)
    parser.add_argument("--out_fp", default=None, help="Optional path to save per-spectrum results (one CSV per --frag_dp, suffixed by label).")
    parser.add_argument(
        "--uwpr_feasibility_csv", default=None,
        help="Optional UWPR feasibility CSV (from scripts/eval/uwpr_feasibility.py). When set, "
        "peaks flagged 'contaminant_only' are removed per spectrum before scoring, giving the "
        "denoised oracle ceiling; the per-spectrum output gains an n_uwpr_removed column.",
    )
    args = parser.parse_args()

    labels = args.frag_dp_labels if args.frag_dp_labels else [os.path.basename(os.path.dirname(fp.rstrip("/"))) for fp in args.frag_dp]
    if len(labels) != len(args.frag_dp):
        raise ValueError(f"--frag_dp_labels must match --frag_dp in length: {len(labels)} != {len(args.frag_dp)}")

    log.info("Loading %s", args.spec_fp)
    spec_df = pd.read_pickle(args.spec_fp)
    spec_df = spec_df[spec_df["prec_type"].isin(PREC_TYPE_TO_MASS_DIFF.keys())]
    spec_df = spec_df[spec_df["peaks"].apply(len) > 0]

    if args.split != "all":
        split_fp = os.path.join(args.split_dp, f"{args.split}_ids.csv")
        log.info("Filtering to split=%s via %s", args.split, split_fp)
        split_spec_ids = set(pd.read_csv(split_fp)["spec_id"])
        spec_df = spec_df[spec_df["spec_id"].isin(split_spec_ids)]

    spec_df = spec_df[["spec_id", "mol_id", "peaks", "prec_mz", "prec_type", "inst_type", "frag_mode"]].copy()

    # Per-spectrum UWPR contaminant m/z to strip before scoring (empty list = no removal).
    if args.uwpr_feasibility_csv is not None:
        log.info("Loading UWPR feasibility from %s", args.uwpr_feasibility_csv)
        feas = pd.read_csv(args.uwpr_feasibility_csv, usecols=["spec_id", "peak_mz", "feasibility"])
        contam = feas[feas["feasibility"] == "contaminant_only"]
        contam_map = contam.groupby("spec_id")["peak_mz"].apply(list).to_dict()
        spec_df["contam_mzs"] = spec_df["spec_id"].map(lambda s: contam_map.get(s, []))
        n_affected = spec_df["contam_mzs"].apply(bool).sum()
        log.info("UWPR removal active: %d/%d spectra have >=1 contaminant_only peak", n_affected, len(spec_df))
    else:
        spec_df["contam_mzs"] = [[] for _ in range(len(spec_df))]

    if args.limit_mols is not None:
        rng = np.random.default_rng(args.seed)
        mol_ids = spec_df["mol_id"].unique()
        keep_mol_ids = rng.choice(mol_ids, size=min(args.limit_mols, len(mol_ids)), replace=False)
        spec_df = spec_df[spec_df["mol_id"].isin(keep_mol_ids)]

    log.info("Scoring %d spectra across %d molecules", len(spec_df), spec_df["mol_id"].nunique())

    all_results = {}
    for frag_dp, label in zip(args.frag_dp, labels):
        results_df = score_dag(
            spec_df, frag_dp,
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
        print(f"\n{'=' * 70}\nComparison (mean)\n{'=' * 70}")
        comparison = pd.DataFrame(
            {label: df[SCORE_COLS].mean() for label, df in all_results.items()}
        ).T.round(4)
        print(comparison.to_string())


if __name__ == "__main__":
    main()
