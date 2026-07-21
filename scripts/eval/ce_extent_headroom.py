"""Measure whether real spectra shift intensity toward more-fragmented species as CE rises.

This is a headroom check for any CE-conditioned "fragmentation extent" mechanism (the CFM-style
flow head's stop probability, ``depth_ce_bias``, a CE x num_cut_bonds bias). All of those encode
the same prior:

    as collision energy rises, mass should move from the precursor toward smaller fragments.

If that trend is not present in the DATA, no such mechanism can help, and the whole line is dead
before a single GPU-hour is spent. This script measures the trend directly from spec_df + the DAG,
with NO model and NO training involved.

Coordinate
----------
Fragmentation extent is measured as the **retained heavy-atom fraction** of the formula that
explains a peak:

    retained_frac(formula) = heavy_atoms(formula) / heavy_atoms(precursor)

1.0 is the intact precursor, small values are heavily fragmented. This is used instead of DAG
``depth`` on purpose: depth is not a fragmentation-extent proxy in this codebase (a 4-cut SMARTS
xanthone node and a 1-cut node are both depth 1 -- see the num_cut_bonds comment in
fragnnet_model.py), and the depth feature is a multi-hot bitmask of every depth a fragment was
reached at, not a scalar. Retained heavy-atom fraction is a property of the formula alone: it is
unaffected by H-shifts, needs no node->formula attribution, and is well defined for both the true
and any predicted spectrum. The denominator is taken as the max heavy-atom count over the DAG's
formulas, which is the root (precursor) fragment by construction.

Ambiguity
---------
A true peak can match several DAG formulas with different heavy-atom counts. Rather than pick one,
every statistic is reported as a **bracket**: ``min`` attributes each peak to the smallest matching
formula, ``max`` to the largest. The truth lies between. If the bracket is tight the ambiguity does
not matter; if it is wide, the reported trend is not identified and that is itself the finding.

Reading the result
------------------
The decisive column is mean intensity-weighted retained_frac per NCE bin:

  - Falls materially with NCE  -> the CE->extent prior has headroom. Build the mechanism, and the
    size of the fall is the effect you are chasing.
  - Flat across NCE            -> no CE-dependent extent mis-allocation exists to model. The flow
    head's stop mechanism, depth_ce_bias, and a CE x cut-count bias are all dead on arrival, and
    the CE degradation lives somewhere else (H-shift states, CMF/CRF selection, peak identity).

Only peaks the DAG can explain are counted, since that is the only mass any of these mechanisms
can redistribute.

Usage:
    python scripts/eval/ce_extent_headroom.py \\
        --frag_dp data/frag/nist20_ma_mi_d3_h4_isoFalse_smpFalse/dags \\
        --split val --limit_mols 2000
"""

from __future__ import annotations

import argparse
import functools
import logging
import os
import re
import sys

import numpy as np
import pandas as pd
import torch as th

import fragnnet.utils.data_utils as data_utils
import fragnnet.utils.frag_utils as frag_utils
from fragnnet.utils.formula_utils import PREC_TYPE_TO_MASS_DIFF
from fragnnet.utils.spec_utils import calculate_match_mzs

# Reuse the DAG/ion-mass machinery rather than reimplementing the CMF/std union.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dag_oracle_scores import (  # noqa: E402
    _allowed_formula_idxs,
    _dag_ion_mzs,
    _detect_is_compressed,
    _tolerance_for_inst_type,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

NCE_BINS = [0, 20, 40, 60, 80, 100, 120, 160, 1000]
_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")


def formula_heavy_atoms(formula: str) -> int:
    """Count non-hydrogen atoms in a formula string.

    Args:
        formula: Formula string as stored in the DAG's ``idx_to_formula``, e.g. "C10H10BrN".
            The empty string (the NULL formula row) counts as 0.

    Returns:
        Number of heavy (non-H) atoms.
    """
    total = 0
    for element, count in _FORMULA_TOKEN.findall(formula):
        if element == "H":
            continue
        total += int(count) if count else 1
    return total


def _extent_for_spectrum(
    true_mzs: th.Tensor,
    true_ints: th.Tensor,
    dag_ion_mzs: th.Tensor,
    heavy_per_ion: th.Tensor,
    n_heavy_prec: int,
    tolerance: float,
    tolerance_min_mz: float,
) -> dict[str, float] | None:
    """Intensity-weighted retained heavy-atom fraction for one spectrum.

    Args:
        true_mzs: True peak m/z of shape (N,).
        true_ints: True peak intensities of shape (N,).
        dag_ion_mzs: Candidate DAG ion m/z of shape (M,).
        heavy_per_ion: Heavy-atom count of the formula behind each candidate ion, shape (M,).
        n_heavy_prec: Heavy-atom count of the precursor (max over DAG formulas).
        tolerance: Relative m/z match tolerance.
        tolerance_min_mz: Divisor floor for the relative tolerance.

    Returns:
        Dict with the min/max-attribution mean retained fractions and the explained
        intensity fraction, or None if the DAG explains no peak in this spectrum.
    """
    match_mask = calculate_match_mzs(
        true_mzs,
        dag_ion_mzs,
        tolerance=tolerance,
        relative=True,
        tolerance_min_mz=tolerance_min_mz,
    )
    matched = match_mask.any(dim=1)
    total_int = float(true_ints.sum())
    if not bool(matched.any()) or total_int <= 0.0:
        return None

    # Per matched peak, the smallest / largest heavy-atom count among the formulas that explain it.
    big = th.full_like(heavy_per_ion, float(n_heavy_prec) + 1.0)
    heavy_min = th.where(match_mask, heavy_per_ion.unsqueeze(0), big.unsqueeze(0)).min(dim=1).values
    heavy_max = th.where(match_mask, heavy_per_ion.unsqueeze(0), th.zeros_like(big).unsqueeze(0)).max(dim=1).values

    ints = true_ints[matched]
    weight = ints / ints.sum()
    frac_min = (heavy_min[matched] / n_heavy_prec * weight).sum()
    frac_max = (heavy_max[matched] / n_heavy_prec * weight).sum()

    # Ambiguity diagnostic: mean number of distinct heavy-atom counts a matched peak admits.
    n_candidates = match_mask[matched].sum(dim=1).float().mean()

    return {
        "retained_frac_min": float(frac_min),
        "retained_frac_max": float(frac_max),
        "explained_int_frac": float(ints.sum()) / total_int,
        "n_matched_peaks": int(matched.sum()),
        "n_true_peaks": int(matched.shape[0]),
        "mean_ion_candidates": float(n_candidates),
        "n_heavy_prec": n_heavy_prec,
    }


def load_pred_peaks(pred_hdf5: str) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Load per-spectrum predicted peaks from a predictions HDF5.

    Args:
        pred_hdf5: Path to a predictions_test.hdf5 written by run_save_inference_eval.py.
            ``pred_mzs``/``pred_ints`` are flat concatenations sliced by start_points/end_points.

    Returns:
        Mapping from spec_id to (pred_mzs, pred_ints) arrays.
    """
    import h5py

    with h5py.File(pred_hdf5, "r") as f:
        spec_ids = f["spec_id"][:]
        starts = f["start_points"][:]
        ends = f["end_points"][:]
        mzs = f["pred_mzs"][:]
        ints = f["pred_ints"][:]
    return {
        int(sid): (mzs[s:e], ints[s:e]) for sid, s, e in zip(spec_ids, starts, ends)
    }


def _process_mol_group(
    mol_id: int,
    group_rows: list[dict],
    frag_dp: str,
    is_compressed: bool,
    max_h_transfer: int | None,
    include_cmf: bool,
) -> list[dict]:
    """Load one molecule's DAG and compute the extent statistic for each of its spectra.

    When a row carries a ``pred_peaks`` entry, the predicted spectrum is pushed through the
    identical DAG-ion attribution as the true spectrum, so the two retained fractions are
    directly comparable.
    """
    dag_d = frag_utils.load_frag_d(mol_id, frag_dp, is_compressed=is_compressed)
    if dag_d is None:
        return []

    allowed_idx = _allowed_formula_idxs(dag_d["idx_by_h_delta"], max_h_transfer)
    idx_t = th.tensor(allowed_idx, dtype=th.long)
    theo_mzs = dag_d["formula_peak_mzs"].index_select(0, idx_t)
    prec_mask = (dag_d["formula_peak_probs"].index_select(0, idx_t) > 0.0).float()

    idx_to_formula = dag_d["idx_to_formula"]
    heavy = th.tensor(
        [formula_heavy_atoms(idx_to_formula[i]) for i in allowed_idx], dtype=th.float32
    )
    n_heavy_prec = int(heavy.max())
    if n_heavy_prec <= 0:
        return []

    # _dag_ion_mzs flattens (num_formulas, num_isotopes) and, when include_cmf, concatenates the
    # CMF block ahead of the std block. Broadcast heavy counts through the same layout so ion
    # column j always maps back to its formula's heavy-atom count.
    n_iso = theo_mzs.shape[1]
    heavy_flat = heavy.unsqueeze(1).expand(-1, n_iso).flatten()
    heavy_per_ion = th.cat([heavy_flat, heavy_flat]) if include_cmf else heavy_flat

    results = []
    for row in group_rows:
        peaks = row["peaks"]
        if not peaks:
            continue
        true_mzs_raw, true_ints_raw = zip(*peaks)
        true_mzs = th.tensor(true_mzs_raw, dtype=th.float32)
        true_ints = th.tensor(true_ints_raw, dtype=th.float32)
        dag_ion_mzs = _dag_ion_mzs(theo_mzs, prec_mask, row["prec_type"], include_cmf=include_cmf)
        if dag_ion_mzs.shape[0] != heavy_per_ion.shape[0]:
            raise ValueError(
                f"ion/heavy layout mismatch for mol {mol_id}: "
                f"{dag_ion_mzs.shape[0]} ions vs {heavy_per_ion.shape[0]} heavy counts"
            )

        inst_tol, inst_tol_min_mz = _tolerance_for_inst_type(row["inst_type"])
        stats = _extent_for_spectrum(
            true_mzs, true_ints, dag_ion_mzs, heavy_per_ion, n_heavy_prec,
            tolerance=inst_tol, tolerance_min_mz=inst_tol_min_mz,
        )
        if stats is None:
            continue

        if row["pred_peaks"] is not None:
            pred_mzs_np, pred_ints_np = row["pred_peaks"]
            pred_stats = _extent_for_spectrum(
                th.tensor(pred_mzs_np, dtype=th.float32),
                th.tensor(pred_ints_np, dtype=th.float32),
                dag_ion_mzs, heavy_per_ion, n_heavy_prec,
                tolerance=inst_tol, tolerance_min_mz=inst_tol_min_mz,
            )
            if pred_stats is None:
                continue
            stats["pred_retained_frac_min"] = pred_stats["retained_frac_min"]
            stats["pred_retained_frac_max"] = pred_stats["retained_frac_max"]

        stats.update(
            {
                "spec_id": row["spec_id"],
                "mol_id": mol_id,
                "nce": row["nce"],
                "inst_type": row["inst_type"],
                "frag_mode": row["frag_mode"],
                "prec_type": row["prec_type"],
            }
        )
        results.append(stats)
    return results


def load_spec_df(spec_fp: str, split_dp: str, split: str, dset: str | None) -> pd.DataFrame:
    """Load spec_df and filter to a split, matching dag_oracle_scores.main's filtering."""
    log.info("Loading %s", spec_fp)
    spec_df = pd.read_pickle(spec_fp)
    if dset:
        spec_df = spec_df[spec_df["dset"] == dset]
    spec_df = spec_df[spec_df["prec_type"].isin(PREC_TYPE_TO_MASS_DIFF.keys())]
    spec_df = spec_df[spec_df["peaks"].apply(len) > 0]
    spec_df = spec_df[spec_df["nce"].notna()]
    if split != "all":
        split_fp = os.path.join(split_dp, f"{split}_ids.csv")
        split_spec_ids = set(pd.read_csv(split_fp)["spec_id"])
        spec_df = spec_df[spec_df["spec_id"].isin(split_spec_ids)]
    return spec_df[
        ["spec_id", "mol_id", "peaks", "prec_mz", "prec_type", "inst_type", "frag_mode", "nce"]
    ]


def summarize(df: pd.DataFrame, min_bin_count: int) -> None:
    """Print the retained-fraction-vs-NCE trend overall and per instrument."""
    df = df.copy()
    df["nce_bin"] = pd.cut(df["nce"], bins=NCE_BINS, right=False)

    print(f"\n{'=' * 104}")
    print("Intensity-weighted retained heavy-atom fraction vs NCE  (1.0 = intact precursor)")
    print(f"{'=' * 104}")
    print(f"spectra: {len(df)}   mols: {df['mol_id'].nunique()}")
    print(
        f"peak->formula ambiguity: mean {df['mean_ion_candidates'].mean():.2f} "
        f"matching DAG ions per explained peak"
    )

    has_pred = "pred_retained_frac_min" in df.columns

    def _table(g: pd.DataFrame, label: str) -> None:
        print(f"\n{label}")
        header = (
            f"  {'NCE bin':>12s} {'n':>7s} {'retained[min]':>14s} {'retained[max]':>14s} "
            f"{'bracket':>9s} {'expl.int':>9s}"
        )
        if has_pred:
            # MAE matters more than the signed mean: a bin's mean bias can be ~0 while every
            # individual spectrum is badly mis-allocated, with the errors cancelling.
            header += f" {'pred[min]':>10s} {'pred-true':>10s} {'MAE':>8s}"
        print(header)
        rows = []
        for nce_bin, gg in g.groupby("nce_bin", observed=True):
            if len(gg) < min_bin_count:
                continue
            lo = gg["retained_frac_min"].mean()
            hi = gg["retained_frac_max"].mean()
            rows.append((nce_bin, len(gg), lo, hi))
            line = (
                f"  {str(nce_bin):>12s} {len(gg):>7d} {lo:>14.4f} {hi:>14.4f} "
                f"{hi - lo:>9.4f} {gg['explained_int_frac'].mean():>9.4f}"
            )
            if has_pred:
                plo = gg["pred_retained_frac_min"].mean()
                mae = (gg["pred_retained_frac_min"] - gg["retained_frac_min"]).abs().mean()
                line += f" {plo:>10.4f} {plo - lo:>+10.4f} {mae:>8.4f}"
            print(line)
        if len(rows) >= 2:
            d_lo = rows[-1][2] - rows[0][2]
            d_hi = rows[-1][3] - rows[0][3]
            print(
                f"  -> change from {str(rows[0][0])} to {str(rows[-1][0])}: "
                f"min-attr {d_lo:+.4f}, max-attr {d_hi:+.4f}"
            )
            # Intensity-weighted correlation is the headline: does extent track CE at all?
            r_lo = g[["nce", "retained_frac_min"]].corr().iloc[0, 1]
            r_hi = g[["nce", "retained_frac_max"]].corr().iloc[0, 1]
            print(f"  -> pearson r(nce, retained): min-attr {r_lo:+.3f}, max-attr {r_hi:+.3f}")

    _table(df, "OVERALL")
    for inst, g in df.groupby("inst_type", observed=True):
        if len(g) < min_bin_count * 2:
            continue
        _table(g, f"inst_type = {inst}")

    print(f"\n{'-' * 104}")
    print("READ: a material FALL in retained fraction as NCE rises means a CE-conditioned extent")
    print("      mechanism (flow stop head / depth_ce_bias / CE x cut-count) has headroom.")
    print("      A FLAT trend means no such mis-allocation exists to model -- do not build it.")
    print("      A bracket as wide as the trend means peak->formula ambiguity hides the answer.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--frag_dp", required=True, help="DAG dir the model is trained on.")
    parser.add_argument(
        "--pred_hdf5",
        default=None,
        help="Optional predictions_*.hdf5. Adds the model's retained fraction alongside the "
        "true one, so the per-NCE-bin gap between them is the headroom a CE-conditioned "
        "extent mechanism could recover.",
    )
    parser.add_argument("--spec_fp", default="data/proc/nist20/spec_df.pkl")
    parser.add_argument("--dset", default="nist20_hr", help="Filter spec_df to this dataset label.")
    parser.add_argument("--split_dp", default="data/split/nist20_ma_mi_inchikey")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="val")
    parser.add_argument("--limit_mols", type=int, default=None, help="Subsample this many mols.")
    parser.add_argument("--max_h_transfer", type=int, default=None)
    parser.add_argument("--include_cmf", action="store_true", default=True)
    parser.add_argument("--min_bin_count", type=int, default=50, help="Skip NCE bins smaller than this.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_jobs", type=int, default=data_utils.JOBLIB_N_JOBS)
    parser.add_argument("--out_fp", default=None, help="Optional CSV of the per-spectrum table.")
    args = parser.parse_args()

    spec_df = load_spec_df(args.spec_fp, args.split_dp, args.split, args.dset)

    pred_peaks = {}
    if args.pred_hdf5 is not None:
        pred_peaks = load_pred_peaks(args.pred_hdf5)
        log.info("Loaded predicted peaks for %d spectra from %s", len(pred_peaks), args.pred_hdf5)
        spec_df = spec_df[spec_df["spec_id"].isin(pred_peaks.keys())]
        log.info("Restricted to %d spectra with predictions", len(spec_df))

    if args.limit_mols is not None:
        rng = np.random.default_rng(args.seed)
        mol_ids = spec_df["mol_id"].unique()
        keep = rng.choice(mol_ids, size=min(args.limit_mols, len(mol_ids)), replace=False)
        spec_df = spec_df[spec_df["mol_id"].isin(keep)]

    groups = spec_df.groupby("mol_id")
    sample_mol_id = int(next(iter(groups.groups.keys())))
    is_compressed = _detect_is_compressed(args.frag_dp, sample_mol_id)
    mol_inputs = []
    for mol_id, g in groups:
        rows = g.to_dict("records")
        for row in rows:
            # Always set the key so the worker can use direct access; None means "true side only".
            row["pred_peaks"] = pred_peaks.get(int(row["spec_id"]))
        mol_inputs.append((int(mol_id), rows))

    worker = functools.partial(
        _process_mol_group,
        frag_dp=args.frag_dp,
        is_compressed=is_compressed,
        max_h_transfer=args.max_h_transfer,
        include_cmf=args.include_cmf,
    )
    log.info("Processing %d mols (%d spectra)", len(mol_inputs), len(spec_df))
    per_mol = data_utils.par_apply(mol_inputs, worker, need_unpack=True, n_jobs=args.n_jobs)

    rows = [r for mol_results in per_mol for r in mol_results]
    if not rows:
        raise ValueError("No spectra produced an extent statistic; check --frag_dp and --split.")
    df = pd.DataFrame(rows)
    log.info("Got extent stats for %d spectra", len(df))

    summarize(df, args.min_bin_count)

    if args.out_fp:
        df.to_csv(args.out_fp, index=False)
        log.info("Saved per-spectrum table to %s", args.out_fp)


if __name__ == "__main__":
    main()
