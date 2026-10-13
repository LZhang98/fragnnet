"""Compare isotope detection strategies on NIST20 FT spectra, DAG-sourced and subformula-sourced.

Uses NIST ann_df isotope annotations as ground truth.

Methods compared:
  current        - detect_isotope_peaks_for_training (prec residual + aggressive fallback,
                   m/z-proxy expected ratio)
  dag_native     - shared detector with DAG fragment formulas and k=1 corrected ratio
  dag_native_protect
                 - dag_native plus protection of peaks that are also DAG monoisotopic
  dag_filter     - current, but also require parent M+0 in DAG (or = prec_mz)
  dag_protect    - current, then un-flag any peak whose m/z matches a DAG M+0 node
                   (i.e. confirmed real fragment → cannot be an isotope)
  dag_formula    - current, then re-check each flagged pair with the exact fragment
                   formula from the DAG; un-flag if the ratio gate fails under the
                   exact expected ratio (tighter precision, same recall logic)
  dag_formula_group
                 - group_f, then the same DAG-formula post-check

  sub_native     - shared detector with the fragment composition decomposed from the peak m/z
                   against the precursor formula, instead of read from a DAG (no frag_dp needed)
  sub_group      - sub_native plus the group-level co-isolation fraction
  sub_protect    - sub_native plus protection of peaks a subformula explains on their own
  sub_infeasible - no detector at all: the peaks `filter_subformula_peaks` already drops,
                   which is the share of isotope removal that filter does for free

Key questions:
  - dag_filter:   Does requiring DAG parent confirmation reduce FP?
  - dag_protect:  Does protecting confirmed DAG fragments reduce FP?
  - dag_formula:  Does exact formula ratio tighten the ratio gate, reducing FP?
  - sub_native:   Does a DAG-free composition source match the DAG one, peak for peak?

Run from repo root:
    conda run -n FRAGNNET-GPU python scripts/debug/isotopes/debug_dag_iso_peak_detection.py
"""
# ruff: noqa: I001

from __future__ import annotations

import argparse
import collections
import logging
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from fragnnet.utils.dag_format_utils import load_frag_d
from fragnnet.utils.formula_utils import C13_SPACING, parse_formula
from fragnnet.utils.isotope_cleanup_utils import (
    _expected_mk_ratio,
    _parse_adduct_counts,
    dag_formula_source,
    detect_isotope_peaks_for_training,
    estimate_coisolation_fraction_from_precursor,
    subformula_formula_source,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths & params
# ---------------------------------------------------------------------------
PROC_DP = Path("data/proc/nist20")
FRAG_DP = Path("data/frag/nist20_ma_mi_d3_h4_isoFalse_cmf/dags")
FIG_DIR = Path("figs/dag_iso_detection")

N_SAMPLES = 500      # spectra to analyse
N_PLOTS = 30         # example plots to save (cases where methods differ)
MZ_TOL_ANN = 0.015   # Da — match observed peak to NIST annotation
DAG_MZ_TOL = 0.015   # Da — match observed M+0 to a DAG node M+0 m/z
SUBFORMULA_MZ_TOL = 0.004  # Da — FT accuracy, the window subformula decomposition uses
PREC_TOL = 0.05      # Da — for recognising precursor M+0 as a valid parent
RATIO_LO = 0.2       # same gates as detect_isotope_peaks_for_training defaults
RATIO_HI = 5.0


# ---------------------------------------------------------------------------
# DAG loading
# ---------------------------------------------------------------------------

class DagInfo:
    """Holds pre-processed DAG data for one molecule."""

    def __init__(self, m0_mzs: np.ndarray, idx2formula: dict[int, str]) -> None:
        self.m0_mzs = m0_mzs                     # sorted float64 (n_formulas,)
        self.idx2formula = idx2formula            # formula_idx -> formula string
        self.formula_strs = np.array(
            [idx2formula.get(i, "") for i in range(len(m0_mzs))],
            dtype=object,
        )
        # Build reverse: sorted array of (mz, formula_idx) for fast lookup
        nonzero = m0_mzs > 0
        self._mzs_sorted = np.sort(m0_mzs[nonzero])
        nz_idx = np.where(nonzero)[0]
        order = np.argsort(m0_mzs[nonzero])
        self._fidx_sorted = nz_idx[order].astype(np.int32)

    def find_formula(self, query_mz: float, tol: float = DAG_MZ_TOL) -> str | None:
        """Return formula string for the closest DAG node to query_mz, or None."""
        lo = np.searchsorted(self._mzs_sorted, query_mz - tol, side="left")
        hi = np.searchsorted(self._mzs_sorted, query_mz + tol, side="right")
        if lo >= hi:
            return None
        best = lo + int(np.argmin(np.abs(self._mzs_sorted[lo:hi] - query_mz)))
        fidx = int(self._fidx_sorted[best])
        return self.idx2formula.get(fidx)

    def has_mz(self, query_mz: float, tol: float = DAG_MZ_TOL) -> bool:
        """Return True if any DAG M+0 node is within tol of query_mz."""
        lo = np.searchsorted(self._mzs_sorted, query_mz - tol, side="left")
        hi = np.searchsorted(self._mzs_sorted, query_mz + tol, side="right")
        return lo < hi


def load_dag_info(mol_id: int | str) -> DagInfo | None:
    """Load and return DagInfo for mol_id, or None on failure."""
    try:
        entry = load_frag_d(mol_id, str(FRAG_DP), is_compressed=True)
    except Exception:
        return None
    fpm = entry.get("formula_peak_mzs")
    idx2f = entry.get("idx_to_formula", {})
    if fpm is None:
        return None
    fpm_np = fpm.numpy() if isinstance(fpm, torch.Tensor) else np.asarray(fpm, dtype=np.float64)
    return DagInfo(fpm_np[:, 0].astype(np.float64), idx2f)


# ---------------------------------------------------------------------------
# Three DAG-augmented methods (all post-process current_mask)
# ---------------------------------------------------------------------------

def method_dag_filter(
    mzs: np.ndarray,
    current_mask: np.ndarray,
    dag: DagInfo,
    prec_mz: float,
) -> np.ndarray:
    """Keep only isotope flags whose parent M+0 is in the DAG (or = prec_mz).

    This is the strictest filter: the parent fragment must be explicitly in
    the fragmentation DAG.  Precursor M+0 is also accepted as a parent.
    """
    out = np.zeros(len(mzs), dtype=bool)
    for i in np.where(current_mask)[0]:
        for k in range(1, 3):
            parent = mzs[i] - k * C13_SPACING
            if parent <= 0:
                continue
            if prec_mz > 0 and abs(parent - prec_mz) <= PREC_TOL:
                out[i] = True
                break
            if dag.has_mz(parent):
                out[i] = True
                break
    return out


def method_dag_protect(
    mzs: np.ndarray,
    current_mask: np.ndarray,
    dag: DagInfo,
) -> np.ndarray:
    """Un-flag any peak that the DAG confirms as a real monoisotopic fragment.

    A peak at mzs[i] that matches a DAG M+0 node cannot be an M+k satellite —
    it is a real fragment.  This removes false positives without touching
    peaks the DAG has no opinion on.
    """
    out = current_mask.copy()
    for i in np.where(current_mask)[0]:
        if dag.has_mz(mzs[i]):
            out[i] = False  # DAG says this m/z is a real M+0 — un-flag
    return out


def method_dag_formula(
    mzs: np.ndarray,
    ints: np.ndarray,
    current_mask: np.ndarray,
    dag: DagInfo,
    prec_mz: float,
    formula: str,
    prec_type: str,
    ratio_lo: float = RATIO_LO,
    ratio_hi: float = RATIO_HI,
) -> np.ndarray:
    """Re-check each flagged pair using the exact fragment formula from the DAG.

    For each (M+0, M+k) pair flagged by current_mask:
      1. Find the M+0 fragment's formula in the DAG.
      2. Compute exact R_exp_frag from that formula.
      3. Estimate f (co-isolation fraction) from the precursor residual peaks
         and the precursor formula.
      4. Un-flag if the observed ratio falls outside [ratio_lo, ratio_hi] ×
         (R_exp_frag × f).  This tightens or relaxes the gate vs the m/z proxy.

    Falls back to current_mask for any pair where formula lookup fails.
    """
    if not formula:
        return current_mask.copy()

    try:
        mol_counts = parse_formula(formula)
        adduct_counts, adduct_delta = _parse_adduct_counts(prec_type if prec_type else None)
    except Exception:
        return current_mask.copy()

    prec_neutral = prec_mz - adduct_delta if prec_mz > 0 else 0.0

    # Estimate R_k from precursor residuals (same as step 1 of the core function)
    sort_idx = np.argsort(mzs)
    smzs = mzs[sort_idx]
    sints = ints[sort_idx]

    def _find_peak(target: float, tol: float) -> tuple[float, float] | None:
        lo = np.searchsorted(smzs, target - tol, side="left")
        hi = np.searchsorted(smzs, target + tol, side="right")
        if lo >= hi:
            return None
        best = lo + int(np.argmax(sints[lo:hi]))
        return float(smzs[best]), float(sints[best])

    base_int = float(sints.max()) if len(sints) > 0 else 0.0
    M0_TOL = 0.05
    MIN_PREC_M0_FRAC = 0.005
    prec_m0 = _find_peak(prec_mz, M0_TOL)
    prec_m0_int: float | None = None
    if (
        prec_m0 is not None
        and prec_m0[1] > 0
        and (base_int <= 0 or prec_m0[1] / base_int >= MIN_PREC_M0_FRAC)
    ):
        prec_m0_int = prec_m0[1]

    # f[k] = R_k / R_exp_prec[k]  (co-isolation fraction for order k)
    f_coiso: dict[int, float] = {}
    if prec_m0_int is not None:
        for k in range(1, 3):
            k_tol = 0.004 if k == 1 else 0.008  # FT tight tolerance
            peak = _find_peak(prec_mz + k * C13_SPACING, k_tol)
            if peak is None or peak[0] <= prec_mz + k_tol:
                continue
            R_k_obs = peak[1] / prec_m0_int
            r_exp_prec_k = _expected_mk_ratio(mol_counts, 1.0, k, adduct_counts)
            if r_exp_prec_k > 0:
                f = R_k_obs / r_exp_prec_k
                if 0.01 <= f <= 1.5:
                    f_coiso[k] = f

    if not f_coiso:
        # No reliable f estimate — fall back to current_mask unchanged
        return current_mask.copy()

    out = current_mask.copy()
    for i in np.where(current_mask)[0]:
        for k in (1, 2):
            k_tol = 0.004 if k == 1 else 0.008
            parent_mz = mzs[i] - k * C13_SPACING
            if parent_mz <= 0:
                continue
            # Find M+0 parent in observed spectrum
            m0 = _find_peak(parent_mz, k_tol)
            if m0 is None or m0[1] <= 0 or m0[1] <= ints[i]:
                continue

            obs_ratio = ints[i] / m0[1]

            # --- proxy expected ratio (current method) ---
            if prec_neutral > 0:
                neutral_frag = m0[0] - adduct_delta
                mol_scale_proxy = max(0.0, min(1.0, neutral_frag / prec_neutral))
            else:
                mol_scale_proxy = 0.5
            r_proxy = _expected_mk_ratio(mol_counts, mol_scale_proxy, k, adduct_counts)

            # --- exact expected ratio from DAG formula ---
            dag_formula_str = dag.find_formula(m0[0])
            if dag_formula_str:
                try:
                    frag_counts = parse_formula(dag_formula_str)
                    r_exact = _expected_mk_ratio(frag_counts, 1.0, k)
                except Exception:
                    r_exact = r_proxy  # fallback
            else:
                r_exact = r_proxy  # no DAG coverage — keep current

            # f estimated from prec residuals
            f = f_coiso.get(k)
            if f is None:
                continue  # no f for this k — no change

            r_expected_exact = r_exact * f
            # Re-check ratio gate with exact expected ratio
            if not (ratio_lo * r_expected_exact <= obs_ratio <= ratio_hi * r_expected_exact):
                out[i] = False  # fails exact gate — un-flag
            break  # only process the lowest k that pairs

    return out


# ---------------------------------------------------------------------------
# Annotation matching
# ---------------------------------------------------------------------------

def match_ann_peaks(
    obs_mzs: np.ndarray,
    ann_mzs_raw: list,
    ann_iso_ks_raw: list,
    mz_tol: float = MZ_TOL_ANN,
) -> tuple[np.ndarray, np.ndarray]:
    """Match NIST annotations to observed peaks.

    Returns:
        iso_k_arr: int array (N,), -1=unannotated, 0=mono, k>0=isotope
        is_ann: bool array (N,)
    """
    iso_k_arr = np.full(len(obs_mzs), -1, dtype=int)
    is_ann = np.zeros(len(obs_mzs), dtype=bool)
    for ann_mz, iso_k in zip(ann_mzs_raw, ann_iso_ks_raw):
        try:
            mz_f, k_i = float(ann_mz), int(float(iso_k))
        except (TypeError, ValueError):
            continue
        d = np.abs(obs_mzs - mz_f)
        best = int(np.argmin(d))
        if d[best] <= mz_tol:
            if not is_ann[best] or k_i > iso_k_arr[best]:
                iso_k_arr[best] = k_i
                is_ann[best] = True
    return iso_k_arr, is_ann


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def memoized_source(source):
    """Cache a formula source per spectrum: the detector asks about the same m/z repeatedly."""
    cache: dict[int, list[dict[str, int]]] = {}

    def wrapped(query_mz: float) -> list[dict[str, int]]:
        key = int(round(query_mz * 1e4))
        if key not in cache:
            cache[key] = source(query_mz)
        return cache[key]

    return wrapped


def _metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "p": p, "r": r, "f1": f1}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_spectrum(
    h: dict,
    out_path: Path,
    idx: int,
    total: int,
) -> None:
    """Comparison plot for GT, current, and DAG-aware methods."""
    mzs = h["mzs"]
    ints = h["ints"]
    prec_mz = h["prec_mz"]
    ints_norm = ints / ints.max() * 100 if ints.max() > 0 else ints.copy()
    gt_iso = h["iso_k_arr"] > 0

    methods = [
        ("GT (NIST)", gt_iso, "tomato"),
        ("Current", h["current"], "darkorange"),
        ("Group-f", h["group_f"], "goldenrod"),
        ("DAG-native", h["dag_native"], "seagreen"),
        ("DAG+Group-f", h["dag_group"], "darkgreen"),
        ("DAG-native-prot", h["dag_native_protect"], "teal"),
        ("DAG-protect", h["dag_protect"], "steelblue"),
        ("DAG-formula", h["dag_formula"], "mediumseagreen"),
        ("DAG-form+Group", h["dag_formula_group"], "forestgreen"),
    ]

    fig, axes = plt.subplots(1, len(methods), figsize=(5.5 * len(methods), 5), sharey=True)
    for ax, (title, mask, col) in zip(axes, methods):
        for i, (mz, rel_int) in enumerate(zip(mzs, ints_norm)):
            if mask[i]:
                c, lw, zo, al = col, 2.0, 3, 0.95
            else:
                c, lw, zo, al = "grey", 0.7, 1, 0.35
            ax.vlines(mz, 0, rel_int, colors=c, lw=lw, alpha=al, zorder=zo)
        if prec_mz > 0:
            ax.axvline(prec_mz, color="black", lw=0.8, ls="--", alpha=0.4)
        ax.set_ylim(0, 130)
        ax.set_xlabel("m/z", fontsize=8)
        tp = int((mask & gt_iso).sum())
        fp = int((mask & ~gt_iso).sum())
        ax.set_title(f"{title}\nTP={tp} FP={fp} (flagged={int(mask.sum())})", fontsize=9)

    axes[0].set_ylabel("Rel. intensity (%)", fontsize=8)
    fig.suptitle(
        f"[{idx+1}/{total}]  spec_id={h['spec_id']}  |  {h['prec_type']}  |  "
        f"formula={h['formula']}  |  prec_mz={prec_mz:.4f}",
        fontsize=9,
        y=1.01,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare current and DAG-aware isotope peak detection on NIST FT spectra.",
    )
    parser.add_argument("--proc-dp", type=Path, default=PROC_DP, help="Processed data directory")
    parser.add_argument("--frag-dp", type=Path, default=FRAG_DP, help="Fragment DAG directory")
    parser.add_argument("--fig-dir", type=Path, default=FIG_DIR, help="Output figure directory")
    parser.add_argument(
        "--subformula-tol",
        type=float,
        default=SUBFORMULA_MZ_TOL,
        help="Da window for subformula decomposition (FT accuracy)",
    )
    parser.add_argument(
        "--candidate-rule",
        default="median",
        choices=["median", "all", "any"],
        help="How the subformula arms resolve several candidate compositions",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=N_SAMPLES,
        help="Number of spectra to analyze; 0 means all available spectra",
    )
    parser.add_argument("--n-plots", type=int, default=N_PLOTS, help="Number of disagreement plots to save")
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=1000,
        help="Log progress every N processed spectra",
    )
    parser.add_argument(
        "--group-min-count",
        type=int,
        default=1,
        help="Minimum precursor-derived f estimates per group/order",
    )
    parser.add_argument(
        "--group-max-cv",
        type=float,
        default=1.0,
        help="Maximum CV for group/order f aggregation",
    )
    parser.add_argument(
        "--group-estimate-samples",
        type=int,
        default=0,
        help="Limit rows used to estimate group f; 0 means use all candidate rows",
    )
    return parser.parse_args()


def _print_summary(label: str, rows: list[dict], key: str) -> None:
    tp = sum(r[key]["tp"] for r in rows)
    fp = sum(r[key]["fp"] for r in rows)
    fn = sum(r[key]["fn"] for r in rows)
    p, r_val, f1 = (
        tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        0.0,
    )
    f1 = 2 * p * r_val / (p + r_val) if (p + r_val) > 0 else 0.0
    log.info(f"  {label:<16} {tp:>6} {fp:>6} {fn:>6}  {p:>6.3f}  {r_val:>6.3f}  {f1:>6.3f}")


def build_group_f_lookup(
    df: pd.DataFrame,
    min_count: int,
    max_cv: float,
) -> dict[tuple[int, str, str], dict[int, float]]:
    """Build precursor-derived group-level f by (mol_id, prec_type, inst_type)."""
    vals_by_group_k: dict[tuple[tuple[int, str, str], int], list[float]] = collections.defaultdict(list)
    n_with_f = 0

    for _, row in df.iterrows():
        peaks = row["peaks"]
        if not peaks or len(peaks) < 3:
            continue

        formula = str(row.get("formula", ""))
        if not formula:
            continue

        mzs = np.array([p[0] for p in peaks], dtype=np.float64)
        ints = np.array([p[1] for p in peaks], dtype=np.float64)
        prec_mz = float(row["prec_mz"]) if pd.notna(row["prec_mz"]) else 0.0
        group_key = (
            int(row["mol_id"]),
            str(row.get("prec_type", "")),
            str(row.get("inst_type", "")),
        )
        f_by_k = estimate_coisolation_fraction_from_precursor(
            mzs,
            ints,
            prec_mz=prec_mz,
            inst_type=str(row.get("inst_type", "")),
            formula=formula,
            prec_type=group_key[1] or None,
        )
        if f_by_k:
            n_with_f += 1
        for k, f in f_by_k.items():
            vals_by_group_k[(group_key, int(k))].append(float(f))

    group_f: dict[tuple[int, str, str], dict[int, float]] = {}
    for (group_key, k), vals in vals_by_group_k.items():
        arr = np.array(vals, dtype=np.float64)
        if len(arr) < min_count:
            continue
        med = float(np.median(arr))
        if med <= 0:
            continue
        cv = float(np.std(arr) / med) if len(arr) > 1 else 0.0
        if cv > max_cv:
            continue
        group_f.setdefault(group_key, {})[k] = med

    log.info(
        f"  group f: spectra with local precursor f={n_with_f:,}, "
        f"groups={len(group_f):,}, group/order estimates={sum(len(v) for v in group_f.values()):,}"
    )
    return group_f


def main() -> None:
    global FIG_DIR, FRAG_DP, N_PLOTS, N_SAMPLES, PROC_DP

    args = parse_args()
    PROC_DP = args.proc_dp
    FRAG_DP = args.frag_dp
    FIG_DIR = args.fig_dir
    N_SAMPLES = args.n_samples
    N_PLOTS = args.n_plots

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading NIST20 data…")
    ann_df = pd.read_pickle(PROC_DP / "ann_df.pkl")[
        ["spec_id", "mol_id", "formula", "prec_type", "ann_peak_mzs", "ann_isotopes"]
    ].copy()
    spec_df = pd.read_pickle(PROC_DP / "spec_df.pkl")[
        ["spec_id", "mol_id", "inst_type", "prec_mz", "peaks"]
    ].copy()

    spec_df_ft = spec_df[spec_df["inst_type"] == "FT"].copy()
    log.info(f"  FT spectra: {len(spec_df_ft):,}")

    ann_df["n_iso"] = ann_df["ann_isotopes"].apply(
        lambda x: sum(1 for v in x if float(v) != 0)
    )
    ann_df_iso = ann_df[ann_df["n_iso"] > 0].copy()
    log.info(f"  ann_df rows with ≥1 isotope peak: {len(ann_df_iso):,}")

    merged = spec_df_ft.merge(
        ann_df_iso[["spec_id", "mol_id", "formula", "prec_type", "ann_peak_mzs", "ann_isotopes"]],
        on=["spec_id", "mol_id"],
        how="inner",
    ).reset_index(drop=True)
    log.info(f"  FT spectra with annotated isotopes: {len(merged):,}")
    merged = merged.sample(frac=1, random_state=42).reset_index(drop=True)

    log.info("Estimating group-level f from precursor residuals…")
    group_source = merged
    if args.group_estimate_samples > 0:
        group_source = merged.head(args.group_estimate_samples)
    group_f_lookup = build_group_f_lookup(
        group_source,
        min_count=args.group_min_count,
        max_cv=args.group_max_cv,
    )

    stats_rows: list[dict] = []
    plot_hits: list[dict] = []
    dag_cache: dict[int, DagInfo | None] = {}
    n_no_dag = 0
    n_no_subformula = 0
    n_processed = 0
    n_target = len(merged) if N_SAMPLES <= 0 else N_SAMPLES

    for _, row in merged.iterrows():
        if N_SAMPLES > 0 and n_processed >= N_SAMPLES:
            break

        peaks = row["peaks"]
        if not peaks or len(peaks) < 3:
            continue

        mzs = np.array([p[0] for p in peaks], dtype=np.float64)
        ints = np.array([p[1] for p in peaks], dtype=np.float64)
        prec_mz = float(row["prec_mz"]) if pd.notna(row["prec_mz"]) else 0.0
        mol_id = int(row["mol_id"])
        formula = str(row.get("formula", ""))
        prec_type = str(row.get("prec_type", ""))
        group_key = (mol_id, prec_type, "FT")
        group_f = group_f_lookup.get(group_key)

        iso_k_arr, is_ann = match_ann_peaks(mzs, row["ann_peak_mzs"], row["ann_isotopes"])
        gt_iso = iso_k_arr > 0

        current_mask = detect_isotope_peaks_for_training(
            mzs, ints, prec_mz=prec_mz, inst_type="FT",
            formula=formula if formula else None,
            prec_type=prec_type if prec_type else None,
        )
        group_mask = detect_isotope_peaks_for_training(
            mzs, ints, prec_mz=prec_mz, inst_type="FT",
            formula=formula if formula else None,
            prec_type=prec_type if prec_type else None,
            coiso_fraction_by_k=group_f,
        )

        if mol_id not in dag_cache:
            dag_cache[mol_id] = load_dag_info(mol_id)
        dag = dag_cache[mol_id]
        if dag is None:
            n_no_dag += 1
            continue

        dag_native_mask = detect_isotope_peaks_for_training(
            mzs, ints, prec_mz=prec_mz, inst_type="FT",
            formula=formula if formula else None,
            prec_type=prec_type if prec_type else None,
            formula_source=dag_formula_source(dag.m0_mzs, dag.formula_strs, DAG_MZ_TOL),
        )
        dag_group_mask = detect_isotope_peaks_for_training(
            mzs, ints, prec_mz=prec_mz, inst_type="FT",
            formula=formula if formula else None,
            prec_type=prec_type if prec_type else None,
            coiso_fraction_by_k=group_f,
            formula_source=dag_formula_source(dag.m0_mzs, dag.formula_strs, DAG_MZ_TOL),
        )
        dag_native_protect_mask = detect_isotope_peaks_for_training(
            mzs, ints, prec_mz=prec_mz, inst_type="FT",
            formula=formula if formula else None,
            prec_type=prec_type if prec_type else None,
            formula_source=dag_formula_source(dag.m0_mzs, dag.formula_strs, DAG_MZ_TOL),
            protect_known_mono=True,
        )
        filter_mask = method_dag_filter(mzs, current_mask, dag, prec_mz)
        protect_mask = method_dag_protect(mzs, current_mask, dag)
        formula_mask = method_dag_formula(mzs, ints, current_mask, dag, prec_mz, formula, prec_type)
        formula_group_mask = method_dag_formula(
            mzs,
            ints,
            group_mask,
            dag,
            prec_mz,
            formula,
            prec_type,
        )

        # Subformula arms: the same detector with the composition sourced from a
        # precursor-bounded decomposition instead of the DAG. `sub_infeasible` is what
        # `filter_subformula_peaks` already removes on its own - no detector involved.
        try:
            sub_source = memoized_source(
                subformula_formula_source(formula, prec_type, args.subformula_tol)
            )
        except ValueError:
            sub_source = None

        if sub_source is None:
            n_no_subformula += 1
            sub_native_mask = np.zeros(len(mzs), dtype=bool)
            sub_group_mask = np.zeros(len(mzs), dtype=bool)
            sub_protect_mask = np.zeros(len(mzs), dtype=bool)
            sub_infeasible_mask = np.zeros(len(mzs), dtype=bool)
        else:
            sub_native_mask = detect_isotope_peaks_for_training(
                mzs, ints, prec_mz=prec_mz, inst_type="FT",
                formula=formula if formula else None,
                prec_type=prec_type if prec_type else None,
                formula_source=sub_source,
                candidate_rule=args.candidate_rule,
            )
            sub_group_mask = detect_isotope_peaks_for_training(
                mzs, ints, prec_mz=prec_mz, inst_type="FT",
                formula=formula if formula else None,
                prec_type=prec_type if prec_type else None,
                coiso_fraction_by_k=group_f,
                formula_source=sub_source,
                candidate_rule=args.candidate_rule,
            )
            sub_protect_mask = detect_isotope_peaks_for_training(
                mzs, ints, prec_mz=prec_mz, inst_type="FT",
                formula=formula if formula else None,
                prec_type=prec_type if prec_type else None,
                formula_source=sub_source,
                candidate_rule=args.candidate_rule,
                protect_known_mono=True,
            )
            sub_infeasible_mask = np.array([not sub_source(float(mz)) for mz in mzs])

        s = {
            "spec_id": int(row["spec_id"]),
            "mol_id": mol_id,
            "formula": formula,
            "n_gt_iso": int(gt_iso.sum()),
            "current":    _metrics(current_mask, gt_iso),
            "group_f": _metrics(group_mask, gt_iso),
            "dag_native": _metrics(dag_native_mask, gt_iso),
            "dag_group": _metrics(dag_group_mask, gt_iso),
            "dag_native_protect": _metrics(dag_native_protect_mask, gt_iso),
            "dag_filter": _metrics(filter_mask, gt_iso),
            "dag_protect": _metrics(protect_mask, gt_iso),
            "dag_formula": _metrics(formula_mask, gt_iso),
            "dag_formula_group": _metrics(formula_group_mask, gt_iso),
            "sub_native": _metrics(sub_native_mask, gt_iso),
            "sub_group": _metrics(sub_group_mask, gt_iso),
            "sub_protect": _metrics(sub_protect_mask, gt_iso),
            "sub_infeasible": _metrics(sub_infeasible_mask, gt_iso),
            "dag_vs_sub_agree": int((dag_native_mask == sub_native_mask).sum()),
            "n_peaks": int(len(mzs)),
        }
        stats_rows.append(s)
        n_processed += 1

        # Collect plots where methods disagree
        any_differs = any(
            s[m]["tp"] != s["current"]["tp"] or s[m]["fp"] != s["current"]["fp"]
            for m in (
                "group_f",
                "dag_native",
                "dag_group",
                "dag_native_protect",
                "dag_filter",
                "dag_protect",
                "dag_formula",
                "dag_formula_group",
            )
        )
        if len(plot_hits) < N_PLOTS and any_differs:
            plot_hits.append({
                "spec_id": s["spec_id"],
                "formula": formula,
                "prec_type": prec_type,
                "prec_mz": prec_mz,
                "mzs": mzs,
                "ints": ints,
                "iso_k_arr": iso_k_arr,
                "current": current_mask,
                "group_f": group_mask,
                "dag_native": dag_native_mask,
                "dag_group": dag_group_mask,
                "dag_native_protect": dag_native_protect_mask,
                "dag_protect": protect_mask,
                "dag_formula": formula_mask,
                "dag_formula_group": formula_group_mask,
            })

        if args.progress_interval > 0 and n_processed % args.progress_interval == 0:
            log.info(f"  processed {n_processed}/{n_target} (no_dag={n_no_dag})")

    log.info(
        f"Done: {n_processed} spectra, {n_no_dag} skipped (no DAG), "
        f"{n_no_subformula} without a subformula source"
    )

    if not stats_rows:
        log.error("No stats collected.")
        return

    total_gt = sum(r["n_gt_iso"] for r in stats_rows)
    log.info("")
    log.info("=" * 72)
    log.info(f"  Spectra analysed : {len(stats_rows)}   |   Total GT iso peaks: {total_gt}")
    log.info("")
    log.info(f"  {'Method':<16} {'TP':>6} {'FP':>6} {'FN':>6}  {'Prec':>6}  {'Recall':>6}  {'F1':>6}")
    log.info(f"  {'-'*72}")
    for label, key in [
        ("Current",     "current"),
        ("Group-f",     "group_f"),
        ("DAG-native",  "dag_native"),
        ("DAG+Group-f", "dag_group"),
        ("DAG-native-prot", "dag_native_protect"),
        ("DAG-filter",  "dag_filter"),
        ("DAG-protect", "dag_protect"),
        ("DAG-formula", "dag_formula"),
        ("DAG-form+Group", "dag_formula_group"),
        ("Subform-native", "sub_native"),
        ("Subform+Group-f", "sub_group"),
        ("Subform-protect", "sub_protect"),
        ("Subform-infeas", "sub_infeasible"),
    ]:
        _print_summary(label, stats_rows, key)
    log.info("=" * 72)

    # Head to head: how often the two sources decide a peak differently at all.
    total_peaks = sum(r["n_peaks"] for r in stats_rows)
    agree = sum(r["dag_vs_sub_agree"] for r in stats_rows)
    log.info(
        f"  DAG-native vs Subform-native: {agree}/{total_peaks} peak decisions agree "
        f"({agree / max(total_peaks, 1):.4f})"
    )

    # Per-method delta vs current
    log.info("")
    for label, key in [
        ("Group-f", "group_f"),
        ("DAG-native", "dag_native"),
        ("DAG+Group-f", "dag_group"),
        ("DAG-native-prot", "dag_native_protect"),
        ("DAG-filter", "dag_filter"),
        ("DAG-protect", "dag_protect"),
        ("DAG-formula", "dag_formula"),
        ("DAG-form+Group", "dag_formula_group"),
        ("Subform-native", "sub_native"),
        ("Subform+Group-f", "sub_group"),
        ("Subform-protect", "sub_protect"),
    ]:
        fp_delta = np.array([r["current"]["fp"] - r[key]["fp"] for r in stats_rows])
        tp_delta = np.array([r["current"]["tp"] - r[key]["tp"] for r in stats_rows])
        fp_reduced_clean = sum(1 for r in stats_rows if r[key]["fp"] < r["current"]["fp"] and r[key]["tp"] == r["current"]["tp"])
        tp_lost = sum(1 for r in stats_rows if r[key]["tp"] < r["current"]["tp"])
        log.info(f"  {label}:")
        log.info(f"    FP removed: total={int(fp_delta.sum())}  mean/spec={fp_delta.mean():.2f}  spectra with FP↓ no TP loss={fp_reduced_clean}")
        log.info(f"    TP removed: total={int(tp_delta.sum())}  mean/spec={tp_delta.mean():.2f}  spectra with TP↓={tp_lost}")

    # Save plots
    log.info(f"\nSaving {len(plot_hits)} plots to {FIG_DIR}…")
    for i, h in enumerate(plot_hits):
        out_path = FIG_DIR / f"{i+1:03d}_spec{h['spec_id']}.png"
        plot_spectrum(h, out_path, i, len(plot_hits))
        if (i + 1) % 10 == 0:
            log.info(f"  {i+1}/{len(plot_hits)} saved")

    log.info(f"Done. Plots in {FIG_DIR}/")


if __name__ == "__main__":
    main()
