"""Compare DAG-based isotope detection strategies on NIST20 FT spectra.

Uses NIST ann_df isotope annotations as ground truth.

Four methods compared:
  current        - detect_isotope_peaks_for_training (prec residual + aggressive fallback,
                   m/z-proxy expected ratio)
  dag_filter     - current, but also require parent M+0 in DAG (or = prec_mz)
  dag_protect    - current, then un-flag any peak whose m/z matches a DAG M+0 node
                   (i.e. confirmed real fragment → cannot be an isotope)
  dag_formula    - current, then re-check each flagged pair with the exact fragment
                   formula from the DAG; un-flag if the ratio gate fails under the
                   exact expected ratio (tighter precision, same recall logic)

Key questions:
  - dag_filter:   Does requiring DAG parent confirmation reduce FP?
  - dag_protect:  Does protecting confirmed DAG fragments reduce FP?
  - dag_formula:  Does exact formula ratio tighten the ratio gate, reducing FP?

Run from repo root:
    conda run -n FRAGNNET-GPU python scripts/debug/debug_dag_iso_peak_detection.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from fragnnet.utils.frag_utils import load_frag_d
from fragnnet.utils.formula_utils import C13_SPACING, parse_formula
from fragnnet.utils.spec_utils import (
    _expected_mk_ratio,
    _parse_adduct_counts,
    detect_isotope_peaks_for_training,
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
    """4-panel comparison plot: GT | current | dag_protect | dag_formula."""
    mzs = h["mzs"]
    ints = h["ints"]
    prec_mz = h["prec_mz"]
    ints_norm = ints / ints.max() * 100 if ints.max() > 0 else ints.copy()
    gt_iso = h["iso_k_arr"] > 0

    methods = [
        ("GT (NIST)", gt_iso, "tomato"),
        ("Current", h["current"], "darkorange"),
        ("DAG-protect", h["dag_protect"], "steelblue"),
        ("DAG-formula", h["dag_formula"], "mediumseagreen"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(24, 5), sharey=True)
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


def main() -> None:
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

    stats_rows: list[dict] = []
    plot_hits: list[dict] = []
    n_no_dag = 0
    n_processed = 0

    for _, row in merged.iterrows():
        if n_processed >= N_SAMPLES:
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

        iso_k_arr, is_ann = match_ann_peaks(mzs, row["ann_peak_mzs"], row["ann_isotopes"])
        gt_iso = iso_k_arr > 0

        current_mask = detect_isotope_peaks_for_training(
            mzs, ints, prec_mz=prec_mz, inst_type="FT",
            formula=formula if formula else None,
            prec_type=prec_type if prec_type else None,
        )

        dag = load_dag_info(mol_id)
        if dag is None:
            n_no_dag += 1
            continue

        filter_mask = method_dag_filter(mzs, current_mask, dag, prec_mz)
        protect_mask = method_dag_protect(mzs, current_mask, dag)
        formula_mask = method_dag_formula(mzs, ints, current_mask, dag, prec_mz, formula, prec_type)

        s = {
            "spec_id": int(row["spec_id"]),
            "mol_id": mol_id,
            "formula": formula,
            "n_gt_iso": int(gt_iso.sum()),
            "current":    _metrics(current_mask, gt_iso),
            "dag_filter": _metrics(filter_mask, gt_iso),
            "dag_protect": _metrics(protect_mask, gt_iso),
            "dag_formula": _metrics(formula_mask, gt_iso),
        }
        stats_rows.append(s)
        n_processed += 1

        # Collect plots where methods disagree
        any_differs = any(
            s[m]["tp"] != s["current"]["tp"] or s[m]["fp"] != s["current"]["fp"]
            for m in ("dag_filter", "dag_protect", "dag_formula")
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
                "dag_protect": protect_mask,
                "dag_formula": formula_mask,
            })

        if n_processed % 100 == 0:
            log.info(f"  processed {n_processed}/{N_SAMPLES} (no_dag={n_no_dag})")

    log.info(f"Done: {n_processed} spectra, {n_no_dag} skipped (no DAG)")

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
        ("DAG-filter",  "dag_filter"),
        ("DAG-protect", "dag_protect"),
        ("DAG-formula", "dag_formula"),
    ]:
        _print_summary(label, stats_rows, key)
    log.info("=" * 72)

    # Per-method delta vs current
    log.info("")
    for label, key in [("DAG-filter", "dag_filter"), ("DAG-protect", "dag_protect"), ("DAG-formula", "dag_formula")]:
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
