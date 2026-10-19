"""Diagnose whether the model gets the top-K dominant MS/MS peaks right, and how much of the
residual failure is the fragmentation DAG's fault rather than the model's.

For every spectrum the true peaks are ranked by intensity and the top ``K_MAX`` are followed
through the prediction. Each dominant true peak lands in exactly one of four outcomes:

  - ``hit``          -- the peak is in the exported prediction AND the model ranks it inside its
                        own top-K predicted peaks. The model got it.
  - ``under_ranked`` -- the peak is in the exported prediction but the model ranks it outside its
                        top-K. A pure ranking failure: the fragment was available and the model
                        down-weighted it.
  - ``zero_pred``    -- the DAG reaches that m/z, but the model gave it so little intensity that
                        the ``min_peak_int`` export filter dropped it. Still a model failure --
                        an extreme form of under-ranking -- not a coverage failure.
  - ``no_dag``       -- no fragment formula in the DAG reaches that m/z at instrument tolerance.
                        A genuinely *unexplained fragment*. No amount of training fixes this; it
                        is an enumeration/coverage failure and it caps the achievable score.

Separating these is the point: ``under_ranked`` + ``zero_pred`` is model headroom, ``no_dag`` is
DAG headroom, and the two call for completely different work. The ``zero_pred`` split is why
``--frag_dp`` is required. The exported ``pred_mzs`` is NOT the DAG support: inference applies a
``min_peak_int`` floor (1.0 on the 0-1000 scale in the standard eval script), which drops roughly
90% of the candidate peaks. Classifying against the export alone would misattribute the dropped
candidates to the DAG. The DAG support is rebuilt from ``formula_peak_mzs``/``formula_peak_probs``
in both the std and CMF ion-mass conventions, matching ``_dag_ion_mzs`` in ``dag_oracle_scores.py``.

Peak matching reuses the eval-time per-instrument tolerance (FT 10 ppm / floor 200 Da, QTOF
20 ppm / floor 750 Da, others fall back to 10 ppm / 200 Da) and the ``1_above_prec`` filter tier
(true peaks above the precursor m/z are dropped -- they are outside the DAG's candidate support),
matching ``dag_oracle_scores.py`` so the numbers here are comparable to the oracle ceiling.

Outputs (to ``--out_dp``):
  - ``dominant_peak_records.csv``      one row per (spectrum, dominant true peak rank)
  - ``dominant_peak_spectra.csv``      one row per spectrum (base-peak outcome, unexplained mass)
  - ``dominant_peak_summary.csv``      outcome shares per K and per stratum
  - ``dominant_peak_topk_outcome.png`` outcome vs K, instrument, adduct
  - ``dominant_peak_ce_and_class.png`` outcome vs collision energy, ClassyFire class, cosine
  - ``dominant_peak_miss_anatomy.png`` where the misses go: rank, m/z, neutral-loss offset

Usage:
    python scripts/debug/oracle/dominant_peak_diagnostic.py \\
        --pred_fp output/inference_eval/pqx7ywc9/predictions_test.hdf5 \\
        --spec_df_fp data/proc/nist20/spec_df.pkl \\
        --frag_dp data/frag/nist20_ma_mi_d3_h4_isoFalse_smpFalse/dags \\
        --classyfire_fp data/classyfire/nist20_mol_classyfire.pkl \\
        --out_dp output/diagnostics/dominant_peak_pqx7ywc9_test
"""

from __future__ import annotations

import argparse
import bz2
import logging
import pickle
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from fragnnet.utils.formula_utils import PREC_TYPE_TO_CMF_MASS_DIFF, PREC_TYPE_TO_MASS_DIFF

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Per-instrument m/z tolerance, matching INST_TYPE_LOSS_TOL in dag_oracle_scores.py and
# inst_type_loss_tol in config/fragnnet/exp_fraggnn_ma_mi/fraggnn_d3_ma_mi_nist20_amolc_all.yml.
INST_TYPE_LOSS_TOL = {
    "ORB": (1.0e-5, 200.0),  # 10 ppm, floor 200 Da
    "ORB_F": (1.0e-5, 200.0),  # 10 ppm, floor 200 Da
    "QTOF": (2.0e-5, 750.0),  # 20 ppm, floor 750 Da
}
DEFAULT_TOL = (1.0e-5, 200.0)

K_MAX = 10  # deepest dominant-peak rank followed
K_GRID = (1, 3, 5, 10)  # K values reported in the headline stacked bars

OUTCOMES = ("hit", "under_ranked", "zero_pred", "no_dag")
OUTCOME_LABELS = {
    "hit": "Hit (in model top-K)",
    "under_ranked": "In DAG, under-ranked",
    "zero_pred": "In DAG, predicted ~zero",
    "no_dag": "Not in DAG (unexplained)",
}
# dataviz skill categorical steps. Validated all-pairs on the light surface: worst CVD dE 6.9
# (deutan), tritan 17.1, normal-vision dE 20.8. The 6-8 CVD band is legal only with secondary
# encoding, so every stacked segment below carries a visible share label.
OUTCOME_COLORS = {
    "hit": "#1baf7a",
    "under_ranked": "#eda100",
    "zero_pred": "#4a3aa7",
    "no_dag": "#e34948",
}

# Common neutral losses, used to label the base-peak m/z offset spectrum. These are the
# monoisotopic masses of the losses that dominate the observed offset histogram.
NEUTRAL_LOSSES = [
    ("H", 1.0078),
    ("CH3", 15.0235),
    ("NH3", 17.0265),
    ("H2O", 18.0106),
    ("C2H2", 26.0157),
    ("CO", 27.9949),
    ("CH2O", 30.0106),
    ("C2H2O", 42.0106),
    ("CO2", 43.9898),
    ("HCOOH", 46.0055),
]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#d9d8d3"
SEQ_HUE = "#2a78d6"  # sequential / single-series hue


class DagSupport:
    """Candidate ion m/z of every fragment formula in a molecule's DAG.

    The exported prediction is intensity-filtered, so it cannot answer "could the DAG reach this
    m/z at all". This reads the DAG directly and reproduces the candidate set exactly as
    ``_dag_ion_mzs`` in ``dag_oracle_scores.py`` does: every non-padding isotope peak of every
    formula row, offset into observed m/z space under both the std and CMF ion-mass conventions.

    Attributes:
        frag_dp: Directory of ``{mol_id}.pkl.bz2`` DAG files.
    """

    def __init__(self, frag_dp: Path) -> None:
        """Initialize the loader.

        Args:
            frag_dp: Directory holding one ``{mol_id}.pkl.bz2`` DAG file per molecule.
        """
        self.frag_dp = frag_dp
        self._neutral: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._ion: dict[tuple[int, str], np.ndarray] = {}

    def get(self, mol_id: int, prec_type: str) -> np.ndarray:
        """Return the sorted unique candidate ion m/z for one molecule and adduct.

        Args:
            mol_id: Molecule id, naming the DAG file.
            prec_type: Precursor adduct type, e.g. ``"[M+H]+"``.

        Returns:
            Sorted unique candidate ion m/z of shape (num_candidates,).

        Raises:
            FileNotFoundError: If the molecule has no DAG file.
        """
        key = (int(mol_id), prec_type)
        if key in self._ion:
            return self._ion[key]
        if int(mol_id) not in self._neutral:
            fp = self.frag_dp / f"{int(mol_id)}.pkl.bz2"
            with bz2.open(fp, "rb") as f:
                dag = pickle.load(f)
            theo = np.asarray(dag["formula_peak_mzs"], dtype=float)
            mask = (np.asarray(dag["formula_peak_probs"], dtype=float) > 0).astype(float)
            self._neutral[int(mol_id)] = (theo, mask)
        theo, mask = self._neutral[int(mol_id)]
        std = theo + mask * PREC_TYPE_TO_MASS_DIFF[prec_type]
        cmf = theo + mask * PREC_TYPE_TO_CMF_MASS_DIFF.get(
            prec_type, PREC_TYPE_TO_MASS_DIFF[prec_type]
        )
        ion = np.unique(np.concatenate([std.ravel(), cmf.ravel()]))
        # Keep the cache bounded: only the molecule currently being streamed is reused.
        if len(self._ion) > 64:
            self._ion.clear()
            self._neutral = {int(mol_id): (theo, mask)}
        self._ion[key] = ion
        return ion


def _apply_style() -> None:
    """Set recessive axes and a neutral chart surface for every figure."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "text.color": INK,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "font.size": 9,
            "legend.frameon": False,
        }
    )


def load_predictions(pred_fp: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load per-spectrum index metadata and the flat predicted peak arrays.

    Args:
        pred_fp: Path to the predictions HDF5.

    Returns:
        Tuple of (per-spectrum DataFrame, flat pred m/z array, flat pred intensity array).
    """
    with h5py.File(pred_fp, "r") as f:
        meta = pd.DataFrame(
            {
                "spec_id": f["spec_id"][:],
                "mol_id": f["mol_id"][:],
                "start_points": f["start_points"][:],
                "end_points": f["end_points"][:],
                "cos_hun_inst_tol": f["cos_hun_inst_tol"][:],
            }
        )
        pred_mzs = f["pred_mzs"][:]
        pred_ints = f["pred_ints"][:]
    log.info("Loaded %d spectra, %d predicted peaks from %s", len(meta), len(pred_mzs), pred_fp)
    return meta, pred_mzs, pred_ints


def build_records(
    meta: pd.DataFrame,
    pred_mzs: np.ndarray,
    pred_ints: np.ndarray,
    spec_df: pd.DataFrame,
    dag: DagSupport,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Follow the top-K dominant true peaks of every spectrum into the prediction.

    Args:
        meta: Per-spectrum index metadata from the predictions HDF5.
        pred_mzs: Flat predicted m/z array indexed by ``start_points``/``end_points``.
        pred_ints: Flat predicted intensity array, same indexing.
        spec_df: Ground-truth spectra with ``peaks``, ``prec_mz``, ``inst_type``, ``prec_type``,
            ``nce``, ``ace``.
        dag: Loader for the unfiltered DAG candidate m/z support.

    Returns:
        Tuple of (per-dominant-peak records, per-spectrum summary).
    """
    gt = spec_df.set_index("spec_id")
    cols = ["peaks", "prec_mz", "inst_type", "prec_type", "nce", "ace"]
    gt = gt.loc[meta["spec_id"].to_numpy(), cols]

    peaks_all = gt["peaks"].to_numpy()
    prec_mz_all = gt["prec_mz"].to_numpy(dtype=float)
    inst_all = gt["inst_type"].to_numpy()
    prec_type_all = gt["prec_type"].to_numpy()
    nce_all = gt["nce"].to_numpy(dtype=float)
    ace_all = gt["ace"].to_numpy(dtype=float)
    starts = meta["start_points"].to_numpy(dtype=int)
    ends = meta["end_points"].to_numpy(dtype=int)
    spec_ids = meta["spec_id"].to_numpy()
    mol_ids = meta["mol_id"].to_numpy()
    cos_all = meta["cos_hun_inst_tol"].to_numpy(dtype=float)

    rec_spec, rec_rank, rec_mz, rec_frac = [], [], [], []
    rec_matched, rec_in_dag, rec_pred_rank, rec_pred_frac = [], [], [], []
    spec_rows = []

    for i in range(len(meta)):
        peaks = np.asarray(peaks_all[i], dtype=float)
        if peaks.ndim != 2 or peaks.shape[0] == 0:
            continue
        prec_mz = prec_mz_all[i]
        inst = inst_all[i]
        rel, min_mz = INST_TYPE_LOSS_TOL.get(inst, DEFAULT_TOL)

        # Filter tier 1_above_prec: drop true peaks above the precursor m/z.
        t_mzs, t_ints = peaks[:, 0], peaks[:, 1]
        keep = t_mzs <= prec_mz + max(prec_mz, min_mz) * rel
        t_mzs, t_ints = t_mzs[keep], t_ints[keep]
        if len(t_mzs) == 0 or t_ints.sum() <= 0:
            continue
        t_frac = t_ints / t_ints.sum()

        s, e = starts[i], ends[i]
        p_mzs, p_ints = pred_mzs[s:e], pred_ints[s:e]
        if len(p_mzs) == 0 or p_ints.sum() <= 0:
            continue
        p_frac = p_ints / p_ints.sum()
        # Rank of every predicted peak by descending intensity (0 = model's own base peak).
        p_rank = np.empty(len(p_ints), dtype=np.int64)
        p_rank[np.argsort(-p_ints, kind="stable")] = np.arange(len(p_ints))

        order = np.argsort(-t_ints, kind="stable")
        top = order[:K_MAX]

        # DAG coverage over ALL true peaks, against the UNFILTERED DAG support -- how much true
        # intensity no fragment formula can reach at all.
        d_mzs = dag.get(mol_ids[i], prec_type_all[i])
        tol_all = rel * np.maximum(t_mzs, min_mz)
        in_dag_all = (np.abs(t_mzs[:, None] - d_mzs[None, :]) <= tol_all[:, None]).any(axis=1)
        unexpl_frac = float(t_frac[~in_dag_all].sum())

        for rank_k, ti in enumerate(top):
            tol = rel * max(t_mzs[ti], min_mz)
            hits = np.flatnonzero(np.abs(p_mzs - t_mzs[ti]) <= tol)
            rec_spec.append(spec_ids[i])
            rec_rank.append(rank_k)
            rec_mz.append(t_mzs[ti])
            rec_frac.append(t_frac[ti])
            rec_in_dag.append(bool(in_dag_all[ti]))
            if len(hits) == 0:
                rec_matched.append(False)
                rec_pred_rank.append(-1)
                rec_pred_frac.append(0.0)
            else:
                # A true m/z can match several formula peaks; credit the strongest.
                best = hits[np.argmax(p_ints[hits])]
                rec_matched.append(True)
                rec_pred_rank.append(int(p_rank[best]))
                rec_pred_frac.append(float(p_frac[best]))

        pb = int(np.argmax(p_ints))
        tb = int(order[0])
        spec_rows.append(
            (
                spec_ids[i],
                mol_ids[i],
                inst,
                prec_type_all[i],
                nce_all[i],
                ace_all[i],
                prec_mz,
                cos_all[i],
                len(t_mzs),
                len(p_mzs),
                int(in_dag_all.sum()),
                unexpl_frac,
                t_mzs[tb],
                float(t_frac[tb]),
                float(p_mzs[pb]),
                float(p_frac[pb]),
            )
        )

    records = pd.DataFrame(
        {
            "spec_id": rec_spec,
            "true_rank": rec_rank,
            "true_mz": rec_mz,
            "true_int_frac": rec_frac,
            "matched": rec_matched,
            "in_dag": rec_in_dag,
            "pred_rank": rec_pred_rank,
            "pred_int_frac": rec_pred_frac,
        }
    )
    spectra = pd.DataFrame(
        spec_rows,
        columns=[
            "spec_id",
            "mol_id",
            "inst_type",
            "prec_type",
            "nce",
            "ace",
            "prec_mz",
            "cos_hun_inst_tol",
            "n_true_peaks",
            "n_pred_peaks",
            "n_in_dag_peaks",
            "unexpl_int_frac",
            "true_base_mz",
            "true_base_frac",
            "pred_base_mz",
            "pred_base_frac",
        ],
    )
    log.info("Built %d dominant-peak records over %d spectra", len(records), len(spectra))
    return records, spectra


def outcome_at_k(records: pd.DataFrame, k: int) -> pd.Series:
    """Classify each of the top-k dominant true peaks into an outcome.

    Args:
        records: Per-dominant-peak records.
        k: Top-K depth. Only ``true_rank < k`` rows participate, and a peak counts as a hit
            when the model also ranks it inside its own top-k.

    Returns:
        Series of outcome strings aligned to ``records[records.true_rank < k]``.
    """
    sub = records[records["true_rank"] < k]
    matched = sub["matched"].to_numpy()
    out = np.where(
        matched,
        np.where(sub["pred_rank"].to_numpy() < k, "hit", "under_ranked"),
        np.where(sub["in_dag"].to_numpy(), "zero_pred", "no_dag"),
    )
    return pd.Series(out, index=sub.index, name="outcome")


def _shares(outcome: pd.Series, group: pd.Series | None = None) -> pd.DataFrame:
    """Fraction of each outcome, overall or per group.

    Args:
        outcome: Outcome labels.
        group: Optional grouping key aligned to ``outcome``.

    Returns:
        DataFrame with one row per group and one column per outcome, plus ``n``.
    """
    if group is None:
        counts = outcome.value_counts()
        df = pd.DataFrame([counts]).reindex(columns=list(OUTCOMES)).fillna(0.0)
        df.index = ["all"]
    else:
        df = pd.crosstab(group, outcome).reindex(columns=list(OUTCOMES)).fillna(0.0)
    df["n"] = df[list(OUTCOMES)].sum(axis=1)
    for c in OUTCOMES:
        df[c] = df[c] / df["n"]
    return df


def _stacked(ax, shares: pd.DataFrame, labels: list[str], min_label: float = 0.04) -> None:
    """Draw a horizontal stacked outcome bar chart with direct share labels.

    Args:
        ax: Target axes.
        shares: Rows of outcome fractions (columns must include every entry of OUTCOMES).
        labels: Y tick labels, one per row of ``shares``.
        min_label: Minimum segment width that still gets a printed percentage.
    """
    y = np.arange(len(shares))
    left = np.zeros(len(shares))
    for oc in OUTCOMES:
        w = shares[oc].to_numpy()
        # 2px surface gap between segments so adjacent fills never touch.
        ax.barh(
            y, w, left=left, color=OUTCOME_COLORS[oc], height=0.62, edgecolor=SURFACE, linewidth=2
        )
        for yi, (wi, li) in enumerate(zip(w, left)):
            if wi >= min_label:
                ax.text(
                    li + wi / 2, yi, f"{wi:.0%}", ha="center", va="center", fontsize=8, color=INK
                )
        left += w
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of dominant true peaks")
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)


def _legend(fig) -> None:
    """Attach the shared outcome legend below the figure."""
    handles = [Patch(facecolor=OUTCOME_COLORS[o], label=OUTCOME_LABELS[o]) for o in OUTCOMES]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))


def fig_topk_outcome(records: pd.DataFrame, spectra: pd.DataFrame, out_fp: Path) -> None:
    """Outcome of the top-K dominant peaks, overall and split by instrument and adduct."""
    meta = spectra.set_index("spec_id")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))

    rows, labels = [], []
    for k in K_GRID:
        rows.append(_shares(outcome_at_k(records, k)).iloc[0])
        labels.append(f"K = {k}")
    _stacked(axes[0, 0], pd.DataFrame(rows), labels)
    axes[0, 0].set_title("Top-K dominant true peaks, all spectra")

    ax = axes[0, 1]
    ks = np.arange(1, K_MAX + 1)
    hit, expl = [], []
    for k in ks:
        oc = outcome_at_k(records, k)
        hit.append((oc == "hit").mean())
        expl.append((oc != "no_dag").mean())
    ax.plot(ks, expl, color=INK_MUTED, linewidth=2, linestyle="--", marker="o", markersize=5)
    ax.plot(ks, hit, color=SEQ_HUE, linewidth=2, marker="o", markersize=5)
    ax.annotate(
        "DAG coverage ceiling",
        (ks[-1], expl[-1]),
        textcoords="offset points",
        xytext=(-6, 8),
        ha="right",
        color=INK_MUTED,
        fontsize=8,
    )
    ax.annotate(
        "model hit rate",
        (ks[-1], hit[-1]),
        textcoords="offset points",
        xytext=(-6, -14),
        ha="right",
        color=SEQ_HUE,
        fontsize=8,
    )
    ax.set_xlabel("K (dominant-peak depth)")
    ax.set_ylabel("fraction of top-K true peaks")
    ax.set_ylim(0, 1)
    ax.set_xticks(ks)
    ax.set_title("Hit rate vs DAG ceiling — the gap is model headroom")

    row_order: list[str] = []
    for ax, k in ((axes[1, 0], 1), (axes[1, 1], 5)):
        oc = outcome_at_k(records, k)
        sub = records.loc[oc.index]
        inst = meta.loc[sub["spec_id"].to_numpy(), "inst_type"].to_numpy()
        prec = meta.loc[sub["spec_id"].to_numpy(), "prec_type"].to_numpy()
        key = pd.Series([f"{i} · {p}" for i, p in zip(inst, prec)], index=oc.index)
        sh = _shares(oc, key)
        sh = sh[sh["n"] >= 200]
        # Fix the row order on the K=1 panel so both panels stay row-comparable.
        if not row_order:
            row_order = sh.sort_values("hit").index.tolist()
        sh = sh.reindex([r for r in row_order if r in sh.index])
        _stacked(ax, sh, [f"{i}  (n={int(n):,})" for i, n in zip(sh.index, sh["n"])])
        ax.set_title(f"K = {k}, by instrument · adduct")

    _legend(fig)
    fig.suptitle(
        "Dominant MS/MS peaks: does the model rank them where the spectrum does?", fontsize=13
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(out_fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_fp)


def fig_ce_and_class(
    records: pd.DataFrame, spectra: pd.DataFrame, cf: pd.DataFrame | None, out_fp: Path
) -> None:
    """Outcome vs collision energy, ClassyFire class, and achieved cosine."""
    meta = spectra.set_index("spec_id")
    k = 1
    oc = outcome_at_k(records, k)
    sub = records.loc[oc.index]
    sids = sub["spec_id"].to_numpy()

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))

    for ax, (inst, col, unit, edges) in zip(
        (axes[0, 0], axes[0, 1]),
        (
            ("ORB", "nce", "NCE (%)", [0, 20, 40, 60, 80, 100, 120, 200]),
            ("ORB_F", "nce", "NCE (%)", [0, 20, 40, 60, 80, 100, 120, 200]),
            ("QTOF", "ace", "ACE (eV)", [0, 10, 20, 30, 40, 50, 100]),
        ),
    ):
        m = meta.loc[sids]
        keep = (m["inst_type"].to_numpy() == inst) & np.isfinite(m[col].to_numpy(dtype=float))
        ce = m[col].to_numpy(dtype=float)[keep]
        bucket = pd.cut(ce, bins=edges, right=False)
        sh = _shares(oc[keep], pd.Series(bucket.astype(str), index=oc[keep].index))
        sh = sh.reindex([str(b) for b in bucket.categories]).dropna(how="all")
        sh = sh[sh["n"] >= 50]
        _stacked(ax, sh, [f"{i}  (n={int(n):,})" for i, n in zip(sh.index, sh["n"])])
        ax.set_title(f"{inst}: base peak outcome vs {unit}")

    ax = axes[1, 0]
    if cf is not None:
        klass = cf.set_index("mol_id")["klass"]
        mol = meta.loc[sids, "mol_id"].to_numpy()
        kl = pd.Series(klass.reindex(mol).to_numpy(), index=oc.index)
        keep = kl.notna()
        sh = _shares(oc[keep], kl[keep])
        sh = sh[sh["n"] >= 500].sort_values("hit").head(14)
        _stacked(ax, sh, [f"{i[:34]}  (n={int(n):,})" for i, n in zip(sh.index, sh["n"])])
        ax.set_title("Worst 14 ClassyFire classes by base-peak hit rate")
    else:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "no ClassyFire table supplied", ha="center", color=INK_MUTED)

    ax = axes[1, 1]
    cos = meta.loc[sids, "cos_hun_inst_tol"].to_numpy(dtype=float)
    keep = np.isfinite(cos)
    dec = pd.cut(cos[keep], bins=np.linspace(0, 1, 11), right=False, include_lowest=True)
    sh = _shares(oc[keep], pd.Series(dec.astype(str), index=oc[keep].index))
    sh = sh.reindex([str(b) for b in dec.categories]).dropna(how="all")
    _stacked(ax, sh, [f"{i}  (n={int(n):,})" for i, n in zip(sh.index, sh["n"])])
    ax.set_title("Base-peak outcome vs achieved cos_hun_inst_tol")

    _legend(fig)
    fig.suptitle("Where the base peak goes wrong: energy, chemistry, and score", fontsize=13)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(out_fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_fp)


def fig_miss_anatomy(records: pd.DataFrame, spectra: pd.DataFrame, out_fp: Path) -> None:
    """Anatomy of base-peak misses: predicted rank, m/z placement, offset, unexplained mass."""
    base = records[records["true_rank"] == 0].set_index("spec_id")
    meta = spectra.set_index("spec_id")
    joined = meta.join(base[["matched", "in_dag", "pred_rank", "pred_int_frac"]], how="inner")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))

    ax = axes[0, 0]
    m = joined[joined["matched"]]
    r = np.clip(m["pred_rank"].to_numpy(), 0, 20)
    counts = np.bincount(r, minlength=21) / len(r)
    ax.bar(np.arange(21), counts, color=SEQ_HUE, width=0.7, edgecolor=SURFACE, linewidth=2)
    ax.text(
        0,
        counts[0],
        f"  {counts[0]:.0%} at rank 0",
        va="bottom",
        ha="left",
        fontsize=9,
        color=INK,
    )
    nodag = float((~joined["in_dag"]).mean())
    zero = float((joined["in_dag"] & ~joined["matched"]).mean())
    ax.set_xlabel("model's rank for the true base peak (20 = 20 or worse)")
    ax.set_ylabel("fraction of base peaks kept in the export")
    ax.set_title(
        f"Predicted rank of the true base peak\n"
        f"(a further {zero:.1%} predicted ~zero · {nodag:.1%} not in the DAG at all)"
    )

    ax = axes[0, 1]
    miss = joined[joined["matched"] & (joined["pred_rank"] > 0)]
    hb = ax.hexbin(
        miss["true_base_mz"],
        miss["pred_base_mz"],
        gridsize=60,
        bins="log",
        cmap="Blues",
        mincnt=1,
    )
    lim = (0, float(np.nanpercentile(joined["true_base_mz"], 99.5)))
    ax.plot(lim, lim, color=INK_MUTED, linewidth=1.2, linestyle="--")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("true base peak m/z")
    ax.set_ylabel("predicted base peak m/z")
    ax.set_title(f"Where the model puts its base peak instead (n={len(miss):,})")
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("spectra", color=INK_MUTED)

    ax = axes[1, 0]
    d = (miss["pred_base_mz"] - miss["true_base_mz"]).to_numpy()
    span = 60.0
    ax.hist(
        d[np.abs(d) <= span],
        bins=np.arange(-span, span + 0.2, 0.2),
        color=SEQ_HUE,
        edgecolor="none",
    )
    top = ax.get_ylim()[1]
    for j, (name, dm) in enumerate(NEUTRAL_LOSSES):
        for sign in (-1, 1):
            ax.axvline(sign * dm, color=INK_MUTED, linewidth=0.7, linestyle=":", zorder=0)
        # Stagger label heights: neighbouring losses (CH3/NH3/H2O, C2H2O/CO2/HCOOH) are only
        # 1-2 Da apart and would otherwise overprint.
        ax.text(
            dm,
            top * (0.98 if j % 2 == 0 else 0.80),
            name,
            rotation=90,
            ha="right",
            va="top",
            fontsize=7,
            color=INK_MUTED,
        )
    prec_frac = float((np.abs(miss["pred_base_mz"] - miss["prec_mz"]) <= 0.02).mean())
    local_frac = float(np.mean(np.abs(d) <= span))
    up_frac = float(np.mean(d > 0))
    ax.set_xlim(-span, span)
    ax.set_xlabel("predicted base peak m/z − true base peak m/z (Da)")
    ax.set_ylabel("spectra")
    ax.set_title(
        f"The wrong base peak is one neutral loss away\n"
        f"{local_frac:.0%} within ±{span:.0f} Da · {up_frac:.0%} higher m/z · "
        f"{prec_frac:.0%} on the precursor"
    )

    ax = axes[1, 1]
    fin = joined[np.isfinite(joined["cos_hun_inst_tol"])]
    bins = np.linspace(0, 1, 21)
    b = pd.cut(fin["unexpl_int_frac"], bins=bins, include_lowest=True)
    g = fin.groupby(b, observed=True)["cos_hun_inst_tol"].agg(["mean", "size"])
    g = g[g["size"] >= 30]
    centers = np.array([iv.mid for iv in g.index])
    ax.plot(centers, g["mean"], color=SEQ_HUE, linewidth=2, marker="o", markersize=5)
    ax.set_xlabel("fraction of true intensity the DAG cannot explain")
    ax.set_ylabel("mean cos_hun_inst_tol")
    ax.set_ylim(0, 1)
    med = float(fin["unexpl_int_frac"].median())
    ax.axvline(med, color=INK_MUTED, linewidth=1.0, linestyle="--")
    ax.annotate(
        f"median {med:.2f}",
        (med, 0.05),
        textcoords="offset points",
        xytext=(6, 0),
        color=INK_MUTED,
        fontsize=8,
    )
    ax.set_title("Unexplained fragment mass caps the score")

    fig.suptitle("Anatomy of a missed base peak", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_fp)


def write_summary(records: pd.DataFrame, spectra: pd.DataFrame, out_fp: Path) -> None:
    """Write the per-K and per-stratum outcome shares to CSV."""
    meta = spectra.set_index("spec_id")
    rows = []
    for k in range(1, K_MAX + 1):
        oc = outcome_at_k(records, k)
        sub = records.loc[oc.index]
        sh = _shares(oc).iloc[0]
        rows.append({"K": k, "stratum": "all", "group": "all", **sh.to_dict()})
        for name, col in (("inst_type", "inst_type"), ("prec_type", "prec_type")):
            key = pd.Series(meta.loc[sub["spec_id"].to_numpy(), col].to_numpy(), index=oc.index)
            for g, r in _shares(oc, key).iterrows():
                rows.append({"K": k, "stratum": name, "group": g, **r.to_dict()})
    pd.DataFrame(rows).to_csv(out_fp, index=False)
    log.info("Wrote %s", out_fp)


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--pred_fp", required=True, help="Path to the predictions HDF5")
    p.add_argument("--spec_df_fp", required=True, help="Path to the ground-truth spec_df.pkl")
    p.add_argument(
        "--frag_dp",
        required=True,
        help="Directory of {mol_id}.pkl.bz2 DAG files, used for unfiltered DAG coverage",
    )
    p.add_argument("--classyfire_fp", default=None, help="Path to the mol ClassyFire pkl")
    p.add_argument("--out_dp", required=True, help="Output directory for CSVs and figures")
    return p.parse_args()


def main() -> None:
    args = get_args()
    out_dp = Path(args.out_dp)
    out_dp.mkdir(parents=True, exist_ok=True)
    _apply_style()

    meta, pred_mzs, pred_ints = load_predictions(args.pred_fp)
    spec_df = pd.read_pickle(args.spec_df_fp)
    missing = set(meta["spec_id"].tolist()) - set(spec_df["spec_id"].tolist())
    if missing:
        raise ValueError(f"{len(missing)} predicted spec_ids absent from {args.spec_df_fp}")

    dag = DagSupport(Path(args.frag_dp))
    records, spectra = build_records(meta, pred_mzs, pred_ints, spec_df, dag)
    records.to_csv(out_dp / "dominant_peak_records.csv", index=False)
    spectra.to_csv(out_dp / "dominant_peak_spectra.csv", index=False)

    cf = pd.read_pickle(args.classyfire_fp) if args.classyfire_fp else None

    write_summary(records, spectra, out_dp / "dominant_peak_summary.csv")
    fig_topk_outcome(records, spectra, out_dp / "dominant_peak_topk_outcome.png")
    fig_ce_and_class(records, spectra, cf, out_dp / "dominant_peak_ce_and_class.png")
    fig_miss_anatomy(records, spectra, out_dp / "dominant_peak_miss_anatomy.png")

    for k in K_GRID:
        oc = outcome_at_k(records, k)
        sh = _shares(oc).iloc[0]
        log.info(
            "K=%2d  hit %.3f  under_ranked %.3f  zero_pred %.3f  no_dag %.3f  (n=%d)",
            k,
            sh["hit"],
            sh["under_ranked"],
            sh["zero_pred"],
            sh["no_dag"],
            int(sh["n"]),
        )


if __name__ == "__main__":
    main()
