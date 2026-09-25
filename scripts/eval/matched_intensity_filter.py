"""Cosine under the same relative-intensity filter on the prediction and the target.

Offline, on a ``run_save_inference_eval.py`` HDF5 exported with ``--min_peak_int 0`` (full
candidate set) and a dumped eval-target pickle (``spec_id``, ``nce``, ``peaks``). For a threshold
``t`` a peak survives when its intensity is strictly above ``t`` times the spectrum's largest
peak -- the rule of ``spec_utils.batched_rel_filter_func``, i.e. ``target_rel_ints_thresh``.

- **Target:** exactly that function: the maximum is taken over all peaks, then peaks at or above
  ``mz_max`` are dropped.
- **Prediction:** MASSter emits several candidates at one m/z (connected fragments that share a
  formula), and the model sums them before scoring (``round_aggregate_peaks``, 4 decimals). The
  filter therefore acts on those aggregated peaks: a candidate survives when its peak does.
  Exports written after 2026-09-24 already hold summed peaks (the exporter aggregates
  MASSter candidates the same way), and the aggregation here is then a no-op.

Each threshold is scored in three modes: ``both`` (the matched comparison), ``pred_only`` and
``target_only``; ``t = 0`` is the unfiltered row (mode ``none``). Two metrics:

- ``cos_hun``: Hungarian cosine at the FT tolerance (10 ppm, 200 Da floor) on round-aggregated
  peaks, the in-model ``cos_hun``/``cos_hun_inst_tol`` on an FT-only split. The ``none`` row
  reproduces the exported column (mean identical to 5 decimals on 52,020 val spectra).
- ``cos_sim``: 0.01 Da binned cosine with ``batched_bin_func``'s bin edges, in float64. The model
  bins its float32 targets, and a target peak lying exactly on a bin edge can land in the other
  bin, so per spectrum this differs from the exported ``cos_sim_0.01`` on ~3% of spectra (mean
  +0.0001 on val). Peak positions do not depend on ``t``, so read its changes, not its level.

Predictions do not depend on the target, so one raw-target export can be scored against any
target variant; ``targets`` builds a variant by overriding ``spec_params`` cleanup keys of an
experiment config and dumping the dataset's eval targets. ``plot`` compares every run with the one
labelled ``raw`` on the spectra they share.

Stages::

    python scripts/eval/matched_intensity_filter.py targets --config C.yml --split val \\
        --set centroid_peaks=true filter_subformula_peaks=true --out_fp T.pkl
    python scripts/eval/matched_intensity_filter.py score --pred_hdf5 P.hdf5 \\
        --targets_fp T.pkl --out_fp scores.pkl
    python scripts/eval/matched_intensity_filter.py plot --runs raw=a.pkl cleaned=b.pkl \\
        --out_dp docs/figures
    python scripts/eval/matched_intensity_filter.py export --pred_hdf5 P.hdf5 \\
        --scores_fp scores.pkl --targets_fp T.pkl --thresh 0.001 --out_dp D

``export`` writes a both-sides-filtered copy of the export and a matching ``spec_df``, so
``scripts/plotting/plot_predictions_hdf5.py`` can draw one threshold without re-running the model.
"""

from __future__ import annotations

import argparse
import logging
from multiprocessing import Pool
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd
import torch as th
import yaml
from scipy.optimize import linear_sum_assignment

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LOG = logging.getLogger("matched_intensity_filter")

MZ_MAX = 1500.0
BIN_RES = 0.01
FT_TOL_REL, FT_TOL_MIN_MZ = 1e-5, 200.0
THRESHOLDS = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05)
MODES = ("both", "pred_only", "target_only")
# Same edges as spec_utils.batched_bin_func: arange(res, mz_max + res, res), searchsorted right.
BIN_EDGES = th.arange(BIN_RES, MZ_MAX + BIN_RES, step=BIN_RES, dtype=th.float64).numpy()
NCE_BINS = [0, 20, 40, 60, 80, 100, 150, 1000]

# Filled by run_score before the worker pool forks; workers read it by spectrum index.
_DATA: dict[str, np.ndarray | list] = {}


def round_agg(mzs: np.ndarray, ints: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Numpy twin of ``spec_utils.round_aggregate_peaks(decimals=4, agg="sum")``.

    Args:
        mzs: Peak m/z, shape (N,).
        ints: Peak intensities, shape (N,).

    Returns:
        Unique rounded m/z (U,), summed intensity per unique m/z (U,), and the index of each
        input peak's aggregate (N,).
    """
    rounded = th.round(th.as_tensor(mzs), decimals=4).numpy()
    uniq, inv = np.unique(rounded, return_inverse=True)
    return uniq, np.bincount(inv, weights=ints, minlength=len(uniq)), inv


def pred_keep_mask(mzs: np.ndarray, ints: np.ndarray, t: float) -> np.ndarray:
    """Candidates whose aggregated peak is strictly above ``t`` times the largest peak, (N,)."""
    if t <= 0:
        return ints > 0
    _, peak_ints, inv = round_agg(mzs, ints)
    return peak_ints[inv] > t * peak_ints.max()


def pred_peak_filter(mzs: np.ndarray, ints: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
    """Keep candidates whose aggregated peak is strictly above ``t`` times the largest peak."""
    keep = pred_keep_mask(mzs, ints, t)
    return mzs[keep], ints[keep]


def target_filter(mzs: np.ndarray, ints: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
    """``batched_rel_filter_func``: strict ``> t * max`` over all peaks, then ``m/z < MZ_MAX``."""
    keep = ints > t * ints.max() if t > 0 else ints > 0
    keep &= mzs < MZ_MAX
    return mzs[keep], ints[keep]


def cos_binned(pm: np.ndarray, pi: np.ndarray, tm: np.ndarray, ti: np.ndarray) -> float:
    """0.01 Da binned cosine (sum within a bin), as ``batched_bin_func`` + ``cos_sim_helper``."""
    in_range = pm < MZ_MAX
    pb = np.searchsorted(BIN_EDGES, pm[in_range], side="right")
    tb = np.searchsorted(BIN_EDGES, tm, side="right")
    pu, pinv = np.unique(pb, return_inverse=True)
    tu, tinv = np.unique(tb, return_inverse=True)
    pw = np.bincount(pinv, weights=pi[in_range])
    tw = np.bincount(tinv, weights=ti)
    if len(pw) == 0 or len(tw) == 0:
        return 0.0
    _, p_idx, t_idx = np.intersect1d(pu, tu, return_indices=True)
    return float(pw[p_idx] @ tw[t_idx] / (np.linalg.norm(pw) * np.linalg.norm(tw)))


def cos_hungarian(pm: np.ndarray, pi: np.ndarray, tm: np.ndarray, ti: np.ndarray) -> float:
    """Hungarian cosine at the FT tolerance on round-aggregated peaks (in-model ``cos_hun``).

    A predicted and a true peak can match when ``|mz_p - mz_t| < tol_rel * max(mz_t, min_mz)``.
    With both spectra L2-normalized, the score is the maximum-weight one-to-one matching of
    ``p_v * t_u`` over matchable pairs.
    """
    pm, pi, _ = round_agg(pm, pi)
    tm, ti, _ = round_agg(tm, ti)
    if len(pm) == 0 or len(tm) == 0:
        return 0.0
    tol = FT_TOL_REL * np.maximum(tm, FT_TOL_MIN_MZ)
    match = np.abs(tm[:, None] - pm[None, :]) < tol[:, None]
    rows, cols = match.any(axis=1), match.any(axis=0)
    if not rows.any():
        return 0.0
    score = match[rows][:, cols] * (ti[rows, None] / np.linalg.norm(ti)) * (
        pi[None, cols] / np.linalg.norm(pi)
    )
    r, c = linear_sum_assignment(score, maximize=True)
    return float(score[r, c].sum())


def score_spectrum(i: int) -> list[tuple]:
    """Every (mode, threshold) score for spectrum ``i`` of ``_DATA``."""
    s, e = _DATA["starts"][i], _DATA["ends"][i]
    pm0 = _DATA["pred_mzs"][s:e].astype(np.float64)
    pi0 = _DATA["pred_ints"][s:e].astype(np.float64)
    peaks = _DATA["target_peaks"][i]
    settings = [("none", 0.0, 0.0, 0.0)] + [
        (mode, t, t if mode != "target_only" else 0.0, t if mode != "pred_only" else 0.0)
        for t in THRESHOLDS
        for mode in MODES
    ]
    out = []
    for mode, t, t_pred, t_true in settings:
        pm, pi = pred_peak_filter(pm0, pi0, t_pred)
        tm, ti = target_filter(peaks[:, 0], peaks[:, 1], t_true)
        n_pred = len(np.unique(th.round(th.as_tensor(pm), decimals=4).numpy()))
        out.append(
            (i, mode, t, cos_binned(pm, pi, tm, ti), cos_hungarian(pm, pi, tm, ti), n_pred, len(tm))
        )
    return out


def run_targets(args: argparse.Namespace) -> None:
    """Dump the eval targets a config's ``spec_params`` (plus overrides) builds for one split."""
    from fragnnet.runner import load_config
    from fragnnet.utils.script_utils import apply_inference_config
    from fragnnet.utils.setup_utils import get_dataset_cls

    # 1. Resolve the config and apply the spec_params overrides (YAML-parsed values).
    config_d = load_config("config", args.config, [])
    for item in args.set:
        key, value = item.split("=", 1)
        if key not in config_d["spec_params"]:
            raise KeyError(f"spec_params has no key {key!r}")
        config_d["spec_params"][key] = yaml.safe_load(value)
    config_d = apply_inference_config(
        config_d,
        auxiliary_scores=["cos_sim", "cos_hun"],
        skip_extra_losses=False,
        disable_preproc=True,
        output_formula_str=False,
        train_batch_size=4,
        eval_batch_size=512,
        dynamic_batch_sampler=False,
    )
    # 2. Build the split exactly as run_save_inference_eval.py does; cleanup runs at load.
    ds = get_dataset_cls(config_d["model_type"])(split=args.split, **config_d)
    cols = ["spec_id", "mol_id", "nce", "prec_mz", "peaks"]
    ds.spec_df[cols].to_pickle(args.out_fp)
    n_peaks = ds.spec_df["peaks"].map(len)
    LOG.info("wrote %s: %d spectra, median %d peaks", args.out_fp, len(ds.spec_df), n_peaks.median())


def run_score(args: argparse.Namespace) -> None:
    """Score every spectrum of the export; check the unfiltered row against the export."""
    # 1. Load the export and align the targets to its spectrum order. A target variant can have
    #    fewer spectra than the export (cleanup empties some); only shared spectra are scored.
    with h5py.File(args.pred_hdf5, "r") as f:
        spec_ids = f["spec_id"][:]
        ref = {k: f[k][:] for k in ("cos_sim_0.01", "cos_hun") if k in f}
        starts, ends = f["start_points"][:], f["end_points"][:]
        _DATA.update(pred_mzs=f["pred_mzs"][:], pred_ints=f["pred_ints"][:])
    targets = pd.read_pickle(args.targets_fp).set_index("spec_id")
    keep = np.isin(spec_ids, targets.index)
    if not keep.any():
        raise ValueError("No spectrum of the export is in the target file")
    LOG.info("%d of %d exported spectra are in the target file", keep.sum(), len(spec_ids))
    spec_ids, ref = spec_ids[keep], {k: v[keep] for k, v in ref.items()}
    _DATA.update(starts=starts[keep], ends=ends[keep])
    targets = targets.loc[spec_ids]
    _DATA["target_peaks"] = [np.asarray(p, dtype=np.float64) for p in targets["peaks"]]
    LOG.info("%d spectra, %d exported peaks", len(spec_ids), len(_DATA["pred_mzs"]))

    # 2. Score in a forked pool (workers inherit _DATA).
    with Pool(args.n_jobs) as pool:
        rows = [r for rs in pool.imap(score_spectrum, range(len(spec_ids)), chunksize=64) for r in rs]
    df = pd.DataFrame(rows, columns=["idx", "mode", "thresh", "cos_sim", "cos_hun", "n_pred", "n_true"])
    nce = np.array([float(np.ravel(x)[0]) for x in targets["nce"]])
    df["spec_id"] = spec_ids[df["idx"]]
    df["nce"] = nce[df["idx"]]
    df = df.drop(columns="idx")

    # 3. The unfiltered row must reproduce the export's in-model scores, when the export was
    #    scored against this same target.
    none = df[df["mode"] == "none"]
    for key, col in (("cos_hun", "cos_hun"), ("cos_sim_0.01", "cos_sim")):
        if key in ref and not args.skip_export_check:
            d = none[col].to_numpy() - ref[key]
            LOG.info(
                "check %s: offline mean %.5f, export %.5f, n(|d|>1e-5) %d",
                key, none[col].mean(), ref[key].mean(), int((np.abs(d) > 1e-5).sum()),
            )
    df.to_pickle(args.out_fp)
    LOG.info("wrote %s\n%s", args.out_fp, summarize(df).round(5).to_string(index=False))


def run_export(args: argparse.Namespace) -> None:
    """Write a both-sides-filtered copy of an export for ``plot_predictions_hdf5.py``.

    From one unfiltered export and its ``score`` output: the HDF5 keeps only prediction peaks that
    pass ``pred_peak_filter`` at ``--thresh``, with candidates at one m/z summed into one peak
    (``round_agg``, the aggregation the model scores) and each spectrum rescaled to a base peak of
    1000, so downstream plots count and draw peaks rather than candidates. Its score columns are
    replaced by that threshold's both-sides scores (``cos_hun`` at the FT tolerance also fills
    ``cos_hun_inst_tol``, which it equals on FT-only data; ``cos_sim_0.01`` is the offline binned
    cosine). The ``spec_df`` carries the target filtered with ``target_filter`` at the same
    threshold, for the plots' ground truth.
    """
    t = args.thresh
    out_dp = Path(args.out_dp)
    out_dp.mkdir(parents=True, exist_ok=True)

    # 1. Filter the prediction by peak, sum candidates at one m/z, and rebuild the ragged index.
    with h5py.File(args.pred_hdf5, "r") as f:
        spec_ids = f["spec_id"][:]
        starts, ends = f["start_points"][:], f["end_points"][:]
        mzs, ints = f["pred_mzs"][:], f["pred_ints"][:]
        passthrough = {k: f[k][:] for k in ("mol_id", "index_rebased") if k in f}
    out_mzs, out_ints = [], []
    for s, e in zip(starts, ends):
        pm, pi = pred_peak_filter(mzs[s:e].astype(np.float64), ints[s:e].astype(np.float64), t)
        peak_mzs, peak_ints, _ = round_agg(pm, pi)
        out_mzs.append(peak_mzs)
        out_ints.append(1000.0 * peak_ints / peak_ints.max())
    counts = np.array([len(m) for m in out_mzs])
    new_ends = np.cumsum(counts)

    # 2. Scores for this threshold, both sides filtered, in export order.
    scores = pd.read_pickle(args.scores_fp)
    mode = "none" if t == 0 else "both"
    sel = scores[(scores["mode"] == mode) & np.isclose(scores["thresh"], t)].set_index("spec_id")
    sel = sel.loc[spec_ids]

    out_fp = out_dp / Path(args.pred_hdf5).name
    with h5py.File(out_fp, "w") as f:
        f.create_dataset("pred_mzs", data=np.concatenate(out_mzs), compression="gzip")
        f.create_dataset("pred_ints", data=np.concatenate(out_ints).astype(ints.dtype),
                         compression="gzip")
        f.create_dataset("spec_id", data=spec_ids)
        f.create_dataset("start_points", data=new_ends - counts)
        f.create_dataset("end_points", data=new_ends)
        for key, value in passthrough.items():
            f.create_dataset(key, data=value)
        f.create_dataset("cos_hun", data=sel["cos_hun"].to_numpy())
        f.create_dataset("cos_hun_inst_tol", data=sel["cos_hun"].to_numpy())
        f.create_dataset("cos_sim_0.01", data=sel["cos_sim"].to_numpy())
        f.attrs["filter_thresh"] = t
        f.attrs["source_export"] = str(args.pred_hdf5)

    # 3. Ground truth for the plots: spec_df rows with the target filtered the same way.
    targets = pd.read_pickle(args.targets_fp).set_index("spec_id")["peaks"]
    filtered = {
        sid: list(zip(*target_filter(*np.asarray(targets[sid], dtype=np.float64).T, t)))
        for sid in spec_ids
    }
    spec_df = pd.read_pickle(args.spec_df_fp)
    spec_df = spec_df[spec_df["spec_id"].isin(spec_ids)].copy()
    spec_df["peaks"] = spec_df["spec_id"].map(filtered)
    spec_df.to_pickle(out_dp / f"spec_df_{Path(args.pred_hdf5).stem.split('_')[-1]}.pkl")
    LOG.info(
        "wrote %s: %d spectra, %d peaks from %d input rows (median %d peaks per spectrum), "
        "cos_hun %.5f",
        out_fp, len(spec_ids), counts.sum(), len(mzs), np.median(counts), sel["cos_hun"].mean(),
    )


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Mean scores, median peak counts, and change from the unfiltered row per (mode, t)."""
    g = df.groupby(["mode", "thresh"], sort=False).agg(
        cos_hun=("cos_hun", "mean"), cos_sim=("cos_sim", "mean"),
        n_pred_median=("n_pred", "median"), n_true_median=("n_true", "median"),
    ).reset_index()
    base = g[g["mode"] == "none"].iloc[0]
    g["d_cos_hun"] = g["cos_hun"] - base["cos_hun"]
    g["d_cos_sim"] = g["cos_sim"] - base["cos_sim"]
    return g


def _paired_delta(df: pd.DataFrame, mode: str, t: float) -> pd.DataFrame:
    """Per-spectrum ``cos_hun`` change from the unfiltered row, with NCE."""
    a = df[df["mode"] == "none"].set_index("spec_id")
    b = df[(df["mode"] == mode) & (df["thresh"] == t)].set_index("spec_id")
    return pd.DataFrame({"d": b["cos_hun"] - a["cos_hun"], "nce": a["nce"]})


INK, MUTED, GRID = "#0b0b0b", "#52514e", "#c9c8c4"
# dataviz reference palette: categorical slots 1-3 (validated all-pairs, light), fixed order.
MODE_STYLE = {
    "both": ("#2a78d6", "both sides"),
    "pred_only": ("#eb6834", "prediction only"),
    "target_only": ("#1baf7a", "target only"),
}
# dataviz reference sequential blue, ordinal steps 250 and 500: lighter = lower threshold.
THRESH_RAMP = {0.005: "#86b6ef", 0.01: "#256abf"}


def _style(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", color=GRID, lw=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_sweep(runs: dict[str, pd.DataFrame], out_fp: Path) -> None:
    """Change in ``cos_hun`` and median peak count against the threshold, one column per target."""
    fig, axes = plt.subplots(2, len(runs), figsize=(5.2 * len(runs), 7.4), sharex=True,
                             sharey="row", squeeze=False)
    y_span = max(-summarize(df)["d_cos_hun"].min() for df in runs.values())
    for col, (label, df) in enumerate(runs.items()):
        s = summarize(df)
        base = s[s["mode"] == "none"].iloc[0]
        ax = axes[0, col]
        ax.axhline(0, color=MUTED, lw=0.8)
        ends = []
        for mode, (color, name) in MODE_STYLE.items():
            m = s[s["mode"] == mode]
            x, y = 100 * m["thresh"].to_numpy(), m["d_cos_hun"].to_numpy()
            ax.plot(x, y, color=color, lw=2, marker="o", ms=6, mec="white", mew=1.5, label=name)
            ends.append((y[-1], name, x[-1]))
        # Direct labels at the line ends, pushed apart when two ends nearly coincide.
        ends.sort(reverse=True)
        for k, (y, name, x) in enumerate(ends):
            close_above = k > 0 and ends[k - 1][0] - y < 0.04 * y_span
            close_below = k + 1 < len(ends) and y - ends[k + 1][0] < 0.04 * y_span
            dy = -6 if close_above else (6 if close_below else 0)
            ax.annotate(name, (x, y), xytext=(6, dy), textcoords="offset points",
                        va="center", fontsize=9, color=INK)
        ax.set_title(f"{label} target, n={df['spec_id'].nunique():,}\n"
                     f"unfiltered cos_hun {base['cos_hun']:.4f}", fontsize=11)
        _style(ax)
        ax = axes[1, col]
        both = s[s["mode"] == "both"]
        x = 100 * both["thresh"].to_numpy()
        ax.plot(x, both["n_pred_median"], color=INK, lw=2, marker="o", ms=6, label="predicted")
        ax.plot(x, both["n_true_median"], color=MUTED, lw=2, ls="--", marker="s", ms=6,
                label="true")
        for key, name in (("n_pred_median", "predicted"), ("n_true_median", "true")):
            ax.annotate(name, (x[0], both[key].iloc[0]), xytext=(0, 7), textcoords="offset points",
                        ha="center", fontsize=9, color=INK)
        ax.text(0.98, 0.95, f"unfiltered: {base['n_pred_median']:.0f} predicted, "
                f"{base['n_true_median']:.0f} true", transform=ax.transAxes, ha="right",
                va="top", fontsize=9, color=MUTED)
        ax.set_xscale("log")
        ax.set_xticks(100 * np.array(THRESHOLDS), [f"{100 * t:g}" for t in THRESHOLDS])
        ax.set_xlabel("filter threshold (% of base peak)")
        _style(ax)
    axes[0, 0].set_ylabel("change in cos_hun vs unfiltered")
    axes[1, 0].set_ylabel("median peaks per spectrum\n(both sides filtered)")
    axes[0, 0].legend(frameon=False, fontsize=9, loc="lower left")
    axes[1, 0].legend(frameon=False, fontsize=9, loc="center right")
    axes[0, -1].set_xlim(right=100 * THRESHOLDS[-1] * 2.2)
    for ax in axes.flat:  # the two thresholds this comparison was asked about
        for t in THRESH_RAMP:
            ax.axvline(100 * t, color=GRID, lw=1, ls=":", zorder=0)
    fig.tight_layout()
    fig.savefig(out_fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOG.info("wrote %s", out_fp)


NCE_LABELS = [f"{lo}-{hi}" if hi < 1000 else f"{lo}+" for lo, hi in zip(NCE_BINS, NCE_BINS[1:])]
# dataviz reference categorical slots 1-3 in fixed order, one per cleaned target.
TARGET_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]


def _by_nce(d: pd.DataFrame) -> pd.DataFrame:
    cut = pd.cut(d["nce"], NCE_BINS, right=False, labels=NCE_LABELS)
    return d.groupby(cut, observed=False)["d"].agg(["mean", "size"])


def cleanup_gain(raw: pd.DataFrame, cleaned: pd.DataFrame) -> pd.DataFrame:
    """Per-spectrum unfiltered ``cos_hun`` change from the raw target, on the shared spectra."""
    a = raw[raw["mode"] == "none"].set_index("spec_id")
    b = cleaned[cleaned["mode"] == "none"].set_index("spec_id")
    common = a.index.intersection(b.index)
    return pd.DataFrame({"d": b.loc[common, "cos_hun"] - a.loc[common, "cos_hun"],
                         "nce": a.loc[common, "nce"]})


def plot_nce(runs: dict[str, pd.DataFrame], out_fp: Path) -> None:
    """Cleanup gain of every non-raw target, and matched-filter cost on the raw target, by NCE."""
    raw = runs["raw"]
    cleaned = {k: v for k, v in runs.items() if k != "raw"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    x = np.arange(len(NCE_LABELS))
    w = 0.8 / len(cleaned)
    for k, ((label, df), color) in enumerate(zip(cleaned.items(), TARGET_COLORS, strict=False)):
        g = _by_nce(cleanup_gain(raw, df))
        ax1.bar(x + (k - (len(cleaned) - 1) / 2) * w, g["mean"], width=w - 0.03, color=color,
                label=label)
    ax1.set_title("target cleanup gain (cleaned - raw, unfiltered)", fontsize=11)
    ax1.set_ylabel("change in cos_hun")
    if len(cleaned) > 1:
        ax1.legend(frameon=False, fontsize=9, loc="upper left")
    n_raw = _by_nce(_paired_delta(raw, "both", 0.01))["size"]
    ticks = [f"{lab}\nn={n:,}" for lab, n in zip(NCE_LABELS, n_raw)]
    ax1.set_xticks(x, ticks, fontsize=8)
    w = 0.36
    for k, (t, color) in enumerate(THRESH_RAMP.items()):
        d = _by_nce(_paired_delta(raw, "both", t))
        ax2.bar(x + (k - 0.5) * (w + 0.02), d["mean"], width=w, color=color,
                label=f"{100 * t:g}% both sides")
    ax2.axhline(0, color=MUTED, lw=0.8)
    ax2.set_title("matched intensity filter (raw target)", fontsize=11)
    ax2.set_xticks(x, ticks, fontsize=8)
    ax2.legend(frameon=False, fontsize=9, loc="lower left")
    for ax in (ax1, ax2):
        ax.set_xlabel("NCE bin")
        _style(ax)
    fig.tight_layout()
    fig.savefig(out_fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOG.info("wrote %s", out_fp)


def plot_per_spectrum(raw: pd.DataFrame, out_fp: Path) -> None:
    """Distribution of per-spectrum ``cos_hun`` change under the matched filter (raw target)."""
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    deltas = {t: _paired_delta(raw, "both", t)["d"].to_numpy() for t in THRESH_RAMP}
    lo = min(d.min() for d in deltas.values()) - 1e-3
    hi = max(max(d.max() for d in deltas.values()), 0.0) + 1e-3
    edges = np.linspace(lo, hi, 61)
    ax.axvline(0, color=MUTED, lw=0.8)
    for t, color in THRESH_RAMP.items():
        d = deltas[t]
        label = (f"{100 * t:g}% both sides: {np.mean(np.abs(d) < 1e-3):.0%} within ±0.001, "
                 f"{np.mean(d < -0.01):.1%} lose >0.01")
        ax.hist(d, bins=edges, histtype="step", lw=2, color=color, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("per-spectrum change in cos_hun vs unfiltered")
    ax.set_ylabel("spectra")
    ax.set_title("matched intensity filter, per spectrum (raw target)", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(out_fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOG.info("wrote %s", out_fp)


def target_table(runs: dict[str, pd.DataFrame], target_fps: dict[str, str]) -> pd.DataFrame:
    """One row per target: spectra kept, retention against the raw target, paired score change.

    Retention compares each variant with the raw target on shared spectra (raw peaks below
    ``MZ_MAX``): the kept fraction of total intensity and of peak count, and whether a peak lies
    within 20 ppm of the raw base peak. The score columns are unfiltered ``cos_hun``.
    """
    raw_peaks = pd.read_pickle(target_fps["raw"]).set_index("spec_id")["peaks"]
    raw_none = runs["raw"][runs["raw"]["mode"] == "none"].set_index("spec_id")["cos_hun"]
    rows = []
    for label, df in runs.items():
        none = df[df["mode"] == "none"].set_index("spec_id")["cos_hun"]
        shared = none.index.intersection(raw_none.index)
        gain = none.loc[shared] - raw_none.loc[shared]
        row = {
            "target": label,
            "n_spectra": len(none),
            "n_dropped_vs_raw": len(raw_none) - len(shared),
            "cos_hun": none.mean(),
            "raw_cos_hun_on_shared": raw_none.loc[shared].mean(),
            "gain_vs_raw_on_shared": gain.mean(),
            "frac_spectra_up": (gain > 1e-9).mean(),
            "frac_spectra_down": (gain < -1e-9).mean(),
        }
        if label in target_fps:
            peaks = pd.read_pickle(target_fps[label]).set_index("spec_id")["peaks"]
            kept_int, kept_n, base_kept = [], [], []
            for sid in peaks.index.intersection(raw_peaks.index):
                r, q = np.asarray(raw_peaks[sid], float), np.asarray(peaks[sid], float)
                r = r[r[:, 0] < MZ_MAX]
                base = r[np.argmax(r[:, 1]), 0]
                kept_int.append(q[:, 1].sum() / r[:, 1].sum())
                kept_n.append(len(q) / len(r))
                base_kept.append(bool(np.any(np.abs(q[:, 0] - base) <= 20e-6 * base)))
            row.update(intensity_kept=np.mean(kept_int), peaks_kept=np.mean(kept_n),
                       base_peak_kept=np.mean(base_kept),
                       median_target_peaks=float(np.median(peaks.map(len))))
        rows.append(row)
    return pd.DataFrame(rows)


def run_plot(args: argparse.Namespace) -> None:
    """Write the three figures and a summary CSV from ``score`` outputs."""
    runs = {k: pd.read_pickle(v) for k, v in (r.split("=", 1) for r in args.runs)}
    out_dp = Path(args.out_dp)
    out_dp.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix
    plot_sweep(runs, out_dp / f"{prefix}_threshold_sweep.png")
    if "raw" in runs and len(runs) > 1:
        plot_nce(runs, out_dp / f"{prefix}_nce.png")
    plot_per_spectrum(next(iter(runs.values())), out_dp / f"{prefix}_per_spectrum.png")

    # Tables: overall per (target, mode, threshold), and the same by NCE bin with the shared-set
    # gain over the raw target.
    summary = pd.concat([summarize(df).assign(target=k, n_spectra=df["spec_id"].nunique())
                         for k, df in runs.items()])
    summary.to_csv(out_dp / f"{prefix}_summary.csv", index=False)
    by_nce = []
    for k, df in runs.items():
        cut = pd.cut(df["nce"], NCE_BINS, right=False, labels=NCE_LABELS)
        g = df.groupby(["mode", "thresh", cut], observed=True, sort=False)["cos_hun"].agg(
            ["mean", "size"]).reset_index().rename(columns={"mean": "cos_hun", "size": "n"})
        if k != "raw" and "raw" in runs:
            gain = _by_nce(cleanup_gain(runs["raw"], df))["mean"].rename("gain_vs_raw_unfiltered")
            g = g.merge(gain, left_on="nce", right_index=True, how="left")
        by_nce.append(g.assign(target=k))
    pd.concat(by_nce).to_csv(out_dp / f"{prefix}_by_nce.csv", index=False)
    LOG.info("summary\n%s", summary.round(5).to_string(index=False))
    if args.target_files:
        target_fps = dict(t.split("=", 1) for t in args.target_files)
        if "raw" not in target_fps or "raw" not in runs:
            raise ValueError("--target_files needs a raw target, matching a raw run")
        targets = target_table(runs, target_fps)
        targets.to_csv(out_dp / f"{prefix}_targets.csv", index=False)
        LOG.info("targets\n%s", targets.round(4).to_string(index=False))


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="stage", required=True)
    s = sub.add_parser("targets")
    s.add_argument("--config", required=True, help="Experiment config (custom_fp).")
    s.add_argument("--split", default="val")
    s.add_argument("--set", nargs="*", default=[], help="spec_params overrides, key=value.")
    s.add_argument("--out_fp", required=True)
    s = sub.add_parser("score")
    s.add_argument("--pred_hdf5", required=True, help="Export with --min_peak_int 0.")
    s.add_argument("--targets_fp", required=True, help="Pickle with spec_id, nce, peaks.")
    s.add_argument("--out_fp", required=True)
    s.add_argument("--n_jobs", type=int, default=10)
    s.add_argument("--skip_export_check", action="store_true",
                   help="The export was scored against a different target; skip the t=0 check.")
    s = sub.add_parser("export")
    s.add_argument("--pred_hdf5", required=True, help="The unfiltered export that was scored.")
    s.add_argument("--scores_fp", required=True, help="score output for that export and target.")
    s.add_argument("--targets_fp", required=True, help="The target pickle that was scored.")
    s.add_argument("--spec_df_fp", default="data/proc/nist20/spec_df.pkl")
    s.add_argument("--thresh", type=float, required=True, help="One of the scored thresholds.")
    s.add_argument("--out_dp", required=True)
    s = sub.add_parser("plot")
    s.add_argument("--runs", nargs="+", required=True,
                   help="label=scores.pkl; the first run is the per-spectrum panel's.")
    s.add_argument("--out_dp", required=True)
    s.add_argument("--prefix", default="matched_intensity_filter")
    s.add_argument("--target_files", nargs="*", default=[],
                   help="label=targets.pkl for the retention table; needs raw=... as well.")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cli = get_args()
    if cli.stage == "targets":
        run_targets(cli)
    elif cli.stage == "score":
        run_score(cli)
    elif cli.stage == "export":
        run_export(cli)
    else:
        run_plot(cli)
