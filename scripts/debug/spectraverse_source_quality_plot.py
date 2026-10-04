"""Plot how much of each Spectraverse source's signal the labelled structure explains.

A peak is explained when it lies within the instrument tolerance of a precursor-bounded subformula
ion of the labelled molecule and adduct (``preproc_scripts/spectraverse/subformula_explained.py``).

(a) Mean % intensity explained and mean % peaks explained per source, over all spectra. MS-DIAL
    is split by instrument. Reads ``data/proc/spectraverse/subformula_explained.csv`` and
    ``spec_source.csv`` (``preproc_scripts/spectraverse/source_filter_split.py``).
(b), (c) % intensity and % peaks explained minus a chance control, against fragment tolerance in
    ppm (no absolute floor). The control moves every peak by +7 Da and -7 Da, which keeps a
    realistic mass defect. MS-DIAL is split at 50% intensity explained (config tolerance).
    Computed on sampled spectra and cached in ``--sweep-csv``.

See docs/findings/masster/spectraverse_source_replicate_audit_2026_10_03.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fragnnet.utils.clean_spectra_utils import _subformula_masses  # noqa: E402

PPM = [5, 10, 20, 50, 100, 200, 500]
SAMPLE = {"MS-DIAL": 4000, "MSnLib": 1500, "GNPS": 2000}
# Reference categorical palette (light), slots in fixed order; ink and grid are neutral.
SERIES = {
    "MSnLib": "#2a78d6",
    "MS-DIAL, < 50% intensity explained": "#eb6834",
    "GNPS": "#1baf7a",
    "MS-DIAL, ≥ 50% intensity explained": "#eda100",
}
BARS = {"intensity": ("% intensity explained", "#2a78d6"), "peaks": ("% peaks explained", "#eb6834")}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _nearest_ppm(masses: np.ndarray, mz: np.ndarray) -> np.ndarray:
    """Signed ppm error from each m/z to the nearest mass in sorted ``masses``."""
    i = np.clip(np.searchsorted(masses, mz), 1, len(masses) - 1)
    lo, hi = masses[i - 1], masses[i]
    near = np.where(np.abs(lo - mz) < np.abs(hi - mz), lo, hi)
    return (mz - near) / near * 1e6


def sweep(spec: pd.DataFrame, formula: pd.Series, seed: int) -> pd.DataFrame:
    """Per-spectrum % intensity / % peaks explained, and the +/-7 Da control, at each ppm."""
    rows = []
    for src, n in SAMPLE.items():
        part = spec[spec["source"] == src]
        for row in part.sample(min(n, len(part)), random_state=seed).itertuples():
            try:
                masses = _subformula_masses(formula[row.mol_id], row.prec_type)
            except ValueError:
                continue
            if not len(masses):
                continue
            mz = np.array([p[0] for p in row.peaks])
            w = np.array([p[1] for p in row.peaks])
            w = w / w.sum()
            err = np.abs(_nearest_ppm(masses, mz))
            shifted = [np.abs(_nearest_ppm(masses, mz + d)) for d in (7.0, -7.0)]
            out = {"source": src, "frac_intensity_cfg": row.frac_intensity}
            for t in PPM:
                out[f"int{t}"] = w[err <= t].sum()
                out[f"pk{t}"] = (err <= t).mean()
                out[f"int_ctrl{t}"] = np.mean([w[e <= t].sum() for e in shifted])
                out[f"pk_ctrl{t}"] = np.mean([(e <= t).mean() for e in shifted])
            rows.append(out)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec-fp", default="data/proc/spectraverse/spec_df.pkl")
    parser.add_argument("--mol-fp", default="data/proc/spectraverse/mol_df.pkl")
    parser.add_argument("--explained-fp", default="data/proc/spectraverse/subformula_explained.csv")
    parser.add_argument("--source-fp", default="data/proc/spectraverse/spec_source.csv")
    parser.add_argument(
        "--sweep-csv",
        default="output/analysis/spectraverse_source_audit_2026_10_03/ppm_sweep_explained_rows.csv",
        help="per-spectrum sweep rows; reused when present",
    )
    parser.add_argument("--out", default="docs/figures/spectraverse_source_quality_2026_10_04.png")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    # 1. Load spectra with source and explained fractions.
    spec = pd.read_pickle(args.spec_fp)
    spec["spec_id"] = spec["spec_id"].astype(np.int64)
    source = pd.read_csv(args.source_fp).set_index("spec_id")["source"]
    explained = pd.read_csv(args.explained_fp).set_index("spec_id")
    spec["source"] = source.loc[spec["spec_id"]].to_numpy()
    spec = spec.join(explained[["frac_intensity", "frac_peaks"]], on="spec_id")
    spec = spec.dropna(subset=["frac_intensity"])

    # 2. Panel (a): mean explained fractions by source, MS-DIAL split by instrument.
    instrument = spec["inst_type"].map({"QTOF": "QTOF", "ORB": "Orbitrap"}).fillna("unspecified")
    label = spec["source"].where(spec["source"] != "MS-DIAL", "MS-DIAL " + instrument)
    major = ["GNPS", "MSnLib", "MoNA", "RIKEN", "MassBank",
             "MS-DIAL QTOF", "MS-DIAL Orbitrap", "MS-DIAL unspecified"]
    label = label.where(label.isin(major), "15 small sources")
    means = spec.groupby(label).agg(
        intensity=("frac_intensity", "mean"), peaks=("frac_peaks", "mean"), n=("frac_peaks", "size")
    ).sort_values("intensity", ascending=False)

    # 3. Panels (b), (c): ppm sweep, cached.
    sweep_fp = Path(args.sweep_csv)
    if sweep_fp.exists():
        rows = pd.read_csv(sweep_fp)
    else:
        formula = pd.read_pickle(args.mol_fp).set_index("mol_id")["formula"]
        rows = sweep(spec, formula, args.seed)
        sweep_fp.parent.mkdir(parents=True, exist_ok=True)
        rows.to_csv(sweep_fp, index=False)
    rows["group"] = rows["source"]
    msdial = rows["source"] == "MS-DIAL"
    rows.loc[msdial, "group"] = np.where(
        rows.loc[msdial, "frac_intensity_cfg"] < 0.5,
        "MS-DIAL, < 50% intensity explained", "MS-DIAL, ≥ 50% intensity explained",
    )
    excess = {
        key: pd.DataFrame({t: rows[f"{key}{t}"] - rows[f"{key}_ctrl{t}"] for t in PPM})
        .groupby(rows["group"]).mean()
        for key in ("int", "pk")
    }

    # 4. Draw.
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
                         "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK})
    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        1, 3, figsize=(15.5, 4.6), gridspec_kw={"width_ratios": [1.2, 1, 1]}
    )

    y = np.arange(len(means))
    h = 0.36
    for col, offset in (("intensity", -h / 2 - 0.02), ("peaks", h / 2 + 0.02)):
        name, color = BARS[col]
        ax_a.barh(y + offset, means[col], height=h, color=color, label=name)
        for yi, v in enumerate(means[col]):
            ax_a.text(v + 0.01, yi + offset, f"{v:.0%}", va="center", fontsize=7, color=INK_2)
    ax_a.set_yticks(y, [f"{s}  (n={n:,})" for s, n in zip(means.index, means["n"])])
    ax_a.invert_yaxis()
    ax_a.set_xlim(0, 1.12)
    ax_a.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax_a.set_xlabel("mean share explained by a subformula of the labelled structure")
    ax_a.set_title("(a) Explained signal by source, all spectra", loc="left", fontsize=10)
    ax_a.legend(frameon=False, fontsize=8, loc="lower right")

    for ax, key, what in ((ax_b, "int", "% intensity"), (ax_c, "pk", "% peaks")):
        ax.axhline(0, color=INK_2, lw=1)
        for name, color in SERIES.items():
            ax.plot(PPM, excess[key].loc[name, PPM].to_numpy(), color=color, lw=2, marker="o", ms=4)
        ax.set_xscale("log")
        ax.set_xticks(PPM, [str(t) for t in PPM])
        ax.minorticks_off()
        ax.set_xlim(4, 650)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_xlabel("fragment tolerance (ppm)")
        ax.set_ylabel(f"{what} explained minus chance")
    ax_b.set_title("(b) % intensity: wider tolerance does not rescue", loc="left", fontsize=10)
    ax_c.set_title("(c) % peaks: same sweep", loc="left", fontsize=10)
    ax_c.legend([plt.Line2D([], [], color=c, lw=2) for c in SERIES.values()], list(SERIES),
                frameon=False, fontsize=8, loc="upper right")

    for ax in (ax_a, ax_b, ax_c):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x" if ax is ax_a else "y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, facecolor="white")
    print(f"wrote {args.out}")
    print(means.round(3).to_string())
    for key in ("int", "pk"):
        print(key)
        print(excess[key].round(3).to_string())


if __name__ == "__main__":
    main()
