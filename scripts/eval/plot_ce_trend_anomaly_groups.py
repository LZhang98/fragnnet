"""Plot CE-series spectra per group for trend-anomaly dips (isotonic-residual flags).

Each PDF page is one ``(mol_id, adduct, instrument, frag_mode)`` group that contains at least one
trend dip. Spectra are drawn sorted by collision energy with DAG-matched peaks in blue and
unmatched peaks in red; the dip spectra (isotonic residual >= ``--resid_thresh``) are highlighted.
This is the spectrum-level view of ``oracle_ce_trend_anomaly.py``'s flags, mirroring
``plot_ce_sibling_spectrum_groups.py`` but sourcing the flags from the isotonic-residual scores.

Usage:
    python scripts/eval/plot_ce_trend_anomaly_groups.py \\
        --out_dp data/eval/oracle_cos_nist20_default \\
        --resid_thresh 0.4
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from oracle_dataset_utils import resolve_dsets_for_spec_fp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_ce_sibling_spectrum_groups import (  # noqa: E402  (needs sys.path tweak first)
    GROUP_COLS,
    add_ce_columns,
    add_peak_label_columns,
    add_peak_tracks,
    choose_group_rows,
    compute_peak_match_masks,
    plot_group_page,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def get_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--scores_csv",
        default="data/eval/oracle_cos_nist20_default/oracle_ce_trend_anomaly_scores.csv",
        help="Per-spectrum isotonic-residual scores from oracle_ce_trend_anomaly.py.",
    )
    parser.add_argument(
        "--oracle_csv",
        default="data/eval/oracle_cos_nist20_default/oracle_cos_default.csv",
    )
    parser.add_argument("--spec_fp", default="data/proc/nist20/spec_df.pkl")
    parser.add_argument("--mol_fp", default="data/proc/nist20/mol_df.pkl")
    parser.add_argument(
        "--frag_dp",
        default="data/frag/nist20_ma_mi_d3_h4_isoFalse_smpFalse/dags",
        help="DAG directory used to color matched/unmatched peaks.",
    )
    parser.add_argument(
        "--dsets", "--dset", nargs="+", default=None, help="Filter spec_df to these dataset labels."
    )
    parser.add_argument("--out_dp", required=True)
    parser.add_argument("--metric", default="cos_hun_inst_tol_opt")
    parser.add_argument(
        "--resid_thresh",
        type=float,
        default=0.4,
        help="Isotonic-residual cutoff defining a flagged dip (fit - metric >= threshold).",
    )
    parser.add_argument(
        "--max_groups",
        type=int,
        default=None,
        help="Limit number of groups plotted after sorting by deepest dip.",
    )
    parser.add_argument(
        "--max_spectra_per_group",
        type=int,
        default=20,
        help="For large CE groups, keep flagged spectra plus representative siblings up to this cap.",
    )
    parser.add_argument("--mz_max", type=float, default=None)
    parser.add_argument("--min_int_frac", type=float, default=0.005)
    parser.add_argument(
        "--peak_color_mode",
        choices=["match", "metric"],
        default="match",
        help="match: color individual peaks by DAG match status; metric: color whole spectrum.",
    )
    parser.add_argument("--max_h_transfer", type=int, default=None)
    parser.add_argument(
        "--no_cmf",
        action="store_true",
        help="For match coloring, use direct-charge-retention ion masses only.",
    )
    parser.add_argument("--max_peaks_per_spectrum", type=int, default=150)
    args = parser.parse_args()
    args.dsets = resolve_dsets_for_spec_fp(args.spec_fp, args.dsets)
    return args


def main() -> None:
    """Create a multipage PDF of CE groups containing trend-anomaly dips."""
    args = get_args()
    out_dp = Path(args.out_dp)
    out_dp.mkdir(parents=True, exist_ok=True)

    scores = pd.read_csv(args.scores_csv)
    dips = scores[scores["iso_resid"] >= args.resid_thresh].copy()
    if dips.empty:
        raise ValueError(f"No dips at resid_thresh={args.resid_thresh}; nothing to plot.")
    dip_ids = set(dips["spec_id"])
    log.info(
        "Dips at resid>=%.2f: %d spectra across %d groups",
        args.resid_thresh, len(dips), dips[GROUP_COLS].drop_duplicates().shape[0],
    )

    oracle = pd.read_csv(args.oracle_csv, usecols=lambda c: c in {"spec_id", args.metric})
    spec = pd.read_pickle(args.spec_fp)[
        ["spec_id", "mol_id", "prec_type", "inst_type", "frag_mode", "dset", "ace", "nce",
         "prec_mz", "peaks"]
    ]
    if args.dsets:
        spec = spec[spec["dset"].isin(args.dsets)]
    mol_by_id = pd.read_pickle(args.mol_fp).set_index("mol_id")

    # Pull every spectrum of each dip-containing group (the CE siblings are the context).
    groups = dips[GROUP_COLS].drop_duplicates()
    plot_df = spec.merge(groups, on=GROUP_COLS, how="inner").merge(oracle, on="spec_id", how="left")
    plot_df = plot_df.merge(
        dips[["spec_id", "iso_resid", "best_diff_ce"]], on="spec_id", how="left"
    )
    plot_df = add_ce_columns(plot_df)
    plot_df = add_peak_label_columns(plot_df, mol_by_id)
    plot_df["ce_sort"] = plot_df["ce_value"].fillna(np.inf)
    plot_df["is_flagged"] = plot_df["spec_id"].isin(dip_ids)

    # Deepest dip first, so the most suspicious groups lead the PDF.
    group_order = (
        dips.groupby(GROUP_COLS, observed=True)["iso_resid"]
        .max()
        .reset_index(name="max_resid")
        .sort_values("max_resid", ascending=False)
    )
    if args.max_groups is not None:
        group_order = group_order.head(args.max_groups)

    page_groups = []
    selected_frames = []
    for row in group_order.itertuples(index=False):
        key = {c: getattr(row, c) for c in GROUP_COLS}
        mask = np.ones(len(plot_df), dtype=bool)
        for col, val in key.items():
            mask &= plot_df[col].to_numpy() == val
        group = plot_df[mask].sort_values("ce_sort").copy()
        group = choose_group_rows(group, args.metric, args.max_spectra_per_group)
        group, _track_df = add_peak_tracks(group)
        page_groups.append((key, group))
        selected_frames.append(group)

    if args.peak_color_mode == "match" and selected_frames:
        selected_df = pd.concat(selected_frames, ignore_index=True)
        log.info("Computing DAG match masks for %d plotted spectra", len(selected_df))
        match_masks = compute_peak_match_masks(
            selected_df, args.frag_dp, args.max_h_transfer, include_cmf=not args.no_cmf
        )
        page_groups = [
            (key, group.assign(match_mask=group["spec_id"].astype(int).map(match_masks)))
            for key, group in page_groups
        ]

    pdf_fp = out_dp / "oracle_ce_trend_anomaly_groups.pdf"
    index_rows = []
    with PdfPages(pdf_fp) as pdf:
        for page, (key, group) in enumerate(page_groups, start=1):
            mol_id = int(key["mol_id"])
            mol_row = mol_by_id.loc[mol_id] if mol_id in mol_by_id.index else None
            if isinstance(mol_row, pd.DataFrame):
                mol_row = mol_row.iloc[0]
            plot_group_page(
                pdf, group, args.metric, mol_row, args.mz_max, args.min_int_frac,
                args.max_peaks_per_spectrum, page,
            )
            flagged = group[group["is_flagged"]]
            index_rows.append(
                {
                    **key,
                    "page": page,
                    "dsets": ";".join(sorted(group["dset"].dropna().astype(str).unique())),
                    "n_plotted_spectra": len(group),
                    "flagged_spec_ids": ";".join(flagged["spec_id"].astype(str)),
                    "max_resid": flagged["iso_resid"].max(),
                    "min_metric": flagged[args.metric].min(),
                    "best_diff_ce": flagged["best_diff_ce"].max(),
                }
            )
            if page % 100 == 0:
                log.info("Plotted %d/%d groups", page, len(group_order))

    index_fp = out_dp / "oracle_ce_trend_anomaly_groups_index.csv"
    pd.DataFrame(index_rows).to_csv(index_fp, index=False)
    log.info("Wrote %s (%d pages)", pdf_fp, len(page_groups))
    log.info("Wrote %s", index_fp)


if __name__ == "__main__":
    main()
