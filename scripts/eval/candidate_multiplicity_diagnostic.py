"""Measure how many candidate formulas the spectral loss sums over per observed peak.

The spectral loss (both ``gaussian`` / mog_ce_fn and ``peak_marginal`` / pm_ce_fn in
``fragnnet/loss/cross_entropy.py``) scores an observed peak by aggregating EVERY predicted candidate
within the m/z tolerance window:

    log P(peak_i) = logsumexp_{j : |mz_j - mz_i| <= tol} [ log w_j (+ log Gaussian(err_ij)) ]

If several distinct candidates fall inside one window, the loss only constrains their SUM --
it never has to decide which candidate is correct. This script measures that multiplicity.

A candidate here is one softmax entry over ``frag_formula_logprobs`` -- i.e. one (formula,
ion-convention) row, after node/H-transfer routes have already been collapsed by the
``scatter_logsumexp`` at fragnnet_model.py:1648. That collapse is why we count formula rows
and not DAG routes.

Two counts are reported per observed peak, and the difference between them is the point:

  - ``n_cand``       : unique candidate ROWS in the window. This is exactly what the loss
                       logsumexps over.
  - ``n_distinct_mz``: unique candidate m/z CLUSTERS in the window, collapsing candidates that
                       sit at the same m/z to machine precision.

``n_cand > n_distinct_mz`` is *structural aliasing*: candidates the loss can provably never
separate because they are the same m/z. For [M+H]+, CMF(F) and CRF(F-H) are the same ion and
land at a bit-identical m/z, so the CMF/CRF doubling in
``spec_mol_frag_dataset._apply_fragment_mz_change`` plus the H-transfer budget makes this
pervasive rather than incidental.

``n_distinct_mz > 1`` is *real chemical ambiguity*: chemically different formulas that happen
to be near-isobaric within the tolerance. Only this part can be fixed by a loss that forces a
choice among candidates; the aliased part cannot, at any tolerance.

Usage:
    python scripts/eval/candidate_multiplicity_diagnostic.py \\
        --frag_dp data/frag/nist20_ma_mi_d3_h4_isoFalse_smpFalse/dags \\
        --inst_type FT --frag_mode HCD --prec_type '[M+H]+' --max_mols 1500
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

from oracle_dataset_utils import resolve_dsets_for_spec_fp
import torch as th

from fragnnet.utils import dag_format_utils
from fragnnet.utils.formula_utils import PREC_TYPE_TO_CMF_MASS_DIFF, PREC_TYPE_TO_MASS_DIFF

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dag_oracle_scores import _detect_is_compressed  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TOLERANCE = 1.0e-5  # relative ratio (10 ppm), matches default loss_tolerance_rel
TOLERANCE_MIN_MZ = 200.0  # matches config/template.yml loss_tolerance_min_mz
MZ_CLUSTER_ATOL = 1.0e-6  # Da; below any real isobar spacing, above float noise


def build_candidates(dag_d: dict, prec_type: str) -> tuple[np.ndarray, np.ndarray]:
    """Build the model's candidate ion m/z set for one molecule.

    Mirrors ``spec_mol_frag_dataset._apply_fragment_mz_change``: CRF rows for every formula,
    plus CMF rows for every non-NULL formula, each a separate softmax entry.

    Args:
        dag_d: DAG dict from ``frag_utils.load_frag_d``.
        prec_type: Precursor adduct, e.g. "[M+H]+".

    Returns:
        Tuple of (cand_mzs, cand_row_ids), both 1-D and index-aligned, covering only real
        (non-padding) isotope peaks.
    """
    theo = dag_d["formula_peak_mzs"]
    probs = dag_d["formula_peak_probs"]
    if isinstance(theo, th.Tensor):
        theo = theo.numpy()
    if isinstance(probs, th.Tensor):
        probs = probs.numpy()

    crf_diff = PREC_TYPE_TO_MASS_DIFF[prec_type]
    blocks = [(theo + crf_diff, probs, 0)]
    if prec_type in PREC_TYPE_TO_CMF_MASS_DIFF and prec_type not in ("[M]+", "[M-]"):
        cmf_diff = PREC_TYPE_TO_CMF_MASS_DIFF[prec_type]
        # row 0 is the NULL formula and gets no CMF row, matching the dataset's neutral[1:].
        blocks.append((theo[1:] + cmf_diff, probs[1:], theo.shape[0]))

    mzs, rows = [], []
    for block_mzs, block_probs, row_offset in blocks:
        real = block_probs > 0.0
        r_idx, _ = np.nonzero(real)
        mzs.append(block_mzs[real])
        rows.append(r_idx + row_offset)
    return np.concatenate(mzs), np.concatenate(rows)


def count_multiplicity(
    true_mzs: np.ndarray, cand_mzs: np.ndarray, cand_rows: np.ndarray, tolerance: float,
    tolerance_min_mz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Count candidate rows and distinct candidate m/z clusters within tolerance of each peak.

    Args:
        true_mzs: Observed peak m/z of shape (T,).
        cand_mzs: Candidate ion m/z of shape (C,).
        cand_rows: Candidate softmax row id of shape (C,).
        tolerance: Relative m/z tolerance ratio.
        tolerance_min_mz: Floor applied to m/z before scaling the tolerance.

    Returns:
        Tuple of (n_cand, n_distinct_mz), each of shape (T,).
    """
    tol = np.maximum(true_mzs, tolerance_min_mz) * tolerance
    # (T, C) match mask, same convention as spec_utils.calculate_match_mzs.
    diff = np.abs(true_mzs[:, None] - cand_mzs[None, :])
    match = diff <= tol[:, None]

    n_cand = np.zeros(len(true_mzs), dtype=np.int64)
    n_distinct = np.zeros(len(true_mzs), dtype=np.int64)
    for i in range(len(true_mzs)):
        sel = match[i]
        if not sel.any():
            continue
        n_cand[i] = len(np.unique(cand_rows[sel]))
        # Cluster matched candidate m/z: values within MZ_CLUSTER_ATOL are unseparable.
        s = np.sort(cand_mzs[sel])
        n_distinct[i] = 1 + int((np.diff(s) > MZ_CLUSTER_ATOL).sum())
    return n_cand, n_distinct


def process_mol(mol_id: int, rows: list[dict], frag_dp: str, is_compressed: bool) -> list[dict]:
    """Count multiplicity for every observed peak of every spectrum of one molecule."""
    dag_d = dag_format_utils.load_frag_d(mol_id, frag_dp, is_compressed=is_compressed)
    if dag_d is None:
        return []

    out = []
    cand_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for row in rows:
        peaks = row["peaks"]
        if not peaks:
            continue
        prec_type = row["prec_type"]
        if prec_type not in cand_cache:
            cand_cache[prec_type] = build_candidates(dag_d, prec_type)
        cand_mzs, cand_rows = cand_cache[prec_type]

        true_mzs_raw, true_ints_raw = zip(*peaks)
        true_mzs = np.asarray(true_mzs_raw, dtype=np.float64)
        true_ints = np.asarray(true_ints_raw, dtype=np.float64)
        if true_ints.sum() <= 0:
            continue
        true_ints = true_ints / true_ints.sum()

        n_cand, n_distinct = count_multiplicity(
            true_mzs, cand_mzs, cand_rows, TOLERANCE, TOLERANCE_MIN_MZ
        )
        matched = n_cand > 0
        if not matched.any():
            continue
        out.append(
            pd.DataFrame(
                {
                    "spec_id": row["spec_id"],
                    "nce": row["nce"],
                    "mz": true_mzs[matched],
                    "intensity": true_ints[matched],
                    "n_cand": n_cand[matched],
                    "n_distinct_mz": n_distinct[matched],
                }
            )
        )
    return out


def _wmean(g: pd.DataFrame, col: str) -> float:
    """Intensity-weighted mean of a column, matching how the loss weights observed peaks."""
    return float(np.average(g[col], weights=g["intensity"]))


def summarize(df: pd.DataFrame) -> None:
    """Print overall, CE-stratified and m/z-stratified multiplicity summaries."""
    print(f"\n{'=' * 96}")
    print("Candidate multiplicity per observed peak (matched peaks only)")
    print(f"{'=' * 96}")
    print(f"spectra: {df['spec_id'].nunique()}   matched peaks: {len(df)}")

    print(
        f"\nOVERALL   n_cand: mean={df['n_cand'].mean():.2f} median={df['n_cand'].median():.0f} "
        f"int-wtd={_wmean(df, 'n_cand'):.2f}   "
        f">1: {100 * (df['n_cand'] > 1).mean():.1f}%  >3: {100 * (df['n_cand'] > 3).mean():.1f}%"
    )
    print(
        f"          n_distinct_mz: mean={df['n_distinct_mz'].mean():.2f} "
        f"median={df['n_distinct_mz'].median():.0f} int-wtd={_wmean(df, 'n_distinct_mz'):.2f}   "
        f">1: {100 * (df['n_distinct_mz'] > 1).mean():.1f}%  "
        f">3: {100 * (df['n_distinct_mz'] > 3).mean():.1f}%"
    )
    alias = df["n_cand"] - df["n_distinct_mz"]
    print(
        f"\nStructural aliasing (n_cand - n_distinct_mz): mean={alias.mean():.2f}   "
        f"peaks with >=1 aliased candidate: {100 * (alias > 0).mean():.1f}%"
    )

    ce_bins = [0, 20, 40, 60, 80, 100, 120, 500]
    df = df.copy()
    df["ce_bin"] = pd.cut(df["nce"], bins=ce_bins)
    print("\nBy NCE bin:")
    print(f"  {'nce':>12s} {'peaks':>8s} {'n_cand':>8s} {'wtd':>7s} {'distinct':>9s} {'wtd':>7s} {'>1 mz':>7s}")
    for ce, g in df.groupby("ce_bin", observed=True):
        if len(g) < 200:
            continue
        print(
            f"  {str(ce):>12s} {len(g):>8d} {g['n_cand'].mean():>8.2f} "
            f"{_wmean(g, 'n_cand'):>7.2f} {g['n_distinct_mz'].mean():>9.2f} "
            f"{_wmean(g, 'n_distinct_mz'):>7.2f} "
            f"{100 * (g['n_distinct_mz'] > 1).mean():>6.1f}%"
        )

    mz_bins = [0, 100, 150, 200, 300, 400, 600, 2000]
    df["mz_bin"] = pd.cut(df["mz"], bins=mz_bins)
    print("\nBy observed m/z bin:")
    print(f"  {'mz':>12s} {'peaks':>8s} {'n_cand':>8s} {'distinct':>9s} {'>1 mz':>7s}")
    for mzb, g in df.groupby("mz_bin", observed=True):
        if len(g) < 200:
            continue
        print(
            f"  {str(mzb):>12s} {len(g):>8d} {g['n_cand'].mean():>8.2f} "
            f"{g['n_distinct_mz'].mean():>9.2f} "
            f"{100 * (g['n_distinct_mz'] > 1).mean():>6.1f}%"
        )

    print("\nmean n_distinct_mz by (m/z bin x NCE bin)  [the decisive table]:")
    piv = df.pivot_table(
        index="mz_bin", columns="ce_bin", values="n_distinct_mz", aggfunc="mean", observed=True
    )
    print(piv.round(2).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec_fp", default="data/proc/nist20/spec_df.pkl")
    parser.add_argument(
        "--dsets", "--dset", nargs="+", default=None, help="Filter spec_df to these dataset labels."
    )
    parser.add_argument("--split_dp", default="data/split/nist20_ma_mi_inchikey")
    parser.add_argument("--split", default="test")
    parser.add_argument("--frag_dp", default="data/frag/nist20_ma_mi_d3_h4_isoFalse_smpFalse/dags")
    parser.add_argument("--inst_type", default="ORB")
    parser.add_argument("--frag_mode", default="HCD")
    parser.add_argument("--prec_type", default="[M+H]+")
    parser.add_argument("--max_mols", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_csv", default=None)
    args = parser.parse_args()
    args.dsets = resolve_dsets_for_spec_fp(args.spec_fp, args.dsets)

    log.info("Loading %s", args.spec_fp)
    spec_df = pd.read_pickle(args.spec_fp)
    if args.dsets:
        spec_df = spec_df[spec_df["dset"].isin(args.dsets)]
    spec_df = spec_df[spec_df["peaks"].apply(len) > 0]
    spec_df = spec_df[spec_df["prec_type"] == args.prec_type]
    spec_df = spec_df[spec_df["inst_type"] == args.inst_type]
    spec_df = spec_df[spec_df["frag_mode"] == args.frag_mode]
    spec_df = spec_df[spec_df["nce"].notna()]
    if args.split != "all":
        split_fp = os.path.join(args.split_dp, f"{args.split}_ids.csv")
        split_ids = set(pd.read_csv(split_fp)["spec_id"])
        spec_df = spec_df[spec_df["spec_id"].isin(split_ids)]
    if len(spec_df) == 0:
        raise ValueError("no spectra left after filtering")

    mol_ids = spec_df["mol_id"].unique()
    if len(mol_ids) > args.max_mols:
        rng = np.random.default_rng(args.seed)
        mol_ids = rng.choice(mol_ids, size=args.max_mols, replace=False)
        spec_df = spec_df[spec_df["mol_id"].isin(set(mol_ids))]
    log.info("scoring %d spectra over %d mols", len(spec_df), len(mol_ids))

    is_compressed = _detect_is_compressed(args.frag_dp, int(spec_df["mol_id"].iloc[0]))
    cols = ["spec_id", "mol_id", "peaks", "prec_type", "nce"]
    frames = []
    for i, (mol_id, g) in enumerate(spec_df[cols].groupby("mol_id", observed=True)):
        frames.extend(process_mol(int(mol_id), g.to_dict("records"), args.frag_dp, is_compressed))
        if (i + 1) % 200 == 0:
            log.info("  %d/%d mols", i + 1, len(mol_ids))

    df = pd.concat(frames, ignore_index=True)
    summarize(df)
    if args.out_csv:
        df.to_csv(args.out_csv, index=False)
        log.info("wrote %s", args.out_csv)


if __name__ == "__main__":
    main()
