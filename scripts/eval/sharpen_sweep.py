"""Post-hoc temperature (sharpening) sweep on saved predictions -- no retraining.

The intensity diagnostic showed the model's spectra are systematically too flat (right rank,
under-confident heights). "Right rank, too flat" is a temperature problem, so this sweeps a
sharpening exponent gamma on the ALREADY-saved predicted intensities:

    p_sharp = p ** gamma   (then L2-normalized inside cos_hun; gamma>1 sharpens, <1 flattens)

and rescores cos_hun_inst_tol against the true spectrum with the eval's own cos_hun_helper and
per-instrument tolerance. gamma=1 reproduces the stored metric (sanity check). Reports:
  - mean cos_hun at each global gamma (best global gamma = the free inference-time win),
  - the per-spectrum-optimal-gamma ceiling (upper bound of any single-temperature fix; the gap
    between it and the best global gamma tells you how much a LEARNED, per-spectrum temperature
    could add over one constant).

Usage:
    python scripts/eval/sharpen_sweep.py \\
        --pred_hdf5 output/inference_eval/4ews8j6f/predictions_test.hdf5 --split test
"""

from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import pandas as pd

from oracle_dataset_utils import resolve_dsets_for_spec_fp
import torch as th

from fragnnet.utils import parallel_utils
from fragnnet.utils.formula_utils import PREC_TYPE_TO_MASS_DIFF
from fragnnet.utils.spec_utils import calculate_match_mzs, cos_hun_helper

# Reuse the diagnostic's loaders + per-instrument tolerance (FT 10ppm/200Da, QTOF 20ppm/750Da).
from intensity_error_diagnostic import _tolerance_for_inst_type, load_pred_spectra

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DEFAULT_GAMMAS = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0]
_ZERO = th.zeros(1)


def _cos_hun_at_gammas(rec: dict, gammas: list[float]) -> dict | None:
    """cos_hun for one spectrum at every gamma. Match mask is computed once (m/z is fixed)."""
    true_mzs = th.as_tensor(rec["true_mzs"], dtype=th.float32)
    true_ints = th.as_tensor(rec["true_ints"], dtype=th.float32)
    pred_mzs = th.as_tensor(rec["pred_mzs"], dtype=th.float32)
    pred_ints = th.as_tensor(rec["pred_ints"], dtype=th.float32)
    if true_mzs.numel() == 0 or pred_mzs.numel() == 0:
        return None

    tol, tol_min = _tolerance_for_inst_type(rec["inst_type"])
    match_mask = th.as_tensor(
        calculate_match_mzs(true_mzs, pred_mzs, tolerance=tol, relative=True, tolerance_min_mz=tol_min)
    )
    true_mm = match_mask.any(dim=1)
    pred_mm = match_mask.any(dim=0)
    if not bool(true_mm.any()) or not bool(pred_mm.any()):
        return {"spec_id": rec["spec_id"], "inst_type": rec["inst_type"],
                "frag_mode": rec["frag_mode"], "prec_type": rec["prec_type"],
                **{f"g{g}": 0.0 for g in gammas}}

    no_prec = th.zeros_like(true_mm, dtype=th.float32)
    no_prec_p = th.zeros_like(pred_mm, dtype=th.float32)
    # normalize to max 1 so p**gamma can't overflow for large gamma
    base = pred_ints / pred_ints.max().clamp(min=1e-12)

    out = {"spec_id": rec["spec_id"], "inst_type": rec["inst_type"],
           "frag_mode": rec["frag_mode"], "prec_type": rec["prec_type"]}
    for g in gammas:
        p_sharp = base**g
        cos = cos_hun_helper(
            true_ints, p_sharp, match_mask, true_mm, pred_mm, False, no_prec, no_prec_p
        )
        out[f"g{g}"] = float(cos)
    return out


def _chunk_worker(recs: list[dict], gammas: list[float]) -> list[dict]:
    return [r for rec in recs if (r := _cos_hun_at_gammas(rec, gammas)) is not None]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pred_hdf5", required=True)
    parser.add_argument("--spec_fp", default="data/proc/nist20/spec_df.pkl")
    parser.add_argument(
        "--dsets", "--dset", nargs="+", default=None, help="Filter spec_df to these dataset labels."
    )
    parser.add_argument("--split_dp", default="data/split/nist20_ma_mi_inchikey")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--gammas", type=float, nargs="+", default=DEFAULT_GAMMAS)
    parser.add_argument(
        "--min_int_frac", type=float, default=0.0,
        help="Drop peaks below this fraction of base peak in BOTH pred and true before scoring.",
    )
    parser.add_argument("--limit_mols", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_jobs", type=int, default=parallel_utils.JOBLIB_N_JOBS)
    args = parser.parse_args()
    args.dsets = resolve_dsets_for_spec_fp(args.spec_fp, args.dsets)

    log.info("Loading predicted spectra from %s", args.pred_hdf5)
    pred_by_sid = load_pred_spectra(args.pred_hdf5)

    log.info("Loading %s", args.spec_fp)
    spec_df = pd.read_pickle(args.spec_fp)
    if args.dsets:
        spec_df = spec_df[spec_df["dset"].isin(args.dsets)]
    spec_df = spec_df[spec_df["prec_type"].isin(PREC_TYPE_TO_MASS_DIFF.keys())]
    spec_df = spec_df[spec_df["peaks"].apply(len) > 0]
    if args.split != "all":
        split_ids = set(pd.read_csv(os.path.join(args.split_dp, f"{args.split}_ids.csv"))["spec_id"])
        spec_df = spec_df[spec_df["spec_id"].isin(split_ids)]
    spec_df = spec_df[spec_df["spec_id"].isin(pred_by_sid.keys())]
    if args.limit_mols is not None:
        rng = np.random.default_rng(args.seed)
        keep = rng.choice(spec_df["mol_id"].unique(),
                          size=min(args.limit_mols, spec_df["mol_id"].nunique()), replace=False)
        spec_df = spec_df[spec_df["mol_id"].isin(keep)]

    def _threshold(mzs, ints):
        if args.min_int_frac <= 0.0 or ints.size == 0:
            return mzs, ints
        keep = ints >= args.min_int_frac * ints.max()
        return mzs[keep], ints[keep]

    recs = []
    for row in spec_df.itertuples(index=False):
        pm, pi = _threshold(*pred_by_sid[int(row.spec_id)])
        tm, ti = _threshold(*(np.array(a, dtype=np.float32) for a in zip(*row.peaks)))
        recs.append({"spec_id": int(row.spec_id), "inst_type": row.inst_type,
                     "frag_mode": row.frag_mode, "prec_type": row.prec_type,
                     "pred_mzs": pm, "pred_ints": pi, "true_mzs": tm, "true_ints": ti})

    log.info("Scoring %d spectra at %d gammas", len(recs), len(args.gammas))
    n_chunks = max(1, args.n_jobs * 4)
    chunks = [(recs[i::n_chunks], args.gammas) for i in range(n_chunks)]
    per_chunk = parallel_utils.par_apply(chunks, _chunk_worker, need_unpack=True, n_jobs=args.n_jobs)
    df = pd.DataFrame([r for c in per_chunk for r in c])

    gcols = [f"g{g}" for g in args.gammas]
    means = df[gcols].mean()
    base = means["g1.0"]
    best_g = args.gammas[int(np.argmax(means.values))]
    best_mean = means.max()
    # per-spectrum-optimal ceiling: each spectrum takes its own best gamma
    per_spec_best = df[gcols].max(axis=1).mean()

    thr = f" (>= {args.min_int_frac:.0%} of base peak)" if args.min_int_frac > 0 else ""
    print(f"\n{'=' * 78}")
    print(f"Post-hoc sharpening sweep -- cos_hun_inst_tol{thr}")
    print(f"{'=' * 78}")
    print(f"spectra: {len(df)}\n")
    for g in args.gammas:
        bar = "#" * int((means[f'g{g}'] - 0.6) * 400) if means[f"g{g}"] > 0.6 else ""
        star = "  <- baseline" if g == 1.0 else ("  <- BEST global" if g == best_g else "")
        print(f"  gamma={g:<4}  cos_hun={means[f'g{g}']:.4f}{star}  {bar}")
    print(f"\nbaseline (gamma=1):        {base:.4f}")
    print(f"best global gamma={best_g}:      {best_mean:.4f}   (+{best_mean - base:.4f})")
    print(f"per-spectrum-optimal gamma: {per_spec_best:.4f}   (+{per_spec_best - base:.4f}, ceiling)")

    print("\nBest global gamma per stratum:")
    for key in ["inst_type", "frag_mode", "prec_type"]:
        for val, g in df.groupby(key, observed=True):
            if len(g) < 50:
                continue
            m = g[gcols].mean()
            bg = args.gammas[int(np.argmax(m.values))]
            print(f"  {key}={str(val):8s} n={len(g):>6d}  base={m['g1.0']:.4f} "
                  f"best gamma={bg} -> {m.max():.4f} (+{m.max() - m['g1.0']:.4f})")


if __name__ == "__main__":
    main()
