"""Measure how discriminating precursor-bounded subformula decomposition is, by precursor size.

A subformula explanation is only evidence if most m/z have none. This walks the real NIST20
formula distribution and reports, per precursor-mass bin: the share of random m/z that some
subformula ion explains within the instrument window, how many candidates such an m/z draws,
and what one decomposition costs.

Run from repo root:
    conda run -n FRAGNNET-GPU python scripts/debug/fragnnet/debug_subformula_grid_density.py
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from fragnnet.utils.clean_spectra_utils import subformula_candidates
from fragnnet.utils.formula_utils import formula_to_mass

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

MASS_BIN_EDGES = [0, 200, 300, 400, 500, 700, 1000, np.inf]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proc-dp", type=Path, default=Path("data/proc/nist20"))
    parser.add_argument("--n-mols", type=int, default=2000, help="Molecules to sample")
    parser.add_argument("--n-queries", type=int, default=200, help="Random m/z per molecule")
    parser.add_argument("--tol", type=float, default=0.004, help="Da decomposition window (FT)")
    parser.add_argument("--prec-type", default="[M+H]+")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    # 1. Sample the real formula distribution.
    args = parse_args()
    mol_df = pd.read_pickle(args.proc_dp / "mol_df.pkl")
    formulas = mol_df["formula"].dropna().astype(str)
    formulas = formulas.sample(
        min(args.n_mols, len(formulas)), random_state=args.seed
    ).tolist()
    log.info(f"Sampled {len(formulas):,} formulas from {args.proc_dp / 'mol_df.pkl'}")

    # 2. For each molecule, probe random m/z below the precursor and time the decomposition.
    rng = np.random.default_rng(args.seed)
    rows = []
    skipped = 0
    for formula in formulas:
        try:
            prec_mass = float(formula_to_mass(formula))
        except Exception:
            skipped += 1
            continue
        if prec_mass <= 120:
            skipped += 1
            continue
        queries = rng.uniform(100.0, prec_mass, args.n_queries)
        explained = 0
        n_candidates = 0
        start = time.perf_counter()
        try:
            for query in queries:
                candidates = subformula_candidates(
                    float(query), formula, args.prec_type, args.tol
                )
                n_candidates += len(candidates)
                explained += bool(candidates)
        except ValueError:
            skipped += 1
            continue
        elapsed_ms = (time.perf_counter() - start) / len(queries) * 1e3
        rows.append(
            {
                "formula": formula,
                "prec_mass": prec_mass,
                "coverage": explained / len(queries),
                "candidates_per_query": n_candidates / len(queries),
                "ms_per_query": elapsed_ms,
            }
        )

    df = pd.DataFrame(rows)
    log.info(f"Measured {len(df):,} molecules ({skipped} skipped)")

    # 3. Report by precursor-mass bin; coverage is the discriminating-power number.
    df["mass_bin"] = pd.cut(df["prec_mass"], MASS_BIN_EDGES)
    summary = df.groupby("mass_bin", observed=True).agg(
        n_mols=("formula", "size"),
        coverage_mean=("coverage", "mean"),
        coverage_p90=("coverage", lambda s: float(np.percentile(s, 90))),
        candidates_mean=("candidates_per_query", "mean"),
        ms_per_query_mean=("ms_per_query", "mean"),
        ms_per_query_p99=("ms_per_query", lambda s: float(np.percentile(s, 99))),
    )
    log.info("")
    log.info("Coverage = share of random m/z that some subformula ion explains within the window")
    log.info("")
    log.info(summary.to_string(float_format=lambda v: f"{v:8.4f}"))

    for threshold in (0.25, 0.50):
        past = df[df["coverage"] > threshold]
        share = len(past) / max(len(df), 1)
        if past.empty:
            log.info(f"\nNo molecule exceeds coverage {threshold:.2f}")
            continue
        log.info(
            f"\nCoverage > {threshold:.2f}: {len(past):,} molecules ({share:.3%}), "
            f"lightest at {past['prec_mass'].min():.1f} Da, "
            f"median precursor mass {past['prec_mass'].median():.1f} Da"
        )

    log.info("")
    log.info(
        f"Overall: coverage mean {df['coverage'].mean():.4f}, "
        f"p50 {df['coverage'].median():.4f}, p99 {np.percentile(df['coverage'], 99):.4f}; "
        f"cost mean {df['ms_per_query'].mean():.3f} ms/query, "
        f"p99 {np.percentile(df['ms_per_query'], 99):.3f} ms/query"
    )


if __name__ == "__main__":
    main()
