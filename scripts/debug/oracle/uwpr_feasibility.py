"""Fill in the DAG feasibility verdict for NIST20 peaks matched to UWPR contaminants.

The input ``nist20_uwpr_hits.csv`` lists every NIST20 peak whose m/z falls within
tolerance of a known UWPR/MaConDa contaminant ion. Its ``feasibility`` column was never
computed (all rows are ``no_dag_file``/``no_cont_formula``). This script fills it in:
for each matched peak, it asks whether the molecule's fragmentation DAG can explain the
peak's m/z under the same adduct/ion-mass + H-transfer conventions used elsewhere in eval.

Verdict per matched peak:
  - ``dag_feasible``      : peak m/z matches a DAG fragment ion -> plausibly a real fragment
                            (the contaminant match is coincidental); do NOT flag.
  - ``contaminant_only``  : no DAG fragment explains the peak, but a contaminant does ->
                            high-confidence contamination in the reference spectrum.
  - ``no_dag_file``       : molecule has no DAG on disk -> cannot decide.
  - ``unsupported_prec``  : precursor adduct not in the ion-mass mapping -> cannot decide.
  - ``missing_metadata``  : precursor adduct or instrument type is missing -> cannot decide.

Usage:
    python scripts/debug/oracle/uwpr_feasibility.py \\
        --hits_csv data/uwpr/nist20_uwpr_hits.csv \\
        --frag_dp data/frag/nist20_ma_mi_d3_h4_isoFalse_smpFalse_mbfs/dags \\
        --spec_fp data/proc/nist20/spec_df.pkl \\
        --out_csv data/uwpr/nist20_uwpr_hits_feasibility.csv
"""

from __future__ import annotations

import argparse
import functools
import logging

import pandas as pd
import torch as th

# Reuse the exact ion-mass / H-transfer / tolerance conventions from the oracle scorer.
from dag_oracle_scores import (  # noqa: E402  (local script import)
    _allowed_formula_idxs,
    _dag_ion_mzs,
    _detect_is_compressed,
    _tolerance_for_inst_type,
)
from oracle_dataset_utils import resolve_dsets_for_spec_fp

from fragnnet.utils import dag_format_utils, parallel_utils
from fragnnet.utils.formula_utils import PREC_TYPE_TO_MASS_DIFF
from fragnnet.utils.spec_similarity import calculate_match_mzs

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def _feasibility_for_mol(
    mol_id: int,
    rows: pd.DataFrame,
    frag_dp: str,
    is_compressed: bool,
    max_h_transfer: int | None,
    include_cmf: bool,
) -> pd.Series:
    """Return the feasibility verdict for every matched peak of one molecule.

    Args:
        mol_id: Molecule identifier.
        rows: Matched-peak rows for this molecule; must have columns ``peak_mz``,
            ``prec_type``, ``inst_type`` and a stable index.
        frag_dp: Directory of per-molecule DAG pickles.
        is_compressed: Whether DAG pickles are bz2-compressed.
        max_h_transfer: H-transfer cap; None uses the DAG's full budget.
        include_cmf: Whether to union the CMF ion-mass convention with std.

    Returns:
        Series of verdict strings indexed like ``rows``.
    """
    verdict = pd.Series("no_dag_file", index=rows.index, dtype=object)

    try:
        dag_d = dag_format_utils.load_frag_d(mol_id, frag_dp, is_compressed=is_compressed)
    except FileNotFoundError:
        dag_d = None
    if dag_d is None:
        return verdict

    verdict.loc[:] = "missing_metadata"

    allowed_idx = _allowed_formula_idxs(dag_d["idx_by_h_delta"], max_h_transfer)
    idx_t = th.tensor(allowed_idx, dtype=th.long)
    theo_mzs = dag_d["formula_peak_mzs"].index_select(0, idx_t)
    prec_mask = (dag_d["formula_peak_probs"].index_select(0, idx_t) > 0.0).float()

    valid_rows = rows.dropna(subset=["prec_type", "inst_type"])
    for prec_type, pt_rows in valid_rows.groupby("prec_type"):
        if prec_type not in PREC_TYPE_TO_MASS_DIFF:
            verdict.loc[pt_rows.index] = "unsupported_prec"
            continue
        dag_ion_mzs = _dag_ion_mzs(theo_mzs, prec_mask, prec_type, include_cmf=include_cmf)
        # inst-type tolerance can differ within a prec_type group, so batch per inst_type.
        for inst_type, it_rows in pt_rows.groupby("inst_type"):
            tol, tol_min_mz = _tolerance_for_inst_type(inst_type)
            it_peak_mzs = th.tensor(it_rows["peak_mz"].to_numpy(), dtype=th.float32)
            match = calculate_match_mzs(
                it_peak_mzs, dag_ion_mzs, tolerance=tol, relative=True, tolerance_min_mz=tol_min_mz
            )
            feasible = match.any(dim=1).numpy()
            verdict.loc[it_rows.index] = [
                "dag_feasible" if f else "contaminant_only" for f in feasible
            ]
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser(description="Fill in UWPR contaminant DAG-feasibility verdict")
    parser.add_argument("--hits_csv", default="data/uwpr/nist20_uwpr_hits.csv")
    parser.add_argument(
        "--frag_dp", default="data/frag/nist20_ma_mi_d3_h4_isoFalse_smpFalse_mbfs/dags"
    )
    parser.add_argument("--spec_fp", default="data/proc/nist20/spec_df.pkl")
    parser.add_argument(
        "--dsets", "--dset", nargs="+", default=None, help="Filter spec_df to these dataset labels."
    )
    parser.add_argument("--out_csv", default="data/uwpr/nist20_uwpr_hits_feasibility.csv")
    parser.add_argument("--max_h_transfer", type=int, default=None)
    parser.add_argument("--no_cmf", action="store_true", help="Use std ion-mass convention only.")
    parser.add_argument("--n_jobs", type=int, default=parallel_utils.JOBLIB_N_JOBS)
    args = parser.parse_args()
    args.dsets = resolve_dsets_for_spec_fp(args.spec_fp, args.dsets)

    log.info("Loading hits from %s", args.hits_csv)
    hits = pd.read_csv(args.hits_csv)
    if hits.empty:
        raise ValueError(f"UWPR hits CSV is empty: {args.hits_csv}")
    log.info(
        "Loaded %d matched peaks (%d spectra, %d mols)",
        len(hits),
        hits["spec_id"].nunique(),
        hits["mol_id"].nunique(),
    )

    # The hits CSV has ion_mode but not the precursor adduct; join it from spec_df.
    spec_df = pd.read_pickle(args.spec_fp)[["spec_id", "dset", "prec_type"]]
    if args.dsets:
        spec_df = spec_df[spec_df["dset"].isin(args.dsets)]
        n_before_scope = len(hits)
        hits = hits[hits["spec_id"].isin(spec_df["spec_id"])]
        log.info(
            "Scoped hits to dsets=%s: retained %d/%d rows",
            args.dsets,
            len(hits),
            n_before_scope,
        )
    spec_df = spec_df.drop(columns=["dset"])
    hits = hits.merge(spec_df, on="spec_id", how="left")
    n_no_prec = hits["prec_type"].isna().sum()
    if n_no_prec:
        log.warning("%d rows had no prec_type after join (spec_id not in spec_df)", n_no_prec)
    if hits.empty:
        hits["feasibility"] = pd.Series(index=hits.index, dtype=object)
        hits.to_csv(args.out_csv, index=False)
        log.warning(
            "No UWPR hits remain after dataset scoping; saved an empty result to %s", args.out_csv
        )
        return

    is_compressed = _detect_is_compressed(args.frag_dp, int(hits["mol_id"].iloc[0]))
    worker = functools.partial(
        _feasibility_for_mol,
        frag_dp=args.frag_dp,
        is_compressed=is_compressed,
        max_h_transfer=args.max_h_transfer,
        include_cmf=not args.no_cmf,
    )
    mol_inputs = [(int(mol_id), grp) for mol_id, grp in hits.groupby("mol_id")]
    log.info("Scoring feasibility for %d molecules", len(mol_inputs))
    per_mol = parallel_utils.par_apply(mol_inputs, worker, need_unpack=True, n_jobs=args.n_jobs)

    verdict = pd.concat(per_mol).reindex(hits.index)
    hits["feasibility"] = verdict

    log.info("Verdict breakdown:\n%s", hits["feasibility"].value_counts(dropna=False).to_string())
    evaluable = hits[hits["feasibility"].isin(["dag_feasible", "contaminant_only"])]
    if len(evaluable):
        frac_contam = (evaluable["feasibility"] == "contaminant_only").mean()
        log.info(
            "Of %d evaluable matched peaks, %.1f%% are contaminant_only (DAG cannot explain)",
            len(evaluable),
            100 * frac_contam,
        )

    hits.to_csv(args.out_csv, index=False)
    log.info("Saved %s", args.out_csv)


if __name__ == "__main__":
    main()
