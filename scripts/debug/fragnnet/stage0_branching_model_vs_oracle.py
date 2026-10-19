"""Stage 0 headroom gate for FluxMPNN: does the TRAINED model already capture sibling branching?

`branching_headroom.py` established the *necessary* condition model-free: the OBSERVED per-parent
sibling branching is CE-dependent (dominant-sibling flips rise with the NCE gap, well above the
replicate floor). That says a CE-conditioned competition head *could* help. Stage 0 checks the
*sufficient* half: whether the current flat-softmax model already reproduces that observed branching.
If it does, a sibling-softmax residual has nothing to add and Stages 1-3 are dead on arrival --
exactly the discipline `ce_extent_headroom` imposed on the CE->extent prior.

The comparison (per spectrum, per competable parent)
----------------------------------------------------
The model emits a flat categorical P(n) over DAG nodes (read from the fixed inference HDF5, sliced
per spectrum via ``node_start_points``/``node_end_points``; node ``k`` == DAG node ``k``). For a
parent ``p`` with observed explained siblings ``E`` (the children that actually carry matched peak
intensity, ``|E| >= 2``), the model's *implied* branching is the renormalized child mass::

    q_base(p -> c) = P(c) / sum_{c' in E} P(c')      for c in E

This is compared against the OBSERVED branching ``q_obs(p -> c)`` from the true spectrum (the
`branching_headroom` machinery: share of explained intensity per sibling, min/max alias brackets),
restricted to the same support ``E``. Restricting to ``E`` isolates *apportionment among the
siblings that appear* -- which is exactly what the CE-conditioned sibling head would fix -- from
*which siblings appear at all* (coverage), which is the separate halt/escape story.

Baselines, per NCE bin
----------------------
  - ``q_base`` vs ``q_obs``  : the model. TV distance + dominant-sibling mismatch.
  - ``q_unif`` vs ``q_obs``  : uniform over ``E``. The "no branching info" ceiling the model must
    beat (expected dom-mismatch ``1 - 1/|E|``).

Reading the result
------------------
  - model TV low and FLAT across NCE, far below uniform -> the model already apportions siblings
    well everywhere, including high energy. **No headroom. STOP -- do not build Stages 1-3.**
  - model TV rising toward the uniform ceiling at high NCE (model dom-mismatch climbing well above
    the ~0.11 same-NCE replicate floor from `branching_headroom`) -> the model loses branching skill
    exactly where CE matters. **Real headroom. Proceed to Stage 1.**
  - min vs max bracket disagree on the verdict -> mass-alias ambiguity hides the answer.

Usage:
    python scripts/debug/fragnnet/stage0_branching_model_vs_oracle.py \\
        --pred_hdf5 output/inference_eval/gdu9njqa/predictions_val.hdf5 \\
        --frag_dp data/frag/nist20_ma_mi_d3_h4_isoFalse_smpFalse/dags \\
        --split val
"""

from __future__ import annotations

import argparse
import functools
import logging
import os
import sys

import h5py
import numpy as np
import pandas as pd

from fragnnet.utils import parallel_utils

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "oracle")
)
from branching_headroom import (  # noqa: E402
    _node_neutral_mzs,
    _peak_node_match,
    _sibling_dists,
    load_spec_df,
)
from dag_oracle_scores import (  # noqa: E402
    _allowed_formula_idxs,
    _detect_is_compressed,
    _tolerance_for_inst_type,
)
from oracle_dataset_utils import resolve_dsets_for_spec_fp  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Absolute-NCE strata (nce units); chosen from the val nce distribution (median 45, IQR 30-85).
NCE_BINS = [(0.0, 20.0), (20.0, 35.0), (35.0, 50.0), (50.0, 85.0), (85.0, 120.0), (120.0, 1e9)]
NCE_LABELS = ["0-20", "20-35", "35-50", "50-85", "85-120", ">=120"]
BRACKETS = ("q_max", "q_min")


def _nce_label(nce: float) -> str | None:
    """Return the NCE-bin label for a scalar nce, or None if it falls in no bin."""
    for (lo, hi), lab in zip(NCE_BINS, NCE_LABELS):
        if lo <= nce < hi:
            return lab
    return None


def _tv(a: np.ndarray, b: np.ndarray) -> float:
    """Total-variation distance between two aligned distributions."""
    return 0.5 * float(np.abs(a - b).sum())


def _empty_acc() -> dict:
    """Zeroed accumulator keyed by (bin, bracket)."""
    acc = {"n_parent_spectra": 0, "n_competable": 0}
    for lab in NCE_LABELS:
        for br in BRACKETS:
            for stat in ("n", "tv_model", "tv_unif", "dom_mis_model", "dom_mis_unif"):
                acc[f"{lab}__{br}__{stat}"] = 0.0
    return acc


def _process_mol_group(
    mol_id: int,
    group_rows: list[dict],
    frag_dp: str,
    is_compressed: bool,
    max_h_transfer: int | None,
) -> dict | None:
    """Model-vs-oracle branching accumulators for one molecule's spectra.

    Each ``group_rows`` entry carries the true spectrum fields plus ``pn``: the model's flat P(n)
    over this molecule's DAG nodes (aligned so ``pn[k]`` == P(DAG node k)). Returns a dict of
    running sums over NCE bins and alias brackets, or None if no competable parent is explained.
    """
    from fragnnet.utils import dag_format_utils

    dag_d = dag_format_utils.load_frag_d(mol_id, frag_dp, is_compressed=is_compressed)
    if dag_d is None:
        return None
    dag = dag_d["dag"]
    edge_index = dag.edge_index.numpy()
    num_nodes = dag.x.shape[0]
    allowed = None
    if max_h_transfer is not None:
        allowed = set(_allowed_formula_idxs(dag_d["idx_by_h_delta"], max_h_transfer))
    neutral_mz, node_of_cand = _node_neutral_mzs(
        dag.x, dag.node_feat_idxs[0], dag_d["formula_peak_mzs"], allowed
    )
    if neutral_mz.size == 0:
        return None

    parents = np.unique(edge_index[0])
    children_of = {int(p): np.unique(edge_index[1][edge_index[0] == p]) for p in parents}
    children_of = {p: c for p, c in children_of.items() if c.size >= 2}
    if not children_of:
        return None

    acc = _empty_acc()
    for row in group_rows:
        peaks = row["peaks"]
        if not peaks:
            continue
        pn = row["pn"]
        if len(pn) != num_nodes:
            raise ValueError(
                f"P(n) length {len(pn)} != DAG node count {num_nodes} for mol {mol_id}; "
                "frag_dp does not match the run's DAGs."
            )
        nce_lab = _nce_label(float(row["nce"]))
        if nce_lab is None:
            continue
        true_mzs_raw, true_ints_raw = zip(*peaks)
        true_mzs = np.asarray(true_mzs_raw, dtype=np.float64)
        true_ints = np.asarray(true_ints_raw, dtype=np.float64)
        if true_ints.sum() <= 0:
            continue
        tol, tol_min = _tolerance_for_inst_type(row["inst_type"])
        peak_node = _peak_node_match(
            true_mzs, neutral_mz, node_of_cand, num_nodes, row["prec_type"], tol, tol_min
        )
        if peak_node is None:
            continue

        for _p, children in children_of.items():
            acc["n_parent_spectra"] += 1
            dists = _sibling_dists(peak_node, true_ints, children)
            if dists is None:
                continue
            # Explained-sibling support E (children carrying matched intensity under max).
            e_mask = dists["q_max"] > 0.0
            if e_mask.sum() < 2:
                continue
            p_child = pn[children][e_mask].astype(np.float64)
            if p_child.sum() <= 0:  # model gives zero mass to every explained sibling
                continue
            acc["n_competable"] += 1
            q_base = p_child / p_child.sum()
            k = e_mask.sum()
            q_unif = np.full(k, 1.0 / k)
            base_argmax = int(np.argmax(q_base))
            for br in BRACKETS:
                q = dists[br]
                if q is None:
                    continue
                q_e = q[e_mask]
                s = q_e.sum()
                if s <= 0:
                    continue
                q_e = q_e / s
                obs_argmax = int(np.argmax(q_e))
                pre = f"{nce_lab}__{br}__"
                acc[pre + "n"] += 1
                acc[pre + "tv_model"] += _tv(q_base, q_e)
                acc[pre + "tv_unif"] += _tv(q_unif, q_e)
                acc[pre + "dom_mis_model"] += float(base_argmax != obs_argmax)
                acc[pre + "dom_mis_unif"] += 1.0 - 1.0 / k  # expected mismatch of a uniform guess

    if acc["n_competable"] == 0:
        return None
    acc["mol_id"] = mol_id
    return acc


def _load_pred_pn(pred_hdf5: str) -> dict[int, np.ndarray]:
    """Map spec_id -> flat model P(n) over that spectrum's DAG nodes (in DAG node order)."""
    log.info("Loading model P(n) from %s", pred_hdf5)
    with h5py.File(pred_hdf5, "r") as f:
        for req in ("pred_node_probs", "node_start_points", "node_end_points", "spec_id"):
            if req not in f:
                raise KeyError(
                    f"{req} missing from {pred_hdf5}; regenerate with the node-grouping writer."
                )
        pn = f["pred_node_probs"][:].astype(np.float64)
        ns = f["node_start_points"][:]
        ne = f["node_end_points"][:]
        spec_ids = f["spec_id"][:]
    return {int(sid): pn[s:e] for sid, s, e in zip(spec_ids, ns, ne)}


def _print_table(df: pd.DataFrame) -> None:
    """Print the model-vs-oracle branching table per NCE bin, for both alias brackets."""
    n_parent = int(df["n_parent_spectra"].sum())
    n_comp = int(df["n_competable"].sum())
    print(f"\n{'=' * 96}")
    print("Stage 0: does the trained model already capture per-parent sibling branching?")
    print(f"{'=' * 96}")
    print(f"molecules: {df['mol_id'].nunique()}")
    print(
        f"parent-spectra evaluated: {n_parent}   competable (>=2 explained siblings, model mass>0): "
        f"{n_comp} ({(100.0 * n_comp / n_parent if n_parent else 0):.1f}%)"
    )
    print(
        "\nTV = total-variation vs observed branching over the explained-sibling support E.\n"
        "dom-mis = fraction of parent-spectra whose predicted top sibling != observed top sibling.\n"
        "'unif' = uniform-over-E baseline (the no-branching-info ceiling the model must beat).\n"
        "same-NCE replicate floor for dom-change is ~0.11 (branching_headroom)."
    )
    for br in BRACKETS:
        print(f"\n  bracket = {br.replace('q_', '')}-attribution")
        print(
            f"    {'nce':>8s} {'n':>8s} {'TV_model':>9s} {'TV_unif':>9s} "
            f"{'dom_model':>10s} {'dom_unif':>9s}"
        )
        for lab in NCE_LABELS:
            n = float(df[f"{lab}__{br}__n"].sum())
            if n == 0:
                continue
            tvm = df[f"{lab}__{br}__tv_model"].sum() / n
            tvu = df[f"{lab}__{br}__tv_unif"].sum() / n
            dmm = df[f"{lab}__{br}__dom_mis_model"].sum() / n
            dmu = df[f"{lab}__{br}__dom_mis_unif"].sum() / n
            print(
                f"    {lab:>8s} {int(n):>8d} {tvm:>9.4f} {tvu:>9.4f} {dmm:>10.4f} {dmu:>9.4f}"
            )


def summarize(df: pd.DataFrame) -> None:
    """Print the table plus the go/no-go read."""
    _print_table(df)
    print(f"\n{'-' * 96}")
    print("READ (max bracket is primary; min is the alias lower bound):")
    print("  TV_model flat across NCE and far below TV_unif -> model has branching, no headroom, STOP.")
    print("  TV_model climbing toward TV_unif / dom_model rising well above ~0.11 at high NCE")
    print("     -> model loses branching skill where CE matters -> real headroom, proceed to Stage 1.")
    print("  max and min brackets disagreeing on the trend -> mass-alias ambiguity hides the answer.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pred_hdf5", required=True, help="Inference HDF5 with node grouping.")
    parser.add_argument(
        "--frag_dp",
        default="data/frag/nist20_ma_mi_d3_h4_isoFalse_smpFalse/dags",
        help="DAG dir the run was trained on (must match the HDF5 node ordering).",
    )
    parser.add_argument("--spec_fp", default="data/proc/nist20/spec_df.pkl")
    parser.add_argument(
        "--dsets", "--dset", nargs="+", default=None, help="Filter spec_df to these dataset labels."
    )
    parser.add_argument("--split_dp", default="data/split/nist20_ma_mi_inchikey")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="val")
    parser.add_argument("--max_h_transfer", type=int, default=None)
    parser.add_argument("--n_jobs", type=int, default=parallel_utils.JOBLIB_N_JOBS)
    parser.add_argument("--out_fp", default=None, help="Optional CSV of the per-mol accumulators.")
    args = parser.parse_args()
    args.dsets = resolve_dsets_for_spec_fp(args.spec_fp, args.dsets)

    pn_by_spec = _load_pred_pn(args.pred_hdf5)
    spec_df = load_spec_df(args.spec_fp, args.split_dp, args.split, args.dsets)
    spec_df = spec_df[spec_df["spec_id"].isin(pn_by_spec)]
    if spec_df.empty:
        raise ValueError("No overlap between spec_df split and the HDF5 spec_ids.")
    log.info("Matched %d spectra to model P(n)", len(spec_df))

    groups = spec_df.groupby("mol_id")
    sample_mol_id = int(next(iter(groups.groups.keys())))
    is_compressed = _detect_is_compressed(args.frag_dp, sample_mol_id)

    mol_inputs = []
    for mol_id, g in groups:
        records = g.to_dict("records")
        for r in records:
            r["pn"] = pn_by_spec[int(r["spec_id"])]
        mol_inputs.append((int(mol_id), records))

    worker = functools.partial(
        _process_mol_group,
        frag_dp=args.frag_dp,
        is_compressed=is_compressed,
        max_h_transfer=args.max_h_transfer,
    )
    log.info("Processing %d mols (%d spectra)", len(mol_inputs), len(spec_df))
    per_mol = parallel_utils.par_apply(mol_inputs, worker, need_unpack=True, n_jobs=args.n_jobs)

    rows = [r for r in per_mol if r is not None]
    if not rows:
        raise ValueError("No molecule produced a competable parent; check --frag_dp and --split.")
    df = pd.DataFrame(rows)
    log.info("Got model-vs-oracle accumulators for %d molecules", len(df))

    summarize(df)

    if args.out_fp:
        df.to_csv(args.out_fp, index=False)
        log.info("Saved per-mol accumulators to %s", args.out_fp)


if __name__ == "__main__":
    main()
