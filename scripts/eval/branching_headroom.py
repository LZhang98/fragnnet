"""Measure whether the OBSERVED per-parent sibling branching moves with collision energy.

This is a model-free headroom check for a parent-local competitive fragmentation head (the
directional-transition MPNN / sibling edge-softmax). That head can only help if, at a fixed
parent fragment, the *choice among sibling children* shifts with collision energy in a way beyond
replicate noise. If sibling branching is CE-invariant, a CE-conditioned competition head has no
signal to exploit among siblings and is dead on arrival -- exactly the outcome ce_extent_headroom
found for the CE->fragmentation-extent prior. No model and no training are involved: the branching
is read straight off the true spectrum via the DAG.

What "sibling branching" is
---------------------------
The fragmentation DAG stores directed ``parent -> child`` edges (``dag.edge_index``). For a parent
node ``p`` with children ``c1..ck``, the observed branching ``q(p -> ci)`` is the share of the
group's explained intensity that lands on child ``ci``. Each child's candidate ion m/z come from
its formula rows (``h_formulae_idx`` -> ``formula_peak_mzs``), in both the std and CMF ion-mass
conventions; a child "explains" a true peak if any of its candidate m/z matches within the
instrument tolerance.

Mass-alias ambiguity (min/max bracket)
--------------------------------------
A true peak can match several siblings at once. Rather than split its intensity arbitrarily, every
child's weight is reported as a bracket:

  - ``max``: every sibling that matches a peak claims its full intensity (upper bound per child;
    shared peaks counted for all matchers).
  - ``min``: a sibling claims a peak only when it is the *unique* matcher in the group (lower
    bound; shared peaks dropped).

If the min and max brackets agree on the verdict, the ambiguity does not matter; if they disagree,
the ambiguity hides the answer and that is itself the finding.

The CE test (controlled, with a replicate noise floor)
------------------------------------------------------
Spectra of one molecule are grouped by ``(parent, prec_type, inst_type, frag_mode)`` so that within
a group the ONLY thing that varies is NCE. Within each group we form spectrum pairs:

  - ``same-NCE`` pairs (identical nce)  -> the replicate NOISE FLOOR: how much sibling branching
    wobbles at fixed energy from measurement noise alone.
  - ``cross-NCE`` pairs (different nce) -> the SIGNAL: how much it moves when energy changes,
    stratified by the size of the NCE gap.

For each pair we record whether the dominant sibling (argmax child) changed, and the total-variation
distance between the two sibling distributions. Only parents with >= 2 explained siblings are
counted -- a parent with a single explained child has no branching to speak of, and the fraction of
parents that even reach 2 explained siblings is reported as a coverage number (if it is tiny, the
whole competition premise is moot regardless of the CE trend).

Reading the result
------------------
  - cross-NCE change >> same-NCE change, and rising with the NCE gap -> sibling branching is
    genuinely CE-dependent. The necessary condition for a CE-conditioned competition head HOLDS.
    (This is necessary, not sufficient: a follow-up must check the trained model does not already
    capture it.)
  - cross-NCE change ~= same-NCE change (flat in the gap) -> sibling choice is CE-invariant; the
    parent-local competition head has no CE signal among siblings. Do not build it.

Usage:
    python scripts/eval/branching_headroom.py \\
        --frag_dp data/frag/nist20_ma_mi_d3_h4_isoFalse_smpFalse/dags \\
        --split val --limit_mols 3000
"""

from __future__ import annotations

import argparse
import functools
import logging
import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd
import torch as th

import fragnnet.utils.data_utils as data_utils
from fragnnet.utils.feat_utils import get_node_feats
from fragnnet.utils.formula_utils import (
    PREC_TYPE_TO_CMF_MASS_DIFF,
    PREC_TYPE_TO_MASS_DIFF,
)
from fragnnet.utils.spec_utils import calculate_match_mzs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dag_oracle_scores import (  # noqa: E402
    _allowed_formula_idxs,
    _detect_is_compressed,
    _tolerance_for_inst_type,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# NCE-gap strata for cross-energy pairs (nce units). A real CE effect should grow with the gap.
GAP_BINS = [(0.0, 10.0), (10.0, 25.0), (25.0, 50.0), (50.0, 1e9)]
GAP_LABELS = ["<10", "10-25", "25-50", ">=50"]


def _node_neutral_mzs(
    dag_x: th.Tensor,
    node_feat_idxs: th.Tensor,
    formula_peak_mzs: th.Tensor,
    allowed_rows: set[int] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten every node's candidate neutral fragment m/z into one array.

    Args:
        dag_x: DAG node-feature matrix of shape (num_nodes, num_feats).
        node_feat_idxs: Feature-slice offsets of shape (num_feat_groups + 1,).
        formula_peak_mzs: Neutral formula m/z of shape (num_formulas, num_isotopes).
        allowed_rows: If given, only formula rows in this set are kept (H-transfer budget);
            None keeps every row the DAG carries.

    Returns:
        Tuple ``(neutral_mz, node_of_cand)`` of shape (M,) each: the neutral m/z of every
        (node, formula-row, isotope) candidate and the node index it belongs to. Row 0 (the
        NULL formula) and zero-mass padding are dropped.
    """
    h_formulae_idx = get_node_feats(dag_x, node_feat_idxs, "h_formulae_idx").numpy()  # (N, 9)
    fpm = formula_peak_mzs.numpy()  # (F, iso)
    num_nodes = h_formulae_idx.shape[0]
    neutral, node_of = [], []
    for n in range(num_nodes):
        rows = np.unique(h_formulae_idx[n])
        rows = rows[rows != 0]  # 0 == NULL formula / unused H-slot
        if allowed_rows is not None:
            rows = np.array([r for r in rows if r in allowed_rows], dtype=np.int64)
        if rows.size == 0:
            continue
        mzs = fpm[rows].reshape(-1)  # (rows * iso,)
        mzs = mzs[mzs > 0.0]  # drop isotope padding
        if mzs.size == 0:
            continue
        neutral.append(mzs)
        node_of.append(np.full(mzs.shape[0], n, dtype=np.int64))
    if not neutral:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.int64)
    return np.concatenate(neutral).astype(np.float64), np.concatenate(node_of)


def _peak_node_match(
    true_mzs: np.ndarray,
    neutral_mz: np.ndarray,
    node_of_cand: np.ndarray,
    num_nodes: int,
    prec_type: str,
    tolerance: float,
    tolerance_min_mz: float,
) -> np.ndarray | None:
    """Boolean (num_true_peaks, num_nodes) matrix: does node n explain true peak i?

    Candidate ion m/z are built in both the std and CMF conventions for ``prec_type`` and matched
    to the true peaks within the instrument tolerance; a hit on either convention counts.

    Returns:
        The match matrix, or None if the adduct is unsupported.
    """
    if prec_type not in PREC_TYPE_TO_MASS_DIFF:
        return None
    std_diff = PREC_TYPE_TO_MASS_DIFF[prec_type]
    cmf_diff = PREC_TYPE_TO_CMF_MASS_DIFF.get(prec_type, std_diff)
    cand_mz = np.concatenate([neutral_mz + std_diff, neutral_mz + cmf_diff])
    cand_node = np.concatenate([node_of_cand, node_of_cand])

    match = calculate_match_mzs(
        th.tensor(true_mzs, dtype=th.float64),
        th.tensor(cand_mz, dtype=th.float64),
        tolerance=tolerance,
        relative=True,
        tolerance_min_mz=tolerance_min_mz,
    ).numpy()  # (num_true, M)

    peak_node = np.zeros((true_mzs.shape[0], num_nodes), dtype=bool)
    hit_i, hit_j = np.nonzero(match)
    peak_node[hit_i, cand_node[hit_j]] = True
    return peak_node


def _sibling_dists(
    peak_node: np.ndarray,
    true_ints: np.ndarray,
    children: np.ndarray,
) -> dict[str, np.ndarray] | None:
    """Min/max-bracketed sibling branching distributions for one parent group.

    Args:
        peak_node: (num_true_peaks, num_nodes) match matrix.
        true_ints: True peak intensities of shape (num_true_peaks,).
        children: Node indices of this parent's children, shape (k,).

    Returns:
        Dict with ``q_min``/``q_max`` (shape (k,), normalised) and the group's explained
        intensity, or None if fewer than 2 children carry any explained intensity under max
        attribution (nothing to compete over).
    """
    sub = peak_node[:, children]  # (num_true, k)
    matched_peak = sub.any(axis=1)
    if not matched_peak.any():
        return None
    ints = true_ints[:, None]
    w_max = (sub * ints).sum(axis=0)  # shared peaks counted for every matcher
    unique = sub & (sub.sum(axis=1, keepdims=True) == 1)
    w_min = (unique * ints).sum(axis=0)  # only peaks a single sibling explains

    if (w_max > 0).sum() < 2:  # single explained sibling -> no branching to model
        return None
    out = {"expl_int": float(true_ints[matched_peak].sum())}
    for name, w in (("q_min", w_min), ("q_max", w_max)):
        s = w.sum()
        out[name] = w / s if s > 0 else None
    return out


def _pair_stats(qs: list[np.ndarray], nces: list[float]) -> dict:
    """Accumulate same-NCE vs cross-NCE branching-change stats over one controlled group.

    Args:
        qs: Sibling distributions (each shape (k,)) for the spectra in this
            ``(parent, prec_type, inst_type, frag_mode)`` group, aligned to the same children.
        nces: The NCE of each distribution, same order as ``qs``.

    Returns:
        Dict of running sums: same-NCE (noise floor) and cross-NCE (signal, by gap bin) pair
        counts, dominant-sibling changes, and summed total-variation distances.
    """
    acc = {
        "same_n": 0, "same_dom": 0, "same_tv": 0.0,
        "cross_n": 0, "cross_dom": 0, "cross_tv": 0.0,
    }
    for g in GAP_LABELS:
        acc[f"gap_{g}_n"] = 0
        acc[f"gap_{g}_dom"] = 0
        acc[f"gap_{g}_tv"] = 0.0
    for (qi, ni), (qj, nj) in combinations(zip(qs, nces), 2):
        if qi is None or qj is None:
            continue
        dom = int(np.argmax(qi) != np.argmax(qj))
        tv = 0.5 * float(np.abs(qi - qj).sum())
        gap = abs(ni - nj)
        if gap == 0.0:
            acc["same_n"] += 1
            acc["same_dom"] += dom
            acc["same_tv"] += tv
        else:
            acc["cross_n"] += 1
            acc["cross_dom"] += dom
            acc["cross_tv"] += tv
            for (lo, hi), lab in zip(GAP_BINS, GAP_LABELS):
                if lo < gap <= hi:
                    acc[f"gap_{lab}_n"] += 1
                    acc[f"gap_{lab}_dom"] += dom
                    acc[f"gap_{lab}_tv"] += tv
                    break
    return acc


def _process_mol_group(
    mol_id: int,
    group_rows: list[dict],
    frag_dp: str,
    is_compressed: bool,
    max_h_transfer: int | None,
) -> dict | None:
    """Compute branching-change accumulators for one molecule's spectra.

    Returns a dict of running sums for both brackets plus coverage counts, or None if the
    molecule's DAG explains no competable parent in any spectrum.
    """
    import fragnnet.utils.frag_utils as frag_utils

    dag_d = frag_utils.load_frag_d(mol_id, frag_dp, is_compressed=is_compressed)
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

    # Parent -> sorted children (siblings), once per molecule.
    parents = np.unique(edge_index[0])
    children_of = {
        int(p): np.unique(edge_index[1][edge_index[0] == p]) for p in parents
    }
    children_of = {p: c for p, c in children_of.items() if c.size >= 2}
    if not children_of:
        return None

    # per (parent, prec_type, inst_type, frag_mode) -> {"q_min": [...], "q_max": [...], "nce": [...]}
    strata: dict[tuple, dict[str, list]] = {}
    n_active_parent_spectra = 0
    n_parent_spectra = 0
    expl_int_fracs = []

    for row in group_rows:
        peaks = row["peaks"]
        if not peaks:
            continue
        true_mzs_raw, true_ints_raw = zip(*peaks)
        true_mzs = np.asarray(true_mzs_raw, dtype=np.float64)
        true_ints = np.asarray(true_ints_raw, dtype=np.float64)
        if true_ints.sum() <= 0:
            continue
        tol, tol_min = _tolerance_for_inst_type(row["inst_type"])
        peak_node = _peak_node_match(
            true_mzs, neutral_mz, node_of_cand, num_nodes,
            row["prec_type"], tol, tol_min,
        )
        if peak_node is None:
            continue
        ctx = (row["prec_type"], row["inst_type"], row["frag_mode"])
        for p, children in children_of.items():
            n_parent_spectra += 1
            dists = _sibling_dists(peak_node, true_ints, children)
            if dists is None:
                continue
            n_active_parent_spectra += 1
            expl_int_fracs.append(dists["expl_int"] / true_ints.sum())
            key = (p, *ctx)
            slot = strata.setdefault(key, {"q_min": [], "q_max": [], "nce": []})
            slot["q_min"].append(dists["q_min"])
            slot["q_max"].append(dists["q_max"])
            slot["nce"].append(float(row["nce"]))

    # Accumulate pair stats across all controlled groups, per bracket.
    out = {
        "mol_id": mol_id,
        "n_parent_spectra": n_parent_spectra,
        "n_active_parent_spectra": n_active_parent_spectra,
        "sum_expl_int_frac": float(np.sum(expl_int_fracs)) if expl_int_fracs else 0.0,
        "n_expl": len(expl_int_fracs),
        "n_strata_multi": 0,
    }
    for bracket in ("q_min", "q_max"):
        agg = None
        for slot in strata.values():
            if len(slot["nce"]) < 2:
                continue
            ps = _pair_stats(slot[bracket], slot["nce"])
            if agg is None:
                agg = ps
            else:
                for k, v in ps.items():
                    agg[k] += v
        if agg is None:
            agg = _pair_stats([], [])
        for k, v in agg.items():
            out[f"{bracket}__{k}"] = v
    # multi-spectrum strata count (same for both brackets)
    out["n_strata_multi"] = sum(1 for s in strata.values() if len(s["nce"]) >= 2)
    return out


def load_spec_df(spec_fp: str, split_dp: str, split: str) -> pd.DataFrame:
    """Load spec_df and filter to a split (mirrors ce_extent_headroom.load_spec_df)."""
    log.info("Loading %s", spec_fp)
    spec_df = pd.read_pickle(spec_fp)
    spec_df = spec_df[spec_df["prec_type"].isin(PREC_TYPE_TO_MASS_DIFF.keys())]
    spec_df = spec_df[spec_df["peaks"].apply(len) > 0]
    spec_df = spec_df[spec_df["nce"].notna()]
    if split != "all":
        split_fp = os.path.join(split_dp, f"{split}_ids.csv")
        split_spec_ids = set(pd.read_csv(split_fp)["spec_id"])
        spec_df = spec_df[spec_df["spec_id"].isin(split_spec_ids)]
    return spec_df[
        ["spec_id", "mol_id", "peaks", "prec_mz", "prec_type", "inst_type", "frag_mode", "nce"]
    ]


def _bracket_table(rows: pd.DataFrame, bracket: str) -> None:
    """Print the same-NCE (floor) vs cross-NCE (signal, by gap) branching-change table."""
    def s(col: str) -> float:
        return float(rows[f"{bracket}__{col}"].sum())

    print(f"\n  bracket = {bracket.replace('q_', '')}-attribution")
    print(f"    {'stratum':>12s} {'pairs':>9s} {'dom-change':>11s} {'mean-TV':>9s}")
    same_n = s("same_n")
    print(
        f"    {'same-NCE':>12s} {int(same_n):>9d} "
        f"{(s('same_dom') / same_n if same_n else float('nan')):>11.4f} "
        f"{(s('same_tv') / same_n if same_n else float('nan')):>9.4f}   <- noise floor"
    )
    cross_n = s("cross_n")
    print(
        f"    {'cross-NCE':>12s} {int(cross_n):>9d} "
        f"{(s('cross_dom') / cross_n if cross_n else float('nan')):>11.4f} "
        f"{(s('cross_tv') / cross_n if cross_n else float('nan')):>9.4f}"
    )
    for lab in GAP_LABELS:
        n = s(f"gap_{lab}_n")
        if n == 0:
            continue
        print(
            f"    {'  gap ' + lab:>12s} {int(n):>9d} "
            f"{s(f'gap_{lab}_dom') / n:>11.4f} {s(f'gap_{lab}_tv') / n:>9.4f}"
        )
    if same_n and cross_n:
        lift_dom = s("cross_dom") / cross_n - s("same_dom") / same_n
        lift_tv = s("cross_tv") / cross_n - s("same_tv") / same_n
        print(f"    -> cross - floor:  dom-change {lift_dom:+.4f}   mean-TV {lift_tv:+.4f}")


def summarize(df: pd.DataFrame) -> None:
    """Print coverage and the branching-change tables for both brackets."""
    n_parent_spec = int(df["n_parent_spectra"].sum())
    n_active = int(df["n_active_parent_spectra"].sum())
    n_expl = int(df["n_expl"].sum())
    mean_expl_frac = df["sum_expl_int_frac"].sum() / n_expl if n_expl else float("nan")

    print(f"\n{'=' * 92}")
    print("Observed per-parent sibling branching vs collision energy (model-free)")
    print(f"{'=' * 92}")
    print(f"molecules: {df['mol_id'].nunique()}")
    print(
        f"parent-spectra evaluated: {n_parent_spec}   with >=2 explained siblings (competable): "
        f"{n_active} ({(100.0 * n_active / n_parent_spec if n_parent_spec else 0):.1f}%)"
    )
    print(f"mean explained-intensity fraction over competable parent-groups: {mean_expl_frac:.3f}")
    print(f"controlled groups with >=2 NCE-varying spectra: {int(df['n_strata_multi'].sum())}")
    print(
        "\nControlled: pairs share (parent, prec_type, inst_type, frag_mode); only NCE differs.\n"
        "'dom-change' = fraction of pairs whose top sibling flips. 'mean-TV' = sibling-distribution\n"
        "total-variation distance. same-NCE is the replicate noise floor."
    )
    for bracket in ("q_min", "q_max"):
        _bracket_table(df, bracket)

    print(f"\n{'-' * 92}")
    print("READ: cross-NCE dom-change/TV clearly ABOVE the same-NCE floor and RISING with the gap")
    print("      -> sibling branching is CE-dependent; a parent-local competition head has signal.")
    print("      cross-NCE ~= floor and flat in the gap -> CE-invariant sibling choice; do not build.")
    print("      min and max brackets disagreeing -> mass-alias ambiguity hides the answer.")
    print("      A tiny competable-% means few parents even have sibling competition to model.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--frag_dp", required=True, help="DAG dir the model is trained on.")
    parser.add_argument("--spec_fp", default="data/proc/nist20/spec_df.pkl")
    parser.add_argument("--split_dp", default="data/split/nist20_ma_mi_inchikey")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="val")
    parser.add_argument("--limit_mols", type=int, default=None, help="Subsample this many mols.")
    parser.add_argument("--max_h_transfer", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_jobs", type=int, default=data_utils.JOBLIB_N_JOBS)
    parser.add_argument("--out_fp", default=None, help="Optional CSV of the per-mol accumulators.")
    args = parser.parse_args()

    spec_df = load_spec_df(args.spec_fp, args.split_dp, args.split)
    if args.limit_mols is not None:
        rng = np.random.default_rng(args.seed)
        mol_ids = spec_df["mol_id"].unique()
        keep = rng.choice(mol_ids, size=min(args.limit_mols, len(mol_ids)), replace=False)
        spec_df = spec_df[spec_df["mol_id"].isin(keep)]

    groups = spec_df.groupby("mol_id")
    sample_mol_id = int(next(iter(groups.groups.keys())))
    is_compressed = _detect_is_compressed(args.frag_dp, sample_mol_id)
    mol_inputs = [(int(mol_id), g.to_dict("records")) for mol_id, g in groups]

    worker = functools.partial(
        _process_mol_group,
        frag_dp=args.frag_dp,
        is_compressed=is_compressed,
        max_h_transfer=args.max_h_transfer,
    )
    log.info("Processing %d mols (%d spectra)", len(mol_inputs), len(spec_df))
    per_mol = data_utils.par_apply(mol_inputs, worker, need_unpack=True, n_jobs=args.n_jobs)

    rows = [r for r in per_mol if r is not None]
    if not rows:
        raise ValueError("No molecule produced a competable parent; check --frag_dp and --split.")
    df = pd.DataFrame(rows)
    log.info("Got branching accumulators for %d molecules", len(df))

    summarize(df)

    if args.out_fp:
        df.to_csv(args.out_fp, index=False)
        log.info("Saved per-mol accumulators to %s", args.out_fp)


if __name__ == "__main__":
    main()
