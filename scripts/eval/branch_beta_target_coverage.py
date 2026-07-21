"""Audit clean oracle beta-target coverage for sibling-branch supervision.

This is stricter than ``branching_headroom.py``.  It answers whether the proposed
offline beta target has enough unambiguous training signal before adding a loss:

  1. keep only children with exactly one DAG parent;
  2. within each parent, require at least two such children;
  3. for a spectrum, require at least two supervised children to explain true
     peak intensity;
  4. drop the parent-spectrum if any true peak aliases across multiple observed
     supervised siblings.

The resulting parent-spectra are the places where a dominant-sibling CE target
can be built without MAGMa-style intensity splitting or multi-parent credit
assignment.

Usage:
    python scripts/eval/branch_beta_target_coverage.py \\
        --frag_dp data/frag/nist20_ma_mi_d3_h4_isoFalse_smpFalse/dags \\
        --split train --limit_mols 1000
"""

from __future__ import annotations

import argparse
import functools
import logging
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

import fragnnet.utils.data_utils as data_utils

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from branching_headroom import (  # noqa: E402
    _node_neutral_mzs,
    _peak_node_match,
    load_spec_df,
)
from dag_oracle_scores import (  # noqa: E402
    _allowed_formula_idxs,
    _detect_is_compressed,
    _tolerance_for_inst_type,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


NCE_BINS = [(0.0, 20.0), (20.0, 35.0), (35.0, 50.0), (50.0, 85.0), (85.0, 120.0), (120.0, 1e9)]
NCE_LABELS = ["0-20", "20-35", "35-50", "50-85", "85-120", ">=120"]


def _nce_label(nce: float) -> str | None:
    for (lo, hi), lab in zip(NCE_BINS, NCE_LABELS):
        if lo <= nce < hi:
            return lab
    return None


def _empty_acc() -> dict:
    acc = {
        "n_mol": 0,
        "n_spectra": 0,
        "n_parents_total": 0,
        "n_parents_ge2_children": 0,
        "n_parents_ge2_single_parent_children": 0,
        "n_parent_spectra_total": 0,
        "n_parent_spectra_single_parent_eligible": 0,
        "n_parent_spectra_ge2_observed": 0,
        "n_parent_spectra_alias_free": 0,
        "n_parent_spectra_dominant_usable": 0,
        "sum_alias_free_expl_int_frac": 0.0,
        "n_alias_free_expl_int_frac": 0,
        "n_controlled_strata_multi": 0,
        "n_controlled_strata_dom_flip": 0,
    }
    for lab in NCE_LABELS:
        acc[f"nce_{lab}__eligible"] = 0
        acc[f"nce_{lab}__alias_free"] = 0
        acc[f"nce_{lab}__dominant_usable"] = 0
    return acc


def _merge_acc(dst: dict, src: dict) -> None:
    for k, v in src.items():
        if k == "mol_id":
            continue
        dst[k] = dst.get(k, 0) + v


def _process_mol_group(
    mol_id: int,
    group_rows: list[dict],
    frag_dp: str,
    is_compressed: bool,
    max_h_transfer: int | None,
    dominance_margin: float,
) -> dict | None:
    """Return clean beta-target coverage counts for one molecule."""
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

    parents_all = np.unique(edge_index[0])
    indeg = np.bincount(edge_index[1], minlength=num_nodes)

    children_all: dict[int, np.ndarray] = {}
    children_single_parent: dict[int, np.ndarray] = {}
    for p in parents_all:
        children = np.unique(edge_index[1][edge_index[0] == p])
        children_all[int(p)] = children
        sp_children = children[indeg[children] == 1]
        if sp_children.size >= 2:
            children_single_parent[int(p)] = sp_children

    acc = _empty_acc()
    acc["mol_id"] = mol_id
    acc["n_mol"] = 1
    acc["n_spectra"] = len(group_rows)
    acc["n_parents_total"] = int(len(parents_all))
    acc["n_parents_ge2_children"] = int(sum(c.size >= 2 for c in children_all.values()))
    acc["n_parents_ge2_single_parent_children"] = int(len(children_single_parent))

    if not children_single_parent:
        return acc

    # Track whether the clean dominant child actually varies with NCE inside controlled contexts.
    # key: (parent, prec_type, inst_type, frag_mode) -> list[(nce, dominant_child)]
    strata: dict[tuple, list[tuple[float, int]]] = defaultdict(list)

    for row in group_rows:
        peaks = row["peaks"]
        if not peaks:
            continue
        true_mzs_raw, true_ints_raw = zip(*peaks)
        true_mzs = np.asarray(true_mzs_raw, dtype=np.float64)
        true_ints = np.asarray(true_ints_raw, dtype=np.float64)
        total_int = float(true_ints.sum())
        if total_int <= 0:
            continue

        tol, tol_min = _tolerance_for_inst_type(row["inst_type"])
        peak_node = _peak_node_match(
            true_mzs,
            neutral_mz,
            node_of_cand,
            num_nodes,
            row["prec_type"],
            tol,
            tol_min,
        )
        if peak_node is None:
            continue

        lab = _nce_label(float(row["nce"]))
        for p, children in children_single_parent.items():
            acc["n_parent_spectra_total"] += 1
            acc["n_parent_spectra_single_parent_eligible"] += 1
            if lab is not None:
                acc[f"nce_{lab}__eligible"] += 1

            sub = peak_node[:, children]  # true peaks x supervised siblings
            observed_mask = sub.any(axis=0)
            if int(observed_mask.sum()) < 2:
                continue
            acc["n_parent_spectra_ge2_observed"] += 1

            observed_sub = sub[:, observed_mask]
            matched_observed_peak = observed_sub.any(axis=1)
            # Clean m/z-resolvable target: no true peak may be claimed by more than one
            # observed supervised sibling.
            if np.any(observed_sub[matched_observed_peak].sum(axis=1) > 1):
                continue

            obs_children = children[observed_mask]
            weights = (observed_sub * true_ints[:, None]).sum(axis=0).astype(np.float64)
            weight_sum = float(weights.sum())
            if weight_sum <= 0:
                continue

            acc["n_parent_spectra_alias_free"] += 1
            acc["sum_alias_free_expl_int_frac"] += weight_sum / total_int
            acc["n_alias_free_expl_int_frac"] += 1
            if lab is not None:
                acc[f"nce_{lab}__alias_free"] += 1

            order = np.argsort(weights)[::-1]
            top = float(weights[order[0]])
            second = float(weights[order[1]])
            if dominance_margin > 0.0 and (top - second) / weight_sum < dominance_margin:
                continue

            dominant_child = int(obs_children[order[0]])
            acc["n_parent_spectra_dominant_usable"] += 1
            if lab is not None:
                acc[f"nce_{lab}__dominant_usable"] += 1

            key = (p, row["prec_type"], row["inst_type"], row["frag_mode"])
            strata[key].append((float(row["nce"]), dominant_child))

    for vals in strata.values():
        if len(vals) < 2:
            continue
        nces = {nce for nce, _ in vals}
        if len(nces) < 2:
            continue
        acc["n_controlled_strata_multi"] += 1
        by_nce: dict[float, set[int]] = defaultdict(set)
        for nce, child in vals:
            by_nce[nce].add(child)
        # Conservative: count a flip only if some observed dominant child differs across NCEs.
        dom_sets = list(by_nce.values())
        first = dom_sets[0]
        if any(s != first for s in dom_sets[1:]):
            acc["n_controlled_strata_dom_flip"] += 1

    return acc


def _pct(num: float, den: float) -> str:
    if den == 0:
        return "nan"
    return f"{100.0 * num / den:.2f}%"


def summarize(df: pd.DataFrame, dominance_margin: float) -> None:
    total = _empty_acc()
    for row in df.to_dict("records"):
        _merge_acc(total, row)

    mean_expl = (
        total["sum_alias_free_expl_int_frac"] / total["n_alias_free_expl_int_frac"]
        if total["n_alias_free_expl_int_frac"]
        else float("nan")
    )

    print(f"\n{'=' * 96}")
    print("Clean beta-target coverage audit")
    print(f"{'=' * 96}")
    print(f"molecules with DAG rows: {int(total['n_mol'])}")
    print(f"spectra: {int(total['n_spectra'])}")
    print(f"dominance margin: {dominance_margin:.3f} of explained sibling mass")
    print()
    print("Static DAG parent coverage:")
    print(f"  parents total:                         {int(total['n_parents_total'])}")
    print(
        "  parents with >=2 children:             "
        f"{int(total['n_parents_ge2_children'])} "
        f"({_pct(total['n_parents_ge2_children'], total['n_parents_total'])})"
    )
    print(
        "  parents with >=2 single-parent child:  "
        f"{int(total['n_parents_ge2_single_parent_children'])} "
        f"({_pct(total['n_parents_ge2_single_parent_children'], total['n_parents_total'])})"
    )
    print()
    print("Parent-spectrum target coverage:")
    print(
        "  single-parent eligible:                "
        f"{int(total['n_parent_spectra_single_parent_eligible'])}"
    )
    print(
        "  >=2 observed supervised siblings:      "
        f"{int(total['n_parent_spectra_ge2_observed'])} "
        f"({_pct(total['n_parent_spectra_ge2_observed'], total['n_parent_spectra_single_parent_eligible'])})"
    )
    print(
        "  alias-free / m/z-resolvable:           "
        f"{int(total['n_parent_spectra_alias_free'])} "
        f"({_pct(total['n_parent_spectra_alias_free'], total['n_parent_spectra_single_parent_eligible'])})"
    )
    print(
        "  dominant-sibling usable:               "
        f"{int(total['n_parent_spectra_dominant_usable'])} "
        f"({_pct(total['n_parent_spectra_dominant_usable'], total['n_parent_spectra_single_parent_eligible'])})"
    )
    print(f"  mean explained-intensity fraction:      {mean_expl:.4f}")
    print()
    print("NCE bins:")
    print(f"  {'nce':>8s} {'eligible':>12s} {'alias-free':>12s} {'dominant':>12s} {'dom/elig':>10s}")
    for lab in NCE_LABELS:
        elig = int(total[f"nce_{lab}__eligible"])
        af = int(total[f"nce_{lab}__alias_free"])
        dom = int(total[f"nce_{lab}__dominant_usable"])
        if elig == 0:
            continue
        print(f"  {lab:>8s} {elig:>12d} {af:>12d} {dom:>12d} {_pct(dom, elig):>10s}")
    print()
    print("Controlled CE variation among clean dominant targets:")
    print(f"  strata with >=2 NCE values:             {int(total['n_controlled_strata_multi'])}")
    print(
        "  strata with dominant-child flip:        "
        f"{int(total['n_controlled_strata_dom_flip'])} "
        f"({_pct(total['n_controlled_strata_dom_flip'], total['n_controlled_strata_multi'])})"
    )
    print(f"\n{'-' * 96}")
    print("READ:")
    print("  dominant-sibling usable is the effective sample count for the stop-test target.")
    print("  if alias-free coverage is tiny, the clean target is too sparse before any loss work.")
    print("  if controlled dominant flips are tiny, CE-specific beta supervision has little signal.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--frag_dp", required=True, help="DAG dir used for the target model.")
    parser.add_argument("--spec_fp", default="data/proc/nist20/spec_df.pkl")
    parser.add_argument("--split_dp", default="data/split/nist20_ma_mi_inchikey")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="train")
    parser.add_argument("--limit_mols", type=int, default=None, help="Subsample mols for a quick check.")
    parser.add_argument("--max_h_transfer", type=int, default=None)
    parser.add_argument(
        "--dominance_margin",
        type=float,
        default=0.0,
        help="Optional minimum (top-second)/sum margin for dominant-child labels.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_jobs", type=int, default=data_utils.JOBLIB_N_JOBS)
    parser.add_argument("--out_fp", default=None, help="Optional CSV of per-mol accumulators.")
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
        dominance_margin=args.dominance_margin,
    )
    log.info("Processing %d mols (%d spectra)", len(mol_inputs), len(spec_df))
    per_mol = data_utils.par_apply(mol_inputs, worker, need_unpack=True, n_jobs=args.n_jobs)
    rows = [r for r in per_mol if r is not None]
    if not rows:
        raise ValueError("No molecule produced audit rows; check --frag_dp and --split.")

    df = pd.DataFrame(rows)
    summarize(df, args.dominance_margin)

    if args.out_fp:
        df.to_csv(args.out_fp, index=False)
        log.info("Saved per-mol accumulators to %s", args.out_fp)


if __name__ == "__main__":
    main()
