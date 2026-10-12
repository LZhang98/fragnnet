"""Generate and compare baseline vs MBFS DAGs on a processed dataset.

This is intended for small debug datasets such as ``data/proc/debug_tinyv4``.
It writes two DAG directories plus a JSON report with per-molecule structural
differences.

Example:
    python3 scripts/debug/dag/debug_compare_mbfs_dags.py \
        --proc_dp data/proc/debug_tinyv4 \
        --output_root data/frag/debug_tinyv4_mbfs_debug \
        --max_depth 3 \
        --max_h_transfer 4
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from fragnnet.utils import data_utils
from fragnnet.models.fragnnet import frag_utils
from fragnnet.utils.misc_utils import booltype
from fragnnet.utils.proc_utils import filter_spec_mol


def _normalize_optional_list(values: list[str] | None) -> list[str] | None:
    return values if values else None


def _load_filtered_mol_df(args: argparse.Namespace) -> pd.DataFrame:
    mol_df = pd.read_pickle(Path(args.proc_dp) / "mol_df.pkl")
    spec_df = pd.read_pickle(Path(args.proc_dp) / "spec_df.pkl")

    spec_df, mol_df = filter_spec_mol(
        spec_df,
        mol_df,
        elements=args.elements,
        dsets=_normalize_optional_list(args.dsets),
        prec_types=_normalize_optional_list(args.prec_types),
        num_entries=args.num_entries,
        frag_modes=_normalize_optional_list(args.frag_modes),
        ion_modes=_normalize_optional_list(args.ion_modes),
        inst_types=_normalize_optional_list(args.inst_types),
        spec_type=args.spec_type,
    )

    mol_df = mol_df.drop_duplicates(subset=["mol_id"]).reset_index(drop=True)
    return mol_df


def _get_cc_mask_rows(dag_d: dict[str, Any]) -> list[tuple[int, ...]]:
    dag = dag_d["dag"]
    node_feat_idxs = dag.node_feat_idxs[0].tolist()
    cc_long = frag_utils.get_node_feats(dag.x, node_feat_idxs, "cc")
    cc_mask = frag_utils.th_long_to_mask(cc_long).cpu().numpy()
    node_count = int(cc_mask.shape[0])
    mol_atoms = int(cc_mask[0].sum()) if node_count else 0
    return [tuple(int(v) for v in row[:mol_atoms]) for row in cc_mask]


def _get_edge_mask_rows(
    dag_d: dict[str, Any], node_masks: list[tuple[int, ...]]
) -> list[tuple[Any, ...]]:
    edge_pairs = dag_d["dag"].edge_index.t().cpu().tolist()
    nodes_min_depth = dag_d["nodes_min_depth"].tolist()
    edges_min_depth = dag_d["edges_min_depth"].tolist()
    edge_rows = []
    for idx, (src, dst) in enumerate(edge_pairs):
        edge_rows.append(
            (
                node_masks[src],
                node_masks[dst],
                int(nodes_min_depth[src]),
                int(nodes_min_depth[dst]),
                int(edges_min_depth[idx]),
            )
        )
    edge_rows.sort()
    return edge_rows


def summarize_dag(dag_d: dict[str, Any]) -> dict[str, Any]:
    node_masks = _get_cc_mask_rows(dag_d)
    node_set = sorted(node_masks)
    node_depth_pairs = sorted(
        (node_masks[idx], int(depth)) for idx, depth in enumerate(dag_d["nodes_min_depth"].tolist())
    )
    edge_rows = _get_edge_mask_rows(dag_d, node_masks)
    formulas = sorted(set(dag_d["idx_to_formula"].values()))
    non_empty_formulas = sorted(f for f in formulas if f)

    return {
        "num_nodes": int(dag_d["dag_num_nodes"]),
        "num_edges": int(dag_d["dag_num_edges"]),
        "reached_depth": int(dag_d["reached_depth"]),
        "force_stopped": bool(dag_d["force_stopped"]),
        "node_masks": node_set,
        "node_depth_pairs": node_depth_pairs,
        "edge_rows": edge_rows,
        "formula_set": formulas,
        "non_empty_formula_set": non_empty_formulas,
        "nodes_by_depth": {str(k): int(v) for k, v in dag_d["dag_num_nodes_by_depth"].items()},
        "edges_by_depth": {str(k): int(v) for k, v in dag_d["dag_num_edges_by_depth"].items()},
    }


def _sorted_set_diff(lhs: Iterable[Any], rhs: Iterable[Any]) -> list[Any]:
    return sorted(set(lhs) - set(rhs))


def compare_dag_summaries(
    baseline_summary: dict[str, Any], mbfs_summary: dict[str, Any]
) -> dict[str, Any]:
    baseline_nodes = baseline_summary["node_masks"]
    mbfs_nodes = mbfs_summary["node_masks"]
    baseline_edges = baseline_summary["edge_rows"]
    mbfs_edges = mbfs_summary["edge_rows"]
    baseline_formulae = baseline_summary["non_empty_formula_set"]
    mbfs_formulae = mbfs_summary["non_empty_formula_set"]

    return {
        "node_count_delta": mbfs_summary["num_nodes"] - baseline_summary["num_nodes"],
        "edge_count_delta": mbfs_summary["num_edges"] - baseline_summary["num_edges"],
        "reached_depth_delta": mbfs_summary["reached_depth"] - baseline_summary["reached_depth"],
        "force_stopped_changed": mbfs_summary["force_stopped"] != baseline_summary["force_stopped"],
        "new_nodes_in_mbfs": _sorted_set_diff(mbfs_nodes, baseline_nodes),
        "missing_nodes_in_mbfs": _sorted_set_diff(baseline_nodes, mbfs_nodes),
        "new_node_depth_pairs_in_mbfs": _sorted_set_diff(
            mbfs_summary["node_depth_pairs"], baseline_summary["node_depth_pairs"]
        ),
        "missing_node_depth_pairs_in_mbfs": _sorted_set_diff(
            baseline_summary["node_depth_pairs"], mbfs_summary["node_depth_pairs"]
        ),
        "new_edges_in_mbfs": _sorted_set_diff(mbfs_edges, baseline_edges),
        "missing_edges_in_mbfs": _sorted_set_diff(baseline_edges, mbfs_edges),
        "new_formulas_in_mbfs": _sorted_set_diff(mbfs_formulae, baseline_formulae),
        "missing_formulas_in_mbfs": _sorted_set_diff(baseline_formulae, mbfs_formulae),
        "baseline_nodes_by_depth": baseline_summary["nodes_by_depth"],
        "mbfs_nodes_by_depth": mbfs_summary["nodes_by_depth"],
        "baseline_edges_by_depth": baseline_summary["edges_by_depth"],
        "mbfs_edges_by_depth": mbfs_summary["edges_by_depth"],
    }


def _clip_items(items: list[Any], limit: int) -> list[Any]:
    return items[:limit]


def generate_dags(
    mol_df: pd.DataFrame,
    output_dp: Path,
    *,
    multi_cut_bfs: bool,
    max_depth: int,
    max_h_transfer: int,
    max_time: int,
    isotopes: bool,
    nb_isomorphic: bool,
    wl_max_iterations: int,
    max_cut_size: int,
    smarts_prepass: bool,
    min_frag_atoms: int,
    compress_dags: bool,
    compress_format: str,
) -> dict[int, dict[str, Any]]:
    dag_dp = output_dp / "dags"
    dag_dp.mkdir(parents=True, exist_ok=True)

    results: dict[int, dict[str, Any]] = {}
    mol_input_rows = (
        [
            row["mol"],
            int(row["mol_id"]),
            max_depth,
            True,
            max_h_transfer,
            max_time,
            isotopes,
            nb_isomorphic,
            wl_max_iterations,
            multi_cut_bfs,
            max_cut_size,
            smarts_prepass,
            min_frag_atoms,
        ]
        for _, row in mol_df.iterrows()
    )
    frag_results = data_utils.par_apply(
        mol_input_rows,
        frag_utils.timed_get_dags,
        need_unpack=True,
        return_as_generator=False,
    )

    for mol_id, dag_d in frag_results:
        mol_id = int(mol_id)
        results[mol_id] = dag_d
        if dag_d:
            fp = frag_utils.get_dag_output_path(
                str(dag_dp),
                mol_id,
                compress_dags=compress_dags,
                compress_format=compress_format,
            )
            frag_utils.dump_dag_pickle(fp, dag_d)
    return results


def build_report(
    mol_df: pd.DataFrame,
    baseline_dags: dict[int, dict[str, Any]],
    mbfs_dags: dict[int, dict[str, Any]],
    detail_limit: int,
) -> dict[str, Any]:
    per_molecule = []
    changed_molecules = 0
    baseline_failures = 0
    mbfs_failures = 0

    for _, row in mol_df.iterrows():
        mol_id = int(row["mol_id"])
        baseline_dag = baseline_dags.get(mol_id, {})
        mbfs_dag = mbfs_dags.get(mol_id, {})
        baseline_ok = bool(baseline_dag)
        mbfs_ok = bool(mbfs_dag)

        if not baseline_ok:
            baseline_failures += 1
        if not mbfs_ok:
            mbfs_failures += 1

        item: dict[str, Any] = {
            "mol_id": mol_id,
            "smiles": row.get("smiles"),
            "baseline_ok": baseline_ok,
            "mbfs_ok": mbfs_ok,
        }

        if baseline_ok and mbfs_ok:
            baseline_summary = summarize_dag(baseline_dag)
            mbfs_summary = summarize_dag(mbfs_dag)
            diff = compare_dag_summaries(baseline_summary, mbfs_summary)
            changed = any(
                [
                    diff["node_count_delta"] != 0,
                    diff["edge_count_delta"] != 0,
                    diff["reached_depth_delta"] != 0,
                    diff["force_stopped_changed"],
                    bool(diff["new_nodes_in_mbfs"]),
                    bool(diff["missing_nodes_in_mbfs"]),
                    bool(diff["new_edges_in_mbfs"]),
                    bool(diff["missing_edges_in_mbfs"]),
                    bool(diff["new_formulas_in_mbfs"]),
                    bool(diff["missing_formulas_in_mbfs"]),
                ]
            )
            if changed:
                changed_molecules += 1

            item["changed"] = changed
            item["baseline_summary"] = {
                k: v
                for k, v in baseline_summary.items()
                if k
                not in {"node_masks", "node_depth_pairs", "edge_rows", "formula_set", "non_empty_formula_set"}
            }
            item["mbfs_summary"] = {
                k: v
                for k, v in mbfs_summary.items()
                if k
                not in {"node_masks", "node_depth_pairs", "edge_rows", "formula_set", "non_empty_formula_set"}
            }
            item["diff"] = {
                **diff,
                "new_nodes_in_mbfs": _clip_items(diff["new_nodes_in_mbfs"], detail_limit),
                "missing_nodes_in_mbfs": _clip_items(diff["missing_nodes_in_mbfs"], detail_limit),
                "new_node_depth_pairs_in_mbfs": _clip_items(
                    diff["new_node_depth_pairs_in_mbfs"], detail_limit
                ),
                "missing_node_depth_pairs_in_mbfs": _clip_items(
                    diff["missing_node_depth_pairs_in_mbfs"], detail_limit
                ),
                "new_edges_in_mbfs": _clip_items(diff["new_edges_in_mbfs"], detail_limit),
                "missing_edges_in_mbfs": _clip_items(diff["missing_edges_in_mbfs"], detail_limit),
                "new_formulas_in_mbfs": _clip_items(diff["new_formulas_in_mbfs"], detail_limit),
                "missing_formulas_in_mbfs": _clip_items(
                    diff["missing_formulas_in_mbfs"], detail_limit
                ),
            }
            item["detail_truncation_limit"] = detail_limit
        per_molecule.append(item)

    return {
        "summary": {
            "num_molecules": int(len(mol_df)),
            "changed_molecules": int(changed_molecules),
            "baseline_failures": int(baseline_failures),
            "mbfs_failures": int(mbfs_failures),
        },
        "molecules": per_molecule,
    }


def print_report(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print(
        "Summary:"
        f" molecules={summary['num_molecules']},"
        f" changed={summary['changed_molecules']},"
        f" baseline_failures={summary['baseline_failures']},"
        f" mbfs_failures={summary['mbfs_failures']}"
    )

    changed = [m for m in report["molecules"] if m.get("changed")]
    failed = [m for m in report["molecules"] if not m["baseline_ok"] or not m["mbfs_ok"]]

    if failed:
        print("\nFailures:")
        for item in failed:
            print(
                f"  mol_id={item['mol_id']} smiles={item['smiles']}"
                f" baseline_ok={item['baseline_ok']} mbfs_ok={item['mbfs_ok']}"
            )

    if changed:
        print("\nChanged molecules:")
        for item in changed:
            diff = item["diff"]
            print(
                f"  mol_id={item['mol_id']} smiles={item['smiles']}"
                f" node_delta={diff['node_count_delta']}"
                f" edge_delta={diff['edge_count_delta']}"
                f" new_nodes={len(diff['new_nodes_in_mbfs'])}"
                f" missing_nodes={len(diff['missing_nodes_in_mbfs'])}"
                f" new_edges={len(diff['new_edges_in_mbfs'])}"
                f" missing_edges={len(diff['missing_edges_in_mbfs'])}"
                f" new_formulas={len(diff['new_formulas_in_mbfs'])}"
                f" missing_formulas={len(diff['missing_formulas_in_mbfs'])}"
            )
    else:
        print("\nNo structural differences detected between baseline and MBFS DAGs.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate baseline and MBFS DAGs and compare them molecule-by-molecule."
    )
    parser.add_argument("--proc_dp", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--num_entries", type=int, default=-1)
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--max_h_transfer", type=int, default=4)
    parser.add_argument("--max_time", type=int, default=120)
    parser.add_argument("--isotopes", type=booltype, default=False)
    parser.add_argument("--nb_isomorphic", type=booltype, default=False)
    parser.add_argument("--wl_max_iterations", type=int, default=3)
    parser.add_argument("--max_cut_size", type=int, default=2, choices=[1, 2])
    parser.add_argument("--smarts_prepass", type=booltype, default=False)
    parser.add_argument("--min_frag_atoms", type=int, default=0)
    parser.add_argument("--compress_dags", type=booltype, default=False)
    parser.add_argument("--compress_format", type=str, default="bz2", choices=["gz", "bz2"])
    parser.add_argument("--detail_limit", type=int, default=20)
    parser.add_argument("--elements", nargs="+", default=frag_utils.ELEMENTS)
    parser.add_argument("--dsets", nargs="+", default=None)
    parser.add_argument("--prec_types", nargs="+", default=None)
    parser.add_argument("--inst_types", nargs="+", default=None)
    parser.add_argument("--frag_modes", nargs="+", default=None)
    parser.add_argument("--ion_modes", nargs="+", default=None)
    parser.add_argument("--spec_type", type=str, default="MS2")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    mol_df = _load_filtered_mol_df(args)
    baseline_dp = output_root / "baseline"
    mbfs_dp = output_root / "mbfs"

    print(f"Loaded {len(mol_df)} molecules from {args.proc_dp}")
    print(f"Writing baseline DAGs to {baseline_dp}")
    baseline_dags = generate_dags(
        mol_df,
        baseline_dp,
        multi_cut_bfs=False,
        max_depth=args.max_depth,
        max_h_transfer=args.max_h_transfer,
        max_time=args.max_time,
        isotopes=args.isotopes,
        nb_isomorphic=args.nb_isomorphic,
        wl_max_iterations=args.wl_max_iterations,
        max_cut_size=args.max_cut_size,
        smarts_prepass=args.smarts_prepass,
        min_frag_atoms=args.min_frag_atoms,
        compress_dags=args.compress_dags,
        compress_format=args.compress_format,
    )

    print(f"Writing MBFS DAGs to {mbfs_dp}")
    mbfs_dags = generate_dags(
        mol_df,
        mbfs_dp,
        multi_cut_bfs=True,
        max_depth=args.max_depth,
        max_h_transfer=args.max_h_transfer,
        max_time=args.max_time,
        isotopes=args.isotopes,
        nb_isomorphic=args.nb_isomorphic,
        wl_max_iterations=args.wl_max_iterations,
        max_cut_size=args.max_cut_size,
        smarts_prepass=args.smarts_prepass,
        min_frag_atoms=args.min_frag_atoms,
        compress_dags=args.compress_dags,
        compress_format=args.compress_format,
    )

    report = build_report(mol_df, baseline_dags, mbfs_dags, detail_limit=args.detail_limit)
    report["config"] = vars(args)

    report_fp = output_root / "comparison_report.json"
    with open(report_fp, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print_report(report)
    print(f"\nReport written to {report_fp}")


if __name__ == "__main__":
    main()
