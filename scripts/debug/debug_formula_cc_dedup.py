"""Compare DAG stats with and without formula+CC-cut deduplication.

Runs compute_dags on a set of representative molecules and prints node/edge counts,
formula_redundancy, and nodes-by-depth for both the baseline (no dedup) and the new
dedup path. Use D3 first (fast); bump --max_depth 4 to check D4 impact.

Usage:
    python scripts/debug/debug_formula_cc_dedup.py
    python scripts/debug/debug_formula_cc_dedup.py --max_depth 4
    python scripts/debug/debug_formula_cc_dedup.py --max_depth 4 --smiles "CC(=O)OC1=CC=CC=C1C(=O)O"
"""

import argparse
import sys
import time

sys.path.insert(0, "src")
import fragnnet.utils.frag_utils as frag_utils
from fragnnet.frag import compute_frags

MOLECULES = {
    "ethanol": "CCO",
    "aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "methionine": "CSCCC(C(=O)O)N",
    "naphthalene": "c1ccc2ccccc2c1",
    "testosterone": "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",
    "ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "reserpine (large)": "COC(=O)[C@@H]1[C@@H]2CC[C@H](OC)[C@@]2(OC)C[C@@H]1OC(=O)c3cc4cc(OC)c(OC)cc4[nH]3",
}


def get_dag_stats(dag_d: dict) -> dict:
    return {
        "nodes": dag_d.get("dag_num_nodes", 0),
        "edges": dag_d.get("dag_num_edges", 0),
        "formula_redundancy": dag_d.get("formula_redundancy", float("nan")),
        "num_formulae": dag_d.get("num_formulae", 0),
        "by_depth": dag_d.get("dag_num_edges_by_depth", {}),
        "force_stopped": dag_d.get("force_stopped", False),
    }


def run_both(smiles: str, max_depth: int, max_h: int, max_time: int) -> tuple[dict, dict, float, float]:
    """Run baseline (no dedup) and dedup compute_dags for one molecule.

    The dedup is controlled by whether is_cc_bond/elem_idxs_c are passed to
    compute_ccs. We toggle this by temporarily monkey-patching frag_utils to
    skip precomputing those arrays for the baseline run.
    """
    mol_d = frag_utils.extract_mol_info(smiles)

    # Baseline: pass is_cc_bond=None by temporarily replacing compute_ccs
    # with a wrapper that drops the new args.
    _orig_compute_ccs = compute_frags.compute_ccs

    def _compute_ccs_no_dedup(n_nodes, n_edges, nm, edges, em, nte, md, tl, mfa,
                               is_cc_bond=None, elem_idxs_c=None):
        return _orig_compute_ccs(n_nodes, n_edges, nm, edges, em, nte, md, tl, mfa)

    compute_frags.compute_ccs = _compute_ccs_no_dedup
    t0 = time.perf_counter()
    _, dag_base = frag_utils.timed_get_dags(
        mol_d["mol"], 0,
        max_depth=max_depth, h_prior=True, max_h_transfer=max_h,
        max_time=max_time, isotopes=False, nb_isomorphic=False,
        wl_max_iterations=-1, multi_cut_bfs=False,
    )
    t_base = time.perf_counter() - t0
    compute_frags.compute_ccs = _orig_compute_ccs

    # Dedup: normal path (is_cc_bond + elem_idxs_c passed automatically)
    t0 = time.perf_counter()
    _, dag_dedup = frag_utils.timed_get_dags(
        mol_d["mol"], 0,
        max_depth=max_depth, h_prior=True, max_h_transfer=max_h,
        max_time=max_time, isotopes=False, nb_isomorphic=False,
        wl_max_iterations=-1, multi_cut_bfs=False,
    )
    t_dedup = time.perf_counter() - t0

    return get_dag_stats(dag_base), get_dag_stats(dag_dedup), t_base, t_dedup


def fmt_ratio(a: float, b: float) -> str:
    if b == 0:
        return "N/A"
    return f"{b/a:.3f}x"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max_depth", type=int, default=3, help="BFS max depth (default 3)")
    parser.add_argument("--max_h", type=int, default=4, help="Max H transfer (default 4)")
    parser.add_argument("--max_time", type=int, default=60, help="Per-molecule timeout seconds")
    parser.add_argument("--smiles", type=str, default=None, help="Single SMILES to test")
    args = parser.parse_args()

    mols = {args.smiles: args.smiles} if args.smiles else MOLECULES

    print(f"Formula+CC-cut dedup comparison  (depth={args.max_depth}, h={args.max_h})")
    print(f"{'='*100}")

    totals = {"nodes_base": 0, "nodes_dedup": 0, "edges_base": 0, "edges_dedup": 0}

    for name, smiles in mols.items():
        print(f"\n{name}  [{smiles}]")
        try:
            base, dedup, t_base, t_dedup = run_both(smiles, args.max_depth, args.max_h, args.max_time)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        if base["force_stopped"]:
            print("  [TIMEOUT - baseline]")
            continue

        node_ratio = dedup["nodes"] / base["nodes"] if base["nodes"] else float("nan")
        edge_ratio = dedup["edges"] / base["edges"] if base["edges"] else float("nan")
        form_saved = base["num_formulae"] - dedup["num_formulae"]

        print(f"  {'':30s} {'baseline':>12}  {'dedup':>12}  {'ratio':>8}")
        print(f"  {'nodes':<30} {base['nodes']:>12,}  {dedup['nodes']:>12,}  {node_ratio:>7.3f}x")
        print(f"  {'edges':<30} {base['edges']:>12,}  {dedup['edges']:>12,}  {edge_ratio:>7.3f}x")
        print(f"  {'formula_redundancy':<30} {base['formula_redundancy']:>12.3f}  {dedup['formula_redundancy']:>12.3f}")
        print(f"  {'num_formulae':<30} {base['num_formulae']:>12,}  {dedup['num_formulae']:>12,}  (saved {form_saved:,})")
        print(f"  {'time (s)':<30} {t_base:>12.3f}  {t_dedup:>12.3f}")

        # Edges by depth
        base_bd = base["by_depth"]
        dedup_bd = dedup["by_depth"]
        all_depths = sorted(set(base_bd) | set(dedup_bd))
        depth_line = "  edges by depth: "
        for d in all_depths:
            b_e = base_bd.get(d, 0)
            d_e = dedup_bd.get(d, 0)
            depth_line += f"d{d}: {b_e}→{d_e}  "
        print(depth_line)

        totals["nodes_base"] += base["nodes"]
        totals["nodes_dedup"] += dedup["nodes"]
        totals["edges_base"] += base["edges"]
        totals["edges_dedup"] += dedup["edges"]

    if len(mols) > 1:
        print(f"\n{'='*100}")
        print("Totals across all molecules:")
        print(f"  nodes: {totals['nodes_base']:,} → {totals['nodes_dedup']:,}  "
              f"({100*(1 - totals['nodes_dedup']/totals['nodes_base']):.1f}% reduction)")
        print(f"  edges: {totals['edges_base']:,} → {totals['edges_dedup']:,}  "
              f"({100*(1 - totals['edges_dedup']/totals['edges_base']):.1f}% reduction)")


if __name__ == "__main__":
    main()
