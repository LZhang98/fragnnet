"""Compare two fragmentation DAG tar files at the formula level.

Investigates whether one DAG set (CMF) contains the same fragments as another
(smpFalse), aside from very small fragments below a heavy-atom cutoff.

Key findings (2026-04-16):
- CMF has ~35 extra *small* nodes per molecule on average (≤2 heavy atoms) — expected.
- CMF appears to have 6,461 "large" extra formulas across 1,343 molecules. These are NOT
  extra fragment types: they are the same connected components (CCs) as in smpFalse, but
  smpFalse incorrectly assigns "" (invalid) at delta_h=0 for nitro-containing sub-fragments
  due to H-floor overcounting in compute_cc_h_floor.  The fix ``floor = min(floor,
  num_hs_prior)`` in compute_approximate_formula eliminates these differences.
- Two molecules (11588, 8886) in smpFalse have 2602 / 850 unique formulas not in CMF;
  these are very large molecules (~80K nodes) and likely reflect BFS ordering / force-stop
  boundary differences — unrelated to the H-floor bug.

Usage::

    conda activate FRAGNNET-GPU
    python scripts/debug/debug_compare_dag_sets.py \\
        --cmf  data/frag/nist20_ma_mi_d3_h4_isoFalse_smpFalse.tar \\
        --smp  data/frag/nist20_ma_mi_d3_h4_isoFalse_smpFalse.tar \\
        [--min-heavy 3] [--max-mols 0] [--out /tmp/dag_compare.csv]
"""

from __future__ import annotations

import argparse
import bz2
import pickle
import re
import tarfile
from pathlib import Path

import pandas as pd

from fragnnet.utils.frag_utils import NODE_FEAT_TO_IDX

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ELEM_RE = re.compile(r"([A-Z][a-z]?)(\d*)")


def heavy_atom_count(formula: str) -> int:
    """Return the number of non-hydrogen atoms in a formula string."""
    if not formula:
        return 0
    return sum((int(n) if n else 1) for elem, n in _ELEM_RE.findall(formula) if elem != "H")


def load_dag(tf: tarfile.TarFile, name: str) -> dict:
    return pickle.loads(bz2.decompress(tf.extractfile(name).read()))


def index_tar(path: str, ext: str) -> dict[int, str]:
    """Return {mol_id: member_name} for all members ending with *ext*."""
    idx: dict[int, str] = {}
    with tarfile.open(path) as tf:
        for m in tf.getmembers():
            if m.name.endswith(ext):
                match = re.search(r"(\d+)\.", m.name)
                if match:
                    idx[int(match.group(1))] = m.name
    return idx


def get_slot0_formulas(dag_d: dict) -> dict[tuple, str]:
    """Return {cc_bitmask_key: formula_at_delta_h=0} for each node."""
    dag = dag_d["dag"]
    idx2f = dag_d["idx_to_formula"]
    feat_idxs = dag.node_feat_idxs[0].tolist()
    cc_start = feat_idxs[NODE_FEAT_TO_IDX["cc"]]
    cc_end = feat_idxs[NODE_FEAT_TO_IDX["cc"] + 1]
    h_start = feat_idxs[NODE_FEAT_TO_IDX["h_formulae_idx"]]
    result: dict[tuple, str] = {}
    for i in range(dag.x.shape[0]):
        cc_key = tuple(dag.x[i, cc_start:cc_end].tolist())
        fidx = int(dag.x[i, h_start].item())
        result[cc_key] = idx2f.get(fidx, "")
    return result


# ---------------------------------------------------------------------------
# Per-molecule analysis
# ---------------------------------------------------------------------------


def compare_molecule(
    cmf_dag: dict,
    smp_dag: dict,
    min_heavy: int,
) -> dict:
    """Return a comparison record for one molecule."""
    cmf_cc = get_slot0_formulas(cmf_dag)
    smp_cc = get_slot0_formulas(smp_dag)

    cmf_ccs = set(cmf_cc)
    smp_ccs = set(smp_cc)

    # Connected-component level differences
    cmf_only_ccs = cmf_ccs - smp_ccs
    smp_only_ccs = smp_ccs - cmf_ccs
    shared_ccs = cmf_ccs & smp_ccs

    # Extra CCs in CMF — split by size
    cmf_extra_small = sum(1 for cc in cmf_only_ccs if heavy_atom_count(cmf_cc[cc]) < min_heavy)
    cmf_extra_large = sum(1 for cc in cmf_only_ccs if heavy_atom_count(cmf_cc[cc]) >= min_heavy)

    # Formula-level: shared CCs where slot-0 formula differs
    shared_cmf_only_large = 0  # CMF has real formula, SMP has ""
    shared_smp_only_large = 0  # SMP has real formula, CMF has ""
    for cc in shared_ccs:
        cf = cmf_cc[cc]
        sf = smp_cc[cc]
        if heavy_atom_count(cf) >= min_heavy and sf == "":
            shared_cmf_only_large += 1
        elif heavy_atom_count(sf) >= min_heavy and cf == "":
            shared_smp_only_large += 1

    # SMP-only CCs — split by size
    smp_only_large = sum(1 for cc in smp_only_ccs if heavy_atom_count(smp_cc[cc]) >= min_heavy)
    smp_only_small = len(smp_only_ccs) - smp_only_large

    # Unique slot-0 formulas (excluding "" and small)
    cmf_large_forms = {f for f in cmf_cc.values() if heavy_atom_count(f) >= min_heavy}
    smp_large_forms = {f for f in smp_cc.values() if heavy_atom_count(f) >= min_heavy}

    return {
        "cmf_nodes": cmf_dag["dag_num_nodes"],
        "smp_nodes": smp_dag["dag_num_nodes"],
        "cmf_edges": cmf_dag["dag_num_edges"],
        "smp_edges": smp_dag["dag_num_edges"],
        "shared_ccs": len(shared_ccs),
        "cmf_extra_ccs_small": cmf_extra_small,
        "cmf_extra_ccs_large": cmf_extra_large,
        "smp_extra_ccs_small": smp_only_small,
        "smp_extra_ccs_large": smp_only_large,
        # Formula-level differences on shared CCs (H-floor bug indicator)
        "shared_cmf_slot0_large_smp_empty": shared_cmf_only_large,
        "shared_smp_slot0_large_cmf_empty": shared_smp_only_large,
        # Net unique-formula counts
        "n_cmf_large_forms": len(cmf_large_forms),
        "n_smp_large_forms": len(smp_large_forms),
        "cmf_only_large_forms": len(cmf_large_forms - smp_large_forms),
        "smp_only_large_forms": len(smp_large_forms - cmf_large_forms),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cmf", required=True, help="Path to CMF tar file")
    parser.add_argument("--smp", required=True, help="Path to smpFalse tar file")
    parser.add_argument("--min-heavy", type=int, default=3, help="Heavy-atom cutoff (default 3)")
    parser.add_argument(
        "--max-mols",
        type=int,
        default=0,
        help="Limit to first N molecules (0 = all)",
    )
    parser.add_argument("--out", default="/tmp/dag_compare.csv", help="Output CSV path")
    args = parser.parse_args()

    print(f"Indexing {args.cmf} ...", flush=True)
    cmf_idx = index_tar(args.cmf, ".pickle.bz2")
    print(f"Indexing {args.smp} ...", flush=True)
    smp_idx = index_tar(args.smp, ".pkl.bz2")

    common = sorted(set(cmf_idx) & set(smp_idx))
    if args.max_mols > 0:
        common = common[: args.max_mols]
    print(
        f"CMF: {len(cmf_idx)} mols | SMP: {len(smp_idx)} mols | "
        f"common: {len(common)} | min_heavy={args.min_heavy}",
        flush=True,
    )

    rows = []
    with tarfile.open(args.cmf) as cmf_tf, tarfile.open(args.smp) as smp_tf:
        for i, mol_id in enumerate(common):
            if i % 2000 == 0:
                print(f"  [{i}/{len(common)}]", flush=True)
            cmf_dag = load_dag(cmf_tf, cmf_idx[mol_id])
            smp_dag = load_dag(smp_tf, smp_idx[mol_id])
            rec = compare_molecule(cmf_dag, smp_dag, args.min_heavy)
            rec["mol_id"] = mol_id
            rows.append(rec)

    df = pd.DataFrame(rows).set_index("mol_id")

    # -----------------------------------------------------------------------
    # Summary report
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Molecules compared : {len(df)}")
    print(
        f"Node diff CMF-SMP  : mean={df['cmf_nodes'].sub(df['smp_nodes']).mean():.1f}, "
        f"median={df['cmf_nodes'].sub(df['smp_nodes']).median():.1f}"
    )

    print("\n--- CMF has extra CCs not in SMP ---")
    print(
        f"  small (<{args.min_heavy} heavy): total={df['cmf_extra_ccs_small'].sum():,}, "
        f"mean/mol={df['cmf_extra_ccs_small'].mean():.2f}"
    )
    print(
        f"  large (>={args.min_heavy} heavy): total={df['cmf_extra_ccs_large'].sum():,}, "
        f"mols affected={(df['cmf_extra_ccs_large'] > 0).sum()}"
    )

    print("\n--- SMP has extra CCs not in CMF ---")
    print(f"  small (<{args.min_heavy} heavy): total={df['smp_extra_ccs_small'].sum():,}")
    print(
        f"  large (>={args.min_heavy} heavy): total={df['smp_extra_ccs_large'].sum():,}, "
        f"mols affected={(df['smp_extra_ccs_large'] > 0).sum()}"
    )

    print("\n--- Shared CCs with formula assignment difference (H-floor bug) ---")
    n_hfloor = df["shared_cmf_slot0_large_smp_empty"].sum()
    n_mols_hfloor = (df["shared_cmf_slot0_large_smp_empty"] > 0).sum()
    print(
        f"  CMF slot-0 = large formula, SMP slot-0 = '' : total={n_hfloor:,}, mols={n_mols_hfloor}"
    )
    n_rev = df["shared_smp_slot0_large_cmf_empty"].sum()
    print(f"  SMP slot-0 = large formula, CMF slot-0 = '' : total={n_rev:,}")

    print("\n--- Net unique-formula counts (large frags only) ---")
    print(
        f"  CMF-only large formulas: {df['cmf_only_large_forms'].sum():,} "
        f"(mols: {(df['cmf_only_large_forms'] > 0).sum()})"
    )
    print(
        f"  SMP-only large formulas: {df['smp_only_large_forms'].sum():,} "
        f"(mols: {(df['smp_only_large_forms'] > 0).sum()})"
    )

    print("\n--- Top-10 molecules by SMP-only large formulas ---")
    top_smp = df.nlargest(10, "smp_only_large_forms")[
        ["cmf_nodes", "smp_nodes", "cmf_only_large_forms", "smp_only_large_forms"]
    ]
    print(top_smp.to_string())

    print("\n--- Top-10 molecules by CMF-only large formulas ---")
    top_cmf = df.nlargest(10, "cmf_only_large_forms")[
        ["cmf_nodes", "smp_nodes", "cmf_only_large_forms", "shared_cmf_slot0_large_smp_empty"]
    ]
    print(top_cmf.to_string())

    out_path = Path(args.out)
    df.to_csv(out_path)
    print(f"\nFull per-molecule results saved to {out_path}")


if __name__ == "__main__":
    main()
