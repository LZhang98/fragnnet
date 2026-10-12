"""Mine matched-formula recall by a finer bond descriptor (ring membership, element-pair
type, alpha-heteroatom activation) instead of the coarse binary C-C/heteroatom split used
in debug_dag_boundary_bond_stats.py.

For each DAG edge, the actual broken bond(s) are reconstructed from the parent/child atom
bitmasks (node "cc" feature) against the source RDKit mol (data/proc/<dset>/mol_df.pkl),
then classified into a (in_ring, element_pair, alpha_hetero) bucket. This lets us test
"mask bucket X at depth d" policies directly against real recall loss, globally aggregated
(no CE split), per docs/plans/fragnnet/bond_breakability_mask_plan.md step 3.

Usage:
    python scripts/debug/dag/debug_dag_bond_descriptor_stats.py \
        --tar data/frag/nist20_ma_mi_d4_h4_isoFalse_smpFalse.tar \
        --mol_df data/proc/nist20/mol_df.pkl \
        --n_sample 3000 \
        --seed 42
"""

import argparse
import bz2
import os
import pickle
import random
import sys
import tarfile
from collections import defaultdict

import pandas as pd
import torch

sys.path.insert(0, "src")
from fragnnet.utils.formula_utils import th_long_to_mask


def decode_cc(cc_row: torch.Tensor, num_atoms: int) -> frozenset:
    """Decode a packed 128-bit atom bitmask (2 int64 words) into a set of atom indices."""
    mask = th_long_to_mask(cc_row.reshape(1, -1).long())[0]
    return frozenset(torch.nonzero(mask[:num_atoms]).flatten().tolist())


def bond_bucket(mol, a1: int, a2: int) -> tuple:
    """Classify a broken bond into (in_ring, element_pair, alpha_hetero)."""
    bond = mol.GetBondBetweenAtoms(a1, a2)
    in_ring = bond.IsInRing()
    atom1, atom2 = mol.GetAtomWithIdx(a1), mol.GetAtomWithIdx(a2)
    is_hetero1 = atom1.GetSymbol() not in ("C", "H")
    is_hetero2 = atom2.GetSymbol() not in ("C", "H")
    if is_hetero1 and is_hetero2:
        pair = "XX"
    elif is_hetero1 or is_hetero2:
        pair = "CX"
    else:
        pair = "CC"

    alpha_hetero = False
    for atom, other_idx in ((atom1, a2), (atom2, a1)):
        if atom.GetSymbol() != "C":
            continue
        for nbr in atom.GetNeighbors():
            if nbr.GetIdx() != other_idx and nbr.GetSymbol() not in ("C", "H"):
                alpha_hetero = True
                break
        if alpha_hetero:
            break

    return (in_ring, pair, alpha_hetero)


def boundary_buckets(mol, child_atoms: frozenset, complement_atoms: frozenset) -> set:
    """Find all mol bonds between child_atoms and complement_atoms; return their buckets."""
    buckets = set()
    for a in child_atoms:
        atom = mol.GetAtomWithIdx(a)
        for nbr in atom.GetNeighbors():
            b = nbr.GetIdx()
            if b in complement_atoms:
                buckets.add(bond_bucket(mol, a, b))
    return buckets


def analyze_dag(dag: dict, mol) -> dict[int, list[set]] | None:
    """For each matched formula, find its min depth and, for every distinct route that
    reaches it at that depth (a route = one parent edge; there may be several nodes/edges
    tied at min depth, e.g. symmetric fragments or multiple parents), the set of bond
    descriptor buckets involved in that route's cut (a route's cut can itself break more
    than one bond, e.g. ring-release multi-cuts).

    Returns:
        dict mapping depth -> list of routes, where each route is a set of buckets.
        A formula is "lost" under masking bucket-set S at its depth iff every one of its
        routes contains at least one bucket in S (since masking blocks a route entirely
        if any one of its cut bonds is masked).
    """
    g = dag["dag"]
    x = g.x
    edge_index = g.edge_index
    node_feat_idxs = g.node_feat_idxs[0]
    n_nodes = x.shape[0]
    num_atoms = mol.GetNumAtoms()

    cc_start, cc_end = node_feat_idxs[1].item(), node_feat_idxs[2].item()
    h_start, h_end = node_feat_idxs[3].item(), node_feat_idxs[4].item()

    node_cc = [decode_cc(x[ni, cc_start:cc_end], num_atoms) for ni in range(n_nodes)]
    h_formulae_idx = x[:, h_start:h_end]

    nmd = dag["nodes_min_depth"]
    fpp = dag["formula_peak_probs"]

    def node_depth(ni: int) -> int:
        v = nmd[ni]
        return v.item() if hasattr(v, "item") else int(v)

    formula_to_min_depth: dict[int, int] = {}
    for ni in range(n_nodes):
        d = node_depth(ni)
        for h_idx_val in h_formulae_idx[ni]:
            fidx = h_idx_val.item()
            if fidx <= 0 or fidx >= fpp.shape[0]:
                continue
            if fpp[fidx].sum() == 0:
                continue
            if fidx not in formula_to_min_depth or d < formula_to_min_depth[fidx]:
                formula_to_min_depth[fidx] = d

    if not formula_to_min_depth:
        return None

    formula_to_nodes: dict[int, list[int]] = defaultdict(list)
    for ni in range(n_nodes):
        for h_idx_val in h_formulae_idx[ni]:
            fidx = h_idx_val.item()
            if fidx > 0:
                formula_to_nodes[fidx].append(ni)

    parent_of: dict[int, list[int]] = defaultdict(list)
    for ei in range(edge_index.shape[1]):
        parent_of[edge_index[1, ei].item()].append(edge_index[0, ei].item())

    depth_to_routes: dict[int, list[set]] = defaultdict(list)
    edge_bucket_cache: dict[tuple[int, int], set] = {}

    for fidx, min_d in formula_to_min_depth.items():
        if min_d == 0:
            continue
        nodes_at_min_d = [ni for ni in formula_to_nodes[fidx] if node_depth(ni) == min_d]

        routes: list[set] = []
        for ni in nodes_at_min_d:
            child_atoms = node_cc[ni]
            for pi in parent_of[ni]:
                cache_key = (ni, pi)
                if cache_key not in edge_bucket_cache:
                    complement_atoms = node_cc[pi] - child_atoms
                    edge_bucket_cache[cache_key] = (
                        boundary_buckets(mol, child_atoms, complement_atoms)
                        if complement_atoms
                        else None
                    )
                buckets = edge_bucket_cache[cache_key]
                if buckets is not None:
                    routes.append(buckets)

        if routes:
            depth_to_routes[min_d].append(routes)

    return dict(depth_to_routes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tar", required=True)
    parser.add_argument("--mol_df", required=True)
    parser.add_argument("--n_sample", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_nodes", type=int, default=50000)
    args = parser.parse_args()

    print(f"Loading {args.mol_df} ...")
    mol_df = pd.read_pickle(args.mol_df)
    mol_by_id = dict(zip(mol_df["mol_id"], mol_df["mol"], strict=True))

    print(f"Scanning {args.tar} ...")
    with tarfile.open(args.tar, "r") as tf:
        members = [
            m for m in tf.getmembers() if m.name.endswith(".pkl.bz2") and "/dags/" in m.name
        ]
    print(f"  Total DAG files: {len(members):,}")

    random.seed(args.seed)
    sample = random.sample(members, min(args.n_sample, len(members)))
    print(f"  Sampling {len(sample):,} molecules ...")

    depth_to_routes: dict[int, list[list[set]]] = defaultdict(list)
    n_analyzed = 0
    n_skipped_large = 0
    n_skipped_no_mol = 0

    with tarfile.open(args.tar, "r") as tf:
        for i, member in enumerate(sample):
            if i % 20 == 0:
                print(f"  {i}/{len(sample)} ...", flush=True)
            f = tf.extractfile(member)
            if f is None:
                continue
            dag = pickle.loads(bz2.decompress(f.read()))
            if dag["reached_depth"] < 1:
                continue
            if dag["dag"].x.shape[0] > args.max_nodes:
                n_skipped_large += 1
                continue

            mol_id = int(os.path.basename(member.name).split(".")[0])
            mol = mol_by_id.get(mol_id)
            if mol is None:
                n_skipped_no_mol += 1
                continue

            result = analyze_dag(dag, mol)
            if result is None:
                continue
            for d, formula_routes_list in result.items():
                depth_to_routes[d].extend(formula_routes_list)
            n_analyzed += 1

    print(
        f"  Skipped {n_skipped_large:,} molecules (>{args.max_nodes:,} nodes), "
        f"{n_skipped_no_mol:,} molecules (no mol_id match)"
    )
    print(f"\nAnalyzed {n_analyzed} molecules")

    max_d = max(depth_to_routes.keys())
    formulas_at_max = depth_to_routes[max_d]
    total_at_max = len(formulas_at_max)
    print(f"\n{'='*100}")
    print(f"Depth d={max_d}: {total_at_max:,} formulas first-matched here")

    all_buckets = {
        bucket for formula_routes in formulas_at_max for route in formula_routes for bucket in route
    }

    print(f"\n{'='*100}")
    print("Per-bucket masking impact at d=max: a formula is LOST only if EVERY route to it")
    print("(every tied min-depth parent edge / multi-bond cut) contains a bond of this bucket")
    print(f"{'-'*100}")
    print(f"{'in_ring':<9} {'pair':<6} {'alpha_hetero':<13} {'#formulas_lost':>15} {'loss %':>10}")
    single_bucket_loss = []
    for bucket in all_buckets:
        lost = sum(
            1 for formula_routes in formulas_at_max if all(bucket in route for route in formula_routes)
        )
        single_bucket_loss.append((bucket, lost))
    single_bucket_loss.sort(key=lambda kv: kv[1])
    for (in_ring, pair, alpha_hetero), lost in single_bucket_loss:
        pct = 100 * lost / total_at_max
        print(f"{str(in_ring):<9} {pair:<6} {str(alpha_hetero):<13} {lost:>15,} {pct:>9.2f}%")

    print(f"\n{'='*100}")
    print("Greedy combined masking: add buckets one at a time (lowest marginal loss first),")
    print("recomputing recall loss for the CURRENT combined mask set at each step")
    print(f"{'-'*100}")
    masked: set = set()
    remaining = set(all_buckets)
    while remaining:
        best_bucket, best_lost = None, None
        for bucket in remaining:
            trial = masked | {bucket}
            lost = sum(
                1
                for formula_routes in formulas_at_max
                if all(any(b in trial for b in route) for route in formula_routes)
            )
            if best_lost is None or lost < best_lost:
                best_bucket, best_lost = bucket, lost
        masked.add(best_bucket)
        remaining.discard(best_bucket)
        pct = 100 * best_lost / total_at_max
        in_ring, pair, alpha_hetero = best_bucket
        print(
            f"  + mask (in_ring={in_ring}, pair={pair}, alpha_hetero={alpha_hetero}) "
            f"-> cumulative loss {best_lost:,} ({pct:.2f}%)"
        )


if __name__ == "__main__":
    main()
