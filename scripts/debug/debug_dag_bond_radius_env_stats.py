"""Mine matched-formula recall by a radius-N circular bond environment descriptor, instead
of the fixed (in_ring, element_pair, alpha_hetero) bucket used in
debug_dag_bond_descriptor_stats.py. The fixed bucket is itself only a radius-1 approximation
(it looks one bond out to check for an adjacent heteroatom); this script generalizes to an
arbitrary radius N so bonds like the C-C in O=C-C=C (conjugated / alpha to a carbonyl) can be
told apart from a bare alkyl C-C, per docs/plans/dag_fanout_cap_plan.md's open question of whether
hand-mining still works at finer granularity or a learned scorer is needed instead.

This is a cardinality/support diagnostic, not a masking-policy search: at radius >= 2 the
number of distinct bond-environment buckets can be far larger than the ~12 buckets in the
fixed-descriptor script, so the question this script answers is "how many distinct buckets
show up, and how much support (route count) does each one have" -- if most buckets are
singletons, a hand-mined per-bucket table has no statistical legs and a learned scorer (which
can generalize across similar-but-not-identical environments) is the better next step.

Usage:
    python scripts/debug/debug_dag_bond_radius_env_stats.py \
        --tar data/frag/nist20_ma_mi_d4_h4_isoFalse_smpFalse.tar \
        --mol_df data/proc/nist20/mol_df.pkl \
        --n_sample 500 \
        --radii 1,2 \
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
from fragnnet.utils.frag_utils import th_long_to_mask

BOND_SYMBOL = {
    "SINGLE": "-",
    "DOUBLE": "=",
    "TRIPLE": "#",
    "AROMATIC": ":",
}


def decode_cc(cc_row: torch.Tensor, num_atoms: int) -> frozenset:
    """Decode a packed 128-bit atom bitmask (2 int64 words) into a set of atom indices."""
    mask = th_long_to_mask(cc_row.reshape(1, -1).long())[0]
    return frozenset(torch.nonzero(mask[:num_atoms]).flatten().tolist())


def boundary_bond_pairs(mol, child_atoms: frozenset, complement_atoms: frozenset) -> set:
    """Find all mol bonds between child_atoms and complement_atoms; return raw (a, b) pairs."""
    pairs = set()
    for a in child_atoms:
        atom = mol.GetAtomWithIdx(a)
        for nbr in atom.GetNeighbors():
            b = nbr.GetIdx()
            if b in complement_atoms:
                pairs.add((a, b) if a < b else (b, a))
    return pairs


def analyze_dag(dag: dict, mol) -> dict[int, list[list[set]]] | None:
    """Same route-finding logic as debug_dag_bond_descriptor_stats.py, but records the raw
    (a, b) atom-index bond pairs broken by each route instead of pre-bucketing them, so the
    same DAG traversal can be reused to test multiple radii without re-scanning the tar.
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

    depth_to_routes: dict[int, list[list[set]]] = defaultdict(list)
    edge_pairs_cache: dict[tuple[int, int], set] = {}

    for fidx, min_d in formula_to_min_depth.items():
        if min_d == 0:
            continue
        nodes_at_min_d = [ni for ni in formula_to_nodes[fidx] if node_depth(ni) == min_d]

        routes: list[set] = []
        for ni in nodes_at_min_d:
            child_atoms = node_cc[ni]
            for pi in parent_of[ni]:
                cache_key = (ni, pi)
                if cache_key not in edge_pairs_cache:
                    complement_atoms = node_cc[pi] - child_atoms
                    edge_pairs_cache[cache_key] = (
                        boundary_bond_pairs(mol, child_atoms, complement_atoms)
                        if complement_atoms
                        else None
                    )
                pairs = edge_pairs_cache[cache_key]
                if pairs is not None:
                    routes.append(pairs)

        if routes:
            depth_to_routes[min_d].append(routes)

    return dict(depth_to_routes)


def atom_env(mol, atom_idx: int, radius: int, came_from: int, cache: dict) -> tuple:
    """Rooted circular environment of `atom_idx` out to `radius` bonds, excluding the bond
    back toward `came_from` (the other end of the bond being described) so the environment
    reflects "what's attached on this side of the cut", not the cut itself. Memoized per
    molecule since the same (atom, came_from, radius) is revisited across redundant routes.
    """
    key = (atom_idx, came_from, radius)
    if key in cache:
        return cache[key]
    atom = mol.GetAtomWithIdx(atom_idx)
    base = (atom.GetSymbol(), atom.GetIsAromatic(), atom.GetFormalCharge())
    if radius == 0:
        result = base + ((),)
        cache[key] = result
        return result
    children = []
    for nbr in atom.GetNeighbors():
        nbr_idx = nbr.GetIdx()
        if nbr_idx == came_from:
            continue
        bond = mol.GetBondBetweenAtoms(atom_idx, nbr_idx)
        bt = str(bond.GetBondType())
        children.append((bt, atom_env(mol, nbr_idx, radius - 1, atom_idx, cache)))
    children.sort()
    result = base + (tuple(children),)
    cache[key] = result
    return result


def env_to_str(env: tuple) -> str:
    """Human-readable rendering of an atom_env tuple, e.g. C(=O,-C(=C))."""
    symbol, aromatic, charge, children = env
    label = symbol.lower() if aromatic else symbol
    if charge:
        label += f"{charge:+d}"
    if children:
        parts = [BOND_SYMBOL.get(bt, "?") + env_to_str(child) for bt, child in children]
        label += "(" + ",".join(parts) + ")"
    return label


def bond_env_bucket(mol, a: int, b: int, radius: int, cache: dict) -> tuple:
    """Undirected radius-N bond descriptor: (in_ring, sorted pair of rooted atom envs)."""
    bond = mol.GetBondBetweenAtoms(a, b)
    in_ring = bond.IsInRing()
    env_a = atom_env(mol, a, radius, b, cache)
    env_b = atom_env(mol, b, radius, a, cache)
    ordered = tuple(sorted([env_a, env_b]))
    return (in_ring, ordered)


def bond_env_str(bucket: tuple) -> str:
    in_ring, (env_a, env_b) = bucket
    ring_tag = "ring" if in_ring else "chain"
    return f"[{ring_tag}] {env_to_str(env_a)} -- {env_to_str(env_b)}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tar", required=True)
    parser.add_argument("--mol_df", required=True)
    parser.add_argument("--n_sample", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_nodes", type=int, default=50000)
    parser.add_argument("--radii", type=str, default="1,2", help="Comma-separated radii to test.")
    parser.add_argument("--top_k", type=int, default=30)
    args = parser.parse_args()
    radii = [int(r) for r in args.radii.split(",")]

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

    # mol_data[max_d] entries are (mol, formula_routes); formula_routes is a list of
    # routes, each route a set of raw (a, b) bond pairs, per the docstring in analyze_dag.
    max_depth_data: list[tuple] = []
    n_analyzed = 0
    n_skipped_large = 0
    n_skipped_no_mol = 0
    global_max_d = 0

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
            local_max_d = max(result.keys())
            global_max_d = max(global_max_d, local_max_d)
            for formula_routes in result[local_max_d]:
                max_depth_data.append((mol, local_max_d, formula_routes))
            n_analyzed += 1

    print(
        f"  Skipped {n_skipped_large:,} molecules (>{args.max_nodes:,} nodes), "
        f"{n_skipped_no_mol:,} molecules (no mol_id match)"
    )
    print(f"\nAnalyzed {n_analyzed} molecules")

    # Only keep molecules whose own max matched-depth equals the global max, matching the
    # original script's "depth d=max_d" slice (deepest layer is where recall pressure is
    # highest / where masking would bite most).
    formulas_at_max = [
        (mol, routes) for mol, local_max_d, routes in max_depth_data if local_max_d == global_max_d
    ]
    total_at_max = len(formulas_at_max)
    print(f"\n{'='*100}")
    print(f"Depth d={global_max_d}: {total_at_max:,} formulas first-matched here")

    for radius in radii:
        print(f"\n{'='*100}")
        print(f"RADIUS = {radius}")
        print(f"{'-'*100}")

        bucket_support: dict[tuple, int] = defaultdict(int)  # routes containing this bucket
        bucket_loss: dict[tuple, int] = defaultdict(int)  # formulas fully blocked if masked
        mol_cache: dict = {}
        last_mol = None

        for mol, routes in formulas_at_max:
            if mol is not last_mol:
                mol_cache = {}
                last_mol = mol
            route_buckets = []
            for pairs in routes:
                buckets = {bond_env_bucket(mol, a, b, radius, mol_cache) for a, b in pairs}
                route_buckets.append(buckets)
                for bucket in buckets:
                    bucket_support[bucket] += 1
            all_buckets_here = {b for rb in route_buckets for b in rb}
            for bucket in all_buckets_here:
                if all(bucket in rb for rb in route_buckets):
                    bucket_loss[bucket] += 1

        n_buckets = len(bucket_support)
        supports = sorted(bucket_support.values())
        n_singleton = sum(1 for s in supports if s == 1)
        print(f"Distinct buckets: {n_buckets:,}")
        print(
            f"Singleton buckets (support=1): {n_singleton:,} "
            f"({100 * n_singleton / n_buckets:.2f}% of buckets)"
        )
        print(
            f"Support distribution: min={supports[0]}, median={supports[len(supports)//2]}, "
            f"max={supports[-1]}, mean={sum(supports)/len(supports):.2f}"
        )

        print(f"\nTop {args.top_k} buckets by single-bucket masking loss (formulas fully blocked):")
        print(f"{'support':>8} {'loss':>8} {'loss %':>8}  bucket")
        top = sorted(bucket_loss.items(), key=lambda kv: -kv[1])[: args.top_k]
        for bucket, lost in top:
            pct = 100 * lost / total_at_max
            print(
                f"{bucket_support[bucket]:>8,} {lost:>8,} {pct:>7.2f}%  {bond_env_str(bucket)}"
            )


if __name__ == "__main__":
    main()
