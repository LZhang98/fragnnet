"""What does the candidate token's cut description resolve, on MASShatter's own candidates?

Two questions, one population.

1. The cut-set topology numbers in ``docs/plans/masshatter_cut_set_features_plan.md`` (96.8% of
   fragments cut at >= 2 bonds, 55.4% with two cuts sharing a ring) were measured on the depth-3
   BFS DAG, where the cut count is bounded by enumeration depth. The plan says to recompute them
   on real MASShatter candidates before quoting them. This does that.

2. Whether the cut-set topology scalars (plan item A) add anything over the pooled cut-bond
   features that landed on 2026-09-22 (plan item B0). B0 feeds the SUM of the severed bonds'
   input features, which is coarser than their multiset and carries no relation BETWEEN the cuts:
   "two ring bonds were cut" does not say "the same ring". The decisive number here is how often
   two candidates of one molecule share a summed cut-bond vector but differ in ring sharing.

Method. Candidates come from ``build_mass_shatter_candidates`` at the production unique-fragment
settings, so the population is MASShatter's own, not a DAG's. Bond logits are UNIFORM: mask
ranking then depends only on how many bonds a mask cuts, which is the deterministic,
model-independent part of the enumeration. A trained proposer concentrates on a few breakable
bonds and would push mass toward higher cut counts among them, so the cut counts below are a
lower bound, not a mean. Every descriptor is compared WITHIN a molecule, because that is the set
the intensity head ranks against each other.

Usage:
    python scripts/debug/debug_candidate_cut_descriptors.py [num_molecules]
"""

import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch as th
from torch_geometric.data import Batch

from fragnnet.frag.mass_shatter_candidates import build_mass_shatter_candidates
from fragnnet.utils.feat_utils import get_mol_graph

# The c4a_base_s1 bond feature set, which is what `mass_shatter_candidate_cut_bond_features`
# pools. Atom features are irrelevant here and are kept minimal.
NODE_FEATS = ["a_onehot"]
EDGE_FEATS = ["b_degree", "b_ring", "b_conj"]

# Production unique-fragment settings from the c4a campaign.
UNIQUE_FRAGMENT_TARGET = 256
RANKED_MASK_BUDGET = 4096


def _candidate_atom_sets(mol) -> list[frozenset[int]]:
    """Return one atom set per unique MASShatter candidate of this molecule."""
    graph = get_mol_graph(mol, NODE_FEATS, EDGE_FEATS, 0, True)
    batch = Batch.from_data_list([graph])
    num_bonds = int(batch.chemical_bond_index.shape[1])
    if num_bonds == 0:
        return []
    candidates = build_mass_shatter_candidates(
        th.zeros(num_bonds),
        batch.chemical_bond_index,
        batch.batch,
        batch.ptr,
        max_masks=UNIQUE_FRAGMENT_TARGET,
        include_single_bond_cuts=True,
        max_unique_components=UNIQUE_FRAGMENT_TARGET,
        max_enumerated_masks=RANKED_MASK_BUDGET,
    )
    representatives = candidates.unique_representative_component_indices.tolist()
    members = defaultdict(set)
    for atom, component in zip(
        candidates.atom_indices.tolist(),
        candidates.atom_component_indices.tolist(),
    ):
        members[component].add(atom)
    return [frozenset(members[component]) for component in representatives]


def _bond_tables(mol) -> tuple[dict, dict, np.ndarray]:
    """Return per-bond endpoints, the rings each bond belongs to, and its feature row."""
    graph = get_mol_graph(mol, NODE_FEATS, EDGE_FEATS, 0, True)
    endpoints = {
        index: (int(u), int(v))
        for index, (u, v) in enumerate(zip(*graph.chemical_bond_index.tolist()))
    }
    bond_rings = {
        index: {
            ring_index
            for ring_index, ring in enumerate(mol.GetRingInfo().BondRings())
            if index in ring
        }
        for index in range(mol.GetNumBonds())
    }
    return endpoints, bond_rings, graph.chemical_bond_attr.numpy().astype(np.int64)


def _cut_bonds(atom_set: frozenset[int], endpoints: dict) -> list[int]:
    """Bonds with exactly one endpoint inside the fragment -- the candidate's severed bonds."""
    return [index for index, (u, v) in endpoints.items() if (u in atom_set) != (v in atom_set)]


def _topology(
    cuts: list[int],
    atom_set: frozenset[int],
    endpoints: dict,
    bond_rings: dict,
) -> tuple[int, int, int, int]:
    """Ring-sharing pairs, atom-sharing pairs split by side, and distinct rings cut.

    An adjacent pair shares exactly one atom (no multi-edges), and which side that atom is on is
    a different event: a shared INSIDE atom means one retained atom lost two substituents, a
    shared OUTSIDE atom means the fragment expelled a bridging atom (the CO/CH2/S extrusion
    shape).
    """
    ring_pairs = inside_pairs = outside_pairs = 0
    for i in range(len(cuts)):
        for j in range(i + 1, len(cuts)):
            if bond_rings[cuts[i]] & bond_rings[cuts[j]]:
                ring_pairs += 1
            shared = set(endpoints[cuts[i]]) & set(endpoints[cuts[j]])
            if not shared:
                continue
            if shared & atom_set:
                inside_pairs += 1
            else:
                outside_pairs += 1
    rings = set()
    for cut in cuts:
        rings |= bond_rings[cut]
    return ring_pairs, inside_pairs, outside_pairs, len(rings)


def _bucket_stats(keys: list[tuple]) -> tuple[float, float]:
    """Mean candidates sharing a key (candidate-weighted) and the share in a singleton bucket."""
    sizes = Counter(keys)
    per_candidate = np.array([sizes[key] for key in keys], dtype=float)
    return float(per_candidate.mean()), float((per_candidate == 1).mean())


def main(num_molecules: int = 100) -> None:
    mol_df = pd.read_pickle("data/proc/nist20/mol_df.pkl")
    mols = dict(zip(mol_df["mol_id"], mol_df["mol"]))
    del mol_df
    rng = np.random.default_rng(0)
    mol_ids = rng.choice(list(mols), size=num_molecules, replace=False)

    cut_histogram = Counter()
    multi = same_ring = adjacent = total = 0
    # Descriptor -> per-candidate bucket keys, pooled across molecules but keyed by molecule so
    # candidates of different precursors never share a bucket.
    descriptors = defaultdict(list)
    # Candidates sharing a molecule and a summed cut-bond vector, split by which part of the
    # cut-set topology they disagree about.
    ambiguity = Counter()
    shared_sum = 0

    for mol_id in mol_ids:
        mol = mols.get(mol_id)
        if mol is None or mol.GetNumBonds() == 0:
            continue
        atom_sets = _candidate_atom_sets(mol)
        if not atom_sets:
            continue
        endpoints, bond_rings, bond_attr = _bond_tables(mol)

        rows = []
        for atom_set in atom_sets:
            cuts = _cut_bonds(atom_set, endpoints)
            if not cuts:
                continue  # the intact precursor breaks nothing
            topology = _topology(cuts, atom_set, endpoints, bond_rings)
            rows.append(
                {
                    "size": len(atom_set),
                    "cuts": len(cuts),
                    # Exactly the feature `mass_shatter_candidate_cut_bond_features` sums; the
                    # mean half adds nothing once the cut count is known.
                    "cut_sum": tuple(bond_attr[cuts].sum(axis=0).tolist()),
                    "topology": topology,
                }
            )

        for row in rows:
            total += 1
            cut_histogram[min(row["cuts"], 6)] += 1
            if row["cuts"] >= 2:
                multi += 1
                same_ring += row["topology"][0] > 0
                adjacent += (row["topology"][1] + row["topology"][2]) > 0
            descriptors["cut count"].append((mol_id, row["cuts"]))
            descriptors["cut count + size"].append((mol_id, row["cuts"], row["size"]))
            descriptors["cut-bond sum (B0)"].append((mol_id, row["cut_sum"]))
            adjacency_total = row["topology"][1] + row["topology"][2]
            descriptors["cut-bond sum + adjacency count"].append(
                (mol_id, row["cut_sum"], adjacency_total)
            )
            descriptors["cut-bond sum + adjacency in/out"].append(
                (mol_id, row["cut_sum"], row["topology"][1], row["topology"][2])
            )
            descriptors["cut-bond sum + full topology (B0+A)"].append(
                (mol_id, row["cut_sum"], row["topology"])
            )
            descriptors["size + cut count + cut-bond sum"].append(
                (mol_id, row["size"], row["cuts"], row["cut_sum"])
            )
            descriptors["size + cut count + cut-bond sum + adjacency in/out"].append(
                (
                    mol_id,
                    row["size"],
                    row["cuts"],
                    row["cut_sum"],
                    row["topology"][1],
                    row["topology"][2],
                )
            )
            descriptors["size + cut count + cut-bond sum + full topology"].append(
                (mol_id, row["size"], row["cuts"], row["cut_sum"], row["topology"])
            )

        # What A adds over B0: candidates that share a molecule and a summed cut-bond vector but
        # disagree about ring sharing (or about the whole topology triple).
        by_sum = defaultdict(list)
        for row in rows:
            by_sum[row["cut_sum"]].append(row["topology"])
        for topologies in by_sum.values():
            if len(topologies) < 2:
                continue
            shared_sum += len(topologies)
            for label, values in (
                ("ring-sharing flag", {topology[0] > 0 for topology in topologies}),
                (
                    "adjacency flag",
                    {(topology[1] + topology[2]) > 0 for topology in topologies},
                ),
                ("adjacency count", {topology[1] + topology[2] for topology in topologies}),
                ("adjacency in/out split", {(topology[1], topology[2]) for topology in topologies}),
                ("ring-shared pair count", {topology[0] for topology in topologies}),
                ("distinct rings cut", {topology[3] for topology in topologies}),
                ("full topology", set(topologies)),
            ):
                if len(values) > 1:
                    ambiguity[label] += len(topologies)

    if not total:
        print("no candidates enumerated")
        return

    print(f"molecules: {num_molecules}    cut candidates: {total:,}\n")
    print("cut bonds per candidate (MASShatter candidates, uniform bond logits):")
    for key in sorted(cut_histogram):
        label = str(key) if key < 6 else ">=6"
        print(f"  {label:>3s} cut(s): {cut_histogram[key] / total:6.3f}")
    print(f"\ncandidates with >= 2 cuts: {multi / total:.3f}")
    if multi:
        print(f"  of those, >= 2 cuts share a ring:  {same_ring / multi:.3f}")
        print(f"  of those, >= 2 cuts share an atom: {adjacent / multi:.3f}")
        print(
            f"  as a share of ALL candidates: same-ring {same_ring / total:.3f}, "
            f"adjacent {adjacent / total:.3f}"
        )

    print("\nwithin-molecule identifiability of each descriptor:")
    print(f"  {'descriptor':<46s} {'mean/bucket':>12s} {'singletons':>11s}")
    for name, keys in descriptors.items():
        mean_bucket, singleton_share = _bucket_stats(keys)
        print(f"  {name:<46s} {mean_bucket:12.2f} {singleton_share:11.3f}")

    print("\nwhat the topology scalars add over the pooled cut-bond sum (plan A over B0):")
    print(f"  candidates sharing a cut-bond sum with another candidate: {shared_sum / total:.3f}")
    print("  of all candidates, those whose cut-bond sum bucket disagrees about")
    for label in (
        "ring-sharing flag",
        "adjacency flag",
        "adjacency count",
        "adjacency in/out split",
        "ring-shared pair count",
        "distinct rings cut",
        "full topology",
    ):
        print(f"    {label:<24s} {ambiguity[label] / total:6.3f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 100)
