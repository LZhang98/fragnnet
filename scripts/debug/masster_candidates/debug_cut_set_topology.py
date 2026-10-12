"""How much of the candidate set can MASSter's pooled boundary NOT tell apart?

The intensity token summarizes a candidate's cut bonds as a COUNT (`cut_counts`) plus a mean over
per-bond encodings. Mean pooling is permutation-invariant and additive, so any relationship
BETWEEN the cut bonds is destroyed: two cuts in the same ring (a ring opening, which releases no
fragment on its own and is the retro-Diels-Alder signature) is represented identically to two cuts
in unrelated positions.

This counts, over real depth-3 DAG fragments, how often that distinction exists to be lost.

Caveat on transfer: the fragments here come from the depth-3 BFS DAG, where cut count is bounded
by the enumeration depth, so the cut-count DISTRIBUTION is an artifact of that enumeration and is
not MASSter's. MASSter ranks unrestricted multi-bond cut masks, so it sees at least as many
multi-cut candidates. Re-run against real candidate batches before quoting the distribution.

Results and their interpretation: ``docs/plans/masster/masster_cut_set_features_plan.md``.

Usage:
    python scripts/debug/masster_candidates/debug_cut_set_topology.py
"""

from collections import defaultdict

import numpy as np
import pandas as pd

from fragnnet.models.fragnnet.frag_utils import load_frag_d

FRAG_DP = "data/frag/nist20_ma_mi_d3_h4_isoFalse_smpFalse/dags"


def main(num_molecules=200):
    mol_df = pd.read_pickle("data/proc/nist20/mol_df.pkl")
    mols = dict(zip(mol_df["mol_id"], mol_df["mol"]))
    del mol_df

    rng = np.random.default_rng(0)
    mol_ids = rng.choice(list(mols), size=num_molecules, replace=False)

    hist = defaultdict(int)
    same_ring = adjacent = multi = total = 0
    for mol_id in mol_ids:
        mol = mols.get(mol_id)
        if mol is None:
            continue
        try:
            frag = load_frag_d(int(mol_id), FRAG_DP, is_compressed=True)
        except Exception:
            continue
        if frag is None:
            continue
        dag = frag["dag"]
        for name in ("boundary_pair_frag_idxs", "boundary_pair_in_local", "boundary_pair_out_local"):
            if not hasattr(dag, name):
                return print("DAG lacks boundary pair tensors")
        frag_idx = np.asarray(dag.boundary_pair_frag_idxs)
        in_idx = np.asarray(dag.boundary_pair_in_local)
        out_idx = np.asarray(dag.boundary_pair_out_local)

        ring_info = mol.GetRingInfo()
        bond_rings = {}
        for bond in mol.GetBonds():
            key = frozenset((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
            bond_rings[key] = {
                ring_index
                for ring_index, ring in enumerate(ring_info.BondRings())
                if bond.GetIdx() in ring
            }

        cuts_by_frag = defaultdict(set)
        for f, a, b in zip(frag_idx, in_idx, out_idx):
            cuts_by_frag[int(f)].add(frozenset((int(a), int(b))))

        for cuts in cuts_by_frag.values():
            n = len(cuts)
            hist[min(n, 6)] += 1
            total += 1
            if n < 2:
                continue
            multi += 1
            rings = [bond_rings.get(c, set()) for c in cuts]
            if any(rings[i] & rings[j] for i in range(n) for j in range(i + 1, n)):
                same_ring += 1
            atoms = [set(c) for c in cuts]
            if any(atoms[i] & atoms[j] for i in range(n) for j in range(i + 1, n)):
                adjacent += 1

    print(f"fragments with boundary bonds: {total:,}\n")
    print("cut bonds per fragment:")
    for k in sorted(hist):
        label = f"{k}" if k < 6 else ">=6"
        print(f"  {label:>3s} cut(s): {hist[k] / total:6.3f}")
    print(f"\nfragments with >=2 cuts: {multi / total:.3f} of all fragments")
    if multi:
        print(f"  of those, >=2 cuts SHARE A RING (ring opening): {same_ring / multi:.3f}")
        print(f"  of those, >=2 cuts SHARE AN ATOM (adjacent):    {adjacent / multi:.3f}")
        print(f"\n  as a share of ALL fragments: same-ring {same_ring / total:.3f}, "
              f"adjacent {adjacent / total:.3f}")


if __name__ == "__main__":
    main()
