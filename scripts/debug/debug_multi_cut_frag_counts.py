"""Compare single-cut vs multi-cut BFS fragment counts across representative molecules.

Usage:
    python scripts/debug/debug_multi_cut_frag_counts.py
    python scripts/debug/debug_multi_cut_frag_counts.py --max_depth 4
"""

import argparse

import fragnnet.utils.frag_utils as frag_utils

MOLECULES = {
    "ethanol (acyclic)": "CCO",
    "aspirin (2 indep. rings)": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "naphthalene (2 fused rings)": "c1ccc2ccccc2c1",
    "benzotriazole (2 fused rings)": "c1ccc2[nH]nnc2c1",
    "testosterone (4 fused rings)": "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",
    "methoxydiphenylphosphine (2 indep. rings)": "COP(C1=CC=CC=C1)C2=CC=CC=C2",
    "big molecule (47 atoms)": "CC(=O)NC1C(OC2CCC3(C)C(CCC4(C)C3CC=C3C5CC(C)(C)CCC5(C(=O)O)CCC34C)C2(C)C)OC(CO)C(O)C1O",
}


def run_comparison(smiles: str, max_depth: int, max_time: int) -> None:
    mol_d = frag_utils.extract_mol_info(smiles)
    n_atoms = mol_d["atom_mask_arr"].shape[0]
    n_bonds = mol_d["bond_mask_arr"].shape[0]
    n_ring_bonds = int(mol_d["ring_bond_mask"].sum())

    _, d_single = frag_utils.timed_get_dags(
        mol_d["mol"],
        0,
        max_depth=max_depth,
        h_prior=True,
        max_h_transfer=4,
        max_time=max_time,
        isotopes=False,
        nb_isomorphic=False,
        wl_max_iterations=3,
        multi_cut_bfs=False,
    )
    _, d_multi = frag_utils.timed_get_dags(
        mol_d["mol"],
        0,
        max_depth=max_depth,
        h_prior=True,
        max_h_transfer=4,
        max_time=max_time,
        isotopes=False,
        nb_isomorphic=False,
        wl_max_iterations=3,
        multi_cut_bfs=True,
        max_cut_size=2,
    )

    nn_s = d_single.get("dag_num_nodes", 0)
    ne_s = d_single.get("dag_num_edges", 0)
    fs_s = d_single.get("force_stopped", False)

    nn_m = d_multi.get("dag_num_nodes", 0)
    ne_m = d_multi.get("dag_num_edges", 0)
    fs_m = d_multi.get("force_stopped", False)
    by_depth_m = d_multi.get("dag_num_nodes_by_depth", {})

    ratio_n = nn_m / nn_s if nn_s > 0 else float("inf")
    ratio_e = ne_m / ne_s if ne_s > 0 else float("inf")

    print(f"  atoms={n_atoms}, bonds={n_bonds}, ring_bonds={n_ring_bonds}")
    print(
        f"  single : nodes={nn_s:5d}, edges={ne_s:5d}"
        + (" [TIMEOUT]" if fs_s else "")
    )
    print(
        f"  multi2 : nodes={nn_m:5d}, edges={ne_m:5d}, "
        f"ratio={ratio_n:.2f}x nodes / {ratio_e:.2f}x edges"
        + (" [TIMEOUT]" if fs_m else "")
    )
    print(f"  nodes by depth: {by_depth_m}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare fragment counts: single vs multi-cut BFS")
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--max_time", type=int, default=300)
    args = parser.parse_args()

    print(f"max_depth={args.max_depth}, max_time={args.max_time}s\n")
    for name, smi in MOLECULES.items():
        print(f"[{name}]  {smi}")
        run_comparison(smi, args.max_depth, args.max_time)
        print()


if __name__ == "__main__":
    main()
