"""Extract bond SMARTS for bonds broken to produce matched DAG fragments.

For each matched DAG fragment, identifies the bonds that were cut (boundary bonds
between fragment atoms and non-fragment atoms in the parent molecule). Collects two
counters per split:
  - all_bonds  : every bond environment in all dataset molecules
  - broken_bonds: every bond environment that was cut to form a matched fragment

Outputs per split:
  all_bond_smarts_{split}.csv    — bond_smarts, count
  broken_bond_smarts_{split}.csv — bond_smarts, count

Never-broken bond environments = all_bonds.keys() - broken_bonds.keys()

Usage:
    conda run -n FRAGNNET-GPU python preproc_scripts/no_dag_exp/04_extract_bond_smarts.py \\
        --template_fp config \\
        --custom_fp config/debug/debug_tiny.yml \\
        --output_dp ./data/bond_smarts/debug_tiny \\
        --batch_size 10 --num_workers 0 --split_names train --env_radius 0
"""

import argparse
import logging
import os
import time
from collections import Counter

import pandas as pd
import torch as th
from joblib import Parallel, delayed
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from tqdm import tqdm

from fragnnet.models.fragnnet.dag.layout import get_node_feats
from fragnnet.runner import init_dataloader, init_dataset, load_config
from fragnnet.utils.formula_utils import (
    H_MASS,
    PREC_TYPE_TO_CMF_MASS_DIFF,
    PREC_TYPE_TO_MASS_DIFF,
    th_long_to_mask,
)
from fragnnet.utils.mol_utils import mol_from_smiles
from fragnnet.utils.spec_similarity import calculate_match_mzs

INST_TOL: dict[str, dict] = {
    "ORB": {"rel": 1e-5, "min_mz": 200.0},  # 10 ppm, floor 0.002 Da
    "ORB_F": {"rel": 1e-5, "min_mz": 200.0},
    "QTOF": {"rel": 2e-5, "min_mz": 750.0},  # 20 ppm, floor 0.015 Da
    "IT": {"rel": 1e-5, "min_mz": 200.0},  # same as ORB
}


def convert_mask(input_mask: th.Tensor) -> th.Tensor:
    """Convert packed h_formulae_idx to boolean h_delta validity mask."""
    batch_size = input_mask.shape[0]
    num_h = input_mask.shape[1] // 2
    output = th.ones((batch_size, 2 * num_h + 1), dtype=input_mask.dtype, device=input_mask.device)
    odd_indices = th.arange(1, 2 * num_h + 1, 2, dtype=th.long, device=input_mask.device)
    even_indices = th.arange(2, 2 * num_h + 2, 2, dtype=th.long, device=input_mask.device)
    output[:, :num_h] = (input_mask[:, odd_indices] != 0).to(input_mask.dtype)
    output[:, num_h + 1 :] = (input_mask[:, even_indices] != 0).to(input_mask.dtype)
    return output


def get_frag_mw(mol: Chem.Mol, atom_ids: list[int]) -> float:
    """Exact MW of a fragment (atom subset), accounting for broken-bond dummy atoms.

    Args:
        mol: Parent RDKit molecule.
        atom_ids: Atom indices belonging to the fragment.

    Returns:
        Exact monoisotopic MW of the fragment submolecule.
    """
    q_mol = Chem.RWMol()
    atom_map: dict[int, int] = {}
    for aidx in atom_ids:
        new_idx = q_mol.AddAtom(mol.GetAtomWithIdx(aidx))
        atom_map[aidx] = new_idx
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if u in atom_map and v in atom_map:
            q_mol.AddBond(atom_map[u], atom_map[v], bond.GetBondType())
        elif u in atom_map or v in atom_map:
            inner = atom_map[u] if u in atom_map else atom_map[v]
            q_mol.AddAtom(AllChem.Atom(0))
            q_mol.AddBond(inner, q_mol.GetNumAtoms() - 1, bond.GetBondType())
            q_mol.GetAtomWithIdx(inner).SetNumExplicitHs(0)
            q_mol.GetAtomWithIdx(inner).SetNoImplicit(False)
    q_mol.UpdatePropertyCache()
    return Descriptors.ExactMolWt(q_mol)


def atom_smarts(atom: Chem.Atom) -> str:
    """Atom SMARTS encoding element, aromaticity, and ring membership."""
    parts = [f"#{atom.GetAtomicNum()}"]
    parts.append("a" if atom.GetIsAromatic() else "!a")
    parts.append("R" if atom.IsInRing() else "!R")
    return "[" + ";".join(parts) + "]"


def get_bond_smarts(mol: Chem.Mol, u: int, v: int, env_radius: int) -> str:
    """Generate bond environment SMARTS for the bond between atoms u and v.

    Args:
        mol: RDKit molecule.
        u: Index of first atom.
        v: Index of second atom.
        env_radius: Neighborhood radius. 0 = just the two atoms + bond type;
            >=1 = local subgraph of all atoms within env_radius bonds of either endpoint.

    Returns:
        Bond SMARTS string (canonical for radius 0; RDKit-generated for radius >= 1).
    """
    if env_radius == 0:
        bond = mol.GetBondBetweenAtoms(u, v)
        if bond.GetIsAromatic():
            bond_str = ":"
        else:
            bond_str = {
                Chem.BondType.SINGLE: "-",
                Chem.BondType.DOUBLE: "=",
                Chem.BondType.TRIPLE: "#",
            }.get(bond.GetBondType(), "~")
        sa = atom_smarts(mol.GetAtomWithIdx(u))
        sb = atom_smarts(mol.GetAtomWithIdx(v))
        pair = tuple(sorted([sa, sb]))
        return f"{pair[0]}{bond_str}{pair[1]}"
    else:
        env_atoms: set[int] = {u, v}
        for seed in [u, v]:
            for bid in Chem.FindAtomEnvironmentOfRadiusN(mol, env_radius, seed):
                b = mol.GetBondWithIdx(bid)
                env_atoms |= {b.GetBeginAtomIdx(), b.GetEndAtomIdx()}
        atoms_list = sorted(env_atoms)
        atoms_set = set(atoms_list)
        bonds_list = [
            b.GetIdx()
            for b in mol.GetBonds()
            if b.GetBeginAtomIdx() in atoms_set and b.GetEndAtomIdx() in atoms_set
        ]
        return Chem.MolFragmentToSmarts(mol, atoms_list, bonds_list)


def process_b_idx(
    b_idx: int,
    spec_batch_idxs: th.Tensor,
    spec_prec_type_str: list[str],
    spec_inst_type: th.Tensor,
    idx_to_inst_type: dict[int, str],
    data: dict,
    num_hs: int,
    env_radius: int,
) -> tuple[Counter, Counter]:
    """Extract all-bond and broken-bond SMARTS counters for one batch item.

    Args:
        b_idx: Batch item index.
        spec_batch_idxs: Batch index per spectrum peak row, shape (n_peaks,).
        spec_prec_type_str: Precursor type string per batch item.
        spec_inst_type: Instrument type integer index per batch item, shape (batch_size,).
        idx_to_inst_type: Mapping from integer index to instrument type string.
        data: Full batch data dict.
        num_hs: Max H-transfer count.
        env_radius: Neighborhood radius for bond SMARTS.

    Returns:
        Tuple of (all_bonds Counter, broken_bonds Counter).
    """
    b_spec_mask = spec_batch_idxs == b_idx
    mol_smiles = data["mol_smiles"][b_idx]
    b_prec_type = spec_prec_type_str[b_idx]
    b_inst_idx = spec_inst_type[b_idx].item()
    b_inst_type = idx_to_inst_type.get(b_inst_idx, "ORB")
    b_mol = mol_from_smiles(mol_smiles)

    frag_pyg = data["frag_pyg"][b_idx]
    frag_node_feat_idxs = frag_pyg.node_feat_idxs[0]
    frag_node_mask = th_long_to_mask(get_node_feats(frag_pyg.x, frag_node_feat_idxs, "cc"))
    frag_h_mask = convert_mask(get_node_feats(frag_pyg.x, frag_node_feat_idxs, "h_formulae_idx"))

    b_num_nodes = frag_node_mask.shape[0]
    b_num_mw_per_node = 2 * num_hs + 1
    all_atom_ids = th.arange(frag_node_mask.shape[1], dtype=th.long)

    # Build predicted m/z arrays for standard and CMF adduct masses
    std_diff = PREC_TYPE_TO_MASS_DIFF[b_prec_type]
    cmf_diff = PREC_TYPE_TO_CMF_MASS_DIFF.get(b_prec_type, std_diff)
    h_range = th.arange(-num_hs, num_hs + 1, dtype=th.float)

    b_frag_node_ids = th.zeros(b_num_nodes * b_num_mw_per_node, dtype=th.long)
    b_pred_mzs_std = th.zeros(b_num_nodes * b_num_mw_per_node, dtype=th.float)
    b_pred_mzs_cmf = th.zeros(b_num_nodes * b_num_mw_per_node, dtype=th.float)

    for i in range(b_num_nodes):
        frag_atom_ids = all_atom_ids[frag_node_mask[i]].tolist()
        mw = get_frag_mw(b_mol, frag_atom_ids)
        sl = slice(i * b_num_mw_per_node, (i + 1) * b_num_mw_per_node)
        b_frag_node_ids[sl] = i
        b_pred_mzs_std[sl] = mw + H_MASS * h_range + std_diff
        b_pred_mzs_cmf[sl] = mw + H_MASS * h_range + cmf_diff

    # Zero out invalid h_delta entries
    h_validity = frag_h_mask.flatten()
    b_pred_mzs_std = b_pred_mzs_std * h_validity
    b_pred_mzs_cmf = b_pred_mzs_cmf * h_validity

    b_true_mzs = data["spec_mzs"][b_spec_mask]

    # Per-peak absolute tolerance from instrument type
    cfg = INST_TOL[b_inst_type]
    tol_per_true = cfg["rel"] * th.clamp(b_true_mzs, min=cfg["min_mz"])

    b_match_std = calculate_match_mzs(
        b_true_mzs, b_pred_mzs_std, tol_per_true=tol_per_true, relative=False
    )
    if std_diff != cmf_diff:
        b_match_cmf = calculate_match_mzs(
            b_true_mzs, b_pred_mzs_cmf, tol_per_true=tol_per_true, relative=False
        )
        b_pred_match_mask = th.any(b_match_std | b_match_cmf, dim=0)
    else:
        b_pred_match_mask = th.any(b_match_std, dim=0)

    matched_node_ids = b_frag_node_ids[b_pred_match_mask]
    unique_matched = set(th.unique(matched_node_ids).tolist())

    # Enumerate all bond environments in the molecule
    all_bonds: Counter = Counter()
    for bond in b_mol.GetBonds():
        bs = get_bond_smarts(b_mol, bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), env_radius)
        all_bonds[bs] += 1

    # Enumerate cut bonds for each matched fragment
    broken_bonds: Counter = Counter()
    for i in range(b_num_nodes):
        if i not in unique_matched:
            continue
        frag_set = set(all_atom_ids[frag_node_mask[i]].tolist())
        for bond in b_mol.GetBonds():
            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if (u in frag_set) != (v in frag_set):
                bs = get_bond_smarts(b_mol, u, v, env_radius)
                broken_bonds[bs] += 1

    return all_bonds, broken_bonds


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract bond SMARTS for DAG cut bonds.")
    parser.add_argument("--template_fp", type=str, default="config")
    parser.add_argument("--custom_fp", type=str, default="config/debug/debug_tiny.yml")
    parser.add_argument("--output_dp", type=str, default="./data/bond_smarts/debug_tiny")
    parser.add_argument("--batch_size", type=int, default=40)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--split_names", type=str, nargs="+", default=["train", "val", "test"])
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="Parallel jobs for joblib (-1 = all cores).",
    )
    parser.add_argument(
        "--env_radius",
        type=int,
        default=0,
        help="Neighborhood radius for bond SMARTS. 0 = just the two atoms + bond type; "
        ">=1 = local subgraph of all atoms within env_radius bonds of either endpoint.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log = logging.getLogger(__name__)

    args = get_args()
    log.info(
        "Starting bond SMARTS extraction: splits=%s  env_radius=%d  batch_size=%d  n_jobs=%d",
        args.split_names,
        args.env_radius,
        args.batch_size,
        args.n_jobs,
    )
    log.info("Output directory: %s", args.output_dp)

    config_d = load_config(args.template_fp, args.custom_fp, None)
    config_d["train_batch_size"] = args.batch_size
    config_d["eval_batch_size"] = args.batch_size
    config_d["frag_params"]["preload"] = False
    config_d["frag_params"]["preprocess"] = False
    config_d["spec_params"]["prec_type_str"] = True
    config_d["num_workers"] = args.num_workers
    config_d["dynamic_batch_sampler"] = False

    num_hs = config_d["num_hs"]

    os.makedirs(args.output_dp, exist_ok=True)

    dses = init_dataset(config_d, splits=args.split_names)
    for split_name, dset in zip(args.split_names, dses):
        idx_to_inst_type: dict[int, str] = dset.idx_to_inst_type
        dl = init_dataloader(dset, config_d)
        log.info("[%s] %d batches (%d spectra)", split_name, len(dl), len(dset))

        all_bonds: Counter = Counter()
        broken_bonds: Counter = Counter()
        t0 = time.monotonic()

        for batch_idx, data in tqdm(enumerate(dl), total=len(dl), desc=split_name):
            results = Parallel(n_jobs=args.n_jobs)(
                delayed(process_b_idx)(
                    b_idx,
                    data["spec_batch_idxs"],
                    data["spec_prec_type_str"],
                    data["spec_inst_type"],
                    idx_to_inst_type,
                    data,
                    num_hs,
                    args.env_radius,
                )
                for b_idx in range(data["batch_size"].item())
            )
            for ab, bb in results:
                all_bonds.update(ab)
                broken_bonds.update(bb)

            if (batch_idx + 1) % 100 == 0:
                log.info(
                    "[%s] batch %d/%d — %d unique all-bond types, %d unique broken-bond types",
                    split_name,
                    batch_idx + 1,
                    len(dl),
                    len(all_bonds),
                    len(broken_bonds),
                )

        elapsed = time.monotonic() - t0
        log.info(
            "[%s] done in %.1fs — %d unique all-bond types (%s total), "
            "%d unique broken-bond types (%s total)",
            split_name,
            elapsed,
            len(all_bonds),
            f"{sum(all_bonds.values()):,}",
            len(broken_bonds),
            f"{sum(broken_bonds.values()):,}",
        )

        for fname, counter in [
            (f"all_bond_smarts_{split_name}.csv", all_bonds),
            (f"broken_bond_smarts_{split_name}.csv", broken_bonds),
        ]:
            df = (
                pd.DataFrame(counter.items(), columns=["bond_smarts", "count"])
                .sort_values("count", ascending=False)
                .reset_index(drop=True)
            )
            df.to_csv(os.path.join(args.output_dp, fname), index=False)
            log.info("  Wrote %s: %d unique bond SMARTS, %s total", fname, len(df), f"{df['count'].sum():,}")
