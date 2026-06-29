import argparse
from copy import deepcopy

import numpy as np
import pandas as pd
import torch as th
from rdkit import Chem
from rdkit.Chem import Draw
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from tqdm import tqdm

from fragnnet.runner import init_dataloader, init_dataset, load_config
from fragnnet.utils.data_utils import mol_from_smiles
from fragnnet.utils.frag_utils import get_node_feats, th_long_to_mask
from fragnnet.utils.misc_utils import PPM, TOLERANCE_MIN_MZ
from fragnnet.utils.spec_utils import calculate_match_mzs


def remove_atoms_and_group_by_connected(adj_matrix, atom_ids_to_remove):
    """
    Removes specified atom IDs from the adjacency matrix and groups the remaining atoms by connected components.
    Returns original atom IDs of the connected components.

    Parameters:
        adj_matrix (np.array): The adjacency matrix of the molecule.
        atom_ids_to_remove (list): List of atom indices to be removed from the matrix.

    Returns:
        List of connected components: Each component is a list of atom indices.
    """
    adj_matrix = (
        adj_matrix.copy()
    )  # Create a copy of the adjacency matrix to avoid modifying the original
    # Remove rows and columns corresponding to the atoms to be removed
    for atom_id in atom_ids_to_remove:
        adj_matrix[:, atom_id] = 0  # Set the column to zero
        adj_matrix[atom_id, :] = 0  # Set the row to zero

    # Convert the updated adjacency matrix to a sparse CSR matrix for efficient computation
    sparse_matrix = csr_matrix(adj_matrix)

    # Use scipy's connected_components to find the connected components in the graph
    num_components, labels = connected_components(sparse_matrix, directed=False, return_labels=True)

    # Group atom indices by connected components
    connected_atom_ids = [np.where(labels == i)[0].tolist() for i in range(num_components)]
    connected_atom_ids = [
        tuple(atom_ids)
        for atom_ids in connected_atom_ids
        if atom_ids[0] not in set(atom_ids_to_remove)
    ]

    return connected_atom_ids


def draw_highlighted_atoms(mol, atom_ids):
    local_mol = deepcopy(mol)  # Create a deep copy of the molecule to avoid modifying the original
    for i, atom in enumerate(local_mol.GetAtoms()):
        # For each atom, set the property "molAtomMapNumber" to a custom number, let's say, the index of the atom in the molecule
        atom.SetProp("molAtomMapNumber", str(atom.GetIdx()))

    # Create a list of atom indices to highlight
    highlight_atoms = set(atom_ids)

    # Generate image with highlighted atoms
    img = Draw.MolToImage(local_mol, highlightAtoms=highlight_atoms, size=(300, 300))

    return img


def get_substructure_from_smarts(target_smiles, substructure_smart):
    """
    Matches a SMARTS substructure with atom mapping to a target SMILES molecule,
    and returns the rest of the atom index sets grouped by disconnected fragments.

    Parameters:
        target_smiles (str): The SMILES of the molecule to search in.
        substructure_smiles (str): The SMARTS pattern with atom map numbers.

    Returns:
        matches (list of lists): Atom indices in target molecule corresponding to SMARTS-mapped atoms.
        remainders (list of lists of lists): For each match, a list of fragments,
                                             each fragment is a list of atom indices.
    """

    target_mol = Chem.MolFromSmiles(target_smiles)

    qmol = Chem.MolFromSmarts(substructure_smart)
    matches = []

    ind_map = {}  # map atom numbers to indices
    for atom in qmol.GetAtoms():
        map_num = atom.GetAtomMapNum()
        if map_num:
            ind_map[map_num - 1] = atom.GetIdx()

    map_list = [ind_map[x] for x in sorted(ind_map)]  # qmol atom indices list to match

    for match in target_mol.GetSubstructMatches(qmol):
        mas = tuple([match[x] for x in map_list])
        matches.append(mas)

    if len(matches) == 0:
        return [], []

    # get remainders
    remainders = []
    adj_matrix = Chem.rdmolops.GetAdjacencyMatrix(target_mol)
    all_atoms = set(range(target_mol.GetNumAtoms()))

    for match in matches:
        # Compute unmatched atom indices
        unmatched_atoms = list(all_atoms - set(match))

        if not unmatched_atoms:
            continue
        remainders = remove_atoms_and_group_by_connected(adj_matrix, atom_ids_to_remove=match)

    return matches, remainders


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--template_fp", type=str, default="config/template.yml", help="Path to the config file"
    )
    parser.add_argument(
        "--custom_fp",
        type=str,
        default="config/exp_fraggnn_mi_ma/fraggnn_d3_mi_ma.yml",
        help="Path to the custom config file",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--smarts_fp",
        type=str,
        default="./data/smarts/fraggnn_d3_mi_ma/smarts_train.csv",
        help="Path to the output file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # load smarts
    smarts_df = pd.read_csv(args.smarts_fp)
    smarts_df = smarts_df[smarts_df["smarts"].notna()]
    smarts_df = smarts_df.drop_duplicates(subset=["smiles", "prec_type"])
    smarts_df = smarts_df.reset_index(drop=True)
    print(smarts_df.groupby("prec_type")["smiles"].count())

    config_d = load_config(args.template_fp, args.custom_fp)

    config_d["train_batch_size"] = args.batch_size
    config_d["eval_batch_size"] = args.batch_size

    config_d["frag_params"]["preload"] = False
    config_d["frag_params"]["preprocess"] = False
    config_d["spec_params"]["prec_type_str"] = True
    config_d["num_workers"] = args.num_workers

    num_hs = config_d["num_hs"]
    splits = ["val"]  # ,'test']
    dses = init_dataset(config_d, splits=splits)

    ds = dses[0]
    dl = init_dataloader(ds, config_d)
    # val_dl = init_dataloader(val_ds, config_d)

    for batch_idx, data in tqdm(enumerate(dl), total=len(dl), desc="Processing batches"):
        spec_mzs = data["spec_mzs"]
        spec_prec_type = data["spec_prec_type"]
        spec_inst_type = data["spec_inst_type"]
        spec_prec_type_str = data["spec_prec_type_str"]
        frag_pyg = data["frag_pyg"]
        frag_formula_peak_mzs = data["frag_formula_peak_mzs"]
        frag_formula_peak_idxs = data["frag_formula_peak_idxs"]

        spec_batch_idxs = data["spec_batch_idxs"]

        for b_idx in range(data["batch_size"].item()):
            b_spec_mask = spec_batch_idxs == b_idx
            mol_smiles = data["mol_smiles"][b_idx]
            b_spec_prec_type_str = spec_prec_type_str[b_idx]
            b_mol = mol_from_smiles(mol_smiles)

            frag_pyg = data["frag_pyg"][b_idx]
            frag_node_feat_idxs = frag_pyg.node_feat_idxs[0]
            frag_node_mask = th_long_to_mask(get_node_feats(frag_pyg.x, frag_node_feat_idxs, "cc"))
            print(frag_node_mask.shape)
            smarts_list = smarts_df[smarts_df["prec_type"] == b_spec_prec_type_str][
                "smarts"
            ].tolist()
            atom_ids_list = []
            for smarts in tqdm(smarts_list):
                matches, remainders = get_substructure_from_smarts(mol_smiles, smarts)
                atom_ids_list += matches
                atom_ids_list += remainders

            atom_ids_list = list(set(atom_ids_list))
            print(len(atom_ids_list))
            """
            b_pred_mzs = []
            b_frag_node_ids = []
            b_smarts = []

            for i in range(frag_node_mask.shape[0]):
                all_atom_ids = th.arange(0, frag_node_mask.shape[1]).to(th.long)
                frag_atom_ids = all_atom_ids[frag_node_mask[i]].tolist()
                substructure_smarts, substructure_smiles, mw, substructure_formula = atom_ids_to_smarts(b_mol, frag_atom_ids)
                b_pred_mzs += [mw + H_MASS * j + PREC_TYPE_TO_MASS_DIFF[b_spec_prec_type_str] for j in range(-num_hs, num_hs + 1)]
                b_frag_node_ids += [i] * (2 * num_hs + 1)
                b_smarts.append([i, substructure_smarts, substructure_smiles, mw, substructure_formula, b_spec_prec_type_str])
            """
            exit()
            b_true_mzs = data["spec_mzs"][b_spec_mask]
            b_pred_mzs = th.tensor(b_pred_mzs)
            b_frag_node_ids = th.tensor(b_frag_node_ids).to(th.long)

            relative = True
            b_match_mask = calculate_match_mzs(
                b_true_mzs,
                b_pred_mzs,
                tolerance=10 * PPM,
                relative=relative,
                tolerance_min_mz=TOLERANCE_MIN_MZ,
            )

            b_pred_match_mask = th.any(b_match_mask, dim=0)
            matched_frag_node_id = th.unique(b_frag_node_ids[b_pred_match_mask])
            b_smarts_df = pd.DataFrame(
                b_smarts, columns=["node_id", "smarts", "smiles", "mw", "formula", "prec_type"]
            )
            b_smarts_df = b_smarts_df[b_smarts_df["node_id"].isin(matched_frag_node_id.tolist())]
