import argparse

import pandas as pd
import torch as th
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

from fragnnet.runner import init_dataloader, init_dataset, load_config
from fragnnet.utils.data_utils import mol_from_smiles
from fragnnet.utils.frag_utils import get_node_feats, th_long_to_mask
from fragnnet.utils.misc_utils import PPM, TOLERANCE_MIN_MZ
from fragnnet.utils.spec_utils import calculate_match_mzs


def create_fragments_by_reaction(mol):
    smarts_l = [
        "[*+0;!H0:0][*+0:1][*+0:2]>>([*+0:0]=[*+0:1].[*+0:2])",
        "[*+0;!H0:0]=[*+0:1][*+0:2]>>([*+0:0]#[*+0:1].[*+0:2])",
        "[*+0:0][*+0:1][*!H0+0:2]>>([*+0:0].[*+0:1]=[*+0:2])",
        "[*+0:0][*+0:1]=[*!H0+0:2]>>([*+0:0].[*+0:1]#[*+0:2])",
        "[*+0:0]1[*+0:1]=[*+0:2][*+0:3][*+0:4][*+0:5]1>>([*+0:0]=[*+0:1][*+0:2]=[*+0:3].[*+0:4]=[*+0:5])",
        "[*+0:0]1[*+0:1]=[*+0:2][*+0:3][*+0:4]=[*+0:5]1>>([*+0:0]=[*+0:1][*+0:2]=[*+0:3].[*+0:4]#[*+0:5])",
        "[*+0:0]=[*+0:1][*+0:2][*+0:3][*+0;!H0:4]>>([*+0:0][*+0:1]=[*+0:2].[*+0:3]=[*+0:4])",
        "[*+0;!H0:0][*+0:1][*+0:2][*+0;!H0:3]>>([*+0:0]=[*+0:1].[*+0:2]=[*+0:3])",
        "[*+0:0]1[*+0:1][*+0:2][*+0:3][*+0:4][*+0:5]1>>([*+0:0]=[*+0:1].[*+0:2]=[*+0:3].[*+0:4]=[*+0:5])",
        "[*+0:0]1=[*+0:1][*+0:2][*+0:3][*+0:4][*+0:5]1>>([*+0:0]#[*+0:1].[*+0:2]=[*+0:3].[*+0:4]=[*+0:5])",
        "[*+0:0]1=[*+0:1][*+0:2]=[*+0:3][*+0:4][*+0:5]1>>([*+0:0]#[*+0:1].[*+0:2]#[*+0:3].[*+0:4]=[*+0:5])",
        "[*+0:0]1=[*+0:1][*+0:2]=[*+0:3][*+0:4]=[*+0:5]1>>([*+0:0]#[*+0:1].[*+0:2]#[*+0:3].[*+0:4]#[*+0:5])",
        "[*+0;!H0:0][*+0:1]=[*+0:2]>>([*+0:0]=[*+0:1][*+0:2])",
        "[*+0;!H0:0][*+0:1]=[*+0:2][*+0:3]=[*+0:4]>>([*+0:0]=[*+0:1][*+0:2]=[*+0:3][*+0:4])",
        "[*+0;!H0:0][*+0:1]=[*+0:2][*+0:3]=[*+0:4][*+0:5]=[*+0:6]>>([*+0:0]=[*+0:1][*+0:2]=[*+0:3][*+0:4]=[*+0:5][*+0:6])",
        "[*:0]@[C:1](=[O:2])@[*:3]>>([*:0][*:3].[C-:1]#[O+:2])",
        "[*+0;!H0:0][*+0;!H0:1]>>([*+0:0]=[*+0:1])",
        "[*+0;!H0:0]=[*+0;!H0:1]>>([*+0:0]#[*+0:1])",
        "[*+0:0][*+0:1][*+0:2][*+0:3]>>([*+0:0][*+0:3].[*+0:1]=[*+0:2])",
        "[*+0:0][*+0:1]=[*+0:2][*+0:3]>>([*+0:0][*+0:3].[*+0:1]#[*+0:2])",
        "[*+0:0][C+0:1](=[O+0:2])[OH+0:3]>>([*+0:0].[O+0:2]=[C+0:1]=[O+0:3])",
        "[C+0;H1:0](=[O+0:1])[O+0:2][*+0:3]>>([O+0:1]=[C+0:0]=[O+0:2].[*+0:3])",
        "[*+0:0][C+0:1](=[O+0:2])[O+0:3][*+0:4]>>([O+0:2]=[C+0:1]=[O+0:3].[*+0:0][*+0:4])",
        "[N+0,O+0,S+0:0][*+0:1][*:2]>>([N+1,O+1,S+1:0]=[*+0:1].[*+0:2])",
        "[N+0,O+0,S+0:0]=[*+0:1][*:2]>>([N+1,O+1,S+1:0]#[*+0:1].[*+0:2])",
        "[*+0:0][*+0:1]>>([*+1:0].[*-1:1])",
        "[*+0:0]1=[*+0:1][*+0:2]=[*+0:3][*+0:4]=[*+0:5]1>>([*+0:0]1=[*+0:1][*+0:2]=[*+0:3][*+1:4]1.[*:5])",
    ]

    fragments_l = []
    for smarts in smarts_l:
        rxn = AllChem.ReactionFromSmarts(smarts)
        # Prepare three reactants as needed (replace with your actual molecules)

        if mol is None:
            raise ValueError("One of the molecules could not be created from SMILES.")
        products = rxn.RunReactants([mol])
        for product_set in products:
            for product in product_set:
                frags = Chem.GetMolFrags(product, asMols=True, sanitizeFrags=False)
                for frag in frags:
                    reorder_inverse = [
                        int(atom.GetProp("react_atom_idx")) for atom in frag.GetAtoms()
                    ]
                    reorder_inverse.sort()
                    reorder_inverse = tuple(reorder_inverse)
                    fragments_l.append(reorder_inverse)

    fragments_l = list(set(fragments_l))  # Remove duplicates
    fragments_l.sort(key=lambda x: len(x))  # Sort by fragment size
    return fragments_l


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--template_fp", type=str, default="config", help="Path to the config file"
    )
    parser.add_argument(
        "--custom_fp",
        type=str,
        default="config/exp_fraggnn_ma_mi/fraggnn_d3_ma_mi.yml",
        help="Path to the custom config file",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    config_d = load_config(args.template_fp, args.custom_fp)

    config_d["train_batch_size"] = args.batch_size
    config_d["eval_batch_size"] = args.batch_size

    config_d["frag_params"]["preload"] = False
    config_d["frag_params"]["preprocess"] = False
    config_d["spec_params"]["prec_type_str"] = True
    config_d["num_workers"] = args.num_workers
    config_d["dynamic_batch_sampler"] = None
    config_d["dynamic_batch_sampler"] = False

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

            # for smarts in tqdm(smarts_list):
            #       matches, remainders = get_substructure_from_smarts(mol_smiles, smarts)
            atom_ids_list = create_fragments_by_reaction(b_mol)

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
