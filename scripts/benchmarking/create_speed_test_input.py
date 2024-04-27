import os
import pandas as pd
import argparse

from rdkit import RDLogger
import frag_gnn.utils.data_utils as data_utils
from frag_gnn.utils.data_utils import par_apply_series
from frag_gnn.utils.formula_utils import get_elements_set, parse_formula
from frag_gnn.utils.frag_utils import ELEMENTS
from frag_gnn.utils.spec_utils import prec_type_mass_diff, PREC_TYPES

from frag_gnn.frag.compute_frags import (
	MAX_NUM_NODES
)

def get_args():
	
	parser = argparse.ArgumentParser()

 	#gen params
	parser.add_argument("--moldf_fp", type=str,default="data/misc/pubchem-10m-clean.zip", help="")
	parser.add_argument("--save_dp", type=str,default="data/benchmark/speed_benchmark", help="")
	parser.add_argument("--random_seed", type=int,default=42, help="")
	parser.add_argument("--num_sampled", type=int,default=5000, help="")
	args = parser.parse_args()
	return args

if __name__ == "__main__":

	RDLogger.DisableLog("rdApp.*")
	args = get_args()
	os.makedirs(args.save_dp, exist_ok=True)
	allowed_element_set = set(ELEMENTS)
 
	smiles_df = pd.read_csv(args.moldf_fp)
	#print(mol_df)
	mol_df = smiles_df.sample(n=args.num_sampled, random_state=args.random_seed)
	mol_df.columns = ["smiles"]
	mol_df["mol_id"]  = mol_df.index + 1

	mol_df.loc[:,"mol"] = par_apply_series(mol_df["smiles"],data_utils.mol_from_smiles)
	mol_df.loc[:,"inchikey_s"] = par_apply_series(mol_df["mol"],data_utils.mol_to_inchikey_s)
	mol_df.loc[:,"scaffold"] = par_apply_series(mol_df["mol"],data_utils.get_murcko_scaffold)
	mol_df.loc[:,"formula"] = par_apply_series(mol_df["mol"],data_utils.mol_to_formula)
	mol_df.loc[:,"inchi"] = par_apply_series(mol_df["mol"],data_utils.mol_to_inchi)
	mol_df.loc[:,"mw"] = par_apply_series(mol_df["mol"],lambda mol: data_utils.mol_to_mol_weight(mol,exact=False))
	mol_df.loc[:,"exact_mw"] = par_apply_series(mol_df["mol"],lambda mol: data_utils.mol_to_mol_weight(mol,exact=True))
	mol_df.loc[:,"num_atoms"] = par_apply_series(mol_df["mol"],data_utils.mol_to_num_atoms)
	mol_df.loc[:,"num_bonds"] = par_apply_series(mol_df["mol"],data_utils.mol_to_num_bonds)
	mol_df.loc[:,"charge"] = par_apply_series(mol_df["mol"],data_utils.mol_to_charge)
	mol_df.loc[:,"single_mol"] = par_apply_series(mol_df["mol"],data_utils.check_single_mol)
	mol_df.loc[:,"num_radicals"] = par_apply_series(mol_df["mol"],data_utils.mol_to_num_radicals)
	mol_df.loc[:,"elements_set"] = par_apply_series(mol_df["formula"], get_elements_set)
	mol_df.loc[:,"all_elements_allowed"] =  par_apply_series(mol_df["elements_set"],lambda elements_set: len(elements_set - allowed_element_set) == 0)
	
	mol_df = mol_df[mol_df['num_atoms'] <= MAX_NUM_NODES]
	mol_df = mol_df[mol_df['single_mol']]
	mol_df = mol_df[mol_df['num_radicals'] == 0]
	mol_df = mol_df[mol_df['charge'] == 0]
	mol_df = mol_df[mol_df["all_elements_allowed"]]

	mol_df.reset_index()
 
	spec_df = mol_df.copy()
	spec_df['spec_id'] = spec_df['mol_id']
	spec_df['group_id'] = spec_df['mol_id']
	spec_df['prec_type'] = '[M+H]+'
	
	spec_df['nce'] = [[10,20,30] for _ in range(len(spec_df))]
	spec_df['ace'] = [[10,20,30] for _ in range(len(spec_df))]
	spec_df["prec_mz"] = spec_df['exact_mw'] + prec_type_mass_diff('[M+H]+')

	print(f"> selected {len(mol_df)} mol after filter")
	print("> save mol df")
	mol_df = mol_df[['mol_id','mol','smiles','inchi']]
	mol_df.to_pickle(os.path.join(args.save_dp, "mol_df.pkl.gz"))
	print("> save spec df")
	spec_df = spec_df[['spec_id', 'group_id', 'mol_id','prec_type','nce', 'ace', 'prec_mz']]
	spec_df.to_pickle(os.path.join(args.save_dp, "spec_df.pkl.gz"))