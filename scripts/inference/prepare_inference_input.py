"""
Prepare mol_df and spec_df for FragGNN inference from input lists.

This script takes lists of SMILES, precursor types, collision energies, and instrument types
and creates the necessary mol_df and spec_df files that can be used for FragGNN prediction.

Usage:
	python prepare_inference_input.py --smiles_list "CCO" "CC" --prec_types "[M+H]+" --ces 10.0 20.0 --inst_types "FT" --output_dir output/
"""

import argparse
from sys import prefix
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from fragnnet.utils import data_utils
from fragnnet.utils.data_utils import par_apply_series
from fragnnet.utils.misc_utils import booltype
import fragnnet.utils.formula_utils as formula_utils
import fragnnet.utils.frag_utils as frag_utils
import logging
from fragnnet.frag.compute_frags import (
	MAX_NUM_NODES
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_mol_df(smiles_list, ids_list):
	"""
	Create mol_df from a list of SMILES strings.
	
	Args:
		smiles_list (list): List of SMILES strings
		
	Returns:
		pd.DataFrame: mol_df with required columns
	"""
	logger.info(f"Creating mol_df for {len(smiles_list)} SMILES...")
	
	# Create initial DataFrame
	mol_df = pd.DataFrame({
		'smiles': smiles_list,
		'mol_id': range(len(smiles_list)),
		'input_mol_id': np.array([str(i) for i in range(len(smiles_list))]) if ids_list is None else np.where(pd.isna(ids_list), np.nan, [str(x) for x in ids_list])
	})

	# Convert SMILES to RDKit molecules
	logger.info("Converting SMILES to RDKit molecules...")
	mol_df.loc[:, "mol"] = par_apply_series(mol_df["smiles"], data_utils.mol_from_smiles)
	
	# Remove invalid molecules
	initial_count = len(mol_df)
	mol_df = mol_df[mol_df["mol"].notna()]
	valid_count = len(mol_df)
	if initial_count != valid_count:
		logger.warning(f"Removed {initial_count - valid_count} invalid SMILES")
	
	# Generate molecular properties
	logger.info("Computing molecular properties...")
	mol_df.loc[:, "inchikey_s"] = par_apply_series(mol_df["mol"], data_utils.mol_to_inchikey_s)
	mol_df.loc[:, "formula"] = par_apply_series(mol_df["mol"], data_utils.mol_to_formula)
	mol_df.loc[:, "mw"] = par_apply_series(mol_df["mol"], lambda mol: data_utils.mol_to_mol_weight(mol, exact=False))
	mol_df.loc[:, "exact_mw"] = par_apply_series(mol_df["mol"], lambda mol: data_utils.mol_to_mol_weight(mol, exact=True))
	mol_df.loc[:, "num_atoms"] = par_apply_series(mol_df["mol"], data_utils.mol_to_num_atoms)
	mol_df.loc[:, "num_bonds"] = par_apply_series(mol_df["mol"], data_utils.mol_to_num_bonds)
	mol_df.loc[:, "charge"] = par_apply_series(mol_df["mol"], data_utils.mol_to_charge)
	mol_df.loc[:, "single_mol"] = par_apply_series(mol_df["mol"], data_utils.check_single_mol)
	mol_df.loc[:, "num_radicals"] = par_apply_series(mol_df["mol"], data_utils.mol_to_num_radicals)
	# Heavy atoms = non-hydrogen atoms
	mol_df.loc[:, "num_heavy_atoms"] = par_apply_series(mol_df["mol"], lambda m: int(m.GetNumHeavyAtoms()) if m is not None else np.nan)
	mol_df.loc[:, "num_bonds"] = par_apply_series(mol_df["mol"], lambda m: int(m.GetNumBonds()) if m is not None else np.nan)

 	# filter by out_set_elements
	allowed_elements = set(frag_utils.ELEMENTS)
	mol_df.loc[:, 'num_out_set_elements'] = mol_df['formula'].apply(lambda x: len(set(list(formula_utils.parse_formula(x).keys())) - allowed_elements))

	# Filter out molecules based on quality criteria
	logger.info("Filtering molecules based on quality criteria...")
	initial_count_filter = len(mol_df)
	
	# Drop molecules that are not single molecules, have radicals, or have charges
	mol_df = mol_df[
		(mol_df["single_mol"] == True) & 
		(mol_df["num_radicals"] == 0) & 
		(mol_df["charge"] == 0) & 
		(mol_df["num_atoms"] <= 128) & 
		(mol_df["num_out_set_elements"] == 0)
	]
	
	filtered_count = len(mol_df)
	if initial_count_filter != filtered_count:
		logger.warning(f"Filtered out {initial_count_filter - filtered_count} molecules due to quality criteria")
		logger.info(f"  - Kept only single molecules with no radicals and no charge")
	
	# Reset mol_id after filtering
	mol_df = mol_df.reset_index(drop=True)
	mol_df.loc[:, "mol_id"] = list(range(len(mol_df)))
		
	logger.info(f"Created mol_df with {len(mol_df)} molecules")
	return mol_df

def create_spec_df(mol_df, config_d):
	"""
	Create spec_df from mol_df and configuration dictionary.
	
	Args:
		mol_df (pd.DataFrame): Molecule dataframe
		config_d (dict): Configuration dictionary containing combinations of experimental conditions
		
	Returns:
		pd.DataFrame: spec_df with required columns
	"""
	combinations = config_d["combinations"]
	num_molecules = len(mol_df)
	num_combinations = len(combinations)
	merged_spectra = config_d['input_parameters']['merged_spectra']

	logger.info(f"Creating spec_df for {num_molecules} molecules with {num_combinations} experimental combinations...")
	
	# Create all combinations of molecules and experimental conditions
	spec_entries = []
	spec_id = 0
	group_id = 0
	
	for mol_id in mol_df['mol_id']:
		for config_combo in config_d["combinations"]:
			prec_type = config_combo["prec_type"]
			inst_type = config_combo["inst_type"]
			# Choose appropriate collision energy based on instrument type
			if config_combo["ce_unit"] == "NCE":
				nce_value = config_combo["ce"]
				ace_value = np.nan
			else:
				nce_value = np.nan
				ace_value = config_combo["ce"]

			spec_entries.append({
				'spec_id': spec_id,
				'mol_id': mol_id,
				'prec_type': prec_type,
					'nce': nce_value,
					'ace': ace_value,
					'inst_type': inst_type,
					'group_id': group_id,
					'dset': 'inference',
					'dset_spec_id': spec_id,
					'prec_mz': np.nan,  # Will be computed from molecular mass and precursor type
					'peaks': [],        # Empty peaks list for prediction
					'frag_mode': 'CID', # Default fragmentation mode
					'spec_type': 'MS2', # Default spectrum type
					'ion_mode': 'P',    # Positive mode default
					'formula': mol_df.loc[mol_id, 'formula'],
					'inchikey': mol_df.loc[mol_id, 'inchikey_s'],
					'exact_mass': mol_df.loc[mol_id, 'exact_mw']
				})
			spec_id += 1
			group_id += 1
	
	spec_df = pd.DataFrame(spec_entries)
	
	if merged_spectra:
		# Reset group_id based on prec_type, inst_type, and mol groups
		logger.info("Adjusting group_ids for merged spectra based on prec_type, inst_type, and molecule groups...")
		
		# Create a mapping for group assignment
		group_mapping = {}
		current_group_id = 0
		
		# Group by prec_type and inst_type combinations
		for (prec_type, inst_type), group_data in spec_df.groupby(['prec_type', 'inst_type']):
			group_key = (prec_type, inst_type)
			if group_key not in group_mapping:
				group_mapping[group_key] = current_group_id
				current_group_id += 1
		
		# Assign new group_ids
		spec_df.loc[:, 'group_id'] = spec_df.apply(
			lambda row: group_mapping[(row['prec_type'], row['inst_type'])], axis=1
		)
		
		logger.info(f"Created {len(group_mapping)} unique groups based on prec_type and inst_type combinations")
	
	# Compute precursor m/z from molecular mass and precursor type
	logger.info("Computing precursor m/z values...")
	spec_df = spec_df.merge(mol_df[["exact_mw"]], left_on="mol_id", right_index=True, how="inner")
	spec_df.loc[:, "prec_mz"] = spec_df.apply(
		lambda row: data_utils.infer_prec_mz(row), axis=1
	)
	spec_df = spec_df.drop(columns=["exact_mw"])
	
	logger.info(f"Created spec_df with {len(spec_df)} spectra")
	return spec_df

def validate_inputs(smiles_list, prec_types, ces, nces, inst_types):
	"""
	Validate input parameters.
	
	Args:
		smiles_list (list): List of SMILES strings
		prec_types (list): List of precursor types
		ces (list): List of collision energies (eV)
		nces (list): List of normalized collision energies (%)
		inst_types (list): List of instrument types
	"""
	if not smiles_list:
		raise ValueError("SMILES list cannot be empty")
	
	if not prec_types:
		raise ValueError("Precursor types list cannot be empty")
	
	if not ces:
		raise ValueError("Collision energies (eV) list cannot be empty")
		
	if not nces:
		raise ValueError("Normalized collision energies (NCE) list cannot be empty")
	
	if not inst_types:
		raise ValueError("Instrument types list cannot be empty")
	
	# Validate collision energies are numeric
	for ce in ces:
		try:
			float(ce)
		except (ValueError, TypeError):
			raise ValueError(f"Invalid collision energy: {ce}. Must be numeric.")
			
	for nce in nces:
		try:
			float(nce)
		except (ValueError, TypeError):
			raise ValueError(f"Invalid NCE value: {nce}. Must be numeric.")
	
	# Log CE unit expectations for each instrument type
	for inst_type in inst_types:
		if inst_type == "FT":
			logger.info(f"Instrument type '{inst_type}' will use NCE (%) values: {nces}")
		else:
			logger.info(f"Instrument type '{inst_type}' will use ACE (eV) values: {ces}")

def get_config(smiles_list, prec_types, ces, nces, inst_types, output_dir, smiles_file=None, merged_spectra=False):
	"""
	Save the configuration parameters to a JSON file.
	
	Args:
		smiles_list (list): List of SMILES strings
		prec_types (list): List of precursor types
		ces (list): List of collision energies (eV)
		nces (list): List of normalized collision energies (%)
		inst_types (list): List of instrument types
		output_dir (str): Output directory
		smiles_file (str, optional): Path to SMILES file if used
		merged_spectra (bool): Whether to generate merged spectra
	"""
	# Create list of all combinations
	combinations_list = []
	combo_id = 0
	

	for prec_type in prec_types:
		for inst_type in inst_types:
			# Choose appropriate collision energy values based on instrument type
			if inst_type == "FT":  # Orbitrap uses NCE
				energy_values = nces
				ce_unit = "NCE"
			else:  # Other instruments use ACE (eV)
				energy_values = ces
				ce_unit = "eV"
			
			for energy in energy_values:
				combinations_list.append({
					"id": combo_id,
					"prec_type": prec_type,
					"ce": energy,
					"ce_unit": ce_unit,
					"inst_type": inst_type
				})
				combo_id += 1
	
	config = {
		"timestamp": datetime.now().isoformat(),
		"input_parameters": {
			"num_smiles": len(smiles_list),
			"smiles_file": smiles_file,
			"precursor_types": prec_types,
			"collision_energies_eV": ces,
			"normalized_collision_energies_pct": nces,
			"instrument_types": inst_types,
			"output_directory": output_dir,
			"merged_spectra": merged_spectra
		},
		"combinations": combinations_list,
		"output_files": {
			"mol_df": f"{output_dir}/mol_df.pkl.gz",
			"spec_df": f"{output_dir}/spec_df.pkl.gz",
			"config": f"{output_dir}/config.json"
		}
	}

	return config

if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Prepare mol_df and spec_df for FragGNN inference",
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog="""
Examples:
	# Single molecule, single condition
	python prepare_inference_input.py --smiles_list "CCO" --prec_types "[M+H]+" --ces 20.0 --nces 25.0 --inst_types "FT" --output_dir output/
	
	# Multiple molecules, multiple conditions
	python prepare_inference_input.py --smiles_list "CCO" "CC(=O)O" "CCN" --prec_types "[M+H]+" "[M-H]-" --ces 10.0 20.0 30.0 --nces 15.0 25.0 35.0 --inst_types "FT" "QTOF" --output_dir output/
	
	# Merged spectra (one entry per molecule)
	python prepare_inference_input.py --smiles_list "CCO" "CC(=O)O" --prec_types "[M+H]+" --ces 20.0 --nces 25.0 --inst_types "QTOF" "FT" --merged_spectra true --output_dir output/
	
	# From file
	python prepare_inference_input.py --smiles_file smiles.txt --prec_types "[M+H]+" --ces 20.0 --nces 25.0 --inst_types "FT" --output_dir output/
		"""
	)
	
	# Input options
	group = parser.add_mutually_exclusive_group(required=True)
	group.add_argument(
		"--smiles_list", 
		nargs="+", 
		help="List of SMILES strings"
	)
	group.add_argument(
		"--smiles_file", 
		type=str, 
		help="File containing SMILES strings (one per line)"
	)
	parser.add_argument(
		"--smiles_colname", 
		type=str, 
		help="Column name containing SMILES strings"
	)
	parser.add_argument(
		"--id_colname", 
		type=str, 
		help="Column name containing mol IDs"
	)
	parser.add_argument(
		"--prec_types", 
		nargs="+", 
		required=True,
		choices=["[M+H]+", "[M-H]-"], # to do update this list if needed
		help="List of precursor types (e.g., [M+H]+ [M-H]-)"
	)
	parser.add_argument(
		"--ces", 
		nargs="+", 
		type=float,
		required=True,
		help="List of collision energies (eVs) without eVs, used by QToF"
	)
	parser.add_argument(
		"--nces", 
		nargs="+", 
		type=float,
		required=True,
		help="List of NCE values without % (e.g., 10.0 20.0 30.0), used by Orbitrap and Ion Trap"
	)
	parser.add_argument(
		"--inst_types", 
		nargs="+", 
		required=True,
		choices=["FT", "QTOF", "IT", "Q", "TOF", "QQQ"], # to do update this list if needed
		help="List of instrument types (e.g., FT QTOF IT)"
	)
	parser.add_argument(
		"--output_dir", 
		type=str, 
		required=True,
		help="Output directory for mol_df.pkl.gz and spec_df.pkl.gz"
	)
	parser.add_argument(
		"--merged_spectra", 
		type=booltype,
		default=False,
		help="Generate merged spectra entries (one per molecule) instead of individual condition entries. Use true/false, yes/no, 1/0."
	)	
	args = parser.parse_args()
	
	# Read SMILES
	ids_list = None
	if args.smiles_list:
		smiles_list = args.smiles_list
	else:
		logger.info(f"Reading SMILES from {args.smiles_file}")
		if args.smiles_file.endswith('.smi') or args.smiles_file.endswith('.smiles') or args.smiles_file.endswith('.txt'):
			with open(args.smiles_file, 'r') as f:
				smiles_list = [line.strip() for line in f if line.strip()]
		elif args.smiles_file.endswith('.csv') or args.smiles_file.endswith('.tsv'):
			sep = ',' if args.smiles_file.endswith('.csv') else '\t'
			smiles_df = pd.read_csv(args.smiles_file, sep=sep)
			if not args.smiles_colname or args.smiles_colname not in smiles_df.columns:
				raise ValueError(f"Column name for SMILES not specified or not found in CSV columns: {smiles_df.columns.tolist()}")
			smiles_list = smiles_df[args.smiles_colname].astype(str).tolist()
			if args.id_colname and args.id_colname in smiles_df.columns:
				ids_list = smiles_df[args.id_colname].astype(str).tolist()
				if len(ids_list) != len(smiles_list):
					raise ValueError(f"Length of IDs list ({len(ids_list)}) does not match length of SMILES list ({len(smiles_list)})")

	# Validate inputs
	validate_inputs(smiles_list, args.prec_types, args.ces, args.nces, args.inst_types)
	
	logger.info(f"Input summary:")
	logger.info(f"  - {len(smiles_list)} SMILES")
	logger.info(f"  - {len(args.prec_types)} precursor types: {args.prec_types}")
	logger.info(f"  - {len(args.ces)} collision energies (eVs) for QToF: {args.ces}")
	logger.info(f"  - {len(args.nces)} collision energies (NCE) for Orbitrap (FT): {args.nces}")
	logger.info(f"  - {len(args.inst_types)} instrument types: {args.inst_types}")
	
	# Create output directory
	os.makedirs(args.output_dir, exist_ok=True)
		
	# Save configuration
	config_d = get_config(
		smiles_list=smiles_list,
		prec_types=args.prec_types,
		ces=args.ces,
		nces=args.nces,
		inst_types=args.inst_types,
		output_dir=args.output_dir,
		smiles_file=args.smiles_file if hasattr(args, 'smiles_file') and args.smiles_file else None,
		merged_spectra=args.merged_spectra
	)
	logger.info(f"  - Meta combinations for each input molecule: {len(config_d['combinations'])}")
	
 	# Save config file
	config_path = os.path.join(args.output_dir, "config.json")

	with open(config_path, 'w') as f:
		json.dump(config_d, f, indent=2, ensure_ascii=False)
	
	logger.info(f"Configuration saved to {config_path}")

	# Create mol_df
	mol_df = create_mol_df(smiles_list, ids_list)
	# Create spec_df
	spec_df = create_spec_df(mol_df, config_d)
 
	# Save files
	mol_df_path = os.path.join(args.output_dir, f"mol_df.pkl.gz")
	spec_df_path = os.path.join(args.output_dir, f"spec_df.pkl.gz")

	logger.info(f"Saving mol_df to {mol_df_path}")
	mol_df.to_pickle(mol_df_path)
	
	logger.info(f"Saving spec_df to {spec_df_path}")
	spec_df.to_pickle(spec_df_path)

 
	# Print summary
	logger.info("Summary:")
	logger.info(f"  - mol_df: {len(mol_df)} molecules")
	logger.info(f"  - spec_df: {len(spec_df)} spectra")
	logger.info(f"  - Output directory: {args.output_dir}")
	logger.info(f"  - Configuration saved: {config_path}")
	
	# Print first few entries for verification
	logger.info("\nFirst 5 molecules in mol_df:")
	print(mol_df.head(5))

	# Heavy atom statistics
	if "num_heavy_atoms" in mol_df.columns and len(mol_df) > 0:
		stats = mol_df["num_heavy_atoms"].describe()
		logger.info("\nHeavy atom count stats (num_heavy_atoms):")
		logger.info(f"  count: {int(stats['count'])}")
		logger.info(f"  mean:  {stats['mean']:.2f}")
		logger.info(f"  std:   {stats['std']:.2f}")
		logger.info(f"  min:   {int(stats['min'])}")
		logger.info(f"  25%:   {stats['25%']:.2f}")
		logger.info(f"  50%:   {stats['50%']:.2f}")
		logger.info(f"  75%:   {stats['75%']:.2f}")
		logger.info(f"  max:   {int(stats['max'])}")

	# num_bonds statistics
	if "num_bonds" in mol_df.columns and len(mol_df) > 0:
		stats_bonds = mol_df["num_bonds"].describe()
		logger.info("\nBond count stats (num_bonds):")
		logger.info(f"  count: {int(stats_bonds['count'])}")
		logger.info(f"  mean:  {stats_bonds['mean']:.2f}")
		logger.info(f"  std:   {stats_bonds['std']:.2f}")
		logger.info(f"  min:   {int(stats_bonds['min'])}")
		logger.info(f"  25%:   {stats_bonds['25%']:.2f}")
		logger.info(f"  50%:   {stats_bonds['50%']:.2f}")
		logger.info(f"  75%:   {stats_bonds['75%']:.2f}")
		logger.info(f"  max:   {int(stats_bonds['max'])}")
	
	logger.info("\nFirst 5 spectra in spec_df:")
	print(spec_df.head(5))