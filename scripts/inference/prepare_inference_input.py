#!/usr/bin/env python3
"""
Prepare mol_df and spec_df for FragGNN inference from input lists.

This script takes lists of SMILES, precursor types, collision energies, and instrument types
and creates the necessary mol_df and spec_df files that can be used for FragGNN prediction.

Usage:
	python prepare_inference_input.py --smiles_list "CCO" "CC" --prec_types "[M+H]+" --ces 10.0 20.0 --inst_types "FT" --output_dir output/
"""

import argparse
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from itertools import product
from fragnnet.utils import data_utils
from fragnnet.utils.data_utils import par_apply_series
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_mol_df(smiles_list):
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
		'mol_id': range(len(smiles_list))
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
	mol_df.loc[:, "scaffold"] = par_apply_series(mol_df["mol"], data_utils.get_murcko_scaffold)
	mol_df.loc[:, "formula"] = par_apply_series(mol_df["mol"], data_utils.mol_to_formula)
	mol_df.loc[:, "inchi"] = par_apply_series(mol_df["mol"], data_utils.mol_to_inchi)
	mol_df.loc[:, "mw"] = par_apply_series(mol_df["mol"], lambda mol: data_utils.mol_to_mol_weight(mol, exact=False))
	mol_df.loc[:, "exact_mw"] = par_apply_series(mol_df["mol"], lambda mol: data_utils.mol_to_mol_weight(mol, exact=True))
	mol_df.loc[:, "num_atoms"] = par_apply_series(mol_df["mol"], data_utils.mol_to_num_atoms)
	mol_df.loc[:, "num_bonds"] = par_apply_series(mol_df["mol"], data_utils.mol_to_num_bonds)
	mol_df.loc[:, "charge"] = par_apply_series(mol_df["mol"], data_utils.mol_to_charge)
	mol_df.loc[:, "single_mol"] = par_apply_series(mol_df["mol"], data_utils.check_single_mol)
	mol_df.loc[:, "num_radicals"] = par_apply_series(mol_df["mol"], data_utils.mol_to_num_radicals)
	
	# Reset mol_id after filtering
	mol_df = mol_df.reset_index(drop=True)
	mol_df.loc[:, "mol_id"] = list(range(len(mol_df)))
	
	# Set mol_df index to mol_id for easier lookup
	mol_df = mol_df.set_index('mol_id')
	
	logger.info(f"Created mol_df with {len(mol_df)} molecules")
	return mol_df

def create_spec_df(mol_df, prec_types, ces, inst_types):
	"""
	Create spec_df from mol_df and experimental conditions.
	
	Args:
		mol_df (pd.DataFrame): Molecule dataframe
		prec_types (list): List of precursor types (e.g., "[M+H]+", "[M-H]-")
		ces (list): List of collision energies (floats)
		inst_types (list): List of instrument types (e.g., "FT", "QTOF")
		
	Returns:
		pd.DataFrame: spec_df with required columns
	"""
	logger.info(f"Creating spec_df for {len(mol_df)} molecules with {len(prec_types)} prec_types, {len(ces)} CEs, {len(inst_types)} inst_types...")
	
	# Create all combinations of molecules and experimental conditions
	spec_entries = []
	spec_id = 0
	group_id = 0
	
	for mol_id in mol_df.index:
		for prec_type, ce, inst_type in product(prec_types, ces, inst_types):
			# Determine CE unit based on instrument type
			ce_unit = get_ce_unit(inst_type)
			
			# Set NCE or ACE based on instrument type
			if ce_unit == "NCE":
				nce_value = float(ce)
				ace_value = np.nan
			else:  # eV
				nce_value = np.nan
				ace_value = float(ce)
			
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
				'ri': np.nan,       # Retention index not applicable
				'frag_mode': 'CID', # Default fragmentation mode
				'spec_type': 'MS2', # Default spectrum type
				'ion_mode': 'P',    # Positive mode default
				'col_gas': np.nan,  # Collision gas not specified
				'res': np.nan,      # Resolution not specified
				'notes': '',        # Empty notes
				'formula': mol_df.loc[mol_id, 'formula'],
				'inchikey': mol_df.loc[mol_id, 'inchikey_s'],
				'exact_mass': mol_df.loc[mol_id, 'exact_mw']
			})
			spec_id += 1
			group_id += 1
	
	spec_df = pd.DataFrame(spec_entries)
	
	# Compute precursor m/z from molecular mass and precursor type
	logger.info("Computing precursor m/z values...")
	spec_df = spec_df.merge(mol_df[["exact_mw"]], left_on="mol_id", right_index=True, how="inner")
	spec_df.loc[:, "prec_mz"] = spec_df.apply(
		lambda row: data_utils.infer_prec_mz(row), axis=1
	)
	spec_df = spec_df.drop(columns=["exact_mw"])
	
	logger.info(f"Created spec_df with {len(spec_df)} spectra")
	return spec_df

def validate_inputs(smiles_list, prec_types, ces, inst_types):
	"""
	Validate input parameters.
	
	Args:
		smiles_list (list): List of SMILES strings
		prec_types (list): List of precursor types
		ces (list): List of collision energies
		inst_types (list): List of instrument types
	"""
	if not smiles_list:
		raise ValueError("SMILES list cannot be empty")
	
	if not prec_types:
		raise ValueError("Precursor types list cannot be empty")
	
	if not ces:
		raise ValueError("Collision energies list cannot be empty")
	
	if not inst_types:
		raise ValueError("Instrument types list cannot be empty")
	
	# Validate collision energies are numeric
	for ce in ces:
		try:
			float(ce)
		except (ValueError, TypeError):
			raise ValueError(f"Invalid collision energy: {ce}. Must be numeric.")
	
	# Common precursor types
	common_prec_types = ["[M+H]+", "[M-H]-", "[M+Na]+", "[M+K]+", "[M+NH4]+", "[M-H2O+H]+"]
	for prec_type in prec_types:
		if prec_type not in common_prec_types:
			logger.warning(f"Uncommon precursor type: {prec_type}. Common types: {common_prec_types}")
	
	# Common instrument types and their CE units
	inst_type_ce_units = {
		"FT": "NCE",      # Orbitrap uses NCE (%)
		"QTOF": "eV",     # QTOF uses eV
		"IT": "NCE",      # Ion Trap typically uses NCE (%)
		"Q": "eV",        # Single quadrupole uses eV
		"TOF": "eV",      # TOF uses eV
		"QQQ": "eV"       # Triple quadrupole uses eV
	}
	
	common_inst_types = list(inst_type_ce_units.keys())
	for inst_type in inst_types:
		if inst_type not in common_inst_types:
			logger.warning(f"Uncommon instrument type: {inst_type}. Common types: {common_inst_types}")
		else:
			expected_unit = inst_type_ce_units[inst_type]
			logger.info(f"Instrument type '{inst_type}' expects collision energy in {expected_unit}")

def get_ce_unit(inst_type):
	"""
	Get the expected collision energy unit for an instrument type.
	
	Args:
		inst_type (str): Instrument type
		
	Returns:
		str: CE unit ("NCE" or "eV")
	"""
	inst_type_ce_units = {
		"FT": "NCE",      # Orbitrap uses NCE (%)
		"QTOF": "eV",     # QTOF uses eV
		"IT": "NCE",      # Ion Trap typically uses NCE (%)
		"Q": "eV",        # Single quadrupole uses eV
		"TOF": "eV",      # TOF uses eV
		"QQQ": "eV"       # Triple quadrupole uses eV
	}
	return inst_type_ce_units.get(inst_type, "NCE")

def save_config(smiles_list, prec_types, ces, inst_types, output_dir, prefix="", smiles_file=None):
	"""
	Save the configuration parameters to a JSON file.
	
	Args:
		smiles_list (list): List of SMILES strings
		prec_types (list): List of precursor types
		ces (list): List of collision energies
		inst_types (list): List of instrument types
		output_dir (str): Output directory
		prefix (str): File prefix
		smiles_file (str, optional): Path to SMILES file if used
	"""
	# Create list of all combinations
	combinations_list = []
	combo_id = 0
	
	for mol_idx, smiles in enumerate(smiles_list):
		for prec_type in prec_types:
			for ce in ces:
				for inst_type in inst_types:
					ce_unit = get_ce_unit(inst_type)
					combinations_list.append({
						"combination_id": combo_id,
						"precursor_type": prec_type,
						"collision_energy": ce,
						"collision_energy_unit": ce_unit,
						"instrument_type": inst_type
					})
					combo_id += 1
	
	config = {
		"timestamp": datetime.now().isoformat(),
		"input_parameters": {
			"num_smiles": len(smiles_list),
			"smiles_list": smiles_list,
			"smiles_file": smiles_file,
			"precursor_types": prec_types,
			"collision_energies": ces,
			"instrument_types": inst_types,
			"output_directory": output_dir,
			"file_prefix": prefix
		},
		"combinations_summary": {
			"total_molecules": len(smiles_list),
			"total_precursor_types": len(prec_types),
			"total_collision_energies": len(ces),
			"total_instrument_types": len(inst_types),
			"total_combinations": len(combinations_list)
		},
		"combinations": combinations_list,
		"output_files": {
			"mol_df": f"{prefix + '_' if prefix else ''}mol_df.pkl.gz",
			"spec_df": f"{prefix + '_' if prefix else ''}spec_df.pkl.gz",
			"config": f"{prefix + '_' if prefix else ''}config.json"
		}
	}
	
	# Save config file
	config_prefix = prefix + "_" if prefix else ""
	config_path = os.path.join(output_dir, f"{config_prefix}config.json")
	
	with open(config_path, 'w') as f:
		json.dump(config, f, indent=2, ensure_ascii=False)
	
	logger.info(f"Configuration saved to {config_path}")
	return config_path

if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Prepare mol_df and spec_df for FragGNN inference",
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog="""
Examples:
	# Single molecule, single condition
	python prepare_inference_input.py --smiles_list "CCO" --prec_types "[M+H]+" --ces 20.0 --inst_types "FT" --output_dir output/
	
	# Multiple molecules, multiple conditions
	python prepare_inference_input.py --smiles_list "CCO" "CC(=O)O" "CCN" --prec_types "[M+H]+" "[M-H]-" --ces 10.0 20.0 30.0 --inst_types "FT" "QTOF" --output_dir output/
	
	# From file
	python prepare_inference_input.py --smiles_file smiles.txt --prec_types "[M+H]+" --ces 20.0 --inst_types "FT" --output_dir output/
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
		"--prec_types", 
		nargs="+", 
		required=True,
		help="List of precursor types (e.g., [M+H]+ [M-H]-)"
	)
	parser.add_argument(
		"--ces", 
		nargs="+", 
		type=float,
		required=True,
		help="List of collision energies. Units depend on instrument type: NCE (%%) for FT/IT, eV for QTOF/Q/TOF/QQQ"
	)
	parser.add_argument(
		"--inst_types", 
		nargs="+", 
		required=True,
		help="List of instrument types (FT=NCE%%, QTOF=eV, IT=NCE%%, Q=eV, TOF=eV, QQQ=eV)"
	)
	parser.add_argument(
		"--output_dir", 
		type=str, 
		required=True,
		help="Output directory for mol_df.pkl.gz and spec_df.pkl.gz"
	)
	parser.add_argument(
		"--prefix", 
		type=str, 
		default="",
		help="Prefix for output files (default: mol_df.pkl.gz, spec_df.pkl.gz)"
	)
	
	args = parser.parse_args()
	
	# Read SMILES
	if args.smiles_list:
		smiles_list = args.smiles_list
	else:
		logger.info(f"Reading SMILES from {args.smiles_file}")
		with open(args.smiles_file, 'r') as f:
			smiles_list = [line.strip() for line in f if line.strip()]
	
	# Validate inputs
	validate_inputs(smiles_list, args.prec_types, args.ces, args.inst_types)
	
	logger.info(f"Input summary:")
	logger.info(f"  - {len(smiles_list)} SMILES")
	logger.info(f"  - {len(args.prec_types)} precursor types: {args.prec_types}")
	logger.info(f"  - {len(args.ces)} collision energies: {args.ces}")
	logger.info(f"  - {len(args.inst_types)} instrument types: {args.inst_types}")
	logger.info(f"  - Total combinations: {len(smiles_list) * len(args.prec_types) * len(args.ces) * len(args.inst_types)}")
	
	# Create output directory
	os.makedirs(args.output_dir, exist_ok=True)
	
	# Create mol_df
	mol_df = create_mol_df(smiles_list)
	
	# Create spec_df
	spec_df = create_spec_df(mol_df, args.prec_types, args.ces, args.inst_types)
	
	# Save files
	prefix = args.prefix + "_" if args.prefix else ""
	mol_df_path = os.path.join(args.output_dir, f"{prefix}mol_df.pkl.gz")
	spec_df_path = os.path.join(args.output_dir, f"{prefix}spec_df.pkl.gz")
	
	logger.info(f"Saving mol_df to {mol_df_path}")
	mol_df.to_pickle(mol_df_path)
	
	logger.info(f"Saving spec_df to {spec_df_path}")
	spec_df.to_pickle(spec_df_path)
	
	# Save configuration
	config_path = save_config(
		smiles_list=smiles_list,
		prec_types=args.prec_types,
		ces=args.ces,
		inst_types=args.inst_types,
		output_dir=args.output_dir,
		prefix=args.prefix,
		smiles_file=args.smiles_file if hasattr(args, 'smiles_file') and args.smiles_file else None
	)
	
	# Print summary
	logger.info("Summary:")
	logger.info(f"  - mol_df: {len(mol_df)} molecules")
	logger.info(f"  - spec_df: {len(spec_df)} spectra")
	logger.info(f"  - Output directory: {args.output_dir}")
	logger.info(f"  - Configuration saved: {config_path}")
	
	# Print first few entries for verification
	logger.info("\nFirst 3 molecules in mol_df:")
	print(mol_df[['smiles', 'formula', 'exact_mw']].head(3))
	
	logger.info("\nFirst 3 spectra in spec_df:")
	print(spec_df[['spec_id', 'mol_id', 'prec_type', 'nce', 'inst_type', 'prec_mz']].head(3))