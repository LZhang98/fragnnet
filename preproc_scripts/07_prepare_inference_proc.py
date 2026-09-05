import argparse
import itertools
import os

import numpy as np
import pandas as pd
import yaml

import fragnnet.utils.data_utils as data_utils
import fragnnet.utils.frag_utils as frag_utils
from fragnnet.utils.formula_utils import PREC_TYPE_TO_MASS_DIFF
from fragnnet.utils.misc_utils import deep_update


DEFAULT_PREC_TYPES = ["[M+H]+"]
DEFAULT_INST_TYPES = ["FT"]
DEFAULT_FRAG_MODES = ["HCD"]
DEFAULT_ION_MODES = ["P"]
DEFAULT_ACE_VALUES = [20.0, 40.0, 60.0]
SUPPORTED_ELEMENTS = set(frag_utils.ELEMENT_TO_VE)

def load_spec_params(config_fp):
	if config_fp is None:
		return {}
	with open(config_fp, encoding="utf-8") as config_file:
		config = yaml.safe_load(config_file) or {}
	template_fp = config.pop("template_fp", "config/template.yml")
	if not os.path.isabs(template_fp):
		config_dir = os.path.dirname(os.path.abspath(config_fp))
		candidates = [
			os.path.join(config_dir, template_fp),
			os.path.join(config_dir, os.pardir, os.pardir, template_fp),
		]
		template_fp = next((path for path in candidates if os.path.isfile(path)), candidates[0])
	if os.path.isfile(template_fp):
		with open(template_fp, encoding="utf-8") as template_file:
			template = yaml.safe_load(template_file) or {}
		config = deep_update(template, config)
	return config.get("spec_params", {})


def canonicalize_smiles(smiles):
	return data_utils.mol_to_smiles(data_utils.mol_from_smiles(smiles))


def load_smiles_input(smiles_input):
	if len(smiles_input) == 1 and os.path.isfile(smiles_input[0]):
		input_fp = smiles_input[0]
		extension = os.path.splitext(input_fp)[1].lower()
		if extension == ".csv":
			input_df = pd.read_csv(input_fp)
		elif extension in [".pkl", ".pickle"]:
			input_df = pd.read_pickle(input_fp)
		elif extension in [".pq", ".parquet"]:
			input_df = pd.read_parquet(input_fp)
		else:
			raise ValueError("SMILES file must be CSV, pickle, or Parquet")
		if not isinstance(input_df, pd.DataFrame):
			raise TypeError(f"Expected a pandas DataFrame in {input_fp}")
		if "canonical_smiles" in input_df.columns: # TODO: kinda awk
			input_df.rename(columns={"canonical_smiles": "smiles"}, inplace=True)
		if "smiles" not in input_df.columns:
			raise ValueError(f"Input file {input_fp} must contain a 'smiles' column")
		smiles = input_df["smiles"].dropna().astype(str).tolist()
	else:
		if any(os.path.isfile(value) for value in smiles_input):
			raise ValueError("Provide either one SMILES dataframe file or a list of SMILES")
		smiles = smiles_input
	return list(dict.fromkeys(smiles))


def make_mol_df(smiles_input):
	
	mol_df = pd.DataFrame({"smiles": load_smiles_input(smiles_input)})
	mol_df["mol"] = data_utils.par_apply_series(mol_df["smiles"], data_utils.mol_from_smiles)
	mol_df["smiles"] = data_utils.par_apply_series(mol_df["mol"], data_utils.mol_to_smiles)
	mol_df = mol_df.dropna(subset=["mol", "smiles"])
	mol_df = mol_df.drop_duplicates(subset=["smiles"]).sort_values("smiles").reset_index(drop=True)
	initial_count = len(mol_df)
	mol_df = mol_df.loc[
		mol_df["mol"].apply(
			lambda mol: all(atom.GetSymbol() in SUPPORTED_ELEMENTS for atom in mol.GetAtoms())
		)
	]
	mol_df = mol_df.loc[mol_df["mol"].apply(data_utils.mol_to_num_bonds) > 0]
	print(f"> removed {initial_count - len(mol_df)} molecules with unsupported atoms or no bonds")
	mol_df = mol_df.reset_index(drop=True)
	mol_df["mol_id"] = np.arange(mol_df.shape[0])
	mol_df["inchikey_s"] = data_utils.par_apply_series(mol_df["mol"], data_utils.mol_to_inchikey_s)
	mol_df["scaffold"] = data_utils.par_apply_series(mol_df["mol"], data_utils.get_murcko_scaffold)
	mol_df["formula"] = data_utils.par_apply_series(mol_df["mol"], data_utils.mol_to_formula)
	mol_df["inchi"] = data_utils.par_apply_series(mol_df["mol"], data_utils.mol_to_inchi)
	mol_df["mw"] = data_utils.par_apply_series(mol_df["mol"], lambda mol: data_utils.mol_to_mol_weight(mol, exact=False))
	mol_df["exact_mw"] = data_utils.par_apply_series(mol_df["mol"], lambda mol: data_utils.mol_to_mol_weight(mol, exact=True))
	mol_df["num_atoms"] = data_utils.par_apply_series(mol_df["mol"], data_utils.mol_to_num_atoms)
	mol_df["num_bonds"] = data_utils.par_apply_series(mol_df["mol"], data_utils.mol_to_num_bonds)
	mol_df["charge"] = data_utils.par_apply_series(mol_df["mol"], data_utils.mol_to_charge)
	mol_df["single_mol"] = data_utils.par_apply_series(mol_df["mol"], data_utils.check_single_mol)
	mol_df["num_radicals"] = data_utils.par_apply_series(mol_df["mol"], data_utils.mol_to_num_radicals)
	return mol_df


def load_mol_df(input_values):
	if len(input_values) == 1 and os.path.splitext(input_values[0])[1].lower() in [".pkl", ".pickle"]:
		input_df = pd.read_pickle(input_values[0])
		if {"mol_id", "smiles", "mol"}.issubset(input_df.columns):
			if input_df["mol_id"].duplicated().any():
				raise ValueError("Input molecule dataframe contains duplicate mol_id values")
			return input_df.copy().reset_index(drop=True)
	return make_mol_df(input_values)


def make_spec_df(mol_df, dset, prec_types, inst_types, frag_modes, ion_modes,
				 ace_values):
	rows = []
	group_id = 0
	for mol_row in mol_df.itertuples(index=False):
		mol_id = mol_row.mol_id
		for prec_type, inst_type, frag_mode, ion_mode in itertools.product(
				prec_types, inst_types, frag_modes, ion_modes):
			for ace in ace_values:
				rows.append({
					"spec_id": len(rows),
					"mol_id": mol_id,
					"prec_type": prec_type,
					"inst_type": inst_type,
					"frag_mode": frag_mode,
					"spec_type": "MS2",
					"ion_mode": ion_mode,
					"dset": dset,
					"dset_spec_id": f"{dset}_{len(rows)}",
					"ace": ace,
					"prec_mz": mol_row.exact_mw + PREC_TYPE_TO_MASS_DIFF[prec_type],
					"peaks": [(1.0, 1.0)],
					"group_id": group_id,
				})
			group_id += 1
	return pd.DataFrame(rows)


def main(args):
	spec_params = load_spec_params(args.config_fp)
	prec_types = args.prec_types or spec_params.get("prec_types") or DEFAULT_PREC_TYPES
	inst_types = args.inst_types or spec_params.get("inst_types") or DEFAULT_INST_TYPES
	frag_modes = args.frag_modes or DEFAULT_FRAG_MODES
	ion_modes = args.ion_modes or DEFAULT_ION_MODES
	for prec_type in prec_types:
		if prec_type not in PREC_TYPE_TO_MASS_DIFF:
			raise ValueError(f"Unsupported precursor type: {prec_type}")

	ace_values = DEFAULT_ACE_VALUES

	mol_df = load_mol_df(args.input)
	spec_df = make_spec_df(
		mol_df, args.dset, prec_types, inst_types, frag_modes, ion_modes,
		ace_values)
	os.makedirs(args.output_dp, exist_ok=True)
	spec_fp = os.path.join(args.output_dp, "spec_df.pkl")
	mol_fp = os.path.join(args.output_dp, "mol_df.pkl")
	spec_df.to_pickle(spec_fp)
	mol_df.to_pickle(mol_fp)
	print(f"> saved {len(mol_df)} molecules to {mol_fp}")
	print(f"> saved {len(spec_df)} spectra to {spec_fp}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--input", "-i", nargs="+", required=True)
	parser.add_argument("--output_dp", "-o", default="data/proc/inference")
	parser.add_argument("--dset", default="inference")
	parser.add_argument("--config_fp")
	parser.add_argument("--prec_types", nargs="+")
	parser.add_argument("--inst_types", nargs="+")
	parser.add_argument("--frag_modes", nargs="+", default=DEFAULT_FRAG_MODES)
	parser.add_argument("--ion_modes", nargs="+", default=DEFAULT_ION_MODES)
	main(parser.parse_args())
