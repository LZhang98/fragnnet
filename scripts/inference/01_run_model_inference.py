import argparse
import csv
import contextlib
import os
from pathlib import Path
import sys

import pandas as pd
import torch as th
import yaml

from fragnnet.inference import FraGNNetInference


class _OOSWarningFilter:
	def __init__(self, stream):
		self.stream = stream
		self.suppress_next_newline = False

	def write(self, text):
		warning = "Everything is OOS!"
		if warning in text:
			warning_start = text.rfind("\n", 0, text.index(warning)) + 1
			warning_end = text.index(warning) + len(warning)
			text = text[:warning_start] + text[warning_end:]
			self.suppress_next_newline = True
		if self.suppress_next_newline and text == "":
			return 0
		if self.suppress_next_newline:
			if text in ("\n", "\r", "\r\n"):
				self.suppress_next_newline = False
				return len(text)
			if text.startswith("\r\n"):
				text = text[2:]
			elif text.startswith(("\n", "\r")):
				text = text[1:]
			self.suppress_next_newline = False
		if text:
			return self.stream.write(text)
		return len(text)

	def flush(self):
		return self.stream.flush()

	def __getattr__(self, name):
		return getattr(self.stream, name)


def parse_args():
	parser = argparse.ArgumentParser(description="Run FraGNNet inference from a YAML config.")
	parser.add_argument(
		"-c",
		"--config_fp",
		required=True,
		type=str,
		help="Path to the inference YAML configuration.",
	)
	return parser.parse_args()


def filter_invalid_molecules(spec_df, mol_df, frag_dp):
	error_rows = []
	input_mol_ids = set(spec_df["mol_id"])
	mol_df = mol_df.loc[mol_df["mol_id"].isin(input_mol_ids)].copy()

	def add_errors(mol_ids, reason):
		for mol_id in sorted(mol_ids, key=str):
			error_rows.append({
				"mol_id": mol_id,
				"reason": reason,
				"num_spectra": int((spec_df["mol_id"] == mol_id).sum()),
			})

	def has_bonds(mol):
		return mol is not None and mol.GetNumBonds() > 0

	valid_mol_mask = mol_df["mol"].map(has_bonds)
	no_bond_mol_ids = set(mol_df.loc[~valid_mol_mask, "mol_id"])
	add_errors(no_bond_mol_ids, "no_bonds")
	mol_df = mol_df.loc[valid_mol_mask].copy()
	spec_df = spec_df.loc[~spec_df["mol_id"].isin(no_bond_mol_ids)].copy()

	dag_ids = {path.name.removesuffix(".pickle.bz2") for path in Path(frag_dp).glob("*.pickle.bz2")}
	missing_dag_mask = ~mol_df["mol_id"].astype(str).isin(dag_ids)
	missing_dag_mol_ids = set(mol_df.loc[missing_dag_mask, "mol_id"])
	add_errors(missing_dag_mol_ids, "missing_dag")
	if missing_dag_mol_ids:
		print(f">> Dropping {len(missing_dag_mol_ids)} molecules without DAGs")
		mol_df = mol_df.loc[~missing_dag_mask].copy()
		spec_df = spec_df.loc[~spec_df["mol_id"].isin(missing_dag_mol_ids)].copy()

	return spec_df, mol_df, error_rows


def fill_missing_nce(spec_df, default_nce):
	if "nce" not in spec_df:
		return spec_df, []

	missing_nce_mask = pd.to_numeric(spec_df["nce"], errors="coerce").isna()
	if not missing_nce_mask.any():
		return spec_df, []

	missing_mol_ids = set(spec_df.loc[missing_nce_mask, "mol_id"])
	print(f">> Imputing missing NCE with {default_nce} for {int(missing_nce_mask.sum())} spectra")
	spec_df = spec_df.copy()
	spec_df.loc[missing_nce_mask, "nce"] = float(default_nce)
	error_rows = [
		{
			"mol_id": mol_id,
			"reason": "missing_nce_imputed",
			"num_spectra": int((spec_df["mol_id"] == mol_id).sum()),
		}
		for mol_id in sorted(missing_mol_ids, key=str)
	]
	return spec_df, error_rows


def fill_empty_peaks(spec_df):
	empty_peak_mask = spec_df["peaks"].map(lambda peaks: peaks is None or len(peaks) == 0)
	if not empty_peak_mask.any():
		return spec_df, []

	empty_peak_mol_ids = set(spec_df.loc[empty_peak_mask, "mol_id"])
	print(f">> Adding placeholder peaks for {int(empty_peak_mask.sum())} spectra without peaks")
	spec_df = spec_df.copy()
	spec_df.loc[empty_peak_mask, "peaks"] = spec_df.loc[empty_peak_mask, "peaks"].map(
		lambda _: [(1.0, 1.0)]
	)
	error_rows = [
		{
			"mol_id": mol_id,
			"reason": "empty_peaks_placeholder",
			"num_spectra": int((spec_df["mol_id"] == mol_id).sum()),
		}
		for mol_id in sorted(empty_peak_mol_ids, key=str)
	]
	return spec_df, error_rows


def write_error_log(error_log_fp, error_rows):
	os.makedirs(os.path.dirname(error_log_fp) or ".", exist_ok=True)
	with open(error_log_fp, "w", newline="") as error_file:
		writer = csv.DictWriter(error_file, fieldnames=["mol_id", "reason", "num_spectra"])
		writer.writeheader()
		writer.writerows(error_rows)
	print(f">> Logged {len(error_rows)} skipped molecules to {error_log_fp}")


def build_input_metadata(spec_df):
	metadata_columns = [
		column
		for column in ["spec_id", "mol_id", "group_id", "prec_mz"]
		if column in spec_df.columns
	]
	metadata = spec_df[metadata_columns].copy()
	if "nce" in spec_df.columns:
		metadata["ce"] = spec_df["nce"]
	elif "ace" in spec_df.columns:
		metadata["ce"] = spec_df["ace"]
	if "inst_type" in spec_df.columns:
		metadata["inst"] = spec_df["inst_type"]
	if "prec_type" in spec_df.columns:
		metadata["adduct"] = spec_df["prec_type"]
	return metadata.reset_index(drop=True)


def main():
	args = parse_args()
	with open(args.config_fp, "r") as config_file:
		config = yaml.load(config_file, Loader=yaml.FullLoader) or {}

	inference_config = config.get("inference", {})
	split = inference_config.get("split", "predict_only")
	if split != "predict_only":
		raise ValueError("run_model_inference.py currently requires inference.split='predict_only'.")

	print(f">> Loading model from {inference_config['ckpt_fp']}")
	engine = FraGNNetInference.from_config(args.config_fp)

	print(f">> Loading spectra from {config['spec_fp']}")
	spec_df = pd.read_pickle(config["spec_fp"])
	print(f">> Loading molecules from {config['mol_fp']}")
	mol_df = pd.read_pickle(config["mol_fp"])
	frag_dp = config.get("frag_dp")
	if frag_dp is None:
		raise ValueError("frag_dp is required for FraGNNet inference.")
	spec_df, mol_df, error_rows = filter_invalid_molecules(spec_df, mol_df, frag_dp)
	spec_df, nce_error_rows = fill_missing_nce(
		spec_df,
		default_nce=inference_config.get("default_nce", config.get("ce_mean", 60.0)),
	)
	error_rows.extend(nce_error_rows)
	spec_df, peak_error_rows = fill_empty_peaks(spec_df)
	error_rows.extend(peak_error_rows)
	write_error_log(inference_config["error_log_fp"], error_rows)

	output_subset = inference_config.get("output_subset")
	if output_subset is not None:
		output_subset = set(output_subset)

	with contextlib.redirect_stdout(_OOSWarningFilter(sys.stdout)):
		vals = engine.run(
			split=split,
			spec_df=spec_df,
			mol_df=mol_df,
			frag_dp=frag_dp,
			batch_cutoff=inference_config.get("batch_cutoff", int(1e6)),
			output_subset=output_subset,
			untransform_spec=inference_config.get("untransform_spec", False),
		)
	if inference_config.get("include_input_data", True):
		vals["input_spec"] = spec_df.reset_index(drop=True)
		vals["input_mol"] = mol_df.reset_index(drop=True)
		vals["input_metadata"] = build_input_metadata(spec_df)

	output_fp = inference_config["output_fp"]
	os.makedirs(os.path.dirname(output_fp) or ".", exist_ok=True)
	pd.to_pickle(vals, output_fp)
	print(f">> Saved predictions to {output_fp}")


if __name__ == "__main__":
	main()
