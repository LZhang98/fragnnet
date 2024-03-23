import numpy as np
import pandas as pd
import torch as th
from pyteomics.mass import Composition

from frag_gnn.dataset import BaseDataset, SpecMolDataset
from frag_gnn.graff.data_utils import fit_ann_library, graff_preprocess
from frag_gnn.utils.formula_utils import NEUTRON_MASS
from frag_gnn.utils.data_utils import composition_to_string

def calculate_prec_formula(prec_formula,prec_type):

	prec_formula_c = Composition(formula=prec_formula)
	if prec_type == "[M+H]+":
		prec_formula_c = prec_formula_c + Composition(formula="H")
	elif prec_type == "[M-H]-":
		prec_formula_c = prec_formula_c - Composition(formula="H")
	else:
		raise ValueError(f"Unknown precursor type: {prec_type}")
	return prec_formula_c

class SpecMolAnnDataset(SpecMolDataset):

	def __init__(self,
		spec_fp: str,
		mol_fp: str,
		split_dp: str,
		split: str,
		subsample_params: dict,
		spec_params: dict,
		mol_params: dict,
		ann_fp: str,
		ann_params: dict,
		**kwargs
	):
		
		BaseDataset.__init__(self)
		self._base_init(
			spec_fp=spec_fp,
			mol_fp=mol_fp,
			split_dp=split_dp,
			split=split,
			subsample_params=subsample_params,
			spec_params=spec_params
		)
		self.mol_params = mol_params
		self.ann_fp = ann_fp
		self.ann_params = ann_params
		self._preprocess_spec()
		self._preprocess_mol()
		self._preprocess_ann()
	
	def _preprocess_ann(self):

		ann_df = pd.read_pickle(self.ann_fp)
		library_df = fit_ann_library(
			self.um_spec_df.copy(),
			ann_df.copy(), 
			self.ann_params["library_size"],
			self.ann_params["max_isotope"])
		self.library_df = library_df
		# setup mzs stuff
		self.library_mzs = th.as_tensor(library_df["mzs"].values, dtype=th.float32)
		self.library_mzs  = self.library_mzs.unsqueeze(1).repeat(1,self.ann_params["max_isotope"])
		self.library_loss_mask = th.as_tensor(
			self.library_df.loc[:,"kind"].apply(lambda x: int(x == "loss")).values,
			dtype=th.float32)
		self.library_loss_mask = self.library_loss_mask.unsqueeze(1).repeat(1,self.ann_params["max_isotope"])
		for i in range(self.ann_params["max_isotope"]):
			self.library_mzs[:,i] = self.library_mzs[:,i] + (-2*self.library_loss_mask[:,i]+1) * (i*NEUTRON_MASS)
		# setup isotope stuff
		self.isotope_df = ann_df[["spec_id","ann_isotopes"]]
		self.isotope_df.loc[:,"has_isotopes"] = ann_df["ann_isotopes"].apply(any)
		self.isotope_df = self.isotope_df.drop(columns=["ann_isotopes"])
		self.isotope_df = self.isotope_df[self.isotope_df["spec_id"].isin(self.um_spec_df["spec_id"])]
		self.isotope_df = self.isotope_df.merge(self.um_spec_df[["spec_id","group_id"]], on="spec_id", how="right").fillna(False)
		if self.spec_params["merge"]:
			self.isotope_df = self.isotope_df.groupby("group_id")["has_isotopes"].apply(any).reset_index()
		self.isotope_df = self.isotope_df.set_index(self.id_key,drop=False)

	def _process_ann(self, spec_entry, mol_entry):

		# set up mzs
		prec_mz = spec_entry["prec_mz"]
		library_mzs = prec_mz * self.library_loss_mask + self.library_mzs * (-2*self.library_loss_mask+1)
		library_mzs = library_mzs.unsqueeze(0)
		# check for isotopes
		has_isotopes = self.isotope_df.loc[spec_entry[self.id_key]]["has_isotopes"]
		isotope_mask = th.tensor([has_isotopes],dtype=th.float32)
		# look for duplicates
		formula = mol_entry["formula"]
		prec_type = spec_entry["prec_type"]
		prec_formula_c = calculate_prec_formula(formula,prec_type)
		product_formulae_df = self.library_df[self.library_df["kind"] == "product"][["ann_id","formula"]]
		loss_formulae_df = self.library_df[self.library_df["kind"] == "loss"][["ann_id","formula"]]
		loss_formulae_df.loc[:,"formula"] = loss_formulae_df["formula"].apply(lambda formula: composition_to_string(prec_formula_c - Composition(formula=formula)))
		both_formulae_df = product_formulae_df.merge(loss_formulae_df,on="formula",how="inner")
		dup_mask = th.zeros([1,self.library_df.shape[0],2],dtype=th.float32)
		dup_mask[:,both_formulae_df["ann_id_x"].values,0] = 1
		dup_mask[:,both_formulae_df["ann_id_y"].values,1] = 1
		ann_data = {
			"library_mzs": library_mzs,
			"isotope_mask": isotope_mask,
			"dup_mask": dup_mask,
		}
		return ann_data

	def _process_mol(self, mol_entry):

		mol_data = super()._process_mol(mol_entry)
		if self.mol_params["graff"]:
			mol_graff = graff_preprocess(mol_entry["mol"],self.ann_params["num_eigs"])
			mol_data["mol_graff"] = mol_graff
		return mol_data

	def __getitem__(self, idx):

		spec_entry = self.spec_df.iloc[idx]
		mol_id = spec_entry["mol_id"]
		mol_entry = self.mol_df.loc[mol_id]
		if self.spec_params["preprocess"]:
			spec_data = self.spec_datas[idx].copy()
		else:
			spec_data = self._process_spec(spec_entry)
		spec_data = self._process_spec(spec_entry)
		if self.mol_params["preprocess"]:
			mol_data = self.mol_datas[mol_id].copy()
		else:
			mol_data = self._process_mol(mol_entry)
		ann_data = self._process_ann(spec_entry, mol_entry)
		data = {**spec_data, **mol_data, **ann_data}
		return data

	@staticmethod
	def get_collate_fn():

		return SpecMolAnnDataset.collate_fn
	
	# @staticmethod
	# def _special_collate(keys, collate_data):

	# 	pass

	@staticmethod
	def collate_fn(data_list):

		batch_size, keys, collate_data = SpecMolAnnDataset._setup_collate(data_list)
		SpecMolAnnDataset._special_collate(keys, collate_data)
		SpecMolAnnDataset._standard_collate(batch_size,keys,collate_data)
		return collate_data
	
def test_dataset():

	from frag_gnn.runner import load_config
	from frag_gnn.utils.misc_utils import print_shapes
	from frag_gnn.utils.misc_utils import to_device

	template_fp = "config/template.yml"
	custom_fps = [
		"config/debug/debug_graff.yml"
	]

	for custom_fp in custom_fps:

		print(custom_fp)

		config_d = load_config(template_fp, custom_fp)
		device = "cuda:0" if config_d["accelerator"] == "gpu" else "cpu"

		ds = SpecMolAnnDataset(split="train", **config_d)
		dl = th.utils.data.DataLoader(
			ds,
			batch_size=8,
			shuffle=False,
			num_workers=0,
			collate_fn=ds.get_collate_fn(),
		)
		dl_iter = iter(dl)
		batch = next(dl_iter)
		import pdb; pdb.set_trace()
		print("> batch")
		print_shapes(batch)
		# batch = to_device(batch,device)


if __name__ == "__main__":

	from frag_gnn.utils.misc_utils import th_temp_seed

	with th_temp_seed(420):
		test_dataset()