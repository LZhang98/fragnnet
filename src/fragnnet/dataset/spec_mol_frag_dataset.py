
import torch as th
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, Sampler, WeightedRandomSampler
import numpy as np
import torch_geometric as pyg
from torch_geometric.data import Batch
import sys
from tqdm import tqdm
from typing import Iterator, List

from fragnnet.utils.feat_utils import batch_mols_frags,get_frag_graph
from fragnnet.utils.frag_utils import load_frag_d
from fragnnet.utils.misc_utils import get_pyg_memory_usage
from fragnnet.utils.formula_utils import PREC_TYPE_TO_MASS_DIFF
from fragnnet.utils.data_utils import deep_copy_data_dict
from fragnnet.dataset import BaseDataset, SpecMolDataset

class SpecMolFragDataset(SpecMolDataset):

	def __init__(
		self,
		spec_fp: str,
		mol_fp: str,
		split_dp: str,
		split: str,
		subsample_params: dict,
		spec_params: dict,
		mol_params: dict,
		frag_dp: str,
		frag_params: dict,
		spec_pp_sd: dict = None, # preprocessed spec data
		mol_pp_sd: dict = None, # preprocessed mol data
		frag_pl_sd: dict = None, # preloaded frag data
		frag_pp_sd: dict = None, # preprocessed frag data
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
		self.frag_dp = frag_dp
		self.frag_params = frag_params
		
		if spec_pp_sd is None:
			spec_pp_sd = dict()
		if mol_pp_sd  is None:
			mol_pp_sd = dict()
		if frag_pl_sd is None:
			frag_pl_sd = dict()
		if frag_pp_sd is None:
			frag_pp_sd = dict()
   
		self._preprocess_spec(spec_pp_sd)
		self._preprocess_mol(mol_pp_sd)
		self._preprocess_frag(frag_pl_sd, frag_pp_sd)
	
	@staticmethod
	def get_data_dict_types():
		# return list of data dict used for caching
		# Spectral Preprocessed Shared Dictionary: spec_pp_sd
		# Molecular Preprocessed Shared Dictionary: mol_pp_sd
		# Fragment Preloaded Shared Dictionary: frag_pl_sd
		# Fragment Preprocessed Shared Dictionary: frag_pp_sd
		return ["spec_pp_sd", "mol_pp_sd", "frag_pl_sd", "frag_pp_sd"]

	def _preprocess_frag(self, frag_pl_sd: dict, frag_pp_sd: dict):

		# preload frag dags
		if self.frag_params["preload"]:
			self.frag_entries = frag_pl_sd
			total_frag_entry_size = 0
			for mol_id in tqdm(self.mol_df["mol_id"].values,desc="> preload frag",total=len(self.mol_df)):
				frag_entry = load_frag_d(mol_id,self.frag_dp,self.frag_params["compressed"])
				total_frag_entry_size += get_pyg_memory_usage(frag_entry["dag"])
				self.frag_entries[mol_id] = frag_entry
			print(f"> total_frag_entry_size: {total_frag_entry_size/1e6:.2f} MB")
   
		# preprocess frag dags
		if self.frag_params["preprocess"]:
			assert self.frag_params["preload"]
			self.frag_data = frag_pp_sd
			total_frag_data_size = 0
			for k in tqdm(list(self.frag_entries.keys()),desc="> preprocess frag",total=len(self.frag_entries)):
				frag_data = self._process_frag(self.frag_entries.pop(k),None)
				total_frag_data_size += get_pyg_memory_usage(frag_data["frag_pyg"])
				self.frag_data[k] = frag_data
			print(f"> total_frag_data_size: {total_frag_data_size/1e6:.2f} MB")
			# remove them from the entries
			self.frag_entries = {}

	def __getitem__(self, idx):
		spec_entry = self.spec_df.iloc[idx]
		mol_id = spec_entry["mol_id"]
		mol_entry = self.mol_df.loc[mol_id]
		if self.spec_params["preprocess"]:
			spec_data = deep_copy_data_dict(self.spec_datas[idx])
		else:
			spec_data = self._process_spec(spec_entry)
		if self.mol_params["preprocess"]:
			mol_data = deep_copy_data_dict(self.mol_datas[mol_id])
		else:
			mol_data = self._process_mol(mol_entry)
		# frag stuff
		if self.frag_params["preprocess"]:
			frag_data = deep_copy_data_dict(self.frag_data[mol_id])
			# update prec_mass_diff
			prec_type_mass_diff = PREC_TYPE_TO_MASS_DIFF[spec_entry["prec_type"]]
			# this works because tensor addition return a new tensor
			# this no deep copy required
			frag_data["frag_formula_peak_mzs"] = frag_data["frag_formula_peak_mzs"] + prec_type_mass_diff
		elif self.frag_params["preload"]:
			frag_entry = deep_copy_data_dict(self.frag_entries[mol_id])
			frag_data = self._process_frag(frag_entry,spec_entry)
		else:
			frag_entry = self._load_frag_entry(mol_id)
			frag_data = self._process_frag(frag_entry,spec_entry)
		data = {**spec_data,**mol_data,**frag_data}
		return data

	def _load_frag_entry(self,mol_id) -> dict:
		""" load frag entry from precomputed dag on disk

		Args:
			mol_id (str): mol id

		Returns:
			dict: dag dictrionary
		"""

		frag_entry = load_frag_d(
			mol_id,
			self.frag_dp,
			self.frag_params["compressed"]
		)
		return frag_entry

	def _process_frag(self,frag_entry:dict,spec_entry):
		"""
			process the frag dictionary, and spec data
		Args:
			frag_entry (_type_): _description_
			spec_entry (_type_): _description_

		Returns:
			_type_: _description_
		"""
		frag_data = {}
		# case use frag dags as second gnn
		if self.frag_params["pyg"]:
			frag_pyg = frag_entry["dag"]
			frag_pyg = get_frag_graph(
				frag_pyg,
				self.frag_params["pyg_node_feats"],
				self.frag_params["pyg_edge_feats"],
				self.frag_params["pyg_edges"],
				self.frag_params["pyg_bigraph"]
			)
			frag_data["frag_pyg"] = frag_pyg
		if self.frag_params["formula_peak_mzs"]:
			formula_peak_mzs = frag_entry["formula_peak_mzs"]
			formula_peak_mzs = formula_peak_mzs[:,:self.frag_params["num_isotopes"]]
			if spec_entry is not None:
				prec_type_mass_diff = PREC_TYPE_TO_MASS_DIFF[spec_entry["prec_type"]]
				formula_peak_mzs = formula_peak_mzs + prec_type_mass_diff
			frag_data["frag_formula_peak_mzs"] = formula_peak_mzs
		if self.frag_params["formula_peak_probs"]:	
			formula_peak_probs = frag_entry["formula_peak_probs"]
			formula_peak_probs = F.normalize(formula_peak_probs[:,:self.frag_params["num_isotopes"]],dim=1,p=1)
			frag_data["frag_formula_peak_probs"] = formula_peak_probs
		if self.frag_params["formula_str"]:
			# import pdb; pdb.set_trace()
			formula_str = list(frag_entry["idx_to_formula"].values())
			assert formula_str[0] == "", formula_str[0]
			formula_str = np.array(formula_str)
			frag_data["frag_formula_str"] = formula_str
		return frag_data

	@staticmethod
	def get_collate_fn():

		return SpecMolFragDataset.collate_fn

	@staticmethod
	def _special_collate(keys, collate_data):

		if "frag_pyg" in keys:
			assert "mol_pyg" in keys
			assert "frag_formula_peak_mzs" in keys
			assert "frag_formula_peak_probs" in keys
			# process
			batch_mol_frag_data = batch_mols_frags(
				collate_data["mol_pyg"],
				collate_data["frag_pyg"],
				collate_data["frag_formula_peak_mzs"],
				collate_data["frag_formula_peak_probs"]
			)
			for k,v in batch_mol_frag_data.items():
				collate_data[k] = v
			# remove from list
			keys.remove("frag_pyg")
			keys.remove("mol_pyg")
			keys.remove("frag_formula_peak_mzs")
			keys.remove("frag_formula_peak_probs")
		SpecMolDataset._special_collate(keys,collate_data)

	@staticmethod
	def collate_fn(data_list):

		# prevent edge case causing crash
		if len(data_list) == 0:
			return  {"batch_size": th.tensor(0)}
		batch_size, keys, collate_data = SpecMolFragDataset._setup_collate(data_list)
		# special handling
		SpecMolFragDataset._special_collate(keys,collate_data)
		SpecMolFragDataset._standard_collate(batch_size,keys,collate_data)
		return collate_data

def get_batch_memory(batch):

	batch_mem_d = {}
	for k,v in batch.items():
		if isinstance(v, th.Tensor):
			batch_mem_d[k] = v.element_size()*v.nelement()
		elif isinstance(v, Batch):
			batch_mem_d[k] = pyg.profile.get_data_size(v)
		elif isinstance(v, list):
			batch_mem_d[k] = sum(sys.getsizeof(x) for x in v)
		else:
			raise ValueError(f"Unsupported type: {type(v)}")
	batch_mem_total = sum(batch_mem_d.values())
	return batch_mem_d, batch_mem_total

def find_largest_batch(dl):

	batch_mem_ds, batch_mem_totals = [], []
	for batch in iter(dl):
		batch_mem_d, batch_mem_total = get_batch_memory(batch)
		batch_mem_ds.append(batch_mem_d)
		batch_mem_totals.append(batch_mem_total)
	argmax_idx = np.argmax(batch_mem_totals)
	return batch_mem_ds[argmax_idx], batch_mem_totals[argmax_idx]
		
