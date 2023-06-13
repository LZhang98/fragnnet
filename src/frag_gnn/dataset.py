import pandas as pd
import os
import torch as th
from torch.utils.data import Dataset
import numpy as np
import torch_geometric as pyg
from torch_geometric.data import Data, Batch

from frag_gnn.utils.feat_utils import batch_mols_frags, get_mol_fp, get_mol_graph, get_frag_graph
from frag_gnn.utils.spec_utils import bin_func, filter_func, batch_func, prec_type_idx, transform_nce, prec_type_mass_diff
from frag_gnn.utils.dataset_utils import merge_spec_df
from frag_gnn.frag.frag_utils import load_frag_d
from frag_gnn.utils.misc_utils import flatten_lol


SPEC_PARAMS = [
	"merge",
	"subsample_size",
	"subsample_seed",
	"ints_thresh",
	"mz_max",
	"sparse",
	"binned",
	"mz_bin_res",
	"sum_ints",
	"prec_type",
	"prec_mz",
	"prec_mass_diff",
	"nce",
	"ace"
]

MOL_PARAMS = [
	"smiles",
	"fingerprint",
	"fingerprint_morgan",
	"fingerprint_rdkit",
	"fingerprint_maccs",
	"pyg",
	"pyg_node_feats",
	"pyg_edge_feats",
	"pyg_pe_embed_k",
	"pyg_bigraph"
]

FRAG_PARAMS = [
	"pyg",
	"pyg_node_feats",
	"pyg_edge_feats",
	"pyg_bigraph",
	"num_isotopes",
	"formula_peak_mzs",
	"formula_peak_probs",
	"compressed"
]

class SpecMolDataset(Dataset):

	def __init__(
		self,
		spec_fp: str,
		mol_fp: str,
		split_fp: str,
		**kwargs):

		super().__init__()
		self.spec_df = pd.read_pickle(spec_fp)
		self.mol_df = pd.read_pickle(mol_fp)
		self.split_df = pd.read_csv(split_fp)
		
		self.spec_params = {k: kwargs["spec_"+k] for k in SPEC_PARAMS}
		self.mol_params = {k: kwargs["mol_"+k] for k in MOL_PARAMS}

		# select spectra
		self.spec_df = self.spec_df[self.spec_df["spec_id"].isin(self.split_df["spec_id"])]
		# select molecules
		self.mol_df = self.mol_df[self.mol_df["mol_id"].isin(self.split_df["mol_id"])]
		assert np.all(np.unique(self.mol_df["mol_id"]) == np.unique(self.spec_df["mol_id"]))

		# merge spectra
		if self.spec_params["merge"]:
			self.spec_df = merge_spec_df(self.spec_df)            

		# subsample
		if self.spec_params["subsample_size"] > 0:
			self.spec_df = self.spec_df.sample(
				n=self.spec_params["subsample_size"],
				random_state=self.spec_params["subsample_seed"],
				replace=False)
			self.mol_df = self.mol_df[self.mol_df["mol_id"].isin(self.spec_df["mol_id"])]

		# reset indices
		self.spec_df = self.spec_df.reset_index(drop=True)
		self.mol_df = self.mol_df.reset_index(drop=True)

		# use mol_id as index for speedy access
		self.mol_df = self.mol_df.set_index("mol_id",drop=False).sort_index().rename_axis(None)

	def __len__(self):

		return len(self.spec_df)

	def __getitem__(self,idx):

		spec_entry = self.spec_df.iloc[idx]
		mol_id = spec_entry["mol_id"]
		mol_entry = self.mol_df.loc[mol_id]
		spec_data = self._process_spec(spec_entry)
		mol_data = self._process_mol(mol_entry)
		data = {**spec_data,**mol_data}
		return data

	def _process_spec(self,spec_entry):

		spec_data = {}
		# peak data
		peaks = spec_entry["peaks"]
		mzs, ints = [], []
		for peak in peaks:
			p_mz, p_int = peak
			mzs.append(p_mz)
			ints.append(p_int)
		mzs = th.tensor(mzs,dtype=th.float)
		ints = th.tensor(ints,dtype=th.float)
		# mzs, ints = filter_func(mzs, ints, self.spec_params["ints_thresh"], self.spec_params["mz_max"])
		assert mzs.shape[0] > 0
		assert ints.shape[0] > 0
		if self.spec_params["sparse"]:
			# get sparse spectrum
			spec_data["spec_mzs"] = mzs
			spec_data["spec_ints"] = ints
		if self.spec_params["binned"]:
			raise NotImplementedError
			# # bin spectrum
			# spec_bin = bin_func(
			# 	mzs,
			# 	ints,
			# 	self.spec_params["mz_max"],
			# 	self.spec_params["mz_bin_res"],
			# 	False,
			# 	self.spec_params["sum_ints"]
			# )
			# spec_data["spec_bin"] = spec_bin.reshape(1,-1)
		# metadata
		if self.spec_params["prec_type"]:
			prec_type = spec_entry["prec_type"]
			prec_type = th.tensor([prec_type_idx(prec_type)],dtype=th.long)
			spec_data["spec_prec_type"] = prec_type
		if self.spec_params["prec_mass_diff"]:
			prec_type = spec_entry["prec_type"]
			mass_diff = th.tensor([prec_type_mass_diff(prec_type)],dtype=th.float)
			spec_data["spec_prec_mass_diff"] = mass_diff
		if self.spec_params["nce"]:
			assert not self.spec_params["merge"]
			nce = spec_entry["nce"]
			nce = th.tensor([transform_nce(nce)],dtype=th.float)
			spec_data["spec_nce"] = nce
		if self.spec_params["ace"]:
			# TODO: implement ace
			raise NotImplementedError
			# assert not self.merge
			# ace = spec_entry["ace"]
			# ace = th.tensor([transform_ace(ace)],dtype=th.float)
			# spec_data["spec_ace"] = ace
		if self.spec_params["prec_mz"]:
			prec_mz = spec_entry["prec_mz"]
			prec_mz = th.tensor([float(prec_mz)],dtype=th.float)
			spec_data["spec_prec_mz"] = prec_mz
		return spec_data

	def _process_mol(self,mol_entry):

		mol_data = {}
		mol = mol_entry["mol"]
		if self.mol_params["smiles"]:
			smiles = mol_entry["smiles"]
			mol_data["mol_smiles"] = [smiles]
		if self.mol_params["fingerprint"]:
			fingerprint = get_mol_fp(mol,
				self.mol_params["fingerprint_morgan"],
				self.mol_params["fingerprint_rdkit"],
				self.mol_params["fingerprint_maccs"]
			)
			mol_data["mol_fingerprint"] = fingerprint
		if self.mol_params["pyg"]:
			mol_pyg = get_mol_graph(
				mol,
				self.mol_params["pyg_node_feats"],
				self.mol_params["pyg_edge_feats"],
				self.mol_params["pyg_pe_embed_k"],
				self.mol_params["pyg_bigraph"]
			)
			mol_data["mol_pyg"] = mol_pyg
		return mol_data

	@staticmethod
	def get_collate_fn():
		return SpecMolDataset.collate_fn

	@staticmethod
	def collate_fn(data_list):

		batch_size = len(data_list)
		keys = list(data_list[0].keys())
		collate_data = {key: [] for key in keys}
		for data in data_list:
			for key in keys:
				collate_data[key].append(data[key])
		# special handling
		if "spec_ints" in keys and "spec_mzs" in keys:
			# create batch_idxs
			mzs, ints, batch_idxs = batch_func(
				collate_data["spec_mzs"],
				collate_data["spec_ints"]
			)
			collate_data["spec_mzs"] = mzs
			collate_data["spec_ints"] = ints
			collate_data["spec_batch_idxs"] = batch_idxs
			# remove from list
			keys.remove("spec_ints")
			keys.remove("spec_mzs")
		if "mol_pyg" in keys:
			# batch
			collate_data["mol_pyg"] = Batch.from_data_list(collate_data["mol_pyg"])
			# remove from list
			keys.remove("mol_pyg")
		# regular handling
		for key in keys:
			values = collate_data[key]
			if isinstance(values[0],list):
				# flatten
				values = flatten_lol(values)
				collate_data[key] = values
			elif isinstance(values[0],th.Tensor):
				# cat
				values = th.cat(values,dim=0)
				collate_data[key] = values
			else:
				raise TypeError(f"Unsupported type: {type(values[0])}")
		# add batch_size
		collate_data["batch_size"] = th.tensor(batch_size)
		return collate_data


class SpecMolFragDataset(SpecMolDataset):

	def __init__(
		self,
		spec_fp,
		mol_fp,
		frag_dp,
		split_fp,
		**kwargs):

		super().__init__(spec_fp,mol_fp,split_fp,**kwargs)
		self.frag_dp = frag_dp
		self.frag_params = {k: kwargs["frag_"+k] for k in FRAG_PARAMS}

	def __getitem__(self,idx):

		spec_entry = self.spec_df.iloc[idx]
		mol_id = spec_entry["mol_id"]
		mol_entry = self.mol_df.loc[mol_id]
		spec_data = self._process_spec(spec_entry)
		mol_data = self._process_mol(mol_entry)
		frag_entry = self._load_frag_entry(mol_id)
		frag_data = self._process_frag(frag_entry)
		data = {**spec_data,**mol_data,**frag_data}
		return data

	def _load_frag_entry(self,mol_id):

		frag_entry = load_frag_d(mol_id,self.frag_dp,self.frag_params["compressed"])
		return frag_entry

	def _process_frag(self,frag_entry):

		frag_data = {}
		if self.frag_params["pyg"]:
			frag_pyg = frag_entry["dag"]
			frag_pyg = get_frag_graph(
				frag_pyg,
				self.frag_params["pyg_node_feats"],
				self.frag_params["pyg_edge_feats"],
				self.frag_params["pyg_bigraph"]
			)
			frag_data["frag_pyg"] = frag_pyg
		if self.frag_params["formula_peak_mzs"]:
			formula_peak_mzs = frag_entry["formula_peak_mzs"]
			formula_peak_mzs = formula_peak_mzs[:,:self.frag_params["num_isotopes"]]
			frag_data["frag_formula_peak_mzs"] = formula_peak_mzs
		if self.frag_params["formula_peak_probs"]:    
			formula_peak_probs = frag_entry["formula_peak_probs"]
			formula_peak_probs = formula_peak_probs[:,:self.frag_params["num_isotopes"]]
			frag_data["frag_formula_peak_probs"] = formula_peak_probs
		return frag_data

	@staticmethod
	def get_collate_fn():
		return SpecMolFragDataset.collate_fn

	@staticmethod
	def collate_fn(data_list):

		batch_size = len(data_list)
		keys = list(data_list[0].keys())
		collate_data = {key: [] for key in keys}
		for data in data_list:
			for key in keys:
				collate_data[key].append(data[key])
		# special handling
		if "spec_ints" in keys and "spec_mzs" in keys:
			# create batch_idxs
			mzs, ints, batch_idxs = batch_func(
				collate_data["spec_mzs"],
				collate_data["spec_ints"]
			)
			collate_data["spec_mzs"] = mzs
			collate_data["spec_ints"] = ints
			collate_data["spec_batch_idxs"] = batch_idxs
			# remove from list
			keys.remove("spec_ints")
			keys.remove("spec_mzs")
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
		# this only runs if "frag_pyg" is not in keys
		if "mol_pyg" in keys:
			# batch
			collate_data["mol_pyg"] = Batch.from_data_list(collate_data["mol_pyg"])
			# remove from list
			keys.remove("mol_pyg")
		# regular handling
		for key in keys:
			values = collate_data[key]
			if isinstance(values[0],list):
				# flatten
				values = flatten_lol(values)
				collate_data[key] = values
			elif isinstance(values[0],th.Tensor):
				# cat
				values = th.cat(values,dim=0)
				collate_data[key] = values
			else:
				raise TypeError(f"Unsupported type: {type(values[0])}")
		# add batch_size
		collate_data["batch_size"] = th.tensor(batch_size)
		return collate_data

def get_default_fps():

	spec_fp = "data/proc/spec_df.pkl"
	mol_fp = "data/proc/mol_df.pkl"
	frag_dp = "data/frag/debug/dags"
	train_fp = "data/split/debug/train_ids.csv"
	val_fp = "data/split/debug/val_ids.csv"
	test_fp = "data/split/debug/test_ids.csv"
	return spec_fp, mol_fp, frag_dp, train_fp, val_fp, test_fp

def get_default_ds_params(frag=True):

	spec_params = {
		"merge": True,
		"subsample_size": 0,
		"subsample_seed": 420,
		"ints_thresh": 0.0, # dont't filter in dataloader
		"mz_max": 1500.0,
		"sparse": True,
		"binned": False, #True,
		"mz_bin_res": 0.01,
		"sum_ints": True,
		"prec_type": True,
		"prec_mz": True,
		"prec_mass_diff": True,
		"nce": False, # cannot be True if merge is True
		"ace": False # cannot be True if merge is True
	}
	mol_params = {
		"smiles": True,
		"fingerprint": False,
		"fingerprint_morgan": True,
		"fingerprint_rdkit": True,
		"fingerprint_maccs": True,
		"pyg": True,
		"pyg_node_feats": [
			"a_onehot",
			"a_degree",
			"a_hybrid",
			"a_formal",
			"a_radical",
			"a_ring",
			"a_mass",
			"a_chiral",
		],
		"pyg_edge_feats": ["b_degree"],
		"pyg_pe_embed_k": 0,
		"pyg_bigraph": True,
	}
	frag_params = {
		"pyg": True,
		"pyg_node_feats": [
			"depth",
			"cc",
			"base_formula"
		],
		"pyg_edge_feats": [
			"cc",
			"base_formula"
		],
		"pyg_bigraph": True,
		"num_isotopes": 1,
		"formula_peak_mzs": True,
		"formula_peak_probs": True,
		"compressed": True,
	}
	assert set(spec_params.keys()) == set(SPEC_PARAMS)
	assert set(mol_params.keys()) == set(MOL_PARAMS)
	assert set(frag_params.keys()) == set(FRAG_PARAMS)
	params = {}
	for k,v in spec_params.items():
		params[f"spec_{k}"] = v
	for k,v in mol_params.items():
		params[f"mol_{k}"] = v
	if frag:
		for k,v in frag_params.items():
			params[f"frag_{k}"] = v
	return params

if __name__ == "__main__":

	import pandas as pd
	from torch.utils.data import DataLoader
	from pprint import pprint

	def print_shapes(th_d):
		for k,v in th_d.items():
			if isinstance(v,list):
				print(k,(len(v),))
			elif isinstance(v,th.Tensor):
				print(k,tuple(v.shape))
			elif isinstance(v,Batch):
				print(k,(v.num_nodes,v.num_edges))
			else:
				raise ValueError(f"Unsupported type: {type(v)}")

	spec_fp, mol_fp, frag_dp, train_fp, _, _ = get_default_fps()	

	params = get_default_ds_params()

	print(">>> SpecMolDataset")
	sm_ds = SpecMolDataset(
		spec_fp,
		mol_fp,
		train_fp,
		**params
	)
	sm_dl = DataLoader(
		sm_ds,
		batch_size=2,
		shuffle=False,
		collate_fn=sm_ds.get_collate_fn(),
		num_workers=0
	)
	sm_b = next(iter(sm_dl))
	print_shapes(sm_b)
	
	print(">>> SpecMolFragDataset")
	smf_ds = SpecMolFragDataset(
		spec_fp,
		mol_fp,
		frag_dp,
		train_fp,
		**params
	)
	smf_dl = DataLoader(
		smf_ds,
		batch_size=2,
		shuffle=False,
		collate_fn=smf_ds.get_collate_fn(),
		num_workers=0
	)
	smf_b = next(iter(smf_dl))
	print_shapes(smf_b)