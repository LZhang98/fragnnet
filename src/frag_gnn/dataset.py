import pandas as pd
import os
import torch as th
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, Sampler, WeightedRandomSampler, RandomSampler
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch_geometric as pyg
from torch_geometric.data import Data, Batch
import sys
from tqdm import tqdm
from typing import Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union

from frag_gnn.utils.feat_utils import batch_mols_frags, get_mol_fp, get_mol_graph, get_frag_graph
from frag_gnn.utils.spec_utils import bin_func, filter_func, batch_func, prec_type_idx, prec_type_mass_diff
from frag_gnn.utils.proc_utils import merge_spec_df
from frag_gnn.utils.frag_utils import load_frag_d
from frag_gnn.utils.misc_utils import flatten_lol, get_pyg_memory_usage
from frag_gnn.utils.formula_utils import PREC_TYPE_TO_MASS_DIFF
import frag_gnn.massformer.data_utils as mf_data_utils


def init_dataloader(ds, config_d):

	split = ds.split
	if split in ["train", "val"]:
		dl_param_d = {
			'dataset': ds,
			'num_workers': config_d["num_workers"],
			'collate_fn': ds.get_collate_fn(),
		}
		
		# for group sampler 
		sampler = None
		if config_d['group_sampler'] and split == 'train':
			sampler = GroupSampler(ds,sample_k=config_d['group_sampler_max_per_group'])
		if config_d['simple_group_sampler'] and split == 'train' and not config_d['group_sampler']:
			sampler = get_group_sampler(ds,config_d)

		# for bacth sampler 
		if config_d['dynamic_batch_sampler']:
			chunk_size = config_d['dynamic_batch_sampler_chunk_size']
			max_batch_size = config_d['train_batch_size'] * config_d['accumulate_grad_batches']
	
			if split == "train":
				if sampler is None and not config_d["debug_overfit"]:
					sampler = RandomSampler(ds, False)

				batchsampler = SpecMolFragDynamicBatchSampler(ds, 
										max_num = config_d['dynamic_batch_sampler_max'], 
										limited_by = config_d['dynamic_batch_sampler_mode'], 
										skip_too_big = True,
										return_batch_at = max_batch_size,
										sampler = sampler)
				
			else:
				batchsampler = SpecMolFragDynamicBatchSampler(ds, 
								max_num = config_d['dynamic_batch_sampler_max'], 
								limited_by = config_d['dynamic_batch_sampler_mode'],
								skip_too_big = True)

			dl_param_d['batch_sampler'] = batchsampler
		else:	
			if split == "train":
				dl_param_d['batch_size'] = config_d["train_batch_size"]	
				dl_param_d['drop_last'] = config_d["drop_last"]
				if sampler is not None:
					dl_param_d['sampler'] = sampler
				else:
					dl_param_d["shuffle"] = True
			else:
				dl_param_d['batch_size'] = config_d["eval_batch_size"]
				dl_param_d["drop_last"] = False
				dl_param_d['shuffle'] = False
	
		dl = DataLoader(
			**dl_param_d
		)

	else:

		dl = DataLoader(
			ds,
			batch_size=config_d["eval_batch_size"],
			shuffle=False,
			num_workers=config_d["num_workers"],
			collate_fn=ds.get_collate_fn()
		)
	return dl

class BaseDataset(Dataset):

	def _base_init(self, 
		spec_fp: str,
		mol_fp: str,
		split_dp: str,
		split: str,
		subsample_params: dict,
		spec_params: dict
	):
		
		self.spec_df = pd.read_pickle(spec_fp)
		self.mol_df = pd.read_pickle(mol_fp)
		self.split = split
		# assert split in ["train","val","test","secondary"], split
		split_fp = os.path.join(split_dp,f"{split}_ids.csv")
		self.split_df = pd.read_csv(split_fp)
		self.spec_params = spec_params
		# select spectra
		self.spec_df = self.spec_df[self.spec_df["spec_id"].isin(self.split_df["spec_id"])]
		# select molecules
		self.mol_df = self.mol_df[self.mol_df["mol_id"].isin(self.split_df["mol_id"])]
		assert np.all(np.unique(self.mol_df["mol_id"]) == np.unique(self.spec_df["mol_id"]))
		# merge spectra
		if self.spec_params["merge"]:
			self.spec_df = merge_spec_df(self.spec_df,keep_ces=self.spec_params["merge_keep_ces"])
		# subsample
		if subsample_params.get(split,False) and subsample_params["subsample_size"] > 0:
			if isinstance(subsample_params["subsample_size"],int):
				n = subsample_params["subsample_size"]
				frac = None
			else:
				assert isinstance(subsample_params["subsample_size"],float)
				n = None
				frac = subsample_params["subsample_size"]
			self.spec_df = self.spec_df.sample(
				n=n,
				frac=frac,
				random_state=subsample_params["subsample_seed"],
				replace=False)
			self.mol_df = self.mol_df[self.mol_df["mol_id"].isin(self.spec_df["mol_id"])]
		# reset indices
		self.spec_df = self.spec_df.reset_index(drop=True)
		self.mol_df = self.mol_df.reset_index(drop=True)
		# use mol_id as index for speedy access
		self.mol_df = self.mol_df.set_index("mol_id",drop=False).sort_index().rename_axis(None)
		self._compute_counts()

	def _compute_counts(self):

		self.group_per_mol = self.spec_df[["mol_id","group_id"]].drop_duplicates().groupby("mol_id").size().to_dict()
		if self.spec_params["merge"]:
			self.spec_per_mol = {mol_id: 1 for mol_id in self.spec_df["mol_id"].unique()}
			self.spec_per_group = {group_id: 1 for group_id in self.spec_df["group_id"].unique()}
		else:
			self.spec_per_mol = self.spec_df[["mol_id","spec_id"]].drop_duplicates().groupby("mol_id").size().to_dict()
			self.spec_per_group = self.spec_df[["group_id","spec_id"]].drop_duplicates().groupby("group_id").size().to_dict()

	def _preprocess_spec(self):

		# preload and pre-process spectra
		if self.spec_params["preprocess"]:
			self.spec_datas = {}
			for idx, spec_entry in tqdm(self.spec_df.iterrows(),desc="> preprocess spec",total=len(self.spec_df)):
				spec_data = self._process_spec(spec_entry)
				self.spec_datas[idx] = spec_data

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
			assert (not self.spec_params["merge"]) or self.spec_params["merge_keep_ces"]
			nce = spec_entry["nce"]
			if self.spec_params["merge"]:
				assert self.spec_params["merge_keep_ces"]
				assert isinstance(nce,list), type(nce)
				nce = th.tensor(nce,dtype=th.float)
				spec_data["spec_m_nce"] = nce
			else:
				assert isinstance(nce,float), type(nce)
				nce = th.tensor([nce],dtype=th.float)
				spec_data["spec_nce"] = nce
		if self.spec_params["ace"]:
			raise NotImplementedError()
		if self.spec_params["prec_mz"]:
			prec_mz = spec_entry["prec_mz"]
			prec_mz = th.tensor([float(prec_mz)],dtype=th.float)
			spec_data["spec_prec_mz"] = prec_mz
		if self.spec_params["unique_id"]:
			if self.spec_params["merge"]:
				id_key = "group_id"
			else:
				id_key = "spec_id"
			unique_id = spec_entry[id_key]
			unique_id = th.tensor([unique_id],dtype=th.long) 
			spec_data["spec_unique_id"] = unique_id
			spec_data['group_id'] =  th.tensor([spec_entry['group_id']],dtype=th.long)
			spec_data['mol_id'] = th.tensor([spec_entry['mol_id']],dtype=th.long)
		if self.spec_params["counts"]:
			spec_per_mol = self.spec_per_mol[spec_entry["mol_id"]]
			spec_per_mol = th.tensor([spec_per_mol],dtype=th.long)
			spec_data["spec_per_mol"] = spec_per_mol
			group_per_mol = self.group_per_mol[spec_entry["mol_id"]]
			group_per_mol = th.tensor([group_per_mol],dtype=th.long)
			spec_data["group_per_mol"] = group_per_mol
			spec_per_group = self.spec_per_group[spec_entry["group_id"]]
			spec_per_group = th.tensor([spec_per_group],dtype=th.long)
			spec_data["spec_per_group"] = spec_per_group
		return spec_data

	def __getitem__(self, idx):

		raise NotImplementedError()

	def __len__(self):

		return len(self.spec_df)

	@staticmethod
	def get_collate_fn():
		
		return BaseDataset.collate_fn

	@staticmethod
	def _setup_collate(data_list):

		batch_size = len(data_list)
		keys = list(data_list[0].keys())
		collate_data = {key: [] for key in keys}
		for data in data_list:
			for key in keys:
				collate_data[key].append(data[key])
		return batch_size, keys, collate_data

	@staticmethod
	def _special_collate(keys, collate_data):

		# handle sparse spectra
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
		# handle sparse ces
		if "spec_m_nce" in keys:
			# create batch_idxs
			nces, _, batch_idxs = batch_func(
				collate_data["spec_m_nce"],
				collate_data["spec_m_nce"] # duplicate for compatibility
			)
			collate_data["spec_m_nce"] = nces
			collate_data["spec_m_nce_batch_idxs"] = batch_idxs
			# remove from list
			keys.remove("spec_m_nce")

	@staticmethod
	def _standard_collate(batch_size,keys,collate_data):
		""" mutates keys and collate_data """

		# handle generic data
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
		# remove everything
		keys.clear()
		# add batch size
		collate_data["batch_size"] = th.tensor(batch_size, dtype=th.long)

	@staticmethod
	def collate_fn(data_list):

		raise NotImplementedError()

class SpecMolDataset(BaseDataset):

	def __init__(
		self,
		spec_fp: str,
		mol_fp: str,
		split_dp: str,
		split: str,
		subsample_params: dict,
		spec_params: dict,
		mol_params: dict,
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
		self._preprocess_spec()
		self._preprocess_mol()
		
	def _get_mol_graph_size(self, mol_data):

		if self.mol_params["pyg"]:
			mol_pyg = mol_data["mol_pyg"]
			mol_graph_size = get_pyg_memory_usage(mol_pyg)
		else:
			mol_graph_size = 0
		return mol_graph_size

	def _preprocess_mol(self):

		# preload and pre-process molecules
		if self.mol_params["preprocess"]:
			self.mol_datas = {}
			total_mol_graph_size = 0
			for idx, mol_entry in tqdm(self.mol_df.iterrows(),desc="> preprocess mol",total=len(self.mol_df)):
				mol_data = self._process_mol(mol_entry)
				total_mol_graph_size += self._get_mol_graph_size(mol_data)
				self.mol_datas[mol_entry["mol_id"]] = mol_data
			print(f"> total_mol_graph_size: {total_mol_graph_size/1e6:.2f} MB")

	def __getitem__(self,idx):
		spec_entry = self.spec_df.iloc[idx]
		mol_id = spec_entry["mol_id"]
		mol_entry = self.mol_df.loc[mol_id]
		if self.spec_params["preprocess"]:
			spec_data = self.spec_datas[idx].copy()
		else:
			spec_data = self._process_spec(spec_entry)
		if self.mol_params["preprocess"]:
			mol_data = self.mol_datas[mol_id].copy()
		else:
			mol_data = self._process_mol(mol_entry)
		data = {**spec_data,**mol_data}
		return data

	def _process_mol(self,mol_entry):

		mol_data = {}
		mol = mol_entry["mol"]
		if self.mol_params["smiles"]:
			smiles = mol_entry["smiles"]
			mol_data["mol_smiles"] = [smiles]
		if self.mol_params["fingerprint"]:
			fingerprint = get_mol_fp(
				mol,
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
		if self.mol_params["mf"]:
			mol_mf = mf_data_utils.gf_preprocess(mol,-1)
			mol_data["mol_mf"] = mol_mf
		return mol_data

	@staticmethod
	def get_collate_fn():

		return SpecMolDataset.collate_fn

	@staticmethod
	def _special_collate(keys, collate_data):

		if "mol_pyg" in keys:
			# batch
			collate_data["mol_pyg"] = Batch.from_data_list(collate_data["mol_pyg"])
			# remove from list
			keys.remove("mol_pyg")
		if "mol_mf" in keys:
			# batch
			mol_mf_d = mf_data_utils.collator(collate_data["mol_mf"])
			for k, v in mol_mf_d.items():
				collate_data["mol_mf_"+k] = v
			# remove from list
			collate_data.pop("mol_mf")
			keys.remove("mol_mf")
		BaseDataset._special_collate(keys,collate_data)

	@staticmethod
	def collate_fn(data_list):
		
		batch_size, keys, collate_data = SpecMolDataset._setup_collate(data_list)
		SpecMolDataset._special_collate(keys,collate_data)
		SpecMolDataset._standard_collate(batch_size,keys,collate_data)
		return collate_data

class SpecMolFragDataset(SpecMolDataset):

	def __init__(
		self,
		spec_fp: str,
		mol_fp: str,
		split_dp: str,
		split: str,
		subsample_params: dict,
		frag_dp: str,
		spec_params: dict,
		mol_params: dict,
		frag_params: dict,
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
		self.frag_dp = frag_dp
		self.mol_params = mol_params
		self.frag_params = frag_params
		self._preprocess_spec()
		self._preprocess_mol()
		self._preprocess_frag()
		
	def _preprocess_frag(self):

		# preload frag dags
		if self.frag_params["preload"]:
			self.frag_entries = {}
			total_frag_entry_size = 0
			for mol_id in tqdm(self.mol_df["mol_id"].values,desc="> preload frag",total=len(self.mol_df)):
				frag_entry = load_frag_d(mol_id,self.frag_dp,self.frag_params["compressed"])
				total_frag_entry_size += get_pyg_memory_usage(frag_entry["dag"])
				self.frag_entries[mol_id] = frag_entry
			print(f"> total_frag_entry_size: {total_frag_entry_size/1e6:.2f} MB")
		# preprocess frag dags
		if self.frag_params["preprocess"]:
			assert self.frag_params["preload"]
			self.frag_data = {}
			total_frag_data_size = 0
			for k in tqdm(list(self.frag_entries.keys()),desc="> preprocess frag",total=len(self.frag_entries)):
				frag_data = self._process_frag(self.frag_entries.pop(k),None)
				total_frag_data_size += get_pyg_memory_usage(frag_data["frag_pyg"])
				self.frag_data[k] = frag_data
			print(f"> total_frag_data_size: {total_frag_data_size/1e6:.2f} MB")

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
		# frag stuff
		if self.frag_params["preprocess"]:
			frag_data = self.frag_data[mol_id].copy()
			# update prec_mass_diff
			prec_type_mass_diff = PREC_TYPE_TO_MASS_DIFF[spec_entry["prec_type"]]
			frag_data["frag_formula_peak_mzs"] = frag_data["frag_formula_peak_mzs"] + prec_type_mass_diff
		elif self.frag_params["preload"]:
			frag_entry = self.frag_entries[mol_id].copy()
			frag_data = self._process_frag(frag_entry,spec_entry)
		else:
			frag_entry = self._load_frag_entry(mol_id)
			frag_data = self._process_frag(frag_entry,spec_entry)
		data = {**spec_data,**mol_data,**frag_data}
		return data

	def _load_frag_entry(self,mol_id):

		frag_entry = load_frag_d(
			mol_id,
			self.frag_dp,
			self.frag_params["compressed"]
		)
		return frag_entry

	def _process_frag(self,frag_entry,spec_entry):

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
			if spec_entry is not None:
				prec_type_mass_diff = PREC_TYPE_TO_MASS_DIFF[spec_entry["prec_type"]]
				formula_peak_mzs = formula_peak_mzs + prec_type_mass_diff
			frag_data["frag_formula_peak_mzs"] = formula_peak_mzs
		if self.frag_params["formula_peak_probs"]:	
			formula_peak_probs = frag_entry["formula_peak_probs"]
			formula_peak_probs = F.normalize(formula_peak_probs[:,:self.frag_params["num_isotopes"]],dim=1,p=1)
			frag_data["frag_formula_peak_probs"] = formula_peak_probs
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
		
class SpecMolFragDynamicBatchSampler(BatchSampler):
	"""Dynamically adds samples to a mini-batch up to a maximum size  either based on number of nodes on frag DAG or number of edges on frag DAG.
	 This is used to avoid CUDA OOM errors, implmentaion is inspired by PyG DynamicBatchSampler, and this should be used to replace default BatchSampler
	 This should have the same random sampling beheivor as RandomSampler
	"""
	def __init__(self, data_source: SpecMolFragDataset, max_num: int, limited_by: str = 'frag_edge',
			  	 	skip_too_big: bool = False, num_samples  = None, 
					return_batch_at = 0, sampler=None) -> None:
		"""_summary_

		Args:
			dataset (Dataset): 
			max_num (int): _description_
			mode (str, optional): _description_. Defaults to 'node'.
			shuffle (bool, optional): Samples elements randomly each epoch. Defaults to False.
			skip_too_big (bool, optional): _description_. Defaults to False.
			num_samples (Optional[int], optional): num of samples to draw. Defaults to None. if None set to all samples in dataset
			generator
			return_batch_at 
			max_batch_sample_size
		Raises:
			ValueError: _description_
			ValueError: _description_
		"""
		if not isinstance(max_num, int) or max_num <= 0:
			raise ValueError("`dag_node` should be a positive integer value "
							"(got {max_num}).")
		if limited_by not in ['frag_node', 'frag_edge']:
			raise ValueError("`limited_by` choice should be either "
							f"'frag_node' or 'dag_edge' (got '{limited_by}').")

		if num_samples is None:
			num_samples = len(data_source)

		self._data_source = data_source
		self._max_num = max_num
		self._limited_by = limited_by
		self._skip_too_big = skip_too_big
		self._max_sampling_step = num_samples
		self._return_batch_at = return_batch_at
		self._batches = []
		self._data_meta = []
		self._sampler = sampler
		self._pre_compute_batches()
		#print("_return_batch_at", self._return_batch_at, return_batch_at)

	def _pre_compute_batches(self):
		self._batches = []

		# get data meta once and cache them
		if len(self._data_meta) == 0:
			expected_total = 0
			warning_msg = ""
			for dataset_idx in range(len(self._data_source)):
				data = self._data_source[dataset_idx]
				self._data_meta.append((data['frag_pyg'].num_nodes,data['frag_pyg'].num_edges))
				n = self._data_meta[dataset_idx][0] if self._limited_by == 'frag_node' else self._data_meta[dataset_idx][1]
				if not (n > self._max_num and self._skip_too_big):
					expected_total += 1
				else:
					warning_msg += "Size of data sample at index " +\
						f"{dataset_idx} is larger than " +\
						f"{self._max_num} {self._limited_by}s " +\
						f"Got {n} {self._limited_by}s." +\
						"This sample can not fit into batch. "
					if self._skip_too_big:
						warning_msg += "Sampler will skip this to prevent CUDA OOM ERROR \n"
					else:
						warning_msg += "Attempting to fit this in to batch, this may cause CUDA OOM ERROR  \n"
					#warnings.warn(warning_msg)
					#print(warning_msg)
			if warning_msg != "":
				print("[specmolfrag_dynamic_batch_sampler]", warning_msg)
			print(f"[SpecMolFragDynamicBatchSampler] Expecting {expected_total}/{len(self._data_source)} samples with skip_too_big:{self._skip_too_big}")
		
		if self._sampler is not None:
			indices = th.tensor([idx for idx in self._sampler])
		else:
			indices = th.arange(len(self._data_source), dtype=th.long)

		# limited index to _max_sampling_step
		indices = indices[:self._max_sampling_step]

		num_processed = 0
		batch = []
		batch_n = 0
		batch_filled = False
		# while num_processed < len(indices) and num_processed < self._max_sampling_step:
		# Fill batch
		for idx in indices: #[num_processed:]:
			# Size of sample
			data = self._data_source[idx.item()]
			#n1 = data['frag_pyg'].num_nodes if  self._limited_by == 'frag_node' else data['frag_pyg'].num_edges
			n = self._data_meta[idx.item()][0] if self._limited_by == 'frag_node' else self._data_meta[idx.item()][1]
			#print(idx, n, n1)
			# assert n == n1

			if n > self._max_num and self._skip_too_big:
				continue 

			# check batch_filled condition
			if batch_n + n > self._max_num:
				# no more budget left, mini-batch filled
				batch_filled = True
						
			# check we need return at this point for ga
			#print("self._return_batch_at", self._return_batch_at)
			if self._return_batch_at > 0 \
				and num_processed > 0 \
				and num_processed % self._return_batch_at == 0:
				#print(">num_processed", num_processed)
				# Mini-batch filled
				batch_filled = True

			if batch_filled:
				self._batches.append(batch)
				batch_n = 0
				batch = []
				batch_filled = False
			
			# Add sample to current batch
			#print(idx.item())
			batch.append(idx.item())
			num_processed += 1
			batch_n += n
			
			
		if len(batch) > 0:
			self._batches.append(batch)
		#print(self._batches)
		print(f"[SpecMolFragDynamicBatchSampler]Batch indices computed, Expecting {len(self._batches)} mini-batch for next epoch")

	def __iter__(self) -> Iterator[List[int]]:
		""" we use a pre computed batche list, this way we could have a correct batch size
			PL uses last batch in progress to toggle val step in training

		Yields:
			Iterator[List[int]]: _description_
		"""
		for batch in self._batches:
			yield batch

	def __len__(self) -> int:
		""" Note The __len__() method isn't strictly required by DataLoader, but is expected in any calculation involving the length of a DataLoader.
			ref: https://pytorch.org/docs/stable/data.html#torch.utils.data.Sampler
		Returns:
			int: length of datasource
		"""
		return len(self._batches)

class GroupSampler(Sampler):

	def __init__(self, data_source: SpecMolFragDataset, sample_k=None) -> None:
		"""_summary_

		Args:
			data_source (SpecMolFragDataset): _description_
			sample_k (_type_, optional): _description_. Defaults to None.
		"""
		#if num_samples is None:
		num_samples = len(data_source)

		self._data_source = data_source
		self._num_samples = num_samples
		self._data_meta_d = {}
		self._sample_k = 3 if sample_k is None else sample_k
		self._pre_compute()

	def _pre_compute(self):

		for dataset_idx in range(len(self._data_source)):
			data = self._data_source[dataset_idx]
			group_id = data['group_id'].item()
			#mol_id = data['mol_id']

			if group_id not in self._data_meta_d:
				self._data_meta_d[group_id] = []
			self._data_meta_d[group_id].append(dataset_idx)
		
		for group_id in self._data_meta_d:
			self._data_meta_d[group_id] =  th.tensor(self._data_meta_d[group_id])

	@property
	def num_samples(self) -> int:
		# dataset size might change at runtime
		if self._num_samples is None:
			return len(self.data_source)
		return self._num_samples

	def __iter__(self) -> Iterator[int]:
		""" 
		Yields:
			Iterator[List[int]]: _description_
		"""
		sampled_indices = []
		for group_id in self._data_meta_d:
			#print(group_id)
			group_indices =self._data_meta_d[group_id]
			#print(group_indices)
			sampled_group_indices = group_indices[th.randperm(min(len(group_indices),self._sample_k))]
			#print(sampled_group_indices)
			sampled_indices.append(sampled_group_indices)
		
		sampled_indices = th.cat(sampled_indices)
		
		self._num_samples = len(sampled_indices)
		#print(sampled_indices)
		for i in th.randperm(len(sampled_indices)):
			#print(sampled_indices[i])
			yield sampled_indices[i].item()
	
	@property
	def num_samples(self) -> int:
		return self._num_samples

	def __len__(self) -> int:
		""" Note The __len__() method isn't strictly required by DataLoader
			ref: https://pytorch.org/docs/stable/data.html#torch.utils.data.Sampler
		Returns:
			int: length of datasource
		"""
		return self._num_samples 
	
def get_group_sampler(ds: SpecMolFragDataset, config_d: dict) -> WeightedRandomSampler:
	"""get WeightedRandomSampler based on input

	Args:
		ds (SpecMolFragDataset): _description_
		config_d (dict): _description_

	Returns:
		_type_: _description_
	"""
	if config_d['simple_group_sampler_type']  == 'group':
		sample_weights = 1.0/th.cat([ i['spec_per_group'] for i in ds])
	elif config_d['simple_group_sampler_type']  == 'mol':
		sample_weights = 1.0/th.cat([ i['spec_per_mol'] for i in ds])
	elif config_d['simple_group_sampler_type']  == 'group_mol':
		sample_weights = 1.0/th.cat([ i['group_per_mol'] * i['spec_per_group']  for i in ds])
	else:
		return None
	
	num_samples = min(config_d['simple_group_sampler_max'], len(ds))
	sampler = WeightedRandomSampler(sample_weights,num_samples=num_samples,replacement=False)
	return sampler

def test_dataset():

	from frag_gnn.runner import load_config
	from frag_gnn.utils.misc_utils import print_shapes
	from frag_gnn.utils.misc_utils import to_device

	template_fp = "config/template.yml"
	custom_fps = [
		"config/debug/debug_d3_nb.yml",
		# "config/debug/debug_d3_nb_nce.yml",
		"config/debug/debug_d3_neims.yml"
	]

	for custom_fp in custom_fps:

		print(custom_fp)

		config_d = load_config(template_fp, custom_fp)
		device = "cuda:0" if config_d["accelerator"] == "gpu" else "cpu"

		ds = SpecMolFragDataset(split="train", **config_d)
		dl = th.utils.data.DataLoader(
			ds,
			batch_size=8,
			shuffle=False,
			num_workers=0,
			collate_fn=ds.get_collate_fn(),
		)
		dl_iter = iter(dl)
		batch = next(dl_iter)
		print("> batch")
		print_shapes(batch)
		batch = to_device(batch,device)


if __name__ == "__main__":

	from frag_gnn.utils.misc_utils import th_temp_seed

	with th_temp_seed(420):
		test_dataset()