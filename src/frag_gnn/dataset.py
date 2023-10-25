import pandas as pd
import os
import torch as th
import torch.nn.functional as F
from torch.utils.data.sampler import Sampler, BatchSampler
from torch.utils.data import Dataset
import numpy as np
import torch_geometric as pyg
from torch_geometric.data import Data, Batch
import sys
from tqdm import tqdm
from typing import Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union
import warnings

from frag_gnn.utils.feat_utils import batch_mols_frags, get_mol_fp, get_mol_graph, get_frag_graph
from frag_gnn.utils.spec_utils import bin_func, filter_func, batch_func, prec_type_idx, transform_nce, prec_type_mass_diff
from frag_gnn.utils.proc_utils import merge_spec_df
from frag_gnn.utils.frag_utils import load_frag_d
from frag_gnn.utils.misc_utils import flatten_lol, get_pyg_memory_usage
from frag_gnn.utils.formula_utils import PREC_TYPE_TO_MASS_DIFF


class BaseDataset(Dataset):

	def _base_init(self, spec_fp, mol_fp, split_dp, split, **kwargs):
		
		self.spec_df = pd.read_pickle(spec_fp)
		self.mol_df = pd.read_pickle(mol_fp)
		assert split in ["train","val","test"], split
		split_fp = os.path.join(split_dp,f"{split}_ids.csv")
		self.split_df = pd.read_csv(split_fp)
		self.spec_params = kwargs["spec_params"]
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
			assert not self.spec_params["merge"]
			nce = spec_entry["nce"]
			nce = th.tensor([transform_nce(nce)],dtype=th.float)
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
	def _standard_collate(batch_size,keys,collate_data):
		""" mutates collate_data """

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
			keys.remove(key)
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
		**kwargs):

		BaseDataset.__init__(self)
		self._base_init(
			spec_fp=spec_fp,
			mol_fp=mol_fp,
			split_dp=split_dp,
			split=split,
			**kwargs
		)
		self.mol_params = kwargs["mol_params"]
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
		return mol_data

	@staticmethod
	def get_collate_fn():
		return SpecMolDataset.collate_fn

	@staticmethod
	def collate_fn(data_list):
		
		batch_size, keys, collate_data = SpecMolDataset._setup_collate(data_list)
		if "mol_pyg" in keys:
			# batch
			collate_data["mol_pyg"] = Batch.from_data_list(collate_data["mol_pyg"])
			# remove from list
			keys.remove("mol_pyg")
		SpecMolDataset._standard_collate(batch_size,keys,collate_data)
		return collate_data

class SpecMolFragDataset(SpecMolDataset):

	def __init__(
		self,
		spec_fp,
		mol_fp,
		split_dp,
		split,
		frag_dp,
		**kwargs):

		BaseDataset.__init__(self)
		self._base_init(
			spec_fp=spec_fp,
			mol_fp=mol_fp,
			split_dp=split_dp,
			split=split,
			**kwargs
		)
		self.frag_dp = frag_dp
		self.mol_params = kwargs["mol_params"]
		self.frag_params = kwargs["frag_params"]
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
	def collate_fn(data_list):

		# prevent edge case causing crash
		if len(data_list) == 0:
			return  {"batch_size": th.tensor(0)}
		batch_size, keys, collate_data = SpecMolFragDataset._setup_collate(data_list)
		# special handling
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
	def __init__(self, data_source: SpecMolFragDataset, max_num: int, limited_by: str = 'frag_node',
					shuffle: bool = False, skip_too_big: bool = False, num_samples  = None, generator = None,
					return_batch_at = 0) -> None:
		"""_summary_

		Args:
			dataset (Dataset): 
			max_num (int): _description_
			mode (str, optional): _description_. Defaults to 'node'.
			shuffle (bool, optional): Samples elements randomly each epoch. Defaults to False.
			skip_too_big (bool, optional): _description_. Defaults to False.
			num_samples (Optional[int], optional): num of samples to draw. Defaults to None. if None set to all samples in dataset

		Raises:
			ValueError: _description_
			ValueError: _description_
		"""
		if not isinstance(max_num, int) or max_num <= 0:
			raise ValueError("`dag_node` should be a positive integer value "
							"(got {max_num}).")
		if limited_by not in ['frag_node', 'dag_edge']:
			raise ValueError("`limited_by` choice should be either "
							f"'frag_node' or 'dag_edge' (got '{limited_by}').")

		if num_samples is None:
			num_samples = len(data_source)

		self.data_source = data_source
		self.max_num = max_num
		self.limited_by = limited_by
		self.skip_too_big = skip_too_big
		self.max_sampling_step = num_samples
		self.shuffle = shuffle
		self.generator = generator
		self.return_batch_at = return_batch_at
		self.batches = []
		self._pre_compute_batches()
	
	def _pre_compute_batches(self):
		self.batches = []
		batch = []
		batch_n = 0
		num_steps = 0
		num_processed = 0

		# setup generator
		if self.generator is None:
			seed = int(th.empty((), dtype=th.int64).random_().item())
			generator = th.Generator()
			generator.manual_seed(seed)
		else:
			generator = self.generator

		# beacuse this is an iter, shuffle only happen once per batch
		if self.shuffle:
			indices = th.randperm(len(self.data_source), dtype=th.long)
		else:
			indices = th.arange(len(self.data_source), dtype=th.long)

		while (num_processed < len(self.data_source)
			   and num_steps < self.max_sampling_step):
			# Fill batch
			for idx in indices[num_processed:]:
				# Size of sample
				data = self.data_source[idx.item()]
				
				n = data['frag_pyg'].num_nodes if data['frag_pyg'].num_edges == 'frag_node' else data['frag_pyg'].num_edges

				if n > self.max_num:
					warning_msg = "Size of data sample at index " +\
						f"{idx} is larger than " +\
						f"{self.max_num} {self.limited_by}s " +\
						f"(got {n} {self.limited_by}s. " +\
						"This sample can not fit into batch "
					if self.skip_too_big:
						warning_msg += "and sampler will skip this to prevent CUDA OOM ERROR"
					else:
						warning_msg += "attempt to fit this in to batch and this may let to CUDA OOM ERROR"
					warnings.warn(warning_msg)
					#if skip too big is true, just move to the next iteration
					# this will have edge case where a "batch" can be empty if this is the last batch
					if self.skip_too_big:
						continue


				if  batch_n + n > self.max_num:
					# no more budget left
					break

				# Add sample to current batch
				batch.append(idx.item())
				num_processed += 1
				batch_n += n
		
				if (self.return_batch_at > 0 and num_processed > 0 and num_processed % self.return_batch_at == 0):
					# Mini-batch filled
					break

			self.batches.append(batch)
			batch = []
			batch_n = 0
			num_steps += 1

	def __iter__(self) -> Iterator[List[int]]:
		""" we use a pre computed batche list, this way we could have a correct batch size
			PL uses last batch in progress to toggle val step in training

		Yields:
			Iterator[List[int]]: _description_
		"""
		# print(">Sampler current batch", self.batches)
		for batch in self.batches:
			yield batch
		self._pre_compute_batches()

	def __len__(self) -> int:
		""" Note The __len__() method isn't strictly required by DataLoader, but is expected in any calculation involving the length of a DataLoader.
			ref: https://pytorch.org/docs/stable/data.html#torch.utils.data.Sampler
		Returns:
			int: length of datasource
		"""
		return len(self.batches)


if __name__ == "__main__":

	import pandas as pd
	from torch.utils.data import DataLoader
	from pprint import pprint
	from frag_gnn.utils.misc_utils import print_shapes

	# TODO: read stuff from a config file

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

	# get the largest batch size
	smf_b_mem_d, smf_b_mem_total = find_largest_batch(smf_dl)
	print(f"largest batch size: {smf_b_mem_total/1e6:.3f} MB")
	pprint({k:f"{v/1e6:.3f} MB" for k,v in smf_b_mem_d.items()})