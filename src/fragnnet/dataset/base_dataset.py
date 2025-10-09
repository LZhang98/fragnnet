import pandas as pd
import os
import torch as th
from torch.utils.data import Dataset
import numpy as np

from tqdm import tqdm
import copy

from fragnnet.utils.spec_utils import batch_func
from fragnnet.utils.proc_utils import merge_spec_df
from fragnnet.utils.misc_utils import flatten_lol, get_tensor_dict_memory_usage
from fragnnet.utils.data_utils import fill_missing_ace, fill_missing_nce, seq_apply_df_rows
from fragnnet.utils.formula_utils import PREC_TYPE_TO_MASS_DIFF

class BaseDataset(Dataset):

	def _base_init(
		self,
		spec_fp: str,
		mol_fp: str,
		split_dp: str,
		split: str,
		subsample_params: dict,
		spec_params: dict):

		self.split = split
		self.subsample_params = subsample_params
		self.spec_params = spec_params
		spec_df, mol_df, um_spec_df, split_df, id_key, ce_key = BaseDataset._setup_dfs(
			spec_fp_or_df=spec_fp,
			mol_fp_or_df=mol_fp,
			split_dp=split_dp,
			splits=[split],
			subsample_params=subsample_params,
			spec_params=spec_params
		)
		self.spec_df = spec_df
		self.mol_df = mol_df
		self.um_spec_df = um_spec_df
		self.split_df = split_df
		self.id_key = id_key
		self.ce_key = ce_key
		self._compute_counts()
		self._setup_prec_type_to_idx()
		self._setup_inst_type_to_idx()

	@staticmethod
	def _setup_dfs( 
		spec_fp_or_df: str|pd.DataFrame,
		mol_fp_or_df: str|pd.DataFrame,
		split_dp: str,
		splits: list[str],
		subsample_params: dict,
		spec_params: dict):
		
		spec_df = spec_fp_or_df if isinstance(spec_fp_or_df, pd.DataFrame) else pd.read_pickle(spec_fp_or_df)
		mol_df = mol_fp_or_df if isinstance(mol_fp_or_df, pd.DataFrame) else pd.read_pickle(mol_fp_or_df)
  
		split_dfs = []
		for split in splits:
			# assert split in ["train","val","test","secondary","predict_only"], split
			if split == "predict_only":
				assert len(splits) == 1, splits
				# predict_all split, just include everything, this is used for prediction
				split_df = pd.DataFrame()
				# fill these to keep compatible
				split_df["spec_id"] = spec_df["spec_id"]
				split_df["mol_id"] = spec_df["mol_id"]
				split_df["group_id"] = spec_df["spec_id"]
			else:	
				split_fp = os.path.join(split_dp,f"{split}_ids.csv")
				assert os.path.isfile(split_fp), split_fp
				split_df = pd.read_csv(split_fp)
			split_dfs.append(split_df)
		split_df = pd.concat(split_dfs,ignore_index=True).reset_index(drop=True)
   
		# select spectra
		spec_df = spec_df[spec_df["spec_id"].isin(split_df["spec_id"])]
		# select molecules
		mol_df = mol_df[mol_df["mol_id"].isin(split_df["mol_id"])]
		assert np.all(np.unique(mol_df["mol_id"]) == np.unique(spec_df["mol_id"]))
		
		# covert ace and nce if there is data
		assert not (spec_params["ace"] and spec_params["nce"])
		if spec_params["ace"]:
			ce_key = "ace"
			spec_df.loc[:,ce_key] = seq_apply_df_rows(spec_df, lambda row: fill_missing_ace(row, ace_colname="ace", nce_colname="nce"))
			spec_df.loc[:,f"{ce_key}_extra_1"] = seq_apply_df_rows(spec_df, lambda row: fill_missing_ace(row, ace_colname="ace_extra_1", nce_colname="nce_extra_1"))
			spec_df.loc[:,f"{ce_key}_extra_2"] = seq_apply_df_rows(spec_df, lambda row: fill_missing_ace(row, ace_colname="ace_extra_2", nce_colname="nce_extra_2"))
		elif spec_params["nce"]:
			ce_key = "nce"
			spec_df.loc[:,ce_key] = seq_apply_df_rows(spec_df, lambda row: fill_missing_nce(row, ace_colname="ace", nce_colname="nce"))
			spec_df.loc[:,f"{ce_key}_extra_1"] = seq_apply_df_rows(spec_df, lambda row: fill_missing_nce(row, ace_colname="ace_extra_1", nce_colname="nce_extra_1"))
			spec_df.loc[:,f"{ce_key}_extra_2"] = seq_apply_df_rows(spec_df, lambda row: fill_missing_nce(row, ace_colname="ace_extra_2", nce_colname="nce_extra_2"))
		else:
			ce_key = None

		if ce_key is not None:
			print(spec_df[spec_df[ce_key] >= spec_df[f"{ce_key}_extra_1"]])
			print(spec_df[spec_df[ce_key] >= spec_df[f"{ce_key}_extra_2"]])
			print(f">> replace missing {ce_key} with -1, and fill ramped/stepped ces accordingly")
			# First fill extra columns with -1 (these stay as floats)
			spec_df.loc[:,f"{ce_key}_extra_1"] = spec_df.loc[:,f"{ce_key}_extra_1"].fillna(-1)
			spec_df.loc[:,f"{ce_key}_extra_2"] = spec_df.loc[:,f"{ce_key}_extra_2"].fillna(-1)
			
			def _fill_ce_list(row, ce_key):
				ce = row[ce_key]
				ce_extra_1 = row[f"{ce_key}_extra_1"]
				ce_extra_2 = row[f"{ce_key}_extra_2"]
				
				# Handle missing main CE value
				if pd.isna(ce):
					ce = -1
					
				if ce != -1 and ce_extra_1 != -1 and ce_extra_2 != -1:
					# for stepped ce
					# ce is the lower bound, ce_extra_1 is the middle, ce_extra_2 is the upper bound
					assert ce < ce_extra_1 < ce_extra_2, row
					return [ce, ce_extra_1, ce_extra_2]
				elif ce != -1 and ce_extra_1 != -1:
					# for ramped ce
					ramped_step_size = 1.0
					# assert ce_extra_1 > ce, (ce, ce_extra_1)
					# assert (ce_extra_1 - ce) > ramped_step_size, (ce, ce_extra_1, ramped_step_size)
					ces = np.arange(ce, ce_extra_1 + 0.1, ramped_step_size).tolist()
					return ces
				else:
					# for single ce
					return [ce]
			
			# Apply the function to create lists - this will set the correct dtype
			result_series = seq_apply_df_rows(spec_df, lambda row: _fill_ce_list(row, ce_key))
			spec_df.loc[:, ce_key] = result_series.astype(object)

		if spec_params["test_ces"] is not None and ce_key is not None and "test" in splits:
			test_split_fp = os.path.join(split_dp,"test_ids.csv")
			test_split_df = pd.read_csv(test_split_fp)
			orig_test_spec_df = spec_df[spec_df["spec_id"].isin(test_split_df["spec_id"])]
			test_drop_ids = orig_test_spec_df[~(orig_test_spec_df[ce_key].isin(np.array(spec_params["test_ces"])))][["spec_id","group_id"]]
			spec_df = spec_df[~(spec_df["spec_id"].isin(test_drop_ids["spec_id"]))]
			new_test_spec_df = spec_df[spec_df["spec_id"].isin(test_split_df["spec_id"])]
			orig_test_spec_count = orig_test_spec_df["spec_id"].nunique()
			orig_test_group_count = orig_test_spec_df["group_id"].nunique()
			new_test_spec_count = new_test_spec_df["spec_id"].nunique()
			new_test_group_count = new_test_spec_df["group_id"].nunique()
			print(f">> Dropping spectra unless {ce_key} is one of {spec_params['test_ces']}")
			print(f"> Before drop: {orig_test_spec_count} spectra, {orig_test_group_count} groups")
			print(f"> After drop: {new_test_spec_count} spectra, {new_test_group_count} groups")

		# merge spectra
		um_spec_df = spec_df[["dset","dset_spec_id","spec_id","group_id","mol_id","peaks"]].copy()
		if spec_params["merge"]:
			spec_df = merge_spec_df(spec_df,keep_ces=spec_params["merge_keep_ces"])
			id_key = "group_id"
		else:
			id_key = "spec_id"
   
		# subsample
		if subsample_params.get(split,False) and subsample_params["subsample_size"] > 0:
			if isinstance(subsample_params["subsample_size"],int):
				n = subsample_params["subsample_size"]
				frac = None
			else:
				assert isinstance(subsample_params["subsample_size"],float)
				n = None
				frac = subsample_params["subsample_size"]
			spec_df = spec_df.sample(
				n=n,
				frac=frac,
				random_state=subsample_params["subsample_seed"],
				replace=False)
			mol_df = mol_df[mol_df["mol_id"].isin(spec_df["mol_id"])]
			um_spec_df = um_spec_df[um_spec_df[id_key].isin(spec_df[id_key])]

		# reset indices
		spec_df = spec_df.reset_index(drop=True)
		mol_df = mol_df.reset_index(drop=True)
		# use mol_id as index for speedy access
		mol_df = mol_df.set_index("mol_id",drop=False).sort_index().rename_axis(None)
		return spec_df, mol_df, um_spec_df, split_df, id_key, ce_key

	def _compute_counts(self):

		self.group_per_mol = self.spec_df[["mol_id","group_id"]].drop_duplicates().groupby("mol_id").size().to_dict()
		if self.spec_params["merge"]:
			self.spec_per_mol = copy.deepcopy(self.group_per_mol)
			self.spec_per_group = {group_id: 1 for group_id in self.spec_df["group_id"].unique()}
		else:
			self.spec_per_mol = self.spec_df[["mol_id","spec_id"]].drop_duplicates().groupby("mol_id").size().to_dict()
			self.spec_per_group = self.spec_df[["group_id","spec_id"]].drop_duplicates().groupby("group_id").size().to_dict()

	def get_group_mol_stats(self):

		group_ids = []
		mol_ids = []
		spec_per_group_stats = []
		spec_per_mol_stats = []
		group_per_mol_stats = []
		for row_idx, row in self.spec_df.iterrows():
			spec_entry = row
			mol_id = spec_entry["mol_id"]
			group_id = spec_entry["group_id"]
			spec_per_group = self.spec_per_group[group_id]
			spec_per_mol = self.spec_per_mol[mol_id]
			group_per_mol = self.group_per_mol[mol_id]
			group_ids.append(group_id)
			spec_per_group_stats.append(spec_per_group)
			mol_ids.append(mol_id)
			spec_per_mol_stats.append(spec_per_mol)
			group_per_mol_stats.append(group_per_mol)
		group_ids = th.tensor(group_ids)
		mol_ids = th.tensor(mol_ids)
		spec_per_group_stats = th.tensor(spec_per_group_stats)
		spec_per_mol_stats = th.tensor(spec_per_mol_stats)
		group_per_mol_stats = th.tensor(group_per_mol_stats)
		return group_ids, mol_ids, spec_per_group_stats, spec_per_mol_stats, group_per_mol_stats

	@staticmethod
	def get_data_dict_types():
		# return list of data dict type used for caching
		# Spectral Preprocessed Shared Dictionary: spec_pp_sd
		return ["spec_pp_sd"]

	def _preprocess_spec(self,spec_pp_sd: dict):

		# preload and pre-process spectra
		if self.spec_params["preprocess"]:
			self.spec_datas = spec_pp_sd
			total_spec_data_size = 0
			for idx, spec_entry in tqdm(self.spec_df.iterrows(),desc="> preprocess spec",total=len(self.spec_df)):
				spec_data = self._process_spec(spec_entry)
				total_spec_data_size += get_tensor_dict_memory_usage(**spec_data)
				self.spec_datas[idx] = spec_data
			print(f"> total_spec_data_size: {total_spec_data_size/1e6:.2f} MB")

	@staticmethod
	def _get_mzs_ints(peaks:list):
		"""
  		convert peaks to tensors, it is caller's responsibility to make sure data is valid 
		"""
		mzs, ints = [], []
		for peak in peaks:
			p_mz, p_int = peak
			mzs.append(p_mz)
			ints.append(p_int)

		mzs = th.tensor(mzs,dtype=th.float)
		ints = th.tensor(ints,dtype=th.float)
		# mzs, ints = filter_func(mzs, ints, self.spec_params["ints_thresh"], self.spec_params["mz_max"])
		return mzs, ints
	
	def _setup_prec_type_to_idx(self):

		prec_types = sorted(self.spec_params["prec_types"])
		assert all(prec_type in PREC_TYPE_TO_MASS_DIFF for prec_type in prec_types), prec_types
		self.prec_type_to_idx = {prec_type: idx for idx, prec_type in enumerate(prec_types)}
		self.idx_to_prec_type = {idx: prec_type for idx, prec_type in enumerate(prec_types)}
		self.num_prec_types = len(prec_types)

	def _setup_inst_type_to_idx(self):

		inst_types = sorted(self.spec_params["inst_types"])
		self.inst_type_to_idx = {inst_type: idx for idx, inst_type in enumerate(inst_types)}
		self.idx_to_inst_type = {idx: inst_type for idx, inst_type in enumerate(inst_types)}
		self.num_inst_types = len(inst_types)

	def _process_spec(self,spec_entry):
		spec_data = {}
		# peak data
		mzs, ints = BaseDataset._get_mzs_ints(spec_entry["peaks"])
		if self.spec_params["sparse"]:
			# get sparse spectrum
			spec_data["spec_mzs"] = mzs
			spec_data["spec_ints"] = ints
		# metadata
		if self.spec_params["prec_type"]:
			prec_type = spec_entry["prec_type"]
			prec_type = th.tensor([self.prec_type_to_idx[prec_type]],dtype=th.long)
			spec_data["spec_prec_type"] = prec_type
		if self.spec_params["prec_type_str"]:
			prec_type_str = spec_entry["prec_type"]
			spec_data["spec_prec_type_str"] = np.array([prec_type_str])
		if self.spec_params["inst_type"]:
			inst_type = spec_entry["inst_type"]
			inst_type = th.tensor([self.inst_type_to_idx[inst_type]],dtype=th.long)
			spec_data["spec_inst_type"] = inst_type
		if self.spec_params["prec_mass_diff"]:
			prec_type = spec_entry["prec_type"]
			mass_diff = th.tensor([PREC_TYPE_TO_MASS_DIFF[prec_type]],dtype=th.float)
			spec_data["spec_prec_mass_diff"] = mass_diff
		if self.spec_params["nce"] or self.spec_params["ace"]:
			assert self.ce_key is not None
			assert (not self.spec_params["merge"]) or self.spec_params["merge_keep_ces"]
			assert not (self.spec_params["ace"] and self.spec_params["nce"])
			ce = spec_entry[self.ce_key]
			# both merged and unmerged spectra are stored as list of ces
			# for merged spectra, it is list of ces used to merge
			# for unmerged spectra, it is list of ces available include both ramped/stepped ces
			assert isinstance(ce,list), type(ce)
			ce = th.tensor(ce,dtype=th.float)
			spec_data["spec_ce"] = ce
		if self.spec_params["prec_mz"]:
			prec_mz = spec_entry["prec_mz"]
			prec_mz = th.tensor([float(prec_mz)],dtype=th.float)
			spec_data["spec_prec_mz"] = prec_mz
		if self.spec_params["unique_id"]:
			unique_id = spec_entry[self.id_key]
			unique_id = th.tensor([unique_id],dtype=th.long)
			spec_data["spec_unique_id"] = unique_id
			spec_data['group_id'] =  th.tensor([spec_entry['group_id']],dtype=th.long)
			spec_data['mol_id'] = spec_entry['mol_id'] # mol id does not need to be an int #th.tensor([spec_entry['mol_id']],dtype=th.long)
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
		if 'spec_ce' in keys:
			# create batch_idxs
			ces, _, batch_idxs = batch_func(
				collate_data['spec_ce'],
				collate_data['spec_ce'] # duplicate for compatibility
			)
			collate_data['spec_ce'] = ces
			collate_data["spec_ce_batch_idxs"] = batch_idxs
			# remove from list
			keys.remove('spec_ce')
				
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
			elif isinstance(values[0],np.ndarray):
				# cat
				values = np.concatenate(values,axis=0)
				collate_data[key] = values
			elif key in ["mol_id"]:
				collate_data[key] = values
			else:
				raise TypeError(f"Unsupported type: {key} {type(values[0])}")
		# remove everything
		keys.clear()
		# add batch size
		collate_data["batch_size"] = th.tensor(batch_size, dtype=th.long)

	@staticmethod
	def collate_fn(data_list):

		raise NotImplementedError()
