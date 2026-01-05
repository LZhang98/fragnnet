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
from fragnnet.utils.frag_utils import load_frag_d, NODE_FEAT_TO_IDX
from fragnnet.utils.misc_utils import get_pyg_memory_usage
from fragnnet.utils.formula_utils import PREC_TYPE_TO_MASS_DIFF, PREC_TYPE_TO_CMF_MASS_DIFF
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
		spec_pp_sd: dict|None = None, # preprocessed spec data
		mol_pp_sd: dict|None = None, # preprocessed mol data
		frag_pl_sd: dict|None = None, # preloaded frag data
		frag_pp_sd: dict|None = None, # preprocessed frag data
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
				frag_data = self._process_neutral_frag(self.frag_entries.pop(k))
				total_frag_data_size += get_pyg_memory_usage(frag_data["frag_pyg"])
				self.frag_data[k] = frag_data
			print(f"> total_frag_data_size: {total_frag_data_size/1e6:.2f} MB")
			# remove them from the entries
			self.frag_entries = {}

	def __getitem__(self, idx):
		spec_entry = self.spec_df.iloc[idx]
		mol_id = spec_entry["mol_id"]
		mol_entry = self.mol_df.loc[mol_id]
  
		# spectra data
		if self.spec_params["preprocess"]:
			# we don't need to deep copy here because spec data won't be modified
			spec_data = self.spec_datas[idx]
		else:
			spec_data = self._process_spec(spec_entry)
   
		# molecule data
		if self.mol_params["preprocess"]:
	  		# we don't need to deep copy here because mol data won't be modified
			mol_data = self.mol_datas[mol_id]
		else:
			mol_data = self._process_mol(mol_entry)
   
		# frag data
		if self.frag_params["preprocess"]:
			frag_data = self.frag_data[mol_id].copy()
   
			#formula_peak_mzs = frag_entry["formula_peak_mzs"]
			#formula_peak_mzs = formula_peak_mzs[:,:self.frag_params["num_isotopes"]]
			# in this case we have not applied precursor type changes yet
			# frag_formula_peak_mzs  is in neutral form
			neutral_peaks = frag_data["frag_formula_peak_mzs"]
			frag_data = self._apply_fragment_mz_change(frag_data, spec_entry['prec_type'], neutral_peaks, clone_pyg=True)
	
		elif self.frag_params["preload"]:
			frag_entry = self.frag_entries[mol_id]
			#frag_entry['frag_pyg'] = frag_entry['frag_pyg'].clone() # clone to avoid modifying the original one
			frag_data = self._process_neutral_frag(frag_entry)
			frag_data = self._apply_fragment_mz_change(frag_data, spec_entry['prec_type'], frag_data["frag_formula_peak_mzs"], clone_pyg=True)
		else:
			frag_entry = self._load_frag_entry(mol_id)
			frag_data = self._process_neutral_frag(frag_entry)
			frag_data = self._apply_fragment_mz_change(frag_data, spec_entry['prec_type'], frag_data["frag_formula_peak_mzs"], clone_pyg=False)
   
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


	def _process_neutral_frag(self, frag_entry: dict) -> dict:
		"""
		Process the fragmentation dictionary to generate neutral fragment data.

		This method processes the fragmentation graph (frag DAG) and associated data,
		such as formula peak probabilities, formula strings, and m/z values.
		Args:
			frag_entry (dict): Fragmentation entry containing the DAG and related data.

		Returns:
			dict: A dictionary containing processed fragmentation data, including:
				  - 'frag_pyg': PyG graph of the fragmentation DAG.
				  - 'frag_formula_peak_probs': Normalized formula peak probabilities.
				  - 'frag_formula_str': Array of formula strings.
				  - 'frag_formula_peak_mzs': Adjusted m/z values for the fragmentation peaks.
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
   
		if self.frag_params["formula_peak_mzs"]:
			formula_peak_mzs = frag_entry["formula_peak_mzs"]
			formula_peak_mzs = formula_peak_mzs[:,:self.frag_params["num_isotopes"]]
			frag_data["frag_formula_peak_mzs"] = formula_peak_mzs

		return frag_data

	def _apply_fragment_mz_change(self, frag_data: dict, prec_type: str, neutral_peaks: th.Tensor, clone_pyg: bool = False) -> dict:
		"""
		Apply precursor mass difference for Charge Remote Fragmentation (CRF) and 
		Charge Migration Fragmentation (CMF) to frag_data.

		This method updates 'frag_formula_peak_mzs', 'frag_formula_peak_probs', 
		'frag_formula_str', and 'frag_pyg' in frag_data.

		Args:
			frag_data (dict): The dictionary containing fragmentation data.
			prec_type (str): The precursor type string.
			neutral_peaks (th.Tensor): The neutral formula peak m/z values.
			clone_pyg (bool): Whether to clone the PyG object before modification. 
							  Set to True if frag_data['frag_pyg'] is a reference to cached data.

		Returns:
			dict: The updated frag_data dictionary.
		"""
		prec_type_mass_diff = PREC_TYPE_TO_MASS_DIFF[prec_type]
		crf_peaks = neutral_peaks + prec_type_mass_diff
		
		# Charge Migration Fragmentation stuff
		if self.frag_params["include_cmf"] and prec_type not in ["[M+H]+","[M-H]-", "[M]+", "M-"]:
			# Check if cmf_h_formulae_idx exists in pyg node features:
			assert "cmf_h_formulae_idx" in self.frag_params["pyg_node_feats"], "cmf_h_formulae_idx not in pyg_node_feats"
			cmf_type_mass_diff = PREC_TYPE_TO_CMF_MASS_DIFF[prec_type]
			num_formulae = neutral_peaks.shape[0]
			cmf_peaks = neutral_peaks[1:] + cmf_type_mass_diff # first peak is for NULL

			# Double all formula-related arrays
			if self.frag_params["formula_peak_mzs"]:
				frag_data["frag_formula_peak_mzs"] = th.cat([crf_peaks, cmf_peaks], dim=0)
			if self.frag_params["formula_peak_probs"]:
				frag_data["frag_formula_peak_probs"] = th.cat(
					[frag_data["frag_formula_peak_probs"], frag_data["frag_formula_peak_probs"][1:]], dim=0
				)
			if "frag_formula_str" in frag_data:
				frag_data["frag_formula_str"] = np.concatenate(
					[frag_data["frag_formula_str"], frag_data["frag_formula_str"][1:]], axis=0
				)
	
			# update pyg
			if clone_pyg:
				frag_pyg = frag_data["frag_pyg"].clone()
			else:
				frag_pyg = frag_data["frag_pyg"]
				
			frag_node_feat_idxs = frag_pyg.node_feat_idxs[0]
			crf_node_feat_idx = NODE_FEAT_TO_IDX['h_formulae_idx']
			cmf_node_feat_idx = NODE_FEAT_TO_IDX['cmf_h_formulae_idx']
			crf_formula_matrix = frag_pyg.x[:,frag_node_feat_idxs[crf_node_feat_idx]:frag_node_feat_idxs[crf_node_feat_idx+1]]
			
			# update cmf formula matrix if crf_formula_matrix value is not zero
			# if crf_formula_matrix is zero, that means h_transfer is not possible, then cmf formula matrix should be zero
			# cmf formula matrix is shifted by num_formulae-1, as forumlea[0] is null formula
			frag_pyg.x[:,frag_node_feat_idxs[cmf_node_feat_idx]:frag_node_feat_idxs[cmf_node_feat_idx+1]] = \
				th.where(crf_formula_matrix != 0, crf_formula_matrix + num_formulae-1, crf_formula_matrix)
			frag_data["frag_pyg"] = frag_pyg
		else:
			frag_data["frag_formula_peak_mzs"] = crf_peaks
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
		
