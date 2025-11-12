
from torch_geometric.data import Batch
from tqdm import tqdm

from fragnnet.utils.feat_utils import get_mol_fp, get_mol_graph
from fragnnet.utils.misc_utils import get_mol_graph_size, get_pyg_memory_usage
from fragnnet.dataset import BaseDataset
# TODO move this another Class
import fragnnet.massformer.data_utils as mf_data_utils

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
		spec_pp_sd: dict = None,
		mol_pp_sd: dict = None,
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
  
		if spec_pp_sd is None:
			spec_pp_sd = dict()
		if mol_pp_sd  is None:
			mol_pp_sd = dict()
   
		self.mol_params = mol_params
		self._preprocess_spec(spec_pp_sd)
		self._preprocess_mol(mol_pp_sd)
	
	@staticmethod
	def get_data_dict_types():
		# return list of data dict type used for caching
		# Spectral Preprocessed Shared Dictionary: spec_pp_sd
  		# Molecular Preprocessed Shared Dictionary: mol_pp_sd
		return ["spec_pp_sd", "mol_pp_sd"]

	def _get_mol_graph_size(self, mol_data):

		#if self.mol_params["pyg"]:
		#	mol_pyg = mol_data["mol_pyg"]
		#	mol_graph_size = get_pyg_memory_usage(mol_pyg)
		#else:
		#	mol_graph_size = 0
		mol_graph_size = get_mol_graph_size(mol_data, self.mol_params)
		return mol_graph_size

	def _preprocess_mol(self, mol_pp_sd: dict):

		# preload and pre-process molecules
		if self.mol_params["preprocess"]:
			self.mol_datas = mol_pp_sd
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
		if "mol_graff" in keys:
			# batch
			collate_data["mol_graff"] = Batch.from_data_list(collate_data["mol_graff"])
			# remove from list
			keys.remove("mol_graff")
		BaseDataset._special_collate(keys,collate_data)

	@staticmethod
	def collate_fn(data_list):
		
		batch_size, keys, collate_data = SpecMolDataset._setup_collate(data_list)
		SpecMolDataset._special_collate(keys,collate_data)
		SpecMolDataset._standard_collate(batch_size,keys,collate_data)
		return collate_data

	def training_data_sanity_check(self):
		"""
  		basic data sanity check for training time only 
		"""
		assert self.spec_df['peaks'].isna().any() == False
		assert (self.spec_df['peaks'] == '').any() == False