import h5py
import pandas as pd
from fragnnet.utils.feat_utils import get_mol_graph
from fragnnet.utils.misc_utils import get_mol_graph_size
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Dict, Any, Union, List, Optional
import torch as th

class MCESDataset(Dataset):
	"""Dataset for MCES (Maximum Common Edge Substructure) pairs.

	Each item is a pair of molecules (i, j) with their SMILES, MCES value, and PyG graphs.
	The dataset is constructed from a .npz file containing arrays for 'smiles' and 'mces'.
	"""

	def __init__(
		self, 
		split: str, 
		mces_fp: str, 
		mol_fp: str, 
		mol_params: Dict[str, Any], 
		mol_pp_sd: Optional[Dict[Union[str, int], Dict[str, Any]]] = None, 
		**kwargs
	) -> None:
		# Store split attribute (required by runner.py)
		self.split = split
		
		# Load data from .npz file (memory-mapped for efficiency)
		self.data = h5py.File(mces_fp, 'r')
		self.smiles = self.data['smiles'][:]  # Array of SMILES strings
		self.num_samples = len(self.data['mces'])  # Number of MCES pairs

		self.mol_params = mol_params
		self.mol_df = pd.read_pickle(mol_fp)
		
		# Initialize shared dict for preprocessing
		if mol_pp_sd is None:
			mol_pp_sd = dict()
		self.mol_pp_sd = mol_pp_sd
		
		# Only preprocess if data is empty (main process only)
		if self.mol_params.get("preprocess", False) and len(mol_pp_sd) == 0:
			print("WARNING: Preprocessing in dataset init - should be done in main process!")
			self._preprocess_mol(mol_pp_sd)
	
	def _preprocess_mol(self, mol_pp_sd: Dict[Union[str, int], Dict[str, Any]]) -> None:
		"""Preprocess all molecules and store in shared dictionary."""
		if self.mol_params.get("preprocess", False):
			total_mol_graph_size = 0
			for idx, mol_entry in tqdm(self.mol_df.iterrows(), desc="> preprocess mol", total=len(self.mol_df)):
				mol_data = self._process_mol(mol_entry)
				total_mol_graph_size += get_mol_graph_size(mol_data)
				mol_id: Union[str, int] = mol_entry.get("mol_id", idx)  # Use index if no mol_id column
				mol_pp_sd[mol_id] = mol_data
			print(f"> total_mol_graph_size: {total_mol_graph_size/1e6:.2f} MB")
	
	def _process_mol(self, mol_entry: pd.Series) -> Dict[str, Any]:
		"""
		Process a single molecular entry into PyG graph format only.
		
		Args:
			mol_entry: DataFrame row containing molecular data
			
		Returns:
			dict: Dictionary containing PyG graph representation
		"""
		mol_data = {}
		mol = mol_entry["mol"]
		
		# Generate PyTorch Geometric graph representation (PyG only for MCES)
		if self.mol_params.get("pyg", False):
			mol_pyg = get_mol_graph(
				mol,
				self.mol_params.get("pyg_node_feats", []),
				self.mol_params.get("pyg_edge_feats", []),
				self.mol_params.get("pyg_pe_embed_k", 0),
				self.mol_params.get("pyg_bigraph", False)
			)
			mol_data["mol_pyg"] = mol_pyg
			
		return mol_data

	def __getitem__(self, idx: int) -> Dict[str, Union[float, th.Tensor, Any]]:
		# Get MCES data
		mces: float = self.data['mces'][idx, -1]
		i: int
		j: int
		i, j = self.data['computation_indices'][idx]
		
		# Get molecule entries
		mol_entry_i: pd.Series = self.mol_df.iloc[i]
		mol_entry_j: pd.Series = self.mol_df.iloc[j]
		
		# Get molecule IDs for preprocessing lookup
		mol_id_i: Union[str, int] = mol_entry_i.get("mol_id", i)  # Use index if no mol_id column
		mol_id_j: Union[str, int] = mol_entry_j.get("mol_id", j)
		
		# Process molecules (either from cache or on-demand)
		mol_data_i: Dict[str, Any]
		mol_data_j: Dict[str, Any]
		
		if self.mol_params.get("preprocess", False) and mol_id_i in self.mol_pp_sd:
			mol_data_i = self.mol_pp_sd[mol_id_i].copy()
		else:
			mol_data_i = self._process_mol(mol_entry_i)
			
		if self.mol_params.get("preprocess", False) and mol_id_j in self.mol_pp_sd:
			mol_data_j = self.mol_pp_sd[mol_id_j].copy()
		else:
			mol_data_j = self._process_mol(mol_entry_j)
		
		# Build return dictionary
		result: Dict[str, Union[float, th.Tensor, Any]] = {'mces': mces}
		
		# Add molecule data with _i and _j suffixes
		for key, value in mol_data_i.items():
			result[f"{key}_i"] = value
		for key, value in mol_data_j.items():
			result[f"{key}_j"] = value
			
		return result

	def __len__(self) -> int:
		# Return the number of MCES pairs in the dataset
		return self.num_samples

	@staticmethod
	def get_collate_fn():
		"""
		Return the collation function for batching MCES data samples.
		
		Returns:
			function: Collation function for DataLoader
		"""
		return MCESDataset.collate_fn

	@staticmethod
	def collate_fn(data_list: List[Dict[str, Union[float, th.Tensor, Any]]]) -> Dict[str, Union[th.Tensor, List[Any]]]:
		"""
		Collation function for batching MCES dataset samples.
		
		Args:
			data_list: List of data dictionaries from dataset samples
			
		Returns:
			dict: Batched data dictionary ready for model input
		"""
		from torch_geometric.data import Batch
		import torch as th
		
		batch_size = len(data_list)
		keys = list(data_list[0].keys())
		
		# Initialize collection structure
		collate_data = {key: [] for key in keys}
		
		# Collect data for each key across all samples
		for data in data_list:
			for key in keys:
				collate_data[key].append(data[key])
		
		# Handle different data types
		for key in list(keys):
			values = collate_data[key]
			
			if key == "mces":
				# MCES values are floats, convert to tensor
				collate_data[key] = th.tensor(values, dtype=th.float)
			elif key.startswith("mol_pyg"):
				# PyG molecular graphs need special batching
				collate_data[key] = Batch.from_data_list(values)
			elif key.startswith("mol_smiles"):
				# SMILES strings - keep as list of lists
				collate_data[key] = values
			elif isinstance(values[0], th.Tensor):
				# Standard tensors - concatenate
				collate_data[key] = th.cat(values, dim=0)
			elif isinstance(values[0], list):
				# Lists - flatten
				flattened = []
				for sublist in values:
					flattened.extend(sublist)
				collate_data[key] = flattened
			else:
				# Keep as-is for other types
				collate_data[key] = values
		
		# Add batch size for reference
		collate_data["batch_size"] = th.tensor(batch_size, dtype=th.long)
		
		return collate_data

	@staticmethod
	def get_data_dict_types() -> List[str]:
		# Return list of data dict type used for caching
		# Molecular Preprocessed Shared Dictionary: mol_pp_sd
		return ["mol_pp_sd"]