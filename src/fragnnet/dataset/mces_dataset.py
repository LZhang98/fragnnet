
import h5py
import pandas as pd
from fragnnet.utils.data_utils import mol_from_smiles, par_apply_series
from fragnnet.utils.feat_utils import get_mol_graph
from torch.utils.data import Dataset

class MCESDataset(Dataset):
	"""Dataset for MCES (Maximum Common Edge Substructure) pairs.

	Each item is a pair of molecules (i, j) with their SMILES, MCES value, and PyG graphs.
	The dataset is constructed from a .npz file containing arrays for 'smiles' and 'mces'.
	"""

	def __init__(self, split, mces_fp, mol_fp, mol_params: dict, **kwargs):
		# Load data from .npz file (memory-mapped for efficiency)
		self.data = h5py.File(mces_fp, 'r')
		self.smiles = self.data['smiles'][:]  # Array of SMILES strings
		self.num_samples = len(self.data['mces'])  # Number of MCES pairs

		#self.mol_df = mol_from_smiles(self.smiles, mol_fp=mol_fp)
		self.mol_params = mol_params
		self.mol_df = pd.read_pickle(mol_fp) #pd.DataFrame({'smiles': self.smiles})
		#(self.mol_df.head())
		self.mol_df.loc[:, "mol"] = par_apply_series(self.mol_df["smiles"], lambda s: mol_from_smiles(s, ml_standardize=False))
		
  		#self.mol_df.loc[:, "pyg"] = par_apply_series(self.mol_df["mol"], lambda mol: get_mol_graph(mol, self.mol_params["pyg_node_feats"],
		#		self.mol_params["pyg_edge_feats"],
		#		self.mol_params["pyg_pe_embed_k"],
		#		self.mol_params["pyg_bigraph"]))
		#self.mol_pygs = [get_mol_graph(mol, self.mol_params["pyg_node_feats"],
		#		self.mol_params["pyg_edge_feats"],
		#		self.mol_params["pyg_pe_embed_k"],
		#		self.mol_params["pyg_bigraph"]) 
        #for mol in self.mol_df["mol"]]
  
	def __getitem__(self, idx):
		# TODO Fix this,
		mces = self.data['mces'][idx,-1]
		i,j = self.data['computation_indices'][idx]
		#smiles_i = self.smiles[i]
		#smiles_j = self.smiles[j]
		mol_pyg_i = self.mol_df["pyg"][i]
		mol_pyg_j = self.mol_df["pyg"][j]
		return {
			#'smiles_i': smiles_i,
			#'smiles_j': smiles_j,
			'mces': mces,
			'mol_pyg_i': mol_pyg_i,
			'mol_pyg_j': mol_pyg_j,
		}

	def __len__(self):
		# Return the number of MCES pairs in the dataset
		return self.num_samples

	@staticmethod
	def get_data_dict_types():
		# Return list of data dict type used for caching
		return []