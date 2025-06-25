
import numpy as np
import h5py
from fragnnet.utils.data_utils import mol_from_smiles
from fragnnet.utils.feat_utils import get_mol_graph
from torch.utils.data import Dataset

class McesDataset(Dataset):
	"""Dataset for MCES (Maximum Common Edge Substructure) pairs.

	Each item is a pair of molecules (i, j) with their SMILES, MCES value, and PyG graphs.
	The dataset is constructed from a .npz file containing arrays for 'smiles' and 'mces'.
	"""

	def __init__(self, hdf5_fp):
		# Load data from .npz file (memory-mapped for efficiency)
		self.data = h5py.File(hdf5_fp, 'r')
		self.smiles = self.data['smiles'][:]  # Array of SMILES strings
		self.mces = self.data['mces'][:]   # Array of MCES values (pairwise)
		self.num_samples = len(self.mces)  # Number of MCES pairs
		# TODO test this, we may need to do this on the fly
  		# Precompute RDKit molecule objects and PyG graphs for each SMILES
		self.mols = [mol_from_smiles(x) for x in self.smiles]
		self.mol_pygs = [get_mol_graph(x) for x in self.mols]
	
	def __getitem__(self, idx):

		mces = self.data['mces'][idx,0]
		i,j = self.data['computation_indices'][idx,:-2]
		smiles_i = self.smiles[i]
		smiles_j = self.smiles[j]
		mol_pyg_i = self.mol_pygs[i]
		mol_pyg_j = self.mol_pygs[j]
		return {
			'smiles_i': smiles_i,
			'smiles_j': smiles_j,
			'mces': mces,
			'mol_pyg_i': mol_pyg_i,
			'mol_pyg_j': mol_pyg_j,
		}

	def __len__(self):
		# Return the number of MCES pairs in the dataset
		return self.num_samples