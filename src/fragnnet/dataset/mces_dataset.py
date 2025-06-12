
class McesDataset(Dataset):
	"""Dataset for MCES (Maximum Common Edge Substructure) pairs.

	Each item is a pair of molecules (i, j) with their SMILES, MCES value, and PyG graphs.
	The dataset is constructed from a .npz file containing arrays for 'smiles' and 'mces'.
	"""

	def __init__(self, npz_fp):
		# Load data from .npz file (memory-mapped for efficiency)
		self.data = np.load(npz_fp, mmap_mode='r')
		self.smiles = self.data['smiles']  # Array of SMILES strings
		self.mces = self.data['mces']      # Array of MCES values (pairwise)
		self.num_samples = len(self.mces)  # Number of MCES pairs
		# Precompute RDKit molecule objects and PyG graphs for each SMILES
		self.mols = [mol_from_smiles(x) for x in self.smiles]
		self.mol_pygs = [get_mol_graph(x) for x in self.mols]

	def _upper_triangular_index_to_pair(self, idx):
		"""
		Inverse mapping: flat index -> (i, j) for upper-triangular matrix storage.
		Used to recover the original molecule pair from a flat MCES index.
		See: https://stackoverflow.com/a/27086195
		"""
		N = self.num_samples
		i = N - 2 - int(np.floor(np.sqrt(-8*idx + 4*N*(N-1)-7)/2.0 - 0.5))
		j = idx + i + 1 - N*(N-1)//2 + (N-i)*((N-i)-1)//2
		return i, j
	
	def __getitem__(self, idx):
		# Get the (i, j) molecule pair for this MCES index
		i, j = self._upper_triangular_index_to_pair(idx)
		smiles_i = self.smiles[i]
		smiles_j = self.smiles[j]
		mces = self.data['mces'][idx]
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