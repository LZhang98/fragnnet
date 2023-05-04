""" mol_graph.py (from SCARF)

Classes to featurize molecules into a graph with onehot concat feats on atoms
and bonds. Inspired by the dgllife library.

"""
from rdkit import Chem
import numpy as np
import torch as th
import networkx as nx
import torch_geometric as pyg


atom_feat_registry = {}
bond_feat_registry = {}


def register_bond_feat(cls):
	"""register_bond_feat."""
	bond_feat_registry[cls.name] = {"fn": cls.featurize, "feat_size": cls.feat_size}
	return cls


def register_atom_feat(cls):
	"""register_atom_feat."""
	atom_feat_registry[cls.name] = {"fn": cls.featurize, "feat_size": cls.feat_size}
	return cls


class MolGraph:
	def __init__(
		self,
		atom_feats: list = [
			"a_onehot",
			"a_degree",
			"a_hybrid",
			"a_formal",
			"a_radical",
			"a_ring",
			"a_mass",
			"a_chiral",
		],
		bond_feats: list = ["b_degree"],
		pe_embed_k: int = 0,
	):
		"""__init__

		Args:
			atom_feats (list)
			bond_feats (list)
			pe_embed_k (int)

		"""
		if pe_embed_k > 0:
			raise NotImplementedError
		self.pe_embed_k = pe_embed_k
		self.atom_feats = atom_feats
		self.bond_feats = bond_feats
		self.a_featurizers = []
		self.b_featurizers = []

		self.num_atom_feats = 0
		self.num_bond_feats = 0

		for i in self.atom_feats:
			if i not in atom_feat_registry:
				raise ValueError(f"Feat {i} not recognized")
			feat_obj = atom_feat_registry[i]
			self.num_atom_feats += feat_obj["feat_size"]
			self.a_featurizers.append(feat_obj["fn"])

		for i in self.bond_feats:
			if i not in bond_feat_registry:
				raise ValueError(f"Feat {i} not recognized")
			feat_obj = bond_feat_registry[i]
			self.num_bond_feats += feat_obj["feat_size"]
			self.b_featurizers.append(feat_obj["fn"])

		self.num_atom_feats += self.pe_embed_k

	def get_mol_graph(
		self,
		mol: Chem.Mol,
		bigraph: str = True,
	) -> dict:
		"""get_mol_graph.

		Args:
			mol (Chem.Mol):
			bigraph (bool): If true, double all edges.

		Return:
			dict:
				"atom_feats": np.ndarray (|N| x d_n)
				"bond_feats": np.ndarray (|E| x d_e)
				"bond_tuples": np.ndarray (|E| x 2)

		"""
		all_atoms = mol.GetAtoms()
		all_bonds = mol.GetBonds()
		bond_feats = []
		bond_tuples = []
		atom_feats = []
		for bond in all_bonds:
			strt = bond.GetBeginAtomIdx()
			end = bond.GetEndAtomIdx()
			bond_tuples.append((strt, end))
			bond_feat = []
			for fn in self.b_featurizers:
				bond_feat.extend(fn(bond))
			bond_feats.append(bond_feat)

		for atom in all_atoms:
			atom_feat = []
			for fn in self.a_featurizers:
				atom_feat.extend(fn(atom))
			atom_feats.append(atom_feat)

		atom_feats = np.array(atom_feats)
		bond_feats = np.array(bond_feats)
		bond_tuples = np.array(bond_tuples)

		# Add doubles
		if bigraph:
			rev_bonds = np.vstack([bond_tuples[:, 1], bond_tuples[:, 0]]).transpose(
				1, 0
			)
			bond_tuples = np.vstack([bond_tuples, rev_bonds])
			bond_feats = np.vstack([bond_feats, bond_feats])
		return {
			"atom_feats": atom_feats,
			"bond_feats": bond_feats,
			"bond_tuples": bond_tuples,
		}

	def get_networkx_graph(self, mol: Chem.Mol, bigraph: str = True):
		"""get_networkx_graph.

		Args:
			mol (Chem.Mol):
			bigraph (bool): If true, double all edges.

		Return:
			networkx graph object
		"""
		mol_graph = self.get_mol_graph(mol, bigraph=bigraph)

		bond_inds = mol_graph["bond_tuples"]
		bond_feats = mol_graph["bond_feats"]
		atom_feats = mol_graph["atom_feats"]

		g = nx.Graph()
		g.add_nodes_from(range(atom_feats.shape[0]))
		g.add_edges_from(bond_inds)

		nx.set_node_attributes(g, values=atom_feats, name="h")
		nx.set_edge_attributes(g, values=bond_feats, name="e")

		return g

	def get_pyg_graph(self, mol: Chem.Mol, bigraph: str = True):

		mol_graph = self.get_mol_graph(mol, bigraph=bigraph)

		bond_inds = mol_graph["bond_tuples"]
		bond_feats = mol_graph["bond_feats"]
		atom_feats = mol_graph["atom_feats"]

		g = pyg.data.Data(
			x=th.from_numpy(atom_feats).float(),
			edge_index=th.from_numpy(bond_inds).long().transpose(1, 0),
			edge_attr=th.from_numpy(bond_feats).float(),
		)

		if self.pe_embed_k > 0:
			raise NotImplementedError()
			# pe_embeds = nn_utils.random_walk_pe(
			# 	g,
			# 	k=self.pe_embed_k,
			# )
			# x = th.cat((x, pe_embeds), dim=-1)
		
		return g		

class FeatBase:
	"""FeatBase.

	Extend this class for atom and bond featurizers

	"""

	feat_size = 0
	name = "base"

	@classmethod
	def featurize(cls, x) -> list:
		raise NotImplementedError()


@register_atom_feat
class AtomOneHot(FeatBase):
	name = "a_onehot"
	allowable_set = [
		"C",
		"N",
		"O",
		"S",
		"F",
		"Si",
		"P",
		"Cl",
		"Br",
		"Mg",
		"Na",
		"Ca",
		"Fe",
		"As",
		"Al",
		"I",
		"B",
		"V",
		"K",
		"Tl",
		"Yb",
		"Sb",
		"Sn",
		"Ag",
		"Pd",
		"Co",
		"Se",
		"Ti",
		"Zn",
		"H",
		"Li",
		"Ge",
		"Cu",
		"Au",
		"Ni",
		"Cd",
		"In",
		"Mn",
		"Zr",
		"Cr",
		"Pt",
		"Hg",
		"Pb",
	]
	feat_size = len(allowable_set) + 1

	@classmethod
	def featurize(cls, x) -> int:
		return one_hot_encoding(x.GetSymbol(), cls.allowable_set, True)


@register_atom_feat
class AtomDegree(FeatBase):
	name = "a_degree"
	allowable_set = list(range(11))
	feat_size = len(allowable_set) + 1 + 2

	@classmethod
	def featurize(cls, x) -> int:
		deg = [x.GetDegree(), x.GetTotalDegree()]
		onehot = one_hot_encoding(deg, cls.allowable_set, True)
		return deg + onehot


@register_atom_feat
class AtomHybrid(FeatBase):

	name = "a_hybrid"
	allowable_set = [
		Chem.rdchem.HybridizationType.SP,
		Chem.rdchem.HybridizationType.SP2,
		Chem.rdchem.HybridizationType.SP3,
		Chem.rdchem.HybridizationType.SP3D,
		Chem.rdchem.HybridizationType.SP3D2,
	]
	feat_size = len(allowable_set) + 1

	@classmethod
	def featurize(cls, x) -> int:
		onehot = one_hot_encoding(x.GetHybridization(), cls.allowable_set, True)
		return onehot


@register_atom_feat
class AtomFormal(FeatBase):

	name = "a_formal"
	allowable_set = list(range(-2, 3))
	feat_size = len(allowable_set) + 1

	@classmethod
	def featurize(cls, x) -> int:
		onehot = one_hot_encoding(x.GetFormalCharge(), cls.allowable_set, True)
		return onehot


@register_atom_feat
class AtomRadical(FeatBase):

	name = "a_radical"
	allowable_set = list(range(5))
	feat_size = len(allowable_set) + 1

	@classmethod
	def featurize(cls, x) -> int:
		onehot = one_hot_encoding(x.GetNumRadicalElectrons(), cls.allowable_set, True)
		return onehot


@register_atom_feat
class AtomRing(FeatBase):

	name = "a_ring"
	allowable_set = [True, False]
	feat_size = len(allowable_set) * 2

	@classmethod
	def featurize(cls, x) -> int:
		onehot_ring = one_hot_encoding(x.IsInRing(), cls.allowable_set, False)
		onehot_aromatic = one_hot_encoding(x.GetIsAromatic(), cls.allowable_set, False)
		return onehot_ring + onehot_aromatic


@register_atom_feat
class AtomChiral(FeatBase):

	name = "a_chiral"
	allowable_set = [
		Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
		Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
		Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
		Chem.rdchem.ChiralType.CHI_OTHER,
	]
	feat_size = len(allowable_set) + 1

	@classmethod
	def featurize(cls, x) -> int:
		chiral_onehot = one_hot_encoding(x.GetChiralTag(), cls.allowable_set, True)
		return chiral_onehot


@register_atom_feat
class AtomMass(FeatBase):

	name = "a_mass"
	coef = 0.01
	feat_size = 1

	@classmethod
	def featurize(cls, x) -> int:
		return [x.GetMass() * cls.coef]


@register_bond_feat
class BondDegree(FeatBase):

	name = "b_degree"
	allowable_set = [
		Chem.rdchem.BondType.SINGLE,
		Chem.rdchem.BondType.DOUBLE,
		Chem.rdchem.BondType.TRIPLE,
		Chem.rdchem.BondType.AROMATIC,
	]
	feat_size = len(allowable_set) + 1

	@classmethod
	def featurize(cls, x) -> int:
		return one_hot_encoding(x.GetBondType(), cls.allowable_set, True)


@register_bond_feat
class BondStereo(FeatBase):

	name = "b_stereo"
	allowable_set = [
		Chem.rdchem.BondStereo.STEREONONE,
		Chem.rdchem.BondStereo.STEREOANY,
		Chem.rdchem.BondStereo.STEREOZ,
		Chem.rdchem.BondStereo.STEREOE,
		Chem.rdchem.BondStereo.STEREOCIS,
		Chem.rdchem.BondStereo.STEREOTRANS,
	]
	feat_size = len(allowable_set) + 1

	@classmethod
	def featurize(cls, x) -> int:
		return one_hot_encoding(x.GetStereo(), cls.allowable_set, True)


@register_bond_feat
class BondConj(FeatBase):

	name = "b_ring"
	feat_size = 2

	@classmethod
	def featurize(cls, x) -> int:
		return one_hot_encoding(x.IsInRing(), [False, True], False)


@register_bond_feat
class BondConj(FeatBase):

	name = "b_conj"
	feat_size = 2

	@classmethod
	def featurize(cls, x) -> int:
		return one_hot_encoding(x.GetIsConjugated(), [False, True], False)


def one_hot_encoding(x, allowable_set, encode_unknown=False) -> list:
	"""One_hot encoding.

	Code taken from dgllife library
	https://lifesci.dgl.ai/_modules/dgllife/utils/featurizers.html

	Args:
		x: Val to encode
		allowable_set: Options
		encode_unknown: If true, encode unk

	Return:
		list of bools
	"""

	if encode_unknown and (allowable_set[-1] is not None):
		allowable_set.append(None)

	if encode_unknown and (x not in allowable_set):
		x = None

	return list(map(lambda s: int(x == s), allowable_set))


class BatchedMolDAG:

	def __init__(
		self,
		mol_pyg_list,
		dag_pyg_list,
		formula_peak_mzs,
		formula_peak_probs,
		device=None
	):

		assert len(mol_pyg_list) == len(dag_pyg_list), (len(mol_pyg_list),len(dag_pyg_list))
		self.batch_size = len(mol_pyg_list)
		if device is not None:
			self._set_device(device)
		else:
			self._set_device("cpu")
		self.batch_size = th.tensor(self.batch_size,device=self.device)
		self.mol_pyg = pyg.data.Batch.from_data_list(mol_pyg_list).to(self.device)
		self.dag_pyg = pyg.data.Batch.from_data_list(dag_pyg_list).to(self.device)
		assert self.mol_pyg.num_graphs == self.dag_pyg.num_graphs == self.batch_size
		assert all(th.all(self.dag_pyg.node_feat_idxs[0] == self.dag_pyg.node_feat_idxs[i]) for i in range(self.batch_size))
		assert all(th.all(self.dag_pyg.edge_feat_idxs[0] == self.dag_pyg.edge_feat_idxs[i]) for i in range(self.batch_size))
		self._compute_batched_node_offsets(mol_pyg_list,dag_pyg_list)
		dag_formula_peak_idxs = []
		dag_formula_peak_mzs = []
		dag_formula_peak_probs = []
		dag_formula_sizes = []
		dag_formula_peak_sizes = []
		for mzs, probs in zip(formula_peak_mzs,formula_peak_probs):
			# sparsify
			idx = th.nonzero(probs)
			mzs = mzs[idx[:,0],idx[:,1]]
			probs = probs[idx[:,0],idx[:,1]]
			dag_formula_peak_idxs.append(idx[:,0])
			dag_formula_peak_mzs.append(mzs)
			dag_formula_peak_probs.append(probs)
			dag_formula_sizes.append(th.unique(idx,sorted=True).shape[0])
			dag_formula_peak_sizes.append(idx.shape[0])
		self.dag_formula_peak_idxs = th.cat(dag_formula_peak_idxs,dim=0).to(self.device)
		self.dag_formula_peak_mzs = th.cat(dag_formula_peak_mzs,dim=0).to(self.device)
		self.dag_formula_peak_probs = th.cat(dag_formula_peak_probs,dim=0).to(self.device)
		self.dag_formula_sizes = th.tensor(dag_formula_sizes,device=self.device)
		self.dag_formula_cumsizes = th.cumsum(th.tensor([0]+dag_formula_sizes,device=self.device),dim=0)
		self.dag_formula_peak_sizes = th.tensor(dag_formula_peak_sizes,device=self.device)

	def _compute_batched_node_offsets(self,mol_pyg_list,dag_pyg_list):

		# import pdb; pdb.set_trace()
		mol_num_nodes = [g.num_nodes for g in mol_pyg_list]
		dag_num_nodes = [g.num_nodes for g in dag_pyg_list]
		self.mol_num_nodes = th.cumsum(th.tensor([0]+mol_num_nodes,device=self.device),dim=0)
		self.dag_num_nodes = th.cumsum(th.tensor([0]+dag_num_nodes,device=self.device),dim=0)

	def to(self,device):

		self._set_device(device)
		self.batch_size = self.batch_size.to(self.device)
		self.mol_pyg = self.mol_pyg.to(self.device)
		self.dag_pyg = self.dag_pyg.to(self.device)
		self.mol_num_nodes = self.mol_num_nodes.to(self.device)
		self.dag_num_nodes = self.dag_num_nodes.to(self.device)
		self.dag_formula_peak_idxs = self.dag_formula_peak_idxs.to(self.device)
		self.dag_formula_peak_mzs = self.dag_formula_peak_mzs.to(self.device)
		self.dag_formula_peak_probs = self.dag_formula_peak_probs.to(self.device)
		self.dag_formula_sizes = self.dag_formula_sizes.to(self.device)
		self.dag_formula_cumsizes = self.dag_formula_cumsizes.to(self.device)
		self.dag_formula_peak_sizes = self.dag_formula_peak_sizes.to(self.device)
		return self

	def _set_device(self,device):

		if isinstance(device,str):
			device = th.device(device)
		self.device = device

	def __str__(self):

		str_format = f"BatchedMolDAG( batch_size={self.batch_size}, device={self.device} )"
		return str_format

	def __dict__(self):

		# break down pyg graphs into dict
		return {
			"batch_size": self.batch_size,
			# "device": self.device,
			"mol_x": self.mol_pyg.x,
			"mol_edge_index": self.mol_pyg.edge_index,
			"mol_edge_attr": self.mol_pyg.edge_attr,
			"dag_x": self.dag_pyg.x,
			"dag_edge_index": self.dag_pyg.edge_index,
			"dag_edge_attr": self.dag_pyg.edge_attr,
			"dag_node_feat_idxs": self.dag_pyg.node_feat_idxs[0],
			"dag_edge_feat_idxs": self.dag_pyg.edge_feat_idxs[0],
			"mol_num_nodes": self.mol_num_nodes,
			"dag_num_nodes": self.dag_num_nodes,
			"dag_formula_peak_idxs": self.dag_formula_peak_idxs,
			"dag_formula_peak_mzs": self.dag_formula_peak_mzs,
			"dag_formula_peak_probs": self.dag_formula_peak_probs,
			"dag_formula_sizes": self.dag_formula_sizes,
			"dag_formula_cumsizes": self.dag_formula_cumsizes,
			"dag_formula_peak_sizes": self.dag_formula_peak_sizes
		}

if __name__ == "__main__":

	import rdkit
	import rdkit.Chem as Chem
	from pprint import pprint

	smiles = "CC(=O)OC1=CC=CC=C1C(=O)O" # aspirin
	mol = Chem.MolFromSmiles(smiles)
	mg = MolGraph()
	g = mg.get_networkx_graph(mol)
	pprint(dict(g.nodes(data=True)))
	pprint(list(g.edges(data=True)))