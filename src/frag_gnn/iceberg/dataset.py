""" dag_data.py

Fragment dataset to build out model class

"""
import logging
from pathlib import Path
from typing import List
import json
import copy

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import dgl
from torch.utils.data.dataset import Dataset

import frag_gnn.iceberg.common as common
import frag_gnn.iceberg.nn_utils as nn_utils
import frag_gnn.iceberg.fragmentation as fragmentation
from frag_gnn.dataset import BaseDataset


class TreeProcessor:
	"""TreeProcessor.

	Hold key functionalities to read in a magma dag and proces it.

	"""

	def __init__(
		self,
		pe_embed_k: int = 10,
		root_encode: str = "gnn",
		binned_targs: bool = False,
		add_hs: bool = False,
	):
		""" """
		self.pe_embed_k = pe_embed_k
		self.root_encode = root_encode
		self.binned_targs = binned_targs
		self.add_hs = add_hs

		# Hard coded bins (for now)
		self.bins = np.linspace(0, 1500, 15000)

	def featurize_frag(
		self,
		frag: int,
		engine: fragmentation.FragmentEngine,
		add_random_walk: bool = False,
	) -> False:
		"""featurize_frag.

		Prev.  dgl_from_frag

		"""

		num_atoms = engine.natoms
		atom_symbols = engine.atom_symbols
		# Need to find all kept atoms and all kept bonds between
		kept_atom_inds, kept_atom_symbols = engine.get_present_atoms(frag)
		kept_bond_orders, kept_bonds = engine.get_present_edges(frag)

		# H count
		form = engine.formula_from_kept_inds(kept_atom_inds)

		# Need to re index the targets to match the new graph size
		num_kept = len(kept_atom_inds)
		new_inds = np.arange(num_kept)
		old_inds = kept_atom_inds

		old_to_new = np.zeros(num_atoms, dtype=int)
		old_to_new[old_inds] = new_inds

		# Keep new_to_old for autoregressive predictions
		new_to_old = np.zeros(num_kept, dtype=int)
		new_to_old[new_inds] = old_inds

		# Remap new bond inds
		new_bond_inds = np.empty((0, 2), dtype=int)
		if len(kept_bonds) > 0:
			new_bond_inds = old_to_new[np.vstack(kept_bonds)]

		if self.add_hs:
			h_adds = np.array(engine.atom_hs)[kept_atom_inds]
		else:
			h_adds = None

		# Make dgl graphs for new targets
		graph = self.dgl_featurize(
			np.array(atom_symbols)[kept_atom_inds],
			h_adds=h_adds,
			bond_inds=new_bond_inds,
			bond_types=np.array(kept_bond_orders),
		)

		if add_random_walk:
			self.add_pe_embed(graph)
		return {
			"graph": graph,
			"new_to_old": new_to_old,
			"old_to_new": old_to_new,
			"form": form,
		}

	def _convert_to_dgl(
		self,
		tree: dict,
		include_targets: bool = True,
		last_row=False,
	):
		"""_convert_to_dgl.

		Args:
			tree (dict): tree dictionary
			include_targets (bool): Try to add inten targets for supervising
				the inten model
			last_row:
		"""
		root_inchi = tree["root_inchi"]
		engine = fragmentation.FragmentEngine(mol_str=root_inchi, mol_str_type="inchi")
		# bottom_depth = engine.max_broken_bonds
		bottom_depth = engine.max_tree_depth
		if self.root_encode == "gnn":
			root_frag = engine.get_root_frag()
			root_graph_dict = self.featurize_frag(
				frag=root_frag,
				engine=engine,
			)
			root_repr = root_graph_dict["graph"]
		elif self.root_encode == "fp":
			root_repr = common.get_morgan_fp_inchi(root_inchi)
		else:
			raise ValueError()

		root_form = common.form_from_inchi(root_inchi)

		# Need to include mass and inten targets here, maybe not necessary in
		# all cases?
		masses, inten_frag_ids, dgl_inputs, inten_targets, frag_targets, max_broken = (
			[],
			[],
			[],
			[],
			[],
			[],
		)
		forms = []
		max_remove_hs, max_add_hs = [], []
		for k, sub_frag in tree["frags"].items():
			max_broken_num = sub_frag["max_broken"]
			tree_depth = sub_frag["tree_depth"]

			# Skip because we never fragment last row
			if (not last_row) and (tree_depth == bottom_depth):
				continue

			binary_targs = sub_frag["atoms_pulled"]
			frag = sub_frag["frag"]

			# Get frag dict and target
			frag_dict = self.featurize_frag(
				frag,
				engine,
			)
			forms.append(frag_dict["form"])
			old_to_new = frag_dict["old_to_new"]
			graph = frag_dict["graph"]
			max_broken.append(max_broken_num)

			max_remove_hs.append(sub_frag["max_remove_hs"])
			max_add_hs.append(sub_frag["max_add_hs"])

			if include_targets and not self.binned_targs:
				inten_targs = sub_frag["intens"]
				inten_targets.append(inten_targs)

			inten_frag_ids.append(k)

			# For gen model only!!
			targ_vec = np.zeros(graph.num_nodes())
			for j in old_to_new[binary_targs]:
				targ_vec[j] = 1

			graph = frag_dict["graph"]

			# Define targ vec
			dgl_inputs.append(graph)
			masses.append(sub_frag["base_mass"])
			frag_targets.append(torch.from_numpy(targ_vec))

		if include_targets and self.binned_targs:
			inten_targets = np.array(tree["raw_spec"])

		masses = engine.shift_bucket_masses[None, :] + np.array(masses)[:, None]
		max_remove_hs = np.array(max_remove_hs)
		max_add_hs = np.array(max_add_hs)
		max_broken = np.array(max_broken)

		# Feat each form
		all_form_vecs = [common.formula_to_dense(i) for i in forms]
		all_form_vecs = np.array(all_form_vecs)
		root_form_vec = common.formula_to_dense(root_form)

		out_dict = {
			"root_repr": root_repr,
			"dgl_frags": dgl_inputs,
			"masses": masses,
			"inten_targs": np.array(inten_targets) if include_targets else None,
			"inten_frag_ids": inten_frag_ids,
			"max_remove_hs": max_remove_hs,
			"max_add_hs": max_add_hs,
			"max_broken": max_broken,
			"targs": frag_targets,
			"form_vecs": all_form_vecs,
			"root_form_vec": root_form_vec,
		}
		return out_dict

	def _process_tree(
		self,
		tree: dict,
		include_targets: bool = True,
		last_row=False,
		convert_to_dgl=True,
	):
		"""_process_tree.

		Args:
			tree (dict): tree dictionary
			include_targets (bool): Try to add inten targets for supervising
				the inten model
			last_row:
			pickle_input: If pickle_input, this
		"""
		if convert_to_dgl:
			out_dict = self._convert_to_dgl(tree, include_targets, last_row)
		else:
			out_dict = tree

		dgl_inputs = out_dict["dgl_frags"]
		root_repr = out_dict["root_repr"]

		if self.pe_embed_k > 0:
			for graph in dgl_inputs:
				self.add_pe_embed(graph)

			if isinstance(root_repr, dgl.DGLGraph):
				self.add_pe_embed(root_repr)

		if include_targets and self.binned_targs:
			intens = out_dict["inten_targs"]
			bin_posts = np.digitize(intens[:, 0], self.bins)
			new_out = np.zeros_like(self.bins)
			for bin_post, inten in zip(bin_posts, intens[:, 1]):
				new_out[bin_post] = max(new_out[bin_post], inten)
			inten_targets = new_out
			out_dict["inten_targs"] = inten_targets
		return out_dict

	def add_pe_embed(self, graph):
		pe_embeds = nn_utils.random_walk_pe(
			graph, k=self.pe_embed_k, eweight_name="e_ind"
		)
		graph.ndata["h"] = torch.cat((graph.ndata["h"], pe_embeds), -1).float()
		return graph

	def process_tree_gen(self, tree: dict, convert_to_dgl=True):
		proc_out = self._process_tree(
			tree, include_targets=False, last_row=False, convert_to_dgl=convert_to_dgl
		)
		keys = {
			"root_repr",
			"dgl_frags",
			"targs",
			"max_broken",
			"form_vecs",
			"root_form_vec",
		}
		dgl_tree = {i: proc_out[i] for i in keys}
		return {"dgl_tree": dgl_tree, "tree": tree}

	def process_tree_inten(self, tree, convert_to_dgl=True):
		proc_out = self._process_tree(
			tree, include_targets=True, last_row=True, convert_to_dgl=convert_to_dgl
		)
		keys = {
			"root_repr",
			"dgl_frags",
			"masses",
			"inten_targs",
			"inten_frag_ids",
			"max_remove_hs",
			"max_add_hs",
			"max_broken",
			"form_vecs",
			"root_form_vec",
		}
		dgl_tree = {i: proc_out[i] for i in keys}
		return {"dgl_tree": dgl_tree, "tree": tree}

	def process_tree_inten_pred(self, tree: dict, convert_to_dgl=True):
		proc_out = self._process_tree(
			tree, include_targets=False, last_row=True, convert_to_dgl=convert_to_dgl
		)
		keys = {
			"root_repr",
			"dgl_frags",
			"masses",
			"inten_targs",
			"inten_frag_ids",
			"max_remove_hs",
			"max_add_hs",
			"max_broken",
			"form_vecs",
			"root_form_vec",
		}
		dgl_tree = {i: proc_out[i] for i in keys}
		return {"dgl_tree": dgl_tree, "tree": tree}

	def dgl_featurize(
		self,
		atom_symbols: List[str],
		h_adds: np.ndarray,
		bond_inds: np.ndarray,
		bond_types: np.ndarray,
	):
		"""dgl_featurize.

		Args:
			atom_symbols (List[str]): node_types
			h_adds (np.ndarray): h_adds
			bond_inds (np.ndarray): bond_inds
			bond_types (np.ndarray)
		"""
		node_types = [common.element_to_position[el] for el in atom_symbols]
		node_types = np.vstack(node_types)
		num_nodes = node_types.shape[0]

		src, dest = bond_inds[:, 0], bond_inds[:, 1]
		src_tens_, dest_tens_ = torch.from_numpy(src), torch.from_numpy(dest)
		bond_types = torch.from_numpy(bond_types)
		src_tens = torch.cat([src_tens_, dest_tens_])
		dest_tens = torch.cat([dest_tens_, src_tens_])
		bond_types = torch.cat([bond_types, bond_types])
		bond_featurizer = torch.eye(fragmentation.MAX_BONDS)

		bond_types_onehot = bond_featurizer[bond_types.long()]
		node_data = torch.from_numpy(node_types)

		# H data is defined, add that
		if h_adds is None:
			zero_vec = torch.zeros((node_data.shape[0], common.MAX_H))
			node_data = torch.hstack([node_data, zero_vec])
		else:
			h_featurizer = torch.eye(common.MAX_H)
			h_adds_vec = torch.from_numpy(h_adds)
			node_data = torch.hstack([node_data, h_featurizer[h_adds_vec]])

		g = dgl.graph(data=(src_tens, dest_tens), num_nodes=num_nodes)
		g.ndata["h"] = node_data.float()
		g.edata["e"] = bond_types_onehot.float()
		g.edata["e_ind"] = bond_types.long()
		return g

	def get_node_feats(self):
		return self.pe_embed_k + common.ELEMENT_DIM + common.MAX_H


class SpecMagmaDataset(BaseDataset):
	""" """

	def __init__(
		self,
		spec_fp: str,
		mol_fp: str,
		split_dp: str,
		split: str,
		magma_dp: str,
		**kwargs,
	):
		"""__init__.
		"""

		BaseDataset.__init__()
		self._base_init(
			spec_fp=spec_fp,
			mol_fp=mol_fp,
			split_dp=split_dp,
			split=split,
			**kwargs
		)
		self.magma_dp = magma_dp
		self.magma_params = kwargs["magma_params"]
		magma_dir = Path(magma_dp)
		magma_tree_dir = magma_dir / "magma_tree"
		tree_group_ids, tree_files = [], []
		for fp in magma_tree_dir.glob("*.json"):
			tree_group_ids.append(int(fp.stem))
			tree_files.append(Path(fp))
		self.group_id_to_tree_file = {group_id: tree_file for group_id, tree_file in zip(tree_group_ids, tree_files)}
		self.read_fn = lambda x: self._read_tree(self._load_tree(x))

	def _load_tree(self, group_id):
		
		json_file = self.group_id_to_tree_file[group_id]
		with open(str(filename), "r") as fp:
			tree_json = json.load(fp)
		return tree_json

	def _read_tree(self, tree):

		raise NotImplementedError()

	def __getitem__(self, idx):
		"""__getitem__."""

		spec_entry = self.spec_df.iloc[idx]
		name = str(spec_entry["group_id"])
		adduct = common.ion2onehot_pos[spec_entry["prec_type"]]
		dgl_entry = self.read_fn(name)["dgl_tree"]
		dgl_entry = self._process_dgl_entry(dgl_entry)
		outdict = {"name": name, "adduct": adduct}
		# Convert this into a list of graphs with a list of targets
		outdict.update(dgl_entry)
		return outdict

	def get_node_feats(self) -> int:
		"""get_node_feats."""
		
		return self.tree_processor.get_node_feats()


class SpecMagmaGenDataset(SpecMagmaDataset):
	"""TODO: fold this into the parent class"""

	def __init__(
		self,
		spec_fp: str,
		mol_fp: str,
		split_dp: str,
		split: str,
		magma_dp: str,
		**kwargs,
	):
		"""__init__.

		Args:
		"""
		super().__init__(
			spec_fp=spec_fp,
			mol_fp=mol_fp,
			split_dp=split_dp,
			split=split,
			magma_dp=magma_dp,
			**kwargs
		)
		tree_group_ids = list(self.group_id_to_tree_file.keys())
		assert self.spec_df["group_id"].isin(tree_group_ids).all()
		self.tree_processor = TreeProcessor(
			pe_embed_k=kwargs["iceberg_pe_embed_k"],
			root_encode=kwargs["iceberg_root_encode"],
			add_hs=kwargs["iceberg_add_hs"],
			binned_targs=False
		)
		
	def _read_tree(self, tree_json):
		
		tree = self.tree_processor.process_tree_gen(tree_json)
		return tree

	def _process_dgl_entry(self, dgl_entry):

		new_dgl_entry = {}
		new_dgl_entry["names"] = [dgl_entry["name"]]
		if isinstance(dgl_entry["root_repr"], dgl.DGLGraph):
			new_dgl_entry["root_reprs"] = dgl_entry["root_repr"]
		else:
			assert isinstance(dgl_entry["root_repr"], np.ndarray)
			new_dgl_entry["root_reprs"] = torch.tensor(dgl_entry["root_repr"], dtype=th.float).unsqueeze(0)
		new_dgl_entry["frag_graphs"] = dgl_entry["dgl_frags"]
		new_dgl_entry["targ_atoms"] = torch.tensor([j for j in dgl_entry["targs"]], dtype=th.long).unsqueeze(0) ##
		new_dgl_entry["frag_atoms"] = torch.tensor([i.num_nodes() for i in dgl_entry["dgl_frags"]], dtype=th.long) ##
		new_dgl_entry["inds"] = torch.zeros([len(dgl_entry["dgl_frags"])], dtype=torch.long) ##
		new_dgl_entry["broken_bonds"] = torch.tensor([dgl_entry["max_broken"]], dtype=th.long)
		new_dgl_entry["adducts"] = torch.tensor([dgl_entry["adduct"]], dtype=th.float)
		new_dgl_entry["frag_form_vecs"] = torch.tensor([j for j in dgl_entry["form_vecs"]], dtype=th.long) ##
		new_dgl_entry["root_form_vecs"] = torch.tensor([dgl_entry["root_form_vec"]], dtype=th.long)
		return new_dgl_entry

	@classmethod
	def get_collate_fn(cls):
		return SpecMagmaGenDataset.collate_fn

	@staticmethod
	def collate_fn(data_list):
		"""collate_fn.

		Batch a list of graphs and convert.

		Should return dict containing:
			1. Vector of names.
			2. Batched root reprs
			3. Batched fragment graphs
			4. Batched target atoms
			6. Number of atoms per frag graph
			7. Indices mapping batched fragment graphs to roots
			8. Conversion mapping indices between root and children?

		"""

		batch_size, keys, collate_data = SpecMagmaGenDataset._setup_collate(data_list)
		###
		frag_graphs = collate_data["frag_graphs"]
		frag_graphs_e = [j for i in frag_graphs for j in i]
		num_frags = torch.LongTensor([len(i) for i in frag_graphs])
		###
		collate_data["names"] = flatten_lol(dgl_entry["names"])
		if isinstance(dgl_entry["root_repr"], dgl.DGLGraph):
			collate_data["root_reprs"] = dgl.batch(dgl_entry["root_reprs"])
		else:
			assert isinstance(dgl_entry["root_repr"], th.Tensor)
			collate_data["root_reprs"] = th.cat(dgl_entry["root_reprs"],dim=0)
		collate_data["frag_graphs"] = dgl.batch(frag_graphs_e)
		collate_data["targ_atoms"] = torch.nn.utils.rnn.pad_sequence(
			[targ_atoms.squeeze(0) for targ_atoms in collate_data["targ_atoms"]], 
			batch_first=True
		)
		collate_data["frag_atoms"] = th.cat(collate_data["frag_atoms"], dim=0)
		collate_data["inds"] = torch.arange(len(frag_graphs)).repeat_interleave(num_frags).long()
		collate_data["broken_bonds"] = th.cat(collate_data["broken_bonds"], dim=0)
		collate_data["adducts"] = th.cat(collate_data["adducts"], dim=0)
		collate_data["frag_form_vecs"] = th.cat(collate_data["frag_form_vecs"], dim=0)
		collate_data["root_form_vecs"] = th.cat(collate_data["root_form_vecs"], dim=0)
		###
		# add batch_size
		collate_data["batch_size"] = th.tensor(batch_size)
		return collate_data


if __name__ == "__main__":

	config_d = dict(
		spec_dp="data/proc/spec_df.pkl",
		mol_df="data/proc/mol_df.pkl",
		split_dp="data/split/d3_h4_isoTrue_v2",
		split="train",
		magma_dp="data/proc/magma",
	)

	dataset = SpecMagmaGenDataset(**config_d)

	item = dataset[0]

	import pdb; pdb.set_trace()

