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
import torch as th
import dgl
from torch.utils.data.dataset import Dataset

import frag_gnn.iceberg.common as common
import frag_gnn.iceberg.nn_utils as nn_utils
import frag_gnn.iceberg.fragmentation as fragmentation
from frag_gnn.dataset import BaseDataset, SpecMolDataset
from frag_gnn.utils.misc_utils import flatten_lol, print_shapes
from frag_gnn.utils.spec_utils import batch_func


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
		mz_max: float = 1500.0,
		mz_bin_res: float = 0.1,
		sum_ints: bool = True,
	):
		""" """
		self.pe_embed_k = pe_embed_k
		self.root_encode = root_encode
		self.binned_targs = binned_targs
		self.add_hs = add_hs

		# Hard coded bins (for now)
		self.bins = np.arange(mz_bin_res, mz_max + mz_bin_res, mz_bin_res)
		self.sum_ints = sum_ints

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
			frag_targets.append(th.from_numpy(targ_vec))

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
			# print(np.sort(intens[:,0]))
			bin_posts = np.digitize(intens[:, 0], self.bins, right=True)
			new_out = np.zeros_like(self.bins)
			for bin_post, inten in zip(bin_posts, intens[:, 1]):
				if self.sum_ints:
					new_out[bin_post] += inten
				else:
					new_out[bin_post] = max(new_out[bin_post], inten)
			# print(new_out.nonzero()[0])
			out_dict["inten_targs"] = new_out
			# ###
			# from frag_gnn.utils.spec_utils import bin_func
			# bin_spec = bin_func(
			# 	th.as_tensor(intens[:,0]).reshape(-1), 
			# 	th.as_tensor(intens[:,1]).reshape(-1), 
			# 	1500, 
			# 	0.1, 
			# 	False, 
			# 	False
			# )
			# print(bin_spec.numpy().nonzero()[0])
			# import pdb; pdb.set_trace()
			# ###
		return out_dict

	def add_pe_embed(self, graph):
		pe_embeds = nn_utils.random_walk_pe(
			graph, k=self.pe_embed_k, eweight_name="e_ind"
		)
		graph.ndata["h"] = th.cat((graph.ndata["h"], pe_embeds), -1).float()
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
		src_tens_, dest_tens_ = th.from_numpy(src), th.from_numpy(dest)
		bond_types = th.from_numpy(bond_types)
		src_tens = th.cat([src_tens_, dest_tens_])
		dest_tens = th.cat([dest_tens_, src_tens_])
		bond_types = th.cat([bond_types, bond_types])
		bond_featurizer = th.eye(fragmentation.MAX_BONDS)

		bond_types_onehot = bond_featurizer[bond_types.long()]
		node_data = th.from_numpy(node_types)

		# H data is defined, add that
		if h_adds is None:
			zero_vec = th.zeros((node_data.shape[0], common.MAX_H))
			node_data = th.hstack([node_data, zero_vec])
		else:
			h_featurizer = th.eye(common.MAX_H)
			h_adds_vec = th.from_numpy(h_adds)
			node_data = th.hstack([node_data, h_featurizer[h_adds_vec]])

		g = dgl.graph(data=(src_tens, dest_tens), num_nodes=num_nodes)
		g.ndata["h"] = node_data.float()
		g.edata["e"] = bond_types_onehot.float()
		g.edata["e_ind"] = bond_types.long()
		return g

	def get_node_feats(self):
		return self.pe_embed_k + common.ELEMENT_DIM + common.MAX_H


class SpecMolMagmaDataset(SpecMolDataset):
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

		BaseDataset.__init__(self)
		self._base_init(
			spec_fp=spec_fp,
			mol_fp=mol_fp,
			split_dp=split_dp,
			split=split,
			**kwargs
		)
		self.magma_dp = magma_dp
		self.mol_params = kwargs["mol_params"]
		self.magma_params = kwargs["magma_params"]
		self._init_magma()
		tree_group_ids = [int(name) for name in self.name_to_tree_file.keys()]
		assert self.spec_df["group_id"].isin(tree_group_ids).all()
		self._preprocess_spec()
		self._preprocess_mol()
		# self._preprocess_magma()

	def _init_magma(self):

		magma_dir = Path(self.magma_dp)
		magma_tree_dir = magma_dir / "magma_tree"
		assert magma_tree_dir.exists(), magma_tree_dir
		tree_names, tree_files = [], []
		for fp in magma_tree_dir.glob("*.json"):
			tree_names.append(fp.stem)
			tree_files.append(Path(fp))
		self.name_to_tree_file = {tree_name: tree_file for tree_name, tree_file in zip(tree_names, tree_files)}
		self.tree_fn = lambda x: self._read_tree(self._load_tree(x))
		self.tree_processor = self.__class__.init_tree_processor(**self.magma_params)

	def _load_tree(self, name):
		
		json_file = self.name_to_tree_file[name]
		with open(str(json_file), "r") as file:
			tree_json = json.load(file)
		return tree_json

	def _read_tree(self, tree):

		raise NotImplementedError()

	def __getitem__(self, idx):
		"""__getitem__."""

		spec_entry = self.spec_df.iloc[idx]
		spec_data = self._process_spec(spec_entry)
		mol_entry = self.mol_df.iloc[idx]
		mol_data = self._process_mol(mol_entry)
		name = str(spec_entry["group_id"])
		adduct = common.ion2onehot_pos[spec_entry["prec_type"]]
		magma_entry = self.tree_fn(name)["dgl_tree"]
		magma_entry = {
			"name": name, 
			"adduct": adduct,
			**magma_entry
		}
		magma_data = self._process_magma(magma_entry)
		data = {**spec_data, **mol_data, **magma_data}
		return data


class SpecMolMagmaGenDataset(SpecMolMagmaDataset):
	"""TODO: fold this into the parent class"""

	@staticmethod
	def init_tree_processor(pe_embed_k,root_encode,add_hs,**magma_params):

		tree_processor = TreeProcessor(
			pe_embed_k=pe_embed_k,
			root_encode=root_encode,
			add_hs=add_hs,
			binned_targs=False
		)
		return tree_processor

	def _read_tree(self, tree_json):
		
		tree = self.tree_processor.process_tree_gen(tree_json)
		return tree

	def _process_magma(self, magma_entry):

		magma_data = {}
		magma_data["magma_names"] = [magma_entry["name"]]
		if isinstance(magma_entry["root_repr"], dgl.DGLGraph):
			magma_data["magma_root_reprs"] = magma_entry["root_repr"]
		else:
			assert isinstance(magma_entry["root_repr"], np.ndarray)
			magma_data["magma_root_reprs"] = th.tensor(magma_entry["root_repr"], dtype=th.float).unsqueeze(0)
		magma_data["magma_frag_graphs"] = magma_entry["dgl_frags"]
		magma_data["magma_targ_atoms"] = th.nn.utils.rnn.pad_sequence([i for i in magma_entry["targs"]], batch_first=True) ##
		magma_data["magma_frag_atoms"] = th.tensor([i.num_nodes() for i in magma_entry["dgl_frags"]], dtype=th.long) ##
		magma_data["magma_inds"] = th.zeros([len(magma_entry["dgl_frags"])], dtype=th.long) ##
		magma_data["magma_broken_bonds"] = th.tensor(magma_entry["max_broken"], dtype=th.long)
		magma_data["magma_adducts"] = th.tensor([magma_entry["adduct"]], dtype=th.float)
		magma_data["magma_frag_form_vecs"] = th.tensor(np.array([i for i in magma_entry["form_vecs"]]), dtype=th.long) ##
		magma_data["magma_root_form_vecs"] = th.tensor(np.array([magma_entry["root_form_vec"]]), dtype=th.long)
		return magma_data

	@classmethod
	def get_collate_fn(cls):
		
		return SpecMolMagmaGenDataset.collate_fn

	@staticmethod
	def collate_fn(data_list):

		batch_size, keys, collate_data = SpecMolMagmaGenDataset._setup_collate(data_list)
		###
		frag_graphs = collate_data["magma_frag_graphs"]
		num_frags = th.tensor([len(i) for i in collate_data["magma_inds"]],dtype=th.long)
		targ_atoms = []
		for i in collate_data["magma_targ_atoms"]:
			targ_atoms.extend([j.squeeze(0) for j in th.split(i,1,dim=0)])
		###
		collate_data["magma_names"] = flatten_lol(collate_data["magma_names"])
		keys.remove("magma_names")
		if isinstance(collate_data["magma_root_reprs"][0], dgl.DGLGraph):
			collate_data["magma_root_reprs"] = dgl.batch(collate_data["magma_root_reprs"])
		else:
			assert isinstance(collate_data["magma_root_reprs"][0], th.Tensor)
			collate_data["magma_root_reprs"] = th.cat(collate_data["magma_root_reprs"],dim=0)
		keys.remove("magma_root_reprs")
		collate_data["magma_frag_graphs"] = dgl.batch(flatten_lol(frag_graphs))
		keys.remove("magma_frag_graphs")
		collate_data["magma_targ_atoms"] = th.nn.utils.rnn.pad_sequence(targ_atoms, batch_first=True)
		keys.remove("magma_targ_atoms")
		collate_data["magma_frag_atoms"] = th.cat(collate_data["magma_frag_atoms"], dim=0)
		keys.remove("magma_frag_atoms")
		collate_data["magma_inds"] = th.arange(len(frag_graphs),dtype=th.long).repeat_interleave(num_frags)
		keys.remove("magma_inds")
		collate_data["magma_broken_bonds"] = th.cat(collate_data["magma_broken_bonds"], dim=0)
		keys.remove("magma_broken_bonds")
		collate_data["magma_adducts"] = th.cat(collate_data["magma_adducts"], dim=0)
		keys.remove("magma_adducts")
		collate_data["magma_frag_form_vecs"] = th.cat(collate_data["magma_frag_form_vecs"], dim=0)
		keys.remove("magma_frag_form_vecs")
		collate_data["magma_root_form_vecs"] = th.cat(collate_data["magma_root_form_vecs"], dim=0)
		keys.remove("magma_root_form_vecs")
		###
		SpecMolMagmaGenDataset._standard_collate(batch_size, keys, collate_data)
		return collate_data


class SpecMolMagmaIntenDataset(SpecMolMagmaDataset):

	def __init__(
		self,
		spec_fp: str,
		mol_fp: str,
		split_dp: str,
		split: str,
		magma_dp: str,
		**kwargs,
	):

		BaseDataset.__init__(self)
		self._base_init(
			spec_fp=spec_fp,
			mol_fp=mol_fp,
			split_dp=split_dp,
			split=split,
			**kwargs
		)
		self.magma_dp = magma_dp
		self.mol_params = kwargs["mol_params"]
		self.magma_params = kwargs["magma_params"]
		self._init_magma()
		tree_group_ids = [int(name) for name in self.name_to_tree_file.keys()]
		assert self.spec_df["group_id"].isin(tree_group_ids).all()
		formula_group_ids = [int(name) for name in self.name_to_formula_file.keys()]
		assert self.spec_df["group_id"].isin(formula_group_ids).all()
		self._preprocess_spec()
		self._preprocess_mol()
		# self._preprocess_magma()

	def _init_magma(self):

		super()._init_magma()
		magma_dir = Path(self.magma_dp)
		formula_dir = magma_dir / "magma_formula"
		assert formula_dir.exists(), formula_dir
		formula_names, formula_files = [], []
		for fp in formula_dir.glob("*.json"):
			formula_names.append(fp.stem)
			formula_files.append(Path(fp))
		self.name_to_formula_file = {formula_name: formula_file for formula_name, formula_file in zip(formula_names, formula_files)}

	@staticmethod
	def init_tree_processor(
		pe_embed_k,
		root_encode,
		add_hs,
		binned_targs,
		mz_bin_res,
		mz_max,
		sum_ints,
		**magma_params
	):

		tree_processor = TreeProcessor(
			pe_embed_k=pe_embed_k,
			root_encode=root_encode,
			add_hs=add_hs,
			binned_targs=binned_targs,
			mz_bin_res=mz_bin_res,
			mz_max=mz_max,
			sum_ints=sum_ints
		)
		return tree_processor

	def _load_tree(self, name):
		
		tree_json = super()._load_tree(name)
		formula_json = self._load_formula(name)
		raw_spec, adduct = self._read_formula(formula_json)
		tree_json["raw_spec"] = raw_spec
		tree_json["adduct"] = adduct
		return tree_json

	def _read_tree(self, tree_json):
		
		adduct = tree_json.pop("adduct")
		tree = self.tree_processor.process_tree_inten(tree_json)
		# adduct correction on masses
		# print("before masses")
		# print(tree["dgl_tree"]["masses"])
		tree["dgl_tree"]["masses"] = tree["dgl_tree"]["masses"] + common.ion2mass[adduct]
		# print("after masses")
		# print(tree["dgl_tree"]["masses"])
		return tree

	def _load_formula(self, name):

		json_file = self.name_to_formula_file[name]
		with open(str(json_file), "r") as file:
			formula_json = json.load(file)
		return formula_json

	def _read_formula(self, formula_json):

		true_tbl = formula_json["output_tbl"]
		true_adduct = formula_json["cand_ion"]
		# raw_spec = list(zip(true_tbl["formula_mass_no_adduct"], true_tbl["rel_inten"]))
		# print("before raw_spec")
		# print(raw_spec)
		mz_diff = common.ion2mass[true_adduct]
		raw_spec = []
		for mz, inten in zip(true_tbl["formula_mass_no_adduct"], true_tbl["rel_inten"]):
			# adduct correction on raw_spec
			raw_spec.append((mz + mz_diff, inten))
		# print("after raw_spec")
		# print(raw_spec)
		return raw_spec, true_adduct

	@classmethod
	def get_collate_fn(cls):
		
		return SpecMolMagmaIntenDataset.collate_fn

	def _process_magma(self, magma_entry):

		magma_data = {}
		magma_data["magma_names"] = [magma_entry["name"]]
		if isinstance(magma_entry["root_repr"], dgl.DGLGraph):
			magma_data["magma_root_reprs"] = magma_entry["root_repr"]
		else:
			assert isinstance(magma_entry["root_repr"], np.ndarray)
			magma_data["magma_root_reprs"] = th.tensor(magma_entry["root_repr"], dtype=th.float).unsqueeze(0)
		magma_data["magma_frag_graphs"] = magma_entry["dgl_frags"]
		magma_data["magma_frag_atoms"] = th.tensor([i.num_nodes() for i in magma_entry["dgl_frags"]], dtype=th.long) ##
		magma_data["magma_inds"] = th.zeros([len(magma_entry["dgl_frags"])], dtype=th.long)
		magma_data["magma_broken_bonds"] = th.tensor(magma_entry["max_broken"], dtype=th.long).unsqueeze(0)
		magma_data["magma_adducts"] = th.tensor([magma_entry["adduct"]], dtype=th.float)
		magma_data["magma_frag_form_vecs"] = th.tensor(np.array([i for i in magma_entry["form_vecs"]]), dtype=th.long).unsqueeze(0)
		magma_data["magma_root_form_vecs"] = th.tensor(np.array([magma_entry["root_form_vec"]]), dtype=th.long)
		###
		magma_data["magma_num_frags"] = th.tensor([len(magma_entry["dgl_frags"])], dtype=th.long)
		if self.magma_params["binned_targs"]:
			magma_data["magma_inten_targs"] = th.tensor(magma_entry["inten_targs"], dtype=th.float).unsqueeze(0)
		magma_data["magma_masses"] = th.tensor(magma_entry["masses"], dtype=th.float).unsqueeze(0)
		magma_data["magma_max_add_hs"] = th.tensor(magma_entry["max_add_hs"], dtype=th.float).unsqueeze(0)
		magma_data["magma_max_remove_hs"] = th.tensor(magma_entry["max_remove_hs"], dtype=th.float).unsqueeze(0)
		magma_data["magma_inten_frag_ids"] = [magma_entry["inten_frag_ids"]]
		return magma_data

	@staticmethod
	def collate_fn(data_list):

		batch_size, keys, collate_data = SpecMolMagmaIntenDataset._setup_collate(data_list)
		###
		frag_graphs = collate_data["magma_frag_graphs"]
		num_frags = th.tensor([len(i) for i in collate_data["magma_inds"]],dtype=th.long)
		assert th.all(num_frags == th.cat(collate_data["magma_num_frags"],dim=0))
		###
		collate_data["magma_names"] = flatten_lol(collate_data["magma_names"])
		keys.remove("magma_names")
		if isinstance(collate_data["magma_root_reprs"][0], dgl.DGLGraph):
			collate_data["magma_root_reprs"] = dgl.batch(collate_data["magma_root_reprs"])
		else:
			assert isinstance(collate_data["magma_root_reprs"][0], th.Tensor)
			collate_data["magma_root_reprs"] = th.cat(collate_data["magma_root_reprs"],dim=0)
		keys.remove("magma_root_reprs")
		collate_data["magma_frag_graphs"] = dgl.batch(flatten_lol(frag_graphs))
		keys.remove("magma_frag_graphs")
		collate_data["magma_frag_atoms"] = th.cat(collate_data["magma_frag_atoms"], dim=0)
		keys.remove("magma_frag_atoms")
		collate_data["magma_inds"] = th.arange(len(frag_graphs),dtype=th.long).repeat_interleave(num_frags)
		keys.remove("magma_inds")
		collate_data["magma_broken_bonds"] = th.nn.utils.rnn.pad_sequence(
			[i.squeeze(0) for i in collate_data["magma_broken_bonds"]], 
			batch_first=True
		)
		keys.remove("magma_broken_bonds")
		collate_data["magma_adducts"] = th.cat(collate_data["magma_adducts"], dim=0)
		keys.remove("magma_adducts")
		collate_data["magma_frag_form_vecs"] = th.nn.utils.rnn.pad_sequence(
			[i.squeeze(0) for i in collate_data["magma_frag_form_vecs"]], 
			batch_first=True
		)
		keys.remove("magma_frag_form_vecs")
		collate_data["magma_root_form_vecs"] = th.cat(collate_data["magma_root_form_vecs"], dim=0)
		keys.remove("magma_root_form_vecs")
		###
		collate_data["magma_num_frags"] = th.cat(collate_data["magma_num_frags"], dim=0)
		keys.remove("magma_num_frags")
		if "magma_inten_targs" in keys:
			collate_data["magma_inten_targs"] = th.cat(collate_data["magma_inten_targs"], dim=0) ### not sure
			keys.remove("magma_inten_targs")
		collate_data["magma_masses"] = th.nn.utils.rnn.pad_sequence(
			[i.squeeze(0) for i in collate_data["magma_masses"]], 
			batch_first=True
		)
		keys.remove("magma_masses")
		collate_data["magma_max_add_hs"] = th.nn.utils.rnn.pad_sequence(
			[i.squeeze(0) for i in collate_data["magma_max_add_hs"]], 
			batch_first=True
		)
		keys.remove("magma_max_add_hs")
		collate_data["magma_max_remove_hs"] = th.nn.utils.rnn.pad_sequence(
			[i.squeeze(0) for i in collate_data["magma_max_remove_hs"]], 
			batch_first=True
		)
		keys.remove("magma_max_remove_hs")
		collate_data["magma_inten_frag_ids"] = flatten_lol(collate_data["magma_inten_frag_ids"])
		keys.remove("magma_inten_frag_ids")
		###
		SpecMolMagmaIntenDataset._standard_collate(batch_size, keys, collate_data)
		return collate_data

if __name__ == "__main__":

	from frag_gnn.utils.misc_utils import to_gpu

	config_d = dict(
		spec_fp="data/proc/spec_df.pkl",
		mol_fp="data/proc/mol_df.pkl",
		split_dp="data/split/d3_h4_isoTrue_v2",
		split="train",
		magma_dp="data/magma",
		spec_params=dict(
			merge=True,
			subsample_size=0,
			subsample_seed=420,
			sparse=True,
			binned=False,
			prec_type=True,
			prec_mz=True,
			prec_mass_diff=True,
			nce=False,
			ace=False,
			preprocess=False,
			unique_id=True
		),
		mol_params=dict(
			smiles=True,
			fingerprint=False,
			fingerprint_morgan=True,
			fingerprint_rdkit=True,
			fingerprint_maccs=True,
			pyg=False,
			pyg_node_feats=[
				"a_onehot",
				"a_degree",
				"a_hybrid",
				"a_formal",
				"a_radical",
				"a_ring",
				"a_mass",
				"a_chiral"
			],	
			pyg_edge_feats=[
				"b_degree"
			],
			pyg_pe_embed_k=0,
			pyg_bigraph=True,
			preprocess=False
		),
		magma_params=dict(
			pe_embed_k=0,
			root_encode="gnn",
			add_hs=False,
			binned_targs=True,
			mz_bin_res=0.1,
			mz_max=1500.0,
			sum_ints=True,
		),
		mz_max=1500.0,
		mz_bin_res=0.1,
		sum_ints=True,
	)

	# for ds_cls in [SpecMolMagmaGenDataset, SpecMolMagmaIntenDataset]:

	# 	ds = ds_cls(**config_d)
	# 	item = ds[0]
	# 	print(f">>> {ds_cls} item")
	# 	print_shapes(item)
	# 	item = to_gpu(item)

	# for ds_cls in [SpecMolMagmaGenDataset, SpecMolMagmaIntenDataset]:

	# 	ds = ds_cls(**config_d)
	# 	dl = th.utils.data.DataLoader(
	# 		ds,
	# 		batch_size=2,
	# 		shuffle=False,
	# 		num_workers=0,
	# 		collate_fn=ds.get_collate_fn(),
	# 	)
	# 	dl_iter = iter(dl)
	# 	batch = next(dl_iter)
	# 	print(">>> batch")
	# 	print_shapes(batch)
	# 	batch = to_gpu(batch)

	from frag_gnn.utils.spec_utils import bin_func, batched_bin_func, binned_cosine_similarity
	from frag_gnn.utils.formula_utils import PREC_TYPE_TO_MASS_DIFF
	import torch.nn.functional as F

	diff = PREC_TYPE_TO_MASS_DIFF["[M+H]+"]

	ds = SpecMolMagmaIntenDataset(**config_d)
	item = ds[0]
	bin_spec_1 = bin_func(
		item["spec_mzs"], 
		item["spec_ints"], 
		1500.0, 
		0.1,
		False,
		True
	)
	bin_spec_1 = th.sqrt(F.normalize(bin_spec_1,p=0,dim=0))
	bin_spec_2 = item["magma_inten_targs"].squeeze(0)
	print("> shapes")
	print(bin_spec_1.shape)
	print(bin_spec_2.shape)
	print("> mzs")
	print(item["spec_mzs"])
	print("> nonzero idxs")
	print(bin_spec_1.nonzero().squeeze(1))
	print(bin_spec_2.nonzero().squeeze(1))
	print(th.all(bin_spec_1.nonzero()==bin_spec_2.nonzero()))
	print("> cosine similarity")
	print(binned_cosine_similarity(bin_spec_1.unsqueeze(0),bin_spec_2.unsqueeze(0)))
	import pdb; pdb.set_trace()

	dl = th.utils.data.DataLoader(
		ds,
		batch_size=2,
		shuffle=False,
		num_workers=0,
		collate_fn=ds.get_collate_fn(),
	)
	dl_iter = iter(dl)
	batch = next(dl_iter)
	print(">>> batch")
	print_shapes(batch)
	bin_spec_1 = batched_bin_func(
		batch["spec_mzs"], 
		batch["spec_ints"], 
		batch["spec_batch_idxs"],
		1500.0,
		0.1,
		True
	)
	bin_spec_2 = batch["magma_inten_targs"]
	print(bin_spec_1.shape)
	print(bin_spec_2.shape)
	print(binned_cosine_similarity(bin_spec_1,bin_spec_2))
	import pdb; pdb.set_trace()