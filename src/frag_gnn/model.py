import torch as th
import torch.nn as nn

from frag_gnn.utils.frag_utils import get_node_feats, get_edge_feats, th_long_to_mask
from frag_gnn.utils.misc_utils import scatter_logsumexp, scatter_logsoftmax, scatter_reduce, scatter_masked_softmax, scatter_masked_logsumexp
from frag_gnn.utils.feat_utils import get_mol_feats_sizes, get_mol_fp_size
from frag_gnn.utils.nn_utils import *
from frag_gnn.form_embedder import get_embedder
from frag_gnn.utils.misc_utils import safelog, check_pyg_compile, LOG_ZERO
from frag_gnn.utils.spec_utils import transform_nce, NCE_MAX


class FragGNNModel(nn.Module):

	def __init__(
		self,
		num_depth: int,
		num_hs: int,
		num_elements: int,
		int_embedder: str,
		int_embedder_tight: bool,
		mol_node_feats: list[str],
		mol_edge_feats: list[str],
		mol_pe_embed_k: int,
		mol_hidden_size: int,
		mol_num_layers: int,
		mol_gnn_type: str,
		mol_normalization: str,
		mol_dropout: float,
		mol_pool_type: str,
		frag_node_feats: list[str],
		frag_edge_feats: list[str],
		frag_hidden_size: int,
		frag_num_layers: int,
		frag_gnn_type: str,
		frag_normalization: str,
		frag_dropout: float,
		frag_pool_type: str,
		frag_remove_edges: bool,
		mlp_output_format: str,
		mlp_hidden_size: int,
		mlp_normalization: str,
		mlp_dropout: float,
		mlp_num_layers: int,
		mlp_use_residuals: bool,
		cc_interstage_type: str,
		skip_edge_loss: bool,
		mask_null_formula: bool,
		predict_oos: bool,
		ce_insert_mol: bool,
		ce_insert_frag: bool,
		ce_insert_mlp: bool,
		ce_insert_mlp_lin: bool,
		m_ce_insert_mlp_embed: bool,
		m_ce_insert_mlp_bin: bool,
	):

		super().__init__()
		self.num_depth = num_depth
		self.num_hs = num_hs
		self.num_elements = num_elements

		# check ce insertion type
		self.ce_insert_mol = ce_insert_mol
		self.ce_insert_frag = ce_insert_frag
		self.ce_insert_mlp = ce_insert_mlp
		self.ce_insert_mlp_lin = ce_insert_mlp_lin
		self.m_ce_insert_mlp_embed = m_ce_insert_mlp_embed
		self.m_ce_insert_mlp_bin = m_ce_insert_mlp_bin

		# calculate node/edge feats sizes
		self.mol_node_feats = mol_node_feats
		self.mol_edge_feats = mol_edge_feats
		self.mol_pe_embed_k = mol_pe_embed_k
		self._compute_mol_feats_sizes()

		# setup mol gnn
		mol_kwargs = {
			"node_feats_size": self.mol_node_feats_size,
			"edge_feats_size": self.mol_edge_feats_size,
			"hidden_size": mol_hidden_size,
			"num_layers": mol_num_layers,
			"gnn_type": mol_gnn_type,
			"dropout": mol_dropout,
			"normalization": mol_normalization,
		}
		
		# Mol GNN
		self.mol_embedder = GNN(**mol_kwargs)
		self.mol_pool_type = mol_pool_type
		self.mol_pool = build_pool_module(mol_pool_type,mol_hidden_size)
		if int_embedder_tight:
			formula_d = {"max_count_int": 255}
			depth_d = {"max_count_int": num_depth+1}
			complement_d = {"max_count_int": 2}
		else:
			formula_d = depth_d = complement_d = {}
		self.formula_embedder = get_embedder(int_embedder,**formula_d)
		self.depth_embedder = get_embedder(int_embedder,**depth_d)
		self.complement_embedder = get_embedder(int_embedder,**complement_d)
		self.frag_node_feats = frag_node_feats
		self.frag_edge_feats = frag_edge_feats
		self._compute_frag_feats_sizes()
		self.frag_remove_edges = frag_remove_edges

		# define interstage
		assert cc_interstage_type in ["add","sub","linear","direct"]
		self.cc_interstage_type = cc_interstage_type
		if self.cc_interstage_type == "linear":
			self.cc_interstage = nn.Linear(mol_hidden_size * 2, mol_hidden_size)

		frag_kwargs = {
			"node_feats_size": self.frag_node_feats_size,
			"edge_feats_size": self.frag_edge_feats_size,
			"hidden_size": frag_hidden_size,
			"num_layers": frag_num_layers,
			"gnn_type": frag_gnn_type,
			"dropout": frag_dropout,
			"normalization": frag_normalization
		}

		self.frag_embedder = GNN(**frag_kwargs)
		self.frag_pool_type = frag_pool_type
		self.frag_pool = build_pool_module(frag_pool_type,frag_hidden_size)
		self.mlp_output_format = mlp_output_format

		# determine mlp input dims
		# we mlp insert for ce is enabled, add ce here
		mlp_input_dim = 2*self.frag_embedder.hidden_size
		if self.ce_insert_mlp:
			self.ce_transform = transform_nce
			self.ce_embedder = nn.Identity()
			mlp_input_dim += 1
		if self.ce_insert_mlp_lin:
			self.ce_transform = transform_nce
			self.ce_embedder = nn.Linear(1,max(self.frag_embedder.hidden_size//8,2))
			mlp_input_dim += max(self.frag_embedder.hidden_size//8,2)
		if self.m_ce_insert_mlp_embed:
			def m_ce_transform(m_ce):
				m_ce = th.clamp(m_ce, min=0, max=int(NCE_MAX)-1)
				m_ce = th.round(m_ce, decimals=0).long()
				m_ce = m_ce.reshape(-1,1)
				return m_ce
			self.m_ce_transform = m_ce_transform
			embedder = get_embedder(int_embedder, max_count_int=int(NCE_MAX))
			self.m_ce_embedder = nn.Sequential(
				embedder,
				nn.Linear(embedder.num_dim,max(self.frag_embedder.hidden_size//8,2))
			)	
			mlp_input_dim += max(self.frag_embedder.hidden_size//8,2)
		if self.m_ce_insert_mlp_bin:
			def m_ce_transform(m_ce):
				m_ce = th.clamp(m_ce, min=0, max=int(NCE_MAX)-10)
				m_ce = th.round(m_ce, decimals=-1).long() // 10
				m_ce = F.one_hot(m_ce, num_classes=int(NCE_MAX)//10).float()
				return m_ce
			self.m_ce_transform = m_ce_transform
			self.m_ce_embedder = nn.Linear(int(NCE_MAX)//10,max(self.frag_embedder.hidden_size//8,2))
			mlp_input_dim += max(self.frag_embedder.hidden_size//8,2)
		# if self.m_ce_insert_mlp_lin:
		# 	def m_ce_transform(m_ce):
		# 		m_ce = transform_nce(m_ce)
		# 		m_ce = m_ce.reshape(-1,1)
		# 		return m_ce
		# 	self.m_ce_transform = m_ce_transform
		# 	self.m_ce_embedder = nn.Linear(1,max(self.frag_embedder.hidden_size//8,2))
		# 	mlp_input_dim += max(self.frag_embedder.hidden_size//8,2)

		if self.mlp_output_format in ["formula","node_formula"]:
			formula_mlp_kwargs = {
				"input_size": mlp_input_dim,
				"output_size": 2*self.num_hs+1,
				"hidden_size": mlp_hidden_size,
				"num_layers": mlp_num_layers,
				"dropout": mlp_dropout,
				"use_residuals": mlp_use_residuals,
				"normalization": mlp_normalization
			}
			self.formula_module = MLPBlocks(**formula_mlp_kwargs)

		if self.mlp_output_format in ["node_formula"]:
			node_mlp_kwargs = {
				"input_size": 2*self.frag_embedder.hidden_size,
				"output_size": 1,
				"hidden_size": mlp_hidden_size,
				"num_layers": mlp_num_layers,
				"dropout": mlp_dropout,
				"use_residuals": mlp_use_residuals,
				"normalization": mlp_normalization
			}
			self.node_module = MLPBlocks(**node_mlp_kwargs)
		else:
			self.node_module = None
		
		self.predict_oos = predict_oos
		if self.predict_oos:
			oos_mlp_kwargs = {
				"input_size": self.mol_embedder.hidden_size+self.frag_embedder.hidden_size,
				"output_size": 1,
				"hidden_size": mlp_hidden_size,
				"num_layers": mlp_num_layers,
				"dropout": mlp_dropout,
				"use_residuals": mlp_use_residuals,
				"normalization": mlp_normalization
			}
			self.oos_module = MLPBlocks(**oos_mlp_kwargs)
		else:
			self.oos_module = None

		self.skip_edge_loss = skip_edge_loss
		self.mask_null_formula = mask_null_formula

		# this is required
		assert "h_formulae_idx" in self.frag_node_feats

	def _compute_mol_feats_sizes(self):
		""" method compute mol feature size
			these features don't rely on any model parameters
		"""
		self.mol_node_feats_size, self.mol_edge_feats_size = get_mol_feats_sizes(
			self.mol_node_feats, 
			self.mol_edge_feats, 
			self.mol_pe_embed_k
		)
		if self.ce_insert_mol:
			self.mol_node_feats_size += 1

	def _compute_frag_feats_sizes(self):
		""" method compute frag-graph feature size
			these features do depend on model parameters
		"""
		# nodes
		self.frag_node_feats_size = 0
		if "cc" in self.frag_node_feats:
			self.frag_node_feats_size += self.mol_embedder.hidden_size
		if "base_formula" in self.frag_node_feats:
			self.frag_node_feats_size += self.num_elements*self.formula_embedder.num_dim
		if "depth" in self.frag_node_feats:
			self.frag_node_feats_size += self.num_depth*self.depth_embedder.num_dim
		if self.frag_node_feats_size == 0:
			self.frag_node_feats_size = 1
		# edges
		self.frag_edge_feats_size = 0
		if "cc" in self.frag_edge_feats:
			self.frag_edge_feats_size += self.mol_embedder.hidden_size
		if "base_formula" in self.frag_edge_feats:
			self.frag_edge_feats_size += self.num_elements*self.formula_embedder.num_dim
		if "complement" in self.frag_edge_feats:
			self.frag_edge_feats_size += self.complement_embedder.num_dim
		# if self.frag_edge_feats_size == 0:
		# 	self.frag_edge_feats_size = 1
		if self.ce_insert_frag:
			self.frag_node_feats_size += 1
			self.frag_edge_feats_size += 1

	def compile_submodules(self,**kwargs):
		""" pyg does not support dynamic shape compiling """
		self.formula_embedder = th.compile(self.formula_embedder,**kwargs)
		self.depth_embedder = th.compile(self.depth_embedder,**kwargs)
		self.complement_embedder = th.compile(self.complement_embedder,**kwargs)
		if hasattr(self,"ce_embedder"):
			self.ce_embedder = th.compile(self.ce_embedder,**kwargs)
		if hasattr(self,"m_ce_embedder"):
			self.m_ce_embedder = th.compile(self.m_ce_embedder,**kwargs)
		if check_pyg_compile():
			self.mol_embedder = pyg.compile(self.mol_embedder,**kwargs)
			self.frag_embedder = pyg.compile(self.frag_embedder,**kwargs)

	def forward(
		self, 
		mol_pyg: pyg.data.Data,
		frag_pyg: pyg.data.Data,
		mol_num_nodes: th.Tensor,
		frag_num_nodes: th.Tensor,
		frag_formula_peak_idxs: th.Tensor,
		frag_formula_peak_mzs: th.Tensor,
		frag_formula_peak_probs: th.Tensor,
		frag_formula_sizes: th.Tensor,
		frag_formula_cumsizes: th.Tensor,
		frag_formula_peak_sizes: th.Tensor,
		spec_nce: th.Tensor = None,
		spec_m_nce: th.Tensor = None,
		spec_m_nce_batch_idxs: th.Tensor = None,
		**kwargs
	):
		"""forward methods for joint predictor

		Args:
			mol_pyg (pyg.data.Data): molecule pyg data object
			frag_pyg (pyg.data.Data): fragmentation graph pyg data object
			mol_num_nodes (th.Tensor): number of nodes in molecule graph
			frag_num_nodes (th.Tensor): number of nodes in fragmentation graph
			frag_formula_peak_idxs (th.Tensor): _description_
			frag_formula_peak_mzs (th.Tensor): _description_
			frag_formula_peak_probs (th.Tensor): _description_
			frag_formula_sizes (th.Tensor): _description_
			frag_formula_cumsizes (th.Tensor): _description_
			frag_formula_peak_sizes (th.Tensor): _description_

		Returns:
			_type_: _description_
		"""

		# mol_x: mol level node feature matrix
		# mol_edge_index: mol graph connectivity in COO format with shape [2, num_edges]
		# edge_attr: mol graph edge feature matrix with shape [num_edges, num_edge_features]
		# batch: sample idx repsect to current batch
		mol_x, mol_edge_index, mol_edge_attr, mol_batch = mol_pyg.x, mol_pyg.edge_index, mol_pyg.edge_attr, mol_pyg.batch
		# frag_x: frag-graph level node feature matrix
		# frag_edge_index: frag graph connectivity in COO format with shape [2, num_edges]
		# frag_edge_attr: frag graph edge feature matrix with shape [num_edges, num_edge_features]
		# batch: sample idx repsect to current batch
		frag_x, frag_edge_index, frag_edge_attr, frag_batch = frag_pyg.x, frag_pyg.edge_index, frag_pyg.edge_attr, frag_pyg.batch

		device = mol_num_nodes.device
		# int_dtype = mol_edge_index.dtype
		float_dtype = mol_edge_attr.dtype
		frag_node_feat_idxs = frag_pyg.node_feat_idxs[0]
		frag_edge_feat_idxs = frag_pyg.edge_feat_idxs[0]
		
		# get ce value)
		ce = spec_nce
		if ce is not None:
			ce = ce.reshape(-1,1)
		m_ce = spec_m_nce
		m_ce_batch_idxs = spec_m_nce_batch_idxs

		if self.ce_insert_mol:
			mol_ce_embed = th.repeat_interleave(ce,th.unique(mol_batch,return_counts=True)[1],dim=0)
			mol_x = th.cat([mol_x,mol_ce_embed],dim=1)

		# get per-atom embeddings
		mol_embed_gnn = self.mol_embedder(
			mol_x,
			mol_batch,
			mol_edge_index,
			mol_edge_attr
		)
		mol_embed_gnn_pool = self.mol_pool(mol_embed_gnn,mol_batch)

		# process dag
		# create interstage
		frag_ndata, frag_edata = [], []
		batch_frag_num_nodes = frag_x.shape[0]
		batch_frag_num_edges = frag_edge_index.shape[1]
		batch_frag_num_formulae = frag_formula_cumsizes[-1]
		batch_size = frag_batch[-1]+1
		# node atom embeddings
		if "cc" in self.frag_node_feats:
			frag_node_mask = th_long_to_mask(get_node_feats(frag_x,frag_node_feat_idxs,"cc").to(device))
			frag_node_mask_idxs = th.nonzero(frag_node_mask).long()
			frag_node_offsets = mol_num_nodes[th.bucketize(frag_node_mask_idxs[:,0],frag_num_nodes,right=True)-1]
			frag_node_mask_idxs[:,1] = frag_node_mask_idxs[:,1] + frag_node_offsets
			frag_node_mask_embed = scatter_reduce(
				src=mol_embed_gnn[frag_node_mask_idxs[:,1]],
				index=frag_node_mask_idxs[:,0:1].expand(-1,mol_embed_gnn.shape[1]),
				reduce="sum",
				dim=0,
				dim_size=batch_frag_num_nodes
			)
			# frag_node_mask_embed can be one of following
			# 1. frag_node_mask_embed plus mol_embed_gnn_pool
			# 2. frag_node_mask_embed only, with out pooled embed for entire mol
			# 3. a linear layer between masked embed and pooled embed
			if self.cc_interstage_type == "add":
				frag_node_mask_embed = frag_node_mask_embed + mol_embed_gnn_pool[frag_batch]
			elif self.cc_interstage_type == "sub":
				frag_node_mask_embed = frag_node_mask_embed - mol_embed_gnn_pool[frag_batch]
			elif self.cc_interstage_type == "linear":
				frag_node_mask_embed = self.cc_interstage(th.cat([frag_node_mask_embed,mol_embed_gnn_pool[frag_batch]], dim = 1))
			else:
				assert self.cc_interstage_type == "direct", self.cc_interstage_type
				frag_node_mask_embed = frag_node_mask_embed
			frag_ndata.append(frag_node_mask_embed)
		# edge atom embeddings
		if "cc" in self.frag_edge_feats:
			# connected competents
			frag_edge_mask = th_long_to_mask(get_edge_feats(frag_edge_attr,frag_edge_feat_idxs,"cc").to(device))
			frag_edge_mask_idxs = th.nonzero(frag_edge_mask).long()
			frag_edge_node_idxs = frag_edge_index[0][frag_edge_mask_idxs[:,0]]
			frag_edge_offsets = mol_num_nodes[th.bucketize(frag_edge_node_idxs,frag_num_nodes,right=True)-1]
			frag_edge_mask_idxs[:,1] = frag_edge_mask_idxs[:,1] + frag_edge_offsets
			frag_edge_mask_embed = scatter_reduce(
				src=mol_embed_gnn[frag_edge_mask_idxs[:,1]],
				index=frag_edge_mask_idxs[:,0:1].expand(-1,mol_embed_gnn.shape[1]),
				reduce="sum",
				dim=0,
				dim_size=batch_frag_num_edges
			)
			frag_edata.append(frag_edge_mask_embed)
		# node formulae
		if "base_formula" in self.frag_node_feats:
			frag_node_formula = self.formula_embedder(
				get_node_feats(frag_x,frag_node_feat_idxs,"base_formula").reshape(batch_frag_num_nodes,-1)
			)
			frag_ndata.append(frag_node_formula)
		# edge formulae
		if "base_formula" in self.frag_edge_feats:
			frag_edge_formula = self.formula_embedder(
				get_edge_feats(frag_edge_attr,frag_edge_feat_idxs,"base_formula").reshape(batch_frag_num_edges,-1)
			)
			frag_edata.append(frag_edge_formula)
		# node depth
		if "depth" in self.frag_node_feats:
			frag_depth = self.depth_embedder(
				get_node_feats(frag_x,frag_node_feat_idxs,"depth").reshape(batch_frag_num_nodes,-1)
			)
			frag_ndata.append(frag_depth)
		# edge complement
		if "complement" in self.frag_edge_feats:
			frag_edge_complement = self.complement_embedder(
				get_edge_feats(frag_edge_attr,frag_edge_feat_idxs,"complement").reshape(batch_frag_num_edges,-1)
			)
			frag_edata.append(frag_edge_complement)
		# empty feats check
		if len(frag_ndata) == 0:
			assert self.frag_node_feats_size == 1, self.frag_node_feats_size
			frag_ndata.append(th.zeros([batch_frag_num_nodes,1],dtype=float_dtype,device=device))
		if len(frag_edata) == 0:
			assert self.frag_edge_feats_size == 0, self.frag_edge_feats_size
			frag_edata.append(th.zeros([batch_frag_num_edges,0],dtype=float_dtype,device=device))

		if self.ce_insert_frag:
			edge_ce = th.repeat_interleave(ce,th.unique(frag_batch[frag_edge_index[0]],return_counts=True)[1],dim=0)
			frag_edata.append(edge_ce)
			node_ce = th.repeat_interleave(ce,th.unique(frag_batch,return_counts=True)[1],dim=0)
			frag_ndata.append(node_ce)

		# get output formula aggregation 
		frag_node_batch_idxs = frag_batch
		frag_node_offsets = frag_formula_cumsizes[frag_node_batch_idxs]
		frag_formula_idxs = (get_node_feats(frag_x,frag_node_feat_idxs,"h_formulae_idx")+frag_node_offsets.unsqueeze(-1)).flatten()
		# get isomorphism aggregation
		frag_nb_idxs = get_node_feats(frag_x,frag_node_feat_idxs,"nb_iso_idx").flatten()
		frag_nb_offsets = scatter_reduce(
			frag_nb_idxs,
			frag_node_batch_idxs,
			reduce="amax",
			dim=0,
			dim_size=batch_size
		)
		frag_nb_offsets = th.cat(
			[
				th.zeros([1],dtype=frag_nb_offsets.dtype,device=device),
				frag_nb_offsets+1
			], dim=0
		)
		frag_nb_offsets = th.cumsum(frag_nb_offsets,dim=0)
		batch_frag_num_nodes_nb = frag_nb_offsets[-1].item()
		frag_nb_offsets = th.gather(
			input=frag_nb_offsets[:-1],
			index=frag_node_batch_idxs,
			dim=0
		)
		frag_nb_idxs = frag_nb_idxs + frag_nb_offsets
		assert th.max(frag_nb_idxs) < batch_frag_num_nodes_nb, (th.max(frag_nb_idxs),batch_frag_num_nodes_nb)
		frag_nb_un_idxs, frag_nb_inv_idxs = th.unique(frag_nb_idxs,return_inverse=True)

		# assemble all features for dag
		# concatenate everything 
		frag_x_embed = th.cat(frag_ndata,dim=-1)
		# concatenate everything 
		frag_edge_attr_embed = th.cat(frag_edata,dim=-1)

		# define frag network
		if self.frag_remove_edges:
			assert self.frag_edge_feats_size == 0, self.frag_edge_feats_size
			frag_edge_index = frag_edge_index[:,:0]
			frag_edge_attr_embed = frag_edge_attr_embed[:0,:]
		frag_embed_gnn = self.frag_embedder(
			frag_x_embed,
			frag_node_batch_idxs,
			frag_edge_index,
			frag_edge_attr_embed
		)
		frag_embed_gnn_pool = self.frag_pool(frag_embed_gnn,frag_batch)
		frag_embed_node = self.frag_embedder.input_project(frag_x_embed)
		frag_embed_node_pool = self.frag_pool(frag_embed_node,frag_batch)

		# get frag dag embedding
		frag_embed_parts = [
			frag_embed_node - frag_embed_node_pool[frag_batch],
			frag_embed_gnn - frag_embed_gnn_pool[frag_batch]
		]
		if self.ce_insert_mlp or self.ce_insert_mlp_lin:
			mlp_ce_embed = self.ce_embedder(self.ce_transform(ce))
			mlp_ce_embed = th.repeat_interleave(mlp_ce_embed,th.unique(frag_node_batch_idxs,return_counts=True)[1],dim=0)
			frag_embed_parts.append(mlp_ce_embed)
		if self.m_ce_insert_mlp_embed or self.m_ce_insert_mlp_bin: #or self.m_ce_insert_mlp_lin:
			mlp_m_ce_embed = self.m_ce_transform(m_ce)
			mlp_m_ce_embed = self.m_ce_embedder(mlp_m_ce_embed)
			mlp_m_ce_embed = scatter_reduce(
				src=mlp_m_ce_embed,
				index=m_ce_batch_idxs.unsqueeze(1).expand_as(mlp_m_ce_embed),
				reduce="mean",
				dim=0,
				dim_size=batch_size
			)
			mlp_m_ce_embed = th.repeat_interleave(mlp_m_ce_embed,th.unique(frag_node_batch_idxs,return_counts=True)[1],dim=0)
			frag_embed_parts.append(mlp_m_ce_embed)

		frag_embed = th.cat(frag_embed_parts, dim=1)

		frag_joint_batch_idxs = th.repeat_interleave(frag_node_batch_idxs,2*self.num_hs+1)
		frag_joint_mask = (~th.isin(frag_formula_idxs,frag_formula_cumsizes[:-1])).float()

		if self.mlp_output_format == "formula":

			# log p(f,n)
			frag_joint_logits = self.formula_module(frag_embed)
			assert frag_joint_logits.shape[1] == 2*self.num_hs+1, (frag_joint_logits.shape[1],2*self.num_hs+1)
			frag_joint_logits = frag_joint_logits.flatten()
			
			# compute total NULL probability (before renormalization)
			frag_joint_logprobs = scatter_logsoftmax(
				frag_joint_logits,
				frag_joint_batch_idxs,
				dim=0
			)
			frag_null_formula_logprob = scatter_logsumexp(
				(1.-frag_joint_mask) * frag_joint_logprobs + frag_joint_mask * (-1e9),
				frag_joint_batch_idxs,
				dim=0
			)

			if self.mask_null_formula:
				# compute non-NULL renormalized intensity
				frag_joint_logprobs = scatter_masked_softmax(
					frag_joint_logits,
					frag_joint_mask,
					frag_joint_batch_idxs,
					dim=0,
					log=True
				)
			
			# reshape
			frag_joint_logprobs = frag_joint_logprobs.reshape(-1,2*self.num_hs+1)

			# aggregate formula logits by node, then normalize
			# log p(n) = logsumexp_f log p(f,n)
			frag_node_logits = th.logsumexp(frag_joint_logprobs,dim=1)
			frag_node_logprobs = scatter_logsoftmax(
				frag_node_logits,
				frag_node_batch_idxs,
				dim=0
			)
			
			# calculate conditional probability
			# log p(f|n) = log p(f,n) - log p(n)
			frag_node_formula_logprobs = frag_joint_logprobs - frag_node_logprobs.unsqueeze(-1)
			frag_node_formula_logprobs = th.log_softmax(frag_node_formula_logprobs,dim=1)

		else:

			assert self.mlp_output_format == "node_formula", self.mlp_output_format

			# log p(f|n)
			frag_node_formula_logits = self.formula_module(frag_embed)
			assert frag_node_formula_logits.shape[1] == 2*self.num_hs+1, (frag_node_formula_logits.shape[1],2*self.num_hs+1)
			frag_node_formula_logprobs = th.log_softmax(frag_node_formula_logits,dim=1)

			# log p(n)
			frag_node_logits = self.node_module(frag_embed).squeeze(1)
			frag_node_logprobs = scatter_logsoftmax(
				frag_node_logits,
				frag_node_batch_idxs,
				dim=0
			)

			# log p(f,n) = log p(f|n) + log p(n)
			frag_joint_logprobs = frag_node_formula_logprobs + frag_node_logprobs.unsqueeze(-1)
			frag_joint_logprobs = frag_joint_logprobs.flatten()

			# compute total NULL probability (before renormalization)
			frag_null_formula_logprob = scatter_logsumexp(
				(1.-frag_joint_mask) * frag_joint_logprobs + frag_joint_mask * (-1e9),
				frag_joint_batch_idxs,
				dim=0
			)

			if self.mask_null_formula:
				# compute non-NULL renormalized intensity
				frag_joint_logprobs = scatter_masked_softmax(
					frag_joint_logprobs,
					frag_joint_mask,
					frag_joint_batch_idxs,
					dim=0,
					log=True
				)

		# aggregate by formula
		# log p(f) = logsumexp_n log p(f,n)
		frag_formula_batch_idxs = th.bucketize(
			th.arange(frag_formula_cumsizes[-1],device=device),
			frag_formula_cumsizes,
			right=True
		)-1
		frag_formula_mask = th.ones_like(frag_formula_batch_idxs,dtype=float_dtype)
		frag_formula_mask[frag_formula_cumsizes[:-1]] = 0.
		# aggregate by formula
		frag_formula_logprobs = scatter_logsumexp(
			frag_joint_logprobs.flatten(),
			frag_formula_idxs,
			dim=0
		)

		# softmax over formulae
		if self.mask_null_formula:
			frag_formula_logprobs = scatter_masked_softmax(
				frag_formula_logprobs,
				frag_formula_mask,
				frag_formula_batch_idxs,
				dim=0,
				log=True
			)
		else:
			frag_formula_logprobs = scatter_masked_softmax(
				frag_formula_logprobs,
				th.ones_like(frag_formula_logprobs),
				frag_formula_batch_idxs,
				dim=0,
				log=True
			)

		if self.predict_oos:
			# get OOS logits and logprobs
			oos_logits = self.oos_module(th.cat([mol_embed_gnn_pool,frag_embed_gnn_pool],dim=1))
			oos_logits = oos_logits.flatten()
			oos_logprobs = F.logsigmoid(oos_logits)
			not_oos_logprobs = F.logsigmoid(-oos_logits)
		else:
			# set them to 0
			oos_logprobs = (-1e9)*th.ones([batch_size],device=device)
			not_oos_logprobs = th.zeros([batch_size],device=device)
		# adjust frag_formula_logprobs
		frag_formula_oos_logprobs = frag_formula_logprobs + \
			th.repeat_interleave(not_oos_logprobs, frag_formula_sizes, dim=0)

		# convert to spectrum
		frag_formula_offsets = th.repeat_interleave(
			frag_formula_cumsizes[:-1],
			frag_formula_sizes-1 # -1 is for NULL formulae
		)
		spec_mzs = frag_formula_peak_mzs
		spec_logprobs = frag_formula_oos_logprobs[frag_formula_peak_idxs+frag_formula_offsets] + th.log(frag_formula_peak_probs)

		# get batch idxs
		spec_batch_idxs = th.repeat_interleave(
			th.arange(frag_formula_peak_sizes.shape[0],device=device),
			frag_formula_peak_sizes
		)

		if not self.skip_edge_loss:

			frag_edge_batch_idxs = frag_batch[frag_edge_index[0]]
			frag_edge_logits = frag_node_logprobs[frag_edge_index[0]] + frag_node_logprobs[frag_edge_index[1]]
			frag_edge_logprobs = scatter_logsoftmax(
				frag_edge_logits,
				frag_edge_batch_idxs,
				dim=0
			)
			# print(scatter_logsumexp(frag_edge_logprobs,frag_edge_batch_idxs,dim=0))
			frag_node_h_counts = get_node_feats(frag_x,frag_node_feat_idxs,"h_counts")
			frag_edge_h_ranges = get_edge_feats(frag_edge_attr,frag_edge_feat_idxs,"h_range")
			frag_edge_h_diffs = frag_node_h_counts[frag_edge_index[0]].unsqueeze(1) - frag_node_h_counts[frag_edge_index[1]].unsqueeze(2)
			frag_edge_h_diffs = frag_edge_h_diffs.reshape(frag_edge_h_diffs.shape[0],-1)
			frag_edge_h_range_masks = th.logical_or(
				frag_edge_h_diffs<frag_edge_h_ranges[:,0].unsqueeze(-1), 
				frag_edge_h_diffs>frag_edge_h_ranges[:,1].unsqueeze(-1)
			)
			frag_edge_h_logprobs = (frag_node_formula_logprobs[frag_edge_index[0]]).unsqueeze(1)  \
				+ (frag_node_formula_logprobs[frag_edge_index[1]]).unsqueeze(2)
			frag_edge_h_logprobs = frag_edge_h_logprobs.reshape(frag_edge_h_logprobs.shape[0],-1) 

		else:

			frag_edge_logprobs = None
			frag_edge_h_diffs = None
			frag_edge_h_range_masks = None
			frag_edge_h_logprobs = None
			frag_edge_batch_idxs = None

		frag_joint_node_idxs = th.repeat_interleave(th.arange(frag_joint_mask.shape[0]//(2*self.num_hs+1),device=device),2*self.num_hs+1)
		frag_joint_formula_idxs = frag_formula_idxs
		# select (will remove all NULL formula idxs, potentially some node idxs too if they contain only NULLs)
		frag_real_joint_logits = frag_joint_logits[frag_joint_mask.bool()]
		frag_real_joint_node_idxs = frag_joint_node_idxs[frag_joint_mask.bool()]
		frag_real_joint_formula_idxs = frag_joint_formula_idxs[frag_joint_mask.bool()]
		frag_real_joint_batch_idxs = frag_joint_batch_idxs[frag_joint_mask.bool()]
		frag_real_joint_logprobs = scatter_logsoftmax(
			frag_real_joint_logits,
			frag_real_joint_batch_idxs,
			dim=0
		)
		# sum, renormalize, but keep all-NULL nodes (as zeros)
		frag_real_node_logprobs = scatter_logsumexp(
			frag_real_joint_logprobs,
			frag_real_joint_node_idxs,
			dim=0,
			dim_size=batch_frag_num_nodes
		)
		frag_real_node_logprobs = scatter_logsoftmax(
			frag_real_node_logprobs,
			frag_node_batch_idxs,
			dim=0
		)
		# sum, renormalize, but keep NULL formulae (as zeros)
		frag_real_formula_logprobs = scatter_logsumexp(
			frag_real_joint_logprobs,
			frag_real_joint_formula_idxs,
			dim=0,
			dim_size=batch_frag_num_formulae
		)
		frag_real_formula_logprobs = scatter_logsoftmax(
			frag_real_formula_logprobs,
			frag_formula_batch_idxs,
			dim=0
		)
		# remove all NULLs from conditionals
		frag_real_node_formula_logprobs = frag_real_joint_logprobs - frag_real_node_logprobs[frag_real_joint_node_idxs]
		frag_real_node_formula_logprobs = scatter_logsoftmax(
			frag_real_node_formula_logprobs,
			frag_real_joint_node_idxs,
			dim=0,
		)
		frag_real_formula_node_logprobs = frag_real_joint_logprobs - frag_real_formula_logprobs[frag_real_joint_formula_idxs]
		frag_real_formula_node_logprobs = scatter_logsoftmax(
			frag_real_formula_node_logprobs,
			frag_real_joint_formula_idxs,
			dim=0,
		)
		# calculate isomorphic distributions
		# P'(n)
		frag_node_nb_logprobs = scatter_logsumexp(
			frag_real_node_logprobs,
			frag_nb_idxs,
			dim=0,
			dim_size=batch_frag_num_nodes_nb
		)
		frag_node_nb_batch_idxs = scatter_reduce(
			frag_node_batch_idxs,
			frag_nb_idxs,
			reduce="amax",
			dim=0,
			dim_size=batch_frag_num_nodes_nb
		)
		frag_node_nb_idxs = frag_nb_un_idxs
		# P'(n|f)
		frag_joint_nb_idxs = frag_nb_idxs[frag_real_joint_node_idxs]
		frag_joint_nb_both_idxs = th.stack(
			[
				frag_joint_nb_idxs,
				frag_real_joint_formula_idxs
			], dim=1
		)
		frag_joint_nb_both_un_idxs, frag_joint_nb_both_inv_idxs = th.unique(
			frag_joint_nb_both_idxs,
			return_inverse=True,
			dim=0
		)
		frag_joint_nb_node_idxs = frag_joint_nb_both_un_idxs[:,0]
		frag_joint_nb_formula_idxs = frag_joint_nb_both_un_idxs[:,1]
		frag_joint_nb_batch_idxs = frag_node_nb_batch_idxs[frag_joint_nb_node_idxs]
		frag_formula_node_nb_logprobs = scatter_logsumexp(
			frag_real_node_formula_logprobs,
			frag_joint_nb_both_inv_idxs,
			dim=0,
			dim_size=frag_joint_nb_both_un_idxs.shape[0]
		)
		# P'(f|n)
		frag_node_rel_logprobs = frag_real_node_logprobs - frag_node_nb_logprobs[frag_nb_inv_idxs]
		frag_node_rel_logprobs = (frag_real_node_logprobs<=LOG_ZERO).float() * LOG_ZERO + \
			(frag_real_node_logprobs>LOG_ZERO).float() * frag_node_rel_logprobs
		lse_debug = scatter_logsumexp(frag_node_rel_logprobs, frag_nb_idxs, dim=0, dim_size=batch_frag_num_nodes_nb)
		assert th.all(lse_debug < 1e-6)
		frag_node_formula_nb_logprobs = frag_real_node_formula_logprobs + frag_node_rel_logprobs[frag_real_joint_node_idxs]
		frag_node_formula_nb_logprobs = scatter_logsumexp(
			frag_node_formula_nb_logprobs,
			frag_joint_nb_both_inv_idxs,
			dim=0,
			dim_size=frag_joint_nb_both_un_idxs.shape[0]
		)

		out_d = {
			"pred_mzs": spec_mzs,
			"pred_logprobs": spec_logprobs,
			"pred_batch_idxs": spec_batch_idxs,
			"pred_formula_logprobs": frag_real_formula_logprobs,
			"pred_formula_batch_idxs": frag_formula_batch_idxs,
			"pred_node_logprobs": frag_real_node_logprobs,
			"pred_node_batch_idxs": frag_node_batch_idxs,
			"pred_node_formula_logprobs": frag_real_node_formula_logprobs,
			"pred_formula_node_logprobs": frag_real_formula_node_logprobs,
			"pred_joint_node_idxs": frag_real_joint_node_idxs,
			"pred_joint_formula_idxs": frag_real_joint_formula_idxs,
			"pred_joint_batch_idxs": frag_real_joint_batch_idxs,
			"pred_null_formula_logprob": frag_null_formula_logprob,
			"pred_edge_logprobs": frag_edge_logprobs,
			"pred_edge_h_diffs": frag_edge_h_diffs,
			"pred_edge_h_range_masks": frag_edge_h_range_masks,
			"pred_edge_h_logprobs": frag_edge_h_logprobs,
			"pred_edge_batch_idxs": frag_edge_batch_idxs,
			"pred_oos_logprobs": oos_logprobs,
			# nb stuff
			"pred_node_nb_logprobs": frag_node_nb_logprobs,
			"pred_node_formula_nb_logprobs": frag_node_formula_nb_logprobs,
			"pred_formula_node_nb_logprobs": frag_formula_node_nb_logprobs,
			"pred_node_nb_idxs": frag_node_nb_idxs,
			"pred_node_nb_batch_idxs": frag_node_nb_batch_idxs,
			"pred_joint_nb_node_idxs": frag_joint_nb_node_idxs,
			"pred_joint_nb_formula_idxs": frag_joint_nb_formula_idxs,
			"pred_joint_nb_batch_idxs": frag_joint_nb_batch_idxs,
		}

		return out_d

class NeimsModel(nn.Module):

	def __init__(
		self,
		mol_fingerprint_morgan: bool,
		mol_fingerprint_rdkit: bool,
		mol_fingerprint_maccs: bool,
		mlp_hidden_size: int,
		mlp_dropout: float,
		mlp_num_layers: int,
		mlp_use_residuals: bool,
		mz_max: int,
		mz_bin_res: float,
		ff_prec_mz_offset: int,
		ff_bidirectional: bool,
		ce_insert_mol: bool,
		ce_insert_frag: bool,
		ce_insert_mlp: bool,
		ce_insert_mlp_lin: bool,
	):

		super().__init__()
		self.mol_fingerprint_morgan = mol_fingerprint_morgan
		self.mol_fingerprint_rdkit = mol_fingerprint_rdkit
		self.mol_fingerprint_maccs = mol_fingerprint_maccs
		self.mz_max = mz_max
		self.mz_bin_res = mz_bin_res
		self.prec_mz_offset = ff_prec_mz_offset
		self.bidirectional = ff_bidirectional
		self.use_residuals = mlp_use_residuals
		
		# check ce insertion type
		self.ce_insert_mol = ce_insert_mol
		self.ce_insert_frag = ce_insert_frag
		self.ce_insert_mlp = ce_insert_mlp
		self.ce_insert_mlp_lin = ce_insert_mlp_lin

		self._compute_input_size()
		self._compute_output_size()

		if self.ce_insert_mlp_lin:
			self.ce_embedder = nn.Linear(1,12)
		self.in_layer = nn.Linear(self.input_size, mlp_hidden_size)
		self.ff_layers = nn.ModuleList([])
		# self.ff_layers.append(nn.Linear(mlp_hidden_size, mlp_hidden_size))
		for i in range(mlp_num_layers):
			self.ff_layers.append(
				NeimsBlock(
					mlp_hidden_size,
					mlp_hidden_size,
					mlp_dropout))
		if self.bidirectional:
			# assumes gating, mass masking
			self.forw_out_layer = nn.Linear(mlp_hidden_size, self.output_size)
			self.rev_out_layer = nn.Linear(mlp_hidden_size, self.output_size)
			self.out_gate = nn.Sequential(
				*[nn.Linear(mlp_hidden_size, self.output_size), nn.Sigmoid()])
		else:
			self.out_layer = nn.Linear(mlp_hidden_size, self.output_size)
			self.out_gate = nn.Sequential(
				*[nn.Linear(mlp_hidden_size, self.output_size), nn.Sigmoid()])
		self.out_activation = nn.ReLU()
		self.out_normalization = lambda x: F.normalize(x, p=1, dim=1)

	def _compute_input_size(self):

		self.mol_fp_size = get_mol_fp_size(self.mol_fingerprint_morgan, self.mol_fingerprint_rdkit, self.mol_fingerprint_maccs)
		self.input_size = self.mol_fp_size
		if self.ce_insert_mlp:
			self.input_size += 1
		if self.ce_insert_mlp_lin:
			self.input_size += 12

	def _compute_output_size(self):

		self.mz_bins = nn.Parameter(th.arange(self.mz_bin_res,self.mz_max+self.mz_bin_res,self.mz_bin_res))
		self.mzs = nn.Parameter(self.mz_bins-0.5*self.mz_bin_res)
		self.output_size = self.mzs.shape[0]

	def _prec_mz_to_idx(self,prec_mz):

		prec_mz_idx = th.bucketize(prec_mz,self.mz_bins.to(prec_mz.device),right=True)
		assert th.max(prec_mz_idx) < self.output_size, (prec_mz_idx,self.output_size)
		return prec_mz_idx

	def forward(
		self,
		mol_fingerprint: th.Tensor, 
		spec_prec_mz: th.Tensor,
		spec_nce: th.Tensor = None,
		**kwargs
	):

		prec_mz_idxs = self._prec_mz_to_idx(spec_prec_mz)
		# get ce
		ce = spec_nce
		fh = mol_fingerprint.reshape(-1,self.mol_fp_size)
		# add CE information
		if self.ce_insert_mlp:
			ce_embed = ce.reshape(-1,1)
			fh = th.cat([fh,ce_embed],dim=1)
		if self.ce_insert_mlp_lin:
			ce_embed = self.ce_embedder(ce.reshape(-1,1))
			fh = th.cat([fh,ce_embed],dim=1)
		# process fingerprint
		fh = self.in_layer(fh)
		# big MLP
		for layer in self.ff_layers:
			if self.use_residuals:
				fh = fh + layer(fh)
			else:
				fh = layer(fh)
		# bidirectional prediction
		if self.bidirectional:
			ff = self.forw_out_layer(fh)
			fr = reverse_prediction(
				self.rev_out_layer(fh),
				prec_mz_idxs,
				self.prec_mz_offset)
			fg = self.out_gate(fh)
			fo = ff * fg + fr * (1. - fg)
			fo = mask_prediction_by_mass(
				fo, prec_mz_idxs, self.prec_mz_offset)
		else:
			# apply output layer
			fo = self.out_layer(fh)
			# apply gating
			fg = self.out_gate(fh)
			fo = fg * fo
		fo = self.out_activation(fo)
		spec = self.out_normalization(fo)
		# handle all zeroes (set to first bin by default)
		all_zero_mask = th.max(spec,dim=1)[0]<=0.
		all_zero_bonus = th.zeros_like(spec)
		all_zero_bonus[all_zero_mask,0] = 1.
		spec = spec + all_zero_bonus
		# convert dense spectrum to sparse
		mask = spec>0.
		pred_mzs = (self.mzs.unsqueeze(0).expand(spec.shape[0],-1))[mask]
		pred_logprobs = safelog(spec)[mask]
		pred_batch_idxs = th.arange(spec.shape[0],device=spec.device).unsqueeze(1).expand(-1,spec.shape[1])[mask]
		pred_specs = spec
		out_d = {
			"pred_mzs": pred_mzs,
			"pred_logprobs": pred_logprobs,
			"pred_batch_idxs": pred_batch_idxs,
			"pred_specs": pred_specs
		}
		return out_d
